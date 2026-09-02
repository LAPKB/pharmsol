//! Symbolic differentiation of compiled dynamics with respect to the states.
//!
//! Produces the row-major state Jacobian `J[i][j] = d(dx_i)/d(x_j)` as an
//! ordinary [`ModelFunction`], so it lowers through the same backend path as
//! every other model function.
//!
//! Two consumers:
//!
//! - numeric solvers, which otherwise approximate `J v` by finite differences;
//! - [`is_state_free`], which reports whether the Jacobian is constant in the
//!   states. A state-free Jacobian means the system is linear, so a solver may
//!   propagate it in closed form instead of integrating it.
//!
//! Differentiation is deliberately partial. Anything it cannot handle exactly
//! is reported as a [`JacobianDecline`] and callers fall back to their existing
//! numeric path; nothing here is allowed to return an approximate derivative.

use crate::execution::{
    ExecutionBlock, ExecutionCall, ExecutionExpr, ExecutionExprKind, ExecutionIfStmt,
    ExecutionLoad, ExecutionModel, ExecutionProgram, ExecutionStateRef, ExecutionStmt,
    ExecutionStmtKind, ExecutionTarget, ExecutionTargetKind, FunctionBody, ModelFunction,
    ModelFunctionKind,
};
use crate::{
    AnalyzedBinaryOp, AnalyzedUnaryOp, ConstValue, MathFunction, ModelKind, Span, ValueType,
};
use std::collections::BTreeMap;

/// Why a model's Jacobian could not be derived symbolically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianDecline {
    pub reason: String,
    pub span: Span,
}

impl JacobianDecline {
    fn new(reason: impl Into<String>, span: Span) -> Self {
        Self {
            reason: reason.into(),
            span,
        }
    }
}

impl std::fmt::Display for JacobianDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// Derive the state Jacobian of a model's dynamics (ODE) or drift (SDE).
pub fn build_jacobian(model: &ExecutionModel) -> Result<ModelFunction, JacobianDecline> {
    let source_kind = match model.kind {
        ModelKind::Ode => ModelFunctionKind::Dynamics,
        ModelKind::Sde => ModelFunctionKind::Drift,
        ModelKind::Analytical => {
            return Err(JacobianDecline::new(
                "analytical models have no dynamics to differentiate",
                model.span,
            ))
        }
    };

    let function = model.function(source_kind).ok_or_else(|| {
        JacobianDecline::new("model declares no dynamics function", model.span)
    })?;
    let FunctionBody::Statements(program) = &function.body else {
        return Err(JacobianDecline::new(
            "dynamics function is not a statement program",
            function.span,
        ));
    };

    let states = model.layout.state_buffer.len;
    let mut context = Differentiator {
        states,
        local_derivatives: BTreeMap::new(),
    };
    let statements = context.differentiate_block(&program.body.statements)?;

    Ok(ModelFunction {
        kind: ModelFunctionKind::Jacobian,
        signature: crate::execution::jacobian_signature(),
        body: FunctionBody::Statements(ExecutionProgram {
            locals: program.locals.clone(),
            body: ExecutionBlock {
                statements,
                span: program.body.span,
            },
        }),
        span: function.span,
    })
}

/// `true` when no expression in the function reads a state, i.e. the Jacobian
/// is constant over an interval and the underlying system is linear in `x`.
pub fn is_state_free(function: &ModelFunction) -> bool {
    let FunctionBody::Statements(program) = &function.body else {
        return false;
    };
    !block_reads(&program.body.statements, &|load| {
        matches!(load, ExecutionLoad::State(_))
    })
}

/// `true` when any expression in the function performs a load matching
/// `predicate`.
pub fn function_reads(function: &ModelFunction, predicate: &dyn Fn(&ExecutionLoad) -> bool) -> bool {
    let FunctionBody::Statements(program) = &function.body else {
        return false;
    };
    block_reads(&program.body.statements, predicate)
}

fn block_reads(statements: &[ExecutionStmt], predicate: &dyn Fn(&ExecutionLoad) -> bool) -> bool {
    statements.iter().any(|statement| match &statement.kind {
        ExecutionStmtKind::Let(let_stmt) => expr_reads(&let_stmt.value, predicate),
        ExecutionStmtKind::Assign(assign) => {
            target_reads(&assign.target, predicate) || expr_reads(&assign.value, predicate)
        }
        ExecutionStmtKind::If(if_stmt) => {
            expr_reads(&if_stmt.condition, predicate)
                || block_reads(&if_stmt.then_branch, predicate)
                || if_stmt
                    .else_branch
                    .as_deref()
                    .is_some_and(|branch| block_reads(branch, predicate))
        }
        ExecutionStmtKind::For(for_stmt) => {
            expr_reads(&for_stmt.range.start, predicate)
                || expr_reads(&for_stmt.range.end, predicate)
                || block_reads(&for_stmt.body, predicate)
        }
    })
}

fn target_reads(target: &ExecutionTarget, predicate: &dyn Fn(&ExecutionLoad) -> bool) -> bool {
    let state_ref = match &target.kind {
        ExecutionTargetKind::StateInit(state)
        | ExecutionTargetKind::StateDerivative(state)
        | ExecutionTargetKind::StateNoise(state) => state,
        _ => return false,
    };
    state_ref
        .index
        .as_deref()
        .is_some_and(|index| expr_reads(index, predicate))
}

fn expr_reads(expr: &ExecutionExpr, predicate: &dyn Fn(&ExecutionLoad) -> bool) -> bool {
    match &expr.kind {
        ExecutionExprKind::Literal(_) => false,
        ExecutionExprKind::Load(load) => {
            if predicate(load) {
                return true;
            }
            match load {
                ExecutionLoad::State(state) => state
                    .index
                    .as_deref()
                    .is_some_and(|index| expr_reads(index, predicate)),
                _ => false,
            }
        }
        ExecutionExprKind::Unary { expr, .. } => expr_reads(expr, predicate),
        ExecutionExprKind::Binary { lhs, rhs, .. } => {
            expr_reads(lhs, predicate) || expr_reads(rhs, predicate)
        }
        ExecutionExprKind::Call { args, .. } => {
            args.iter().any(|arg| expr_reads(arg, predicate))
        }
    }
}

struct Differentiator {
    states: usize,
    /// `local slot -> (state column -> derivative expression)`. Only non-zero
    /// columns are stored.
    local_derivatives: BTreeMap<usize, BTreeMap<usize, ExecutionExpr>>,
}

impl Differentiator {
    fn differentiate_block(
        &mut self,
        statements: &[ExecutionStmt],
    ) -> Result<Vec<ExecutionStmt>, JacobianDecline> {
        let mut out = Vec::with_capacity(statements.len());
        for statement in statements {
            match &statement.kind {
                ExecutionStmtKind::Let(let_stmt) => {
                    // The value itself is still needed: product and quotient
                    // rules refer to it.
                    out.push(statement.clone());
                    let mut columns = BTreeMap::new();
                    for column in 0..self.states {
                        let derivative = self.differentiate(&let_stmt.value, column)?;
                        if !is_zero(&derivative) {
                            columns.insert(column, derivative);
                        }
                    }
                    self.local_derivatives.insert(let_stmt.local, columns);
                }
                ExecutionStmtKind::Assign(assign) => {
                    let ExecutionTargetKind::StateDerivative(state) = &assign.target.kind else {
                        return Err(JacobianDecline::new(
                            "dynamics assigns a target that is not a state derivative",
                            statement.span,
                        ));
                    };
                    let row = constant_state_offset(state)?;
                    // Every column of the row is written, including zeros, so a
                    // later assignment to the same derivative fully replaces an
                    // earlier one instead of leaving stale entries behind.
                    for column in 0..self.states {
                        let derivative = self.differentiate(&assign.value, column)?;
                        out.push(ExecutionStmt {
                            kind: ExecutionStmtKind::Assign(
                                crate::execution::ExecutionAssignStmt {
                                    target: ExecutionTarget {
                                        kind: ExecutionTargetKind::JacobianEntry {
                                            row,
                                            col: column,
                                            states: self.states,
                                        },
                                        span: assign.target.span,
                                    },
                                    value: derivative,
                                },
                            ),
                            span: statement.span,
                        });
                    }
                }
                ExecutionStmtKind::If(if_stmt) => {
                    if expr_reads(&if_stmt.condition, &|load| {
                        matches!(load, ExecutionLoad::State(_))
                    }) {
                        return Err(JacobianDecline::new(
                            "a branch condition depends on a state, making the dynamics piecewise",
                            statement.span,
                        ));
                    }
                    let then_branch = self.differentiate_block(&if_stmt.then_branch)?;
                    let else_branch = match &if_stmt.else_branch {
                        Some(branch) => Some(self.differentiate_block(branch)?),
                        None => None,
                    };
                    out.push(ExecutionStmt {
                        kind: ExecutionStmtKind::If(ExecutionIfStmt {
                            condition: if_stmt.condition.clone(),
                            then_branch,
                            else_branch,
                        }),
                        span: statement.span,
                    });
                }
                ExecutionStmtKind::For(_) => {
                    return Err(JacobianDecline::new(
                        "loops in dynamics are not differentiated symbolically yet",
                        statement.span,
                    ))
                }
            }
        }
        Ok(out)
    }

    fn differentiate(
        &self,
        expr: &ExecutionExpr,
        column: usize,
    ) -> Result<ExecutionExpr, JacobianDecline> {
        let span = expr.span;
        match &expr.kind {
            ExecutionExprKind::Literal(_) => Ok(zero(span)),
            ExecutionExprKind::Load(load) => self.differentiate_load(load, column, span),
            ExecutionExprKind::Unary { op, expr: inner } => {
                let derivative = self.differentiate(inner, column)?;
                match op {
                    AnalyzedUnaryOp::Plus => Ok(derivative),
                    AnalyzedUnaryOp::Minus => Ok(neg(derivative)),
                    AnalyzedUnaryOp::Not => Err(JacobianDecline::new(
                        "boolean negation cannot appear in a differentiated expression",
                        span,
                    )),
                }
            }
            ExecutionExprKind::Binary { op, lhs, rhs } => {
                self.differentiate_binary(*op, lhs, rhs, column, span)
            }
            ExecutionExprKind::Call { callee, args } => {
                self.differentiate_call(callee, args, column, span)
            }
        }
    }

    fn differentiate_load(
        &self,
        load: &ExecutionLoad,
        column: usize,
        span: Span,
    ) -> Result<ExecutionExpr, JacobianDecline> {
        match load {
            ExecutionLoad::State(state) => {
                let offset = constant_state_offset(state)?;
                Ok(if offset == column {
                    one(span)
                } else {
                    zero(span)
                })
            }
            ExecutionLoad::Local(slot) => Ok(self
                .local_derivatives
                .get(slot)
                .and_then(|columns| columns.get(&column))
                .cloned()
                .unwrap_or_else(|| zero(span))),
            ExecutionLoad::Time
            | ExecutionLoad::Parameter(_)
            | ExecutionLoad::Covariate(_)
            | ExecutionLoad::Derived(_)
            | ExecutionLoad::RouteInput { .. } => Ok(zero(span)),
        }
    }

    fn differentiate_binary(
        &self,
        op: AnalyzedBinaryOp,
        lhs: &ExecutionExpr,
        rhs: &ExecutionExpr,
        column: usize,
        span: Span,
    ) -> Result<ExecutionExpr, JacobianDecline> {
        match op {
            AnalyzedBinaryOp::Add => Ok(add(
                self.differentiate(lhs, column)?,
                self.differentiate(rhs, column)?,
            )),
            AnalyzedBinaryOp::Sub => Ok(sub(
                self.differentiate(lhs, column)?,
                self.differentiate(rhs, column)?,
            )),
            AnalyzedBinaryOp::Mul => {
                let dl = self.differentiate(lhs, column)?;
                let dr = self.differentiate(rhs, column)?;
                Ok(add(mul(dl, rhs.clone()), mul(lhs.clone(), dr)))
            }
            AnalyzedBinaryOp::Div => {
                let dl = self.differentiate(lhs, column)?;
                let dr = self.differentiate(rhs, column)?;
                if is_zero(&dr) {
                    return Ok(div(dl, rhs.clone()));
                }
                let numerator = sub(mul(dl, rhs.clone()), mul(lhs.clone(), dr));
                Ok(div(numerator, mul(rhs.clone(), rhs.clone())))
            }
            AnalyzedBinaryOp::Pow => self.differentiate_pow(lhs, rhs, column, span),
            AnalyzedBinaryOp::Or
            | AnalyzedBinaryOp::And
            | AnalyzedBinaryOp::Eq
            | AnalyzedBinaryOp::NotEq
            | AnalyzedBinaryOp::Lt
            | AnalyzedBinaryOp::LtEq
            | AnalyzedBinaryOp::Gt
            | AnalyzedBinaryOp::GtEq => Err(JacobianDecline::new(
                "a comparison appears where a numeric expression is differentiated",
                span,
            )),
        }
    }

    fn differentiate_pow(
        &self,
        base: &ExecutionExpr,
        exponent: &ExecutionExpr,
        column: usize,
        span: Span,
    ) -> Result<ExecutionExpr, JacobianDecline> {
        let dbase = self.differentiate(base, column)?;
        let dexponent = self.differentiate(exponent, column)?;
        if is_zero(&dbase) && is_zero(&dexponent) {
            return Ok(zero(span));
        }
        if is_zero(&dexponent) {
            // d/dx a^b = b * a^(b-1) * a'
            let reduced = sub(exponent.clone(), one(span));
            return Ok(mul(
                mul(exponent.clone(), call2(MathFunction::Pow, base.clone(), reduced)),
                dbase,
            ));
        }
        // d/dx a^b = a^b * (b' * ln(a) + b * a'/a)
        let power = call2(MathFunction::Pow, base.clone(), exponent.clone());
        let term = add(
            mul(dexponent, call1(MathFunction::Ln, base.clone())),
            mul(exponent.clone(), div(dbase, base.clone())),
        );
        Ok(mul(power, term))
    }

    fn differentiate_call(
        &self,
        callee: &ExecutionCall,
        args: &[ExecutionExpr],
        column: usize,
        span: Span,
    ) -> Result<ExecutionExpr, JacobianDecline> {
        let ExecutionCall::Math(function) = callee;
        let derivatives = args
            .iter()
            .map(|arg| self.differentiate(arg, column))
            .collect::<Result<Vec<_>, _>>()?;
        if derivatives.iter().all(is_zero) {
            return Ok(zero(span));
        }

        let inner = args.first().cloned().ok_or_else(|| {
            JacobianDecline::new("intrinsic call has no arguments to differentiate", span)
        })?;
        let d_inner = derivatives[0].clone();

        let derivative = match function {
            MathFunction::Exp => mul(call1(MathFunction::Exp, inner), d_inner),
            MathFunction::Ln | MathFunction::Log => div(d_inner, inner),
            MathFunction::Log10 => div(
                d_inner,
                mul(inner, literal(std::f64::consts::LN_10, span)),
            ),
            MathFunction::Log2 => div(d_inner, mul(inner, literal(std::f64::consts::LN_2, span))),
            MathFunction::Sqrt => div(
                d_inner,
                mul(literal(2.0, span), call1(MathFunction::Sqrt, inner)),
            ),
            MathFunction::Sin => mul(call1(MathFunction::Cos, inner), d_inner),
            MathFunction::Cos => neg(mul(call1(MathFunction::Sin, inner), d_inner)),
            MathFunction::Tan => {
                let cosine = call1(MathFunction::Cos, inner);
                div(d_inner, mul(cosine.clone(), cosine))
            }
            MathFunction::Pow => {
                let exponent = args.get(1).cloned().ok_or_else(|| {
                    JacobianDecline::new("`pow` requires two arguments", span)
                })?;
                return self.differentiate_pow(&args[0], &exponent, column, span);
            }
            MathFunction::Abs
            | MathFunction::Ceil
            | MathFunction::Floor
            | MathFunction::Round
            | MathFunction::Min
            | MathFunction::Max => {
                return Err(JacobianDecline::new(
                    format!(
                        "`{}` is not differentiable where it depends on a state",
                        function.name()
                    ),
                    span,
                ))
            }
        };
        Ok(derivative)
    }
}

fn constant_state_offset(state: &ExecutionStateRef) -> Result<usize, JacobianDecline> {
    match state.index.as_deref() {
        None => Ok(state.base_offset),
        Some(index) => {
            let element = index
                .constant
                .as_ref()
                .and_then(ConstValue::as_i64)
                .filter(|value| *value >= 0)
                .ok_or_else(|| {
                    JacobianDecline::new(
                        "state index is not known at compile time",
                        index.span,
                    )
                })?;
            Ok(state.base_offset + element as usize)
        }
    }
}

// ─────────────────────────── folding constructors ───────────────────────────
//
// Every node is built through these so that the `0` and `1` factors that
// dominate a symbolic derivative collapse immediately. Without folding the
// derivative of even a small dynamics block grows past anything worth
// compiling.

fn literal(value: f64, span: Span) -> ExecutionExpr {
    ExecutionExpr {
        kind: ExecutionExprKind::Literal(ConstValue::Real(value)),
        ty: ValueType::Real,
        constant: Some(ConstValue::Real(value)),
        span,
    }
}

fn zero(span: Span) -> ExecutionExpr {
    literal(0.0, span)
}

fn one(span: Span) -> ExecutionExpr {
    literal(1.0, span)
}

fn constant_of(expr: &ExecutionExpr) -> Option<f64> {
    expr.constant.as_ref().and_then(ConstValue::as_f64)
}

fn is_zero(expr: &ExecutionExpr) -> bool {
    constant_of(expr) == Some(0.0)
}

fn is_one(expr: &ExecutionExpr) -> bool {
    constant_of(expr) == Some(1.0)
}

fn add(lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    if is_zero(&lhs) {
        return rhs;
    }
    if is_zero(&rhs) {
        return lhs;
    }
    if let (Some(a), Some(b)) = (constant_of(&lhs), constant_of(&rhs)) {
        return literal(a + b, lhs.span.join(rhs.span));
    }
    binary(AnalyzedBinaryOp::Add, lhs, rhs)
}

fn sub(lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    if is_zero(&rhs) {
        return lhs;
    }
    if is_zero(&lhs) {
        return neg(rhs);
    }
    if let (Some(a), Some(b)) = (constant_of(&lhs), constant_of(&rhs)) {
        return literal(a - b, lhs.span.join(rhs.span));
    }
    binary(AnalyzedBinaryOp::Sub, lhs, rhs)
}

fn mul(lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    if is_zero(&lhs) || is_zero(&rhs) {
        return zero(lhs.span.join(rhs.span));
    }
    if is_one(&lhs) {
        return rhs;
    }
    if is_one(&rhs) {
        return lhs;
    }
    if let (Some(a), Some(b)) = (constant_of(&lhs), constant_of(&rhs)) {
        return literal(a * b, lhs.span.join(rhs.span));
    }
    binary(AnalyzedBinaryOp::Mul, lhs, rhs)
}

fn div(lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    if is_zero(&lhs) {
        return zero(lhs.span.join(rhs.span));
    }
    if is_one(&rhs) {
        return lhs;
    }
    binary(AnalyzedBinaryOp::Div, lhs, rhs)
}

fn neg(expr: ExecutionExpr) -> ExecutionExpr {
    if let Some(value) = constant_of(&expr) {
        return literal(-value, expr.span);
    }
    let span = expr.span;
    ExecutionExpr {
        kind: ExecutionExprKind::Unary {
            op: AnalyzedUnaryOp::Minus,
            expr: Box::new(expr),
        },
        ty: ValueType::Real,
        constant: None,
        span,
    }
}

fn binary(op: AnalyzedBinaryOp, lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    let span = lhs.span.join(rhs.span);
    ExecutionExpr {
        kind: ExecutionExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ty: ValueType::Real,
        constant: None,
        span,
    }
}

fn call1(function: MathFunction, arg: ExecutionExpr) -> ExecutionExpr {
    let span = arg.span;
    ExecutionExpr {
        kind: ExecutionExprKind::Call {
            callee: ExecutionCall::Math(function),
            args: vec![arg],
        },
        ty: ValueType::Real,
        constant: None,
        span,
    }
}

fn call2(function: MathFunction, lhs: ExecutionExpr, rhs: ExecutionExpr) -> ExecutionExpr {
    let span = lhs.span.join(rhs.span);
    ExecutionExpr {
        kind: ExecutionExprKind::Call {
            callee: ExecutionCall::Math(function),
            args: vec![lhs, rhs],
        },
        ty: ValueType::Real,
        constant: None,
        span,
    }
}
