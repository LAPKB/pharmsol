//! Detects models whose dynamics are linear and time-invariant over an
//! integration interval.
//!
//! Such a system, `dx/dt = A x + b` with `A` and `b` constant on the interval,
//! has an exact solution and does not need a numeric integrator. The test here
//! is deliberately conservative: anything it cannot prove constant is reported
//! as a [`LtiDecline`] and the caller keeps its numeric path.
//!
//! `A` is the symbolic state Jacobian from [`crate::differentiate`]. `b` is the
//! dynamics evaluated at `x = 0`, which is why no separate program is needed.

use std::collections::BTreeSet;

use crate::differentiate::{find_load, function_reads};
use crate::execution::{
    ExecutionExpr, ExecutionExprKind, ExecutionLoad, ExecutionModel, ExecutionStmt,
    ExecutionStmtKind, ExecutionTargetKind, FunctionBody, ModelFunction, ModelFunctionKind,
};
use crate::{ModelKind, Span};

/// Why a model cannot be propagated in closed form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtiDecline {
    pub reason: String,
    pub span: Span,
}

impl LtiDecline {
    fn new(reason: impl Into<String>, span: Span) -> Self {
        Self {
            reason: reason.into(),
            span,
        }
    }
}

impl std::fmt::Display for LtiDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// What a model needs in order to be propagated in closed form.
///
/// Linearity and the absence of explicit time dependence are settled at
/// compile time. Covariate interpolation is not: a model declares an
/// expectation, but the subject data decides how values are interpolated, so
/// the covariates whose values feed the coefficients are reported here for the
/// runtime to check against the data it actually has.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LtiRequirements {
    /// Covariates that must be piecewise constant for the coefficients to hold
    /// still between integration boundaries.
    pub piecewise_constant_covariates: Vec<String>,
}

/// Report whether a model's dynamics are linear and time-invariant between
/// integration boundaries.
pub fn classify_linear_time_invariant(
    model: &ExecutionModel,
) -> Result<LtiRequirements, LtiDecline> {
    if model.kind != ModelKind::Ode {
        return Err(LtiDecline::new(
            "closed-form propagation applies to ODE models only",
            model.span,
        ));
    }

    let jacobian = model.function(ModelFunctionKind::Jacobian).ok_or_else(|| {
        match &model.jacobian_decline {
            Some(decline) => LtiDecline::new(decline.reason.clone(), decline.span),
            None => LtiDecline::new("model has no derivable state Jacobian", model.span),
        }
    })?;

    if let Some(span) = find_load(jacobian, &|load| matches!(load, ExecutionLoad::State(_))) {
        return Err(LtiDecline::new(
            "the state Jacobian still depends on the states, so the system is nonlinear",
            span,
        ));
    }

    let dynamics = model
        .function(ModelFunctionKind::Dynamics)
        .ok_or_else(|| LtiDecline::new("model declares no dynamics function", model.span))?;

    // `A` and `b` must both hold still across the interval, so the dynamics get
    // the same treatment as the Jacobian.
    let varying = TimeVarying::analyze(model);
    let mut required = BTreeSet::new();
    for function in [jacobian, dynamics] {
        if let Some(span) = find_load(function, &|load| matches!(load, ExecutionLoad::Time)) {
            return Err(LtiDecline::new(
                "`t` appears in the dynamics, so the system is not time-invariant",
                span,
            ));
        }
        if let Some((reason, span)) = varying.time_dependent_derived(function) {
            return Err(LtiDecline::new(reason, span));
        }
        required.extend(varying.covariates_reached(function));
    }

    Ok(LtiRequirements {
        piecewise_constant_covariates: required.into_iter().collect(),
    })
}

/// Loads whose value can change between integration boundaries, and the derived
/// slots that inherit that variability.
struct TimeVarying {
    /// Derived slots that vary with `t` itself, which no data can make constant.
    time_dependent_derived: BTreeSet<usize>,
    /// `derived slot -> covariates it transitively reads`.
    derived_covariates: Vec<BTreeSet<usize>>,
    covariate_names: Vec<String>,
    derived_names: Vec<String>,
}

impl TimeVarying {
    fn analyze(model: &ExecutionModel) -> Self {
        let derived_count = model.metadata.derived.len();
        let mut analysis = Self {
            time_dependent_derived: BTreeSet::new(),
            derived_covariates: vec![BTreeSet::new(); derived_count],
            covariate_names: model
                .metadata
                .covariates
                .iter()
                .map(|covariate| covariate.name.clone())
                .collect(),
            derived_names: model
                .metadata
                .derived
                .iter()
                .map(|derived| derived.name.clone())
                .collect(),
        };
        if let Some(derive) = model.function(ModelFunctionKind::Derive) {
            if let FunctionBody::Statements(program) = &derive.body {
                let mut locals = LocalDependencies::default();
                analysis.propagate(
                    &program.body.statements,
                    &mut locals,
                    &Dependencies::default(),
                );
            }
        }
        analysis
    }

    /// Thread dependencies through the derive program so a derived value
    /// inherits whatever its inputs depend on, including via branch conditions.
    fn propagate(
        &mut self,
        statements: &[ExecutionStmt],
        locals: &mut LocalDependencies,
        inherited: &Dependencies,
    ) {
        for statement in statements {
            match &statement.kind {
                ExecutionStmtKind::Let(let_stmt) => {
                    let mut dependencies = self.expr_dependencies(&let_stmt.value, locals);
                    dependencies.merge(inherited);
                    locals.slots.insert(let_stmt.local, dependencies);
                }
                ExecutionStmtKind::Assign(assign) => {
                    if let ExecutionTargetKind::Derived(slot) = assign.target.kind {
                        let mut dependencies = self.expr_dependencies(&assign.value, locals);
                        dependencies.merge(inherited);
                        if dependencies.time {
                            self.time_dependent_derived.insert(slot);
                        }
                        if let Some(entry) = self.derived_covariates.get_mut(slot) {
                            entry.extend(dependencies.covariates);
                        }
                    }
                }
                ExecutionStmtKind::If(if_stmt) => {
                    let mut branch = self.expr_dependencies(&if_stmt.condition, locals);
                    branch.merge(inherited);
                    self.propagate(&if_stmt.then_branch, locals, &branch);
                    if let Some(else_branch) = &if_stmt.else_branch {
                        self.propagate(else_branch, locals, &branch);
                    }
                }
                ExecutionStmtKind::For(for_stmt) => {
                    let mut branch = self.expr_dependencies(&for_stmt.range.start, locals);
                    branch.merge(&self.expr_dependencies(&for_stmt.range.end, locals));
                    branch.merge(inherited);
                    self.propagate(&for_stmt.body, locals, &branch);
                }
            }
        }
    }

    fn expr_dependencies(&self, expr: &ExecutionExpr, locals: &LocalDependencies) -> Dependencies {
        match &expr.kind {
            ExecutionExprKind::Literal(_) => Dependencies::default(),
            ExecutionExprKind::Load(load) => self.load_dependencies(load, locals),
            ExecutionExprKind::Unary { expr, .. } => self.expr_dependencies(expr, locals),
            ExecutionExprKind::Binary { lhs, rhs, .. } => {
                let mut dependencies = self.expr_dependencies(lhs, locals);
                dependencies.merge(&self.expr_dependencies(rhs, locals));
                dependencies
            }
            ExecutionExprKind::Call { args, .. } => {
                let mut dependencies = Dependencies::default();
                for arg in args {
                    dependencies.merge(&self.expr_dependencies(arg, locals));
                }
                dependencies
            }
        }
    }

    fn load_dependencies(&self, load: &ExecutionLoad, locals: &LocalDependencies) -> Dependencies {
        let mut dependencies = Dependencies::default();
        match load {
            ExecutionLoad::Time => dependencies.time = true,
            ExecutionLoad::Covariate(index) => {
                dependencies.covariates.insert(*index);
            }
            ExecutionLoad::Derived(index) => {
                if self.time_dependent_derived.contains(index) {
                    dependencies.time = true;
                }
                if let Some(entry) = self.derived_covariates.get(*index) {
                    dependencies.covariates.extend(entry.iter().copied());
                }
            }
            ExecutionLoad::Local(index) => {
                if let Some(entry) = locals.slots.get(index) {
                    dependencies.merge(entry);
                }
            }
            // Route inputs are piecewise constant between infusion boundaries,
            // and states are handled by the linearity check.
            ExecutionLoad::Parameter(_)
            | ExecutionLoad::RouteInput { .. }
            | ExecutionLoad::State(_) => {}
        }
        dependencies
    }

    fn time_dependent_derived(&self, function: &ModelFunction) -> Option<(String, Span)> {
        self.time_dependent_derived.iter().find_map(|index| {
            find_load(function, &|load| {
                matches!(load, ExecutionLoad::Derived(slot) if slot == index)
            })
            .map(|span| {
                let name = self
                    .derived_names
                    .get(*index)
                    .map(String::as_str)
                    .unwrap_or("<derived>");
                (
                    format!(
                        "derived value `{name}` depends on `t`, so the coefficients vary within an interval"
                    ),
                    span,
                )
            })
        })
    }

    /// Covariates whose values reach `function`, directly or through a derived
    /// value.
    fn covariates_reached(&self, function: &ModelFunction) -> BTreeSet<String> {
        let mut reached = BTreeSet::new();
        for (index, name) in self.covariate_names.iter().enumerate() {
            let direct = function_reads(
                function,
                &|load| matches!(load, ExecutionLoad::Covariate(slot) if *slot == index),
            );
            let via_derived = self
                .derived_covariates
                .iter()
                .enumerate()
                .any(|(slot, set)| {
                    set.contains(&index)
                        && function_reads(
                            function,
                            &|load| matches!(load, ExecutionLoad::Derived(other) if *other == slot),
                        )
                });
            if direct || via_derived {
                reached.insert(name.clone());
            }
        }
        reached
    }
}

/// Inputs an expression's value can change with.
#[derive(Debug, Clone, Default)]
struct Dependencies {
    time: bool,
    covariates: BTreeSet<usize>,
}

impl Dependencies {
    fn merge(&mut self, other: &Dependencies) {
        self.time |= other.time;
        self.covariates.extend(other.covariates.iter().copied());
    }
}

#[derive(Debug, Default)]
struct LocalDependencies {
    slots: std::collections::BTreeMap<usize, Dependencies>,
}
