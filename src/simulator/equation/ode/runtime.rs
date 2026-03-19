use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, Bdf, NalgebraContext,
    NewtonNonlinearSolver, NoLineSearch, OdeBuilder, OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::data::error_model::AssayErrorModels;
use crate::prelude::simulator::SubjectPredictions;
use crate::simulator::{Fa, Lag, Neqs, M, V};
use crate::{Covariates, Infusion, Observation, PharmsolError, Subject};

use super::closure::PMProblem;
use super::{Equation, EquationPriv, EquationTypes};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

fn no_lag(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

fn no_fa(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeOdeModel {
    pub states: Vec<String>,
    pub parameters: Vec<String>,
    pub outputs: Vec<String>,
    pub derivatives: Vec<String>,
    pub output_equations: Vec<String>,
    #[serde(default)]
    pub covariates: Vec<String>,
    #[serde(default)]
    pub init: Vec<String>,
}

impl RuntimeOdeModel {
    pub fn from_json(json: &str) -> Result<Self, PharmsolError> {
        serde_json::from_str(json)
            .map_err(|e| PharmsolError::OtherError(format!("invalid runtime ODE JSON: {e}")))
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeODE {
    lag: Lag,
    fa: Fa,
    neqs: Neqs,
    program: Arc<RuntimeProgram>,
}

impl RuntimeODE {
    pub fn from_model(model: RuntimeOdeModel) -> Result<Self, PharmsolError> {
        let program = RuntimeProgram::compile(model)?;
        Ok(Self {
            lag: no_lag,
            fa: no_fa,
            neqs: (program.nstates(), program.noutputs()),
            program: Arc::new(program),
        })
    }

    pub fn from_json(json: &str) -> Result<Self, PharmsolError> {
        let model = RuntimeOdeModel::from_json(json)?;
        Self::from_model(model)
    }
}

impl EquationTypes for RuntimeODE {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for RuntimeODE {
    #[inline(always)]
    fn lag(&self) -> &Lag {
        &self.lag
    }

    #[inline(always)]
    fn fa(&self) -> &Fa {
        &self.fa
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }

    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError> {
        if start_time == end_time {
            return Ok(());
        }

        let program = self.program.clone();
        let eval_error = Arc::new(RefCell::new(None::<PharmsolError>));
        let error_ref = eval_error.clone();

        let runtime_rhs =
            move |x: &V, p: &V, t: f64, dx: &mut V, bolus: &V, rateiv: &V, cov: &Covariates| {
                if error_ref.borrow().is_some() {
                    dx.fill(0.0);
                    return;
                }

                if let Err(err) = program.eval_derivatives_inplace(x, p, t, bolus, rateiv, cov, dx)
                {
                    *error_ref.borrow_mut() = Some(err);
                    dx.fill(0.0);
                }
            };

        let p_as_v: V = DVector::from_vec(support_point.clone()).into();
        let infusion_refs: Vec<&Infusion> = infusions.iter().collect();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![ATOL])
            .rtol(RTOL)
            .t0(start_time)
            .h0(1e-3)
            .p(support_point.clone())
            .build_from_eqn(PMProblem::with_params_v(
                runtime_rhs,
                self.get_nstates(),
                support_point.clone(),
                p_as_v,
                covariates,
                infusion_refs.as_slice(),
                state.clone().into(),
            ))?;

        let mut solver: Bdf<
            '_,
            PMProblem<_>,
            NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>, NoLineSearch>,
        > = problem.bdf::<diffsol::NalgebraLU<f64>>()?;

        match solver.set_stop_time(end_time) {
            Ok(_) => loop {
                match solver.step() {
                    Ok(OdeSolverStopReason::InternalTimestep) => continue,
                    Ok(OdeSolverStopReason::TstopReached) => break,
                    Ok(OdeSolverStopReason::RootFound(_)) => continue,
                    Err(err) => match err {
                        diffsol::error::DiffsolError::OdeSolverError(
                            OdeSolverError::StepSizeTooSmall { .. },
                        ) => {
                            return Err(PharmsolError::OtherError(
                                "The step size of the ODE solver went to zero. Check whether parameters are too close to zero or infinite.".to_string(),
                            ));
                        }
                        _ => {
                            return Err(PharmsolError::OtherError(format!(
                                "Unexpected solver error: {err}"
                            )));
                        }
                    },
                }
            },
            Err(e) => match e {
                diffsol::error::DiffsolError::OdeSolverError(
                    OdeSolverError::StopTimeAtCurrentTime,
                ) => {}
                _ => {
                    return Err(PharmsolError::OtherError(format!(
                        "Unexpected solver error: {e}"
                    )));
                }
            },
        }

        if let Some(err) = eval_error.borrow_mut().take() {
            return Err(err);
        }

        *state = solver.state().y.clone();
        Ok(())
    }

    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_models: Option<&AssayErrorModels>,
        time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
        let p: V = DVector::from_vec(support_point.clone()).into();
        let zero = V::zeros(self.get_nstates(), NalgebraContext);

        self.program
            .eval_outputs_inplace(x, &p, time, &zero, &zero, covariates, &mut y)?;

        let pred = y[observation.outeq()];
        let pred = observation.to_prediction(pred, x.as_slice().to_vec());
        if let Some(error_models) = error_models {
            likelihood.push(pred.log_likelihood(error_models)?.exp());
        }
        output.add_prediction(pred);
        Ok(())
    }

    fn initial_state(
        &self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S {
        let mut x = V::zeros(self.get_nstates(), NalgebraContext);
        if occasion_index != 0 {
            return x;
        }

        let p: V = DVector::from_vec(support_point.clone()).into();
        let zero = V::zeros(self.get_nstates(), NalgebraContext);
        let _ =
            self.program
                .eval_init_inplace(&x.clone(), &p, 0.0, &zero, &zero, covariates, &mut x);
        x
    }
}

impl Equation for RuntimeODE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = self.estimate_predictions(subject, support_point)?;
        Ok(ypred.log_likelihood(error_models)?.exp())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = self.estimate_predictions(subject, support_point)?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }
}

#[derive(Clone, Debug)]
struct RuntimeProgram {
    state_names: Vec<String>,
    covariate_names: Vec<String>,
    derivative_exprs: Vec<Expr>,
    output_exprs: Vec<Expr>,
    init_exprs: Vec<Expr>,
}

impl RuntimeProgram {
    fn compile(model: RuntimeOdeModel) -> Result<Self, PharmsolError> {
        if model.states.is_empty() {
            return Err(PharmsolError::OtherError(
                "runtime ODE requires at least one state".to_string(),
            ));
        }
        if model.outputs.is_empty() {
            return Err(PharmsolError::OtherError(
                "runtime ODE requires at least one output".to_string(),
            ));
        }
        if model.derivatives.len() != model.states.len() {
            return Err(PharmsolError::OtherError(format!(
                "runtime ODE derivatives count ({}) must match states count ({})",
                model.derivatives.len(),
                model.states.len()
            )));
        }
        if model.output_equations.len() != model.outputs.len() {
            return Err(PharmsolError::OtherError(format!(
                "runtime ODE output equations count ({}) must match outputs count ({})",
                model.output_equations.len(),
                model.outputs.len()
            )));
        }

        let mut init = model.init;
        if init.is_empty() {
            init = vec!["0.0".to_string(); model.states.len()];
        }
        if init.len() != model.states.len() {
            return Err(PharmsolError::OtherError(format!(
                "runtime ODE init count ({}) must match states count ({})",
                init.len(),
                model.states.len()
            )));
        }

        ensure_unique("state", &model.states)?;
        ensure_unique("parameter", &model.parameters)?;
        ensure_unique("covariate", &model.covariates)?;
        ensure_unique("output", &model.outputs)?;

        let mut symbols = HashMap::new();
        for (i, name) in model.states.iter().enumerate() {
            if symbols.insert(name.clone(), VarRef::State(i)).is_some() {
                return Err(PharmsolError::OtherError(format!(
                    "symbol '{name}' is declared multiple times"
                )));
            }
        }
        for (i, name) in model.parameters.iter().enumerate() {
            if symbols.insert(name.clone(), VarRef::Param(i)).is_some() {
                return Err(PharmsolError::OtherError(format!(
                    "symbol '{name}' is declared multiple times"
                )));
            }
        }
        for (i, name) in model.covariates.iter().enumerate() {
            if symbols.insert(name.clone(), VarRef::Cov(i)).is_some() {
                return Err(PharmsolError::OtherError(format!(
                    "symbol '{name}' is declared multiple times"
                )));
            }
        }

        let parser_ctx = ParserContext { symbols };

        let derivative_exprs = model
            .derivatives
            .iter()
            .map(|s| ExprParser::new(s, &parser_ctx).parse())
            .collect::<Result<Vec<_>, _>>()?;

        let output_exprs = model
            .output_equations
            .iter()
            .map(|s| ExprParser::new(s, &parser_ctx).parse())
            .collect::<Result<Vec<_>, _>>()?;

        let init_exprs = init
            .iter()
            .map(|s| ExprParser::new(s, &parser_ctx).parse())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            state_names: model.states,
            covariate_names: model.covariates,
            derivative_exprs,
            output_exprs,
            init_exprs,
        })
    }

    fn nstates(&self) -> usize {
        self.state_names.len()
    }

    fn noutputs(&self) -> usize {
        self.output_exprs.len()
    }

    fn eval_derivatives_inplace(
        &self,
        x: &V,
        p: &V,
        t: f64,
        bolus: &V,
        rateiv: &V,
        covariates: &Covariates,
        out: &mut V,
    ) -> Result<(), PharmsolError> {
        for (i, expr) in self.derivative_exprs.iter().enumerate() {
            out[i] = expr.eval(&EvalContext {
                x,
                p,
                t,
                bolus,
                rateiv,
                covariates,
                covariate_names: &self.covariate_names,
            })?;
        }
        Ok(())
    }

    fn eval_outputs_inplace(
        &self,
        x: &V,
        p: &V,
        t: f64,
        bolus: &V,
        rateiv: &V,
        covariates: &Covariates,
        out: &mut V,
    ) -> Result<(), PharmsolError> {
        for (i, expr) in self.output_exprs.iter().enumerate() {
            out[i] = expr.eval(&EvalContext {
                x,
                p,
                t,
                bolus,
                rateiv,
                covariates,
                covariate_names: &self.covariate_names,
            })?;
        }
        Ok(())
    }

    fn eval_init_inplace(
        &self,
        x: &V,
        p: &V,
        t: f64,
        bolus: &V,
        rateiv: &V,
        covariates: &Covariates,
        out: &mut V,
    ) -> Result<(), PharmsolError> {
        for (i, expr) in self.init_exprs.iter().enumerate() {
            out[i] = expr.eval(&EvalContext {
                x,
                p,
                t,
                bolus,
                rateiv,
                covariates,
                covariate_names: &self.covariate_names,
            })?;
        }
        Ok(())
    }
}

fn ensure_unique(kind: &str, values: &[String]) -> Result<(), PharmsolError> {
    let mut seen = std::collections::HashSet::new();
    for value in values {
        if !seen.insert(value.clone()) {
            return Err(PharmsolError::OtherError(format!(
                "duplicate {kind} name '{value}'"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct EvalContext<'a> {
    x: &'a V,
    p: &'a V,
    t: f64,
    bolus: &'a V,
    rateiv: &'a V,
    covariates: &'a Covariates,
    covariate_names: &'a [String],
}

#[derive(Clone, Copy, Debug)]
enum VarRef {
    Time,
    State(usize),
    Param(usize),
    Cov(usize),
    X(usize),
    P(usize),
    Bolus(usize),
    RateIv(usize),
}

#[derive(Clone, Debug)]
enum Func {
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
    Min,
    Max,
    Pow,
}

#[derive(Clone, Debug)]
enum Expr {
    Number(f64),
    Var(VarRef),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

impl Expr {
    fn eval(&self, ctx: &EvalContext<'_>) -> Result<f64, PharmsolError> {
        match self {
            Self::Number(v) => Ok(*v),
            Self::Var(var) => eval_var(*var, ctx),
            Self::Neg(a) => Ok(-a.eval(ctx)?),
            Self::Add(a, b) => Ok(a.eval(ctx)? + b.eval(ctx)?),
            Self::Sub(a, b) => Ok(a.eval(ctx)? - b.eval(ctx)?),
            Self::Mul(a, b) => Ok(a.eval(ctx)? * b.eval(ctx)?),
            Self::Div(a, b) => Ok(a.eval(ctx)? / b.eval(ctx)?),
            Self::Pow(a, b) => Ok(a.eval(ctx)?.powf(b.eval(ctx)?)),
            Self::Call(func, args) => {
                let evaluated = args
                    .iter()
                    .map(|a| a.eval(ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                match func {
                    Func::Exp => check_arity_and_eval("exp", 1, &evaluated, |a| a[0].exp()),
                    Func::Ln => check_arity_and_eval("ln", 1, &evaluated, |a| a[0].ln()),
                    Func::Log10 => check_arity_and_eval("log10", 1, &evaluated, |a| a[0].log10()),
                    Func::Sqrt => check_arity_and_eval("sqrt", 1, &evaluated, |a| a[0].sqrt()),
                    Func::Abs => check_arity_and_eval("abs", 1, &evaluated, |a| a[0].abs()),
                    Func::Min => check_arity_and_eval("min", 2, &evaluated, |a| a[0].min(a[1])),
                    Func::Max => check_arity_and_eval("max", 2, &evaluated, |a| a[0].max(a[1])),
                    Func::Pow => check_arity_and_eval("pow", 2, &evaluated, |a| a[0].powf(a[1])),
                }
            }
        }
    }
}

fn check_arity_and_eval(
    name: &str,
    expected: usize,
    args: &[f64],
    op: impl Fn(&[f64]) -> f64,
) -> Result<f64, PharmsolError> {
    if args.len() != expected {
        return Err(PharmsolError::OtherError(format!(
            "{name} expects {expected} arguments but got {}",
            args.len()
        )));
    }
    Ok(op(args))
}

fn eval_var(var: VarRef, ctx: &EvalContext<'_>) -> Result<f64, PharmsolError> {
    match var {
        VarRef::Time => Ok(ctx.t),
        VarRef::State(i) | VarRef::X(i) => {
            if i < ctx.x.len() {
                Ok(ctx.x[i])
            } else {
                Err(PharmsolError::OtherError(format!(
                    "state index {i} is out of bounds"
                )))
            }
        }
        VarRef::Param(i) | VarRef::P(i) => {
            if i < ctx.p.len() {
                Ok(ctx.p[i])
            } else {
                Err(PharmsolError::OtherError(format!(
                    "parameter index {i} is out of bounds"
                )))
            }
        }
        VarRef::Bolus(i) => {
            if i < ctx.bolus.len() {
                Ok(ctx.bolus[i])
            } else {
                Err(PharmsolError::OtherError(format!(
                    "bolus index {i} is out of bounds"
                )))
            }
        }
        VarRef::RateIv(i) => {
            if i < ctx.rateiv.len() {
                Ok(ctx.rateiv[i])
            } else {
                Err(PharmsolError::OtherError(format!(
                    "rateiv index {i} is out of bounds"
                )))
            }
        }
        VarRef::Cov(i) => {
            let name = ctx
                .covariate_names
                .get(i)
                .ok_or_else(|| PharmsolError::OtherError(format!("covariate index {i} missing")))?;
            let cov = ctx.covariates.get_covariate(name).ok_or_else(|| {
                PharmsolError::OtherError(format!("covariate '{name}' not found"))
            })?;
            cov.interpolate(ctx.t).map_err(PharmsolError::from)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

#[derive(Clone, Debug)]
struct ParserContext {
    symbols: HashMap<String, VarRef>,
}

struct ExprParser<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    pos: usize,
    ctx: &'a ParserContext,
}

impl<'a> ExprParser<'a> {
    fn new(source: &'a str, ctx: &'a ParserContext) -> Self {
        Self {
            source,
            tokens: vec![],
            pos: 0,
            ctx,
        }
    }

    fn parse(mut self) -> Result<Expr, PharmsolError> {
        self.tokens = tokenize(self.source)?;
        let expr = self.parse_expr(0)?;
        if self.pos != self.tokens.len() {
            return Err(PharmsolError::OtherError(format!(
                "unexpected token in expression '{}'",
                self.source
            )));
        }
        Ok(expr)
    }

    fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, PharmsolError> {
        let mut lhs = self.parse_prefix()?;

        while let Some((prec, right_assoc, op)) = self.current_binary_op() {
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let next_min_prec = if right_assoc { prec } else { prec + 1 };
            let rhs = self.parse_expr(next_min_prec)?;
            lhs = match op {
                Token::Plus => Expr::Add(Box::new(lhs), Box::new(rhs)),
                Token::Minus => Expr::Sub(Box::new(lhs), Box::new(rhs)),
                Token::Star => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                Token::Slash => Expr::Div(Box::new(lhs), Box::new(rhs)),
                Token::Caret => Expr::Pow(Box::new(lhs), Box::new(rhs)),
                _ => unreachable!(),
            };
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, PharmsolError> {
        let token = self
            .tokens
            .get(self.pos)
            .ok_or_else(|| PharmsolError::OtherError("unexpected end of expression".to_string()))?
            .clone();
        self.pos += 1;

        match token {
            Token::Number(v) => Ok(Expr::Number(v)),
            Token::Minus => Ok(Expr::Neg(Box::new(self.parse_expr(4)?))),
            Token::LParen => {
                let expr = self.parse_expr(0)?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::Ident(name) => self.parse_ident(name),
            _ => Err(PharmsolError::OtherError(format!(
                "unexpected token in expression '{}'",
                self.source
            ))),
        }
    }

    fn parse_ident(&mut self, name: String) -> Result<Expr, PharmsolError> {
        if matches!(self.peek(), Some(Token::LParen)) {
            self.pos += 1;
            let args = self.parse_arg_list()?;
            let func = match name.as_str() {
                "exp" => Func::Exp,
                "ln" => Func::Ln,
                "log10" => Func::Log10,
                "sqrt" => Func::Sqrt,
                "abs" => Func::Abs,
                "min" => Func::Min,
                "max" => Func::Max,
                "pow" => Func::Pow,
                _ => {
                    return Err(PharmsolError::OtherError(format!(
                        "unknown function '{name}' in expression '{}'",
                        self.source
                    )));
                }
            };
            return Ok(Expr::Call(func, args));
        }

        if matches!(self.peek(), Some(Token::LBracket)) {
            self.pos += 1;
            let idx = self.parse_index()?;
            self.expect(Token::RBracket)?;
            let var = match name.as_str() {
                "x" => VarRef::X(idx),
                "p" => VarRef::P(idx),
                "bolus" => VarRef::Bolus(idx),
                "rateiv" => VarRef::RateIv(idx),
                _ => {
                    return Err(PharmsolError::OtherError(format!(
                        "indexed symbol '{name}[..]' is not supported"
                    )));
                }
            };
            return Ok(Expr::Var(var));
        }

        if name == "t" {
            return Ok(Expr::Var(VarRef::Time));
        }

        self.ctx
            .symbols
            .get(&name)
            .copied()
            .map(Expr::Var)
            .ok_or_else(|| {
                PharmsolError::OtherError(format!(
                    "unknown symbol '{name}' in expression '{}'",
                    self.source
                ))
            })
    }

    fn parse_index(&mut self) -> Result<usize, PharmsolError> {
        match self.tokens.get(self.pos) {
            Some(Token::Number(v)) if v.fract() == 0.0 && *v >= 0.0 => {
                self.pos += 1;
                Ok(*v as usize)
            }
            _ => Err(PharmsolError::OtherError(format!(
                "index must be a non-negative integer in expression '{}'",
                self.source
            ))),
        }
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Expr>, PharmsolError> {
        if matches!(self.peek(), Some(Token::RParen)) {
            self.pos += 1;
            return Ok(Vec::new());
        }

        let mut args = Vec::new();
        loop {
            args.push(self.parse_expr(0)?);
            match self.peek() {
                Some(Token::Comma) => {
                    self.pos += 1;
                }
                Some(Token::RParen) => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(PharmsolError::OtherError(format!(
                        "expected ',' or ')' in expression '{}'",
                        self.source
                    )));
                }
            }
        }
        Ok(args)
    }

    fn expect(&mut self, expected: Token) -> Result<(), PharmsolError> {
        let actual = self
            .tokens
            .get(self.pos)
            .ok_or_else(|| PharmsolError::OtherError("unexpected end of expression".to_string()))?;
        if *actual == expected {
            self.pos += 1;
            Ok(())
        } else {
            Err(PharmsolError::OtherError(format!(
                "expected token {:?} but found {:?} in expression '{}'",
                expected, actual, self.source
            )))
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn current_binary_op(&self) -> Option<(u8, bool, Token)> {
        match self.peek()? {
            Token::Plus => Some((1, false, Token::Plus)),
            Token::Minus => Some((1, false, Token::Minus)),
            Token::Star => Some((2, false, Token::Star)),
            Token::Slash => Some((2, false, Token::Slash)),
            Token::Caret => Some((3, true, Token::Caret)),
            _ => None,
        }
    }
}

fn tokenize(source: &str) -> Result<Vec<Token>, PharmsolError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                i += 1;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch.is_ascii_digit() || ch == '.' {
                        i += 1;
                        continue;
                    }
                    if ch == 'e' || ch == 'E' {
                        i += 1;
                        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                            i += 1;
                        }
                        continue;
                    }
                    break;
                }
                let lexeme: String = chars[start..i].iter().collect();
                let value = lexeme
                    .parse::<f64>()
                    .map_err(|_| PharmsolError::OtherError(format!("invalid number '{lexeme}'")))?;
                tokens.push(Token::Number(value));
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                i += 1;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            _ => {
                return Err(PharmsolError::OtherError(format!(
                    "invalid character '{}' in expression '{}'",
                    c, source
                )));
            }
        }
    }

    Ok(tokens)
}

impl fmt::Display for RuntimeOdeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RuntimeOdeModel:")?;
        writeln!(f, "  states: {:?}", self.states)?;
        writeln!(f, "  parameters: {:?}", self.parameters)?;
        writeln!(f, "  outputs: {:?}", self.outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::prelude::{equation, Equation};
    use crate::SubjectBuilderExt;

    use super::*;

    #[test]
    fn runtime_ode_matches_closure_ode_for_basic_one_compartment_model() {
        let model = RuntimeOdeModel {
            states: vec!["A".to_string()],
            parameters: vec!["ke".to_string(), "v".to_string()],
            outputs: vec!["cp".to_string()],
            derivatives: vec!["-ke * A + rateiv[0]".to_string()],
            output_equations: vec!["A / v".to_string()],
            covariates: vec![],
            init: vec![],
        };
        let runtime = RuntimeODE::from_model(model).unwrap();

        let closure_ode = equation::ODE::new(
            |x, p, _t, dx, _b, rateiv, _cov| {
                let ke = p[0];
                dx[0] = -ke * x[0] + rateiv[0];
            },
            |_p, _t, _cov| HashMap::new(),
            |_p, _t, _cov| HashMap::new(),
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                let v = p[1];
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let subject = Subject::builder("id1")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(0.5, 1.0, 0)
            .observation(1.0, 1.0, 0)
            .observation(2.0, 1.0, 0)
            .build();
        let spp = vec![0.8, 200.0];

        let runtime_pred = runtime.estimate_predictions(&subject, &spp).unwrap();
        let closure_pred = closure_ode.estimate_predictions(&subject, &spp).unwrap();

        let lhs = runtime_pred.flat_predictions();
        let rhs = closure_pred.flat_predictions();
        assert_eq!(lhs.len(), rhs.len());

        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() < 2e-3, "runtime={a}, closure={b}");
        }
    }

    #[test]
    fn runtime_ode_supports_covariates() {
        let model = RuntimeOdeModel {
            states: vec!["A".to_string()],
            parameters: vec!["ke".to_string(), "v".to_string()],
            outputs: vec!["cp".to_string()],
            derivatives: vec!["-(ke * wt / 70.0) * A + rateiv[0]".to_string()],
            output_equations: vec!["A / v".to_string()],
            covariates: vec!["wt".to_string()],
            init: vec![],
        };
        let runtime = RuntimeODE::from_model(model).unwrap();

        let subject = Subject::builder("id1")
            .covariate("wt", 0.0, 70.0)
            .covariate("wt", 1.0, 84.0)
            .bolus(0.0, 100.0, 0)
            .observation(0.5, 1.0, 0)
            .observation(1.5, 1.0, 0)
            .build();
        let spp = vec![0.6, 30.0];

        let pred = runtime.estimate_predictions(&subject, &spp).unwrap();
        assert_eq!(pred.flat_predictions().len(), 2);
    }
}
