/// Load an ODE IR file and return an IrEquation and its Meta.
pub fn load_ir_ode<P: Into<PathBuf>>(path: P) -> Result<(IrEquation, Meta), std::io::Error> {
    let eq = IrEquation::from_ir_file(path.into())?;
    let meta = eq.model.meta();
    Ok((eq, meta))
}
use std::collections::HashMap;
use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::PathBuf;

use meval::Expr;
use serde::Deserialize;

use crate::simulator::equation::Meta;
use crate::simulator::{Covariates, Infusion, Event, Observation, PharmsolError, Subject};
use crate::simulator::{T as SimT, V as SimV};

#[derive(Debug, Deserialize)]
struct StateDef {
    name: String,
    init: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct IrFile {
    ir_version: String,
    kind: String,
    params: Vec<String>,
    // rhs may be provided as an explicit JSON array of expressions
    rhs: Option<Vec<String>>,
    // optional output expressions (out equations)
    outputs: Option<Vec<String>>,
    // states may be provided with initial values
    states: Option<Vec<StateDef>>,
    // fallback: raw model text with one expression per line
    model_text: Option<String>,
}

/// A minimal interpreter-backed model representation.
///
/// This struct intentionally focuses on expression-based ODE RHS evaluation.
/// It is NOT yet a full `Equation` implementation; rather it provides a small
/// runtime that can parse IR JSON produced by `exa::build::emit_ir` and evaluate
/// RHS expressions using a simple expression evaluator.
pub struct IrModel {
    pub params: Vec<String>,
    pub rhs_exprs: Vec<String>,
    // Pre-bound evaluators: each takes a slice of variable values and returns f64
    evaluators: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    // Scratch buffer reused for variable values: [t, params..., x1, x2, ...]
    scratch: RefCell<Vec<f64>>,
    // Optional initial state values parsed from IR (length == nstates)
    initial_states: Vec<f64>,
    // Optional output evaluators compiled from IR 'outputs'
    output_evaluators: Option<Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>>,
}

impl IrModel {
    /// Load IR from a file produced by `emit_ir`.
    pub fn load_from_file(path: PathBuf) -> Result<Self, io::Error> {
        let s = fs::read_to_string(path)?;
        Self::from_json(&s)
    }

    /// Parse IR JSON string and compile RHS expressions.
    pub fn from_json(s: &str) -> Result<Self, io::Error> {
        let ir: IrFile = serde_json::from_str(s)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("json: {}", e)))?;

        // Determine RHS expressions: prefer explicit `rhs` array, otherwise fall back to `model_text` lines.
        let rhs_exprs: Vec<String> = if let Some(rhs) = ir.rhs {
            rhs
        } else if let Some(mt) = ir.model_text {
            mt.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).map(|l| l.to_string()).collect()
        } else {
            Vec::new()
        };

        // Compile expressions with meval and bind them to indexed variable arrays.
        // Variable ordering: ["t", params..., x1, x2, ...]
        let nstates = rhs_exprs.len();
        let mut var_names: Vec<String> = Vec::with_capacity(1 + ir.params.len() + nstates);
        var_names.push("t".to_string());
        for p in &ir.params {
            var_names.push(p.clone());
        }
        for i in 0..nstates {
            var_names.push(format!("x{}", i + 1));
        }

        // Convert var_names to Vec<&str> for binding
        let var_name_refs: Vec<&str> = var_names.iter().map(|s| s.as_str()).collect();

        let mut evaluators: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = Vec::with_capacity(nstates);
        for expr in &rhs_exprs {
            let e = expr
                .parse::<Expr>()
                .map_err(|pe| io::Error::new(io::ErrorKind::InvalidData, format!("parse error: {}", pe)))?;
            // Bind expression to positional variable array
            let f = e
                .bind(var_name_refs.as_slice())
                .map_err(|pe| io::Error::new(io::ErrorKind::InvalidData, format!("bind error: {}", pe)))?;
            evaluators.push(Box::new(f));
        }

        // Compile output expressions if present
        let mut output_evaluators: Option<Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>> = None;
        if let Some(outputs) = ir.outputs {
            let mut outs = Vec::with_capacity(outputs.len());
            for expr in outputs.iter() {
                let e = expr
                    .parse::<Expr>()
                    .map_err(|pe| io::Error::new(io::ErrorKind::InvalidData, format!("output parse error: {}", pe)))?;
                let f = e
                    .bind(var_name_refs.as_slice())
                    .map_err(|pe| io::Error::new(io::ErrorKind::InvalidData, format!("output bind error: {}", pe)))?;
                outs.push(Box::new(f) as Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);
            }
            output_evaluators = Some(outs);
        }

        // Prepare scratch buffer length: 1 + params + nstates
        let scratch_len = 1 + ir.params.len() + nstates;
        let scratch = RefCell::new(vec![0.0f64; scratch_len]);

        // Parse initial states if provided in IR
        let mut initial_states = vec![0.0f64; nstates];
        if let Some(states_def) = ir.states {
            for (i, sdef) in states_def.into_iter().enumerate().take(nstates) {
                if let Some(v) = sdef.init {
                    initial_states[i] = v;
                }
            }
        }

        Ok(IrModel {
            params: ir.params,
            rhs_exprs,
            evaluators,
            scratch,
            initial_states,
            output_evaluators,
        })
    }

    /// Evaluate RHS expressions.
    ///
    /// `t` is current time, `states` is slice of state variables (x0..xn),
    /// `params` is slice of parameter values (in the order declared in IR),
    /// `derivs` is an output slice to be filled with computed derivatives.
    pub fn eval_rhs(&self, t: f64, states: &[f64], params: &[f64], derivs: &mut [f64]) -> Result<(), String> {
        let n = self.evaluators.len();
        if derivs.len() < n {
            return Err("derivs buffer too small".into());
        }


        // Fill scratch: [t, params..., x1, x2, ...] using interior mutability to avoid allocations
        let mut s = self.scratch.borrow_mut();
        let mut idx = 0;
        s[idx] = t;
        idx += 1;
        for i in 0..self.params.len() {
            s[idx] = if i < params.len() { params[i] } else { 0.0 };
            idx += 1;
        }
        // states slice length may be <= n
        for i in 0..n {
            s[idx] = if i < states.len() { states[i] } else { 0.0 };
            idx += 1;
        }

        // Evaluate using bound evaluators which accept the scratch slice
        for i in 0..n {
            derivs[i] = (self.evaluators[i])(&s);
        }

        Ok(())
    }

    /// Evaluate output expressions (if any) and return their values.
    pub fn eval_outputs(&self, t: f64, states: &[f64], params: &[f64]) -> Result<Vec<f64>, String> {
        if self.output_evaluators.is_none() {
            return Ok(Vec::new());
        }
        let outs = self.output_evaluators.as_ref().unwrap();
        // Prepare scratch like eval_rhs
        let mut s = self.scratch.borrow_mut();
        let mut idx = 0;
        s[idx] = t;
        idx += 1;
        for i in 0..self.params.len() {
            s[idx] = if i < params.len() { params[i] } else { 0.0 };
            idx += 1;
        }
        for i in 0..outs.len().max(self.evaluators.len()) {
            s[idx] = if i < states.len() { states[i] } else { 0.0 };
            idx += 1;
        }

        let mut res = Vec::with_capacity(outs.len());
        for f in outs.iter() {
            res.push(f(&s));
        }
        Ok(res)
    }

    /// Produce a `Meta` object from the IR parameters.
    pub fn meta(&self) -> Meta {
        let refs: Vec<&str> = self.params.iter().map(|s| s.as_str()).collect();
        Meta::new(refs)
    }
}

/// A lightweight adapter that implements the `Equation` trait using an `IrModel` as RHS.
#[derive(Clone)]
pub struct IrEquation {
    model: IrModel,
    nstates: usize,
}

impl IrEquation {
    pub fn from_ir_file(path: PathBuf) -> Result<Self, io::Error> {
        let m = IrModel::load_from_file(path)?;
        let n = m.evaluators.len();
        Ok(Self { model: m, nstates: n })
    }

    pub fn from_ir_json(s: &str) -> Result<Self, io::Error> {
        let m = IrModel::from_json(s)?;
        let n = m.evaluators.len();
        Ok(Self { model: m, nstates: n })
    }
}

// Default lag and fa implementations: return empty maps
fn default_lag(_v: &SimV, _t: SimT, _cov: &Covariates) -> std::collections::HashMap<usize, SimT> {
    std::collections::HashMap::new()
}

fn default_fa(_v: &SimV, _t: SimT, _cov: &Covariates) -> std::collections::HashMap<usize, SimT> {
    std::collections::HashMap::new()
}

use crate::simulator::equation::{Equation, EquationPriv, EquationTypes};
use crate::simulator::{Fa, Lag, Neqs};

impl EquationTypes for IrEquation {
    type S = SimV;
    type P = crate::simulator::likelihood::SubjectPredictions;
}

impl EquationPriv for IrEquation {
    fn lag(&self) -> &Lag {
        &default_lag
    }

    fn fa(&self) -> &Fa {
        &default_fa
    }

    fn get_nstates(&self) -> usize {
        self.nstates
    }

    fn get_nouteqs(&self) -> usize {
        // If the IR defines outputs, use that count; otherwise default to state count
        if let Some(ref outs) = self.model.output_evaluators {
            outs.len()
        } else {
            self.get_nstates()
        }
    }

    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        _covariates: &Covariates,
        _infusions: &Vec<Infusion>,
        _start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError> {
        // Use diffsol OdeBuilder + PMProblem to integrate the expression-based RHS.
        use diffsol::{Bdf, NalgebraContext, OdeBuilder};
        use diffsol::error::OdeSolverError;
        use diffsol::ode_solver::method::OdeSolverMethod;

        // Prepare infusions references vector
        let inf_refs: Vec<&Infusion> = _infusions.iter().collect();

        // Build a closure that adapts the DiffEq signature to the IrModel evaluator
        let func = |x: &SimV, p: &SimV, t: SimT, y: &mut SimV, _bolus: SimV, _rateiv: SimV, _cov: &Covariates| {
            // Use slices directly to avoid allocations and copies
            let x_slice = x.as_slice();
            let p_slice = p.as_slice();
            // Evaluate RHS directly into y's storage to avoid temporary allocations
            let y_slice = y.as_mut_slice();
            // propagate errors as panics inside the closure; diffsol will handle solver errors
            self.model
                .eval_rhs(t, x_slice, p_slice, y_slice)
                .expect("eval_rhs failed");
        };

        // Create PMProblem
        let init_v = state.clone();
        let problem = OdeBuilder::<diffsol::NalgebraMat<SimT>>::new()
            .t0(0.0)
            .h0(1e-3)
            .p(support_point.clone())
            .build_from_eqn(crate::simulator::equation::ode::closure::PMProblem::new(
                func,
                self.get_nstates(),
                support_point.clone(),
                _covariates,
                inf_refs,
                init_v.into(),
            ))?;

        let mut solver = problem.bdf::<diffsol::NalgebraLU<SimT>>()?;

        // integrate until end_time
        match solver.set_stop_time(end_time) {
            Ok(_) => loop {
                let ret = solver.step();
                match ret {
                    Ok(diffsol::ode_solver::OdeSolverStopReason::InternalTimestep) => continue,
                    Ok(diffsol::ode_solver::OdeSolverStopReason::TstopReached) => break,
                    Err(err) => match err {
                        diffsol::error::DiffsolError::OdeSolverError(
                            OdeSolverError::StepSizeTooSmall { time: _ },
                        ) => {
                            return Err(PharmsolError::OtherError(
                                "The step size of the ODE solver went to zero".to_string(),
                            ));
                        }
                        _ => panic!("Unexpected solver error: {:?}", err),
                    },
                    _ => panic!("Unexpected solver return value: {:?}", ret),
                }
            },
            Err(e) => match e {
                diffsol::error::DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime) => {
                    // nothing to do
                }
                _ => panic!("Unexpected solver error: {:?}", e),
            },
        }

        // Copy back final state from solver
        let final_y = solver.state().y;
        for i in 0..self.get_nstates() {
            state[i] = final_y[i];
        }

        Ok(())
    }

    fn nparticles(&self) -> usize {
        1
    }

    fn process_observation(
        &self,
        _support_point: &Vec<f64>,
        _observation: &Observation,
        _error_models: Option<&crate::error::ErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        // Compute outputs either from compiled output expressions or fall back to state mapping
        let state_slice = _x.as_slice();
        let mut outputs: Vec<f64> = Vec::new();
        if self.model.output_evaluators.is_some() {
            // evaluate outputs using support_point as params
            match self.model.eval_outputs(_time, state_slice, _support_point.as_slice()) {
                Ok(o) => outputs = o,
                Err(_) => outputs = state_slice.to_vec(),
            }
        } else {
            outputs = state_slice.to_vec();
        }

        // Determine the output index
        let outeq = _observation.outeq();
        let pred = if outeq < outputs.len() { outputs[outeq] } else { 0.0 };

        let pred_obj = _observation.to_prediction(pred, outputs);
        if let Some(error_models) = _error_models {
            // compute likelihood and push to vector; ignore errors here
            match pred_obj.likelihood(error_models) {
                Ok(l) => _likelihood.push(l),
                Err(_) => (),
            }
        }
        _output.add_prediction(pred_obj);
        Ok(())
    }

    fn initial_state(&self, _support_point: &Vec<f64>, _covariates: &Covariates, _occasion_index: usize) -> Self::S {
        use diffsol::NalgebraContext;
        let mut v = SimV::zeros(self.get_nstates(), NalgebraContext);
        // If IR included initial states, set them
        for i in 0..self.get_nstates() {
            if i < self.model.initial_states.len() {
                v[i] = self.model.initial_states[i];
            }
        }
        v
    }
}

impl Equation for IrEquation {
    fn estimate_likelihood(
        &self,
        _subject: &Subject,
        _support_point: &Vec<f64>,
        _error_models: &crate::error_model::ErrorModels,
        _cache: bool,
    ) -> Result<f64, PharmsolError> {
        // Use the default simulate_subject implementation to produce predictions
        let (preds, _) = self.simulate_subject(_subject, _support_point, Some(_error_models))?;
        // Compute joint likelihood
        preds.likelihood(_error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_ir_eval() {
        let json = r#"{
            "ir_version": "1.0",
            "kind": "ode",
            "params": ["k10", "k12", "k21"],
            "model_text": "-k10*x1 - k12*x1 + k21*x2\n k12*x1 - k21*x2"
        }"#;

        let m = IrModel::from_json(json).expect("load");
        let states = [100.0, 0.0];
        let params = [0.1, 0.05, 0.02];
        let mut d = vec![0.0; 2];
        m.eval_rhs(0.0, &states, &params, &mut d).expect("eval");
        // check signs and rough values
        assert!(d[0] < 0.0);
        assert!(d[1] >= 0.0);
    }

    #[test]
    fn simple_ir_outputs() {
        let json = r#"{
            "ir_version": "1.0",
            "kind": "ode",
            "params": ["k"],
            "model_text": "-k*x1\n 0",
            "outputs": ["x1", "x1 * k"]
        }"#;

        let m = IrModel::from_json(json).expect("load");
        let states = [10.0, 0.0];
        let params = [0.5];
        let outs = m.eval_outputs(0.0, &states, &params).expect("eval outputs");
        // first output is state x1
        assert_eq!(outs[0], 10.0);
        // second output is x1 * k
        assert!((outs[1] - 5.0).abs() < 1e-12);
    }
}
