//! `JitOde` — an [`Equation`](crate::Equation) implementation backed by
//! Cranelift JIT-compiled native code.
//!
//! This mirrors the structure of the existing [`crate::ODE`] type, but stores
//! a [`JitArtifact`] (the executable model) instead of plain `fn` pointers, so
//! it can host closures that capture the JIT module.

use std::cell::RefCell;
use std::sync::Arc;

use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, OdeBuilder,
    OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    simulator::{
        cache::{PredictionCache, DEFAULT_CACHE_SIZE},
        equation::{
            ode::{ExplicitRkTableau, OdeSolver, SdirkTableau},
            spphash, Cache, EqnKind, Equation, EquationPriv, EquationTypes, Predictions, State,
        },
        likelihood::SubjectPredictions,
        Fa, Lag, M, V,
    },
    Event, Observation, PharmsolError, Subject,
};

use super::codegen::JitArtifact;

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

/// No-op `Lag` function pointer — never produces lag adjustments.
fn no_lag(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
    std::collections::HashMap::new()
}

/// No-op `Fa` function pointer — bioavailability is always 1.
fn no_fa(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
    std::collections::HashMap::new()
}

/// JIT-compiled ODE model.
///
/// Build one with [`crate::jit::Model::compile`].
#[derive(Clone, Debug)]
pub struct JitOde {
    artifact: Arc<JitArtifact>,
    nstates: usize,
    ndrugs: usize,
    nout: usize,
    /// Compartment names in declaration order. The index of a name is also its
    /// `input` index in the data layer (`bolus`/`infusion`).
    compartments: Arc<Vec<String>>,
    /// Output names in declaration order. The index of a name is also its
    /// `outeq` index in the data layer (`observation`).
    outputs: Arc<Vec<String>>,
    /// Covariate names in the order expected by the JIT'd function.
    covariates: Arc<Vec<String>>,
    /// Parameter names — kept for diagnostics and `Meta` consumers.
    params: Arc<Vec<String>>,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
    cache: Option<PredictionCache>,
    lag_fn: Lag,
    fa_fn: Fa,
}

impl JitOde {
    pub(crate) fn new(
        artifact: Arc<JitArtifact>,
        nstates: usize,
        ndrugs: usize,
        nout: usize,
        compartments: Vec<String>,
        outputs: Vec<String>,
        covariates: Vec<String>,
        params: Vec<String>,
    ) -> Self {
        Self {
            artifact,
            nstates,
            ndrugs,
            nout,
            compartments: Arc::new(compartments),
            outputs: Arc::new(outputs),
            covariates: Arc::new(covariates),
            params: Arc::new(params),
            solver: OdeSolver::default(),
            rtol: RTOL,
            atol: ATOL,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
            lag_fn: no_lag as Lag,
            fa_fn: no_fa as Fa,
        }
    }

    /// Compartment names in declaration order. Index `i` is the value to pass
    /// as `input` when constructing a [`crate::Subject`] bolus or infusion at
    /// this compartment.
    pub fn compartments(&self) -> &[String] {
        &self.compartments
    }

    /// Output names in declaration order. Index `i` is the value to pass as
    /// `outeq` for an observation of this output.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Parameter names in the order expected by the support point.
    pub fn params(&self) -> &[String] {
        &self.params
    }

    /// Covariate names referenced by the model.
    pub fn covariates(&self) -> &[String] {
        &self.covariates
    }

    /// Look up the input/state index of a compartment by name.
    ///
    /// Returns `None` if no compartment with that name is declared. Use this
    /// to avoid hard-coding integers when constructing data:
    ///
    /// ```ignore
    /// let central = ode.compartment_index("central").unwrap();
    /// Subject::builder("p").bolus(0.0, 100.0, central).build();
    /// ```
    pub fn compartment_index(&self, name: &str) -> Option<usize> {
        self.compartments.iter().position(|c| c == name)
    }

    /// Look up the `outeq` index of an output by name.
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o == name)
    }

    /// Look up a compartment index by name, panicking with a helpful message
    /// (listing the available names) if the name is not found. Intended for
    /// inline use in test/example data construction.
    #[track_caller]
    pub fn cmt(&self, name: &str) -> usize {
        self.compartment_index(name).unwrap_or_else(|| {
            panic!(
                "unknown compartment {name:?}; declared: [{}]",
                self.compartments.join(", ")
            )
        })
    }

    /// Look up an output index by name, panicking with a helpful message
    /// (listing the available names) if the name is not found. Intended for
    /// inline use in test/example data construction.
    #[track_caller]
    pub fn outeq(&self, name: &str) -> usize {
        self.output_index(name).unwrap_or_else(|| {
            panic!(
                "unknown output {name:?}; declared: [{}]",
                self.outputs.join(", ")
            )
        })
    }

    /// Set the ODE solver algorithm.
    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set absolute and relative tolerances.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }
}

impl Cache for JitOde {
    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self
    }
    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self
    }
    fn clear_cache(&self) {
        if let Some(c) = &self.cache {
            c.invalidate_all();
        }
    }
    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

impl EquationTypes for JitOde {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for JitOde {
    fn lag(&self) -> &Lag {
        &self.lag_fn
    }
    fn fa(&self) -> &Fa {
        &self.fa_fn
    }
    fn get_nstates(&self) -> usize {
        self.nstates
    }
    fn get_ndrugs(&self) -> usize {
        self.ndrugs
    }
    fn get_nouteqs(&self) -> usize {
        self.nout
    }
    fn solve(
        &self,
        _state: &mut Self::S,
        _support_point: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("JitOde::solve is not used; simulate_subject overrides the default")
    }
    fn process_observation(
        &self,
        _support_point: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!(
            "JitOde::process_observation is not used; observations are handled in simulate_subject"
        )
    }
    fn initial_state(&self, support_point: &[f64], covariates: &Covariates, _occasion: usize) -> V {
        let mut x = V::zeros(self.nstates, NalgebraContext);
        if let Some(init_fn) = self.artifact.init {
            // Fill a temporary covariate buffer for t=0.
            let mut cov_buf = vec![0.0_f64; self.covariates.len()];
            fill_cov_buffer(&self.covariates, covariates, 0.0, &mut cov_buf);
            unsafe {
                init_fn(
                    0.0,
                    support_point.as_ptr(),
                    cov_buf.as_ptr(),
                    x.as_mut_slice().as_mut_ptr(),
                );
            }
        }
        x
    }
}

/// Resolve the per-step covariate values into `buf` for the JIT'd function.
///
/// Missing covariates are filled with NaN — this surfaces as a NaN in the
/// computation so the user notices, rather than silently treating it as zero.
#[inline]
fn fill_cov_buffer(names: &[String], cov: &Covariates, t: f64, buf: &mut [f64]) {
    for (i, n) in names.iter().enumerate() {
        buf[i] = match cov.get_covariate(n) {
            Some(c) => c.interpolate(t).unwrap_or(f64::NAN),
            None => f64::NAN,
        };
    }
}

/// Apply JIT-compiled `lag()` and `fa()` to all bolus events in place, then
/// re-sort to maintain pharmsol's event ordering invariant
/// (Observation < Bolus < Infusion at the same time).
fn apply_lag_and_fa(
    events: &mut Vec<Event>,
    artifact: &JitArtifact,
    cov_names: &[String],
    covariates: &Covariates,
    support_point: &[f64],
    ndrugs: usize,
) {
    let mut cov_buf = vec![0.0_f64; cov_names.len()];
    let mut lag_buf = vec![0.0_f64; ndrugs];
    let mut fa_buf = vec![1.0_f64; ndrugs];
    let p_ptr = support_point.as_ptr();

    for ev in events.iter_mut() {
        if let Event::Bolus(bolus) = ev {
            let input = bolus.input();
            if input >= ndrugs {
                continue;
            }
            let t = bolus.time();
            fill_cov_buffer(cov_names, covariates, t, &mut cov_buf);

            if let Some(lag_fn) = artifact.lag {
                unsafe {
                    lag_fn(t, p_ptr, cov_buf.as_ptr(), lag_buf.as_mut_ptr());
                }
                let l = lag_buf[input];
                if l != 0.0 {
                    *bolus.mut_time() += l;
                    // Re-fetch covariates at the shifted time before applying fa.
                    fill_cov_buffer(cov_names, covariates, bolus.time(), &mut cov_buf);
                }
            }
            if let Some(fa_fn) = artifact.fa {
                unsafe {
                    fa_fn(bolus.time(), p_ptr, cov_buf.as_ptr(), fa_buf.as_mut_ptr());
                }
                let f = fa_buf[input];
                if f != 1.0 {
                    bolus.set_amount(bolus.amount() * f);
                }
            }
        }
    }

    events.sort_by(|a, b| {
        #[inline]
        fn order(e: &Event) -> u8 {
            match e {
                Event::Observation(_) => 1,
                Event::Bolus(_) => 2,
                Event::Infusion(_) => 3,
            }
        }
        let t_cmp = a.time().partial_cmp(&b.time());
        match t_cmp {
            Some(std::cmp::Ordering::Equal) => order(a).cmp(&order(b)),
            Some(o) => o,
            None => std::cmp::Ordering::Equal,
        }
    });
}

fn _subject_predictions(
    eqn: &JitOde,
    subject: &Subject,
    support_point: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = &eqn.cache {
        let key = (subject.hash(), spphash(support_point));
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }
        let result = eqn.simulate_subject(subject, support_point, None)?.0;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        Ok(eqn.simulate_subject(subject, support_point, None)?.0)
    }
}

impl Equation for JitOde {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = _subject_predictions(self, subject, support_point)?;
        Ok(ypred.log_likelihood(error_models)?.exp())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = _subject_predictions(self, subject, support_point)?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> EqnKind {
        EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        use crate::simulator::equation::ode::closure_helpers::PMProblem;

        let mut output = Self::P::new(1);

        let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
        let mut likelihood: Vec<f64> = Vec::with_capacity(event_count);

        let nstates = self.nstates;
        let ndrugs = self.ndrugs;

        let mut state_with_bolus = V::zeros(nstates, NalgebraContext);
        let mut state_without_bolus = V::zeros(nstates, NalgebraContext);
        let zero_bolus = V::zeros(ndrugs, NalgebraContext);
        let zero_rateiv = V::zeros(ndrugs, NalgebraContext);
        let mut bolus_v = V::zeros(ndrugs, NalgebraContext);
        let spp_v: V = DVector::from_vec(support_point.to_vec()).into();
        let mut y_out = V::zeros(self.nout, NalgebraContext);

        // Per-call covariate buffer (held inside RefCell so the diffsol closure can
        // mutate it without requiring `Fn` mutability).
        let cov_buf = RefCell::new(vec![0.0_f64; self.covariates.len()]);

        for occasion in subject.occasions() {
            let covariates = occasion.covariates();
            let infusions = occasion.infusions_ref();
            // Validate infusion input channels up-front so the user gets a
            // dedicated error rather than an opaque solver-side panic.
            for inf in infusions.iter() {
                if inf.input() >= ndrugs {
                    return Err(PharmsolError::InputOutOfRange {
                        input: inf.input(),
                        ndrugs,
                    });
                }
            }
            let mut events = occasion.process_events(
                Some((self.fa(), self.lag(), support_point, covariates)),
                true,
            );

            // Apply JIT-compiled lag and fa to bolus events. Pharmsol's own
            // `process_events` only knows about the `Lag`/`Fa` `fn` pointers
            // we stored (no-ops), so we adjust here using the artifact's
            // closure-capable lag/fa functions.
            if self.artifact.lag.is_some() || self.artifact.fa.is_some() {
                apply_lag_and_fa(
                    &mut events,
                    &self.artifact,
                    &self.covariates,
                    covariates,
                    support_point,
                    self.ndrugs,
                );
            }

            // Closure that satisfies `Fn(&V, &V, T, &mut V, &V, &V, &Covariates)`.
            let artifact = Arc::clone(&self.artifact);
            let cov_names = Arc::clone(&self.covariates);
            let cov_buf_ref = &cov_buf;
            let diffeq = move |x: &V,
                               p: &V,
                               t: f64,
                               dx: &mut V,
                               _bolus: &V,
                               rateiv: &V,
                               cov: &Covariates| {
                let mut buf = cov_buf_ref.borrow_mut();
                fill_cov_buffer(&cov_names, cov, t, &mut buf);
                unsafe {
                    (artifact.rhs)(
                        t,
                        x.as_slice().as_ptr(),
                        dx.as_mut_slice().as_mut_ptr(),
                        p.as_slice().as_ptr(),
                        rateiv.as_slice().as_ptr(),
                        buf.as_ptr(),
                    );
                }
            };

            let problem = OdeBuilder::<M>::new()
                .atol(vec![self.atol])
                .rtol(self.rtol)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.to_vec())
                .build_from_eqn(PMProblem::with_params_v(
                    diffeq,
                    nstates,
                    ndrugs,
                    support_point.to_vec(),
                    spp_v.clone(),
                    covariates,
                    infusions.as_slice(),
                    self.initial_state(support_point, covariates, occasion.index()),
                )?)?;

            macro_rules! run {
                ($solver:expr) => {{
                    let mut solver = $solver?;
                    self.run_events(
                        &mut solver,
                        &events,
                        &spp_v,
                        covariates,
                        error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                        &cov_buf,
                    )?;
                }};
            }

            match &self.solver {
                OdeSolver::Bdf => run!(problem.bdf::<diffsol::NalgebraLU<f64>>()),
                OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => run!(problem.tsit45()),
                OdeSolver::Sdirk(SdirkTableau::TrBdf2) => {
                    run!(problem.tr_bdf2::<diffsol::NalgebraLU<f64>>())
                }
                OdeSolver::Sdirk(SdirkTableau::Esdirk34) => {
                    run!(problem.esdirk34::<diffsol::NalgebraLU<f64>>())
                }
            }
        }

        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}

impl JitOde {
    /// Event loop: applies bolus deltas, processes observations, and steps the
    /// solver between events. Mirrors `ODE::run_events`.
    #[allow(clippy::too_many_arguments)]
    fn run_events<'a, F, S>(
        &self,
        solver: &mut S,
        events: &[Event],
        spp_v: &V,
        covariates: &Covariates,
        error_models: Option<&AssayErrorModels>,
        bolus_v: &mut V,
        zero_bolus: &V,
        zero_rateiv: &V,
        state_with_bolus: &mut V,
        state_without_bolus: &mut V,
        y_out: &mut V,
        likelihood: &mut Vec<f64>,
        output: &mut SubjectPredictions,
        cov_buf: &RefCell<Vec<f64>>,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, crate::simulator::equation::ode::closure_helpers::PMProblem<'a, F>>,
    {
        for (index, event) in events.iter().enumerate() {
            let next_event = events.get(index + 1);
            match event {
                Event::Bolus(bolus) => {
                    if bolus.input() >= bolus_v.len() {
                        return Err(PharmsolError::InputOutOfRange {
                            input: bolus.input(),
                            ndrugs: bolus_v.len(),
                        });
                    }
                    bolus_v.fill(0.0);
                    bolus_v[bolus.input()] = bolus.amount();
                    state_with_bolus.fill(0.0);
                    state_without_bolus.fill(0.0);

                    let mut buf = cov_buf.borrow_mut();
                    fill_cov_buffer(&self.covariates, covariates, event.time(), &mut buf);
                    let cov_ptr = buf.as_ptr();

                    unsafe {
                        (self.artifact.rhs)(
                            event.time(),
                            solver.state().y.as_slice().as_ptr(),
                            state_without_bolus.as_mut_slice().as_mut_ptr(),
                            spp_v.as_slice().as_ptr(),
                            zero_rateiv.as_slice().as_ptr(),
                            cov_ptr,
                        );
                        (self.artifact.rhs)(
                            event.time(),
                            solver.state().y.as_slice().as_ptr(),
                            state_with_bolus.as_mut_slice().as_mut_ptr(),
                            spp_v.as_slice().as_ptr(),
                            zero_rateiv.as_slice().as_ptr(),
                            cov_ptr,
                        );
                    }
                    drop(buf);
                    // Difference is the bolus contribution to dx — apply it as a state delta.
                    // For pharmsol's bolus semantics this is equivalent to calling
                    // `state.add_bolus(input, amount)` because the RHS of a JIT model
                    // does not depend on `bolus[]`. We use the same pattern as ODE for
                    // safety in case future expression support adds bolus dependence.
                    state_with_bolus.axpy(-1.0, state_without_bolus, 1.0);
                    // bolus_v contribution: in pharmsol's ODE path the diffeq is called
                    // again with bolus_v passed in. Since our model treats `bolus[i]`
                    // as 0.0, the difference above is zero and we must apply the bolus
                    // directly:
                    solver
                        .state_mut()
                        .y
                        .add_bolus(bolus.input(), bolus.amount());
                }
                Event::Infusion(_) => {}
                Event::Observation(observation) => {
                    y_out.fill(0.0);
                    let mut buf = cov_buf.borrow_mut();
                    fill_cov_buffer(&self.covariates, covariates, observation.time(), &mut buf);
                    unsafe {
                        (self.artifact.out)(
                            observation.time(),
                            solver.state().y.as_slice().as_ptr(),
                            spp_v.as_slice().as_ptr(),
                            buf.as_ptr(),
                            y_out.as_mut_slice().as_mut_ptr(),
                        );
                    }
                    drop(buf);
                    let outeq = observation.outeq();
                    if outeq >= y_out.len() {
                        return Err(PharmsolError::OuteqOutOfRange {
                            outeq,
                            nout: y_out.len(),
                        });
                    }
                    let pred = y_out[outeq];
                    let pred =
                        observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                    if let Some(error_models) = error_models {
                        likelihood.push(pred.log_likelihood(error_models)?.exp());
                    }
                    output.add_prediction(pred);
                }
            }

            if let Some(next_event) = next_event {
                if !event.time().eq(&next_event.time()) {
                    match solver.set_stop_time(next_event.time()) {
                        Ok(_) => loop {
                            match solver.step() {
                                Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                Ok(OdeSolverStopReason::TstopReached) => break,
                                Err(diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StepSizeTooSmall { time },
                                )) => {
                                    return Err(PharmsolError::OtherError(format!(
                                        "ODE solver step size went to zero at t = {time:.4} (target t = {:.4}).",
                                        next_event.time()
                                    )));
                                }
                                Err(_) | Ok(_) => {
                                    return Err(PharmsolError::OtherError(
                                        "Unexpected solver error".to_string(),
                                    ));
                                }
                            }
                        },
                        Err(diffsol::error::DiffsolError::OdeSolverError(
                            OdeSolverError::StopTimeAtCurrentTime,
                        )) => continue,
                        Err(_) => {
                            return Err(PharmsolError::OtherError(
                                "Unexpected solver error".to_string(),
                            ));
                        }
                    }
                }
            }
        }
        let _ = zero_bolus;
        Ok(())
    }
}
