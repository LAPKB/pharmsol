mod closure;

/// Re-export of the diffsol `OdeEquations` adapter so the JIT module can build
/// `OdeBuilder` problems with closures (rather than plain `fn` pointers).
///
/// This helper is shared by the legacy JIT path and the native
/// runtime wrappers.
#[cfg(feature = "dsl-jit")]
pub(crate) mod closure_helpers {
    pub(crate) use super::closure::PMProblem;
}

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, Parameters, PharmsolError, Subject,
};

use super::parameters_hash;
use crate::simulator::cache::{
    BoundErrorModelCache, PredictionCache, DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE, DEFAULT_CACHE_SIZE,
};
use crate::simulator::equation::Predictions;
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, OdeBuilder,
    OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;
use pharmsol_dsl::ModelKind;
use thiserror::Error;

use super::{
    EqnKind, Equation, EquationPriv, EquationTypes, ModelMetadata, ModelMetadataError, State,
    ValidatedModelMetadata,
};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

/// ODE solver selection.
///
/// Each variant corresponds to a solver family from diffsol.
/// `Sdirk` and `ExplicitRk` take a tableau that determines the specific method.
///
/// ```ignore
/// // Implicit multistep (stiff, default):
/// OdeSolver::Bdf
///
/// // Implicit single-step with a chosen tableau:
/// OdeSolver::Sdirk(SdirkTableau::TrBdf2)
/// OdeSolver::Sdirk(SdirkTableau::Esdirk34)
///
/// // Explicit Runge-Kutta — fastest for non-stiff problems:
/// OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)
/// ```
#[derive(Clone, Debug, Default)]
pub enum OdeSolver {
    /// Backward Differentiation Formulae — implicit multistep, best for stiff problems
    #[default]
    Bdf,
    /// Singly Diagonally Implicit Runge-Kutta
    Sdirk(SdirkTableau),
    /// Explicit Runge-Kutta — no Jacobian needed
    ExplicitRk(ExplicitRkTableau),
}

/// Tableau for [`OdeSolver::Sdirk`].
#[derive(Clone, Debug)]
pub enum SdirkTableau {
    /// TR-BDF2 — good all-rounder for moderately stiff problems
    TrBdf2,
    /// ESDIRK3(4) — higher accuracy for stiff problems
    Esdirk34,
}

/// Tableau for [`OdeSolver::ExplicitRk`].
#[derive(Clone, Debug)]
pub enum ExplicitRkTableau {
    /// Tsitouras 5(4) — fastest for non-stiff problems
    Tsit45,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum OdeMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("ODE declares {declared} state metadata entries but model has {expected} states")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error("ODE declares {declared} route metadata entries but model has {expected} inputs")]
    RouteCountMismatch { expected: usize, declared: usize },
    #[error("ODE declares {declared} output metadata entries but model has {expected} outputs")]
    OutputCountMismatch { expected: usize, declared: usize },
}

#[derive(Clone, Debug)]
pub struct ODE {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
    metadata: Option<ValidatedModelMetadata>,
    cache: Option<PredictionCache>,
    error_model_cache: Option<BoundErrorModelCache>,
}

impl ODE {
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Self {
            diffeq,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            solver: OdeSolver::default(),
            rtol: RTOL,
            atol: ATOL,
            metadata: None,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
            error_model_cache: Some(BoundErrorModelCache::new(
                DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
            )),
        }
    }

    /// Set the number of state variables (ODE compartments).
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.neqs.nstates = nstates;
        self.invalidate_metadata();
        self
    }

    /// Set the number of drug inputs (size of bolus[] and rateiv[]).
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.neqs.ndrugs = ndrugs;
        self.invalidate_metadata();
        self
    }

    /// Set the number of output equations.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.neqs.nout = nout;
        self.invalidate_metadata();
        self
    }

    /// Set the ODE solver algorithm.
    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the relative and absolute tolerances for the ODE solver.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Attach validated handwritten-model metadata to this ODE.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, OdeMetadataError> {
        let metadata = metadata.validate_for(ModelKind::Ode)?;
        validate_metadata_dimensions(&metadata, &self.neqs)?;
        self.metadata = Some(metadata);
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
        Ok(self)
    }

    /// Access the validated metadata attached to this ODE, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.state_index(name)
    }

    fn invalidate_metadata(&mut self) {
        self.metadata = None;
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
    }
}

fn validate_metadata_dimensions(
    metadata: &ValidatedModelMetadata,
    neqs: &Neqs,
) -> Result<(), OdeMetadataError> {
    let declared_states = metadata.states().len();
    if declared_states != neqs.nstates {
        return Err(OdeMetadataError::StateCountMismatch {
            expected: neqs.nstates,
            declared: declared_states,
        });
    }

    let declared_routes = metadata.route_input_count();
    if declared_routes != neqs.ndrugs {
        return Err(OdeMetadataError::RouteCountMismatch {
            expected: neqs.ndrugs,
            declared: declared_routes,
        });
    }

    let declared_outputs = metadata.outputs().len();
    if declared_outputs != neqs.nout {
        return Err(OdeMetadataError::OutputCountMismatch {
            expected: neqs.nout,
            declared: declared_outputs,
        });
    }

    Ok(())
}

impl super::Cache for ODE {
    fn with_cache_capacity(mut self, size: usize) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
        if let Some(cache) = &self.error_model_cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self.error_model_cache = None;
        self
    }
}

impl State for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    parameters: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    let bound_error_models = ode.bind_error_models(error_models)?;
    let ypred = _subject_predictions(ode, subject, parameters)?;
    Ok(ypred.log_likelihood(&bound_error_models)?.exp())
}

#[inline(always)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    parameters: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = &ode.cache {
        let key = (subject.hash(), parameters_hash(parameters));
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = _simulate_subject_dense(ode, subject, parameters, None)?.0;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        Ok(_simulate_subject_dense(ode, subject, parameters, None)?.0)
    }
}

fn _simulate_subject_dense(
    ode: &ODE,
    subject: &Subject,
    parameters: &[f64],
    error_models: Option<&AssayErrorModels>,
) -> Result<(SubjectPredictions, Option<f64>), PharmsolError> {
    let bound_error_models = match error_models {
        Some(error_models) => Some(ode.bind_error_models(error_models)?),
        None => None,
    };
    let bound_error_models = bound_error_models.as_ref().map(|models| &**models);

    let mut output = SubjectPredictions::new(ode.nparticles());

    let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
    let mut likelihood = Vec::with_capacity(event_count);

    let nstates = ode.get_nstates();
    let ndrugs = ode.get_ndrugs();

    let mut state_with_bolus = V::zeros(nstates, NalgebraContext::new());
    let mut state_without_bolus = V::zeros(nstates, NalgebraContext::new());
    let zero_bolus = V::zeros(ndrugs, NalgebraContext::new());
    let zero_rateiv = V::zeros(ndrugs, NalgebraContext::new());
    let mut bolus_v = V::zeros(ndrugs, NalgebraContext::new());
    // Scratch for refreshing the solver's derivative at infusion boundaries.
    let mut dy_scratch = V::zeros(nstates, NalgebraContext::new());
    let parameters_vec = parameters.to_vec();
    let parameters_v: V = DVector::from_vec(parameters_vec.clone()).into();

    let mut y_out = V::zeros(ode.get_nouteqs(), NalgebraContext::new());

    for occasion in subject.occasions() {
        // Run one occasion in a closure so any error can be tagged with the
        // subject and support point in a single place below.
        let occasion_result: Result<(), PharmsolError> = (|| {
            let covariates = occasion.covariates();
            let events = ode.resolve_occasion_events(occasion, parameters, covariates)?;

            let problem = OdeBuilder::<M>::new()
                .atol(vec![ode.atol])
                .rtol(ode.rtol)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(parameters_vec.clone())
                .build_from_eqn(PMProblem::with_params_v(
                    move |x, p, t, dx, bolus, rateiv, cov| {
                        (ode.diffeq)(x, p, t, dx, bolus, rateiv, cov);
                    },
                    nstates,
                    ndrugs,
                    parameters_v.clone(),
                    covariates,
                    events.iter().filter_map(|event| match event {
                        Event::Infusion(infusion) => Some(infusion),
                        _ => None,
                    }),
                    ode.initial_state(parameters, covariates, occasion.index()),
                )?)?;

            match &ode.solver {
                OdeSolver::Bdf => {
                    let mut solver = problem.bdf::<diffsol::NalgebraLU<f64>>()?;
                    ODE::run_events(
                        ode,
                        &mut solver,
                        &events,
                        &parameters_v,
                        covariates,
                        bound_error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut dy_scratch,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => {
                    let mut solver = problem.tsit45()?;
                    ODE::run_events(
                        ode,
                        &mut solver,
                        &events,
                        &parameters_v,
                        covariates,
                        bound_error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut dy_scratch,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::Sdirk(SdirkTableau::TrBdf2) => {
                    let mut solver = problem.tr_bdf2::<diffsol::NalgebraLU<f64>>()?;
                    ODE::run_events(
                        ode,
                        &mut solver,
                        &events,
                        &parameters_v,
                        covariates,
                        bound_error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut dy_scratch,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::Sdirk(SdirkTableau::Esdirk34) => {
                    let mut solver = problem.esdirk34::<diffsol::NalgebraLU<f64>>()?;
                    ODE::run_events(
                        ode,
                        &mut solver,
                        &events,
                        &parameters_v,
                        covariates,
                        bound_error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut dy_scratch,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
            }
            Ok(())
        })();
        occasion_result.map_err(|e| {
            let names = ode
                .metadata()
                .map(|m| m.parameter_names())
                .unwrap_or_default();
            e.with_subject_context(subject.id(), parameters, &names)
        })?;
    }

    let ll = bound_error_models.map(|_| likelihood.iter().product::<f64>());
    Ok((output, ll))
}

impl EquationTypes for ODE {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for ODE {
    //#[inline(always)]
    // fn get_lag(&self, parameters: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let parameters = DVector::from_vec(parameters.to_vec());
    //     Some((self.lag)(&parameters))
    // }

    // #[inline(always)]
    // fn get_fa(&self, parameters: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let parameters = DVector::from_vec(parameters.to_vec());
    //     Some((self.fa)(&parameters))
    // }
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
        self.neqs.nstates
    }

    #[inline(always)]
    fn get_ndrugs(&self) -> usize {
        self.neqs.ndrugs
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.nout
    }

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    #[inline(always)]
    fn solve(
        &self,
        _state: &mut Self::S,
        _parameters: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve not implemented for ODE");
    }
    #[inline(always)]
    fn process_observation(
        &self,
        _parameters: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation not implemented for ODE");
    }

    #[inline(always)]
    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates(), NalgebraContext::new());
        if occasion_index == 0 {
            let parameters = DVector::from_vec(parameters.to_vec());
            (init)(&parameters.into(), 0.0, covariates, &mut x);
        }
        x
    }
}

/// Restart the solver after a state or RHS discontinuity.
///
/// The multi-step history and the internal Jacobian were built for the
/// pre-discontinuity state/RHS, so the first step into the new segment must not reuse
/// them as-is:
/// - `set_state` forces diffsol to recompute its BDF coefficients and
///   reinitialize the internal Jacobian for the current state;
/// - the stored derivative is refreshed against the post-boundary
///   (right-continuous) RHS so a first-order restart predicts with the new
///   dynamics instead of the pre-boundary ones;
/// - `state_mut` marks the state as modified so the next step restarts the
///   multi-step method at first order.
///
/// Shared with the DSL/JIT ODE path ([`crate::dsl::native::NativeOdeModel`]),
/// which must apply the same discontinuity semantics as the closure-based
/// [`ODE`] event loop.
pub(crate) fn reinitialize_at_boundary<'a, F, S>(solver: &mut S, dy_scratch: &mut V)
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state = solver.state_clone();
    solver.set_state(state);

    let t = solver.state().t;
    {
        let y = solver.state().y;
        solver
            .problem()
            .eqn
            .refresh_state_derivative(t, y, dy_scratch);
    }
    let state = solver.state_mut();
    state.dy.copy_from(dy_scratch);
}

/// Whether a requested solver stop is effectively at the current state time.
///
/// diffsol reports `StopTimeAtCurrentTime` not only for a stop exactly at the
/// current time, but also when its internal state time has landed a few ULPs
/// past the requested stop (adaptive steps may end slightly beyond a stop).
/// Dense output grids built with floating-point arithmetic routinely place
/// requested times a few ULPs away from event times (e.g. a `t += dt`
/// accumulation puts a point ~16 ULPs after a bolus at `t = 12`), so accept a
/// stop within a small relative tolerance of the current time instead of
/// erroring. The tolerance stays far below any meaningful time difference:
/// ~64-128 ULPs of the current time, i.e. at most ~1e-13 at `t = 12`.
///
/// Shared with the DSL/JIT ODE path ([`crate::dsl::native::NativeOdeModel`]).
pub(crate) fn stop_time_reached(stop_time: f64, state_t: f64) -> bool {
    let tolerance = f64::EPSILON * state_t.abs().max(1.0) * 64.0;
    (stop_time - state_t).abs() <= tolerance
}

impl ODE {
    /// Generic event-loop runner, parameterized over the concrete solver type.
    #[allow(clippy::too_many_arguments)]
    fn run_events<'a, F, S>(
        &self,
        solver: &mut S,
        events: &[Event],
        parameters_v: &V,
        covariates: &Covariates,
        error_models: Option<&AssayErrorModels>,
        bolus_v: &mut V,
        zero_bolus: &V,
        zero_rateiv: &V,
        state_with_bolus: &mut V,
        state_without_bolus: &mut V,
        dy_scratch: &mut V,
        y_out: &mut V,
        likelihood: &mut Vec<f64>,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    {
        let infusion_boundary_times = solver.problem().eqn.infusion_boundary_times();
        let mut infusion_boundary_cursor = 0usize;
        let mut index = 0usize;
        // Set when the previous event changed the state or the previous stop
        // was an infusion boundary: the solver must be restarted before the
        // first step of the next segment. Deferred until `set_stop_time`
        // succeeds so a stop that is already reached does not trigger a
        // restart for a zero-length segment.
        let mut pending_reinit = false;
        while index < events.len() {
            let event = &events[index];
            let next_event = events.get(index + 1);

            match event {
                Event::Bolus(bolus) => {
                    let input = bolus.input_index().ok_or_else(|| {
                        let available = self
                            .metadata()
                            .map(|m| m.route_labels())
                            .unwrap_or_default();
                        PharmsolError::unknown_input_label(bolus.input(), &available)
                    })?;

                    if input >= bolus_v.len() {
                        return Err(PharmsolError::InputOutOfRange {
                            input,
                            ndrugs: bolus_v.len(),
                        });
                    }
                    bolus_v.fill(0.0);
                    bolus_v[input] = bolus.amount();

                    state_with_bolus.fill(0.0);
                    state_without_bolus.fill(0.0);

                    (self.diffeq)(
                        solver.state().y,
                        parameters_v,
                        event.time(),
                        state_without_bolus,
                        zero_bolus,
                        zero_rateiv,
                        covariates,
                    );

                    (self.diffeq)(
                        solver.state().y,
                        parameters_v,
                        event.time(),
                        state_with_bolus,
                        bolus_v,
                        zero_rateiv,
                        covariates,
                    );

                    state_with_bolus.axpy(-1.0, state_without_bolus, 1.0);
                    solver.state_mut().y.axpy(1.0, state_with_bolus, 1.0);
                    pending_reinit = true;
                }
                Event::Infusion(_) => {
                    // Infusions are handled within the ODE function itself
                }
                Event::Observation(observation) => {
                    y_out.fill(0.0);
                    (self.out)(
                        solver.state().y,
                        parameters_v,
                        observation.time(),
                        covariates,
                        y_out,
                    );
                    let outeq = observation.outeq_index().ok_or_else(|| {
                        let available = self
                            .metadata()
                            .map(|m| m.output_labels())
                            .unwrap_or_default();
                        PharmsolError::unknown_output_label(observation.outeq(), &available)
                    })?;
                    let pred = y_out[outeq];
                    let pred =
                        observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                    if let Some(error_models) = error_models {
                        likelihood.push(pred.log_likelihood(error_models)?.exp());
                    }
                    output.add_prediction(pred);
                }
            }

            // Advance to the next event time if it exists
            if let Some(next_event) = next_event {
                let next_event_time = next_event.time();
                while next_event_time > solver.state().t {
                    while infusion_boundary_cursor < infusion_boundary_times.len()
                        && infusion_boundary_times[infusion_boundary_cursor] <= solver.state().t
                    {
                        infusion_boundary_cursor += 1;
                    }

                    let (stop_time, is_infusion_boundary) = if let Some(stop_time) =
                        infusion_boundary_times.get(infusion_boundary_cursor)
                    {
                        if *stop_time <= next_event_time {
                            infusion_boundary_cursor += 1;
                            (*stop_time, true)
                        } else {
                            (next_event_time, false)
                        }
                    } else {
                        (next_event_time, false)
                    };

                    solver
                        .problem()
                        .eqn
                        .set_left_continuity_time(if is_infusion_boundary {
                            Some(stop_time)
                        } else {
                            None
                        });

                    match solver.set_stop_time(stop_time) {
                        Ok(_) => {
                            if pending_reinit {
                                reinitialize_at_boundary(solver, dy_scratch);
                                pending_reinit = false;
                            }
                            loop {
                                match solver.step() {
                                    Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                    Ok(OdeSolverStopReason::TstopReached) => {
                                        solver.problem().eqn.set_left_continuity_time(None);
                                        if is_infusion_boundary {
                                            pending_reinit = true;
                                        }
                                        break;
                                    }
                                    Ok(OdeSolverStopReason::RootFound(_, _)) => {
                                        return Err(PharmsolError::OtherError(format!(
                                            "solver stopped at an unexpected root at t = {:.4} \
                                             (root finding is not configured)",
                                            stop_time
                                        )));
                                    }
                                    Err(err) => {
                                        return Err(PharmsolError::from_solver_error(
                                            err, stop_time,
                                        ));
                                    }
                                }
                            }
                        }
                        Err(diffsol::error::DiffsolError::OdeSolverError(
                            OdeSolverError::StopTimeAtCurrentTime,
                        )) => {
                            solver.problem().eqn.set_left_continuity_time(None);
                            let state_t = solver.state().t;
                            let stop_reached = stop_time_reached(stop_time, state_t);

                            if stop_reached {
                                if is_infusion_boundary {
                                    pending_reinit = true;
                                }
                                // The requested stop is the current time within
                                // a small relative tolerance. If it is an
                                // infusion boundary before the next subject
                                // event, keep integrating toward the event;
                                // break only when the reached stop is the
                                // event time itself. Breaking early would skip
                                // the remaining interval and leave the solver
                                // state at the boundary when the observation is
                                // evaluated.
                                if stop_time < next_event_time {
                                    continue;
                                }
                                break;
                            }
                            return Err(PharmsolError::from_solver_error(
                                diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StopTimeAtCurrentTime,
                                ),
                                stop_time,
                            ));
                        }
                        Err(err) => {
                            solver.problem().eqn.set_left_continuity_time(None);
                            return Err(PharmsolError::from_solver_error(err, stop_time));
                        }
                    }
                }
            }
            index += 1;
        }
        Ok(())
    }
}

impl Equation for ODE {
    fn bound_error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.error_model_cache.as_ref()
    }

    fn estimate_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, parameters.as_slice(), error_models)
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        _subject_predictions(self, subject, parameters.as_slice())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let ypred = _subject_predictions(self, subject, parameters.as_slice())?;
        ypred.log_likelihood(&bound_error_models)
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        _subject_predictions(self, subject, parameters)
    }

    fn estimate_log_likelihood_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let ypred = _subject_predictions(self, subject, parameters)?;
        ypred.log_likelihood(&bound_error_models)
    }

    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        _simulate_subject_dense(self, subject, parameters, error_models)
    }

    fn kind() -> EqnKind {
        EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        _simulate_subject_dense(self, subject, parameters.as_slice(), error_models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fa, lag, Subject, SubjectBuilderExt};
    use approx::assert_relative_eq;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static PREDICTION_CACHE_DIFFEQ_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn simple_ode() -> ODE {
        ODE::new(
            |_x, _p, _t, _dx, _b, _rateiv, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |_x, _p, _t, _cov, _y| {},
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
    }

    fn route_policy_subject() -> Subject {
        Subject::builder("route_policy")
            .bolus(0.0, 100.0, "oral")
            .infusion(0.0, 100.0, "iv", 1.0)
            .observation(1.0, 0.0, "cp")
            .build()
    }

    fn explicit_route_function(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        b: &V,
        rateiv: &V,
        _cov: &Covariates,
    ) {
        dx[0] = b[0] + rateiv[0];
    }

    fn injected_route_function(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _b: &V,
        _rateiv: &V,
        _cov: &Covariates,
    ) {
        dx[0] = 0.0;
    }

    fn zero_lag(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
        std::collections::HashMap::new()
    }

    fn unit_fa(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
        std::collections::HashMap::new()
    }

    fn zero_init(_p: &V, _t: f64, _cov: &Covariates, _x: &mut V) {}

    fn state_output(x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V) {
        y[0] = x[0];
    }

    fn infusion_rate_function(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _b: &V,
        rateiv: &V,
        _cov: &Covariates,
    ) {
        dx[0] = rateiv[0];
    }

    fn run_infusions(subject: &Subject, solver: OdeSolver) -> Vec<f64> {
        let ode = ODE::new(
            infusion_rate_function,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_solver(solver)
        .with_metadata(
            super::super::metadata::new("infusion_rate_ode")
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata should validate");

        let predictions = ode
            .simulate_subject(subject, &crate::parameters::dense([]), None)
            .expect("infusion simulation should succeed")
            .0;

        predictions
            .predictions()
            .iter()
            .map(|p| p.prediction())
            .collect()
    }

    fn counting_function(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _b: &V,
        _rateiv: &V,
        _cov: &Covariates,
    ) {
        PREDICTION_CACHE_DIFFEQ_CALLS.fetch_add(1, Ordering::SeqCst);
        dx[0] = 0.0;
    }

    #[test]
    fn handwritten_ode_metadata_exposes_name_lookup() {
        let ode = simple_ode()
            .with_metadata(
                super::super::metadata::new("bimodal_ke")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::infusion("iv").to_state("central")),
            )
            .expect("metadata attachment should validate");
        let metadata = ode.metadata().expect("metadata exists");

        assert_eq!(ode.parameter_index("ke"), Some(0));
        assert_eq!(ode.parameter_index("v"), Some(1));
        assert_eq!(ode.state_index("central"), Some(0));
        assert!(metadata.route("iv").is_some());
        assert!(metadata.output("cp").is_some());
        assert_eq!(metadata.kind(), ModelKind::Ode);
    }

    #[test]
    fn handwritten_ode_without_metadata_keeps_raw_path() {
        let ode = simple_ode();

        assert!(ode.metadata().is_none());
        assert_eq!(ode.state_index("central"), None);
    }

    #[test]
    fn handwritten_ode_rejects_dimension_mismatches() {
        let error = simple_ode()
            .with_metadata(
                super::super::metadata::new("wrong_outputs")
                    .parameters(["ke"])
                    .states(["central"])
                    .outputs(["cp", "auc"])
                    .route(super::super::Route::infusion("iv").to_state("central")),
            )
            .expect_err("output-count mismatches must fail");

        assert_eq!(
            error,
            OdeMetadataError::OutputCountMismatch {
                expected: 1,
                declared: 2,
            }
        );
    }

    #[test]
    fn handwritten_ode_rejects_invalid_metadata() {
        let error = simple_ode()
            .with_metadata(
                super::super::metadata::new("missing_destination")
                    .parameters(["ke"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::infusion("iv")),
            )
            .expect_err("invalid metadata must fail during attachment");

        assert_eq!(
            error,
            OdeMetadataError::Validation(ModelMetadataError::MissingRouteDestination {
                route: "iv".to_string(),
            })
        );
    }

    #[test]
    fn handwritten_ode_defaults_to_explicit_route_vectors() {
        let ode = ODE::new(
            explicit_route_function,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            super::super::metadata::new("explicit_routes")
                .states(["central"])
                .outputs(["cp"])
                .routes([
                    super::super::Route::bolus("oral").to_state("central"),
                    super::super::Route::infusion("iv").to_state("central"),
                ]),
        )
        .expect("metadata attachment should validate");

        let predictions = ode
            .simulate_subject(&route_policy_subject(), &crate::parameters::dense([]), None)
            .expect("simulation should succeed")
            .0;
        let metadata = ode.metadata().expect("metadata exists");

        assert_eq!(
            metadata.route("oral").map(|route| route.input_index()),
            Some(0)
        );
        assert_eq!(
            metadata.route("iv").map(|route| route.input_index()),
            Some(0)
        );
        assert_relative_eq!(
            predictions.predictions()[0].prediction(),
            200.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn handwritten_ode_metadata_input_policy_is_descriptive_only() {
        let ode = ODE::new(
            injected_route_function,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            super::super::metadata::new("injected_routes")
                .states(["central"])
                .outputs(["cp"])
                .routes([
                    super::super::Route::bolus("oral")
                        .to_state("central")
                        .inject_input_to_destination(),
                    super::super::Route::infusion("iv")
                        .to_state("central")
                        .inject_input_to_destination(),
                ]),
        )
        .expect("metadata attachment should validate");

        let predictions = ode
            .simulate_subject(&route_policy_subject(), &crate::parameters::dense([]), None)
            .expect("simulation should succeed")
            .0;

        assert_relative_eq!(
            predictions.predictions()[0].prediction(),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn handwritten_ode_metadata_resolves_raw_numeric_aliases_against_canonical_labels() {
        let ode = ODE::new(
            explicit_route_function,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            super::super::metadata::new("numeric_alias_ode")
                .states(["central"])
                .outputs(["outeq_1"])
                .route(super::super::Route::infusion("input_1").to_state("central")),
        )
        .expect("metadata attachment should validate");

        let canonical = Subject::builder("canonical")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build();
        let aliased = Subject::builder("aliased")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "1")
            .build();

        let canonical_predictions = ode
            .simulate_subject(&canonical, &crate::parameters::dense([]), None)
            .expect("canonical labels should simulate")
            .0;
        let aliased_predictions = ode
            .simulate_subject(&aliased, &crate::parameters::dense([]), None)
            .expect("raw numeric aliases should simulate")
            .0;

        assert_relative_eq!(
            canonical_predictions.predictions()[0].prediction(),
            aliased_predictions.predictions()[0].prediction(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn changing_dimensions_after_metadata_clears_route_metadata() {
        let ode = simple_ode()
            .with_metadata(
                super::super::metadata::new("bimodal_ke")
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::infusion("iv").to_state("central")),
            )
            .expect("metadata attachment should validate")
            .with_ndrugs(2);

        assert!(ode.metadata().is_none());
    }

    #[test]
    fn handwritten_ode_estimate_predictions_uses_prediction_cache() {
        PREDICTION_CACHE_DIFFEQ_CALLS.store(0, Ordering::SeqCst);

        let ode = ODE::new(
            counting_function,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);
        let subject = Subject::builder("cached_predictions")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 0.0, 0)
            .build();

        let first = ode
            .estimate_predictions(&subject, &crate::parameters::dense([]))
            .expect("first prediction run should succeed");
        let first_calls = PREDICTION_CACHE_DIFFEQ_CALLS.load(Ordering::SeqCst);
        assert!(first_calls > 0);

        let second = ode
            .estimate_predictions(&subject, &crate::parameters::dense([]))
            .expect("second prediction run should succeed");
        let second_calls = PREDICTION_CACHE_DIFFEQ_CALLS.load(Ordering::SeqCst);

        assert_eq!(first.predictions().len(), second.predictions().len());
        assert_eq!(first_calls, second_calls);
    }

    #[test]
    fn ode_infusions_short_dose_conserved_after_infusion() {
        let subject = Subject::builder("short_infusion")
            .infusion(0.0, 100.0, "iv", 0.1)
            .observation(0.5, 0.0, "cp")
            .build();

        let predictions = run_infusions(&subject, OdeSolver::Bdf);

        assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusions_observation_at_infusion_end_uses_active_left_rate() {
        let subject = Subject::builder("infusion_end_observation")
            .infusion(0.0, 100.0, "iv", 0.1)
            .observation(0.1, 0.0, "cp")
            .observation(0.5, 0.0, "cp")
            .build();

        let predictions = run_infusions(&subject, OdeSolver::Bdf);

        assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
        assert_relative_eq!(predictions[1], 100.0, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusions_dose_conservation_with_multiple_solvers() {
        // Genuinely short infusion (0.01 h), representative of the reported
        // failures; the dose must be fully conserved across all solver types.
        let subject = Subject::builder("short_infusion_multi_solver")
            .infusion(0.0, 100.0, "iv", 0.01)
            .observation(0.01, 0.0, "cp")
            .build();

        let solvers = [
            (OdeSolver::Bdf, "Bdf"),
            (OdeSolver::Sdirk(SdirkTableau::TrBdf2), "TrBdf2"),
            (OdeSolver::Sdirk(SdirkTableau::Esdirk34), "Esdirk34"),
            (OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45), "Tsit45"),
        ];

        for (solver, _label) in solvers {
            let predictions = run_infusions(&subject, solver);
            assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_infusions_delayed_short_dose_with_bdf() {
        let subject = Subject::builder("short_infusion_delayed_start_bdf")
            .observation(0.0, 0.0, "cp")
            .infusion(0.5, 100.0, "iv", 0.01)
            .observation(0.52, 0.0, "cp")
            .build();

        let predictions = run_infusions(&subject, OdeSolver::Bdf);

        assert_relative_eq!(predictions[1], 100.0, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusions_back_to_back_discontinuities_conserve_dose() {
        let subject = Subject::builder("back_to_back_infusions")
            .infusion(0.0, 100.0, "iv", 0.5)
            .infusion(0.5, 100.0, "iv", 0.5)
            .observation(1.0, 0.0, "cp")
            .build();

        let predictions = run_infusions(&subject, OdeSolver::Bdf);

        assert_relative_eq!(predictions[0], 200.0, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusions_accepted_stop_before_event_keeps_integrating() {
        // The infusion ends exactly one ULP after the observation at t = 10.
        // After landing on the observation stop the solver is already within
        // diffsol's round-off of the end boundary, so `set_stop_time` reports
        // `StopTimeAtCurrentTime` and the loop accepts it through the
        // same-ULP check. The reached boundary is *before* the observation at
        // t = 20, so the event loop must keep integrating toward it; breaking
        // early would evaluate the observation with the state frozen at the
        // boundary and miss the exponential decay.
        let subject = Subject::builder("accepted_stop_before_event")
            .infusion(5.0, 100.0, "iv", 10.0_f64.next_up() - 5.0)
            .observation(10.0, 0.0, "cp")
            .observation(20.0, 0.0, "cp")
            .build();

        let ode = ODE::new(
            |x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, rateiv: &V, _cov: &Covariates| {
                dx[0] = rateiv[0] - 0.5 * x[0];
            },
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_solver(OdeSolver::Bdf)
        .with_metadata(
            super::super::metadata::new("accepted_stop_ode")
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata should validate");

        let predictions = ode
            .simulate_subject(&subject, &crate::parameters::dense([]), None)
            .expect("infusion simulation should succeed")
            .0;

        // Dose delivered over [5, 10], then exponential decay for 10 h:
        // (100 * (1 - exp(-0.5 * 5)) / (0.5 * 5)) * exp(-0.5 * 10).
        let delivered = 100.0 * (1.0 - (-2.5_f64).exp()) / 2.5;
        let expected = delivered * (-5.0_f64).exp();
        let pred = predictions.predictions()[1].prediction();
        assert_relative_eq!(pred, expected, max_relative = 1e-3);
    }

    #[test]
    fn dense_observations_a_few_ulps_from_event_time_do_not_error() {
        // Regression test for the 0.28.5 event loop: dense prediction grids
        // built with floating-point accumulation (e.g. a `t += 0.05` grid over
        // 0..24) place requested times a few ULPs away from event times. Here
        // the bolus at t = 12 meets observations ~16 ULPs on either side;
        // `set_stop_time` reports `StopTimeAtCurrentTime` for those stops and
        // the loop must treat them as reached instead of erroring.
        let ulp = 12.0f64.next_up() - 12.0;
        let subject = Subject::builder("dense_grid_near_event")
            .bolus(0.0, 200.0, 0)
            .bolus(12.0, 100.0, 0)
            .missing_observation(0.0, "cp")
            .missing_observation(12.0 - 16.0 * ulp, "cp")
            .missing_observation(12.0 + 16.0 * ulp, "cp")
            .missing_observation(24.0, "cp")
            .build();

        let solvers = [
            (OdeSolver::Bdf, "Bdf"),
            (OdeSolver::Sdirk(SdirkTableau::TrBdf2), "TrBdf2"),
            (OdeSolver::Sdirk(SdirkTableau::Esdirk34), "Esdirk34"),
            (OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45), "Tsit45"),
        ];

        for (solver, label) in solvers {
            let ode = ODE::new(
                |x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, _rateiv: &V, _cov: &Covariates| {
                    dx[0] = -0.3 * x[0];
                },
                zero_lag,
                unit_fa,
                zero_init,
                state_output,
            )
            .with_nstates(1)
            .with_ndrugs(1)
            .with_nout(1)
            .with_solver(solver)
            .with_metadata(
                super::super::metadata::new("dense_grid_near_event_ode")
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::bolus("input_0").to_state("central")),
            )
            .expect("metadata should validate");

            let predictions = ode
                .simulate_subject_dense(&subject, &crate::parameters::dense([]), None)
                .unwrap_or_else(|error| panic!("{label}: dense grid near event failed: {error}"))
                .0;
            assert_eq!(predictions.predictions().len(), 4);
        }
    }

    // debug/script.R hybrid phage model (subject `1` of dat.csv), with the
    // parameters hardcoded. A large infusion rate (1e9 over 1.25e-3 h) makes
    // the first Newton solve after every infusion boundary fail once with
    // diffsol's convergence heuristics and then recover via step-size
    // reduction; over a long horizon those individually recovered failures
    // accumulate in diffsol's absolute `number_of_nonlinear_solver_fails`
    // counter and blow up the run unless the boundary restart avoids them.
    fn hybrid_phage_diffeq(
        x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _b: &V,
        rateiv: &V,
        _cov: &Covariates,
    ) {
        let eps = 1.0e-12_f64;
        let soft = |v: f64| 0.5 * (v + (v * v + eps * eps).sqrt());

        let kep = 20.799022436141968;
        let k12 = 3.611151695251465;
        let k21 = 0.20569434165954592;
        let kdep = 3.674600839614868;
        let kcl_air = 98.17452669143677;
        let kgr = 2.072104573249817;
        let kinf = 1.909232258796692e-6;
        let c50 = 427933.12072753906;
        let klysis = 0.8622971177101135;
        let burst = 1.591451644897461;
        let ksp = 4.387639760971069;
        let kdp = 0.0917521107196808;
        let kn = 1.147785520553589;
        let va = 12.829959392547607;

        let phage_air = soft(x[2]);
        let bacc_pos = soft(x[3]);
        let binf_pos = soft(x[4]);
        let bprot_pos = soft(x[5]);
        let tb = bacc_pos + binf_pos + bprot_pos;
        let bmax = 10_f64.powi(10);
        let cair = phage_air / va;
        let inf_eff = kinf * cair / (1.0 + cair / c50);

        dx[0] = -(kep + k12 + kdep) * x[0] + k21 * x[1] + rateiv[0];
        dx[1] = k12 * x[0] - k21 * x[1];
        dx[2] = kdep * x[0] - kcl_air * x[2] - inf_eff * bacc_pos + burst * klysis * binf_pos;
        dx[3] = kgr * bacc_pos * (1.0 - tb / bmax) - inf_eff * bacc_pos - ksp * x[3] + kdp * x[5]
            - kn * x[3];
        dx[4] = inf_eff * bacc_pos - klysis * x[4];
        dx[5] = ksp * x[3] - kdp * x[5];
    }

    fn hybrid_phage_init(_p: &V, _t: f64, _cov: &Covariates, x: &mut V) {
        x[3] = 3.0 * 10_f64.powf(5.5);
    }

    fn run_hybrid_phage_infusions(subject: &Subject) -> Vec<f64> {
        let ode = ODE::new(
            hybrid_phage_diffeq,
            zero_lag,
            unit_fa,
            hybrid_phage_init,
            state_output,
        )
        .with_nstates(6)
        .with_ndrugs(1)
        .with_nout(1)
        .with_solver(OdeSolver::Bdf)
        .with_metadata(
            super::super::metadata::new("hybrid_phage_infusions")
                .states(["plasma", "peripheral", "airway", "bacc", "binf", "bprot"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("plasma")),
        )
        .expect("metadata should validate");

        let predictions = ode
            .simulate_subject(subject, &crate::parameters::dense([]), None)
            .expect("hybrid phage simulation should succeed")
            .0;

        predictions
            .predictions()
            .iter()
            .map(|p| p.prediction())
            .collect()
    }

    #[test]
    fn ode_infusions_long_horizon_boundary_failures_do_not_accumulate() {
        // The dosing and observation schedule of debug/dat.csv subject `1`
        // (103 short infusions): enough boundaries that, without the boundary
        // restart handling, the individually-recovered Newton failures would
        // accumulate past diffsol's 50-failure limit and abort the run.
        let mut builder = Subject::builder("long_horizon_short_infusions");
        builder = builder.infusion(0.0, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(0.005, "cp");
        builder = builder.missing_observation(0.01791667, "cp");
        builder = builder.missing_observation(0.02208333, "cp");
        builder = builder.missing_observation(0.02208333, "cp");
        builder = builder.missing_observation(0.03458333, "cp");
        builder = builder.missing_observation(0.03833333, "cp");
        builder = builder.missing_observation(0.03833333, "cp");
        builder = builder.missing_observation(0.08458333, "cp");
        builder = builder.missing_observation(0.18625, "cp");
        builder = builder.infusion(0.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(1.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(1.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(2.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(2.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(3.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(3.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(4.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(4.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(5.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(5.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(6.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(6.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(7.0, 1e+09, "iv", 0.00125);
        builder = builder.infusion(7.5, 1e+09, "iv", 0.00125);
        builder = builder.infusion(8.490833, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(8.991667, "cp");
        builder = builder.infusion(8.992917, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(8.995833, "cp");
        builder = builder.missing_observation(9.010417, "cp");
        builder = builder.missing_observation(9.074167, "cp");
        builder = builder.missing_observation(9.166667, "cp");
        builder = builder.infusion(9.492917, 1e+09, "iv", 0.00125);
        builder = builder.infusion(9.992917, 1e+09, "iv", 0.00125);
        builder = builder.infusion(10.49292, 1e+09, "iv", 0.00125);
        builder = builder.infusion(10.99292, 1e+09, "iv", 0.00125);
        builder = builder.infusion(11.49292, 1e+09, "iv", 0.00125);
        builder = builder.infusion(12.01458, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(12.03958, "cp");
        builder = builder.missing_observation(12.03958, "cp");
        builder = builder.missing_observation(13.01375, "cp");
        builder = builder.infusion(13.01542, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(13.01792, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.26875, "cp");
        builder = builder.infusion(14.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(15.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(16.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(17.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(18.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(19.01542, 3e+09, "iv", 0.00125);
        builder = builder.infusion(20.04917, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(21.03333, "cp");
        builder = builder.infusion(21.04917, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(21.055, "cp");
        builder = builder.missing_observation(21.06792, "cp");
        builder = builder.infusion(22.04917, 3e+09, "iv", 0.00125);
        builder = builder.infusion(23.04917, 3e+09, "iv", 0.00125);
        builder = builder.infusion(26.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(27.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(28.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(29.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(30.05125, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(33.05333, "cp");
        builder = builder.infusion(33.06042, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(33.06375, "cp");
        builder = builder.missing_observation(34.0375, "cp");
        builder = builder.missing_observation(34.0375, "cp");
        builder = builder.infusion(34.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(35.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(36.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(37.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(40.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(41.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(42.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(43.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(44.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(47.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(48.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(49.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(50.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(51.06042, 3e+09, "iv", 0.00125);
        builder = builder.infusion(54.03667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(55.03667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(56.17417, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(56.21667, "cp");
        builder = builder.missing_observation(56.21667, "cp");
        builder = builder.infusion(57.17417, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(58.01542, "cp");
        builder = builder.missing_observation(58.01542, "cp");
        builder = builder.infusion(58.17417, 3e+09, "iv", 0.00125);
        builder = builder.infusion(61.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(62.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(63.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(64.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(65.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(68.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(69.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(70.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(71.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(72.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(75.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(76.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(77.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(78.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(79.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(82.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(83.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(84.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(85.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(86.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(89.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(90.17583, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(91.05958, "cp");
        builder = builder.infusion(91.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(92.17583, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(93.07417, "cp");
        builder = builder.missing_observation(93.07417, "cp");
        builder = builder.infusion(93.17583, 3e+09, "iv", 0.00125);
        builder = builder.infusion(96.02792, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(97.02792, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(98.02792, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(99.02792, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(100.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(103.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(104.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(105.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(106.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(107.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(110.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(111.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(112.0271, "cp");
        builder = builder.missing_observation(112.0271, "cp");
        builder = builder.infusion(112.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(112.0742, "cp");
        builder = builder.missing_observation(112.0742, "cp");
        builder = builder.infusion(113.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(114.0279, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(114.1333, "cp");
        builder = builder.missing_observation(114.1333, "cp");
        builder = builder.infusion(117.05, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(118.05, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(119.05, 4.5e+09, "iv", 0.00125);
        let subject = builder.build();

        let predictions = run_hybrid_phage_infusions(&subject);

        // The simulation must complete and produce a finite, positive plasma
        // prediction (the crash is the regression this guards against).
        assert!(predictions[0].is_finite());
        assert!(predictions[0] > 0.0);
    }
}
