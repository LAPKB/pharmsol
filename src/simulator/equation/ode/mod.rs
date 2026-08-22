mod closure;

/// Re-export of the diffsol `OdeEquations` adapter so the JIT module can build
/// `OdeBuilder` problems with closures (rather than plain `fn` pointers).
///
/// This helper is shared by the legacy JIT path and the native
/// runtime wrappers.
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load"))]
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
    OdeSolverConfig, OdeSolverStopReason, Vector, VectorHost,
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
/// Headroom above diffsol's hard minimum after a close stop, allowing the
/// controller to shrink a restarted step before reaching that floor again.
const MINIMUM_TIMESTEP_HEADROOM: f64 = 4.0;
/// Default dimensionless TSIT45 accepted-step limit for one exact smooth
/// segment. This is a progress-check window, not model time, tolerance, or
/// wall-clock work.
pub(crate) const DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SEGMENT: usize = 500_000;
/// Minimum dimensionless fraction of the original exact segment gap that an
/// accepted-step window must cover before the same segment may continue.
const MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW: f64 = 1e-4;
/// Default cumulative dimensionless TSIT45 accepted-step limit for one
/// `PMProblem`, which corresponds to one subject occasion. The 10,000,000
/// value is supported by successful calibration: standalone and fresh isolated
/// Pmetrics matrices both passed 16/16, standalone script2/TSIT45 took
/// 109.681 s (replay 111.68 s), fresh Pmetrics script2/TSIT45 took 21.079 s,
/// the watchdog took about 3.2 s, and unmodified script8.R completed 100
/// cycles with objective 12034.912411317557 and a generated report. These are
/// calibration evidence, not completion of the remaining audit and acceptance
/// gates.
pub(crate) const DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SESSION: usize = 10_000_000;

/// Accepted-step limits configured once for one `PMProblem`/occasion.
///
/// Both dimensions count accepted solver steps only. Starting another exact
/// event-loop segment resets the progress window but never the session count;
/// coordinate rebases and residual integrations do neither. `None` disables
/// both limits for implicit solver sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExplicitStepBudgetConfig {
    pub(crate) max_accepted_steps_per_segment: usize,
    pub(crate) max_accepted_steps_per_session: usize,
}

/// Return the accepted-step budgets selected for one solver session.
///
/// Only TSIT45 uses the explicit work guard. The implicit methods retain their
/// ordinary diffsol failure semantics and are never limited by either budget.
pub(crate) fn accepted_step_limits_for_solver(
    solver: &OdeSolver,
) -> Option<ExplicitStepBudgetConfig> {
    match solver {
        OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => Some(ExplicitStepBudgetConfig {
            max_accepted_steps_per_segment: DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SEGMENT,
            max_accepted_steps_per_session: DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SESSION,
        }),
        OdeSolver::Bdf
        | OdeSolver::Sdirk(SdirkTableau::TrBdf2)
        | OdeSolver::Sdirk(SdirkTableau::Esdirk34) => None,
    }
}

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

    fn initial_state_at_time(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
        time: f64,
    ) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates(), NalgebraContext::new());
        if occasion_index == 0 {
            let parameters = DVector::from_vec(parameters.to_vec());
            (init)(&parameters.into(), time, covariates, &mut x);
        }
        x
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
    let bound_error_models = bound_error_models.as_deref();

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
    // Scratch for refreshing the solver's derivative at discontinuity boundaries.
    let mut dy_scratch = V::zeros(nstates, NalgebraContext::new());
    let parameters_vec = parameters.to_vec();
    let parameters_v: V = DVector::from_vec(parameters_vec.clone()).into();

    let mut y_out = V::zeros(ode.get_nouteqs(), NalgebraContext::new());

    for occasion in subject.occasions() {
        // Run one occasion in a closure so any error can be tagged with the
        // subject and support point in a single place below.
        let occasion_result: Result<(), PharmsolError> = (|| {
            let covariates = occasion.covariates();
            covariates.validate_for_ode()?;
            let events = ode.resolve_occasion_events(occasion, parameters, covariates)?;
            let time_origin = validate_resolved_ode_schedule(&events)?;

            let problem = OdeBuilder::<M>::new()
                .atol(vec![ode.atol])
                .rtol(ode.rtol)
                .t0(0.0)
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
                    ode.initial_state_at_time(
                        parameters,
                        covariates,
                        occasion.index(),
                        time_origin,
                    ),
                    time_origin,
                )?)?;
            problem
                .eqn
                .configure_explicit_step_guard(accepted_step_limits_for_solver(&ode.solver));

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

fn checked_local_time(
    time_origin: f64,
    absolute_time: f64,
    context: &str,
) -> Result<f64, PharmsolError> {
    let local_time = absolute_time - time_origin;
    let round_trip = time_origin + local_time;
    if !time_origin.is_finite()
        || !absolute_time.is_finite()
        || !local_time.is_finite()
        || !round_trip.is_finite()
        || round_trip != absolute_time
    {
        return Err(PharmsolError::OtherError(format!(
            "{context}: absolute time {absolute_time:?} cannot be represented relative to origin {time_origin:?} (local = {local_time:?}, round trip = {round_trip:?})"
        )));
    }
    Ok(local_time)
}

/// Validate a resolved ODE schedule and return its absolute-time origin.
///
/// Validation happens after route resolution, lag, bioavailability, and event
/// reordering so the solver never has to interpret an invalid schedule. The
/// solver starts at local `t = 0` at the first resolved event. Schedule
/// validation is absolute-time validation; a later time may need a coordinate
/// shift before it is passed to diffsol.
pub(crate) fn validate_resolved_ode_schedule(events: &[Event]) -> Result<f64, PharmsolError> {
    let first_event_time = events.first().map(Event::time).unwrap_or(0.0);
    if !first_event_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "invalid ODE event schedule: first resolved event time {first_event_time:?} is not finite"
        )));
    }
    let time_origin = first_event_time;

    let mut required_times = Vec::with_capacity(events.len() * 2);
    let mut previous_event_time = None;

    for (index, event) in events.iter().enumerate() {
        let time = event.time();
        if !time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "invalid ODE event schedule: resolved event {index} has non-finite time {time:?}"
            )));
        }
        if let Some(previous) = previous_event_time {
            if time < previous {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: resolved event times are not nondecreasing; event {index} at t = {time:.16e} follows t = {previous:.16e}"
                )));
            }
            let gap = time - previous;
            if !gap.is_finite() || (time > previous && gap <= 0.0) {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: gap from t = {previous:.16e} to t = {time:.16e} is not representable"
                )));
            }
        }
        previous_event_time = Some(time);
        required_times.push(time);

        match event {
            Event::Bolus(bolus) => {
                if !bolus.amount().is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: bolus at t = {time:.16e} has non-finite amount {:?}",
                        bolus.amount()
                    )));
                }
            }
            Event::Infusion(infusion) => {
                let amount = infusion.amount();
                if !amount.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite amount {amount:?}"
                    )));
                }

                let duration = infusion.duration();
                if !duration.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite duration {duration:?}"
                    )));
                }
                if duration <= 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} must have positive duration, got {duration:.16e}"
                    )));
                }
                let rate = amount / duration;
                if !rate.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite rate from amount {amount:.16e} and duration {duration:.16e}"
                    )));
                }
                if amount != 0.0 && rate == 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} with amount {amount:.16e} and duration {duration:.16e} has a zero rate, which would silently lose nonzero infusion material"
                    )));
                }
                let endpoint = time + duration;
                if !endpoint.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion endpoint {time:.16e} + {duration:.16e} is not finite"
                    )));
                }
                let endpoint_gap = endpoint - time;
                if !endpoint_gap.is_finite() || endpoint_gap <= 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion endpoint {endpoint:.16e} is not representably after start t = {time:.16e}"
                    )));
                }
                required_times.push(endpoint);
            }
            Event::Observation(_) => {}
        }
    }

    let mut previous_time: Option<f64> = None;
    required_times.sort_by(f64::total_cmp);
    for time in required_times {
        if let Some(previous_time) = previous_time.filter(|previous_time| time > *previous_time) {
            let gap = time - previous_time;
            if !gap.is_finite() || gap <= 0.0 {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: required positive gap from t = {previous_time:.16e} to t = {time:.16e} is not representable"
                )));
            }
        }
        previous_time = Some(time);
    }

    Ok(time_origin)
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
    solver.state_mut().dy.copy_from(dy_scratch);
}

/// Keep an exact-stop alignment step from poisoning the next segment. This
/// changes only the next-step proposal after the current segment succeeded;
/// the accepted state and time remain unchanged.
fn restore_timestep_after_successful_stop<'a, F, S>(solver: &mut S)
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let minimum_timestep = *solver.config().as_base_ref().minimum_timestep;
    let restart_timestep = MINIMUM_TIMESTEP_HEADROOM * minimum_timestep;
    let step = solver.state().h;
    if minimum_timestep > 0.0 && step.abs() < restart_timestep {
        *solver.state_mut().h = restart_timestep.copysign(step);
    }
}

/// Shift diffsol's independent variable without changing the accepted state
/// or either absolute endpoint. The selected origin and both local times must
/// round-trip exactly; this helper never approximates model time. It also
/// discards solver history/Jacobian assumptions and refreshes the RHS at the
/// accepted absolute state.
fn shift_solver_coordinate<'a, F, S>(
    solver: &mut S,
    new_origin: f64,
    target_time: f64,
    dy_scratch: &mut V,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state_time = solver.state().t;
    let old_origin = solver.problem().eqn.time_origin();
    let current_time = old_origin + state_time;
    if !state_time.is_finite()
        || !old_origin.is_finite()
        || !current_time.is_finite()
        || !target_time.is_finite()
    {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift requires finite times: origin = {old_origin:?}, local state = {state_time:?}, current = {current_time:?}, target = {target_time:?}"
        )));
    }
    if target_time <= current_time {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot shift to a non-positive residual interval from t = {current_time:.16e} to t = {target_time:.16e}"
        )));
    }

    let new_state_time = checked_local_time(
        new_origin,
        current_time,
        "ODE solver coordinate-shift current conversion",
    )?;
    let new_target_time = checked_local_time(
        new_origin,
        target_time,
        "ODE solver coordinate-shift target conversion",
    )?;
    let remaining = new_target_time - new_state_time;
    if !remaining.is_finite() || remaining <= 0.0 {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift cannot preserve the positive interval from absolute t = {current_time:.16e} to t = {target_time:.16e} with origin {new_origin:.16e} (local state = {new_state_time:?}, local target = {new_target_time:?})"
        )));
    }

    let current_step = solver.state().h.abs();
    if !current_step.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift cannot retain a non-finite accepted step at absolute t = {current_time:.16e}: h = {:?}",
            solver.state().h
        )));
    }
    let bounded_step = if current_step == 0.0 {
        remaining
    } else {
        current_step.min(remaining)
    };

    solver.problem().eqn.rebase_time_origin(new_origin);
    {
        let state = solver.state_mut();
        *state.t = new_state_time;
        *state.h = bounded_step;
    }
    let state = solver.state_clone();
    solver.set_state(state);

    let y = solver.state().y;
    solver
        .problem()
        .eqn
        .refresh_state_derivative(new_state_time, y, dy_scratch);
    solver.state_mut().dy.copy_from(dy_scratch);
    Ok(())
}

/// Pick a deterministic origin that represents the accepted absolute state
/// and the next absolute stop exactly. Zero is preferred because every finite
/// f64 absolute time round-trips relative to it; the accepted current time is
/// the deterministic fallback and is accepted only when both conversions and
/// the positive local gap are exact.
fn coordinate_shift_origin(current_time: f64, target_time: f64) -> Result<f64, PharmsolError> {
    if !current_time.is_finite() || !target_time.is_finite() || target_time <= current_time {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot choose a coordinate origin for current t = {current_time:?} and target t = {target_time:?}; expected finite target after current time"
        )));
    }

    for new_origin in [0.0, current_time] {
        let Ok(local_current) = checked_local_time(
            new_origin,
            current_time,
            "ODE solver coordinate-shift current conversion",
        ) else {
            continue;
        };
        let Ok(local_target) = checked_local_time(
            new_origin,
            target_time,
            "ODE solver coordinate-shift target conversion",
        ) else {
            continue;
        };
        let local_gap = local_target - local_current;
        if local_gap.is_finite() && local_gap > 0.0 {
            return Ok(new_origin);
        }
    }

    Err(PharmsolError::OtherError(format!(
        "ODE solver cannot preserve exact absolute coordinates for current t = {current_time:.16e} and target t = {target_time:.16e}; tried deterministic origins 0 and the current absolute time, but no finite positive local interval round-tripped exactly. No model-time approximation was attempted"
    )))
}

/// Ensure the accepted current state and an absolute stop can both be passed
/// to diffsol under the active origin. A coordinate shift is performed once
/// when the active origin loses either round trip, and native callback errors
/// are drained immediately after the refresh.
fn ensure_solver_stop_coordinates<'a, F, S, H>(
    solver: &mut S,
    absolute_stop_time: f64,
    dy_scratch: &mut V,
    after_step: &mut H,
) -> Result<f64, PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    let state_time = solver.state().t;
    let time_origin = solver.problem().eqn.time_origin();
    let current_time = time_origin + state_time;
    if !state_time.is_finite()
        || !time_origin.is_finite()
        || !current_time.is_finite()
        || !absolute_stop_time.is_finite()
    {
        return Err(PharmsolError::OtherError(format!(
            "ODE event coordinate normalization requires finite times: origin = {time_origin:?}, local state = {state_time:?}, current = {current_time:?}, target = {absolute_stop_time:?}"
        )));
    }
    if absolute_stop_time < current_time {
        return Err(stop_time_before_current_time(
            absolute_stop_time,
            current_time,
        ));
    }
    if absolute_stop_time == current_time {
        return checked_local_time(
            time_origin,
            absolute_stop_time,
            "ODE event equal-target conversion",
        );
    }

    let active_state_time = checked_local_time(
        time_origin,
        current_time,
        "ODE event current-state conversion",
    );
    let active_stop_time =
        checked_local_time(time_origin, absolute_stop_time, "ODE event stop conversion");
    if let (Ok(active_state_time), Ok(active_stop_time)) = (active_state_time, active_stop_time) {
        let local_gap = active_stop_time - active_state_time;
        if active_state_time == state_time && local_gap.is_finite() && local_gap > 0.0 {
            return Ok(active_stop_time);
        }
    }

    let new_origin = coordinate_shift_origin(current_time, absolute_stop_time)?;
    shift_solver_coordinate::<F, S>(solver, new_origin, absolute_stop_time, dy_scratch)?;
    after_step()?;

    let shifted_stop_time = checked_local_time(
        solver.problem().eqn.time_origin(),
        absolute_stop_time,
        "ODE event shifted stop conversion",
    )?;
    let shifted_state_time = solver.state().t;
    let shifted_gap = shifted_stop_time - shifted_state_time;
    if !shifted_gap.is_finite() || shifted_gap <= 0.0 {
        return Err(PharmsolError::OtherError(format!(
            "ODE event coordinate shift did not preserve a positive local interval: current absolute t = {current_time:.16e}, target = {absolute_stop_time:.16e}, local state = {shifted_state_time:?}, local target = {shifted_stop_time:?}"
        )));
    }
    Ok(shifted_stop_time)
}

/// Rebase a close accepted stop at its absolute current time. Unlike the
/// general coordinate normalization, this intentionally sets local state time
/// to zero and permits one residual integration attempt.
fn rebase_solver_time<'a, F, S>(
    solver: &mut S,
    absolute_time: f64,
    target_time: f64,
    dy_scratch: &mut V,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    shift_solver_coordinate::<F, S>(solver, absolute_time, target_time, dy_scratch)
}

fn ensure_no_material_infusion_is_skipped<'a, F, S>(
    solver: &S,
    start_time: f64,
    stop_time: f64,
    rtol: f64,
    atol: f64,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let skipped_infusion = solver
        .problem()
        .eqn
        .infusion_amount_between(start_time, stop_time);
    if !skipped_infusion.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot safely resolve the infusion boundary from t = {start_time:.16e} to t = {stop_time:.16e}: the scheduled infusion amount is non-finite. Check infusion amounts, durations, and time units"
        )));
    }
    let state_scale = solver
        .state()
        .y
        .as_slice()
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let material_tolerance = atol.abs() + rtol.abs() * state_scale;
    if skipped_infusion > material_tolerance {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot safely resolve the infusion boundary from t = \
             {start_time:.16e} to t = {stop_time:.16e}: advancing without another solver \
             step would skip infusion amount {skipped_infusion:.6e}, above tolerance \
             {material_tolerance:.6e}. No dose was skipped. Check the infusion duration, \
             rate, and model time units"
        )));
    }
    Ok(())
}

fn stop_time_before_current_time(stop_time: f64, state_time: f64) -> PharmsolError {
    PharmsolError::from_solver_error(
        diffsol::error::DiffsolError::OdeSolverError(OdeSolverError::StopTimeBeforeCurrentTime {
            stop_time,
            state_time,
        }),
        stop_time,
    )
}

fn solver_error_at_absolute_time<'a, F, S>(
    solver: &S,
    error: diffsol::error::DiffsolError,
    absolute_target_time: f64,
) -> PharmsolError
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    use diffsol::error::DiffsolError;

    let error = match error {
        DiffsolError::OdeSolverError(error) => {
            let error = match error {
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time,
                    state_time,
                } => OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: solver.problem().eqn.absolute_time(stop_time),
                    state_time: solver.problem().eqn.absolute_time(state_time),
                },
                OdeSolverError::TooManyNonlinearSolverFailures { time, num_failures } => {
                    OdeSolverError::TooManyNonlinearSolverFailures {
                        time: solver.problem().eqn.absolute_time(time),
                        num_failures,
                    }
                }
                OdeSolverError::TooManyErrorTestFailures { time, num_failures } => {
                    OdeSolverError::TooManyErrorTestFailures {
                        time: solver.problem().eqn.absolute_time(time),
                        num_failures,
                    }
                }
                OdeSolverError::StepSizeTooSmall { time } => OdeSolverError::StepSizeTooSmall {
                    time: solver.problem().eqn.absolute_time(time),
                },
                other => other,
            };
            DiffsolError::OdeSolverError(error)
        }
        other => other,
    };
    PharmsolError::from_solver_error(error, absolute_target_time)
}

fn cannot_integrate_distinct_stop(
    start_time: f64,
    stop_time: f64,
    last_accepted_time: f64,
) -> PharmsolError {
    let interval = stop_time - start_time;
    PharmsolError::OtherError(format!(
        "ODE solver cannot integrate distinct stop from t = {start_time:.16e} to \
         t = {stop_time:.16e} (gap {interval:.6e}); the last accepted solver time was \
         t = {last_accepted_time:.16e}. No state dynamics were skipped. Check the event \
         schedule: use identical times only for truly simultaneous records; otherwise \
         rescale the model's time unit so this interval is numerically resolvable"
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReachedStop {
    Exact,
    Interpolated,
    Rebased,
}

/// Reconcile the accepted state after diffsol reports `TstopReached`.
///
/// Diffsol's stop comparison is intentionally tolerant: an accepted state may
/// be a few ULPs short of the requested stop. Such a state must not be
/// relabeled. It is rebased once at its accepted absolute time so the complete
/// residual interval can be integrated. A state past the stop is instead
/// interpolated with diffsol's `state_mut_back` contract.
#[allow(clippy::too_many_arguments)]
fn normalize_reached_stop<'a, F, S>(
    solver: &mut S,
    stop_time: f64,
    absolute_stop_time: f64,
    segment_start_time: f64,
    allow_rebase: bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
) -> Result<ReachedStop, PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state_time = solver.state().t;
    if !stop_time.is_finite() || !absolute_stop_time.is_finite() || !state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop with non-finite time coordinates: state = {state_time:?}, stop = {stop_time:?}, absolute stop = {absolute_stop_time:?}"
        )));
    }
    let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
    if !absolute_state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop whose absolute state time is not finite: local state = {state_time:?}, absolute state = {absolute_state_time:?}"
        )));
    }
    if state_time == stop_time && absolute_state_time == absolute_stop_time {
        return Ok(ReachedStop::Exact);
    }
    if absolute_state_time > absolute_stop_time {
        if let Err(error) = solver.state_mut_back(stop_time) {
            return Err(solver_error_at_absolute_time(
                solver,
                error,
                absolute_stop_time,
            ));
        }
        return Ok(ReachedStop::Interpolated);
    }

    let step = solver.state().h;
    if !step.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop with a non-finite accepted step h = {step:?}"
        )));
    }
    let passed_stop =
        (step >= 0.0 && state_time > stop_time) || (step < 0.0 && state_time < stop_time);
    if passed_stop {
        if let Err(error) = solver.state_mut_back(stop_time) {
            return Err(solver_error_at_absolute_time(
                solver,
                error,
                absolute_stop_time,
            ));
        }
        return Ok(ReachedStop::Interpolated);
    }

    let residual = absolute_stop_time - absolute_state_time;
    if !residual.is_finite() || residual <= 0.0 {
        ensure_no_material_infusion_is_skipped(
            solver,
            absolute_state_time,
            absolute_stop_time,
            rtol,
            atol,
        )?;
        return Err(cannot_integrate_distinct_stop(
            segment_start_time,
            absolute_stop_time,
            absolute_state_time,
        ));
    }
    if !allow_rebase {
        ensure_no_material_infusion_is_skipped(
            solver,
            absolute_state_time,
            absolute_stop_time,
            rtol,
            atol,
        )?;
        return Err(cannot_integrate_distinct_stop(
            segment_start_time,
            absolute_stop_time,
            absolute_state_time,
        ));
    }

    rebase_solver_time(solver, absolute_state_time, absolute_stop_time, dy_scratch)?;
    Ok(ReachedStop::Rebased)
}

/// Advance the solver to `next_event_time`, stopping at every scheduled
/// integration boundary in between.
///
/// This is the single implementation of the event-to-event integration loop
/// shared by the closure-based [`ODE`] equation and the DSL runtime ODE path
/// ([`crate::dsl::native::NativeOdeModel`]): stop selection at integration
/// boundaries, left-continuity handling at covariate and infusion knots, exact
/// restarts for true RHS discontinuities, and safe normalization of stops that
/// diffsol has already reached.
///
/// `after_step` runs after every step; the DSL path uses it to surface
/// model-function errors raised inside the RHS callback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_solver_to_event<'a, F, S, H>(
    solver: &mut S,
    next_event_time: f64,
    integration_boundary_cursor: &mut usize,
    pending_reinit: &mut bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
    after_step: &mut H,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    let initial_state_time = solver.state().t;
    if !next_event_time.is_finite() || !initial_state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance requires finite target and solver state times: target = {next_event_time:?}, state = {initial_state_time:?}"
        )));
    }
    let initial_current_time = solver.problem().eqn.absolute_time(initial_state_time);
    if !initial_current_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance produced a non-finite absolute current time from solver state {initial_state_time:?}"
        )));
    }
    if next_event_time < initial_current_time {
        return Err(stop_time_before_current_time(
            next_event_time,
            initial_current_time,
        ));
    }
    if next_event_time == initial_current_time {
        return Ok(());
    }
    if !(next_event_time - initial_current_time).is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance gap from t = {initial_current_time:.16e} to t = {next_event_time:.16e} is not representable"
        )));
    }

    loop {
        let state_time = solver.state().t;
        if !state_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance encountered a non-finite solver state time {state_time:?}"
            )));
        }
        let current_time = solver.problem().eqn.absolute_time(state_time);
        if !current_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance encountered a non-finite absolute current time from solver state {state_time:?}"
            )));
        }
        if next_event_time < current_time {
            return Err(stop_time_before_current_time(next_event_time, current_time));
        }
        if next_event_time == current_time {
            return Ok(());
        }
        if !(next_event_time - current_time).is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance gap from t = {current_time:.16e} to t = {next_event_time:.16e} is not representable"
            )));
        }

        let integration_boundary_times = solver.problem().eqn.integration_boundary_times();
        while *integration_boundary_cursor < integration_boundary_times.len()
            && integration_boundary_times[*integration_boundary_cursor] <= current_time
        {
            *integration_boundary_cursor += 1;
        }

        let (absolute_stop_time, is_integration_boundary) =
            match integration_boundary_times.get(*integration_boundary_cursor) {
                Some(&boundary_time) if boundary_time <= next_event_time => {
                    *integration_boundary_cursor += 1;
                    (boundary_time, true)
                }
                _ => (next_event_time, false),
            };
        let is_discontinuity_boundary = is_integration_boundary
            && solver
                .problem()
                .eqn
                .is_discontinuity_time(absolute_stop_time);
        if !absolute_stop_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance selected a non-finite stop time {absolute_stop_time:?}"
            )));
        }
        if absolute_stop_time <= current_time {
            return Err(stop_time_before_current_time(
                absolute_stop_time,
                current_time,
            ));
        }
        if !(absolute_stop_time - current_time).is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event segment gap from t = {current_time:.16e} to t = {absolute_stop_time:.16e} is not representable"
            )));
        }

        solver
            .problem()
            .eqn
            .set_left_continuity_time(if is_integration_boundary {
                Some(absolute_stop_time)
            } else {
                None
            });

        let segment_start_time = current_time;
        solver
            .problem()
            .eqn
            .begin_explicit_step_segment(segment_start_time, absolute_stop_time);
        let configured_minimum_timestep = *solver.config().as_base_ref().minimum_timestep;
        let short_segment = absolute_stop_time - segment_start_time
            <= MINIMUM_TIMESTEP_HEADROOM * configured_minimum_timestep;
        if short_segment {
            *solver.config_mut().as_base_mut().minimum_timestep = 0.0;
        }

        let segment_result = (|| -> Result<(), PharmsolError> {
            let mut rebased = false;
            loop {
                let stop_time = ensure_solver_stop_coordinates::<F, S, H>(
                    solver,
                    absolute_stop_time,
                    dy_scratch,
                    after_step,
                )?;
                let state_time = solver.state().t;
                let local_gap = stop_time - state_time;
                if !stop_time.is_finite() || !state_time.is_finite() || !local_gap.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "ODE event coordinate conversion is not finite: current local t = {state_time:?}, absolute stop = {absolute_stop_time:?}, local stop = {stop_time:?}, gap = {local_gap:?}"
                    )));
                }
                match solver.set_stop_time(stop_time) {
                    Ok(()) => {
                        if *pending_reinit {
                            reinitialize_at_boundary(solver, dy_scratch);
                            if let Err(error) = after_step() {
                                solver.problem().eqn.set_left_continuity_time(None);
                                return Err(error);
                            }
                            *pending_reinit = false;
                        }
                        integrate_to_stop(
                            solver,
                            stop_time,
                            absolute_stop_time,
                            segment_start_time,
                            is_discontinuity_boundary,
                            pending_reinit,
                            !rebased,
                            dy_scratch,
                            rtol,
                            atol,
                            after_step,
                        )?;
                        return Ok(());
                    }
                    Err(diffsol::error::DiffsolError::OdeSolverError(
                        OdeSolverError::StopTimeAtCurrentTime,
                    )) => {
                        let state_time = solver.state().t;
                        let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
                        if !state_time.is_finite() || !absolute_state_time.is_finite() {
                            return Err(PharmsolError::OtherError(format!(
                                "ODE event advance encountered a non-finite state after set_stop_time: local = {state_time:?}, absolute = {absolute_state_time:?}"
                            )));
                        }
                        if absolute_state_time > absolute_stop_time {
                            solver.problem().eqn.set_left_continuity_time(None);
                            return Err(stop_time_before_current_time(
                                absolute_stop_time,
                                absolute_state_time,
                            ));
                        }
                        if absolute_state_time == absolute_stop_time {
                            solver.problem().eqn.set_left_continuity_time(None);
                            if is_discontinuity_boundary {
                                *pending_reinit = true;
                            }
                            return Ok(());
                        }
                        if rebased {
                            solver.problem().eqn.set_left_continuity_time(None);
                            ensure_no_material_infusion_is_skipped(
                                solver,
                                absolute_state_time,
                                absolute_stop_time,
                                rtol,
                                atol,
                            )?;
                            return Err(cannot_integrate_distinct_stop(
                                segment_start_time,
                                absolute_stop_time,
                                absolute_state_time,
                            ));
                        }
                        rebase_solver_time(
                            solver,
                            absolute_state_time,
                            absolute_stop_time,
                            dy_scratch,
                        )?;
                        if let Err(error) = after_step() {
                            solver.problem().eqn.set_left_continuity_time(None);
                            return Err(error);
                        }
                        *pending_reinit = false;
                        rebased = true;
                    }
                    Err(err) => {
                        solver.problem().eqn.set_left_continuity_time(None);
                        after_step()?;
                        return Err(solver_error_at_absolute_time(
                            solver,
                            err,
                            absolute_stop_time,
                        ));
                    }
                }
            }
        })();

        if short_segment {
            *solver.config_mut().as_base_mut().minimum_timestep = configured_minimum_timestep;
        }
        if segment_result.is_ok() {
            restore_timestep_after_successful_stop(solver);
        }
        if segment_result.is_err() {
            solver.problem().eqn.set_left_continuity_time(None);
        }
        segment_result?;
    }
}

/// Step the solver until the stop set by `set_stop_time`.
///
/// Solver failures are returned directly. Retrying every nonlinear or linear
/// algebra failure can hide structural model errors and has no generally safe,
/// unit-independent restart step.
#[allow(clippy::too_many_arguments)]
fn integrate_to_stop<'a, F, S, H>(
    solver: &mut S,
    stop_time: f64,
    absolute_stop_time: f64,
    segment_start_time: f64,
    is_discontinuity_boundary: bool,
    pending_reinit: &mut bool,
    allow_close_rebase: bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
    after_step: &mut H,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    loop {
        if solver.problem().eqn.explicit_step_budget_check_due() {
            // Drain callback errors before deciding whether this is a
            // productive window reset or a fail-closed guard error.
            if let Err(error) = after_step() {
                solver.problem().eqn.set_left_continuity_time(None);
                return Err(error);
            }
            if let Some(error) = solver.problem().eqn.check_explicit_step_budget() {
                solver.problem().eqn.set_left_continuity_time(None);
                return Err(error);
            }
        }

        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                if let Err(error) = after_step() {
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(error);
                }
            }
            Ok(OdeSolverStopReason::TstopReached) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                // Keep left-continuity active while reconciling the accepted
                // state. In particular, a residual close segment ending at a
                // covariate or infusion boundary must use the segment on the
                // left until the exact boundary is reached.
                if let Err(error) = after_step() {
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(error);
                }
                let reached = match normalize_reached_stop(
                    solver,
                    stop_time,
                    absolute_stop_time,
                    segment_start_time,
                    allow_close_rebase,
                    dy_scratch,
                    rtol,
                    atol,
                ) {
                    Ok(reached) => reached,
                    Err(error) => {
                        solver.problem().eqn.set_left_continuity_time(None);
                        return Err(error);
                    }
                };

                match reached {
                    ReachedStop::Exact | ReachedStop::Interpolated => {
                        solver.problem().eqn.set_left_continuity_time(None);
                        if matches!(reached, ReachedStop::Interpolated) {
                            *pending_reinit = true;
                        }
                        if is_discontinuity_boundary {
                            *pending_reinit = true;
                        }
                        return Ok(());
                    }
                    ReachedStop::Rebased => {
                        // `rebase_solver_time` refreshes the RHS and Jacobian;
                        // native callbacks may report an error during that
                        // refresh. Surface it before asking diffsol to accept
                        // another stop time, and leave the segment cleanup to
                        // the same restoration path as every other failure.
                        let configured_minimum_timestep =
                            *solver.config().as_base_ref().minimum_timestep;
                        if let Err(error) = after_step() {
                            *solver.config_mut().as_base_mut().minimum_timestep =
                                configured_minimum_timestep;
                            solver.problem().eqn.set_left_continuity_time(None);
                            return Err(error);
                        }
                        *solver.config_mut().as_base_mut().minimum_timestep = 0.0;
                        let residual_result = (|| -> Result<(), PharmsolError> {
                            let residual_stop = ensure_solver_stop_coordinates::<F, S, H>(
                                solver,
                                absolute_stop_time,
                                dy_scratch,
                                after_step,
                            )?;
                            let residual_state = solver.state().t;
                            let residual_gap = residual_stop - residual_state;
                            if !residual_stop.is_finite()
                                || !residual_state.is_finite()
                                || !residual_gap.is_finite()
                                || residual_gap <= 0.0
                            {
                                let absolute_state_time =
                                    solver.problem().eqn.absolute_time(residual_state);
                                ensure_no_material_infusion_is_skipped(
                                    solver,
                                    absolute_state_time,
                                    absolute_stop_time,
                                    rtol,
                                    atol,
                                )?;
                                return Err(cannot_integrate_distinct_stop(
                                    segment_start_time,
                                    absolute_stop_time,
                                    absolute_state_time,
                                ));
                            }

                            match solver.set_stop_time(residual_stop) {
                                Ok(()) => integrate_to_stop(
                                    solver,
                                    residual_stop,
                                    absolute_stop_time,
                                    segment_start_time,
                                    is_discontinuity_boundary,
                                    pending_reinit,
                                    false,
                                    dy_scratch,
                                    rtol,
                                    atol,
                                    after_step,
                                ),
                                Err(diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StopTimeAtCurrentTime,
                                )) => {
                                    let absolute_state_time =
                                        solver.problem().eqn.absolute_time(solver.state().t);
                                    ensure_no_material_infusion_is_skipped(
                                        solver,
                                        absolute_state_time,
                                        absolute_stop_time,
                                        rtol,
                                        atol,
                                    )?;
                                    Err(cannot_integrate_distinct_stop(
                                        segment_start_time,
                                        absolute_stop_time,
                                        absolute_state_time,
                                    ))
                                }
                                Err(error) => {
                                    after_step()?;
                                    Err(solver_error_at_absolute_time(
                                        solver,
                                        error,
                                        absolute_stop_time,
                                    ))
                                }
                            }
                        })();
                        *solver.config_mut().as_base_mut().minimum_timestep =
                            configured_minimum_timestep;
                        if residual_result.is_err() {
                            solver.problem().eqn.set_left_continuity_time(None);
                        }
                        residual_result?;
                        return Ok(());
                    }
                }
            }
            Ok(OdeSolverStopReason::RootFound(root_time, _)) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                if let Err(error) = after_step() {
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(error);
                }
                solver.problem().eqn.set_left_continuity_time(None);
                let absolute_root_time = solver.problem().eqn.absolute_time(root_time);
                return Err(PharmsolError::OtherError(format!(
                    "solver stopped at an unexpected root at t = {:.4} \
                     (root finding is not configured)",
                    absolute_root_time
                )));
            }
            Err(diffsol::error::DiffsolError::OdeSolverError(
                OdeSolverError::StopTimeAtCurrentTime,
            )) => {
                if let Err(error) = after_step() {
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(error);
                }
                let state_time = solver.state().t;
                let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
                if !state_time.is_finite() || !absolute_state_time.is_finite() {
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(PharmsolError::OtherError(format!(
                        "ODE solver reached a stop with non-finite state time: local = {state_time:?}, absolute = {absolute_state_time:?}"
                    )));
                }
                if absolute_state_time > absolute_stop_time {
                    if let Err(error) = solver.state_mut_back(stop_time) {
                        solver.problem().eqn.set_left_continuity_time(None);
                        return Err(solver_error_at_absolute_time(
                            solver,
                            error,
                            absolute_stop_time,
                        ));
                    }
                    solver.problem().eqn.set_left_continuity_time(None);
                    *pending_reinit = true;
                    return Ok(());
                }
                if absolute_state_time < absolute_stop_time {
                    ensure_no_material_infusion_is_skipped(
                        solver,
                        absolute_state_time,
                        absolute_stop_time,
                        rtol,
                        atol,
                    )?;
                    solver.problem().eqn.set_left_continuity_time(None);
                    return Err(cannot_integrate_distinct_stop(
                        segment_start_time,
                        absolute_stop_time,
                        absolute_state_time,
                    ));
                }
                solver.problem().eqn.set_left_continuity_time(None);
                if is_discontinuity_boundary {
                    *pending_reinit = true;
                }
                return Ok(());
            }
            Err(err) => {
                solver.problem().eqn.set_left_continuity_time(None);
                // A model-function error raised inside the RHS is the root
                // cause when present; surface it over the solver error.
                after_step()?;
                return Err(solver_error_at_absolute_time(
                    solver,
                    err,
                    absolute_stop_time,
                ));
            }
        }
    }
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
        let mut integration_boundary_cursor = 0usize;
        let mut index = 0usize;
        // Set when the previous event changed the state or the previous stop
        // was a discontinuity boundary: the solver must be restarted before the
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
                advance_solver_to_event(
                    solver,
                    next_event.time(),
                    &mut integration_boundary_cursor,
                    &mut pending_reinit,
                    dy_scratch,
                    self.rtol,
                    self.atol,
                    &mut || Ok(()),
                )?;
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

    type GuardTestRhs = fn(&V, &V, f64, &mut V, &V, &V, &Covariates);

    fn unit_rhs(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _bolus: &V,
        _rateiv: &V,
        _covariates: &Covariates,
    ) {
        dx[0] = 1.0;
    }

    fn guard_test_problem(h0: f64) -> diffsol::OdeSolverProblem<PMProblem<'static, GuardTestRhs>> {
        let covariates = Covariates::default();
        OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(h0)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    unit_rhs as GuardTestRhs,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    std::iter::empty::<&Infusion>(),
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed")
    }

    macro_rules! build_guard_test_problem {
        ($h0:expr) => {{
            let covariates = Covariates::default();
            OdeBuilder::<M>::new()
                .atol(vec![1e-4])
                .rtol(1e-4)
                .t0(0.0)
                .h0($h0)
                .p(Vec::<f64>::new())
                .build_from_eqn(
                    PMProblem::with_params_v(
                        unit_rhs as GuardTestRhs,
                        1,
                        0,
                        V::zeros(0, NalgebraContext::new()),
                        &covariates,
                        std::iter::empty::<&Infusion>(),
                        V::zeros(1, NalgebraContext::new()),
                        0.0,
                    )
                    .expect("finite test ODE problem"),
                )
                .expect("test ODE builder should succeed")
        }};
    }

    #[test]
    fn explicit_step_guard_is_selected_only_for_tsit45() {
        assert_eq!(accepted_step_limits_for_solver(&OdeSolver::Bdf), None);
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
            None
        );
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
            None
        );
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
            Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 500_000,
                max_accepted_steps_per_session: 10_000_000,
            })
        );
    }

    #[test]
    fn explicit_step_guard_rejects_stagnant_window_and_accumulates_session() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 4,
            }));
        problem.eqn.begin_explicit_step_segment(10.0, 20.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        problem.eqn.rebase_time_origin(10.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        assert!(problem.eqn.explicit_step_budget_check_due());
        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("a stagnant window must fail closed");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("segment start = 1.0000000000000000e1"));
        assert!(message.contains("target = 2.0000000000000000e1"));
        assert!(message.contains("numeric gap (target - start) = 1.0000000000000000e1"));
        assert!(message.contains("progress window start = 1.0000000000000000e1"));
        assert!(message.contains("actual progress = 0.0000000000000000e0"));
        assert!(message.contains("required progress = 1.0000000000000000e-3"));
        assert!(message.contains("progress fraction = 0.0000000000000000e0"));
        assert!(message.contains("minimum progress fraction = 1.0000000000000000e-4"));
        assert!(message.contains("last accepted absolute time = 1.0000000000000000e1"));
        assert!(message.contains("window count/limit = 2/2"));
        assert!(message.contains("cumulative count/limit = 2/4"));
        assert!(message.contains("no incomplete state was returned"));
        assert!(message.contains("model stiffness and state/time/unit scaling"));
        assert!(message.contains("explicitly choose an implicit solver"));

        problem.eqn.begin_explicit_step_segment(20.0, 30.0);
        assert!(!problem.eqn.explicit_step_budget_check_due());
        problem.eqn.record_explicit_accepted_step(20.0);
        problem.eqn.record_explicit_accepted_step(20.0);
        let second_segment_error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("the session count must survive a new segment");
        let second_segment_message = second_segment_error.to_string();
        assert!(second_segment_message.contains("budget scope = session total"));
        assert!(second_segment_message.contains("segment start = 2.0000000000000000e1"));
        assert!(second_segment_message.contains("target = 3.0000000000000000e1"));
        assert!(second_segment_message.contains("progress window start = 2.0000000000000000e1"));
        assert!(second_segment_message.contains("window count/limit = 2/2"));
        assert!(second_segment_message.contains("cumulative count/limit = 4/4"));
    }

    #[test]
    fn explicit_step_guard_rejects_zero_progress_for_positive_subnormal_gap() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let positive_subnormal_gap = f64::from_bits(1);
        problem
            .eqn
            .begin_explicit_step_segment(0.0, positive_subnormal_gap);
        problem.eqn.record_explicit_accepted_step(0.0);
        assert!(problem.eqn.explicit_step_budget_check_due());

        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("zero progress must fail closed even when required progress underflows");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("required progress = 0.0000000000000000e0"));
        assert!(message.contains("progress fraction = 0.0000000000000000e0"));
        assert!(problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn explicit_step_guard_continues_after_productive_window() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 4,
            }));
        problem.eqn.begin_explicit_step_segment(10.0, 20.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        problem.eqn.record_explicit_accepted_step(10.01);
        assert!(problem.eqn.explicit_step_budget_check_due());
        assert!(problem.eqn.check_explicit_step_budget().is_none());
        assert!(!problem.eqn.explicit_step_budget_check_due());
        problem.eqn.record_explicit_accepted_step(10.02);
        assert!(!problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn rebase_and_residual_do_not_reset_session_budget() {
        let problem = build_guard_test_problem!(1.0);
        let target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 3,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        problem.eqn.begin_explicit_step_segment(0.0, target);
        problem.eqn.record_explicit_accepted_step(0.0);
        rebase_solver_time(&mut solver, 0.0, target, &mut dy_scratch)
            .expect("the test rebase should preserve a positive residual");

        let mut pending_reinit = false;
        let _error = integrate_to_stop(
            &mut solver,
            target,
            target,
            0.0,
            false,
            &mut pending_reinit,
            false,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the residual must stop before requesting another step");

        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("the residual accepted step must retain the session count");
        let message = error.to_string();
        assert!(message.contains("budget scope = session total"));
        assert!(message.contains("window count/limit = 2/3"));
        assert!(message.contains("cumulative count/limit = 2/2"));
    }

    #[test]
    fn advance_resets_segment_budget_but_not_session_budget() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let first_target = 1.0e-4_f64.next_down();

        advance_solver_to_event(
            &mut solver,
            first_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("the first segment should use its one-step budget");

        let current_time = solver.state().t;
        let next_target = current_time + solver.state().h;
        assert!(next_target > current_time);
        advance_solver_to_event(
            &mut solver,
            next_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a new exact segment should reset only its segment budget");

        let state_before_third = solver.state().t;
        let third_target = state_before_third + solver.state().h;
        assert!(third_target > state_before_third);
        let error = advance_solver_to_event(
            &mut solver,
            third_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the cumulative session budget must block the next segment");
        let message = error.to_string();
        assert!(message.contains("budget scope = session total"));
        assert!(message.contains("window count/limit = 0/1"));
        assert!(message.contains("cumulative count/limit = 2/2"));
        assert!(message.contains("numeric gap (target - start)"));
        assert_eq!(solver.state().t, state_before_third);
    }

    #[test]
    fn disabled_explicit_step_guard_never_limits_implicit_session() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(accepted_step_limits_for_solver(&OdeSolver::Bdf));
        problem.eqn.begin_explicit_step_segment(0.0, 2.0);
        for time in 1..=4 {
            problem.eqn.record_explicit_accepted_step(f64::from(time));
        }
        assert!(!problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn explicit_step_guard_checks_before_requesting_next_step() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            10.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the second accepted step must be blocked");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("numeric gap (target - start)"));
        assert!(message.contains("progress window start ="));
        assert!(message.contains("actual progress ="));
        assert!(message.contains("required progress ="));
        assert!(message.contains("progress fraction ="));
        assert!(message.contains("window count/limit = 1/1"));
        assert!(message.contains("cumulative count/limit = 1/2"));
        assert!(message.contains("last accepted absolute time"));
        assert!(message.contains("no incomplete state was returned"));
        assert!(message.contains("explicitly choose an implicit solver"));
        assert!(solver.state().t > 0.0);
        assert!(solver.state().t < 10.0);
    }

    #[test]
    fn explicit_step_guard_accepts_exact_completion_on_final_step() {
        let problem = build_guard_test_problem!(1.0);
        let final_target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        advance_solver_to_event(
            &mut solver,
            final_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a final permitted step that reaches the target is complete");
        assert_eq!(solver.state().t, final_target);
    }

    #[test]
    fn explicit_step_guard_accepts_exact_completion_on_final_session_step() {
        let problem = build_guard_test_problem!(1.0);
        let final_target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 1,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        advance_solver_to_event(
            &mut solver,
            final_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a final permitted session step that reaches the target is complete");
        assert_eq!(solver.state().t, final_target);
    }

    #[test]
    fn callback_error_precedes_set_stop_time_error() {
        let problem = build_guard_test_problem!(1.0);
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        *solver.state_mut().h = -1.0;
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Err(PharmsolError::OtherError("native callback error".into())),
        )
        .expect_err("callback failure must precede the set_stop_time error");
        assert!(error.to_string().contains("native callback error"));
    }

    #[test]
    fn callback_error_precedes_explicit_step_budget_error() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 0,
                max_accepted_steps_per_session: 1,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Err(PharmsolError::OtherError("native callback error".into())),
        )
        .expect_err("callback failure must retain precedence");
        assert!(error.to_string().contains("native callback error"));
    }

    #[test]
    fn resolved_schedule_rejects_positive_nonzero_infusion_rate_underflow() {
        let amount = f64::from_bits(1);
        let events = [Event::Infusion(Infusion::new(1.5, amount, "0", 2.0, 0))];
        let error = validate_resolved_ode_schedule(&events)
            .expect_err("positive nonzero infusion underflow must be rejected");
        let message = error.to_string();
        assert!(message.contains("t = 1.5000000000000000e0"));
        assert!(message.contains("amount"));
        assert!(message.contains("duration"));
        assert!(message.contains("silently lose nonzero infusion material"));
    }

    #[test]
    fn resolved_schedule_rejects_negative_nonzero_infusion_rate_underflow() {
        let amount = -f64::from_bits(1);
        let events = [Event::Infusion(Infusion::new(1.5, amount, "0", 2.0, 0))];
        let error = validate_resolved_ode_schedule(&events)
            .expect_err("negative nonzero infusion underflow must be rejected");
        let message = error.to_string();
        assert!(message.contains("t = 1.5000000000000000e0"));
        assert!(message.contains("amount"));
        assert!(message.contains("duration"));
        assert!(message.contains("silently lose nonzero infusion material"));
    }

    #[test]
    fn resolved_schedule_accepts_zero_amount_infusion_with_zero_rate() {
        let events = [Event::Infusion(Infusion::new(1.5, 0.0, "0", 2.0, 0))];
        assert_eq!(
            validate_resolved_ode_schedule(&events).expect("zero amount is material-free"),
            1.5
        );
    }

    #[test]
    fn shared_advance_helper_rejects_backward_targets() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    std::iter::empty::<&Infusion>(),
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let error = advance_solver_to_event(
            &mut solver,
            -1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("backward event target must be rejected");

        assert!(error.to_string().contains("before current time"));
        assert_eq!(solver.state().t, 0.0);

        let mut callback_called = false;
        advance_solver_to_event(
            &mut solver,
            0.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || {
                callback_called = true;
                Ok(())
            },
        )
        .expect("equal target should be a zero-length no-op");
        assert!(!callback_called);
        assert_eq!(solver.state().t, 0.0);
    }

    #[test]
    fn shared_advance_helper_normalizes_unrepresentable_local_targets() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    std::iter::empty::<&Infusion>(),
                    V::zeros(1, NalgebraContext::new()),
                    -192.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        advance_solver_to_event(
            &mut solver,
            0.35,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("finite cancellation should be normalized before set_stop_time");

        let absolute_time = solver.problem().eqn.absolute_time(solver.state().t);
        assert_eq!(
            absolute_time,
            0.35,
            "origin = {}, local state = {}",
            solver.problem().eqn.time_origin(),
            solver.state().t
        );
    }

    #[test]
    fn normalize_reached_stop_does_not_relabel_local_time_equality() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    std::iter::empty::<&Infusion>(),
                    V::zeros(1, NalgebraContext::new()),
                    -1.0e16,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        *solver.state_mut().t = 1.0e16;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let reached = normalize_reached_stop(
            &mut solver,
            1.0e16,
            1.0,
            0.0,
            true,
            &mut dy_scratch,
            1e-4,
            1e-4,
        )
        .expect("mismatched local equality should use the checked rebase");

        assert_eq!(reached, ReachedStop::Rebased);
        assert_eq!(solver.problem().eqn.time_origin(), 0.0);
        assert_eq!(solver.state().t, 0.0);
    }

    #[test]
    fn rebase_callback_error_precedes_followup_solver_error() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    std::iter::empty::<&Infusion>(),
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let large_time = 1.0e6_f64;
        advance_solver_to_event(
            &mut solver,
            large_time,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("the solver should reach the large event before the close target");

        let close_target = large_time.next_up();
        let error = advance_solver_to_event(
            &mut solver,
            close_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || {
                Err(PharmsolError::OtherError(
                    "callback error during rebase".into(),
                ))
            },
        )
        .expect_err("the callback error must be returned");

        assert!(error.to_string().contains("callback error during rebase"));
        assert_eq!(solver.problem().eqn.time_origin(), large_time);
        assert_eq!(solver.state().t, 0.0);
    }

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
        // `StopTimeAtCurrentTime`, which confirms the boundary is reached.
        // The reached boundary is *before* the observation at
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
        let kdp = 0.09175211071968079;
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
        // Exact finite-f64 reproduction of debug/dat.csv subject `1`: 145
        // events (103 short infusions and 42 observations), including duplicate
        // observations. Without boundary restarts, individually recovered
        // Newton failures accumulate past diffsol's 50-failure limit.
        let mut builder = Subject::builder("long_horizon_short_infusions");
        builder = builder.infusion(0.0, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(0.005, "cp");
        builder = builder.missing_observation(0.0179166666666667, "cp");
        builder = builder.missing_observation(0.0220833333333333, "cp");
        builder = builder.missing_observation(0.0220833333333333, "cp");
        builder = builder.missing_observation(0.0345833333333333, "cp");
        builder = builder.missing_observation(0.0383333333333333, "cp");
        builder = builder.missing_observation(0.0383333333333333, "cp");
        builder = builder.missing_observation(0.0845833333333333, "cp");
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
        builder = builder.infusion(8.49083333333333, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(8.99166666666667, "cp");
        builder = builder.infusion(8.99291666666667, 1e+09, "iv", 0.00125);
        builder = builder.missing_observation(8.99583333333333, "cp");
        builder = builder.missing_observation(9.01041666666667, "cp");
        builder = builder.missing_observation(9.07416666666667, "cp");
        builder = builder.missing_observation(9.16666666666667, "cp");
        builder = builder.infusion(9.49291666666667, 1e+09, "iv", 0.00125);
        builder = builder.infusion(9.99291666666667, 1e+09, "iv", 0.00125);
        builder = builder.infusion(10.4929166666667, 1e+09, "iv", 0.00125);
        builder = builder.infusion(10.9929166666667, 1e+09, "iv", 0.00125);
        builder = builder.infusion(11.4929166666667, 1e+09, "iv", 0.00125);
        builder = builder.infusion(12.0145833333333, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(12.0395833333333, "cp");
        builder = builder.missing_observation(12.0395833333333, "cp");
        builder = builder.missing_observation(13.01375, "cp");
        builder = builder.infusion(13.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(13.0179166666667, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.1925, "cp");
        builder = builder.missing_observation(13.26875, "cp");
        builder = builder.infusion(14.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(15.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(16.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(17.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(18.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(19.0154166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(20.0491666666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(21.0333333333333, "cp");
        builder = builder.infusion(21.0491666666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(21.055, "cp");
        builder = builder.missing_observation(21.0679166666667, "cp");
        builder = builder.infusion(22.0491666666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(23.0491666666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(26.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(27.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(28.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(29.05125, 3e+09, "iv", 0.00125);
        builder = builder.infusion(30.05125, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(33.0533333333333, "cp");
        builder = builder.infusion(33.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(33.06375, "cp");
        builder = builder.missing_observation(34.0375, "cp");
        builder = builder.missing_observation(34.0375, "cp");
        builder = builder.infusion(34.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(35.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(36.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(37.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(40.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(41.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(42.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(43.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(44.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(47.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(48.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(49.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(50.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(51.0604166666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(54.0366666666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(55.0366666666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(56.1741666666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(56.2166666666667, "cp");
        builder = builder.missing_observation(56.2166666666667, "cp");
        builder = builder.infusion(57.1741666666667, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(58.0154166666667, "cp");
        builder = builder.missing_observation(58.0154166666667, "cp");
        builder = builder.infusion(58.1741666666667, 3e+09, "iv", 0.00125);
        builder = builder.infusion(61.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(62.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(63.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(64.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(65.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(68.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(69.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(70.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(71.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(72.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(75.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(76.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(77.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(78.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(79.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(82.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(83.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(84.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(85.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(86.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(89.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(90.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(91.0595833333333, "cp");
        builder = builder.infusion(91.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(92.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.missing_observation(93.0741666666667, "cp");
        builder = builder.missing_observation(93.0741666666667, "cp");
        builder = builder.infusion(93.1758333333333, 3e+09, "iv", 0.00125);
        builder = builder.infusion(96.0279166666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(97.0279166666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(98.0279166666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(99.0279166666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(100.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(103.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(104.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(105.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(106.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(107.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(110.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(111.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(112.027083333333, "cp");
        builder = builder.missing_observation(112.027083333333, "cp");
        builder = builder.infusion(112.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(112.074166666667, "cp");
        builder = builder.missing_observation(112.074166666667, "cp");
        builder = builder.infusion(113.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.infusion(114.027916666667, 4.5e+09, "iv", 0.00125);
        builder = builder.missing_observation(114.133333333333, "cp");
        builder = builder.missing_observation(114.133333333333, "cp");
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
