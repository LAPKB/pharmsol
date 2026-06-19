mod closure;

/// Re-export of the diffsol `OdeEquations` adapter so the JIT module can build
/// `OdeBuilder` problems with closures (rather than plain `fn` pointers).
///
/// This helper is shared by the legacy JIT path and the native
/// runtime wrappers.
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load", feature = "dsl-wasm"))]
pub(crate) mod closure_helpers {
    pub(crate) use super::closure::PMProblem;
}

use crate::{
    core::{ModelInfo, Solver},
    data::Covariates,
    error_model::AssayErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, PharmsolError, Subject,
};

use crate::core::Predictions;
use crate::simulator::backends::parameters_hash;
use crate::simulator::cache::{BoundErrorModelCache, PredictionCache, DEFAULT_CACHE_SIZE};
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, OdeBuilder,
    OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;
use pharmsol_dsl::ModelKind;
use thiserror::Error;

use crate::core::metadata::{ModelMetadata, ModelMetadataError, ValidatedModelMetadata};
use crate::core::State;

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
    core: crate::core::ModelCore<PredictionCache>,
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
}

impl ODE {
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Self {
            core: crate::core::ModelCore::new(Some(PredictionCache::new(DEFAULT_CACHE_SIZE))),
            diffeq,
            lag,
            fa,
            init,
            out,
            solver: OdeSolver::default(),
            rtol: RTOL,
            atol: ATOL,
        }
    }

    /// Set the number of state variables (ODE compartments).
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.core = self.core.with_nstates(nstates);
        self
    }

    /// Set the number of drug inputs (size of bolus[] and rateiv[]).
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.core = self.core.with_ndrugs(ndrugs);
        self
    }

    /// Set the number of output equations.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.core = self.core.with_nout(nout);
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
        let validated = metadata
            .validate_for(ModelKind::Ode)
            .map_err(OdeMetadataError::Validation)?;
        validate_metadata_dimensions(&validated, &self.core.dims())?;
        self.core.set_metadata(validated);
        Ok(self)
    }

    /// Access the validated metadata attached to this ODE, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.core.metadata()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.state_index(name)
    }

    pub(crate) fn run_events<'a, F, S>(
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
        y_out: &mut V,
        likelihood: &mut Vec<f64>,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    {
        for (index, event) in events.iter().enumerate() {
            let next_event = events.get(index + 1);

            match event {
                Event::Bolus(bolus) => {
                    let input =
                        bolus
                            .input_index()
                            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
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
                }
                Event::Infusion(_) => {}
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
                        PharmsolError::UnknownOutputLabel {
                            label: observation.outeq().to_string(),
                        }
                    })?;
                    let pred = y_out[outeq];
                    let pred =
                        observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                    if let Some(em) = error_models {
                        likelihood.push(pred.log_likelihood(em)?.exp());
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
                                        "ODE solver step size went to zero at t = {time:.4}",
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
        Ok(())
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
    let bound_error_models = crate::core::simulate::bind_error_models_inner(ode, error_models)?;
    let ypred = _subject_predictions(ode, subject, parameters)?;
    Ok(ypred.log_likelihood(&bound_error_models)?.exp())
}

#[inline(always)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    parameters: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = ode.core.cache() {
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
        Some(error_models) => Some(crate::core::simulate::bind_error_models_inner(
            ode,
            error_models,
        )?),
        None => None,
    };
    let bound_error_models = bound_error_models.as_deref();

    let mut output = SubjectPredictions::new(ode.nparticles());

    let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
    let mut likelihood = Vec::with_capacity(event_count);

    let nstates = ode.nstates();
    let ndrugs = ode.ndrugs();

    let mut state_with_bolus = V::zeros(nstates, NalgebraContext);
    let mut state_without_bolus = V::zeros(nstates, NalgebraContext);
    let zero_bolus = V::zeros(ndrugs, NalgebraContext);
    let zero_rateiv = V::zeros(ndrugs, NalgebraContext);
    let mut bolus_v = V::zeros(ndrugs, NalgebraContext);
    let parameters_vec = parameters.to_vec();
    let parameters_v: V = DVector::from_vec(parameters_vec.clone()).into();

    let mut y_out = V::zeros(ode.nout(), NalgebraContext);

    for occasion in subject.occasions() {
        let covariates = occasion.covariates();
        let events = ode.resolve_events(occasion, parameters, covariates)?;

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
                    &mut y_out,
                    &mut likelihood,
                    &mut output,
                )?;
            }
        }
    }

    let ll = bound_error_models.map(|_| likelihood.iter().product::<f64>());
    Ok((output, ll))
}

// ── New core traits ─────────────────────────────────────────────────────────

impl crate::core::Solver for ODE {
    type State = V;
    type Predictions = SubjectPredictions;

    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.nstates(), NalgebraContext);
        if occasion_index == 0 {
            (init)(
                &V::from_vec(parameters.to_vec(), NalgebraContext),
                0.0,
                covariates,
                &mut x,
            );
        }
        x
    }

    fn nparticles(&self) -> usize {
        1
    }

    fn is_batch(&self) -> bool {
        true
    }
}

impl crate::core::ModelInfo for ODE {
    fn nstates(&self) -> usize {
        self.core.nstates()
    }

    fn ndrugs(&self) -> usize {
        self.core.ndrugs()
    }

    fn nout(&self) -> usize {
        self.core.nout()
    }

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.core.metadata()
    }

    fn lag(&self) -> &Lag {
        &self.lag
    }

    fn fa(&self) -> &Fa {
        &self.fa
    }
}

impl crate::core::Caching for ODE {
    fn prediction_cache(&self) -> Option<&PredictionCache> {
        self.core.cache()
    }

    fn error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.core.error_model_cache()
    }

    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.core = self.core.with_cache_capacity(PredictionCache::new(size));
        self
    }

    fn without_cache(mut self) -> Self {
        self.core = self.core.without_cache();
        self
    }

    fn clear_cache(&self) {
        self.core.clear_cache();
        if let Some(cache) = self.core.cache() {
            cache.invalidate_all();
        }
    }
}

impl crate::core::Simulate for ODE {
    fn simulate_subject(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<Self::Predictions, PharmsolError> {
        if error_models.is_none() {
            if let Some(cache) = self.core.cache() {
                let key = (subject.hash(), super::parameters_hash(params));
                if let Some(cached) = cache.get(&key) {
                    return Ok(cached);
                }
            }
        }

        let (predictions, _likelihood) =
            _simulate_subject_dense(self, subject, params, error_models)?;

        if error_models.is_none() {
            if let Some(cache) = self.core.cache() {
                let key = (subject.hash(), super::parameters_hash(params));
                cache.insert(key, predictions.clone());
            }
        }

        Ok(predictions)
    }

    fn log_likelihood(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models =
            crate::core::simulate::bind_error_models_inner(self, error_models)?;
        let ypred = _subject_predictions(self, subject, params)?;
        ypred.log_likelihood(&bound_error_models)
    }

    fn kind() -> pharmsol_dsl::ModelKind {
        pharmsol_dsl::ModelKind::Ode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Simulate;
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

    fn explicit_route_kernel(
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

    fn injected_route_kernel(
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

    fn counting_kernel(
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
                crate::core::metadata::new("bimodal_ke")
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
                crate::core::metadata::new("wrong_outputs")
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
                crate::core::metadata::new("missing_destination")
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
            explicit_route_kernel,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            crate::core::metadata::new("explicit_routes")
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
            .expect("simulation should succeed");
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
            injected_route_kernel,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            crate::core::metadata::new("injected_routes")
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
            .expect("simulation should succeed");

        assert_relative_eq!(
            predictions.predictions()[0].prediction(),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn handwritten_ode_metadata_resolves_raw_numeric_aliases_against_canonical_labels() {
        let ode = ODE::new(
            explicit_route_kernel,
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            crate::core::metadata::new("numeric_alias_ode")
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
            .expect("canonical labels should simulate");
        let aliased_predictions = ode
            .simulate_subject(&aliased, &crate::parameters::dense([]), None)
            .expect("raw numeric aliases should simulate");

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
                crate::core::metadata::new("bimodal_ke")
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

        let ode = ODE::new(counting_kernel, zero_lag, unit_fa, zero_init, state_output)
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
}
