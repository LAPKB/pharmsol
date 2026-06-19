mod em;

use diffsol::{NalgebraContext, Vector};
use nalgebra::DVector;
use ndarray::Array2;
use pharmsol_dsl::ModelKind;
use rand::{rng, RngExt};
use rayon::prelude::*;
use thiserror::Error;

use crate::core::{ModelInfo, Simulate};
use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::Prediction,
    simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Observation, PharmsolError, Subject,
};

use crate::simulator::backends::parameters_hash;
use crate::simulator::cache::{
    BoundErrorModelCache, PredictionCache, SdeLikelihoodCache, DEFAULT_CACHE_SIZE,
};

use diffsol::VectorCommon;

use crate::core::metadata::{ModelMetadata, ModelMetadataError, ValidatedModelMetadata};
use crate::core::{Predictions, State};
use crate::simulator::likelihood::{LikelihoodModel, ParticleLikelihood};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SdeMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("SDE declares {declared} state metadata entries but model has {expected} states")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error("SDE declares {declared} route metadata entries but model has {expected} inputs")]
    RouteCountMismatch { expected: usize, declared: usize },
    #[error("SDE declares {declared} output metadata entries but model has {expected} outputs")]
    OutputCountMismatch { expected: usize, declared: usize },
}

#[derive(Clone, Debug, Default)]
struct InjectedBolusMappings {
    destinations: Vec<Option<usize>>,
}

impl InjectedBolusMappings {
    fn explicit(ndrugs: usize) -> Self {
        Self {
            destinations: vec![None; ndrugs],
        }
    }

    fn from_destinations(ndrugs: usize, destinations: &[Option<usize>]) -> Self {
        let mut mappings = Self::explicit(ndrugs);
        for (input, destination) in destinations.iter().copied().take(ndrugs).enumerate() {
            mappings.destinations[input] = destination;
        }
        mappings
    }

    #[allow(dead_code)]
    fn invalidate_for_ndrugs(&mut self, ndrugs: usize) {
        *self = Self::explicit(ndrugs);
    }

    fn apply(&self, state: &mut [DVector<f64>], input: usize, amount: f64) -> bool {
        let Some(destination) = self.destinations.get(input).copied().flatten() else {
            return false;
        };
        state.par_iter_mut().for_each(|particle| {
            particle[destination] += amount;
        });
        true
    }
}

/// Simulate a stochastic differential equation (SDE) event.
///
/// This function advances the SDE system from time `ti` to `tf` using
/// the Euler-Maruyama method implemented in the `em` module.
///
/// # Arguments
///
/// * `drift` - Function defining the deterministic component of the SDE
/// * `difussion` - Function defining the stochastic component of the SDE
/// * `x` - Current state vector
/// * `parameters` - Parameter vector for the model
/// * `cov` - Covariates that may influence the system dynamics
/// * `infusions` - Infusion events to be applied during simulation
/// * `ti` - Starting time
/// * `tf` - Ending time
///
/// # Returns
///
/// The state vector at time `tf` after simulation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    parameters: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ndrugs: usize,
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }

    let parameters_v = V::from_vec(parameters.to_vec(), NalgebraContext);
    let covariates = cov.clone();
    let infusion_events = infusions.to_vec();
    let drift_fn = *drift;
    let diffusion_fn = *difussion;

    let parameters_for_drift = parameters_v.clone();
    let drift_closure = move |time: f64, state: &DVector<f64>, out: &mut DVector<f64>| {
        let mut rateiv = V::zeros(ndrugs, NalgebraContext);
        for infusion in &infusion_events {
            if time >= infusion.time() && time <= infusion.duration() + infusion.time() {
                let input = infusion
                    .input_index()
                    .expect("resolved infusions should use numeric input labels");
                rateiv[input] += infusion.amount() / infusion.duration();
            }
        }

        let state_v: V = state.clone().into();
        let mut out_v = V::zeros(state.len(), NalgebraContext);
        drift_fn(
            &state_v,
            &parameters_for_drift,
            time,
            &mut out_v,
            &rateiv,
            &covariates,
        );
        out.copy_from(out_v.inner());
    };

    let diffusion_closure = move |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| {
        let mut out_v = V::zeros(out.len(), NalgebraContext);
        diffusion_fn(&parameters_v, &mut out_v);
        out.copy_from(out_v.inner());
    };

    simulate_sde_event_with(drift_closure, diffusion_closure, x.inner().clone(), ti, tf).into()
}

pub(crate) fn simulate_sde_event_with<D, G>(
    drift: D,
    diffusion: G,
    initial_state: DVector<f64>,
    ti: f64,
    tf: f64,
) -> DVector<f64>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
{
    if ti == tf {
        return initial_state;
    }

    let mut sde = em::EM::new(drift, diffusion, initial_state, 1e-2, 1e-2);
    let (_time, solution) = sde.solve(ti, tf);
    solution.last().unwrap().clone()
}

/// Stochastic Differential Equation solver for pharmacometric models.
///
/// This struct represents a stochastic differential equation system and provides
/// methods to simulate particles and estimate likelihood for PKPD modeling.
///
/// SDE models introduce stochasticity into the system dynamics, allowing for more
/// realistic modeling of biological variability and uncertainty.
#[derive(Clone, Debug)]
pub struct SDE {
    core: crate::core::ModelCore<SdeLikelihoodCache>,
    drift: Drift,
    diffusion: Diffusion,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    nparticles: usize,
    injected_bolus_mappings: InjectedBolusMappings,
}

impl Predictions for Array2<Prediction> {
    fn new(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }
    fn squared_error(&self) -> f64 {
        unimplemented!();
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        if self.is_empty() || self.ncols() == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.ncols());
        for col in 0..self.ncols() {
            let column = self.column(col);
            let mean_prediction: f64 = column
                .iter()
                .map(|pred: &Prediction| pred.prediction())
                .sum::<f64>()
                / self.nrows() as f64;
            let mut prediction = column.first().unwrap().clone();
            prediction.set_prediction(mean_prediction);
            result.push(prediction);
        }
        result
    }
    fn log_likelihood(&self, model: &dyn LikelihoodModel) -> Result<f64, crate::PharmsolError> {
        let predictions = self.get_predictions();
        if predictions.is_empty() {
            return Ok(0.0);
        }
        let log_liks: Result<Vec<f64>, _> = predictions
            .iter()
            .filter(|p| p.observation().is_some())
            .map(|p| model.observation_log_likelihood(p))
            .collect();
        log_liks.map(|lls| lls.iter().sum())
    }
}

impl crate::core::PredictionsContainer for Array2<Prediction> {
    fn new(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }

    fn push(&mut self, pred: Prediction) {
        let col = Array2::from_shape_vec((self.nrows(), 1), vec![pred; self.nrows()]).unwrap();
        *self = ndarray::concatenate(ndarray::Axis(1), &[self.view(), col.view()]).unwrap();
    }

    fn predictions(&self) -> &[Prediction] {
        // Array2 doesn't support slicing to &[Prediction] directly
        unimplemented!("predictions() not supported for Array2 — use get_predictions()")
    }

    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, crate::PharmsolError> {
        let predictions: Vec<Prediction> = Predictions::get_predictions(self);
        if predictions.is_empty() {
            return Ok(0.0);
        }
        let log_liks: Result<Vec<f64>, _> = predictions
            .iter()
            .filter(|p| p.observation().is_some())
            .map(|p| p.log_likelihood(error_models))
            .collect();
        log_liks.map(|lls| lls.iter().sum())
    }
}

impl SDE {
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        nparticles: usize,
    ) -> Self {
        Self {
            core: crate::core::ModelCore::new(Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE))),
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
            nparticles,
            injected_bolus_mappings: InjectedBolusMappings::default(),
        }
    }

    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.core = self.core.with_nstates(nstates);
        self
    }

    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.core = self.core.with_ndrugs(ndrugs);
        self
    }

    pub fn with_nout(mut self, nout: usize) -> Self {
        self.core = self.core.with_nout(nout);
        self
    }

    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, SdeMetadataError> {
        let validated = metadata
            .validate_for_with_particles(ModelKind::Sde, self.nparticles)
            .map_err(SdeMetadataError::Validation)?;
        validate_metadata_dimensions(&validated, &self.core.dims())?;
        self.core.set_metadata(validated);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn with_injected_bolus_inputs(mut self, destinations: &[Option<usize>]) -> Self {
        self.injected_bolus_mappings =
            InjectedBolusMappings::from_destinations(self.core.ndrugs(), destinations);
        self
    }

    /// Access the validated metadata attached to this SDE model, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.core.metadata()
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
}

impl State for Vec<DVector<f64>> {
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.par_iter_mut().for_each(|particle| {
            particle[input] += amount;
        });
    }
}

fn validate_metadata_dimensions(
    metadata: &ValidatedModelMetadata,
    neqs: &Neqs,
) -> Result<(), SdeMetadataError> {
    let declared_states = metadata.states().len();
    if declared_states != neqs.nstates {
        return Err(SdeMetadataError::StateCountMismatch {
            expected: neqs.nstates,
            declared: declared_states,
        });
    }

    let declared_routes = metadata.route_input_count();
    if declared_routes != neqs.ndrugs {
        return Err(SdeMetadataError::RouteCountMismatch {
            expected: neqs.ndrugs,
            declared: declared_routes,
        });
    }

    let declared_outputs = metadata.outputs().len();
    if declared_outputs != neqs.nout {
        return Err(SdeMetadataError::OutputCountMismatch {
            expected: neqs.nout,
            declared: declared_outputs,
        });
    }

    Ok(())
}

// ── New core traits ─────────────────────────────────────────────────────────

impl crate::core::Solver for SDE {
    type State = Vec<DVector<f64>>;
    type Predictions = ParticleLikelihood;

    fn solve(
        &self,
        state: &mut Self::State,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        let ndrugs = self.ndrugs();
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone().into(),
                parameters,
                covariates,
                infusions,
                ndrugs,
                ti,
                tf,
            )
            .inner()
            .clone();
        });
        Ok(())
    }

    fn process_bolus(&self, state: &mut Self::State, input: usize, amount: f64) {
        if !self.injected_bolus_mappings.apply(state, input, amount) {
            state.add_bolus(input, amount);
        }
    }

    fn process_observation(
        &self,
        x: &mut Self::State,
        parameters: &[f64],
        observation: &Observation,
        likelihood: Option<&dyn LikelihoodModel>,
        covariates: &Covariates,
    ) -> Result<(Prediction, Option<f64>), PharmsolError> {
        // Compute predictions across all particles
        let nparticles = self.nparticles;
        let mut preds = vec![Prediction::default(); nparticles];

        preds.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.nout(), NalgebraContext);
            (self.out)(
                &x[i].clone().into(),
                &V::from_vec(parameters.to_vec(), NalgebraContext),
                observation.time(),
                covariates,
                &mut y,
            );
            let outeq = observation
                .outeq_index()
                .expect("resolved observations should use numeric output labels");
            *p = observation.to_prediction(y[outeq], x[i].as_slice().to_vec());
        });

        // Resampling — mutate state to concentrate particles on high-likelihood regions
        let lik = if let Some(model) = likelihood {
            let q: Vec<f64> = preds
                .iter()
                .map(|p| {
                    model
                        .observation_log_likelihood(p)
                        .map(f64::exp)
                        .unwrap_or(0.0)
                })
                .collect();
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let indices = sysresample(&w);
            *x = indices.iter().map(|&i| x[i].clone()).collect();
            Some(sum_q / nparticles as f64)
        } else {
            None
        };

        // Return the mean prediction across particles
        let mean_pred: f64 = preds.iter().map(|p| p.prediction()).sum::<f64>() / nparticles as f64;
        let mut prediction = preds[0].clone();
        prediction.set_prediction(mean_pred);

        Ok((prediction, lik))
    }

    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Vec<DVector<f64>> {
        let mut x = Vec::with_capacity(self.nparticles);
        for _ in 0..self.nparticles {
            let mut state: V = DVector::zeros(self.nstates()).into();
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(parameters.to_vec(), NalgebraContext),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state.inner().clone());
        }
        x
    }

    fn nparticles(&self) -> usize {
        self.nparticles
    }
}

impl crate::core::ModelInfo for SDE {
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

impl crate::core::Caching for SDE {
    fn prediction_cache(&self) -> Option<&PredictionCache> {
        self.core.cache().map(|_| unimplemented!()) /* SDE uses SdeLikelihoodCache */
        // SDE uses SdeLikelihoodCache, not PredictionCache
    }

    fn error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.core.error_model_cache()
    }

    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.core = self.core.with_cache_capacity(SdeLikelihoodCache::new(size));
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

impl SDE {
    /// Estimate the per-particle prediction matrix (dense trajectory) for a subject.
    ///
    /// The SDE [`Simulate`](crate::core::Simulate) output is a
    /// [`ParticleLikelihood`]; use this method when you need the particle
    /// prediction matrix itself (e.g. to inspect the mean trajectory).
    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &crate::Parameters,
    ) -> Result<Array2<Prediction>, PharmsolError> {
        let (predictions, _likelihood) = crate::core::standard_event_loop::<
            Self,
            Array2<Prediction>,
        >(self, subject, parameters.as_slice(), None)?;
        Ok(predictions)
    }
}

impl crate::core::Simulate for SDE {
    fn simulate_subject(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<Self::Predictions, PharmsolError> {
        let (_predictions, likelihood) = crate::core::standard_event_loop::<
            Self,
            Array2<Prediction>,
        >(self, subject, params, error_models)?;
        Ok(ParticleLikelihood::new(likelihood.unwrap_or(0.0)))
    }

    fn log_likelihood(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        // Use cached likelihood path
        _estimate_likelihood(self, subject, params, error_models)
    }

    fn kind() -> pharmsol_dsl::ModelKind {
        pharmsol_dsl::ModelKind::Sde
    }
}

#[inline(always)]
fn _estimate_likelihood(
    sde: &SDE,
    subject: &Subject,
    parameters: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    if let Some(cache) = sde.core.cache() {
        let key = (
            subject.hash(),
            parameters_hash(parameters),
            error_models.hash(),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let ypred =
            <SDE as Simulate>::simulate_subject(sde, subject, parameters, Some(error_models))?;
        let result = ypred.value();
        cache.insert(key, result);
        Ok(result)
    } else {
        let ypred =
            <SDE as Simulate>::simulate_subject(sde, subject, parameters, Some(error_models))?;
        Ok(ypred.value())
    }
}

/// Performs systematic resampling of particles based on weights.
///
/// # Arguments
///
/// * `q` - Vector of particle weights
///
/// # Returns
///
/// Vector of indices to use for resampling.
fn sysresample(q: &[f64]) -> Vec<usize> {
    let mut qc = vec![0.0; q.len()];
    qc[0] = q[0];
    for i in 1..q.len() {
        qc[i] = qc[i - 1] + q[i];
    }
    let m = q.len();
    let mut rng = rng();
    let u: Vec<f64> = (0..m)
        .map(|i| (i as f64 + rng.random::<f64>()) / m as f64)
        .collect();
    let mut i = vec![0; m];
    let mut k = 0;
    for j in 0..m {
        while qc[k] < u[j] {
            k += 1;
        }
        i[j] = k;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metadata::{Covariate, Route};
    use crate::SubjectBuilderExt;
    use crate::{fa, fetch_params, lag};

    fn simple_sde() -> SDE {
        let drift = |x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx[0] = rateiv[0] - x[0];
        };
        let diffusion = |_p: &V, g: &mut V| {
            g[0] = 1.0;
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x[0] = 0.0;
        };
        let out = |x: &V, p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        };

        SDE::new(drift, diffusion, lag, fa, init, out, 128)
            .with_nstates(1)
            .with_ndrugs(1)
            .with_nout(1)
    }

    fn route_policy_sde(drift: Drift) -> SDE {
        let diffusion = |_p: &V, sigma: &mut V| {
            sigma.fill(0.0);
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[1];
        };

        SDE::new(drift, diffusion, lag, fa, init, out, 16)
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
    }

    #[test]
    fn handwritten_sde_metadata_exposes_name_lookup_and_particles() {
        let sde = simple_sde()
            .with_metadata(
                crate::core::metadata::new("one_cmt_sde")
                    .parameters(["ke", "v"])
                    .covariates([Covariate::continuous("wt")])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect("SDE metadata attachment should validate");

        let metadata = sde.metadata().expect("metadata exists");
        assert_eq!(metadata.kind(), ModelKind::Sde);
        assert_eq!(metadata.particles(), Some(128));
        assert_eq!(sde.parameter_index("ke"), Some(0));
        assert_eq!(sde.parameter_index("v"), Some(1));
        assert_eq!(sde.covariate_index("wt"), Some(0));
        assert_eq!(sde.state_index("central"), Some(0));
        assert!(metadata.route("iv").is_some());
        assert!(metadata.output("cp").is_some());
    }

    #[test]
    fn handwritten_sde_metadata_resolves_raw_numeric_aliases_against_canonical_labels() {
        let drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let diffusion = |_p: &V, sigma: &mut V| {
            sigma.fill(0.0);
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[1];
        };

        let sde = SDE::new(drift, diffusion, lag, fa, init, out, 16)
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(
                crate::core::metadata::new("numeric_alias_sde")
                    .states(["depot", "central"])
                    .outputs(["outeq_1"])
                    .route(Route::infusion("input_1").to_state("central"))
                    .particles(16),
            )
            .expect("SDE metadata attachment should validate");

        let canonical = Subject::builder("canonical")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build();
        let aliased = Subject::builder("aliased")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "1")
            .build();

        let canonical_predictions = sde
            .estimate_predictions(&canonical, &crate::parameters::dense([]))
            .expect("canonical labels should simulate");
        let aliased_predictions = sde
            .estimate_predictions(&aliased, &crate::parameters::dense([]))
            .expect("raw numeric aliases should simulate");

        assert!(
            (canonical_predictions[[0, 0]].prediction() - aliased_predictions[[0, 0]].prediction())
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn handwritten_sde_without_metadata_keeps_raw_path() {
        let sde = simple_sde();

        assert!(sde.metadata().is_none());
        assert_eq!(sde.parameter_index("ke"), None);
    }

    #[test]
    fn handwritten_sde_rejects_dimension_mismatches() {
        let error = simple_sde()
            .with_metadata(
                crate::core::metadata::new("bad_sde")
                    .parameters(["ke", "v"])
                    .states(["central", "peripheral"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect_err("mismatched state metadata must fail");

        assert_eq!(
            error,
            SdeMetadataError::StateCountMismatch {
                expected: 1,
                declared: 2,
            }
        );
    }

    #[test]
    fn handwritten_sde_rejects_particle_mismatch() {
        let error = simple_sde()
            .with_metadata(
                crate::core::metadata::new("particle_conflict")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(64),
            )
            .expect_err("mismatched SDE particles must fail");

        assert_eq!(
            error,
            SdeMetadataError::Validation(ModelMetadataError::ParticleCountConflict {
                declared: 64,
                fallback: 128,
            })
        );
    }

    #[test]
    fn changing_dimensions_after_metadata_clears_sde_metadata() {
        let sde = simple_sde()
            .with_metadata(
                crate::core::metadata::new("one_cmt_sde")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect("metadata attachment should validate")
            .with_nout(2);

        assert!(sde.metadata().is_none());
    }

    #[test]
    fn sde_metadata_input_policy_is_descriptive_only_for_bolus_routes() {
        let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
        };

        let explicit = route_policy_sde(zero_drift)
            .with_metadata(
                crate::core::metadata::new("explicit_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::bolus("oral").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(zero_drift)
            .with_metadata(
                crate::core::metadata::new("injected_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::bolus("oral")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate");

        let subject = Subject::builder("bolus_route")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.1, "cp")
            .build();

        let explicit_predictions = explicit
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();
        let injected_predictions = injected
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert_eq!(explicit_predictions[[0, 0]].prediction(), 0.0);
        assert_eq!(injected_predictions[[0, 0]].prediction(), 0.0);
    }

    #[test]
    fn sde_metadata_input_policy_does_not_change_explicit_rateiv_behavior() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };

        let explicit = route_policy_sde(rateiv_drift)
            .with_metadata(
                crate::core::metadata::new("explicit_infusion")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(rateiv_drift)
            .with_metadata(
                crate::core::metadata::new("injected_infusion")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::infusion("iv")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate");

        let subject = Subject::builder("infusion_route")
            .infusion(0.0, 100.0, "iv", 1.0)
            .missing_observation(1.0, "cp")
            .build();

        let explicit_predictions = explicit
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();
        let injected_predictions = injected
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        let explicit_prediction = explicit_predictions[[0, 0]].prediction();
        let injected_prediction = injected_predictions[[0, 0]].prediction();

        assert!(explicit_prediction > 0.0);
        assert!((injected_prediction - explicit_prediction).abs() < 1e-8);
    }

    #[test]
    fn clearing_sde_metadata_preserves_raw_bolus_behavior() {
        let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
        };

        let sde = route_policy_sde(zero_drift)
            .with_metadata(
                crate::core::metadata::new("injected_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::bolus("oral")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate")
            .with_nout(1);

        let subject = Subject::builder("bolus_route")
            .bolus(0.0, 100.0, 0)
            .missing_observation(0.1, 0)
            .build();

        let predictions = sde
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert!(sde.metadata().is_none());
        assert_eq!(predictions[[0, 0]].prediction(), 0.0);
    }
}
