mod em;

use diffsol::{NalgebraContext, Vector};
use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use pharmsol_dsl::ModelKind;
use rand::{rng, RngExt};
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::Prediction,
    simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Subject,
};

use super::spphash;
use crate::simulator::cache::{SdeLikelihoodCache, DEFAULT_CACHE_SIZE};

use diffsol::VectorCommon;

use crate::PharmsolError;

use super::{
    EqnKind, Equation, EquationPriv, EquationTypes, ModelMetadata, ModelMetadataError, Predictions,
    State, ValidatedModelMetadata,
};

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
/// * `support_point` - Parameter vector for the model
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
    support_point: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ndrugs: usize,
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }

    let params_v = V::from_vec(support_point.to_vec(), NalgebraContext);
    let covariates = cov.clone();
    let infusion_events = infusions.to_vec();
    let drift_fn = *drift;
    let diffusion_fn = *difussion;

    let params_for_drift = params_v.clone();
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
            &params_for_drift,
            time,
            &mut out_v,
            &rateiv,
            &covariates,
        );
        out.copy_from(out_v.inner());
    };

    let diffusion_closure = move |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| {
        let mut out_v = V::zeros(out.len(), NalgebraContext);
        diffusion_fn(&params_v, &mut out_v);
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
    drift: Drift,
    diffusion: Diffusion,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    nparticles: usize,
    metadata: Option<ValidatedModelMetadata>,
    injected_bolus_mappings: InjectedBolusMappings,
    cache: Option<SdeLikelihoodCache>,
}

impl SDE {
    /// Creates a new stochastic differential equation solver with default Neqs.
    ///
    /// Use builder methods to configure dimensions:
    /// ```ignore
    /// SDE::new(drift, diffusion, lag, fa, init, out, nparticles)
    ///     .with_nstates(2)
    ///     .with_ndrugs(1)
    ///     .with_nout(1)
    /// ```
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
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            nparticles,
            metadata: None,
            injected_bolus_mappings: InjectedBolusMappings::default(),
            cache: Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    /// Set the number of state variables.
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

    /// Attach validated handwritten-model metadata to this SDE model.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, SdeMetadataError> {
        let metadata = metadata.validate_for_with_particles(ModelKind::Sde, self.nparticles)?;
        validate_metadata_dimensions(&metadata, &self.neqs)?;
        self.metadata = Some(metadata);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn with_injected_bolus_inputs(mut self, destinations: &[Option<usize>]) -> Self {
        self.injected_bolus_mappings =
            InjectedBolusMappings::from_destinations(self.neqs.ndrugs, destinations);
        self
    }

    /// Access the validated metadata attached to this SDE model, if any.
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

    pub fn route_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.route_index(name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.output_index(name)
    }

    fn invalidate_metadata(&mut self) {
        self.metadata = None;
        self.injected_bolus_mappings
            .invalidate_for_ndrugs(self.neqs.ndrugs);
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

impl super::Cache for SDE {
    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(size));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

/// State trait implementation for particle-based SDE simulation.
///
/// This implementation allows adding bolus doses to all particles in the system.
impl State for Vec<DVector<f64>> {
    /// Adds a bolus dose to a specific input compartment across all particles.
    ///
    /// # Arguments
    ///
    /// * `input` - Index of the input compartment
    /// * `amount` - Amount to add to the compartment
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.par_iter_mut().for_each(|particle| {
            particle[input] += amount;
        });
    }
}

/// Predictions implementation for particle-based SDE simulation outputs.
///
/// This implementation manages and processes predictions from multiple particles.
impl Predictions for Array2<Prediction> {
    fn new(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }
    fn squared_error(&self) -> f64 {
        unimplemented!();
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        // Make this return the mean prediction across all particles
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
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, crate::PharmsolError> {
        // For SDE, compute log-likelihood using mean predictions across particles
        let predictions = self.get_predictions();
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

impl EquationTypes for SDE {
    type S = Vec<DVector<f64>>; // Vec -> particles, DVector -> state
    type P = Array2<Prediction>; // Rows -> particles, Columns -> time
}

impl EquationPriv for SDE {
    // #[inline(always)]
    // fn get_init(&self) -> &Init {
    //     &self.init
    // }

    // #[inline(always)]
    // fn get_out(&self) -> &Out {
    //     &self.out
    // }

    // #[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.lag)(&V::from_vec(spp.to_owned())))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.fa)(&V::from_vec(spp.to_owned())))
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
        state: &mut Self::S,
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        let ndrugs = self.get_ndrugs();
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone().into(),
                support_point,
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
    fn nparticles(&self) -> usize {
        self.nparticles
    }

    fn is_sde(&self) -> bool {
        true
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &[f64],
        observation: &crate::Observation,
        error_models: Option<&AssayErrorModels>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut pred = vec![Prediction::default(); self.nparticles];

        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
            (self.out)(
                &x[i].clone().into(),
                &V::from_vec(support_point.to_vec(), NalgebraContext),
                observation.time(),
                covariates,
                &mut y,
            );
            let outeq = observation
                .outeq_index()
                .expect("resolved observations should use numeric output labels");
            *p = observation.to_prediction(y[outeq], x[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.nparticles, 1), pred.clone())?;
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_models {
            let mut q: Vec<f64> = Vec::with_capacity(self.nparticles);

            pred.iter().for_each(|p| {
                let lik = p.log_likelihood(em).map(f64::exp);
                match lik {
                    Ok(l) => q.push(l),
                    Err(e) => panic!("Error in likelihood calculation: {:?}", e),
                }
            });
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let i = sysresample(&w);
            let a: Vec<DVector<f64>> = i.iter().map(|&i| x[i].clone()).collect();
            *x = a;
            likelihood.push(sum_q / self.nparticles as f64);
            // let qq: Vec<f64> = i.iter().map(|&i| q[i]).collect();
            // likelihood.push(qq.iter().sum::<f64>() / self.nparticles as f64);
        }
        Ok(())
    }
    #[inline(always)]
    fn initial_state(
        &self,
        support_point: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S {
        let mut x = Vec::with_capacity(self.nparticles);
        for _ in 0..self.nparticles {
            let mut state: V = DVector::zeros(self.get_nstates()).into();
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(support_point.to_vec(), NalgebraContext),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state.inner().clone());
        }
        x
    }

    fn simulate_event(
        &self,
        support_point: &[f64],
        event: &crate::Event,
        next_event: Option<&crate::Event>,
        error_models: Option<&AssayErrorModels>,
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        match event {
            crate::Event::Bolus(bolus) => {
                let input =
                    bolus
                        .input_index()
                        .ok_or_else(|| PharmsolError::UnknownInputLabel {
                            label: bolus.input().to_string(),
                        })?;

                if input >= self.get_ndrugs() {
                    return Err(PharmsolError::InputOutOfRange {
                        input,
                        ndrugs: self.get_ndrugs(),
                    });
                }
                if !self.injected_bolus_mappings.apply(x, input, bolus.amount()) {
                    x.add_bolus(input, bolus.amount());
                }
            }
            crate::Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            crate::Event::Observation(observation) => {
                self.process_observation(
                    support_point,
                    observation,
                    error_models,
                    event.time(),
                    covariates,
                    x,
                    likelihood,
                    output,
                )?;
            }
        }

        if let Some(next_event) = next_event {
            self.solve(
                x,
                support_point,
                covariates,
                infusions,
                event.time(),
                next_event.time(),
            )?;
        }
        Ok(())
    }
}

impl Equation for SDE {
    /// Estimates the likelihood of observed data given a model and parameters.
    ///
    /// # Arguments
    ///
    /// * `subject` - Subject data containing observations
    /// * `support_point` - Parameter vector for the model
    /// * `error_model` - Error model to use for likelihood calculations
    ///
    /// # Returns
    ///
    /// The log-likelihood of the observed data given the model and parameters.
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        // For SDE, the particle filter computes likelihood in regular space.
        // We compute it directly and then take the log.
        let lik = _estimate_likelihood(self, subject, support_point, error_models)?;

        if lik > 0.0 {
            Ok(lik.ln())
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }

    fn kind() -> EqnKind {
        EqnKind::SDE
    }
}

#[inline(always)]
fn _estimate_likelihood(
    sde: &SDE,
    subject: &Subject,
    support_point: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    if let Some(cache) = &sde.cache {
        let key = (subject.hash(), spphash(support_point), error_models.hash());
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let ypred = sde.simulate_subject(subject, support_point, Some(error_models))?;
        let result = ypred.1.unwrap();
        cache.insert(key, result);
        Ok(result)
    } else {
        let ypred = sde.simulate_subject(subject, support_point, Some(error_models))?;
        Ok(ypred.1.unwrap())
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
    use crate::simulator::equation::{self, Covariate, Route};
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
                equation::metadata::new("one_cmt_sde")
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
        assert_eq!(sde.route_index("iv"), Some(0));
        assert_eq!(sde.output_index("cp"), Some(0));
    }

    #[test]
    fn handwritten_sde_without_metadata_keeps_raw_path() {
        let sde = simple_sde();

        assert!(sde.metadata().is_none());
        assert_eq!(sde.parameter_index("ke"), None);
        assert_eq!(sde.route_index("iv"), None);
        assert_eq!(sde.output_index("cp"), None);
    }

    #[test]
    fn handwritten_sde_rejects_dimension_mismatches() {
        let error = simple_sde()
            .with_metadata(
                equation::metadata::new("bad_sde")
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
                equation::metadata::new("particle_conflict")
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
                equation::metadata::new("one_cmt_sde")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect("metadata attachment should validate")
            .with_nout(2);

        assert!(sde.metadata().is_none());
        assert_eq!(sde.route_index("iv"), None);
        assert_eq!(sde.output_index("cp"), None);
    }

    #[test]
    fn sde_metadata_input_policy_is_descriptive_only_for_bolus_routes() {
        let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
        };

        let explicit = route_policy_sde(zero_drift)
            .with_metadata(
                equation::metadata::new("explicit_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::bolus("oral").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(zero_drift)
            .with_metadata(
                equation::metadata::new("injected_bolus")
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

        let explicit_predictions = explicit.estimate_predictions(&subject, &[0.0]).unwrap();
        let injected_predictions = injected.estimate_predictions(&subject, &[0.0]).unwrap();

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
                equation::metadata::new("explicit_infusion")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(rateiv_drift)
            .with_metadata(
                equation::metadata::new("injected_infusion")
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

        let explicit_predictions = explicit.estimate_predictions(&subject, &[0.0]).unwrap();
        let injected_predictions = injected.estimate_predictions(&subject, &[0.0]).unwrap();

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
                equation::metadata::new("injected_bolus")
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

        let predictions = sde.estimate_predictions(&subject, &[0.0]).unwrap();

        assert!(sde.metadata().is_none());
        assert_eq!(predictions[[0, 0]].prediction(), 0.0);
    }
}
