//! Stochastic models simulated with particles.
//!
//! [`SDE`] pairs an [`SdeModel`] with a particle count, a random seed and a
//! likelihood cache. Between events every particle is advanced with adaptive
//! Euler-Maruyama steps. At each observation the particles are weighted by
//! the observation's likelihood and resampled (a bootstrap particle filter).
//!
//! The weights come from an [`ObservationModel`], which pharmsol does not
//! implement: the error models live downstream. [`SDE::predict`] needs none
//! and runs the particles free. [`SDE::log_likelihood`] returns the filter's
//! (cached) likelihood estimate, and [`SDE::filter`] returns the filtered
//! particles along with it.
//!
//! Simulation is deterministic: the random streams depend only on the seed,
//! the subject and the event being simulated, never on the parameters. Two
//! support points are therefore compared under the same noise, and repeated
//! calls return the same result.

mod em;
mod native;
mod observation;

pub use native::NativeSde;
pub use observation::ObservationModel;

use std::sync::Arc;

use diffsol::{NalgebraContext, Vector};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use em::EmScratch;

use super::model::{
    fill_active_rates, prepare_occasion, with_subject_context, InfusionDose, Model, Step,
};
use super::{parameters_hash, Cache, Equation, ValidatedModelMetadata};
use crate::data::event::Observation;
use crate::simulator::cache::{SdeLikelihoodCache, DEFAULT_CACHE_SIZE};
use crate::simulator::predictions::ParticlePredictions;
use crate::simulator::V;
use crate::{Covariates, Occasion, PharmsolError, Subject};

/// Model functions for an [`SDE`].
pub trait SdeModel: Model {
    /// Write the drift of `x` at `time` into `dx`, which is zeroed on entry.
    #[allow(clippy::too_many_arguments)]
    fn drift(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError>;

    /// Write the diffusion of each state at `time` into `sigma`, which is
    /// zeroed on entry. Each state receives independent noise.
    #[allow(clippy::too_many_arguments)]
    fn diffusion(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        sigma: &mut V,
    ) -> Result<(), PharmsolError>;
}

/// An [`SdeModel`] simulated with particles, with a likelihood cache.
///
/// The cache holds [`Self::log_likelihood`] estimates for observation models
/// whose [`ObservationModel::cache_key`] is `Some`. Caching is on by
/// default; see [`Cache`]. Clones share the cache.
#[derive(Clone, Debug)]
pub struct SDE<M = NativeSde> {
    model: M,
    nparticles: usize,
    seed: u64,
    cache: Option<SdeLikelihoodCache>,
}

impl<M: SdeModel> SDE<M> {
    /// Wrap a model with `nparticles` particles, the default seed and the
    /// default cache.
    pub fn from_model(model: M, nparticles: usize) -> Self {
        Self {
            model,
            nparticles,
            seed: 0,
            cache: Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    /// The model functions.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Mutable access to the model functions.
    ///
    /// Cached results were computed with the old model, so this equation
    /// switches to a fresh, empty cache of the same size (clones keep theirs).
    pub(crate) fn model_mut(&mut self) -> &mut M {
        self.cache = self.cache.as_ref().map(|cache| cache.emptied());
        &mut self.model
    }

    /// Number of particles.
    pub fn nparticles(&self) -> usize {
        self.nparticles
    }

    /// The random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Swap in a model with a different particle count, keeping the seed and
    /// the (shared) cache. Cache keys include the particle count.
    #[cfg(feature = "dsl")]
    pub(crate) fn with_model(mut self, model: M, nparticles: usize) -> Self {
        self.model = model;
        self.nparticles = nparticles;
        self
    }

    /// Use a different random seed.
    ///
    /// Results are deterministic for a given seed; change it to draw an
    /// independent set of particle paths.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.state_index(name)
    }

    /// Simulate a subject's particles; see [`Mode`].
    fn run_particles<O: ObservationModel + ?Sized>(
        &self,
        subject: &Subject,
        parameters: &[f64],
        mode: Mode<'_, O>,
    ) -> Result<ParticleRun, PharmsolError> {
        let in_context = |error| with_subject_context(&self.model, error, subject.id(), parameters);
        if self.nparticles == 0 {
            return Err(PharmsolError::OtherError(
                "an SDE needs at least one particle".to_string(),
            ));
        }
        self.model
            .check_parameters(parameters)
            .map_err(in_context)?;

        let p = V::from_vec(parameters.to_vec(), NalgebraContext::new());
        let mut run = ParticleRun {
            observations: Vec::new(),
            values: Vec::new(),
            weights: Vec::new(),
            log_likelihood: 0.0,
            streams: RandomStreams::new(self.seed, subject.hash()),
        };
        for occasion in subject.occasions() {
            self.simulate_occasion(occasion, &p, mode, &mut run)
                .map_err(in_context)?;
            if mode.stops_at_zero_likelihood() && run.log_likelihood == f64::NEG_INFINITY {
                break;
            }
        }
        Ok(run)
    }

    fn simulate_occasion<O: ObservationModel + ?Sized>(
        &self,
        occasion: &Occasion,
        p: &V,
        mode: Mode<'_, O>,
        run: &mut ParticleRun,
    ) -> Result<(), PharmsolError> {
        let covariates = occasion.covariates();
        let mut ws = self.model.workspace();
        let schedule = prepare_occasion(&self.model, &mut ws, occasion, p)?;
        let mut initial = V::zeros(self.model.nstates(), NalgebraContext::new());
        self.model.init(&mut ws, 0.0, p, covariates, &mut initial)?;
        let mut particles = vec![initial; self.nparticles];
        let mut rates = V::zeros(self.model.ninputs(), NalgebraContext::new());
        let mut y = V::zeros(self.model.noutputs(), NalgebraContext::new());

        for (index, step) in schedule.steps.iter().enumerate() {
            match *step {
                Step::Bolus {
                    time,
                    input,
                    amount,
                } => {
                    for particle in &mut particles {
                        self.model
                            .bolus(&mut ws, time, particle, p, covariates, input, amount)?;
                    }
                }
                // Infusions enter through the rates the drift receives.
                Step::Infusion(_) => {}
                Step::Observation { observation, outeq } => {
                    fill_active_rates(&schedule.infusions, observation.time(), &mut rates);
                    let first = run.values.len();
                    for particle in &particles {
                        y.fill(0.0);
                        self.model.outputs(
                            &mut ws,
                            observation.time(),
                            particle,
                            p,
                            covariates,
                            &rates,
                            &mut y,
                        )?;
                        run.values.push(y[outeq]);
                    }

                    // Particles stay equally weighted unless the filter runs.
                    let mut weights = None;
                    let observation_model = mode
                        .observation_model()
                        .filter(|_| run.log_likelihood > f64::NEG_INFINITY);
                    if let Some(observation_model) = observation_model {
                        let column = &run.values[first..];
                        let rng = &mut run.streams.next();
                        match weigh_and_resample(observation, column, observation_model, rng)? {
                            Some(filtered) => {
                                run.log_likelihood += filtered.log_mean;
                                particles = filtered
                                    .indices
                                    .iter()
                                    .map(|&i| particles[i].clone())
                                    .collect();
                                weights = Some(filtered.weights);
                            }
                            None => run.log_likelihood = f64::NEG_INFINITY,
                        }
                    }
                    if mode.keeps_predictions() {
                        run.observations.push(observation.clone());
                        if mode.observation_model().is_some() {
                            match weights {
                                Some(weights) => run.weights.extend(weights),
                                None => run.weights.extend(std::iter::repeat_n(
                                    1.0 / self.nparticles as f64,
                                    self.nparticles,
                                )),
                            }
                        }
                    } else {
                        run.values.truncate(first);
                    }
                    if mode.stops_at_zero_likelihood() && run.log_likelihood == f64::NEG_INFINITY {
                        return Ok(());
                    }
                }
            }

            if let Some(next) = schedule.steps.get(index + 1) {
                self.advance(
                    &mut particles,
                    p,
                    covariates,
                    &schedule.infusions,
                    step.time(),
                    next.time(),
                    run.streams.next_seed(),
                )?;
            }
        }
        Ok(())
    }

    /// Advance every particle from `start` to `end`, in parallel.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        particles: &mut [V],
        p: &V,
        covariates: &Covariates,
        infusions: &[InfusionDose],
        start: f64,
        end: f64,
        seed: u64,
    ) -> Result<(), PharmsolError> {
        if start == end {
            return Ok(());
        }

        let model = &self.model;
        particles.par_iter_mut().enumerate().try_for_each_init(
            || ParticleScratch::new(model),
            |scratch, (index, particle)| {
                let ParticleScratch { ws, rates, em } = scratch;
                let mut rng = StdRng::seed_from_u64(mix(seed, index as u64));
                // Drift and diffusion are evaluated one after the other, never
                // at once, so they can share the workspace and rates.
                let shared = std::cell::RefCell::new((ws, rates));
                let mut drift = |time: f64, x: &V, dx: &mut V| {
                    let (ws, rates) = &mut *shared.borrow_mut();
                    fill_active_rates(infusions, time, rates);
                    model.drift(ws, time, x, p, covariates, rates, dx)
                };
                let mut diffusion = |time: f64, x: &V, sigma: &mut V| {
                    let (ws, rates) = &mut *shared.borrow_mut();
                    fill_active_rates(infusions, time, rates);
                    model.diffusion(ws, time, x, p, covariates, rates, sigma)
                };
                em::advance(
                    &mut drift,
                    &mut diffusion,
                    particle,
                    start,
                    end,
                    &mut rng,
                    em,
                )
            },
        )
    }

    /// Particle-filter estimate of the log-likelihood of the subject's
    /// observations under `observation_model`.
    ///
    /// The estimate is deterministic for a given seed. It is cached when
    /// caching is enabled and [`ObservationModel::cache_key`] returns `Some`.
    ///
    /// Errors before simulating if `observation_model` names an output this
    /// equation does not declare or fails [`ObservationModel::validate`].
    /// The filter stops early, returning negative infinity, at the first
    /// observation that rules out every particle.
    pub fn log_likelihood<O: ObservationModel + ?Sized>(
        &self,
        subject: &Subject,
        parameters: &[f64],
        observation_model: &O,
    ) -> Result<f64, PharmsolError> {
        self.validate_observation_model(observation_model)?;
        let simulate = || {
            self.run_particles(subject, parameters, Mode::Score(observation_model))
                .map(|run| run.log_likelihood)
        };
        match (&self.cache, observation_model.cache_key()) {
            (Some(cache), Some(observation_key)) => {
                let key = (
                    mix(mix(subject.hash(), self.seed), self.nparticles as u64),
                    parameters_hash(parameters),
                    observation_key,
                );
                cache.get_or_try_insert(key, simulate)
            }
            _ => simulate(),
        }
    }

    /// One particle-filter run through the whole subject: the filtered
    /// particle predictions with their weights, and the log-likelihood
    /// estimate they produced (the same value as [`Self::log_likelihood`]).
    ///
    /// Uncached. Every observation gets predictions, even after one ruled
    /// out every particle; from then on the particles are no longer
    /// filtered and carry uniform weights.
    pub fn filter<O: ObservationModel + ?Sized>(
        &self,
        subject: &Subject,
        parameters: &[f64],
        observation_model: &O,
    ) -> Result<FilteredParticles, PharmsolError> {
        self.validate_observation_model(observation_model)?;
        let run = self.run_particles(
            subject,
            parameters,
            Mode::ScoreAndPredict(observation_model),
        )?;
        let log_likelihood = run.log_likelihood;
        Ok(FilteredParticles {
            predictions: run.into_predictions(self.nparticles)?,
            log_likelihood,
        })
    }

    /// Check an observation model's own settings and that every label it
    /// covers names one of this equation's outputs.
    fn validate_observation_model<O: ObservationModel + ?Sized>(
        &self,
        observation_model: &O,
    ) -> Result<(), PharmsolError> {
        observation_model.validate()?;
        let metadata = self.model.metadata();
        for label in observation_model.labels() {
            let declared = match metadata {
                Some(metadata) => metadata.output_for_label(label.as_str()).is_some(),
                None => label
                    .index()
                    .is_some_and(|slot| slot < self.model.noutputs()),
            };
            if !declared {
                let available = metadata.map(|metadata| metadata.output_labels());
                return Err(PharmsolError::unknown_output_label(
                    label,
                    available.as_deref().unwrap_or_default(),
                ));
            }
        }
        Ok(())
    }
}

/// The result of [`SDE::filter`]: one particle-filter run.
#[derive(Debug, Clone)]
pub struct FilteredParticles {
    /// The filtered particles with their normalized weights (see
    /// [`ParticlePredictions`]).
    pub predictions: ParticlePredictions,
    /// The run's particle-filter estimate of the log-likelihood; negative
    /// infinity when some observation ruled out every particle.
    pub log_likelihood: f64,
}

/// What a particle run computes.
enum Mode<'a, O: ?Sized> {
    /// Particles evolve independently; no likelihood.
    Predict,
    /// Run the particle filter (resample at every observation) for the
    /// likelihood only. Stops at the first observation no particle explains.
    Score(&'a O),
    /// Run the particle filter and keep every prediction and weight, through
    /// the whole subject.
    ScoreAndPredict(&'a O),
}

impl<O: ?Sized> Clone for Mode<'_, O> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<O: ?Sized> Copy for Mode<'_, O> {}

impl<'a, O: ?Sized> Mode<'a, O> {
    fn observation_model(self) -> Option<&'a O> {
        match self {
            Mode::Predict => None,
            Mode::Score(model) | Mode::ScoreAndPredict(model) => Some(model),
        }
    }

    fn keeps_predictions(self) -> bool {
        !matches!(self, Mode::Score(_))
    }

    fn stops_at_zero_likelihood(self) -> bool {
        matches!(self, Mode::Score(_))
    }
}

/// Buffers one rayon worker reuses across the particles it advances.
struct ParticleScratch<W> {
    ws: W,
    rates: V,
    em: EmScratch,
}

impl<W> ParticleScratch<W> {
    fn new<M: Model<Workspace = W>>(model: &M) -> Self {
        Self {
            ws: model.workspace(),
            rates: V::zeros(model.ninputs(), NalgebraContext::new()),
            em: EmScratch::new(model.nstates()),
        }
    }
}

/// The outcome of simulating one subject.
struct ParticleRun {
    /// The observations simulated (empty in [`Mode::Score`]).
    observations: Vec<Observation>,
    /// Predicted values in observation order, `nparticles` per observation
    /// (empty in [`Mode::Score`]).
    values: Vec<f64>,
    /// Normalized filter weights, laid out like `values` (filled only in
    /// [`Mode::ScoreAndPredict`]).
    weights: Vec<f64>,
    log_likelihood: f64,
    streams: RandomStreams,
}

impl ParticleRun {
    /// Particle predictions with one row per particle and one column per
    /// observation.
    fn into_predictions(self, nparticles: usize) -> Result<ParticlePredictions, PharmsolError> {
        let shape = (self.observations.len(), nparticles);
        let by_particle = |values: Vec<f64>| -> Result<Array2<f64>, PharmsolError> {
            Ok(Array2::from_shape_vec(shape, values)?
                .reversed_axes()
                .as_standard_layout()
                .into_owned())
        };
        let weights = match self.weights.is_empty() {
            true => None,
            false => Some(by_particle(self.weights)?),
        };
        ParticlePredictions::new(self.observations, by_particle(self.values)?, weights)
    }
}

/// One observation's particles after weighting.
struct Filtered {
    /// Log of the mean particle likelihood.
    log_mean: f64,
    /// Normalized particle weights.
    weights: Vec<f64>,
    /// Indices of the resampled particles.
    indices: Vec<usize>,
}

/// Weight one observation's particles and resample them.
///
/// Returns `None` when every particle has zero likelihood.
fn weigh_and_resample<O: ObservationModel + ?Sized>(
    observation: &Observation,
    values: &[f64],
    observation_model: &O,
    rng: &mut StdRng,
) -> Result<Option<Filtered>, PharmsolError> {
    let mut weights = Vec::with_capacity(values.len());
    let mut max_log_weight = f64::NEG_INFINITY;
    for &value in values {
        // A missing observation leaves every particle equally likely.
        let log_density = match observation.value() {
            None => Ok(0.0),
            Some(observed) => observation_model.log_density(
                observation.outeq(),
                observed,
                value,
                observation.censoring(),
                observation.errorpoly(),
            ),
        };
        // A particle the observation cannot have come from (including one
        // whose state diverged to NaN) gets zero weight. A non-finite density
        // returned as `Ok` is treated the same as `NonFiniteLikelihood`, so an
        // implementation that skips the check cannot turn the weights into NaN.
        let log_weight = match log_density {
            Ok(log_density) if log_density.is_finite() => log_density,
            Ok(_) | Err(PharmsolError::NonFiniteLikelihood(_)) => f64::NEG_INFINITY,
            Err(error) => return Err(error),
        };
        max_log_weight = max_log_weight.max(log_weight);
        weights.push(log_weight);
    }
    if !max_log_weight.is_finite() {
        return Ok(None);
    }
    // Shift before exponentiating to keep normalization in range.
    weights
        .iter_mut()
        .for_each(|weight| *weight = (*weight - max_log_weight).exp());
    let total: f64 = weights.iter().sum();
    let log_mean = max_log_weight + (total / weights.len() as f64).ln();
    weights.iter_mut().for_each(|weight| *weight /= total);
    let indices = systematic_resample(&weights, rng);
    Ok(Some(Filtered {
        log_mean,
        weights,
        indices,
    }))
}

/// Systematic resampling: indices drawn from `weights` (summing to one) with
/// a single uniform offset.
fn systematic_resample(weights: &[f64], rng: &mut StdRng) -> Vec<usize> {
    let n = weights.len();
    let offset: f64 = rng.random();
    let mut indices = Vec::with_capacity(n);
    let mut cumulative = weights[0];
    let mut k = 0;
    for j in 0..n {
        let u = (j as f64 + offset) / n as f64;
        while cumulative < u && k + 1 < n {
            k += 1;
            cumulative += weights[k];
        }
        indices.push(k);
    }
    indices
}

/// Independent random streams for one subject, one per simulation step.
struct RandomStreams {
    base: u64,
    step: u64,
}

impl RandomStreams {
    fn new(seed: u64, subject: u64) -> Self {
        Self {
            base: mix(seed, subject),
            step: 0,
        }
    }

    fn next_seed(&mut self) -> u64 {
        self.step += 1;
        mix(self.base, self.step)
    }

    fn next(&mut self) -> StdRng {
        StdRng::seed_from_u64(self.next_seed())
    }
}

/// Combine two values into a well-mixed seed (SplitMix64 finalizer).
fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl<M: SdeModel> Cache for SDE<M> {
    fn with_cache_capacity(mut self, size: usize) -> Self {
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

impl<M: SdeModel> Equation for SDE<M> {
    type P = ParticlePredictions;

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata()
    }

    fn nstates(&self) -> usize {
        self.model.nstates()
    }

    fn nouteqs(&self) -> usize {
        self.model.noutputs()
    }

    /// Free-running particle predictions: each particle follows its own path
    /// through every observation. Uncached.
    fn predict_shared(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Arc<ParticlePredictions>, PharmsolError> {
        self.run_particles::<dyn ObservationModel>(subject, parameters, Mode::Predict)?
            .into_predictions(self.nparticles)
            .map(Arc::new)
    }
}

#[cfg(test)]
pub(crate) mod test_observation;
#[cfg(test)]
mod tests;
