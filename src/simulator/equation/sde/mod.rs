//! Stochastic models simulated with particles.
//!
//! [`SDE`] pairs an [`SdeModel`] with a particle count, a random seed and a
//! likelihood cache. Between events every particle is advanced with adaptive
//! Euler-Maruyama steps. At each observation the particles are weighted by
//! the observation's likelihood and resampled (a bootstrap particle filter).
//!
//! Simulation is deterministic: the random streams depend only on the seed,
//! the subject and the event being simulated, never on the parameters. Two
//! support points are therefore compared under the same noise, and repeated
//! calls return the same result.

mod em;
mod native;

pub use native::NativeSde;

use diffsol::{NalgebraContext, Vector, VectorHost};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use em::EmScratch;

use super::model::{fill_active_rates, infusions, prepare_occasion, with_subject_context, Model};
use super::{parameters_hash, Cache, Equation, Predictions, ValidatedModelMetadata};
use crate::data::error_model::AssayErrorModels;
use crate::prelude::simulator::Prediction;
use crate::simulator::cache::{SdeLikelihoodCache, DEFAULT_CACHE_SIZE};
use crate::simulator::V;
use crate::{Covariates, Event, Infusion, Occasion, PharmsolError, Subject};

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
/// Caching is on by default; see [`Cache`]. Clones share the cache.
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

    /// Number of particles.
    pub fn nparticles(&self) -> usize {
        self.nparticles
    }

    /// The random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn cache_enabled(&self) -> bool {
        self.cache.is_some()
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

    /// Simulate a subject's particles.
    ///
    /// With error models, run the particle filter: resample at every
    /// observation and return the log-likelihood estimate. Without, the
    /// particles evolve independently and the log-likelihood is zero.
    fn simulate(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<ParticleRun, PharmsolError> {
        let in_context = |error| with_subject_context(&self.model, error, subject.id(), parameters);
        self.model
            .check_parameters(parameters)
            .map_err(in_context)?;

        let p = V::from_vec(parameters.to_vec(), NalgebraContext::new());
        let mut run = ParticleRun {
            predictions: Vec::new(),
            log_likelihood: 0.0,
            streams: RandomStreams::new(self.seed, subject.hash()),
        };
        for occasion in subject.occasions() {
            self.simulate_occasion(occasion, &p, error_models, &mut run)
                .map_err(in_context)?;
            if run.log_likelihood == f64::NEG_INFINITY {
                break;
            }
        }
        Ok(run)
    }

    fn simulate_occasion(
        &self,
        occasion: &Occasion,
        p: &V,
        error_models: Option<&AssayErrorModels>,
        run: &mut ParticleRun,
    ) -> Result<(), PharmsolError> {
        let covariates = occasion.covariates();
        let mut ws = self.model.workspace();
        let events = prepare_occasion(&self.model, &mut ws, occasion, p)?;
        let infusions = infusions(&events);
        let mut initial = V::zeros(self.model.nstates(), NalgebraContext::new());
        self.model.init(&mut ws, 0.0, p, covariates, &mut initial)?;
        let mut particles = vec![initial; self.nparticles];
        let mut rates = V::zeros(self.model.ninputs(), NalgebraContext::new());
        let mut y = V::zeros(self.model.noutputs(), NalgebraContext::new());

        for (index, event) in events.iter().enumerate() {
            match event {
                Event::Bolus(bolus) => {
                    let input = bolus
                        .input_index()
                        .expect("prepared boluses carry a dense input index");
                    for particle in &mut particles {
                        self.model.bolus(
                            &mut ws,
                            event.time(),
                            particle,
                            p,
                            covariates,
                            input,
                            bolus.amount(),
                        )?;
                    }
                }
                // Infusions enter through the rates the drift receives.
                Event::Infusion(_) => {}
                Event::Observation(observation) => {
                    fill_active_rates(&infusions, observation.time(), &mut rates);
                    let outeq = observation
                        .outeq_slot()
                        .expect("prepared observations carry an output slot");
                    let first = run.predictions.len();
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
                        run.predictions.push(
                            observation.to_prediction(y[outeq], particle.as_slice().to_vec()),
                        );
                    }

                    if let Some(error_models) = error_models {
                        let column = &run.predictions[first..];
                        let resampled = filter(column, error_models, &mut run.streams.next())?;
                        match resampled {
                            Some((log_mean, indices)) => {
                                run.log_likelihood += log_mean;
                                particles = indices.iter().map(|&i| particles[i].clone()).collect();
                            }
                            None => {
                                run.log_likelihood = f64::NEG_INFINITY;
                                return Ok(());
                            }
                        }
                    }
                }
            }

            if let Some(next_event) = events.get(index + 1) {
                self.advance(
                    &mut particles,
                    p,
                    covariates,
                    &infusions,
                    event.time(),
                    next_event.time(),
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
        infusions: &[Infusion],
        start: f64,
        end: f64,
        seed: u64,
    ) -> Result<(), PharmsolError> {
        if start == end {
            return Ok(());
        }

        particles
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(index, particle)| {
                let model = &self.model;
                let mut ws = model.workspace();
                let mut rates = V::zeros(model.ninputs(), NalgebraContext::new());
                let mut rng = StdRng::seed_from_u64(mix(seed, index as u64));
                let mut scratch = EmScratch::new(model.nstates());
                // Drift and diffusion are evaluated one after the other, never
                // at once, so they can share the workspace and rates.
                let shared = std::cell::RefCell::new((&mut ws, &mut rates));
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
                    &mut scratch,
                )
            })
    }

    /// Cached particle-filter log-likelihood.
    fn log_likelihood(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let Some(cache) = &self.cache else {
            return Ok(self
                .simulate(subject, parameters, Some(error_models))?
                .log_likelihood);
        };
        let key = (
            mix(subject.hash(), self.seed),
            parameters_hash(parameters),
            error_models.hash(),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }
        let log_likelihood = self
            .simulate(subject, parameters, Some(error_models))?
            .log_likelihood;
        cache.insert(key, log_likelihood);
        Ok(log_likelihood)
    }
}

/// The outcome of simulating one subject.
struct ParticleRun {
    /// Predictions in observation order, `nparticles` per observation.
    predictions: Vec<Prediction>,
    log_likelihood: f64,
    streams: RandomStreams,
}

impl ParticleRun {
    /// Particle predictions with one row per particle and one column per
    /// observation.
    fn into_matrix(self, nparticles: usize) -> Result<Array2<Prediction>, PharmsolError> {
        let observations = self.predictions.len() / nparticles.max(1);
        let by_observation = Array2::from_shape_vec((observations, nparticles), self.predictions)?;
        Ok(by_observation
            .reversed_axes()
            .as_standard_layout()
            .into_owned())
    }
}

/// Weight one observation's particles and resample them.
///
/// Returns the log of the mean particle likelihood and the indices of the
/// resampled particles, or `None` when every particle has zero likelihood.
fn filter(
    predictions: &[Prediction],
    error_models: &AssayErrorModels,
    rng: &mut StdRng,
) -> Result<Option<(f64, Vec<usize>)>, PharmsolError> {
    let mut weights = Vec::with_capacity(predictions.len());
    for prediction in predictions {
        weights.push(prediction.log_likelihood(error_models)?.exp());
    }
    let total: f64 = weights.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return Ok(None);
    }
    let log_mean = (total / weights.len() as f64).ln();
    weights.iter_mut().for_each(|weight| *weight /= total);
    Ok(Some((log_mean, systematic_resample(&weights, rng))))
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

impl Predictions for Array2<Prediction> {
    /// Mean prediction across particles, one per observation.
    fn get_predictions(&self) -> Vec<Prediction> {
        if self.is_empty() || self.ncols() == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.ncols());
        for column in self.columns() {
            let mean = column.iter().map(Prediction::prediction).sum::<f64>() / self.nrows() as f64;
            let mut prediction = column[0].clone();
            prediction.set_prediction(mean);
            result.push(prediction);
        }
        result
    }

    /// Log-likelihood of the mean prediction across particles.
    ///
    /// This scores the particle cloud as if it were deterministic. For the
    /// particle-filter estimate, use [`Equation::estimate_log_likelihood`].
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        self.get_predictions()
            .iter()
            .filter(|prediction| prediction.observation().is_some())
            .map(|prediction| prediction.log_likelihood(error_models))
            .sum()
    }
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
    /// One row per particle, one column per observation.
    type P = Array2<Prediction>;

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata()
    }

    fn nstates(&self) -> usize {
        self.model.nstates()
    }

    fn nouteqs(&self) -> usize {
        self.model.noutputs()
    }

    /// Particle predictions without filtering: each particle follows its own
    /// path through every observation.
    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Array2<Prediction>, PharmsolError> {
        self.simulate(subject, parameters, None)?
            .into_matrix(self.nparticles)
    }

    /// Particle-filter estimate of the log-likelihood.
    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        self.validate_error_models(error_models)?;
        self.log_likelihood(subject, parameters, error_models)
    }
}

#[cfg(test)]
mod tests;
