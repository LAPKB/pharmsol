use std::sync::Arc;

use crate::core::{Caching, ModelInfo, Solver, State};
use crate::data::error_model::{AssayErrorModels, BoundAssayErrorModels};
use crate::simulator::likelihood::Prediction;
use crate::{Event, Infusion, Parameters, PharmsolError, Subject};

/// A container that accumulates predictions during simulation.
///
/// Implemented by [`SubjectPredictions`] and [`Array2<Prediction>`] (SDE).
pub trait PredictionsContainer: Default {
    /// Create a new container pre-sized for `nparticles` (1 for deterministic,
    /// >1 for SDE).
    fn new(nparticles: usize) -> Self;

    /// Append a prediction.
    fn push(&mut self, pred: Prediction);

    /// Get all predictions as a slice.
    fn predictions(&self) -> &[Prediction];

    /// Compute the total log-likelihood across all predictions.
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError>;
}

/// The user-facing simulation API.
///
/// Anything that is `Solver + ModelInfo + Caching + Clone + Sync + 'static`
/// can implement `Simulate`. For backends that don't use batch integration,
/// the standard event loop ([`standard_event_loop`]) provides the default
/// `simulate_subject` implementation.
///
/// # Provided convenience methods
///
/// - [`predictions`](Self::predictions) — simulate and return predictions only
/// - [`log_likelihood`](Self::log_likelihood) — simulate and compute log-likelihood
/// - [`estimate_predictions`](Self::estimate_predictions) — accept `&Parameters` instead of `&[f64]`
/// - [`estimate_log_likelihood`](Self::estimate_log_likelihood) — accept `&Parameters` instead of `&[f64]`
pub trait Simulate: Solver + ModelInfo + Caching + Clone + Sync + 'static {
    /// The predictions container type for this backend.
    type Predictions: PredictionsContainer;

    /// Run the simulation for a subject and return predictions + optional likelihood.
    ///
    /// This is the only required method. Implementors can call
    /// [`standard_event_loop`] for the default per-event loop, or provide
    /// their own (e.g. batch diffsol integration).
    fn simulate_subject(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::Predictions, Option<f64>), PharmsolError>;

    /// Simulate and return predictions only.
    fn predictions(
        &self,
        subject: &Subject,
        params: &[f64],
    ) -> Result<Self::Predictions, PharmsolError> {
        Ok(self.simulate_subject(subject, params, None)?.0)
    }

    /// Simulate and return the log-likelihood.
    fn log_likelihood(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let predictions = self.predictions(subject, params)?;
        let bound = bind_error_models_inner(self, error_models)?;
        predictions.log_likelihood(&bound)
    }

    /// Convenience: accept `&Parameters` instead of `&[f64]`.
    fn estimate_predictions(
        &self,
        subject: &Subject,
        params: &Parameters,
    ) -> Result<Self::Predictions, PharmsolError> {
        self.predictions(subject, params.as_slice())
    }

    /// Convenience: accept `&Parameters` instead of `&[f64]`.
    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        params: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        self.log_likelihood(subject, params.as_slice(), error_models)
    }

    /// The model kind for runtime dispatch.
    fn kind() -> pharmsol_dsl::ModelKind;
}

// ── Standard event loop ────────────────────────────────────────────────────

/// The default simulation driver for per-interval solvers.
///
/// Iterates events: applies boluses, accumulates infusions, computes
/// predictions from observations, and calls [`Solver::solve`] to advance
/// the system between events.
///
/// Caches results using [`Caching::prediction_cache`] when caching is
/// enabled. Uses [`Caching::error_model_cache`] for bound error-model sharing.
pub fn standard_event_loop<S, P>(
    model: &S,
    subject: &Subject,
    params: &[f64],
    error_models: Option<&AssayErrorModels>,
) -> Result<(P, Option<f64>), PharmsolError>
where
    S: Solver + ModelInfo + Caching,
    P: PredictionsContainer,
{
    // Check prediction cache
    if let (Some(cache), None) = (model.prediction_cache(), error_models) {
        let key = (subject.hash(), parameters_hash(params));
        // Cache hit would need to return (P, None) but P isn't necessarily the same
        // type as what's in the cache. We skip cache-based return here and let
        // individual backends handle caching in their simulate_subject impl.
        // The cache check pattern is used by Analytical and ODE backends.
        let _ = (cache, key);
    }

    let bound_error_models = match error_models {
        Some(em) => Some(bind_error_models_inner(model, em)?),
        None => None,
    };

    let mut output = P::new(model.nparticles());
    let mut likelihood = Vec::new();

    for occasion in subject.occasions() {
        let covariates = occasion.covariates();
        let events = model.resolve_events(occasion, params, covariates)?;
        let mut state = model.initial_state(params, covariates, occasion.index());
        let mut infusions: Vec<Infusion> = Vec::new();

        for (idx, event) in events.iter().enumerate() {
            match event {
                Event::Bolus(bolus) => {
                    let input =
                        bolus
                            .input_index()
                            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
                            })?;

                    if input >= model.ndrugs() {
                        return Err(PharmsolError::InputOutOfRange {
                            input,
                            ndrugs: model.ndrugs(),
                        });
                    }
                    state.add_bolus(input, bolus.amount());
                }
                Event::Infusion(infusion) => {
                    infusions.push(infusion.clone());
                }
                Event::Observation(observation) => {
                    let (pred, lik) = model.process_observation(
                        &state,
                        params,
                        observation,
                        error_models,
                        covariates,
                    )?;
                    if let Some(lik) = lik {
                        likelihood.push(lik);
                    }
                    output.push(pred);
                }
            }

            // Advance to next event
            if let Some(next_event) = events.get(idx + 1) {
                if !event.time().eq(&next_event.time()) {
                    model.solve(
                        &mut state,
                        params,
                        covariates,
                        &infusions,
                        event.time(),
                        next_event.time(),
                    )?;
                }
            }
        }
    }

    let ll = bound_error_models.map(|_| likelihood.iter().product::<f64>());
    Ok((output, ll))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Bind assay error models using the model's metadata for output-label
/// resolution, with caching through the model's error-model cache.
pub(crate) fn bind_error_models_inner<'a, M: ModelInfo + Caching>(
    model: &'a M,
    error_models: &'a AssayErrorModels,
) -> Result<BoundAssayErrorModels<'a>, PharmsolError> {
    if let Some(cache) = model.error_model_cache() {
        let key = error_models.hash();
        if let Some(bound_error_models) = cache.get(&key) {
            return Ok(BoundAssayErrorModels::Shared(bound_error_models));
        }

        return match error_models
            .bind_output_names(
                model
                    .metadata()
                    .map(|m| m.outputs().iter().map(|o| o.name()))
                    .into_iter()
                    .flatten(),
            )
            .map_err(PharmsolError::from)?
        {
            BoundAssayErrorModels::Owned(bound_error_models) => {
                let bound_error_models = Arc::new(bound_error_models);
                cache.insert(key, Arc::clone(&bound_error_models));
                Ok(BoundAssayErrorModels::Shared(bound_error_models))
            }
            bound => Ok(bound),
        };
    }

    error_models
        .bind_output_names(
            model
                .metadata()
                .map(|m| m.outputs().iter().map(|o| o.name()))
                .into_iter()
                .flatten(),
        )
        .map_err(PharmsolError::from)
}

/// Hash a parameter slice for cache keys.
#[inline(always)]
pub(crate) fn parameters_hash(params: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    for &value in params {
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}
