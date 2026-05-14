//! Handwritten equation families and their shared simulation interfaces.
//!
//! This module is the public home for handwritten [`ODE`], [`Analytical`], and
//! [`SDE`] models, plus the shared [`Equation`] trait and the metadata types
//! that attach public names to parameters, states, routes, and outputs.
//!
//! Use this module when you want to:
//! - choose between deterministic ODE, analytical, and stochastic SDE models
//! - attach metadata so dataset labels such as `"iv"` and `"cp"` resolve by
//!   name instead of by dense numeric index
//! - work with prediction or likelihood APIs across equation families
//!
//! # Equation Families
//!
//! - [`ODE`] for deterministic models that must be numerically integrated.
//! - [`Analytical`] for supported closed-form models.
//! - [`SDE`] for stochastic models that use particles.
//!
//! # Labels And Metadata
//!
//! Input and output labels arrive from public data APIs as strings.
//!
//! - Without metadata, handwritten models fall back to numeric labels such as
//!   `0` or `1`.
//! - With [`metadata::ModelMetadata`] attached, route and output labels are
//!   resolved by name against the declared routes and outputs before
//!   simulation.
//!
//! That label-first path is the preferred public workflow for current authoring.
//!
//! # Example
//!
//! ```rust
//! use pharmsol::{metadata, ModelKind};
//!
//! let metadata = metadata::new("one_cmt")
//!     .kind(ModelKind::Ode)
//!     .parameters(["cl", "v"])
//!     .states(["central"])
//!     .outputs(["cp"])
//!     .route(metadata::Route::infusion("iv").to_state("central"))
//!     .validate()
//!     .unwrap();
//!
//! assert_eq!(metadata.route("iv").unwrap().destination(), "central");
//! assert!(metadata.output("cp").is_some());
//! ```

use std::{fmt::Debug, sync::Arc};
pub mod analytical;
pub mod metadata;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use metadata::*;
pub use ode::*;
pub use pharmsol_dsl::{AnalyticalKernel, ModelKind};
use pharmsol_dsl::{NUMERIC_OUTPUT_PREFIX, NUMERIC_ROUTE_PREFIX};
pub use sde::*;

use crate::{
    error_model::{AssayErrorModels, BoundAssayErrorModels},
    simulator::{cache::BoundErrorModelCache, Fa, Lag},
    Covariates, Event, Infusion, InputLabel, Observation, Occasion, OutputLabel, Parameters,
    PharmsolError, Subject,
};

use super::likelihood::Prediction;

/// Trait for state vectors that can receive bolus doses.
pub trait State {
    /// Add a bolus dose to the state at the specified resolved input index.
    ///
    /// # Parameters
    /// - `input`: The resolved dense input index used by the execution layer
    /// - `amount`: The bolus amount
    fn add_bolus(&mut self, input: usize, amount: f64);
}

/// Trait for prediction containers.
pub trait Predictions: Default {
    /// Create a new prediction container with specified capacity.
    ///
    /// # Parameters
    /// - `nparticles`: Number of particles (for SDE)
    ///
    /// # Returns
    /// A new predictions container
    fn new(_nparticles: usize) -> Self {
        Default::default()
    }

    /// Calculate the sum of squared errors for all predictions.
    ///
    /// # Returns
    /// The sum of squared errors
    fn squared_error(&self) -> f64;

    /// Get all predictions as a vector.
    ///
    /// # Returns
    /// Vector of prediction objects
    fn get_predictions(&self) -> Vec<Prediction>;

    /// Calculate the log-likelihood of the predictions given an error model.
    ///
    /// This is numerically more stable than computing the likelihood and taking its log,
    /// especially for extreme values or many observations.
    ///
    /// # Parameters
    /// - `error_models`: The error models for computing observation variance
    ///
    /// # Returns
    /// The sum of log-likelihoods for all predictions
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError>;
}

/// Trait for enabling prediction caching on equation types.
///
/// Caching is **enabled by default** with a capacity of 100,000 entries.
/// Use these methods to adjust capacity, clear entries, or disable caching.
///
/// # Example
/// ```ignore
/// use pharmsol::*;
///
/// // Caching is on by default:
/// let ode = ODE::new(diffeq, lag, fa, init, out);
///
/// // Adjust capacity:
/// let ode = ODE::new(diffeq, lag, fa, init, out)
///     .with_cache_capacity(50_000);
///
/// // Disable caching:
/// let ode = ODE::new(diffeq, lag, fa, init, out)
///     .disable_cache();
/// ```
pub trait Cache: Sized {
    /// Enable caching with the given maximum number of entries.
    ///
    /// When caching is enabled, results for the same inputs are stored and reused.
    /// Cloned equations share the same cache.
    ///
    /// If caching is already enabled, this **replaces** the cache with a new, empty
    /// one of the given size — all previously cached entries are discarded.
    fn with_cache_capacity(self, size: u64) -> Self;

    /// Enable caching with the default size (100,000 entries).
    ///
    /// If caching is already enabled, this **replaces** the cache with a new,
    /// empty one — all previously cached entries are discarded.
    fn enable_cache(self) -> Self;

    /// Clear all entries from this equation's cache, if caching is enabled.
    ///
    /// The cache itself remains active (with the same capacity).
    /// Does nothing if caching is not enabled.
    fn clear_cache(&self);

    /// Disable caching.
    ///
    /// Disables caching and discards all cached entries.
    fn disable_cache(self) -> Self;
}

/// Associated state and prediction container types for an equation family.
pub trait EquationTypes {
    /// The state vector type
    type S: State + Debug;
    /// The predictions container type
    type P: Predictions;
}

pub(crate) trait EquationPriv: EquationTypes {
    // fn get_init(&self) -> &Init;
    // fn get_out(&self) -> &Out;
    fn lag(&self) -> &Lag;
    fn fa(&self) -> &Fa;
    fn get_nstates(&self) -> usize;
    fn get_ndrugs(&self) -> usize;
    fn get_nouteqs(&self) -> usize;
    fn metadata(&self) -> Option<&ValidatedModelMetadata>;
    fn solve(
        &self,
        state: &mut Self::S,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError>;
    fn nparticles(&self) -> usize {
        1
    }

    fn resolve_input_label(
        &self,
        label: &InputLabel,
        expected_kind: RouteKind,
    ) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            let route = metadata
                .route(label.as_str())
                .or_else(|| {
                    canonical_numeric_alias(label.as_str(), NUMERIC_ROUTE_PREFIX)
                        .and_then(|alias| metadata.route(alias.as_str()))
                })
                .ok_or_else(|| PharmsolError::UnknownInputLabel {
                    label: label.to_string(),
                })?;

            if route.kind() != expected_kind {
                return Err(PharmsolError::UnsupportedInputRouteKind {
                    input: route.input_index(),
                    kind: match expected_kind {
                        RouteKind::Bolus => pharmsol_dsl::RouteKind::Bolus,
                        RouteKind::Infusion => pharmsol_dsl::RouteKind::Infusion,
                    },
                });
            }

            return Ok(route.input_index());
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                label: label.to_string(),
            })
    }

    fn resolve_output_label(&self, label: &OutputLabel) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            return metadata
                .output_index(label.as_str())
                .or_else(|| {
                    canonical_numeric_alias(label.as_str(), NUMERIC_OUTPUT_PREFIX)
                        .and_then(|alias| metadata.output_index(alias.as_str()))
                })
                .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                    label: label.to_string(),
                });
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                label: label.to_string(),
            })
    }

    fn resolve_occasion_events(
        &self,
        occasion: &Occasion,
        parameters: &[f64],
        covariates: &Covariates,
    ) -> Result<Vec<Event>, PharmsolError> {
        let mut resolved = occasion.clone();

        for event in resolved.events_iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    let input = self.resolve_input_label(bolus.input(), RouteKind::Bolus)?;
                    bolus.set_input(input);
                }
                Event::Infusion(infusion) => {
                    let input = self.resolve_input_label(infusion.input(), RouteKind::Infusion)?;
                    infusion.set_input(input);
                }
                Event::Observation(observation) => {
                    let outeq = self.resolve_output_label(observation.outeq())?;
                    observation.set_outeq(outeq);
                }
            }
        }

        Ok(resolved.process_events(Some((self.fa(), self.lag(), parameters, covariates)), true))
    }
    #[allow(dead_code)]
    fn is_sde(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn process_observation(
        &self,
        parameters: &[f64],
        observation: &Observation,
        error_models: Option<&AssayErrorModels>,
        time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError>;

    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S;

    #[allow(clippy::too_many_arguments)]
    fn simulate_event(
        &self,
        parameters: &[f64],
        event: &Event,
        next_event: Option<&Event>,
        error_models: Option<&AssayErrorModels>,
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        match event {
            Event::Bolus(bolus) => {
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
                x.add_bolus(input, bolus.amount());
            }
            Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            Event::Observation(observation) => {
                self.process_observation(
                    parameters,
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
                parameters,
                covariates,
                infusions,
                event.time(),
                next_event.time(),
            )?;
        }
        Ok(())
    }
}

fn canonical_numeric_alias(label: &str, prefix: &str) -> Option<String> {
    if label.is_empty() || !label.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(format!("{prefix}{label}"))
}

/// Trait for handwritten model equations that can be simulated.
///
/// [`Equation`] is the shared interface implemented by handwritten [`ODE`],
/// [`Analytical`], and [`SDE`] models.
///
/// Subject data enters this layer through public labels on dose and observation
/// events. If metadata is attached to the equation, those labels are resolved by
/// name before simulation. Otherwise, the execution layer expects numeric labels
/// that can be interpreted as dense indices.
///
/// # Likelihood Calculation
///
/// Use [`estimate_log_likelihood`](Self::estimate_log_likelihood) for numerically stable
/// likelihood computation. The deprecated [`estimate_likelihood`](Self::estimate_likelihood)
/// is provided for backward compatibility.
#[allow(private_bounds)]
pub trait Equation: EquationPriv + 'static + Clone + Sync {
    #[doc(hidden)]
    fn bound_error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        None
    }

    #[doc(hidden)]
    fn bind_error_models<'a>(
        &'a self,
        error_models: &'a AssayErrorModels,
    ) -> Result<BoundAssayErrorModels<'a>, PharmsolError> {
        if let Some(cache) = self.bound_error_model_cache() {
            let key = error_models.hash();
            if let Some(bound_error_models) = cache.get(&key) {
                return Ok(BoundAssayErrorModels::Shared(bound_error_models));
            }

            return match error_models.bind_to(self)? {
                BoundAssayErrorModels::Owned(bound_error_models) => {
                    let bound_error_models = Arc::new(bound_error_models);
                    cache.insert(key, Arc::clone(&bound_error_models));
                    Ok(BoundAssayErrorModels::Shared(bound_error_models))
                }
                bound_error_models => Ok(bound_error_models),
            };
        }

        Ok(error_models.bind_to(self)?)
    }

    /// Estimate the likelihood of the subject given the parameters and error model.
    ///
    /// **Deprecated**: Use [`estimate_log_likelihood`](Self::estimate_log_likelihood) instead
    /// for better numerical stability, especially with many observations or extreme parameter values.
    ///
    /// This function calculates how likely the observed data is given the model
    /// parameters and error model. It may use caching for performance.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `parameters`: The parameter values
    /// - `error_model`: The error model
    ///
    /// # Returns
    /// The likelihood value (product of individual observation likelihoods)
    #[deprecated(
        since = "0.23.0",
        note = "Use estimate_log_likelihood() instead for better numerical stability"
    )]
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError>;

    /// Estimate the log-likelihood of the subject given the parameters and error model.
    ///
    /// This function calculates the log of how likely the observed data is given the model
    /// parameters and error model. It is numerically more stable than `estimate_likelihood`
    /// for extreme values or many observations.
    ///
    /// Uses observation-based sigma, appropriate for non-parametric algorithms.
    /// For parametric algorithms (SAEM, FOCE), use [`crate::ResidualErrorModels`] directly.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `parameters`: The parameter values
    /// - `error_models`: The error model
    ///
    /// # Returns
    /// The log-likelihood value (sum of individual observation log-likelihoods)
    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError>;

    fn kind() -> EqnKind;

    #[doc(hidden)]
    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        Ok(self.simulate_subject_dense(subject, parameters, None)?.0)
    }

    #[doc(hidden)]
    fn estimate_log_likelihood_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let predictions = self.estimate_predictions_dense(subject, parameters)?;
        predictions.log_likelihood(&bound_error_models)
    }

    #[doc(hidden)]
    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let bound_error_models = match error_models {
            Some(error_models) => Some(self.bind_error_models(error_models)?),
            None => None,
        };
        let bound_error_models = bound_error_models.as_ref().map(|models| &**models);

        let mut output = Self::P::new(self.nparticles());
        let mut likelihood = Vec::new();
        for occasion in subject.occasions() {
            let covariates = occasion.covariates();

            let mut x = self.initial_state(parameters, covariates, occasion.index());
            let mut infusions = Vec::new();
            let events = self.resolve_occasion_events(occasion, parameters, covariates)?;
            for (index, event) in events.iter().enumerate() {
                self.simulate_event(
                    parameters,
                    event,
                    events.get(index + 1),
                    bound_error_models,
                    covariates,
                    &mut x,
                    &mut infusions,
                    &mut likelihood,
                    &mut output,
                )?;
            }
        }
        let ll = bound_error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }

    /// Generate predictions for a subject with given parameters.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `parameters`: The parameter values
    ///
    /// # Returns
    /// Predicted concentrations
    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        self.estimate_predictions_dense(subject, parameters.as_slice())
    }

    /// Get the number of output equations in the model.
    fn nouteqs(&self) -> usize {
        self.get_nouteqs()
    }

    /// Get the number of state variables in the model.
    fn nstates(&self) -> usize {
        self.get_nstates()
    }

    /// Build a label-aware [`AssayErrorModels`] set for this equation.
    ///
    /// Handwritten equations resolve output labels from attached metadata.
    /// Equations without metadata fall back to an explicit unbound set so dense
    /// output-slot workflows remain available without adding runtime lookup cost.
    #[doc(hidden)]
    fn assay_error_models(&self) -> AssayErrorModels {
        self.metadata()
            .map(|metadata| {
                AssayErrorModels::with_output_names(
                    metadata.outputs().iter().map(|output| output.name()),
                )
            })
            .unwrap_or_else(AssayErrorModels::empty)
    }

    /// Simulate a subject with given parameters and optionally calculate likelihood.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `parameters`: The parameter values
    /// - `error_model`: The error model (optional)
    ///
    /// # Returns
    /// A tuple containing predictions and optional likelihood
    fn simulate_subject(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        self.simulate_subject_dense(subject, parameters.as_slice(), error_models)
    }
}

/// Runtime family tag for handwritten equations.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum EqnKind {
    ODE = 0,
    Analytical = 1,
    SDE = 2,
}

impl EqnKind {
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::ODE => "EqnKind::ODE",
            Self::Analytical => "EqnKind::Analytical",
            Self::SDE => "EqnKind::SDE",
        }
    }
}

/// Hash parameter vectors to a u64 for cache key generation.
#[inline(always)]
pub(crate) fn parameters_hash(parameters: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    for &value in parameters {
        // Normalize -0.0 to 0.0 for consistent hashing
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}
