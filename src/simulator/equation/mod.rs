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
//! - With [`metadata::ModelMetadata`] attached, route and output labels are
//!   resolved by exact name against the declared routes and outputs before
//!   simulation. Numeric names such as `"1"` are ordinary names and never
//!   resolve by position.
//! - Without metadata, handwritten models declare no names, so labels are
//!   dense indices such as `0` or `1`.
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

use std::fmt::Debug;
pub mod analytical;
pub mod metadata;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use metadata::*;
pub use ode::*;
pub use pharmsol_dsl::{AnalyticalKernel, ModelKind};
pub use sde::*;

use crate::{
    error_model::{AssayErrorModels, ErrorModelError},
    simulator::{Fa, Lag},
    Covariates, Event, Infusion, InputLabel, Observation, Occasion, OutputLabel, PharmsolError,
    Subject,
};

use super::likelihood::Prediction;

/// Prediction container returned by an [`Equation`].
pub trait Predictions {
    /// Get all predictions as a vector.
    ///
    /// Particle containers return the mean prediction across particles.
    fn get_predictions(&self) -> Vec<Prediction>;

    /// Calculate the log-likelihood of the predictions given an error model.
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
    fn with_cache_capacity(self, size: usize) -> Self;

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

pub(crate) trait EquationPriv: Equation {
    type S: Debug;
    fn empty_predictions(&self) -> Self::P;
    // fn get_init(&self) -> &Init;
    // fn get_out(&self) -> &Out;
    fn lag(&self) -> &Lag;
    fn fa(&self) -> &Fa;
    fn get_nstates(&self) -> usize;
    fn get_ndrugs(&self) -> usize;
    fn get_nouteqs(&self) -> usize;
    fn solve(
        &self,
        state: &mut Self::S,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError>;
    fn add_bolus(&self, x: &mut Self::S, input: usize, amount: f64);

    fn resolve_input_label(
        &self,
        label: &InputLabel,
        expected_kind: RouteKind,
    ) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            // A bolus and an infusion may share a label, so resolution must
            // match the kind as well: bolus events resolve bolus routes and
            // infusion events resolve infusion routes.
            let route = match metadata.route_for_label(label.as_str(), expected_kind) {
                Some(route) => route,
                None => {
                    let other_kind = match expected_kind {
                        RouteKind::Bolus => RouteKind::Infusion,
                        RouteKind::Infusion => RouteKind::Bolus,
                    };
                    if let Some(other) = metadata.route_for_label(label.as_str(), other_kind) {
                        return Err(PharmsolError::UnsupportedInputRouteKind {
                            input: other.input_index(),
                            kind: match expected_kind {
                                RouteKind::Bolus => pharmsol_dsl::RouteKind::Bolus,
                                RouteKind::Infusion => pharmsol_dsl::RouteKind::Infusion,
                            },
                        });
                    }
                    return Err(PharmsolError::unknown_input_label(
                        label.as_str(),
                        &metadata.route_labels(),
                    ));
                }
            };

            return Ok(route.input_index());
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::unknown_input_label(label.as_str(), &[]))
    }

    fn resolve_output_label(&self, label: &OutputLabel) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            return metadata.output_for_label(label.as_str()).ok_or_else(|| {
                PharmsolError::unknown_output_label(label.as_str(), &metadata.output_labels())
            });
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::unknown_output_label(label.as_str(), &[]))
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
                    observation.set_outeq_slot(outeq);
                }
            }
        }

        Ok(resolved.process_events(Some((self.fa(), self.lag(), parameters, covariates))))
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

    fn initial_state(&self, parameters: &[f64], covariates: &Covariates) -> Self::S;

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
                let input = bolus.input_index().ok_or_else(|| {
                    let available = self
                        .metadata()
                        .map(|m| m.route_labels())
                        .unwrap_or_default();
                    PharmsolError::unknown_input_label(bolus.input(), &available)
                })?;

                if input >= self.get_ndrugs() {
                    return Err(PharmsolError::InputOutOfRange {
                        input,
                        ndrugs: self.get_ndrugs(),
                    });
                }
                self.add_bolus(x, input, bolus.amount());
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

/// A model that can be simulated against subject data.
///
/// [`Equation`] is the shared interface of the [`ODE`], [`Analytical`], and
/// [`SDE`] families, whether their model functions are handwritten closures or
/// compiled from the DSL.
///
/// Subject data enters this layer through public labels on dose and observation
/// events. If the equation carries metadata, those labels are resolved by exact
/// name before simulation. Otherwise, the equation declares no names and labels
/// must be dense indices.
///
/// Parameters are passed as a dense slice in the model's parameter order.
/// [`Parameters`](crate::Parameters) dereferences to a slice, so
/// `&parameters` can be passed directly.
pub trait Equation: Clone + Send + Sync + 'static {
    /// Prediction container: [`SubjectPredictions`](crate::prelude::simulator::SubjectPredictions)
    /// for deterministic models, a particle matrix for [`SDE`].
    type P: Predictions;

    /// Validated metadata attached to this equation, if any.
    fn metadata(&self) -> Option<&ValidatedModelMetadata>;

    /// Number of state variables.
    fn nstates(&self) -> usize;

    /// Number of output equations.
    fn nouteqs(&self) -> usize;

    /// Simulate a subject and return its predictions.
    ///
    /// Results may be served from the equation's cache.
    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError>;

    /// Log-likelihood of the subject's observations.
    ///
    /// Uses observation-based sigma, appropriate for non-parametric algorithms.
    /// For parametric algorithms (SAEM, FOCE), use [`crate::ResidualErrorModels`]
    /// directly.
    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError>;

    /// Simulate a subject and, when error models are given, also return the
    /// likelihood (not log-likelihood) of its observations.
    fn simulate_subject(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let predictions = self.estimate_predictions(subject, parameters)?;
        let likelihood = match error_models {
            Some(error_models) => Some(
                self.estimate_log_likelihood(subject, parameters, error_models)?
                    .exp(),
            ),
            None => None,
        };
        Ok((predictions, likelihood))
    }

    /// Check that every label names one of this equation's outputs.
    ///
    /// Run once per call, before simulating, so an error model for an output
    /// the equation does not declare fails up front instead of being ignored.
    #[doc(hidden)]
    fn validate_output_labels<'a>(
        &self,
        labels: impl IntoIterator<Item = &'a OutputLabel>,
    ) -> Result<(), PharmsolError> {
        for label in labels {
            let declared = match self.metadata() {
                Some(metadata) => metadata.output_for_label(label.as_str()).is_some(),
                None => label.index().is_some_and(|slot| slot < self.nouteqs()),
            };
            if !declared {
                return Err(ErrorModelError::UnknownOutputLabel(label.to_string()).into());
            }
        }
        Ok(())
    }

    /// [`Self::validate_output_labels`] for an assay error-model set.
    #[doc(hidden)]
    fn validate_error_models(&self, error_models: &AssayErrorModels) -> Result<(), PharmsolError> {
        self.validate_output_labels(error_models.labels())
    }
}

/// Run the event loop of an [`EquationPriv`] family.
///
/// Returns the predictions and, when error models are given, the product of the
/// per-observation likelihoods.
pub(crate) fn simulate_events<E: EquationPriv>(
    equation: &E,
    subject: &Subject,
    parameters: &[f64],
    error_models: Option<&AssayErrorModels>,
) -> Result<(E::P, Option<f64>), PharmsolError> {
    if let Some(error_models) = error_models {
        equation.validate_error_models(error_models)?;
    }

    let mut output = equation.empty_predictions();
    let mut likelihood = Vec::new();
    for occasion in subject.occasions() {
        let covariates = occasion.covariates();

        let mut x = equation.initial_state(parameters, covariates);
        let mut infusions = Vec::new();
        let events = equation.resolve_occasion_events(occasion, parameters, covariates)?;
        for (index, event) in events.iter().enumerate() {
            equation.simulate_event(
                parameters,
                event,
                events.get(index + 1),
                error_models,
                covariates,
                &mut x,
                &mut infusions,
                &mut likelihood,
                &mut output,
            )?;
        }
    }
    let ll = error_models.map(|_| likelihood.iter().product::<f64>());
    Ok((output, ll))
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
