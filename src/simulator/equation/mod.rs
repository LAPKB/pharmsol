//! Equation families and their shared simulation interfaces.
//!
//! This module is the public home for the [`ODE`], [`Analytical`], and [`SDE`]
//! families, the shared [`Equation`] trait, and the metadata types that attach
//! public names to parameters, states, routes, and outputs.
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
//! # Models And Drivers
//!
//! Each family is a driver that advances a model through a subject's events
//! and caches the results. The model supplies the functions it evaluates: the
//! shared [`Model`] trait (initial state, doses, lag, bioavailability,
//! outputs) plus one family trait ([`OdeModel`], [`AnalyticalModel`],
//! [`SdeModel`]).
//!
//! Handwritten models ([`NativeOde`], [`NativeAnalytical`], [`NativeSde`]) are
//! the default, so `ODE::new(...)` works as before. Models compiled from the
//! DSL use the same drivers with a `JitModel`.
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

pub mod analytical;
pub mod metadata;
pub mod model;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use metadata::*;
pub use model::{AttachMetadataError, Model};
pub use ode::*;
pub use pharmsol_dsl::{AnalyticalKernel, ModelKind};
pub use sde::*;

use std::borrow::Cow;

use crate::{error_model::ErrorModelError, OutputLabel, PharmsolError, Subject};

use super::likelihood::{ObservationModel, Prediction, SubjectPredictions};

/// Prediction container returned by an [`Equation`], reduced to one
/// prediction per observation.
///
/// Deterministic equations return
/// [`SubjectPredictions`],
/// which already are one prediction per observation and are borrowed.
/// [`SDE`] returns [`ParticlePredictions`](crate::ParticlePredictions),
/// whose point is the (weighted) particle mean, built on each call.
///
/// The trait has no likelihood: a stochastic model's likelihood is not a
/// function of any one set of predictions. Use [`Equation::log_likelihood`].
pub trait PointPredictions {
    /// One prediction per observation.
    fn point(&self) -> Cow<'_, [Prediction]>;
}

/// The result of [`Equation::simulate`].
#[derive(Debug, Clone)]
pub struct Simulation<P> {
    /// The subject's predictions.
    pub predictions: P,
    /// Log-likelihood of the subject's observations, when an observation
    /// model was given.
    pub log_likelihood: Option<f64>,
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
    /// Prediction container: [`SubjectPredictions`]
    /// for deterministic models, [`ParticlePredictions`](crate::ParticlePredictions)
    /// for [`SDE`].
    type P: PointPredictions;

    /// Validated metadata attached to this equation, if any.
    fn metadata(&self) -> Option<&ValidatedModelMetadata>;

    /// Number of state variables.
    fn nstates(&self) -> usize;

    /// Number of output equations.
    fn nouteqs(&self) -> usize;

    /// Simulate a subject and return its predictions.
    ///
    /// Results may be served from the equation's cache. An [`SDE`] returns
    /// free-running particles.
    fn predict(&self, subject: &Subject, parameters: &[f64]) -> Result<Self::P, PharmsolError>;

    /// Log-likelihood of the subject's observations under `observation_model`.
    ///
    /// Deterministic equations sum the log-density of each observation given
    /// its (possibly cached) prediction. An [`SDE`] returns its particle-filter
    /// estimate. Usually `observation_model` is a [`crate::OutputModels`],
    /// whose outputs take sigma from the observation (assay error) or from
    /// the prediction (residual error).
    ///
    /// Errors if `observation_model` names an output this equation does not
    /// declare, before simulating.
    fn log_likelihood<O: ObservationModel + ?Sized>(
        &self,
        subject: &Subject,
        parameters: &[f64],
        observation_model: &O,
    ) -> Result<f64, PharmsolError>;

    /// Simulate a subject and, when an observation model is given, also
    /// return the log-likelihood of its observations from the same run.
    ///
    /// For deterministic equations this is [`Self::predict`] plus the sum of
    /// observation log-densities. An [`SDE`] given an observation model runs
    /// its particle filter once, uncached, and returns the filtered particles
    /// with their weights alongside the likelihood estimate.
    fn simulate(
        &self,
        subject: &Subject,
        parameters: &[f64],
        observation_model: Option<&dyn ObservationModel>,
    ) -> Result<Simulation<Self::P>, PharmsolError>;

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

    /// [`Self::validate_output_labels`] for an observation model.
    #[doc(hidden)]
    fn validate_observation_model<O: ObservationModel + ?Sized>(
        &self,
        observation_model: &O,
    ) -> Result<(), PharmsolError> {
        self.validate_output_labels(observation_model.labels())
    }
}

/// [`Equation::simulate`] for an equation whose predictions are
/// deterministic: predict, then score the predictions.
pub(crate) fn simulate_deterministic<E>(
    equation: &E,
    subject: &Subject,
    parameters: &[f64],
    observation_model: Option<&dyn ObservationModel>,
) -> Result<Simulation<SubjectPredictions>, PharmsolError>
where
    E: Equation<P = SubjectPredictions>,
{
    if let Some(observation_model) = observation_model {
        equation.validate_observation_model(observation_model)?;
    }
    let predictions = equation.predict(subject, parameters)?;
    let log_likelihood = observation_model
        .map(|observation_model| predictions.log_likelihood(observation_model))
        .transpose()?;
    Ok(Simulation {
        predictions,
        log_likelihood,
    })
}

/// Hash parameter vectors to a u64 for cache key generation.
#[inline(always)]
pub(crate) fn parameters_hash(parameters: &[f64]) -> u64 {
    salted_parameters_hash(parameters, 0)
}

/// [`parameters_hash`] combined with `salt`, e.g. a hash of solver settings
/// that change the result.
#[inline(always)]
pub(crate) fn salted_parameters_hash(parameters: &[f64], salt: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    salt.hash(&mut hasher);
    for &value in parameters {
        // Normalize -0.0 to 0.0 for consistent hashing
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}
