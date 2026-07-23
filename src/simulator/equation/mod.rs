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
//! - generate noiseless predictions across equation families
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
    simulator::{Fa, Lag},
    Covariates, Event, Infusion, InputLabel, Observation, Occasion, OutputLabel, Parameters,
    PharmsolError, Subject,
};

use super::prediction::Prediction;

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

    /// Get all predictions as a vector.
    ///
    /// # Returns
    /// Vector of prediction objects
    fn get_predictions(&self) -> Vec<Prediction>;

    /// Visit each effective prediction without requiring callers to own a `Vec`.
    fn for_each_prediction(&self, mut f: impl FnMut(&Prediction)) {
        let predictions = self.get_predictions();
        for prediction in &predictions {
            f(prediction);
        }
    }

    /// Record the subject identifier these predictions belong to.
    ///
    /// The default implementation is a no-op for containers that do not carry a
    /// subject identifier (for example the particle grid used by SDE models).
    fn set_subject_id(&mut self, _id: &str) {}
}

/// Trait for prediction caching on deterministic ODE and analytical equations.
///
/// Stochastic SDE equations do not implement this trait. Caching is **enabled by
/// default** for deterministic equations with a capacity of 100,000 entries.
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
            let route = metadata.route_for_label(label.as_str()).ok_or_else(|| {
                PharmsolError::unknown_input_label(label.as_str(), &metadata.route_labels())
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

            let input = route.input_index();
            if input >= self.get_ndrugs() {
                return Err(PharmsolError::InputOutOfRange {
                    input,
                    ndrugs: self.get_ndrugs(),
                });
            }
            return Ok(input);
        }

        let input = label
            .index()
            .ok_or_else(|| PharmsolError::unknown_input_label(label.as_str(), &[]))?;
        if input >= self.get_ndrugs() {
            return Err(PharmsolError::InputOutOfRange {
                input,
                ndrugs: self.get_ndrugs(),
            });
        }
        Ok(input)
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

    /// Resolve the public output label for a dense output index.
    ///
    /// When metadata is attached, this returns the declared output name. Without
    /// metadata, it falls back to the numeric index as a label.
    fn output_label(&self, index: usize) -> OutputLabel {
        self.metadata()
            .and_then(|metadata| metadata.output_labels().get(index).map(OutputLabel::new))
            .unwrap_or_else(|| OutputLabel::from(index))
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

        Ok(resolved.process_events(Some((self.fa(), self.lag(), parameters, covariates))))
    }
    fn process_observation(
        &self,
        parameters: &[f64],
        observation: &Observation,
        time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
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
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
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
                x.add_bolus(input, bolus.amount());
            }
            Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            Event::Observation(observation) => {
                self.process_observation(
                    parameters,
                    observation,
                    event.time(),
                    covariates,
                    x,
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
/// # Estimation boundary
///
/// Estimation algorithms call [`estimate_predictions`](Self::estimate_predictions)
/// and perform all scoring outside this crate. Equation simulation exposes only
/// structural predictions.
#[allow(private_bounds)]
pub trait Equation: EquationPriv + 'static + Clone + Sync {
    fn kind() -> EqnKind;

    #[doc(hidden)]
    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        self.simulate_subject_dense(subject, parameters)
    }

    #[doc(hidden)]
    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        let mut output = Self::P::new(self.nparticles());
        output.set_subject_id(subject.id());
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
                    covariates,
                    &mut x,
                    &mut infusions,
                    &mut output,
                )?;
            }
        }
        Ok(output)
    }

    /// Generate predictions for a subject with the given parameter vector.
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

    /// Simulate a subject and return prediction data.
    fn simulate_subject(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        self.simulate_subject_dense(subject, parameters.as_slice())
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
