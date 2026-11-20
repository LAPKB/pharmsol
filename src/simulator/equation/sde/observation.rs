//! The observation model the SDE particle filter weights particles with.
//!
//! pharmsol does not compute likelihoods, with one exception: an
//! [`SDE`](crate::SDE)'s likelihood is not a function of any one set of
//! predictions, so its particle filter has to score observations while it
//! simulates. [`ObservationModel`] is how it does that. pharmsol ships no
//! implementation; error-model crates (such as PMcore) implement it.

use crate::data::event::OutputLabel;
use crate::{Censor, ErrorPoly, PharmsolError};

/// The distribution of an observed value given the model's prediction.
///
/// [`SDE::log_likelihood`](crate::SDE::log_likelihood) and
/// [`SDE::filter`](crate::SDE::filter) weight every particle at every
/// observation with [`Self::log_density`]. pharmsol ships no implementation
/// of this trait.
///
/// Implementations are keyed by output label, so one model can cover several
/// outputs. The trait is object safe, and the SDE methods accept unsized
/// implementations such as `&dyn ObservationModel`.
///
/// # Contract
///
/// An implementation must guarantee:
///
/// - [`Self::log_density`] is a pure function of its arguments: the same
///   inputs always give the same result, from any thread.
/// - A density that is not finite (an observation the particle cannot have
///   produced, including a non-finite prediction from a diverged particle) is
///   reported as [`PharmsolError::NonFiniteLikelihood`]. The filter gives such
///   a particle zero weight; any other error aborts the simulation. As a
///   safeguard, a non-finite value returned as `Ok` is also given zero weight.
/// - [`Self::labels`] lists every output it scores, and only declared
///   outputs; the SDE rejects labels its model does not declare before
///   simulating.
/// - [`Self::cache_key`], when it returns `Some`, identifies the densities
///   exactly (see its documentation).
pub trait ObservationModel: Send + Sync {
    /// Log-density of `observation` given `prediction`, for output `outeq`.
    ///
    /// `censoring` says whether `observation` is a quantification limit
    /// rather than a measured value (the log-CDF or log-survival function
    /// then applies), and `errorpoly` is the observation's own error
    /// polynomial, if the data set carries one.
    ///
    /// Returns [`PharmsolError::NonFiniteLikelihood`] when the density is not
    /// finite, which the particle filter treats as a particle the observation
    /// rules out.
    fn log_density(
        &self,
        outeq: &OutputLabel,
        observation: f64,
        prediction: f64,
        censoring: Censor,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError>;

    /// The output labels this model covers.
    ///
    /// The SDE checks these against its declared outputs before simulating,
    /// so a model for an output the equation does not have fails up front.
    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_>;

    /// Check the model's own settings, e.g. distribution parameters.
    ///
    /// The SDE calls this, with [`Self::labels`], before simulating, so a
    /// model can be built without error handling and still fail up front.
    /// The default accepts everything.
    fn validate(&self) -> Result<(), PharmsolError> {
        Ok(())
    }

    /// A hash of everything that changes [`Self::log_density`], or `None` to
    /// opt out of caching.
    ///
    /// [`SDE::log_likelihood`](crate::SDE::log_likelihood) caches its
    /// estimates keyed on this value, and uses the cache only when it is
    /// `Some`. Two models with the same key must give the same densities for
    /// every input; implementations should mix in a tag for their own type so
    /// different model types never collide. The default, `None`, always
    /// simulates.
    fn cache_key(&self) -> Option<u64> {
        None
    }
}
