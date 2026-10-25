//! The observation model: how an observed value is distributed around a
//! prediction.
//!
//! [`ObservationModel`] is what every likelihood in pharmsol is computed
//! through. Deterministic equations sum its log-density over their
//! predictions; the SDE particle filter weights each particle with it.
//!
//! The crate's implementation is [`OutputModels`](crate::OutputModels): one
//! [`OutputModel`](crate::OutputModel) per output, each choosing where sigma
//! comes from (observation or prediction), the distribution and the scale.
//! Implement the trait yourself for anything those cannot express.

use crate::data::event::OutputLabel;
use crate::{Censor, ErrorPoly, PharmsolError};

/// The distribution of an observed value given the model's prediction.
///
/// Implementations are keyed by output label, so one model can cover several
/// outputs. The trait is object safe: [`Equation::simulate`](crate::Equation::simulate)
/// takes a `&dyn ObservationModel`, while the likelihood paths are generic and
/// dispatch statically.
pub trait ObservationModel: Send + Sync {
    /// Log-density of `observation` given `prediction`, for output `outeq`.
    ///
    /// `censoring` says whether `observation` is a quantification limit
    /// rather than a measured value, and `errorpoly` is the observation's own
    /// error polynomial, if the data set carries one.
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
    /// Equations check these against their declared outputs before
    /// simulating, so a model for an output the equation does not have fails
    /// up front.
    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_>;

    /// A hash of everything that changes [`Self::log_density`].
    ///
    /// Likelihood caches (see [`SDE`](crate::SDE)) key on it, so two models
    /// with the same key must give the same densities. Implementations should
    /// mix in a tag for their own type so different model types never collide.
    fn cache_key(&self) -> u64;
}

/// Finite log-density, or [`PharmsolError::NonFiniteLikelihood`].
#[inline]
pub(crate) fn finite(log_density: f64) -> Result<f64, PharmsolError> {
    if log_density.is_finite() {
        Ok(log_density)
    } else {
        Err(PharmsolError::NonFiniteLikelihood(log_density))
    }
}
