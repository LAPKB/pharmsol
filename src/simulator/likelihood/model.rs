//! Pluggable likelihood models.
//!
//! A [`LikelihoodModel`] defines *how* the log-likelihood contribution of a
//! single observation–prediction pair is computed. This is the extension point
//! for supporting different probability models: Gaussian assay error today,
//! and e.g. Poisson, Student-t, or other distributions in the future.
//!
//! To add a new method of computing likelihood, implement this trait for a new
//! type — no changes to the solvers or prediction containers are required, as
//! they accept any `&dyn LikelihoodModel`.
//!
//! ```ignore
//! struct PoissonLikelihood;
//!
//! impl LikelihoodModel for PoissonLikelihood {
//!     fn observation_log_likelihood(&self, prediction: &Prediction) -> Result<f64, PharmsolError> {
//!         let lambda = prediction.prediction();
//!         match prediction.observation() {
//!             Some(obs) => Ok(poisson_log_pmf(obs, lambda)),
//!             None => Ok(0.0),
//!         }
//!     }
//! }
//! ```

use crate::data::error_model::AssayErrorModels;
use crate::PharmsolError;

use super::prediction::Prediction;

/// A strategy for scoring an observation against a model prediction.
///
/// Implementors define a probability model used to compute the log-likelihood
/// contribution of a single observation. This trait is object-safe so that
/// solvers and prediction containers can accept `&dyn LikelihoodModel` and
/// remain agnostic to the concrete likelihood method in use.
pub trait LikelihoodModel {
    /// Log-likelihood contribution of one observation–prediction pair.
    ///
    /// Missing observations must contribute `0.0` (the log of probability 1),
    /// so that absent measurements do not affect the joint likelihood.
    fn observation_log_likelihood(&self, prediction: &Prediction) -> Result<f64, PharmsolError>;
}

/// Gaussian (log-normal) assay-error likelihood.
///
/// This is the default likelihood method, appropriate for continuous
/// concentration measurements with observation-based sigma. It delegates to the
/// numerically stable per-prediction computation in
/// [`Prediction::log_likelihood`].
impl LikelihoodModel for AssayErrorModels {
    fn observation_log_likelihood(&self, prediction: &Prediction) -> Result<f64, PharmsolError> {
        prediction.log_likelihood(self)
    }
}
