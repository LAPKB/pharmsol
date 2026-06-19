use crate::simulator::likelihood::{LikelihoodModel, Prediction};
use crate::PharmsolError;

/// Trait for prediction containers.
///
/// Implemented by [`SubjectPredictions`] (ODE/Analytical) and
/// [`ParticleLikelihood`] (SDE). The concrete container a backend produces is
/// selected through the [`Solver::Predictions`](super::Solver::Predictions)
/// associated type, so the simulation output can differ per solver — rich
/// predictions for deterministic models, a likelihood for particle-based ones.
///
/// The likelihood is computed through a pluggable [`LikelihoodModel`], so the
/// same container supports Gaussian assay error today and other methods
/// (Poisson, Student-t, …) in the future without changes here.
///
/// For the push-based accumulation interface used during simulation, see
/// [`super::PredictionsContainer`].
///
/// [`SubjectPredictions`]: crate::simulator::likelihood::SubjectPredictions
/// [`ParticleLikelihood`]: crate::simulator::likelihood::ParticleLikelihood
pub trait Predictions: Default {
    /// Create a new prediction container with specified capacity.
    ///
    /// # Parameters
    /// - `nparticles`: Number of particles (1 for deterministic, >1 for SDE)
    fn new(_nparticles: usize) -> Self {
        Default::default()
    }

    /// Calculate the sum of squared errors for all predictions.
    fn squared_error(&self) -> f64;

    /// Get all predictions as an owned vector.
    fn get_predictions(&self) -> Vec<Prediction>;

    /// Calculate the log-likelihood of the predictions under a likelihood model.
    ///
    /// The `model` selects how each observation is scored (Gaussian assay
    /// error, Poisson, …). This is numerically more stable than computing the
    /// likelihood and taking its log, especially for extreme values or many
    /// observations.
    fn log_likelihood(&self, model: &dyn LikelihoodModel) -> Result<f64, PharmsolError>;
}
