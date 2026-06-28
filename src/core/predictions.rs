use crate::data::error_model::AssayErrorModels;
use crate::simulator::likelihood::Prediction;
use crate::PharmsolError;

/// Trait for prediction containers.
///
/// Implemented by [`SubjectPredictions`] (ODE/Analytical) and
/// [`Array2<Prediction>`] (SDE).
///
/// For the push-based accumulation interface used during simulation, see
/// [`super::PredictionsContainer`].
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

    /// Calculate the log-likelihood of the predictions given an error model.
    ///
    /// This is numerically more stable than computing the likelihood and
    /// taking its log, especially for extreme values or many observations.
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError>;
}
