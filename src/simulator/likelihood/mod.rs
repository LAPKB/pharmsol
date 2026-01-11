//! Likelihood calculation module for pharmacometric analyses.
//!
//! This module provides functions and types for computing log-likelihoods
//! in pharmacometric population modeling. It supports both:
//!
//! - **Non-parametric algorithms** (NPAG, NPOD): Use [`ErrorModels`] with observation-based sigma
//! - **Parametric algorithms** (SAEM, FOCE): Use [`ResidualErrorModels`] with prediction-based sigma
//!
//! # Module Organization
//!
//! - [`distributions`]: Statistical distribution functions (log-normal PDF, CDF)
//! - [`prediction`]: Single observation-prediction pairs
//! - [`subject`]: Subject-level prediction collections
//! - [`matrix`]: Population-level log-likelihood matrix computation
//!
//! # Key Functions
//!
//! ## For Non-Parametric Algorithms
//!
//! Use [`log_likelihood_matrix`] to compute a matrix of log-likelihoods across
//! all subjects and support points:
//!
//! ```ignore
//! use pharmsol::prelude::simulator::{log_likelihood_matrix, LikelihoodMatrixOptions};
//!
//! let log_liks = log_likelihood_matrix(
//!     &equation,
//!     &data,
//!     &support_points,
//!     &error_models,
//!     LikelihoodMatrixOptions::new().with_progress(),
//! )?;
//! ```
//!
//! ## For Parametric Algorithms
//!
//! Use [`log_likelihood_batch`] when each subject has individual parameters:
//!
//! ```ignore
//! use pharmsol::prelude::simulator::log_likelihood_batch;
//!
//! let log_liks = log_likelihood_batch(
//!     &equation,
//!     &data,
//!     &parameters,
//!     &residual_error_models,
//! )?;
//! ```
//!
//! # Numerical Stability
//!
//! All likelihood functions operate in log-space for numerical stability.
//! The deprecated `likelihood()` and `psi()` functions are provided for
//! backward compatibility but should be avoided in new code.

mod distributions;
mod matrix;
mod prediction;
mod progress;
mod subject;

// Re-export main types
pub use matrix::{log_likelihood_matrix, LikelihoodMatrixOptions};
pub use prediction::Prediction;
pub use subject::{PopulationPredictions, SubjectPredictions};

// Deprecated re-exports for backward compatibility
#[allow(deprecated)]
pub use matrix::{log_psi, psi};

use ndarray::Array2;
use rayon::prelude::*;

use crate::{Data, Equation, PharmsolError, Predictions, Subject};

/// Compute log-likelihoods for all subjects in parallel, where each subject
/// has its own parameter vector.
///
/// This function simulates each subject with their individual parameters and
/// computes log-likelihood using prediction-based sigma (appropriate for
/// parametric algorithms like SAEM, FOCE).
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subjects`: The subject data (N subjects)
/// - `parameters`: Parameter vectors for each subject (N × P matrix, row i = params for subject i)
/// - `residual_error_models`: The residual error models (prediction-based sigma)
///
/// # Returns
/// A vector of N log-likelihoods, one per subject. Returns `f64::NEG_INFINITY`
/// for subjects where simulation fails.
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::simulator::log_likelihood_batch;
/// use pharmsol::{ResidualErrorModel, ResidualErrorModels};
///
/// let residual_error = ResidualAssayErrorModels::new()
///     .add(0, ResidualErrorModel::constant(0.5));
///
/// let log_liks = log_likelihood_batch(
///     &equation,
///     &data,
///     &parameters,
///     &residual_error,
/// )?;
/// ```
pub fn log_likelihood_batch(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    residual_error_models: &crate::ResidualErrorModels,
) -> Result<Vec<f64>, PharmsolError> {
    let subjects_vec = subjects.subjects();
    let n_subjects = subjects_vec.len();

    if parameters.nrows() != n_subjects {
        return Err(PharmsolError::OtherError(format!(
            "parameters has {} rows but there are {} subjects",
            parameters.nrows(),
            n_subjects
        )));
    }

    // Parallel computation across subjects
    let results: Vec<f64> = (0..n_subjects)
        .into_par_iter()
        .map(|i| {
            let subject = &subjects_vec[i];
            let params = parameters.row(i).to_vec();

            // Simulate to get predictions
            let predictions = match equation.estimate_predictions(subject, &params) {
                Ok(preds) => preds,
                Err(_) => return f64::NEG_INFINITY,
            };

            // Extract (outeq, observation, prediction) tuples and compute log-likelihood
            let obs_pred_pairs = predictions
                .get_predictions()
                .into_iter()
                .filter_map(|pred| {
                    pred.observation()
                        .map(|obs| (pred.outeq(), obs, pred.prediction()))
                });

            residual_error_models.total_log_likelihood(obs_pred_pairs)
        })
        .collect();

    Ok(results)
}

/// Compute log-likelihood for a single subject using prediction-based sigma.
///
/// This is the single-subject equivalent of [`log_likelihood_batch`].
/// It simulates the model, extracts observation-prediction pairs, and computes
/// the log-likelihood using [`crate::ResidualErrorModels`].
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subject`: The subject data
/// - `params`: Parameter vector for this subject
/// - `residual_error_models`: The residual error models (prediction-based sigma)
///
/// # Returns
/// The log-likelihood for this subject. Returns `f64::NEG_INFINITY` on simulation error.
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::simulator::log_likelihood_subject;
///
/// let log_lik = log_likelihood_subject(
///     &equation,
///     &subject,
///     &params,
///     &residual_error_models,
/// );
/// ```
pub fn log_likelihood_subject(
    equation: &impl Equation,
    subject: &Subject,
    params: &[f64],
    residual_error_models: &crate::ResidualErrorModels,
) -> f64 {
    // Simulate to get predictions
    let predictions = match equation.estimate_predictions(subject, &params.to_vec()) {
        Ok(preds) => preds,
        Err(_) => return f64::NEG_INFINITY,
    };

    // Extract (outeq, observation, prediction) tuples and compute log-likelihood
    let obs_pred_pairs = predictions
        .get_predictions()
        .into_iter()
        .filter_map(|pred| {
            pred.observation()
                .map(|obs| (pred.outeq(), obs, pred.prediction()))
        });

    residual_error_models.total_log_likelihood(obs_pred_pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::error_model::{ErrorModel, ErrorPoly};
    use crate::data::event::Observation;
    use crate::Censor;

    #[test]
    fn test_log_likelihood_equals_log_of_likelihood() {
        // Create a prediction with an observation
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 10.5,
            outeq: 0,
            errorpoly: None,
            state: vec![10.5],
            occasion: 0,
            censoring: Censor::None,
        };

        // Create error model with additive error
        let error_models = crate::AssayErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        #[allow(deprecated)]
        let lik = prediction.likelihood(&error_models).unwrap();
        let log_lik = prediction.log_likelihood(&error_models).unwrap();

        // log_likelihood should equal ln(likelihood)
        let expected_log_lik = lik.ln();
        assert!(
            (log_lik - expected_log_lik).abs() < 1e-10,
            "log_likelihood ({}) should equal ln(likelihood) ({})",
            log_lik,
            expected_log_lik
        );
    }

    #[test]
    fn test_subject_predictions_log_likelihood() {
        let predictions = vec![
            Prediction {
                time: 1.0,
                observation: Some(10.0),
                prediction: 10.1,
                outeq: 0,
                errorpoly: None,
                state: vec![10.1],
                occasion: 0,
                censoring: Censor::None,
            },
            Prediction {
                time: 2.0,
                observation: Some(8.0),
                prediction: 8.2,
                outeq: 0,
                errorpoly: None,
                state: vec![8.2],
                occasion: 0,
                censoring: Censor::None,
            },
        ];

        let subject_predictions = SubjectPredictions::from(predictions);
        let error_models = crate::AssayErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        #[allow(deprecated)]
        let lik = subject_predictions.likelihood(&error_models).unwrap();
        let log_lik = subject_predictions.log_likelihood(&error_models).unwrap();

        // Sum of log likelihoods should equal log of product of likelihoods
        let expected_log_lik = lik.ln();
        assert!(
            (log_lik - expected_log_lik).abs() < 1e-10,
            "Subject log_likelihood ({}) should equal ln(likelihood) ({})",
            log_lik,
            expected_log_lik
        );
    }

    #[test]
    fn test_empty_predictions_have_neutral_log_likelihood() {
        let preds = SubjectPredictions::default();
        let errors = crate::AssayErrorModels::new();
        assert_eq!(preds.log_likelihood(&errors).unwrap(), 0.0); // log(1) = 0
    }

    #[test]
    fn test_log_likelihood_combines_observations() {
        let mut preds = SubjectPredictions::default();
        let obs = Observation::new(0.0, Some(1.0), 0, None, 0, Censor::None);
        preds.add_prediction(obs.to_prediction(1.0, vec![]));

        let error_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0);
        let errors = crate::AssayErrorModels::new().add(0, error_model).unwrap();

        let log_lik = preds.log_likelihood(&errors).unwrap();
        assert!(log_lik.is_finite());
        assert!(log_lik <= 0.0); // Log likelihood is always <= 0
    }

    #[test]
    fn test_lognormpdf_direct() {
        use super::distributions::lognormpdf;

        // Test the helper function directly
        let obs = 0.0;
        let pred = 0.0;
        let sigma = 1.0;

        let log_pdf = lognormpdf(obs, pred, sigma);

        // At mean of standard normal, log PDF = -0.5 * ln(2π)
        let expected = -0.5 * distributions::LOG_2PI;
        assert!(
            (log_pdf - expected).abs() < 1e-12,
            "lognormpdf at mean should be -0.5*ln(2π)"
        );
    }
}
