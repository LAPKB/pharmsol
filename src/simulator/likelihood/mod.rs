//! Likelihood calculation module for pharmacometric analyses.
//!
//! This module provides functions and types for computing log-likelihoods
//! in pharmacometric population modeling. Observations are scored with an
//! [`ObservationModel`], usually a [`crate::OutputModels`] whose outputs take
//! sigma from the observation (assay error, typical for non-parametric
//! algorithms such as NPAG, NPOD) or from the prediction (residual error,
//! typical for parametric algorithms such as SAEM, FOCE).
//!
//! Every function here accepts any [`ObservationModel`].
//! All of them score subjects through [`Equation::log_likelihood`], so a
//! deterministic equation sums the observation log-densities of its
//! (cached) predictions and an [`SDE`](crate::SDE) runs its particle filter.
//!
//! # Types
//!
//! - [`Prediction`]: Single observation-prediction pairs
//! - [`SubjectPredictions`]: A deterministic subject's predictions
//! - [`ParticlePredictions`]: A stochastic subject's particle predictions
//! - [`ObservationModel`]: The distribution of observations around predictions
//!
//! # Key Functions
//!
//! ## For Non-Parametric Algorithms
//!
//! Use [`log_likelihood_matrix`] to compute a matrix of log-likelihoods across
//! all subjects and support points.
//!
//! Validate any external support-point column order once with
//! [`crate::ParameterOrder`] before calling [`log_likelihood_matrix`]. The
//! runtime still consumes a dense model-order matrix:
//!
//! ```ignore
//! use ndarray::array;
//! use pharmsol::{ParameterOrder, prelude::simulator::log_likelihood_matrix};
//!
//! let order = ParameterOrder::with_model(&equation, ["ka", "ke"])?;
//! let support_points_in_source_order = array![[0.1, 0.3], [0.2, 0.4]];
//! let support_points = order.matrix(support_points_in_source_order)?;
//!
//! let log_liks = log_likelihood_matrix(
//!     &equation,
//!     &data,
//!     &support_points,
//!     &error_models,
//!     false,
//! )?;
//! ```
//!
//! ## For Parametric Algorithms
//!
//! Use [`log_likelihood_batch`] when each subject has individual parameters:
//!
//! ```ignore
//! use ndarray::array;
//! use pharmsol::prelude::simulator::log_likelihood_batch;
//!
//! let params = array![[0.1, 0.3], [0.2, 0.4]];
//!
//! let log_liks = log_likelihood_batch(
//!     &equation,
//!     &data,
//!     &params,
//!     &residual_error_models,
//! )?;
//! ```
//!
//! # Numerical Stability
//!
//! The deprecated `likelihood()` and `psi()` functions are provided for
//! backward compatibility but should be avoided in new code.

pub(crate) mod distributions;
mod matrix;
pub(crate) mod observation;
mod particles;
mod prediction;
mod progress;
mod subject;

// Re-export main types
pub use matrix::log_likelihood_matrix;
#[allow(deprecated)]
pub use matrix::log_psi;
#[allow(deprecated)]
pub use matrix::psi;
pub use observation::ObservationModel;
pub use particles::ParticlePredictions;
pub use prediction::Prediction;
pub use subject::{PopulationPredictions, SubjectPredictions};

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::{Data, Equation, PharmsolError, Subject};

/// Compute log-likelihoods for all subjects in parallel, where each subject
/// has its own parameter vector.
///
/// Each subject is scored with [`Equation::log_likelihood`], so cached
/// predictions are reused without copying them and an [`SDE`](crate::SDE)
/// is scored with its particle filter. Usually `observation_model` is a
/// [`crate::OutputModels`] with residual (prediction-based) error, as
/// parametric algorithms like SAEM and FOCE use, but any [`ObservationModel`]
/// works.
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subjects`: The subject data (N subjects)
/// - `parameters`: Parameter vectors for each subject (N × P matrix, row i = params for subject i)
/// - `observation_model`: The observation model, e.g. residual error models
///
/// # Returns
/// A vector of N log-likelihoods, one per subject. Returns `f64::NEG_INFINITY`
/// for subjects where simulation or scoring fails. Errors if the parameter
/// matrix does not have one row per subject, or if the observation model
/// names an output the equation does not declare.
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::simulator::log_likelihood_batch;
/// use pharmsol::{OutputModels, ResidualErrorModel};
///
/// let residual_error = OutputModels::new()
///     .add("cp", ResidualErrorModel::constant(0.5))?;
///
/// let log_liks = log_likelihood_batch(
///     &equation,
///     &data,
///     &parameters,
///     &residual_error,
/// )?;
/// ```
pub fn log_likelihood_batch<O: ObservationModel + ?Sized>(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    observation_model: &O,
) -> Result<Vec<f64>, PharmsolError> {
    let subject_slice = subjects.subjects_slice();
    let n_subjects = subject_slice.len();

    if parameters.nrows() != n_subjects {
        return Err(PharmsolError::OtherError(format!(
            "parameters has {} rows but there are {} subjects",
            parameters.nrows(),
            n_subjects
        )));
    }

    equation.validate_observation_model(observation_model)?;

    let score_subject = |subject: &Subject, parameter_row: &[f64]| {
        equation
            .log_likelihood(subject, parameter_row, observation_model)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let results: Vec<f64> = if let Some(flat_parameters) = parameters.as_slice() {
        let width = parameters.ncols();
        subject_slice
            .par_iter()
            .enumerate()
            .map(|(i, subject)| {
                let start = i * width;
                score_subject(subject, &flat_parameters[start..start + width])
            })
            .collect()
    } else {
        let parameter_rows = parameters
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        subject_slice
            .par_iter()
            .enumerate()
            .map(|(i, subject)| score_subject(subject, &parameter_rows[i]))
            .collect()
    };

    Ok(results)
}

/// Compute the log-likelihood for a single subject.
///
/// This is the single-subject equivalent of [`log_likelihood_batch`]: it is
/// [`Equation::log_likelihood`] with every error mapped to
/// `f64::NEG_INFINITY`.
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subject`: The subject data
/// - `params`: Parameter vector for this subject
/// - `observation_model`: The observation model, e.g. residual error models
///
/// # Returns
/// The log-likelihood for this subject. Returns `f64::NEG_INFINITY` on simulation error
/// or when the observation model names an output the equation does not declare
/// (use [`Equation::log_likelihood`] to get that error back).
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
pub fn log_likelihood_subject<O: ObservationModel + ?Sized>(
    equation: &impl Equation,
    subject: &Subject,
    params: &crate::Parameters,
    observation_model: &O,
) -> f64 {
    equation
        .log_likelihood(subject, params.as_slice(), observation_model)
        .unwrap_or(f64::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    use crate::data::event::{Observation, OutputLabel};
    use crate::Censor;

    #[test]
    fn test_log_likelihood_equals_log_of_likelihood() {
        // Create a prediction with an observation
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 10.5,
            outeq: OutputLabel::from(0),
            errorpoly: None,
            state: vec![10.5],
            occasion: 0,
            censoring: Censor::None,
        };

        // Create error model with additive error
        let error_models = crate::OutputModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
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
                outeq: OutputLabel::from(0),
                errorpoly: None,
                state: vec![10.1],
                occasion: 0,
                censoring: Censor::None,
            },
            Prediction {
                time: 2.0,
                observation: Some(8.0),
                prediction: 8.2,
                outeq: OutputLabel::from(0),
                errorpoly: None,
                state: vec![8.2],
                occasion: 0,
                censoring: Censor::None,
            },
        ];

        let subject_predictions = SubjectPredictions::from(predictions);
        let error_models = crate::OutputModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
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
        let errors = crate::OutputModels::new();
        assert_eq!(preds.log_likelihood(&errors).unwrap(), 0.0); // log(1) = 0
    }

    #[test]
    fn test_log_likelihood_combines_observations() {
        let mut preds = SubjectPredictions::default();
        let obs = Observation::new(0.0, Some(1.0), 0, None, 0, Censor::None);
        preds.add_prediction(obs.to_prediction(1.0, vec![]));

        let error_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0);
        let errors = crate::OutputModels::new().add(0, error_model).unwrap();

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
