//! Subject-level predictions and likelihood calculations.
//!
//! This module contains [`SubjectPredictions`] for holding all predictions
//! for a single subject, and [`PopulationPredictions`] for population-level
//! predictions.

use ndarray::{Array2, ShapeBuilder};

use crate::data::error_model::AssayErrorModels;
use crate::{PharmsolError, Predictions};

use super::prediction::Prediction;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject along with methods
/// for calculating aggregate likelihood and error metrics.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
}

impl Predictions for SubjectPredictions {
    fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .filter_map(|p| p.observation().map(|obs| (obs - p.prediction()).powi(2)))
            .sum()
    }

    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }

    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        SubjectPredictions::log_likelihood(self, error_models)
    }
}

impl SubjectPredictions {
    /// Calculate the log-likelihood of all predictions given an error model.
    ///
    /// This sums the log-likelihood of each prediction to get the joint log-likelihood.
    /// This is numerically stable and avoids underflow issues that can occur
    /// when computing products of small probabilities.
    ///
    /// # Error Model
    /// Uses observation-based sigma from [`AssayErrorModels`], which is appropriate
    /// for non-parametric algorithms (NPAG, NPOD). For parametric algorithms
    /// (SAEM, FOCE), use [`crate::ResidualErrorModels`] directly.
    ///
    /// # Parameters
    /// - `error_models`: The error models to use for calculating the likelihood
    ///
    /// # Returns
    /// The sum of all individual prediction log-likelihoods.
    /// Returns 0.0 for empty prediction sets (log of 1.0).
    ///
    /// # Example
    /// ```ignore
    /// let log_lik = subject_predictions.log_likelihood(&error_models)?;
    /// ```
    pub fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        if self.predictions.is_empty() {
            return Ok(0.0);
        }

        let log_liks: Result<Vec<f64>, _> = self
            .predictions
            .iter()
            .filter(|p| p.observation().is_some())
            .map(|p| p.log_likelihood(error_models))
            .collect();

        log_liks.map(|lls| lls.iter().sum())
    }

    /// Calculate the likelihood of all predictions.
    ///
    /// **Deprecated**: Use [`log_likelihood`](Self::log_likelihood) instead for
    /// better numerical stability. This method exponentiates the log-likelihood.
    ///
    /// # Parameters
    /// - `error_models`: The error models to use for calculating the likelihood
    ///
    /// # Returns
    /// The product of all individual prediction likelihoods.
    /// Returns 1.0 for empty prediction sets.
    #[deprecated(
        since = "0.23.0",
        note = "Use log_likelihood() instead for better numerical stability"
    )]
    pub fn likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        match self.predictions.is_empty() {
            true => Ok(1.0),
            false => {
                let log_lik = self.log_likelihood(error_models)?;
                Ok(log_lik.exp())
            }
        }
    }

    /// Add a new prediction to the collection.
    ///
    /// # Parameters
    /// - `prediction`: The prediction to add
    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.predictions.push(prediction);
    }

    /// Get a reference to the vector of predictions.
    pub fn predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }

    /// Return a flat vector of prediction values.
    pub fn flat_predictions(&self) -> Vec<f64> {
        self.predictions.iter().map(|p| p.prediction()).collect()
    }

    /// Return a flat vector of time points.
    pub fn flat_times(&self) -> Vec<f64> {
        self.predictions.iter().map(|p| p.time()).collect()
    }

    /// Return a flat vector of observations.
    pub fn flat_observations(&self) -> Vec<Option<f64>> {
        self.predictions.iter().map(|p| p.observation()).collect()
    }
}

impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self { predictions }
    }
}

/// Container for predictions across a population of subjects.
///
/// This struct holds predictions for multiple subjects organized in a 2D array
/// where rows represent subjects and columns represent support points (or
/// other groupings).
pub struct PopulationPredictions {
    /// 2D array of subject predictions
    pub subject_predictions: Array2<SubjectPredictions>,
}

impl Default for PopulationPredictions {
    fn default() -> Self {
        Self {
            subject_predictions: Array2::default((0, 0).f()),
        }
    }
}

impl From<Array2<SubjectPredictions>> for PopulationPredictions {
    fn from(subject_predictions: Array2<SubjectPredictions>) -> Self {
        Self {
            subject_predictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    use crate::data::event::Observation;
    use crate::Censor;

    fn create_error_models() -> AssayErrorModels {
        AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .unwrap()
    }

    #[test]
    fn test_empty_predictions_log_likelihood() {
        let preds = SubjectPredictions::default();
        let errors = create_error_models();
        assert_eq!(preds.log_likelihood(&errors).unwrap(), 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_empty_predictions_likelihood() {
        let preds = SubjectPredictions::default();
        let errors = create_error_models();
        assert_eq!(preds.likelihood(&errors).unwrap(), 1.0);
    }

    #[test]
    fn test_log_likelihood_with_observations() {
        let mut preds = SubjectPredictions::default();
        let obs = Observation::new(0.0, Some(1.0), 0, None, 0, Censor::None);
        preds.add_prediction(obs.to_prediction(1.0, vec![]));

        let error_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0);
        let errors = AssayErrorModels::new().add(0, error_model).unwrap();

        let log_lik = preds.log_likelihood(&errors).unwrap();
        assert!(log_lik.is_finite());
        assert!(log_lik <= 0.0); // Log likelihood should be <= 0
    }

    #[test]
    fn test_multiple_observations() {
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
        let error_models = create_error_models();

        let log_lik = subject_predictions.log_likelihood(&error_models).unwrap();
        assert!(log_lik.is_finite());

        // Log-likelihood of multiple observations should be sum of individual log-likelihoods
        // (more negative than single observation)
    }

    #[test]
    fn test_flat_vectors() {
        let predictions = vec![
            Prediction {
                time: 1.0,
                observation: Some(10.0),
                prediction: 11.0,
                ..Default::default()
            },
            Prediction {
                time: 2.0,
                observation: Some(8.0),
                prediction: 9.0,
                ..Default::default()
            },
        ];

        let subject_predictions = SubjectPredictions::from(predictions);

        assert_eq!(subject_predictions.flat_times(), vec![1.0, 2.0]);
        assert_eq!(subject_predictions.flat_predictions(), vec![11.0, 9.0]);
        assert_eq!(
            subject_predictions.flat_observations(),
            vec![Some(10.0), Some(8.0)]
        );
    }
}
