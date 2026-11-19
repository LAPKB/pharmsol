//! Subject-level and population-level prediction containers.
//!
//! This module contains [`SubjectPredictions`] for holding all predictions
//! for a single subject, and [`PopulationPredictions`] for population-level
//! predictions.

use std::borrow::Cow;

use ndarray::{Array2, ShapeBuilder};

use super::prediction::Prediction;
use super::PointPredictions;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject along with methods
/// for aggregate error metrics.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
}

impl PointPredictions for SubjectPredictions {
    /// The predictions themselves, borrowed.
    fn point(&self) -> Cow<'_, [Prediction]> {
        Cow::Borrowed(&self.predictions)
    }
}

impl SubjectPredictions {
    /// Sum of squared errors over all observed predictions.
    pub fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .filter_map(|p| p.observation().map(|obs| (obs - p.prediction()).powi(2)))
            .sum()
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

    /// Take the vector of predictions.
    pub fn into_predictions(self) -> Vec<Prediction> {
        self.predictions
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
    use crate::data::event::OutputLabel;
    use crate::Censor;

    #[test]
    fn test_squared_error_skips_missing_observations() {
        let predictions = vec![
            Prediction {
                time: 1.0,
                observation: Some(10.0),
                prediction: 10.5,
                outeq: OutputLabel::from(0),
                errorpoly: None,
                state: vec![10.5],
                occasion: 0,
                censoring: Censor::None,
            },
            Prediction {
                time: 2.0,
                observation: None,
                prediction: 8.2,
                ..Default::default()
            },
            Prediction {
                time: 3.0,
                observation: Some(8.0),
                prediction: 6.0,
                ..Default::default()
            },
        ];

        let subject_predictions = SubjectPredictions::from(predictions);
        assert_eq!(subject_predictions.squared_error(), 4.25);
        assert_eq!(SubjectPredictions::default().squared_error(), 0.0);
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
