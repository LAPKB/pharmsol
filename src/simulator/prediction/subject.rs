//! Subject-level prediction containers.
//!
//! This module contains [`SubjectPredictions`] for holding all predictions
//! for a single subject.

use crate::Predictions;

use super::point::Prediction;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject and supports borrowed
/// visitation of each point.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
}

impl Predictions for SubjectPredictions {
    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }

    fn for_each_prediction(&self, mut f: impl FnMut(&Prediction)) {
        for prediction in &self.predictions {
            f(prediction);
        }
    }
}

impl SubjectPredictions {
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

#[cfg(test)]
mod tests {
    use super::*;
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
