//! Subject-level prediction containers.
//!
//! This module contains [`SubjectPredictions`] for holding all predictions
//! for a single subject.

use crate::{simulator::prediction::Prediction, Predictions};

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject, across every occasion, and
/// supports borrowed visitation of each point. The occasion index that produced
/// each prediction is tracked in parallel to [`SubjectPredictions::predictions`]
/// so it stays discernible without living on the core [`Prediction`] type.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    id: String,
    predictions: Vec<Prediction>,
    occasions: Vec<usize>,
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

    fn set_subject_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl SubjectPredictions {
    /// Add a new prediction to the collection.
    ///
    /// # Parameters
    /// - `prediction`: The prediction to add
    /// - `occasion`: The occasion index that produced this prediction
    pub fn add_prediction(&mut self, prediction: Prediction, occasion: usize) {
        self.predictions.push(prediction);
        self.occasions.push(occasion);
    }

    /// Get the subject identifier these predictions belong to.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Set the subject identifier these predictions belong to.
    pub fn set_id(&mut self, id: impl Into<String>) {
        self.id = id.into();
    }

    /// Get a reference to the vector of predictions.
    pub fn predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }

    /// Get the occasion index for each prediction, parallel to
    /// [`SubjectPredictions::predictions`].
    pub fn occasions(&self) -> &Vec<usize> {
        &self.occasions
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
        let occasions = vec![0; predictions.len()];
        Self {
            id: String::new(),
            predictions,
            occasions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputLabel;
    #[test]
    fn test_flat_vectors() {
        let predictions = vec![
            Prediction {
                time: 1.0,
                observation: Some(10.0),
                prediction: 11.0,
                outeq: OutputLabel::new("cp"),
                ..Default::default()
            },
            Prediction {
                time: 2.0,
                observation: Some(8.0),
                prediction: 9.0,
                outeq: OutputLabel::new("cp"),
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
