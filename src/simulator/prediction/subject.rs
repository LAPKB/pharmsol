//! Subject-level prediction containers.
//!
//! This module contains [`SubjectPredictions`] for holding all predictions
//! for a single subject.

use crate::simulator::prediction::Prediction;
use std::collections::BTreeMap;
type OccasionIndex = usize;

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
    occasions: Vec<OccasionIndex>,
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

    /// Create a new empty `SubjectPredictions` with a given subject identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            predictions: Vec::new(),
            occasions: Vec::new(),
        }
    }

    /// Returns a tuple vector with references to each prediction and occasion
    pub fn predictions(&self) -> Vec<(&Prediction, &usize)> {
        self.predictions.iter().zip(self.occasions.iter()).collect()
    }

    /// Return a BTreeMap of occasions to vectors of predictions for that occasion
    pub fn predictions_map(&self) -> BTreeMap<usize, Vec<&Prediction>> {
        let mut map: BTreeMap<usize, Vec<&Prediction>> = BTreeMap::new();
        for (prediction, occasion) in self.predictions() {
            map.entry(*occasion).or_default().push(prediction);
        }
        map
    }

    /// Get the subject identifier these predictions belong to.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Set the subject identifier these predictions belong to.
    pub fn set_id(&mut self, id: impl Into<String>) {
        self.id = id.into();
    }

    /// Get the occasion index for each prediction, parallel to
    /// [`SubjectPredictions::predictions`].
    pub fn occasions(&self) -> &Vec<OccasionIndex> {
        &self.occasions
    }

    /// Iterate over a reference to each prediction and its occasion index in parallel.
    pub fn iter(&self) -> impl Iterator<Item = (&Prediction, &OccasionIndex)> {
        self.predictions.iter().zip(self.occasions.iter())
    }

    /// Iterate over a mutable reference to each prediction and its occasion index in parallel.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Prediction, &mut OccasionIndex)> {
        self.predictions.iter_mut().zip(self.occasions.iter_mut())
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

/// Iterate over each subject prediction and its occasion index in parallel.
impl IntoIterator for SubjectPredictions {
    type Item = (Prediction, OccasionIndex);
    type IntoIter =
        std::iter::Zip<std::vec::IntoIter<Prediction>, std::vec::IntoIter<OccasionIndex>>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.into_iter().zip(self.occasions.into_iter())
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

        assert_eq!(subject_predictions.predictions.len(), 2);
        assert_eq!(subject_predictions.occasions.len(), 2);
    }
}
