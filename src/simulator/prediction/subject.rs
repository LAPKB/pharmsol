//! Subject-level prediction containers.
//!
//! These containers mirror how observed data is stored: a [`SubjectPredictions`]
//! owns one [`OccasionPredictions`] per occasion, just as a
//! [`Subject`](crate::Subject) owns one [`Occasion`](crate::Occasion) per
//! occasion.

use crate::simulator::prediction::Prediction;
use std::collections::BTreeMap;
type OccasionIndex = usize;

/// Container for predictions associated with a occasion for a subject
#[derive(Debug, Clone, Default)]
pub struct OccasionPredictions {
    occasion: OccasionIndex,
    predictions: Vec<Prediction>,
}

impl OccasionPredictions {
    /// Create a new empty container for the given occasion index.
    pub fn new(occasion: OccasionIndex) -> Self {
        Self {
            occasion,
            predictions: Vec::new(),
        }
    }

    /// Get the occasion index these predictions belong to.
    pub fn occasion(&self) -> OccasionIndex {
        self.occasion
    }

    /// Get the predictions recorded for this occasion.
    pub fn predictions(&self) -> &[Prediction] {
        &self.predictions
    }

    /// Add a prediction to this occasion.
    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.predictions.push(prediction);
    }

    /// Iterate over a reference to each prediction in this occasion.
    pub fn iter(&self) -> std::slice::Iter<'_, Prediction> {
        self.predictions.iter()
    }

    /// Iterate over a mutable reference to each prediction in this occasion.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Prediction> {
        self.predictions.iter_mut()
    }

    /// Get the number of predictions in this occasion.
    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    /// Check if this occasion holds no predictions.
    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

impl IntoIterator for OccasionPredictions {
    type Item = Prediction;
    type IntoIter = std::vec::IntoIter<Prediction>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.into_iter()
    }
}

impl<'a> IntoIterator for &'a OccasionPredictions {
    type Item = &'a Prediction;
    type IntoIter = std::slice::Iter<'a, Prediction>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.iter()
    }
}

impl<'a> IntoIterator for &'a mut OccasionPredictions {
    type Item = &'a mut Prediction;
    type IntoIter = std::slice::IterMut<'a, Prediction>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.iter_mut()
    }
}

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject, grouped by occasion.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    id: String,
    occasions: Vec<OccasionPredictions>,
}

impl SubjectPredictions {
    /// Create a new empty `SubjectPredictions` with a given subject identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            occasions: Vec::new(),
        }
    }

    /// Add a new prediction to the collection.
    ///
    /// The container for `occasion` is created on first use.
    ///
    /// # Parameters
    /// - `prediction`: The prediction to add
    /// - `occasion`: The occasion index that produced this prediction
    pub fn add_prediction(&mut self, prediction: Prediction, occasion: OccasionIndex) {
        match self
            .occasions
            .iter_mut()
            .find(|entry| entry.occasion == occasion)
        {
            Some(entry) => entry.add_prediction(prediction),
            None => {
                let mut entry = OccasionPredictions::new(occasion);
                entry.add_prediction(prediction);
                self.occasions.push(entry);
            }
        }
    }

    /// Returns a vector with references to every prediction, across occasions.
    pub fn predictions(&self) -> Vec<&Prediction> {
        self.predictions_iter().collect()
    }

    /// Returns the occasion index of every prediction, parallel to
    /// [`SubjectPredictions::predictions`].
    pub fn prediction_occasions(&self) -> Vec<OccasionIndex> {
        self.occasions
            .iter()
            .flat_map(|entry| std::iter::repeat_n(entry.occasion, entry.len()))
            .collect()
    }

    /// Returns every prediction paired with the occasion index that produced it.
    pub fn predictions_with_occasions(&self) -> Vec<(&Prediction, OccasionIndex)> {
        self.occasions
            .iter()
            .flat_map(|entry| entry.iter().map(move |p| (p, entry.occasion)))
            .collect()
    }

    /// Return a BTreeMap of occasions to vectors of predictions for that occasion
    pub fn predictions_map(&self) -> BTreeMap<OccasionIndex, Vec<&Prediction>> {
        let mut map: BTreeMap<OccasionIndex, Vec<&Prediction>> = BTreeMap::new();
        for entry in &self.occasions {
            map.entry(entry.occasion).or_default().extend(entry.iter());
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

    /// Get the per-occasion prediction containers.
    pub fn occasions(&self) -> &[OccasionPredictions] {
        &self.occasions
    }

    /// Get the predictions recorded for a given occasion index.
    pub fn occasion(&self, occasion: OccasionIndex) -> Option<&OccasionPredictions> {
        self.occasions
            .iter()
            .find(|entry| entry.occasion == occasion)
    }

    /// Iterate over a reference to each occasion.
    pub fn iter(&self) -> std::slice::Iter<'_, OccasionPredictions> {
        self.occasions.iter()
    }

    /// Iterate over a mutable reference to each occasion.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, OccasionPredictions> {
        self.occasions.iter_mut()
    }

    /// Iterate over a reference to every prediction, across occasions.
    pub fn predictions_iter(&self) -> impl Iterator<Item = &Prediction> {
        self.occasions.iter().flat_map(OccasionPredictions::iter)
    }

    /// Iterate over a mutable reference to every prediction, across occasions.
    pub fn predictions_iter_mut(&mut self) -> impl Iterator<Item = &mut Prediction> {
        self.occasions
            .iter_mut()
            .flat_map(OccasionPredictions::iter_mut)
    }

    /// Get the number of occasions holding predictions.
    pub fn len(&self) -> usize {
        self.occasions.len()
    }

    /// Check if the subject holds no occasions.
    pub fn is_empty(&self) -> bool {
        self.occasions.is_empty()
    }
}

impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            id: String::new(),
            occasions: vec![OccasionPredictions {
                occasion: 0,
                predictions,
            }],
        }
    }
}

/// Iterate over each occasion of the subject.
impl IntoIterator for SubjectPredictions {
    type Item = OccasionPredictions;
    type IntoIter = std::vec::IntoIter<OccasionPredictions>;

    fn into_iter(self) -> Self::IntoIter {
        self.occasions.into_iter()
    }
}

impl<'a> IntoIterator for &'a SubjectPredictions {
    type Item = &'a OccasionPredictions;
    type IntoIter = std::slice::Iter<'a, OccasionPredictions>;

    fn into_iter(self) -> Self::IntoIter {
        self.occasions.iter()
    }
}

impl<'a> IntoIterator for &'a mut SubjectPredictions {
    type Item = &'a mut OccasionPredictions;
    type IntoIter = std::slice::IterMut<'a, OccasionPredictions>;

    fn into_iter(self) -> Self::IntoIter {
        self.occasions.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputLabel;

    fn prediction(time: f64, value: f64) -> Prediction {
        Prediction {
            time,
            observation: Some(value),
            prediction: value + 1.0,
            outeq: OutputLabel::new("cp"),
            ..Default::default()
        }
    }

    #[test]
    fn from_vec_collapses_into_a_single_occasion() {
        let subject_predictions =
            SubjectPredictions::from(vec![prediction(1.0, 10.0), prediction(2.0, 8.0)]);

        assert_eq!(subject_predictions.occasions().len(), 1);
        assert_eq!(subject_predictions.predictions().len(), 2);
        assert_eq!(subject_predictions.prediction_occasions(), vec![0, 0]);
    }

    #[test]
    fn add_prediction_groups_by_occasion() {
        let mut subject_predictions = SubjectPredictions::new("id");
        subject_predictions.add_prediction(prediction(1.0, 10.0), 0);
        subject_predictions.add_prediction(prediction(2.0, 8.0), 1);
        subject_predictions.add_prediction(prediction(3.0, 6.0), 0);

        assert_eq!(subject_predictions.id(), "id");
        assert_eq!(subject_predictions.occasions().len(), 2);
        assert_eq!(subject_predictions.occasion(0).unwrap().len(), 2);
        assert_eq!(subject_predictions.occasion(1).unwrap().len(), 1);
        assert_eq!(
            subject_predictions.prediction_occasions(),
            vec![0usize, 0, 1]
        );

        let paired = subject_predictions.predictions_with_occasions();
        assert_eq!(paired.len(), 3);
        assert_eq!(paired[1].0.time(), 3.0);
        assert_eq!(paired[1].1, 0);

        let map = subject_predictions.predictions_map();
        assert_eq!(map[&0].len(), 2);
        assert_eq!(map[&1].len(), 1);
    }
}
