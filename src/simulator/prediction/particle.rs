//! Particle-level prediction containers produced by stochastic (SDE) models.

use crate::simulator::prediction::Prediction;
use crate::PharmsolError;
use ndarray::{concatenate, Array2, Axis};

type OccasionIndex = usize;

/// Container for particle predictions produced by a stochastic model.
///
/// Predictions are stored as a matrix with one row per particle and one column
/// per observation, in observation order. The occasion index that produced each
/// column is tracked alongside the matrix.
#[derive(Debug, Clone, Default)]
pub struct ParticlePredictions {
    id: String,
    nparticles: usize,
    predictions: Array2<Prediction>,
    occasions: Vec<OccasionIndex>,
}

impl ParticlePredictions {
    /// Create a new empty container for `nparticles` particles.
    pub fn new(id: impl Into<String>, nparticles: usize) -> Self {
        Self {
            id: id.into(),
            nparticles,
            predictions: Array2::from_shape_fn((nparticles, 0), |_| Prediction::default()),
            occasions: Vec::new(),
        }
    }

    /// Append one observation column, holding one prediction per particle.
    ///
    /// # Errors
    /// Returns an error if `predictions` does not hold exactly one entry per particle.
    pub fn add_predictions(
        &mut self,
        predictions: Vec<Prediction>,
        occasion: OccasionIndex,
    ) -> Result<(), PharmsolError> {
        let column = Array2::from_shape_vec((self.nparticles, 1), predictions)?;
        self.predictions = concatenate(Axis(1), &[self.predictions.view(), column.view()])?;
        self.occasions.push(occasion);
        Ok(())
    }

    /// Get the subject identifier these predictions belong to.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Set the subject identifier these predictions belong to.
    pub fn set_id(&mut self, id: impl Into<String>) {
        self.id = id.into();
    }

    /// Get the number of particles, i.e. the number of rows.
    pub fn nparticles(&self) -> usize {
        self.nparticles
    }

    /// Get the number of observations, i.e. the number of columns.
    pub fn nobservations(&self) -> usize {
        self.predictions.ncols()
    }

    /// Check if the container holds no observations.
    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty() || self.predictions.ncols() == 0
    }

    /// Borrow the full particle matrix, with particles as rows and observations as columns.
    pub fn particles(&self) -> &Array2<Prediction> {
        &self.predictions
    }

    /// Borrow the prediction for a single particle at a single observation.
    pub fn get(&self, particle: usize, observation: usize) -> Option<&Prediction> {
        self.predictions.get((particle, observation))
    }

    /// Get the occasion index of each observation, parallel to the matrix columns.
    pub fn occasions(&self) -> &[OccasionIndex] {
        &self.occasions
    }

    /// Get one prediction per observation, averaged across all particles.
    pub fn mean_predictions(&self) -> Vec<Prediction> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.predictions.ncols());

        for col in 0..self.predictions.ncols() {
            let column = self.predictions.column(col);

            let mean_prediction: f64 = column
                .iter()
                .map(|pred: &Prediction| pred.prediction())
                .sum::<f64>()
                / self.predictions.nrows() as f64;

            let mut prediction = column.first().unwrap().clone();
            prediction.set_prediction(mean_prediction);
            result.push(prediction);
        }

        result
    }
}

impl std::ops::Index<[usize; 2]> for ParticlePredictions {
    type Output = Prediction;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.predictions[index]
    }
}

impl std::ops::Index<(usize, usize)> for ParticlePredictions {
    type Output = Prediction;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.predictions[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputLabel;

    fn prediction(value: f64) -> Prediction {
        Prediction {
            time: 1.0,
            observation: None,
            prediction: value,
            outeq: OutputLabel::new("cp"),
            ..Default::default()
        }
    }

    #[test]
    fn mean_predictions_average_across_particles() {
        let mut predictions = ParticlePredictions::new("id", 2);
        predictions
            .add_predictions(vec![prediction(1.0), prediction(3.0)], 0)
            .unwrap();
        predictions
            .add_predictions(vec![prediction(2.0), prediction(6.0)], 1)
            .unwrap();

        assert_eq!(predictions.nparticles(), 2);
        assert_eq!(predictions.nobservations(), 2);
        assert_eq!(predictions.occasions(), &[0, 1]);
        assert_eq!(predictions[[1, 0]].prediction(), 3.0);

        let means: Vec<f64> = predictions
            .mean_predictions()
            .iter()
            .map(Prediction::prediction)
            .collect();
        assert_eq!(means, vec![2.0, 4.0]);
    }

    #[test]
    fn empty_container_has_no_means() {
        let predictions = ParticlePredictions::new("id", 4);
        assert!(predictions.is_empty());
        assert!(predictions.mean_predictions().is_empty());
    }
}
