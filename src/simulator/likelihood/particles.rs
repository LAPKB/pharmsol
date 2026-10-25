//! Particle predictions from a stochastic model.
//!
//! [`ParticlePredictions`] is what an [`SDE`](crate::SDE) returns: one
//! predicted value per particle and observation, stored as a matrix next to a
//! single list of the observations they predict.

use std::borrow::Cow;

use ndarray::{Array2, ArrayView1};

use crate::data::event::Observation;
use crate::{PharmsolError, PointPredictions};

use super::{Prediction, SubjectPredictions};

/// Particle predictions for one subject.
///
/// The observations (time, output, observed value, censoring, error
/// polynomial) are stored once and shared by every particle; the predicted
/// values are an `nparticles × nobservations` matrix.
///
/// The particles either ran free or were filtered:
///
/// - **Free-running** ([`SDE::predict`](crate::Equation::predict)): each row is
///   one particle's own path through every observation, and every particle
///   counts equally.
/// - **Filtered** ([`SDE::simulate`](crate::Equation::simulate) with an
///   observation model): at each observation the particles were weighted by
///   the observation and then resampled. Column `j` holds the particles just
///   before that resampling, with their normalized weights in
///   [`Self::weights`]. Resampling reorders particles, so a row is not one
///   particle's path.
///
/// Summaries ([`Self::mean`], [`Self::quantile`], [`Self::point_summary`])
/// use the weights when there are any, so for filtered particles they
/// describe the filtering distribution at each observation.
#[derive(Debug, Clone)]
pub struct ParticlePredictions {
    observations: Vec<Observation>,
    values: Array2<f64>,
    weights: Option<Array2<f64>>,
}

impl ParticlePredictions {
    /// Build particle predictions from their parts.
    ///
    /// `values` has one row per particle and one column per observation.
    /// `weights`, when given, has the same shape and marks the particles as
    /// filtered; each column should sum to one.
    pub fn new(
        observations: Vec<Observation>,
        values: Array2<f64>,
        weights: Option<Array2<f64>>,
    ) -> Result<Self, PharmsolError> {
        if values.ncols() != observations.len() {
            return Err(PharmsolError::OtherError(format!(
                "particle values have {} columns but there are {} observations",
                values.ncols(),
                observations.len()
            )));
        }
        if let Some(weights) = &weights {
            if weights.dim() != values.dim() {
                return Err(PharmsolError::OtherError(format!(
                    "particle weights have shape {:?} but values have shape {:?}",
                    weights.dim(),
                    values.dim()
                )));
            }
        }
        Ok(Self {
            observations,
            values,
            weights,
        })
    }

    /// Number of particles.
    pub fn nparticles(&self) -> usize {
        self.values.nrows()
    }

    /// Number of observations.
    pub fn nobservations(&self) -> usize {
        self.observations.len()
    }

    /// `(nparticles, nobservations)`.
    pub fn dim(&self) -> (usize, usize) {
        self.values.dim()
    }

    /// The observations, in the order of the value columns.
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// Predicted values: one row per particle, one column per observation.
    pub fn values(&self) -> &Array2<f64> {
        &self.values
    }

    /// Normalized filter weights, shaped like [`Self::values`], for filtered
    /// particles; `None` for free-running ones.
    pub fn weights(&self) -> Option<&Array2<f64>> {
        self.weights.as_ref()
    }

    /// Whether the particles were filtered against the observations (see the
    /// type documentation), rather than running free.
    pub fn is_filtered(&self) -> bool {
        self.weights.is_some()
    }

    /// Mean prediction at each observation, weighted for filtered particles.
    ///
    /// Particles with zero weight are left out, so one that diverged (and was
    /// ruled out by the observation) does not turn the mean into NaN.
    pub fn mean(&self) -> Vec<f64> {
        (0..self.nobservations())
            .map(|j| {
                let values = self.values.column(j);
                match &self.weights {
                    None => values.sum() / values.len() as f64,
                    Some(weights) => values
                        .iter()
                        .zip(weights.column(j))
                        .filter(|(_, &weight)| weight > 0.0)
                        .map(|(value, weight)| value * weight)
                        .sum(),
                }
            })
            .collect()
    }

    /// The `q`-quantile of the predictions at each observation, weighted for
    /// filtered particles.
    ///
    /// This is the inverse of the (weighted) empirical distribution: the
    /// smallest particle value whose cumulative weight reaches `q`. Errors
    /// unless `0 <= q <= 1`.
    pub fn quantile(&self, q: f64) -> Result<Vec<f64>, PharmsolError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PharmsolError::OtherError(format!(
                "quantile must be between 0 and 1, got {q}"
            )));
        }
        Ok((0..self.nobservations())
            .map(|j| {
                weighted_quantile(
                    self.values.column(j),
                    self.weights.as_ref().map(|weights| weights.column(j)),
                    q,
                )
            })
            .collect())
    }

    /// One particle's predictions, or `None` if `index` is out of range.
    ///
    /// For filtered particles this is the `index`-th particle at each
    /// observation, not one path (see the type documentation).
    pub fn particle(&self, index: usize) -> Option<SubjectPredictions> {
        let row = self.values.outer_iter().nth(index)?;
        Some(self.with_values(row.iter().copied()))
    }

    /// One prediction per observation, carrying the [`Self::mean`].
    ///
    /// The predictions have no state vector.
    pub fn point_summary(&self) -> SubjectPredictions {
        self.with_values(self.mean())
    }

    fn with_values(&self, values: impl IntoIterator<Item = f64>) -> SubjectPredictions {
        self.observations
            .iter()
            .zip(values)
            .map(|(observation, value)| observation.to_prediction(value, Vec::new()))
            .collect::<Vec<Prediction>>()
            .into()
    }
}

impl PointPredictions for ParticlePredictions {
    /// The [`ParticlePredictions::point_summary`]: the (weighted) mean
    /// prediction at each observation.
    fn point(&self) -> Cow<'_, [Prediction]> {
        Cow::Owned(self.point_summary().into_predictions())
    }
}

/// Smallest value whose cumulative weight reaches `q` of the total.
fn weighted_quantile(values: ArrayView1<f64>, weights: Option<ArrayView1<f64>>, q: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = match weights {
        None => values.iter().map(|&value| (value, 1.0)).collect(),
        Some(weights) => values
            .iter()
            .zip(weights)
            .filter(|(_, &weight)| weight > 0.0)
            .map(|(&value, &weight)| (value, weight))
            .collect(),
    };
    if pairs.is_empty() {
        return f64::NAN;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let target = q * pairs.iter().map(|(_, weight)| weight).sum::<f64>();
    let mut cumulative = 0.0;
    for &(value, weight) in &pairs {
        cumulative += weight;
        if cumulative >= target {
            return value;
        }
    }
    pairs[pairs.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Censor;
    use ndarray::array;

    fn observations() -> Vec<Observation> {
        vec![
            Observation::new(1.0, Some(2.0), 0, None, 0, Censor::None),
            Observation::new(2.0, None, 0, None, 0, Censor::None),
        ]
    }

    #[test]
    fn free_running_summaries_count_particles_equally() {
        let particles = ParticlePredictions::new(
            observations(),
            array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            None,
        )
        .unwrap();

        assert!(!particles.is_filtered());
        assert_eq!(particles.dim(), (4, 2));
        assert_eq!(particles.mean(), vec![2.5, 25.0]);
        assert_eq!(particles.quantile(0.0).unwrap(), vec![1.0, 10.0]);
        assert_eq!(particles.quantile(0.5).unwrap(), vec![2.0, 20.0]);
        assert_eq!(particles.quantile(1.0).unwrap(), vec![4.0, 40.0]);
        assert!(particles.quantile(1.5).is_err());

        let summary = particles.point_summary();
        assert_eq!(summary.flat_predictions(), vec![2.5, 25.0]);
        assert_eq!(summary.flat_times(), vec![1.0, 2.0]);
        assert_eq!(summary.flat_observations(), vec![Some(2.0), None]);

        let third = particles.particle(2).unwrap();
        assert_eq!(third.flat_predictions(), vec![3.0, 30.0]);
        assert!(particles.particle(4).is_none());
    }

    #[test]
    fn filtered_summaries_use_the_weights() {
        let particles = ParticlePredictions::new(
            observations(),
            array![[1.0, f64::NAN], [3.0, 5.0]],
            Some(array![[0.25, 0.0], [0.75, 1.0]]),
        )
        .unwrap();

        assert!(particles.is_filtered());
        // The zero-weight NaN particle is left out.
        assert_eq!(particles.mean(), vec![2.5, 5.0]);
        assert_eq!(particles.quantile(0.2).unwrap(), vec![1.0, 5.0]);
        assert_eq!(particles.quantile(0.5).unwrap(), vec![3.0, 5.0]);
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        assert!(ParticlePredictions::new(observations(), array![[1.0]], None).is_err());
        assert!(ParticlePredictions::new(
            observations(),
            array![[1.0, 2.0]],
            Some(array![[1.0], [1.0]]),
        )
        .is_err());
    }
}
