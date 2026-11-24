//! A single observation paired with its prediction.
//!
//! This module contains the [`Prediction`] struct which holds a single
//! observation-prediction pair along with the observation's metadata
//! (output, censoring, error polynomial), so downstream code can score it.

use crate::data::event::{Observation, OutputLabel};
use crate::{Censor, ErrorPoly};

/// Prediction holds an observation and its prediction at a single time point.
///
/// Besides the predicted value it carries everything about the observation a
/// downstream error model needs to score it: the observed value, output
/// label, censoring and the observation's own error polynomial, if any.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub(crate) time: f64,
    pub(crate) observation: Option<f64>,
    pub(crate) prediction: f64,
    pub(crate) outeq: OutputLabel,
    pub(crate) errorpoly: Option<ErrorPoly>,
    pub(crate) state: Vec<f64>,
    pub(crate) occasion: usize,
    pub(crate) censoring: Censor,
}

impl Prediction {
    /// Get the time point of this prediction.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the observed value.
    pub fn observation(&self) -> Option<f64> {
        self.observation
    }

    /// Get the predicted value.
    pub fn prediction(&self) -> f64 {
        self.prediction
    }

    /// Get the output label of this prediction.
    pub fn outeq(&self) -> &OutputLabel {
        &self.outeq
    }

    /// Get the error polynomial coefficients, if available.
    pub fn errorpoly(&self) -> Option<ErrorPoly> {
        self.errorpoly
    }

    /// Calculate the raw prediction error (prediction - observation).
    pub fn prediction_error(&self) -> Option<f64> {
        self.observation.map(|obs| self.prediction - obs)
    }

    /// Calculate the percentage error as (prediction - observation)/observation * 100.
    pub fn percentage_error(&self) -> Option<f64> {
        self.observation
            .map(|obs| ((self.prediction - obs) / obs) * 100.0)
    }

    /// Calculate the absolute error |prediction - observation|.
    pub fn absolute_error(&self) -> Option<f64> {
        self.observation.map(|obs| (self.prediction - obs).abs())
    }

    /// Calculate the squared error (prediction - observation)².
    pub fn squared_error(&self) -> Option<f64> {
        self.observation.map(|obs| (self.prediction - obs).powi(2))
    }

    /// Get the state vector at this prediction point
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Get the occasion index
    pub fn occasion(&self) -> usize {
        self.occasion
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        &mut self.occasion
    }

    /// Get the censoring status
    pub fn censoring(&self) -> Censor {
        self.censoring
    }

    /// Create an [Observation] from this prediction
    pub fn to_observation(&self) -> Observation {
        Observation::new(
            self.time,
            self.observation,
            self.outeq.clone(),
            self.errorpoly,
            self.occasion,
            self.censoring,
        )
    }
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            time: 0.0,
            observation: None,
            prediction: 0.0,
            outeq: OutputLabel::from(0),
            errorpoly: None,
            state: vec![],
            occasion: 0,
            censoring: Censor::None,
        }
    }
}

impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let obs_str = match self.observation {
            Some(obs) => format!("{:.4}", obs),
            None => "NA".to_string(),
        };
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.4}\tOuteq: {}",
            self.time, obs_str, self.prediction, self.outeq
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prediction(obs: f64, pred: f64) -> Prediction {
        Prediction {
            time: 1.0,
            observation: Some(obs),
            prediction: pred,
            outeq: OutputLabel::from(0),
            errorpoly: None,
            state: vec![pred],
            occasion: 0,
            censoring: Censor::None,
        }
    }

    #[test]
    fn test_error_metrics() {
        let prediction = create_test_prediction(10.0, 12.0);

        assert_eq!(prediction.prediction_error(), Some(2.0));
        assert_eq!(prediction.absolute_error(), Some(2.0));
        assert_eq!(prediction.squared_error(), Some(4.0));
        assert_eq!(prediction.percentage_error(), Some(20.0));
    }

    #[test]
    fn missing_observation_has_no_error_metrics() {
        let prediction = Prediction {
            observation: None,
            prediction: 10.0,
            ..Default::default()
        };
        assert_eq!(prediction.prediction_error(), None);
        assert_eq!(prediction.squared_error(), None);
    }

    #[test]
    fn to_observation_keeps_the_observation_metadata() {
        let poly = ErrorPoly::new(0.1, 0.2, 0.0, 0.0);
        let observation = Observation::new(2.0, Some(3.0), "cp", Some(poly), 1, Censor::BLOQ);
        let prediction = observation.to_prediction(4.0, vec![4.0]);

        assert_eq!(prediction.errorpoly(), Some(poly));
        assert_eq!(prediction.censoring(), Censor::BLOQ);
        assert_eq!(prediction.outeq().as_str(), "cp");
        let back = prediction.to_observation();
        assert_eq!(back.time(), observation.time());
        assert_eq!(back.value(), observation.value());
        assert_eq!(back.errorpoly(), observation.errorpoly());
        assert_eq!(back.occasion(), observation.occasion());
    }
}
