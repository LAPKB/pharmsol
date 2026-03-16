//! Single-point prediction and likelihood calculation.
//!
//! This module contains the [`Prediction`] struct which holds a single
//! observation-prediction pair along with metadata needed for likelihood
//! calculation.

use crate::data::error_model::AssayErrorModels;
use crate::data::event::Observation;
use crate::{Censor, ErrorPoly, PharmsolError};

use super::distributions::{lognormccdf, lognormcdf, lognormpdf};

/// Prediction holds an observation and its prediction at a single time point.
///
/// This struct contains all information needed to calculate the likelihood
/// contribution of a single observation.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub(crate) time: f64,
    pub(crate) observation: Option<f64>,
    pub(crate) prediction: f64,
    pub(crate) outeq: usize,
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

    /// Set the predicted value
    pub(crate) fn set_prediction(&mut self, prediction: f64) {
        self.prediction = prediction;
    }

    /// Get the output equation index.
    pub fn outeq(&self) -> usize {
        self.outeq
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

    /// Calculate the log-likelihood of this prediction given an error model.
    ///
    /// This method is numerically stable and handles:
    /// - Regular observations: uses log-normal PDF
    /// - BLOQ (below limit of quantification): uses log-CDF
    /// - ALOQ (above limit of quantification): uses log-survival function
    ///
    /// # Error Model
    /// Uses observation-based sigma from [`AssayErrorModels`], which is appropriate
    /// for non-parametric algorithms (NPAG, NPOD). For parametric algorithms
    /// (SAEM, FOCE), use [`crate::ResidualErrorModels`] directly.
    ///
    /// # Parameters
    /// - `error_models`: The error models to use for sigma calculation
    ///
    /// # Returns
    /// The log-likelihood value, or an error if:
    /// - The observation is missing
    /// - The log-likelihood is non-finite
    ///
    /// # Example
    /// ```ignore
    /// let log_lik = prediction.log_likelihood(&error_models)?;
    /// ```
    #[inline]
    pub fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        if self.observation.is_none() {
            return Err(PharmsolError::MissingObservation);
        }

        let sigma = error_models.sigma(self)?;
        let obs = self.observation.unwrap();

        let log_lik = match self.censoring {
            Censor::None => lognormpdf(obs, self.prediction, sigma),
            Censor::BLOQ => lognormcdf(obs, self.prediction, sigma)?,
            Censor::ALOQ => lognormccdf(obs, self.prediction, sigma)?,
        };

        if log_lik.is_finite() {
            Ok(log_lik)
        } else {
            Err(PharmsolError::NonFiniteLikelihood(log_lik))
        }
    }

    /// Calculate the likelihood of this prediction.
    ///
    /// **Deprecated**: Use [`log_likelihood`](Self::log_likelihood) instead for
    /// better numerical stability. This method is provided for backward
    /// compatibility and simply exponentiates the log-likelihood.
    ///
    /// # Parameters
    /// - `error_models`: The error models to use for sigma calculation
    ///
    /// # Returns
    /// The likelihood value (exp of log-likelihood)
    #[deprecated(
        since = "0.23.0",
        note = "Use log_likelihood() instead for better numerical stability"
    )]
    pub fn likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
        let log_lik = self.log_likelihood(error_models)?;
        let lik = log_lik.exp();

        if lik.is_finite() {
            Ok(lik)
        } else if lik == 0.0 {
            Err(PharmsolError::ZeroLikelihood)
        } else {
            Err(PharmsolError::NonFiniteLikelihood(lik))
        }
    }

    /// Get the state vector at this prediction point
    pub fn state(&self) -> &Vec<f64> {
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
            self.outeq,
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
            outeq: 0,
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
            "Time: {:.2}\tObs: {:.4}\tPred: {:.4}\tOuteq: {:.2}",
            self.time, obs_str, self.prediction, self.outeq
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};

    fn create_test_prediction(obs: f64, pred: f64) -> Prediction {
        Prediction {
            time: 1.0,
            observation: Some(obs),
            prediction: pred,
            outeq: 0,
            errorpoly: None,
            state: vec![pred],
            occasion: 0,
            censoring: Censor::None,
        }
    }

    fn create_error_models() -> AssayErrorModels {
        AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
            )
            .unwrap()
    }

    #[test]
    fn test_log_likelihood_basic() {
        let prediction = create_test_prediction(10.0, 10.5);
        let error_models = create_error_models();

        let log_lik = prediction.log_likelihood(&error_models).unwrap();
        assert!(log_lik.is_finite());
        assert!(log_lik < 0.0); // Log likelihood should be negative
    }

    #[test]
    fn test_log_likelihood_numerical_stability() {
        // Test with values that would cause very small likelihood
        let prediction = create_test_prediction(10.0, 30.0); // 20 sigma away
        let error_models = create_error_models();

        let log_lik = prediction.log_likelihood(&error_models).unwrap();
        assert!(log_lik.is_finite());
        assert!(log_lik < -100.0); // Should be very negative
    }

    #[test]
    fn test_log_likelihood_extreme() {
        // Test with truly extreme values
        let prediction = create_test_prediction(10.0, 50.0); // 40 sigma away
        let error_models = create_error_models();

        let log_lik = prediction.log_likelihood(&error_models).unwrap();
        assert!(log_lik.is_finite());
        assert!(
            log_lik < -700.0 && log_lik > -900.0,
            "log_lik ({}) should be approximately -800",
            log_lik
        );
    }

    #[test]
    fn test_missing_observation() {
        let prediction = Prediction {
            time: 1.0,
            observation: None,
            prediction: 10.0,
            ..Default::default()
        };
        let error_models = create_error_models();

        let result = prediction.log_likelihood(&error_models);
        assert!(matches!(result, Err(PharmsolError::MissingObservation)));
    }

    #[test]
    fn test_error_metrics() {
        let prediction = create_test_prediction(10.0, 12.0);

        assert_eq!(prediction.prediction_error(), Some(2.0));
        assert_eq!(prediction.absolute_error(), Some(2.0));
        assert_eq!(prediction.squared_error(), Some(4.0));
        assert_eq!(prediction.percentage_error(), Some(20.0));
    }
}
