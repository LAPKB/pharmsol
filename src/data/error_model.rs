use crate::simulator::likelihood::Prediction;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [ErrorModel] defines how the standard deviation of observations is calculated
/// based on the type of error model used and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorModel {
    /// Additive error model, where error is independent of concentration
    ///
    /// Contains:
    /// * `lambda` - Lambda parameter for scaling errors
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    Additive {
        /// Lambda parameter for scaling errors
        lambda: f64,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: (f64, f64, f64, f64),
    },

    /// Proportional error model, where error scales with concentration
    ///
    /// Contains:
    /// * `gamma` - Gamma parameter for scaling errors
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    Proportional {
        /// Gamma parameter for scaling errors
        gamma: f64,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: (f64, f64, f64, f64),
    },
}

impl ErrorModel {
    /// Create a new additive error model
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `lambda` - Lambda parameter for scaling errors
    ///
    /// # Returns
    ///
    /// A new additive error model
    pub fn additive(poly: (f64, f64, f64, f64), lambda: f64) -> Self {
        Self::Additive { lambda, poly }
    }

    /// Create a new proportional error model
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gamma` - Gamma parameter for scaling errors
    ///
    /// # Returns
    ///
    /// A new proportional error model
    pub fn proportional(poly: (f64, f64, f64, f64), gamma: f64) -> Self {
        Self::Proportional { gamma, poly }
    }

    /// Get the error polynomial coefficients
    ///
    /// # Returns
    ///
    /// The error polynomial coefficients (c0, c1, c2, c3)
    pub fn polynomial(&self) -> (f64, f64, f64, f64) {
        match self {
            Self::Additive { poly, .. } => *poly,
            Self::Proportional { poly, .. } => *poly,
        }
    }

    /// Set the error polynomial coefficients
    ///
    /// # Arguments
    ///
    /// * `poly` - New error polynomial coefficients (c0, c1, c2, c3)
    ///
    /// # Returns
    ///
    /// The updated error model with the new polynomial coefficients
    pub fn set_polynomial(&mut self, poly: (f64, f64, f64, f64)) {
        match self {
            Self::Additive { poly: p, .. } => *p = poly,
            Self::Proportional { poly: p, .. } => *p = poly,
        }
    }

    /// Get the scaling parameter
    pub fn scalar(&self) -> f64 {
        match self {
            Self::Additive { lambda, .. } => *lambda,
            Self::Proportional { gamma, .. } => *gamma,
        }
    }

    /// Set the scaling parameter
    pub fn set_scalar(&mut self, scalar: f64) {
        match self {
            Self::Additive { lambda, .. } => *lambda = scalar,
            Self::Proportional { gamma, .. } => *gamma = scalar,
        }
    }

    /// Estimate the standard deviation for a prediction
    ///
    /// Calculates the standard deviation based on the error model type,
    /// using either observation-specific error polynomial coefficients or
    /// the model's default coefficients.
    ///
    /// # Arguments
    ///
    /// * `prediction` - The prediction for which to estimate the standard deviation
    ///
    /// # Returns
    ///
    /// The estimated standard deviation of the prediction
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        // Get appropriate polynomial coefficients from prediction or default
        let (c0, c1, c2, c3) = match prediction.errorpoly() {
            Some(poly) => poly,
            None => self.polynomial(),
        };

        // Calculate alpha term
        let alpha = c0
            + c1 * prediction.observation()
            + c2 * prediction.observation().powi(2)
            + c3 * prediction.observation().powi(3);

        // Calculate standard deviation based on error model type
        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma * alpha,
        };

        if sigma < 0.0 {
            Err(ErrorModelError::NegativeSigma)
        } else if !sigma.is_finite() {
            Err(ErrorModelError::NonFiniteSigma)
        } else {
            Ok(sigma)
        }
    }

    /// Estimate the variance of the observation
    ///
    /// This is a conveniecen function which calls [ErrorModel::sigma], and squares the result.
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        let sigma = self.sigma(prediction)?;
        Ok(sigma.powi(2))
    }

    /// Estimate the standard deviation for a raw observation value
    ///
    /// Calculates the standard deviation based on the error model type,
    /// using the model's default coefficients and a provided observation value.
    ///
    /// # Arguments
    ///
    /// * `value` - The observation value for which to estimate the standard deviation
    ///
    /// # Returns
    ///
    /// The estimated standard deviation for the given value
    pub fn sigma_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        // Get polynomial coefficients from the model
        let (c0, c1, c2, c3) = self.polynomial();

        // Calculate alpha term
        let alpha = c0 + c1 * value + c2 * value.powi(2) + c3 * value.powi(3);

        // Calculate standard deviation based on error model type
        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma * alpha,
        };

        if sigma < 0.0 {
            Err(ErrorModelError::NegativeSigma)
        } else if !sigma.is_finite() {
            Err(ErrorModelError::NonFiniteSigma)
        } else {
            Ok(sigma)
        }
    }

    /// Estimate the variance for a raw observation value
    ///
    /// This is a conveniecen function which calls [ErrorModel::sigma_from_value], and squares the result.
    pub fn variance_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        let sigma = self.sigma_from_value(value)?;
        Ok(sigma.powi(2))
    }
}

#[derive(Error, Debug, Clone)]
pub enum ErrorModelError {
    #[error("The computed standard deviation is negative.")]
    NegativeSigma,
    #[error("The computed standard deviation is non-finite")]
    NonFiniteSigma,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Observation;

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::additive((1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::proportional((1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma(&prediction).unwrap(), 2.0);
    }

    #[test]
    fn test_polynomial() {
        let model = ErrorModel::additive((1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.polynomial(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_set_polynomial() {
        let mut model = ErrorModel::additive((1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.polynomial(), (1.0, 2.0, 3.0, 4.0));
        model.set_polynomial((5.0, 6.0, 7.0, 8.0));
        assert_eq!(model.polynomial(), (5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn test_set_scalar() {
        let mut model = ErrorModel::additive((1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.scalar(), 5.0);
        model.set_scalar(10.0);
        assert_eq!(model.scalar(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = ErrorModel::additive((1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = ErrorModel::proportional((1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
    }
}
