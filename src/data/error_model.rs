use serde::{Deserialize, Serialize};

use crate::simulator::likelihood::Prediction;

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [ErrorModel] defines how the standard deviation of observations is calculated,
/// using error polynomial coefficients and a gamma parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorModel {
    /// Error polynomial coefficients (c0, c1, c2, c3)
    poly: (f64, f64, f64, f64),
    /// Gamma parameter for scaling errors
    gamlam: f64,
    /// Error type (additive or proportional)
    error_type: ErrorType,
}

impl ErrorModel {
    /// Get the gamma parameter
    ///
    /// # Returns
    ///
    /// The gamma parameter value
    pub fn gl(&self) -> f64 {
        self.gamlam
    }

    /// Create a new error model
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gamlam` - Gamma parameter for scaling errors
    /// * `error_type` - Error type (additive or proportional)
    pub fn new(poly: (f64, f64, f64, f64), gamlam: f64, error_type: ErrorType) -> Self {
        Self {
            poly,
            gamlam,
            error_type,
        }
    }
}

/// Types of error models for pharmacometric observations
///
/// Different error types define how observation variability scales with concentration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    /// Additive error model, where error is independent of concentration
    Additive,
    /// Proportional error model, where error scales with concentration
    Proportional,
}
#[allow(clippy::extra_unused_lifetimes)]
impl ErrorModel {
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
    /// The estimated standard deviation
    ///
    /// # Panics
    ///
    /// Panics if the computed standard deviation is NaN or negative
    pub(crate) fn estimate_sigma(&self, prediction: &Prediction) -> f64 {
        let (c0, c1, c2, c3) = match prediction.errorpoly() {
            Some((c0, c1, c2, c3)) => (c0, c1, c2, c3),
            None => (self.poly.0, self.poly.1, self.poly.2, self.poly.3),
        };
        let alpha = c0
            + c1 * prediction.observation()
            + c2 * prediction.observation().powi(2)
            + c3 * prediction.observation().powi(3);

        let res = match self.error_type {
            ErrorType::Additive => (alpha.powi(2) + self.gamlam.powi(2)).sqrt(),
            ErrorType::Proportional => self.gamlam * alpha,
        };

        if res.is_nan() || res < 0.0 {
            panic!("The computed standard deviation is either NaN or negative (SD = {}), coercing to 0", res);
            // tracing::error!(
            //     "The computed standard deviation is either NaN or negative (SD = {}), coercing to 0",
            //     res
            // );
            // 0.0
        } else {
            res
        }
    }
}
