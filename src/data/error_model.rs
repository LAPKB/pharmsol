use crate::simulator::likelihood::Prediction;

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [ErrorModel] defines how the standard deviation of observations is calculated
/// based on the type of error model used and its parameters.
#[derive(Debug, Clone)]
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
    /// The estimated standard deviation
    ///
    /// # Panics
    ///
    /// Panics if the computed standard deviation is NaN or negative
    pub fn sigma(&self, prediction: &Prediction) -> f64 {
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
        let res = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma * alpha,
        };

        if !res.is_finite() || res < 0.0 {
            panic!(
                "The computed standard deviation is either non-finite or negative. The standard devation for the prediction {} is {}",
                prediction,
                res
            );
        } else {
            res
        }
    }
}
