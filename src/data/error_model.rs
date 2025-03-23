use crate::simulator::likelihood::Prediction;

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [ErrorModel] defines how the standard deviation of observations is calculated,
/// using error polynomial coefficients and a gamma parameter.
#[derive(Debug, Clone)]
pub struct ErrorModel<'a> {
    /// Error polynomial coefficients (c0, c1, c2, c3)
    c: (f64, f64, f64, f64),
    /// Gamma parameter for scaling errors
    gl: f64,
    /// Error type (additive or proportional)
    e_type: &'a ErrorType,
}

impl<'a> ErrorModel<'a> {
    /// Get the gamma parameter
    ///
    /// # Returns
    ///
    /// The gamma parameter value
    pub fn gl(&self) -> f64 {
        self.gl
    }

    /// Create a new error model
    ///
    /// # Arguments
    ///
    /// * `c` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gl` - Gamma parameter for scaling errors
    /// * `e_type` - Error type (additive or proportional)
    pub fn new(c: (f64, f64, f64, f64), gl: f64, e_type: &'a ErrorType) -> Self {
        Self { c, gl, e_type }
    }
}

/// Types of error models for pharmacometric observations
///
/// Different error types define how observation variability scales with concentration.
#[derive(Debug, Clone)]
pub enum ErrorType {
    /// Additive error model, where error is independent of concentration
    Add,
    /// Proportional error model, where error scales with concentration
    Prop,
}
#[allow(clippy::extra_unused_lifetimes)]
impl<'a> ErrorModel<'_> {
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
            None => (self.c.0, self.c.1, self.c.2, self.c.3),
        };
        let alpha = c0
            + c1 * prediction.observation()
            + c2 * prediction.observation().powi(2)
            + c3 * prediction.observation().powi(3);

        let res = match self.e_type {
            ErrorType::Add => (alpha.powi(2) + self.gl.powi(2)).sqrt(),
            ErrorType::Prop => self.gl * alpha,
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
