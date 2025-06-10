use std::hash::{DefaultHasher, Hash, Hasher};

use crate::simulator::likelihood::Prediction;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const NOUTEQS: usize = 10; // Maximum number of output equations

/// Error polynomial coefficients for the error model
///
/// This struct holds the coefficients for a polynomial used to model
/// the error in pharmacometric analyses. It represents the error associated with quantification
/// of e.g. the drug concentration in a biological sample, such as blood or plasma.
/// More simply, it is the error associated with the observed value.
/// The polynomial is defined as:
///
/// ```text
/// error = c0 + c1 * observation + c2 * observation^2 + c3 * observation^3
/// ```
///
/// where `c0`, `c1`, `c2`, and `c3` are the coefficients of the polynomial.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct ErrorPoly {
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl ErrorPoly {
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Get the coefficients of the error polynomial
    pub fn coefficients(&self) -> (f64, f64, f64, f64) {
        (self.c0, self.c1, self.c2, self.c3)
    }

    pub fn c0(&self) -> f64 {
        self.c0
    }
    pub fn c1(&self) -> f64 {
        self.c1
    }
    pub fn c2(&self) -> f64 {
        self.c2
    }
    pub fn c3(&self) -> f64 {
        self.c3
    }

    /// Set the coefficients of the error polynomial
    pub fn set_coefficients(&mut self, c0: f64, c1: f64, c2: f64, c3: f64) {
        self.c0 = c0;
        self.c1 = c1;
        self.c2 = c2;
        self.c3 = c3;
    }
}

impl From<[ErrorModel; NOUTEQS]> for ErrorModels {
    fn from(models: [ErrorModel; NOUTEQS]) -> Self {
        Self { models }
    }
}

/// Collection of error models for all possible outputs in the model/dataset
/// This struct holds a vector of error models, each corresponding to a specific output
/// in the pharmacometric analysis.
///
/// This is a wrapper around a vector of [ErrorModel]s, its size is determined by the number of outputs in the model/dataset.

pub struct ErrorModels {
    models: [ErrorModel; NOUTEQS],
}

impl ErrorModels {
    pub fn new() -> Self {
        Self {
            models: core::array::from_fn(|_| ErrorModel::default()),
        }
    }

    pub fn add(mut self, outeq: usize, model: ErrorModel) -> Result<Self, ErrorModelError> {
        if outeq >= NOUTEQS {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] != ErrorModel::None {
            return Err(ErrorModelError::ExistingOutputEquation(outeq));
        }
        self.models[outeq] = model;
        Ok(self)
    }
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        for outeq in 0..NOUTEQS {
            // Find the model with the matching outeq ID

            let model = &self.models[outeq];
            outeq.hash(&mut hasher);

            match model {
                ErrorModel::Additive { lambda, poly: _ } => {
                    0u8.hash(&mut hasher); // Use 0 for additive model
                    lambda.to_bits().hash(&mut hasher);
                }
                ErrorModel::Proportional { gamma, poly: _ } => {
                    1u8.hash(&mut hasher); // Use 1 for proportional model
                    gamma.to_bits().hash(&mut hasher);
                }
                ErrorModel::None => {
                    2u8.hash(&mut hasher); // Use 2 for no model
                }
            }
        }

        hasher.finish()
    }
    /// Returns the number of error models in the collection.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns the error polynomial associated with the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// The [`ErrorPoly`] for the given output equation.
    pub fn errorpoly(&self, outeq: usize) -> Result<ErrorPoly, ErrorModelError> {
        self.models[outeq].errorpoly()
    }

    /// Returns the scalar value associated with the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// The scalar value for the given output equation.
    pub fn scalar(&self, outeq: usize) -> Result<f64, ErrorModelError> {
        self.models[outeq].scalar()
    }

    /// Sets the error polynomial for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `poly` - The new [`ErrorPoly`] to set.
    pub fn set_polynomial(&mut self, outeq: usize, poly: ErrorPoly) {
        self.models[outeq].set_polynomial(poly);
    }

    /// Sets the scalar value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `scalar` - The new scalar value to set.
    pub fn set_scalar(&mut self, outeq: usize, scalar: f64) {
        self.models[outeq].set_scalar(scalar);
    }

    /// Computes the standard deviation (sigma) for the specified output equation and prediction.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `prediction` - The [`Prediction`] to use for the calculation.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the computed sigma value or an [`ErrorModelError`] if the calculation fails.
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.models[prediction.outeq].sigma(prediction)
    }

    /// Computes the variance for the specified output equation and prediction.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `prediction` - The [`Prediction`] to use for the calculation.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the computed variance or an [`ErrorModelError`] if the calculation fails.
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.models[prediction.outeq].variance(prediction)
    }

    /// Computes the standard deviation (sigma) for the specified output equation and value.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `value` - The value to use for the calculation.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the computed sigma value or an [`ErrorModelError`] if the calculation fails.
    pub fn sigma_from_value(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.models[outeq].sigma_from_value(value)
    }

    /// Computes the variance for the specified output equation and value.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `value` - The value to use for the calculation.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the computed variance or an [`ErrorModelError`] if the calculation fails.
    pub fn variance_from_value(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.models[outeq].variance_from_value(value)
    }
}

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [ErrorModel] defines how the standard deviation of observations is calculated
/// based on the type of error model used and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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
        poly: ErrorPoly,
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
        poly: ErrorPoly,
    },
    #[default]
    None,
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
    pub fn additive(poly: ErrorPoly, lambda: f64) -> Self {
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
    pub fn proportional(poly: ErrorPoly, gamma: f64) -> Self {
        Self::Proportional { gamma, poly }
    }

    /// Get the error polynomial coefficients
    ///
    /// # Returns
    ///
    /// The error polynomial coefficients (c0, c1, c2, c3)
    fn errorpoly(&self) -> Result<ErrorPoly, ErrorModelError> {
        match self {
            Self::Additive { poly, .. } => Ok(*poly),
            Self::Proportional { poly, .. } => Ok(*poly),
            Self::None => Err(ErrorModelError::MissingErrorModel),
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
    fn set_polynomial(&mut self, poly: ErrorPoly) {
        match self {
            Self::Additive { poly: p, .. } => *p = poly,
            Self::Proportional { poly: p, .. } => *p = poly,
            Self::None => {}
        }
    }

    /// Get the scaling parameter
    fn scalar(&self) -> Result<f64, ErrorModelError> {
        match self {
            Self::Additive { lambda, .. } => Ok(*lambda),
            Self::Proportional { gamma, .. } => Ok(*gamma),
            Self::None => Err(ErrorModelError::MissingErrorModel),
        }
    }

    /// Set the scaling parameter
    fn set_scalar(&mut self, scalar: f64) {
        match self {
            Self::Additive { lambda, .. } => *lambda = scalar,
            Self::Proportional { gamma, .. } => *gamma = scalar,
            Self::None => {}
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
    fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        // Get appropriate polynomial coefficients from prediction or default
        let errorpoly = match prediction.errorpoly() {
            Some(poly) => poly,
            None => self.errorpoly()?,
        };

        let (c0, c1, c2, c3) = (errorpoly.c0, errorpoly.c1, errorpoly.c2, errorpoly.c3);

        // Calculate alpha term
        let alpha = c0
            + c1 * prediction.observation()
            + c2 * prediction.observation().powi(2)
            + c3 * prediction.observation().powi(3);

        // Calculate standard deviation based on error model type
        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma * alpha,
            Self::None => {
                return Err(ErrorModelError::MissingErrorModel);
            }
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
    fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
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
    fn sigma_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        // Get polynomial coefficients from the model
        let (c0, c1, c2, c3) = self.errorpoly()?.coefficients();

        // Calculate alpha term
        let alpha = c0 + c1 * value + c2 * value.powi(2) + c3 * value.powi(3);

        // Calculate standard deviation based on error model type
        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma * alpha,
            Self::None => {
                return Err(ErrorModelError::MissingErrorModel);
            }
        };

        if sigma < 0.0 {
            Err(ErrorModelError::NegativeSigma)
        } else if !sigma.is_finite() {
            Err(ErrorModelError::NonFiniteSigma)
        } else if sigma == 0.0 {
            Err(ErrorModelError::ZeroSigma)
        } else {
            Ok(sigma)
        }
    }

    /// Estimate the variance for a raw observation value
    ///
    /// This is a conveniecen function which calls [ErrorModel::sigma_from_value], and squares the result.
    fn variance_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        let sigma = self.sigma_from_value(value)?;
        Ok(sigma.powi(2))
    }
}

#[derive(Error, Debug, Clone)]
pub enum ErrorModelError {
    #[error("The computed standard deviation is negative")]
    NegativeSigma,
    #[error("The computed standard deviation is zero")]
    ZeroSigma,
    #[error("The computed standard deviation is non-finite")]
    NonFiniteSigma,
    #[error("The output equation index {0} is invalid")]
    InvalidOutputEquation(usize),
    #[error("The output equation number {0} already exists")]
    ExistingOutputEquation(usize),
    #[error("An output equation does not have an error model defined")]
    MissingErrorModel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Observation;

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma(&prediction).unwrap(), 2.0);
    }

    #[test]
    fn test_polynomial() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_set_polynomial() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
        model.set_polynomial(ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (5.0, 6.0, 7.0, 8.0)
        );
    }

    #[test]
    fn test_set_scalar() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.scalar().unwrap(), 5.0);
        model.set_scalar(10.0);
        assert_eq!(model.scalar().unwrap(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
    }
}
