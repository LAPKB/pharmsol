use std::hash::{DefaultHasher, Hash, Hasher};

use crate::simulator::likelihood::Prediction;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

/// Collection of error models for all possible outputs in the model/dataset
/// This struct holds a vector of error models, each corresponding to a specific output
/// in the pharmacometric analysis.
///
/// This is a wrapper around a vector of [ErrorModel]s, its size is determined by the number of outputs in the model/dataset.

pub struct ErrorModels {
    models: Vec<(usize, ErrorModel)>,
}

impl Default for ErrorModels {
    fn default() -> Self {
        Self { models: Vec::new() }
    }
}

impl ErrorModels {
    /// Create a new collection of [ErrorModels]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error model for a specific output equation.
    pub fn add(mut self, outeq: usize, model: ErrorModel) -> Result<Self, ErrorModelError> {
        if self.models.iter().any(|(id, _)| *id == outeq) {
            return Err(ErrorModelError::ExistingOutputEquation(outeq));
        }
        self.models.push((outeq, model));
        Ok(self)
    }

    /// Returns the number of error models in the collection.
    pub fn len(&self) -> usize {
        self.models.len()
    }
    /// Calculates a hash of the ErrorModels collection
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Sort by output equation index to ensure consistent hashing
        let mut sorted_models = self.models.clone();
        sorted_models.sort_by_key(|(outeq, _)| *outeq);

        for (outeq, model) in sorted_models {
            outeq.hash(&mut hasher);
            // Hash the discriminant and fields of the ErrorModel enum
            match model {
                ErrorModel::Additive { lambda, poly } => {
                    0u8.hash(&mut hasher); // discriminant for Additive
                    lambda.to_bits().hash(&mut hasher);
                    poly.coefficients().0.to_bits().hash(&mut hasher);
                    poly.coefficients().1.to_bits().hash(&mut hasher);
                    poly.coefficients().2.to_bits().hash(&mut hasher);
                    poly.coefficients().3.to_bits().hash(&mut hasher);
                }
                ErrorModel::Proportional { gamma, poly } => {
                    1u8.hash(&mut hasher); // discriminant for Proportional
                    gamma.to_bits().hash(&mut hasher);
                    poly.coefficients().0.to_bits().hash(&mut hasher);
                    poly.coefficients().1.to_bits().hash(&mut hasher);
                    poly.coefficients().2.to_bits().hash(&mut hasher);
                    poly.coefficients().3.to_bits().hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }
    /// Returns the scalar value associated with the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the scalar value for the given output equation or an [`ErrorModelError`] if the index is invalid.
    pub fn scalar(&self, outeq: usize) -> Result<f64, ErrorModelError> {
        self.models
            .iter()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))
            .map(|(_, model)| model.scalar())
    }

    /// Sets the scalar value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `scalar` - The new scalar value to set.
    ///
    /// # Returns
    ///
    /// A [`Result`] indicating success or an [`ErrorModelError`] if the index is invalid.
    pub fn set_scalar(&mut self, outeq: usize, scalar: f64) -> Result<(), ErrorModelError> {
        self.models
            .iter_mut()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))?
            .1
            .set_scalar(scalar);
        Ok(())
    }

    /// Returns the error polynomial for the specified output equation.
    ///
    ///  # Arguments
    ///
    ///  * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///  A [`Result`] containing the [`ErrorPoly`] for the given output equation or an [`ErrorModelError`] if the index is invalid.
    pub fn errorpoly(&self, outeq: usize) -> Result<ErrorPoly, ErrorModelError> {
        self.models
            .iter()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))
            .map(|(_, model)| model.errorpoly())
    }

    /// Sets the error polynomial for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `poly` - The new [`ErrorPoly`] to set.
    pub fn set_errorpoly(&mut self, outeq: usize, poly: ErrorPoly) -> Result<(), ErrorModelError> {
        self.models
            .iter_mut()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))?
            .1
            .set_errorpoly(poly);
        Ok(())
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
        self.models
            .iter()
            .find(|(id, _)| *id == prediction.outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(prediction.outeq))?
            .1
            .sigma(prediction)
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
        self.models
            .iter()
            .find(|(id, _)| *id == prediction.outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(prediction.outeq))?
            .1
            .variance(prediction)
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
        self.models
            .iter()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))?
            .1
            .sigma_from_value(value)
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
        self.models
            .iter()
            .find(|(id, _)| *id == outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))?
            .1
            .variance_from_value(value)
    }
}

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
    pub fn errorpoly(&self) -> ErrorPoly {
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
    pub fn set_errorpoly(&mut self, poly: ErrorPoly) {
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
    fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        // Get appropriate polynomial coefficients from prediction or default
        let errorpoly = match prediction.errorpoly() {
            Some(poly) => poly,
            None => self.errorpoly(),
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
        let (c0, c1, c2, c3) = self.errorpoly().coefficients();

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
        assert_eq!(model.errorpoly().coefficients(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_set_errorpoly() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.errorpoly().coefficients(), (1.0, 2.0, 3.0, 4.0));
        model.set_errorpoly(ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(model.errorpoly().coefficients(), (5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn test_set_scalar() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.scalar(), 5.0);
        model.set_scalar(10.0);
        assert_eq!(model.scalar(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
    }

    // Tests for ErrorModels collection
    #[test]
    fn test_error_models_new() {
        let models = ErrorModels::new();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_error_models_default() {
        let models = ErrorModels::default();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_error_models_add_single() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_error_models_add_multiple() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let models = ErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(1, model2)
            .unwrap();

        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_error_models_add_duplicate_outeq_fails() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let result = ErrorModels::new().add(0, model1).unwrap().add(0, model2); // Same outeq should fail

        assert!(result.is_err());
        match result {
            Err(ErrorModelError::ExistingOutputEquation(outeq)) => assert_eq!(outeq, 0),
            _ => panic!("Expected ExistingOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_scalar() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.scalar(0).unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_scalar_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let result = models.scalar(1);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_set_scalar() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.scalar(0).unwrap(), 5.0);
        models.set_scalar(0, 10.0).unwrap();
        assert_eq!(models.scalar(0).unwrap(), 10.0);
    }

    #[test]
    fn test_error_models_set_scalar_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        let result = models.set_scalar(1, 10.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_errorpoly() {
        let poly = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let model = ErrorModel::additive(poly, 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let retrieved_poly = models.errorpoly(0).unwrap();
        assert_eq!(retrieved_poly.coefficients(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_error_models_errorpoly_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let result = models.errorpoly(1);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_set_errorpoly() {
        let poly1 = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let poly2 = ErrorPoly::new(5.0, 6.0, 7.0, 8.0);
        let model = ErrorModel::additive(poly1, 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(
            models.errorpoly(0).unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
        models.set_errorpoly(0, poly2).unwrap();
        assert_eq!(
            models.errorpoly(0).unwrap().coefficients(),
            (5.0, 6.0, 7.0, 8.0)
        );
    }

    #[test]
    fn test_error_models_set_errorpoly_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        let result = models.set_errorpoly(1, ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_sigma() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);

        let sigma = models.sigma(&prediction).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 1, None, false); // outeq=1 not in models
        let prediction = observation.to_prediction(10.0, vec![]);

        let result = models.sigma(&prediction);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_variance() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);

        let variance = models.variance(&prediction).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 1, None, false); // outeq=1 not in models
        let prediction = observation.to_prediction(10.0, vec![]);

        let result = models.variance(&prediction);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_sigma_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let sigma = models.sigma_from_value(0, 20.0).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_from_value_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let result = models.sigma_from_value(1, 20.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_variance_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let variance = models.variance_from_value(0, 20.0).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_from_value_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let result = models.variance_from_value(1, 20.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_hash_consistency() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let models1 = ErrorModels::new()
            .add(0, model1.clone())
            .unwrap()
            .add(1, model2.clone())
            .unwrap();

        let models2 = ErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(1, model2)
            .unwrap();

        // Same models should produce same hash
        assert_eq!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_hash_order_independence() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        // Add in different orders
        let models1 = ErrorModels::new()
            .add(0, model1.clone())
            .unwrap()
            .add(1, model2.clone())
            .unwrap();

        let models2 = ErrorModels::new()
            .add(1, model2)
            .unwrap()
            .add(0, model1)
            .unwrap();

        // Hash should be the same regardless of insertion order
        assert_eq!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_hash_different_for_different_models() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = ErrorModel::additive(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 5.0); // Different poly

        let models1 = ErrorModels::new().add(0, model1).unwrap();
        let models2 = ErrorModels::new().add(0, model2).unwrap();

        // Different models should produce different hashes
        assert_ne!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_multiple_outeqs() {
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5);
        let proportional_model = ErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.1);

        let models = ErrorModels::new()
            .add(0, additive_model)
            .unwrap()
            .add(1, proportional_model)
            .unwrap();

        assert_eq!(models.len(), 2);

        // Test scalar retrieval for different outeqs
        assert_eq!(models.scalar(0).unwrap(), 0.5);
        assert_eq!(models.scalar(1).unwrap(), 0.1);

        // Test polynomial retrieval for different outeqs
        assert_eq!(
            models.errorpoly(0).unwrap().coefficients(),
            (1.0, 0.1, 0.0, 0.0)
        );
        assert_eq!(
            models.errorpoly(1).unwrap().coefficients(),
            (0.0, 0.05, 0.0, 0.0)
        );
    }

    #[test]
    fn test_error_models_with_predictions_different_outeqs() {
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let models = ErrorModels::new()
            .add(0, additive_model)
            .unwrap()
            .add(1, proportional_model)
            .unwrap();

        // Test with outeq=0 (additive model)
        let obs1 = Observation::new(0.0, 20.0, 0, None, false);
        let pred1 = obs1.to_prediction(10.0, vec![]);
        let sigma1 = models.sigma(&pred1).unwrap();
        assert_eq!(sigma1, (26.0_f64).sqrt()); // additive: sqrt(alpha^2 + lambda^2) = sqrt(1^2 + 5^2) = sqrt(26)

        // Test with outeq=1 (proportional model)
        let obs2 = Observation::new(0.0, 20.0, 1, None, false);
        let pred2 = obs2.to_prediction(10.0, vec![]);
        let sigma2 = models.sigma(&pred2).unwrap();
        assert_eq!(sigma2, 2.0); // proportional: gamma * alpha = 2 * 1 = 2
    }
}
