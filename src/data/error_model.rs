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

impl From<Vec<ErrorModel>> for ErrorModels {
    fn from(models: Vec<ErrorModel>) -> Self {
        Self { models }
    }
}

/// Collection of error models for all possible outputs in the model/dataset
/// This struct holds a vector of error models, each corresponding to a specific output
/// in the pharmacometric analysis.
///
/// This is a wrapper around a vector of [ErrorModel]s, its size is determined by the number of outputs in the model/dataset.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct ErrorModels {
    models: Vec<ErrorModel>,
}

impl Default for ErrorModels {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorModels {
    /// Create a new instance of ErrorModels with an empty vector
    ///
    /// # Returns
    /// A new instance of ErrorModels with an empty vector of models.
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    /// Get the error model for a specific output equation
    ///
    /// # Arguments
    /// * `outeq` - The index of the output equation for which to retrieve the error model.
    /// # Returns
    /// A reference to the [ErrorModel] for the specified output equation.
    /// # Errors
    /// If the output equation index is invalid, an [ErrorModelError::InvalidOutputEquation] is returned.
    pub fn get_error_model(&self, outeq: usize) -> Result<&ErrorModel, ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        Ok(&self.models[outeq])
    }

    /// Add a new error model for a specific output equation
    /// # Arguments
    /// * `outeq` - The index of the output equation for which to add the error model.
    /// * `model` - The [ErrorModel] to add for the specified output equation.
    /// # Returns
    /// A new instance of ErrorModels with the added model.
    /// # Errors
    /// If the output equation index is invalid or if a model already exists for that output equation, an [ErrorModelError::ExistingOutputEquation] is returned.
    pub fn add(mut self, outeq: usize, model: ErrorModel) -> Result<Self, ErrorModelError> {
        if outeq >= self.models.len() {
            self.models.resize(outeq + 1, ErrorModel::None);
        }
        if self.models[outeq] != ErrorModel::None {
            return Err(ErrorModelError::ExistingOutputEquation(outeq));
        }
        self.models[outeq] = model;
        Ok(self)
    }
    /// Returns an iterator over the error models in the collection.
    ///
    /// # Returns
    /// An iterator that yields tuples containing the index and a reference to each [ErrorModel].
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ErrorModel)> {
        self.models.iter().enumerate()
    }

    /// Returns an iterator that yields mutable references to the error models in the collection.
    /// # Returns
    /// An iterator that yields tuples containing the index and a mutable reference to each [ErrorModel].
    pub fn into_iter(self) -> impl Iterator<Item = (usize, ErrorModel)> {
        self.models.into_iter().enumerate()
    }

    /// Returns a mutable iterator that yields mutable references to the error models in the collection.
    /// # Returns
    /// An iterator that yields tuples containing the index and a mutable reference to each [ErrorModel].
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut ErrorModel)> {
        self.models.iter_mut().enumerate()
    }

    /// Computes a hash for the error models collection.
    /// This hash is based on the output equations and their associated error models.
    /// # Returns
    /// A `u64` hash value representing the error models collection.
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        for outeq in 0..self.models.len() {
            // Find the model with the matching outeq ID

            let model = &self.models[outeq];
            outeq.hash(&mut hasher);

            match model {
                ErrorModel::Additive {
                    lambda,
                    poly: _,
                    lloq: _,
                } => {
                    0u8.hash(&mut hasher); // Use 0 for additive model
                    lambda.to_bits().hash(&mut hasher);
                }
                ErrorModel::Proportional {
                    gamma,
                    poly: _,
                    lloq: _,
                } => {
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
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
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
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        Ok(self.models[outeq].scalar()?)
    }

    /// Sets the error polynomial for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `poly` - The new [`ErrorPoly`] to set.
    pub fn set_errorpoly(&mut self, outeq: usize, poly: ErrorPoly) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].set_errorpoly(poly);
        Ok(())
    }

    /// Sets the scalar value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `scalar` - The new scalar value to set.
    pub fn set_scalar(&mut self, outeq: usize, scalar: f64) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].set_scalar(scalar);
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
        let outeq = prediction.outeq;
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
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
        let outeq = prediction.outeq;
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
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
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
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
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
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
        /// Optional: lloq (Lower Limit of Quantification) of the analytical method
        lloq: Option<f64>,
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
        /// Optional: lloq (Lower Limit of Quantification) of the analytical method
        lloq: Option<f64>,
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
    pub fn additive(poly: ErrorPoly, lambda: f64, lloq: Option<f64>) -> Self {
        Self::Additive { lambda, poly, lloq }
    }

    /// Get the lloq (Lower Limit of Quantification) value, if available.
    ///
    /// # Returns
    ///
    /// An `Option<f64>` containing the lloq value if it exists, otherwise `None`.
    pub fn lloq(&self) -> Option<f64> {
        match self {
            Self::Additive { lloq, .. } => *lloq,
            Self::Proportional { lloq, .. } => *lloq,
            Self::None => None,
        }
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
    pub fn proportional(poly: ErrorPoly, gamma: f64, lloq: Option<f64>) -> Self {
        Self::Proportional { gamma, poly, lloq }
    }

    /// Get the error polynomial coefficients
    ///
    /// # Returns
    ///
    /// The error polynomial coefficients (c0, c1, c2, c3)
    pub fn errorpoly(&self) -> Result<ErrorPoly, ErrorModelError> {
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
    pub fn set_errorpoly(&mut self, poly: ErrorPoly) {
        match self {
            Self::Additive { poly: p, .. } => *p = poly,
            Self::Proportional { poly: p, .. } => *p = poly,
            Self::None => {}
        }
    }

    /// Get the scaling parameter
    pub fn scalar(&self) -> Result<f64, ErrorModelError> {
        match self {
            Self::Additive { lambda, .. } => Ok(*lambda),
            Self::Proportional { gamma, .. } => Ok(*gamma),
            Self::None => Err(ErrorModelError::MissingErrorModel),
        }
    }

    /// Set the scaling parameter
    pub fn set_scalar(&mut self, scalar: f64) {
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
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
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
    pub fn variance_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
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
    #[error("The output equation index {0} is of type ErrorModel::None")]
    NoneErrorModel(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Observation;

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0, None);
        assert_eq!(model.sigma(&prediction).unwrap(), 2.0);
    }

    #[test]
    fn test_polynomial() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0, None);
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_set_errorpoly() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0, None);
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
        model.set_errorpoly(ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (5.0, 6.0, 7.0, 8.0)
        );
    }

    #[test]
    fn test_set_scalar() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0, None);
        assert_eq!(model.scalar().unwrap(), 5.0);
        model.set_scalar(10.0);
        assert_eq!(model.scalar().unwrap(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0, None);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
    }

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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_error_models_add_multiple() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0, None);

        let models = ErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(1, model2)
            .unwrap();

        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_error_models_add_duplicate_outeq_fails() {
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0, None);

        let result = ErrorModels::new().add(0, model1).unwrap().add(0, model2); // Same outeq should fail

        assert!(result.is_err());
        match result {
            Err(ErrorModelError::ExistingOutputEquation(outeq)) => assert_eq!(outeq, 0),
            _ => panic!("Expected ExistingOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_scalar() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.scalar(0).unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_scalar_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.scalar(0).unwrap(), 5.0);
        models.set_scalar(0, 10.0).unwrap();
        assert_eq!(models.scalar(0).unwrap(), 10.0);
    }

    #[test]
    fn test_error_models_set_scalar_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(poly, 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        let retrieved_poly = models.errorpoly(0).unwrap();
        assert_eq!(retrieved_poly.coefficients(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_error_models_errorpoly_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(poly1, 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);

        let sigma = models.sigma(&prediction).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, 20.0, 0, None, false);
        let prediction = observation.to_prediction(10.0, vec![]);

        let variance = models.variance(&prediction).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        let sigma = models.sigma_from_value(0, 20.0).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_from_value_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let models = ErrorModels::new().add(0, model).unwrap();

        let variance = models.variance_from_value(0, 20.0).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_from_value_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
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
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0, None);

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
        let model1 = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let model2 = ErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0, None);

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
    fn test_error_models_multiple_outeqs() {
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5, None);
        let proportional_model =
            ErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.1, None);

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
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0, None);
        let proportional_model =
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0, None);

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
