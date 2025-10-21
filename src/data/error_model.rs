use std::hash::{DefaultHasher, Hash, Hasher};

use crate::simulator::likelihood::Prediction;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Parameter that can be either fixed or variable for estimation
///
/// This enum allows specifying whether a factor parameter (like lambda or gamma)
/// should be fixed at a specific value or allowed to vary during estimation.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub enum Factor {
    /// Parameter can be estimated/varied during optimization
    Variable(f64),
    /// Parameter is fixed at this value and won't be estimated
    Fixed(f64),
}

impl Factor {
    /// Get the current value of the parameter
    pub fn value(&self) -> f64 {
        match self {
            Self::Variable(val) | Self::Fixed(val) => *val,
        }
    }

    /// Check if the parameter is fixed
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Check if the parameter is variable (can be estimated)
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Set the value while preserving the fixed/variable state
    pub fn set_value(&mut self, new_value: f64) {
        match self {
            Self::Variable(val) => *val = new_value,
            Self::Fixed(val) => *val = new_value,
        }
    }

    /// Convert the parameter to fixed at its current value
    pub fn make_fixed(&mut self) {
        if let Self::Variable(val) = self {
            *self = Self::Fixed(*val);
        }
    }

    /// Convert the parameter to variable at its current value
    pub fn make_variable(&mut self) {
        if let Self::Fixed(val) = self {
            *self = Self::Variable(*val);
        }
    }

    /// Replace the current factor with a new factor value
    pub fn set_factor(&mut self, factor: &Factor) {
        match factor {
            Factor::Variable(val) => *self = Self::Variable(*val),
            Factor::Fixed(val) => *self = Self::Fixed(*val),
        }
    }
}

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
    /// Create a new instance of [ErrorModels]
    ///
    /// # Returns
    /// A new instance of [ErrorModels].
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
    pub fn error_model(&self, outeq: usize) -> Result<&ErrorModel, ErrorModelError> {
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
                ErrorModel::Additive { lambda, poly: _ } => {
                    0u8.hash(&mut hasher); // Use 0 for additive model
                    lambda.value().to_bits().hash(&mut hasher);
                    lambda.is_fixed().hash(&mut hasher); // Include fixed/variable state in hash
                }
                ErrorModel::Proportional { gamma, poly: _ } => {
                    1u8.hash(&mut hasher); // Use 1 for proportional model
                    gamma.value().to_bits().hash(&mut hasher);
                    gamma.is_fixed().hash(&mut hasher); // Include fixed/variable state in hash
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

    /// Returns the factor value associated with the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// The factor value for the given output equation.
    pub fn factor(&self, outeq: usize) -> Result<f64, ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        Ok(self.models[outeq].factor()?)
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

    /// Sets the factor value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `factor` - The new factor value to set.
    pub fn set_factor(&mut self, outeq: usize, factor: f64) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].set_factor(factor);
        Ok(())
    }

    /// Gets the factor parameter (including fixed/variable state) for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// The [`Factor`] for the given output equation.
    pub fn factor_param(&self, outeq: usize) -> Result<Factor, ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].factor_param()
    }

    /// Sets the factor parameter (including fixed/variable state) for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    /// * `param` - The new [`Factor`] to set.
    pub fn set_factor_param(&mut self, outeq: usize, param: Factor) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].set_factor_param(param);
        Ok(())
    }

    /// Checks if the factor parameter is fixed for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    ///
    /// # Returns
    ///
    /// `true` if the factor parameter is fixed, `false` if it's variable.
    pub fn is_factor_fixed(&self, outeq: usize) -> Result<bool, ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].is_factor_fixed()
    }

    /// Makes the factor parameter fixed at its current value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    pub fn fix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].fix_factor();
        Ok(())
    }

    /// Makes the factor parameter variable at its current value for the specified output equation.
    ///
    /// # Arguments
    ///
    /// * `outeq` - The index of the output equation.
    pub fn unfix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError> {
        if outeq >= self.models.len() {
            return Err(ErrorModelError::InvalidOutputEquation(outeq));
        }
        if self.models[outeq] == ErrorModel::None {
            return Err(ErrorModelError::NoneErrorModel(outeq));
        }
        self.models[outeq].unfix_factor();
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

impl IntoIterator for ErrorModels {
    type Item = (usize, ErrorModel);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.models
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a> IntoIterator for &'a ErrorModels {
    type Item = (usize, &'a ErrorModel);
    type IntoIter = std::iter::Enumerate<std::slice::Iter<'a, ErrorModel>>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter().enumerate()
    }
}

impl<'a> IntoIterator for &'a mut ErrorModels {
    type Item = (usize, &'a mut ErrorModel);
    type IntoIter = std::iter::Enumerate<std::slice::IterMut<'a, ErrorModel>>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter_mut().enumerate()
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
    /// * `lambda` - Lambda parameter for scaling errors (can be fixed or variable)
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    Additive {
        /// Lambda parameter for scaling errors (can be fixed or variable)
        lambda: Factor,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: ErrorPoly,
    },

    /// Proportional error model, where error scales with concentration
    ///
    /// Contains:
    /// * `gamma` - Gamma parameter for scaling errors (can be fixed or variable)
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    Proportional {
        /// Gamma parameter for scaling errors (can be fixed or variable)
        gamma: Factor,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: ErrorPoly,
    },
    #[default]
    None,
}

impl ErrorModel {
    /// Create a new additive error model with a variable lambda parameter
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `lambda` - Lambda parameter for scaling errors (will be variable)
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new additive error model
    pub fn additive(poly: ErrorPoly, lambda: f64) -> Self {
        Self::Additive {
            lambda: Factor::Variable(lambda),
            poly,
        }
    }

    /// Create a new additive error model with a fixed lambda parameter
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `lambda` - Lambda parameter for scaling errors (will be fixed)
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new additive error model with fixed lambda
    pub fn additive_fixed(poly: ErrorPoly, lambda: f64) -> Self {
        Self::Additive {
            lambda: Factor::Fixed(lambda),
            poly,
        }
    }

    /// Create a new additive error model with a specified Factor for lambda
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `lambda` - Lambda parameter (can be Variable or Fixed) using [Factor]
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new additive error model
    pub fn additive_with_param(poly: ErrorPoly, lambda: Factor) -> Self {
        Self::Additive { lambda, poly }
    }

    /// Create a new proportional error model with a variable gamma parameter
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gamma` - Gamma parameter for scaling errors (will be variable)
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new proportional error model
    pub fn proportional(poly: ErrorPoly, gamma: f64) -> Self {
        Self::Proportional {
            gamma: Factor::Variable(gamma),
            poly,
        }
    }

    /// Create a new proportional error model with a fixed gamma parameter
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gamma` - Gamma parameter for scaling errors (will be fixed)
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new proportional error model with fixed gamma
    pub fn proportional_fixed(poly: ErrorPoly, gamma: f64) -> Self {
        Self::Proportional {
            gamma: Factor::Fixed(gamma),
            poly,
        }
    }

    /// Create a new proportional error model with a specified Factor for gamma
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `gamma` - Gamma parameter (can be Variable or Fixed) using [Factor]
    /// * `lloq` - Optional lower limit of quantification
    ///
    /// # Returns
    ///
    /// A new proportional error model
    pub fn proportional_with_param(poly: ErrorPoly, gamma: Factor) -> Self {
        Self::Proportional { gamma, poly }
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

    /// Get the scaling parameter value
    pub fn factor(&self) -> Result<f64, ErrorModelError> {
        match self {
            Self::Additive { lambda, .. } => Ok(lambda.value()),
            Self::Proportional { gamma, .. } => Ok(gamma.value()),
            Self::None => Err(ErrorModelError::MissingErrorModel),
        }
    }

    /// Set the scaling parameter value (preserves fixed/variable state)
    pub fn set_factor(&mut self, factor: f64) {
        match self {
            Self::Additive { lambda, .. } => lambda.set_value(factor),
            Self::Proportional { gamma, .. } => gamma.set_value(factor),
            Self::None => {}
        }
    }

    /// Get the scaling parameter (including its fixed/variable state)
    pub fn factor_param(&self) -> Result<Factor, ErrorModelError> {
        match self {
            Self::Additive { lambda, .. } => Ok(*lambda),
            Self::Proportional { gamma, .. } => Ok(*gamma),
            Self::None => Err(ErrorModelError::MissingErrorModel),
        }
    }

    /// Set the scaling parameter (including its fixed/variable state)
    pub fn set_factor_param(&mut self, param: Factor) {
        match self {
            Self::Additive { lambda, .. } => *lambda = param,
            Self::Proportional { gamma, .. } => *gamma = param,
            Self::None => {}
        }
    }

    /// Check if the scaling parameter is fixed
    pub fn is_factor_fixed(&self) -> Result<bool, ErrorModelError> {
        match self {
            Self::Additive { lambda, .. } => Ok(lambda.is_fixed()),
            Self::Proportional { gamma, .. } => Ok(gamma.is_fixed()),
            Self::None => Err(ErrorModelError::MissingErrorModel),
        }
    }

    /// Make the scaling parameter fixed at its current value
    pub fn fix_factor(&mut self) {
        match self {
            Self::Additive { lambda, .. } => lambda.make_fixed(),
            Self::Proportional { gamma, .. } => gamma.make_fixed(),
            Self::None => {}
        }
    }

    /// Make the scaling parameter variable at its current value
    pub fn unfix_factor(&mut self) {
        match self {
            Self::Additive { lambda, .. } => lambda.make_variable(),
            Self::Proportional { gamma, .. } => gamma.make_variable(),
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
        if prediction.observation.is_none() {
            return Err(ErrorModelError::MissingObservation);
        }

        // Get appropriate polynomial coefficients from prediction or default
        let errorpoly = match prediction.errorpoly() {
            Some(poly) => poly,
            None => self.errorpoly()?,
        };

        let (c0, c1, c2, c3) = (errorpoly.c0, errorpoly.c1, errorpoly.c2, errorpoly.c3);

        // Calculate alpha term
        let alpha = c0
            + c1 * prediction.observation().unwrap()
            + c2 * prediction.observation().unwrap().powi(2)
            + c3 * prediction.observation().unwrap().powi(3);

        // Calculate standard deviation based on error model type
        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.value().powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma.value() * alpha,
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
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.value().powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma.value() * alpha,
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

    /// Get a boolean indicating if the error model should be optimized
    ///
    /// In other words, if the error model is not None, and the [Factor] is variable, it should be optimized.
    pub fn optimize(&self) -> bool {
        match self {
            Self::Additive { lambda, .. } => lambda.is_variable(),
            Self::Proportional { gamma, .. } => gamma.is_variable(),
            Self::None => false,
        }
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
    #[error("The prediction does not have an observation associated with it")]
    MissingObservation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Censor, Observation};

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
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
    fn test_set_errorpoly() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
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
    fn test_set_factor() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.factor().unwrap(), 5.0);
        model.set_factor(10.0);
        assert_eq!(model.factor().unwrap(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
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
    fn test_error_models_factor() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.factor(0).unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_factor_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let result = models.factor(1);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            _ => panic!("Expected InvalidOutputEquation error"),
        }
    }

    #[test]
    fn test_error_models_set_factor() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.factor(0).unwrap(), 5.0);
        models.set_factor(0, 10.0).unwrap();
        assert_eq!(models.factor(0).unwrap(), 10.0);
    }

    #[test]
    fn test_error_models_set_factor_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = ErrorModels::new().add(0, model).unwrap();

        let result = models.set_factor(1, 10.0);
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

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let sigma = models.sigma(&prediction).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None); // outeq=1 not in models
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

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let variance = models.variance(&prediction).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_invalid_outeq() {
        let model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = ErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None); // outeq=1 not in models
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
    fn test_error_models_multiple_outeqs() {
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5);
        let proportional_model = ErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.1);

        let models = ErrorModels::new()
            .add(0, additive_model)
            .unwrap()
            .add(1, proportional_model)
            .unwrap();

        assert_eq!(models.len(), 2);

        // Test factor retrieval for different outeqs
        assert_eq!(models.factor(0).unwrap(), 0.5);
        assert_eq!(models.factor(1).unwrap(), 0.1);

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
        let obs1 = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let pred1 = obs1.to_prediction(10.0, vec![]);
        let sigma1 = models.sigma(&pred1).unwrap();
        assert_eq!(sigma1, (26.0_f64).sqrt()); // additive: sqrt(alpha^2 + lambda^2) = sqrt(1^2 + 5^2) = sqrt(26)

        // Test with outeq=1 (proportional model)
        let obs2 = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None);
        let pred2 = obs2.to_prediction(10.0, vec![]);
        let sigma2 = models.sigma(&pred2).unwrap();
        assert_eq!(sigma2, 2.0); // proportional: gamma * alpha = 2 * 1 = 2
    }

    #[test]
    fn test_factor_param_new_constructors() {
        // Test variable constructors (default behavior)
        let additive = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(additive.factor().unwrap(), 5.0);
        assert!(!additive.is_factor_fixed().unwrap());

        let proportional = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(proportional.factor().unwrap(), 2.0);
        assert!(!proportional.is_factor_fixed().unwrap());

        // Test fixed constructors
        let additive_fixed = ErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(additive_fixed.factor().unwrap(), 5.0);
        assert!(additive_fixed.is_factor_fixed().unwrap());

        let proportional_fixed =
            ErrorModel::proportional_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(proportional_fixed.factor().unwrap(), 2.0);
        assert!(proportional_fixed.is_factor_fixed().unwrap());

        // Test Factor constructors
        let additive_with_param =
            ErrorModel::additive_with_param(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), Factor::Fixed(5.0));
        assert_eq!(additive_with_param.factor().unwrap(), 5.0);
        assert!(additive_with_param.is_factor_fixed().unwrap());

        let proportional_with_param = ErrorModel::proportional_with_param(
            ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
            Factor::Variable(2.0),
        );
        assert_eq!(proportional_with_param.factor().unwrap(), 2.0);
        assert!(!proportional_with_param.is_factor_fixed().unwrap());
    }

    #[test]
    fn test_factor_param_methods() {
        let mut model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

        // Test initial state
        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(!model.is_factor_fixed().unwrap());

        // Test fixing parameter
        model.fix_factor();
        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(model.is_factor_fixed().unwrap());

        // Test unfixing parameter
        model.unfix_factor();
        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(!model.is_factor_fixed().unwrap());

        // Test setting factor param directly
        model.set_factor_param(Factor::Fixed(10.0));
        assert_eq!(model.factor().unwrap(), 10.0);
        assert!(model.is_factor_fixed().unwrap());

        // Test getting factor param
        let param = model.factor_param().unwrap();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_fixed());
    }

    #[test]
    fn test_factor_param_functionality() {
        let mut param = Factor::Variable(5.0);

        // Test basic functionality
        assert_eq!(param.value(), 5.0);
        assert!(param.is_variable());
        assert!(!param.is_fixed());

        // Test setting value
        param.set_value(10.0);
        assert_eq!(param.value(), 10.0);
        assert!(param.is_variable());

        // Test making fixed
        param.make_fixed();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_fixed());
        assert!(!param.is_variable());

        // Test making variable again
        param.make_variable();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_variable());
        assert!(!param.is_fixed());
    }

    #[test]
    fn test_error_models_factor_param_methods() {
        let additive_model = ErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let mut models = ErrorModels::new()
            .add(0, additive_model)
            .unwrap()
            .add(1, proportional_model)
            .unwrap();

        // Test factor param retrieval
        let param0 = models.factor_param(0).unwrap();
        assert_eq!(param0.value(), 5.0);
        assert!(param0.is_fixed());

        let param1 = models.factor_param(1).unwrap();
        assert_eq!(param1.value(), 2.0);
        assert!(param1.is_variable());

        // Test is_factor_fixed
        assert!(models.is_factor_fixed(0).unwrap());
        assert!(!models.is_factor_fixed(1).unwrap());

        // Test fixing/unfixing
        models.fix_factor(1).unwrap();
        assert!(models.is_factor_fixed(1).unwrap());

        models.unfix_factor(0).unwrap();
        assert!(!models.is_factor_fixed(0).unwrap());

        // Test setting factor param
        models.set_factor_param(0, Factor::Fixed(10.0)).unwrap();
        assert_eq!(models.factor(0).unwrap(), 10.0);
        assert!(models.is_factor_fixed(0).unwrap());
    }

    #[test]
    fn test_fixed_parameters_in_calculations() {
        // Test that fixed and variable parameters produce the same calculation results
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let model_variable = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model_fixed = ErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

        let sigma_variable = model_variable.sigma(&prediction).unwrap();
        let sigma_fixed = model_fixed.sigma(&prediction).unwrap();

        assert_eq!(sigma_variable, sigma_fixed);
        assert_eq!(sigma_variable, (26.0_f64).sqrt());

        // Test with sigma_from_value
        let sigma_variable_val = model_variable.sigma_from_value(20.0).unwrap();
        let sigma_fixed_val = model_fixed.sigma_from_value(20.0).unwrap();

        assert_eq!(sigma_variable_val, sigma_fixed_val);
        assert_eq!(sigma_variable_val, (26.0_f64).sqrt());
    }

    #[test]
    fn test_hash_includes_fixed_state() {
        let model1_variable = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model1_fixed = ErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

        let models1 = ErrorModels::new().add(0, model1_variable).unwrap();
        let models2 = ErrorModels::new().add(0, model1_fixed).unwrap();

        // Different fixed/variable states should produce different hashes
        assert_ne!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_into_iter_functionality() {
        let additive_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model = ErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let mut models = ErrorModels::new()
            .add(0, additive_model)
            .unwrap()
            .add(1, proportional_model)
            .unwrap();

        // Verify initial state - both should be variable
        assert!(!models.is_factor_fixed(0).unwrap());
        assert!(!models.is_factor_fixed(1).unwrap());
        assert_eq!(models.factor(0).unwrap(), 5.0);
        assert_eq!(models.factor(1).unwrap(), 2.0);

        // First iteration: update values using iter_mut
        for (outeq, model) in models.iter_mut() {
            match outeq {
                0 => model.set_factor(10.0), // Update additive lambda from 5.0 to 10.0
                1 => model.set_factor(4.0),  // Update proportional gamma from 2.0 to 4.0
                _ => {}
            }
        }

        // Verify values were updated
        assert_eq!(models.factor(0).unwrap(), 10.0);
        assert_eq!(models.factor(1).unwrap(), 4.0);
        assert!(!models.is_factor_fixed(0).unwrap()); // Still variable
        assert!(!models.is_factor_fixed(1).unwrap()); // Still variable

        // Second iteration: fix all parameters using iter_mut
        for (_outeq, model) in models.iter_mut() {
            model.fix_factor();
        }

        // Verify all parameters are now fixed
        assert!(models.is_factor_fixed(0).unwrap());
        assert!(models.is_factor_fixed(1).unwrap());
        assert_eq!(models.factor(0).unwrap(), 10.0); // Values should remain the same
        assert_eq!(models.factor(1).unwrap(), 4.0);

        // Test read-only iteration with iter()
        let mut count = 0;
        for (outeq, model) in models.iter() {
            count += 1;
            match outeq {
                0 => {
                    assert!(model.is_factor_fixed().unwrap());
                    assert_eq!(model.factor().unwrap(), 10.0);
                }
                1 => {
                    assert!(model.is_factor_fixed().unwrap());
                    assert_eq!(model.factor().unwrap(), 4.0);
                }
                _ => panic!("Unexpected outeq: {}", outeq),
            }
        }
        assert_eq!(count, 2);

        // Test consuming iteration with into_iter()
        let collected_models: Vec<(usize, ErrorModel)> = models.into_iter().collect();
        assert_eq!(collected_models.len(), 2);

        // Verify the collected models retain their state
        let (outeq0, model0) = &collected_models[0];
        let (outeq1, model1) = &collected_models[1];

        assert_eq!(*outeq0, 0);
        assert_eq!(*outeq1, 1);
        assert!(model0.is_factor_fixed().unwrap());
        assert!(model1.is_factor_fixed().unwrap());
        assert_eq!(model0.factor().unwrap(), 10.0);
        assert_eq!(model1.factor().unwrap(), 4.0);
    }
}
