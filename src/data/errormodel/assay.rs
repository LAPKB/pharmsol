use std::hash::{Hash, Hasher};

use crate::simulator::likelihood::Prediction;
use serde::{Deserialize, Serialize};

use super::{ErrorModel, ErrorModelError, ErrorModels};

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

/// Error polynomial coefficients for the assay error model
///
/// This struct holds the coefficients for a polynomial used to model the
/// error in observations. It represents the error associated with quantification
/// of e.g. the drug concentration in a biological sample, such as blood or plasma.
///
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

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [AssayErrorModel] defines how the standard deviation of observations is calculated
/// based on the type of error model used and its parameters.
///
/// Implements the [`ErrorModel`] trait, where [`ErrorModel::sigma`] computes sigma
/// from the **observation** value (appropriate for non-parametric algorithms).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssayErrorModel {
    /// Additive error model, where error is independent of concentration
    ///
    /// sigma = sqrt(alpha^2 + lambda^2)
    Additive {
        /// Lambda parameter for scaling errors (can be fixed or variable)
        lambda: Factor,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: ErrorPoly,
    },

    /// Proportional error model, where error scales with concentration
    ///
    /// sigma = gamma * alpha
    Proportional {
        /// Gamma parameter for scaling errors (can be fixed or variable)
        gamma: Factor,
        /// Error polynomial coefficients (c0, c1, c2, c3)
        poly: ErrorPoly,
    },
}

impl ErrorModel for AssayErrorModel {
    /// Compute sigma from a raw value (typically the observation).
    ///
    /// Uses the model's default polynomial coefficients to compute alpha, then
    /// applies the additive or proportional formula.
    fn sigma(&self, value: f64) -> Result<f64, ErrorModelError> {
        let (c0, c1, c2, c3) = self.errorpoly().coefficients();

        let alpha = c0 + c1 * value + c2 * value.powi(2) + c3 * value.powi(3);

        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.value().powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma.value() * alpha,
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
}

impl AssayErrorModel {
    /// Create a new additive error model with a variable lambda parameter
    pub fn additive(poly: ErrorPoly, lambda: f64) -> Self {
        Self::Additive {
            lambda: Factor::Variable(lambda),
            poly,
        }
    }

    /// Create a new additive error model with a fixed lambda parameter
    pub fn additive_fixed(poly: ErrorPoly, lambda: f64) -> Self {
        Self::Additive {
            lambda: Factor::Fixed(lambda),
            poly,
        }
    }

    /// Create a new additive error model with a specified Factor for lambda
    pub fn additive_with_param(poly: ErrorPoly, lambda: Factor) -> Self {
        Self::Additive { lambda, poly }
    }

    /// Create a new proportional error model with a variable gamma parameter
    pub fn proportional(poly: ErrorPoly, gamma: f64) -> Self {
        Self::Proportional {
            gamma: Factor::Variable(gamma),
            poly,
        }
    }

    /// Create a new proportional error model with a fixed gamma parameter
    pub fn proportional_fixed(poly: ErrorPoly, gamma: f64) -> Self {
        Self::Proportional {
            gamma: Factor::Fixed(gamma),
            poly,
        }
    }

    /// Create a new proportional error model with a specified Factor for gamma
    pub fn proportional_with_param(poly: ErrorPoly, gamma: Factor) -> Self {
        Self::Proportional { gamma, poly }
    }

    /// Get the error polynomial coefficients
    pub fn errorpoly(&self) -> ErrorPoly {
        match self {
            Self::Additive { poly, .. } => *poly,
            Self::Proportional { poly, .. } => *poly,
        }
    }

    /// Set the error polynomial coefficients
    pub fn set_errorpoly(&mut self, poly: ErrorPoly) {
        match self {
            Self::Additive { poly: p, .. } => *p = poly,
            Self::Proportional { poly: p, .. } => *p = poly,
        }
    }

    /// Get the scaling parameter value
    pub fn factor(&self) -> f64 {
        match self {
            Self::Additive { lambda, .. } => lambda.value(),
            Self::Proportional { gamma, .. } => gamma.value(),
        }
    }

    /// Set the scaling parameter value (preserves fixed/variable state)
    pub fn set_factor(&mut self, factor: f64) {
        match self {
            Self::Additive { lambda, .. } => lambda.set_value(factor),
            Self::Proportional { gamma, .. } => gamma.set_value(factor),
        }
    }

    /// Get the scaling parameter (including its fixed/variable state)
    pub fn factor_param(&self) -> Factor {
        match self {
            Self::Additive { lambda, .. } => *lambda,
            Self::Proportional { gamma, .. } => *gamma,
        }
    }

    /// Set the scaling parameter (including its fixed/variable state)
    pub fn set_factor_param(&mut self, param: Factor) {
        match self {
            Self::Additive { lambda, .. } => *lambda = param,
            Self::Proportional { gamma, .. } => *gamma = param,
        }
    }

    /// Check if the scaling parameter is fixed
    pub fn is_factor_fixed(&self) -> bool {
        match self {
            Self::Additive { lambda, .. } => lambda.is_fixed(),
            Self::Proportional { gamma, .. } => gamma.is_fixed(),
        }
    }

    /// Make the scaling parameter fixed at its current value
    pub fn fix_factor(&mut self) {
        match self {
            Self::Additive { lambda, .. } => lambda.make_fixed(),
            Self::Proportional { gamma, .. } => gamma.make_fixed(),
        }
    }

    /// Make the scaling parameter variable at its current value
    pub fn unfix_factor(&mut self) {
        match self {
            Self::Additive { lambda, .. } => lambda.make_variable(),
            Self::Proportional { gamma, .. } => gamma.make_variable(),
        }
    }

    /// Check if this is a proportional error model
    pub fn is_proportional(&self) -> bool {
        matches!(self, Self::Proportional { .. })
    }

    /// Check if this is an additive error model
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Additive { .. })
    }

    /// Estimate the standard deviation for a prediction (uses observation value from the prediction)
    ///
    /// This method computes sigma using the **observation** from the prediction, which is
    /// appropriate for non-parametric algorithms (NPAG, NPOD).
    pub fn sigma_from_prediction(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        if prediction.observation.is_none() {
            return Err(ErrorModelError::MissingObservation);
        }

        // Get appropriate polynomial coefficients from prediction or default
        let errorpoly = match prediction.errorpoly() {
            Some(poly) => poly,
            None => self.errorpoly(),
        };

        let (c0, c1, c2, c3) = (errorpoly.c0, errorpoly.c1, errorpoly.c2, errorpoly.c3);
        let obs = prediction.observation().unwrap();

        let alpha = c0 + c1 * obs + c2 * obs.powi(2) + c3 * obs.powi(3);

        let sigma = match self {
            Self::Additive { lambda, .. } => (alpha.powi(2) + lambda.value().powi(2)).sqrt(),
            Self::Proportional { gamma, .. } => gamma.value() * alpha,
        };

        if sigma < 0.0 {
            Err(ErrorModelError::NegativeSigma)
        } else if !sigma.is_finite() {
            Err(ErrorModelError::NonFiniteSigma)
        } else {
            Ok(sigma)
        }
    }

    /// Estimate the variance for a prediction
    pub fn variance_from_prediction(
        &self,
        prediction: &Prediction,
    ) -> Result<f64, ErrorModelError> {
        let sigma = self.sigma_from_prediction(prediction)?;
        Ok(sigma.powi(2))
    }

    /// Get a boolean indicating if the error model should be optimized
    ///
    /// If the [Factor] is variable, it should be optimized.
    pub fn optimize(&self) -> bool {
        match self {
            Self::Additive { lambda, .. } => lambda.is_variable(),
            Self::Proportional { gamma, .. } => gamma.is_variable(),
        }
    }
}

// ── AssayErrorModels: type alias + extension methods ──────────────────────

/// Collection of assay/measurement error models for all outputs.
///
/// This is a type alias for [`ErrorModels<AssayErrorModel>`]. Use the extension methods
/// on [`AssayErrorModelsExt`] for assay-specific operations like computing sigma from
/// a [`Prediction`] object.
pub type AssayErrorModels = ErrorModels<AssayErrorModel>;

/// Deprecated alias for [`AssayErrorModels`].
#[deprecated(
    since = "0.23.0",
    note = "Use AssayErrorModels instead. ErrorModels has been renamed to better reflect its purpose (assay/measurement error)."
)]
pub type OldErrorModels = AssayErrorModels;

/// Extension trait providing assay-specific operations on [`AssayErrorModels`].
///
/// These methods work with [`Prediction`] objects and are specific to observation-based
/// error models used by non-parametric algorithms.
pub trait AssayErrorModelsExt {
    /// Compute sigma using the observation from a [`Prediction`].
    fn sigma_from_prediction(&self, prediction: &Prediction) -> Result<f64, ErrorModelError>;

    /// Compute variance using the observation from a [`Prediction`].
    fn variance_from_prediction(&self, prediction: &Prediction) -> Result<f64, ErrorModelError>;

    /// Returns the error polynomial for the specified output equation.
    fn errorpoly(&self, outeq: usize) -> Result<ErrorPoly, ErrorModelError>;

    /// Returns the factor value for the specified output equation.
    fn factor(&self, outeq: usize) -> Result<f64, ErrorModelError>;

    /// Sets the error polynomial for the specified output equation.
    fn set_errorpoly(&mut self, outeq: usize, poly: ErrorPoly) -> Result<(), ErrorModelError>;

    /// Sets the factor value for the specified output equation.
    fn set_factor(&mut self, outeq: usize, factor: f64) -> Result<(), ErrorModelError>;

    /// Gets the factor parameter (including fixed/variable state).
    fn factor_param(&self, outeq: usize) -> Result<Factor, ErrorModelError>;

    /// Sets the factor parameter (including fixed/variable state).
    fn set_factor_param(&mut self, outeq: usize, param: Factor) -> Result<(), ErrorModelError>;

    /// Checks if the factor parameter is fixed.
    fn is_factor_fixed(&self, outeq: usize) -> Result<bool, ErrorModelError>;

    /// Makes the factor parameter fixed at its current value.
    fn fix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError>;

    /// Makes the factor parameter variable at its current value.
    fn unfix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError>;

    /// Check if the error model for the output equation is proportional.
    fn is_proportional(&self, outeq: usize) -> bool;

    /// Check if the error model for the output equation is additive.
    fn is_additive(&self, outeq: usize) -> bool;

    /// Compute a hash for the error models collection.
    fn hash(&self) -> u64;
}

impl AssayErrorModelsExt for AssayErrorModels {
    fn sigma_from_prediction(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        let outeq = prediction.outeq;
        self.get(outeq)?.sigma_from_prediction(prediction)
    }

    fn variance_from_prediction(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        let outeq = prediction.outeq;
        self.get(outeq)?.variance_from_prediction(prediction)
    }

    fn errorpoly(&self, outeq: usize) -> Result<ErrorPoly, ErrorModelError> {
        Ok(self.get(outeq)?.errorpoly())
    }

    fn factor(&self, outeq: usize) -> Result<f64, ErrorModelError> {
        Ok(self.get(outeq)?.factor())
    }

    fn set_errorpoly(&mut self, outeq: usize, poly: ErrorPoly) -> Result<(), ErrorModelError> {
        self.get_mut(outeq)?.set_errorpoly(poly);
        Ok(())
    }

    fn set_factor(&mut self, outeq: usize, factor: f64) -> Result<(), ErrorModelError> {
        self.get_mut(outeq)?.set_factor(factor);
        Ok(())
    }

    fn factor_param(&self, outeq: usize) -> Result<Factor, ErrorModelError> {
        Ok(self.get(outeq)?.factor_param())
    }

    fn set_factor_param(&mut self, outeq: usize, param: Factor) -> Result<(), ErrorModelError> {
        self.get_mut(outeq)?.set_factor_param(param);
        Ok(())
    }

    fn is_factor_fixed(&self, outeq: usize) -> Result<bool, ErrorModelError> {
        Ok(self.get(outeq)?.is_factor_fixed())
    }

    fn fix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError> {
        self.get_mut(outeq)?.fix_factor();
        Ok(())
    }

    fn unfix_factor(&mut self, outeq: usize) -> Result<(), ErrorModelError> {
        self.get_mut(outeq)?.unfix_factor();
        Ok(())
    }

    fn is_proportional(&self, outeq: usize) -> bool {
        self.get(outeq)
            .map(|m| m.is_proportional())
            .unwrap_or(false)
    }

    fn is_additive(&self, outeq: usize) -> bool {
        self.get(outeq).map(|m| m.is_additive()).unwrap_or(false)
    }

    fn hash(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();

        for (outeq, model) in self.iter() {
            outeq.hash(&mut hasher);
            match model {
                AssayErrorModel::Additive { lambda, .. } => {
                    0u8.hash(&mut hasher);
                    lambda.value().to_bits().hash(&mut hasher);
                    lambda.is_fixed().hash(&mut hasher);
                }
                AssayErrorModel::Proportional { gamma, .. } => {
                    1u8.hash(&mut hasher);
                    gamma.value().to_bits().hash(&mut hasher);
                    gamma.is_fixed().hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Censor, Observation};

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(
            model.sigma_from_prediction(&prediction).unwrap(),
            (26.0_f64).sqrt()
        );
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma_from_prediction(&prediction).unwrap(), 2.0);
    }

    #[test]
    fn test_trait_sigma() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        // ErrorModel trait: sigma from raw value (observation = 20.0)
        // alpha = 1.0, sigma = sqrt(1.0 + 25.0) = sqrt(26)
        assert_eq!(ErrorModel::sigma(&model, 20.0).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_polynomial() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.errorpoly().coefficients(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_set_errorpoly() {
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.errorpoly().coefficients(), (1.0, 2.0, 3.0, 4.0));
        model.set_errorpoly(ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(model.errorpoly().coefficients(), (5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn test_set_factor() {
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.factor(), 5.0);
        model.set_factor(10.0);
        assert_eq!(model.factor(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(ErrorModel::sigma(&model, 20.0).unwrap(), (26.0_f64).sqrt());

        let model = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(ErrorModel::sigma(&model, 20.0).unwrap(), 2.0);
    }

    #[test]
    fn test_error_models_new() {
        let models = AssayErrorModels::new();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_error_models_default() {
        let models = AssayErrorModels::default();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_error_models_add_single() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_error_models_add_multiple() {
        let model1 = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = AssayErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let models = AssayErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(1, model2)
            .unwrap();

        assert_eq!(models.len(), 2);
    }
}
