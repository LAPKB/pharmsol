use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use crate::simulator::likelihood::distributions::{lognormccdf, lognormcdf, lognormpdf};
use crate::simulator::likelihood::observation::finite;
use crate::{
    data::event::OutputLabel,
    simulator::likelihood::{ObservationModel, Prediction},
    Censor, PharmsolError,
};
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

/// Collection of assay/measurement error models, keyed by output label.
///
/// This struct represents **measurement/assay noise** - the error associated with
/// quantification of drug concentration in biological samples. Sigma is computed
/// from the **observation** value.
///
/// Used by non-parametric algorithms (NPAG, NPOD, etc.).
///
/// For parametric algorithms (SAEM, FOCE), use [`crate::ResidualErrorModels`] instead,
/// which computes sigma from the **prediction**.
///
/// Every model is addressed by the output label it was added for. Labels match
/// an equation's declared output names exactly: `"1"` matches an output
/// declared as `1`, and `"cp"` an output declared as `cp`. Equations without
/// metadata declare no output names; their outputs are labeled by position
/// (`"0"`, `"1"`, ...), like the observation labels resolved against them.
///
/// ```rust
/// # use pharmsol::prelude::*;
/// let mut error_models = AssayErrorModels::new()
///     .add("cp", AssayErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.5))?;
///
/// for (label, model) in error_models.iter_mut() {
///     if model.optimize() {
///         model.set_factor(model.factor()? * 2.0);
///     }
///     assert_eq!(label.as_str(), "cp");
/// }
/// assert_eq!(error_models.factor("cp")?, 1.0);
/// # Ok::<(), pharmsol::data::error_model::ErrorModelError>(())
/// ```
#[derive(Serialize, Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(transparent)]
pub struct AssayErrorModels {
    models: BTreeMap<OutputLabel, AssayErrorModel>,
}

/// Deprecated alias for [`AssayErrorModels`].
///
/// This type alias is provided for backward compatibility.
/// New code should use [`AssayErrorModels`] directly.
#[deprecated(
    since = "0.23.0",
    note = "Use AssayErrorModels instead. ErrorModels has been renamed to better reflect its purpose (assay/measurement error)."
)]
pub type ErrorModels = AssayErrorModels;

fn hash_model<H: Hasher>(hasher: &mut H, model: &AssayErrorModel) {
    match model {
        AssayErrorModel::Additive { lambda, poly } => {
            0u8.hash(hasher);
            lambda.value().to_bits().hash(hasher);
            lambda.is_fixed().hash(hasher);
            hash_poly(hasher, poly);
        }
        AssayErrorModel::Proportional { gamma, poly } => {
            1u8.hash(hasher);
            gamma.value().to_bits().hash(hasher);
            gamma.is_fixed().hash(hasher);
            hash_poly(hasher, poly);
        }
        AssayErrorModel::None => {
            2u8.hash(hasher);
        }
    }
}

/// The polynomial is part of the hash because likelihood caches (SDE) key on
/// it: changing only the polynomial must not reuse a cached likelihood.
fn hash_poly<H: Hasher>(hasher: &mut H, poly: &ErrorPoly) {
    let (c0, c1, c2, c3) = poly.coefficients();
    for coefficient in [c0, c1, c2, c3] {
        coefficient.to_bits().hash(hasher);
    }
}

impl AssayErrorModels {
    /// Create an empty [`AssayErrorModels`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error model for an output label.
    ///
    /// # Arguments
    /// * `outeq` - The output label, matched exactly against the model's declared outputs.
    /// * `model` - The [AssayErrorModel] to add for that output.
    /// # Returns
    /// A new instance of AssayErrorModels with the added model.
    /// # Errors
    /// [ErrorModelError::ExistingOutputLabel] if a model already exists for that label.
    pub fn add(
        mut self,
        outeq: impl ToString,
        model: AssayErrorModel,
    ) -> Result<Self, ErrorModelError> {
        let label = OutputLabel::new(outeq);
        if self.models.contains_key(&label) {
            return Err(ErrorModelError::ExistingOutputLabel(label.to_string()));
        }
        self.models.insert(label, model);
        Ok(self)
    }

    /// Get the error model for an output label, if one was added.
    pub fn get(&self, outeq: impl ToString) -> Option<&AssayErrorModel> {
        self.models.get(outeq.to_string().as_str())
    }

    /// Get a mutable reference to the error model for an output label, if one was added.
    pub fn get_mut(&mut self, outeq: impl ToString) -> Option<&mut AssayErrorModel> {
        self.models.get_mut(outeq.to_string().as_str())
    }

    /// Whether an error model was added for this output label.
    pub fn contains(&self, outeq: impl ToString) -> bool {
        self.models.contains_key(outeq.to_string().as_str())
    }

    /// Get the error model for an output label.
    ///
    /// # Errors
    /// [ErrorModelError::UnknownOutputLabel] if no model was added for that label.
    pub fn error_model(&self, outeq: impl ToString) -> Result<&AssayErrorModel, ErrorModelError> {
        let label = outeq.to_string();
        self.models
            .get(label.as_str())
            .ok_or(ErrorModelError::UnknownOutputLabel(label))
    }

    /// The error model for `label`, which must exist and not be [AssayErrorModel::None].
    ///
    /// Looks the label up as a `&str`, so the likelihood path never allocates.
    #[inline]
    fn defined_label(&self, label: &str) -> Result<&AssayErrorModel, ErrorModelError> {
        match self.models.get(label) {
            None => Err(ErrorModelError::UnknownOutputLabel(label.to_string())),
            Some(AssayErrorModel::None) => Err(ErrorModelError::NoneErrorModel(label.to_string())),
            Some(model) => Ok(model),
        }
    }

    /// [`Self::defined_label`] for any label-like value.
    fn defined(&self, outeq: impl ToString) -> Result<&AssayErrorModel, ErrorModelError> {
        self.defined_label(&outeq.to_string())
    }

    /// Mutable variant of [`Self::defined`].
    fn defined_mut(
        &mut self,
        outeq: impl ToString,
    ) -> Result<&mut AssayErrorModel, ErrorModelError> {
        let label = outeq.to_string();
        match self.models.get_mut(label.as_str()) {
            None => Err(ErrorModelError::UnknownOutputLabel(label)),
            Some(AssayErrorModel::None) => Err(ErrorModelError::NoneErrorModel(label)),
            Some(model) => Ok(model),
        }
    }

    /// Iterate over `(label, model)` pairs, in label order.
    pub fn iter(&self) -> impl Iterator<Item = (&OutputLabel, &AssayErrorModel)> {
        self.models.iter()
    }

    /// Iterate mutably over `(label, model)` pairs, in label order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&OutputLabel, &mut AssayErrorModel)> {
        self.models.iter_mut()
    }

    /// Iterate over the output labels that have an error model.
    pub fn labels(&self) -> impl Iterator<Item = &OutputLabel> {
        self.models.keys()
    }

    /// Computes a hash for the error models collection.
    /// This hash is based on the output labels and their associated error models.
    /// # Returns
    /// A `u64` hash value representing the error models collection.
    pub fn hash(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        for (label, model) in &self.models {
            label.hash(&mut hasher);
            hash_model(&mut hasher, model);
        }
        hasher.finish()
    }

    /// Returns the number of error models in the collection.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns whether the collection contains no error models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Returns the error polynomial for an output label.
    pub fn errorpoly(&self, outeq: impl ToString) -> Result<ErrorPoly, ErrorModelError> {
        self.defined(outeq)?.errorpoly()
    }

    /// Returns the factor value (gamma or lambda) for an output label.
    pub fn factor(&self, outeq: impl ToString) -> Result<f64, ErrorModelError> {
        self.defined(outeq)?.factor()
    }

    /// Sets the error polynomial for an output label.
    pub fn set_errorpoly(
        &mut self,
        outeq: impl ToString,
        poly: ErrorPoly,
    ) -> Result<(), ErrorModelError> {
        self.defined_mut(outeq)?.set_errorpoly(poly);
        Ok(())
    }

    /// Sets the factor value (gamma or lambda) for an output label.
    pub fn set_factor(&mut self, outeq: impl ToString, factor: f64) -> Result<(), ErrorModelError> {
        self.defined_mut(outeq)?.set_factor(factor);
        Ok(())
    }

    /// Gets the factor parameter (including fixed/variable state) for an output label.
    pub fn factor_param(&self, outeq: impl ToString) -> Result<Factor, ErrorModelError> {
        self.defined(outeq)?.factor_param()
    }

    /// Sets the factor parameter (including fixed/variable state) for an output label.
    pub fn set_factor_param(
        &mut self,
        outeq: impl ToString,
        param: Factor,
    ) -> Result<(), ErrorModelError> {
        self.defined_mut(outeq)?.set_factor_param(param);
        Ok(())
    }

    /// Checks if the factor parameter is fixed for an output label.
    pub fn is_factor_fixed(&self, outeq: impl ToString) -> Result<bool, ErrorModelError> {
        self.defined(outeq)?.is_factor_fixed()
    }

    /// Makes the factor parameter fixed at its current value for an output label.
    pub fn fix_factor(&mut self, outeq: impl ToString) -> Result<(), ErrorModelError> {
        self.defined_mut(outeq)?.fix_factor();
        Ok(())
    }

    /// Makes the factor parameter variable at its current value for an output label.
    pub fn unfix_factor(&mut self, outeq: impl ToString) -> Result<(), ErrorModelError> {
        self.defined_mut(outeq)?.unfix_factor();
        Ok(())
    }

    /// Whether the error model for an output label is proportional.
    pub fn is_proportional(&self, outeq: impl ToString) -> bool {
        self.get(outeq)
            .is_some_and(AssayErrorModel::is_proportional)
    }

    /// Whether the error model for an output label is additive.
    pub fn is_additive(&self, outeq: impl ToString) -> bool {
        self.get(outeq).is_some_and(AssayErrorModel::is_additive)
    }

    /// Computes the standard deviation (sigma) for a prediction, using the
    /// error model of the prediction's output label.
    ///
    /// This always uses the **observation** value to compute sigma, which is appropriate
    /// for non-parametric algorithms (NPAG, NPOD). For parametric algorithms (SAEM, FOCE),
    /// use [`crate::ResidualErrorModels`] instead, which computes sigma from the prediction.
    #[inline]
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.defined_label(prediction.outeq().as_str())?
            .sigma(prediction)
    }

    /// Computes the variance for a prediction, using the error model of the
    /// prediction's output label.
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.defined_label(prediction.outeq().as_str())?
            .variance(prediction)
    }

    /// Computes the standard deviation (sigma) for an output label and value.
    pub fn sigma_from_value(
        &self,
        outeq: impl ToString,
        value: f64,
    ) -> Result<f64, ErrorModelError> {
        self.defined(outeq)?.sigma_from_value(value)
    }

    /// Computes the variance for an output label and value.
    pub fn variance_from_value(
        &self,
        outeq: impl ToString,
        value: f64,
    ) -> Result<f64, ErrorModelError> {
        self.defined(outeq)?.variance_from_value(value)
    }
}

/// Tag mixed into [`AssayErrorModels`] cache keys, so they never collide with
/// another observation model's.
const ASSAY_CACHE_TAG: u64 = 0x6173_7361_7900_0001;

impl ObservationModel for AssayErrorModels {
    /// Normal density with sigma from the observed value; censored
    /// observations use the normal CDF (BLOQ) or survival function (ALOQ).
    #[inline]
    fn log_density(
        &self,
        outeq: &OutputLabel,
        observation: f64,
        prediction: f64,
        censoring: Censor,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        let sigma = self
            .defined_label(outeq.as_str())?
            .sigma_for(observation, errorpoly)?;
        let log_density = match censoring {
            Censor::None => lognormpdf(observation, prediction, sigma),
            Censor::BLOQ => lognormcdf(observation, prediction, sigma)?,
            Censor::ALOQ => lognormccdf(observation, prediction, sigma)?,
        };
        finite(log_density)
    }

    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_> {
        Box::new(self.models.keys())
    }

    fn cache_key(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        ASSAY_CACHE_TAG.hash(&mut hasher);
        self.hash().hash(&mut hasher);
        hasher.finish()
    }
}

impl IntoIterator for AssayErrorModels {
    type Item = (OutputLabel, AssayErrorModel);
    type IntoIter = std::collections::btree_map::IntoIter<OutputLabel, AssayErrorModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.into_iter()
    }
}

impl<'a> IntoIterator for &'a AssayErrorModels {
    type Item = (&'a OutputLabel, &'a AssayErrorModel);
    type IntoIter = std::collections::btree_map::Iter<'a, OutputLabel, AssayErrorModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter()
    }
}

impl<'a> IntoIterator for &'a mut AssayErrorModels {
    type Item = (&'a OutputLabel, &'a mut AssayErrorModel);
    type IntoIter = std::collections::btree_map::IterMut<'a, OutputLabel, AssayErrorModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter_mut()
    }
}

/// Model for calculating observation errors in pharmacometric analyses
///
/// An [AssayErrorModel] defines how the standard deviation of observations is calculated
/// based on the type of error model used and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum AssayErrorModel {
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

/// Deprecated alias for [`AssayErrorModel`].
///
/// This type alias is provided for backward compatibility.
/// New code should use [`AssayErrorModel`] directly.
#[deprecated(
    since = "0.23.0",
    note = "Use AssayErrorModel instead. ErrorModel has been renamed to better reflect its purpose (assay/measurement error)."
)]
pub type ErrorModel = AssayErrorModel;

impl AssayErrorModel {
    /// Create a new additive error model with a variable lambda parameter
    ///
    /// # Arguments
    ///
    /// * `poly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `lambda` - Lambda parameter for scaling errors (will be variable)
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

    /// Check if this is a proportional error model
    ///
    /// # Returns
    ///
    /// `true` if this is a `Proportional` variant, `false` otherwise
    pub fn is_proportional(&self) -> bool {
        matches!(self, Self::Proportional { .. })
    }

    /// Check if this is an additive error model
    ///
    /// # Returns
    ///
    /// `true` if this is an `Additive` variant, `false` otherwise
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Additive { .. })
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
        let observation = prediction
            .observation()
            .ok_or(ErrorModelError::MissingObservation)?;
        self.sigma_for(observation, prediction.errorpoly())
    }

    /// [`Self::sigma`] for an observed value and its error polynomial, if
    /// it carries one (otherwise the model's own polynomial is used).
    #[inline]
    pub(crate) fn sigma_for(
        &self,
        observation: f64,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, ErrorModelError> {
        // Get appropriate polynomial coefficients from the observation or default
        let errorpoly = match errorpoly {
            Some(poly) => poly,
            None => self.errorpoly()?,
        };

        let (c0, c1, c2, c3) = (errorpoly.c0, errorpoly.c1, errorpoly.c2, errorpoly.c3);

        // Calculate alpha term
        let alpha = c0 + c1 * observation + c2 * observation.powi(2) + c3 * observation.powi(3);

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
    /// This is a convenience function which calls [AssayErrorModel::sigma], and squares the result.
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
    /// This is a convenience function which calls [AssayErrorModel::sigma_from_value], and squares the result.
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
    #[error("The output label `{0}` is not declared in this error model context")]
    UnknownOutputLabel(String),
    #[error("The output label `{0}` already exists in this assay error model specification")]
    ExistingOutputLabel(String),
    #[error("An output equation does not have an error model defined")]
    MissingErrorModel,
    #[error("The error model for output `{0}` is of type ErrorModel::None")]
    NoneErrorModel(String),
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
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma(&prediction).unwrap(), 2.0);
    }

    #[test]
    fn test_polynomial() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(
            model.errorpoly().unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_set_errorpoly() {
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
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
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.factor().unwrap(), 5.0);
        model.set_factor(10.0);
        assert_eq!(model.factor().unwrap(), 10.0);
    }

    #[test]
    fn test_sigma_from_value() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
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

    #[test]
    fn test_error_models_duplicate_label_fails() {
        let result = AssayErrorModels::new()
            .add(
                "cp",
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap()
            .add(
                "cp",
                AssayErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0),
            );

        match result {
            Err(ErrorModelError::ExistingOutputLabel(label)) => assert_eq!(label, "cp"),
            _ => panic!("Expected ExistingOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_sigma_uses_prediction_label() {
        let models = AssayErrorModels::new()
            .add(
                "cp",
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();

        let observation = Observation::new(0.0, Some(20.0), "cp", None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        assert_eq!(models.sigma(&prediction).unwrap(), (26.0_f64).sqrt());

        let observation = Observation::new(0.0, Some(20.0), "effect", None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        match models.sigma(&prediction) {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "effect"),
            other => panic!("Expected UnknownOutputLabel error, got {other:?}"),
        }
    }

    #[test]
    fn test_error_models_add_duplicate_numeric_label_fails() {
        let model1 = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = AssayErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let result = AssayErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(0, model2); // Same outeq should fail

        assert!(result.is_err());
        match result {
            Err(ErrorModelError::ExistingOutputLabel(label)) => assert_eq!(label, "0"),
            _ => panic!("Expected ExistingOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_factor() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.factor(0).unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_factor_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.factor(1);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_set_factor() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = AssayErrorModels::new().add(0, model).unwrap();

        assert_eq!(models.factor(0).unwrap(), 5.0);
        models.set_factor(0, 10.0).unwrap();
        assert_eq!(models.factor(0).unwrap(), 10.0);
    }

    #[test]
    fn test_error_models_set_factor_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.set_factor(1, 10.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_errorpoly() {
        let poly = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let model = AssayErrorModel::additive(poly, 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let retrieved_poly = models.errorpoly(0).unwrap();
        assert_eq!(retrieved_poly.coefficients(), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_error_models_errorpoly_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.errorpoly(1);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_set_errorpoly() {
        let poly1 = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let poly2 = ErrorPoly::new(5.0, 6.0, 7.0, 8.0);
        let model = AssayErrorModel::additive(poly1, 5.0);
        let mut models = AssayErrorModels::new().add(0, model).unwrap();

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
    fn test_error_models_set_errorpoly_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let mut models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.set_errorpoly(1, ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_sigma() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        // Non-parametric: sigma from observation
        let sigma = models.sigma(&prediction).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None); // outeq=1 not in models
        let prediction = observation.to_prediction(10.0, vec![]);

        let result = models.sigma(&prediction);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_variance() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let variance = models.variance(&prediction).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None); // outeq=1 not in models
        let prediction = observation.to_prediction(10.0, vec![]);

        let result = models.variance(&prediction);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_sigma_from_value() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let sigma = models.sigma_from_value(0, 20.0).unwrap();
        assert_eq!(sigma, (26.0_f64).sqrt());
    }

    #[test]
    fn test_error_models_sigma_from_value_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.sigma_from_value(1, 20.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_variance_from_value() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let variance = models.variance_from_value(0, 20.0).unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(variance, expected_sigma.powi(2));
    }

    #[test]
    fn test_error_models_variance_from_value_unknown_label() {
        let model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let models = AssayErrorModels::new().add(0, model).unwrap();

        let result = models.variance_from_value(1, 20.0);
        assert!(result.is_err());
        match result {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "1"),
            _ => panic!("Expected UnknownOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_hash_consistency() {
        let model1 = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = AssayErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        let models1 = AssayErrorModels::new()
            .add(0, model1.clone())
            .unwrap()
            .add(1, model2.clone())
            .unwrap();

        let models2 = AssayErrorModels::new()
            .add(0, model1)
            .unwrap()
            .add(1, model2)
            .unwrap();

        // Same models should produce same hash
        assert_eq!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_hash_order_independence() {
        let model1 = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model2 = AssayErrorModel::proportional(ErrorPoly::new(2.0, 0.0, 0.0, 0.0), 3.0);

        // Add in different orders
        let models1 = AssayErrorModels::new()
            .add(0, model1.clone())
            .unwrap()
            .add(1, model2.clone())
            .unwrap();

        let models2 = AssayErrorModels::new()
            .add(1, model2)
            .unwrap()
            .add(0, model1)
            .unwrap();

        // Hash should be the same regardless of insertion order
        assert_eq!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_multiple_outeqs() {
        let additive_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5);
        let proportional_model =
            AssayErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.1);

        let models = AssayErrorModels::new()
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
        let additive_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model =
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let models = AssayErrorModels::new()
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
        let additive = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(additive.factor().unwrap(), 5.0);
        assert!(!additive.is_factor_fixed().unwrap());

        let proportional = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(proportional.factor().unwrap(), 2.0);
        assert!(!proportional.is_factor_fixed().unwrap());

        // Test fixed constructors
        let additive_fixed =
            AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(additive_fixed.factor().unwrap(), 5.0);
        assert!(additive_fixed.is_factor_fixed().unwrap());

        let proportional_fixed =
            AssayErrorModel::proportional_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(proportional_fixed.factor().unwrap(), 2.0);
        assert!(proportional_fixed.is_factor_fixed().unwrap());

        // Test Factor constructors
        let additive_with_param = AssayErrorModel::additive_with_param(
            ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
            Factor::Fixed(5.0),
        );
        assert_eq!(additive_with_param.factor().unwrap(), 5.0);
        assert!(additive_with_param.is_factor_fixed().unwrap());

        let proportional_with_param = AssayErrorModel::proportional_with_param(
            ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
            Factor::Variable(2.0),
        );
        assert_eq!(proportional_with_param.factor().unwrap(), 2.0);
        assert!(!proportional_with_param.is_factor_fixed().unwrap());
    }

    #[test]
    fn test_factor_param_methods() {
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

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
        let additive_model =
            AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model =
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let mut models = AssayErrorModels::new()
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

        let model_variable = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model_fixed = AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

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
        let model1_variable = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let model1_fixed = AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

        let models1 = AssayErrorModels::new().add(0, model1_variable).unwrap();
        let models2 = AssayErrorModels::new().add(0, model1_fixed).unwrap();

        // Different fixed/variable states should produce different hashes
        assert_ne!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_into_iter_functionality() {
        let additive_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        let proportional_model =
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);

        let mut models = AssayErrorModels::new()
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
            match outeq.as_str() {
                "0" => model.set_factor(10.0), // Update additive lambda from 5.0 to 10.0
                "1" => model.set_factor(4.0),  // Update proportional gamma from 2.0 to 4.0
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
            match outeq.as_str() {
                "0" => {
                    assert!(model.is_factor_fixed().unwrap());
                    assert_eq!(model.factor().unwrap(), 10.0);
                }
                "1" => {
                    assert!(model.is_factor_fixed().unwrap());
                    assert_eq!(model.factor().unwrap(), 4.0);
                }
                _ => panic!("Unexpected outeq: {}", outeq),
            }
        }
        assert_eq!(count, 2);

        // Test consuming iteration with into_iter()
        let collected_models: Vec<(OutputLabel, AssayErrorModel)> = models.into_iter().collect();
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

    #[test]
    fn error_model_hash_deterministic() {
        let models = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();
        assert_eq!(models.hash(), models.hash());
    }

    #[test]
    fn error_model_hash_differs_on_value() {
        let a = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();
        let b = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 10.0),
            )
            .unwrap();
        assert_ne!(a.hash(), b.hash());
    }

    #[test]
    fn numeric_and_text_labels_name_the_same_output() {
        let mut models = AssayErrorModels::new()
            .add(
                1,
                AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0),
            )
            .unwrap();
        assert!(!models.is_empty());

        // `1` and `"1"` are the same label; `"01"` is a different one.
        assert!(models.contains(1) && models.contains("1"));
        assert!(!models.contains("01"));
        assert_eq!(models.get(1), models.get("1"));
        assert!(models.error_model("01").is_err());
        assert!(models.is_proportional("1") && !models.is_additive(1));
        assert!(!models.is_proportional("01"));

        // Changes through one spelling are seen through the other.
        models.get_mut(1).unwrap().set_factor(3.0);
        assert_eq!(models.factor("1").unwrap(), 3.0);

        // Iterating `&mut` reaches every model, as an optimizer loop would.
        for (_, model) in &mut models {
            model.fix_factor();
        }
        assert!(models.is_factor_fixed("1").unwrap());
        assert_eq!((&models).into_iter().count(), 1);
    }

    #[test]
    fn error_model_hash_differs_on_polynomial() {
        for model in [
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
        ] {
            let a = AssayErrorModels::new().add("cp", model).unwrap();
            let mut b = a.clone();
            b.set_errorpoly("cp", ErrorPoly::new(1.0, 0.1, 0.0, 0.0))
                .unwrap();
            assert_ne!(a.hash(), b.hash());
        }
    }

    #[test]
    fn error_model_hash_differs_on_type() {
        let a = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();
        let b = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();
        assert_ne!(a.hash(), b.hash());
    }
}
