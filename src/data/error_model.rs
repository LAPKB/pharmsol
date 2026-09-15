use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use crate::{data::event::OutputLabel, simulator::likelihood::Prediction};
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
/// Every model is stored against the public [`OutputLabel`] it was added under,
/// and there is no index anywhere in the type. Labels are resolved to dense
/// output slots exactly once, by the equation, when the set is bound for
/// simulation or likelihood evaluation. The resulting dense view is
/// [`DenseAssayErrorModels`], which is internal to the runtime path.
#[derive(Serialize, Debug, Clone, Deserialize, Default)]
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

impl AssayErrorModels {
    /// Create a new, empty label-keyed [`AssayErrorModels`] definition.
    ///
    /// Output labels are resolved once per equation when the error models are
    /// used through simulation or likelihood entrypoints.
    ///
    /// This lets the same public definition be reused safely across multiple
    /// equations while keeping the dense bound representation internal to the
    /// runtime path.
    ///
    /// ```rust
    /// # use pharmsol::prelude::*;
    /// let error_models = AssayErrorModels::new()
    ///     .add("cp", AssayErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0))?;
    ///
    /// assert_eq!(error_models.len(), 1);
    /// assert_eq!(error_models.factor("cp")?, 0.0);
    /// # Ok::<(), pharmsol::data::error_model::ErrorModelError>(())
    /// ```
    pub fn new() -> Self {
        Self {
            models: BTreeMap::new(),
        }
    }

    /// Add a new error model for an output label.
    ///
    /// A bare number is treated as its OUTEQ number (`1` matches a declared
    /// `outeq_1`), exactly like data labels, never as a dense slot. The label is
    /// resolved when the collection is bound to an equation, using the same
    /// resolver that observation labels go through.
    ///
    /// # Arguments
    /// * `label` - The public output label, or the OUTEQ number for that output.
    /// * `model` - The [AssayErrorModel] to add for the specified output.
    ///
    /// # Returns
    /// A new instance of [`AssayErrorModels`] with the added model.
    ///
    /// # Errors
    /// [`ErrorModelError::ExistingOutputLabel`] if a model was already added for
    /// that label.
    pub fn add(
        mut self,
        label: impl ToString,
        model: AssayErrorModel,
    ) -> Result<Self, ErrorModelError> {
        let label = OutputLabel::new(label);
        if self.models.contains_key(&label) {
            return Err(ErrorModelError::ExistingOutputLabel(label.to_string()));
        }
        self.models.insert(label, model);
        Ok(self)
    }

    /// Returns an iterator over the label/model pairs in the collection.
    ///
    /// Iteration order is the label order, so it does not depend on insertion
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (&OutputLabel, &AssayErrorModel)> {
        self.models.iter()
    }

    /// Returns an iterator yielding mutable references to the models, together
    /// with the label each one was added under.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&OutputLabel, &mut AssayErrorModel)> {
        self.models.iter_mut()
    }

    /// Returns the number of error models in the collection.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns whether the collection contains no error models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Returns `true` if a model has been added for `label`.
    pub fn contains(&self, label: impl ToString) -> bool {
        self.models.contains_key(&OutputLabel::new(label))
    }

    /// Returns the labels that have a model, in label order.
    pub fn labels(&self) -> impl Iterator<Item = &OutputLabel> {
        self.models.keys()
    }

    /// Computes a hash for the error models collection.
    ///
    /// The hash covers every label and its model. [`BTreeMap`] iterates in label
    /// order, so the hash does not depend on insertion order.
    pub fn hash(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        for (label, model) in &self.models {
            label.hash(&mut hasher);
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
                AssayErrorModel::None => {
                    2u8.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Get the error model added for an output label.
    ///
    /// # Errors
    /// [`ErrorModelError::UnknownOutputLabel`] if no model was added for `label`.
    pub fn error_model(&self, label: impl ToString) -> Result<&AssayErrorModel, ErrorModelError> {
        let label = OutputLabel::new(label);
        self.models
            .get(&label)
            .ok_or_else(|| ErrorModelError::UnknownOutputLabel(label.to_string()))
    }

    /// Get a mutable reference to the error model added for an output label.
    ///
    /// # Errors
    /// [`ErrorModelError::UnknownOutputLabel`] if no model was added for `label`.
    pub fn error_model_mut(
        &mut self,
        label: impl ToString,
    ) -> Result<&mut AssayErrorModel, ErrorModelError> {
        let label = OutputLabel::new(label);
        self.models
            .get_mut(&label)
            .ok_or_else(|| ErrorModelError::UnknownOutputLabel(label.to_string()))
    }

    /// Returns the error polynomial associated with the given output label.
    pub fn errorpoly(&self, label: impl ToString) -> Result<ErrorPoly, ErrorModelError> {
        self.error_model(label)?.errorpoly()
    }

    /// Sets the error polynomial for the given output label.
    pub fn set_errorpoly(
        &mut self,
        label: impl ToString,
        poly: ErrorPoly,
    ) -> Result<(), ErrorModelError> {
        let model = self.error_model_mut(label)?;
        if model == &AssayErrorModel::None {
            return Err(ErrorModelError::MissingErrorModel);
        }
        model.set_errorpoly(poly);
        Ok(())
    }

    /// Returns the factor value (lambda or gamma) for the given output label.
    pub fn factor(&self, label: impl ToString) -> Result<f64, ErrorModelError> {
        self.error_model(label)?.factor()
    }

    /// Sets the factor value (lambda or gamma) for the given output label.
    pub fn set_factor(&mut self, label: impl ToString, factor: f64) -> Result<(), ErrorModelError> {
        let model = self.error_model_mut(label)?;
        if model == &AssayErrorModel::None {
            return Err(ErrorModelError::MissingErrorModel);
        }
        model.set_factor(factor);
        Ok(())
    }

    /// Gets the factor parameter (including fixed/variable state) for the given output label.
    pub fn factor_param(&self, label: impl ToString) -> Result<Factor, ErrorModelError> {
        self.error_model(label)?.factor_param()
    }

    /// Sets the factor parameter (including fixed/variable state) for the given output label.
    pub fn set_factor_param(
        &mut self,
        label: impl ToString,
        param: Factor,
    ) -> Result<(), ErrorModelError> {
        let model = self.error_model_mut(label)?;
        if model == &AssayErrorModel::None {
            return Err(ErrorModelError::MissingErrorModel);
        }
        model.set_factor_param(param);
        Ok(())
    }

    /// Checks if the factor parameter is fixed for the given output label.
    pub fn is_factor_fixed(&self, label: impl ToString) -> Result<bool, ErrorModelError> {
        self.error_model(label)?.is_factor_fixed()
    }

    /// Makes the factor parameter fixed at its current value for the given output label.
    pub fn fix_factor(&mut self, label: impl ToString) -> Result<(), ErrorModelError> {
        let model = self.error_model_mut(label)?;
        if model == &AssayErrorModel::None {
            return Err(ErrorModelError::MissingErrorModel);
        }
        model.fix_factor();
        Ok(())
    }

    /// Makes the factor parameter variable at its current value for the given output label.
    pub fn unfix_factor(&mut self, label: impl ToString) -> Result<(), ErrorModelError> {
        let model = self.error_model_mut(label)?;
        if model == &AssayErrorModel::None {
            return Err(ErrorModelError::MissingErrorModel);
        }
        model.unfix_factor();
        Ok(())
    }

    /// Check if the error model for the given output label is proportional.
    pub fn is_proportional(&self, label: impl ToString) -> bool {
        self.error_model(label)
            .map(AssayErrorModel::is_proportional)
            .unwrap_or(false)
    }

    /// Check if the error model for the given output label is additive.
    pub fn is_additive(&self, label: impl ToString) -> bool {
        self.error_model(label)
            .map(AssayErrorModel::is_additive)
            .unwrap_or(false)
    }

    /// Computes the standard deviation (sigma) for the given output label and value.
    pub fn sigma_from_value(
        &self,
        label: impl ToString,
        value: f64,
    ) -> Result<f64, ErrorModelError> {
        self.error_model(label)?.sigma_from_value(value)
    }

    /// Computes the variance for the given output label and value.
    pub fn variance_from_value(
        &self,
        label: impl ToString,
        value: f64,
    ) -> Result<f64, ErrorModelError> {
        self.error_model(label)?.variance_from_value(value)
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

/// Dense, equation-bound view of a set of [`AssayErrorModels`].
///
/// This is the runtime representation used on the per-observation hot path. It
/// is produced only by binding a public [`AssayErrorModels`] set to an equation,
/// which sizes it to the equation's output count and resolves every label
/// through the same resolver that observation labels use. Indexing it with a
/// resolved `outeq` is therefore always meaningful.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct DenseAssayErrorModels(Vec<AssayErrorModel>);

impl DenseAssayErrorModels {
    /// Build a dense set directly from already-resolved output slots.
    ///
    /// Only the binding step and in-crate tests construct one this way; public
    /// input always goes through [`AssayErrorModels`].
    pub(crate) fn from_dense(models: Vec<AssayErrorModel>) -> Self {
        Self(models)
    }

    /// Number of dense output slots, which equals the equation's output count.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether there are no output slots at all.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Hash of the dense models, used as a likelihood-cache key.
    pub fn hash(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        for (outeq, model) in self.0.iter().enumerate() {
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
                AssayErrorModel::None => {
                    2u8.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Get the error model for a resolved output equation index.
    #[inline]
    pub fn error_model(&self, outeq: usize) -> Result<&AssayErrorModel, ErrorModelError> {
        self.0
            .get(outeq)
            .ok_or(ErrorModelError::InvalidOutputEquation(outeq))
    }

    /// Get the error model for a resolved output equation index, rejecting
    /// slots that have no model.
    #[inline]
    fn defined_model(&self, outeq: usize) -> Result<&AssayErrorModel, ErrorModelError> {
        match self.0.get(outeq) {
            None => Err(ErrorModelError::InvalidOutputEquation(outeq)),
            Some(AssayErrorModel::None) => Err(ErrorModelError::NoneErrorModel(outeq)),
            Some(model) => Ok(model),
        }
    }

    /// Returns the error polynomial for a resolved output equation index.
    pub fn errorpoly(&self, outeq: usize) -> Result<ErrorPoly, ErrorModelError> {
        self.defined_model(outeq)?.errorpoly()
    }

    /// Returns the factor value for a resolved output equation index.
    pub fn factor(&self, outeq: usize) -> Result<f64, ErrorModelError> {
        self.defined_model(outeq)?.factor()
    }

    /// Computes the standard deviation (sigma) for a prediction.
    ///
    /// This always uses the **observation** value to compute sigma, which is
    /// appropriate for non-parametric algorithms (NPAG, NPOD). For parametric
    /// algorithms (SAEM, FOCE), use [`crate::ResidualErrorModels`] instead.
    #[inline]
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.defined_model(prediction.outeq)?.sigma(prediction)
    }

    /// Computes the variance for a prediction.
    #[inline]
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.defined_model(prediction.outeq)?.variance(prediction)
    }

    /// Computes the standard deviation (sigma) for a raw value.
    pub fn sigma_from_value(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.defined_model(outeq)?.sigma_from_value(value)
    }

    /// Computes the variance for a raw value.
    pub fn variance_from_value(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.defined_model(outeq)?.variance_from_value(value)
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
    #[error("The output equation index {0} is invalid")]
    InvalidOutputEquation(usize),
    #[error("The output label `{0}` is not declared in this error model context")]
    UnknownOutputLabel(String),
    #[error("The output label `{0}` already exists in this assay error model specification")]
    ExistingOutputLabel(String),
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

    fn additive(c0: f64, lambda: f64) -> AssayErrorModel {
        AssayErrorModel::additive(ErrorPoly::new(c0, 0.0, 0.0, 0.0), lambda)
    }

    fn proportional(c0: f64, gamma: f64) -> AssayErrorModel {
        AssayErrorModel::proportional(ErrorPoly::new(c0, 0.0, 0.0, 0.0), gamma)
    }

    #[test]
    fn test_additive_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = additive(1.0, 5.0);
        assert_eq!(model.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_proportional_error_model() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);
        let model = proportional(1.0, 2.0);
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
        let model = additive(1.0, 5.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), (26.0_f64).sqrt());

        let model = proportional(1.0, 2.0);
        assert_eq!(model.sigma_from_value(20.0).unwrap(), 2.0);
    }

    #[test]
    fn test_error_models_new() {
        let models = AssayErrorModels::new();
        assert_eq!(models.len(), 0);
        assert!(models.is_empty());
    }

    #[test]
    fn test_error_models_default() {
        let models = AssayErrorModels::default();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_error_models_add_single() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        assert_eq!(models.len(), 1);
        assert!(models.contains("cp"));
    }

    #[test]
    fn test_error_models_add_multiple() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("effect", proportional(2.0, 3.0))
            .unwrap();

        assert_eq!(models.len(), 2);
        assert_eq!(
            models
                .labels()
                .map(OutputLabel::to_string)
                .collect::<Vec<_>>(),
            vec!["cp".to_string(), "effect".to_string()]
        );
    }

    /// Regression: `AssayErrorModel::None` is a public variant, and the old
    /// dense storage reused the first `None` slot for the next labelled model.
    /// Two labels then aliased onto one slot and `len()` reported 1.
    #[test]
    fn test_none_model_does_not_alias_another_label() {
        let models = AssayErrorModels::new()
            .add("a", AssayErrorModel::None)
            .unwrap()
            .add("b", additive(1.0, 5.0))
            .unwrap();

        assert_eq!(models.len(), 2);
        assert_eq!(models.error_model("a").unwrap(), &AssayErrorModel::None);
        assert_eq!(models.error_model("b").unwrap().factor().unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_unknown_label_is_reported() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();

        match models.factor("effect") {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "effect"),
            other => panic!("Expected UnknownOutputLabel error, got {other:?}"),
        }
    }

    #[test]
    fn test_error_models_duplicate_label_fails() {
        let result = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("cp", proportional(2.0, 3.0));

        match result {
            Err(ErrorModelError::ExistingOutputLabel(label)) => assert_eq!(label, "cp"),
            _ => panic!("Expected ExistingOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_add_duplicate_numeric_label_fails() {
        let result = AssayErrorModels::new()
            .add(0, additive(1.0, 5.0))
            .unwrap()
            .add(0, proportional(2.0, 3.0));

        match result {
            Err(ErrorModelError::ExistingOutputLabel(label)) => assert_eq!(label, "0"),
            _ => panic!("Expected ExistingOutputLabel error"),
        }
    }

    #[test]
    fn test_error_models_factor() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        assert_eq!(models.factor("cp").unwrap(), 5.0);
    }

    #[test]
    fn test_error_models_set_factor() {
        let mut models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();

        assert_eq!(models.factor("cp").unwrap(), 5.0);
        models.set_factor("cp", 10.0).unwrap();
        assert_eq!(models.factor("cp").unwrap(), 10.0);
    }

    #[test]
    fn test_error_models_set_factor_unknown_label() {
        let mut models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();

        match models.set_factor("effect", 10.0) {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "effect"),
            other => panic!("Expected UnknownOutputLabel error, got {other:?}"),
        }
    }

    #[test]
    fn test_error_models_errorpoly() {
        let poly = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let models = AssayErrorModels::new()
            .add("cp", AssayErrorModel::additive(poly, 5.0))
            .unwrap();

        assert_eq!(
            models.errorpoly("cp").unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_error_models_set_errorpoly() {
        let poly1 = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        let poly2 = ErrorPoly::new(5.0, 6.0, 7.0, 8.0);
        let mut models = AssayErrorModels::new()
            .add("cp", AssayErrorModel::additive(poly1, 5.0))
            .unwrap();

        assert_eq!(
            models.errorpoly("cp").unwrap().coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
        models.set_errorpoly("cp", poly2).unwrap();
        assert_eq!(
            models.errorpoly("cp").unwrap().coefficients(),
            (5.0, 6.0, 7.0, 8.0)
        );
    }

    #[test]
    fn test_error_models_set_errorpoly_unknown_label() {
        let mut models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();

        match models.set_errorpoly("effect", ErrorPoly::new(5.0, 6.0, 7.0, 8.0)) {
            Err(ErrorModelError::UnknownOutputLabel(label)) => assert_eq!(label, "effect"),
            other => panic!("Expected UnknownOutputLabel error, got {other:?}"),
        }
    }

    #[test]
    fn test_error_models_sigma_from_value() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        assert_eq!(
            models.sigma_from_value("cp", 20.0).unwrap(),
            (26.0_f64).sqrt()
        );
    }

    #[test]
    fn test_error_models_variance_from_value() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(
            models.variance_from_value("cp", 20.0).unwrap(),
            expected_sigma.powi(2)
        );
    }

    #[test]
    fn test_error_models_multiple_outputs() {
        let models = AssayErrorModels::new()
            .add(
                "cp",
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5),
            )
            .unwrap()
            .add(
                "effect",
                AssayErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.1),
            )
            .unwrap();

        assert_eq!(models.len(), 2);
        assert_eq!(models.factor("cp").unwrap(), 0.5);
        assert_eq!(models.factor("effect").unwrap(), 0.1);
        assert_eq!(
            models.errorpoly("cp").unwrap().coefficients(),
            (1.0, 0.1, 0.0, 0.0)
        );
        assert_eq!(
            models.errorpoly("effect").unwrap().coefficients(),
            (0.0, 0.05, 0.0, 0.0)
        );
        assert!(models.is_additive("cp"));
        assert!(models.is_proportional("effect"));
        assert!(!models.is_additive("missing"));
    }

    #[test]
    fn test_error_models_hash_consistency() {
        let models1 = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("effect", proportional(2.0, 3.0))
            .unwrap();

        let models2 = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("effect", proportional(2.0, 3.0))
            .unwrap();

        assert_eq!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_error_models_hash_order_independence() {
        let models1 = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("effect", proportional(2.0, 3.0))
            .unwrap();

        let models2 = AssayErrorModels::new()
            .add("effect", proportional(2.0, 3.0))
            .unwrap()
            .add("cp", additive(1.0, 5.0))
            .unwrap();

        assert_eq!(models1.hash(), models2.hash());
    }

    /// Regression: the old `hash()` skipped every labelled slot in its second
    /// loop, so a label rename could go unnoticed by the bound-model cache.
    #[test]
    fn test_error_models_hash_depends_on_label() {
        let a = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        let b = AssayErrorModels::new()
            .add("effect", additive(1.0, 5.0))
            .unwrap();

        assert_ne!(a.hash(), b.hash());
    }

    #[test]
    fn error_model_hash_deterministic() {
        let models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        assert_eq!(models.hash(), models.hash());
    }

    #[test]
    fn error_model_hash_differs_on_value() {
        let a = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        let b = AssayErrorModels::new()
            .add("cp", additive(1.0, 10.0))
            .unwrap();
        assert_ne!(a.hash(), b.hash());
    }

    #[test]
    fn error_model_hash_differs_on_type() {
        let a = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        let b = AssayErrorModels::new()
            .add("cp", proportional(1.0, 5.0))
            .unwrap();
        assert_ne!(a.hash(), b.hash());
    }

    #[test]
    fn test_hash_includes_fixed_state() {
        let models1 = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap();
        let models2 = AssayErrorModels::new()
            .add(
                "cp",
                AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap();

        assert_ne!(models1.hash(), models2.hash());
    }

    #[test]
    fn test_factor_param_new_constructors() {
        let additive = additive(1.0, 5.0);
        assert_eq!(additive.factor().unwrap(), 5.0);
        assert!(!additive.is_factor_fixed().unwrap());

        let proportional = proportional(1.0, 2.0);
        assert_eq!(proportional.factor().unwrap(), 2.0);
        assert!(!proportional.is_factor_fixed().unwrap());

        let additive_fixed =
            AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);
        assert_eq!(additive_fixed.factor().unwrap(), 5.0);
        assert!(additive_fixed.is_factor_fixed().unwrap());

        let proportional_fixed =
            AssayErrorModel::proportional_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0);
        assert_eq!(proportional_fixed.factor().unwrap(), 2.0);
        assert!(proportional_fixed.is_factor_fixed().unwrap());

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
        let mut model = additive(1.0, 5.0);

        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(!model.is_factor_fixed().unwrap());

        model.fix_factor();
        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(model.is_factor_fixed().unwrap());

        model.unfix_factor();
        assert_eq!(model.factor().unwrap(), 5.0);
        assert!(!model.is_factor_fixed().unwrap());

        model.set_factor_param(Factor::Fixed(10.0));
        assert_eq!(model.factor().unwrap(), 10.0);
        assert!(model.is_factor_fixed().unwrap());

        let param = model.factor_param().unwrap();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_fixed());
    }

    #[test]
    fn test_factor_param_functionality() {
        let mut param = Factor::Variable(5.0);

        assert_eq!(param.value(), 5.0);
        assert!(param.is_variable());
        assert!(!param.is_fixed());

        param.set_value(10.0);
        assert_eq!(param.value(), 10.0);
        assert!(param.is_variable());

        param.make_fixed();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_fixed());
        assert!(!param.is_variable());

        param.make_variable();
        assert_eq!(param.value(), 10.0);
        assert!(param.is_variable());
        assert!(!param.is_fixed());
    }

    #[test]
    fn test_error_models_factor_param_methods() {
        let mut models = AssayErrorModels::new()
            .add(
                "cp",
                AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0),
            )
            .unwrap()
            .add("effect", proportional(1.0, 2.0))
            .unwrap();

        let cp = models.factor_param("cp").unwrap();
        assert_eq!(cp.value(), 5.0);
        assert!(cp.is_fixed());

        let effect = models.factor_param("effect").unwrap();
        assert_eq!(effect.value(), 2.0);
        assert!(effect.is_variable());

        assert!(models.is_factor_fixed("cp").unwrap());
        assert!(!models.is_factor_fixed("effect").unwrap());

        models.fix_factor("effect").unwrap();
        assert!(models.is_factor_fixed("effect").unwrap());

        models.unfix_factor("cp").unwrap();
        assert!(!models.is_factor_fixed("cp").unwrap());

        models.set_factor_param("cp", Factor::Fixed(10.0)).unwrap();
        assert_eq!(models.factor("cp").unwrap(), 10.0);
        assert!(models.is_factor_fixed("cp").unwrap());
    }

    #[test]
    fn test_fixed_parameters_in_calculations() {
        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let model_variable = additive(1.0, 5.0);
        let model_fixed = AssayErrorModel::additive_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 5.0);

        let sigma_variable = model_variable.sigma(&prediction).unwrap();
        let sigma_fixed = model_fixed.sigma(&prediction).unwrap();

        assert_eq!(sigma_variable, sigma_fixed);
        assert_eq!(sigma_variable, (26.0_f64).sqrt());

        let sigma_variable_val = model_variable.sigma_from_value(20.0).unwrap();
        let sigma_fixed_val = model_fixed.sigma_from_value(20.0).unwrap();

        assert_eq!(sigma_variable_val, sigma_fixed_val);
        assert_eq!(sigma_variable_val, (26.0_f64).sqrt());
    }

    /// The NPAG gamma/lambda loop: collect the optimizable labels, then update
    /// each one through the label-keyed accessors.
    #[test]
    fn test_optimizer_loop_over_labels() {
        let mut models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add(
                "effect",
                AssayErrorModel::proportional_fixed(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0),
            )
            .unwrap();

        let optimizable: Vec<OutputLabel> = models
            .iter()
            .filter(|(_, model)| model.optimize())
            .map(|(label, _)| label.clone())
            .collect();
        assert_eq!(optimizable, vec![OutputLabel::new("cp")]);

        for label in &optimizable {
            let updated = models.factor(label).unwrap() * 2.0;
            models.set_factor(label, updated).unwrap();
        }

        assert_eq!(models.factor("cp").unwrap(), 10.0);
        assert_eq!(models.factor("effect").unwrap(), 2.0);
    }

    #[test]
    fn test_error_models_iteration() {
        let mut models = AssayErrorModels::new()
            .add("cp", additive(1.0, 5.0))
            .unwrap()
            .add("effect", proportional(1.0, 2.0))
            .unwrap();

        assert_eq!(models.iter().count(), models.len());
        assert_eq!(models.iter_mut().count(), 2);

        for (label, model) in models.iter_mut() {
            match label.as_str() {
                "cp" => model.set_factor(10.0),
                "effect" => model.set_factor(4.0),
                other => panic!("Unexpected label: {other}"),
            }
        }

        assert_eq!(models.factor("cp").unwrap(), 10.0);
        assert_eq!(models.factor("effect").unwrap(), 4.0);

        for (_label, model) in models.iter_mut() {
            model.fix_factor();
        }
        assert!(models.is_factor_fixed("cp").unwrap());
        assert!(models.is_factor_fixed("effect").unwrap());

        let collected: Vec<(OutputLabel, AssayErrorModel)> = models.into_iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, OutputLabel::new("cp"));
        assert_eq!(collected[1].0, OutputLabel::new("effect"));
        assert_eq!(collected[0].1.factor().unwrap(), 10.0);
        assert_eq!(collected[1].1.factor().unwrap(), 4.0);
    }

    // ───────────────────────── DenseAssayErrorModels ─────────────────────────

    #[test]
    fn test_dense_error_models_sigma() {
        let models = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        assert_eq!(models.sigma(&prediction).unwrap(), (26.0_f64).sqrt());
    }

    #[test]
    fn test_dense_error_models_variance() {
        let models = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);

        let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        let expected_sigma = (26.0_f64).sqrt();
        assert_eq!(
            models.variance(&prediction).unwrap(),
            expected_sigma.powi(2)
        );
    }

    #[test]
    fn test_dense_error_models_different_outeqs() {
        let models =
            DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0), proportional(1.0, 2.0)]);

        let obs1 = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
        let pred1 = obs1.to_prediction(10.0, vec![]);
        assert_eq!(models.sigma(&pred1).unwrap(), (26.0_f64).sqrt());

        let obs2 = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None);
        let pred2 = obs2.to_prediction(10.0, vec![]);
        assert_eq!(models.sigma(&pred2).unwrap(), 2.0);
    }

    #[test]
    fn test_dense_error_models_invalid_outeq() {
        let models = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        match models.sigma(&prediction) {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            other => panic!("Expected InvalidOutputEquation error, got {other:?}"),
        }
        match models.factor(1) {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            other => panic!("Expected InvalidOutputEquation error, got {other:?}"),
        }
        match models.errorpoly(1) {
            Err(ErrorModelError::InvalidOutputEquation(outeq)) => assert_eq!(outeq, 1),
            other => panic!("Expected InvalidOutputEquation error, got {other:?}"),
        }
    }

    /// Regression: a bound set is sized by the equation's output count, so an
    /// output with no model reports `NoneErrorModel` rather than the misleading
    /// `InvalidOutputEquation` the short dense vector used to produce.
    #[test]
    fn test_dense_error_models_unset_slot_is_none_not_invalid() {
        let models =
            DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0), AssayErrorModel::None]);
        assert_eq!(models.len(), 2);

        let observation = Observation::new(0.0, Some(20.0), 1, None, 0, Censor::None);
        let prediction = observation.to_prediction(10.0, vec![]);

        match models.sigma(&prediction) {
            Err(ErrorModelError::NoneErrorModel(outeq)) => assert_eq!(outeq, 1),
            other => panic!("Expected NoneErrorModel error, got {other:?}"),
        }
    }

    #[test]
    fn test_dense_error_models_value_helpers() {
        let models = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);
        let expected_sigma = (26.0_f64).sqrt();

        assert_eq!(models.sigma_from_value(0, 20.0).unwrap(), expected_sigma);
        assert_eq!(
            models.variance_from_value(0, 20.0).unwrap(),
            expected_sigma.powi(2)
        );
        assert_eq!(models.error_model(0).unwrap().factor().unwrap(), 5.0);
        assert!(DenseAssayErrorModels::default().is_empty());
    }

    #[test]
    fn test_dense_error_models_hash() {
        let a = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);
        let b = DenseAssayErrorModels::from_dense(vec![additive(1.0, 5.0)]);
        let c = DenseAssayErrorModels::from_dense(vec![AssayErrorModel::None, additive(1.0, 5.0)]);

        assert_eq!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
    }
}
