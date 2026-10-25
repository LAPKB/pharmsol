use std::hash::{Hash, Hasher};

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

/// A plain value is a variable factor.
impl From<f64> for Factor {
    fn from(value: f64) -> Self {
        Factor::Variable(value)
    }
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

/// Hash an assay error model, including whether its factor is fixed.
pub(crate) fn hash_model<H: Hasher>(hasher: &mut H, model: &AssayErrorModel) {
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

/// Assay error: sigma computed from the **observed value**.
///
/// The assay's own standard deviation at an observed value `y` comes from
/// the [`ErrorPoly`]:
///
/// ```text
/// α(y) = c0 + c1·y + c2·y² + c3·y³
/// ```
///
/// The variant says how the extra, unexplained noise of the [`Factor`] is
/// combined with it:
///
/// - [`AssayErrorModel::Additive`] (λ): `σ = sqrt(α² + λ²)`, adding noise
///   independent of the assay's.
/// - [`AssayErrorModel::Proportional`] (γ): `σ = γ·α`, scaling the assay's
///   standard deviation.
///
/// Note that "proportional" here scales α, unlike
/// [`ResidualErrorModel::proportional`](crate::ResidualErrorModel::proportional),
/// which scales the prediction. An observation that carries its own
/// polynomial (from the data set) uses it in place of the model's.
///
/// Non-parametric algorithms (NPAG, NPOD) estimate λ or γ unless the factor
/// is [`Factor::Fixed`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssayErrorModel {
    /// `σ = sqrt(α² + λ²)`.
    Additive {
        /// Lambda: standard deviation of the noise added to the assay's.
        lambda: Factor,
        /// The assay error polynomial α.
        poly: ErrorPoly,
    },

    /// `σ = γ·α`.
    Proportional {
        /// Gamma: the factor the assay's standard deviation is scaled by.
        gamma: Factor,
        /// The assay error polynomial α.
        poly: ErrorPoly,
    },
}

impl AssayErrorModel {
    /// Additive assay error, `σ = sqrt(α² + λ²)`.
    ///
    /// A plain `f64` lambda is estimated; pass [`Factor::Fixed`] to fix it.
    ///
    /// ```rust
    /// use pharmsol::prelude::*;
    /// use pharmsol::data::error_model::Factor;
    ///
    /// let poly = ErrorPoly::new(0.0, 0.1, 0.0, 0.0);
    /// assert!(AssayErrorModel::additive(poly, 0.5).optimize());
    /// assert!(!AssayErrorModel::additive(poly, Factor::Fixed(0.5)).optimize());
    /// ```
    pub fn additive(poly: ErrorPoly, lambda: impl Into<Factor>) -> Self {
        Self::Additive {
            lambda: lambda.into(),
            poly,
        }
    }

    /// Proportional assay error, `σ = γ·α`.
    ///
    /// A plain `f64` gamma is estimated; pass [`Factor::Fixed`] to fix it.
    pub fn proportional(poly: ErrorPoly, gamma: impl Into<Factor>) -> Self {
        Self::Proportional {
            gamma: gamma.into(),
            poly,
        }
    }

    fn factor_ref(&self) -> &Factor {
        match self {
            Self::Additive { lambda, .. } => lambda,
            Self::Proportional { gamma, .. } => gamma,
        }
    }

    fn factor_mut(&mut self) -> &mut Factor {
        match self {
            Self::Additive { lambda, .. } => lambda,
            Self::Proportional { gamma, .. } => gamma,
        }
    }

    /// The assay error polynomial.
    pub fn errorpoly(&self) -> ErrorPoly {
        match self {
            Self::Additive { poly, .. } | Self::Proportional { poly, .. } => *poly,
        }
    }

    /// Replace the assay error polynomial.
    pub fn set_errorpoly(&mut self, new_poly: ErrorPoly) {
        match self {
            Self::Additive { poly, .. } | Self::Proportional { poly, .. } => *poly = new_poly,
        }
    }

    /// The factor's value (lambda or gamma).
    pub fn factor(&self) -> f64 {
        self.factor_ref().value()
    }

    /// Set the factor's value, keeping whether it is fixed.
    pub fn set_factor(&mut self, factor: f64) {
        self.factor_mut().set_value(factor);
    }

    /// The factor, with whether it is fixed.
    pub fn factor_param(&self) -> Factor {
        *self.factor_ref()
    }

    /// Replace the factor, including whether it is fixed.
    pub fn set_factor_param(&mut self, param: Factor) {
        *self.factor_mut() = param;
    }

    /// Whether the factor is fixed.
    pub fn is_factor_fixed(&self) -> bool {
        self.factor_ref().is_fixed()
    }

    /// Fix the factor at its current value.
    pub fn fix_factor(&mut self) {
        self.factor_mut().make_fixed();
    }

    /// Make the factor variable at its current value.
    pub fn unfix_factor(&mut self) {
        self.factor_mut().make_variable();
    }

    /// Whether this is [`AssayErrorModel::Proportional`].
    pub fn is_proportional(&self) -> bool {
        matches!(self, Self::Proportional { .. })
    }

    /// Whether this is [`AssayErrorModel::Additive`].
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Additive { .. })
    }

    /// Whether the factor should be estimated, i.e. it is not fixed.
    pub fn optimize(&self) -> bool {
        self.factor_ref().is_variable()
    }

    /// Sigma for a prediction's observation, using the observation's own
    /// polynomial if it carries one.
    ///
    /// Errors with [`ErrorModelError::MissingObservation`] if the prediction
    /// has no observation.
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
        let (c0, c1, c2, c3) = errorpoly.unwrap_or(self.errorpoly()).coefficients();
        let alpha = c0 + c1 * observation + c2 * observation.powi(2) + c3 * observation.powi(3);
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

    /// Variance for a prediction's observation: [`Self::sigma`] squared.
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        Ok(self.sigma(prediction)?.powi(2))
    }

    /// Sigma for an observed value, using the model's polynomial.
    ///
    /// Unlike [`Self::sigma`], a zero sigma is an error.
    pub fn sigma_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        match self.sigma_for(value, None)? {
            0.0 => Err(ErrorModelError::ZeroSigma),
            sigma => Ok(sigma),
        }
    }

    /// Variance for an observed value: [`Self::sigma_from_value`] squared.
    pub fn variance_from_value(&self, value: f64) -> Result<f64, ErrorModelError> {
        Ok(self.sigma_from_value(value)?.powi(2))
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
    #[error("The prediction does not have an observation associated with it")]
    MissingObservation,
    #[error("The error model for output `{0}` takes sigma from the prediction, not from an assay error polynomial")]
    NotAssayError(String),
    #[error("A log-transformed observation must be positive, got {0}")]
    NonPositiveObservation(f64),
    #[error("Invalid distribution: {0}")]
    InvalidFamily(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::output_model::OutputModels;
    use crate::{Censor, Observation};

    fn poly(c0: f64) -> ErrorPoly {
        ErrorPoly::new(c0, 0.0, 0.0, 0.0)
    }

    fn prediction(observation: Option<f64>, errorpoly: Option<ErrorPoly>) -> Prediction {
        Observation::new(0.0, observation, 0, errorpoly, 0, Censor::None)
            .to_prediction(10.0, vec![])
    }

    fn hash(model: &AssayErrorModel) -> u64 {
        let mut hasher = ahash::AHasher::default();
        hash_model(&mut hasher, model);
        hasher.finish()
    }

    #[test]
    fn additive_and_proportional_sigma() {
        let observed = prediction(Some(20.0), None);
        let additive = AssayErrorModel::additive(poly(1.0), 5.0);
        assert_eq!(additive.sigma(&observed).unwrap(), 26.0_f64.sqrt());
        assert!((additive.variance(&observed).unwrap() - 26.0).abs() < 1e-12);

        let proportional = AssayErrorModel::proportional(poly(1.0), 2.0);
        assert_eq!(proportional.sigma(&observed).unwrap(), 2.0);

        // The polynomial is evaluated at the observation, not the prediction.
        let linear = AssayErrorModel::proportional(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 1.0);
        assert!((linear.sigma(&observed).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn an_observation_polynomial_overrides_the_model_polynomial() {
        let model = AssayErrorModel::proportional(poly(1.0), 2.0);
        assert_eq!(
            model
                .sigma(&prediction(Some(20.0), Some(poly(3.0))))
                .unwrap(),
            6.0
        );
    }

    #[test]
    fn sigma_needs_an_observation() {
        let model = AssayErrorModel::additive(poly(1.0), 5.0);
        assert!(matches!(
            model.sigma(&prediction(None, None)),
            Err(ErrorModelError::MissingObservation)
        ));
    }

    #[test]
    fn invalid_sigmas_are_errors() {
        let negative = AssayErrorModel::proportional(poly(1.0), -1.0);
        assert!(matches!(
            negative.sigma(&prediction(Some(1.0), None)),
            Err(ErrorModelError::NegativeSigma)
        ));
        let infinite = AssayErrorModel::proportional(poly(f64::INFINITY), 1.0);
        assert!(matches!(
            infinite.sigma(&prediction(Some(1.0), None)),
            Err(ErrorModelError::NonFiniteSigma)
        ));
        let zero = AssayErrorModel::proportional(poly(0.0), 1.0);
        assert_eq!(zero.sigma(&prediction(Some(1.0), None)).unwrap(), 0.0);
        assert!(matches!(
            zero.sigma_from_value(1.0),
            Err(ErrorModelError::ZeroSigma)
        ));
    }

    #[test]
    fn sigma_from_value_uses_the_model_polynomial() {
        let additive = AssayErrorModel::additive(poly(1.0), 5.0);
        assert_eq!(additive.sigma_from_value(20.0).unwrap(), 26.0_f64.sqrt());
        assert!((additive.variance_from_value(20.0).unwrap() - 26.0).abs() < 1e-12);
        let proportional = AssayErrorModel::proportional(poly(1.0), 2.0);
        assert_eq!(proportional.sigma_from_value(20.0).unwrap(), 2.0);
    }

    #[test]
    fn errorpoly_can_be_read_and_replaced() {
        let mut model = AssayErrorModel::additive(ErrorPoly::new(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(model.errorpoly().coefficients(), (1.0, 2.0, 3.0, 4.0));
        model.set_errorpoly(ErrorPoly::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(model.errorpoly().coefficients(), (5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn a_plain_factor_is_variable_and_factor_fixed_is_fixed() {
        for model in [
            AssayErrorModel::additive(poly(1.0), 5.0),
            AssayErrorModel::proportional(poly(1.0), 5.0),
        ] {
            assert_eq!(model.factor(), 5.0);
            assert!(!model.is_factor_fixed());
            assert!(model.optimize());
        }
        for model in [
            AssayErrorModel::additive(poly(1.0), Factor::Fixed(5.0)),
            AssayErrorModel::proportional(poly(1.0), Factor::Fixed(5.0)),
        ] {
            assert_eq!(model.factor(), 5.0);
            assert!(model.is_factor_fixed());
            assert!(!model.optimize());
        }
        assert!(AssayErrorModel::additive(poly(1.0), 1.0).is_additive());
        assert!(AssayErrorModel::proportional(poly(1.0), 1.0).is_proportional());
    }

    #[test]
    fn factor_value_and_state_change_independently() {
        let mut model = AssayErrorModel::proportional(poly(1.0), 2.0);
        model.set_factor(4.0);
        assert_eq!(model.factor_param(), Factor::Variable(4.0));

        model.fix_factor();
        assert_eq!(model.factor_param(), Factor::Fixed(4.0));
        model.set_factor(3.0);
        assert_eq!(model.factor_param(), Factor::Fixed(3.0));

        model.unfix_factor();
        assert_eq!(model.factor_param(), Factor::Variable(3.0));
        model.set_factor_param(Factor::Fixed(1.5));
        assert!(model.is_factor_fixed());
        assert_eq!(model.factor(), 1.5);
        // A fixed factor still enters sigma.
        assert_eq!(model.sigma(&prediction(Some(1.0), None)).unwrap(), 1.5);
    }

    #[test]
    fn factor_helpers() {
        let mut factor = Factor::Variable(1.0);
        factor.make_fixed();
        assert!(factor.is_fixed());
        factor.set_value(2.0);
        assert_eq!(factor, Factor::Fixed(2.0));
        factor.make_variable();
        assert!(factor.is_variable());
        factor.set_factor(&Factor::Fixed(3.0));
        assert_eq!(factor, Factor::Fixed(3.0));
        assert_eq!(Factor::from(0.5), Factor::Variable(0.5));
    }

    #[test]
    fn hash_covers_type_value_state_and_polynomial() {
        let base = AssayErrorModel::additive(poly(1.0), 5.0);
        assert_eq!(hash(&base), hash(&base.clone()));
        for other in [
            AssayErrorModel::proportional(poly(1.0), 5.0),
            AssayErrorModel::additive(poly(1.0), 10.0),
            AssayErrorModel::additive(poly(1.0), Factor::Fixed(5.0)),
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        ] {
            assert_ne!(hash(&base), hash(&other), "{other:?}");
        }
    }

    #[test]
    fn models_are_reached_by_exact_label() {
        let mut models = OutputModels::new()
            .add(1, AssayErrorModel::proportional(poly(1.0), 2.0))
            .unwrap();
        assert!(matches!(
            OutputModels::new()
                .add("cp", AssayErrorModel::additive(poly(1.0), 5.0))
                .unwrap()
                .add("cp", AssayErrorModel::proportional(poly(2.0), 3.0)),
            Err(ErrorModelError::ExistingOutputLabel(label)) if label == "cp"
        ));

        // `1` and `"1"` are the same label; `"01"` is a different one.
        assert!(models.contains(1) && models.contains("1"));
        assert!(!models.contains("01"));
        assert_eq!(models.get(1), models.get("1"));
        assert!(models.output_model("01").is_err());
        assert!(matches!(
            models.factor("01"),
            Err(ErrorModelError::UnknownOutputLabel(label)) if label == "01"
        ));

        // Changes through one spelling are seen through the other.
        models.set_factor(1, 3.0).unwrap();
        assert_eq!(models.factor("1").unwrap(), 3.0);

        // Iterating `&mut` reaches every model, as an optimizer loop would.
        for (_, model) in &mut models {
            model.assay_mut().unwrap().fix_factor();
        }
        assert!(models.get("1").unwrap().assay().unwrap().is_factor_fixed());
        assert_eq!((&models).into_iter().count(), 1);
        assert_eq!(models.into_iter().count(), 1);
    }
}
