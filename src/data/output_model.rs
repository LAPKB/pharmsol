//! Observation models, one per output.
//!
//! An [`OutputModel`] says how an output's observations are distributed around
//! the model's predictions, and [`OutputModels`] holds one per output label.
//! `OutputModels` is the [`ObservationModel`] every likelihood in pharmsol is
//! computed with.
//!
//! ```rust
//! use pharmsol::prelude::*;
//!
//! let models = OutputModels::new()
//!     // Assay error from the assay's polynomial [c0, c1], plus lambda = 0.5.
//!     .with("cp", OutputModel::assay([0.0, 0.1]).additive(0.5))
//!     // Heavy-tailed combined residual error.
//!     .with("effect", OutputModel::combined(0.5, 0.1).student_t(4.0))
//!     // Exponential error: constant sigma 0.2 on the log scale.
//!     .with("metabolite", OutputModel::exponential(0.2));
//! assert_eq!(models.len(), 3);
//! ```
//!
//! # Building a model
//!
//! Start from where sigma comes from:
//!
//! | Constructor | Sigma |
//! |---|---|
//! | [`OutputModel::assay`]`(poly)` | the assay's polynomial `α(y)` at the **observation** |
//! | `… .additive(λ)` | `sqrt(α² + λ²)` (the default, with λ = 0) |
//! | `… .multiplicative(γ)` | `γ·α` |
//! | [`OutputModel::constant`]`(a)` | `a` |
//! | [`OutputModel::proportional`]`(b)` | `b·|f|`, at the **prediction** |
//! | [`OutputModel::combined`]`(a, b)` | `sqrt(a² + b²·f²)` |
//! | [`OutputModel::exponential`]`(σ)` | `σ` on the log scale |
//!
//! Then optionally change the distribution with `.student_t(nu)` (normal by
//! default) or compare on the log scale with `.log()`. Assay factors are
//! estimated unless `.fixed()`.
//!
//! Building never fails. Each model is checked when it is used: an output
//! the equation does not declare, or invalid degrees of freedom, is an error
//! before anything is simulated.
//!
//! # The parts
//!
//! Every model is three independent choices, which [`OutputModel::scale`],
//! [`OutputModel::family`] and [`OutputModel::transform`] return:
//!
//! - [`Scale`]: where sigma comes from. [`Scale::Observation`] computes it
//!   from the observed value with an [`AssayErrorModel`] (assay error, the
//!   usual choice for non-parametric algorithms such as NPAG);
//!   [`Scale::Prediction`] computes it from the prediction with a
//!   [`ResidualErrorModel`] (residual error, the usual choice for parametric
//!   algorithms such as SAEM and FOCE).
//! - [`Family`]: [`Family::Normal`] or [`Family::StudentT`].
//! - [`Transform`]: [`Transform::Identity`] or [`Transform::Log`].
//!
//! Censored observations (BLOQ, ALOQ) use the family's CDF or survival
//! function, whatever the scale.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::data::error_model::{self, AssayErrorModel, ErrorModelError, ErrorPoly, Factor};
use crate::data::event::OutputLabel;
use crate::data::residual_error::{self, ResidualErrorModel};
use crate::simulator::likelihood::distributions;
use crate::simulator::likelihood::observation::finite;
use crate::simulator::likelihood::{ObservationModel, Prediction};
use crate::{Censor, PharmsolError};

/// Where an output's sigma comes from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Scale {
    /// Sigma from the **observed value**, through an assay error polynomial.
    Observation(AssayErrorModel),
    /// Sigma from the **prediction**.
    Prediction(ResidualErrorModel),
}

impl Scale {
    /// Sigma for one observation.
    ///
    /// `errorpoly` is the observation's own error polynomial, which takes
    /// precedence over the assay model's; residual error ignores it.
    #[inline]
    pub fn sigma(
        &self,
        observation: f64,
        prediction: f64,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, ErrorModelError> {
        match self {
            Scale::Observation(model) => model.sigma_for(observation, errorpoly),
            Scale::Prediction(model) => Ok(model.sigma(prediction)),
        }
    }

    /// The scale's parameters, in a fixed order, with whether each is fixed:
    ///
    /// - [`AssayErrorModel::Additive`]: `[λ]`; [`AssayErrorModel::Multiplicative`]: `[γ]`.
    /// - [`ResidualErrorModel::Constant`]: `[a]`; [`ResidualErrorModel::Proportional`]:
    ///   `[b]`; [`ResidualErrorModel::Combined`]: `[a, b]`. Residual
    ///   parameters are always [`Factor::Variable`].
    ///
    /// The assay polynomial is data about the assay, not a parameter.
    pub fn parameters(&self) -> Vec<Factor> {
        match self {
            Scale::Observation(model) => vec![model.factor_param()],
            Scale::Prediction(model) => match *model {
                ResidualErrorModel::Constant { a } => vec![Factor::Variable(a)],
                ResidualErrorModel::Proportional { b } => vec![Factor::Variable(b)],
                ResidualErrorModel::Combined { a, b } => {
                    vec![Factor::Variable(a), Factor::Variable(b)]
                }
            },
        }
    }

    /// Set the values of [`Self::parameters`], in the same order, keeping
    /// whether each is fixed.
    ///
    /// Errors with [`ErrorModelError::ParameterCount`] unless `values` has
    /// one value per parameter.
    pub fn set_parameters(&mut self, values: &[f64]) -> Result<(), ErrorModelError> {
        let expected = match self {
            Scale::Prediction(ResidualErrorModel::Combined { .. }) => 2,
            _ => 1,
        };
        if values.len() != expected {
            return Err(ErrorModelError::ParameterCount {
                expected,
                got: values.len(),
            });
        }
        match self {
            Scale::Observation(model) => model.set_factor(values[0]),
            Scale::Prediction(ResidualErrorModel::Constant { a }) => *a = values[0],
            Scale::Prediction(ResidualErrorModel::Proportional { b }) => *b = values[0],
            Scale::Prediction(ResidualErrorModel::Combined { a, b }) => {
                *a = values[0];
                *b = values[1];
            }
        }
        Ok(())
    }
}

impl From<AssayErrorModel> for Scale {
    fn from(model: AssayErrorModel) -> Self {
        Scale::Observation(model)
    }
}

impl From<ResidualErrorModel> for Scale {
    fn from(model: ResidualErrorModel) -> Self {
        Scale::Prediction(model)
    }
}

/// The distribution of observations around the prediction.
///
/// Every family is a location-scale family: the prediction is the location
/// and sigma (from the [`Scale`]) the scale parameter.
///
/// Sigma is the standard deviation only for [`Family::Normal`]. For
/// [`Family::StudentT`] the standard deviation is `σ·sqrt(ν / (ν − 2))` when
/// `ν > 2`, and undefined otherwise, so a t model's sigma is not comparable
/// with a normal model's.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub enum Family {
    /// Normal distribution; sigma is the standard deviation.
    #[default]
    Normal,
    /// Student's t-distribution with `nu` degrees of freedom: heavier tails
    /// than the normal, so outliers weigh less. Approaches the normal as
    /// `nu` grows. Sigma is the scale parameter, not the standard deviation.
    StudentT {
        /// Degrees of freedom, positive and finite. Checked when the model
        /// is used (see [`ObservationModel::validate`]), not when it is built.
        nu: f64,
    },
}

impl Family {
    /// Student's t-distribution with `nu` degrees of freedom.
    pub fn student_t(nu: f64) -> Self {
        Family::StudentT { nu }
    }

    /// Errors with [`ErrorModelError::InvalidFamily`] unless the family's
    /// parameters are valid.
    pub fn validate(&self) -> Result<(), ErrorModelError> {
        match *self {
            Family::StudentT { nu } if !(nu > 0.0 && nu.is_finite()) => {
                Err(ErrorModelError::InvalidFamily(format!(
                    "Student's t needs finite nu > 0, got {nu}"
                )))
            }
            _ => Ok(()),
        }
    }

    /// Log-density of `y` given location `f` and scale `sigma`.
    #[inline]
    pub fn log_pdf(&self, y: f64, f: f64, sigma: f64) -> Result<f64, ErrorModelError> {
        match *self {
            Family::Normal => Ok(distributions::lognormpdf(y, f, sigma)),
            Family::StudentT { nu } => {
                self.validate()?;
                distributions::log_t_pdf(y, f, sigma, nu)
            }
        }
    }

    /// Log of the CDF at `y` (the probability of a value at most `y`).
    #[inline]
    pub fn log_cdf(&self, y: f64, f: f64, sigma: f64) -> Result<f64, ErrorModelError> {
        match *self {
            Family::Normal => distributions::lognormcdf(y, f, sigma),
            Family::StudentT { nu } => {
                self.validate()?;
                distributions::log_t_cdf(y, f, sigma, nu)
            }
        }
    }

    /// Log of the survival function at `y` (the probability of a value
    /// above `y`).
    #[inline]
    pub fn log_sf(&self, y: f64, f: f64, sigma: f64) -> Result<f64, ErrorModelError> {
        match *self {
            Family::Normal => distributions::lognormccdf(y, f, sigma),
            Family::StudentT { nu } => {
                self.validate()?;
                distributions::log_t_sf(y, f, sigma, nu)
            }
        }
    }
}

/// The scale observation and prediction are compared on.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Transform {
    /// Compare them as they are.
    #[default]
    Identity,
    /// Compare their natural logarithms. Observations must be positive; a
    /// prediction that is not positive gets zero likelihood.
    ///
    /// The [`Scale`] still gives sigma in the units of the observation. It is
    /// carried to the log scale by the delta method, `σ_log = σ / x`, where
    /// `x` is what the scale computes sigma from: the observation for
    /// [`Scale::Observation`], the prediction for [`Scale::Prediction`]. So a
    /// proportional residual error `σ = b·f` becomes a constant `b` on the
    /// log scale (see [`OutputModel::exponential`]), and an assay polynomial
    /// becomes the assay's coefficient of variation.
    ///
    /// The density includes the Jacobian `-ln(y)`, so it is a density of the
    /// observation on its original scale, comparable with
    /// [`Transform::Identity`].
    Log,
}

/// The observation model for one output: a [`Scale`], a [`Family`] and a
/// [`Transform`].
///
/// Sigma is always computed from the untransformed observation or
/// prediction, in its units, then carried to the transformed scale (see
/// [`Transform::Log`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputModel {
    scale: Scale,
    #[serde(default)]
    family: Family,
    #[serde(default)]
    transform: Transform,
}

impl OutputModel {
    /// A normal, untransformed model with this scale.
    ///
    /// The named constructors ([`Self::assay`], [`Self::constant`],
    /// [`Self::proportional`], [`Self::combined`], [`Self::exponential`])
    /// cover the usual models.
    pub fn new(scale: impl Into<Scale>) -> Self {
        Self {
            scale: scale.into(),
            family: Family::Normal,
            transform: Transform::Identity,
        }
    }

    /// Assay error: sigma from the observed value, through the assay's
    /// error polynomial (lowest degree first, e.g. `[c0, c1]`).
    ///
    /// Returns the [`AssayErrorModel`], additive with λ = 0 until finished
    /// with [`AssayErrorModel::additive`] or
    /// [`AssayErrorModel::multiplicative`]; it converts into an
    /// `OutputModel` wherever one is expected.
    ///
    /// ```rust
    /// use pharmsol::prelude::*;
    ///
    /// let models = OutputModels::new()
    ///     .with("cp", OutputModel::assay([0.0, 0.1]).additive(0.5))
    ///     .with("pd", OutputModel::assay([0.01, 0.1]).multiplicative(1.5).fixed())
    ///     .with("met", OutputModel::assay([0.0, 0.1]).student_t(4.0));
    /// assert_eq!(models.len(), 3);
    /// ```
    pub fn assay(poly: impl Into<ErrorPoly>) -> AssayErrorModel {
        AssayErrorModel::new(poly)
    }

    /// Constant residual error, `σ = a`.
    pub fn constant(a: f64) -> Self {
        Self::new(ResidualErrorModel::constant(a))
    }

    /// Proportional residual error, `σ = b·|f|`.
    pub fn proportional(b: f64) -> Self {
        Self::new(ResidualErrorModel::proportional(b))
    }

    /// Combined residual error, `σ = sqrt(a² + b²·f²)`.
    pub fn combined(a: f64, b: f64) -> Self {
        Self::new(ResidualErrorModel::combined(a, b))
    }

    /// Exponential error: normal with constant `sigma` on the log scale.
    ///
    /// The same as `OutputModel::proportional(sigma).log()`: the delta method
    /// makes a proportional error a constant `sigma` on the log scale.
    pub fn exponential(sigma: f64) -> Self {
        Self::proportional(sigma).log()
    }

    /// Student's t-distributed observations with `nu` degrees of freedom,
    /// for heavy tails. `nu` must be positive and finite; that is checked
    /// when the model is used.
    pub fn student_t(self, nu: f64) -> Self {
        self.with_family(Family::student_t(nu))
    }

    /// Compare observation and prediction on the log scale; see
    /// [`Transform::Log`].
    pub fn log(self) -> Self {
        self.with_transform(Transform::Log)
    }

    /// Use another distribution.
    pub fn with_family(mut self, family: Family) -> Self {
        self.family = family;
        self
    }

    /// Compare observation and prediction on another scale.
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Where sigma comes from.
    pub fn scale(&self) -> &Scale {
        &self.scale
    }

    /// Mutable access to the scale, e.g. to update its parameters.
    pub fn scale_mut(&mut self) -> &mut Scale {
        &mut self.scale
    }

    /// The distribution.
    pub fn family(&self) -> Family {
        self.family
    }

    /// The transform.
    pub fn transform(&self) -> Transform {
        self.transform
    }

    /// The assay error model, if sigma comes from the observation.
    pub fn as_assay(&self) -> Option<&AssayErrorModel> {
        match &self.scale {
            Scale::Observation(model) => Some(model),
            Scale::Prediction(_) => None,
        }
    }

    /// Mutable [`Self::as_assay`].
    pub fn as_assay_mut(&mut self) -> Option<&mut AssayErrorModel> {
        match &mut self.scale {
            Scale::Observation(model) => Some(model),
            Scale::Prediction(_) => None,
        }
    }

    /// The residual error model, if sigma comes from the prediction.
    pub fn as_residual(&self) -> Option<&ResidualErrorModel> {
        match &self.scale {
            Scale::Prediction(model) => Some(model),
            Scale::Observation(_) => None,
        }
    }

    /// Mutable [`Self::as_residual`].
    pub fn as_residual_mut(&mut self) -> Option<&mut ResidualErrorModel> {
        match &mut self.scale {
            Scale::Prediction(model) => Some(model),
            Scale::Observation(_) => None,
        }
    }

    /// Sigma for a prediction and its observation, on the scale they are
    /// compared on (the log scale for [`Transform::Log`]).
    ///
    /// Errors with [`ErrorModelError::MissingObservation`] when sigma comes
    /// from the observation and there is none.
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        let observation = match (&self.scale, prediction.observation()) {
            (_, Some(observation)) => observation,
            (Scale::Prediction(_), None) => f64::NAN,
            (Scale::Observation(_), None) => return Err(ErrorModelError::MissingObservation),
        };
        self.transformed_sigma(observation, prediction.prediction(), prediction.errorpoly())
    }

    /// Sigma from the [`Scale`], carried to the transformed scale.
    #[inline]
    fn transformed_sigma(
        &self,
        observation: f64,
        prediction: f64,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, ErrorModelError> {
        let sigma = self.scale.sigma(observation, prediction, errorpoly)?;
        match self.transform {
            Transform::Identity => Ok(sigma),
            // Delta method: sd(ln x) ≈ sd(x) / x, at the value sigma came from.
            Transform::Log => match self.scale {
                Scale::Observation(_) if observation > 0.0 => Ok(sigma / observation),
                Scale::Observation(_) => Err(ErrorModelError::NonPositiveObservation(observation)),
                Scale::Prediction(_) if prediction > 0.0 => Ok(sigma / prediction),
                Scale::Prediction(_) => Err(ErrorModelError::NonPositivePrediction(prediction)),
            },
        }
    }

    /// Log-density of one observation given its prediction.
    ///
    /// Censored observations (`observation` is the quantification limit) use
    /// the family's CDF (BLOQ) or survival function (ALOQ).
    #[inline]
    pub fn log_density(
        &self,
        observation: f64,
        prediction: f64,
        censoring: Censor,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        let (y, f, jacobian) = match self.transform {
            Transform::Identity => (observation, prediction, 0.0),
            Transform::Log => {
                if observation.is_nan() || observation <= 0.0 {
                    return Err(ErrorModelError::NonPositiveObservation(observation).into());
                }
                if prediction.is_nan() || prediction <= 0.0 {
                    return Err(PharmsolError::NonFiniteLikelihood(f64::NEG_INFINITY));
                }
                let y = observation.ln();
                (y, prediction.ln(), -y)
            }
        };
        let sigma = self.transformed_sigma(observation, prediction, errorpoly)?;
        let log_density = match censoring {
            Censor::None => self.family.log_pdf(y, f, sigma)? + jacobian,
            Censor::BLOQ => self.family.log_cdf(y, f, sigma)?,
            Censor::ALOQ => self.family.log_sf(y, f, sigma)?,
        };
        finite(log_density)
    }

    fn hash_into<H: Hasher>(&self, hasher: &mut H) {
        match &self.scale {
            Scale::Observation(model) => {
                0u8.hash(hasher);
                error_model::hash_model(hasher, model);
            }
            Scale::Prediction(model) => {
                1u8.hash(hasher);
                residual_error::hash_model(hasher, model);
            }
        }
        match self.family {
            Family::Normal => 0u8.hash(hasher),
            Family::StudentT { nu } => {
                1u8.hash(hasher);
                nu.to_bits().hash(hasher);
            }
        }
        match self.transform {
            Transform::Identity => 0u8.hash(hasher),
            Transform::Log => 1u8.hash(hasher),
        }
    }
}

impl From<AssayErrorModel> for OutputModel {
    fn from(model: AssayErrorModel) -> Self {
        Self::new(model)
    }
}

impl From<ResidualErrorModel> for OutputModel {
    fn from(model: ResidualErrorModel) -> Self {
        Self::new(model)
    }
}

impl From<Scale> for OutputModel {
    fn from(scale: Scale) -> Self {
        Self::new(scale)
    }
}

/// One [`OutputModel`] per output label: the observation model likelihoods
/// are computed with.
///
/// Labels match an equation's declared output names exactly: `"1"` matches an
/// output declared as `1`, and `"cp"` an output declared as `cp`. Equations
/// without metadata declare no output names; their outputs are labeled by
/// position (`"0"`, `"1"`, ...), like the observation labels resolved against
/// them.
///
/// [`Self::parameters`] and [`Self::set_parameters`] reach an output's
/// error parameters by label, assay or residual; [`Self::factor`] and
/// [`Self::set_factor`] reach just the assay factor (λ or γ), as an NPAG
/// loop does. Anything else about an
/// output's model goes through [`Self::get`] or [`Self::get_mut`] and the
/// [`OutputModel`] accessors.
///
/// ```rust
/// # use pharmsol::prelude::*;
/// let mut models =
///     OutputModels::new().with("cp", OutputModel::assay([0.0, 0.05]).additive(0.5));
///
/// for (label, model) in models.iter_mut() {
///     if let Some(assay) = model.as_assay_mut().filter(|assay| assay.optimize()) {
///         assay.set_factor(assay.factor() * 2.0);
///     }
///     assert_eq!(label.as_str(), "cp");
/// }
/// assert_eq!(models.factor("cp")?, 1.0);
///
/// models.set_factor("cp", 0.25)?;
/// models.get_mut("cp").and_then(OutputModel::as_assay_mut).unwrap().fix_factor();
/// assert!(!models.get("cp").and_then(OutputModel::as_assay).unwrap().optimize());
/// # Ok::<(), pharmsol::data::error_model::ErrorModelError>(())
/// ```
#[derive(Serialize, Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(transparent)]
pub struct OutputModels {
    models: BTreeMap<OutputLabel, OutputModel>,
}

/// Tag mixed into [`OutputModels`] cache keys, so they never collide with
/// another observation model's.
const CACHE_TAG: u64 = 0x6f75_7470_7574_0001;

impl OutputModels {
    /// An empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model for an output label, replacing any model it had.
    ///
    /// Takes anything that converts into an [`OutputModel`], including an
    /// [`AssayErrorModel`] or a [`ResidualErrorModel`] on its own. Use
    /// [`Self::add`] to have a repeated label caught instead.
    pub fn with(mut self, outeq: impl ToString, model: impl Into<OutputModel>) -> Self {
        self.models.insert(OutputLabel::new(outeq), model.into());
        self
    }

    /// Add the model for an output label, like [`Self::with`] but refusing
    /// to replace one.
    ///
    /// # Errors
    /// [ErrorModelError::ExistingOutputLabel] if a model already exists for
    /// that label.
    pub fn add(
        mut self,
        outeq: impl ToString,
        model: impl Into<OutputModel>,
    ) -> Result<Self, ErrorModelError> {
        let label = OutputLabel::new(outeq);
        if self.models.contains_key(&label) {
            return Err(ErrorModelError::ExistingOutputLabel(label.to_string()));
        }
        self.models.insert(label, model.into());
        Ok(self)
    }

    /// The model for an output label, if one was added.
    pub fn get(&self, outeq: impl ToString) -> Option<&OutputModel> {
        self.models.get(outeq.to_string().as_str())
    }

    /// Mutable [`Self::get`].
    pub fn get_mut(&mut self, outeq: impl ToString) -> Option<&mut OutputModel> {
        self.models.get_mut(outeq.to_string().as_str())
    }

    /// Whether a model was added for this output label.
    pub fn contains(&self, outeq: impl ToString) -> bool {
        self.models.contains_key(outeq.to_string().as_str())
    }

    /// The model for an output label.
    ///
    /// # Errors
    /// [ErrorModelError::UnknownOutputLabel] if no model was added for that label.
    pub fn output_model(&self, outeq: impl ToString) -> Result<&OutputModel, ErrorModelError> {
        self.model_for(&outeq.to_string())
    }

    /// Looks the label up as a `&str`, so the likelihood path never allocates.
    #[inline]
    fn model_for(&self, label: &str) -> Result<&OutputModel, ErrorModelError> {
        self.models
            .get(label)
            .ok_or_else(|| ErrorModelError::UnknownOutputLabel(label.to_string()))
    }

    /// The assay model for `outeq`, which must exist and take sigma from
    /// the observation.
    fn assay(&self, outeq: impl ToString) -> Result<&AssayErrorModel, ErrorModelError> {
        let label = outeq.to_string();
        self.model_for(&label)?
            .as_assay()
            .ok_or(ErrorModelError::NotAssayError(label))
    }

    /// Mutable [`Self::assay`].
    fn assay_mut(&mut self, outeq: impl ToString) -> Result<&mut AssayErrorModel, ErrorModelError> {
        let label = outeq.to_string();
        match self.models.get_mut(label.as_str()) {
            None => Err(ErrorModelError::UnknownOutputLabel(label)),
            Some(model) => model
                .as_assay_mut()
                .ok_or(ErrorModelError::NotAssayError(label)),
        }
    }

    /// Iterate over `(label, model)` pairs, in label order.
    pub fn iter(&self) -> impl Iterator<Item = (&OutputLabel, &OutputModel)> {
        self.models.iter()
    }

    /// Iterate mutably over `(label, model)` pairs, in label order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&OutputLabel, &mut OutputModel)> {
        self.models.iter_mut()
    }

    /// Iterate over the output labels that have a model.
    pub fn labels(&self) -> impl Iterator<Item = &OutputLabel> {
        self.models.keys()
    }

    /// A hash of the labels and every model's parameters.
    pub fn hash(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        for (label, model) in &self.models {
            label.hash(&mut hasher);
            model.hash_into(&mut hasher);
        }
        hasher.finish()
    }

    /// Number of models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Sigma for a prediction, using the model of its output label.
    #[inline]
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        self.model_for(prediction.outeq().as_str())?
            .sigma(prediction)
    }

    /// Variance for a prediction, using the model of its output label.
    pub fn variance(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        Ok(self.sigma(prediction)?.powi(2))
    }

    /// The assay factor (lambda or gamma) of an output.
    ///
    /// # Errors
    /// [ErrorModelError::UnknownOutputLabel] for an output without a model,
    /// [ErrorModelError::NotAssayError] for one whose sigma comes from the
    /// prediction.
    pub fn factor(&self, outeq: impl ToString) -> Result<f64, ErrorModelError> {
        Ok(self.assay(outeq)?.factor())
    }

    /// Set the assay factor (lambda or gamma) of an output, keeping whether
    /// it is fixed. Errors as [`Self::factor`] does.
    pub fn set_factor(&mut self, outeq: impl ToString, factor: f64) -> Result<(), ErrorModelError> {
        self.assay_mut(outeq)?.set_factor(factor);
        Ok(())
    }

    /// The [`Scale::parameters`] of an output, assay or residual.
    ///
    /// Errors with [`ErrorModelError::UnknownOutputLabel`] if the output has
    /// no model.
    pub fn parameters(&self, outeq: impl ToString) -> Result<Vec<Factor>, ErrorModelError> {
        Ok(self.model_for(&outeq.to_string())?.scale().parameters())
    }

    /// Set the [`Scale::parameters`] of an output, assay or residual.
    ///
    /// Errors as [`Self::parameters`] and [`Scale::set_parameters`] do.
    pub fn set_parameters(
        &mut self,
        outeq: impl ToString,
        values: &[f64],
    ) -> Result<(), ErrorModelError> {
        let label = outeq.to_string();
        self.models
            .get_mut(label.as_str())
            .ok_or(ErrorModelError::UnknownOutputLabel(label))?
            .scale_mut()
            .set_parameters(values)
    }
}

impl ObservationModel for OutputModels {
    #[inline]
    fn log_density(
        &self,
        outeq: &OutputLabel,
        observation: f64,
        prediction: f64,
        censoring: Censor,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        self.model_for(outeq.as_str())?
            .log_density(observation, prediction, censoring, errorpoly)
    }

    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_> {
        Box::new(self.models.keys())
    }

    fn validate(&self) -> Result<(), PharmsolError> {
        for (label, model) in &self.models {
            model.family.validate().map_err(|error| match error {
                ErrorModelError::InvalidFamily(reason) => {
                    ErrorModelError::InvalidFamily(format!("output `{label}`: {reason}"))
                }
                other => other,
            })?;
        }
        Ok(())
    }

    fn cache_key(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        CACHE_TAG.hash(&mut hasher);
        self.hash().hash(&mut hasher);
        hasher.finish()
    }
}

/// Later entries replace earlier ones for the same label, as in
/// [`OutputModels::with`].
impl<L: ToString, M: Into<OutputModel>> FromIterator<(L, M)> for OutputModels {
    fn from_iter<I: IntoIterator<Item = (L, M)>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Self::new(), |models, (label, model)| {
                models.with(label, model)
            })
    }
}

/// ```rust
/// use pharmsol::prelude::*;
///
/// let models = OutputModels::from([
///     ("cp", OutputModel::assay([0.0, 0.1]).additive(0.5)),
///     ("pd", OutputModel::assay([0.0, 0.2]).into()),
/// ]);
/// assert_eq!(models.len(), 2);
/// ```
impl<L: ToString, M: Into<OutputModel>, const N: usize> From<[(L, M); N]> for OutputModels {
    fn from(entries: [(L, M); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl IntoIterator for OutputModels {
    type Item = (OutputLabel, OutputModel);
    type IntoIter = std::collections::btree_map::IntoIter<OutputLabel, OutputModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.into_iter()
    }
}

impl<'a> IntoIterator for &'a OutputModels {
    type Item = (&'a OutputLabel, &'a OutputModel);
    type IntoIter = std::collections::btree_map::Iter<'a, OutputLabel, OutputModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter()
    }
}

impl<'a> IntoIterator for &'a mut OutputModels {
    type Item = (&'a OutputLabel, &'a mut OutputModel);
    type IntoIter = std::collections::btree_map::IterMut<'a, OutputLabel, OutputModel>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Observation;

    fn constant(sigma: f64) -> OutputModel {
        OutputModel::constant(sigma)
    }

    fn density(model: &OutputModel, y: f64, f: f64) -> f64 {
        model.log_density(y, f, Censor::None, None).unwrap()
    }

    #[test]
    fn residual_normal_matches_the_closed_form() {
        let model = constant(1.0);
        let expected = -0.5 * (std::f64::consts::TAU.ln() + 1.0);
        assert!((density(&model, 1.0, 0.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn assay_normal_uses_the_observation_for_sigma() {
        let model = OutputModel::new(OutputModel::assay([0.0, 0.1]));
        // sigma = 0.1 * 20 = 2
        let expected = distributions::lognormpdf(20.0, 18.0, 2.0);
        assert_eq!(density(&model, 20.0, 18.0), expected);
        // The observation's own polynomial wins.
        let own = model
            .log_density(
                20.0,
                18.0,
                Censor::None,
                Some(ErrorPoly::new(1.0, 0.0, 0.0, 0.0)),
            )
            .unwrap();
        assert_eq!(own, distributions::lognormpdf(20.0, 18.0, 1.0));
    }

    #[test]
    fn student_t_has_heavier_tails() {
        let normal = constant(1.0);
        let t = constant(1.0).student_t(3.0);
        assert!(density(&t, 0.0, 0.0) < density(&normal, 0.0, 0.0));
        assert!(density(&t, 6.0, 0.0) > density(&normal, 6.0, 0.0));

        // Standard t with 1 degree of freedom is the Cauchy distribution.
        let cauchy = constant(1.0).student_t(1.0);
        let expected = -(std::f64::consts::PI * 2.0).ln();
        assert!((density(&cauchy, 1.0, 0.0) - expected).abs() < 1e-12);

        // Large nu approaches the normal.
        let wide = constant(1.0).student_t(1.0e7);
        assert!((density(&wide, 1.3, 0.0) - density(&normal, 1.3, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn invalid_degrees_of_freedom_are_rejected_at_use() {
        // Building never fails; scoring and validation do.
        for nu in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let model = constant(1.0).student_t(nu);
            assert!(model.log_density(1.0, 0.0, Censor::None, None).is_err());
            let models = OutputModels::new().with("cp", model);
            assert!(matches!(
                models.validate(),
                Err(PharmsolError::ErrorModelError(ErrorModelError::InvalidFamily(reason)))
                    if reason.starts_with("output `cp`")
            ));
        }
        let family: Family = serde_json::from_str(r#"{"StudentT":{"nu":4.0}}"#).unwrap();
        assert_eq!(family, Family::student_t(4.0));
    }

    #[test]
    fn censoring_uses_the_family_tails() {
        for family in [Family::Normal, Family::student_t(4.0)] {
            let model = constant(1.0).with_family(family);
            // At the prediction, half the mass is on either side.
            for censoring in [Censor::BLOQ, Censor::ALOQ] {
                let log_p = model.log_density(2.0, 2.0, censoring, None).unwrap();
                assert!(
                    (log_p - 0.5_f64.ln()).abs() < 1e-9,
                    "{family:?} {censoring:?}"
                );
            }
            // Below and above sum to one.
            let below = model.log_density(1.0, 2.0, Censor::BLOQ, None).unwrap();
            let above = model.log_density(1.0, 2.0, Censor::ALOQ, None).unwrap();
            assert!((below.exp() + above.exp() - 1.0).abs() < 1e-9, "{family:?}");
        }
    }

    #[test]
    fn residual_error_now_honors_censoring() {
        let model = constant(1.0);
        let censored = model.log_density(1.0, 2.0, Censor::BLOQ, None).unwrap();
        assert_eq!(censored, distributions::lognormcdf(1.0, 2.0, 1.0).unwrap());
    }

    #[test]
    fn log_transform_is_the_lognormal_density() {
        let model = OutputModel::exponential(0.3);
        let (y, f): (f64, f64) = (2.0, 1.5);
        // Lognormal density of y with log-mean ln(f) and log-sd 0.3.
        let expected = distributions::lognormpdf(y.ln(), f.ln(), 0.3) - y.ln();
        assert!((density(&model, y, f) - expected).abs() < 1e-12);

        // Integrates to one over y (trapezoid on a wide grid).
        let (lo, hi, n) = (1e-3, 20.0, 200_000);
        let h = (hi - lo) / n as f64;
        let total: f64 = (0..=n)
            .map(|i| {
                let y = lo + i as f64 * h;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                w * density(&model, y, f).exp()
            })
            .sum::<f64>()
            * h;
        assert!((total - 1.0).abs() < 1e-3, "{total}");
    }

    #[test]
    fn log_transform_carries_sigma_by_the_delta_method() {
        let (y, f): (f64, f64) = (20.0, 10.0);
        let expected =
            |sigma_log: f64| distributions::lognormpdf(y.ln(), f.ln(), sigma_log) - y.ln();

        // Residual: sigma is computed at the prediction, divided by it.
        let constant_log = constant(2.0).log();
        assert!((density(&constant_log, y, f) - expected(2.0 / f)).abs() < 1e-12);
        let combined = OutputModel::combined(3.0, 0.4).log();
        let sigma = (9.0 + 0.16 * f * f).sqrt();
        assert!((density(&combined, y, f) - expected(sigma / f)).abs() < 1e-12);

        // Assay: sigma is computed at the observation, divided by it, so a
        // purely linear polynomial is a constant coefficient of variation.
        let assay = OutputModel::new(OutputModel::assay([0.0, 0.1])).log();
        assert!((density(&assay, y, f) - expected(0.1)).abs() < 1e-12);

        // `sigma` reports the log-scale sigma too.
        let prediction =
            Observation::new(0.0, Some(y), 0, None, 0, Censor::None).to_prediction(f, vec![]);
        assert!((assay.sigma(&prediction).unwrap() - 0.1).abs() < 1e-12);
        assert!((OutputModel::exponential(0.3).sigma(&prediction).unwrap() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn log_transform_rejects_non_positive_values() {
        let model = OutputModel::exponential(0.3);
        assert!(matches!(
            model.log_density(0.0, 1.0, Censor::None, None),
            Err(PharmsolError::ErrorModelError(
                ErrorModelError::NonPositiveObservation(_)
            ))
        ));
        assert!(matches!(
            model.log_density(1.0, 0.0, Censor::None, None),
            Err(PharmsolError::NonFiniteLikelihood(_))
        ));
    }

    #[test]
    fn outputs_mix_scales_families_and_transforms() {
        let models = OutputModels::new()
            .with("cp", OutputModel::assay([1.0]))
            .with("effect", constant(0.5).student_t(3.0))
            .with("metabolite", OutputModel::exponential(0.2));

        let cp = OutputLabel::new("cp");
        let effect = OutputLabel::new("effect");
        assert_eq!(
            models
                .log_density(&cp, 3.0, 2.0, Censor::None, None)
                .unwrap(),
            distributions::lognormpdf(3.0, 2.0, 1.0)
        );
        assert_eq!(
            models
                .log_density(&effect, 3.0, 2.0, Censor::None, None)
                .unwrap(),
            distributions::log_t_pdf(3.0, 2.0, 0.5, 3.0).unwrap()
        );
        assert!(models
            .log_density(&OutputLabel::new("other"), 3.0, 2.0, Censor::None, None)
            .is_err());

        // Assay helpers reach assay outputs only.
        assert_eq!(models.factor("cp").unwrap(), 0.0);
        assert!(matches!(
            models.factor("effect"),
            Err(ErrorModelError::NotAssayError(label)) if label == "effect"
        ));
        assert!(matches!(
            models.factor("other"),
            Err(ErrorModelError::UnknownOutputLabel(_))
        ));
    }

    #[test]
    fn sigma_follows_each_scale() {
        let models = OutputModels::new()
            .with(0, OutputModel::assay([0.0, 0.1]))
            .with(1, ResidualErrorModel::proportional(0.2));
        let assay =
            Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None).to_prediction(10.0, vec![]);
        let residual =
            Observation::new(0.0, None, 1, None, 0, Censor::None).to_prediction(10.0, vec![]);
        assert!((models.sigma(&assay).unwrap() - 2.0).abs() < 1e-12);
        assert!((models.sigma(&residual).unwrap() - 2.0).abs() < 1e-12);
        let missing =
            Observation::new(0.0, None, 0, None, 0, Censor::None).to_prediction(10.0, vec![]);
        assert!(matches!(
            models.sigma(&missing),
            Err(ErrorModelError::MissingObservation)
        ));
    }

    #[test]
    fn building_reads_like_the_model() {
        // Polynomials from their leading coefficients.
        assert_eq!(ErrorPoly::from([0.5]), ErrorPoly::new(0.5, 0.0, 0.0, 0.0));
        assert_eq!(
            ErrorPoly::from([1.0, 2.0, 3.0, 4.0]),
            ErrorPoly::new(1.0, 2.0, 3.0, 4.0)
        );

        // An unfinished assay model is pure assay error, with lambda estimated.
        let assay = OutputModel::assay([0.0, 0.1]);
        assert_eq!(assay.factor_param(), Factor::Variable(0.0));
        assert!(assay.is_additive());
        let fixed = OutputModel::assay([0.0, 0.1]).multiplicative(1.5).fixed();
        assert_eq!(fixed.factor_param(), Factor::Fixed(1.5));
        assert_eq!(fixed.errorpoly(), ErrorPoly::from([0.0, 0.1]));

        // Shorthands and the general setters agree.
        assert_eq!(
            OutputModel::proportional(0.1).log(),
            OutputModel::new(ResidualErrorModel::proportional(0.1)).with_transform(Transform::Log)
        );
        assert_eq!(
            OutputModel::exponential(0.1),
            OutputModel::proportional(0.1).log()
        );
        assert_eq!(
            OutputModel::assay([0.0, 0.1]).student_t(4.0),
            OutputModel::new(OutputModel::assay([0.0, 0.1])).with_family(Family::student_t(4.0))
        );

        // `with` replaces, `add` refuses, `from` takes the last.
        let replaced = OutputModels::new()
            .with("cp", OutputModel::constant(1.0))
            .with("cp", OutputModel::constant(2.0));
        assert_eq!(replaced.len(), 1);
        assert_eq!(replaced.get("cp"), Some(&OutputModel::constant(2.0)));
        assert!(matches!(
            replaced.clone().add("cp", OutputModel::constant(3.0)),
            Err(ErrorModelError::ExistingOutputLabel(label)) if label == "cp"
        ));
        let from = OutputModels::from([
            ("cp", OutputModel::constant(1.0)),
            ("cp", OutputModel::constant(2.0)),
        ]);
        assert_eq!(from, replaced);
    }

    #[test]
    fn parameters_cover_assay_and_residual_scales() {
        let mut models = OutputModels::new()
            .with("cp", OutputModel::assay([1.0]).multiplicative(2.0).fixed())
            .with("effect", ResidualErrorModel::combined(0.5, 0.1));

        assert_eq!(models.parameters("cp").unwrap(), vec![Factor::Fixed(2.0)]);
        assert_eq!(
            models.parameters("effect").unwrap(),
            vec![Factor::Variable(0.5), Factor::Variable(0.1)]
        );

        models.set_parameters("cp", &[3.0]).unwrap();
        models.set_parameters("effect", &[0.7, 0.2]).unwrap();
        // Setting keeps a fixed factor fixed.
        assert_eq!(models.parameters("cp").unwrap(), vec![Factor::Fixed(3.0)]);
        assert_eq!(
            models.get("effect").unwrap().as_residual(),
            Some(&ResidualErrorModel::combined(0.7, 0.2))
        );

        assert!(matches!(
            models.set_parameters("effect", &[1.0]),
            Err(ErrorModelError::ParameterCount {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            models.parameters("other"),
            Err(ErrorModelError::UnknownOutputLabel(_))
        ));
    }

    #[test]
    fn hash_covers_scale_family_and_transform() {
        let base = OutputModels::new().with("cp", constant(1.0));
        let variants = [
            OutputModels::new().with("cp", constant(2.0)),
            OutputModels::new().with("cp", ResidualErrorModel::proportional(1.0)),
            OutputModels::new().with("cp", constant(1.0).student_t(3.0)),
            OutputModels::new().with("cp", constant(1.0).student_t(4.0)),
            OutputModels::new().with("cp", constant(1.0).log()),
            OutputModels::new().with("effect", constant(1.0)),
        ];
        for variant in &variants {
            assert_ne!(base.hash(), variant.hash(), "{variant:?}");
        }
        assert_eq!(base.hash(), base.clone().hash());
    }

    #[test]
    fn serde_round_trips_and_defaults_family_and_transform() {
        let models = OutputModels::new()
            .with("cp", constant(1.0).student_t(5.0))
            .with("pd", OutputModel::exponential(0.1));
        let json = serde_json::to_string(&models).unwrap();
        let back: OutputModels = serde_json::from_str(&json).unwrap();
        assert_eq!(back, models);

        let minimal: OutputModels =
            serde_json::from_str(r#"{"cp": {"scale": {"Prediction": {"Constant": {"a": 0.5}}}}}"#)
                .unwrap();
        let model = minimal.get("cp").unwrap();
        assert_eq!(model.family(), Family::Normal);
        assert_eq!(model.transform(), Transform::Identity);
    }
}
