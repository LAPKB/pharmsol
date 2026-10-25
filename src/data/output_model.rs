//! Observation models, one per output.
//!
//! An [`OutputModel`] says how an output's observations are distributed around
//! the model's predictions. It is built from three independent choices:
//!
//! - [`Scale`]: where sigma comes from. [`Scale::Observation`] computes it
//!   from the observed value with an [`AssayErrorModel`] (assay error, the
//!   usual choice for non-parametric algorithms such as NPAG);
//!   [`Scale::Prediction`] computes it from the prediction with a
//!   [`ResidualErrorModel`] (residual error, the usual choice for parametric
//!   algorithms such as SAEM and FOCE).
//! - [`Family`]: the distribution: [`Family::Normal`] (the default),
//!   [`Family::StudentT`] for heavy tails, or [`Family::Laplace`].
//! - [`Transform`]: the scale observation and prediction are compared on:
//!   [`Transform::Identity`] (the default) or [`Transform::Log`].
//!
//! Censored observations (BLOQ, ALOQ) use the family's CDF or survival
//! function, whatever the scale.
//!
//! [`OutputModels`] collects one [`OutputModel`] per output label and is the
//! [`ObservationModel`] every likelihood in pharmsol is computed with. Outputs
//! in one collection can mix scales, families and transforms.
//!
//! ```rust
//! use pharmsol::prelude::*;
//! use pharmsol::{Family, OutputModel, ResidualErrorModel};
//!
//! let models = OutputModels::new()
//!     // Assay error on the concentration.
//!     .add("cp", AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.5))?
//!     // Heavy-tailed residual error on the effect.
//!     .add(
//!         "effect",
//!         OutputModel::new(ResidualErrorModel::combined(0.5, 0.1))
//!             .with_family(Family::student_t(4.0)?),
//!     )?
//!     // Exponential error: constant sigma on the log scale.
//!     .add("metabolite", OutputModel::exponential(0.2))?;
//! assert_eq!(models.len(), 3);
//! # Ok::<(), pharmsol::data::error_model::ErrorModelError>(())
//! ```

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::data::error_model::{self, AssayErrorModel, ErrorModelError, ErrorPoly};
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
/// and sigma (from the [`Scale`]) the scale parameter. Sigma is the standard
/// deviation only for [`Family::Normal`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub enum Family {
    /// Normal distribution.
    #[default]
    Normal,
    /// Student's t-distribution with `nu` degrees of freedom: heavier tails
    /// than the normal, so outliers weigh less. Approaches the normal as
    /// `nu` grows.
    StudentT {
        /// Degrees of freedom, positive and finite.
        nu: f64,
    },
    /// Laplace (double exponential) distribution, with scale sigma.
    Laplace,
}

impl Family {
    /// Student's t-distribution with `nu` degrees of freedom.
    ///
    /// Errors unless `nu` is positive and finite.
    pub fn student_t(nu: f64) -> Result<Self, ErrorModelError> {
        let family = Family::StudentT { nu };
        family.validate()?;
        Ok(family)
    }

    fn validate(&self) -> Result<(), ErrorModelError> {
        match *self {
            Family::StudentT { nu } if !(nu > 0.0 && nu.is_finite()) => Err(
                ErrorModelError::InvalidFamily(format!("Student's t needs nu > 0, got {nu}")),
            ),
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
            Family::Laplace => Ok(distributions::log_laplace_pdf(y, f, sigma)),
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
            Family::Laplace => Ok(distributions::log_laplace_cdf(y, f, sigma)),
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
            // The Laplace distribution is symmetric.
            Family::Laplace => Ok(distributions::log_laplace_cdf(f, y, sigma)),
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
    /// The density includes the Jacobian `-ln(y)`, so it is a density of the
    /// observation on its original scale, comparable with
    /// [`Transform::Identity`].
    Log,
}

/// The observation model for one output: a [`Scale`], a [`Family`] and a
/// [`Transform`].
///
/// Sigma is always computed from the untransformed observation or
/// prediction, then used on the transformed scale.
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
    pub fn new(scale: impl Into<Scale>) -> Self {
        Self {
            scale: scale.into(),
            family: Family::Normal,
            transform: Transform::Identity,
        }
    }

    /// Exponential error: normal with constant `sigma` on the log scale.
    pub fn exponential(sigma: f64) -> Self {
        Self::new(ResidualErrorModel::constant(sigma)).with_transform(Transform::Log)
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
    pub fn assay(&self) -> Option<&AssayErrorModel> {
        match &self.scale {
            Scale::Observation(model) => Some(model),
            Scale::Prediction(_) => None,
        }
    }

    /// Mutable [`Self::assay`].
    pub fn assay_mut(&mut self) -> Option<&mut AssayErrorModel> {
        match &mut self.scale {
            Scale::Observation(model) => Some(model),
            Scale::Prediction(_) => None,
        }
    }

    /// The residual error model, if sigma comes from the prediction.
    pub fn residual(&self) -> Option<&ResidualErrorModel> {
        match &self.scale {
            Scale::Prediction(model) => Some(model),
            Scale::Observation(_) => None,
        }
    }

    /// Mutable [`Self::residual`].
    pub fn residual_mut(&mut self) -> Option<&mut ResidualErrorModel> {
        match &mut self.scale {
            Scale::Prediction(model) => Some(model),
            Scale::Observation(_) => None,
        }
    }

    /// Sigma for a prediction and its observation.
    ///
    /// Errors with [`ErrorModelError::MissingObservation`] when sigma comes
    /// from the observation and there is none.
    pub fn sigma(&self, prediction: &Prediction) -> Result<f64, ErrorModelError> {
        let observation = match (&self.scale, prediction.observation()) {
            (_, Some(observation)) => observation,
            (Scale::Prediction(_), None) => f64::NAN,
            (Scale::Observation(_), None) => return Err(ErrorModelError::MissingObservation),
        };
        self.scale
            .sigma(observation, prediction.prediction(), prediction.errorpoly())
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
        let sigma = self.scale.sigma(observation, prediction, errorpoly)?;
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
            Family::Laplace => 2u8.hash(hasher),
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
/// [`Self::factor`] and [`Self::set_factor`] reach the assay factor (λ or γ)
/// of an output by label, as an NPAG loop does. Anything else about an
/// output's model goes through [`Self::get`] or [`Self::get_mut`] and the
/// [`OutputModel`] accessors.
///
/// ```rust
/// # use pharmsol::prelude::*;
/// let mut models = OutputModels::new()
///     .add("cp", AssayErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.5))?;
///
/// for (label, model) in models.iter_mut() {
///     if let Some(assay) = model.assay_mut().filter(|assay| assay.optimize()) {
///         assay.set_factor(assay.factor() * 2.0);
///     }
///     assert_eq!(label.as_str(), "cp");
/// }
/// assert_eq!(models.factor("cp")?, 1.0);
///
/// models.set_factor("cp", 0.25)?;
/// models.get_mut("cp").and_then(OutputModel::assay_mut).unwrap().fix_factor();
/// assert!(!models.get("cp").and_then(OutputModel::assay).unwrap().optimize());
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

    /// Add the model for an output label.
    ///
    /// Takes anything that converts into an [`OutputModel`], including an
    /// [`AssayErrorModel`] or a [`ResidualErrorModel`] on its own.
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
            .assay()
            .ok_or(ErrorModelError::NotAssayError(label))
    }

    /// Mutable [`Self::assay`].
    fn assay_mut(&mut self, outeq: impl ToString) -> Result<&mut AssayErrorModel, ErrorModelError> {
        let label = outeq.to_string();
        match self.models.get_mut(label.as_str()) {
            None => Err(ErrorModelError::UnknownOutputLabel(label)),
            Some(model) => model
                .assay_mut()
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

    fn cache_key(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        CACHE_TAG.hash(&mut hasher);
        self.hash().hash(&mut hasher);
        hasher.finish()
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
        OutputModel::new(ResidualErrorModel::constant(sigma))
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
        let model = OutputModel::new(AssayErrorModel::additive(
            ErrorPoly::new(0.0, 0.1, 0.0, 0.0),
            0.0,
        ));
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
        let t = constant(1.0).with_family(Family::student_t(3.0).unwrap());
        assert!(density(&t, 0.0, 0.0) < density(&normal, 0.0, 0.0));
        assert!(density(&t, 6.0, 0.0) > density(&normal, 6.0, 0.0));

        // Standard t with 1 degree of freedom is the Cauchy distribution.
        let cauchy = constant(1.0).with_family(Family::student_t(1.0).unwrap());
        let expected = -(std::f64::consts::PI * 2.0).ln();
        assert!((density(&cauchy, 1.0, 0.0) - expected).abs() < 1e-12);

        // Large nu approaches the normal.
        let wide = constant(1.0).with_family(Family::student_t(1.0e7).unwrap());
        assert!((density(&wide, 1.3, 0.0) - density(&normal, 1.3, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn invalid_degrees_of_freedom_are_rejected() {
        assert!(Family::student_t(0.0).is_err());
        assert!(Family::student_t(f64::NAN).is_err());
        let model = constant(1.0).with_family(Family::StudentT { nu: -1.0 });
        assert!(model.log_density(1.0, 0.0, Censor::None, None).is_err());
    }

    #[test]
    fn laplace_matches_the_closed_form() {
        let model = constant(2.0).with_family(Family::Laplace);
        let expected = -(4.0_f64).ln() - 1.5 / 2.0;
        assert!((density(&model, 1.5, 0.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn censoring_uses_the_family_tails() {
        for family in [
            Family::Normal,
            Family::student_t(4.0).unwrap(),
            Family::Laplace,
        ] {
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
            .add(
                "cp",
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
            )
            .unwrap()
            .add("effect", constant(0.5).with_family(Family::Laplace))
            .unwrap()
            .add("metabolite", OutputModel::exponential(0.2))
            .unwrap();

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
            distributions::log_laplace_pdf(3.0, 2.0, 0.5)
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
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
            )
            .unwrap()
            .add(1, ResidualErrorModel::proportional(0.2))
            .unwrap();
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
    fn hash_covers_scale_family_and_transform() {
        let base = OutputModels::new().add("cp", constant(1.0)).unwrap();
        let variants = [
            OutputModels::new().add("cp", constant(2.0)).unwrap(),
            OutputModels::new()
                .add("cp", ResidualErrorModel::proportional(1.0))
                .unwrap(),
            OutputModels::new()
                .add("cp", constant(1.0).with_family(Family::Laplace))
                .unwrap(),
            OutputModels::new()
                .add(
                    "cp",
                    constant(1.0).with_family(Family::student_t(3.0).unwrap()),
                )
                .unwrap(),
            OutputModels::new()
                .add(
                    "cp",
                    constant(1.0).with_family(Family::student_t(4.0).unwrap()),
                )
                .unwrap(),
            OutputModels::new()
                .add("cp", constant(1.0).with_transform(Transform::Log))
                .unwrap(),
            OutputModels::new().add("effect", constant(1.0)).unwrap(),
        ];
        for variant in &variants {
            assert_ne!(base.hash(), variant.hash(), "{variant:?}");
        }
        assert_eq!(base.hash(), base.clone().hash());
    }

    #[test]
    fn serde_round_trips_and_defaults_family_and_transform() {
        let models = OutputModels::new()
            .add(
                "cp",
                constant(1.0).with_family(Family::student_t(5.0).unwrap()),
            )
            .unwrap()
            .add("pd", OutputModel::exponential(0.1))
            .unwrap();
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
