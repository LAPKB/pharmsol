//! Event types and public label wrappers for subject schedules.
//!
//! These types are the low-level representation behind the higher-level
//! builder and parsing APIs. Most users can start with
//! [`crate::data::builder::SubjectBuilder`], then inspect or transform
//! [`Event`] values after construction.
//!
//! Dose events carry an [`InputLabel`], and observations carry an
//! [`OutputLabel`]. Prefer stable strings such as `"depot"`, `"iv"`, and
//! `"cp"`. Numeric values are accepted, but they remain labels until a
//! downstream workflow explicitly interprets them as indices.

use crate::data::error_model::ErrorPoly;
use crate::prelude::simulator::Prediction;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Shared Analysis Types
// ============================================================================

/// Administration route classification used by downstream analyses.
///
/// [`Route`] is a coarse route category, not the original public input label.
/// In the current data-side heuristic:
/// - [`Event::Infusion`] maps to [`Route::IVInfusion`]
/// - [`Event::Bolus`] with input label `0` maps to [`Route::Extravascular`]
/// - [`Event::Bolus`] with any other label maps to [`Route::IVBolus`]
///
/// If you need the original model-facing label, read [`Bolus::input`] or
/// [`Infusion::input`] instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// Intravenous bolus
    IVBolus,
    /// Intravenous infusion
    IVInfusion,
    /// Extravascular (oral, SC, IM, etc.)
    #[default]
    Extravascular,
}

/// AUC calculation method
///
/// Controls how the area under the concentration-time curve is computed.
/// This is a general trapezoidal method applicable to any AUC calculation,
/// not specific to NCA analysis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AUCMethod {
    /// Linear trapezoidal rule
    Linear,
    /// Linear up / log down (industry standard)
    #[default]
    LinUpLogDown,
    /// Linear before Tmax, log-linear after Tmax (PKNCA "lin-log")
    ///
    /// Uses linear trapezoidal before and at Tmax, then log-linear for
    /// descending portions after Tmax. Falls back to linear if either
    /// concentration is zero or non-positive.
    LinLog,
}

/// BLQ (Below Limit of Quantification) handling rule
///
/// Controls how observations marked with [`Censor::BLOQ`] are handled
/// during analysis. Applicable to NCA, AUC calculations, and any
/// observation-processing pipeline.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum BLQRule {
    /// Replace BLQ with zero
    Zero,
    /// Replace BLQ with LOQ/2
    LoqOver2,
    /// Exclude BLQ values from analysis
    #[default]
    Exclude,
    /// Position-aware handling (PKNCA default): first=keep(0), middle=drop, last=keep(0)
    ///
    /// This is the FDA-recommended approach.
    /// It keeps the first BLQ (before tfirst) as 0 to anchor the profile start,
    /// drops middle BLQ values (between tfirst and tlast) to avoid deflating AUC,
    /// and keeps the last BLQ (at or after tlast) as 0 to define profile end.
    Positional,
    /// Tmax-relative handling: different rules before vs after Tmax
    ///
    /// Contains `(before_tmax_rule, after_tmax_rule)`.
    /// Each rule can either keep BLQ as 0 or drop it from analysis.
    /// Default PKNCA behavior is `before.tmax=drop` and `after.tmax=keep`.
    TmaxRelative {
        /// Rule for BLQ before Tmax: true=keep as 0, false=drop
        before_tmax_keep: bool,
        /// Rule for BLQ at or after Tmax: true=keep as 0, false=drop
        after_tmax_keep: bool,
    },
}

/// One scheduled item in a subject record.
///
/// Events are the low-level representation for doses and observations:
/// - [`Bolus`] for instantaneous input
/// - [`Infusion`] for input over a duration
/// - [`Observation`] for measured or missing outputs
///
/// Most users create these through `Subject::builder(...)`, row ingestion, or
/// file parsing rather than constructing them all by hand.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub enum Event {
    /// A bolus dose (instantaneous drug input)
    Bolus(Bolus),
    /// An infusion (continuous drug input over a duration)
    Infusion(Infusion),
    /// An observation of drug concentration or other measure
    Observation(Observation),
}

/// Public label for a dosing input or route.
///
/// [`Bolus`] and [`Infusion`] store the original user-facing route name in
/// this type.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InputLabel(String);

impl InputLabel {
    /// Create a new public label.
    ///
    /// Prefer stable names when the model declares named routes.
    pub fn new(label: impl ToString) -> Self {
        Self(label.to_string())
    }

    /// Borrow the stored label as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Try to interpret the label as a numeric index.
    ///
    /// This is mainly a compatibility helper for lower-level paths that still
    /// operate on dense indices after label resolution.
    pub fn index(&self) -> Option<usize> {
        self.0.parse::<usize>().ok()
    }
}

impl From<String> for InputLabel {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for InputLabel {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<usize> for InputLabel {
    fn from(value: usize) -> Self {
        Self(value.to_string())
    }
}

impl AsRef<str> for InputLabel {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for InputLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<usize> for InputLabel {
    fn eq(&self, other: &usize) -> bool {
        self.index() == Some(*other)
    }
}

impl PartialEq<InputLabel> for usize {
    fn eq(&self, other: &InputLabel) -> bool {
        other == self
    }
}

impl PartialEq<usize> for &InputLabel {
    fn eq(&self, other: &usize) -> bool {
        (**self).eq(other)
    }
}

impl PartialEq<&InputLabel> for usize {
    fn eq(&self, other: &&InputLabel) -> bool {
        other.eq(self)
    }
}

/// Public label for an observation output.
///
/// [`Observation`] stores the original user-facing output name in this type.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OutputLabel(String);

impl OutputLabel {
    /// Create a new public label.
    ///
    /// Prefer stable names when the model declares named outputs.
    pub fn new(label: impl ToString) -> Self {
        Self(label.to_string())
    }

    /// Borrow the stored label as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Try to interpret the label as a numeric index.
    ///
    /// This is mainly a compatibility helper for lower-level paths that still
    /// operate on dense indices after label resolution.
    pub fn index(&self) -> Option<usize> {
        self.0.parse::<usize>().ok()
    }
}

impl From<String> for OutputLabel {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for OutputLabel {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<usize> for OutputLabel {
    fn from(value: usize) -> Self {
        Self(value.to_string())
    }
}

impl AsRef<str> for OutputLabel {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for OutputLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<usize> for OutputLabel {
    fn eq(&self, other: &usize) -> bool {
        self.index() == Some(*other)
    }
}

impl PartialEq<OutputLabel> for usize {
    fn eq(&self, other: &OutputLabel) -> bool {
        other == self
    }
}

impl PartialEq<usize> for &OutputLabel {
    fn eq(&self, other: &usize) -> bool {
        (**self).eq(other)
    }
}

impl PartialEq<&OutputLabel> for usize {
    fn eq(&self, other: &&OutputLabel) -> bool {
        other.eq(self)
    }
}

impl Event {
    /// Get the time of the event
    pub fn time(&self) -> f64 {
        match self {
            Event::Bolus(bolus) => bolus.time,
            Event::Infusion(infusion) => infusion.time,
            Event::Observation(observation) => observation.time,
        }
    }
    /// Increment the event time by a specified delta
    pub(crate) fn inc_time(&mut self, dt: f64) {
        match self {
            Event::Bolus(bolus) => bolus.time += dt,
            Event::Infusion(infusion) => infusion.time += dt,
            Event::Observation(observation) => observation.time += dt,
        }
    }

    /// Get the occasion index for this event
    pub fn occasion(&self) -> usize {
        match self {
            Event::Bolus(bolus) => bolus.occasion,
            Event::Infusion(infusion) => infusion.occasion,
            Event::Observation(observation) => observation.occasion,
        }
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        match self {
            Event::Bolus(bolus) => bolus.mut_occasion(),
            Event::Infusion(infusion) => infusion.mut_occasion(),
            Event::Observation(observation) => observation.mut_occasion(),
        }
    }

    /// Set the occasion index for this event
    pub fn set_occasion(&mut self, occasion: usize) {
        match self {
            Event::Bolus(_) => {
                *self.mut_occasion() = occasion;
            }
            Event::Infusion(_) => {
                *self.mut_occasion() = occasion;
            }
            Event::Observation(_) => {
                *self.mut_occasion() = occasion;
            }
        }
    }
}

/// Instantaneous dose input.
///
/// A [`Bolus`] records one discrete amount at one time, tagged with the public
/// input label that should be matched against the model.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Bolus {
    time: f64,
    amount: f64,
    input: InputLabel,
    occasion: usize,
}
impl Bolus {
    /// Create a new bolus event
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the bolus dose
    /// * `amount` - Amount of drug administered
    /// * `input` - The route label receiving the dose
    pub fn new(time: f64, amount: f64, input: impl ToString, occasion: usize) -> Self {
        Bolus {
            time,
            amount,
            input: InputLabel::new(input),
            occasion,
        }
    }

    /// Get the amount of drug in the bolus
    pub fn amount(&self) -> f64 {
        self.amount
    }

    /// Get the route label that receives the bolus
    pub fn input(&self) -> &InputLabel {
        &self.input
    }

    /// Try to interpret the input label as a numeric index.
    ///
    /// Prefer [`Bolus::input`] when working with the public label itself.
    pub fn input_index(&self) -> Option<usize> {
        self.input.index()
    }

    /// Get the time of the bolus administration
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Set the amount of drug in the bolus
    pub fn set_amount(&mut self, amount: f64) {
        self.amount = amount;
    }

    /// Set the route label that receives the bolus
    pub fn set_input(&mut self, input: impl ToString) {
        self.input = InputLabel::new(input);
    }

    /// Set the time of the bolus administration
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Get a mutable reference to the amount of drug in the bolus
    pub fn mut_amount(&mut self) -> &mut f64 {
        &mut self.amount
    }

    /// Get a mutable reference to the route label that receives the bolus
    pub fn mut_input(&mut self) -> &mut InputLabel {
        &mut self.input
    }

    /// Get a mutable reference to the time of the bolus administration
    pub fn mut_time(&mut self) -> &mut f64 {
        &mut self.time
    }

    /// Get the occasion index for this bolus
    pub fn occasion(&self) -> usize {
        self.occasion
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        &mut self.occasion
    }
}

/// Continuous dose input over a duration.
///
/// An [`Infusion`] records the total amount, start time, duration, and public
/// input label for one infusion event.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Infusion {
    time: f64,
    amount: f64,
    input: InputLabel,
    duration: f64,
    occasion: usize,
}
impl Infusion {
    /// Create a new infusion event
    ///
    /// # Arguments
    ///
    /// * `time` - Start time of the infusion
    /// * `amount` - Total amount of drug to be administered
    /// * `input` - The route label receiving the dose
    /// * `duration` - Duration of the infusion in time units
    pub fn new(
        time: f64,
        amount: f64,
        input: impl ToString,
        duration: f64,
        occasion: usize,
    ) -> Self {
        Infusion {
            time,
            amount,
            input: InputLabel::new(input),
            duration,
            occasion,
        }
    }

    /// Get the total amount of drug provided over the infusion
    pub fn amount(&self) -> f64 {
        self.amount
    }

    /// Get the route label that receives the infusion
    pub fn input(&self) -> &InputLabel {
        &self.input
    }

    /// Try to interpret the input label as a numeric index.
    ///
    /// Prefer [`Infusion::input`] when working with the public label itself.
    pub fn input_index(&self) -> Option<usize> {
        self.input.index()
    }

    /// Get the duration of the infusion
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get the start time of the infusion
    ///
    /// The infusion continues from this time until time + duration.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Set the amount of drug in the infusion
    pub fn set_amount(&mut self, amount: f64) {
        self.amount = amount;
    }

    /// Set the route label that receives the infusion
    pub fn set_input(&mut self, input: impl ToString) {
        self.input = InputLabel::new(input);
    }

    /// Set the time of the infusion administration
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Set the duration of the infusion
    pub fn set_duration(&mut self, duration: f64) {
        self.duration = duration;
    }

    /// Set the amount of drug in the infusion
    pub fn mut_amount(&mut self) -> &mut f64 {
        &mut self.amount
    }

    /// Get a mutable reference to the route label that receives the infusion
    pub fn mut_input(&mut self) -> &mut InputLabel {
        &mut self.input
    }

    /// Set the time of the infusion administration
    pub fn mut_time(&mut self) -> &mut f64 {
        &mut self.time
    }

    /// Set the duration of the infusion
    pub fn mut_duration(&mut self) -> &mut f64 {
        &mut self.duration
    }

    /// Get the occasion index for this infusion
    pub fn occasion(&self) -> usize {
        self.occasion
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        &mut self.occasion
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Censor {
    /// No censoring
    #[default]
    None,
    /// Below the lower limit of quantification
    BLOQ,
    /// Above the limit of quantification
    ALOQ,
}

/// Observation of a model output.
///
/// An [`Observation`] can carry a measured value or `None` for a prediction-only
/// time point. Observations also carry the public output label, optional assay
/// error polynomial, occasion index, and censoring state.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Observation {
    time: f64,
    value: Option<f64>,
    outeq: OutputLabel,
    errorpoly: Option<ErrorPoly>,
    occasion: usize,
    censoring: Censor,
}
impl Observation {
    /// Create a new observation
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Output label corresponding to this observation
    /// * `errorpoly` - Optional error polynomial coefficients (c0, c1, c2, c3)
    /// * `occasion` - Occasion index
    /// * `censoring` - Censoring type for this observation
    pub(crate) fn new(
        time: f64,
        value: Option<f64>,
        outeq: impl ToString,
        errorpoly: Option<ErrorPoly>,
        occasion: usize,
        censoring: Censor,
    ) -> Self {
        Observation {
            time,
            value,
            outeq: OutputLabel::new(outeq),
            errorpoly,
            occasion,
            censoring,
        }
    }

    /// Get the time of the observation
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the value of the observation.
    ///
    /// `None` means this is a prediction-only or missing-observation slot.
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Get the output label corresponding to this observation
    pub fn outeq(&self) -> &OutputLabel {
        &self.outeq
    }

    /// Try to interpret the output label as a numeric index.
    ///
    /// Prefer [`Observation::outeq`] when working with the public label itself.
    pub fn outeq_index(&self) -> Option<usize> {
        self.outeq.index()
    }

    /// Get the error polynomial coefficients (c0, c1, c2, c3) if available
    ///
    /// The error polynomial is used to model the observation error.
    pub fn errorpoly(&self) -> Option<ErrorPoly> {
        self.errorpoly
    }

    /// Set the time of the observation
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Set the value of the observation (e.g., drug concentration)
    pub fn set_value(&mut self, value: Option<f64>) {
        self.value = value;
    }

    /// Set the output label corresponding to this observation
    pub fn set_outeq(&mut self, outeq: impl ToString) {
        self.outeq = OutputLabel::new(outeq);
    }

    /// Set the [ErrorPoly] for this observation
    pub fn set_errorpoly(&mut self, errorpoly: Option<ErrorPoly>) {
        self.errorpoly = errorpoly;
    }

    /// Get a mutable reference to the time of the observation
    pub fn mut_time(&mut self) -> &mut f64 {
        &mut self.time
    }

    /// Get a mutable reference to the value of the observation
    pub fn mut_value(&mut self) -> &mut Option<f64> {
        &mut self.value
    }

    /// Get a mutable reference to the output label
    pub fn mut_outeq(&mut self) -> &mut OutputLabel {
        &mut self.outeq
    }

    /// Get a mutable reference to the error polynomial
    pub fn mut_errorpoly(&mut self) -> &mut Option<ErrorPoly> {
        &mut self.errorpoly
    }

    /// Get the occasion index for this observation
    pub fn occasion(&self) -> usize {
        self.occasion
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        &mut self.occasion
    }

    /// Create a [`Prediction`] from this observation.
    ///
    /// This is a low-level helper for code paths that already operate on a
    /// resolved or numeric output index. Named output labels must be resolved by
    /// the caller before this conversion happens.
    pub fn to_prediction(&self, pred: f64, state: Vec<f64>) -> Prediction {
        Prediction {
            time: self.time(),
            observation: self.value(),
            prediction: pred,
            outeq: self
                .outeq_index()
                .expect("prediction requires a resolved or numeric output label"),
            errorpoly: self.errorpoly(),
            state,
            occasion: self.occasion(),
            censoring: self.censoring(),
        }
    }

    /// Check if the observation is censored
    pub fn censored(&self) -> bool {
        match self.censoring {
            Censor::None => false,
            Censor::ALOQ => true,
            Censor::BLOQ => true,
        }
    }

    /// Get the censoring type of the observation
    pub fn censoring(&self) -> Censor {
        self.censoring
    }

    /// Set whether the observation is censored
    pub fn censor(&mut self, censor: Censor) {
        self.censoring = censor;
    }

    /// Get a mutable reference to the censoring flag
    pub fn mut_censoring(&mut self) -> &mut Censor {
        &mut self.censoring
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Bolus(bolus) => write!(
                f,
                "Bolus at time {:.2} with amount {:.2} in compartment {}",
                bolus.time, bolus.amount, bolus.input
            ),
            Event::Infusion(infusion) => write!(
                f,
                "Infusion starting at {:.2} with amount {:.2} over {:.2} hours in compartment {}",
                infusion.time, infusion.amount, infusion.duration, infusion.input
            ),
            Event::Observation(observation) => {
                let errpoly_desc = match observation.errorpoly {
                    Some(errorpoly) => {
                        format!(
                            "with error poly {} {} {} {}",
                            errorpoly.coefficients().0,
                            errorpoly.coefficients().1,
                            errorpoly.coefficients().2,
                            errorpoly.coefficients().3
                        )
                    }
                    None => "".to_string(),
                };
                write!(
                    f,
                    "Observation at time {:.2}: {:#?} (outeq {}) {}",
                    observation.time, observation.value, observation.outeq, errpoly_desc
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bolus_creation() {
        let bolus = Bolus::new(2.5, 100.0, 1, 0);
        assert_eq!(bolus.time(), 2.5);
        assert_eq!(bolus.amount(), 100.0);
        assert_eq!(bolus.input(), 1);
        assert_eq!(bolus.input().as_str(), "1");
    }

    #[test]
    fn test_bolus_setters() {
        let mut bolus = Bolus::new(2.5, 100.0, 1, 0);

        bolus.set_time(3.0);
        assert_eq!(bolus.time(), 3.0);

        bolus.set_amount(150.0);
        assert_eq!(bolus.amount(), 150.0);

        bolus.set_input(2);
        assert_eq!(bolus.input(), 2);
    }

    #[test]
    fn test_infusion_creation() {
        let infusion = Infusion::new(1.0, 200.0, 1, 2.5, 0);
        assert_eq!(infusion.time(), 1.0);
        assert_eq!(infusion.amount(), 200.0);
        assert_eq!(infusion.input(), 1);
        assert_eq!(infusion.input().as_str(), "1");
        assert_eq!(infusion.duration(), 2.5);
    }

    #[test]
    fn test_infusion_setters() {
        let mut infusion = Infusion::new(1.0, 200.0, 1, 2.5, 0);

        infusion.set_time(1.5);
        assert_eq!(infusion.time(), 1.5);

        infusion.set_amount(250.0);
        assert_eq!(infusion.amount(), 250.0);

        infusion.set_input(2);
        assert_eq!(infusion.input(), 2);

        infusion.set_duration(3.0);
        assert_eq!(infusion.duration(), 3.0);
    }

    #[test]
    fn test_observation_creation() {
        let error_poly = Some(ErrorPoly::new(0.1, 0.2, 0.3, 0.4));
        let observation = Observation::new(5.0, Some(75.5), 2, error_poly, 0, Censor::None);

        assert_eq!(observation.time(), 5.0);
        assert_eq!(observation.value(), Some(75.5));
        assert_eq!(observation.outeq(), 2);
        assert_eq!(observation.outeq().as_str(), "2");
        assert_eq!(observation.errorpoly(), error_poly);
    }

    #[test]
    fn test_observation_setters() {
        let mut observation = Observation::new(
            5.0,
            Some(75.5),
            2,
            Some(ErrorPoly::new(0.1, 0.2, 0.3, 0.4)),
            0,
            Censor::None,
        );

        observation.set_time(6.0);
        assert_eq!(observation.time(), 6.0);

        observation.set_value(Some(80.0));
        assert_eq!(observation.value(), Some(80.0));

        observation.set_outeq(3);
        assert_eq!(observation.outeq(), 3);

        let new_error_poly = Some(ErrorPoly::new(0.2, 0.3, 0.4, 0.5));
        observation.set_errorpoly(new_error_poly);
        assert_eq!(observation.errorpoly(), new_error_poly);
    }

    #[test]
    fn test_event_time_operations() {
        let mut bolus_event = Event::Bolus(Bolus::new(1.0, 100.0, 1, 0));
        let mut infusion_event = Event::Infusion(Infusion::new(2.0, 200.0, 1, 2.5, 0));
        let mut observation_event =
            Event::Observation(Observation::new(3.0, Some(75.5), 2, None, 0, Censor::None));

        assert_eq!(bolus_event.time(), 1.0);
        assert_eq!(infusion_event.time(), 2.0);
        assert_eq!(observation_event.time(), 3.0);

        bolus_event.inc_time(0.5);
        infusion_event.inc_time(0.5);
        observation_event.inc_time(0.5);

        assert_eq!(bolus_event.time(), 1.5);
        assert_eq!(infusion_event.time(), 2.5);
        assert_eq!(observation_event.time(), 3.5);
    }
}
