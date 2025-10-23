use crate::data::error_model::ErrorPoly;
use crate::prelude::simulator::Prediction;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a pharmacokinetic/pharmacodynamic event
///
/// Events represent key occurrences in a PK/PD profile, including:
/// - [Bolus] doses (instantaneous drug input)
/// - [Infusion]s (continuous drug input over a duration)
/// - [Observation]s (measured concentrations or other values)
#[derive(Serialize, Debug, Clone, Deserialize)]
pub enum Event {
    /// A bolus dose (instantaneous drug input)
    Bolus(Bolus),
    /// An infusion (continuous drug input over a duration)
    Infusion(Infusion),
    /// An observation of drug concentration or other measure
    Observation(Observation),
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

/// Represents an instantaneous input of drug
///
/// A [Bolus] is a discrete amount of drug added to a specific compartment at a specific time.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Bolus {
    time: f64,
    amount: f64,
    input: usize,
    occasion: usize,
}
impl Bolus {
    /// Create a new bolus event
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the bolus dose
    /// * `amount` - Amount of drug administered
    /// * `input` - The compartment number (zero-indexed) receiving the dose
    pub fn new(time: f64, amount: f64, input: usize, occasion: usize) -> Self {
        Bolus {
            time,
            amount,
            input,
            occasion,
        }
    }

    /// Get the amount of drug in the bolus
    pub fn amount(&self) -> f64 {
        self.amount
    }

    /// Get the compartment number (zero-indexed) that receives the bolus
    pub fn input(&self) -> usize {
        self.input
    }

    /// Get the time of the bolus administration
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Set the amount of drug in the bolus
    pub fn set_amount(&mut self, amount: f64) {
        self.amount = amount;
    }

    /// Set the compartment number (zero-indexed) that receives the bolus
    pub fn set_input(&mut self, input: usize) {
        self.input = input;
    }

    /// Set the time of the bolus administration
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Get a mutable reference to the amount of drug in the bolus
    pub fn mut_amount(&mut self) -> &mut f64 {
        &mut self.amount
    }

    /// Get a mutable reference to the compartment number that receives the bolus
    pub fn mut_input(&mut self) -> &mut usize {
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

/// Represents a continuous dose of drug over time
///
/// An [Infusion] administers drug at a constant rate over a specified duration.
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Infusion {
    time: f64,
    amount: f64,
    input: usize,
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
    /// * `input` - The compartment number (zero-indexed) receiving the dose
    /// * `duration` - Duration of the infusion in time units
    pub fn new(time: f64, amount: f64, input: usize, duration: f64, occasion: usize) -> Self {
        Infusion {
            time,
            amount,
            input,
            duration,
            occasion,
        }
    }

    /// Get the total amount of drug provided over the infusion
    pub fn amount(&self) -> f64 {
        self.amount
    }

    /// Get the compartment number (zero-indexed) that receives the infusion
    pub fn input(&self) -> usize {
        self.input
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

    /// Set the compartment number (zero-indexed) that receives the infusion
    pub fn set_input(&mut self, input: usize) {
        self.input = input;
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

    /// Set the compartment number (zero-indexed) that receives the infusion
    pub fn mut_input(&mut self) -> &mut usize {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Censor {
    /// No censoring
    None,
    /// Below the lower limit of quantification
    BLOQ,
    /// Above the limit of quantification
    ALOQ,
}

/// Represents an observation of drug concentration or other measured value
#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Observation {
    time: f64,
    value: Option<f64>,
    outeq: usize,
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
    /// * `outeq` - Output equation number (zero-indexed) corresponding to this observation
    /// * `errorpoly` - Optional error polynomial coefficients (c0, c1, c2, c3)
    /// * `ignore` - Whether to ignore this observation in calculations
    pub(crate) fn new(
        time: f64,
        value: Option<f64>,
        outeq: usize,
        errorpoly: Option<ErrorPoly>,
        occasion: usize,
        censoring: Censor,
    ) -> Self {
        Observation {
            time,
            value,
            outeq,
            errorpoly,
            occasion,
            censoring,
        }
    }

    /// Get the time of the observation
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the value of the observation (e.g., drug concentration)
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Get the output equation number (zero-indexed) corresponding to this observation
    pub fn outeq(&self) -> usize {
        self.outeq
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

    /// Set the output equation number (zero-indexed) corresponding to this observation
    pub fn set_outeq(&mut self, outeq: usize) {
        self.outeq = outeq;
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

    /// Get a mutable reference to the output equation number
    pub fn mut_outeq(&mut self) -> &mut usize {
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

    /// Create a [Prediction] from this observation
    pub fn to_prediction(&self, pred: f64, state: Vec<f64>) -> Prediction {
        Prediction {
            time: self.time(),
            observation: self.value(),
            prediction: pred,
            outeq: self.outeq(),
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
