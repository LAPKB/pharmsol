use std::fmt;

use serde::Deserialize;

/// Represents a pharmacokinetic/pharmacodynamic event
///
/// Events represent key occurrences in a PK/PD profile, including:
/// - [Bolus] doses (instantaneous drug input)
/// - [Infusion]s (continuous drug input over a duration)
/// - [Observation]s (measured concentrations or other values)
#[derive(serde::Serialize, Debug, Clone, Deserialize)]
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
    pub(crate) fn time(&self) -> f64 {
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
}

/// Represents an instantaneous input of drug
///
/// A [Bolus] is a discrete amount of drug added to a specific compartment at a specific time.
#[derive(serde::Serialize, Debug, Clone, Deserialize)]
pub struct Bolus {
    time: f64,
    amount: f64,
    input: usize,
}
impl Bolus {
    /// Create a new bolus event
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the bolus dose
    /// * `amount` - Amount of drug administered
    /// * `input` - The compartment number (zero-indexed) receiving the dose
    pub(crate) fn new(time: f64, amount: f64, input: usize) -> Self {
        Bolus {
            time,
            amount,
            input,
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
    /// Get a mutable reference to the time of the bolus
    pub(crate) fn mut_time(&mut self) -> &mut f64 {
        &mut self.time
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
}

/// Represents a continuous dose of drug over time
///
/// An [Infusion] administers drug at a constant rate over a specified duration.
#[derive(serde::Serialize, Debug, Clone, Deserialize)]
pub struct Infusion {
    time: f64,
    amount: f64,
    input: usize,
    duration: f64,
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
    pub(crate) fn new(time: f64, amount: f64, input: usize, duration: f64) -> Self {
        Infusion {
            time,
            amount,
            input,
            duration,
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
}

/// Represents an observation of drug concentration or other measured value
#[derive(serde::Serialize, Debug, Clone, Deserialize)]
pub struct Observation {
    time: f64,
    value: f64,
    outeq: usize,
    errorpoly: Option<(f64, f64, f64, f64)>,
    ignore: bool,
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
        value: f64,
        outeq: usize,
        errorpoly: Option<(f64, f64, f64, f64)>,
        ignore: bool,
    ) -> Self {
        Observation {
            time,
            value,
            outeq,
            errorpoly,
            ignore,
        }
    }
    /// Get the time of the observation
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Get the value of the observation (e.g., drug concentration)
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Get the output equation number (zero-indexed) corresponding to this observation
    pub fn outeq(&self) -> usize {
        self.outeq
    }
    /// Get the error polynomial coefficients (c0, c1, c2, c3) if available
    ///
    /// The error polynomial is used to model the observation error.
    pub fn errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
        self.errorpoly
    }
    /// Check if this observation should be ignored in calculations
    pub fn ignore(&self) -> bool {
        self.ignore
    }

    /// Set the time of the observation
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Set the value of the observation (e.g., drug concentration)
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    /// Set the output equation number (zero-indexed) corresponding to this observation
    pub fn set_outeq(&mut self, outeq: usize) {
        self.outeq = outeq;
    }

    /// Set the error polynomial coefficients (c0, c1, c2, c3) if available
    pub fn set_errorpoly(&mut self, errorpoly: Option<(f64, f64, f64, f64)>) {
        self.errorpoly = errorpoly;
    }

    /// Set whether to ignore this observation in calculations
    pub fn set_ignore(&mut self, ignore: bool) {
        self.ignore = ignore;
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
                    Some((c0, c1, c2, c3)) => {
                        format!("with error poly =  ({}, {}, {}, {})", c0, c1, c2, c3)
                    }
                    None => "".to_string(),
                };
                write!(
                    f,
                    "Observation at time {:.2}: {} (outeq {}) {}",
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
        let bolus = Bolus::new(2.5, 100.0, 1);
        assert_eq!(bolus.time(), 2.5);
        assert_eq!(bolus.amount(), 100.0);
        assert_eq!(bolus.input(), 1);
    }

    #[test]
    fn test_bolus_setters() {
        let mut bolus = Bolus::new(2.5, 100.0, 1);

        bolus.set_time(3.0);
        assert_eq!(bolus.time(), 3.0);

        bolus.set_amount(150.0);
        assert_eq!(bolus.amount(), 150.0);

        bolus.set_input(2);
        assert_eq!(bolus.input(), 2);
    }

    #[test]
    fn test_infusion_creation() {
        let infusion = Infusion::new(1.0, 200.0, 1, 2.5);
        assert_eq!(infusion.time(), 1.0);
        assert_eq!(infusion.amount(), 200.0);
        assert_eq!(infusion.input(), 1);
        assert_eq!(infusion.duration(), 2.5);
    }

    #[test]
    fn test_infusion_setters() {
        let mut infusion = Infusion::new(1.0, 200.0, 1, 2.5);

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
        let error_poly = Some((0.1, 0.2, 0.3, 0.4));
        let observation = Observation::new(5.0, 75.5, 2, error_poly, false);

        assert_eq!(observation.time(), 5.0);
        assert_eq!(observation.value(), 75.5);
        assert_eq!(observation.outeq(), 2);
        assert_eq!(observation.errorpoly(), error_poly);
        assert_eq!(observation.ignore(), false);
    }

    #[test]
    fn test_observation_setters() {
        let mut observation = Observation::new(5.0, 75.5, 2, Some((0.1, 0.2, 0.3, 0.4)), false);

        observation.set_time(6.0);
        assert_eq!(observation.time(), 6.0);

        observation.set_value(80.0);
        assert_eq!(observation.value(), 80.0);

        observation.set_outeq(3);
        assert_eq!(observation.outeq(), 3);

        let new_error_poly = Some((0.2, 0.3, 0.4, 0.5));
        observation.set_errorpoly(new_error_poly);
        assert_eq!(observation.errorpoly(), new_error_poly);

        observation.set_ignore(true);
        assert_eq!(observation.ignore(), true);
    }

    #[test]
    fn test_event_time_operations() {
        let mut bolus_event = Event::Bolus(Bolus::new(1.0, 100.0, 1));
        let mut infusion_event = Event::Infusion(Infusion::new(2.0, 200.0, 1, 2.5));
        let mut observation_event = Event::Observation(Observation::new(3.0, 75.5, 2, None, false));

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
