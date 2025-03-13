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
    pub(crate) fn get_time(&self) -> f64 {
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
