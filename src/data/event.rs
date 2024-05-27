use std::fmt;

use serde::Deserialize;

/// An Event can be a Bolus, Infusion, or Observation
#[derive(Debug, Clone, Deserialize)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}
impl Event {
    pub(crate) fn get_time(&self) -> f64 {
        match self {
            Event::Bolus(bolus) => bolus.time,
            Event::Infusion(infusion) => infusion.time,
            Event::Observation(observation) => observation.time,
        }
    }
}

/// An instantaenous input of drug
#[derive(Debug, Clone, Deserialize)]
pub struct Bolus {
    time: f64,
    amount: f64,
    input: usize,
}
impl Bolus {
    pub(crate) fn new(time: f64, amount: f64, input: usize) -> Self {
        Bolus {
            time,
            amount,
            input,
        }
    }
    pub(crate) fn amount(&self) -> f64 {
        self.amount
    }
    pub(crate) fn input(&self) -> usize {
        self.input
    }
    pub(crate) fn time(&self) -> f64 {
        self.time
    }
    pub(crate) fn mut_time(&mut self) -> &mut f64 {
        &mut self.time
    }
}

/// A continuous dose of drug
#[derive(Debug, Clone, Deserialize)]
pub struct Infusion {
    time: f64,
    amount: f64,
    input: usize,
    duration: f64,
}
impl Infusion {
    pub(crate) fn new(time: f64, amount: f64, input: usize, duration: f64) -> Self {
        Infusion {
            time,
            amount,
            input,
            duration,
        }
    }
    pub(crate) fn amount(&self) -> f64 {
        self.amount
    }

    pub(crate) fn input(&self) -> usize {
        self.input
    }

    pub(crate) fn duration(&self) -> f64 {
        self.duration
    }

    pub(crate) fn time(&self) -> f64 {
        self.time
    }
}

/// An observation of drug concentration or covariates
#[derive(Debug, Clone, Deserialize)]
pub struct Observation {
    time: f64,
    value: f64,
    outeq: usize,
    errorpoly: Option<(f64, f64, f64, f64)>,
    ignore: bool,
}
impl Observation {
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
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn outeq(&self) -> usize {
        self.outeq
    }
    pub(crate) fn errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
        self.errorpoly
    }
    pub(crate) fn ignore(&self) -> bool {
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
