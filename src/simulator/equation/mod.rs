use std::fmt::Debug;
pub mod analytical;
pub mod meta;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use meta::*;
pub use ode::*;
pub use sde::*;

use crate::{
    error_model::ErrorModels,
    simulator::{Fa, Lag},
    Covariates, Event, Infusion, Observation, PharmsolError, Subject,
};

use super::likelihood::Prediction;

/// Trait for state vectors that can receive bolus doses.
pub trait State {
    /// Add a bolus dose to the state at the specified input compartment.
    ///
    /// # Parameters
    /// - `input`: The compartment index
    /// - `amount`: The bolus amount
    fn add_bolus(&mut self, input: usize, amount: f64);
}

/// Trait for prediction containers.
pub trait Predictions: Default {
    /// Create a new prediction container with specified capacity.
    ///
    /// # Parameters
    /// - `nparticles`: Number of particles (for SDE)
    ///
    /// # Returns
    /// A new predictions container
    fn new(_nparticles: usize) -> Self {
        Default::default()
    }

    /// Calculate the sum of squared errors for all predictions.
    ///
    /// # Returns
    /// The sum of squared errors
    fn squared_error(&self) -> f64;

    /// Get all predictions as a vector.
    ///
    /// # Returns
    /// Vector of prediction objects
    fn get_predictions(&self) -> Vec<Prediction>;
}

/// Trait defining the associated types for equations.
pub trait EquationTypes {
    /// The state vector type
    type S: State + Debug;
    /// The predictions container type
    type P: Predictions;
}

pub(crate) trait EquationPriv: EquationTypes {
    // fn get_init(&self) -> &Init;
    // fn get_out(&self) -> &Out;
    fn lag(&self) -> &Lag;
    fn fa(&self) -> &Fa;
    fn get_nstates(&self) -> usize;
    fn get_nouteqs(&self) -> usize;
    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError>;
    fn nparticles(&self) -> usize {
        1
    }
    #[allow(dead_code)]
    fn is_sde(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_models: Option<&ErrorModels>,
        time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError>;

    fn initial_state(
        &self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S;

    #[allow(clippy::too_many_arguments)]
    fn simulate_event(
        &self,
        support_point: &Vec<f64>,
        event: &Event,
        next_event: Option<&Event>,
        error_models: Option<&ErrorModels>,
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        match event {
            Event::Bolus(bolus) => {
                x.add_bolus(bolus.input(), bolus.amount());
            }
            Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            Event::Observation(observation) => {
                self.process_observation(
                    support_point,
                    observation,
                    error_models,
                    event.time(),
                    covariates,
                    x,
                    likelihood,
                    output,
                )?;
            }
        }

        if let Some(next_event) = next_event {
            self.solve(
                x,
                support_point,
                covariates,
                infusions,
                event.time(),
                next_event.time(),
            )?;
        }
        Ok(())
    }
}

/// Trait for model equations that can be simulated.
///
/// This trait defines the interface for different types of model equations
/// (ODE, SDE, analytical) that can be simulated to generate predictions
/// and estimate parameters.
#[allow(private_bounds)]
pub trait Equation: EquationPriv + 'static + Clone + Sync {
    /// Estimate the likelihood of the subject given the support point and error model.
    ///
    /// This function calculates how likely the observed data is given the model
    /// parameters and error model. It may use caching for performance.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    /// - `error_model`: The error model
    /// - `cache`: Whether to use caching
    ///
    /// # Returns
    /// The log-likelihood value
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError>;

    fn kind() -> EqnKind;

    /// Generate predictions for a subject with given parameters.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    ///
    /// # Returns
    /// Predicted concentrations
    fn estimate_predictions(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
    ) -> Result<Self::P, PharmsolError> {
        Ok(self.simulate_subject(subject, support_point, None)?.0)
    }

    /// Get the number of output equations in the model.
    fn nouteqs(&self) -> usize {
        self.get_nouteqs()
    }

    /// Get the number of state variables in the model.
    fn nstates(&self) -> usize {
        self.get_nstates()
    }

    /// Simulate a subject with given parameters and optionally calculate likelihood.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    /// - `error_model`: The error model (optional)
    ///
    /// # Returns
    /// A tuple containing predictions and optional likelihood
    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: Option<&ErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let mut output = Self::P::new(self.nparticles());
        let mut likelihood = Vec::new();
        for occasion in subject.occasions() {
            let covariates = occasion.covariates();

            let mut x = self.initial_state(support_point, covariates, occasion.index());
            let mut infusions = Vec::new();
            let events = occasion.process_events(
                Some((self.fa(), self.lag(), support_point, covariates)),
                true,
            );
            for (index, event) in events.iter().enumerate() {
                self.simulate_event(
                    support_point,
                    event,
                    events.get(index + 1),
                    error_models,
                    covariates,
                    &mut x,
                    &mut infusions,
                    &mut likelihood,
                    &mut output,
                )?;
            }
        }
        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub enum EqnKind {
    ODE = 0,
    Analytical = 1,
    SDE = 2,
}

impl EqnKind {
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::ODE => "EqnKind::ODE",
            Self::Analytical => "EqnKind::Analytical",
            Self::SDE => "EqnKind::SDE",
        }
    }
}
