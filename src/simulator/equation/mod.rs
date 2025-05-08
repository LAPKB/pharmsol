use std::{collections::HashMap, fmt::Debug};
pub mod analytical;
pub mod meta;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use meta::*;
pub use ode::*;
pub use sde::*;

use crate::{error_model::ErrorModel, Covariates, Event, Infusion, Observation, Subject};

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
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>>;
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>>;
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
    );
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
        error_model: Option<&ErrorModel>,
        time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    );

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
        error_model: Option<&ErrorModel>,
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
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
                    error_model,
                    event.time(),
                    covariates,
                    x,
                    likelihood,
                    output,
                );
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
            );
        }
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
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64;

    /// Generate predictions for a subject with given parameters.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    ///
    /// # Returns
    /// Predicted concentrations
    fn estimate_predictions(&self, subject: &Subject, support_point: &Vec<f64>) -> Self::P {
        self.simulate_subject(subject, support_point, None).0
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
        error_model: Option<&ErrorModel>,
    ) -> (Self::P, Option<f64>) {
        let lag = self.get_lag(support_point);
        let fa = self.get_fa(support_point);
        let mut output = Self::P::new(self.nparticles());
        let mut likelihood = Vec::new();
        for occasion in subject.occasions() {
            let covariates = occasion.get_covariates().unwrap();
            let mut x = self.initial_state(support_point, covariates, occasion.index());
            let mut infusions = Vec::new();
            let events = occasion.get_events(&lag, &fa, true);
            for (index, event) in events.iter().enumerate() {
                self.simulate_event(
                    support_point,
                    event,
                    events.get(index + 1),
                    error_model,
                    covariates,
                    &mut x,
                    &mut infusions,
                    &mut likelihood,
                    &mut output,
                );
            }
        }
        let ll = error_model.map(|_| likelihood.iter().product::<f64>());
        (output, ll)
    }
}
