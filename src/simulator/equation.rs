use std::collections::HashMap;
pub mod analytical;
pub mod ode;
pub mod sde;

pub use analytical::*;
pub use ode::*;
pub use sde::*;

use crate::{error_model::ErrorModel, Covariates, Event, Infusion, Observation, Subject};

use super::{likelihood::Prediction, Init, Out};

pub trait SimulationState {
    fn add_bolus(&mut self, input: usize, amount: f64);
}

pub trait Predictions: Default {
    fn new(_nparticles: usize) -> Self {
        Default::default()
    }
    fn squared_error(&self) -> f64;
    fn get_predictions(&self) -> &Vec<Prediction>;
}

pub trait Equation: 'static + Clone + Sync {
    type S: SimulationState;
    type P: Predictions;

    fn subject_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64;

    fn nparticles(&self) -> usize {
        1
    }

    fn is_sde(&self) -> bool {
        false
    }
    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    );
    fn get_init(&self) -> &Init;
    fn get_out(&self) -> &Out;
    fn get_lag(&self, spp: &[f64]) -> HashMap<usize, f64>;
    fn get_fa(&self, spp: &[f64]) -> HashMap<usize, f64>;
    fn get_nstates(&self) -> usize;
    fn get_nouteqs(&self) -> usize;
    #[doc(hidden)]
    fn _process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    );
    #[doc(hidden)]
    fn _simulate_event(
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
                self._process_observation(
                    support_point,
                    observation,
                    error_model,
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
                &infusions,
                event.get_time(),
                next_event.get_time(),
            );
        }
    }
    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: Option<&ErrorModel>,
    ) -> (Self::P, Option<f64>) {
        let lag_closure = self.get_lag(support_point);
        let fa_closure = self.get_fa(support_point);
        let mut output = Self::P::new(self.nparticles());
        let mut likelihood = Vec::new();
        for occasion in subject.occasions() {
            let covariates = occasion.get_covariates().unwrap();
            let mut x = self._initial_state(support_point, covariates, occasion.index());
            let mut infusions = Vec::new();
            let events = occasion.get_events(Some(&lag_closure), Some(&fa_closure), true);
            for (index, event) in events.iter().enumerate() {
                self._simulate_event(
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
        let ll = match error_model {
            Some(_) => Some(likelihood.iter().product::<f64>()),
            None => None,
        };
        (output, ll)
    }
    #[doc(hidden)]
    fn _initial_state(
        &self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S;
}
