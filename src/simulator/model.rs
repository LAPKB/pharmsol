use std::collections::HashMap;

use crate::simulator::equation::Outputs;
use crate::{Covariates, Equation, ErrorModel, Event, Infusion, Observation, Subject};

// Define Model as a trait
pub trait Model<'a> {
    type Eq: Equation<'a>;

    fn new(equation: &'a Self::Eq, subject: &'a Subject, spp: &[f64]) -> Self
    where
        Self: Sized;

    fn equation(&self) -> &Self::Eq;
    fn subject(&self) -> &Subject;
    fn state(&self) -> &<Self::Eq as Equation<'a>>::S;
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>>;
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>>;
    fn solve(
        &mut self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    );

    fn process_observation(
        &mut self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation<'a>>::P,
    );

    fn initial_state(
        &mut self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    );
    fn add_bolus(&mut self, input: usize, amount: f64);

    fn simulate_event(
        &mut self,
        support_point: &Vec<f64>,
        event: &Event,
        next_event: Option<&Event>,
        error_model: Option<&ErrorModel>,
        covariates: &Covariates,
        infusions: &mut Vec<Infusion>,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation<'a>>::P,
    ) {
        match event {
            Event::Bolus(bolus) => {
                self.add_bolus(bolus.input(), bolus.amount());
            }
            Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            Event::Observation(observation) => {
                self.process_observation(
                    support_point,
                    observation,
                    error_model,
                    event.get_time(),
                    covariates,
                    likelihood,
                    output,
                );
            }
        }

        if let Some(next_event) = next_event {
            self.solve(
                support_point,
                covariates,
                infusions,
                event.get_time(),
                next_event.get_time(),
            );
        }
    }
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
        &mut self,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64;

    fn nparticles(&self) -> usize {
        //TODO: remove
        self.equation().nparticles()
    }

    /// Generate Outputs for a subject with given parameters.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    ///
    /// # Returns
    /// Predicted concentrations
    fn estimate_outputs(&mut self, support_point: &Vec<f64>) -> <Self::Eq as Equation<'a>>::P {
        self.simulate_subject(support_point, None).0
    }

    /// Simulate a subject with given parameters and optionally calculate likelihood.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    /// - `error_model`: The error model (optional)
    ///
    /// # Returns
    /// A tuple containing Outputs and optional likelihood
    fn simulate_subject(
        &mut self,
        support_point: &Vec<f64>,
        error_model: Option<&ErrorModel>,
    ) -> (<Self::Eq as Equation<'a>>::P, Option<f64>) {
        let lag = self.get_lag(support_point);
        let fa = self.get_fa(support_point);

        let mut output = <Self::Eq as Equation>::P::new_outputs(self.nparticles());
        let mut likelihood = Vec::new();

        for occasion in self.subject().clone().occasions() {
            let covariates = occasion.get_covariates().unwrap();
            self.initial_state(support_point, covariates, occasion.index());
            let mut infusions = Vec::new();
            let events = occasion.get_events(&lag, &fa, true);
            for (index, event) in events.iter().enumerate() {
                self.simulate_event(
                    support_point,
                    event,
                    events.get(index + 1),
                    error_model,
                    covariates,
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

// Implementation for ODE
