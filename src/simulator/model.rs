use std::collections::HashMap;

use crate::simulator::equation::*;
use crate::{Covariates, Equation, ErrorModel, Event, Infusion, Observation, Occasion, Subject};
// Define Model as a trait
pub trait Model<'a>
where
    Self: 'a,
{
    type Eq: Equation<'a>;

    fn new(equation: &'a Self::Eq, subject: &'a Subject, spp: Vec<f64>) -> Self
    where
        Self: Sized;

    // fn equation(&self) -> &Self::Eq;
    // fn subject(&self) -> &Subject;
    // fn state(&mut self) -> &mut <Self::Eq as Equation<'a>>::S;
    fn get_lag(&self) -> Option<HashMap<usize, f64>>;
    fn get_fa(&self) -> Option<HashMap<usize, f64>>;
    fn solve(
        &mut self,
        covariates: &Covariates,
        infusions: Vec<&Infusion>,
        start_time: f64,
        end_time: f64,
    );

    fn process_observation(
        &mut self,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation<'a>>::P,
    );

    fn initial_state(&mut self, occasion: &Occasion);

    fn add_bolus(&mut self, input: usize, amount: f64);

    fn simulate_event(
        &mut self,
        event: &Event,
        next_event: Option<&Event>,
        error_model: Option<&ErrorModel>,
        occasion: &Occasion,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation<'a>>::P,
    ) where
        Self: Sized,
    {
        let binding = Covariates::new();
        let covariates = occasion.get_covariates().unwrap_or(&binding);
        let infusions = occasion.infusions_ref();
        match event {
            Event::Bolus(bolus) => {
                self.add_bolus(bolus.input(), bolus.amount());
            }
            Event::Infusion(_infusion) => {
                //Nothing to do here
            }
            Event::Observation(observation) => {
                self.process_observation(
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
    fn estimate_likelihood(self, error_model: &ErrorModel, cache: bool) -> f64;

    fn nparticles(&self) -> usize {
        //TODO: remove
        self.equation().nparticles()
    }

    fn equation(&self) -> &Self::Eq;

    /// Generate Outputs for a subject with given parameters.
    ///
    /// # Parameters
    /// - `subject`: The subject data
    /// - `support_point`: The parameter values
    ///
    /// # Returns
    /// Predicted concentrations
    fn estimate_outputs(self) -> <Self::Eq as Equation<'a>>::P
    where
        Self: Sized,
    {
        self.simulate_subject(None).0
    }

    fn subject(&self) -> &Subject;

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
        mut self,
        error_model: Option<&ErrorModel>,
    ) -> (<Self::Eq as Equation<'a>>::P, Option<f64>)
    where
        Self: Sized,
    {
        let lag = self.get_lag();
        let fa = self.get_fa();

        let mut output = <Self::Eq as Equation>::P::empty(self.nparticles());
        let mut likelihood = Vec::new();
        // Clone the occasions so we don’t immutably borrow `self` across mutable calls.
        let mut occasions: Vec<Occasion> = vec![];
        for occasion in self.subject().occasions() {
            // Clone the occasion to avoid borrowing issues
            occasions.push(occasion.clone());
        }

        for occasion in occasions {
            self.initial_state(&occasion);

            let events = occasion.get_events(&lag, &fa, true);
            for (index, event) in events.iter().enumerate() {
                self.simulate_event(
                    event,
                    events.get(index + 1),
                    error_model,
                    &occasion,
                    &mut likelihood,
                    &mut output,
                );
            }
        }
        let ll = error_model.map(|_| likelihood.iter().product::<f64>());
        (output, ll)
    }
}
