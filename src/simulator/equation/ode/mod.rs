mod closure;

use core::panic;
use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::{Prediction, SubjectPredictions},
    simulator::{likelihood::ToPrediction, model::Model, DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, Occasion, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use closure::PMProblem;
use diffsol::OdeSolverState;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, Bdf, BdfState,
    NewtonNonlinearSolver, OdeBuilder, OdeSolverProblem, OdeSolverStopReason,
};
use nalgebra::{DMatrix, DVector};

use super::{Equation, Predictions, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct ODE {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
}

pub struct ODEModel<'a> {
    equation: &'a ODE,
    subject: &'a Subject,
    support_point: Vec<f64>,
    problem: Option<OdeSolverProblem<PMProblem<'a, DiffEq>>>,
    state: Option<BdfState<DVector<f64>, DMatrix<f64>>>,
}

impl ODE {
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out, neqs: Neqs) -> Self {
        Self {
            diffeq,
            lag,
            fa,
            init,
            out,
            neqs,
        }
    }
}

impl State for BdfState<DVector<f64>, DMatrix<f64>> {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.as_mut().y[input] += amount;
    }
}

impl Predictions for SubjectPredictions {
    fn empty(_nparticles: usize) -> Self {
        SubjectPredictions::default()
    }

    fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .map(|p| p.squared_error())
            .sum::<f64>()
    }

    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }
}

impl<'a> Equation<'a> for ODE {
    type S = BdfState<DVector<f64>, DMatrix<f64>>;
    type P = SubjectPredictions;
    type Mod = ODEModel<'a>;
    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn initialize_model(&'a self, subject: &'a Subject, spp: Vec<f64>) -> Self::Mod {
        ODEModel::new(self, subject, spp)
    }
}

impl<'a> Model<'a> for ODEModel<'a> {
    type Eq = ODE;

    fn new(equation: &'a Self::Eq, subject: &'a Subject, spp: Vec<f64>) -> Self {
        Self {
            equation,
            subject,
            support_point: spp,
            problem: None,
            state: None,
        }
    }
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        match self.state {
            Some(ref mut state) => {
                state.add_bolus(input, amount);
            }
            None => {
                panic!("State is not initialized");
            }
        }
    }

    #[inline(always)]
    fn equation(&self) -> &Self::Eq {
        self.equation
    }

    #[inline(always)]
    fn subject(&self) -> &Subject {
        self.subject
    }
    #[inline(always)]
    fn get_lag(&self) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(self.support_point.to_vec());
        Some((self.equation.lag)(&spp))
    }

    #[inline(always)]
    fn get_fa(&self) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(self.support_point.to_vec());
        Some((self.equation.fa)(&spp))
    }
    #[inline(always)]
    fn solve(
        &mut self,
        _covariates: &Covariates,
        _infusions: Vec<&Infusion>,
        _start_time: f64,
        end_time: f64,
    ) {
        let mut solver = self
            .problem
            .as_mut()
            .unwrap()
            .bdf_solver::<diffsol::NalgebraLU<f64>>(self.state.take().unwrap())
            .unwrap();
        match solver.set_stop_time(end_time) {
            Ok(_) => loop {
                let ret = solver.step();
                match ret {
                    Ok(OdeSolverStopReason::InternalTimestep) => continue,
                    Ok(OdeSolverStopReason::TstopReached) => break,
                    _ => panic!("Unexpected solver error: {:?}", ret),
                }
            },
            Err(e) => match e {
                diffsol::error::DiffsolError::OdeSolverError(
                    OdeSolverError::StopTimeAtCurrentTime,
                ) => {}
                _ => panic!("Unexpected solver error: {:?}", e),
            },
        }
        self.state = Some(solver.into_state());
    }
    #[inline(always)]
    fn process_observation(
        &mut self,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation>::P,
    ) {
        let mut y = V::zeros(self.equation.get_nouteqs());
        let out = &self.equation.out;
        let spp = DVector::from_vec(self.support_point.clone()); // TODO: Avoid clone
        (out)(
            self.state.as_ref().unwrap().as_ref().y,
            &spp,
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(
            pred,
            self.state.as_ref().unwrap().as_ref().y.as_slice().to_vec(),
        );
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }

    #[inline(always)]
    fn initial_state(&mut self, occasion: &'a Occasion) {
        let covariates = occasion.get_covariates().unwrap();
        let infusions = occasion.infusions_ref();

        let init = &self.equation.init;
        let mut x = V::zeros(self.equation.get_nstates());
        if occasion.index() == 0 {
            let spp = DVector::from_vec(self.support_point.clone());
            (init)(&spp, 0.0, covariates, &mut x);
        }

        let problem = OdeBuilder::<M>::new()
            .atol(vec![ATOL])
            .rtol(RTOL)
            .t0(occasion.initial_time())
            .h0(1e-3)
            .p(self.support_point.clone())
            .build_from_eqn(PMProblem::new(
                self.equation.diffeq,
                self.equation.get_nstates(),
                self.support_point.clone(), //TODO: Avoid cloning the support point
                &covariates,
                infusions,
                x,
            ))
            .unwrap();

        self.problem = Some(problem);
        let solver = self
            .problem
            .as_ref()
            .unwrap()
            .bdf::<diffsol::NalgebraLU<f64>>()
            .unwrap(); // TODO: Result
        let state: BdfState<DVector<f64>, DMatrix<f64>> = solver.into_state();

        self.state = Some(state);
    }
    fn estimate_likelihood(self, error_model: &ErrorModel, cache: bool) -> f64 {
        _estimate_likelihood(self, error_model, cache)
    }
    // fn simulate_subject(
    //     mut self,
    //     error_model: Option<&ErrorModel>,
    // ) -> (<Self::Eq as Equation<'a>>::P, Option<f64>)
    // where
    //     Self: Sized + 'a,
    // {
    //     let lag = self.get_lag();
    //     let fa = self.get_fa();
    //     let mut output = <Self::Eq as Equation<'a>>::P::empty(self.nparticles());
    //     let mut likelihood = Vec::new();
    //     let occasions = self.subject().occasions().clone();

    //     for occasion in occasions {
    //         let covariates = occasion.get_covariates().unwrap();
    //         let infusions = occasion.infusions_ref();
    //         let events = occasion.get_events(&lag, &fa, true);

    //         let problem: OdeSolverProblem<PMProblem<'_, DiffEq>> = OdeBuilder::<M>::new()
    //             .atol(vec![ATOL])
    //             .rtol(RTOL)
    //             .t0(occasion.initial_time())
    //             .h0(1e-3)
    //             .p(self.support_point.clone())
    //             .build_from_eqn(PMProblem::new(
    //                 self.equation.diffeq,
    //                 self.equation.get_nstates(),
    //                 self.support_point.clone(), //TODO: Avoid cloning the support point
    //                 &covariates,
    //                 infusions,
    //                 self.initial_state(support_point, covariates, occasion.index()),
    //             ))
    //             .unwrap();

    //         let mut solver: Bdf<
    //             '_,
    //             PMProblem<DiffEq>,
    //             NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
    //         > = problem.bdf::<diffsol::NalgebraLU<f64>>().unwrap(); // TODO: Result

    //         let state: BdfState<DVector<f64>, DMatrix<f64>> = solver.into_state();
    //         let mut solver = problem.bdf_solver(state);
    //         for (index, event) in events.iter().enumerate() {
    //             let next_event = events.get(index + 1);
    //             //START SIMULATE_EVENT
    //             match event {
    //                 Event::Bolus(bolus) => {
    //                     // x.add_bolus(bolus.input(), bolus.amount());
    //                     solver.state_mut().y[bolus.input()] += bolus.amount();
    //                 }
    //                 Event::Infusion(_infusion) => {
    //                     // infusions.push(infusion.clone());
    //                 }
    //                 Event::Observation(observation) => {
    //                     //START PROCESS_OBSERVATION
    //                     let mut y = V::zeros(self.get_nouteqs());
    //                     let out = &self.out;
    //                     let spp = DVector::from_vec(support_point.clone()); // TODO: Avoid clone
    //                     (out)(
    //                         solver.state().y,
    //                         &spp,
    //                         observation.time(),
    //                         covariates,
    //                         &mut y,
    //                     );
    //                     let pred = y[observation.outeq()];
    //                     let pred =
    //                         observation.to_obs_pred(pred, solver.state().y.as_slice().to_vec());
    //                     if let Some(error_model) = error_model {
    //                         likelihood.push(pred.likelihood(error_model));
    //                     }
    //                     output.add_prediction(pred);
    //                     //END PROCESS_OBSERVATION
    //                 }
    //             }
    //             // START SOLVE
    //             if let Some(next_event) = next_event {
    //                 match solver.set_stop_time(next_event.get_time()) {
    //                     Ok(_) => loop {
    //                         let ret = solver.step();
    //                         match ret {
    //                             Ok(OdeSolverStopReason::InternalTimestep) => continue,
    //                             Ok(OdeSolverStopReason::TstopReached) => break,
    //                             _ => panic!("Unexpected solver error: {:?}", ret),
    //                         }
    //                     },
    //                     Err(e) => {
    //                         match e {
    //                             diffsol::error::DiffsolError::OdeSolverError(
    //                                 OdeSolverError::StopTimeAtCurrentTime,
    //                             ) => {
    //                                 // If the stop time is at the current state time, we can just continue
    //                                 continue;
    //                             }
    //                             _ => panic!("Unexpected solver error: {:?}", e),
    //                         }
    //                     }
    //                 }
    //             }
    //             //End SOLVE
    //             //END SIMULATE_EVENT
    //         }
    //     }
    //     let ll = error_model.map(|_| likelihood.iter().product::<f64>());
    //     (output, ll)
    // }
}

// Hash the support points by converting them to bits and summing them
// The wrapping_add is used to avoid overflow, and prevent panics
fn spphash(spp: &[f64]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    spp.iter().for_each(|&value| {
        // Normalize negative zero to zero, e.g. -0.0 -> 0.0
        let normalized_value = if value == 0.0 && value.is_sign_negative() {
            0.0
        } else {
            value
        };
        // Convert the value to bits and hash it
        let bits = normalized_value.to_bits();
        std::hash::Hash::hash(&bits, &mut hasher);
    });

    std::hash::Hasher::finish(&hasher)
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", model.subject.id(), spphash(&model.support_point)) }"#
)]
fn _subject_predictions(model: ODEModel) -> SubjectPredictions {
    model.simulate_subject(None).0
}

fn _estimate_likelihood(model: ODEModel, error_model: &ErrorModel, cache: bool) -> f64 {
    let ypred = if cache {
        _subject_predictions(model)
    } else {
        _subject_predictions_no_cache(model)
    };
    ypred.likelihood(error_model)
}

// Test spphash
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_spphash() {
        let spp1 = vec![1.0, 2.0, 3.0];
        let spp2 = vec![1.0, 2.0, 3.0];
        let spp3 = vec![3.0, 2.0, 1.0];
        let spp4 = vec![1.0, 2.0, 3.000001];
        // Equal values should have the same hash
        assert_eq!(spphash(&spp1), spphash(&spp2));
        // Mirrored values should have different hashes
        assert_ne!(spphash(&spp1), spphash(&spp3));
        // Very close values should have different hashes
        // Note: Due to f64 precision this will fail for values that are very close, e.g. 3.0 and 3.0000000000000001
        assert_ne!(spphash(&spp1), spphash(&spp4));
    }
}
