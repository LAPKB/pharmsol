mod closure;

use core::panic;
use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use crate::simulator::equation::Predictions;
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, Bdf, NewtonNonlinearSolver,
    OdeBuilder, OdeSolverStopReason,
};
use nalgebra::DVector;

use super::{Equation, EquationPriv, EquationTypes, State};

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

impl State for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
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
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#
)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> SubjectPredictions {
    ode.simulate_subject(subject, support_point, None).0
}

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
    cache: bool,
) -> f64 {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    };
    ypred.likelihood(error_model)
}

impl EquationTypes for ODE {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for ODE {
    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(spp.to_vec());
        Some((self.lag)(&spp))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(spp.to_vec());
        Some((self.fa)(&spp))
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn solve(
        &self,
        _state: &mut Self::S,
        _support_point: &Vec<f64>,
        _covariates: &Covariates,
        _infusions: &Vec<Infusion>,
        _start_time: f64,
        _end_time: f64,
    ) {
        unimplemented!("solve not implemented for ODE");
    }
    #[inline(always)]
    fn process_observation(
        &self,
        _support_point: &Vec<f64>,
        _observation: &Observation,
        _error_model: Option<&ErrorModel>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) {
        unimplemented!("process_observation not implemented for ODE");
    }

    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            let spp = DVector::from_vec(spp.clone());
            (init)(&spp, 0.0, covariates, &mut x);
        }
        x
    }
}

impl Equation for ODE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        _estimate_likelihood(self, subject, support_point, error_model, cache)
    }

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
            let infusions = occasion.infusions_ref();
            let events = occasion.get_events(&lag, &fa, true);

            let problem = OdeBuilder::<M>::new()
                .atol(vec![ATOL])
                .rtol(RTOL)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.clone())
                .build_from_eqn(PMProblem::new(
                    self.diffeq,
                    self.get_nstates(),
                    support_point.clone(), //TODO: Avoid cloning the support point
                    covariates,
                    infusions,
                    self.initial_state(support_point, covariates, occasion.index()),
                ))
                .unwrap();

            let mut solver: Bdf<
                '_,
                PMProblem<DiffEq>,
                NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
            > = problem.bdf::<diffsol::NalgebraLU<f64>>().unwrap(); // TODO: Result

            for (index, event) in events.iter().enumerate() {
                let next_event = events.get(index + 1);
                //START SIMULATE_EVENT
                match event {
                    Event::Bolus(bolus) => {
                        // x.add_bolus(bolus.input(), bolus.amount());
                        solver.state_mut().y[bolus.input()] += bolus.amount();
                    }
                    Event::Infusion(_infusion) => {
                        // infusions.push(infusion.clone());
                    }
                    Event::Observation(observation) => {
                        //START PROCESS_OBSERVATION
                        let mut y = V::zeros(self.get_nouteqs());
                        let out = &self.out;
                        let spp = DVector::from_vec(support_point.clone()); // TODO: Avoid clone
                        (out)(
                            solver.state().y,
                            &spp,
                            observation.time(),
                            covariates,
                            &mut y,
                        );
                        let pred = y[observation.outeq()];
                        let pred =
                            observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                        if let Some(error_model) = error_model {
                            likelihood.push(pred.likelihood(error_model));
                        }
                        output.add_prediction(pred);
                        //END PROCESS_OBSERVATION
                    }
                }
                // START SOLVE
                if let Some(next_event) = next_event {
                    match solver.set_stop_time(next_event.time()) {
                        Ok(_) => loop {
                            let ret = solver.step();
                            match ret {
                                Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                Ok(OdeSolverStopReason::TstopReached) => break,
                                _ => panic!("Unexpected solver error: {:?}", ret),
                            }
                        },
                        Err(e) => {
                            match e {
                                diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StopTimeAtCurrentTime,
                                ) => {
                                    // If the stop time is at the current state time, we can just continue
                                    continue;
                                }
                                _ => panic!("Unexpected solver error: {:?}", e),
                            }
                        }
                    }
                }
                //End SOLVE
                //END SIMULATE_EVENT
            }
        }
        let ll = error_model.map(|_| likelihood.iter().product::<f64>());
        (output, ll)
    }
}

// // Test spphash
// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn test_spphash() {
//         let spp1 = vec![1.0, 2.0, 3.0];
//         let spp2 = vec![1.0, 2.0, 3.0];
//         let spp3 = vec![3.0, 2.0, 1.0];
//         let spp4 = vec![1.0, 2.0, 3.000001];
//         // Equal values should have the same hash
//         assert_eq!(spphash(&spp1), spphash(&spp2));
//         // Mirrored values should have different hashes
//         assert_ne!(spphash(&spp1), spphash(&spp3));
//         // Very close values should have different hashes
//         // Note: Due to f64 precision this will fail for values that are very close, e.g. 3.0 and 3.0000000000000001
//         assert_ne!(spphash(&spp1), spphash(&spp4));
//     }
// }
