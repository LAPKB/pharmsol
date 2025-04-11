mod closure;

use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::{Prediction, SubjectPredictions},
    simulator::{likelihood::ToPrediction, model::Model, DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use closure::PMProblem;
use diffsol::{ode_solver::method::OdeSolverMethod, Bdf, NewtonNonlinearSolver, OdeBuilder};
use nalgebra::DVector;

use super::{Equation, Outputs, State};

// use self::diffsol_traits::build_pm_ode;

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
    data: &'a Subject,
    state: DVector<f64>,
    spp: &'a [f64],
}
impl State for DVector<f64> {}
impl Outputs for SubjectPredictions {
    fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .map(|p| (p.observation - p.prediction).powi(2))
            .sum()
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }
}
impl<'a> Equation<'a> for ODE {
    type S = V;
    type P = SubjectPredictions;
    type Mod = ODEModel<'a>;
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    fn initialize_model(&'a self, subject: &'a Subject, spp: &'a [f64]) -> Self::Mod {
        ODEModel::new(self, subject, spp)
    }
}

impl<'a> Model<'a> for ODEModel<'a> {
    type Eq = ODE;

    fn new(equation: &'a ODE, data: &'a Subject, spp: &'a [f64]) -> Self {
        Self {
            equation,
            data,
            state: DVector::default(),
            spp,
        }
    }

    fn equation(&self) -> &ODE {
        self.equation
    }

    fn subject(&self) -> &Subject {
        self.data
    }

    fn state(&self) -> &<Self::Eq as Equation>::S {
        &self.state
    }
    #[inline(always)]
    fn get_lag(&self) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(self.spp.to_vec());
        Some((self.equation.lag)(&spp))
    }

    #[inline(always)]
    fn get_fa(&self) -> Option<HashMap<usize, f64>> {
        let spp = DVector::from_vec(self.spp.to_vec());
        Some((self.equation.fa)(&spp))
    }

    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.state[input] += amount;
    }
    #[inline(always)]
    fn solve(
        &mut self,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    ) {
        if f64::abs(start_time - end_time) < 1e-8 {
            return;
        }

        let problem = OdeBuilder::<M>::new()
            .atol(vec![ATOL])
            .rtol(RTOL)
            .t0(start_time)
            .h0(1e-3)
            .p(self.spp.to_vec())
            .build_from_eqn(PMProblem::new(
                self.equation.diffeq,
                self.equation.get_nstates(),
                self.spp.to_vec(),
                covariates.clone(),
                infusions.clone(),
                self.state.clone(),
            ))
            .unwrap();

        let mut solver: Bdf<
            '_,
            PMProblem<DiffEq>,
            NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
        > = problem.bdf::<diffsol::NalgebraLU<f64>>().unwrap();
        let (ys, _ts) = solver.solve(end_time).unwrap();

        self.state = ys.column(ys.ncols() - 1).into_owned();
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
        let spp = DVector::from_vec(self.spp.to_vec());
        (out)(&self.state, &spp, observation.time(), covariates, &mut y);
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred, self.state.as_slice().to_vec());
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }

    #[inline(always)]
    fn initial_state(&mut self, covariates: &Covariates, occasion_index: usize) {
        let init = &self.equation.init;
        let mut x = V::zeros(self.equation.get_nstates());
        if occasion_index == 0 {
            let spp = DVector::from_vec(self.spp.to_vec());
            (init)(&spp, 0.0, covariates, &mut x);
        }
        self.state = x;
    }
    fn estimate_likelihood(&mut self, error_model: &ErrorModel, cache: bool) -> f64 {
        _estimate_likelihood(self, error_model, cache)
    }
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
    convert = r#"{ format!("{}{}", model.data.id(), spphash(model.spp)) }"#
)]
fn _subject_predictions(model: &mut ODEModel) -> SubjectPredictions {
    model.simulate_subject(None).0
}

fn _estimate_likelihood(model: &mut ODEModel, error_model: &ErrorModel, cache: bool) -> f64 {
    let ypred = if cache {
        _subject_predictions(model)
    } else {
        _subject_predictions_no_cache(model)
    };
    ypred.likelihood(error_model)
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
