mod closure;
mod diffsol_traits;

use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{likelihood::ToPrediction, DiffEq, Fa, Init, Lag, Neqs, Out, M, T, V},
    Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf};

use self::diffsol_traits::build_pm_ode;

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
        Some((self.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.fa)(&V::from_vec(spp.to_owned())))
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
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    ) {
        if f64::abs(start_time - end_time) < 1e-8 {
            return;
        }
        // dbg!(start_time, end_time);
        let problem = build_pm_ode::<M, _, _>(
            self.diffeq,
            |_p: &V, _t: T| state.clone(),
            V::from_vec(support_point.to_vec()),
            start_time,
            1e-3,
            RTOL,
            ATOL,
            covariates.clone(),
            infusions.clone(),
        )
        .unwrap();

        let mut solver = Bdf::default();
        let sol = solver.solve(&problem, end_time).unwrap();
        *state = sol.0.last().unwrap().clone()
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.clone()),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred);
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }

    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
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
