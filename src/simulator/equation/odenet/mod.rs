use std::collections::HashMap;
mod closure;
mod diffsol_traits;
mod outeq;
pub use outeq::*;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{likelihood::ToPrediction, T, V},
    Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf};
use diffsol_traits::build_network_ode;
use nalgebra::DMatrix;

use super::{Equation, EquationPriv, EquationTypes, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct ODENet {
    linear: Vec<DMatrix<f64>>,
    out: Vec<OutEq>,
    neqs: (usize, usize),
}

impl ODENet {
    pub fn new(linear: Vec<DMatrix<f64>>, out: Vec<OutEq>, neqs: (usize, usize)) -> Self {
        Self { linear, out, neqs }
    }
}

impl State for DMatrix<f64> {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}
fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#
)]
fn _subject_predictions(
    net: &ODENet,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> SubjectPredictions {
    net.simulate_subject(subject, support_point, None).0
}

fn _estimate_likelihood(
    net: &ODENet,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
    cache: bool,
) -> f64 {
    let ypred = if cache {
        _subject_predictions(net, subject, support_point)
    } else {
        _subject_predictions_no_cache(net, subject, support_point)
    };
    ypred.likelihood(error_model)
}

impl EquationTypes for ODENet {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for ODENet {
    #[inline(always)]
    fn get_lag(&self, _spp: &[f64]) -> HashMap<usize, f64> {
        Default::default()
    }

    #[inline(always)]
    fn get_fa(&self, _spp: &[f64]) -> HashMap<usize, f64> {
        Default::default()
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
        if start_time == end_time {
            return;
        }
        let problem = build_network_ode(
            self.linear.clone(),
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
        _covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        let point = V::from_vec(support_point.clone());
        // point.iter_mut().for_each(|x| *x = 1.0 / *x);
        // assuming v = p[2] and y[0] = x[1]/v
        for eq in out.iter() {
            eq.apply(&mut y, &point, x);
        }
        // y[0] = x[0] / point[1];

        // (out)(
        //     x,
        //     &V::from_vec(support_point.clone()),
        //     observation.time(),
        //     covariates,
        //     &mut y,
        // );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred);
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }
    #[inline(always)]
    fn initial_state(
        &self,
        _spp: &Vec<f64>,
        _covariates: &Covariates,
        _occasion_index: usize,
    ) -> V {
        // let init = &self.init;
        let x = V::zeros(self.get_nstates());
        // if occasion_index == 0 {
        //     (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        // }
        x
    }
}

impl Equation for ODENet {
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
