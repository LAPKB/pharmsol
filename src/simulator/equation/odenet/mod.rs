use std::collections::HashMap;
mod closure;
mod diffsol_traits;
mod operations;
pub use operations::*;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{likelihood::ToPrediction, T, V},
    Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use diffsol::{ode_solver::method::OdeSolverMethod, OdeSolverState};
use diffsol_traits::build_network_ode;
use nalgebra::{DMatrix, DVector};

use super::{Equation, EquationPriv, EquationTypes, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct ODENet {
    linear: Vec<DMatrix<f64>>,
    non_linear: Vec<NL>,
    lag: Vec<Lag>,
    fa: Vec<Fa>,
    secondary_equations: Vec<SEq>,
    init: Vec<Init>,
    out: Vec<OutEq>,
    neqs: (usize, usize),
}

impl ODENet {
    pub fn new(
        linear: Vec<DMatrix<f64>>,
        secondary_equations: Vec<SEq>,
        non_linear: Vec<NL>,
        lag: Vec<Lag>,
        fa: Vec<Fa>,
        init: Vec<Init>,
        out: Vec<OutEq>,
        neqs: (usize, usize),
    ) -> Self {
        Self {
            linear,
            non_linear,
            secondary_equations,
            fa,
            lag,
            init,
            out,
            neqs,
        }
    }

    fn calculate_secondary(
        &self,
        point: &DVector<f64>,
        covs: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        self.secondary_equations
            .clone()
            .into_iter()
            .fold(HashMap::new(), |mut acc, x| {
                x.apply(&mut acc, point, covs);
                acc
            })
    }

    fn calculate_non_linear(
        &self,
        p: &DVector<f64>,
        x: &DVector<f64>,
        covs: &HashMap<String, f64>,
    ) -> DVector<f64> {
        self.non_linear
            .clone()
            .into_iter()
            .fold(DVector::zeros(x.len()), |mut acc, nl| {
                nl.apply(&mut acc, p, covs);
                acc
            })
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
    fn get_lag(&self, p: &[f64]) -> Option<HashMap<usize, f64>> {
        let p = V::from_vec(p.to_owned());
        match self.lag.len() {
            0 => None,
            _ => Some(
                self.lag
                    .clone()
                    .into_iter()
                    .fold(HashMap::new(), |mut acc, x| {
                        x.apply(&mut acc, &p);
                        acc
                    }),
            ),
        }
    }

    #[inline(always)]
    fn get_fa(&self, _spp: &[f64]) -> Option<HashMap<usize, f64>> {
        let p = V::from_vec(_spp.to_owned());
        match self.fa.len() {
            0 => None,
            _ => Some(
                self.fa
                    .clone()
                    .into_iter()
                    .fold(HashMap::new(), |mut acc, x| {
                        x.apply(&mut acc, &p);
                        acc
                    }),
            ),
        }
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
        let nl = self.calculate_non_linear(
            &V::from_vec(support_point.clone()),
            state,
            &covariates.to_hashmap(start_time),
        );
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
            nl,
        )
        .unwrap();
        let mut solver = diffsol::Bdf::default();
        // let tableau: diffsol::Tableau<DMatrix<f64>> = diffsol::Tableau::esdirk34();
        // let tableau: diffsol::Tableau<DMatrix<f64>> = diffsol::Tableau::tr_bdf2();
        // let mut solver = diffsol::Sdirk::new(tableau, diffsol::NalgebraLU::default());
        let st = OdeSolverState::new(&problem, &solver).unwrap();
        solver.set_problem(st, &problem).unwrap();
        while solver.state().unwrap().t <= end_time {
            solver.step().unwrap();
        }
        *state = solver.interpolate(end_time).unwrap();
    }

    #[inline(always)]
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
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        let point = V::from_vec(support_point.clone());
        let cov = covariates.to_hashmap(time);
        let sec = self.calculate_secondary(&point, &cov);
        for eq in out.iter() {
            eq.apply(&mut y, &point, x, &cov, &sec);
        }
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred, x.as_slice().to_vec());
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }
    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            for eq in self.init.iter() {
                let p = V::from_vec(spp.clone());
                eq.apply(&mut x, &p, covariates);
            }
        }
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
