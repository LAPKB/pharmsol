mod closure;
mod diffsol_traits;

use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{likelihood::ToPrediction, DiffEq, Fa, Init, Lag, Neqs, Out, M, T, V},
    Observation,
};

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf};

use self::diffsol_traits::build_pm_ode;

use super::{Equation, SimulationState};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[inline(always)]
pub(crate) fn simulate_ode_event(
    diffeq: &DiffEq,
    x: &mut V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ti: f64,
    tf: f64,
) {
    if ti == tf {
        return;
    }
    let problem = build_pm_ode::<M, _, _>(
        *diffeq,
        |_p: &V, _t: T| x.clone(),
        V::from_vec(support_point.to_vec()),
        ti,
        1e-3,
        RTOL,
        ATOL,
        cov.clone(),
        infusions.to_owned(),
    )
    .unwrap();
    let mut solver = Bdf::default();
    let sol = solver.solve(&problem, tf).unwrap();
    *x = sol.0.last().unwrap().clone()
}

#[derive(Clone)]
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

impl SimulationState for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

impl Equation for ODE {
    type S = V;
    type P = SubjectPredictions;

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
    fn get_init(&self) -> &Init {
        &self.init
    }

    #[inline(always)]
    fn get_out(&self) -> &Out {
        &self.out
    }

    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> HashMap<usize, f64> {
        (self.lag)(&V::from_vec(spp.to_owned()))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> HashMap<usize, f64> {
        (self.fa)(&V::from_vec(spp.to_owned()))
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
    fn _process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = self.get_out();
        (out)(
            &x,
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
    fn _initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = self.get_init();
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        }
        x
    }
}
