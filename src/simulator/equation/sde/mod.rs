mod em;

use std::collections::HashMap;

use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
const STEPS: usize = 10;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::Prediction,
    simulator::{likelihood::ToPrediction, Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Subject,
};

use super::{Equation, Predictions, SimulationState};

#[inline(always)]
pub(crate) fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    support_point: &[f64],
    _cov: &Covariates,
    _infusions: &[Infusion],
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }

    let mut sde = em::EM::new(
        drift.clone(),
        difussion.clone(),
        DVector::from_column_slice(support_point),
        x,
    );
    let solution = sde.solve(ti, tf, STEPS);
    solution.last().unwrap().clone().into()
}

#[derive(Clone)]
pub struct SDE {
    drift: Drift,
    diffusion: Diffusion,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    nparticles: usize,
}

impl SDE {
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: Neqs,
        nparticles: usize,
    ) -> Self {
        Self {
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
            neqs,
            nparticles,
        }
    }
    pub fn simulate_trajectories(&self, subject: &Subject, spp: &Vec<f64>) -> Array2<Prediction> {
        self.simulate_subject(subject, spp, None).0
    }
}
impl SimulationState for Vec<DVector<f64>> {
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.par_iter_mut().for_each(|particle| {
            particle[input] += amount;
        });
    }
}

impl Predictions for Array2<Prediction> {
    fn new(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }
    fn squared_error(&self) -> f64 {
        unimplemented!();
    }
    fn get_predictions(&self) -> &Vec<Prediction> {
        unimplemented!();
    }
    fn likelihood(&self, _error_model: &ErrorModel) -> f64 {
        unimplemented!();
    }
}

impl Equation for SDE {
    type S = Vec<DVector<f64>>; // Vec -> particles, DVector -> state
    type P = Array2<Prediction>; // Rows -> particles, Columns -> time

    fn nparticles(&self) -> usize {
        self.nparticles
    }

    fn is_sde(&self) -> bool {
        true
    }

    #[inline(always)]
    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        ti: f64,
        tf: f64,
    ) {
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone(),
                support_point,
                covariates,
                &infusions,
                ti,
                tf,
            );
        });
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
        observation: &crate::Observation,
        error_model: Option<&ErrorModel>,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut pred = vec![Prediction::default(); self.nparticles];
        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.get_nouteqs());
            (self.out)(
                &x[i],
                &V::from_vec(support_point.clone()),
                observation.time(),
                covariates,
                &mut y,
            );
            *p = observation.to_obs_pred(y[observation.outeq()]);
        });
        let out = Array2::from_shape_vec((self.nparticles, 1), pred.clone()).unwrap();
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_model {
            let mut q: Vec<f64> = Vec::with_capacity(self.nparticles);

            pred.iter().for_each(|p| q.push(p.likelihood(em)));
            let sum_q: f64 = q.iter().sum();
            //py = (1 / Np) * sum(q)
            let py = sum_q / self.nparticles as f64;
            //b[t] = log(py)
            // ll.push(py.ln());
            likelihood.push(py);
            //q = q ./ sum(q)
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            //ind = sysresample(q)
            let i = sysresample(&w);
            //x = x[ind,:]
            let a: Vec<DVector<f64>> = i.iter().map(|&i| x[i].clone()).collect();
            *x = a;
        }
    }

    #[inline(always)]
    fn _initial_state(
        &self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S {
        let mut x = Vec::with_capacity(self.nparticles);
        for _ in 0..self.nparticles {
            let mut state = DVector::zeros(self.get_nstates());
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(support_point.to_vec()),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state);
        }
        x
    }
}
fn sysresample(q: &[f64]) -> Vec<usize> {
    let mut qc = vec![0.0; q.len()];
    qc[0] = q[0];
    for i in 1..q.len() {
        qc[i] = qc[i - 1] + q[i];
    }
    let m = q.len();
    let mut rng = thread_rng();
    let u: Vec<f64> = (0..m)
        .map(|i| (i as f64 + rng.gen::<f64>()) / m as f64)
        .collect();
    let mut i = vec![0; m];
    let mut k = 0;
    for j in 0..m {
        while qc[k] < u[j] {
            k += 1;
        }
        i[j] = k;
    }
    i
}