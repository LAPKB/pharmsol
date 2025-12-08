mod em;

use std::collections::HashMap;

use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use rand::{rng, Rng};
use rayon::prelude::*;

use cached::proc_macro::cached;
use cached::UnboundCache;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::Prediction,
    simulator::{likelihood::ToPrediction, Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Subject,
};

use super::{Equation, EquationPriv, EquationTypes, Predictions, State};

#[inline(always)]
pub(crate) fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }

    let mut sde = em::EM::new(
        *drift,
        *difussion,
        DVector::from_column_slice(support_point),
        x,
        cov.clone(),
        infusions.to_vec(),
        1e-2,
        1e-2,
    );
    let (_time, solution) = sde.solve(ti, tf);
    solution.last().unwrap().clone()
}

#[derive(Clone, Debug)]
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
}
impl State for Vec<DVector<f64>> {
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
    fn get_predictions(&self) -> Vec<Prediction> {
        //TODO: This is only returning the first particle, not the best, not the worst, THE FIRST
        // CHANGE THIS
        // let row = self.row(0).to_vec();
        // row
        // Make this return the mean prediction across all particles
        if self.is_empty() || self.ncols() == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.ncols());

        for col in 0..self.ncols() {
            let column = self.column(col);
            let mean_prediction: f64 = column
                .iter()
                .map(|pred: &Prediction| pred.prediction())
                .sum::<f64>()
                / self.nrows() as f64;

            let mut prediction = column.first().unwrap().clone();
            prediction.set_prediction(mean_prediction);
            result.push(prediction);
        }

        result
    }
}

impl EquationTypes for SDE {
    type S = Vec<DVector<f64>>; // Vec -> particles, DVector -> state
    type P = Array2<Prediction>; // Rows -> particles, Columns -> time
}

impl EquationPriv for SDE {
    // #[inline(always)]
    // fn get_init(&self) -> &Init {
    //     &self.init
    // }

    // #[inline(always)]
    // fn get_out(&self) -> &Out {
    //     &self.out
    // }

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
                infusions,
                ti,
                tf,
            );
        });
    }
    fn nparticles(&self) -> usize {
        self.nparticles
    }

    fn is_sde(&self) -> bool {
        true
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &crate::Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
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
            *p = observation.to_obs_pred(y[observation.outeq()], x[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.nparticles, 1), pred.clone()).unwrap();
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_model {
            let mut q: Vec<f64> = Vec::with_capacity(self.nparticles);
            //
            // wmy centering_function is a running Chi^2 w/expected value = support point value
            // let centering_function = p.pred[2]; // move this inside the iteration.
            //

            // if no centering_function {
            //   pred.iter().for_each(|p| q.push(p.likelihood(em)));
            // } else
            // pred.iter().enumerate().for_each(|(i,p)| q.push(p.likelihood(em) * x[i][4])); // pred[i][2]

            // if no centering function // for example w/the ODE
            pred.iter()
                .enumerate()
                .for_each(|(i, p)| q.push(p.likelihood(em))); // pred[i][2]
                                                              // else
                                                              //    pred.iter().enumerate().for_each(|(i,p)| q.push(p.likelihood(em) * x[i][4]));

            // for (i,p) in pred.iter().enumerate() { // does the same thing as above.
            //    q.push(p.likelihood(em) * x[i][4]);
            // }
            //
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let i = sysresample(&w);
            let a: Vec<DVector<f64>> = i.iter().map(|&i| x[i].clone()).collect();
            *x = a;
            likelihood.push(sum_q / self.nparticles as f64);
            // let qq: Vec<f64> = i.iter().map(|&i| q[i]).collect();
            // likelihood.push(qq.iter().sum::<f64>() / self.nparticles as f64);
        }
    }
    #[inline(always)]
    fn initial_state(
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

impl Equation for SDE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        if cache {
            _estimate_likelihood_cached(self, subject, support_point, error_model)
        } else {
            _estimate_likelihood_single(self, subject, support_point, error_model)
        }
    }
}

fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}

/// Adaptive Monte Carlo likelihood estimation with convergence criteria
/// Runs simulations until the mean stabilizes within a tolerance threshold
#[inline(always)]
#[cached(
    ty = "UnboundCache<String, f64>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}{}", subject.id(), spphash(support_point), error_model.gl()) }"#
)]
fn _estimate_likelihood_cached(
    sde: &SDE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
) -> f64 {
    // Configuration for adaptive Monte Carlo
    const MIN_RUNS: usize = 5; // Minimum number of simulations
    const MAX_RUNS: usize = 50; // Maximum number of simulations to prevent infinite loops
    const REL_TOL: f64 = 0.01; // Relative tolerance: 1% change in mean
    const ABS_TOL: f64 = 1e-6; // Absolute tolerance for near-zero likelihoods
    const WINDOW_SIZE: usize = 9; // Check stability over last N runs

    let mut likelihoods = Vec::with_capacity(MAX_RUNS);

    // Run minimum number of simulations
    for _ in 0..MIN_RUNS {
        let ypred = sde.simulate_subject(subject, support_point, Some(error_model));
        likelihoods.push(ypred.1.unwrap());
    }

    // Continue until convergence or max runs
    for run in MIN_RUNS..MAX_RUNS {
        let ypred = sde.simulate_subject(subject, support_point, Some(error_model));
        likelihoods.push(ypred.1.unwrap());

        // Check convergence: compare current window mean with previous window mean
        // Need at least 2*WINDOW_SIZE samples to compare two windows
        if run >= 2 * WINDOW_SIZE - 1 {
            let current_mean: f64 =
                likelihoods[run - WINDOW_SIZE + 1..=run].iter().sum::<f64>() / WINDOW_SIZE as f64;

            let previous_mean: f64 = likelihoods[run - 2 * WINDOW_SIZE + 1..=run - WINDOW_SIZE]
                .iter()
                .sum::<f64>()
                / WINDOW_SIZE as f64;

            // Check relative and absolute change
            let abs_change = (current_mean - previous_mean).abs();
            let rel_change = if previous_mean.abs() > ABS_TOL {
                abs_change / previous_mean.abs()
            } else {
                abs_change
            };

            if rel_change < REL_TOL || abs_change < ABS_TOL {
                // Converged! Return mean of all runs
                return likelihoods.iter().sum::<f64>() / likelihoods.len() as f64;
            }
        }
    }

    // Max runs reached, return mean of all runs
    likelihoods.iter().sum::<f64>() / likelihoods.len() as f64
}

/// Single run likelihood estimation (no caching, no convergence checking)
fn _estimate_likelihood_single(
    sde: &SDE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
) -> f64 {
    let ypred = sde.simulate_subject(subject, support_point, Some(error_model));
    ypred.1.unwrap()
}
fn sysresample(q: &[f64]) -> Vec<usize> {
    let mut qc = vec![0.0; q.len()];
    qc[0] = q[0];
    for i in 1..q.len() {
        qc[i] = qc[i - 1] + q[i];
    }
    let m = q.len();
    let mut rng = rng();
    let u: Vec<f64> = (0..m)
        .map(|i| (i as f64 + rng.random::<f64>()) / m as f64)
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
