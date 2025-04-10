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
    simulator::{
        likelihood::ToPrediction, model::Model, Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V,
    },
    Subject,
};

use super::{Equation, Outputs, State};

/// Simulate a stochastic differential equation (SDE) event.
///
/// This function advances the SDE system from time `ti` to `tf` using
/// the Euler-Maruyama method implemented in the `em` module.
///
/// # Arguments
///
/// * `drift` - Function defining the deterministic component of the SDE
/// * `difussion` - Function defining the stochastic component of the SDE
/// * `x` - Current state vector
/// * `support_point` - Parameter vector for the model
/// * `cov` - Covariates that may influence the system dynamics
/// * `infusions` - Infusion events to be applied during simulation
/// * `ti` - Starting time
/// * `tf` - Ending time
///
/// # Returns
///
/// The state vector at time `tf` after simulation.
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

/// Stochastic Differential Equation solver for pharmacometric models.
///
/// This struct represents a stochastic differential equation system and provides
/// methods to simulate particles and estimate likelihood for PKPD modeling.
///
/// SDE models introduce stochasticity into the system dynamics, allowing for more
/// realistic modeling of biological variability and uncertainty.
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

pub struct SDEModel<'a> {
    equation: &'a SDE,
    data: &'a Subject,
    state: Vec<DVector<f64>>,
}

impl State for Vec<DVector<f64>> {}
impl Outputs for Array2<Prediction> {
    fn new_outputs(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }
    fn squared_error(&self) -> f64 {
        unimplemented!();
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        //TODO: This is only returning the first particle, not the best, not the worst, THE FIRST
        // CHANGE THIS
        let row = self.row(0).to_vec();
        row
    }
}
impl<'a> Equation<'a> for SDE {
    type S = Vec<DVector<f64>>;
    type P = Array2<Prediction>;
    type Mod = SDEModel<'a>;
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    fn is_sde(&self) -> bool {
        true
    }
    fn nparticles(&self) -> usize {
        self.nparticles
    }
    fn initialize_model(&'a self, subject: &'a Subject, spp: &[f64]) -> Self::Mod {
        SDEModel::new(self, subject, spp)
    }
}
impl<'a> Model<'a> for SDEModel<'a> {
    type Eq = SDE;

    fn new(equation: &'a Self::Eq, data: &'a Subject, _spp: &[f64]) -> Self {
        Self {
            equation,
            data,
            state: vec![DVector::zeros(equation.get_nstates()); equation.nparticles()],
        }
    }

    fn equation(&self) -> &Self::Eq {
        self.equation
    }

    fn subject(&self) -> &Subject {
        self.data
    }

    fn state(&self) -> &<Self::Eq as Equation>::S {
        &self.state
    }
    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.equation.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.equation.fa)(&V::from_vec(spp.to_owned())))
    }

    /// Adds a bolus dose to a specific input compartment across all particles.
    ///
    /// # Arguments
    ///
    /// * `input` - Index of the input compartment
    /// * `amount` - Amount to add to the compartment
    fn add_bolus(&mut self, input: usize, amount: f64) {
        // self.state.par_iter_mut().for_each(|particle| {
        //     particle[input] += amount;
        // });
        for particle in &mut self.state {
            particle[input] += amount;
        }
    }

    #[inline(always)]
    fn solve(
        &mut self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        ti: f64,
        tf: f64,
    ) {
        self.state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.equation.drift,
                &self.equation.diffusion,
                particle.clone(),
                support_point,
                covariates,
                infusions,
                ti,
                tf,
            );
        });
    }

    #[inline(always)]
    fn process_observation(
        &mut self,
        support_point: &Vec<f64>,
        observation: &crate::Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation>::P,
    ) {
        let mut pred = vec![Prediction::default(); self.equation.nparticles];
        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.equation.get_nouteqs());
            (self.equation.out)(
                &self.state[i],
                &V::from_vec(support_point.clone()),
                observation.time(),
                covariates,
                &mut y,
            );
            *p = observation.to_obs_pred(y[observation.outeq()], self.state[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.equation.nparticles, 1), pred.clone()).unwrap();
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_model {
            let mut q: Vec<f64> = Vec::with_capacity(self.equation.nparticles);

            pred.iter().for_each(|p| q.push(p.likelihood(em)));
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let i = sysresample(&w);
            let a: Vec<DVector<f64>> = i.iter().map(|&i| self.state[i].clone()).collect();
            self.state = a;
            likelihood.push(sum_q / self.equation.nparticles as f64);
            // let qq: Vec<f64> = i.iter().map(|&i| q[i]).collect();
            // likelihood.push(qq.iter().sum::<f64>() / self.nparticles as f64);
        }
    }
    #[inline(always)]
    fn initial_state(
        &mut self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        occasion_index: usize,
    ) {
        let mut x = Vec::with_capacity(self.equation.nparticles);
        for _ in 0..self.equation.nparticles {
            let mut state = DVector::zeros(self.equation.get_nstates());
            if occasion_index == 0 {
                (self.equation.init)(
                    &V::from_vec(support_point.to_vec()),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state);
        }
        self.state = x;
    }
    fn estimate_likelihood(
        &mut self,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        if cache {
            _estimate_likelihood(self, support_point, error_model)
        } else {
            _estimate_likelihood_no_cache(self, support_point, error_model)
        }
    }
}

impl SDE {
    /// Creates a new stochastic differential equation solver.
    ///
    /// # Arguments
    ///
    /// * `drift` - Function defining the deterministic component of the SDE
    /// * `diffusion` - Function defining the stochastic component of the SDE
    /// * `lag` - Function to compute absorption lag times
    /// * `fa` - Function to compute bioavailability fractions
    /// * `init` - Function to initialize the system state
    /// * `out` - Function to compute output equations
    /// * `neqs` - Tuple containing the number of state and output equations
    /// * `nparticles` - Number of particles to use in the simulation
    ///
    /// # Returns
    ///
    /// A new SDE solver instance configured with the given components.
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
            // state: vec![DVector::zeros(neqs.0); nparticles],
        }
    }
}

/// Computes a hash value for a parameter vector.
///
/// # Arguments
///
/// * `spp` - Parameter vector
///
/// # Returns
///
/// A u64 hash value representing the parameter vector.
fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<String, f64>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}{}", model.data.id(), spphash(support_point), error_model.gl()) }"#
)]
fn _estimate_likelihood(
    model: &mut SDEModel,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
) -> f64 {
    let ypred = model.simulate_subject(support_point, Some(error_model));
    ypred.1.unwrap()
}

/// Performs systematic resampling of particles based on weights.
///
/// # Arguments
///
/// * `q` - Vector of particle weights
///
/// # Returns
///
/// Vector of indices to use for resampling.
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
