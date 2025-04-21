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
    Occasion, Subject,
};

use super::{Equation, Predictions, State};

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
    infusions: &[&Infusion],
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
        infusions
            .iter()
            .map(|inf| (*inf).clone())
            .collect::<Vec<Infusion>>(),
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
    subject: &'a Subject,
    state: Vec<DVector<f64>>,
    support_point: Vec<f64>,
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
        }
    }
}

/// State trait implementation for particle-based SDE simulation.
///
/// This implementation allows adding bolus doses to all particles in the system.
impl State for Vec<DVector<f64>> {
    /// Adds a bolus dose to a specific input compartment across all particles.
    ///
    /// # Arguments
    ///
    /// * `input` - Index of the input compartment
    /// * `amount` - Amount to add to the compartment
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.par_iter_mut().for_each(|particle| {
            particle[input] += amount;
        });
    }
}

/// Predictions implementation for particle-based SDE simulation outputs.
///
/// This implementation manages and processes predictions from multiple particles.
impl Predictions for Array2<Prediction> {
    fn empty(nparticles: usize) -> Self {
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
    type S = Vec<DVector<f64>>; // Vec -> particles, DVector -> state
    type P = Array2<Prediction>; // Rows -> particles, Columns -> time
    type Mod = SDEModel<'a>;

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
        SDEModel::new(self, subject, spp)
    }
    fn nparticles(&self) -> usize {
        self.nparticles
    }

    fn is_sde(&self) -> bool {
        true
    }
}

impl<'a> Model<'a> for SDEModel<'a> {
    type Eq = SDE;
    fn new(equation: &'a Self::Eq, subject: &'a Subject, support_point: Vec<f64>) -> Self {
        Self {
            equation,
            subject,
            state: vec![DVector::zeros(equation.get_nstates()); equation.nparticles()],
            support_point,
        }
    }
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.state.add_bolus(input, amount);
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
        Some((self.equation.lag)(&V::from_vec(
            self.support_point.to_owned(),
        )))
    }

    #[inline(always)]
    fn get_fa(&self) -> Option<HashMap<usize, f64>> {
        Some((self.equation.fa)(&V::from_vec(
            self.support_point.to_owned(),
        )))
    }
    #[inline(always)]
    fn solve(&mut self, covariates: &Covariates, infusions: Vec<&Infusion>, ti: f64, tf: f64) {
        self.state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.equation.drift,
                &self.equation.diffusion,
                particle.clone(),
                &self.support_point,
                covariates,
                &infusions,
                ti,
                tf,
            );
        });
    }
    #[inline(always)]
    fn process_observation(
        &mut self,
        observation: &crate::Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation>::P,
    ) {
        let mut pred = vec![Prediction::default(); self.nparticles()];
        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.equation.get_nouteqs());
            (self.equation.out)(
                &self.state[i],
                &V::from_vec(self.support_point.to_vec()),
                observation.time(),
                covariates,
                &mut y,
            );
            *p = observation.to_obs_pred(y[observation.outeq()], self.state[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.nparticles(), 1), pred.clone()).unwrap();
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_model {
            let mut q: Vec<f64> = Vec::with_capacity(self.nparticles());

            pred.iter().for_each(|p| q.push(p.likelihood(em)));
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let i = sysresample(&w);
            self.state = i.iter().map(|&i| self.state[i].clone()).collect();
            likelihood.push(sum_q / self.nparticles() as f64);
            // let qq: Vec<f64> = i.iter().map(|&i| q[i]).collect();
            // likelihood.push(qq.iter().sum::<f64>() / self.nparticles as f64);
        }
    }
    #[inline(always)]
    fn initial_state(&mut self, occasion: &Occasion) {
        let mut x = Vec::with_capacity(self.nparticles());
        let covariates = occasion.get_covariates();
        for _ in 0..self.nparticles() {
            let mut state = DVector::zeros(self.equation.get_nstates());
            if occasion.index() == 0 {
                (self.equation.init)(
                    &V::from_vec(self.support_point.to_vec()),
                    0.0,
                    covariates.unwrap_or(&Covariates::new()),
                    &mut state,
                );
            }
            x.push(state);
        }
        self.state = x
    }
    fn estimate_likelihood(self, error_model: &ErrorModel, cache: bool) -> f64 {
        if cache {
            _estimate_likelihood(self, error_model)
        } else {
            _estimate_likelihood_no_cache(self, error_model)
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
    convert = r#"{ format!("{}{}{}", model.subject.id(), spphash(&model.support_point), error_model.gl()) }"#
)]
fn _estimate_likelihood(model: SDEModel, error_model: &ErrorModel) -> f64 {
    let ypred = model.simulate_subject(Some(error_model));
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
