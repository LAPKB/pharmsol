mod em;

use diffsol::{NalgebraContext, Vector};
use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use rand::{rng, Rng};
use rayon::prelude::*;

use cached::proc_macro::cached;
use cached::UnboundCache;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModels,
    prelude::simulator::Prediction,
    simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Subject,
};

use diffsol::VectorCommon;

use crate::PharmsolError;

use super::{Equation, EquationPriv, EquationTypes, Predictions, State};

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
        x.inner().clone(),
        cov.clone(),
        infusions.to_vec(),
        1e-2,
        1e-2,
    );
    let (_time, solution) = sde.solve(ti, tf);
    solution.last().unwrap().clone().into()
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
    #[allow(clippy::too_many_arguments)]
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
    fn new(nparticles: usize) -> Self {
        Array2::from_shape_fn((nparticles, 0), |_| Prediction::default())
    }
    fn squared_error(&self) -> f64 {
        unimplemented!();
    }
    fn get_predictions(&self) -> Vec<Prediction> {
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

    // #[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.lag)(&V::from_vec(spp.to_owned())))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.fa)(&V::from_vec(spp.to_owned())))
    // }

    #[inline(always)]
    fn lag(&self) -> &Lag {
        &self.lag
    }

    #[inline(always)]
    fn fa(&self) -> &Fa {
        &self.fa
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
    ) -> Result<(), PharmsolError> {
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone().into(),
                support_point,
                covariates,
                infusions,
                ti,
                tf,
            )
            .inner()
            .clone();
        });
        Ok(())
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
        error_models: Option<&ErrorModels>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut pred = vec![Prediction::default(); self.nparticles];
        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
            (self.out)(
                &x[i].clone().into(),
                &V::from_vec(support_point.clone(), NalgebraContext),
                observation.time(),
                covariates,
                &mut y,
            );
            *p = observation.to_prediction(y[observation.outeq()], x[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.nparticles, 1), pred.clone())?;
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        //e = y[t] .- x[:,1]
        // q = pdf.(Distributions.Normal(0, 0.5), e)
        if let Some(em) = error_models {
            let mut q: Vec<f64> = Vec::with_capacity(self.nparticles);

            pred.iter().for_each(|p| {
                let lik = p.likelihood(em);
                match lik {
                    Ok(l) => q.push(l),
                    Err(e) => panic!("Error in likelihood calculation: {:?}", e),
                }
            });
            let sum_q: f64 = q.iter().sum();
            let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
            let i = sysresample(&w);
            let a: Vec<DVector<f64>> = i.iter().map(|&i| x[i].clone()).collect();
            *x = a;
            likelihood.push(sum_q / self.nparticles as f64);
            // let qq: Vec<f64> = i.iter().map(|&i| q[i]).collect();
            // likelihood.push(qq.iter().sum::<f64>() / self.nparticles as f64);
        }
        Ok(())
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
            let mut state: V = DVector::zeros(self.get_nstates()).into();
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(support_point.to_vec(), NalgebraContext),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state.inner().clone());
        }
        x
    }
}

impl Equation for SDE {
    /// Estimates the likelihood of observed data given a model and parameters.
    ///
    /// # Arguments
    ///
    /// * `subject` - Subject data containing observations
    /// * `support_point` - Parameter vector for the model
    /// * `error_model` - Error model to use for likelihood calculations
    /// * `cache` - Whether to cache likelihood results for reuse
    ///
    /// # Returns
    ///
    /// The log-likelihood of the observed data given the model and parameters.
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        if cache {
            _estimate_likelihood(self, subject, support_point, error_models)
        } else {
            _estimate_likelihood_no_cache(self, subject, support_point, error_models)
        }
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::SDE
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
    convert = r#"{ format!("{}{}{:#?}", subject.id(), spphash(support_point), error_models.hash()) }"#,
    result = "true"
)]
fn _estimate_likelihood(
    sde: &SDE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &ErrorModels,
) -> Result<f64, PharmsolError> {
    let ypred = sde.simulate_subject(subject, support_point, Some(error_models))?;
    Ok(ypred.1.unwrap())
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
