mod em;

use diffsol::{NalgebraContext, Vector};
use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use rand::{rng, RngExt};
use rayon::prelude::*;

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::Prediction,
    simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Subject,
};

use super::spphash;
use crate::simulator::cache::{SdeLikelihoodCache, DEFAULT_CACHE_SIZE};

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
    ndrugs: usize,
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
        ndrugs,
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
    cache: Option<SdeLikelihoodCache>,
}

impl SDE {
    /// Creates a new stochastic differential equation solver with default Neqs.
    ///
    /// Use builder methods to configure dimensions:
    /// ```ignore
    /// SDE::new(drift, diffusion, lag, fa, init, out, nparticles)
    ///     .with_nstates(2)
    ///     .with_ndrugs(1)
    ///     .with_nout(1)
    /// ```
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        nparticles: usize,
    ) -> Self {
        Self {
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            nparticles,
            cache: None,
        }
    }

    /// Set the number of state variables.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.neqs.nstates = nstates;
        self
    }

    /// Set the number of drug input channels (size of bolus[] and rateiv[]).
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.neqs.ndrugs = ndrugs;
        self
    }

    /// Set the number of output equations.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.neqs.nout = nout;
        self
    }

    /// Enable likelihood caching with the given maximum number of entries.
    ///
    /// When caching is enabled, likelihood results for the same
    /// (subject, parameters, error model) triple are stored and reused.
    /// Cloned equations share the same cache.
    pub fn with_cache(mut self, size: u64) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(size));
        self
    }

    /// Enable likelihood caching with the default size (100,000 entries).
    pub fn with_default_cache(mut self) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE));
        self
    }

    /// Clear all entries from this equation's cache, if caching is enabled.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
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
    fn log_likelihood(&self, error_models: &AssayErrorModels) -> Result<f64, crate::PharmsolError> {
        // For SDE, compute log-likelihood using mean predictions across particles
        let predictions = self.get_predictions();
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let log_liks: Result<Vec<f64>, _> = predictions
            .iter()
            .filter(|p| p.observation().is_some())
            .map(|p| p.log_likelihood(error_models))
            .collect();

        log_liks.map(|lls| lls.iter().sum())
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
        self.neqs.nstates
    }

    #[inline(always)]
    fn get_ndrugs(&self) -> usize {
        self.neqs.ndrugs
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.nout
    }
    #[inline(always)]
    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        let ndrugs = self.get_ndrugs();
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone().into(),
                support_point,
                covariates,
                infusions,
                ndrugs,
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
        support_point: &[f64],
        observation: &crate::Observation,
        error_models: Option<&AssayErrorModels>,
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
                &V::from_vec(support_point.to_vec(), NalgebraContext),
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
                let lik = p.log_likelihood(em).map(f64::exp);
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
        support_point: &[f64],
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
    ///
    /// # Returns
    ///
    /// The log-likelihood of the observed data given the model and parameters.
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        // For SDE, the particle filter computes likelihood in regular space.
        // We compute it directly and then take the log.
        let lik = _estimate_likelihood(self, subject, support_point, error_models)?;

        if lik > 0.0 {
            Ok(lik.ln())
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::SDE
    }
}

#[inline(always)]
fn _estimate_likelihood(
    sde: &SDE,
    subject: &Subject,
    support_point: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    if let Some(cache) = &sde.cache {
        let key = (subject.hash(), spphash(support_point), error_models.hash());
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let ypred = sde.simulate_subject(subject, support_point, Some(error_models))?;
        let result = ypred.1.unwrap();
        cache.insert(key, result);
        Ok(result)
    } else {
        let ypred = sde.simulate_subject(subject, support_point, Some(error_models))?;
        Ok(ypred.1.unwrap())
    }
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
