mod em;

use std::collections::HashMap;
use std::marker::PhantomData;

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
    Missing, Provided, Subject,
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
    /// For a more ergonomic API, consider using [`SDEBuilder`] instead.
    ///
    /// # Arguments
    ///
    /// * `drift` - Function defining the deterministic component of the SDE
    /// * `diffusion` - Function defining the stochastic component of the SDE
    /// * `lag` - Function to compute absorption lag times
    /// * `fa` - Function to compute bioavailability fractions
    /// * `init` - Function to initialize the system state
    /// * `out` - Function to compute output equations
    /// * `neqs` - Number of states and output equations (can be a tuple or [`Neqs`])
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
        neqs: impl Into<Neqs>,
        nparticles: usize,
    ) -> Self {
        Self {
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
            neqs: neqs.into(),
            nparticles,
        }
    }

    /// Returns a new [`SDEBuilder`] for constructing an SDE equation.
    ///
    /// # Example
    /// ```ignore
    /// use pharmsol::prelude::*;
    ///
    /// // Minimal builder - only required fields
    /// let sde = SDE::builder()
    ///     .drift(drift)
    ///     .diffusion(diffusion)
    ///     .out(out)
    ///     .nstates(2)
    ///     .nouteqs(1)
    ///     .nparticles(1000)
    ///     .build();
    ///
    /// // With optional fields
    /// let sde = SDE::builder()
    ///     .drift(drift)
    ///     .diffusion(diffusion)
    ///     .out(out)
    ///     .nstates(2)
    ///     .nouteqs(1)
    ///     .nparticles(1000)
    ///     .lag(|p, _t, _cov| lag! { 0 => p[2] })
    ///     .fa(|p, _t, _cov| fa! { 0 => 0.8 })
    ///     .init(|p, _t, _cov, x| { x[0] = p[3]; })
    ///     .build();
    /// ```
    pub fn builder() -> SDEBuilder<Missing, Missing, Missing, Missing, Missing, Missing> {
        SDEBuilder::new()
    }
}

// =============================================================================
// Type-State Builder Pattern
// =============================================================================

// Note: Missing and Provided marker types are defined in the parent module
// and imported via `use crate::{..., Missing, Provided, ...}`

/// Builder for constructing [`SDE`] equations with compile-time validation.
///
/// This builder uses the type-state pattern to ensure all required fields
/// are set before `build()` can be called. Optional fields (`lag`, `fa`, `init`)
/// have sensible defaults.
///
/// # Required Fields (enforced at compile time)
/// - `drift`: The drift (deterministic) function
/// - `diffusion`: The diffusion (stochastic) function
/// - `out`: Output equation function
/// - `nstates`: Number of state variables
/// - `nouteqs`: Number of output equations
/// - `nparticles`: Number of particles for simulation
///
/// # Optional Fields (with defaults)
/// - `lag`: Lag time function (defaults to no lag)
/// - `fa`: Bioavailability function (defaults to 100% bioavailability)
/// - `init`: Initial state function (defaults to zero initial state)
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::*;
///
/// // Minimal example - only required fields
/// let sde = SDE::builder()
///     .drift(|x, p, t, dx, rateiv, cov| { /* ... */ })
///     .diffusion(|x, p, t, dx, cov| { /* ... */ })
///     .out(|x, p, _t, _cov, y| { y[0] = x[0] / p[1]; })
///     .nstates(1)
///     .nouteqs(1)
///     .nparticles(1000)
///     .build();
/// ```
pub struct SDEBuilder<
    DriftState,
    DiffusionState,
    OutState,
    NStatesState,
    NOuteqsState,
    NParticlesState,
> {
    drift: Option<Drift>,
    diffusion: Option<Diffusion>,
    lag: Option<Lag>,
    fa: Option<Fa>,
    init: Option<Init>,
    out: Option<Out>,
    nstates: Option<usize>,
    nouteqs: Option<usize>,
    nparticles: Option<usize>,
    _phantom: PhantomData<(
        DriftState,
        DiffusionState,
        OutState,
        NStatesState,
        NOuteqsState,
        NParticlesState,
    )>,
}

impl SDEBuilder<Missing, Missing, Missing, Missing, Missing, Missing> {
    /// Creates a new SDEBuilder with all required fields unset.
    pub fn new() -> Self {
        Self {
            drift: None,
            diffusion: None,
            lag: None,
            fa: None,
            init: None,
            out: None,
            nstates: None,
            nouteqs: None,
            nparticles: None,
            _phantom: PhantomData,
        }
    }
}

impl Default for SDEBuilder<Missing, Missing, Missing, Missing, Missing, Missing> {
    fn default() -> Self {
        Self::new()
    }
}

impl<DriftState, DiffusionState, OutState, NStatesState, NOuteqsState, NParticlesState>
    SDEBuilder<DriftState, DiffusionState, OutState, NStatesState, NOuteqsState, NParticlesState>
{
    /// Sets the lag time function (optional).
    ///
    /// If not set, defaults to no lag for any compartment.
    pub fn lag(mut self, lag: Lag) -> Self {
        self.lag = Some(lag);
        self
    }

    /// Sets the bioavailability function (optional).
    ///
    /// If not set, defaults to 100% bioavailability for all compartments.
    pub fn fa(mut self, fa: Fa) -> Self {
        self.fa = Some(fa);
        self
    }

    /// Sets the initial state function (optional).
    ///
    /// If not set, defaults to zero initial state for all compartments.
    pub fn init(mut self, init: Init) -> Self {
        self.init = Some(init);
        self
    }
}

impl<DiffusionState, OutState, NStatesState, NOuteqsState, NParticlesState>
    SDEBuilder<Missing, DiffusionState, OutState, NStatesState, NOuteqsState, NParticlesState>
{
    /// Sets the drift (deterministic) function (required).
    ///
    /// The drift function defines the deterministic component of the SDE: dx/dt = f(x, p, t, ...)
    pub fn drift(
        self,
        drift: Drift,
    ) -> SDEBuilder<Provided, DiffusionState, OutState, NStatesState, NOuteqsState, NParticlesState>
    {
        SDEBuilder {
            drift: Some(drift),
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, OutState, NStatesState, NOuteqsState, NParticlesState>
    SDEBuilder<DriftState, Missing, OutState, NStatesState, NOuteqsState, NParticlesState>
{
    /// Sets the diffusion (stochastic) function (required).
    ///
    /// The diffusion function defines the stochastic component of the SDE.
    pub fn diffusion(
        self,
        diffusion: Diffusion,
    ) -> SDEBuilder<DriftState, Provided, OutState, NStatesState, NOuteqsState, NParticlesState>
    {
        SDEBuilder {
            drift: self.drift,
            diffusion: Some(diffusion),
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, DiffusionState, NStatesState, NOuteqsState, NParticlesState>
    SDEBuilder<DriftState, DiffusionState, Missing, NStatesState, NOuteqsState, NParticlesState>
{
    /// Sets the output equation function (required).
    pub fn out(
        self,
        out: Out,
    ) -> SDEBuilder<DriftState, DiffusionState, Provided, NStatesState, NOuteqsState, NParticlesState>
    {
        SDEBuilder {
            drift: self.drift,
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: Some(out),
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, DiffusionState, OutState, NOuteqsState, NParticlesState>
    SDEBuilder<DriftState, DiffusionState, OutState, Missing, NOuteqsState, NParticlesState>
{
    /// Sets the number of state variables (compartments) (required).
    pub fn nstates(
        self,
        nstates: usize,
    ) -> SDEBuilder<DriftState, DiffusionState, OutState, Provided, NOuteqsState, NParticlesState>
    {
        SDEBuilder {
            drift: self.drift,
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(nstates),
            nouteqs: self.nouteqs,
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, DiffusionState, OutState, NStatesState, NParticlesState>
    SDEBuilder<DriftState, DiffusionState, OutState, NStatesState, Missing, NParticlesState>
{
    /// Sets the number of output equations (required).
    pub fn nouteqs(
        self,
        nouteqs: usize,
    ) -> SDEBuilder<DriftState, DiffusionState, OutState, NStatesState, Provided, NParticlesState>
    {
        SDEBuilder {
            drift: self.drift,
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: Some(nouteqs),
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, DiffusionState, OutState, NParticlesState>
    SDEBuilder<DriftState, DiffusionState, OutState, Missing, Missing, NParticlesState>
{
    /// Sets both nstates and nouteqs from a [`Neqs`] struct or tuple (required).
    pub fn neqs(
        self,
        neqs: impl Into<Neqs>,
    ) -> SDEBuilder<DriftState, DiffusionState, OutState, Provided, Provided, NParticlesState> {
        let neqs = neqs.into();
        SDEBuilder {
            drift: self.drift,
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(neqs.nstates),
            nouteqs: Some(neqs.nouteqs),
            nparticles: self.nparticles,
            _phantom: PhantomData,
        }
    }
}

impl<DriftState, DiffusionState, OutState, NStatesState, NOuteqsState>
    SDEBuilder<DriftState, DiffusionState, OutState, NStatesState, NOuteqsState, Missing>
{
    /// Sets the number of particles for simulation (required).
    pub fn nparticles(
        self,
        nparticles: usize,
    ) -> SDEBuilder<DriftState, DiffusionState, OutState, NStatesState, NOuteqsState, Provided>
    {
        SDEBuilder {
            drift: self.drift,
            diffusion: self.diffusion,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            nparticles: Some(nparticles),
            _phantom: PhantomData,
        }
    }
}

/// Default lag function: no lag for any compartment
fn default_lag(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

/// Default fa function: 100% bioavailability for all compartments
fn default_fa(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

/// Default init function: zero initial state
fn default_init(_p: &V, _t: f64, _cov: &Covariates, _x: &mut V) {
    // State is already zero-initialized
}

impl SDEBuilder<Provided, Provided, Provided, Provided, Provided, Provided> {
    /// Builds the [`SDE`] equation.
    ///
    /// This method is only available when all required fields have been set:
    /// - `drift`
    /// - `diffusion`
    /// - `out`
    /// - `nstates`
    /// - `nouteqs`
    /// - `nparticles`
    ///
    /// Optional fields use defaults if not set:
    /// - `lag`: No lag (empty HashMap)
    /// - `fa`: 100% bioavailability (empty HashMap)
    /// - `init`: Zero initial state
    pub fn build(self) -> SDE {
        SDE {
            drift: self.drift.unwrap(),
            diffusion: self.diffusion.unwrap(),
            lag: self.lag.unwrap_or(default_lag),
            fa: self.fa.unwrap_or(default_fa),
            init: self.init.unwrap_or(default_init),
            out: self.out.unwrap(),
            neqs: Neqs::new(self.nstates.unwrap(), self.nouteqs.unwrap()),
            nparticles: self.nparticles.unwrap(),
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
    fn log_likelihood(&self, error_models: &ErrorModels) -> Result<f64, crate::PharmsolError> {
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
    fn get_nouteqs(&self) -> usize {
        self.neqs.nouteqs
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

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        // For SDE, the particle filter computes likelihood in regular space.
        // We take the log of the cached/computed likelihood.
        // Note: For extreme underflow cases, this may return -inf.
        let lik = self.estimate_likelihood(subject, support_point, error_models, cache)?;
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

//TODO: Add hash impl on dedicated structure!
/// Hash support points to a u64 for cache key generation.
/// Uses DefaultHasher for good distribution and collision resistance.
#[inline(always)]

fn spphash(spp: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    for &value in spp {
        // Normalize -0.0 to 0.0 for consistent hashing
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<(u64, u64, u64), f64>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ ((subject.hash()), spphash(support_point), error_models.hash()) }"#,
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
