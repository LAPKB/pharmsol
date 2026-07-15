mod em;

use diffsol::{NalgebraContext, Vector};
use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use pharmsol_dsl::ModelKind;
use rand::Rng;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    data::{Covariates, Infusion},
    prelude::simulator::Prediction,
    simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V},
    Event, Observation, Parameters, Subject,
};

use diffsol::VectorCommon;

use crate::PharmsolError;

use super::{
    EqnKind, Equation, EquationPriv, EquationTypes, ModelMetadata, ModelMetadataError, Predictions,
    State, ValidatedModelMetadata,
};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SdeMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("SDE declares {declared} state metadata entries but model has {expected} states")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error("SDE declares {declared} route metadata entries but model has {expected} inputs")]
    RouteCountMismatch { expected: usize, declared: usize },
    #[error("SDE declares {declared} output metadata entries but model has {expected} outputs")]
    OutputCountMismatch { expected: usize, declared: usize },
}

#[derive(Clone, Debug, Default)]
struct InjectedBolusMappings {
    destinations: Vec<Option<usize>>,
}

impl InjectedBolusMappings {
    fn explicit(ndrugs: usize) -> Self {
        Self {
            destinations: vec![None; ndrugs],
        }
    }

    fn from_destinations(ndrugs: usize, destinations: &[Option<usize>]) -> Self {
        let mut mappings = Self::explicit(ndrugs);
        for (input, destination) in destinations.iter().copied().take(ndrugs).enumerate() {
            mappings.destinations[input] = destination;
        }
        mappings
    }

    fn invalidate_for_ndrugs(&mut self, ndrugs: usize) {
        *self = Self::explicit(ndrugs);
    }

    fn apply(&self, state: &mut [DVector<f64>], input: usize, amount: f64) -> bool {
        let Some(destination) = self.destinations.get(input).copied().flatten() else {
            return false;
        };
        state.par_iter_mut().for_each(|particle| {
            particle[destination] += amount;
        });
        true
    }
}

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
/// * `parameters` - Parameter vector for the model
/// * `cov` - Covariates that may influence the system dynamics
/// * `infusions` - Infusion events to be applied during simulation
/// * `ti` - Starting time
/// * `tf` - Ending time
///
/// # Returns
///
/// The state vector at time `tf` after simulation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    parameters: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ndrugs: usize,
    ti: f64,
    tf: f64,
    discontinuities: &[f64],
) -> V {
    if ti == tf {
        return x;
    }

    let parameters_v = V::from_vec(parameters.to_vec(), NalgebraContext::new());
    let covariates = cov.clone();
    let infusion_events = infusions.to_vec();
    let drift_fn = *drift;
    let diffusion_fn = *difussion;

    let parameters_for_drift = parameters_v.clone();
    let drift_closure = move |time: f64, state: &DVector<f64>, out: &mut DVector<f64>| {
        let mut rateiv = V::zeros(ndrugs, NalgebraContext::new());
        for infusion in &infusion_events {
            if time >= infusion.time() && time < infusion.duration() + infusion.time() {
                let input = infusion
                    .input_index()
                    .expect("resolved infusions should use numeric input labels");
                rateiv[input] += infusion.amount() / infusion.duration();
            }
        }

        let state_v: V = state.clone().into();
        let mut out_v = V::zeros(state.len(), NalgebraContext::new());
        drift_fn(
            &state_v,
            &parameters_for_drift,
            time,
            &mut out_v,
            &rateiv,
            &covariates,
        );
        out.copy_from(out_v.inner());
    };

    let diffusion_closure = move |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| {
        let mut out_v = V::zeros(out.len(), NalgebraContext::new());
        diffusion_fn(&parameters_v, &mut out_v);
        out.copy_from(out_v.inner());
    };

    simulate_sde_event_with(
        drift_closure,
        diffusion_closure,
        x.inner().clone(),
        ti,
        tf,
        discontinuities,
    )
    .into()
}

pub(crate) fn infusion_discontinuities(infusions: &[Infusion], ti: f64, tf: f64) -> Vec<f64> {
    let mut discontinuities = infusions
        .iter()
        .flat_map(|infusion| [infusion.time(), infusion.time() + infusion.duration()])
        .filter(|&time| time > ti && time < tf)
        .collect::<Vec<_>>();
    discontinuities.sort_by(f64::total_cmp);
    discontinuities.dedup_by(|left, right| left.total_cmp(right).is_eq());
    discontinuities
}

pub(crate) fn simulate_sde_event_with<D, G>(
    drift: D,
    diffusion: G,
    initial_state: DVector<f64>,
    ti: f64,
    tf: f64,
    discontinuities: &[f64],
) -> DVector<f64>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
{
    if ti == tf {
        return initial_state;
    }

    let mut rng = rand::rng();
    solve_sde_event_with_rng(
        drift,
        diffusion,
        initial_state,
        ti,
        tf,
        discontinuities,
        &mut rng,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_sde_event_with_rng<D, G, R>(
    drift: D,
    diffusion: G,
    initial_state: DVector<f64>,
    ti: f64,
    tf: f64,
    discontinuities: &[f64],
    rng: &mut R,
) -> DVector<f64>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    R: Rng + ?Sized,
{
    let mut sde = em::EM::new(drift, diffusion, initial_state.clone(), 1e-2, 1e-2);
    let mut segment_start = ti;
    let mut final_state = initial_state;
    for segment_end in discontinuities.iter().copied().chain(std::iter::once(tf)) {
        let (_times, mut solution) = sde.solve_with_rng(segment_start, segment_end, rng);
        final_state = solution.pop().unwrap();
        segment_start = segment_end;
    }
    final_state
}

/// Stochastic Differential Equation solver for pharmacometric models.
///
/// This struct represents a stochastic differential equation system and provides
/// methods to generate structural predictions from multiple particles.
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
    metadata: Option<ValidatedModelMetadata>,
    injected_bolus_mappings: InjectedBolusMappings,
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
            metadata: None,
            injected_bolus_mappings: InjectedBolusMappings::default(),
        }
    }

    /// Set the number of state variables.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.neqs.nstates = nstates;
        self.invalidate_metadata();
        self
    }

    /// Set the number of drug inputs (size of bolus[] and rateiv[]).
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.neqs.ndrugs = ndrugs;
        self.invalidate_metadata();
        self
    }

    /// Set the number of output equations.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.neqs.nout = nout;
        self.invalidate_metadata();
        self
    }

    /// Attach validated handwritten-model metadata to this SDE model.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, SdeMetadataError> {
        let metadata = metadata.validate_for_with_particles(ModelKind::Sde, self.nparticles)?;
        validate_metadata_dimensions(&metadata, &self.neqs)?;
        self.metadata = Some(metadata);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn with_injected_bolus_inputs(mut self, destinations: &[Option<usize>]) -> Self {
        self.injected_bolus_mappings =
            InjectedBolusMappings::from_destinations(self.neqs.ndrugs, destinations);
        self
    }

    /// Access the validated metadata attached to this SDE model, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.state_index(name)
    }

    fn invalidate_metadata(&mut self) {
        self.metadata = None;
        self.injected_bolus_mappings
            .invalidate_for_ndrugs(self.neqs.ndrugs);
    }

    /// Start a stateful particle simulation controlled by the supplied generator.
    ///
    /// The session stops at every observation. The caller must explicitly keep
    /// the cloud or select ancestors before asking for the next observation.
    pub fn particle_session<'a, R: Rng + ?Sized>(
        &'a self,
        subject: &Subject,
        parameters: &'a Parameters,
        particle_count: usize,
        rng: &'a mut R,
    ) -> Result<SdeParticleSession<'a, R>, SdeSessionError> {
        if particle_count == 0 {
            return Err(SdeSessionError::EmptyCloud);
        }

        let mut events = Vec::with_capacity(subject.occasions().len());
        let mut covariates = Vec::with_capacity(subject.occasions().len());
        for occasion in subject.occasions() {
            covariates.push(occasion.covariates().clone());
            events.push(self.resolve_occasion_events(
                occasion,
                parameters.as_slice(),
                occasion.covariates(),
            )?);
        }

        let states = if let Some(occasion) = subject.occasions().first() {
            self.initial_particles(
                parameters.as_slice(),
                occasion.covariates(),
                occasion.index(),
                particle_count,
            )
        } else {
            Vec::new()
        };

        Ok(SdeParticleSession {
            model: self,
            parameters: parameters.as_slice(),
            events,
            covariates,
            occasion_indices: subject
                .occasions()
                .iter()
                .map(|occasion| occasion.index())
                .collect(),
            occasion: 0,
            event: 0,
            states,
            spare: Vec::with_capacity(particle_count),
            infusions: Vec::new(),
            predictions: Vec::with_capacity(particle_count),
            observation: None,
            waiting: false,
            rng,
            particle_count,
        })
    }

    fn initial_particles(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
        particle_count: usize,
    ) -> Vec<DVector<f64>> {
        let mut particles = Vec::with_capacity(particle_count);
        for _ in 0..particle_count {
            let mut state: V = DVector::zeros(self.get_nstates()).into();
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(parameters.to_vec(), NalgebraContext),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            particles.push(state.inner().clone());
        }
        particles
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_particle<R: Rng + ?Sized>(
        &self,
        state: DVector<f64>,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
        discontinuities: &[f64],
        rng: &mut R,
    ) -> DVector<f64> {
        if ti == tf {
            return state;
        }
        let parameter_values = V::from_vec(parameters.to_vec(), NalgebraContext);
        let infusion_events = infusions.to_vec();
        let ndrugs = self.get_ndrugs();
        let drift = self.drift;
        let diffusion = self.diffusion;
        let drift_parameters = parameter_values.clone();
        let covariates = covariates.clone();

        let drift_closure = move |time: f64, state: &DVector<f64>, out: &mut DVector<f64>| {
            let mut rateiv = V::zeros(ndrugs, NalgebraContext);
            for infusion in &infusion_events {
                if time >= infusion.time() && time < infusion.duration() + infusion.time() {
                    let input = infusion
                        .input_index()
                        .expect("resolved infusions should use numeric input labels");
                    rateiv[input] += infusion.amount() / infusion.duration();
                }
            }
            let mut out_v = V::zeros(state.len(), NalgebraContext);
            drift(
                &state.clone().into(),
                &drift_parameters,
                time,
                &mut out_v,
                &rateiv,
                &covariates,
            );
            out.copy_from(out_v.inner());
        };
        let diffusion_closure = move |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| {
            let mut out_v = V::zeros(out.len(), NalgebraContext);
            diffusion(&parameter_values, &mut out_v);
            out.copy_from(out_v.inner());
        };
        solve_sde_event_with_rng(
            drift_closure,
            diffusion_closure,
            state,
            ti,
            tf,
            discontinuities,
            rng,
        )
    }
}

/// Errors raised by caller-controlled SDE particle sessions.
#[derive(Debug, Error)]
pub enum SdeSessionError {
    /// The underlying simulation rejected the subject, labels, or parameters.
    #[error(transparent)]
    Simulation(#[from] PharmsolError),
    /// A session was requested with zero particles.
    #[error("a particle session requires at least one particle")]
    EmptyCloud,
    /// The caller requested another observation before choosing how to resume.
    #[error("choose how to resume from the current observation boundary before advancing")]
    BoundaryPending,
    /// A resume operation was requested when no observation is pending.
    #[error("no observation boundary is currently pending")]
    NoBoundary,
    /// The replacement ancestor list does not contain one index per particle.
    #[error("expected {expected} ancestor indices, received {actual}")]
    AncestorCount {
        /// Required number of indices.
        expected: usize,
        /// Number supplied by the caller.
        actual: usize,
    },
    /// An ancestor index does not identify a particle in the current cloud.
    #[error("ancestor index {index} is outside the particle cloud of size {particle_count}")]
    AncestorOutOfRange {
        /// Invalid ancestor index.
        index: usize,
        /// Number of particles in the current cloud.
        particle_count: usize,
    },
}

/// Borrowed data exposed while a session is stopped at an observation.
#[derive(Debug)]
pub struct SdeParticleObservation<'a> {
    observation: &'a Observation,
    predictions: &'a [Prediction],
    states: &'a [DVector<f64>],
}

impl<'a> SdeParticleObservation<'a> {
    /// Borrow the source observation at this boundary.
    pub fn observation(&self) -> &'a Observation {
        self.observation
    }

    /// Borrow one noiseless model prediction per particle.
    pub fn predictions(&self) -> &'a [Prediction] {
        self.predictions
    }

    /// Borrow the full particle states at this boundary.
    pub fn states(&self) -> &'a [DVector<f64>] {
        self.states
    }

    /// Return the observation time.
    pub fn time(&self) -> f64 {
        self.observation.time()
    }

    /// Return the resolved dense output index.
    pub fn output_index(&self) -> usize {
        self.observation
            .outeq_index()
            .expect("session observations are resolved")
    }
}

/// Stateful particle simulation that pauses at each observation boundary.
pub struct SdeParticleSession<'a, R: Rng + ?Sized> {
    model: &'a SDE,
    parameters: &'a [f64],
    events: Vec<Vec<Event>>,
    covariates: Vec<Covariates>,
    occasion_indices: Vec<usize>,
    occasion: usize,
    event: usize,
    states: Vec<DVector<f64>>,
    spare: Vec<DVector<f64>>,
    infusions: Vec<Infusion>,
    predictions: Vec<Prediction>,
    observation: Option<Observation>,
    waiting: bool,
    rng: &'a mut R,
    particle_count: usize,
}

impl<R: Rng + ?Sized> SdeParticleSession<'_, R> {
    /// Return the fixed number of particles in this session.
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// Advance to the next observation, or return `None` when the schedule ends.
    pub fn next_observation(
        &mut self,
    ) -> Result<Option<SdeParticleObservation<'_>>, SdeSessionError> {
        if self.waiting {
            return Err(SdeSessionError::BoundaryPending);
        }

        loop {
            if self.occasion >= self.events.len() {
                return Ok(None);
            }
            if self.event >= self.events[self.occasion].len() {
                self.occasion += 1;
                self.event = 0;
                self.infusions.clear();
                if self.occasion >= self.events.len() {
                    return Ok(None);
                }
                self.states = self.model.initial_particles(
                    self.parameters,
                    &self.covariates[self.occasion],
                    self.occasion_indices[self.occasion],
                    self.particle_count,
                );
                continue;
            }

            let event = self.events[self.occasion][self.event].clone();
            match &event {
                Event::Bolus(bolus) => {
                    let input =
                        bolus
                            .input_index()
                            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
                            })?;
                    if input >= self.model.get_ndrugs() {
                        return Err(PharmsolError::InputOutOfRange {
                            input,
                            ndrugs: self.model.get_ndrugs(),
                        }
                        .into());
                    }
                    if !self.model.injected_bolus_mappings.apply(
                        &mut self.states,
                        input,
                        bolus.amount(),
                    ) {
                        self.states.add_bolus(input, bolus.amount());
                    }
                    self.event += 1;
                    self.advance_after(event.time());
                }
                Event::Infusion(infusion) => {
                    self.infusions.push(infusion.clone());
                    self.event += 1;
                    self.advance_after(event.time());
                }
                Event::Observation(observation) => {
                    self.predictions.clear();
                    let output_index = observation
                        .outeq_index()
                        .expect("session observations are resolved");
                    let parameter_values = V::from_vec(self.parameters.to_vec(), NalgebraContext);
                    for state in &self.states {
                        let mut output = V::zeros(self.model.get_nouteqs(), NalgebraContext);
                        (self.model.out)(
                            &state.clone().into(),
                            &parameter_values,
                            observation.time(),
                            &self.covariates[self.occasion],
                            &mut output,
                        );
                        self.predictions.push(observation.to_prediction_resolved(
                            output_index,
                            output[output_index],
                            state.as_slice().to_vec(),
                        ));
                    }
                    self.observation = Some(observation.clone());
                    self.event += 1;
                    self.waiting = true;
                    return Ok(Some(SdeParticleObservation {
                        observation: self.observation.as_ref().unwrap(),
                        predictions: &self.predictions,
                        states: &self.states,
                    }));
                }
            }
        }
    }

    /// Resume after the current boundary without changing particle states.
    pub fn retain_particles(&mut self) -> Result<(), SdeSessionError> {
        if !self.waiting {
            return Err(SdeSessionError::NoBoundary);
        }
        let time = self.observation.as_ref().unwrap().time();
        self.waiting = false;
        self.advance_after(time);
        Ok(())
    }

    /// Replace particle states from `ancestors`, then resume after the boundary.
    ///
    /// The slice must contain exactly one in-range ancestor index for each
    /// particle in the session. Indices may repeat.
    pub fn select_ancestors(&mut self, ancestors: &[usize]) -> Result<(), SdeSessionError> {
        if !self.waiting {
            return Err(SdeSessionError::NoBoundary);
        }
        if ancestors.len() != self.particle_count {
            return Err(SdeSessionError::AncestorCount {
                expected: self.particle_count,
                actual: ancestors.len(),
            });
        }
        if let Some(&index) = ancestors
            .iter()
            .find(|&&index| index >= self.particle_count)
        {
            return Err(SdeSessionError::AncestorOutOfRange {
                index,
                particle_count: self.particle_count,
            });
        }

        self.spare.clear();
        self.spare.reserve(self.particle_count);
        self.spare
            .extend(ancestors.iter().map(|&index| self.states[index].clone()));
        std::mem::swap(&mut self.states, &mut self.spare);
        let time = self.observation.as_ref().unwrap().time();
        self.waiting = false;
        self.advance_after(time);
        Ok(())
    }

    fn advance_after(&mut self, time: f64) {
        let Some(next) = self.events[self.occasion].get(self.event) else {
            return;
        };
        let end = next.time();
        if time == end {
            return;
        }
        let discontinuities = infusion_discontinuities(&self.infusions, time, end);
        let old_states = std::mem::take(&mut self.states);
        self.states = old_states
            .into_iter()
            .map(|state| {
                self.model.advance_particle(
                    state,
                    self.parameters,
                    &self.covariates[self.occasion],
                    &self.infusions,
                    time,
                    end,
                    &discontinuities,
                    self.rng,
                )
            })
            .collect();
    }
}

fn validate_metadata_dimensions(
    metadata: &ValidatedModelMetadata,
    neqs: &Neqs,
) -> Result<(), SdeMetadataError> {
    let declared_states = metadata.states().len();
    if declared_states != neqs.nstates {
        return Err(SdeMetadataError::StateCountMismatch {
            expected: neqs.nstates,
            declared: declared_states,
        });
    }

    let declared_routes = metadata.route_input_count();
    if declared_routes != neqs.ndrugs {
        return Err(SdeMetadataError::RouteCountMismatch {
            expected: neqs.ndrugs,
            declared: declared_routes,
        });
    }

    let declared_outputs = metadata.outputs().len();
    if declared_outputs != neqs.nout {
        return Err(SdeMetadataError::OutputCountMismatch {
            expected: neqs.nout,
            declared: declared_outputs,
        });
    }

    Ok(())
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
    // fn get_lag(&self, parameters: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.lag)(&V::from_vec(parameters.to_owned())))
    // }

    // #[inline(always)]
    // fn get_fa(&self, parameters: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.fa)(&V::from_vec(parameters.to_owned())))
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

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    #[inline(always)]
    fn solve(
        &self,
        state: &mut Self::S,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        let ndrugs = self.get_ndrugs();
        let discontinuities = infusion_discontinuities(infusions, ti, tf);
        state.par_iter_mut().for_each(|particle| {
            *particle = simulate_sde_event(
                &self.drift,
                &self.diffusion,
                particle.clone().into(),
                parameters,
                covariates,
                infusions,
                ndrugs,
                ti,
                tf,
                &discontinuities,
            )
            .inner()
            .clone();
        });
        Ok(())
    }
    fn nparticles(&self) -> usize {
        self.nparticles
    }

    #[inline(always)]
    fn process_observation(
        &self,
        parameters: &[f64],
        observation: &crate::Observation,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut pred = vec![Prediction::default(); self.nparticles];

        pred.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut y = V::zeros(self.get_nouteqs(), NalgebraContext::new());
            (self.out)(
                &x[i].clone().into(),
                &V::from_vec(parameters.to_vec(), NalgebraContext::new()),
                observation.time(),
                covariates,
                &mut y,
            );
            let outeq = observation
                .outeq_index()
                .expect("resolved observations should use numeric output labels");
            *p = observation.to_prediction_resolved(outeq, y[outeq], x[i].as_slice().to_vec());
        });
        let out = Array2::from_shape_vec((self.nparticles, 1), pred)?;
        *output = concatenate(Axis(1), &[output.view(), out.view()]).unwrap();
        Ok(())
    }
    #[inline(always)]
    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::S {
        let mut x = Vec::with_capacity(self.nparticles);
        for _ in 0..self.nparticles {
            let mut state: V = DVector::zeros(self.get_nstates()).into();
            if occasion_index == 0 {
                (self.init)(
                    &V::from_vec(parameters.to_vec(), NalgebraContext::new()),
                    0.0,
                    covariates,
                    &mut state,
                );
            }
            x.push(state.inner().clone());
        }
        x
    }

    fn simulate_event(
        &self,
        parameters: &[f64],
        event: &crate::Event,
        next_event: Option<&crate::Event>,
        covariates: &Covariates,
        x: &mut Self::S,
        infusions: &mut Vec<Infusion>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        match event {
            crate::Event::Bolus(bolus) => {
                let input = bolus.input_index().ok_or_else(|| {
                    let available = self
                        .metadata()
                        .map(|m| m.route_labels())
                        .unwrap_or_default();
                    PharmsolError::unknown_input_label(bolus.input(), &available)
                })?;

                if input >= self.get_ndrugs() {
                    return Err(PharmsolError::InputOutOfRange {
                        input,
                        ndrugs: self.get_ndrugs(),
                    });
                }
                if !self.injected_bolus_mappings.apply(x, input, bolus.amount()) {
                    x.add_bolus(input, bolus.amount());
                }
            }
            crate::Event::Infusion(infusion) => {
                infusions.push(infusion.clone());
            }
            crate::Event::Observation(observation) => {
                self.process_observation(
                    parameters,
                    observation,
                    event.time(),
                    covariates,
                    x,
                    output,
                )?;
            }
        }

        if let Some(next_event) = next_event {
            self.solve(
                x,
                parameters,
                covariates,
                infusions,
                event.time(),
                next_event.time(),
            )?;
        }
        Ok(())
    }
}

impl Equation for SDE {
    fn kind() -> EqnKind {
        EqnKind::SDE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::equation::{self, Covariate, Route};
    use crate::{fa, fetch_params, lag};
    use crate::{Subject, SubjectBuilderExt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn simple_sde() -> SDE {
        let drift = |x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx[0] = rateiv[0] - x[0];
        };
        let diffusion = |_p: &V, g: &mut V| {
            g[0] = 1.0;
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x[0] = 0.0;
        };
        let out = |x: &V, p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        };

        SDE::new(drift, diffusion, lag, fa, init, out, 128)
            .with_nstates(1)
            .with_ndrugs(1)
            .with_nout(1)
    }

    fn route_policy_sde(drift: Drift) -> SDE {
        let diffusion = |_p: &V, sigma: &mut V| {
            sigma.fill(0.0);
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[1];
        };

        SDE::new(drift, diffusion, lag, fa, init, out, 16)
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
    }

    #[test]
    fn handwritten_sde_metadata_exposes_name_lookup_and_particles() {
        let sde = simple_sde()
            .with_metadata(
                equation::metadata::new("one_cmt_sde")
                    .parameters(["ke", "v"])
                    .covariates([Covariate::continuous("wt")])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect("SDE metadata attachment should validate");

        let metadata = sde.metadata().expect("metadata exists");
        assert_eq!(metadata.kind(), ModelKind::Sde);
        assert_eq!(metadata.particles(), Some(128));
        assert_eq!(sde.parameter_index("ke"), Some(0));
        assert_eq!(sde.parameter_index("v"), Some(1));
        assert_eq!(sde.covariate_index("wt"), Some(0));
        assert_eq!(sde.state_index("central"), Some(0));
        assert!(metadata.route("iv").is_some());
        assert!(metadata.output("cp").is_some());
    }

    #[test]
    fn handwritten_sde_metadata_resolves_raw_numeric_aliases_against_canonical_labels() {
        let drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let diffusion = |_p: &V, sigma: &mut V| {
            sigma.fill(0.0);
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
        let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[1];
        };

        let sde = SDE::new(drift, diffusion, lag, fa, init, out, 16)
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(
                equation::metadata::new("numeric_alias_sde")
                    .states(["depot", "central"])
                    .outputs(["outeq_1"])
                    .route(Route::infusion("input_1").to_state("central"))
                    .particles(16),
            )
            .expect("SDE metadata attachment should validate");

        let canonical = Subject::builder("canonical")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build();
        let aliased = Subject::builder("aliased")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "1")
            .build();

        let canonical_predictions = sde
            .estimate_predictions(&canonical, &crate::parameters::dense([]))
            .expect("canonical labels should simulate");
        let aliased_predictions = sde
            .estimate_predictions(&aliased, &crate::parameters::dense([]))
            .expect("raw numeric aliases should simulate");

        assert!(
            (canonical_predictions[[0, 0]].prediction() - aliased_predictions[[0, 0]].prediction())
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn handwritten_sde_without_metadata_keeps_raw_path() {
        let sde = simple_sde();

        assert!(sde.metadata().is_none());
        assert_eq!(sde.parameter_index("ke"), None);
    }

    #[test]
    fn handwritten_sde_rejects_dimension_mismatches() {
        let error = simple_sde()
            .with_metadata(
                equation::metadata::new("bad_sde")
                    .parameters(["ke", "v"])
                    .states(["central", "peripheral"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect_err("mismatched state metadata must fail");

        assert_eq!(
            error,
            SdeMetadataError::StateCountMismatch {
                expected: 1,
                declared: 2,
            }
        );
    }

    #[test]
    fn handwritten_sde_rejects_particle_mismatch() {
        let error = simple_sde()
            .with_metadata(
                equation::metadata::new("particle_conflict")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(64),
            )
            .expect_err("mismatched SDE particles must fail");

        assert_eq!(
            error,
            SdeMetadataError::Validation(ModelMetadataError::ParticleCountConflict {
                declared: 64,
                fallback: 128,
            })
        );
    }

    #[test]
    fn changing_dimensions_after_metadata_clears_sde_metadata() {
        let sde = simple_sde()
            .with_metadata(
                equation::metadata::new("one_cmt_sde")
                    .parameters(["ke", "v"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(128),
            )
            .expect("metadata attachment should validate")
            .with_nout(2);

        assert!(sde.metadata().is_none());
    }

    #[test]
    fn sde_metadata_input_policy_is_descriptive_only_for_bolus_routes() {
        let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
        };

        let explicit = route_policy_sde(zero_drift)
            .with_metadata(
                equation::metadata::new("explicit_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::bolus("oral").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(zero_drift)
            .with_metadata(
                equation::metadata::new("injected_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::bolus("oral")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate");

        let subject = Subject::builder("bolus_route")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.1, "cp")
            .build();

        let explicit_predictions = explicit
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();
        let injected_predictions = injected
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert_eq!(explicit_predictions[[0, 0]].prediction(), 0.0);
        assert_eq!(injected_predictions[[0, 0]].prediction(), 0.0);
    }

    #[test]
    fn sde_metadata_input_policy_does_not_change_explicit_rateiv_behavior() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };

        let explicit = route_policy_sde(rateiv_drift)
            .with_metadata(
                equation::metadata::new("explicit_infusion")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(Route::infusion("iv").to_state("central"))
                    .particles(16),
            )
            .expect("explicit metadata should validate");

        let injected = route_policy_sde(rateiv_drift)
            .with_metadata(
                equation::metadata::new("injected_infusion")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::infusion("iv")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate");

        let subject = Subject::builder("infusion_route")
            .infusion(0.0, 100.0, "iv", 1.0)
            .missing_observation(1.0, "cp")
            .build();

        let explicit_predictions = explicit
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();
        let injected_predictions = injected
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        let explicit_prediction = explicit_predictions[[0, 0]].prediction();
        let injected_prediction = injected_predictions[[0, 0]].prediction();

        assert!(explicit_prediction > 0.0);
        assert!((injected_prediction - explicit_prediction).abs() < 1e-8);
    }

    #[test]
    fn standard_sde_short_infusion_stops_at_boundary() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let sde = route_policy_sde(rateiv_drift);
        let subject = Subject::builder("short-infusion")
            .infusion(0.0, 2.5, 0, 0.025)
            .missing_observation(0.025, 0)
            .missing_observation(0.05, 0)
            .build();

        let predictions = sde
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert!((predictions[[0, 0]].prediction() - 2.5).abs() < 1e-12);
        assert!((predictions[[1, 0]].prediction() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn standard_sde_segments_at_infusion_end_between_events() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let sde = route_policy_sde(rateiv_drift);
        let subject = Subject::builder("segmented-standard")
            .infusion(0.0, 3.0, 0, 0.075)
            .missing_observation(0.1, 0)
            .build();

        let predictions = sde
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert!((predictions[[0, 0]].prediction() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn standard_sde_infusion_is_inactive_at_end_and_after() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let sde = route_policy_sde(rateiv_drift);
        let subject = Subject::builder("half-open-standard")
            .infusion(0.0, 100.0, 0, 1.0)
            .missing_observation(1.0, 0)
            .missing_observation(1.2, 0)
            .build();

        let predictions = sde
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert!((predictions[[0, 0]].prediction() - 100.0).abs() < 1e-10);
        assert!((predictions[[1, 0]].prediction() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn stateful_sde_infusion_is_inactive_at_end_and_after() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let sde = route_policy_sde(rateiv_drift);
        let subject = Subject::builder("half-open-session")
            .infusion(0.0, 100.0, 0, 1.0)
            .missing_observation(1.0, 0)
            .missing_observation(1.2, 0)
            .build();
        let parameters = crate::parameters::dense([0.0]);
        let mut rng = StdRng::seed_from_u64(41);
        let mut session = sde
            .particle_session(&subject, &parameters, 4, &mut rng)
            .unwrap();

        {
            let boundary = session.next_observation().unwrap().unwrap();
            assert_eq!(boundary.time(), 1.0);
            assert!(boundary
                .predictions()
                .iter()
                .all(|prediction| (prediction.prediction() - 100.0).abs() < 1e-10));
        }
        session.retain_particles().unwrap();
        {
            let boundary = session.next_observation().unwrap().unwrap();
            assert_eq!(boundary.time(), 1.2);
            assert!(boundary
                .predictions()
                .iter()
                .all(|prediction| (prediction.prediction() - 100.0).abs() < 1e-10));
        }
    }

    #[test]
    fn stateful_sde_segments_at_infusion_end_between_events() {
        let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
            dx[1] = rateiv[0];
        };
        let sde = route_policy_sde(rateiv_drift);
        let subject = Subject::builder("segmented-session")
            .infusion(0.0, 3.0, 0, 0.075)
            .missing_observation(0.1, 0)
            .build();
        let parameters = crate::parameters::dense([0.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut session = sde
            .particle_session(&subject, &parameters, 4, &mut rng)
            .unwrap();

        let boundary = session.next_observation().unwrap().unwrap();
        assert_eq!(boundary.time(), 0.1);
        assert!(boundary
            .predictions()
            .iter()
            .all(|prediction| (prediction.prediction() - 3.0).abs() < 1e-12));
    }

    #[test]
    fn handwritten_sde_rejects_out_of_range_numeric_event_labels() {
        let model = simple_sde();
        let parameters = crate::parameters::dense([0.0, 1.0]);
        for subject in [
            Subject::builder("bad-bolus").bolus(0.0, 1.0, 1).build(),
            Subject::builder("bad-infusion")
                .infusion(0.0, 1.0, 1, 1.0)
                .build(),
        ] {
            assert!(matches!(
                model.simulate_subject(&subject, &parameters),
                Err(PharmsolError::InputOutOfRange {
                    input: 1,
                    ndrugs: 1
                })
            ));
        }
        let subject = Subject::builder("bad-output")
            .observation(0.0, 0.0, 1)
            .build();
        assert!(matches!(
            model.simulate_subject(&subject, &parameters),
            Err(PharmsolError::OuteqOutOfRange { outeq: 1, nout: 1 })
        ));
    }

    #[test]
    fn particle_session_validates_boundaries_and_ancestor_selection() {
        let model = simple_sde();
        let subject = Subject::builder("session-validation")
            .missing_observation(1.0, 0)
            .missing_observation(1.0, 0)
            .build();
        let parameters = crate::parameters::dense([0.0, 1.0]);
        let mut rng = StdRng::seed_from_u64(99);
        let mut session = model
            .particle_session(&subject, &parameters, 4, &mut rng)
            .unwrap();

        assert!(matches!(
            session.retain_particles(),
            Err(SdeSessionError::NoBoundary)
        ));
        let original = session
            .next_observation()
            .unwrap()
            .unwrap()
            .states()
            .to_vec();
        assert!(matches!(
            session.next_observation(),
            Err(SdeSessionError::BoundaryPending)
        ));
        assert!(matches!(
            session.select_ancestors(&[0, 1]),
            Err(SdeSessionError::AncestorCount {
                expected: 4,
                actual: 2
            })
        ));
        assert!(matches!(
            session.select_ancestors(&[0, 1, 2, 4]),
            Err(SdeSessionError::AncestorOutOfRange {
                index: 4,
                particle_count: 4
            })
        ));

        session.select_ancestors(&[3, 2, 1, 0]).unwrap();
        let selected = session.next_observation().unwrap().unwrap();
        for (actual, expected) in selected.states().iter().zip(original.iter().rev()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn clearing_sde_metadata_preserves_raw_bolus_behavior() {
        let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
            dx.fill(0.0);
        };

        let sde = route_policy_sde(zero_drift)
            .with_metadata(
                equation::metadata::new("injected_bolus")
                    .parameters(["theta"])
                    .states(["depot", "central"])
                    .outputs(["cp"])
                    .route(
                        Route::bolus("oral")
                            .to_state("central")
                            .inject_input_to_destination(),
                    )
                    .particles(16),
            )
            .expect("injected metadata should validate")
            .with_nout(1);

        let subject = Subject::builder("bolus_route")
            .bolus(0.0, 100.0, 0)
            .missing_observation(0.1, 0)
            .build();

        let predictions = sde
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert!(sde.metadata().is_none());
        assert_eq!(predictions[[0, 0]].prediction(), 0.0);
    }
}
