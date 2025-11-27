mod closure;

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, PharmsolError, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use crate::simulator::equation::Predictions;
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, Bdf, NalgebraContext,
    NewtonNonlinearSolver, OdeBuilder, OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;

use super::{Equation, EquationPriv, EquationTypes, Missing, Provided, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ODE {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
}

impl ODE {
    /// Creates a new ODE equation.
    ///
    /// For a more ergonomic API, consider using [`ODEBuilder`] instead.
    ///
    /// # Parameters
    /// - `diffeq`: The differential equation closure
    /// - `lag`: Lag time function
    /// - `fa`: Bioavailability function
    /// - `init`: Initial state function
    /// - `out`: Output equation function
    /// - `neqs`: Number of states and output equations (can be a tuple or [`Neqs`])
    pub fn new(
        diffeq: DiffEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: impl Into<Neqs>,
    ) -> Self {
        Self {
            diffeq,
            lag,
            fa,
            init,
            out,
            neqs: neqs.into(),
        }
    }

    /// Returns a new [`ODEBuilder`] for constructing an ODE equation.
    ///
    /// # Example
    /// ```ignore
    /// use pharmsol::prelude::*;
    ///
    /// // Minimal builder - only required fields
    /// let ode = ODE::builder()
    ///     .diffeq(diffeq)
    ///     .out(out)
    ///     .nstates(2)
    ///     .nouteqs(1)
    ///     .build();
    ///
    /// // With optional fields
    /// let ode = ODE::builder()
    ///     .diffeq(diffeq)
    ///     .out(out)
    ///     .nstates(2)
    ///     .nouteqs(1)
    ///     .lag(|p, _t, _cov| lag! { 0 => p[2] })
    ///     .fa(|p, _t, _cov| fa! { 0 => 0.8 })
    ///     .init(|p, _t, _cov, x| { x[0] = p[3]; })
    ///     .build();
    /// ```
    pub fn builder() -> ODEBuilder<Missing, Missing, Missing, Missing> {
        ODEBuilder::new()
    }
}

// =============================================================================
// Type-State Builder Pattern
// =============================================================================

// Note: Missing and Provided marker types are defined in the parent module
// and imported via `use super::{..., Missing, Provided, ...}`

/// Builder for constructing [`ODE`] equations with compile-time validation.
///
/// This builder uses the type-state pattern to ensure all required fields
/// are set before `build()` can be called. Optional fields (`lag`, `fa`, `init`)
/// have sensible defaults.
///
/// # Required Fields (enforced at compile time)
/// - `diffeq`: The differential equation closure
/// - `out`: Output equation function
/// - `nstates`: Number of state variables
/// - `nouteqs`: Number of output equations
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
/// let ode = ODE::builder()
///     .diffeq(|x, p, _t, dx, _b, rateiv, _cov| {
///         fetch_params!(p, ke, _v);
///         dx[0] = -ke * x[0] + rateiv[0];
///     })
///     .out(|x, p, _t, _cov, y| {
///         fetch_params!(p, _ke, v);
///         y[0] = x[0] / v;
///     })
///     .nstates(1)
///     .nouteqs(1)
///     .build();
/// ```
pub struct ODEBuilder<DiffEqState, OutState, NStatesState, NOuteqsState> {
    diffeq: Option<DiffEq>,
    lag: Option<Lag>,
    fa: Option<Fa>,
    init: Option<Init>,
    out: Option<Out>,
    nstates: Option<usize>,
    nouteqs: Option<usize>,
    _phantom: PhantomData<(DiffEqState, OutState, NStatesState, NOuteqsState)>,
}

impl ODEBuilder<Missing, Missing, Missing, Missing> {
    /// Creates a new ODEBuilder with all required fields unset.
    pub fn new() -> Self {
        Self {
            diffeq: None,
            lag: None,
            fa: None,
            init: None,
            out: None,
            nstates: None,
            nouteqs: None,
            _phantom: PhantomData,
        }
    }
}

impl Default for ODEBuilder<Missing, Missing, Missing, Missing> {
    fn default() -> Self {
        Self::new()
    }
}

impl<DiffEqState, OutState, NStatesState, NOuteqsState>
    ODEBuilder<DiffEqState, OutState, NStatesState, NOuteqsState>
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

impl<OutState, NStatesState, NOuteqsState>
    ODEBuilder<Missing, OutState, NStatesState, NOuteqsState>
{
    /// Sets the differential equation closure (required).
    ///
    /// This closure defines the system of ODEs: dx/dt = f(x, p, t, ...)
    ///
    /// # Parameters
    /// The closure receives:
    /// - `x`: Current state vector
    /// - `p`: Parameter vector
    /// - `t`: Current time
    /// - `dx`: Output vector for derivatives (mutated by the closure)
    /// - `bolus`: Bolus amounts
    /// - `rateiv`: IV infusion rates
    /// - `cov`: Covariates
    pub fn diffeq(
        self,
        diffeq: DiffEq,
    ) -> ODEBuilder<Provided, OutState, NStatesState, NOuteqsState> {
        ODEBuilder {
            diffeq: Some(diffeq),
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<DiffEqState, NStatesState, NOuteqsState>
    ODEBuilder<DiffEqState, Missing, NStatesState, NOuteqsState>
{
    /// Sets the output equation function (required).
    ///
    /// This closure computes observable outputs from the state.
    pub fn out(self, out: Out) -> ODEBuilder<DiffEqState, Provided, NStatesState, NOuteqsState> {
        ODEBuilder {
            diffeq: self.diffeq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: Some(out),
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<DiffEqState, OutState, NOuteqsState> ODEBuilder<DiffEqState, OutState, Missing, NOuteqsState> {
    /// Sets the number of state variables (compartments) (required).
    pub fn nstates(
        self,
        nstates: usize,
    ) -> ODEBuilder<DiffEqState, OutState, Provided, NOuteqsState> {
        ODEBuilder {
            diffeq: self.diffeq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(nstates),
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<DiffEqState, OutState, NStatesState> ODEBuilder<DiffEqState, OutState, NStatesState, Missing> {
    /// Sets the number of output equations (required).
    pub fn nouteqs(
        self,
        nouteqs: usize,
    ) -> ODEBuilder<DiffEqState, OutState, NStatesState, Provided> {
        ODEBuilder {
            diffeq: self.diffeq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: Some(nouteqs),
            _phantom: PhantomData,
        }
    }
}

impl<DiffEqState, OutState> ODEBuilder<DiffEqState, OutState, Missing, Missing> {
    /// Sets both nstates and nouteqs from a [`Neqs`] struct or tuple (required).
    pub fn neqs(
        self,
        neqs: impl Into<Neqs>,
    ) -> ODEBuilder<DiffEqState, OutState, Provided, Provided> {
        let neqs = neqs.into();
        ODEBuilder {
            diffeq: self.diffeq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(neqs.nstates),
            nouteqs: Some(neqs.nouteqs),
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

impl ODEBuilder<Provided, Provided, Provided, Provided> {
    /// Builds the [`ODE`] equation.
    ///
    /// This method is only available when all required fields have been set:
    /// - `diffeq`
    /// - `out`
    /// - `nstates`
    /// - `nouteqs`
    ///
    /// Optional fields use defaults if not set:
    /// - `lag`: No lag (empty HashMap)
    /// - `fa`: 100% bioavailability (empty HashMap)
    /// - `init`: Zero initial state
    pub fn build(self) -> ODE {
        ODE {
            diffeq: self.diffeq.unwrap(),
            lag: self.lag.unwrap_or(default_lag),
            fa: self.fa.unwrap_or(default_fa),
            init: self.init.unwrap_or(default_init),
            out: self.out.unwrap(),
            neqs: Neqs::new(self.nstates.unwrap(), self.nouteqs.unwrap()),
        }
    }
}

impl State for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

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

/// Hash a subject ID string to u64 for cache key generation.

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &ErrorModels,
    cache: bool,
) -> Result<f64, PharmsolError> {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    }?;
    ypred.likelihood(error_models)
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<(u64, u64), SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ ((subject.hash()), spphash(support_point)) }"#,
    result = "true"
)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> Result<SubjectPredictions, PharmsolError> {
    Ok(ode.simulate_subject(subject, support_point, None)?.0)
}

impl EquationTypes for ODE {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for ODE {
    //#[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let spp = DVector::from_vec(spp.to_vec());
    //     Some((self.lag)(&spp))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let spp = DVector::from_vec(spp.to_vec());
    //     Some((self.fa)(&spp))
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
        _state: &mut Self::S,
        _support_point: &Vec<f64>,
        _covariates: &Covariates,
        _infusions: &Vec<Infusion>,
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve not implemented for ODE");
    }
    #[inline(always)]
    fn process_observation(
        &self,
        _support_point: &Vec<f64>,
        _observation: &Observation,
        _error_models: Option<&ErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation not implemented for ODE");
    }

    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates(), NalgebraContext);
        if occasion_index == 0 {
            let spp = DVector::from_vec(spp.clone());
            (init)(&spp.into(), 0.0, covariates, &mut x);
        }
        x
    }
}

impl Equation for ODE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models, cache)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        let ypred = if cache {
            _subject_predictions(self, subject, support_point)
        } else {
            _subject_predictions_no_cache(self, subject, support_point)
        }?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: Option<&ErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let mut output = Self::P::new(self.nparticles());

        // Preallocate likelihood vector
        let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
        let mut likelihood = Vec::with_capacity(event_count);

        // Cache nstates to avoid repeated method calls
        let nstates = self.get_nstates();

        // Preallocate reusable vectors for bolus computation
        let mut state_with_bolus = V::zeros(nstates, NalgebraContext);
        let mut state_without_bolus = V::zeros(nstates, NalgebraContext);
        let zero_vector = V::zeros(nstates, NalgebraContext);
        let mut bolus_v = V::zeros(nstates, NalgebraContext);
        let spp_v: V = DVector::from_vec(support_point.clone()).into();

        // Pre-allocate output vector for observations
        let mut y_out = V::zeros(self.get_nouteqs(), NalgebraContext);

        // Iterate over occasions
        for occasion in subject.occasions() {
            let covariates = occasion.covariates();
            let infusions = occasion.infusions_ref();
            let events = occasion.process_events(
                Some((self.fa(), self.lag(), support_point, covariates)),
                true,
            );

            let problem = OdeBuilder::<M>::new()
                .atol(vec![ATOL])
                .rtol(RTOL)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.clone())
                .build_from_eqn(PMProblem::with_params_v(
                    self.diffeq,
                    nstates,
                    support_point.clone(),
                    spp_v.clone(), // Reuse pre-converted V
                    covariates,
                    infusions.as_slice(),
                    self.initial_state(support_point, covariates, occasion.index())
                        .into(),
                ))?;

            let mut solver: Bdf<
                '_,
                PMProblem<DiffEq>,
                NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
            > = problem.bdf::<diffsol::NalgebraLU<f64>>()?;

            // Iterate over events
            for (index, event) in events.iter().enumerate() {
                // Check if we have a next event
                let next_event = events.get(index + 1);

                // Handle events accordingly
                match event {
                    Event::Bolus(bolus) => {
                        // Reset and reuse the pre-allocated bolus vector
                        bolus_v.fill(0.0);
                        bolus_v[bolus.input()] = bolus.amount();

                        // Reset and reuse the bolus changes vectors
                        state_with_bolus.fill(0.0);
                        state_without_bolus.fill(0.0);

                        // Call the differential equation closure without bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_without_bolus,
                            &zero_vector,
                            &zero_vector,
                            covariates,
                        );

                        // Call the differential equation closure with bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_with_bolus,
                            &bolus_v,
                            &zero_vector,
                            covariates,
                        );

                        // The difference between the two states is the actual bolus effect
                        // Apply the computed changes to the state using vectorized operations
                        // state_with_bolus now contains (with_bolus - without_bolus) after axpy
                        state_with_bolus.axpy(-1.0, &state_without_bolus, 1.0);

                        // Add the difference to the solver state
                        solver.state_mut().y.axpy(1.0, &state_with_bolus, 1.0);
                    }
                    Event::Infusion(_infusion) => {
                        // Infusions are handled within the ODE function itself
                    }
                    Event::Observation(observation) => {
                        // Reuse pre-allocated output vector
                        y_out.fill(0.0);
                        let out = &self.out;
                        (out)(
                            solver.state().y,
                            &spp_v,
                            observation.time(),
                            covariates,
                            &mut y_out,
                        );
                        let pred = y_out[observation.outeq()];
                        let pred =
                            observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                        if let Some(error_models) = error_models {
                            likelihood.push(pred.likelihood(error_models)?);
                        }
                        output.add_prediction(pred);
                    }
                }

                // Advance to the next event time if it exists
                if let Some(next_event) = next_event {
                    if !event.time().eq(&next_event.time()) {
                        match solver.set_stop_time(next_event.time()) {
                            Ok(_) => loop {
                                let ret = solver.step();
                                match ret {
                                    Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                    Ok(OdeSolverStopReason::TstopReached) => break,
                                    Err(err) => match err {
                                        diffsol::error::DiffsolError::OdeSolverError(
                                            OdeSolverError::StepSizeTooSmall { time },
                                        ) => {
                                            let _time = time;
                                            return Err(PharmsolError::OtherError("The step size of the ODE solver went to zero, this means one of your parameters is getting really close to 0.0 or INFINITE. Check your model".to_string()));
                                        }
                                        _ => {
                                            return Err(PharmsolError::OtherError(
                                                "Unexpected solver error: {:?}".to_string(),
                                            ));
                                        }
                                    },
                                    _ => {
                                        return Err(PharmsolError::OtherError(
                                            "Unexpected solver error: {:?}".to_string(),
                                        ));
                                    }
                                }
                            },
                            Err(e) => {
                                match e {
                                    diffsol::error::DiffsolError::OdeSolverError(
                                        OdeSolverError::StopTimeAtCurrentTime,
                                    ) => {
                                        // If the stop time is at the current state time, we can just continue
                                        continue;
                                    }
                                    _ => {
                                        return Err(PharmsolError::OtherError(
                                            "Unexpected solver error: {:?}".to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}
