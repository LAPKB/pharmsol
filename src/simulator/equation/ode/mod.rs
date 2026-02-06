mod closure;

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, PharmsolError, Subject,
};

use super::id_hash;
use super::spphash;
use crate::simulator::cache::{cache_enabled, ode_cache_lock_read};
use crate::simulator::equation::Predictions;
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, OdeBuilder,
    OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;

use super::{Equation, EquationPriv, EquationTypes, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

/// ODE solver selection.
///
/// Each variant corresponds to a solver family from diffsol.
/// `Sdirk` and `ExplicitRk` take a tableau that determines the specific method.
///
/// ```ignore
/// // Implicit multistep (stiff, default):
/// OdeSolver::Bdf
///
/// // Implicit single-step with a chosen tableau:
/// OdeSolver::Sdirk(SdirkTableau::TrBdf2)
/// OdeSolver::Sdirk(SdirkTableau::Esdirk34)
///
/// // Explicit Runge-Kutta — fastest for non-stiff problems:
/// OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)
/// ```
#[derive(Clone, Debug, Default)]
pub enum OdeSolver {
    /// Backward Differentiation Formulae — implicit multistep, best for stiff problems
    #[default]
    Bdf,
    /// Singly Diagonally Implicit Runge-Kutta
    Sdirk(SdirkTableau),
    /// Explicit Runge-Kutta — no Jacobian needed
    ExplicitRk(ExplicitRkTableau),
}

/// Tableau for [`OdeSolver::Sdirk`].
#[derive(Clone, Debug)]
pub enum SdirkTableau {
    /// TR-BDF2 — good all-rounder for moderately stiff problems
    TrBdf2,
    /// ESDIRK3(4) — higher accuracy for stiff problems
    Esdirk34,
}

/// Tableau for [`OdeSolver::ExplicitRk`].
#[derive(Clone, Debug)]
pub enum ExplicitRkTableau {
    /// Tsitouras 5(4) — fastest for non-stiff problems
    Tsit45,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ODE {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
}

impl ODE {
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Self {
            diffeq,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            solver: OdeSolver::default(),
            rtol: RTOL,
            atol: ATOL,
        }
    }

    /// Set the number of state variables (ODE compartments).
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

    /// Set the ODE solver algorithm.
    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the relative and absolute tolerances for the ODE solver.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }
}

impl State for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    let ypred = _subject_predictions(ode, subject, support_point)?;
    Ok(ypred.log_likelihood(error_models)?.exp())
}

#[inline(always)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> Result<SubjectPredictions, PharmsolError> {
    if cache_enabled() {
        let key = (id_hash(subject.id()), spphash(support_point));
        let cache_guard = ode_cache_lock_read()?;
        if let Some(cached) = cache_guard.get(&key) {
            return Ok(cached);
        }
        drop(cache_guard);

        let result = ode.simulate_subject(subject, support_point, None)?.0;
        let cache_guard = ode_cache_lock_read()?;
        cache_guard.insert(key, result.clone());
        Ok(result)
    } else {
        Ok(ode.simulate_subject(subject, support_point, None)?.0)
    }
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
        _error_models: Option<&AssayErrorModels>,
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

impl ODE {
    /// Generic event-loop runner, parameterized over the concrete solver type.
    #[allow(clippy::too_many_arguments)]
    fn run_events<'a, S: OdeSolverMethod<'a, PMProblem<'a, DiffEq>>>(
        &self,
        solver: &mut S,
        events: &[Event],
        spp_v: &V,
        covariates: &Covariates,
        error_models: Option<&AssayErrorModels>,
        bolus_v: &mut V,
        zero_bolus: &V,
        zero_rateiv: &V,
        state_with_bolus: &mut V,
        state_without_bolus: &mut V,
        y_out: &mut V,
        likelihood: &mut Vec<f64>,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError> {
        for (index, event) in events.iter().enumerate() {
            let next_event = events.get(index + 1);

            match event {
                Event::Bolus(bolus) => {
                    if bolus.input() >= bolus_v.len() {
                        return Err(PharmsolError::InputOutOfRange {
                            input: bolus.input(),
                            ndrugs: bolus_v.len(),
                        });
                    }
                    bolus_v.fill(0.0);
                    bolus_v[bolus.input()] = bolus.amount();

                    state_with_bolus.fill(0.0);
                    state_without_bolus.fill(0.0);

                    (self.diffeq)(
                        solver.state().y,
                        spp_v,
                        event.time(),
                        state_without_bolus,
                        zero_bolus,
                        zero_rateiv,
                        covariates,
                    );

                    (self.diffeq)(
                        solver.state().y,
                        spp_v,
                        event.time(),
                        state_with_bolus,
                        bolus_v,
                        zero_rateiv,
                        covariates,
                    );

                    state_with_bolus.axpy(-1.0, state_without_bolus, 1.0);
                    solver.state_mut().y.axpy(1.0, state_with_bolus, 1.0);
                }
                Event::Infusion(_) => {
                    // Infusions are handled within the ODE function itself
                }
                Event::Observation(observation) => {
                    y_out.fill(0.0);
                    (self.out)(
                        solver.state().y,
                        spp_v,
                        observation.time(),
                        covariates,
                        y_out,
                    );
                    let pred = y_out[observation.outeq()];
                    let pred =
                        observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                    if let Some(error_models) = error_models {
                        likelihood.push(pred.log_likelihood(error_models)?.exp());
                    }
                    output.add_prediction(pred);
                }
            }

            // Advance to the next event time if it exists
            if let Some(next_event) = next_event {
                if !event.time().eq(&next_event.time()) {
                    match solver.set_stop_time(next_event.time()) {
                        Ok(_) => loop {
                            match solver.step() {
                                Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                Ok(OdeSolverStopReason::TstopReached) => break,
                                Err(diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StepSizeTooSmall { time },
                                )) => {
                                    return Err(PharmsolError::OtherError(format!(
                                        "ODE solver step size went to zero at t = {time:.4} (target t = {:.4}). \
                                         A parameter is likely near 0 or infinite.",
                                        next_event.time()
                                    )));
                                }
                                Err(_) | Ok(_) => {
                                    return Err(PharmsolError::OtherError(
                                        "Unexpected solver error".to_string(),
                                    ));
                                }
                            }
                        },
                        Err(diffsol::error::DiffsolError::OdeSolverError(
                            OdeSolverError::StopTimeAtCurrentTime,
                        )) => {
                            continue;
                        }
                        Err(_) => {
                            return Err(PharmsolError::OtherError(
                                "Unexpected solver error".to_string(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Equation for ODE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = _subject_predictions(self, subject, support_point)?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let mut output = Self::P::new(self.nparticles());

        // Preallocate likelihood vector
        let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
        let mut likelihood = Vec::with_capacity(event_count);

        // Cache dimensions to avoid repeated method calls
        let nstates = self.get_nstates();
        let ndrugs = self.get_ndrugs();

        let state_buffer_size = nstates;
        let output_buffer_size = self.get_nouteqs();

        // Preallocate reusable vectors for bolus computation
        let mut state_with_bolus = V::zeros(state_buffer_size, NalgebraContext);
        let mut state_without_bolus = V::zeros(state_buffer_size, NalgebraContext);
        let zero_vector = V::zeros(state_buffer_size, NalgebraContext);
        let mut bolus_v = V::zeros(state_buffer_size, NalgebraContext);
        let spp_v: V = DVector::from_vec(support_point.clone()).into();

        // Pre-allocate output vector for observations
        let mut y_out = V::zeros(output_buffer_size, NalgebraContext);

        // Iterate over occasions
        for occasion in subject.occasions() {
            let covariates = occasion.covariates();
            let infusions = occasion.infusions_ref();
            let events = occasion.process_events(
                Some((self.fa(), self.lag(), support_point, covariates)),
                true,
            );

            let problem = OdeBuilder::<M>::new()
                .atol(vec![self.atol])
                .rtol(self.rtol)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.clone())
                .build_from_eqn(PMProblem::with_params_v(
                    self.diffeq,
                    nstates,
                    ndrugs,
                    support_point.clone(),
                    spp_v.clone(),
                    covariates,
                    infusions.as_slice(),
                    self.initial_state(support_point, covariates, occasion.index())
                        .into(),
                )?)?;

            match &self.solver {
                OdeSolver::Bdf => {
                    let mut solver = problem.bdf::<diffsol::NalgebraLU<f64>>()?;
                    Self::run_events(
                        self,
                        &mut solver,
                        &events,
                        &spp_v,
                        covariates,
                        error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => {
                    let mut solver = problem.tsit45()?;
                    Self::run_events(
                        self,
                        &mut solver,
                        &events,
                        &spp_v,
                        covariates,
                        error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::Sdirk(SdirkTableau::TrBdf2) => {
                    let mut solver = problem.tr_bdf2::<diffsol::NalgebraLU<f64>>()?;
                    Self::run_events(
                        self,
                        &mut solver,
                        &events,
                        &spp_v,
                        covariates,
                        error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
                OdeSolver::Sdirk(SdirkTableau::Esdirk34) => {
                    let mut solver = problem.esdirk34::<diffsol::NalgebraLU<f64>>()?;
                    Self::run_events(
                        self,
                        &mut solver,
                        &events,
                        &spp_v,
                        covariates,
                        error_models,
                        &mut bolus_v,
                        &zero_bolus,
                        &zero_rateiv,
                        &mut state_with_bolus,
                        &mut state_without_bolus,
                        &mut y_out,
                        &mut likelihood,
                        &mut output,
                    )?;
                }
            }
        }
        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}
