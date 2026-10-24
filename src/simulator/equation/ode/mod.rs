//! Models integrated with an ODE solver.
//!
//! [`ODE`] pairs an [`OdeModel`] with a diffsol solver and a prediction cache.
//! The same driver runs handwritten models ([`NativeOde`], built by
//! [`ODE::new`] or [`ode!`](crate::ode!)) and DSL-compiled ones.

mod advance;
mod closure;
mod native;

pub(crate) use advance::{advance_solver_to_event, validate_resolved_ode_schedule};
pub use native::{NativeOde, NativeOdeWorkspace};

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, NalgebraLU,
    OdeBuilder, OdeSolverConfig, OdeSolverStopReason, Vector, VectorHost,
};

use crate::{
    data::Covariates,
    error_model::AssayErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{M as Matrix, V},
    Occasion, PharmsolError, Subject,
};

use super::model::{
    fill_active_rates, prepare_occasion, with_subject_context, InfusionDose, Model, Step,
};
use super::{salted_parameters_hash, Cache, Equation, ValidatedModelMetadata};
use crate::simulator::cache::{PredictionCache, PredictionKey, DEFAULT_CACHE_SIZE};
use closure::PMProblem;

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;
/// Headroom above diffsol's hard minimum after a close stop, allowing the
/// controller to shrink a restarted step before reaching that floor again.
const MINIMUM_TIMESTEP_HEADROOM: f64 = 4.0;
/// Default dimensionless TSIT45 accepted-step limit for one exact smooth
/// segment. This is a progress-check window, not model time, tolerance, or
/// wall-clock work.
pub(crate) const DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SEGMENT: usize = 500_000;
/// Minimum dimensionless fraction of the original exact segment gap that an
/// accepted-step window must cover before the same segment may continue.
const MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW: f64 = 1e-4;
/// Default cumulative dimensionless TSIT45 accepted-step limit for one
/// `PMProblem` (one subject occasion). Calibrated against long Pmetrics/PMcore
/// runs; hitting it indicates a stalled trajectory, not normal work.
pub(crate) const DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SESSION: usize = 10_000_000;

/// Accepted-step limits configured once for one `PMProblem`/occasion.
///
/// Both dimensions count accepted solver steps only. Starting another exact
/// event-loop segment resets the progress window but never the session count;
/// coordinate rebases and residual integrations do neither. `None` disables
/// both limits for implicit solver sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExplicitStepBudgetConfig {
    pub(crate) max_accepted_steps_per_segment: usize,
    pub(crate) max_accepted_steps_per_session: usize,
}

/// Return the accepted-step budgets selected for one solver session.
///
/// Only TSIT45 uses the explicit work guard. The implicit methods retain their
/// ordinary diffsol failure semantics and are never limited by either budget.
pub(crate) fn accepted_step_limits_for_solver(
    solver: &OdeSolver,
) -> Option<ExplicitStepBudgetConfig> {
    match solver {
        OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => Some(ExplicitStepBudgetConfig {
            max_accepted_steps_per_segment: DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SEGMENT,
            max_accepted_steps_per_session: DEFAULT_TSIT45_MAX_ACCEPTED_STEPS_PER_SESSION,
        }),
        OdeSolver::Bdf
        | OdeSolver::Sdirk(SdirkTableau::TrBdf2)
        | OdeSolver::Sdirk(SdirkTableau::Esdirk34) => None,
    }
}

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

/// Model functions for an [`ODE`].
pub trait OdeModel: Model {
    /// Write the derivative of `x` at `time` into `dx`.
    ///
    /// `rates` holds the active infusion rate per input.
    #[allow(clippy::too_many_arguments)]
    fn rhs(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError>;
}

/// An [`OdeModel`] solved numerically, with a prediction cache.
///
/// Caching is on by default; see [`Cache`]. Clones share the cache.
#[derive(Clone, Debug)]
pub struct ODE<M = NativeOde> {
    model: M,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
    cache: Option<PredictionCache>,
}

impl<M: OdeModel> ODE<M> {
    /// Wrap a model with the default solver, tolerances and cache.
    pub fn from_model(model: M) -> Self {
        Self {
            model,
            solver: OdeSolver::default(),
            rtol: RTOL,
            atol: ATOL,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    /// The model functions.
    pub fn model(&self) -> &M {
        &self.model
    }

    #[cfg(all(test, feature = "dsl"))]
    pub(crate) fn prediction_cache(&self) -> Option<&PredictionCache> {
        self.cache.as_ref()
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

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.state_index(name)
    }

    /// Predictions for a subject, from the cache when possible.
    fn predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Arc<SubjectPredictions>, PharmsolError> {
        let simulate = || self.simulate(subject, parameters).map(Arc::new);
        match &self.cache {
            Some(cache) => {
                cache.get_or_try_insert(self.prediction_key(subject, parameters), simulate)
            }
            None => simulate(),
        }
    }

    pub(crate) fn prediction_key(&self, subject: &Subject, parameters: &[f64]) -> PredictionKey {
        (
            subject.hash(),
            salted_parameters_hash(parameters, self.settings_salt()),
        )
    }

    /// Hash of the settings that change predictions, so clones with
    /// different settings never share cache entries.
    fn settings_salt(&self) -> u64 {
        let solver = match self.solver {
            OdeSolver::Bdf => 1,
            OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => 2,
            OdeSolver::Sdirk(SdirkTableau::TrBdf2) => 3,
            OdeSolver::Sdirk(SdirkTableau::Esdirk34) => 4,
        };
        salted_parameters_hash(&[self.rtol, self.atol], solver)
    }

    fn simulate(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        let in_context = |error| with_subject_context(&self.model, error, subject.id(), parameters);
        self.model
            .check_parameters(parameters)
            .map_err(in_context)?;

        let p = V::from_vec(parameters.to_vec(), NalgebraContext::new());
        let mut scratch = Scratch::new(&self.model);
        let mut output = SubjectPredictions::default();
        for occasion in subject.occasions() {
            self.simulate_occasion(occasion, &p, &mut scratch, &mut output)
                .map_err(in_context)?;
        }
        Ok(output)
    }

    fn simulate_occasion(
        &self,
        occasion: &Occasion,
        p: &V,
        scratch: &mut Scratch,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError> {
        let covariates = occasion.covariates();
        covariates.validate_for_ode()?;

        let mut ws = self.model.workspace();
        let schedule = prepare_occasion(&self.model, &mut ws, occasion, p)?;
        let time_origin = validate_resolved_ode_schedule(&schedule.steps)?;
        let mut x0 = V::zeros(self.model.nstates(), NalgebraContext::new());
        self.model
            .init(&mut ws, time_origin, p, covariates, &mut x0)?;

        let ws = RefCell::new(ws);
        let rhs_error = RhsError::default();
        let rhs = |x: &V, p: &V, t: f64, dx: &mut V, _bolus: &V, rates: &V, cov: &Covariates| {
            if rhs_error.is_set() {
                dx.fill(0.0);
                return;
            }
            if let Err(error) = self
                .model
                .rhs(&mut ws.borrow_mut(), t, x, p, cov, rates, dx)
            {
                dx.fill(0.0);
                rhs_error.set(error);
            }
        };

        let problem = OdeBuilder::<Matrix>::new()
            .atol(vec![self.atol])
            .rtol(self.rtol)
            .t0(0.0)
            .h0(1e-3)
            .p(p.as_slice().to_vec())
            .build_from_eqn(PMProblem::with_params_v(
                rhs,
                self.model.nstates(),
                self.model.ninputs(),
                p.clone(),
                covariates,
                &schedule.infusions,
                x0,
                time_origin,
            )?)?;
        problem
            .eqn
            .configure_explicit_step_guard(accepted_step_limits_for_solver(&self.solver));

        let run = OccasionRun {
            steps: &schedule.steps,
            infusions: &schedule.infusions,
            p,
            covariates,
            ws: &ws,
            rhs_error: &rhs_error,
        };
        match &self.solver {
            OdeSolver::Bdf => self.run_events(
                &mut problem.bdf::<NalgebraLU<f64>>()?,
                &run,
                scratch,
                output,
            ),
            OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => {
                self.run_events(&mut problem.tsit45()?, &run, scratch, output)
            }
            OdeSolver::Sdirk(SdirkTableau::TrBdf2) => self.run_events(
                &mut problem.tr_bdf2::<NalgebraLU<f64>>()?,
                &run,
                scratch,
                output,
            ),
            OdeSolver::Sdirk(SdirkTableau::Esdirk34) => self.run_events(
                &mut problem.esdirk34::<NalgebraLU<f64>>()?,
                &run,
                scratch,
                output,
            ),
        }?;

        rhs_error.take()
    }

    /// Walk the events of one occasion, integrating between them.
    fn run_events<'a, F, S>(
        &self,
        solver: &mut S,
        run: &OccasionRun<'_, M>,
        scratch: &mut Scratch,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    {
        let mut integration_boundary_cursor = 0usize;
        // Set when the previous event changed the state or the previous stop
        // was a discontinuity boundary: the solver must be restarted before the
        // first step of the next segment. Deferred until `set_stop_time`
        // succeeds so a stop that is already reached does not trigger a
        // restart for a zero-length segment.
        let mut pending_reinit = false;

        for (index, step) in run.steps.iter().enumerate() {
            match *step {
                Step::Bolus {
                    time,
                    input,
                    amount,
                } => {
                    self.model.bolus(
                        &mut run.ws.borrow_mut(),
                        time,
                        solver.state_mut().y,
                        run.p,
                        run.covariates,
                        input,
                        amount,
                    )?;
                    pending_reinit = true;
                }
                // Infusions enter through the rates the RHS receives.
                Step::Infusion(_) => {}
                Step::Observation { observation, outeq } => {
                    run.rhs_error.take()?;
                    fill_active_rates(run.infusions, observation.time(), &mut scratch.rates);
                    scratch.y.fill(0.0);
                    self.model.outputs(
                        &mut run.ws.borrow_mut(),
                        observation.time(),
                        solver.state().y,
                        run.p,
                        run.covariates,
                        &scratch.rates,
                        &mut scratch.y,
                    )?;
                    output.add_prediction(
                        observation
                            .to_prediction(scratch.y[outeq], solver.state().y.as_slice().to_vec()),
                    );
                }
            }

            if let Some(next) = run.steps.get(index + 1) {
                advance_solver_to_event(
                    solver,
                    next.time(),
                    &mut integration_boundary_cursor,
                    &mut pending_reinit,
                    &mut scratch.dy,
                    self.rtol,
                    self.atol,
                    &mut || run.rhs_error.take(),
                )?;
            }
        }
        Ok(())
    }
}

/// Everything [`ODE::run_events`] needs about the occasion being solved.
struct OccasionRun<'a, M: Model> {
    steps: &'a [Step<'a>],
    infusions: &'a [InfusionDose],
    p: &'a V,
    covariates: &'a Covariates,
    ws: &'a RefCell<M::Workspace>,
    rhs_error: &'a RhsError,
}

/// The first error raised inside the RHS.
///
/// diffsol cannot propagate errors out of the RHS, so a failure is parked
/// here (the RHS returns zeros from then on) and surfaced at the next stop.
#[derive(Default)]
struct RhsError {
    is_set: Cell<bool>,
    error: RefCell<Option<PharmsolError>>,
}

impl RhsError {
    fn is_set(&self) -> bool {
        self.is_set.get()
    }

    fn set(&self, error: PharmsolError) {
        self.is_set.set(true);
        *self.error.borrow_mut() = Some(error);
    }

    /// Surface the parked error, if any, and clear it.
    fn take(&self) -> Result<(), PharmsolError> {
        if !self.is_set.replace(false) {
            return Ok(());
        }
        match self.error.borrow_mut().take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Buffers reused across a subject's occasions.
struct Scratch {
    /// Derivative refreshed at discontinuity boundaries.
    dy: V,
    /// Outputs at an observation.
    y: V,
    /// Active infusion rates at an observation.
    rates: V,
}

impl Scratch {
    fn new(model: &impl Model) -> Self {
        let ctx = NalgebraContext::new();
        Self {
            dy: V::zeros(model.nstates(), ctx),
            y: V::zeros(model.noutputs(), ctx),
            rates: V::zeros(model.ninputs(), ctx),
        }
    }
}

impl<M: OdeModel> Cache for ODE<M> {
    fn with_cache_capacity(mut self, size: usize) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

impl<M: OdeModel> Equation for ODE<M> {
    type P = SubjectPredictions;

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata()
    }

    fn nstates(&self) -> usize {
        self.model.nstates()
    }

    fn nouteqs(&self) -> usize {
        self.model.noutputs()
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.predictions(subject, parameters)
            .map(Arc::unwrap_or_clone)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        self.validate_error_models(error_models)?;
        self.predictions(subject, parameters)?
            .log_likelihood(error_models)
    }
}

#[cfg(test)]
mod tests;
