use super::{ExplicitStepBudgetConfig, MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW};
use crate::{Covariates, Infusion, PharmsolError};
use diffsol::{
    ConstantOp, LinearOp, MatrixCommon, NalgebraContext, NalgebraMat, NonLinearOp,
    NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op, UnitCallable, Vector,
};
use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    marker::PhantomData,
};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

/// Evaluate one model RHS using the absolute model time and the active route
/// rates. Passing `active_rateiv` lets a directional derivative reuse exactly
/// the same rates for its base and perturbed states.
#[allow(clippy::too_many_arguments)]
fn evaluate_rhs<F>(
    func: &F,
    p_as_v: &V,
    covariates: &Covariates,
    zero_bolus: &V,
    integration_schedule: &IntegrationSchedule,
    time_origin: &Cell<f64>,
    rateiv_buffer: &RefCell<V>,
    x: &V,
    t: f64,
    active_rateiv: Option<&V>,
    y: &mut V,
) where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    let absolute_time = time_origin.get() + t;
    match active_rateiv {
        Some(rateiv) => {
            (func)(x, p_as_v, absolute_time, y, zero_bolus, rateiv, covariates);
        }
        None => {
            let mut rateiv = rateiv_buffer.borrow_mut();
            integration_schedule.fill_rate_vector(absolute_time, &mut rateiv);
            (func)(x, p_as_v, absolute_time, y, zero_bolus, &rateiv, covariates);
        }
    }
}

#[derive(Debug, Clone)]
struct InfusionTrack {
    input: usize,
    event_times: Vec<f64>,
    cumulative_rates: Vec<f64>,
}

impl InfusionTrack {
    fn new(input: usize, mut events: Vec<(f64, f64)>) -> Self {
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut event_times = Vec::with_capacity(events.len());
        let mut cumulative_rates = Vec::with_capacity(events.len());
        let mut current_rate = 0.0;

        for (time, delta) in events {
            current_rate += delta;
            event_times.push(time);
            cumulative_rates.push(current_rate);
        }

        Self {
            input,
            event_times,
            cumulative_rates,
        }
    }

    fn rate_at(&self, time: f64, left_continuity_time: Option<f64>) -> f64 {
        if let Some(left_continuity_time) = left_continuity_time {
            if left_continuity_time == time {
                return self.rate_at_left(time);
            }
        }

        self.rate_at_right(time)
    }

    fn rate_at_left(&self, time: f64) -> f64 {
        if self.event_times.is_empty() {
            return 0.0;
        }

        let event_search = self
            .event_times
            .binary_search_by(|probe| probe.partial_cmp(&time).unwrap_or(Ordering::Less));

        match event_search {
            Ok(mut idx) => {
                while idx > 0 && self.event_times[idx - 1] == time {
                    idx -= 1;
                }

                if idx == 0 {
                    0.0
                } else {
                    self.cumulative_rates[idx - 1]
                }
            }
            Err(0) => 0.0,
            Err(idx) => self.cumulative_rates[idx - 1],
        }
    }

    fn rate_at_right(&self, time: f64) -> f64 {
        if self.event_times.is_empty() {
            return 0.0;
        }

        match self
            .event_times
            .binary_search_by(|probe| probe.partial_cmp(&time).unwrap_or(Ordering::Less))
        {
            Ok(mut idx) => {
                while idx + 1 < self.event_times.len()
                    && self.event_times[idx + 1] == self.event_times[idx]
                {
                    idx += 1;
                }
                self.cumulative_rates[idx]
            }
            Err(0) => 0.0,
            Err(idx) => self.cumulative_rates[idx - 1],
        }
    }
}

/// Owns infusion tracks and the exact integration/discontinuity boundaries
/// required by one ODE solver session.
#[derive(Debug, Clone, Default)]
struct IntegrationSchedule {
    tracks: Vec<InfusionTrack>,
    /// All exact stops needed to avoid crossing an infusion or covariate knot.
    integration_boundary_times: Vec<f64>,
    /// The subset whose RHS value changes and therefore needs a restart.
    discontinuity_times: Vec<f64>,
    left_continuity_time: Cell<Option<f64>>,
}

impl IntegrationSchedule {
    fn new<'a, I>(
        ndrugs: usize,
        infusions: I,
        covariate_breakpoints: &[f64],
        covariate_discontinuities: &[f64],
    ) -> Result<Self, PharmsolError>
    where
        I: IntoIterator<Item = &'a Infusion>,
    {
        let mut integration_boundary_times = covariate_breakpoints.to_vec();
        let mut discontinuity_times = covariate_discontinuities.to_vec();
        if ndrugs == 0 {
            integration_boundary_times.sort_by(f64::total_cmp);
            integration_boundary_times.dedup();
            discontinuity_times.sort_by(f64::total_cmp);
            discontinuity_times.dedup();
            return Ok(Self {
                tracks: Vec::new(),
                integration_boundary_times,
                discontinuity_times,
                left_continuity_time: Cell::new(None),
            });
        }

        let mut per_input: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ndrugs];
        let mut saw_infusion = false;
        for infusion in infusions {
            saw_infusion = true;
            if infusion.duration() <= 0.0 {
                continue;
            }

            let input = infusion
                .input_index()
                .ok_or_else(|| PharmsolError::unknown_input_label(infusion.input(), &[]))?;
            if input >= ndrugs {
                return Err(PharmsolError::InputOutOfRange { input, ndrugs });
            }

            let rate = infusion.amount() / infusion.duration();
            let end = infusion.time() + infusion.duration();

            per_input[input].push((infusion.time(), rate));
            per_input[input].push((end, -rate));
            integration_boundary_times.push(infusion.time());
            integration_boundary_times.push(end);
            discontinuity_times.push(infusion.time());
            discontinuity_times.push(end);
        }

        integration_boundary_times.sort_by(f64::total_cmp);
        integration_boundary_times.dedup();
        discontinuity_times.sort_by(f64::total_cmp);
        discontinuity_times.dedup();

        if !saw_infusion {
            return Ok(Self {
                tracks: Vec::new(),
                integration_boundary_times,
                discontinuity_times,
                left_continuity_time: Cell::new(None),
            });
        }

        let tracks = per_input
            .into_iter()
            .enumerate()
            .filter_map(|(input, events)| {
                if events.is_empty() {
                    None
                } else {
                    Some(InfusionTrack::new(input, events))
                }
            })
            .collect();

        Ok(Self {
            tracks,
            integration_boundary_times,
            discontinuity_times,
            left_continuity_time: Cell::new(None),
        })
    }

    fn set_left_continuity_time(&self, time: Option<f64>) {
        self.left_continuity_time.set(time);
    }

    fn integration_boundary_times(&self) -> &[f64] {
        &self.integration_boundary_times
    }

    fn is_discontinuity_time(&self, time: f64) -> bool {
        self.discontinuity_times
            .binary_search_by(|probe| probe.total_cmp(&time))
            .is_ok()
    }

    /// Absolute infusion input in an interval that the backend could not
    /// integrate. The event loop uses this to produce a dose-specific failure
    /// rather than silently advancing across the unresolved boundary.
    fn infusion_amount_between(&self, from: f64, to: f64) -> f64 {
        if to <= from {
            return 0.0;
        }
        let duration = to - from;
        self.tracks
            .iter()
            .map(|track| track.rate_at_left(to).abs() * duration)
            .sum()
    }

    fn fill_rate_vector(&self, time: f64, rateiv: &mut V) {
        let left_continuity_time = self.left_continuity_time.get();
        rateiv.fill(0.0);
        for track in &self.tracks {
            let rate = track.rate_at(time, left_continuity_time);
            if rate != 0.0 {
                rateiv[track.input] = rate;
            }
        }
    }
}

/// Per-PMProblem/occasion accepted-work guard for TSIT45.
///
/// The window and session limits count accepted solver steps only. The window
/// is a dimensionless progress check: a productive window starts a new window
/// at its last accepted absolute time, while a stagnant window fails closed.
/// A new exact segment resets the window; coordinate rebases and residual
/// integrations reset neither the window nor the session total.
#[derive(Debug, Clone)]
struct ExplicitStepGuard {
    config: Cell<Option<ExplicitStepBudgetConfig>>,
    accepted_steps_in_window: Cell<usize>,
    accepted_steps_in_session: Cell<usize>,
    segment_start_time: Cell<f64>,
    segment_target_time: Cell<f64>,
    progress_window_start_time: Cell<f64>,
    last_accepted_time: Cell<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExplicitStepBudgetScope {
    InsufficientSegmentProgress,
    SessionTotal,
}

impl Default for ExplicitStepGuard {
    fn default() -> Self {
        Self {
            config: Cell::new(None),
            accepted_steps_in_window: Cell::new(0),
            accepted_steps_in_session: Cell::new(0),
            segment_start_time: Cell::new(f64::NAN),
            segment_target_time: Cell::new(f64::NAN),
            progress_window_start_time: Cell::new(f64::NAN),
            last_accepted_time: Cell::new(f64::NAN),
        }
    }
}

impl ExplicitStepGuard {
    fn configure(&self, config: Option<ExplicitStepBudgetConfig>) {
        self.config.set(config);
        self.accepted_steps_in_window.set(0);
        self.accepted_steps_in_session.set(0);
        self.segment_start_time.set(f64::NAN);
        self.segment_target_time.set(f64::NAN);
        self.progress_window_start_time.set(f64::NAN);
        self.last_accepted_time.set(f64::NAN);
    }

    fn begin_segment(&self, start_time: f64, target_time: f64) {
        if self.config.get().is_none() {
            return;
        }
        self.accepted_steps_in_window.set(0);
        self.segment_start_time.set(start_time);
        self.segment_target_time.set(target_time);
        self.progress_window_start_time.set(start_time);
        self.last_accepted_time.set(start_time);
    }

    fn record_accepted_step(&self, absolute_time: f64) {
        if self.config.get().is_some() {
            self.accepted_steps_in_window
                .set(self.accepted_steps_in_window.get().saturating_add(1));
            self.accepted_steps_in_session
                .set(self.accepted_steps_in_session.get().saturating_add(1));
            self.last_accepted_time.set(absolute_time);
        }
    }

    fn progress_stats(&self) -> (f64, f64, f64, f64) {
        let numeric_gap = self.segment_target_time.get() - self.segment_start_time.get();
        let actual_progress = self.last_accepted_time.get() - self.progress_window_start_time.get();
        let required_progress = numeric_gap * MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW;
        let progress_fraction = actual_progress / numeric_gap;
        (
            numeric_gap,
            actual_progress,
            required_progress,
            progress_fraction,
        )
    }

    fn window_is_productive(&self) -> bool {
        let (numeric_gap, actual_progress, _, progress_fraction) = self.progress_stats();
        numeric_gap.is_finite()
            && numeric_gap > 0.0
            && actual_progress.is_finite()
            && actual_progress >= 0.0
            && progress_fraction.is_finite()
            && progress_fraction >= MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW
    }

    fn check_due(&self) -> bool {
        let Some(config) = self.config.get() else {
            return false;
        };
        self.accepted_steps_in_session.get() >= config.max_accepted_steps_per_session
            || self.accepted_steps_in_window.get() >= config.max_accepted_steps_per_segment
    }

    /// Check the guard immediately before requesting another solver step.
    /// Productive-window state is reset only here, never while rebasing a
    /// residual or changing the solver coordinate origin.
    fn check_at_next_step(&self) -> Option<PharmsolError> {
        let config = self.config.get()?;
        if self.accepted_steps_in_session.get() >= config.max_accepted_steps_per_session {
            return Some(self.error_for(config, ExplicitStepBudgetScope::SessionTotal));
        }
        if self.accepted_steps_in_window.get() >= config.max_accepted_steps_per_segment {
            if self.window_is_productive() {
                self.accepted_steps_in_window.set(0);
                self.progress_window_start_time
                    .set(self.last_accepted_time.get());
                return None;
            }
            return Some(
                self.error_for(config, ExplicitStepBudgetScope::InsufficientSegmentProgress),
            );
        }
        None
    }

    fn error_for(
        &self,
        config: ExplicitStepBudgetConfig,
        scope: ExplicitStepBudgetScope,
    ) -> PharmsolError {
        let scope_name = match scope {
            ExplicitStepBudgetScope::InsufficientSegmentProgress => "insufficient segment progress",
            ExplicitStepBudgetScope::SessionTotal => "session total",
        };
        let segment_start_time = self.segment_start_time.get();
        let segment_target_time = self.segment_target_time.get();
        let (numeric_gap, actual_progress, required_progress, progress_fraction) =
            self.progress_stats();
        PharmsolError::OtherError(format!(
            "TSIT45 accepted-step budget exhausted (budget scope = {scope_name}) for exact smooth segment; segment start = {segment_start_time:.16e}; target = {segment_target_time:.16e}; numeric gap (target - start) = {numeric_gap:.16e}; progress window start = {:.16e}; actual progress = {actual_progress:.16e}; required progress = {required_progress:.16e}; progress fraction = {progress_fraction:.16e}; minimum progress fraction = {MIN_TSIT45_PROGRESS_FRACTION_PER_WINDOW:.16e}; last accepted absolute time = {:.16e}; window count/limit = {}/{}; cumulative count/limit = {}/{}; no incomplete state was returned. This is a dimensionless computational work budget, independent of model time and tolerance heuristics. Inspect model stiffness and state/time/unit scaling, or explicitly choose an implicit solver (BDF, TRBDF2, or ESDIRK34).",
            self.progress_window_start_time.get(),
            self.last_accepted_time.get(),
            self.accepted_steps_in_window.get(),
            config.max_accepted_steps_per_segment,
            self.accepted_steps_in_session.get(),
            config.max_accepted_steps_per_session,
        ))
    }
}

pub struct PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    nstates: usize,
    nparams: usize,
    integration_schedule: &'a IntegrationSchedule,
    time_origin: &'a Cell<f64>,
    covariates: &'a Covariates,
    p_as_v: &'a V,
    func: &'a F,
    jacobian: Option<&'a RhsJacobianFn<'a>>,
    jacobian_buffer: &'a RefCell<Vec<f64>>,
    rateiv_buffer: &'a RefCell<V>,
    jvp_x_buffer: &'a RefCell<V>,
    jvp_base_buffer: &'a RefCell<V>,
    jvp_perturbed_buffer: &'a RefCell<V>,
    zero_bolus: &'a V,
}

impl<F> Op for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmMass {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmMass {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmInit<'a> {
    nstates: usize,
    nout: usize,
    nparams: usize,
    init: &'a V,
}

impl Op for PmInit<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

impl ConstantOp for PmInit<'_> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(self.init);
    }
}

pub struct PmRoot {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmRoot {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmOut {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmOut {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

impl<F> NonLinearOp for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        evaluate_rhs(
            self.func,
            self.p_as_v,
            self.covariates,
            self.zero_bolus,
            self.integration_schedule,
            self.time_origin,
            self.rateiv_buffer,
            x,
            t,
            None,
            y,
        );
    }
}

impl<F> NonLinearOpJacobian for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        if let Some(jacobian) = self.jacobian {
            self.exact_jac_mul_inplace(jacobian, x, t, v, y);
            return;
        }

        // Forward directional difference:
        //
        //     J(x, t)v ~= (f(x + h v, t) - f(x, t)) / h
        //
        // with h = sqrt(eps) * max(1, ||x||_inf) / ||v||_inf.  The
        // normalized construction below avoids forming an overflowing h
        // when v is very small while making the state perturbation explicit.
        let mut x_scale: f64 = 1.0;
        let mut v_scale: f64 = 0.0;
        let mut v_is_zero = true;
        let mut finite_inputs = true;
        for index in 0..self.nstates {
            let xi = x.get_index(index);
            let vi = v.get_index(index);
            v_is_zero &= vi == 0.0;
            if xi.is_finite() {
                x_scale = x_scale.max(xi.abs());
            } else {
                finite_inputs = false;
            }
            if vi.is_finite() {
                v_scale = v_scale.max(vi.abs());
            } else {
                finite_inputs = false;
            }
        }

        // NonLinearOpJacobian has no error return. Invalid numeric inputs must
        // remain visibly non-finite rather than becoming a silent zero JVP.
        if !finite_inputs {
            y.fill(f64::NAN);
            return;
        }

        // The finite zero direction is exact and must not call a user callback.
        if v_is_zero {
            y.fill(0.0);
            return;
        }

        let delta = f64::EPSILON.sqrt() * x_scale;
        if !v_scale.is_finite() || v_scale == 0.0 || !delta.is_finite() || delta == 0.0 {
            y.fill(f64::NAN);
            return;
        }

        let mut signed_delta = delta;
        let mut x_perturbed = self.jvp_x_buffer.borrow_mut();
        let mut perturbation_is_finite = true;
        for index in 0..self.nstates {
            let candidate = x.get_index(index) + delta * (v.get_index(index) / v_scale);
            if candidate.is_finite() {
                x_perturbed.set_index(index, candidate);
            } else {
                perturbation_is_finite = false;
                break;
            }
        }
        if !perturbation_is_finite {
            // Use the opposite one-sided perturbation if the first direction
            // would overflow a state component near the edge of f64.
            signed_delta = -delta;
            perturbation_is_finite = true;
            for index in 0..self.nstates {
                let candidate = x.get_index(index) - delta * (v.get_index(index) / v_scale);
                if candidate.is_finite() {
                    x_perturbed.set_index(index, candidate);
                } else {
                    perturbation_is_finite = false;
                    break;
                }
            }
        }
        if !perturbation_is_finite {
            y.fill(f64::NAN);
            return;
        }

        // Build the active rate vector once. Both evaluations below receive
        // the same borrow, including the schedule's left/right continuity
        // choice at this absolute time.
        let mut rateiv = self.rateiv_buffer.borrow_mut();
        let absolute_time = self.time_origin.get() + t;
        self.integration_schedule
            .fill_rate_vector(absolute_time, &mut rateiv);
        let mut base = self.jvp_base_buffer.borrow_mut();
        let mut perturbed = self.jvp_perturbed_buffer.borrow_mut();
        evaluate_rhs(
            self.func,
            self.p_as_v,
            self.covariates,
            self.zero_bolus,
            self.integration_schedule,
            self.time_origin,
            self.rateiv_buffer,
            x,
            t,
            Some(&rateiv),
            &mut base,
        );
        evaluate_rhs(
            self.func,
            self.p_as_v,
            self.covariates,
            self.zero_bolus,
            self.integration_schedule,
            self.time_origin,
            self.rateiv_buffer,
            &x_perturbed,
            t,
            Some(&rateiv),
            &mut perturbed,
        );

        // Scale the output by v_scale / signed_delta rather than dividing by
        // h directly. An unusable scale must remain visibly non-finite rather
        // than being sanitized after the callback evaluations.
        let output_scale = v_scale / signed_delta;
        if !output_scale.is_finite() || output_scale == 0.0 {
            y.fill(f64::NAN);
            return;
        }

        // Preserve NaN/Inf from the model's RHS difference. No callback retry
        // or exceptional-value sanitization belongs in this no-error API.
        for index in 0..self.nstates {
            let difference = perturbed.get_index(index) - base.get_index(index);
            y.set_index(index, difference * output_scale);
        }
    }
}

impl<F> PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    /// `J(x, t) v` from an exact Jacobian, avoiding both the two RHS
    /// evaluations and the `sqrt(eps)` accuracy of the difference quotient.
    fn exact_jac_mul_inplace(
        &self,
        jacobian: &RhsJacobianFn<'_>,
        x: &V,
        t: T,
        v: &V,
        y: &mut V,
    ) {
        let absolute_time = self.time_origin.get() + t;
        let mut rateiv = self.rateiv_buffer.borrow_mut();
        self.integration_schedule
            .fill_rate_vector(absolute_time, &mut rateiv);

        let mut entries = self.jacobian_buffer.borrow_mut();
        entries.fill(0.0);
        (jacobian)(
            x,
            self.p_as_v,
            absolute_time,
            &rateiv,
            self.covariates,
            &mut entries,
        );

        for row in 0..self.nstates {
            let mut sum = 0.0;
            for column in 0..self.nstates {
                sum += entries[row * self.nstates + column] * v.get_index(column);
            }
            y.set_index(row, sum);
        }
    }
}

impl LinearOp for PmMass {
    fn gemv_inplace(&self, _x: &Self::V, _t: Self::T, _beta: Self::T, _y: &mut Self::V) {}
}

impl NonLinearOp for PmRoot {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

impl NonLinearOp for PmOut {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

/// Fills a row-major `n x n` state Jacobian for the model RHS.
///
/// Supplied by backends that can derive one symbolically; solvers otherwise
/// fall back to a finite-difference directional derivative.
pub(crate) type RhsJacobianFn<'a> = dyn Fn(&V, &V, T, &V, &Covariates, &mut [f64]) + 'a;

// Completely revised PMProblem to fix lifetime issues and improve performance
pub(crate) struct PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    func: F,
    // Borrowed rather than owned: an owned trait object would make drop-check
    // require `'a` to strictly outlive every borrow of the problem.
    jacobian: Option<&'a RhsJacobianFn<'a>>,
    jacobian_buffer: RefCell<Vec<f64>>,
    nstates: usize,
    nparams: usize,
    init: V,
    p_as_v: V,
    zero_bolus: V,
    covariates: Covariates,
    integration_schedule: IntegrationSchedule,
    time_origin: Cell<f64>,
    explicit_step_guard: ExplicitStepGuard,
    rateiv_buffer: RefCell<V>,
    jvp_x_buffer: RefCell<V>,
    jvp_base_buffer: RefCell<V>,
    jvp_perturbed_buffer: RefCell<V>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, F> PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    /// Convert diffsol's local independent variable to model/data time.
    pub(crate) fn absolute_time(&self, solver_time: f64) -> f64 {
        self.time_origin.get() + solver_time
    }

    /// Return diffsol's current absolute-time origin.
    pub(crate) fn time_origin(&self) -> f64 {
        self.time_origin.get()
    }

    /// Start a new local coordinate at an accepted absolute-time state.
    pub(crate) fn rebase_time_origin(&self, absolute_time: f64) {
        self.time_origin.set(absolute_time);
    }

    /// Configure the private accepted-work guard once for this solver
    /// session. `None` keeps both limits disabled for implicit sessions.
    pub(crate) fn configure_explicit_step_guard(&self, config: Option<ExplicitStepBudgetConfig>) {
        self.explicit_step_guard.configure(config);
    }

    /// Start a new exact event-loop integration segment. A coordinate rebase
    /// and its residual integration deliberately do not call this method.
    pub(crate) fn begin_explicit_step_segment(&self, start_time: f64, target_time: f64) {
        self.explicit_step_guard
            .begin_segment(start_time, target_time);
    }

    pub(crate) fn record_explicit_accepted_step(&self, absolute_time: f64) {
        self.explicit_step_guard.record_accepted_step(absolute_time);
    }

    pub(crate) fn explicit_step_budget_check_due(&self) -> bool {
        self.explicit_step_guard.check_due()
    }

    pub(crate) fn check_explicit_step_budget(&self) -> Option<PharmsolError> {
        self.explicit_step_guard.check_at_next_step()
    }

    pub(crate) fn set_left_continuity_time(&self, time: Option<f64>) {
        self.covariates.set_left_continuity_time(time);
        self.integration_schedule.set_left_continuity_time(time);
    }

    pub(crate) fn integration_boundary_times(&self) -> &[f64] {
        self.integration_schedule.integration_boundary_times()
    }

    pub(crate) fn is_discontinuity_time(&self, time: f64) -> bool {
        self.integration_schedule.is_discontinuity_time(time)
    }

    pub(crate) fn infusion_amount_between(&self, from: f64, to: f64) -> f64 {
        self.integration_schedule.infusion_amount_between(from, to)
    }

    /// Evaluate the full RHS (including the currently scheduled infusion
    /// rates) at local solver time `t` into `dx`.
    ///
    /// Used at discontinuity boundaries to refresh the solver's stored
    /// derivative against the post-boundary (right-continuous) RHS, so a solver
    /// restart predicts with the new dynamics instead of the pre-boundary ones.
    pub(crate) fn refresh_state_derivative(&self, t: f64, x: &V, dx: &mut V) {
        evaluate_rhs(
            &self.func,
            &self.p_as_v,
            &self.covariates,
            &self.zero_bolus,
            &self.integration_schedule,
            &self.time_origin,
            &self.rateiv_buffer,
            x,
            t,
            None,
            dx,
        );
    }

    /// Creates a new PMProblem with a pre-converted parameter vector and an
    /// absolute-time origin for diffsol's local clock.
    ///
    /// This avoids an allocation when the caller already has a V representation
    /// while keeping all model callbacks in absolute time.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params_v<'b, I>(
        func: F,
        nstates: usize,
        ndrugs: usize,
        p_as_v: V,
        covariates: &Covariates,
        infusions: I,
        init: V,
        time_origin: f64,
    ) -> Result<Self, PharmsolError>
    where
        I: IntoIterator<Item = &'b Infusion>,
    {
        if !time_origin.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "invalid ODE time origin {time_origin:?}: the resolved event schedule must be finite"
            )));
        }

        let nparams = p_as_v.len();
        let rateiv_buffer = RefCell::new(V::zeros(ndrugs, NalgebraContext::new()));
        let covariates = covariates.clone();
        let covariate_breakpoints = covariates.ode_breakpoint_times()?;
        let covariate_discontinuities = covariates.ode_discontinuity_times()?;
        let integration_schedule = IntegrationSchedule::new(
            ndrugs,
            infusions,
            &covariate_breakpoints,
            &covariate_discontinuities,
        )?;
        // Pre-allocate zero bolus and directional-derivative scratch vectors.
        let zero_bolus = V::zeros(ndrugs, NalgebraContext::new());
        let jvp_x_buffer = RefCell::new(V::zeros(nstates, NalgebraContext::new()));
        let jvp_base_buffer = RefCell::new(V::zeros(nstates, NalgebraContext::new()));
        let jvp_perturbed_buffer = RefCell::new(V::zeros(nstates, NalgebraContext::new()));

        Ok(Self {
            func,
            jacobian: None,
            jacobian_buffer: RefCell::new(vec![0.0; nstates * nstates]),
            nstates,
            nparams,
            init,
            p_as_v,
            zero_bolus,
            covariates,
            integration_schedule,
            time_origin: Cell::new(time_origin),
            explicit_step_guard: ExplicitStepGuard::default(),
            rateiv_buffer,
            jvp_x_buffer,
            jvp_base_buffer,
            jvp_perturbed_buffer,
            _lifetime: PhantomData,
        })
    }

    /// Supply an exact state Jacobian, replacing the finite-difference
    /// directional derivative the solver would otherwise use.
    pub(crate) fn with_jacobian(mut self, jacobian: &'a RhsJacobianFn<'a>) -> Self {
        self.jacobian = Some(jacobian);
        self
    }
}

impl<'a, F> Op for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

// Implement OdeEquationsRef for PMProblem for any lifetime 'b
impl<'a, 'b, F> OdeEquationsRef<'b> for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    type Rhs = PmRhs<'b, F>;
    type Mass = PmMass;
    type Init = PmInit<'b>;
    type Root = PmRoot;
    type Out = PmOut;
    type Reset = UnitCallable<M>;
}

// Implement OdeEquations with correct lifetime handling
impl<'a, F> OdeEquations for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    fn rhs(&self) -> PmRhs<'_, F> {
        PmRhs {
            nstates: self.nstates,
            nparams: self.nparams,
            integration_schedule: &self.integration_schedule,
            time_origin: &self.time_origin,
            covariates: &self.covariates,
            p_as_v: &self.p_as_v,
            func: &self.func,
            jacobian: self.jacobian,
            jacobian_buffer: &self.jacobian_buffer,
            rateiv_buffer: &self.rateiv_buffer,
            jvp_x_buffer: &self.jvp_x_buffer,
            jvp_base_buffer: &self.jvp_base_buffer,
            jvp_perturbed_buffer: &self.jvp_perturbed_buffer,
            zero_bolus: &self.zero_bolus,
        }
    }

    fn mass(&self) -> Option<PmMass> {
        None
    }

    fn init(&self) -> PmInit<'_> {
        PmInit {
            nstates: self.nstates,
            nout: self.nstates,
            nparams: self.nparams,
            init: &self.init,
        }
    }

    fn get_params(&self, p: &mut V) {
        if p.len() == self.p_as_v.len() {
            p.copy_from(&self.p_as_v);
        } else {
            *p = self.p_as_v.clone();
        }
    }

    fn root(&self) -> Option<PmRoot> {
        None
    }

    fn out(&self) -> Option<PmOut> {
        None
    }

    fn reset(&self) -> Option<UnitCallable<M>> {
        None
    }

    fn set_params(&mut self, p: &V) {
        if self.p_as_v.len() == p.len() {
            self.p_as_v.copy_from(p);
        } else {
            self.p_as_v = p.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Covariate;
    use diffsol::{NonLinearOpJacobian, OdeEquations, Vector};
    use std::cell::Cell;

    #[test]
    fn pm_rhs_jvp_zero_direction_skips_callback() {
        let callback_count = Cell::new(0);
        let covariates = Covariates::default();
        let infusions = Vec::<Infusion>::new();
        let problem = PMProblem::with_params_v(
            |x, _p, _t, dx, _bolus, _rateiv, _cov| {
                callback_count.set(callback_count.get() + 1);
                dx[0] = x[0];
            },
            1,
            0,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(1, NalgebraContext::new()),
            0.0,
        )
        .unwrap_or_else(|error| panic!("zero-direction RHS problem should build: {error}"));
        let rhs = problem.rhs();
        let x = V::from_vec(vec![3.0], NalgebraContext::new());
        let v = V::zeros(1, NalgebraContext::new());
        let mut jvp = V::from_vec(vec![f64::NAN], NalgebraContext::new());

        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);

        assert_eq!(jvp.get_index(0), 0.0);
        assert_eq!(callback_count.get(), 0);
    }

    #[test]
    fn pm_rhs_jvp_preserves_nonfinite_inputs() {
        let callback_count = Cell::new(0);
        let covariates = Covariates::default();
        let infusions = Vec::<Infusion>::new();
        let problem = PMProblem::with_params_v(
            |_x, _p, _t, dx, _bolus, _rateiv, _cov| {
                callback_count.set(callback_count.get() + 1);
                dx[0] = 1.0;
            },
            1,
            0,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(1, NalgebraContext::new()),
            0.0,
        )
        .unwrap_or_else(|error| panic!("non-finite-input RHS problem should build: {error}"));
        let rhs = problem.rhs();
        let mut jvp = V::zeros(1, NalgebraContext::new());

        let x = V::from_vec(vec![f64::NAN], NalgebraContext::new());
        let v = V::from_vec(vec![1.0], NalgebraContext::new());
        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);
        assert!(
            jvp.get_index(0).is_nan(),
            "NaN state must remain visible: {jvp:?}"
        );

        let x = V::from_vec(vec![1.0], NalgebraContext::new());
        let v = V::from_vec(vec![f64::INFINITY], NalgebraContext::new());
        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);
        assert!(
            jvp.get_index(0).is_nan(),
            "infinite direction must remain visible: {jvp:?}"
        );
        assert_eq!(callback_count.get(), 0);
    }

    #[test]
    fn pm_rhs_jvp_preserves_nonfinite_rhs_differences_without_retry() {
        let callback_count = Cell::new(0);
        let covariates = Covariates::default();
        let infusions = Vec::<Infusion>::new();
        let problem = PMProblem::with_params_v(
            |_x, _p, _t, dx, _bolus, _rateiv, _cov| {
                callback_count.set(callback_count.get() + 1);
                dx[0] = f64::NAN;
                dx[1] = if _x[1] > 1.0 { f64::INFINITY } else { 0.0 };
            },
            2,
            0,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(2, NalgebraContext::new()),
            0.0,
        )
        .unwrap_or_else(|error| panic!("non-finite-RHS problem should build: {error}"));
        let rhs = problem.rhs();
        let x = V::from_vec(vec![1.0, 1.0], NalgebraContext::new());
        let v = V::from_vec(vec![1.0, 1.0], NalgebraContext::new());
        let mut jvp = V::zeros(2, NalgebraContext::new());

        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);

        assert!(
            jvp.get_index(0).is_nan(),
            "NaN RHS difference was sanitized: {jvp:?}"
        );
        assert!(
            jvp.get_index(1).is_infinite(),
            "infinite RHS difference was sanitized: {jvp:?}"
        );
        assert_eq!(callback_count.get(), 2);
    }

    #[test]
    fn pm_rhs_jvp_matches_nonlinear_square_rhs() {
        let covariates = Covariates::default();
        let infusions = Vec::<Infusion>::new();
        let problem = PMProblem::with_params_v(
            |x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = x[0] * x[0],
            1,
            0,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(1, NalgebraContext::new()),
            0.0,
        )
        .unwrap_or_else(|error| panic!("square RHS problem should build: {error}"));
        let rhs = problem.rhs();
        let x = V::from_vec(vec![3.0], NalgebraContext::new());
        let v = V::from_vec(vec![2.0], NalgebraContext::new());
        let mut jvp = V::zeros(1, NalgebraContext::new());

        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);

        assert!((jvp.get_index(0) - 12.0).abs() < 1e-6, "Jv = {jvp:?}");
    }

    #[test]
    fn pm_rhs_jvp_uses_absolute_time_and_active_infusion_rate() {
        let covariates = Covariates::default();
        let infusions = [Infusion::new(10.0, 3.0, 0, 1.0, 0)];
        let problem = PMProblem::with_params_v(
            |x, _p, t, dx, _bolus, rateiv, _cov| {
                dx[0] = (rateiv[0] + t) * x[0] * x[0];
            },
            1,
            1,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(1, NalgebraContext::new()),
            10.0,
        )
        .unwrap_or_else(|error| panic!("rate-dependent RHS problem should build: {error}"));
        let rhs = problem.rhs();
        let x = V::from_vec(vec![2.0], NalgebraContext::new());
        let v = V::from_vec(vec![0.5], NalgebraContext::new());
        let mut jvp = V::zeros(1, NalgebraContext::new());

        rhs.jac_mul_inplace(&x, 0.25, &v, &mut jvp);
        assert!((jvp.get_index(0) - 26.5).abs() < 1e-6, "Jv = {jvp:?}");

        problem.set_left_continuity_time(Some(10.0));
        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);
        assert!((jvp.get_index(0) - 20.0).abs() < 1e-6, "left Jv = {jvp:?}");

        problem.set_left_continuity_time(None);
        rhs.jac_mul_inplace(&x, 0.0, &v, &mut jvp);
        assert!((jvp.get_index(0) - 26.0).abs() < 1e-6, "right Jv = {jvp:?}");
    }

    #[test]
    fn discontinuity_boundaries_merge_covariate_knots_and_infusions_exactly() {
        let mut covariates = Covariates::new();
        let mut covariate = Covariate::new("rate".into(), false);
        covariate.add_observation(0.0, 1.0);
        covariate.add_observation(1.0, 2.0);
        covariate.add_observation(2.0, 4.0);
        covariates.add_covariate("rate".into(), covariate);
        let infusions = [Infusion::new(0.0, 1.0, 0, 1.0, 0)];

        let problem = PMProblem::with_params_v(
            |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
            1,
            1,
            V::zeros(0, NalgebraContext::new()),
            &covariates,
            infusions.iter(),
            V::zeros(1, NalgebraContext::new()),
            0.0,
        )
        .expect("merged-boundary problem should build");

        assert_eq!(problem.integration_boundary_times(), &[0.0, 1.0, 2.0]);
        assert!(problem.is_discontinuity_time(0.0));
        assert!(problem.is_discontinuity_time(1.0));
        assert!(!problem.is_discontinuity_time(2.0));
    }

    #[test]
    fn solver_covariate_continuity_isolated_from_source_covariates() {
        let mut source = Covariates::new();
        let mut covariate = Covariate::new("rate".into(), true);
        covariate.add_observation(0.0, 1.0);
        covariate.add_observation(1.0, 2.0);
        source.add_covariate("rate".into(), covariate);

        let problem = PMProblem::with_params_v(
            |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
            1,
            0,
            V::zeros(0, NalgebraContext::new()),
            &source,
            std::iter::empty::<&Infusion>(),
            V::zeros(1, NalgebraContext::new()),
            0.0,
        )
        .expect("isolated-covariate problem should build");

        problem.set_left_continuity_time(Some(1.0));
        assert_eq!(
            source
                .get_covariate("rate")
                .expect("source covariate")
                .interpolate(1.0)
                .unwrap(),
            2.0
        );
        assert_eq!(
            problem
                .covariates
                .get_covariate("rate")
                .expect("solver covariate")
                .interpolate(1.0)
                .unwrap(),
            1.0
        );

        problem.set_left_continuity_time(None);
        assert_eq!(
            problem
                .covariates
                .get_covariate("rate")
                .expect("solver covariate")
                .interpolate(1.0)
                .unwrap(),
            2.0
        );
    }
}
