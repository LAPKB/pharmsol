//! Event-to-event integration on top of diffsol.
//!
//! [`advance_solver_to_event`] moves a solver from the current event to the
//! next one while respecting infusion and covariate discontinuities, keeping
//! diffsol's local clock representable, and enforcing the explicit-solver step
//! budget. Every ODE model, native or compiled, goes through it.

use super::closure::PMProblem;
use super::*;
use crate::simulator::equation::model::Step;

fn checked_local_time(
    time_origin: f64,
    absolute_time: f64,
    context: &str,
) -> Result<f64, PharmsolError> {
    let local_time = absolute_time - time_origin;
    let round_trip = time_origin + local_time;
    if !time_origin.is_finite()
        || !absolute_time.is_finite()
        || !local_time.is_finite()
        || !round_trip.is_finite()
        || round_trip != absolute_time
    {
        return Err(PharmsolError::OtherError(format!(
            "{context}: absolute time {absolute_time:?} cannot be represented relative to origin {time_origin:?} (local = {local_time:?}, round trip = {round_trip:?})"
        )));
    }
    Ok(local_time)
}

/// Validate a resolved ODE schedule and return its absolute-time origin.
///
/// Validation happens after route resolution, lag, bioavailability, and event
/// reordering so the solver never has to interpret an invalid schedule. The
/// solver starts at local `t = 0` at the first resolved event. Schedule
/// validation is absolute-time validation; a later time may need a coordinate
/// shift before it is passed to diffsol.
pub(crate) fn validate_resolved_ode_schedule(steps: &[Step]) -> Result<f64, PharmsolError> {
    let first_event_time = steps.first().map(Step::time).unwrap_or(0.0);
    if !first_event_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "invalid ODE event schedule: first resolved event time {first_event_time:?} is not finite"
        )));
    }
    let time_origin = first_event_time;

    let mut required_times = Vec::with_capacity(steps.len() * 2);
    let mut previous_event_time = None;

    for (index, step) in steps.iter().enumerate() {
        let time = step.time();
        if !time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "invalid ODE event schedule: resolved event {index} has non-finite time {time:?}"
            )));
        }
        if let Some(previous) = previous_event_time {
            if time < previous {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: resolved event times are not nondecreasing; event {index} at t = {time:.16e} follows t = {previous:.16e}"
                )));
            }
            let gap = time - previous;
            if !gap.is_finite() || (time > previous && gap <= 0.0) {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: gap from t = {previous:.16e} to t = {time:.16e} is not representable"
                )));
            }
        }
        previous_event_time = Some(time);
        required_times.push(time);

        match step {
            Step::Bolus { amount, .. } => {
                if !amount.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: bolus at t = {time:.16e} has non-finite amount {amount:?}"
                    )));
                }
            }
            Step::Infusion(infusion) => {
                let amount = infusion.amount;
                if !amount.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite amount {amount:?}"
                    )));
                }

                let duration = infusion.duration;
                if !duration.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite duration {duration:?}"
                    )));
                }
                if duration <= 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} must have positive duration, got {duration:.16e}"
                    )));
                }
                let rate = amount / duration;
                if !rate.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} has non-finite rate from amount {amount:.16e} and duration {duration:.16e}"
                    )));
                }
                if amount != 0.0 && rate == 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion at t = {time:.16e} with amount {amount:.16e} and duration {duration:.16e} has a zero rate, which would silently lose nonzero infusion material"
                    )));
                }
                let endpoint = time + duration;
                if !endpoint.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion endpoint {time:.16e} + {duration:.16e} is not finite"
                    )));
                }
                let endpoint_gap = endpoint - time;
                if !endpoint_gap.is_finite() || endpoint_gap <= 0.0 {
                    return Err(PharmsolError::OtherError(format!(
                        "invalid ODE event schedule: infusion endpoint {endpoint:.16e} is not representably after start t = {time:.16e}"
                    )));
                }
                required_times.push(endpoint);
            }
            Step::Observation { .. } => {}
        }
    }

    let mut previous_time: Option<f64> = None;
    required_times.sort_by(f64::total_cmp);
    for time in required_times {
        if let Some(previous_time) = previous_time.filter(|previous_time| time > *previous_time) {
            let gap = time - previous_time;
            if !gap.is_finite() || gap <= 0.0 {
                return Err(PharmsolError::OtherError(format!(
                    "invalid ODE event schedule: required positive gap from t = {previous_time:.16e} to t = {time:.16e} is not representable"
                )));
            }
        }
        previous_time = Some(time);
    }

    Ok(time_origin)
}

/// Restart the solver after a state or RHS discontinuity.
///
/// The multi-step history and the internal Jacobian were built for the
/// pre-discontinuity state/RHS, so the first step into the new segment must not reuse
/// them as-is:
/// - `set_state` forces diffsol to recompute its BDF coefficients and
///   reinitialize the internal Jacobian for the current state;
/// - the stored derivative is refreshed against the post-boundary
///   (right-continuous) RHS so a first-order restart predicts with the new
///   dynamics instead of the pre-boundary ones;
/// - `state_mut` marks the state as modified so the next step restarts the
///   multi-step method at first order.
///
/// Every [`ODE`] model, native or compiled, goes through this, so all of them
/// apply the same discontinuity semantics.
pub(crate) fn reinitialize_at_boundary<'a, F, S>(solver: &mut S, dy_scratch: &mut V)
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state = solver.state_clone();
    solver.set_state(state);

    let t = solver.state().t;
    {
        let y = solver.state().y;
        solver
            .problem()
            .eqn
            .refresh_state_derivative(t, y, dy_scratch);
    }
    solver.state_mut().dy.copy_from(dy_scratch);
}

/// Keep an exact-stop alignment step from poisoning the next segment. This
/// changes only the next-step proposal after the current segment succeeded;
/// the accepted state and time remain unchanged.
fn restore_timestep_after_successful_stop<'a, F, S>(solver: &mut S)
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let minimum_timestep = *solver.config().as_base_ref().minimum_timestep;
    let restart_timestep = MINIMUM_TIMESTEP_HEADROOM * minimum_timestep;
    let step = solver.state().h;
    if minimum_timestep > 0.0 && step.abs() < restart_timestep {
        *solver.state_mut().h = restart_timestep.copysign(step);
    }
}

/// Shift diffsol's independent variable without changing the accepted state
/// or either absolute endpoint. The selected origin and both local times must
/// round-trip exactly; this helper never approximates model time. It also
/// discards solver history/Jacobian assumptions and refreshes the RHS at the
/// accepted absolute state.
fn shift_solver_coordinate<'a, F, S>(
    solver: &mut S,
    new_origin: f64,
    target_time: f64,
    dy_scratch: &mut V,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state_time = solver.state().t;
    let old_origin = solver.problem().eqn.time_origin();
    let current_time = old_origin + state_time;
    if !state_time.is_finite()
        || !old_origin.is_finite()
        || !current_time.is_finite()
        || !target_time.is_finite()
    {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift requires finite times: origin = {old_origin:?}, local state = {state_time:?}, current = {current_time:?}, target = {target_time:?}"
        )));
    }
    if target_time <= current_time {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot shift to a non-positive residual interval from t = {current_time:.16e} to t = {target_time:.16e}"
        )));
    }

    let new_state_time = checked_local_time(
        new_origin,
        current_time,
        "ODE solver coordinate-shift current conversion",
    )?;
    let new_target_time = checked_local_time(
        new_origin,
        target_time,
        "ODE solver coordinate-shift target conversion",
    )?;
    let remaining = new_target_time - new_state_time;
    if !remaining.is_finite() || remaining <= 0.0 {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift cannot preserve the positive interval from absolute t = {current_time:.16e} to t = {target_time:.16e} with origin {new_origin:.16e} (local state = {new_state_time:?}, local target = {new_target_time:?})"
        )));
    }

    let current_step = solver.state().h.abs();
    if !current_step.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver coordinate shift cannot retain a non-finite accepted step at absolute t = {current_time:.16e}: h = {:?}",
            solver.state().h
        )));
    }
    let bounded_step = if current_step == 0.0 {
        remaining
    } else {
        current_step.min(remaining)
    };

    solver.problem().eqn.rebase_time_origin(new_origin);
    {
        let state = solver.state_mut();
        *state.t = new_state_time;
        *state.h = bounded_step;
    }
    let state = solver.state_clone();
    solver.set_state(state);

    let y = solver.state().y;
    solver
        .problem()
        .eqn
        .refresh_state_derivative(new_state_time, y, dy_scratch);
    solver.state_mut().dy.copy_from(dy_scratch);
    Ok(())
}

/// Pick a deterministic origin that represents the accepted absolute state
/// and the next absolute stop exactly. Zero is preferred because every finite
/// f64 absolute time round-trips relative to it; the accepted current time is
/// the deterministic fallback and is accepted only when both conversions and
/// the positive local gap are exact.
fn coordinate_shift_origin(current_time: f64, target_time: f64) -> Result<f64, PharmsolError> {
    if !current_time.is_finite() || !target_time.is_finite() || target_time <= current_time {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot choose a coordinate origin for current t = {current_time:?} and target t = {target_time:?}; expected finite target after current time"
        )));
    }

    for new_origin in [0.0, current_time] {
        let Ok(local_current) = checked_local_time(
            new_origin,
            current_time,
            "ODE solver coordinate-shift current conversion",
        ) else {
            continue;
        };
        let Ok(local_target) = checked_local_time(
            new_origin,
            target_time,
            "ODE solver coordinate-shift target conversion",
        ) else {
            continue;
        };
        let local_gap = local_target - local_current;
        if local_gap.is_finite() && local_gap > 0.0 {
            return Ok(new_origin);
        }
    }

    Err(PharmsolError::OtherError(format!(
        "ODE solver cannot preserve exact absolute coordinates for current t = {current_time:.16e} and target t = {target_time:.16e}; tried deterministic origins 0 and the current absolute time, but no finite positive local interval round-tripped exactly. No model-time approximation was attempted"
    )))
}

/// Ensure the accepted current state and an absolute stop can both be passed
/// to diffsol under the active origin. A coordinate shift is performed once
/// when the active origin loses either round trip, and native callback errors
/// are drained immediately after the refresh.
fn ensure_solver_stop_coordinates<'a, F, S, H>(
    solver: &mut S,
    absolute_stop_time: f64,
    dy_scratch: &mut V,
    after_step: &mut H,
) -> Result<f64, PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    let state_time = solver.state().t;
    let time_origin = solver.problem().eqn.time_origin();
    let current_time = time_origin + state_time;
    if !state_time.is_finite()
        || !time_origin.is_finite()
        || !current_time.is_finite()
        || !absolute_stop_time.is_finite()
    {
        return Err(PharmsolError::OtherError(format!(
            "ODE event coordinate normalization requires finite times: origin = {time_origin:?}, local state = {state_time:?}, current = {current_time:?}, target = {absolute_stop_time:?}"
        )));
    }
    if absolute_stop_time < current_time {
        return Err(stop_time_before_current_time(
            absolute_stop_time,
            current_time,
        ));
    }
    if absolute_stop_time == current_time {
        return checked_local_time(
            time_origin,
            absolute_stop_time,
            "ODE event equal-target conversion",
        );
    }

    let active_state_time = checked_local_time(
        time_origin,
        current_time,
        "ODE event current-state conversion",
    );
    let active_stop_time =
        checked_local_time(time_origin, absolute_stop_time, "ODE event stop conversion");
    if let (Ok(active_state_time), Ok(active_stop_time)) = (active_state_time, active_stop_time) {
        let local_gap = active_stop_time - active_state_time;
        if active_state_time == state_time && local_gap.is_finite() && local_gap > 0.0 {
            return Ok(active_stop_time);
        }
    }

    let new_origin = coordinate_shift_origin(current_time, absolute_stop_time)?;
    shift_solver_coordinate::<F, S>(solver, new_origin, absolute_stop_time, dy_scratch)?;
    after_step()?;

    let shifted_stop_time = checked_local_time(
        solver.problem().eqn.time_origin(),
        absolute_stop_time,
        "ODE event shifted stop conversion",
    )?;
    let shifted_state_time = solver.state().t;
    let shifted_gap = shifted_stop_time - shifted_state_time;
    if !shifted_gap.is_finite() || shifted_gap <= 0.0 {
        return Err(PharmsolError::OtherError(format!(
            "ODE event coordinate shift did not preserve a positive local interval: current absolute t = {current_time:.16e}, target = {absolute_stop_time:.16e}, local state = {shifted_state_time:?}, local target = {shifted_stop_time:?}"
        )));
    }
    Ok(shifted_stop_time)
}

/// Rebase a close accepted stop at its absolute current time. Unlike the
/// general coordinate normalization, this intentionally sets local state time
/// to zero and permits one residual integration attempt.
fn rebase_solver_time<'a, F, S>(
    solver: &mut S,
    absolute_time: f64,
    target_time: f64,
    dy_scratch: &mut V,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    shift_solver_coordinate::<F, S>(solver, absolute_time, target_time, dy_scratch)
}

fn ensure_no_material_infusion_is_skipped<'a, F, S>(
    solver: &S,
    start_time: f64,
    stop_time: f64,
    rtol: f64,
    atol: f64,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let skipped_infusion = solver
        .problem()
        .eqn
        .infusion_amount_between(start_time, stop_time);
    if !skipped_infusion.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot safely resolve the infusion boundary from t = {start_time:.16e} to t = {stop_time:.16e}: the scheduled infusion amount is non-finite. Check infusion amounts, durations, and time units"
        )));
    }
    let state_scale = solver
        .state()
        .y
        .as_slice()
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let material_tolerance = atol.abs() + rtol.abs() * state_scale;
    if skipped_infusion > material_tolerance {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver cannot safely resolve the infusion boundary from t = \
             {start_time:.16e} to t = {stop_time:.16e}: advancing without another solver \
             step would skip infusion amount {skipped_infusion:.6e}, above tolerance \
             {material_tolerance:.6e}. No dose was skipped. Check the infusion duration, \
             rate, and model time units"
        )));
    }
    Ok(())
}

fn stop_time_before_current_time(stop_time: f64, state_time: f64) -> PharmsolError {
    PharmsolError::from_solver_error(
        diffsol::error::DiffsolError::OdeSolverError(OdeSolverError::StopTimeBeforeCurrentTime {
            stop_time,
            state_time,
        }),
        stop_time,
    )
}

fn solver_error_at_absolute_time<'a, F, S>(
    solver: &S,
    error: diffsol::error::DiffsolError,
    absolute_target_time: f64,
) -> PharmsolError
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    use diffsol::error::DiffsolError;

    let error = match error {
        DiffsolError::OdeSolverError(error) => {
            let error = match error {
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time,
                    state_time,
                } => OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: solver.problem().eqn.absolute_time(stop_time),
                    state_time: solver.problem().eqn.absolute_time(state_time),
                },
                OdeSolverError::TooManyNonlinearSolverFailures { time, num_failures } => {
                    OdeSolverError::TooManyNonlinearSolverFailures {
                        time: solver.problem().eqn.absolute_time(time),
                        num_failures,
                    }
                }
                OdeSolverError::TooManyErrorTestFailures { time, num_failures } => {
                    OdeSolverError::TooManyErrorTestFailures {
                        time: solver.problem().eqn.absolute_time(time),
                        num_failures,
                    }
                }
                OdeSolverError::StepSizeTooSmall { time } => OdeSolverError::StepSizeTooSmall {
                    time: solver.problem().eqn.absolute_time(time),
                },
                other => other,
            };
            DiffsolError::OdeSolverError(error)
        }
        other => other,
    };
    PharmsolError::from_solver_error(error, absolute_target_time)
}

fn cannot_integrate_distinct_stop(
    start_time: f64,
    stop_time: f64,
    last_accepted_time: f64,
) -> PharmsolError {
    let interval = stop_time - start_time;
    PharmsolError::OtherError(format!(
        "ODE solver cannot integrate distinct stop from t = {start_time:.16e} to \
         t = {stop_time:.16e} (gap {interval:.6e}); the last accepted solver time was \
         t = {last_accepted_time:.16e}. No state dynamics were skipped. Check the event \
         schedule: use identical times only for truly simultaneous records; otherwise \
         rescale the model's time unit so this interval is numerically resolvable"
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReachedStop {
    Exact,
    Interpolated,
    Rebased,
}

/// Reconcile the accepted state after diffsol reports `TstopReached`.
///
/// Diffsol's stop comparison is intentionally tolerant: an accepted state may
/// be a few ULPs short of the requested stop. Such a state must not be
/// relabeled. It is rebased once at its accepted absolute time so the complete
/// residual interval can be integrated. A state past the stop is instead
/// interpolated with diffsol's `state_mut_back` contract.
#[allow(clippy::too_many_arguments)]
fn normalize_reached_stop<'a, F, S>(
    solver: &mut S,
    stop_time: f64,
    absolute_stop_time: f64,
    segment_start_time: f64,
    allow_rebase: bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
) -> Result<ReachedStop, PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
{
    let state_time = solver.state().t;
    if !stop_time.is_finite() || !absolute_stop_time.is_finite() || !state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop with non-finite time coordinates: state = {state_time:?}, stop = {stop_time:?}, absolute stop = {absolute_stop_time:?}"
        )));
    }
    let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
    if !absolute_state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop whose absolute state time is not finite: local state = {state_time:?}, absolute state = {absolute_state_time:?}"
        )));
    }
    // Absolute model time is authoritative. Diffsol may report `TstopReached`
    // with its local state one ULP below the local stop even though adding the
    // time origin maps that accepted state exactly to the requested event.
    // In that case there is no absolute residual—and no dynamics—to integrate.
    if absolute_state_time == absolute_stop_time {
        return Ok(ReachedStop::Exact);
    }
    if absolute_state_time > absolute_stop_time {
        if let Err(error) = solver.state_mut_back(stop_time) {
            return Err(solver_error_at_absolute_time(
                solver,
                error,
                absolute_stop_time,
            ));
        }
        return Ok(ReachedStop::Interpolated);
    }

    let step = solver.state().h;
    if !step.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE solver reached a stop with a non-finite accepted step h = {step:?}"
        )));
    }
    let passed_stop =
        (step >= 0.0 && state_time > stop_time) || (step < 0.0 && state_time < stop_time);
    if passed_stop {
        if let Err(error) = solver.state_mut_back(stop_time) {
            return Err(solver_error_at_absolute_time(
                solver,
                error,
                absolute_stop_time,
            ));
        }
        return Ok(ReachedStop::Interpolated);
    }

    let residual = absolute_stop_time - absolute_state_time;
    if !residual.is_finite() || residual <= 0.0 {
        ensure_no_material_infusion_is_skipped(
            solver,
            absolute_state_time,
            absolute_stop_time,
            rtol,
            atol,
        )?;
        return Err(cannot_integrate_distinct_stop(
            segment_start_time,
            absolute_stop_time,
            absolute_state_time,
        ));
    }
    if !allow_rebase {
        ensure_no_material_infusion_is_skipped(
            solver,
            absolute_state_time,
            absolute_stop_time,
            rtol,
            atol,
        )?;
        return Err(cannot_integrate_distinct_stop(
            segment_start_time,
            absolute_stop_time,
            absolute_state_time,
        ));
    }

    rebase_solver_time(solver, absolute_state_time, absolute_stop_time, dy_scratch)?;
    Ok(ReachedStop::Rebased)
}

/// Advance the solver to `next_event_time`, stopping at every scheduled
/// integration boundary in between.
///
/// This is the single implementation of the event-to-event integration loop
/// for every [`ODE`] model: stop selection at integration boundaries,
/// left-continuity handling at covariate and infusion knots, exact restarts
/// for true RHS discontinuities, and safe normalization of stops that diffsol
/// has already reached.
///
/// `after_step` runs after every step; the driver uses it to surface
/// model-function errors raised inside the RHS callback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_solver_to_event<'a, F, S, H>(
    solver: &mut S,
    next_event_time: f64,
    integration_boundary_cursor: &mut usize,
    pending_reinit: &mut bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
    after_step: &mut H,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    let initial_state_time = solver.state().t;
    if !next_event_time.is_finite() || !initial_state_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance requires finite target and solver state times: target = {next_event_time:?}, state = {initial_state_time:?}"
        )));
    }
    let initial_current_time = solver.problem().eqn.absolute_time(initial_state_time);
    if !initial_current_time.is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance produced a non-finite absolute current time from solver state {initial_state_time:?}"
        )));
    }
    if next_event_time < initial_current_time {
        return Err(stop_time_before_current_time(
            next_event_time,
            initial_current_time,
        ));
    }
    if next_event_time == initial_current_time {
        return Ok(());
    }
    if !(next_event_time - initial_current_time).is_finite() {
        return Err(PharmsolError::OtherError(format!(
            "ODE event advance gap from t = {initial_current_time:.16e} to t = {next_event_time:.16e} is not representable"
        )));
    }

    loop {
        let state_time = solver.state().t;
        if !state_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance encountered a non-finite solver state time {state_time:?}"
            )));
        }
        let current_time = solver.problem().eqn.absolute_time(state_time);
        if !current_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance encountered a non-finite absolute current time from solver state {state_time:?}"
            )));
        }
        if next_event_time < current_time {
            return Err(stop_time_before_current_time(next_event_time, current_time));
        }
        if next_event_time == current_time {
            return Ok(());
        }
        if !(next_event_time - current_time).is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance gap from t = {current_time:.16e} to t = {next_event_time:.16e} is not representable"
            )));
        }

        let integration_boundary_times = solver.problem().eqn.integration_boundary_times();
        while *integration_boundary_cursor < integration_boundary_times.len()
            && integration_boundary_times[*integration_boundary_cursor] <= current_time
        {
            *integration_boundary_cursor += 1;
        }

        let (absolute_stop_time, is_integration_boundary) =
            match integration_boundary_times.get(*integration_boundary_cursor) {
                Some(&boundary_time) if boundary_time <= next_event_time => {
                    *integration_boundary_cursor += 1;
                    (boundary_time, true)
                }
                _ => (next_event_time, false),
            };
        let is_discontinuity_boundary = is_integration_boundary
            && solver
                .problem()
                .eqn
                .is_discontinuity_time(absolute_stop_time);
        if !absolute_stop_time.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event advance selected a non-finite stop time {absolute_stop_time:?}"
            )));
        }
        if absolute_stop_time <= current_time {
            return Err(stop_time_before_current_time(
                absolute_stop_time,
                current_time,
            ));
        }
        if !(absolute_stop_time - current_time).is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "ODE event segment gap from t = {current_time:.16e} to t = {absolute_stop_time:.16e} is not representable"
            )));
        }

        solver
            .problem()
            .eqn
            .set_left_continuity_time(if is_integration_boundary {
                Some(absolute_stop_time)
            } else {
                None
            });

        let segment_start_time = current_time;
        solver
            .problem()
            .eqn
            .begin_explicit_step_segment(segment_start_time, absolute_stop_time);
        let configured_minimum_timestep = *solver.config().as_base_ref().minimum_timestep;
        let short_segment = absolute_stop_time - segment_start_time
            <= MINIMUM_TIMESTEP_HEADROOM * configured_minimum_timestep;
        if short_segment {
            *solver.config_mut().as_base_mut().minimum_timestep = 0.0;
        }

        let segment_result = (|| -> Result<(), PharmsolError> {
            let mut rebased = false;
            loop {
                let stop_time = ensure_solver_stop_coordinates::<F, S, H>(
                    solver,
                    absolute_stop_time,
                    dy_scratch,
                    after_step,
                )?;
                let state_time = solver.state().t;
                let local_gap = stop_time - state_time;
                if !stop_time.is_finite() || !state_time.is_finite() || !local_gap.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "ODE event coordinate conversion is not finite: current local t = {state_time:?}, absolute stop = {absolute_stop_time:?}, local stop = {stop_time:?}, gap = {local_gap:?}"
                    )));
                }
                match solver.set_stop_time(stop_time) {
                    Ok(()) => {
                        if *pending_reinit {
                            reinitialize_at_boundary(solver, dy_scratch);
                            after_step()?;
                            *pending_reinit = false;
                        }
                        integrate_to_stop(
                            solver,
                            stop_time,
                            absolute_stop_time,
                            segment_start_time,
                            is_discontinuity_boundary,
                            pending_reinit,
                            !rebased,
                            dy_scratch,
                            rtol,
                            atol,
                            after_step,
                        )?;
                        return Ok(());
                    }
                    Err(diffsol::error::DiffsolError::OdeSolverError(
                        OdeSolverError::StopTimeAtCurrentTime,
                    )) => {
                        let state_time = solver.state().t;
                        let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
                        if !state_time.is_finite() || !absolute_state_time.is_finite() {
                            return Err(PharmsolError::OtherError(format!(
                                "ODE event advance encountered a non-finite state after set_stop_time: local = {state_time:?}, absolute = {absolute_state_time:?}"
                            )));
                        }
                        if absolute_state_time > absolute_stop_time {
                            return Err(stop_time_before_current_time(
                                absolute_stop_time,
                                absolute_state_time,
                            ));
                        }
                        if absolute_state_time == absolute_stop_time {
                            if is_discontinuity_boundary {
                                *pending_reinit = true;
                            }
                            return Ok(());
                        }
                        if rebased {
                            ensure_no_material_infusion_is_skipped(
                                solver,
                                absolute_state_time,
                                absolute_stop_time,
                                rtol,
                                atol,
                            )?;
                            return Err(cannot_integrate_distinct_stop(
                                segment_start_time,
                                absolute_stop_time,
                                absolute_state_time,
                            ));
                        }
                        rebase_solver_time(
                            solver,
                            absolute_state_time,
                            absolute_stop_time,
                            dy_scratch,
                        )?;
                        after_step()?;
                        *pending_reinit = false;
                        rebased = true;
                    }
                    Err(err) => {
                        after_step()?;
                        return Err(solver_error_at_absolute_time(
                            solver,
                            err,
                            absolute_stop_time,
                        ));
                    }
                }
            }
        })();

        if short_segment {
            *solver.config_mut().as_base_mut().minimum_timestep = configured_minimum_timestep;
        }
        // The marker armed above is segment-scoped; this is its only clear.
        solver.problem().eqn.set_left_continuity_time(None);
        if segment_result.is_ok() {
            restore_timestep_after_successful_stop(solver);
        }
        segment_result?;
    }
}

/// Step the solver until the stop set by `set_stop_time`.
///
/// Solver failures are returned directly. Retrying every nonlinear or linear
/// algebra failure can hide structural model errors and has no generally safe,
/// unit-independent restart step.
///
/// The left-continuity marker is owned by [`advance_solver_to_event`], which
/// clears it after every segment; no exit path here manages it.
#[allow(clippy::too_many_arguments)]
fn integrate_to_stop<'a, F, S, H>(
    solver: &mut S,
    stop_time: f64,
    absolute_stop_time: f64,
    segment_start_time: f64,
    is_discontinuity_boundary: bool,
    pending_reinit: &mut bool,
    allow_close_rebase: bool,
    dy_scratch: &mut V,
    rtol: f64,
    atol: f64,
    after_step: &mut H,
) -> Result<(), PharmsolError>
where
    F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
    S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    H: FnMut() -> Result<(), PharmsolError>,
{
    loop {
        if solver.problem().eqn.explicit_step_budget_check_due() {
            // Drain callback errors before deciding whether this is a
            // productive window reset or a fail-closed guard error.
            after_step()?;
            if let Some(error) = solver.problem().eqn.check_explicit_step_budget() {
                return Err(error);
            }
        }

        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                after_step()?;
            }
            Ok(OdeSolverStopReason::TstopReached) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                // Keep left-continuity active while reconciling the accepted
                // state. In particular, a residual close segment ending at a
                // covariate or infusion boundary must use the segment on the
                // left until the exact boundary is reached.
                after_step()?;
                let reached = normalize_reached_stop(
                    solver,
                    stop_time,
                    absolute_stop_time,
                    segment_start_time,
                    allow_close_rebase,
                    dy_scratch,
                    rtol,
                    atol,
                )?;

                match reached {
                    ReachedStop::Exact | ReachedStop::Interpolated => {
                        if matches!(reached, ReachedStop::Interpolated) {
                            *pending_reinit = true;
                        }
                        if is_discontinuity_boundary {
                            *pending_reinit = true;
                        }
                        return Ok(());
                    }
                    ReachedStop::Rebased => {
                        // `rebase_solver_time` refreshes the RHS and Jacobian;
                        // native callbacks may report an error during that
                        // refresh. Surface it before asking diffsol to accept
                        // another stop time.
                        let configured_minimum_timestep =
                            *solver.config().as_base_ref().minimum_timestep;
                        after_step()?;
                        *solver.config_mut().as_base_mut().minimum_timestep = 0.0;
                        let residual_result = (|| -> Result<(), PharmsolError> {
                            let residual_stop = ensure_solver_stop_coordinates::<F, S, H>(
                                solver,
                                absolute_stop_time,
                                dy_scratch,
                                after_step,
                            )?;
                            let residual_state = solver.state().t;
                            let residual_gap = residual_stop - residual_state;
                            if !residual_stop.is_finite()
                                || !residual_state.is_finite()
                                || !residual_gap.is_finite()
                                || residual_gap <= 0.0
                            {
                                let absolute_state_time =
                                    solver.problem().eqn.absolute_time(residual_state);
                                ensure_no_material_infusion_is_skipped(
                                    solver,
                                    absolute_state_time,
                                    absolute_stop_time,
                                    rtol,
                                    atol,
                                )?;
                                return Err(cannot_integrate_distinct_stop(
                                    segment_start_time,
                                    absolute_stop_time,
                                    absolute_state_time,
                                ));
                            }

                            match solver.set_stop_time(residual_stop) {
                                Ok(()) => integrate_to_stop(
                                    solver,
                                    residual_stop,
                                    absolute_stop_time,
                                    segment_start_time,
                                    is_discontinuity_boundary,
                                    pending_reinit,
                                    false,
                                    dy_scratch,
                                    rtol,
                                    atol,
                                    after_step,
                                ),
                                Err(diffsol::error::DiffsolError::OdeSolverError(
                                    OdeSolverError::StopTimeAtCurrentTime,
                                )) => {
                                    let absolute_state_time =
                                        solver.problem().eqn.absolute_time(solver.state().t);
                                    ensure_no_material_infusion_is_skipped(
                                        solver,
                                        absolute_state_time,
                                        absolute_stop_time,
                                        rtol,
                                        atol,
                                    )?;
                                    Err(cannot_integrate_distinct_stop(
                                        segment_start_time,
                                        absolute_stop_time,
                                        absolute_state_time,
                                    ))
                                }
                                Err(error) => {
                                    after_step()?;
                                    Err(solver_error_at_absolute_time(
                                        solver,
                                        error,
                                        absolute_stop_time,
                                    ))
                                }
                            }
                        })();
                        *solver.config_mut().as_base_mut().minimum_timestep =
                            configured_minimum_timestep;
                        residual_result?;
                        return Ok(());
                    }
                }
            }
            Ok(OdeSolverStopReason::RootFound(root_time, _)) => {
                let accepted_time = solver.problem().eqn.absolute_time(solver.state().t);
                solver
                    .problem()
                    .eqn
                    .record_explicit_accepted_step(accepted_time);
                after_step()?;
                let absolute_root_time = solver.problem().eqn.absolute_time(root_time);
                return Err(PharmsolError::OtherError(format!(
                    "solver stopped at an unexpected root at t = {:.4} \
                     (root finding is not configured)",
                    absolute_root_time
                )));
            }
            Err(diffsol::error::DiffsolError::OdeSolverError(
                OdeSolverError::StopTimeAtCurrentTime,
            )) => {
                after_step()?;
                let state_time = solver.state().t;
                let absolute_state_time = solver.problem().eqn.absolute_time(state_time);
                if !state_time.is_finite() || !absolute_state_time.is_finite() {
                    return Err(PharmsolError::OtherError(format!(
                        "ODE solver reached a stop with non-finite state time: local = {state_time:?}, absolute = {absolute_state_time:?}"
                    )));
                }
                if absolute_state_time > absolute_stop_time {
                    if let Err(error) = solver.state_mut_back(stop_time) {
                        return Err(solver_error_at_absolute_time(
                            solver,
                            error,
                            absolute_stop_time,
                        ));
                    }
                    *pending_reinit = true;
                    return Ok(());
                }
                if absolute_state_time < absolute_stop_time {
                    ensure_no_material_infusion_is_skipped(
                        solver,
                        absolute_state_time,
                        absolute_stop_time,
                        rtol,
                        atol,
                    )?;
                    return Err(cannot_integrate_distinct_stop(
                        segment_start_time,
                        absolute_stop_time,
                        absolute_state_time,
                    ));
                }
                if is_discontinuity_boundary {
                    *pending_reinit = true;
                }
                return Ok(());
            }
            Err(err) => {
                // A model-function error raised inside the RHS is the root
                // cause when present; surface it over the solver error.
                after_step()?;
                return Err(solver_error_at_absolute_time(
                    solver,
                    err,
                    absolute_stop_time,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::equation::model::InfusionDose;
    use crate::simulator::M;
    use crate::Covariates;

    fn infusion_step(amount: f64) -> Step<'static> {
        Step::Infusion(InfusionDose {
            time: 1.5,
            duration: 2.0,
            amount,
            input: 0,
        })
    }
    use diffsol::{NalgebraContext, OdeBuilder};

    type GuardTestRhs = fn(&V, &V, f64, &mut V, &V, &V, &Covariates);

    fn unit_rhs(
        _x: &V,
        _p: &V,
        _t: f64,
        dx: &mut V,
        _bolus: &V,
        _rateiv: &V,
        _covariates: &Covariates,
    ) {
        dx[0] = 1.0;
    }

    fn guard_test_problem(h0: f64) -> diffsol::OdeSolverProblem<PMProblem<'static, GuardTestRhs>> {
        let covariates = Covariates::default();
        OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(h0)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    unit_rhs as GuardTestRhs,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed")
    }

    macro_rules! build_guard_test_problem {
        ($h0:expr) => {{
            let covariates = Covariates::default();
            OdeBuilder::<M>::new()
                .atol(vec![1e-4])
                .rtol(1e-4)
                .t0(0.0)
                .h0($h0)
                .p(Vec::<f64>::new())
                .build_from_eqn(
                    PMProblem::with_params_v(
                        unit_rhs as GuardTestRhs,
                        1,
                        0,
                        V::zeros(0, NalgebraContext::new()),
                        &covariates,
                        &[],
                        V::zeros(1, NalgebraContext::new()),
                        0.0,
                    )
                    .expect("finite test ODE problem"),
                )
                .expect("test ODE builder should succeed")
        }};
    }

    #[test]
    fn explicit_step_guard_is_selected_only_for_tsit45() {
        assert_eq!(accepted_step_limits_for_solver(&OdeSolver::Bdf), None);
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
            None
        );
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
            None
        );
        assert_eq!(
            accepted_step_limits_for_solver(&OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
            Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 500_000,
                max_accepted_steps_per_session: 10_000_000,
            })
        );
    }

    #[test]
    fn explicit_step_guard_rejects_stagnant_window_and_accumulates_session() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 4,
            }));
        problem.eqn.begin_explicit_step_segment(10.0, 20.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        problem.eqn.rebase_time_origin(10.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        assert!(problem.eqn.explicit_step_budget_check_due());
        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("a stagnant window must fail closed");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("segment start = 1.0000000000000000e1"));
        assert!(message.contains("target = 2.0000000000000000e1"));
        assert!(message.contains("numeric gap (target - start) = 1.0000000000000000e1"));
        assert!(message.contains("progress window start = 1.0000000000000000e1"));
        assert!(message.contains("actual progress = 0.0000000000000000e0"));
        assert!(message.contains("required progress = 1.0000000000000000e-3"));
        assert!(message.contains("progress fraction = 0.0000000000000000e0"));
        assert!(message.contains("minimum progress fraction = 1.0000000000000000e-4"));
        assert!(message.contains("last accepted absolute time = 1.0000000000000000e1"));
        assert!(message.contains("window count/limit = 2/2"));
        assert!(message.contains("cumulative count/limit = 2/4"));
        assert!(message.contains("no incomplete state was returned"));
        assert!(message.contains("model stiffness and state/time/unit scaling"));
        assert!(message.contains("explicitly choose an implicit solver"));

        problem.eqn.begin_explicit_step_segment(20.0, 30.0);
        assert!(!problem.eqn.explicit_step_budget_check_due());
        problem.eqn.record_explicit_accepted_step(20.0);
        problem.eqn.record_explicit_accepted_step(20.0);
        let second_segment_error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("the session count must survive a new segment");
        let second_segment_message = second_segment_error.to_string();
        assert!(second_segment_message.contains("budget scope = session total"));
        assert!(second_segment_message.contains("segment start = 2.0000000000000000e1"));
        assert!(second_segment_message.contains("target = 3.0000000000000000e1"));
        assert!(second_segment_message.contains("progress window start = 2.0000000000000000e1"));
        assert!(second_segment_message.contains("window count/limit = 2/2"));
        assert!(second_segment_message.contains("cumulative count/limit = 4/4"));
    }

    #[test]
    fn explicit_step_guard_rejects_zero_progress_for_positive_subnormal_gap() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let positive_subnormal_gap = f64::from_bits(1);
        problem
            .eqn
            .begin_explicit_step_segment(0.0, positive_subnormal_gap);
        problem.eqn.record_explicit_accepted_step(0.0);
        assert!(problem.eqn.explicit_step_budget_check_due());

        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("zero progress must fail closed even when required progress underflows");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("required progress = 0.0000000000000000e0"));
        assert!(message.contains("progress fraction = 0.0000000000000000e0"));
        assert!(problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn explicit_step_guard_continues_after_productive_window() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 4,
            }));
        problem.eqn.begin_explicit_step_segment(10.0, 20.0);
        problem.eqn.record_explicit_accepted_step(10.0);
        problem.eqn.record_explicit_accepted_step(10.01);
        assert!(problem.eqn.explicit_step_budget_check_due());
        assert!(problem.eqn.check_explicit_step_budget().is_none());
        assert!(!problem.eqn.explicit_step_budget_check_due());
        problem.eqn.record_explicit_accepted_step(10.02);
        assert!(!problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn rebase_and_residual_do_not_reset_session_budget() {
        let problem = build_guard_test_problem!(1.0);
        let target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 3,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        problem.eqn.begin_explicit_step_segment(0.0, target);
        problem.eqn.record_explicit_accepted_step(0.0);
        rebase_solver_time(&mut solver, 0.0, target, &mut dy_scratch)
            .expect("the test rebase should preserve a positive residual");

        let mut pending_reinit = false;
        let _error = integrate_to_stop(
            &mut solver,
            target,
            target,
            0.0,
            false,
            &mut pending_reinit,
            false,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the residual must stop before requesting another step");

        let error = problem
            .eqn
            .check_explicit_step_budget()
            .expect("the residual accepted step must retain the session count");
        let message = error.to_string();
        assert!(message.contains("budget scope = session total"));
        assert!(message.contains("window count/limit = 2/3"));
        assert!(message.contains("cumulative count/limit = 2/2"));
    }

    #[test]
    fn advance_resets_segment_budget_but_not_session_budget() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let first_target = 1.0e-4_f64.next_down();

        advance_solver_to_event(
            &mut solver,
            first_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("the first segment should use its one-step budget");

        let current_time = solver.state().t;
        let next_target = current_time + solver.state().h;
        assert!(next_target > current_time);
        advance_solver_to_event(
            &mut solver,
            next_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a new exact segment should reset only its segment budget");

        let state_before_third = solver.state().t;
        let third_target = state_before_third + solver.state().h;
        assert!(third_target > state_before_third);
        let error = advance_solver_to_event(
            &mut solver,
            third_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the cumulative session budget must block the next segment");
        let message = error.to_string();
        assert!(message.contains("budget scope = session total"));
        assert!(message.contains("window count/limit = 0/1"));
        assert!(message.contains("cumulative count/limit = 2/2"));
        assert!(message.contains("numeric gap (target - start)"));
        assert_eq!(solver.state().t, state_before_third);
    }

    #[test]
    fn disabled_explicit_step_guard_never_limits_implicit_session() {
        let problem = guard_test_problem(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(accepted_step_limits_for_solver(&OdeSolver::Bdf));
        problem.eqn.begin_explicit_step_segment(0.0, 2.0);
        for time in 1..=4 {
            problem.eqn.record_explicit_accepted_step(f64::from(time));
        }
        assert!(!problem.eqn.explicit_step_budget_check_due());
    }

    #[test]
    fn explicit_step_guard_checks_before_requesting_next_step() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            10.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("the second accepted step must be blocked");
        let message = error.to_string();
        assert!(message.contains("budget scope = insufficient segment progress"));
        assert!(message.contains("numeric gap (target - start)"));
        assert!(message.contains("progress window start ="));
        assert!(message.contains("actual progress ="));
        assert!(message.contains("required progress ="));
        assert!(message.contains("progress fraction ="));
        assert!(message.contains("window count/limit = 1/1"));
        assert!(message.contains("cumulative count/limit = 1/2"));
        assert!(message.contains("last accepted absolute time"));
        assert!(message.contains("no incomplete state was returned"));
        assert!(message.contains("explicitly choose an implicit solver"));
        assert!(solver.state().t > 0.0);
        assert!(solver.state().t < 10.0);
    }

    #[test]
    fn explicit_step_guard_accepts_exact_completion_on_final_step() {
        let problem = build_guard_test_problem!(1.0);
        let final_target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 1,
                max_accepted_steps_per_session: 2,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        advance_solver_to_event(
            &mut solver,
            final_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a final permitted step that reaches the target is complete");
        assert_eq!(solver.state().t, final_target);
    }

    #[test]
    fn explicit_step_guard_accepts_exact_completion_on_final_session_step() {
        let problem = build_guard_test_problem!(1.0);
        let final_target = 1.0e-4_f64.next_down();
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 2,
                max_accepted_steps_per_session: 1,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        advance_solver_to_event(
            &mut solver,
            final_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("a final permitted session step that reaches the target is complete");
        assert_eq!(solver.state().t, final_target);
    }

    #[test]
    fn callback_error_precedes_set_stop_time_error() {
        let problem = build_guard_test_problem!(1.0);
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        *solver.state_mut().h = -1.0;
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Err(PharmsolError::OtherError("native callback error".into())),
        )
        .expect_err("callback failure must precede the set_stop_time error");
        assert!(error.to_string().contains("native callback error"));
    }

    #[test]
    fn callback_error_precedes_explicit_step_budget_error() {
        let problem = build_guard_test_problem!(1.0);
        problem
            .eqn
            .configure_explicit_step_guard(Some(ExplicitStepBudgetConfig {
                max_accepted_steps_per_segment: 0,
                max_accepted_steps_per_session: 1,
            }));
        let mut solver = problem.tsit45().expect("test TSIT45 solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let error = advance_solver_to_event(
            &mut solver,
            1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Err(PharmsolError::OtherError("native callback error".into())),
        )
        .expect_err("callback failure must retain precedence");
        assert!(error.to_string().contains("native callback error"));
    }

    #[test]
    fn resolved_schedule_rejects_positive_nonzero_infusion_rate_underflow() {
        let amount = f64::from_bits(1);
        let events = [infusion_step(amount)];
        let error = validate_resolved_ode_schedule(&events)
            .expect_err("positive nonzero infusion underflow must be rejected");
        let message = error.to_string();
        assert!(message.contains("t = 1.5000000000000000e0"));
        assert!(message.contains("amount"));
        assert!(message.contains("duration"));
        assert!(message.contains("silently lose nonzero infusion material"));
    }

    #[test]
    fn resolved_schedule_rejects_negative_nonzero_infusion_rate_underflow() {
        let amount = -f64::from_bits(1);
        let events = [infusion_step(amount)];
        let error = validate_resolved_ode_schedule(&events)
            .expect_err("negative nonzero infusion underflow must be rejected");
        let message = error.to_string();
        assert!(message.contains("t = 1.5000000000000000e0"));
        assert!(message.contains("amount"));
        assert!(message.contains("duration"));
        assert!(message.contains("silently lose nonzero infusion material"));
    }

    #[test]
    fn resolved_schedule_accepts_zero_amount_infusion_with_zero_rate() {
        let events = [infusion_step(0.0)];
        assert_eq!(
            validate_resolved_ode_schedule(&events).expect("zero amount is material-free"),
            1.5
        );
    }

    #[test]
    fn shared_advance_helper_rejects_backward_targets() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let error = advance_solver_to_event(
            &mut solver,
            -1.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect_err("backward event target must be rejected");

        assert!(error.to_string().contains("before current time"));
        assert_eq!(solver.state().t, 0.0);

        let mut callback_called = false;
        advance_solver_to_event(
            &mut solver,
            0.0,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || {
                callback_called = true;
                Ok(())
            },
        )
        .expect("equal target should be a zero-length no-op");
        assert!(!callback_called);
        assert_eq!(solver.state().t, 0.0);
    }

    #[test]
    fn shared_advance_helper_normalizes_unrepresentable_local_targets() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    -192.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        advance_solver_to_event(
            &mut solver,
            0.35,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("finite cancellation should be normalized before set_stop_time");

        let absolute_time = solver.problem().eqn.absolute_time(solver.state().t);
        assert_eq!(
            absolute_time,
            0.35,
            "origin = {}, local state = {}",
            solver.problem().eqn.time_origin(),
            solver.state().t
        );
    }

    #[test]
    fn normalize_reached_stop_does_not_relabel_local_time_equality() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    -1.0e16,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        *solver.state_mut().t = 1.0e16;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let reached = normalize_reached_stop(
            &mut solver,
            1.0e16,
            1.0,
            0.0,
            true,
            &mut dy_scratch,
            1e-4,
            1e-4,
        )
        .expect("mismatched local equality should use the checked rebase");

        assert_eq!(reached, ReachedStop::Rebased);
        assert_eq!(solver.problem().eqn.time_origin(), 0.0);
        assert_eq!(solver.state().t, 0.0);
    }

    #[test]
    fn normalize_reached_stop_accepts_exact_absolute_time_with_local_roundoff() {
        let origin = f64::from_bits(0x3ff8_4444_4444_439d);
        let local_state = f64::from_bits(0x3ff8_0000_0000_00b6);
        let local_stop = f64::from_bits(0x3ff8_0000_0000_00b7);
        let absolute_stop = f64::from_bits(0x4008_2222_2222_222a);
        assert_ne!(local_state, local_stop);
        assert_eq!(origin + local_state, absolute_stop);

        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    origin,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        *solver.state_mut().t = local_state;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());

        let reached = normalize_reached_stop(
            &mut solver,
            local_stop,
            absolute_stop,
            origin,
            true,
            &mut dy_scratch,
            1e-4,
            1e-4,
        )
        .expect("exact absolute time should not require a residual integration");

        assert_eq!(reached, ReachedStop::Exact);
        assert_eq!(solver.state().t, local_state);
        assert_eq!(
            solver.problem().eqn.absolute_time(local_state),
            absolute_stop
        );
    }

    #[test]
    fn rebase_callback_error_precedes_followup_solver_error() {
        let covariates = Covariates::default();
        let problem = OdeBuilder::<M>::new()
            .atol(vec![1e-4])
            .rtol(1e-4)
            .t0(0.0)
            .h0(1e-3)
            .p(Vec::<f64>::new())
            .build_from_eqn(
                PMProblem::with_params_v(
                    |_x, _p, _t, dx, _bolus, _rateiv, _cov| dx[0] = 0.0,
                    1,
                    0,
                    V::zeros(0, NalgebraContext::new()),
                    &covariates,
                    &[],
                    V::zeros(1, NalgebraContext::new()),
                    0.0,
                )
                .expect("finite test ODE problem"),
            )
            .expect("test ODE builder should succeed");
        let mut solver = problem
            .bdf::<diffsol::NalgebraLU<f64>>()
            .expect("test BDF solver should build");
        let mut integration_boundary_cursor = 0;
        let mut pending_reinit = false;
        let mut dy_scratch = V::zeros(1, NalgebraContext::new());
        let large_time = 1.0e6_f64;
        advance_solver_to_event(
            &mut solver,
            large_time,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || Ok(()),
        )
        .expect("the solver should reach the large event before the close target");

        let close_target = large_time.next_up();
        let error = advance_solver_to_event(
            &mut solver,
            close_target,
            &mut integration_boundary_cursor,
            &mut pending_reinit,
            &mut dy_scratch,
            1e-4,
            1e-4,
            &mut || {
                Err(PharmsolError::OtherError(
                    "callback error during rebase".into(),
                ))
            },
        )
        .expect_err("the callback error must be returned");

        assert!(error.to_string().contains("callback error during rebase"));
        assert_eq!(solver.problem().eqn.time_origin(), large_time);
        assert_eq!(solver.state().t, 0.0);
    }
}
