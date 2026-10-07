use thiserror::Error;

use pharmsol_dsl::RouteKind;

use crate::data::error_model::ErrorModelError;
use crate::data::parser::pmetrics::DataError;

use crate::parameters::ParameterError;
use crate::CovariateError;

use ndarray::ShapeError;

#[derive(Error, Debug, Clone)]
pub enum PharmsolError {
    #[error("Parameter error: {0}")]
    ParameterError(#[from] ParameterError),
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
    #[error("Covariate error: {0}")]
    CovariateError(#[from] CovariateError),
    #[error("Shape error: {0}")]
    NdarrayShapeError(#[from] ShapeError),
    #[error("Error parsing data: {0}")]
    DataError(#[from] DataError),
    #[error("Diffsol error: {0}")]
    DiffsolError(String),
    #[error("Other error: {0}")]
    OtherError(String),
    #[error("Error setting up progress bar: {0}")]
    ProgressBarError(String),
    #[error("Likelihood is not finite: {0}")]
    NonFiniteLikelihood(f64),
    #[error("The calculated likelihood is zero")]
    ZeroLikelihood,
    #[error("Missing observation in prediction")]
    MissingObservation,
    #[error("Input label `{label}` could not be resolved to a route input{available}")]
    UnknownInputLabel { label: String, available: String },
    #[error("Output label `{label}` could not be resolved to an output{available}")]
    UnknownOutputLabel { label: String, available: String },
    #[error("Input index {input} does not support route kind {kind:?}")]
    UnsupportedInputRouteKind { input: usize, kind: RouteKind },
    #[error("Input index {input} is out of range (ndrugs = {ndrugs})")]
    InputOutOfRange { input: usize, ndrugs: usize },
    #[error("Output equation {outeq} is out of range (nout = {nout})")]
    OuteqOutOfRange { outeq: usize, nout: usize },
    #[error("Compiled model `{model}` has invalid runtime metadata: {detail}")]
    InvalidMetadata { model: String, detail: String },
}

impl From<diffsol::error::DiffsolError> for PharmsolError {
    fn from(error: diffsol::error::DiffsolError) -> Self {
        PharmsolError::DiffsolError(describe_diffsol_error(&error, None))
    }
}

impl PharmsolError {
    /// Build a descriptive [`PharmsolError`] from a diffsol solver error,
    /// adding the integration target time and the likely root cause.
    pub fn from_solver_error(error: diffsol::error::DiffsolError, target_time: f64) -> Self {
        PharmsolError::DiffsolError(describe_diffsol_error(&error, Some(target_time)))
    }
}

impl PharmsolError {
    /// Tag a simulation error with the failing subject ID and support point.
    ///
    /// If `parameter_names` matches `parameters` in length the support point is
    /// rendered as `name=value` pairs, otherwise as a bare value list. Only
    /// [`PharmsolError::DiffsolError`] and [`PharmsolError::OtherError`] are
    /// augmented; other variants pass through unchanged.
    pub fn with_subject_context(
        self,
        subject_id: &str,
        parameters: &[f64],
        parameter_names: &[&str],
    ) -> Self {
        let support_point = if parameter_names.len() == parameters.len() {
            let pairs = parameter_names
                .iter()
                .zip(parameters)
                .map(|(name, value)| format!("{name}={value:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{pairs}}}")
        } else {
            format!("{parameters:?}")
        };
        let context = format!(" [subject `{subject_id}`, support point {support_point}]");
        match self {
            PharmsolError::DiffsolError(msg) => {
                PharmsolError::DiffsolError(format!("{msg}{context}"))
            }
            PharmsolError::OtherError(msg) => PharmsolError::OtherError(format!("{msg}{context}")),
            other => other,
        }
    }

    /// Build an [`UnknownInputLabel`](PharmsolError::UnknownInputLabel) error,
    /// listing the valid route labels when they are known (empty otherwise).
    pub fn unknown_input_label(label: impl std::fmt::Display, available: &[&str]) -> Self {
        PharmsolError::UnknownInputLabel {
            label: label.to_string(),
            available: format_available(available),
        }
    }

    /// Build an [`UnknownOutputLabel`](PharmsolError::UnknownOutputLabel) error,
    /// listing the valid output labels when they are known (empty otherwise).
    pub fn unknown_output_label(label: impl std::fmt::Display, available: &[&str]) -> Self {
        PharmsolError::UnknownOutputLabel {
            label: label.to_string(),
            available: format_available(available),
        }
    }
}

/// Snapshot of the most recent right-hand-side evaluation that produced
/// non-finite derivatives.
///
/// Recorded by the compiled-model dynamics callback and cleared as soon as a
/// later evaluation is finite, so it is only still present when the solver gave
/// up straight after a non-finite evaluation, which is the usual cause of
/// "step size collapsed" and "Newton solver failed" errors.
#[cfg_attr(not(feature = "dsl"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct NonFiniteRhs {
    /// Solver-local time of the evaluation.
    pub time: f64,
    pub state: Vec<f64>,
    pub derivative: Vec<f64>,
}

/// Ratio of largest to smallest non-zero state magnitude above which a solver
/// failure is annotated with a rescaling hint.
#[cfg_attr(not(feature = "dsl"), allow(dead_code))]
const STATE_RANGE_HINT_RATIO: f64 = 1e8;
/// Maximum number of entries listed in a diagnostic note.
#[cfg_attr(not(feature = "dsl"), allow(dead_code))]
const MAX_LISTED_STATES: usize = 6;

#[cfg_attr(not(feature = "dsl"), allow(dead_code))]
impl PharmsolError {
    /// Append actionable context to a solver failure.
    ///
    /// Adds (when applicable): the non-finite derivatives of the last RHS
    /// evaluation together with the state that produced them, non-finite entries
    /// in the last accepted state, and a rescaling hint when the accepted state
    /// spans many orders of magnitude. Only [`PharmsolError::DiffsolError`] is
    /// augmented; all other variants pass through unchanged.
    ///
    /// `state_name` maps a state index to a display label. `rhs` pairs the
    /// recorded evaluation with its absolute time.
    pub(crate) fn with_solver_diagnostics(
        self,
        accepted_time: f64,
        accepted_state: &[f64],
        atol: f64,
        state_name: &dyn Fn(usize) -> String,
        rhs: Option<(&NonFiniteRhs, f64)>,
    ) -> Self {
        let PharmsolError::DiffsolError(message) = self else {
            return self;
        };
        let mut notes = Vec::new();

        if let Some((rhs, absolute_time)) = rhs {
            let bad = rhs
                .derivative
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.is_finite())
                .map(|(i, value)| format!("d{}/dt = {value}", state_name(i)))
                .take(MAX_LISTED_STATES)
                .collect::<Vec<_>>()
                .join(", ");
            let state = list_states(&rhs.state, state_name);
            notes.push(format!(
                "the last right-hand-side evaluation (t = {absolute_time:.4}) returned non-finite derivatives ({bad}) for state [{state}]; check for division by zero, negative bases of fractional powers, or overflow. A state that overshoots slightly below zero inside a stiff step can turn terms such as (c/x)^h into NaN"
            ));
        }

        let non_finite = accepted_state
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_finite())
            .map(|(i, value)| format!("{} = {value}", state_name(i)))
            .take(MAX_LISTED_STATES)
            .collect::<Vec<_>>();
        if !non_finite.is_empty() {
            notes.push(format!(
                "the last accepted state (t = {accepted_time:.4}) contains non-finite values ({})",
                non_finite.join(", ")
            ));
        } else {
            let magnitudes = accepted_state
                .iter()
                .enumerate()
                .map(|(i, value)| (i, value.abs()))
                .filter(|(_, magnitude)| *magnitude > 0.0);
            let largest = magnitudes.clone().max_by(|a, b| a.1.total_cmp(&b.1));
            let smallest = magnitudes.min_by(|a, b| a.1.total_cmp(&b.1));
            if let (Some((hi, max)), Some((lo, min))) = (largest, smallest) {
                if max / min >= STATE_RANGE_HINT_RATIO {
                    notes.push(format!(
                        "the last accepted state (t = {accepted_time:.4}) spans {:.0} orders of magnitude ({} = {max:.3e}, {} = {min:.3e}) with atol = {atol:e}; consider rescaling states (e.g. counts in units of 1e6) or bounding unchecked exponential growth",
                        (max / min).log10().floor(),
                        state_name(hi),
                        state_name(lo),
                    ));
                }
            }
        }

        if notes.is_empty() {
            return PharmsolError::DiffsolError(message);
        }
        PharmsolError::DiffsolError(format!("{message} Diagnostics: {}.", notes.join("; ")))
    }
}

#[cfg_attr(not(feature = "dsl"), allow(dead_code))]
fn list_states(state: &[f64], state_name: &dyn Fn(usize) -> String) -> String {
    let mut listed = state
        .iter()
        .enumerate()
        .take(MAX_LISTED_STATES)
        .map(|(i, value)| format!("{} = {value:.4e}", state_name(i)))
        .collect::<Vec<_>>()
        .join(", ");
    if state.len() > MAX_LISTED_STATES {
        listed.push_str(", ...");
    }
    listed
}

/// Render a ` (available: a, b, c)` suffix, or an empty string when the list is
/// empty (e.g. index-only models without named labels).
fn format_available(labels: &[&str]) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!(" (available: {})", labels.join(", "))
    }
}

/// Build a concise diagnostic message from a [`diffsol::error::DiffsolError`].
///
/// When `target_time` is set it is appended to time-dependent failures to show
/// how far the solver was advancing. Matched variants add the likely cause.
fn describe_diffsol_error(
    error: &diffsol::error::DiffsolError,
    target_time: Option<f64>,
) -> String {
    use diffsol::error::{DiffsolError, NonLinearSolverError, OdeSolverError};
    use diffsol_la::error::{LaError, LinearSolverError};

    // Suffix describing where the solver was headed, used for time-dependent failures.
    let toward = match target_time {
        Some(t) => format!(" while advancing toward t = {t:.4}"),
        None => String::new(),
    };

    match error {
        DiffsolError::OdeSolverError(ode) => match ode {
            OdeSolverError::StepSizeTooSmall { time } => format!(
                "step size collapsed to zero at t = {time:.4}{toward}; \
                 a parameter is likely near zero, infinite, or NaN, or the system is too stiff."
            ),
            OdeSolverError::TooManyNonlinearSolverFailures { time, num_failures } => format!(
                "Newton solver failed {num_failures} times at t = {time:.4}{toward}; \
                 the system is likely stiff or ill-conditioned."
            ),
            OdeSolverError::TooManyErrorTestFailures { time, num_failures } => format!(
                "error test failed {num_failures} times at t = {time:.4}{toward}; \
                 tolerances may be too tight or the parameters implausible."
            ),
            OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time,
                state_time,
            } => format!(
                "stop time t = {stop_time:.4} is before current time t = {state_time:.4}; \
                 event times may be out of order."
            ),
            OdeSolverError::StateProblemMismatch => {
                format!("initial state is inconsistent with the model equations{toward}.")
            }
            OdeSolverError::InvalidTableau(msg) => format!("invalid solver tableau: {msg}"),
            OdeSolverError::BuilderError(msg) => format!("failed to build ODE problem: {msg}"),
            other => format!("ODE solver error{toward}: {other}"),
        },
        DiffsolError::NonLinearSolverError(nl) => match nl {
            NonLinearSolverError::NewtonDiverged => {
                format!("Newton iteration diverged{toward}; the system may be unstable.")
            }
            NonLinearSolverError::NewtonMaxIterations => {
                format!("Newton iteration did not converge{toward}; the system is likely stiff.")
            }
            NonLinearSolverError::InitialConditionDidNotConverge => {
                format!("could not find consistent initial conditions{toward}.")
            }
            other => format!("nonlinear solver error{toward}: {other}"),
        },
        DiffsolError::LaError(LaError::LinearSolverError(lin)) => match lin {
            LinearSolverError::LuSolveFailed | LinearSolverError::LuNotInitialized => format!(
                "linear (LU) solve failed{toward}; the Jacobian is singular or near-singular."
            ),
            other => format!("linear solver error{toward}: {other}"),
        },
        DiffsolError::LaError(la) => format!("linear algebra error{toward}: {la}"),
        other => match target_time {
            Some(_) => format!("solver error{toward}: {other}"),
            None => other.to_string(),
        },
    }
}

#[cfg(test)]
mod solver_diagnostics_tests {
    use super::*;

    fn solver_error() -> PharmsolError {
        PharmsolError::DiffsolError("Newton solver failed".to_string())
    }

    fn name(i: usize) -> String {
        ["drug", "bacteria"]
            .get(i)
            .map_or_else(|| format!("x[{i}]"), |n| n.to_string())
    }

    #[test]
    fn wide_state_range_adds_rescaling_hint() {
        let message = solver_error()
            .with_solver_diagnostics(3.0, &[1.5, 1.6e10], 1e-4, &name, None)
            .to_string();
        assert!(message.contains("10 orders of magnitude"), "{message}");
        assert!(message.contains("bacteria = 1.600e10"), "{message}");
        assert!(message.contains("rescaling"), "{message}");
    }

    #[test]
    fn moderate_state_range_is_not_annotated() {
        let original = solver_error().to_string();
        let message = solver_error()
            .with_solver_diagnostics(3.0, &[1.5, 1.0e5], 1e-4, &name, None)
            .to_string();
        assert_eq!(message, original);
    }

    #[test]
    fn non_finite_accepted_state_is_named() {
        let message = solver_error()
            .with_solver_diagnostics(3.0, &[1.0, f64::NAN], 1e-4, &name, None)
            .to_string();
        assert!(message.contains("bacteria = NaN"), "{message}");
    }

    #[test]
    fn non_finite_rhs_lists_derivatives_and_state() {
        let rhs = NonFiniteRhs {
            time: 2.0,
            state: vec![1.0, -1e-9],
            derivative: vec![0.5, f64::NAN],
        };
        let message = solver_error()
            .with_solver_diagnostics(2.0, &[1.0, 1.0], 1e-4, &name, Some((&rhs, 12.0)))
            .to_string();
        assert!(message.contains("dbacteria/dt = NaN"), "{message}");
        assert!(message.contains("t = 12.0000"), "{message}");
        assert!(message.contains("bacteria = -1.0000e-9"), "{message}");
    }

    #[test]
    fn non_solver_errors_pass_through() {
        let error = PharmsolError::OtherError("boom".to_string()).with_solver_diagnostics(
            0.0,
            &[f64::NAN],
            1e-4,
            &name,
            None,
        );
        assert_eq!(error.to_string(), "Other error: boom");
    }
}
