use thiserror::Error;

use pharmsol_dsl::RouteKind;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::data::error_model::ErrorModelError;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::data::row::DataError;

use crate::parameters::ParameterError;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::CovariateError;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use ndarray::ShapeError;

#[derive(Error, Debug, Clone)]
pub enum PharmsolError {
    #[error("Parameter error: {0}")]
    ParameterError(#[from] ParameterError),
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[error("Covariate error: {0}")]
    CovariateError(#[from] CovariateError),
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[error("Shape error: {0}")]
    NdarrayShapeError(#[from] ShapeError),
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
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

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl From<diffsol::error::DiffsolError> for PharmsolError {
    fn from(error: diffsol::error::DiffsolError) -> Self {
        PharmsolError::DiffsolError(describe_diffsol_error(&error, None))
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
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
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn describe_diffsol_error(
    error: &diffsol::error::DiffsolError,
    target_time: Option<f64>,
) -> String {
    use diffsol::error::{DiffsolError, LinearSolverError, NonLinearSolverError, OdeSolverError};

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
        DiffsolError::LinearSolverError(lin) => match lin {
            LinearSolverError::LuSolveFailed | LinearSolverError::LuNotInitialized => format!(
                "linear (LU) solve failed{toward}; the Jacobian is singular or near-singular."
            ),
            other => format!("linear solver error{toward}: {other}"),
        },
        other => match target_time {
            Some(_) => format!("solver error{toward}: {other}"),
            None => other.to_string(),
        },
    }
}
