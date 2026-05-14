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
    #[error("Input label `{label}` could not be resolved to a route input")]
    UnknownInputLabel { label: String },
    #[error("Output label `{label}` could not be resolved to an output")]
    UnknownOutputLabel { label: String },
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
        PharmsolError::DiffsolError(error.to_string())
    }
}
