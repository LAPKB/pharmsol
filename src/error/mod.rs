use thiserror::Error;

use crate::data::error_model::ErrorModelError;
use crate::data::parser::pmetrics::PmetricsError;

use crate::CovariateError;

use ndarray::ShapeError;

#[derive(Error, Debug, Clone)]
pub enum PharmsolError {
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
    #[error("Covariate error: {0}")]
    CovariateError(#[from] CovariateError),
    #[error("Shape error: {0}")]
    NdarrayShapeError(#[from] ShapeError),
    #[error("Error parsing Pmetrics datafile: {0}")]
    PmetricsError(#[from] PmetricsError),
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
}

impl From<diffsol::error::DiffsolError> for PharmsolError {
    fn from(error: diffsol::error::DiffsolError) -> Self {
        PharmsolError::DiffsolError(error.to_string())
    }
}
