use thiserror::Error;

use crate::data::error_model::ErrorModelError;
use crate::data::parse_pmetrics::PmetricsError;
use ndarray::ShapeError;

#[derive(Error, Debug, Clone)]
pub enum PharmsolError {
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
    #[error("Shape error: {0}")]
    NdarrayShapeError(#[from] ShapeError),
    #[error("Error parsing Pmetrics datafile: {0}")]
    PmetricsError(#[from] PmetricsError),
    #[error("Other error: {0}")]
    OtherError(String),
}
