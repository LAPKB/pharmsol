use thiserror::Error;

use crate::data::error_model::ErrorModelError;
use ndarray::ShapeError;

#[derive(Error, Debug, Clone)]
pub enum PharmsolError {
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
    #[error("Shape error: {0}")]
    NdarrayShapeError(#[from] ShapeError),
}
