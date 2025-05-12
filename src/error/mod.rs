use thiserror::Error;

use crate::data::error_model::ErrorModelError;

#[derive(Error, Debug)]
pub enum PharmsolError {
    #[error("Error in the error model: {0}")]
    ErrorModelError(#[from] ErrorModelError),
}
