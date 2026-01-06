//! NCA error types

use thiserror::Error;

/// Errors that can occur during NCA analysis
#[derive(Error, Debug, Clone)]
pub enum NCAError {
    /// No observations found for the specified output equation
    #[error("No observations found for outeq {outeq}")]
    NoObservations { outeq: usize },

    /// Insufficient data points for analysis
    #[error("Insufficient data: {n} points, need at least {required}")]
    InsufficientData { n: usize, required: usize },

    /// Occasion not found
    #[error("Occasion {index} not found")]
    OccasionNotFound { index: usize },

    /// Subject not found
    #[error("Subject '{id}' not found")]
    SubjectNotFound { id: String },

    /// All concentrations are zero or BLQ
    #[error("All concentrations are zero or below LOQ")]
    AllBLQ,

    /// Lambda-z estimation failed
    #[error("Lambda-z estimation failed: {reason}")]
    LambdaZFailed { reason: String },

    /// Invalid time sequence
    #[error("Invalid time sequence: times must be monotonically increasing")]
    InvalidTimeSequence,

    /// Invalid parameter value
    #[error("Invalid parameter: {param} = {value}")]
    InvalidParameter { param: String, value: String },
}
