//! NCA error types

use thiserror::Error;

/// Errors that can occur during NCA analysis
#[derive(Error, Debug, Clone)]
pub enum NCAError {
    /// An error from observation data processing (BLQ filtering, profile construction)
    #[error(transparent)]
    Observation(#[from] crate::data::observation_error::ObservationError),

    /// An error from observation metrics computation
    #[error(transparent)]
    Metrics(#[from] crate::data::traits::MetricsError),

    /// Lambda-z estimation failed
    #[error("Lambda-z estimation failed: {reason}")]
    LambdaZFailed { reason: String },

    /// Invalid parameter value
    #[error("Invalid parameter: {param} = {value}")]
    InvalidParameter { param: String, value: String },


}
