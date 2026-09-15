//! Error types for observation data processing
//!
//! [`ObservationError`] covers errors that arise during observation extraction,
//! BLQ filtering, and profile construction. These are data-level errors that
//! don't depend on NCA analysis — they can occur whenever working with
//! concentration-time data.
//!
//! NCA code can propagate these via the [`From`] impl on `NCAError`.

use thiserror::Error;

/// Errors arising from observation data processing
///
/// These represent problems with the input data itself, not with NCA analysis.
/// Used by [`ObservationProfile`](crate::nca::observation::ObservationProfile)
/// construction methods.
#[derive(Error, Debug, Clone)]
pub enum ObservationError {
    /// Insufficient data points for the requested operation
    #[error("Insufficient data: {n} points, need at least {required}")]
    InsufficientData {
        /// Number of points available
        n: usize,
        /// Minimum number required
        required: usize,
    },

    /// Time values are not monotonically increasing
    #[error("Invalid time sequence: times must be monotonically increasing")]
    InvalidTimeSequence,

    /// All values are zero or below the limit of quantification
    #[error("All values are zero or below quantification limit")]
    AllBelowLOQ,

    /// No observations carry the requested output label
    ///
    /// The requested label is reported together with the labels that *are*
    /// present, so a typo or a stale numeric column is immediately visible.
    #[error("No observations found for output `{outeq}`{}", if available.is_empty() {
        " (this occasion has no observations)".to_string()
    } else {
        format!(" (present in this occasion: {})", available.join(", "))
    })]
    NoObservations {
        /// The requested output label
        outeq: String,
        /// Output labels that are present in the data
        available: Vec<String>,
    },

    /// Array length mismatch between parallel input arrays
    #[error("Array length mismatch: {description}")]
    ArrayLengthMismatch {
        /// Description of which arrays mismatched and their lengths
        description: String,
    },
}
