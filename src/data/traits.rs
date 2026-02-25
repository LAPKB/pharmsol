//! Extension traits for observation-level pharmacokinetic metrics
//!
//! These traits provide convenient access to AUC, Cmax, Tmax, and other
//! observation-derived metrics on [`Data`], [`Subject`], and [`Occasion`].
//! These are generic observation-level computations, not NCA-specific —
//! they belong in the data layer because they operate on raw observed data
//! and are useful for any downstream analysis (NCA, BestDose, model diagnostics, etc.).
//!
//! # Example
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//!
//! let subject = Subject::builder("pt1")
//!     .bolus(0.0, 100.0, 0)
//!     .observation(1.0, 10.0, 0)
//!     .observation(2.0, 8.0, 0)
//!     .observation(4.0, 4.0, 0)
//!     .build();
//!
//! let auc = subject.auc(0, &AUCMethod::Linear, &BLQRule::Exclude);
//! let cmax = subject.cmax(0, &BLQRule::Exclude);
//! let cmax_val = subject.cmax_first(0, &BLQRule::Exclude).unwrap();
//! ```

use crate::data::event::{AUCMethod, BLQRule};
use crate::data::observation::ObservationProfile;
use crate::data::observation_error::ObservationError;
use crate::{Data, Occasion, Subject};
use rayon::prelude::*;

/// Error type for observation metric computations
///
/// Wraps [`ObservationError`] with optional context about which subject,
/// occasion, or output equation failed. This provides better error messages
/// than bare `ObservationError`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MetricsError {
    /// An error from observation data processing
    #[error(transparent)]
    Observation(#[from] ObservationError),

    /// Output equation not found in subject data
    #[error("Output equation {outeq} not found in subject{}", subject_id.as_ref().map(|id| format!(" '{}'", id)).unwrap_or_default())]
    OutputEquationNotFound {
        /// The requested output equation index
        outeq: usize,
        /// Optional subject identifier for context
        subject_id: Option<String>,
    },
}

/// Extension trait for observation-level pharmacokinetic metrics
///
/// Provides convenient access to AUC, Cmax, Tmax, etc. without running
/// full NCA analysis. Each method returns one result per occasion.
///
/// For single-occasion convenience, use the `_first()` variants which
/// return a single `Result` instead of `Vec<Result<...>>`.
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::prelude::*;
///
/// let subject = Subject::builder("pt1")
///     .bolus(0.0, 100.0, 0)
///     .observation(1.0, 10.0, 0)
///     .observation(2.0, 8.0, 0)
///     .observation(4.0, 4.0, 0)
///     .build();
///
/// // Per-occasion results
/// let auc = subject.auc(0, &AUCMethod::Linear, &BLQRule::Exclude);
///
/// // Single-occasion convenience
/// let cmax = subject.cmax_first(0, &BLQRule::Exclude).unwrap();
/// ```
pub trait ObservationMetrics {
    /// Calculate AUC from time 0 to Tlast
    fn auc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    /// Calculate partial AUC over a time interval
    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    /// Get Cmax (maximum concentration)
    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Tmax (time of maximum concentration)
    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Clast (last quantifiable concentration)
    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Tlast (time of last quantifiable concentration)
    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Calculate AUMC (Area Under the first Moment Curve)
    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    /// Get filtered observation profiles
    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>>;

    // ========================================================================
    // Convenience methods for the single-occasion common case
    // ========================================================================

    /// Calculate AUC for the first occasion
    ///
    /// Convenience for the common single-occasion case. Avoids `[0].unwrap()`.
    fn auc_first(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Result<f64, MetricsError> {
        self.auc(outeq, method, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Cmax for the first occasion
    fn cmax_first(&self, outeq: usize, blq_rule: &BLQRule) -> Result<f64, MetricsError> {
        self.cmax(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Tmax for the first occasion
    fn tmax_first(&self, outeq: usize, blq_rule: &BLQRule) -> Result<f64, MetricsError> {
        self.tmax(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Clast for the first occasion
    fn clast_first(&self, outeq: usize, blq_rule: &BLQRule) -> Result<f64, MetricsError> {
        self.clast(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Tlast for the first occasion
    fn tlast_first(&self, outeq: usize, blq_rule: &BLQRule) -> Result<f64, MetricsError> {
        self.tlast(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Calculate AUMC for the first occasion
    fn aumc_first(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Result<f64, MetricsError> {
        self.aumc(outeq, method, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Calculate partial AUC for the first occasion
    fn auc_interval_first(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Result<f64, MetricsError> {
        self.auc_interval(outeq, start, end, method, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get filtered observation profile for the first occasion
    fn filtered_observations_first(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Result<ObservationProfile, ObservationError> {
        self.filtered_observations(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(ObservationError::InsufficientData {
                n: 0,
                required: 2,
            }))
    }
}

// ============================================================================
// Occasion implementations (core logic)
// ============================================================================

impl ObservationMetrics for Occasion {
    fn auc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        vec![auc_occasion(self, outeq, method, blq_rule)]
    }

    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        vec![auc_interval_occasion(
            self, outeq, start, end, method, blq_rule,
        )]
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![cmax_occasion(self, outeq, blq_rule)]
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![tmax_occasion(self, outeq, blq_rule)]
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![clast_occasion(self, outeq, blq_rule)]
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![tlast_occasion(self, outeq, blq_rule)]
    }

    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        vec![aumc_occasion(self, outeq, method, blq_rule)]
    }

    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>> {
        vec![ObservationProfile::from_occasion(self, outeq, blq_rule)]
    }
}

// ============================================================================
// Subject implementations (iterate occasions)
// ============================================================================

impl ObservationMetrics for Subject {
    fn auc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| auc_occasion(o, outeq, method, blq_rule))
            .collect()
    }

    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| auc_interval_occasion(o, outeq, start, end, method, blq_rule))
            .collect()
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| cmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| tmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| clast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| tlast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| aumc_occasion(o, outeq, method, blq_rule))
            .collect()
    }

    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>> {
        self.occasions()
            .par_iter()
            .map(|o| ObservationProfile::from_occasion(o, outeq, blq_rule))
            .collect()
    }
}

// ============================================================================
// Data implementations (iterate subjects, flatten)
// ============================================================================

impl ObservationMetrics for Data {
    fn auc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.auc(outeq, method, blq_rule))
            .collect()
    }

    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.auc_interval(outeq, start, end, method, blq_rule))
            .collect()
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.cmax(outeq, blq_rule))
            .collect()
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.tmax(outeq, blq_rule))
            .collect()
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.clast(outeq, blq_rule))
            .collect()
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.tlast(outeq, blq_rule))
            .collect()
    }

    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.aumc(outeq, method, blq_rule))
            .collect()
    }

    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.filtered_observations(outeq, blq_rule))
            .collect()
    }
}

// ============================================================================
// Private helper functions for Occasion-level implementations
// ============================================================================

fn auc_occasion(
    occasion: &Occasion,
    outeq: usize,
    method: &AUCMethod,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.auc_last(method))
}

fn auc_interval_occasion(
    occasion: &Occasion,
    outeq: usize,
    start: f64,
    end: f64,
    method: &AUCMethod,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.auc_interval(start, end, method))
}

fn cmax_occasion(
    occasion: &Occasion,
    outeq: usize,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.cmax())
}

fn tmax_occasion(
    occasion: &Occasion,
    outeq: usize,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.tmax())
}

fn clast_occasion(
    occasion: &Occasion,
    outeq: usize,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.clast())
}

fn tlast_occasion(
    occasion: &Occasion,
    outeq: usize,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.tlast())
}

fn aumc_occasion(
    occasion: &Occasion,
    outeq: usize,
    method: &AUCMethod,
    blq_rule: &BLQRule,
) -> Result<f64, MetricsError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.aumc_last(method))
}
