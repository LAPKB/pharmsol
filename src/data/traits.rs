//! Extension traits for observation-level pharmacokinetic metrics
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
//! let auc = subject.auc(0, &AUCMethod::Linear);
//! let cmax = subject.cmax(0);
//!
//! // With BLQ handling
//! let auc_blq = subject.auc_blq(0, &AUCMethod::Linear, &BLQRule::Zero);
//! ```

use crate::data::event::{AUCMethod, BLQRule};
use crate::data::observation_error::ObservationError;
use crate::nca::observation::ObservationProfile;
use crate::{Data, Occasion, Subject};
use rayon::prelude::*;

/// Error type for observation metric computations
///
/// Wraps [`ObservationError`] with optional context about which subject,
/// occasion, or output equation failed. This provides better error messages
/// Wraps [`ObservationError`] with context about which subject or output equation failed.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MetricsError {
    #[error(transparent)]
    Observation(#[from] ObservationError),

    #[error("Output equation {outeq} not found in subject{}", subject_id.as_ref().map(|id| format!(" '{}'", id)).unwrap_or_default())]
    OutputEquationNotFound {
        outeq: usize,
        subject_id: Option<String>,
    },
}

/// Observation-level pharmacokinetic metrics (AUC, Cmax, Tmax, etc.)
///
/// Methods without `_blq` default to [`BLQRule::Exclude`].
/// The `_first` variants return a single result for the first occasion.
///
/// ```rust,ignore
/// use pharmsol::prelude::*;
///
/// let auc = subject.auc(0, &AUCMethod::Linear);
/// let cmax_val = subject.cmax_first(0).unwrap();
/// ```
pub trait ObservationMetrics {
    // ========================================================================
    // Required methods — with explicit BLQ rule
    // ========================================================================

    /// Calculate AUC from time 0 to Tlast with explicit BLQ handling
    fn auc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    /// Calculate partial AUC over a time interval with explicit BLQ handling
    fn auc_interval_blq(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    /// Get Cmax with explicit BLQ handling
    fn cmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Tmax with explicit BLQ handling
    fn tmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Clast with explicit BLQ handling
    fn clast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Get Tlast with explicit BLQ handling
    fn tlast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>>;

    /// Calculate AUMC with explicit BLQ handling
    fn aumc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>>;

    // ========================================================================
    // Ergonomic defaults — BLQ observations excluded
    // ========================================================================

    /// Calculate AUC from time 0 to Tlast
    fn auc(&self, outeq: usize, method: &AUCMethod) -> Vec<Result<f64, MetricsError>> {
        self.auc_blq(outeq, method, &BLQRule::Exclude)
    }

    /// Calculate partial AUC over a time interval
    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
    ) -> Vec<Result<f64, MetricsError>> {
        self.auc_interval_blq(outeq, start, end, method, &BLQRule::Exclude)
    }

    /// Get Cmax
    fn cmax(&self, outeq: usize) -> Vec<Result<f64, MetricsError>> {
        self.cmax_blq(outeq, &BLQRule::Exclude)
    }

    /// Get Tmax
    fn tmax(&self, outeq: usize) -> Vec<Result<f64, MetricsError>> {
        self.tmax_blq(outeq, &BLQRule::Exclude)
    }

    /// Get Clast
    fn clast(&self, outeq: usize) -> Vec<Result<f64, MetricsError>> {
        self.clast_blq(outeq, &BLQRule::Exclude)
    }

    /// Get Tlast
    fn tlast(&self, outeq: usize) -> Vec<Result<f64, MetricsError>> {
        self.tlast_blq(outeq, &BLQRule::Exclude)
    }

    /// Calculate AUMC
    fn aumc(&self, outeq: usize, method: &AUCMethod) -> Vec<Result<f64, MetricsError>> {
        self.aumc_blq(outeq, method, &BLQRule::Exclude)
    }

    // ========================================================================
    // Single-occasion convenience — no BLQ
    // ========================================================================

    /// Calculate AUC for the first occasion
    fn auc_first(&self, outeq: usize, method: &AUCMethod) -> Result<f64, MetricsError> {
        self.auc(outeq, method)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Cmax for the first occasion
    fn cmax_first(&self, outeq: usize) -> Result<f64, MetricsError> {
        self.cmax(outeq)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Tmax for the first occasion
    fn tmax_first(&self, outeq: usize) -> Result<f64, MetricsError> {
        self.tmax(outeq)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Clast for the first occasion
    fn clast_first(&self, outeq: usize) -> Result<f64, MetricsError> {
        self.clast(outeq)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Tlast for the first occasion
    fn tlast_first(&self, outeq: usize) -> Result<f64, MetricsError> {
        self.tlast(outeq)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Calculate AUMC for the first occasion
    fn aumc_first(&self, outeq: usize, method: &AUCMethod) -> Result<f64, MetricsError> {
        self.aumc(outeq, method)
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
    ) -> Result<f64, MetricsError> {
        self.auc_interval(outeq, start, end, method)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    // ========================================================================
    // Single-occasion convenience — with BLQ
    // ========================================================================

    /// Calculate AUC for the first occasion with explicit BLQ handling
    fn auc_blq_first(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Result<f64, MetricsError> {
        self.auc_blq(outeq, method, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Get Cmax for the first occasion with explicit BLQ handling
    fn cmax_blq_first(&self, outeq: usize, blq_rule: &BLQRule) -> Result<f64, MetricsError> {
        self.cmax_blq(outeq, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }

    /// Calculate partial AUC for the first occasion with explicit BLQ handling
    fn auc_interval_blq_first(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Result<f64, MetricsError> {
        self.auc_interval_blq(outeq, start, end, method, blq_rule)
            .into_iter()
            .next()
            .unwrap_or(Err(MetricsError::Observation(
                ObservationError::InsufficientData { n: 0, required: 2 },
            )))
    }
}

// ============================================================================
// Occasion implementations (core logic)
// ============================================================================

impl ObservationMetrics for Occasion {
    fn auc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        vec![auc_occasion(self, outeq, method, blq_rule)]
    }

    fn auc_interval_blq(
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

    fn cmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![cmax_occasion(self, outeq, blq_rule)]
    }

    fn tmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![tmax_occasion(self, outeq, blq_rule)]
    }

    fn clast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![clast_occasion(self, outeq, blq_rule)]
    }

    fn tlast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        vec![tlast_occasion(self, outeq, blq_rule)]
    }

    fn aumc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        vec![aumc_occasion(self, outeq, method, blq_rule)]
    }
}

// ============================================================================
// Subject implementations (iterate occasions)
// ============================================================================

impl ObservationMetrics for Subject {
    fn auc_blq(
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

    fn auc_interval_blq(
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

    fn cmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| cmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| tmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn clast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| clast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tlast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.occasions()
            .par_iter()
            .map(|o| tlast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn aumc_blq(
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
}

// ============================================================================
// Data implementations (iterate subjects, flatten)
// ============================================================================

impl ObservationMetrics for Data {
    fn auc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.auc_blq(outeq, method, blq_rule))
            .collect()
    }

    fn auc_interval_blq(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.auc_interval_blq(outeq, start, end, method, blq_rule))
            .collect()
    }

    fn cmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.cmax_blq(outeq, blq_rule))
            .collect()
    }

    fn tmax_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.tmax_blq(outeq, blq_rule))
            .collect()
    }

    fn clast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.clast_blq(outeq, blq_rule))
            .collect()
    }

    fn tlast_blq(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.tlast_blq(outeq, blq_rule))
            .collect()
    }

    fn aumc_blq(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, MetricsError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.aumc_blq(outeq, method, blq_rule))
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
    Ok(profile.auc_last(method)?)
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
    Ok(profile.auc_interval(start, end, method)?)
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
    Ok(profile.aumc_last(method)?)
}
