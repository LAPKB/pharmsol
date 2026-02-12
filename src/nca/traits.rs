//! Extension traits for NCA analysis on pharmsol data types
//!
//! These traits add NCA functionality to [`Data`], [`Subject`], and [`Occasion`]
//! without creating a dependency from `data` → `nca`. Import them via the prelude:
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//!
//! let results = subject.nca(&NCAOptions::default(), 0);
//! ```

use crate::data::event::{AUCMethod, BLQRule};
use crate::data::observation::ObservationProfile;
use crate::data::observation_error::ObservationError;
use crate::nca::analyze::analyze;
use crate::nca::calc::tlag_from_raw;
use crate::nca::error::NCAError;
use crate::nca::types::{DoseContext, NCAOptions, NCAResult};
use crate::{Data, Occasion, Subject};
use rayon::prelude::*;

// ============================================================================
// Trait 1: Full NCA analysis
// ============================================================================

/// Extension trait for Non-Compartmental Analysis
///
/// Provides the `.nca()` method on [`Data`], [`Subject`], and [`Occasion`].
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::prelude::*;
/// use pharmsol::nca::NCAOptions;
///
/// let subject = Subject::builder("patient_001")
///     .bolus(0.0, 100.0, 0)
///     .observation(1.0, 10.0, 0)
///     .observation(2.0, 8.0, 0)
///     .observation(4.0, 4.0, 0)
///     .build();
///
/// let results = subject.nca(&NCAOptions::default(), 0);
/// if let Ok(res) = &results[0] {
///     println!("Cmax: {:.2}", res.exposure.cmax);
/// }
/// ```
pub trait NCA {
    /// Perform Non-Compartmental Analysis
    ///
    /// # Arguments
    ///
    /// * `options` - NCA calculation options
    /// * `outeq` - Output equation index to analyze (0-indexed)
    ///
    /// # Returns
    ///
    /// Vector of `Result<NCAResult, NCAError>` for each occasion
    fn nca(&self, options: &NCAOptions, outeq: usize) -> Vec<Result<NCAResult, NCAError>>;

    /// Perform NCA on the first occasion and return a single result
    ///
    /// Convenience method that avoids the `Vec<Result<...>>` pattern when
    /// you only have one occasion (the common case).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use pharmsol::prelude::*;
    /// use pharmsol::nca::NCAOptions;
    ///
    /// let result = subject.nca_first(&NCAOptions::default(), 0)?;
    /// println!("Cmax: {:.2}", result.exposure.cmax);
    /// ```
    fn nca_first(&self, options: &NCAOptions, outeq: usize) -> Result<NCAResult, NCAError> {
        self.nca(options, outeq)
            .into_iter()
            .next()
            .unwrap_or(Err(NCAError::InvalidParameter {
                param: "occasion".to_string(),
                value: "none found".to_string(),
            }))
    }
}

impl NCA for Occasion {
    fn nca(&self, options: &NCAOptions, outeq: usize) -> Vec<Result<NCAResult, NCAError>> {
        vec![nca_occasion(self, options, outeq)]
    }
}

impl NCA for Subject {
    fn nca(&self, options: &NCAOptions, outeq: usize) -> Vec<Result<NCAResult, NCAError>> {
        self.occasions()
            .iter()
            .map(|occasion| {
                let mut result = nca_occasion(occasion, options, outeq)?;
                result.subject_id = Some(self.id().to_string());
                Ok(result)
            })
            .collect()
    }
}

impl NCA for Data {
    fn nca(&self, options: &NCAOptions, outeq: usize) -> Vec<Result<NCAResult, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|subject| subject.nca(options, outeq))
            .collect()
    }
}

/// Core NCA implementation for a single occasion
fn nca_occasion(
    occasion: &Occasion,
    options: &NCAOptions,
    outeq: usize,
) -> Result<NCAResult, NCAError> {
    // Build profile directly from the occasion
    let profile = ObservationProfile::from_occasion(occasion, outeq, &options.blq_rule)?;

    // Compute tlag from raw (unfiltered) data to match PKNCA
    let (times, concs, censoring) = occasion.get_observations(outeq);
    let raw_tlag = tlag_from_raw(&times, &concs, &censoring);

    // Build dose context from introspection methods
    let dose = dose_info(occasion);

    // Calculate NCA directly on the profile
    let mut result = analyze(&profile, dose.as_ref(), options, raw_tlag)?;
    result.occasion = Some(occasion.index());

    Ok(result)
}

/// Build dose context from an occasion's dose events
///
/// Returns `Some(DoseContext)` if the occasion contains dose events,
/// or `None` if there are no doses.
fn dose_info(occasion: &Occasion) -> Option<DoseContext> {
    if occasion.total_dose() > 0.0 {
        Some(DoseContext {
            amount: occasion.total_dose(),
            duration: occasion.infusion_duration(),
            route: occasion.route(),
        })
    } else {
        None
    }
}

// ============================================================================
// Trait 2: Observation metric convenience methods
// ============================================================================

/// Extension trait for observation-level pharmacokinetic metrics
///
/// Provides convenient access to AUC, Cmax, Tmax, etc. without running
/// full NCA analysis. Each method returns one result per occasion.
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
/// let auc = subject.auc(0, &AUCMethod::Linear, &BLQRule::Exclude);
/// let cmax = subject.cmax(0, &BLQRule::Exclude);
/// ```
pub trait ObservationMetrics {
    /// Calculate AUC from time 0 to Tlast
    fn auc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>>;

    /// Calculate partial AUC over a time interval
    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>>;

    /// Get Cmax (maximum concentration)
    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>>;

    /// Get Tmax (time of maximum concentration)
    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>>;

    /// Get Clast (last quantifiable concentration)
    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>>;

    /// Get Tlast (time of last quantifiable concentration)
    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>>;

    /// Calculate AUMC (Area Under the first Moment Curve)
    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>>;

    /// Get filtered observation profiles
    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>>;
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
    ) -> Vec<Result<f64, NCAError>> {
        vec![auc_occasion(self, outeq, method, blq_rule)]
    }

    fn auc_interval(
        &self,
        outeq: usize,
        start: f64,
        end: f64,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>> {
        vec![auc_interval_occasion(
            self, outeq, start, end, method, blq_rule,
        )]
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        vec![cmax_occasion(self, outeq, blq_rule)]
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        vec![tmax_occasion(self, outeq, blq_rule)]
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        vec![clast_occasion(self, outeq, blq_rule)]
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        vec![tlast_occasion(self, outeq, blq_rule)]
    }

    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>> {
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
    ) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
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
    ) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| auc_interval_occasion(o, outeq, start, end, method, blq_rule))
            .collect()
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| cmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| tmax_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| clast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| tlast_occasion(o, outeq, blq_rule))
            .collect()
    }

    fn aumc(
        &self,
        outeq: usize,
        method: &AUCMethod,
        blq_rule: &BLQRule,
    ) -> Vec<Result<f64, NCAError>> {
        self.occasions()
            .iter()
            .map(|o| aumc_occasion(o, outeq, method, blq_rule))
            .collect()
    }

    fn filtered_observations(
        &self,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Vec<Result<ObservationProfile, ObservationError>> {
        self.occasions()
            .iter()
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
    ) -> Vec<Result<f64, NCAError>> {
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
    ) -> Vec<Result<f64, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.auc_interval(outeq, start, end, method, blq_rule))
            .collect()
    }

    fn cmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.cmax(outeq, blq_rule))
            .collect()
    }

    fn tmax(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.tmax(outeq, blq_rule))
            .collect()
    }

    fn clast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|s| s.clast(outeq, blq_rule))
            .collect()
    }

    fn tlast(&self, outeq: usize, blq_rule: &BLQRule) -> Vec<Result<f64, NCAError>> {
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
    ) -> Vec<Result<f64, NCAError>> {
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
) -> Result<f64, NCAError> {
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
) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.auc_interval(start, end, method))
}

fn cmax_occasion(occasion: &Occasion, outeq: usize, blq_rule: &BLQRule) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.cmax())
}

fn tmax_occasion(occasion: &Occasion, outeq: usize, blq_rule: &BLQRule) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.tmax())
}

fn clast_occasion(occasion: &Occasion, outeq: usize, blq_rule: &BLQRule) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.clast())
}

fn tlast_occasion(occasion: &Occasion, outeq: usize, blq_rule: &BLQRule) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.tlast())
}

fn aumc_occasion(
    occasion: &Occasion,
    outeq: usize,
    method: &AUCMethod,
    blq_rule: &BLQRule,
) -> Result<f64, NCAError> {
    let profile = ObservationProfile::from_occasion(occasion, outeq, blq_rule)?;
    Ok(profile.aumc_last(method))
}
