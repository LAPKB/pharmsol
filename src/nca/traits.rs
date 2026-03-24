//! Extension traits for NCA analysis and observation-level pharmacokinetic metrics
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//!
//! // Full NCA
//! let result = subject.nca(&NCAOptions::default())?;
//!
//! // Quick metrics
//! let auc = subject.auc(0, &AUCMethod::Linear);
//! let cmax = subject.cmax(0);
//! ```

use super::observation::ObservationProfile;
use crate::data::event::{AUCMethod, BLQRule};
use crate::data::observation_error::ObservationError;
use crate::nca::analyze::{analyze, AnalysisContext};
use crate::nca::calc::tlag_from_raw;
use crate::nca::error::NCAError;
use crate::nca::types::{NCAOptions, NCAResult, Warning};
use crate::{Data, Occasion, Subject};
use rayon::prelude::*;

/// Structured NCA result for a single subject
///
/// Groups occasion-level results under a subject identifier,
/// making it easy to associate results back to subjects.
#[derive(Debug, Clone)]
pub struct SubjectNCAResult {
    /// Subject identifier
    pub subject_id: String,
    /// NCA results for each occasion
    pub occasions: Vec<Result<NCAResult, NCAError>>,
}

impl SubjectNCAResult {
    /// Collect all successful NCA results across occasions
    pub fn successes(&self) -> Vec<&NCAResult> {
        self.occasions
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .collect()
    }

    /// Collect all errors across occasions
    pub fn errors(&self) -> Vec<&NCAError> {
        self.occasions
            .iter()
            .filter_map(|r| r.as_ref().err())
            .collect()
    }
}

// ============================================================================
// Trait: Full NCA analysis
// ============================================================================

/// Extension trait for Non-Compartmental Analysis
///
/// Provides `.nca()` (first occasion) and `.nca_all()` (all occasions)
/// on [`Data`], [`Subject`], and [`Occasion`].
///
/// The output equation is controlled by [`NCAOptions::outeq`] (default 0).
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
/// // Single-occasion (the common case)
/// let result = subject.nca(&NCAOptions::default())?;
/// println!("Cmax: {:.2}", result.exposure.cmax);
///
/// // All occasions
/// let all = subject.nca_all(&NCAOptions::default());
/// ```
pub trait NCA {
    /// NCA on the first occasion (the common case). Returns a single result.
    fn nca(&self, options: &NCAOptions) -> Result<NCAResult, NCAError>;

    /// NCA on all occasions. Returns a Vec of results.
    fn nca_all(&self, options: &NCAOptions) -> Vec<Result<NCAResult, NCAError>>;
}

/// Extension trait for structured population-level NCA
///
/// Returns results grouped by subject, making it easy to associate
/// NCA results back to their source subjects.
pub trait NCAPopulation {
    /// Perform NCA and return results grouped by subject
    ///
    /// Unlike [`NCA::nca_all`] which returns a flat `Vec`, this returns
    /// a `Vec<SubjectNCAResult>` where each entry groups all occasion
    /// results for a single subject.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use pharmsol::prelude::*;
    /// use pharmsol::nca::{NCAOptions, NCAPopulation};
    ///
    /// let population_results = data.nca_grouped(&NCAOptions::default());
    /// for subject_result in &population_results {
    ///     println!("Subject {}: {} occasions", subject_result.subject_id, subject_result.occasions.len());
    /// }
    /// ```
    fn nca_grouped(&self, options: &NCAOptions) -> Vec<SubjectNCAResult>;
}

// ============================================================================
// NCA on ObservationProfile (simulated / raw data)
// ============================================================================

use crate::data::Route;

impl Occasion {
    /// Run NCA with an explicit dose, overriding what is embedded in the occasion.
    ///
    /// Use this when you want to supply or override dose amount, route, or infusion
    /// duration without modifying the underlying data.
    pub fn nca_with_dose(
        &self,
        dose_amount: f64,
        route: Route,
        infusion_duration: Option<f64>,
        options: &NCAOptions,
    ) -> Result<NCAResult, NCAError> {
        let profile = ObservationProfile::from_occasion(self, options.outeq, &options.blq_rule)?;
        let (times, concs, censoring) = self.get_observations(options.outeq);
        let raw_tlag = tlag_from_raw(&times, &concs, &censoring);
        analyze(&AnalysisContext {
            profile: &profile,
            dose_amount: Some(dose_amount),
            route,
            infusion_duration,
            options,
            raw_tlag,
            subject_id: None,
            occasion: Some(self.index()),
        })
    }
}

impl NCA for Occasion {
    fn nca(&self, options: &NCAOptions) -> Result<NCAResult, NCAError> {
        nca_occasion(self, options, None)
    }

    fn nca_all(&self, options: &NCAOptions) -> Vec<Result<NCAResult, NCAError>> {
        vec![self.nca(options)]
    }
}

impl Subject {
    /// Run NCA with an explicit dose on the first occasion, overriding what is
    /// embedded in the subject's events.
    ///
    /// Use this when you want to supply or override dose amount, route, or infusion
    /// duration without modifying the underlying data.
    pub fn nca_with_dose(
        &self,
        dose_amount: f64,
        route: Route,
        infusion_duration: Option<f64>,
        options: &NCAOptions,
    ) -> Result<NCAResult, NCAError> {
        let occasion = self
            .occasions()
            .iter()
            .next()
            .ok_or(NCAError::InvalidParameter {
                param: "occasion".to_string(),
                value: "none found".to_string(),
            })?;
        occasion.nca_with_dose(dose_amount, route, infusion_duration, options)
    }
}

impl NCA for Subject {
    fn nca(&self, options: &NCAOptions) -> Result<NCAResult, NCAError> {
        self.occasions()
            .first()
            .map(|occ| nca_occasion(occ, options, Some(self.id())))
            .unwrap_or(Err(NCAError::InvalidParameter {
                param: "occasion".to_string(),
                value: "none found".to_string(),
            }))
    }

    fn nca_all(&self, options: &NCAOptions) -> Vec<Result<NCAResult, NCAError>> {
        self.occasions()
            .par_iter()
            .map(|occasion| nca_occasion(occasion, options, Some(self.id())))
            .collect()
    }
}

impl NCA for Data {
    fn nca(&self, options: &NCAOptions) -> Result<NCAResult, NCAError> {
        self.subjects()
            .first()
            .map(|s| s.nca(options))
            .unwrap_or(Err(NCAError::InvalidParameter {
                param: "subject".to_string(),
                value: "none found".to_string(),
            }))
    }

    fn nca_all(&self, options: &NCAOptions) -> Vec<Result<NCAResult, NCAError>> {
        self.subjects()
            .par_iter()
            .flat_map(|subject| subject.nca_all(options))
            .collect()
    }
}

impl NCAPopulation for Data {
    fn nca_grouped(&self, options: &NCAOptions) -> Vec<SubjectNCAResult> {
        self.subjects()
            .par_iter()
            .map(|subject| {
                let occasions = subject
                    .occasions()
                    .par_iter()
                    .map(|occasion| nca_occasion(occasion, options, Some(subject.id())))
                    .collect();
                SubjectNCAResult {
                    subject_id: subject.id().to_string(),
                    occasions,
                }
            })
            .collect()
    }
}

/// Core NCA implementation for a single occasion
fn nca_occasion(
    occasion: &Occasion,
    options: &NCAOptions,
    subject_id: Option<&str>,
) -> Result<NCAResult, NCAError> {
    let outeq = options.outeq;

    // Build profile directly from the occasion
    let profile = ObservationProfile::from_occasion(occasion, outeq, &options.blq_rule)?;

    // Compute tlag from raw (unfiltered) data to match PKNCA
    let (times, concs, censoring) = occasion.get_observations(outeq);
    let raw_tlag = tlag_from_raw(&times, &concs, &censoring);

    // Extract dose info from Occasion directly (no DoseContext)
    let dose_amount = {
        let d = occasion.total_dose();
        if d > 0.0 {
            Some(d)
        } else {
            None
        }
    };
    let route = options.route_override.unwrap_or_else(|| occasion.route());
    let infusion_duration = occasion.infusion_duration();

    // Calculate NCA directly on the profile
    let mut result = analyze(&AnalysisContext {
        profile: &profile,
        dose_amount,
        route,
        infusion_duration,
        options,
        raw_tlag,
        subject_id,
        occasion: Some(occasion.index()),
    })?;

    // Warn about mixed routes if no explicit override was given
    let routes = occasion.routes();
    if routes.len() > 1 && options.route_override.is_none() {
        result
            .quality
            .warnings
            .push(Warning::MixedRoutes { routes });
    }

    Ok(result)
}

// ============================================================================
// Observation-level metrics (AUC, Cmax, Tmax, etc.)
// ============================================================================

/// Error type for observation metric computations
///
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
    // Required methods — with explicit BLQ rule

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

    // Ergonomic defaults — BLQ observations excluded

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

    // Single-occasion convenience — no BLQ

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

    // Single-occasion convenience — with BLQ

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
// Occasion implementations (core metrics logic)
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
// Private helpers for Occasion-level metric implementations
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
