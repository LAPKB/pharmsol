//! Extension traits for NCA analysis on pharmsol data types
//!
//! The [`NCA`] trait adds full non-compartmental analysis to [`Data`], [`Subject`],
//! and [`Occasion`] without creating a dependency from `data` → `nca`.
//!
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//!
//! let result = subject.nca(&NCAOptions::default())?;
//! ```

use crate::data::observation::ObservationProfile;
use crate::nca::analyze::analyze;
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

impl ObservationProfile {
    /// Run NCA directly on an observation profile with explicit dose information.
    ///
    /// This is the entry point for simulated or predicted data where there is
    /// no `Subject` or `Occasion` to attach to.
    ///
    /// # Arguments
    /// * `dose_amount` - Total dose amount (None = no dose-normalized params)
    /// * `route` - Administration route
    /// * `infusion_duration` - Duration of infusion (for IV infusion route)
    /// * `options` - NCA options (outeq is ignored; the profile is already filtered)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use pharmsol::data::observation::ObservationProfile;
    /// use pharmsol::nca::NCAOptions;
    /// use pharmsol::data::Route;
    ///
    /// let profile = ObservationProfile::from_raw(
    ///     &[0.0, 1.0, 2.0, 4.0, 8.0],
    ///     &[0.0, 10.0, 8.0, 4.0, 1.0],
    /// );
    /// let result = profile.nca_with_dose(Some(100.0), Route::Extravascular, None, &NCAOptions::default())?;
    /// println!("Cmax: {:.2}", result.exposure.cmax);
    /// ```
    pub fn nca_with_dose(
        &self,
        dose_amount: Option<f64>,
        route: Route,
        infusion_duration: Option<f64>,
        options: &NCAOptions,
    ) -> Result<NCAResult, NCAError> {
        analyze(
            self,
            dose_amount,
            route,
            infusion_duration,
            options,
            None,
            None,
            None,
        )
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
    let mut result = analyze(
        &profile,
        dose_amount,
        route,
        infusion_duration,
        options,
        raw_tlag,
        subject_id,
        Some(occasion.index()),
    )?;

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
