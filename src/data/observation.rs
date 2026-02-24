//! Observation profile: filtered, validated concentration-time data
//!
//! [`ObservationProfile`] is a **crate-internal** value object. It is the single source
//! of truth for working with concentration-time profiles inside the NCA pipeline.
//!
//! External callers should use the `nca_with_dose` methods on [`crate::Subject`],
//! [`crate::Occasion`], and [`crate::Data`] instead of constructing profiles directly.
//!
//! Internally it owns:
//!
//! - **Struct + construction**: BLQ filtering, validation, index caching
//! - **Basic accessors**: Cmax, Tmax, Cmin, Clast, Tlast
//! - **AUC methods**: delegate to [`crate::data::auc`] primitives

use crate::data::auc;
use crate::data::event::{AUCMethod, BLQRule, Censor};
use crate::Occasion;

// ============================================================================
// Types
// ============================================================================

/// Action to take for a BLQ observation based on position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlqAction {
    Keep,
    Drop,
}

/// A filtered, validated view of observations ready for NCA analysis.
///
/// This is a **crate-internal** type. External callers should use
/// `Subject::nca_with_dose`, `Occasion::nca_with_dose`, or the [`crate::nca::NCA`]
/// trait rather than constructing profiles directly.
///
/// Contains time-concentration data after BLQ filtering, with cached
/// indices for Cmax, Cmin, and Tlast for efficient access.
#[derive(Debug, Clone)]
pub(crate) struct ObservationProfile {
    /// Time points (sorted, ascending)
    pub(crate) times: Vec<f64>,
    /// Concentration values (parallel to times)
    pub(crate) concentrations: Vec<f64>,
    /// Index of Cmax in the arrays
    pub(crate) cmax_idx: usize,
    /// Index of Cmin in the arrays
    pub(crate) cmin_idx: usize,
    /// Index of Clast (last positive concentration)
    pub(crate) tlast_idx: usize,
}

// ============================================================================
// Error type
// ============================================================================

use crate::data::observation_error::ObservationError;

// ============================================================================
// Accessors
// ============================================================================

impl ObservationProfile {
    /// Get Cmax value
    #[inline]
    pub fn cmax(&self) -> f64 {
        self.concentrations[self.cmax_idx]
    }

    /// Get Tmax value
    #[inline]
    pub fn tmax(&self) -> f64 {
        self.times[self.cmax_idx]
    }

    /// Get Cmin value (minimum concentration)
    #[inline]
    pub fn cmin(&self) -> f64 {
        self.concentrations[self.cmin_idx]
    }

    /// Get Clast value (last positive concentration)
    #[inline]
    pub fn clast(&self) -> f64 {
        self.concentrations[self.tlast_idx]
    }

    /// Get Tlast value (time of last positive concentration)
    #[inline]
    pub fn tlast(&self) -> f64 {
        self.times[self.tlast_idx]
    }

    /// Number of data points
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether the profile has no data points
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

impl std::fmt::Display for ObservationProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ObservationProfile ({} points)", self.len())?;
        writeln!(f, "  Cmax: {:.4} at t={:.2}", self.cmax(), self.tmax())?;
        writeln!(f, "  Cmin: {:.4}", self.cmin())?;
        writeln!(f, "  Clast: {:.4} at t={:.2}", self.clast(), self.tlast())?;
        writeln!(
            f,
            "  Time range: [{:.2}, {:.2}]",
            self.times.first().copied().unwrap_or(0.0),
            self.times.last().copied().unwrap_or(0.0)
        )?;
        Ok(())
    }
}

// ============================================================================
// Construction
// ============================================================================

impl ObservationProfile {
    /// Create a profile from an [`Occasion`]
    ///
    /// Extracts observations for the given `outeq`, applies BLQ filtering,
    /// and validates the result.
    ///
    /// # Arguments
    /// * `occasion` - The occasion containing events
    /// * `outeq` - Output equation index to extract
    /// * `blq_rule` - How to handle BLQ observations
    ///
    /// # Errors
    /// Returns error if data is insufficient or invalid
    pub(crate) fn from_occasion(
        occasion: &Occasion,
        outeq: usize,
        blq_rule: &BLQRule,
    ) -> Result<Self, ObservationError> {
        let (times, concs, censoring) = occasion.get_observations(outeq);
        Self::from_arrays(&times, &concs, &censoring, blq_rule.clone())
    }

    /// Build a profile from raw arrays with BLQ filtering
    ///
    /// This is the core construction logic. It validates inputs, applies BLQ rules,
    /// and produces a finalized profile.
    ///
    /// # Arguments
    /// * `times` - Sorted time points
    /// * `concentrations` - Concentration values (parallel to `times`)
    /// * `censoring` - Censoring flags (parallel to `times`)
    /// * `blq_rule` - How to handle BLQ observations
    ///
    /// # Errors
    /// Returns error if arrays mismatch, data is insufficient, or all values are BLQ
    fn from_arrays(
        times: &[f64],
        concentrations: &[f64],
        censoring: &[Censor],
        blq_rule: BLQRule,
    ) -> Result<Self, ObservationError> {
        if times.len() != concentrations.len() || times.len() != censoring.len() {
            return Err(ObservationError::ArrayLengthMismatch {
                description: format!(
                    "times={}, concentrations={}, censoring={}",
                    times.len(),
                    concentrations.len(),
                    censoring.len()
                ),
            });
        }

        if times.is_empty() {
            return Err(ObservationError::InsufficientData { n: 0, required: 2 });
        }

        // Check time sequence is valid
        for i in 1..times.len() {
            if times[i] < times[i - 1] {
                return Err(ObservationError::InvalidTimeSequence);
            }
        }

        // For Positional rule, we need tfirst and tlast first
        // For TmaxRelative, we need tmax
        let (tfirst_idx, tlast_idx) = if matches!(blq_rule, BLQRule::Positional) {
            find_tfirst_tlast(concentrations, censoring)
        } else {
            (None, None)
        };

        let tmax_idx = if matches!(blq_rule, BLQRule::TmaxRelative { .. }) {
            find_tmax_idx(concentrations, censoring)
        } else {
            None
        };

        let mut proc_times = Vec::with_capacity(times.len());
        let mut proc_concs = Vec::with_capacity(concentrations.len());

        for i in 0..times.len() {
            let time = times[i];
            let conc = concentrations[i];
            let censor = censoring[i];

            let is_blq = matches!(censor, Censor::BLOQ);

            if is_blq {
                match blq_rule {
                    BLQRule::Zero => {
                        proc_times.push(time);
                        proc_concs.push(0.0);
                    }
                    BLQRule::LoqOver2 => {
                        proc_times.push(time);
                        proc_concs.push(conc / 2.0);
                    }
                    BLQRule::Exclude => {
                        // Skip
                    }
                    BLQRule::Positional => {
                        let action = get_positional_action(i, tfirst_idx, tlast_idx);
                        match action {
                            BlqAction::Keep => {
                                proc_times.push(time);
                                proc_concs.push(0.0);
                            }
                            BlqAction::Drop => {
                                // Skip middle BLQ points
                            }
                        }
                    }
                    BLQRule::TmaxRelative {
                        before_tmax_keep,
                        after_tmax_keep,
                    } => {
                        let is_before_tmax = tmax_idx.map(|t| i < t).unwrap_or(true);
                        let keep = if is_before_tmax {
                            before_tmax_keep
                        } else {
                            after_tmax_keep
                        };
                        if keep {
                            proc_times.push(time);
                            proc_concs.push(0.0);
                        }
                    }
                }
            } else {
                proc_times.push(time);
                proc_concs.push(conc);
            }
        }

        finalize(proc_times, proc_concs)
    }
}

// ============================================================================
// AUC methods
// ============================================================================

impl ObservationProfile {
    /// Calculate AUC from time 0 to Tlast
    ///
    /// Delegates to [`crate::data::auc::auc`] over `times[..=tlast_idx]`.
    pub fn auc_last(&self, method: &AUCMethod) -> f64 {
        let end = self.tlast_idx + 1;
        auc::auc(&self.times[..end], &self.concentrations[..end], method)
    }

    /// Calculate AUC over a specific time interval
    ///
    /// Delegates to [`crate::data::auc::auc_interval`].
    pub fn auc_interval(&self, start: f64, end: f64, method: &AUCMethod) -> f64 {
        auc::auc_interval(&self.times, &self.concentrations, start, end, method)
    }

    /// Calculate AUMC from time 0 to Tlast
    ///
    /// Delegates to [`crate::data::auc::aumc`] over `times[..=tlast_idx]`.
    pub fn aumc_last(&self, method: &AUCMethod) -> f64 {
        let end = self.tlast_idx + 1;
        auc::aumc(&self.times[..end], &self.concentrations[..end], method)
    }

    /// Linear interpolation of concentration at a given time
    ///
    /// Delegates to [`crate::data::auc::interpolate_linear`].
    pub fn interpolate(&self, time: f64) -> f64 {
        auc::interpolate_linear(&self.times, &self.concentrations, time)
    }
}

// ============================================================================
// Helper functions (private)
// ============================================================================

/// Find tfirst and tlast indices for positional BLQ handling
fn find_tfirst_tlast(
    concentrations: &[f64],
    censoring: &[Censor],
) -> (Option<usize>, Option<usize>) {
    let mut tfirst_idx = None;
    let mut tlast_idx = None;

    for i in 0..concentrations.len() {
        let is_blq = matches!(censoring[i], Censor::BLOQ);
        if !is_blq && concentrations[i] > 0.0 {
            if tfirst_idx.is_none() {
                tfirst_idx = Some(i);
            }
            tlast_idx = Some(i);
        }
    }

    (tfirst_idx, tlast_idx)
}

/// Find index of Tmax (first maximum concentration) among non-BLQ points
fn find_tmax_idx(concentrations: &[f64], censoring: &[Censor]) -> Option<usize> {
    let mut max_conc = f64::NEG_INFINITY;
    let mut tmax_idx = None;

    for i in 0..concentrations.len() {
        let is_blq = matches!(censoring[i], Censor::BLOQ);
        if !is_blq && concentrations[i] > max_conc {
            max_conc = concentrations[i];
            tmax_idx = Some(i);
        }
    }

    tmax_idx
}

/// Determine action for a BLQ observation based on its position
fn get_positional_action(
    idx: usize,
    tfirst_idx: Option<usize>,
    tlast_idx: Option<usize>,
) -> BlqAction {
    match (tfirst_idx, tlast_idx) {
        (Some(tfirst), Some(tlast)) => {
            if idx <= tfirst {
                BlqAction::Keep
            } else if idx >= tlast {
                BlqAction::Keep
            } else {
                BlqAction::Drop
            }
        }
        _ => BlqAction::Keep,
    }
}

/// Finalize profile construction by finding Cmax/Cmin/Tlast indices
fn finalize(
    proc_times: Vec<f64>,
    proc_concs: Vec<f64>,
) -> Result<ObservationProfile, ObservationError> {
    if proc_times.len() < 2 {
        return Err(ObservationError::InsufficientData {
            n: proc_times.len(),
            required: 2,
        });
    }

    // Check if all values are zero
    if proc_concs.iter().all(|&c| c <= 0.0) {
        return Err(ObservationError::AllBelowLOQ);
    }

    // Find Cmax index (first occurrence in case of ties, matching PKNCA)
    let cmax_idx = proc_concs
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(max_i, max_c), (i, &c)| {
            if c > max_c {
                (i, c)
            } else {
                (max_i, max_c)
            }
        })
        .0;

    // Find Cmin index (first occurrence of minimum)
    let cmin_idx = proc_concs
        .iter()
        .enumerate()
        .fold((0, f64::INFINITY), |(min_i, min_c), (i, &c)| {
            if c < min_c {
                (i, c)
            } else {
                (min_i, min_c)
            }
        })
        .0;

    // Find Tlast index (last positive concentration)
    let tlast_idx = proc_concs
        .iter()
        .rposition(|&c| c > 0.0)
        .unwrap_or(proc_concs.len() - 1);

    Ok(ObservationProfile {
        times: proc_times,
        concentrations: proc_concs,
        cmax_idx,
        cmin_idx,
        tlast_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builder::SubjectBuilderExt;
    use crate::Subject;

    #[test]
    fn test_from_occasion() {
        let subject = Subject::builder("pt1")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .build();

        let occasion = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occasion, 0, &BLQRule::Exclude).unwrap();

        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.cmax(), 10.0);
        assert_eq!(profile.tmax(), 1.0);
        assert_eq!(profile.clast(), 2.0);
        assert_eq!(profile.tlast(), 8.0);
    }

    #[test]
    fn test_from_raw() {
        let subject = Subject::builder("pt1")
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        assert_eq!(profile.cmax(), 10.0);
        assert_eq!(profile.tmax(), 1.0);
        assert_eq!(profile.cmin(), 0.0);
        assert_eq!(profile.clast(), 2.0);
        assert_eq!(profile.tlast(), 8.0);
    }

    #[test]
    fn test_from_raw_insufficient() {
        // One non-BLQ point → InsufficientData after finalize
        let result =
            ObservationProfile::from_arrays(&[0.0], &[10.0], &[Censor::None], BLQRule::Exclude);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_raw_all_zero() {
        // All non-BLQ concentrations are 0 → AllBelowLOQ
        let result = ObservationProfile::from_arrays(
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[Censor::None, Censor::None],
            BLQRule::Exclude,
        );
        assert!(matches!(result, Err(ObservationError::AllBelowLOQ)));
    }

    #[test]
    fn test_from_raw_bad_time_sequence() {
        let result = ObservationProfile::from_arrays(
            &[2.0, 1.0],
            &[10.0, 5.0],
            &[Censor::None, Censor::None],
            BLQRule::Exclude,
        );
        assert!(matches!(result, Err(ObservationError::InvalidTimeSequence)));
    }

    #[test]
    fn test_cmin() {
        let subject = Subject::builder("pt1")
            .observation(0.0, 2.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 1.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();
        assert_eq!(profile.cmin(), 1.0);
    }

    #[test]
    fn test_blq_handling() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.1, 10.0, 8.0, 4.0, 0.1];
        let censoring = vec![
            Censor::BLOQ,
            Censor::None,
            Censor::None,
            Censor::None,
            Censor::BLOQ,
        ];

        let profile =
            ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude).unwrap();
        assert_eq!(profile.times.len(), 3);

        let profile =
            ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::Zero).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.0);
        assert_eq!(profile.concentrations[4], 0.0);

        let profile =
            ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::LoqOver2).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.05);
        assert_eq!(profile.concentrations[4], 0.05);
    }

    #[test]
    fn test_insufficient_data() {
        let times = vec![0.0];
        let concs = vec![10.0];
        let censoring = vec![Censor::None];

        let result = ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_blq() {
        let times = vec![0.0, 1.0, 2.0];
        let concs = vec![0.1, 0.1, 0.1];
        let censoring = vec![Censor::BLOQ, Censor::BLOQ, Censor::BLOQ];

        let result = ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude);
        assert!(matches!(
            result,
            Err(ObservationError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_positional_blq() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.1, 10.0, 0.1, 4.0, 2.0, 0.1];
        let censoring = vec![
            Censor::BLOQ,
            Censor::None,
            Censor::BLOQ,
            Censor::None,
            Censor::None,
            Censor::BLOQ,
        ];

        let profile =
            ObservationProfile::from_arrays(&times, &concs, &censoring, BLQRule::Positional)
                .unwrap();

        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.times[0], 0.0);
        assert_eq!(profile.times[1], 1.0);
        assert_eq!(profile.times[2], 4.0);
        assert_eq!(profile.times[3], 8.0);
        assert_eq!(profile.times[4], 12.0);
        assert_eq!(profile.concentrations[0], 0.0);
        assert_eq!(profile.concentrations[4], 0.0);
    }

    #[test]
    fn test_auc_last_method() {
        let subject = Subject::builder("pt1")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .build();

        let occasion = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occasion, 0, &BLQRule::Exclude).unwrap();

        let auc_val = profile.auc_last(&AUCMethod::Linear);
        assert!((auc_val - 44.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_last_delegates_to_data_auc() {
        // Verify ObservationProfile.auc_last matches data::auc::auc directly
        let subject = Subject::builder("pt1")
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();
        let method = AUCMethod::Linear;

        let profile_auc = profile.auc_last(&method);
        let direct_auc = auc::auc(&profile.times, &profile.concentrations, &method);

        assert!((profile_auc - direct_auc).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_delegates() {
        let subject = Subject::builder("pt1")
            .observation(0.0, 0.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 6.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        assert!((profile.interpolate(1.0) - 5.0).abs() < 1e-10);
        assert!((profile.interpolate(3.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let subject = Subject::builder("pt1")
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        let display = format!("{}", profile);
        assert!(display.contains("ObservationProfile (5 points)"));
        assert!(display.contains("Cmax"));
        assert!(display.contains("Cmin"));
        assert!(display.contains("Clast"));
    }
}
