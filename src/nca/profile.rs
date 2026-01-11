//! Internal profile representation for NCA analysis
//!
//! The Profile struct is a validated, analysis-ready concentration-time dataset.
//! It handles BLQ processing and caches key indices for efficiency.

use super::error::NCAError;
use super::types::BLQRule;
use crate::Censor;

/// A validated concentration-time profile ready for NCA analysis
///
/// This is an internal structure that normalizes data from various sources
/// (raw arrays, Occasion) into a consistent format with BLQ handling applied.
#[derive(Debug, Clone)]
pub(crate) struct Profile {
    /// Time points (sorted, ascending)
    pub times: Vec<f64>,
    /// Concentration values (parallel to times)
    pub concentrations: Vec<f64>,
    /// Index of Cmax in the arrays
    pub cmax_idx: usize,
    /// Index of Clast (last positive concentration)
    pub tlast_idx: usize,
}

impl Profile {
    /// Create a profile from time/concentration/censoring arrays
    ///
    /// BLQ/ALQ status is determined by the `Censor` marking:
    /// - `Censor::BLOQ`: Below limit of quantification - value is the lower limit
    /// - `Censor::ALOQ`: Above limit of quantification - value is the upper limit
    /// - `Censor::None`: Quantifiable observation - value is the measured concentration
    ///
    /// # Arguments
    /// * `times` - Time points
    /// * `concentrations` - Concentration values (for censored samples, this is the LOQ/ULQ)
    /// * `censoring` - Censoring status for each observation
    /// * `blq_rule` - How to handle BLQ values
    ///
    /// # Errors
    /// Returns error if data is insufficient or invalid
    pub fn from_arrays(
        times: &[f64],
        concentrations: &[f64],
        censoring: &[Censor],
        blq_rule: BLQRule,
    ) -> Result<Self, NCAError> {
        if times.len() != concentrations.len() || times.len() != censoring.len() {
            return Err(NCAError::InvalidParameter {
                param: "arrays".to_string(),
                value: format!(
                    "array lengths mismatch: times={}, concentrations={}, censoring={}",
                    times.len(),
                    concentrations.len(),
                    censoring.len()
                ),
            });
        }

        if times.is_empty() {
            return Err(NCAError::InsufficientData { n: 0, required: 2 });
        }

        // Check time sequence is valid
        for i in 1..times.len() {
            if times[i] < times[i - 1] {
                return Err(NCAError::InvalidTimeSequence);
            }
        }

        // For Positional rule, we need tfirst and tlast first
        // For TmaxRelative, we need tmax
        // Do a preliminary pass to find these indices
        let (tfirst_idx, tlast_idx) = if matches!(blq_rule, BLQRule::Positional) {
            Self::find_tfirst_tlast(concentrations, censoring)
        } else {
            (None, None)
        };

        let tmax_idx = if matches!(blq_rule, BLQRule::TmaxRelative { .. }) {
            Self::find_tmax_idx(concentrations, censoring)
        } else {
            None
        };

        let mut proc_times = Vec::with_capacity(times.len());
        let mut proc_concs = Vec::with_capacity(concentrations.len());

        for i in 0..times.len() {
            let time = times[i];
            let conc = concentrations[i];
            let censor = censoring[i];

            // BLQ is determined by the Censor marking
            // Note: ALOQ values are kept unchanged (follows PKNCA behavior)
            let is_blq = matches!(censor, Censor::BLOQ);

            if is_blq {
                // When censored, `conc` is the LOQ threshold
                match blq_rule {
                    BLQRule::Zero => {
                        proc_times.push(time);
                        proc_concs.push(0.0);
                    }
                    BLQRule::LoqOver2 => {
                        proc_times.push(time);
                        proc_concs.push(conc / 2.0); // conc IS the LOQ
                    }
                    BLQRule::Exclude => {
                        // Skip this point
                    }
                    BLQRule::Positional => {
                        // Position-aware handling: first=keep, middle=drop, last=keep
                        // PKNCA "keep" means keep as 0, not as LOQ
                        let action = Self::get_positional_action(i, tfirst_idx, tlast_idx);
                        match action {
                            super::types::BlqAction::Keep => {
                                // Keep as 0 (PKNCA "keep" behavior preserves the zero)
                                proc_times.push(time);
                                proc_concs.push(0.0);
                            }
                            super::types::BlqAction::SetLoqOver2 => {
                                proc_times.push(time);
                                proc_concs.push(conc / 2.0);
                            }
                            super::types::BlqAction::SetZero => {
                                proc_times.push(time);
                                proc_concs.push(0.0);
                            }
                            super::types::BlqAction::Drop => {
                                // Skip middle BLQ points
                            }
                        }
                    }
                    BLQRule::TmaxRelative {
                        before_tmax_keep,
                        after_tmax_keep,
                    } => {
                        // Tmax-relative handling
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
                        // else: drop the point
                    }
                }
            } else {
                proc_times.push(time);
                proc_concs.push(conc);
            }
        }

        Self::finalize(proc_times, proc_concs)
    }

    /// Find tfirst and tlast indices for positional BLQ handling
    ///
    /// tfirst = index of first positive (non-BLQ) concentration
    /// tlast = index of last positive (non-BLQ) concentration
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
    ///
    /// PKNCA default: first=keep, middle=drop, last=keep
    fn get_positional_action(
        idx: usize,
        tfirst_idx: Option<usize>,
        tlast_idx: Option<usize>,
    ) -> super::types::BlqAction {
        match (tfirst_idx, tlast_idx) {
            (Some(tfirst), Some(tlast)) => {
                if idx <= tfirst {
                    // First position (at or before tfirst): keep
                    super::types::BlqAction::Keep
                } else if idx >= tlast {
                    // Last position (at or after tlast): keep
                    super::types::BlqAction::Keep
                } else {
                    // Middle position: drop
                    super::types::BlqAction::Drop
                }
            }
            _ => {
                // No positive concentrations found - keep everything
                super::types::BlqAction::Keep
            }
        }
    }

    /// Finalize profile construction by finding Cmax/Tlast indices
    fn finalize(proc_times: Vec<f64>, proc_concs: Vec<f64>) -> Result<Self, NCAError> {
        if proc_times.len() < 2 {
            return Err(NCAError::InsufficientData {
                n: proc_times.len(),
                required: 2,
            });
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

        // Find Tlast index (last positive concentration)
        let tlast_idx = proc_concs
            .iter()
            .rposition(|&c| c > 0.0)
            .unwrap_or(proc_concs.len() - 1);

        // Check if all values are zero
        if proc_concs.iter().all(|&c| c <= 0.0) {
            return Err(NCAError::AllBLQ);
        }

        Ok(Self {
            times: proc_times,
            concentrations: proc_concs,
            cmax_idx,
            tlast_idx,
        })
    }

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

    /// Get Clast value
    #[inline]
    pub fn clast(&self) -> f64 {
        self.concentrations[self.tlast_idx]
    }

    /// Get Tlast value
    #[inline]
    pub fn tlast(&self) -> f64 {
        self.times[self.tlast_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_arrays() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0];
        let censoring = vec![Censor::None; 5];

        let profile = Profile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude).unwrap();

        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.cmax(), 10.0);
        assert_eq!(profile.tmax(), 1.0);
        assert_eq!(profile.clast(), 2.0);
        assert_eq!(profile.tlast(), 8.0);
    }

    #[test]
    fn test_profile_blq_handling() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        // First and last are BLOQ with LOQ = 0.1
        let concs = vec![0.1, 10.0, 8.0, 4.0, 0.1];
        let censoring = vec![
            Censor::BLOQ,
            Censor::None,
            Censor::None,
            Censor::None,
            Censor::BLOQ,
        ];

        // Exclude BLQ
        let profile = Profile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude).unwrap();
        assert_eq!(profile.times.len(), 3); // Only 3 points not BLQ

        // Zero substitution
        let profile = Profile::from_arrays(&times, &concs, &censoring, BLQRule::Zero).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.0);
        assert_eq!(profile.concentrations[4], 0.0);

        // LOQ/2 substitution (conc value IS the LOQ when censored)
        let profile = Profile::from_arrays(&times, &concs, &censoring, BLQRule::LoqOver2).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.05); // 0.1 / 2
        assert_eq!(profile.concentrations[4], 0.05);
    }

    #[test]
    fn test_profile_insufficient_data() {
        let times = vec![0.0];
        let concs = vec![10.0];
        let censoring = vec![Censor::None];

        let result = Profile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_all_blq() {
        let times = vec![0.0, 1.0, 2.0];
        let concs = vec![0.1, 0.1, 0.1]; // All are LOQ values
        let censoring = vec![Censor::BLOQ, Censor::BLOQ, Censor::BLOQ];

        let result = Profile::from_arrays(&times, &concs, &censoring, BLQRule::Exclude);
        assert!(matches!(result, Err(NCAError::InsufficientData { .. })));
    }

    #[test]
    fn test_profile_positional_blq() {
        // Profile with BLQ at first, middle, and last positions
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.1, 10.0, 0.1, 4.0, 2.0, 0.1]; // LOQ = 0.1
        let censoring = vec![
            Censor::BLOQ, // first - should keep
            Censor::None, // quantifiable
            Censor::BLOQ, // middle - should drop
            Censor::None, // quantifiable
            Censor::None, // quantifiable (tlast)
            Censor::BLOQ, // last - should keep
        ];

        // Positional BLQ handling: first=keep(0), middle=drop, last=keep(0)
        let profile =
            Profile::from_arrays(&times, &concs, &censoring, BLQRule::Positional).unwrap();

        // Should have 5 points: first BLQ (kept as 0), 3 quantifiable, last BLQ (kept as 0)
        // Middle BLQ at t=2 should be dropped
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.times[0], 0.0); // First BLQ kept
        assert_eq!(profile.times[1], 1.0); // Quantifiable
        assert_eq!(profile.times[2], 4.0); // Middle BLQ dropped, this is the next
        assert_eq!(profile.times[3], 8.0); // Quantifiable
        assert_eq!(profile.times[4], 12.0); // Last BLQ kept

        // First BLQ should be kept as 0 (PKNCA behavior, not LOQ)
        assert_eq!(profile.concentrations[0], 0.0);
        // Last BLQ should be kept as 0 (PKNCA behavior, not LOQ)
        assert_eq!(profile.concentrations[4], 0.0);
    }
}
