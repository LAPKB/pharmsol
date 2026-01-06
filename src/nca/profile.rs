//! Internal profile representation for NCA analysis
//!
//! The Profile struct is a validated, analysis-ready concentration-time dataset.
//! It handles BLQ processing and caches key indices for efficiency.

use super::error::NCAError;
use super::types::BLQRule;

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
    /// Create a profile from raw time/concentration arrays
    ///
    /// # Arguments
    /// * `times` - Time points
    /// * `concentrations` - Concentration values
    /// * `loq` - Limit of quantification
    /// * `blq_rule` - How to handle BLQ values
    ///
    /// # Errors
    /// Returns error if data is insufficient or invalid
    pub fn from_arrays(
        times: &[f64],
        concentrations: &[f64],
        loq: f64,
        blq_rule: BLQRule,
    ) -> Result<Self, NCAError> {
        if times.len() != concentrations.len() {
            return Err(NCAError::InvalidParameter {
                param: "arrays".to_string(),
                value: format!(
                    "times length {} != concentrations length {}",
                    times.len(),
                    concentrations.len()
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

        let mut proc_times = Vec::with_capacity(times.len());
        let mut proc_concs = Vec::with_capacity(concentrations.len());

        for i in 0..times.len() {
            let time = times[i];
            let conc = concentrations[i];

            // Check if BLQ (concentration below LOQ or negative)
            let is_blq = (loq > 0.0 && conc < loq) || conc < 0.0;

            if is_blq {
                match blq_rule {
                    BLQRule::Zero => {
                        proc_times.push(time);
                        proc_concs.push(0.0);
                    }
                    BLQRule::LoqOver2 => {
                        proc_times.push(time);
                        proc_concs.push(loq / 2.0);
                    }
                    BLQRule::Exclude => {
                        // Skip this point
                    }
                }
            } else {
                proc_times.push(time);
                proc_concs.push(conc);
            }
        }

        if proc_times.len() < 2 {
            return Err(NCAError::InsufficientData {
                n: proc_times.len(),
                required: 2,
            });
        }

        // Find Cmax index
        let cmax_idx = proc_concs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

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

        let profile = Profile::from_arrays(&times, &concs, 0.0, BLQRule::Exclude).unwrap();

        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.cmax(), 10.0);
        assert_eq!(profile.tmax(), 1.0);
        assert_eq!(profile.clast(), 2.0);
        assert_eq!(profile.tlast(), 8.0);
    }

    #[test]
    fn test_profile_blq_handling() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.05, 10.0, 8.0, 4.0, 0.05];

        // Exclude BLQ
        let profile = Profile::from_arrays(&times, &concs, 0.1, BLQRule::Exclude).unwrap();
        assert_eq!(profile.times.len(), 3); // Only 3 points above LOQ

        // Zero substitution
        let profile = Profile::from_arrays(&times, &concs, 0.1, BLQRule::Zero).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.0);
        assert_eq!(profile.concentrations[4], 0.0);

        // LOQ/2 substitution
        let profile = Profile::from_arrays(&times, &concs, 0.1, BLQRule::LoqOver2).unwrap();
        assert_eq!(profile.times.len(), 5);
        assert_eq!(profile.concentrations[0], 0.05);
        assert_eq!(profile.concentrations[4], 0.05);
    }

    #[test]
    fn test_profile_insufficient_data() {
        let times = vec![0.0];
        let concs = vec![10.0];

        let result = Profile::from_arrays(&times, &concs, 0.0, BLQRule::Exclude);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_all_blq() {
        let times = vec![0.0, 1.0, 2.0];
        let concs = vec![0.05, 0.05, 0.05];

        let result = Profile::from_arrays(&times, &concs, 0.1, BLQRule::Exclude);
        assert!(matches!(result, Err(NCAError::InsufficientData { .. })));
    }
}
