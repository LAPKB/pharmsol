//! Sparse PK analysis using Bailer's method
//!
//! For studies with destructive sampling (e.g., preclinical) or very sparse designs
//! (e.g., pediatric/oncology), individual subjects don't have enough samples for
//! traditional NCA. Bailer's method computes a population AUC with standard error
//! by using the trapezoidal rule on mean concentrations at each time point.
//!
//! # Usage
//!
//! The simplest way is via [`sparse_auc_from_data`] which accepts a [`Data`] object:
//!
//! ```rust,ignore
//! use pharmsol::nca::sparse::sparse_auc_from_data;
//!
//! let result = sparse_auc_from_data(&data, 0, None).unwrap();
//! println!("Population AUC: {:.2} ± {:.2}", result.auc, result.auc_se);
//! ```
//!
//! Reference: Bailer AJ. "Testing for the equality of area under the curves when
//! using destructive measurement techniques." J Pharmacokinet Biopharm. 1988;16(3):303-309.

use crate::Data;
use serde::{Deserialize, Serialize};

/// Result of sparse PK analysis using Bailer's method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePKResult {
    /// Population AUC estimate (trapezoidal on mean concentrations)
    pub auc: f64,
    /// Standard error of the AUC estimate
    pub auc_se: f64,
    /// 95% confidence interval lower bound
    pub auc_ci_lower: f64,
    /// 95% confidence interval upper bound
    pub auc_ci_upper: f64,
    /// Number of time points
    pub n_timepoints: usize,
    /// Mean concentrations at each time point
    pub mean_concentrations: Vec<f64>,
    /// Number of observations at each time point
    pub n_per_timepoint: Vec<usize>,
    /// Unique time points
    pub times: Vec<f64>,
}

/// Compute population AUC from sparse/destructive sampling using Bailer's method
///
/// Groups observations by time point, computes mean and variance at each time,
/// then applies the trapezoidal rule to the mean concentrations. The standard
/// error is computed using the variance propagation formula for the trapezoidal rule.
///
/// # Arguments
/// * `times` - Observation times (parallel with `concentrations`)
/// * `concentrations` - Observed concentrations (parallel with `times`)
/// * `time_tolerance` - Tolerance for grouping time points (default: exact matching).
///   Observations at times within this tolerance are considered the same nominal time.
///
/// # Returns
/// `None` if fewer than 2 unique time points with data
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::nca::sparse::sparse_auc;
///
/// let times = vec![0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 8.0, 8.0];
/// let concs = vec![0.0, 0.0, 10.5, 12.0, 5.0, 4.5, 1.5, 2.0];
///
/// let result = sparse_auc(&times, &concs, None).unwrap();
/// println!("Population AUC: {:.2} ± {:.2}", result.auc, result.auc_se);
/// println!("95% CI: [{:.2}, {:.2}]", result.auc_ci_lower, result.auc_ci_upper);
/// ```
pub fn sparse_auc(
    times: &[f64],
    concentrations: &[f64],
    time_tolerance: Option<f64>,
) -> Option<SparsePKResult> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    let tol = time_tolerance.unwrap_or(0.0);

    // Group observations by time point
    let mut time_groups: Vec<(f64, Vec<f64>)> = Vec::new();

    // Sort by time using indices
    let mut indices: Vec<usize> = (0..times.len()).collect();
    indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

    for &idx in &indices {
        let t = times[idx];
        let c = concentrations[idx];
        let matched = time_groups.iter_mut().find(|(gt, _)| (t - *gt).abs() <= tol);
        if let Some((_, group)) = matched {
            group.push(c);
        } else {
            time_groups.push((t, vec![c]));
        }
    }

    // Sort by time
    time_groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if time_groups.len() < 2 {
        return None;
    }

    let n_timepoints = time_groups.len();
    let group_times: Vec<f64> = time_groups.iter().map(|(t, _)| *t).collect();
    let n_per_timepoint: Vec<usize> = time_groups.iter().map(|(_, g)| g.len()).collect();

    // Compute mean and variance at each time point
    let mean_concentrations: Vec<f64> = time_groups
        .iter()
        .map(|(_, group)| {
            let n = group.len() as f64;
            group.iter().sum::<f64>() / n
        })
        .collect();

    let variances: Vec<f64> = time_groups
        .iter()
        .map(|(_, group)| {
            let n = group.len() as f64;
            if n < 2.0 {
                return 0.0; // Single observation: no variance estimate
            }
            let mean = group.iter().sum::<f64>() / n;
            group.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .collect();

    // Bailer's AUC: trapezoidal rule on mean concentrations
    let mut auc = 0.0;
    for i in 0..n_timepoints - 1 {
        let dt = group_times[i + 1] - group_times[i];
        auc += (mean_concentrations[i] + mean_concentrations[i + 1]) * dt / 2.0;
    }

    // Bailer's variance: sum of weighted variances
    let mut weights = vec![0.0; n_timepoints];
    for i in 0..n_timepoints - 1 {
        let dt = group_times[i + 1] - group_times[i];
        weights[i] += dt / 2.0;
        weights[i + 1] += dt / 2.0;
    }

    let auc_variance: f64 = (0..n_timepoints)
        .map(|j| {
            let n_j = n_per_timepoint[j] as f64;
            if n_j > 0.0 {
                weights[j].powi(2) * variances[j] / n_j
            } else {
                0.0
            }
        })
        .sum();

    let auc_se = auc_variance.sqrt();

    // 95% CI using normal approximation (z = 1.96)
    let z = 1.96;
    let auc_ci_lower = auc - z * auc_se;
    let auc_ci_upper = auc + z * auc_se;

    Some(SparsePKResult {
        auc,
        auc_se,
        auc_ci_lower,
        auc_ci_upper,
        n_timepoints,
        mean_concentrations,
        n_per_timepoint,
        times: group_times,
    })
}

/// Compute population AUC from sparse/destructive sampling using a [`Data`] dataset
///
/// Extracts all observations for the given `outeq` from every subject and occasion
/// in the dataset, then applies Bailer's method.
///
/// # Arguments
/// * `data` - Population dataset with sparsely-sampled subjects
/// * `outeq` - Output equation index to extract observations for
/// * `time_tolerance` - Tolerance for grouping time points (None = exact matching)
///
/// # Returns
/// `None` if fewer than 2 unique time points with data
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::prelude::*;
/// use pharmsol::nca::sparse::sparse_auc_from_data;
///
/// let data: Data = /* load or build population data */;
/// let result = sparse_auc_from_data(&data, 0, None).unwrap();
/// println!("Population AUC: {:.2} ± {:.2}", result.auc, result.auc_se);
/// ```
pub fn sparse_auc_from_data(
    data: &Data,
    outeq: usize,
    time_tolerance: Option<f64>,
) -> Option<SparsePKResult> {
    let (mut all_times, mut all_concs) = (Vec::new(), Vec::new());
    for subject in data.subjects() {
        for occasion in subject.occasions() {
            let (times, concs, _censoring) = occasion.get_observations(outeq);
            all_times.extend(times);
            all_concs.extend(concs);
        }
    }
    sparse_auc(&all_times, &all_concs, time_tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_auc_basic() {
        // 4 time points, 3 subjects each
        let times = vec![
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            4.0, 4.0, 4.0,
            8.0, 8.0, 8.0,
        ];
        let concs = vec![
            0.0, 0.0, 0.0,
            10.0, 12.0, 11.0,
            5.0, 4.0, 6.0,
            1.0, 1.5, 1.2,
        ];

        let result = sparse_auc(&times, &concs, None).unwrap();

        assert_eq!(result.n_timepoints, 4);
        assert!(result.auc > 0.0);
        assert!(result.auc_se >= 0.0);
        assert!(result.auc_ci_lower <= result.auc);
        assert!(result.auc_ci_upper >= result.auc);

        // Manual: means = [0, 11, 5, ~1.23]
        assert!((result.mean_concentrations[0] - 0.0).abs() < 1e-10);
        assert!((result.mean_concentrations[1] - 11.0).abs() < 1e-10);
        assert!((result.mean_concentrations[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_auc_single_timepoint() {
        let times = vec![0.0, 0.0];
        let concs = vec![10.0, 12.0];

        assert!(sparse_auc(&times, &concs, None).is_none());
    }

    #[test]
    fn test_sparse_auc_with_tolerance() {
        let times = vec![0.0, 0.01, 1.0, 0.99];
        let concs = vec![0.0, 0.0, 10.0, 12.0];

        let result = sparse_auc(&times, &concs, Some(0.05)).unwrap();
        assert_eq!(result.n_timepoints, 2); // Should have 2 groups, not 4
    }

    #[test]
    fn test_sparse_auc_empty() {
        assert!(sparse_auc(&[], &[], None).is_none());
    }

    #[test]
    fn test_sparse_auc_known_values() {
        // If all subjects have the same concentration at each time point,
        // variance = 0, SE = 0, and AUC = simple trapezoidal
        let times = vec![0.0, 0.0, 2.0, 2.0];
        let concs = vec![10.0, 10.0, 5.0, 5.0];

        let result = sparse_auc(&times, &concs, None).unwrap();

        // AUC = (10 + 5) / 2 * 2 = 15
        assert!((result.auc - 15.0).abs() < 1e-10);
        assert!((result.auc_se - 0.0).abs() < 1e-10);
    }
}
