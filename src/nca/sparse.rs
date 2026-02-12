//! Sparse PK analysis using Bailer's method
//!
//! For studies with destructive sampling (e.g., preclinical) or very sparse designs
//! (e.g., pediatric/oncology), individual subjects don't have enough samples for
//! traditional NCA. Bailer's method computes a population AUC with standard error
//! by using the trapezoidal rule on mean concentrations at each time point.
//!
//! Reference: Bailer AJ. "Testing for the equality of area under the curves when
//! using destructive measurement techniques." J Pharmacokinet Biopharm. 1988;16(3):303-309.

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

/// Time-concentration observation for sparse PK
#[derive(Debug, Clone)]
pub struct SparseObservation {
    /// Nominal sampling time
    pub time: f64,
    /// Observed concentration
    pub concentration: f64,
}

/// Compute population AUC from sparse/destructive sampling using Bailer's method
///
/// Groups observations by time point, computes mean and variance at each time,
/// then applies the trapezoidal rule to the mean concentrations. The standard
/// error is computed using the variance propagation formula for the trapezoidal rule.
///
/// # Arguments
/// * `observations` - All concentration-time observations (multiple subjects, sparse per subject)
/// * `time_tolerance` - Tolerance for grouping time points (default: observations at times
///   within this tolerance are considered the same nominal time). If `None`, exact matching is used.
///
/// # Returns
/// `None` if fewer than 2 unique time points with data
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::nca::sparse::{sparse_auc, SparseObservation};
///
/// let obs = vec![
///     SparseObservation { time: 0.0, concentration: 0.0 },  // Subject 1
///     SparseObservation { time: 0.0, concentration: 0.0 },  // Subject 2
///     SparseObservation { time: 1.0, concentration: 10.5 }, // Subject 3
///     SparseObservation { time: 1.0, concentration: 12.0 }, // Subject 4
///     SparseObservation { time: 4.0, concentration: 5.0 },  // Subject 5
///     SparseObservation { time: 4.0, concentration: 4.5 },  // Subject 6
///     SparseObservation { time: 8.0, concentration: 1.5 },  // Subject 7
///     SparseObservation { time: 8.0, concentration: 2.0 },  // Subject 8
/// ];
///
/// let result = sparse_auc(&obs, None).unwrap();
/// println!("Population AUC: {:.2} ± {:.2}", result.auc, result.auc_se);
/// println!("95% CI: [{:.2}, {:.2}]", result.auc_ci_lower, result.auc_ci_upper);
/// ```
pub fn sparse_auc(
    observations: &[SparseObservation],
    time_tolerance: Option<f64>,
) -> Option<SparsePKResult> {
    if observations.is_empty() {
        return None;
    }

    let tol = time_tolerance.unwrap_or(0.0);

    // Group observations by time point
    let mut time_groups: Vec<(f64, Vec<f64>)> = Vec::new();

    // Sort observations by time
    let mut sorted_obs: Vec<&SparseObservation> = observations.iter().collect();
    sorted_obs.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    for obs in &sorted_obs {
        let matched = time_groups.iter_mut().find(|(t, _)| (obs.time - *t).abs() <= tol);
        if let Some((_, group)) = matched {
            group.push(obs.concentration);
        } else {
            time_groups.push((obs.time, vec![obs.concentration]));
        }
    }

    // Sort by time
    time_groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if time_groups.len() < 2 {
        return None;
    }

    let n_timepoints = time_groups.len();
    let times: Vec<f64> = time_groups.iter().map(|(t, _)| *t).collect();
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
        let dt = times[i + 1] - times[i];
        auc += (mean_concentrations[i] + mean_concentrations[i + 1]) * dt / 2.0;
    }

    // Bailer's variance: sum of weighted variances
    // Var(AUC) = Σ (dt_i/2)² × (Var(C_i)/n_i + Var(C_{i+1})/n_{i+1})
    // But the exact formula sums the squared coefficients for each time point
    // The coefficient for time point j in the trapezoidal rule is:
    //   w_0 = dt_0/2, w_j = (dt_{j-1} + dt_j)/2 for 1 ≤ j ≤ k-1, w_k = dt_{k-1}/2
    // Var(AUC) = Σ w_j² × Var(C_j) / n_j

    let mut weights = vec![0.0; n_timepoints];
    for i in 0..n_timepoints - 1 {
        let dt = times[i + 1] - times[i];
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
        times,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_auc_basic() {
        // 4 time points, 3 subjects each
        let obs = vec![
            SparseObservation { time: 0.0, concentration: 0.0 },
            SparseObservation { time: 0.0, concentration: 0.0 },
            SparseObservation { time: 0.0, concentration: 0.0 },
            SparseObservation { time: 1.0, concentration: 10.0 },
            SparseObservation { time: 1.0, concentration: 12.0 },
            SparseObservation { time: 1.0, concentration: 11.0 },
            SparseObservation { time: 4.0, concentration: 5.0 },
            SparseObservation { time: 4.0, concentration: 4.0 },
            SparseObservation { time: 4.0, concentration: 6.0 },
            SparseObservation { time: 8.0, concentration: 1.0 },
            SparseObservation { time: 8.0, concentration: 1.5 },
            SparseObservation { time: 8.0, concentration: 1.2 },
        ];

        let result = sparse_auc(&obs, None).unwrap();

        assert_eq!(result.n_timepoints, 4);
        assert!(result.auc > 0.0);
        assert!(result.auc_se >= 0.0);
        assert!(result.auc_ci_lower <= result.auc);
        assert!(result.auc_ci_upper >= result.auc);

        // Manual: means = [0, 11, 5, ~1.23]
        // AUC ~= (0+11)/2 * 1 + (11+5)/2 * 3 + (5+1.23)/2 * 4 = 5.5 + 24 + 12.47 = 41.97
        assert!((result.mean_concentrations[0] - 0.0).abs() < 1e-10);
        assert!((result.mean_concentrations[1] - 11.0).abs() < 1e-10);
        assert!((result.mean_concentrations[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_auc_single_timepoint() {
        let obs = vec![
            SparseObservation { time: 0.0, concentration: 10.0 },
            SparseObservation { time: 0.0, concentration: 12.0 },
        ];

        assert!(sparse_auc(&obs, None).is_none());
    }

    #[test]
    fn test_sparse_auc_with_tolerance() {
        let obs = vec![
            SparseObservation { time: 0.0, concentration: 0.0 },
            SparseObservation { time: 0.01, concentration: 0.0 },  // Should group with t=0
            SparseObservation { time: 1.0, concentration: 10.0 },
            SparseObservation { time: 0.99, concentration: 12.0 }, // Should group with t=1
        ];

        let result = sparse_auc(&obs, Some(0.05)).unwrap();
        assert_eq!(result.n_timepoints, 2); // Should have 2 groups, not 4
    }

    #[test]
    fn test_sparse_auc_empty() {
        assert!(sparse_auc(&[], None).is_none());
    }

    #[test]
    fn test_sparse_auc_known_values() {
        // If all subjects have the same concentration at each time point,
        // variance = 0, SE = 0, and AUC = simple trapezoidal
        let obs = vec![
            SparseObservation { time: 0.0, concentration: 10.0 },
            SparseObservation { time: 0.0, concentration: 10.0 },
            SparseObservation { time: 2.0, concentration: 5.0 },
            SparseObservation { time: 2.0, concentration: 5.0 },
        ];

        let result = sparse_auc(&obs, None).unwrap();

        // AUC = (10 + 5) / 2 * 2 = 15
        assert!((result.auc - 15.0).abs() < 1e-10);
        assert!((result.auc_se - 0.0).abs() < 1e-10);
    }
}
