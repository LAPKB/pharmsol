//! Population summary statistics for NCA results
//!
//! Computes descriptive statistics across multiple [`NCAResult`]s,
//! including geometric mean, CV%, and percentiles — standard PK reporting metrics.
//!
//! # Example
//!
//! ```rust,ignore
//! use pharmsol::nca::{summarize, NCAOptions, NCA};
//!
//! let results: Vec<NCAResult> = subjects.iter()
//!     .flat_map(|s| s.nca_all(&NCAOptions::default()))
//!     .filter_map(|r| r.ok())
//!     .collect();
//!
//! let summary = summarize(&results);
//! println!("N subjects: {}", summary.n_subjects);
//! for p in &summary.parameters {
//!     println!("{}: mean={:.2} CV%={:.1}", p.name, p.mean, p.cv_pct);
//! }
//! ```

use super::types::NCAResult;
use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// Descriptive statistics for a single NCA parameter across subjects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSummary {
    /// Parameter name (matches keys from `NCAResult::to_params()`)
    pub name: String,
    /// Number of subjects with this parameter
    pub n: usize,
    /// Arithmetic mean
    pub mean: f64,
    /// Standard deviation
    pub sd: f64,
    /// Coefficient of variation (%)
    pub cv_pct: f64,
    /// Median
    pub median: f64,
    /// Minimum
    pub min: f64,
    /// Maximum
    pub max: f64,
    /// Geometric mean (NaN if any values ≤ 0)
    pub geo_mean: f64,
    /// Geometric CV% (NaN if any values ≤ 0)
    pub geo_cv_pct: f64,
    /// 5th percentile
    pub p5: f64,
    /// 25th percentile (Q1)
    pub p25: f64,
    /// 75th percentile (Q3)
    pub p75: f64,
    /// 95th percentile
    pub p95: f64,
}

/// Summary of NCA results across a population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSummary {
    /// Total number of NCA results summarized
    pub n_subjects: usize,
    /// Per-parameter descriptive statistics
    pub parameters: Vec<ParameterSummary>,
}

// ============================================================================
// Public API
// ============================================================================

/// Compute population summary statistics from a collection of NCA results
///
/// Extracts each named parameter via [`NCAResult::to_params()`], then computes
/// descriptive statistics across all results that have that parameter.
///
/// Parameters are returned in a stable alphabetical order.
pub fn summarize(results: &[NCAResult]) -> PopulationSummary {
    if results.is_empty() {
        return PopulationSummary {
            n_subjects: 0,
            parameters: Vec::new(),
        };
    }

    // Collect all parameter names across all results
    let mut all_params: std::collections::BTreeMap<&'static str, Vec<f64>> =
        std::collections::BTreeMap::new();

    for result in results {
        let params = result.to_params();
        for (name, value) in params {
            all_params.entry(name).or_default().push(value);
        }
    }

    // Compute summary for each parameter
    let parameters: Vec<ParameterSummary> = all_params
        .into_iter()
        .map(|(name, values)| compute_parameter_summary(name, &values))
        .collect();

    PopulationSummary {
        n_subjects: results.len(),
        parameters,
    }
}

/// Generate a CSV string from a slice of NCA results
///
/// The CSV has a header row containing `subject_id`, `occasion`, and all
/// parameter names (union across all results). Each subsequent row contains
/// one result. Missing parameters are left empty.
///
/// # Example
///
/// ```rust,ignore
/// let csv = pharmsol::nca::nca_to_csv(&results);
/// std::fs::write("nca_results.csv", csv).unwrap();
/// ```
pub fn nca_to_csv(results: &[NCAResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    // Collect all unique parameter names in stable order
    let mut param_names: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();
    let param_maps: Vec<_> = results
        .iter()
        .map(|r| {
            let p = r.to_params();
            for name in p.keys() {
                param_names.insert(name);
            }
            p
        })
        .collect();

    let ordered_names: Vec<&str> = param_names.into_iter().collect();

    // Build CSV
    let mut csv = String::new();

    // Header
    csv.push_str("subject_id,occasion");
    for name in &ordered_names {
        csv.push(',');
        csv.push_str(name);
    }
    csv.push('\n');

    // Data rows
    for (result, params) in results.iter().zip(param_maps.iter()) {
        // Subject ID
        match &result.subject_id {
            Some(id) => csv.push_str(id),
            None => csv.push_str("NA"),
        }
        csv.push(',');

        // Occasion
        match result.occasion {
            Some(occ) => csv.push_str(&occ.to_string()),
            None => csv.push_str("NA"),
        }

        // Parameters
        for name in &ordered_names {
            csv.push(',');
            if let Some(val) = params.get(name) {
                csv.push_str(&val.to_string());
            }
        }
        csv.push('\n');
    }

    csv
}

// ============================================================================
// Internal helpers
// ============================================================================

fn compute_parameter_summary(name: &str, values: &[f64]) -> ParameterSummary {
    use statrs::statistics::{Data, Distribution, Max, Min, OrderStatistics};

    let n = values.len();
    assert!(n > 0);

    let mut data = Data::new(values.to_vec());

    let mean = data.mean().unwrap_or(f64::NAN);
    let sd = if n > 1 {
        data.std_dev().unwrap_or(0.0)
    } else {
        0.0
    };
    let cv_pct = if mean.abs() > f64::EPSILON {
        (sd / mean) * 100.0
    } else {
        f64::NAN
    };

    let median = data.median();
    let min = data.min();
    let max = data.max();

    // Geometric statistics (only valid for positive values)
    let (geo_mean, geo_cv_pct) = if values.iter().all(|&v| v > 0.0) {
        let log_values: Vec<f64> = values.iter().map(|v| v.ln()).collect();
        let log_data = Data::new(log_values);
        let log_mean = log_data.mean().unwrap_or(f64::NAN);
        let gm = log_mean.exp();

        let log_var = log_data.variance().unwrap_or(0.0);
        // Geometric CV% = sqrt(exp(s²) - 1) * 100
        let gcv = (log_var.exp() - 1.0).sqrt() * 100.0;
        (gm, gcv)
    } else {
        (f64::NAN, f64::NAN)
    };

    ParameterSummary {
        name: name.to_string(),
        n,
        mean,
        sd,
        cv_pct,
        median,
        min,
        max,
        geo_mean,
        geo_cv_pct,
        p5: data.percentile(5),
        p25: data.percentile(25),
        p75: data.percentile(75),
        p95: data.percentile(95),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::event::Route;
    use crate::nca::types::*;

    fn make_result(subject_id: &str, cmax: f64, auc_last: f64, lambda_z: f64) -> NCAResult {
        let half_life = std::f64::consts::LN_2 / lambda_z;
        NCAResult {
            subject_id: Some(subject_id.to_string()),
            occasion: Some(0),
            dose_amount: Some(100.0),
            route: Some(Route::Extravascular),
            infusion_duration: None,
            exposure: ExposureParams {
                cmax,
                tmax: 1.0,
                clast: cmax * 0.1,
                tlast: 24.0,
                tfirst: Some(0.5),
                auc_last,
                auc_inf_obs: Some(auc_last * 1.1),
                auc_inf_pred: Some(auc_last * 1.12),
                auc_pct_extrap_obs: Some(9.1),
                auc_pct_extrap_pred: Some(10.7),
                auc_partial: None,
                aumc_last: None,
                aumc_inf: None,
                tlag: None,
                cmax_dn: Some(cmax / 100.0),
                auc_last_dn: Some(auc_last / 100.0),
                auc_inf_dn: Some(auc_last * 1.1 / 100.0),
                time_above_mic: None,
            },
            terminal: Some(TerminalParams {
                lambda_z,
                half_life,
                regression: Some(RegressionStats {
                    r_squared: 0.99,
                    adj_r_squared: 0.98,
                    corrxy: -0.995,
                    n_points: 5,
                    time_first: 4.0,
                    time_last: 24.0,
                    span_ratio: 3.0,
                }),
                mrt: Some(half_life * 1.44),
                effective_half_life: Some(std::f64::consts::LN_2 * half_life * 1.44),
                kel: Some(1.0 / (half_life * 1.44)),
            }),
            clearance: Some(ClearanceParams {
                cl_f: 100.0 / (auc_last * 1.1),
                vz_f: 100.0 / (auc_last * 1.1 * lambda_z),
                vss: None,
            }),
            route_params: Some(RouteParams::Extravascular),
            steady_state: None,
            multi_dose: None,
            quality: Quality { warnings: vec![] },
        }
    }

    #[test]
    fn test_summarize_basic() {
        let results = vec![
            make_result("S1", 10.0, 100.0, 0.1),
            make_result("S2", 20.0, 200.0, 0.15),
            make_result("S3", 15.0, 150.0, 0.12),
        ];

        let summary = summarize(&results);
        assert_eq!(summary.n_subjects, 3);
        assert!(!summary.parameters.is_empty());

        // Check cmax summary
        let cmax = summary
            .parameters
            .iter()
            .find(|p| p.name == "cmax")
            .unwrap();
        assert_eq!(cmax.n, 3);
        assert!((cmax.mean - 15.0).abs() < 1e-10);
        assert_eq!(cmax.min, 10.0);
        assert_eq!(cmax.max, 20.0);
        assert_eq!(cmax.median, 15.0);
    }

    #[test]
    fn test_summarize_single_result() {
        let results = vec![make_result("S1", 10.0, 100.0, 0.1)];

        let summary = summarize(&results);
        assert_eq!(summary.n_subjects, 1);

        let cmax = summary
            .parameters
            .iter()
            .find(|p| p.name == "cmax")
            .unwrap();
        assert_eq!(cmax.n, 1);
        assert!((cmax.mean - 10.0).abs() < 1e-10);
        assert_eq!(cmax.sd, 0.0);
        assert_eq!(cmax.min, 10.0);
        assert_eq!(cmax.max, 10.0);
    }

    #[test]
    fn test_summarize_empty() {
        let summary = summarize(&[]);
        assert_eq!(summary.n_subjects, 0);
        assert!(summary.parameters.is_empty());
    }

    #[test]
    fn test_summarize_geometric_stats() {
        // Known values for geometric mean
        let results = vec![
            make_result("S1", 10.0, 100.0, 0.1),
            make_result("S2", 10.0, 100.0, 0.1),
        ];

        let summary = summarize(&results);
        let cmax = summary
            .parameters
            .iter()
            .find(|p| p.name == "cmax")
            .unwrap();

        // All same value → geo_mean = 10.0, geo_cv = 0%
        assert!((cmax.geo_mean - 10.0).abs() < 1e-10);
        assert!((cmax.geo_cv_pct - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_summarize_percentiles() {
        // Create 5 results with known cmax values: 10, 20, 30, 40, 50
        let results: Vec<NCAResult> = (1..=5)
            .map(|i| make_result(&format!("S{}", i), i as f64 * 10.0, 100.0, 0.1))
            .collect();

        let summary = summarize(&results);
        let cmax = summary
            .parameters
            .iter()
            .find(|p| p.name == "cmax")
            .unwrap();

        assert_eq!(cmax.n, 5);
        assert!((cmax.mean - 30.0).abs() < 1e-10);
        assert_eq!(cmax.median, 30.0);
        assert_eq!(cmax.min, 10.0);
        assert_eq!(cmax.max, 50.0);
    }

    #[test]
    fn test_summarize_parameters_sorted() {
        let results = vec![make_result("S1", 10.0, 100.0, 0.1)];
        let summary = summarize(&results);

        // Parameters should be in alphabetical order (BTreeMap)
        let names: Vec<&str> = summary.parameters.iter().map(|p| p.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "Parameters should be alphabetically sorted");
    }

    #[test]
    fn test_nca_to_csv_basic() {
        let results = vec![
            make_result("S1", 10.0, 100.0, 0.1),
            make_result("S2", 20.0, 200.0, 0.15),
        ];

        let csv = nca_to_csv(&results);

        // Check header
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines.len() >= 3, "Should have header + 2 data rows");
        assert!(lines[0].starts_with("subject_id,occasion"));

        // Check subject IDs appear
        assert!(lines[1].starts_with("S1,"));
        assert!(lines[2].starts_with("S2,"));
    }

    #[test]
    fn test_nca_to_csv_empty() {
        let csv = nca_to_csv(&[]);
        assert!(csv.is_empty());
    }
}
