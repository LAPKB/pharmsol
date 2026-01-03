//! NCA result structures and options
//!
//! This module defines the main result structures for NCA analysis and
//! configuration options.

use serde::{Deserialize, Serialize};

use super::auc::AUCMethod;
use super::terminal::LambdaZOptions;

/// Complete NCA results for a single subject or profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NCAResult {
    // Identification
    /// Subject ID (if from Subject data)
    pub subject_id: Option<String>,
    /// Occasion index (if from multi-occasion data)
    pub occasion: Option<usize>,
    /// Output equation index (for multiple analytes)
    pub outeq: usize,

    // Primary concentration parameters
    /// Maximum observed concentration
    pub cmax: f64,
    /// Time of maximum concentration
    pub tmax: f64,
    /// Last measurable concentration (> LOQ)
    pub clast: f64,
    /// Time of last measurable concentration
    pub tlast: f64,

    // AUC parameters
    /// AUC from time 0 to Tlast
    pub auc_last: f64,
    /// AUC from time 0 to infinity (observed Clast)
    pub auc_inf_obs: Option<f64>,
    /// AUC from time 0 to infinity (predicted Clast)
    pub auc_inf_pred: Option<f64>,
    /// Percent of AUC extrapolated
    pub auc_pct_extrap: Option<f64>,

    // Terminal phase parameters
    /// Terminal elimination rate constant
    pub lambda_z: Option<f64>,
    /// Terminal half-life
    pub half_life: Option<f64>,
    /// R² for lambda_z regression
    pub r_squared: Option<f64>,
    /// Adjusted R² for lambda_z regression
    pub adj_r_squared: Option<f64>,
    /// Number of points used in lambda_z calculation
    pub n_points_lambda_z: Option<usize>,
    /// Span ratio for lambda_z
    pub span_ratio: Option<f64>,

    // Derived parameters
    /// Apparent clearance (CL/F) - requires dose
    pub clearance: Option<f64>,
    /// Apparent volume of distribution (Vz/F) - requires dose
    pub vz: Option<f64>,

    // MRT parameters
    /// AUMC from 0 to Tlast
    pub aumc_last: Option<f64>,
    /// Mean residence time
    pub mrt: Option<f64>,

    // Dosing information (if available)
    /// Total dose administered
    pub dose: Option<f64>,

    // Quality flags
    /// Warnings generated during calculation
    pub warnings: Vec<String>,
}

impl Default for NCAResult {
    fn default() -> Self {
        Self {
            subject_id: None,
            occasion: None,
            outeq: 0,
            cmax: 0.0,
            tmax: 0.0,
            clast: 0.0,
            tlast: 0.0,
            auc_last: 0.0,
            auc_inf_obs: None,
            auc_inf_pred: None,
            auc_pct_extrap: None,
            lambda_z: None,
            half_life: None,
            r_squared: None,
            adj_r_squared: None,
            n_points_lambda_z: None,
            span_ratio: None,
            clearance: None,
            vz: None,
            aumc_last: None,
            mrt: None,
            dose: None,
            warnings: Vec::new(),
        }
    }
}

impl NCAResult {
    /// Create a new NCAResult with basic identification
    pub fn new(subject_id: Option<String>, occasion: Option<usize>, outeq: usize) -> Self {
        Self {
            subject_id,
            occasion,
            outeq,
            ..Default::default()
        }
    }

    /// Add a warning message
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    /// Check if lambda_z calculation was successful
    pub fn has_lambda_z(&self) -> bool {
        self.lambda_z.is_some()
    }

    /// Check if AUC extrapolation is reliable (< max_extrap %)
    pub fn is_auc_extrap_reliable(&self, max_extrap: f64) -> bool {
        self.auc_pct_extrap
            .map(|pct| pct < max_extrap)
            .unwrap_or(false)
    }
}

/// Options for NCA calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NCAOptions {
    /// AUC calculation method (default: LinUpLogDown)
    pub auc_method: AUCMethod,
    /// Options for lambda_z calculation
    pub lambda_z_options: LambdaZOptions,
    /// Return first Tmax if multiple maxima (default: true)
    pub first_tmax: bool,
    /// Limit of quantification (default: 0.0)
    pub loq: f64,
    /// Maximum acceptable % AUC extrapolation (default: 20.0)
    pub max_auc_extrap: f64,
    /// Calculate AUMC and MRT (default: true)
    pub calculate_mrt: bool,
}

impl Default for NCAOptions {
    fn default() -> Self {
        Self {
            auc_method: AUCMethod::LinUpLogDown,
            lambda_z_options: LambdaZOptions::default(),
            first_tmax: true,
            loq: 0.0,
            max_auc_extrap: 20.0,
            calculate_mrt: true,
        }
    }
}

impl NCAOptions {
    /// Create options with linear trapezoidal method
    pub fn linear() -> Self {
        Self {
            auc_method: AUCMethod::Linear,
            ..Default::default()
        }
    }

    /// Set the AUC calculation method
    pub fn with_auc_method(mut self, method: AUCMethod) -> Self {
        self.auc_method = method;
        self
    }

    /// Set the limit of quantification
    pub fn with_loq(mut self, loq: f64) -> Self {
        self.loq = loq;
        self
    }

    /// Set lambda_z options
    pub fn with_lambda_z_options(mut self, options: LambdaZOptions) -> Self {
        self.lambda_z_options = options;
        self
    }

    /// Set minimum R² for lambda_z
    pub fn with_min_r_squared(mut self, min_r_squared: f64) -> Self {
        self.lambda_z_options.min_r_squared = min_r_squared;
        self
    }

    /// Set minimum points for lambda_z calculation
    pub fn with_min_points(mut self, min_points: usize) -> Self {
        self.lambda_z_options.min_points = min_points;
        self
    }
}

/// Calculate complete NCA from time-concentration arrays
///
/// This is the main NCA calculation function that computes all standard
/// NCA parameters from raw data arrays.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `dose` - Optional dose amount for calculating CL/F and Vz/F
/// * `options` - Calculation options
///
/// # Returns
///
/// `NCAResult` containing all calculated parameters
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{calculate_nca, NCAOptions};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0, 1.0];
///
/// let result = calculate_nca(&times, &concs, Some(100.0), &NCAOptions::default());
///
/// println!("Cmax: {:.2}", result.cmax);
/// println!("AUClast: {:.2}", result.auc_last);
/// if let Some(hl) = result.half_life {
///     println!("t½: {:.2} h", hl);
/// }
/// ```
pub fn calculate_nca(
    times: &[f64],
    concentrations: &[f64],
    dose: Option<f64>,
    options: &NCAOptions,
) -> NCAResult {
    use super::auc::auc_last;
    use super::params::{clast_tlast, cmax_tmax};
    use super::terminal::{
        auc_inf, auc_inf_pred, auc_percent_extrap, aumc_last, clearance, lambda_z, mrt, vz,
    };

    let mut result = NCAResult::default();
    result.dose = dose;

    // Validate inputs
    if times.is_empty() || times.len() != concentrations.len() {
        result.add_warning("Invalid input: empty or mismatched arrays");
        return result;
    }

    // Calculate Cmax/Tmax
    if let Some(cm) = cmax_tmax(times, concentrations, options.first_tmax) {
        result.cmax = cm.cmax;
        result.tmax = cm.tmax;
    } else {
        result.add_warning("Could not calculate Cmax/Tmax");
        return result;
    }

    // Calculate Clast/Tlast
    if let Some(cl) = clast_tlast(times, concentrations, options.loq) {
        result.clast = cl.clast;
        result.tlast = cl.tlast;

        // Calculate AUClast
        result.auc_last = auc_last(times, concentrations, options.auc_method);

        // Calculate lambda_z
        let tmax_idx = times
            .iter()
            .position(|&t| (t - result.tmax).abs() < 1e-10)
            .unwrap_or(0);

        if let Some(lz) = lambda_z(
            times,
            concentrations,
            tmax_idx,
            cl.index,
            &options.lambda_z_options,
        ) {
            result.lambda_z = Some(lz.lambda_z);
            result.half_life = Some(lz.half_life);
            result.r_squared = Some(lz.r_squared);
            result.adj_r_squared = Some(lz.adj_r_squared);
            result.n_points_lambda_z = Some(lz.n_points);
            result.span_ratio = Some(lz.span_ratio);

            // Calculate AUCinf
            let auc_inf_obs_val = auc_inf(result.auc_last, result.clast, lz.lambda_z);
            let auc_inf_pred_val = auc_inf_pred(result.auc_last, lz.clast_pred, lz.lambda_z);

            result.auc_inf_obs = Some(auc_inf_obs_val);
            result.auc_inf_pred = Some(auc_inf_pred_val);
            result.auc_pct_extrap = Some(auc_percent_extrap(result.auc_last, auc_inf_obs_val));

            // Check extrapolation
            if let Some(pct) = result.auc_pct_extrap {
                if pct > options.max_auc_extrap {
                    result.add_warning(format!(
                        "AUC extrapolation ({:.1}%) exceeds maximum ({:.1}%)",
                        pct, options.max_auc_extrap
                    ));
                }
            }

            // Calculate clearance and Vz if dose is provided
            if let Some(d) = dose {
                result.clearance = Some(clearance(d, auc_inf_obs_val));
                result.vz = Some(vz(d, lz.lambda_z, auc_inf_obs_val));
            }
        } else {
            result.add_warning("Could not calculate lambda_z: insufficient data or poor fit");
        }

        // Calculate AUMC and MRT
        if options.calculate_mrt {
            let aumc_val = aumc_last(times, concentrations);
            result.aumc_last = Some(aumc_val);

            if result.auc_last > 0.0 {
                result.mrt = Some(mrt(aumc_val, result.auc_last));
            }
        }
    } else {
        result.add_warning("No concentrations above LOQ");
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_nca_basic() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0, 1.0];

        let result = calculate_nca(&times, &concs, Some(100.0), &NCAOptions::default());

        // Check basic parameters
        assert_eq!(result.cmax, 10.0);
        assert_eq!(result.tmax, 1.0);
        assert_eq!(result.clast, 1.0);
        assert_eq!(result.tlast, 12.0);
        assert!(result.auc_last > 0.0);
    }

    #[test]
    fn test_calculate_nca_with_lambda_z() {
        // Exponential decay data
        let times: Vec<f64> = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let concs: Vec<f64> = times.iter().map(|&t| 10.0 * (-0.1_f64 * t).exp()).collect();

        let mut options = NCAOptions::default();
        options.lambda_z_options.min_r_squared = 0.9;

        let result = calculate_nca(&times, &concs, Some(100.0), &options);

        assert!(result.lambda_z.is_some());
        assert!(result.half_life.is_some());
        assert!(result.auc_inf_obs.is_some());
        assert!(result.clearance.is_some());
    }

    #[test]
    fn test_nca_options_builder() {
        let options = NCAOptions::default()
            .with_auc_method(AUCMethod::Linear)
            .with_loq(0.1)
            .with_min_r_squared(0.95);

        assert_eq!(options.auc_method, AUCMethod::Linear);
        assert_eq!(options.loq, 0.1);
        assert_eq!(options.lambda_z_options.min_r_squared, 0.95);
    }
}
