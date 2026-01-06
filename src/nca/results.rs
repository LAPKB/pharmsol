//! NCA result structures and options
//!
//! This module defines the main result structures for NCA analysis and
//! configuration options.

use serde::{Deserialize, Serialize};

use super::auc::AUCMethod;
use super::terminal::LambdaZOptions;

/// Administration route for pharmacokinetic dosing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdministrationRoute {
    /// Intravenous bolus (instantaneous IV administration)
    IVBolus,
    /// Intravenous infusion (IV administration over time)
    IVInfusion,
    /// Extravascular (oral, IM, SC, etc.)
    Extravascular,
}

impl AdministrationRoute {
    /// Detect administration route from dose characteristics
    /// 
    /// # Logic:
    /// - If is_infusion && duration > 0: IVInfusion
    /// - If is_infusion && duration == 0: IVBolus
    /// - Otherwise: Extravascular
    pub fn detect(is_infusion: bool, duration: Option<f64>) -> Self {
        if is_infusion {
            if let Some(dur) = duration {
                if dur > 0.0 {
                    Self::IVInfusion
                } else {
                    Self::IVBolus
                }
            } else {
                Self::IVBolus
            }
        } else {
            Self::Extravascular
        }
    }

    /// Check if route is intravenous (bolus or infusion)
    pub fn is_iv(&self) -> bool {
        matches!(self, Self::IVBolus | Self::IVInfusion)
    }

    /// Check if route is extravascular
    pub fn is_extravascular(&self) -> bool {
        matches!(self, Self::Extravascular)
    }
}

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

    // Dosing information (if available)
    /// Total dose administered
    pub dose: Option<f64>,
    /// Administration route detected from dose characteristics
    pub admin_route: Option<AdministrationRoute>,

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

    // Route-specific parameters
    /// Initial concentration (IV Bolus only) - extrapolated to time 0
    pub c0: Option<f64>,
    /// Volume of distribution (IV Bolus only) - Dose/C0
    pub vd: Option<f64>,
    /// Volume of distribution at steady state (IV Infusion only)
    pub vss: Option<f64>,
    /// Absorption rate constant (Extravascular only)
    pub ka: Option<f64>,
    /// Lag time (Extravascular only)
    pub tlag: Option<f64>,

    // MRT parameters
    /// AUMC from 0 to Tlast
    pub aumc_last: Option<f64>,
    /// Mean residence time
    pub mrt: Option<f64>,

    // Dose-normalized parameters
    /// Dose-normalized Cmax (Cmax/Dose)
    pub cmax_dn: Option<f64>,
    /// Dose-normalized AUClast (AUClast/Dose)
    pub auc_last_dn: Option<f64>,
    /// Dose-normalized AUCinf (AUCinf/Dose)
    pub auc_inf_dn: Option<f64>,

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
            dose: None,
            admin_route: None,
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
            c0: None,
            vd: None,
            vss: None,
            ka: None,
            tlag: None,
            aumc_last: None,
            mrt: None,            cmax_dn: None,
            auc_last_dn: None,
            auc_inf_dn: None,            warnings: Vec::new(),
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

/// BLQ (Below Limit of Quantification) handling rule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BLQRule {
    /// Replace BLQ values with zero
    Zero,
    /// Replace BLQ values with LOQ/2
    LoqOver2,
    /// Exclude BLQ values from analysis
    Exclude,
}

impl Default for BLQRule {
    fn default() -> Self {
        Self::Exclude
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
    /// BLQ handling rule (default: Exclude)
    pub blq_rule: BLQRule,
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
            blq_rule: BLQRule::default(),
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

    /// Set BLQ handling rule
    pub fn with_blq_rule(mut self, blq_rule: BLQRule) -> Self {
        self.blq_rule = blq_rule;
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

/// Apply BLQ (Below Limit of Quantification) handling rule to concentration data
///
/// # Pre-dose BLQ Handling
///
/// Pre-dose observations (time < first dose) that are BLQ are always excluded,
/// regardless of the BLQ rule. This is because:
/// - Zero/LOQ2 substitution before dosing could bias baseline correction
/// - Pre-dose BLQ is expected and pharmacologically meaningful
///
/// Post-dose BLQ observations are handled according to the specified rule:
/// - `Zero`: Replace with 0.0
/// - `LoqOver2`: Replace with LOQ/2
/// - `Exclude`: Remove from analysis
///
/// # Arguments
///
/// * `times` - Time points
/// * `concentrations` - Original concentration values
/// * `loq` - Limit of quantification
/// * `rule` - BLQ handling rule to apply
/// * `first_dose_time` - Time of first dose (for pre-dose detection)
///
/// # Returns
///
/// Tuple of (processed_times, processed_concentrations, blq_count, predose_blq_count)
fn apply_blq_rule(
    times: &[f64],
    concentrations: &[f64],
    loq: f64,
    rule: BLQRule,
    first_dose_time: f64,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let mut processed_times = Vec::new();
    let mut processed_concs = Vec::new();
    let mut blq_count = 0;
    let mut predose_blq_count = 0;

    for (i, &conc) in concentrations.iter().enumerate() {
        let is_blq = conc < loq && loq > 0.0;
        let is_predose = times[i] < first_dose_time;
        
        if is_blq {
            blq_count += 1;
            
            // Pre-dose BLQ: always exclude
            if is_predose {
                predose_blq_count += 1;
                continue;
            }
            
            // Post-dose BLQ: apply rule
            match rule {
                BLQRule::Zero => {
                    // Replace with zero
                    processed_times.push(times[i]);
                    processed_concs.push(0.0);
                }
                BLQRule::LoqOver2 => {
                    // Replace with LOQ/2
                    processed_times.push(times[i]);
                    processed_concs.push(loq / 2.0);
                }
                BLQRule::Exclude => {
                    // Skip this point
                    continue;
                }
            }
        } else {
            // Keep normal values
            processed_times.push(times[i]);
            processed_concs.push(conc);
        }
    }

    (processed_times, processed_concs, blq_count, predose_blq_count)
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

    // Determine first dose time (assume time 0 for dosing)
    // Pre-dose observations are those with time < 0
    let first_dose_time = 0.0;

    // Apply BLQ handling with pre-dose detection
    let (proc_times, proc_concs, blq_count, predose_blq_count) = apply_blq_rule(
        times, 
        concentrations, 
        options.loq, 
        options.blq_rule,
        first_dose_time,
    );
    
    if predose_blq_count > 0 {
        result.add_warning(format!(
            "Excluded {} pre-dose BLQ observation(s)",
            predose_blq_count
        ));
    }
    
    if blq_count > predose_blq_count {
        result.add_warning(format!(
            "Applied {} BLQ handling rule to {} post-dose point(s)",
            match options.blq_rule {
                BLQRule::Zero => "Zero",
                BLQRule::LoqOver2 => "LOQ/2",
                BLQRule::Exclude => "Exclude",
            },
            blq_count - predose_blq_count
        ));
    }

    if proc_times.is_empty() {
        result.add_warning("No data points after BLQ handling");
        return result;
    }

    // Use processed data for all calculations
    let times = &proc_times;
    let concentrations = &proc_concs;

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

                // Calculate route-specific parameters based on administration route
                if let Some(route) = result.admin_route {
                    match route {
                        AdministrationRoute::IVBolus => {
                            // C0: Initial concentration extrapolated to time 0
                            let c0_val = super::terminal::c0_iv_bolus(
                                result.clast,
                                result.tlast,
                                lz.lambda_z,
                            );
                            if c0_val.is_finite() {
                                result.c0 = Some(c0_val);
                                
                                // Vd: Volume of distribution = Dose/C0
                                let vd_val = super::terminal::vd_iv_bolus(d, c0_val);
                                if vd_val.is_finite() {
                                    result.vd = Some(vd_val);
                                }
                            }

                            // Vss: Volume at steady state (optional for bolus)
                            if let Some(aumc_val) = result.aumc_last {
                                let vss_val = super::terminal::vss_iv(d, aumc_val, result.auc_last);
                                if vss_val.is_finite() {
                                    result.vss = Some(vss_val);
                                }
                            }
                        }
                        AdministrationRoute::IVInfusion => {
                            // Vss: Primary parameter for infusion
                            if let Some(aumc_val) = result.aumc_last {
                                let vss_val = super::terminal::vss_iv(d, aumc_val, result.auc_last);
                                if vss_val.is_finite() {
                                    result.vss = Some(vss_val);
                                }
                            }
                        }
                        AdministrationRoute::Extravascular => {
                            // Ka: Absorption rate constant (if estimatable)
                            if let Some(ka_val) = super::terminal::ka_extravascular(
                                times,
                                concentrations,
                                lz.lambda_z,
                            ) {
                                result.ka = Some(ka_val);
                            }

                            // Tlag: Lag time
                            if let Some(tlag_val) = super::terminal::tlag_extravascular(
                                times,
                                concentrations,
                                options.loq,
                            ) {
                                result.tlag = Some(tlag_val);
                            }
                        }
                    }
                }

                // Calculate dose-normalized parameters
                result.cmax_dn = Some(result.cmax / d);
                result.auc_last_dn = Some(result.auc_last / d);
                if let Some(auc_inf_val) = result.auc_inf_obs {
                    result.auc_inf_dn = Some(auc_inf_val / d);
                }
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
            .with_blq_rule(BLQRule::Zero)
            .with_min_r_squared(0.95);

        assert_eq!(options.auc_method, AUCMethod::Linear);
        assert_eq!(options.loq, 0.1);
        assert_eq!(options.blq_rule, BLQRule::Zero);
        assert_eq!(options.lambda_z_options.min_r_squared, 0.95);
    }

    #[test]
    fn test_blq_handling_zero() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 0.05, 2.0, 0.03]; // Two BLQ at 4h and 12h
        
        let mut options = NCAOptions::default();
        options.loq = 0.1;
        options.blq_rule = BLQRule::Zero;
        
        let result = calculate_nca(&times, &concs, Some(100.0), &options);
        
        // BLQ values should be replaced with zero
        assert_eq!(result.cmax, 10.0);
        assert!(result.warnings.iter().any(|w| w.contains("BLQ")));
    }

    #[test]
    fn test_blq_handling_loq_over_2() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 0.05, 2.0, 0.03]; // Two BLQ
        
        let mut options = NCAOptions::default();
        options.loq = 0.1;
        options.blq_rule = BLQRule::LoqOver2;
        
        let result = calculate_nca(&times, &concs, Some(100.0), &options);
        
        // Should have processed data with LOQ/2 substitution
        assert_eq!(result.cmax, 10.0);
        assert!(result.warnings.iter().any(|w| w.contains("LOQ/2")));
    }

    #[test]
    fn test_blq_handling_exclude() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 0.05, 2.0, 0.03]; // Two BLQ
        
        let mut options = NCAOptions::default();
        options.loq = 0.1;
        options.blq_rule = BLQRule::Exclude;
        
        let result = calculate_nca(&times, &concs, Some(100.0), &options);
        
        // BLQ points should be excluded
        assert_eq!(result.cmax, 10.0);
        assert!(result.warnings.iter().any(|w| w.contains("Exclude")));
    }

    #[test]
    fn test_predose_blq_handling() {
        // Pre-dose BLQ should always be excluded
        let times = vec![-1.0, 0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.05, 0.0, 10.0, 8.0, 4.0, 2.0]; // Pre-dose BLQ at -1h
        
        let mut options = NCAOptions::default();
        options.loq = 0.1;
        options.blq_rule = BLQRule::Zero; // Even with Zero rule
        
        let result = calculate_nca(&times, &concs, Some(100.0), &options);
        
        // Pre-dose BLQ should be excluded
        assert!(result.warnings.iter().any(|w| w.contains("pre-dose")));
        assert_eq!(result.cmax, 10.0);
    }

    #[test]
    fn test_administration_route_enum() {
        // Test IV Bolus detection
        let bolus = AdministrationRoute::detect(true, None);
        assert_eq!(bolus, AdministrationRoute::IVBolus);
        assert!(bolus.is_iv());
        
        // Test IV Infusion detection
        let infusion = AdministrationRoute::detect(true, Some(1.0));
        assert_eq!(infusion, AdministrationRoute::IVInfusion);
        assert!(infusion.is_iv());
        
        // Test Extravascular detection
        let oral = AdministrationRoute::detect(false, None);
        assert_eq!(oral, AdministrationRoute::Extravascular);
        assert!(oral.is_extravascular());
    }
}
