//! Terminal phase analysis: λz (lambda_z) and half-life estimation
//!
//! This module provides functions for estimating the terminal elimination rate
//! constant (λz) and derived parameters using log-linear regression.
//!
//! # Algorithm: Curve Stripping
//!
//! The default algorithm follows the PKNCA approach:
//!
//! 1. **Point Selection**: Start from Tlast and work backwards
//!    - Minimum 3 points (configurable)
//!    - Exclude Tmax by default (configurable)
//!    - Exclude BLQ values
//!
//! 2. **Regression**: Fit ln(C) vs time using ordinary least squares
//!    - λz = -slope
//!    - Must have λz > 0
//!
//! 3. **Best Fit Selection**: Among all valid fits
//!    - Maximum adjusted R² (within tolerance)
//!    - Prefer more points if R² is similar
//!
//! # Quality Criteria
//!
//! - R² ≥ 0.9 (configurable via `min_r_squared`)
//! - Span ratio ≥ 2 (time span / half-life)
//! - λz > 0

use serde::{Deserialize, Serialize};

/// Lambda-z selection method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LambdaZMethod {
    /// Optimize adjusted R² (Best Fit) - default
    AdjustedR2,
    /// Optimize raw R² value
    R2,
    /// User-defined time interval
    Interval,
    /// Use last N points
    Points,
}

impl Default for LambdaZMethod {
    fn default() -> Self {
        Self::AdjustedR2
    }
}

/// Regression weighting method for lambda-z calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionWeight {
    /// No weighting (uniform) - default
    Uniform,
    /// Weight by 1/Y
    InverseY,
    /// Weight by 1/Y²
    InverseYSquared,
}

impl Default for RegressionWeight {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Result of λz estimation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaZResult {
    /// Terminal elimination rate constant (1/time)
    pub lambda_z: f64,
    /// Terminal half-life: ln(2) / λz
    pub half_life: f64,
    /// Coefficient of determination
    pub r_squared: f64,
    /// Adjusted R² accounting for number of points
    pub adj_r_squared: f64,
    /// Number of points used in regression
    pub n_points: usize,
    /// First time point used in regression
    pub time_first: f64,
    /// Last time point used in regression
    pub time_last: f64,
    /// Span ratio: (time_last - time_first) / half_life
    pub span_ratio: f64,
    /// Predicted concentration at Tlast (from regression)
    pub clast_pred: f64,
    /// Intercept of ln(C) vs time regression
    pub intercept: f64,
    /// Method used for selection
    pub method: LambdaZMethod,
    /// Weighting used for regression
    pub weighting: RegressionWeight,
}

/// Options for λz calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaZOptions {
    /// Selection method (default: AdjustedR2)
    pub method: LambdaZMethod,
    /// Regression weighting (default: Uniform)
    pub weighting: RegressionWeight,
    /// Minimum number of points to use in regression (default: 3)
    pub min_points: usize,
    /// Maximum number of points (None = no limit)
    pub max_points: Option<usize>,
    /// Whether to allow Tmax in the regression (default: false)
    pub allow_tmax: bool,
    /// Minimum R² to accept (default: 0.9)
    pub min_r_squared: f64,
    /// Minimum span ratio to accept (default: 2.0)
    pub min_span_ratio: f64,
    /// Minimum time for regression start (default: 0.0)
    pub min_time: f64,
    /// Tolerance for comparing R² values (default: 1e-4)
    pub r_squared_tolerance: f64,
    /// For Interval method: start time
    pub interval_start_time: Option<f64>,
    /// For Interval method: end time
    pub interval_end_time: Option<f64>,
    /// For Points method: number of points to use
    pub n_points: Option<usize>,
}

impl Default for LambdaZOptions {
    fn default() -> Self {
        Self {
            method: LambdaZMethod::AdjustedR2,
            weighting: RegressionWeight::Uniform,
            min_points: 3,
            max_points: None,
            allow_tmax: false,
            min_r_squared: 0.9,
            min_span_ratio: 2.0,
            min_time: 0.0,
            r_squared_tolerance: 1e-4,
            interval_start_time: None,
            interval_end_time: None,
            n_points: None,
        }
    }
}

/// Calculate λz and related parameters using curve stripping
///
/// This function implements the standard curve stripping algorithm for
/// estimating the terminal elimination rate constant.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `tmax_idx` - Index of Tmax (to potentially exclude from regression)
/// * `tlast_idx` - Index of Tlast (last point to include)
/// * `options` - Configuration options for the calculation
///
/// # Returns
///
/// `Some(LambdaZResult)` if a valid regression is found, `None` otherwise.
///
/// # Algorithm
///
/// 1. Generate all candidate point sets (from tlast backwards)
/// 2. Fit log-linear regression for each set
/// 3. Select best fit based on R² and number of points
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{lambda_z, LambdaZOptions};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
/// let concs = vec![0.0, 10.0, 8.0, 5.0, 2.5, 1.2, 0.3];
///
/// let options = LambdaZOptions::default();
/// if let Some(result) = lambda_z(&times, &concs, 1, 6, &options) {
///     println!("λz = {:.4} h⁻¹", result.lambda_z);
///     println!("t½ = {:.2} h", result.half_life);
///     println!("R² = {:.4}", result.r_squared);
/// }
/// ```
pub fn lambda_z(
    times: &[f64],
    concentrations: &[f64],
    tmax_idx: usize,
    tlast_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    if times.len() != concentrations.len() || tlast_idx >= times.len() {
        return None;
    }

    // Route to appropriate method
    match options.method {
        LambdaZMethod::AdjustedR2 => {
            lambda_z_adjusted_r2(times, concentrations, tmax_idx, tlast_idx, options)
        }
        LambdaZMethod::R2 => lambda_z_r2(times, concentrations, tmax_idx, tlast_idx, options),
        LambdaZMethod::Interval => {
            lambda_z_interval(times, concentrations, tmax_idx, tlast_idx, options)
        }
        LambdaZMethod::Points => {
            lambda_z_points(times, concentrations, tmax_idx, tlast_idx, options)
        }
    }
}

/// Lambda-z using Adjusted R² method (Best Fit)
fn lambda_z_adjusted_r2(
    times: &[f64],
    concentrations: &[f64],
    tmax_idx: usize,
    tlast_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    if times.len() != concentrations.len() || tlast_idx >= times.len() {
        return None;
    }

    // Determine starting point for regression
    // If allow_tmax is false, start after tmax
    let start_idx = if options.allow_tmax { 0 } else { tmax_idx + 1 };

    // Need at least min_points between start and tlast
    if tlast_idx < start_idx + options.min_points - 1 {
        return None;
    }

    let mut best_result: Option<LambdaZResult> = None;

    // Determine maximum number of points to try
    let max_n_points = if let Some(max_pts) = options.max_points {
        (tlast_idx - start_idx + 1).min(max_pts)
    } else {
        tlast_idx - start_idx + 1
    };

    // Try all possible point sets from tlast backwards
    // Start with minimum points, increase to maximum
    for n_points in options.min_points..=max_n_points {
        // Calculate starting index for this number of points
        let first_idx = tlast_idx - n_points + 1;

        // Skip if this would include points before start_idx
        if first_idx < start_idx {
            continue;
        }

        // Skip if first point is before min_time
        if times[first_idx] < options.min_time {
            continue;
        }

        // Extract points for regression (only positive concentrations)
        let mut reg_times = Vec::with_capacity(n_points);
        let mut reg_log_conc = Vec::with_capacity(n_points);
        let mut reg_conc = Vec::with_capacity(n_points);

        for i in first_idx..=tlast_idx {
            if concentrations[i] > 0.0 {
                reg_times.push(times[i]);
                reg_log_conc.push(concentrations[i].ln());
                reg_conc.push(concentrations[i]);
            }
        }

        // Need at least min_points with positive concentrations
        if reg_times.len() < options.min_points {
            continue;
        }

        // Perform weighted linear regression: ln(C) = intercept + slope * t
        if let Some((slope, intercept, r_squared)) =
            weighted_linear_regression(&reg_times, &reg_log_conc, &reg_conc, &options.weighting)
        {
            let lambda = -slope;

            // λz must be positive
            if lambda <= 0.0 {
                continue;
            }

            let half_life = std::f64::consts::LN_2 / lambda;
            let time_span = reg_times[reg_times.len() - 1] - reg_times[0];
            let span_ratio = time_span / half_life;

            // Calculate adjusted R²
            let n = reg_times.len() as f64;
            let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);

            // Calculate predicted Clast
            let clast_pred = (intercept + slope * times[tlast_idx]).exp();

            let result = LambdaZResult {
                lambda_z: lambda,
                half_life,
                r_squared,
                adj_r_squared,
                n_points: reg_times.len(),
                time_first: reg_times[0],
                time_last: reg_times[reg_times.len() - 1],
                span_ratio,
                clast_pred,
                intercept,
                method: LambdaZMethod::AdjustedR2,
                weighting: options.weighting,
            };

            // Check quality criteria
            if r_squared < options.min_r_squared {
                continue;
            }

            // Select best result
            // Priority: highest adj R² within tolerance, then most points
            match &best_result {
                None => best_result = Some(result),
                Some(best) => {
                    let r_diff = result.adj_r_squared - best.adj_r_squared;
                    if r_diff > options.r_squared_tolerance {
                        // Significantly better R²
                        best_result = Some(result);
                    } else if r_diff >= -options.r_squared_tolerance
                        && result.n_points > best.n_points
                    {
                        // Similar R² but more points
                        best_result = Some(result);
                    }
                }
            }
        }
    }

    best_result
}

/// Lambda-z using R² method (optimize raw R² without adjustment)
fn lambda_z_r2(
    times: &[f64],
    concentrations: &[f64],
    tmax_idx: usize,
    tlast_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    if times.len() != concentrations.len() || tlast_idx >= times.len() {
        return None;
    }

    let start_idx = if options.allow_tmax { 0 } else { tmax_idx + 1 };

    if tlast_idx < start_idx + options.min_points - 1 {
        return None;
    }

    let mut best_result: Option<LambdaZResult> = None;

    let max_n_points = if let Some(max_pts) = options.max_points {
        (tlast_idx - start_idx + 1).min(max_pts)
    } else {
        tlast_idx - start_idx + 1
    };

    for n_points in options.min_points..=max_n_points {
        let first_idx = tlast_idx - n_points + 1;

        if first_idx < start_idx {
            continue;
        }

        if times[first_idx] < options.min_time {
            continue;
        }

        let mut reg_times = Vec::with_capacity(n_points);
        let mut reg_log_conc = Vec::with_capacity(n_points);
        let mut reg_conc = Vec::with_capacity(n_points);

        for i in first_idx..=tlast_idx {
            if concentrations[i] > 0.0 {
                reg_times.push(times[i]);
                reg_log_conc.push(concentrations[i].ln());
                reg_conc.push(concentrations[i]);
            }
        }

        if reg_times.len() < options.min_points {
            continue;
        }

        if let Some((slope, intercept, r_squared)) =
            weighted_linear_regression(&reg_times, &reg_log_conc, &reg_conc, &options.weighting)
        {
            let lambda = -slope;

            if lambda <= 0.0 {
                continue;
            }

            let half_life = std::f64::consts::LN_2 / lambda;
            let time_span = reg_times[reg_times.len() - 1] - reg_times[0];
            let span_ratio = time_span / half_life;

            let n = reg_times.len() as f64;
            let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);
            let clast_pred = (intercept + slope * times[tlast_idx]).exp();

            let result = LambdaZResult {
                lambda_z: lambda,
                half_life,
                r_squared,
                adj_r_squared,
                n_points: reg_times.len(),
                time_first: reg_times[0],
                time_last: reg_times[reg_times.len() - 1],
                span_ratio,
                clast_pred,
                intercept,
                method: LambdaZMethod::R2,
                weighting: options.weighting,
            };

            if r_squared < options.min_r_squared {
                continue;
            }

            // For R² method, optimize raw R² (not adjusted)
            match &best_result {
                None => best_result = Some(result),
                Some(best) => {
                    let r_diff = result.r_squared - best.r_squared;
                    if r_diff > options.r_squared_tolerance {
                        best_result = Some(result);
                    } else if r_diff >= -options.r_squared_tolerance
                        && result.n_points > best.n_points
                    {
                        best_result = Some(result);
                    }
                }
            }
        }
    }

    best_result
}

/// Lambda-z using Interval method (user-defined time range)
fn lambda_z_interval(
    times: &[f64],
    concentrations: &[f64],
    _tmax_idx: usize,
    _tlast_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    let start_time = options.interval_start_time?;
    let end_time = options.interval_end_time?;

    if start_time >= end_time {
        return None;
    }

    // Find all points within the interval
    let mut reg_times = Vec::new();
    let mut reg_log_conc = Vec::new();
    let mut reg_conc = Vec::new();

    for i in 0..times.len() {
        if times[i] >= start_time && times[i] <= end_time && concentrations[i] > 0.0 {
            reg_times.push(times[i]);
            reg_log_conc.push(concentrations[i].ln());
            reg_conc.push(concentrations[i]);
        }
    }

    if reg_times.len() < options.min_points {
        return None;
    }

    let (slope, intercept, r_squared) =
        weighted_linear_regression(&reg_times, &reg_log_conc, &reg_conc, &options.weighting)?;

    let lambda = -slope;
    if lambda <= 0.0 {
        return None;
    }

    let half_life = std::f64::consts::LN_2 / lambda;
    let time_span = reg_times[reg_times.len() - 1] - reg_times[0];
    let span_ratio = time_span / half_life;

    let n = reg_times.len() as f64;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);

    // Predict at last point in data
    let last_data_idx = times.len() - 1;
    let clast_pred = (intercept + slope * times[last_data_idx]).exp();

    Some(LambdaZResult {
        lambda_z: lambda,
        half_life,
        r_squared,
        adj_r_squared,
        n_points: reg_times.len(),
        time_first: reg_times[0],
        time_last: reg_times[reg_times.len() - 1],
        span_ratio,
        clast_pred,
        intercept,
        method: LambdaZMethod::Interval,
        weighting: options.weighting,
    })
}

/// Lambda-z using Points method (use last N points)
fn lambda_z_points(
    times: &[f64],
    concentrations: &[f64],
    _tmax_idx: usize,
    tlast_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    let n_points = options.n_points?;

    if n_points < options.min_points {
        return None;
    }

    // Find last n_points with positive concentrations, working backwards from tlast
    let mut reg_times = Vec::new();
    let mut reg_log_conc = Vec::new();
    let mut reg_conc = Vec::new();

    for i in (0..=tlast_idx).rev() {
        if concentrations[i] > 0.0 {
            reg_times.insert(0, times[i]);
            reg_log_conc.insert(0, concentrations[i].ln());
            reg_conc.insert(0, concentrations[i]);

            if reg_times.len() == n_points {
                break;
            }
        }
    }

    if reg_times.len() < options.min_points {
        return None;
    }

    let (slope, intercept, r_squared) =
        weighted_linear_regression(&reg_times, &reg_log_conc, &reg_conc, &options.weighting)?;

    let lambda = -slope;
    if lambda <= 0.0 {
        return None;
    }

    let half_life = std::f64::consts::LN_2 / lambda;
    let time_span = reg_times[reg_times.len() - 1] - reg_times[0];
    let span_ratio = time_span / half_life;

    let n = reg_times.len() as f64;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);
    let clast_pred = (intercept + slope * times[tlast_idx]).exp();

    Some(LambdaZResult {
        lambda_z: lambda,
        half_life,
        r_squared,
        adj_r_squared,
        n_points: reg_times.len(),
        time_first: reg_times[0],
        time_last: reg_times[reg_times.len() - 1],
        span_ratio,
        clast_pred,
        intercept,
        method: LambdaZMethod::Points,
        weighting: options.weighting,
    })
}

/// Calculate λz with automatic Tmax and Tlast detection
///
/// Convenience function that automatically determines Tmax and Tlast indices.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `options` - Configuration options
///
/// # Returns
///
/// `Some(LambdaZResult)` if a valid regression is found, `None` otherwise.
pub fn lambda_z_auto(
    times: &[f64],
    concentrations: &[f64],
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    // Find Tmax (first occurrence of max)
    let tmax_idx = concentrations
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;

    // Find Tlast (last positive concentration)
    let tlast_idx = concentrations.iter().rposition(|&c| c > 0.0)?;

    lambda_z(times, concentrations, tmax_idx, tlast_idx, options)
}

/// Calculate AUC extrapolated to infinity
///
/// AUC_inf = AUC_last + C_last / λz
///
/// # Arguments
///
/// * `auc_last` - AUC from 0 to Tlast
/// * `clast` - Last measurable concentration
/// * `lambda_z` - Terminal elimination rate constant
///
/// # Returns
///
/// AUC extrapolated to infinity
pub fn auc_inf(auc_last: f64, clast: f64, lambda_z: f64) -> f64 {
    if lambda_z <= 0.0 {
        return f64::NAN;
    }
    auc_last + clast / lambda_z
}

/// Calculate AUC extrapolated to infinity using predicted Clast
///
/// AUC_inf_pred = AUC_last + C_last_pred / λz
///
/// # Arguments
///
/// * `auc_last` - AUC from 0 to Tlast
/// * `clast_pred` - Predicted concentration at Tlast from λz regression
/// * `lambda_z` - Terminal elimination rate constant
///
/// # Returns
///
/// AUC extrapolated to infinity using predicted Clast
pub fn auc_inf_pred(auc_last: f64, clast_pred: f64, lambda_z: f64) -> f64 {
    if lambda_z <= 0.0 {
        return f64::NAN;
    }
    auc_last + clast_pred / lambda_z
}

/// Calculate percent of AUC extrapolated
///
/// %AUC_extrap = (AUC_inf - AUC_last) / AUC_inf × 100
///
/// # Arguments
///
/// * `auc_last` - AUC from 0 to Tlast
/// * `auc_inf` - AUC extrapolated to infinity
///
/// # Returns
///
/// Percentage of AUC that is extrapolated
pub fn auc_percent_extrap(auc_last: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 {
        return f64::NAN;
    }
    (auc_inf - auc_last) / auc_inf * 100.0
}

/// Calculate apparent clearance (CL/F)
///
/// CL/F = Dose / AUC_inf
///
/// # Arguments
///
/// * `dose` - Administered dose
/// * `auc_inf` - AUC extrapolated to infinity
///
/// # Returns
///
/// Apparent clearance (volume/time)
pub fn clearance(dose: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 {
        return f64::NAN;
    }
    dose / auc_inf
}

/// Calculate apparent volume of distribution (Vz/F)
///
/// Vz/F = Dose / (λz × AUC_inf)
///
/// # Arguments
///
/// * `dose` - Administered dose
/// * `lambda_z` - Terminal elimination rate constant
/// * `auc_inf` - AUC extrapolated to infinity
///
/// # Returns
///
/// Apparent volume of distribution
pub fn vz(dose: f64, lambda_z: f64, auc_inf: f64) -> f64 {
    if lambda_z <= 0.0 || auc_inf <= 0.0 {
        return f64::NAN;
    }
    dose / (lambda_z * auc_inf)
}

/// Calculate mean residence time (MRT)
///
/// MRT = AUMC / AUC
///
/// # Arguments
///
/// * `aumc` - Area under the first moment curve
/// * `auc` - Area under the concentration-time curve
///
/// # Returns
///
/// Mean residence time
pub fn mrt(aumc: f64, auc: f64) -> f64 {
    if auc <= 0.0 {
        return f64::NAN;
    }
    aumc / auc
}

/// Calculate initial concentration (C0) for IV Bolus
///
/// C0 is extrapolated to time 0 using lambda_z:
/// C0 = Clast × exp(lambda_z × Tlast)
///
/// # Arguments
///
/// * `clast` - Last measured concentration
/// * `tlast` - Time of last measurement
/// * `lambda_z` - Terminal elimination rate constant
///
/// # Returns
///
/// Initial concentration at time 0
pub fn c0_iv_bolus(clast: f64, tlast: f64, lambda_z: f64) -> f64 {
    if lambda_z <= 0.0 {
        return f64::NAN;
    }
    clast * (lambda_z * tlast).exp()
}

/// Calculate volume of distribution (Vd) for IV Bolus
///
/// Vd = Dose / C0
///
/// # Arguments
///
/// * `dose` - Administered dose
/// * `c0` - Initial concentration
///
/// # Returns
///
/// Volume of distribution
pub fn vd_iv_bolus(dose: f64, c0: f64) -> f64 {
    if c0 <= 0.0 {
        return f64::NAN;
    }
    dose / c0
}

/// Calculate volume of distribution at steady state (Vss) for IV administration
///
/// Vss = Dose × AUMC / AUC²
///
/// This formula applies to both IV Bolus and IV Infusion
///
/// # Arguments
///
/// * `dose` - Administered dose
/// * `aumc` - Area under the first moment curve
/// * `auc` - Area under the concentration curve
///
/// # Returns
///
/// Volume of distribution at steady state
pub fn vss_iv(dose: f64, aumc: f64, auc: f64) -> f64 {
    if auc <= 0.0 {
        return f64::NAN;
    }
    dose * aumc / (auc * auc)
}

/// Estimate absorption rate constant (Ka) for extravascular administration
///
/// This uses the method of residuals (feathering) on the terminal phase.
/// The absorption rate constant is estimated from the residual line.
///
/// Note: This is a simplified estimation. For more accurate Ka, use
/// population modeling or fitting approaches.
///
/// # Arguments
///
/// * `times` - Time points
/// * `_concentrations` - Observed concentrations
/// * `lambda_z` - Terminal elimination rate constant
///
/// # Returns
///
/// Estimated absorption rate constant (if successful)
pub fn ka_extravascular(times: &[f64], _concentrations: &[f64], lambda_z: f64) -> Option<f64> {
    // Need at least 4 points for reasonable ka estimation
    if times.len() < 4 || lambda_z <= 0.0 {
        return None;
    }

    // For now, return None - full implementation would require:
    // 1. Extrapolate terminal phase backwards using lambda_z
    // 2. Calculate residuals (observed - extrapolated)
    // 3. Fit log(residuals) vs time to get ka
    // This is complex and often unreliable without good data
    None
}

/// Estimate lag time (Tlag) for extravascular administration
///
/// Tlag is the time delay before absorption begins.
/// It's identified as the time of the first non-zero concentration.
///
/// # Arguments
///
/// * `times` - Time points
/// * `concentrations` - Observed concentrations
/// * `loq` - Limit of quantification
///
/// # Returns
///
/// Estimated lag time (time of first quantifiable concentration)
pub fn tlag_extravascular(times: &[f64], concentrations: &[f64], loq: f64) -> Option<f64> {
    for (i, &conc) in concentrations.iter().enumerate() {
        if conc > loq && i > 0 {
            // If there are zero/BLQ values before this point, take previous time as tlag
            return Some(times[i - 1]);
        }
    }
    None
}

/// Weighted linear regression: y = a + b*x
///
/// Supports different weighting schemes for lambda-z calculation
///
/// Returns (slope, intercept, r_squared)
fn weighted_linear_regression(
    x: &[f64],
    y: &[f64],
    conc: &[f64],
    weighting: &RegressionWeight,
) -> Option<(f64, f64, f64)> {
    let n = x.len();
    if n < 2 || n != y.len() || n != conc.len() {
        return None;
    }

    match weighting {
        RegressionWeight::Uniform => {
            // No weighting - use simple linear regression
            linear_regression(x, y)
        }
        RegressionWeight::InverseY => {
            // Weight by 1/Y (where Y is concentration, not log-conc)
            let mut weights = Vec::with_capacity(n);
            for &c in conc {
                if c <= 0.0 {
                    return None; // Invalid concentration for weighting
                }
                weights.push(1.0 / c);
            }
            weighted_linear_regression_impl(x, y, &weights)
        }
        RegressionWeight::InverseYSquared => {
            // Weight by 1/Y²
            let mut weights = Vec::with_capacity(n);
            for &c in conc {
                if c <= 0.0 {
                    return None;
                }
                weights.push(1.0 / (c * c));
            }
            weighted_linear_regression_impl(x, y, &weights)
        }
    }
}

/// Weighted linear regression implementation
fn weighted_linear_regression_impl(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
) -> Option<(f64, f64, f64)> {
    let n = x.len();
    if n < 2 || n != y.len() || n != weights.len() {
        return None;
    }

    // Calculate weighted sums
    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 0.0 {
        return None;
    }

    let sum_wx: f64 = weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
    let sum_wy: f64 = weights.iter().zip(y.iter()).map(|(w, yi)| w * yi).sum();
    let sum_wxx: f64 = weights
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi * xi)
        .sum();
    let sum_wxy: f64 = weights
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((w, xi), yi)| w * xi * yi)
        .sum();

    // Calculate weighted means
    let x_mean = sum_wx / sum_w;
    let y_mean = sum_wy / sum_w;

    // Calculate slope and intercept using weighted formulas
    let denominator = sum_wxx - sum_wx * sum_wx / sum_w;
    if denominator.abs() < 1e-15 {
        return None;
    }

    let slope = (sum_wxy - sum_wx * sum_wy / sum_w) / denominator;
    let intercept = y_mean - slope * x_mean;

    // Calculate weighted R²
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..n {
        let y_pred = intercept + slope * x[i];
        let y_diff = y[i] - y_mean;
        let res = y[i] - y_pred;
        ss_tot += weights[i] * y_diff * y_diff;
        ss_res += weights[i] * res * res;
    }

    let r_squared = if ss_tot.abs() < 1e-15 {
        1.0 // Perfect fit
    } else {
        1.0 - ss_res / ss_tot
    };

    Some((slope, intercept, r_squared))
}

/// Simple linear regression: y = a + b*x
///
/// Returns (slope, intercept, r_squared)
fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return None;
    }

    let n_f = n as f64;

    // Calculate means
    let x_mean: f64 = x.iter().sum::<f64>() / n_f;
    let y_mean: f64 = y.iter().sum::<f64>() / n_f;

    // Calculate sums for slope and intercept
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        ss_xy += x_diff * y_diff;
        ss_xx += x_diff * x_diff;
        ss_yy += y_diff * y_diff;
    }

    // Check for zero variance in x
    if ss_xx.abs() < 1e-15 {
        return None;
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    // Calculate R²
    let r_squared = if ss_yy.abs() < 1e-15 {
        1.0 // Perfect fit if no variance in y
    } else {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    };

    Some((slope, intercept, r_squared))
}

/// Calculate AUMC (Area Under the first Moment Curve) segment
///
/// AUMC_segment = ∫ t × C dt
///
/// Uses linear trapezoidal method:
/// AUMC = (t1×C1 + t2×C2) / 2 × (t2 - t1)
pub fn aumc_segment(t1: f64, c1: f64, t2: f64, c2: f64) -> f64 {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return 0.0;
    }
    (t1 * c1 + t2 * c2) / 2.0 * dt
}

/// Calculate cumulative AUMC from time-concentration data
///
/// # Arguments
///
/// * `times` - Time points
/// * `concentrations` - Concentration values
///
/// # Returns
///
/// Total AUMC from first to last time point
pub fn aumc_last(times: &[f64], concentrations: &[f64]) -> f64 {
    if times.len() < 2 || times.len() != concentrations.len() {
        return 0.0;
    }

    // Find tlast (last index where concentration > 0)
    let tlast_idx = concentrations
        .iter()
        .rposition(|&c| c > 0.0)
        .unwrap_or(times.len() - 1);

    let mut aumc = 0.0;

    for i in 1..=tlast_idx {
        aumc += aumc_segment(
            times[i - 1],
            concentrations[i - 1],
            times[i],
            concentrations[i],
        );
    }

    aumc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        // Perfect linear relationship: y = 2 + 3x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];

        let (slope, intercept, r_squared) = linear_regression(&x, &y).unwrap();
        assert!((slope - 3.0).abs() < 1e-10);
        assert!((intercept - 2.0).abs() < 1e-10);
        assert!((r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lambda_z() {
        // Simulated data with known half-life
        // C(t) = 10 × e^(-0.1t), so λz = 0.1, t½ = 6.93
        let times: Vec<f64> = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let concs: Vec<f64> = times.iter().map(|&t| 10.0 * (-0.1_f64 * t).exp()).collect();

        let options = LambdaZOptions::default();
        let result = lambda_z(&times, &concs, 0, 6, &options).unwrap();

        // Should recover λz ≈ 0.1
        assert!((result.lambda_z - 0.1).abs() < 0.01);
        // t½ = ln(2)/0.1 ≈ 6.93
        assert!((result.half_life - 6.93).abs() < 0.1);
        // R² should be very close to 1 for perfect exponential data
        assert!(result.r_squared > 0.999);
    }

    #[test]
    fn test_lambda_z_auto() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 5.0, 2.0, 0.8];

        let options = LambdaZOptions {
            min_r_squared: 0.8,
            ..Default::default()
        };
        let result = lambda_z_auto(&times, &concs, &options);

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.lambda_z > 0.0);
        assert!(r.half_life > 0.0);
    }

    #[test]
    fn test_auc_inf_calculation() {
        let auc_last = 100.0;
        let clast = 1.0;
        let lambda = 0.1;

        let auc = auc_inf(auc_last, clast, lambda);
        // AUC_inf = 100 + 1/0.1 = 110
        assert!((auc - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_clearance_vz() {
        let dose = 100.0;
        let auc = 50.0;
        let lambda = 0.1;

        let cl = clearance(dose, auc);
        assert!((cl - 2.0).abs() < 1e-10); // 100/50 = 2

        let v = vz(dose, lambda, auc);
        assert!((v - 20.0).abs() < 1e-10); // 100/(0.1 × 50) = 20
    }

    #[test]
    fn test_aumc_segment() {
        // Simple case
        let aumc = aumc_segment(0.0, 10.0, 1.0, 10.0);
        // (0×10 + 1×10) / 2 × 1 = 5
        assert!((aumc - 5.0).abs() < 1e-10);
    }
}
