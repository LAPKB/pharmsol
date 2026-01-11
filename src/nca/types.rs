//! NCA types: results, options, and configuration structures
//!
//! This module defines all public types for NCA analysis including:
//! - [`NCAResult`]: Complete structured results
//! - [`NCAOptions`]: Configuration options
//! - [`Route`]: Administration route
//! - Parameter group structs

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

// ============================================================================
// Configuration Types
// ============================================================================

/// Complete NCA configuration
///
/// Dose and route information are automatically detected from the data.
/// Use these options to control calculation methods and quality thresholds.
#[derive(Debug, Clone)]
pub struct NCAOptions {
    /// AUC calculation method (default: LinUpLogDown)
    pub auc_method: AUCMethod,

    /// BLQ handling rule (default: Exclude)
    ///
    /// When an observation is censored (`Censor::BLOQ` or `Censor::ALOQ`),
    /// its value represents the quantification limit (lower or upper).
    /// This rule determines how such observations are handled in the analysis.
    ///
    /// Note: ALOQ (Above LOQ) values are currently kept unchanged in the analysis.
    /// This follows PKNCA behavior which also does not explicitly handle ALOQ.
    pub blq_rule: BLQRule,

    /// Terminal phase (λz) estimation options
    pub lambda_z: LambdaZOptions,

    /// Dosing interval for steady-state analysis (None = single-dose)
    pub tau: Option<f64>,

    /// Time interval for partial AUC calculation (start, end)
    ///
    /// If specified, `auc_partial` in the result will contain the AUC
    /// over this interval. Useful for regulatory submissions requiring
    /// AUC over specific time windows (e.g., AUC0-4h).
    pub auc_interval: Option<(f64, f64)>,

    /// C0 estimation methods for IV bolus (tried in order)
    ///
    /// Default: `[Observed, LogSlope, FirstConc]`
    pub c0_methods: Vec<C0Method>,

    /// Which Clast to use for extrapolation to infinity
    pub clast_type: ClastType,

    /// Maximum acceptable AUC extrapolation percentage (default: 20.0)
    pub max_auc_extrap_pct: f64,
}

impl Default for NCAOptions {
    fn default() -> Self {
        Self {
            auc_method: AUCMethod::LinUpLogDown,
            blq_rule: BLQRule::Exclude,
            lambda_z: LambdaZOptions::default(),
            tau: None,
            auc_interval: None,
            c0_methods: vec![C0Method::Observed, C0Method::LogSlope, C0Method::FirstConc],
            clast_type: ClastType::Observed,
            max_auc_extrap_pct: 20.0,
        }
    }
}

impl NCAOptions {
    /// FDA Bioequivalence study defaults
    pub fn bioequivalence() -> Self {
        Self {
            lambda_z: LambdaZOptions {
                min_r_squared: 0.90,
                min_points: 3,
                ..Default::default()
            },
            max_auc_extrap_pct: 20.0,
            ..Default::default()
        }
    }

    /// Lenient settings for sparse/exploratory data
    pub fn sparse() -> Self {
        Self {
            lambda_z: LambdaZOptions {
                min_r_squared: 0.80,
                min_points: 3,
                ..Default::default()
            },
            max_auc_extrap_pct: 30.0,
            ..Default::default()
        }
    }

    /// Set AUC calculation method
    pub fn with_auc_method(mut self, method: AUCMethod) -> Self {
        self.auc_method = method;
        self
    }

    /// Set BLQ handling rule
    ///
    /// Censoring is determined by `Censor` markings on observations (`BLOQ`/`ALOQ`),
    /// not by a numeric threshold. This method sets how censored observations
    /// are handled in the analysis.
    pub fn with_blq_rule(mut self, rule: BLQRule) -> Self {
        self.blq_rule = rule;
        self
    }

    /// Set dosing interval for steady-state analysis
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = Some(tau);
        self
    }

    /// Set time interval for partial AUC calculation
    pub fn with_auc_interval(mut self, start: f64, end: f64) -> Self {
        self.auc_interval = Some((start, end));
        self
    }

    /// Set lambda-z options
    pub fn with_lambda_z(mut self, options: LambdaZOptions) -> Self {
        self.lambda_z = options;
        self
    }

    /// Set minimum R² for lambda-z
    pub fn with_min_r_squared(mut self, min_r_squared: f64) -> Self {
        self.lambda_z.min_r_squared = min_r_squared;
        self
    }

    /// Set C0 estimation methods (tried in order)
    pub fn with_c0_methods(mut self, methods: Vec<C0Method>) -> Self {
        self.c0_methods = methods;
        self
    }

    /// Set which Clast to use for AUCinf extrapolation
    pub fn with_clast_type(mut self, clast_type: ClastType) -> Self {
        self.clast_type = clast_type;
        self
    }
}

/// Lambda-z estimation options
#[derive(Debug, Clone)]
pub struct LambdaZOptions {
    /// Point selection method
    pub method: LambdaZMethod,
    /// Minimum number of points for regression (default: 3)
    pub min_points: usize,
    /// Maximum number of points (None = no limit)
    pub max_points: Option<usize>,
    /// Minimum R² to accept (default: 0.90)
    pub min_r_squared: f64,
    /// Minimum span ratio (default: 2.0)
    pub min_span_ratio: f64,
    /// Whether to include Tmax in regression (default: false)
    pub include_tmax: bool,
    /// Factor added to adjusted R² to prefer more points (default: 0.0001, PKNCA default)
    ///
    /// The scoring formula becomes: adj_r_squared + adj_r_squared_factor * n_points
    /// This allows preferring regressions with more points when R² values are similar.
    pub adj_r_squared_factor: f64,
}

impl Default for LambdaZOptions {
    fn default() -> Self {
        Self {
            method: LambdaZMethod::AdjR2,
            min_points: 3,
            max_points: None,
            min_r_squared: 0.90,
            min_span_ratio: 2.0,
            include_tmax: false,
            adj_r_squared_factor: 0.0001, // PKNCA default
        }
    }
}

/// Lambda-z point selection method
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LambdaZMethod {
    /// Best adjusted R² (recommended)
    #[default]
    AdjR2,
    /// Best raw R²
    R2,
    /// Use specific number of terminal points
    Manual(usize),
}

/// AUC calculation method
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AUCMethod {
    /// Linear trapezoidal rule
    Linear,
    /// Linear up / log down (industry standard)
    #[default]
    LinUpLogDown,
    /// Linear before Tmax, log-linear after Tmax (PKNCA "lin-log")
    ///
    /// Uses linear trapezoidal before and at Tmax, then log-linear for
    /// descending portions after Tmax. Falls back to linear if either
    /// concentration is zero or non-positive.
    LinLog,
}

/// BLQ (Below Limit of Quantification) handling rule
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum BLQRule {
    /// Replace BLQ with zero
    Zero,
    /// Replace BLQ with LOQ/2
    LoqOver2,
    /// Exclude BLQ values from analysis
    #[default]
    Exclude,
    /// Position-aware handling (PKNCA default): first=keep(0), middle=drop, last=keep(0)
    ///
    /// This is the FDA-recommended approach that:
    /// - Keeps first BLQ (before tfirst) as 0 to anchor the profile start
    /// - Drops middle BLQ (between tfirst and tlast) to avoid deflating AUC
    /// - Keeps last BLQ (at/after tlast) as 0 to define profile end
    Positional,
    /// Tmax-relative handling: different rules before vs after Tmax
    ///
    /// Contains (before_tmax_rule, after_tmax_rule) where each rule can be:
    /// - "keep" = keep as 0
    /// - "drop" = exclude from analysis
    /// Default PKNCA: before.tmax=drop, after.tmax=keep
    TmaxRelative {
        /// Rule for BLQ before Tmax: true=keep as 0, false=drop
        before_tmax_keep: bool,
        /// Rule for BLQ at or after Tmax: true=keep as 0, false=drop
        after_tmax_keep: bool,
    },
}

/// Action to take for a BLQ observation based on position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlqAction {
    Keep,
    Drop,
}

/// Administration route
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// Intravenous bolus
    IVBolus,
    /// Intravenous infusion
    IVInfusion,
    /// Extravascular (oral, SC, IM, etc.)
    #[default]
    Extravascular,
}

/// C0 (initial concentration) estimation method for IV bolus
///
/// Methods are tried in order until one succeeds. Default cascade:
/// `[Observed, LogSlope, FirstConc]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0Method {
    /// Use observed concentration at dose time if present and non-zero
    Observed,
    /// Semilog back-extrapolation from first two positive concentrations
    LogSlope,
    /// Use first positive concentration after dose time
    FirstConc,
    /// Use minimum positive concentration (for IV infusion steady-state)
    Cmin,
    /// Set C0 = 0 (for extravascular where C0 doesn't exist)
    Zero,
}

/// Which Clast value to use for extrapolation to infinity
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClastType {
    /// Use observed Clast (AUCinf,obs)
    #[default]
    Observed,
    /// Use predicted Clast from λz regression (AUCinf,pred)
    Predicted,
}

// ============================================================================
// Result Types
// ============================================================================

/// Complete NCA result with logical parameter grouping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NCAResult {
    /// Subject identifier
    pub subject_id: Option<String>,
    /// Occasion index
    pub occasion: Option<usize>,

    /// Core exposure parameters (always computed)
    pub exposure: ExposureParams,

    /// Terminal phase parameters (if λz succeeds)
    pub terminal: Option<TerminalParams>,

    /// Clearance parameters (if dose + λz available)
    pub clearance: Option<ClearanceParams>,

    /// IV Bolus-specific parameters
    pub iv_bolus: Option<IVBolusParams>,

    /// IV Infusion-specific parameters
    pub iv_infusion: Option<IVInfusionParams>,

    /// Steady-state parameters (if tau specified)
    pub steady_state: Option<SteadyStateParams>,

    /// Quality metrics and warnings
    pub quality: Quality,
}

impl NCAResult {
    /// Get half-life if available
    pub fn half_life(&self) -> Option<f64> {
        self.terminal.as_ref().map(|t| t.half_life)
    }

    /// Flatten result to parameter name-value pairs for export
    pub fn to_params(&self) -> HashMap<&'static str, f64> {
        let mut p = HashMap::new();

        p.insert("cmax", self.exposure.cmax);
        p.insert("tmax", self.exposure.tmax);
        p.insert("clast", self.exposure.clast);
        p.insert("tlast", self.exposure.tlast);
        p.insert("auc_last", self.exposure.auc_last);

        if let Some(ref t) = self.terminal {
            p.insert("lambda_z", t.lambda_z);
            p.insert("half_life", t.half_life);
        }

        if let Some(ref c) = self.clearance {
            p.insert("cl_f", c.cl_f);
            p.insert("vz_f", c.vz_f);
        }

        p
    }
}

impl fmt::Display for NCAResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════╗")?;
        writeln!(f, "║           NCA Results                ║")?;
        writeln!(f, "╠══════════════════════════════════════╣")?;

        if let Some(ref id) = self.subject_id {
            writeln!(f, "║ Subject: {:<27} ║", id)?;
        }
        if let Some(occ) = self.occasion {
            writeln!(f, "║ Occasion: {:<26} ║", occ)?;
        }

        writeln!(f, "╠══════════════════════════════════════╣")?;
        writeln!(f, "║ EXPOSURE                             ║")?;
        writeln!(
            f,
            "║   Cmax:    {:>10.4} at Tmax={:<6.2} ║",
            self.exposure.cmax, self.exposure.tmax
        )?;
        writeln!(
            f,
            "║   AUClast: {:>10.4}               ║",
            self.exposure.auc_last
        )?;
        writeln!(
            f,
            "║   Clast:   {:>10.4} at Tlast={:<5.2}║",
            self.exposure.clast, self.exposure.tlast
        )?;

        if let Some(ref t) = self.terminal {
            writeln!(f, "╠══════════════════════════════════════╣")?;
            writeln!(f, "║ TERMINAL                             ║")?;
            writeln!(f, "║   λz:      {:>10.5}               ║", t.lambda_z)?;
            writeln!(f, "║   t½:      {:>10.2}               ║", t.half_life)?;
            if let Some(ref reg) = t.regression {
                writeln!(f, "║   R²:      {:>10.4}               ║", reg.r_squared)?;
            }
        }

        if let Some(ref c) = self.clearance {
            writeln!(f, "╠══════════════════════════════════════╣")?;
            writeln!(f, "║ CLEARANCE                            ║")?;
            writeln!(f, "║   CL/F:    {:>10.4}               ║", c.cl_f)?;
            writeln!(f, "║   Vz/F:    {:>10.4}               ║", c.vz_f)?;
        }

        if !self.quality.warnings.is_empty() {
            writeln!(f, "╠══════════════════════════════════════╣")?;
            writeln!(f, "║ WARNINGS                             ║")?;
            for w in &self.quality.warnings {
                writeln!(f, "║   • {:<32} ║", format!("{:?}", w))?;
            }
        }

        writeln!(f, "╚══════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Core exposure parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureParams {
    /// Maximum observed concentration
    pub cmax: f64,
    /// Time of maximum concentration
    pub tmax: f64,
    /// Last quantifiable concentration
    pub clast: f64,
    /// Time of last quantifiable concentration
    pub tlast: f64,
    /// AUC from time 0 to Tlast
    pub auc_last: f64,
    /// AUC extrapolated to infinity
    pub auc_inf: Option<f64>,
    /// Percentage of AUC extrapolated
    pub auc_pct_extrap: Option<f64>,
    /// Partial AUC (if requested)
    pub auc_partial: Option<f64>,
    /// AUMC from time 0 to Tlast
    pub aumc_last: Option<f64>,
    /// AUMC extrapolated to infinity
    pub aumc_inf: Option<f64>,
    /// Lag time (extravascular only)
    pub tlag: Option<f64>,
}

/// Terminal phase parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalParams {
    /// Terminal elimination rate constant
    pub lambda_z: f64,
    /// Terminal half-life
    pub half_life: f64,
    /// Mean residence time
    pub mrt: Option<f64>,
    /// Regression statistics
    pub regression: Option<RegressionStats>,
}

/// Regression statistics for λz estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionStats {
    /// Coefficient of determination
    pub r_squared: f64,
    /// Adjusted R²
    pub adj_r_squared: f64,
    /// Number of points used
    pub n_points: usize,
    /// First time point in regression
    pub time_first: f64,
    /// Last time point in regression
    pub time_last: f64,
    /// Span ratio
    pub span_ratio: f64,
}

/// Clearance parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceParams {
    /// Apparent clearance (CL/F)
    pub cl_f: f64,
    /// Apparent volume of distribution (Vz/F)
    pub vz_f: f64,
    /// Volume at steady state (for IV)
    pub vss: Option<f64>,
}

/// IV Bolus-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVBolusParams {
    /// Back-extrapolated initial concentration
    pub c0: f64,
    /// Volume of distribution
    pub vd: f64,
    /// Volume at steady state
    pub vss: Option<f64>,
}

/// IV Infusion-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVInfusionParams {
    /// Infusion duration
    pub infusion_duration: f64,
    /// MRT corrected for infusion
    pub mrt_iv: Option<f64>,
    /// Volume at steady state
    pub vss: Option<f64>,
}

/// Steady-state parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteadyStateParams {
    /// Dosing interval
    pub tau: f64,
    /// AUC over dosing interval
    pub auc_tau: f64,
    /// Minimum concentration
    pub cmin: f64,
    /// Maximum concentration at steady state
    pub cmax_ss: f64,
    /// Average concentration
    pub cavg: f64,
    /// Percent fluctuation
    pub fluctuation: f64,
    /// Swing
    pub swing: f64,
    /// Accumulation ratio
    pub accumulation: Option<f64>,
}

/// Quality metrics and warnings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Quality {
    /// List of warnings
    pub warnings: Vec<Warning>,
}

/// NCA analysis warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Warning {
    /// High AUC extrapolation
    HighExtrapolation,
    /// Poor lambda-z fit
    PoorFit,
    /// Lambda-z could not be estimated
    LambdaZNotEstimable,
    /// Short terminal phase
    ShortTerminalPhase,
    /// Low Cmax
    LowCmax,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nca_options_default() {
        let opts = NCAOptions::default();
        assert_eq!(opts.auc_method, AUCMethod::LinUpLogDown);
        assert_eq!(opts.blq_rule, BLQRule::Exclude);
        assert!(opts.tau.is_none());
        assert_eq!(opts.max_auc_extrap_pct, 20.0);
    }

    #[test]
    fn test_nca_options_builder() {
        let opts = NCAOptions::default()
            .with_auc_method(AUCMethod::Linear)
            .with_blq_rule(BLQRule::LoqOver2)
            .with_tau(24.0)
            .with_min_r_squared(0.95);

        assert_eq!(opts.auc_method, AUCMethod::Linear);
        assert_eq!(opts.blq_rule, BLQRule::LoqOver2);
        assert_eq!(opts.tau, Some(24.0));
        assert_eq!(opts.lambda_z.min_r_squared, 0.95);
    }

    #[test]
    fn test_nca_options_presets() {
        let be = NCAOptions::bioequivalence();
        assert_eq!(be.lambda_z.min_r_squared, 0.90);
        assert_eq!(be.max_auc_extrap_pct, 20.0);

        let sparse = NCAOptions::sparse();
        assert_eq!(sparse.lambda_z.min_r_squared, 0.80);
        assert_eq!(sparse.max_auc_extrap_pct, 30.0);
    }
}
