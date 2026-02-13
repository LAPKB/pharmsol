//! NCA types: results, options, and configuration structures
//!
//! This module defines all public types for NCA analysis including:
//! - [`NCAResult`]: Complete structured results
//! - [`NCAOptions`]: Configuration options
//! - [`Route`]: Administration route
//! - Parameter group structs

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

use crate::data::event::{AUCMethod, BLQRule, Route};

// ============================================================================
// Configuration Types
// ============================================================================

/// Complete NCA configuration
///
/// Dose and route information are automatically detected from the data.
/// Use these options to control calculation methods and quality thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Maximum acceptable AUC extrapolation percentage (default: 20.0)
    pub max_auc_extrap_pct: f64,

    /// Target concentration for time-above-concentration calculation (None = skip)
    ///
    /// When specified, the result will contain `time_above_mic` — the total time
    /// the concentration profile is above this threshold. Uses linear interpolation
    /// at crossing points. Commonly set to MIC for antibiotics.
    pub concentration_threshold: Option<f64>,

    /// Override the auto-detected route
    ///
    /// By default, the administration route is inferred from dose events
    /// (compartment number). Set this to override the heuristic when the
    /// auto-detection gives wrong results (e.g., models where compartment 1
    /// is a depot, not central).
    pub route_override: Option<Route>,

    /// Output equation index to analyze (default: 0)
    ///
    /// For multi-output models, select which output equation to run NCA on.
    pub outeq: usize,

    /// Dose times for multi-dose NCA (None = single-dose)
    ///
    /// When set, AUC/Cmax/Tmax will be computed for each dosing interval
    /// and stored in [`NCAResult::multi_dose`].
    pub dose_times: Option<Vec<f64>>,
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
            max_auc_extrap_pct: 20.0,
            concentration_threshold: None,
            route_override: None,
            outeq: 0,
            dose_times: None,
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

    /// Set a target concentration threshold for time-above-concentration
    ///
    /// When set, the result will include `time_above_mic` — the total time
    /// the profile is above this concentration.
    pub fn with_concentration_threshold(mut self, threshold: f64) -> Self {
        self.concentration_threshold = Some(threshold);
        self
    }

    /// Override the auto-detected route
    ///
    /// Use this when the auto-detection from compartment numbers gives wrong
    /// results. For example, if your model uses compartment 1 as a depot
    /// (not central), the auto-detection would incorrectly classify it as IV.
    pub fn with_route(mut self, route: Route) -> Self {
        self.route_override = Some(route);
        self
    }

    /// Set output equation index (default: 0)
    pub fn with_outeq(mut self, outeq: usize) -> Self {
        self.outeq = outeq;
        self
    }

    /// Set dose times for multi-dose NCA (interval-based AUC, Cmax, Tmax)
    ///
    /// When set, `analyze` will compute AUC, Cmax, and Tmax for each dosing
    /// interval and store them in [`NCAResult::multi_dose`].
    pub fn with_dose_times(mut self, times: Vec<f64>) -> Self {
        self.dose_times = Some(times);
        self
    }
}

/// Lambda-z estimation options
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Indices of observation points to exclude from λz regression
    ///
    /// These are indices into the observation profile (0-based). Points at these
    /// indices will be skipped when fitting the terminal log-linear regression.
    /// Useful for analyst-directed exclusion of outlier points.
    pub exclude_indices: Vec<usize>,
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
            exclude_indices: Vec::new(),
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

    /// Total dose amount (None if no dose events)
    pub dose_amount: Option<f64>,
    /// Administration route (auto-detected or overridden)
    pub route: Option<Route>,
    /// Infusion duration (None for bolus/extravascular)
    pub infusion_duration: Option<f64>,

    /// Core exposure parameters (always computed)
    pub exposure: ExposureParams,

    /// Terminal phase parameters (if λz succeeds)
    pub terminal: Option<TerminalParams>,

    /// Clearance parameters (if dose + λz available)
    pub clearance: Option<ClearanceParams>,

    /// Route-specific parameters (IV bolus, IV infusion, or extravascular)
    pub route_params: Option<RouteParams>,

    /// Steady-state parameters (if tau specified)
    pub steady_state: Option<SteadyStateParams>,

    /// Multi-dose interval parameters (if dose_times specified)
    pub multi_dose: Option<MultiDoseParams>,

    /// Quality metrics and warnings
    pub quality: Quality,
}

impl NCAResult {
    /// Get half-life if available
    pub fn half_life(&self) -> Option<f64> {
        self.terminal.as_ref().map(|t| t.half_life)
    }

    /// C0 (IV Bolus only) — back-extrapolated initial concentration
    pub fn c0(&self) -> Option<f64> {
        match &self.route_params {
            Some(RouteParams::IVBolus(p)) => Some(p.c0),
            _ => None,
        }
    }

    /// Volume of distribution by back-extrapolated C0 (IV Bolus only)
    pub fn vd(&self) -> Option<f64> {
        match &self.route_params {
            Some(RouteParams::IVBolus(p)) => Some(p.vd),
            _ => None,
        }
    }

    /// Volume of distribution at steady state (from [`ClearanceParams`])
    pub fn vss(&self) -> Option<f64> {
        self.clearance.as_ref().and_then(|c| c.vss)
    }

    /// Concentration at end of infusion (IV Infusion only)
    pub fn ceoi(&self) -> Option<f64> {
        match &self.route_params {
            Some(RouteParams::IVInfusion(p)) => p.ceoi,
            _ => None,
        }
    }

    /// MRT for IV Infusion (adjusted for infusion time)
    pub fn mrt_iv(&self) -> Option<f64> {
        match &self.route_params {
            Some(RouteParams::IVInfusion(p)) => p.mrt_iv,
            _ => None,
        }
    }

    /// Flatten result to parameter name-value pairs for export
    pub fn to_params(&self) -> HashMap<&'static str, f64> {
        let mut p = HashMap::new();

        // Exposure
        p.insert("cmax", self.exposure.cmax);
        p.insert("tmax", self.exposure.tmax);
        p.insert("clast", self.exposure.clast);
        p.insert("tlast", self.exposure.tlast);
        if let Some(v) = self.exposure.tfirst {
            p.insert("tfirst", v);
        }
        p.insert("auc_last", self.exposure.auc_last);
        if let Some(v) = self.exposure.auc_inf_obs {
            p.insert("auc_inf_obs", v);
        }
        if let Some(v) = self.exposure.auc_inf_pred {
            p.insert("auc_inf_pred", v);
        }
        if let Some(v) = self.exposure.auc_pct_extrap_obs {
            p.insert("auc_pct_extrap_obs", v);
        }
        if let Some(v) = self.exposure.auc_pct_extrap_pred {
            p.insert("auc_pct_extrap_pred", v);
        }
        if let Some(v) = self.exposure.auc_partial {
            p.insert("auc_partial", v);
        }
        if let Some(v) = self.exposure.aumc_last {
            p.insert("aumc_last", v);
        }
        if let Some(v) = self.exposure.aumc_inf {
            p.insert("aumc_inf", v);
        }
        if let Some(v) = self.exposure.tlag {
            p.insert("tlag", v);
        }

        // Dose-normalized
        if let Some(v) = self.exposure.cmax_dn {
            p.insert("cmax_dn", v);
        }
        if let Some(v) = self.exposure.auc_last_dn {
            p.insert("auc_last_dn", v);
        }
        if let Some(v) = self.exposure.auc_inf_dn {
            p.insert("auc_inf_dn", v);
        }

        if let Some(v) = self.exposure.time_above_mic {
            p.insert("time_above_mic", v);
        }

        // Dose
        if let Some(v) = self.dose_amount {
            p.insert("dose", v);
        }

        // Terminal
        if let Some(ref t) = self.terminal {
            p.insert("lambda_z", t.lambda_z);
            p.insert("half_life", t.half_life);
            if let Some(mrt) = t.mrt {
                p.insert("mrt", mrt);
            }
            if let Some(eff_hl) = t.effective_half_life {
                p.insert("effective_half_life", eff_hl);
            }
            if let Some(kel) = t.kel {
                p.insert("kel", kel);
            }
            if let Some(ref reg) = t.regression {
                if reg.corrxy.is_finite() {
                    p.insert("lambda_z_corrxy", reg.corrxy);
                }
            }
        }

        // Clearance
        if let Some(ref c) = self.clearance {
            p.insert("cl_f", c.cl_f);
            p.insert("vz_f", c.vz_f);
            if let Some(vss) = c.vss {
                p.insert("vss", vss);
            }
        }

        // Route-specific
        if let Some(ref rp) = self.route_params {
            match rp {
                RouteParams::IVBolus(ref b) => {
                    p.insert("c0", b.c0);
                    p.insert("vd", b.vd);
                }
                RouteParams::IVInfusion(ref inf) => {
                    p.insert("infusion_duration", inf.infusion_duration);
                    if let Some(mrt_iv) = inf.mrt_iv {
                        p.insert("mrt_iv", mrt_iv);
                    }
                    if let Some(ceoi) = inf.ceoi {
                        p.insert("ceoi", ceoi);
                    }
                }
                RouteParams::Extravascular => {}
            }
        }

        // Steady-state
        if let Some(ref ss) = self.steady_state {
            p.insert("tau", ss.tau);
            p.insert("auc_tau", ss.auc_tau);
            p.insert("cmin", ss.cmin);
            p.insert("cmax_ss", ss.cmax_ss);
            p.insert("cavg", ss.cavg);
            p.insert("fluctuation", ss.fluctuation);
            p.insert("swing", ss.swing);
            p.insert("peak_trough_ratio", ss.peak_trough_ratio);
            if let Some(acc) = ss.accumulation {
                p.insert("accumulation", acc);
            }
        }

        p
    }

    /// Flatten result to ordered key-value pairs
    ///
    /// Unlike [`to_params()`](Self::to_params) which returns a HashMap, this returns
    /// a `Vec` with a canonical ordering suitable for tabular display. Optional
    /// parameters that are absent produce `None` values.
    ///
    /// The ordering follows PK reporting convention:
    /// exposure → terminal → clearance → route-specific → steady-state → dose-normalized → quality
    pub fn to_row(&self) -> Vec<(&'static str, Option<f64>)> {
        let mut row = Vec::with_capacity(40);

        // Exposure
        row.push(("cmax", Some(self.exposure.cmax)));
        row.push(("tmax", Some(self.exposure.tmax)));
        row.push(("clast", Some(self.exposure.clast)));
        row.push(("tlast", Some(self.exposure.tlast)));
        row.push(("tfirst", self.exposure.tfirst));
        row.push(("auc_last", Some(self.exposure.auc_last)));
        row.push(("auc_inf_obs", self.exposure.auc_inf_obs));
        row.push(("auc_inf_pred", self.exposure.auc_inf_pred));
        row.push(("auc_pct_extrap_obs", self.exposure.auc_pct_extrap_obs));
        row.push(("auc_pct_extrap_pred", self.exposure.auc_pct_extrap_pred));
        row.push(("auc_partial", self.exposure.auc_partial));
        row.push(("aumc_last", self.exposure.aumc_last));
        row.push(("aumc_inf", self.exposure.aumc_inf));
        row.push(("tlag", self.exposure.tlag));

        // Terminal
        if let Some(ref t) = self.terminal {
            row.push(("lambda_z", Some(t.lambda_z)));
            row.push(("half_life", Some(t.half_life)));
            row.push(("mrt", t.mrt));
            row.push(("effective_half_life", t.effective_half_life));
            row.push(("kel", t.kel));
        } else {
            row.push(("lambda_z", None));
            row.push(("half_life", None));
            row.push(("mrt", None));
            row.push(("effective_half_life", None));
            row.push(("kel", None));
        }

        // Clearance
        if let Some(ref c) = self.clearance {
            row.push(("cl_f", Some(c.cl_f)));
            row.push(("vz_f", Some(c.vz_f)));
            row.push(("vss", c.vss));
        } else {
            row.push(("cl_f", None));
            row.push(("vz_f", None));
            row.push(("vss", None));
        }

        // Route-specific — always emit all columns, None when not applicable
        match self.route_params.as_ref() {
            Some(RouteParams::IVBolus(ref b)) => {
                row.push(("c0", Some(b.c0)));
                row.push(("vd", Some(b.vd)));
                row.push(("infusion_duration", None));
                row.push(("ceoi", None));
            }
            Some(RouteParams::IVInfusion(ref inf)) => {
                row.push(("c0", None));
                row.push(("vd", None));
                row.push(("infusion_duration", Some(inf.infusion_duration)));
                row.push(("ceoi", inf.ceoi));
            }
            Some(RouteParams::Extravascular) | None => {
                row.push(("c0", None));
                row.push(("vd", None));
                row.push(("infusion_duration", None));
                row.push(("ceoi", None));
            }
        }

        // Steady-state — always emit all columns
        if let Some(ref ss) = self.steady_state {
            row.push(("tau", Some(ss.tau)));
            row.push(("auc_tau", Some(ss.auc_tau)));
            row.push(("cmin", Some(ss.cmin)));
            row.push(("cmax_ss", Some(ss.cmax_ss)));
            row.push(("cavg", Some(ss.cavg)));
            row.push(("fluctuation", Some(ss.fluctuation)));
            row.push(("swing", Some(ss.swing)));
            row.push(("peak_trough_ratio", Some(ss.peak_trough_ratio)));
            row.push(("accumulation", ss.accumulation));
        } else {
            row.push(("tau", None));
            row.push(("auc_tau", None));
            row.push(("cmin", None));
            row.push(("cmax_ss", None));
            row.push(("cavg", None));
            row.push(("fluctuation", None));
            row.push(("swing", None));
            row.push(("peak_trough_ratio", None));
            row.push(("accumulation", None));
        }

        // Dose-normalized
        row.push(("cmax_dn", self.exposure.cmax_dn));
        row.push(("auc_last_dn", self.exposure.auc_last_dn));
        row.push(("auc_inf_dn", self.exposure.auc_inf_dn));
        row.push(("time_above_mic", self.exposure.time_above_mic));

        // Dose
        row.push(("dose", self.dose_amount));

        row
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
        if let Some(amount) = self.dose_amount {
            let route_str = self
                .route
                .map(|r| format!("{:?}", r))
                .unwrap_or_else(|| "Unknown".to_string());
            writeln!(
                f,
                "║ Dose: {:<30} ║",
                format!("{:.2} ({})", amount, route_str)
            )?;
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
        if let Some(v) = self.exposure.auc_inf_obs {
            writeln!(f, "║   AUCinf(obs):  {:>10.4}          ║", v)?;
        }
        if let Some(v) = self.exposure.auc_inf_pred {
            writeln!(f, "║   AUCinf(pred): {:>10.4}          ║", v)?;
        }
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
            if let Some(eff_hl) = t.effective_half_life {
                writeln!(f, "║   t½eff:   {:>10.2}               ║", eff_hl)?;
            }
            if let Some(kel) = t.kel {
                writeln!(f, "║   Kel:     {:>10.5}               ║", kel)?;
            }
            if let Some(ref reg) = t.regression {
                writeln!(f, "║   R²:      {:>10.4}               ║", reg.r_squared)?;
                if reg.corrxy.is_finite() {
                    writeln!(f, "║   corrxy:  {:>10.4}               ║", reg.corrxy)?;
                }
            }
        }

        if let Some(ref c) = self.clearance {
            writeln!(f, "╠══════════════════════════════════════╣")?;
            writeln!(f, "║ CLEARANCE                            ║")?;
            writeln!(f, "║   CL/F:    {:>10.4}               ║", c.cl_f)?;
            writeln!(f, "║   Vz/F:    {:>10.4}               ║", c.vz_f)?;
        }

        if let Some(ref rp) = self.route_params {
            match rp {
                RouteParams::IVBolus(ref b) => {
                    writeln!(f, "╠══════════════════════════════════════╣")?;
                    writeln!(f, "║ IV BOLUS                             ║")?;
                    writeln!(f, "║   C0:      {:>10.4}               ║", b.c0)?;
                    writeln!(f, "║   Vd:      {:>10.4}               ║", b.vd)?;
                }
                RouteParams::IVInfusion(ref inf) => {
                    writeln!(f, "╠══════════════════════════════════════╣")?;
                    writeln!(f, "║ IV INFUSION                          ║")?;
                    writeln!(
                        f,
                        "║   Dur:     {:>10.4}               ║",
                        inf.infusion_duration
                    )?;
                }
                RouteParams::Extravascular => {}
            }
        }

        if !self.quality.warnings.is_empty() {
            writeln!(f, "╠══════════════════════════════════════╣")?;
            writeln!(f, "║ WARNINGS                             ║")?;
            for w in &self.quality.warnings {
                writeln!(f, "║   • {:<32} ║", format!("{}", w))?;
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
    /// First measurable (positive) concentration time
    pub tfirst: Option<f64>,
    /// AUC from time 0 to Tlast
    pub auc_last: f64,
    /// AUC extrapolated to infinity using observed Clast
    pub auc_inf_obs: Option<f64>,
    /// AUC extrapolated to infinity using predicted Clast (from λz regression)
    pub auc_inf_pred: Option<f64>,
    /// Percentage of AUC extrapolated (observed Clast)
    pub auc_pct_extrap_obs: Option<f64>,
    /// Percentage of AUC extrapolated (predicted Clast)
    pub auc_pct_extrap_pred: Option<f64>,
    /// Partial AUC (if requested)
    pub auc_partial: Option<f64>,
    /// AUMC from time 0 to Tlast
    pub aumc_last: Option<f64>,
    /// AUMC extrapolated to infinity
    pub aumc_inf: Option<f64>,
    /// Lag time (extravascular only)
    pub tlag: Option<f64>,

    // Dose-normalized parameters (computed when dose > 0)
    /// Cmax normalized by dose (Cmax / dose)
    pub cmax_dn: Option<f64>,
    /// AUClast normalized by dose (AUClast / dose)
    pub auc_last_dn: Option<f64>,
    /// AUCinf(obs) normalized by dose (AUCinf_obs / dose)
    pub auc_inf_dn: Option<f64>,

    /// Total time above a concentration threshold (e.g., MIC)
    ///
    /// Only computed when [`NCAOptions::concentration_threshold`] is set.
    pub time_above_mic: Option<f64>,
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
    /// Effective half-life: ln(2) × MRT
    pub effective_half_life: Option<f64>,
    /// Elimination rate constant: 1 / MRT
    pub kel: Option<f64>,
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
    /// Pearson correlation coefficient (corrxy) — negative for terminal elimination
    pub corrxy: f64,
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
///
/// Note: Volume of distribution at steady state (Vss) is computed from clearance
/// and is therefore located in [`ClearanceParams::vss`], not here. Use
/// [`NCAResult::vss()`] for convenient access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVBolusParams {
    /// Back-extrapolated initial concentration
    pub c0: f64,
    /// Volume of distribution
    pub vd: f64,
    /// Which C0 estimation method succeeded
    pub c0_method: Option<C0Method>,
}

/// IV Infusion-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVInfusionParams {
    /// Infusion duration
    pub infusion_duration: f64,
    /// MRT corrected for infusion
    pub mrt_iv: Option<f64>,
    /// Concentration at end of infusion
    pub ceoi: Option<f64>,
}

/// Route-specific NCA parameters
///
/// Replaces separate `iv_bolus`/`iv_infusion` fields with a single discriminated union.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouteParams {
    /// IV bolus route with back-extrapolated C0, Vd, and optional Vss
    IVBolus(IVBolusParams),
    /// IV infusion route with infusion duration, MRT correction, and optional Vss
    IVInfusion(IVInfusionParams),
    /// Extravascular route (oral, SC, IM, etc.) — tlag is in [`ExposureParams`]
    Extravascular,
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
    /// Peak-to-trough ratio (Cmax / Cmin)
    pub peak_trough_ratio: f64,
    /// Accumulation ratio (AUC_tau / AUC_inf from single dose)
    pub accumulation: Option<f64>,
}

/// Per-interval parameters for multi-dose NCA
///
/// Computed when [`NCAOptions::dose_times`] is set. Contains AUC, Cmax, and Tmax
/// for each dosing interval defined by consecutive dose times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDoseParams {
    /// Dose time marking the start of each interval
    pub dose_times: Vec<f64>,
    /// AUC for each dosing interval (dose_i → dose_{i+1}, or dose_last → tlast)
    pub auc_intervals: Vec<f64>,
    /// Cmax within each dosing interval
    pub cmax_intervals: Vec<f64>,
    /// Tmax within each dosing interval
    pub tmax_intervals: Vec<f64>,
}

/// Quality metrics and warnings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Quality {
    /// List of warnings
    pub warnings: Vec<Warning>,
}

impl Quality {
    /// Get only critical warnings (errors that may invalidate results)
    pub fn errors(&self) -> Vec<&Warning> {
        self.warnings
            .iter()
            .filter(|w| w.severity() == Severity::Error)
            .collect()
    }

    /// Get non-critical warnings (suboptimal but usable results)
    pub fn warnings_only(&self) -> Vec<&Warning> {
        self.warnings
            .iter()
            .filter(|w| w.severity() == Severity::Warning)
            .collect()
    }

    /// Get informational notices
    pub fn info(&self) -> Vec<&Warning> {
        self.warnings
            .iter()
            .filter(|w| w.severity() == Severity::Info)
            .collect()
    }

    /// Check if any critical errors are present
    pub fn has_errors(&self) -> bool {
        self.warnings
            .iter()
            .any(|w| w.severity() == Severity::Error)
    }
}

/// Severity level for NCA warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational — results are valid but of note
    Info,
    /// Warning — results are usable but suboptimal
    Warning,
    /// Error — results may be invalid or analysis failed
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

/// NCA analysis warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Warning {
    /// AUC extrapolation percentage exceeds threshold
    HighExtrapolation {
        /// Actual extrapolation percentage
        pct: f64,
        /// Configured threshold
        threshold: f64,
    },
    /// Poor lambda-z regression fit
    PoorFit {
        /// Actual R² value
        r_squared: f64,
        /// Minimum required R²
        threshold: f64,
    },
    /// Lambda-z could not be estimated
    LambdaZNotEstimable,
    /// Terminal phase span ratio too short
    ShortTerminalPhase {
        /// Actual span ratio
        span_ratio: f64,
        /// Minimum required span ratio
        threshold: f64,
    },
    /// Cmax is zero or negative
    LowCmax,
    /// Multiple routes detected in a single occasion without explicit override
    MixedRoutes {
        /// Routes detected in the occasion
        routes: Vec<Route>,
    },
}

impl Warning {
    /// Get the severity level of this warning
    ///
    /// - **Error**: `LambdaZNotEstimable`, `LowCmax` — analysis may be invalid
    /// - **Warning**: `HighExtrapolation`, `PoorFit` — results usable but suboptimal
    /// - **Info**: `ShortTerminalPhase` — informational only
    pub fn severity(&self) -> Severity {
        match self {
            Warning::LambdaZNotEstimable | Warning::LowCmax => Severity::Error,
            Warning::HighExtrapolation { .. } | Warning::PoorFit { .. } => Severity::Warning,
            Warning::ShortTerminalPhase { .. } | Warning::MixedRoutes { .. } => Severity::Info,
        }
    }
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::HighExtrapolation { pct, threshold } => {
                write!(
                    f,
                    "AUC extrapolation {:.1}% exceeds {:.1}% threshold",
                    pct, threshold
                )
            }
            Warning::PoorFit {
                r_squared,
                threshold,
            } => {
                write!(f, "λz R²={:.4} below minimum {:.4}", r_squared, threshold)
            }
            Warning::LambdaZNotEstimable => write!(f, "λz could not be estimated"),
            Warning::ShortTerminalPhase {
                span_ratio,
                threshold,
            } => {
                write!(
                    f,
                    "Terminal phase span ratio {:.2} below minimum {:.2}",
                    span_ratio, threshold
                )
            }
            Warning::LowCmax => write!(f, "Cmax ≤ 0"),
            Warning::MixedRoutes { routes } => {
                write!(f, "Mixed routes detected: {:?}", routes)
            }
        }
    }
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

    /// Helper: minimal NCAResult with given route_params and clearance
    fn make_result_with(
        route_params: Option<RouteParams>,
        clearance: Option<ClearanceParams>,
    ) -> NCAResult {
        NCAResult {
            subject_id: None,
            occasion: None,
            dose_amount: Some(100.0),
            route: Some(crate::data::Route::Extravascular),
            infusion_duration: None,
            exposure: ExposureParams {
                cmax: 10.0,
                tmax: 1.0,
                clast: 1.0,
                tlast: 8.0,
                tfirst: None,
                auc_last: 50.0,
                auc_inf_obs: None,
                auc_inf_pred: None,
                auc_pct_extrap_obs: None,
                auc_pct_extrap_pred: None,
                auc_partial: None,
                aumc_last: None,
                aumc_inf: None,
                tlag: None,
                cmax_dn: None,
                auc_last_dn: None,
                auc_inf_dn: None,
                time_above_mic: None,
            },
            terminal: None,
            clearance,
            route_params,
            steady_state: None,
            multi_dose: None,
            quality: Quality::default(),
        }
    }

    #[test]
    fn test_accessor_c0_iv_bolus() {
        let result = make_result_with(
            Some(RouteParams::IVBolus(IVBolusParams {
                c0: 25.0,
                vd: 20.0,
                c0_method: None,
            })),
            None,
        );
        assert_eq!(result.c0(), Some(25.0));
        assert_eq!(result.vd(), Some(20.0));
    }

    #[test]
    fn test_accessor_c0_not_bolus() {
        let result = make_result_with(Some(RouteParams::Extravascular), None);
        assert_eq!(result.c0(), None);
        assert_eq!(result.vd(), None);
    }

    #[test]
    fn test_accessor_vss() {
        let result = make_result_with(
            None,
            Some(ClearanceParams {
                cl_f: 5.0,
                vz_f: 10.0,
                vss: Some(15.0),
            }),
        );
        assert_eq!(result.vss(), Some(15.0));
    }

    #[test]
    fn test_accessor_vss_none() {
        let result = make_result_with(None, None);
        assert_eq!(result.vss(), None);
    }

    #[test]
    fn test_accessor_ceoi_infusion() {
        let result = make_result_with(
            Some(RouteParams::IVInfusion(IVInfusionParams {
                infusion_duration: 1.0,
                mrt_iv: Some(4.0),
                ceoi: Some(30.0),
            })),
            None,
        );
        assert_eq!(result.ceoi(), Some(30.0));
        assert_eq!(result.mrt_iv(), Some(4.0));
    }

    #[test]
    fn test_accessor_ceoi_not_infusion() {
        let result = make_result_with(Some(RouteParams::Extravascular), None);
        assert_eq!(result.ceoi(), None);
        assert_eq!(result.mrt_iv(), None);
    }
}
