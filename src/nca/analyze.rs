//! Main NCA analysis orchestrator
//!
//! This module contains the core analysis function that computes all NCA parameters
//! from a validated profile and options.

use super::calc;
use super::error::NCAError;
use super::profile::Profile;
use super::types::*;

// ============================================================================
// Dose Context (internal - auto-detected from data structures)
// ============================================================================

/// Dose and route information detected from data
///
/// This is constructed internally by `Occasion::nca()` from the dose events in the data.
#[derive(Debug, Clone)]
pub(crate) struct DoseContext {
    /// Total dose amount
    pub amount: f64,
    /// Infusion duration (None for bolus)
    pub duration: Option<f64>,
    /// Administration route
    pub route: Route,
}

impl DoseContext {
    /// Create a new dose context
    pub fn new(amount: f64, duration: Option<f64>, route: Route) -> Self {
        Self {
            amount,
            duration,
            route,
        }
    }
}

// ============================================================================
// Main Analysis Function
// ============================================================================

/// Perform complete NCA analysis on a profile
///
/// This is an internal function. External users should use `analyze_arrays`
/// or the `.nca()` method on data structures.
///
/// # Arguments
/// * `profile` - Validated concentration-time profile
/// * `dose` - Dose context (detected from data, None if no dosing info)
/// * `options` - Analysis configuration
pub(crate) fn analyze(
    profile: &Profile,
    dose: Option<&DoseContext>,
    options: &NCAOptions,
) -> Result<NCAResult, NCAError> {
    // When called without raw data, calculate tlag from the (filtered) profile
    #[allow(deprecated)]
    let raw_tlag = calc::tlag(profile);
    analyze_with_raw_tlag(profile, dose, options, raw_tlag)
}

/// Internal analysis with pre-computed raw tlag
fn analyze_with_raw_tlag(
    profile: &Profile,
    dose: Option<&DoseContext>,
    options: &NCAOptions,
    raw_tlag: Option<f64>,
) -> Result<NCAResult, NCAError> {
    if profile.times.is_empty() {
        return Err(NCAError::InsufficientData { n: 0, required: 2 });
    }

    // Core exposure parameters (always calculated)
    let mut exposure = compute_exposure(profile, options, raw_tlag)?;

    // Terminal phase parameters (if lambda-z can be estimated)
    let (terminal, lambda_z_result) = compute_terminal(profile, options);

    // Update exposure with AUCinf if we have terminal phase
    if let Some(ref lz) = lambda_z_result {
        update_exposure_with_terminal(&mut exposure, profile, lz, options);
    }

    // Clearance parameters (if we have dose and terminal phase)
    let clearance = dose
        .and_then(|d| lambda_z_result.as_ref().map(|lz| (d, lz)))
        .map(|(d, lz)| compute_clearance(d.amount, exposure.auc_inf, lz.lambda_z));

    // Route-specific parameters
    let (iv_bolus, iv_infusion) =
        compute_route_specific(profile, dose, lambda_z_result.as_ref(), options);

    // Steady-state parameters (if tau specified)
    let steady_state = options
        .tau
        .map(|tau| compute_steady_state(profile, tau, options));

    // Build quality summary
    let quality = build_quality(
        &exposure,
        terminal.as_ref(),
        lambda_z_result.as_ref(),
        options,
    );

    Ok(NCAResult {
        subject_id: None,
        occasion: None,
        exposure,
        terminal,
        clearance,
        iv_bolus,
        iv_infusion,
        steady_state,
        quality,
    })
}

/// Compute core exposure parameters
fn compute_exposure(
    profile: &Profile,
    options: &NCAOptions,
    raw_tlag: Option<f64>,
) -> Result<ExposureParams, NCAError> {
    let cmax = profile.cmax();
    let tmax = profile.tmax();
    let clast = profile.clast();
    let tlast = profile.tlast();

    let auc_last = calc::auc_last(profile, options.auc_method);
    let aumc_last = calc::aumc_last(profile, options.auc_method);

    // Calculate partial AUC if interval specified
    let auc_partial = options
        .auc_interval
        .map(|(start, end)| calc::auc_interval(profile, start, end, options.auc_method));

    // AUCinf will be computed in terminal phase if lambda-z is available
    Ok(ExposureParams {
        cmax,
        tmax,
        clast,
        tlast,
        auc_last,
        auc_inf: None, // Will be filled in if terminal phase estimated
        auc_pct_extrap: None,
        auc_partial,
        aumc_last: Some(aumc_last),
        aumc_inf: None,
        tlag: raw_tlag,
    })
}

/// Compute terminal phase parameters
fn compute_terminal(
    profile: &Profile,
    options: &NCAOptions,
) -> (Option<TerminalParams>, Option<calc::LambdaZResult>) {
    use crate::nca::types::ClastType;

    let lz_result = calc::lambda_z(profile, &options.lambda_z);

    let terminal = lz_result.as_ref().map(|lz| {
        let half_life = calc::half_life(lz.lambda_z);

        // Choose Clast based on ClastType option
        let clast = match options.clast_type {
            ClastType::Observed => profile.clast(),
            ClastType::Predicted => lz.clast_pred,
        };

        // Compute AUC infinity
        let auc_last_val = calc::auc_last(profile, options.auc_method);
        let auc_inf = calc::auc_inf(auc_last_val, clast, lz.lambda_z);

        // MRT - use aumc with same method as auc for consistency
        let aumc_last_val = calc::aumc_last(profile, options.auc_method);
        let aumc_inf = calc::aumc_inf(aumc_last_val, clast, profile.tlast(), lz.lambda_z);
        let mrt = calc::mrt(aumc_inf, auc_inf);

        TerminalParams {
            lambda_z: lz.lambda_z,
            half_life,
            mrt: Some(mrt),
            regression: Some(lz.clone().into()),
        }
    });

    (terminal, lz_result)
}

/// Compute clearance parameters
fn compute_clearance(dose: f64, auc_inf: Option<f64>, lambda_z: f64) -> ClearanceParams {
    let auc = auc_inf.unwrap_or(f64::NAN);
    let cl = calc::clearance(dose, auc);
    let vz = calc::vz(dose, lambda_z, auc);

    ClearanceParams {
        cl_f: cl,
        vz_f: vz,
        vss: None, // Computed for IV routes
    }
}

/// Pre-computed base values to avoid redundant calculations
struct BaseValues {
    auc_last: f64,
    aumc_last: f64,
    clast: f64,
    tlast: f64,
}

impl BaseValues {
    fn from_profile(profile: &Profile, method: AUCMethod) -> Self {
        Self {
            auc_last: calc::auc_last(profile, method),
            aumc_last: calc::aumc_last(profile, method),
            clast: profile.clast(),
            tlast: profile.tlast(),
        }
    }

    /// Create with predicted clast from lambda-z regression
    fn with_clast_pred(mut self, clast_pred: f64) -> Self {
        self.clast = clast_pred;
        self
    }

    fn auc_inf(&self, lambda_z: f64) -> f64 {
        calc::auc_inf(self.auc_last, self.clast, lambda_z)
    }

    fn aumc_inf(&self, lambda_z: f64) -> f64 {
        calc::aumc_inf(self.aumc_last, self.clast, self.tlast, lambda_z)
    }
}

/// Compute route-specific parameters (IV only - extravascular tlag is in exposure)
fn compute_route_specific(
    profile: &Profile,
    dose: Option<&DoseContext>,
    lz_result: Option<&calc::LambdaZResult>,
    options: &NCAOptions,
) -> (Option<IVBolusParams>, Option<IVInfusionParams>) {
    let route = dose.map(|d| d.route).unwrap_or(Route::Extravascular);

    // Pre-compute base values once to avoid redundant calculations
    let mut base = BaseValues::from_profile(profile, options.auc_method);

    // Apply predicted clast if requested and lambda-z is available
    if matches!(options.clast_type, ClastType::Predicted) {
        if let Some(lz) = lz_result {
            base = base.with_clast_pred(lz.clast_pred);
        }
    }

    match route {
        Route::IVBolus => {
            let lambda_z = lz_result.map(|lz| lz.lambda_z).unwrap_or(f64::NAN);
            let c0 = calc::c0(profile, &options.c0_methods, lambda_z);

            let vd = dose
                .map(|d| calc::vd_bolus(d.amount, c0))
                .unwrap_or(f64::NAN);

            // VSS for IV
            let vss = lz_result.and_then(|lz| {
                dose.map(|d| {
                    let auc_inf = base.auc_inf(lz.lambda_z);
                    let aumc_inf = base.aumc_inf(lz.lambda_z);
                    calc::vss(d.amount, aumc_inf, auc_inf)
                })
            });

            (Some(IVBolusParams { c0, vd, vss }), None)
        }
        Route::IVInfusion => {
            let duration = dose.and_then(|d| d.duration).unwrap_or(0.0);

            // MRT adjusted for infusion
            let mrt_iv = lz_result.map(|lz| {
                let auc_inf = base.auc_inf(lz.lambda_z);
                let aumc_inf = base.aumc_inf(lz.lambda_z);
                let mrt_uncorrected = calc::mrt(aumc_inf, auc_inf);
                calc::mrt_infusion(mrt_uncorrected, duration)
            });

            // VSS for IV infusion
            let vss = lz_result.and_then(|lz| {
                dose.map(|d| {
                    let auc_inf = base.auc_inf(lz.lambda_z);
                    let aumc_inf = base.aumc_inf(lz.lambda_z);
                    calc::vss(d.amount, aumc_inf, auc_inf)
                })
            });

            (
                None,
                Some(IVInfusionParams {
                    infusion_duration: duration,
                    mrt_iv,
                    vss,
                }),
            )
        }
        Route::Extravascular => {
            // Tlag is computed in exposure params
            (None, None)
        }
    }
}

/// Compute steady-state parameters
fn compute_steady_state(profile: &Profile, tau: f64, options: &NCAOptions) -> SteadyStateParams {
    let cmax = profile.cmax();
    let cmin = calc::cmin(profile);
    let auc_tau = calc::auc_interval(profile, 0.0, tau, options.auc_method);
    let cavg = calc::cavg(auc_tau, tau);
    let fluctuation = calc::fluctuation(cmax, cmin, cavg);
    let swing = calc::swing(cmax, cmin);

    SteadyStateParams {
        tau,
        auc_tau,
        cmin,
        cmax_ss: cmax,
        cavg,
        fluctuation,
        swing,
        accumulation: None, // Would need single-dose reference
    }
}

/// Build quality assessment
fn build_quality(
    exposure: &ExposureParams,
    terminal: Option<&TerminalParams>,
    lz_result: Option<&calc::LambdaZResult>,
    options: &NCAOptions,
) -> Quality {
    let mut warnings = Vec::new();

    // Check for issues
    if exposure.cmax <= 0.0 {
        warnings.push(Warning::LowCmax);
    }

    // Check extrapolation percentage
    if let (Some(auc_inf), Some(lz)) = (exposure.auc_inf, lz_result) {
        let pct_extrap = calc::auc_extrap_pct(exposure.auc_last, auc_inf);
        if pct_extrap > options.max_auc_extrap_pct {
            warnings.push(Warning::HighExtrapolation);
        }

        // Check span ratio
        if let Some(stats) = terminal.and_then(|t| t.regression.as_ref()) {
            if stats.span_ratio < options.lambda_z.min_span_ratio {
                warnings.push(Warning::ShortTerminalPhase);
            }
        }

        // Check R²
        if lz.r_squared < options.lambda_z.min_r_squared {
            warnings.push(Warning::PoorFit);
        }
    } else {
        warnings.push(Warning::LambdaZNotEstimable);
    }

    Quality { warnings }
}

/// Update exposure parameters with terminal phase info
fn update_exposure_with_terminal(
    exposure: &mut ExposureParams,
    profile: &Profile,
    lz_result: &calc::LambdaZResult,
    options: &NCAOptions,
) {
    // Choose Clast based on ClastType option
    let clast = match options.clast_type {
        ClastType::Observed => profile.clast(),
        ClastType::Predicted => lz_result.clast_pred,
    };
    let tlast = profile.tlast();

    // AUC infinity
    let auc_inf = calc::auc_inf(exposure.auc_last, clast, lz_result.lambda_z);
    exposure.auc_inf = Some(auc_inf);
    exposure.auc_pct_extrap = Some(calc::auc_extrap_pct(exposure.auc_last, auc_inf));

    // AUMC infinity
    if let Some(aumc_last) = exposure.aumc_last {
        exposure.aumc_inf = Some(calc::aumc_inf(aumc_last, clast, tlast, lz_result.lambda_z));
    }
}

// ============================================================================
// Helper for Data integration
// ============================================================================

/// Analyze from raw arrays with censoring information
///
/// Censoring status is determined by the `Censor` marking:
/// - `Censor::BLOQ`: Below limit of quantification - value is the lower limit
/// - `Censor::ALOQ`: Above limit of quantification - value is the upper limit  
/// - `Censor::None`: Quantifiable observation - value is the measured concentration
///
/// For uncensored data, pass `Censor::None` for all observations.
pub(crate) fn analyze_arrays(
    times: &[f64],
    concentrations: &[f64],
    censoring: &[crate::Censor],
    dose: Option<&DoseContext>,
    options: &NCAOptions,
) -> Result<NCAResult, NCAError> {
    // Calculate tlag from raw data (before BLQ filtering) to match PKNCA
    let raw_tlag = calc::tlag_from_raw(times, concentrations, censoring);

    let profile = Profile::from_arrays(times, concentrations, censoring, options.blq_rule.clone())?;
    analyze_with_raw_tlag(&profile, dose, options, raw_tlag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Censor;

    fn test_profile() -> Profile {
        let censoring = vec![Censor::None; 8];
        Profile::from_arrays(
            &[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            &[0.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.25],
            &censoring,
            BLQRule::Exclude,
        )
        .unwrap()
    }

    #[test]
    fn test_analyze_basic() {
        let profile = test_profile();
        let options = NCAOptions::default();

        let result = analyze(&profile, None, &options).unwrap();

        assert_eq!(result.exposure.cmax, 10.0);
        assert_eq!(result.exposure.tmax, 1.0);
        assert!(result.exposure.auc_last > 0.0);
        // No clearance without dose
        assert!(result.clearance.is_none());
    }

    #[test]
    fn test_analyze_with_dose() {
        let profile = test_profile();
        let options = NCAOptions::default();
        let dose = DoseContext::new(100.0, None, Route::Extravascular);

        let result = analyze(&profile, Some(&dose), &options).unwrap();

        // Should have clearance if terminal phase estimated
        if result.terminal.is_some() {
            assert!(result.clearance.is_some());
        }
        // Tlag is now in exposure, not a separate struct
        // Exposure params are always present
        assert!(result.exposure.auc_last > 0.0);
    }

    #[test]
    fn test_analyze_iv_bolus() {
        let profile = test_profile();
        let options = NCAOptions::default();
        let dose = DoseContext::new(100.0, None, Route::IVBolus);

        let result = analyze(&profile, Some(&dose), &options).unwrap();

        assert!(result.iv_bolus.is_some());
        assert!(result.iv_infusion.is_none());
    }

    #[test]
    fn test_analyze_iv_infusion() {
        let profile = test_profile();
        let options = NCAOptions::default();
        let dose = DoseContext::new(100.0, Some(1.0), Route::IVInfusion);

        let result = analyze(&profile, Some(&dose), &options).unwrap();

        assert!(result.iv_bolus.is_none());
        assert!(result.iv_infusion.is_some());
        assert_eq!(result.iv_infusion.as_ref().unwrap().infusion_duration, 1.0);
    }

    #[test]
    fn test_analyze_steady_state() {
        let profile = test_profile();
        let options = NCAOptions::default().with_tau(12.0);
        let dose = DoseContext::new(100.0, None, Route::Extravascular);

        let result = analyze(&profile, Some(&dose), &options).unwrap();

        assert!(result.steady_state.is_some());
        let ss = result.steady_state.unwrap();
        assert_eq!(ss.tau, 12.0);
        assert!(ss.auc_tau > 0.0);
    }

    #[test]
    fn test_empty_profile() {
        let profile = Profile::from_arrays(&[], &[], &[], BLQRule::Exclude);
        assert!(profile.is_err());
    }
}
