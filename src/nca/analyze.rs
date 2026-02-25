//! Main NCA analysis orchestrator
//!
//! This module contains the core analysis function that computes all NCA parameters
//! from a validated profile and options.
//!
//! # Planned features
//!
//! - **Ka estimation**: Absorption rate constant estimation for extravascular routes
//!   is not yet implemented. <!-- TODO: implement Ka estimation (flip-flop detection, Wagner-Nelson) -->

use super::calc;
use super::error::NCAError;
use super::types::*;
use crate::data::event::{AUCMethod, Route};
use crate::data::observation::ObservationProfile as Profile;
use crate::data::observation_error::ObservationError;

// ============================================================================
// Precomputed values (computed once, threaded through)
// ============================================================================

/// Values computed once at the start of analysis to avoid redundant calculation
struct Precomputed {
    auc_last: f64,
    aumc_last: f64,
    cmax: f64,
    tmax: f64,
    clast: f64,
    tlast: f64,
}

impl Precomputed {
    fn from_profile(profile: &Profile, method: AUCMethod) -> Result<Self, NCAError> {
        Ok(Self {
            auc_last: profile.auc_last(&method)?,
            aumc_last: profile.aumc_last(&method)?,
            cmax: profile.cmax(),
            tmax: profile.tmax(),
            clast: profile.clast(),
            tlast: profile.tlast(),
        })
    }

    fn auc_inf(&self, clast: f64, lambda_z: f64) -> f64 {
        calc::auc_inf(self.auc_last, clast, lambda_z)
    }

    fn aumc_inf(&self, clast: f64, lambda_z: f64) -> f64 {
        calc::aumc_inf(self.aumc_last, clast, self.tlast, lambda_z)
    }
}

// ============================================================================
// Main Analysis Function
// ============================================================================

/// Perform complete NCA analysis on a profile
///
/// This is the primary entry point for NCA analysis.
///
/// # Arguments
/// * `profile` - Validated concentration-time profile
/// * `dose_amount` - Total dose amount (None if no dosing data)
/// * `route` - Administration route
/// * `infusion_duration` - Infusion duration (None for bolus/extravascular)
/// * `options` - Analysis configuration
/// * `raw_tlag` - Tlag computed from raw (unfiltered) data, or None
/// * `subject_id` - Subject identifier (None for ad-hoc profiles)
/// * `occasion` - Occasion index (None for ad-hoc profiles)
pub(crate) fn analyze(
    profile: &Profile,
    dose_amount: Option<f64>,
    route: Route,
    infusion_duration: Option<f64>,
    options: &NCAOptions,
    raw_tlag: Option<f64>,
    subject_id: Option<&str>,
    occasion: Option<usize>,
) -> Result<NCAResult, NCAError> {
    if profile.times.is_empty() {
        return Err(ObservationError::InsufficientData { n: 0, required: 2 }.into());
    }

    // Compute AUC/AUMC once, use everywhere
    let pre = Precomputed::from_profile(profile, options.auc_method)?;

    // Core exposure parameters (always calculated)
    let mut exposure = compute_exposure(&pre, profile, options, raw_tlag)?;

    // Terminal phase parameters (if lambda-z can be estimated)
    let (terminal, lambda_z_result) = compute_terminal(&pre, profile, options);

    // Update exposure with both AUCinf variants if we have terminal phase
    if let Some(ref lz) = lambda_z_result {
        // AUCinf using observed Clast
        let auc_inf_obs = pre.auc_inf(pre.clast, lz.lambda_z);
        exposure.auc_inf_obs = Some(auc_inf_obs);
        exposure.auc_pct_extrap_obs = Some(calc::auc_extrap_pct(pre.auc_last, auc_inf_obs));

        // AUCinf using predicted Clast (from λz regression)
        let auc_inf_pred = pre.auc_inf(lz.clast_pred, lz.lambda_z);
        exposure.auc_inf_pred = Some(auc_inf_pred);
        exposure.auc_pct_extrap_pred = Some(calc::auc_extrap_pct(pre.auc_last, auc_inf_pred));

        if exposure.aumc_last.is_some() {
            // AUMC∞ uses observed Clast by convention
            exposure.aumc_inf = Some(pre.aumc_inf(pre.clast, lz.lambda_z));
        }
    }

    // Clearance parameters (if we have dose and terminal phase)
    // Uses auc_inf_obs by convention (standard practice)
    let clearance = dose_amount
        .and_then(|d| lambda_z_result.as_ref().map(|lz| (d, lz)))
        .map(|(d, lz)| compute_clearance(d, exposure.auc_inf_obs, lz.lambda_z, route, &pre));

    // Route-specific parameters (uses observed Clast for extrapolation)
    let route_params = compute_route_specific(
        &pre,
        profile,
        dose_amount,
        route,
        infusion_duration,
        lambda_z_result.as_ref(),
        pre.clast,
        options,
    );

    // Steady-state parameters (if tau specified)
    let steady_state = options
        .tau
        .map(|tau| compute_steady_state(&pre, profile, tau, options))
        .transpose()?;

    // Dose-normalized parameters
    if let Some(d) = dose_amount {
        if d > 0.0 {
            exposure.cmax_dn = Some(exposure.cmax / d);
            exposure.auc_last_dn = Some(exposure.auc_last / d);
            if let Some(auc_inf_obs) = exposure.auc_inf_obs {
                exposure.auc_inf_dn = Some(auc_inf_obs / d);
            }
        }
    }

    // Multi-dose interval parameters (if dose_times specified)
    let multi_dose = compute_multi_dose(profile, options)?;

    // Build quality summary
    let quality = build_quality(
        &exposure,
        terminal.as_ref(),
        lambda_z_result.as_ref(),
        options,
    );

    Ok(NCAResult {
        subject_id: subject_id.map(|s| s.to_string()),
        occasion,
        dose_amount,
        route: Some(route),
        infusion_duration,
        exposure,
        terminal,
        clearance,
        route_params,
        steady_state,
        multi_dose,
        quality,
    })
}

/// Compute core exposure parameters
fn compute_exposure(
    pre: &Precomputed,
    profile: &Profile,
    options: &NCAOptions,
    raw_tlag: Option<f64>,
) -> Result<ExposureParams, NCAError> {
    // Calculate partial AUC if interval specified
    let auc_partial = options
        .auc_interval
        .map(|(start, end)| profile.auc_interval(start, end, &options.auc_method))
        .transpose()?;

    // Find first measurable (positive) concentration time
    let tfirst = profile
        .times
        .iter()
        .zip(profile.concentrations.iter())
        .find(|(_, &c)| c > 0.0)
        .map(|(&t, _)| t);

    // Time above concentration threshold (if specified)
    let time_above_mic = options.concentration_threshold.map(|threshold| {
        calc::time_above_concentration(&profile.times, &profile.concentrations, threshold)
    });

    Ok(ExposureParams {
        cmax: pre.cmax,
        tmax: pre.tmax,
        clast: pre.clast,
        tlast: pre.tlast,
        tfirst,
        auc_last: pre.auc_last,
        auc_inf_obs: None, // filled in by caller if terminal phase estimated
        auc_inf_pred: None,
        auc_pct_extrap_obs: None,
        auc_pct_extrap_pred: None,
        auc_partial,
        aumc_last: Some(pre.aumc_last),
        aumc_inf: None,
        tlag: raw_tlag,
        cmax_dn: None, // filled in by caller if dose available
        auc_last_dn: None,
        auc_inf_dn: None,
        time_above_mic,
    })
}

/// Compute terminal phase parameters
fn compute_terminal(
    pre: &Precomputed,
    profile: &Profile,
    options: &NCAOptions,
) -> (Option<TerminalParams>, Option<calc::LambdaZResult>) {
    let lz_result = calc::lambda_z(profile, &options.lambda_z);

    let terminal = lz_result.as_ref().map(|lz| {
        let half_life = calc::half_life(lz.lambda_z);

        // MRT uses observed Clast by convention
        let auc_inf = pre.auc_inf(pre.clast, lz.lambda_z);
        let aumc_inf = pre.aumc_inf(pre.clast, lz.lambda_z);
        let mrt = calc::mrt(aumc_inf, auc_inf);

        // Derived terminal parameters
        let effective_half_life = if mrt.is_finite() && mrt > 0.0 {
            Some(calc::effective_half_life(mrt))
        } else {
            None
        };
        let kel = if mrt.is_finite() && mrt > 0.0 {
            Some(calc::kel(mrt))
        } else {
            None
        };

        TerminalParams {
            lambda_z: lz.lambda_z,
            half_life,
            mrt: Some(mrt),
            effective_half_life,
            kel,
            regression: Some(lz.clone().into()),
        }
    });

    (terminal, lz_result)
}

/// Compute clearance parameters
fn compute_clearance(
    dose: f64,
    auc_inf: Option<f64>,
    lambda_z: f64,
    route: Route,
    pre: &Precomputed,
) -> ClearanceParams {
    let auc = auc_inf.unwrap_or(f64::NAN);
    let cl = calc::clearance(dose, auc);
    let vz = calc::vz(dose, lambda_z, auc);

    // Vss is computed for IV routes: Vss = Dose * AUMC_inf / AUC_inf^2
    let vss = match route {
        Route::IVBolus | Route::IVInfusion => {
            let auc_inf_val = pre.auc_inf(pre.clast, lambda_z);
            let aumc_inf_val = pre.aumc_inf(pre.clast, lambda_z);
            Some(calc::vss(dose, aumc_inf_val, auc_inf_val))
        }
        Route::Extravascular => None,
    };

    ClearanceParams {
        cl_f: cl,
        vz_f: vz,
        vss,
    }
}

/// Compute route-specific parameters (IV only — extravascular tlag is in exposure)
fn compute_route_specific(
    pre: &Precomputed,
    profile: &Profile,
    dose_amount: Option<f64>,
    route: Route,
    infusion_duration: Option<f64>,
    lz_result: Option<&calc::LambdaZResult>,
    eff_clast: f64,
    options: &NCAOptions,
) -> Option<RouteParams> {
    match route {
        Route::IVBolus => {
            let lambda_z = lz_result.map(|lz| lz.lambda_z).unwrap_or(f64::NAN);
            let (c0, c0_method) = calc::c0(profile, &options.c0_methods, lambda_z);

            let vd = dose_amount
                .map(|d| calc::vd_bolus(d, c0))
                .unwrap_or(f64::NAN);

            Some(RouteParams::IVBolus(IVBolusParams { c0, vd, c0_method }))
        }
        Route::IVInfusion => {
            let duration = infusion_duration.unwrap_or(0.0);

            // MRT adjusted for infusion
            let mrt_iv = lz_result.map(|lz| {
                let auc_inf = pre.auc_inf(eff_clast, lz.lambda_z);
                let aumc_inf = pre.aumc_inf(eff_clast, lz.lambda_z);
                let mrt_uncorrected = calc::mrt(aumc_inf, auc_inf);
                calc::mrt_infusion(mrt_uncorrected, duration)
            });

            // Concentration at end of infusion (interpolate at dose end time)
            let ceoi = if duration > 0.0 {
                Some(profile.interpolate(duration))
            } else {
                None
            };

            Some(RouteParams::IVInfusion(IVInfusionParams {
                infusion_duration: duration,
                mrt_iv,
                ceoi,
            }))
        }
        Route::Extravascular => Some(RouteParams::Extravascular),
    }
}

/// Compute steady-state parameters
fn compute_steady_state(
    pre: &Precomputed,
    profile: &Profile,
    tau: f64,
    options: &NCAOptions,
) -> Result<SteadyStateParams, NCAError> {
    let cmin = calc::cmin(profile);
    let auc_tau = profile.auc_interval(0.0, tau, &options.auc_method)?;
    let cavg = calc::cavg(auc_tau, tau);
    let fluctuation = calc::fluctuation(pre.cmax, cmin, cavg);
    let swing = calc::swing(pre.cmax, cmin);
    let ptr = calc::peak_trough_ratio(pre.cmax, cmin);

    Ok(SteadyStateParams {
        tau,
        auc_tau,
        cmin,
        cmax_ss: pre.cmax,
        cavg,
        fluctuation,
        swing,
        peak_trough_ratio: ptr,
        accumulation: None, // Would need single-dose reference
    })
}

/// Compute multi-dose interval parameters
fn compute_multi_dose(
    profile: &Profile,
    options: &NCAOptions,
) -> Result<Option<MultiDoseParams>, NCAError> {
    let times = match &options.dose_times {
        None => return Ok(None),
        Some(t) if t.is_empty() => return Ok(None),
        Some(t) => t,
    };

    let mut sorted_times = times.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let last_obs_time = match profile.times.last() {
        Some(t) => *t,
        None => return Ok(None),
    };
    let n = sorted_times.len();

    let mut auc_intervals = Vec::with_capacity(n);
    let mut cmax_intervals = Vec::with_capacity(n);
    let mut tmax_intervals = Vec::with_capacity(n);

    for i in 0..n {
        let start = sorted_times[i];
        let end = if i + 1 < n {
            sorted_times[i + 1]
        } else {
            last_obs_time
        };

        // AUC over interval
        auc_intervals.push(profile.auc_interval(start, end, &options.auc_method)?);

        // Cmax/Tmax within [start, end]
        let (cmax, tmax) = cmax_tmax_in_window(profile, start, end);
        cmax_intervals.push(cmax);
        tmax_intervals.push(tmax);
    }

    Ok(Some(MultiDoseParams {
        dose_times: sorted_times,
        auc_intervals,
        cmax_intervals,
        tmax_intervals,
    }))
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

    // Check extrapolation percentage (uses observed variant)
    if let (Some(auc_inf_obs), Some(lz)) = (exposure.auc_inf_obs, lz_result) {
        let pct_extrap = calc::auc_extrap_pct(exposure.auc_last, auc_inf_obs);
        if pct_extrap > options.max_auc_extrap_pct {
            warnings.push(Warning::HighExtrapolation {
                pct: pct_extrap,
                threshold: options.max_auc_extrap_pct,
            });
        }

        // Check span ratio
        if let Some(stats) = terminal.and_then(|t| t.regression.as_ref()) {
            if stats.span_ratio < options.lambda_z.min_span_ratio {
                warnings.push(Warning::ShortTerminalPhase {
                    span_ratio: stats.span_ratio,
                    threshold: options.lambda_z.min_span_ratio,
                });
            }
        }

        // Check R²
        if lz.r_squared < options.lambda_z.min_r_squared {
            warnings.push(Warning::PoorFit {
                r_squared: lz.r_squared,
                threshold: options.lambda_z.min_r_squared,
            });
        }
    } else {
        warnings.push(Warning::LambdaZNotEstimable);
    }

    Quality { warnings }
}

/// Cmax and Tmax within a time window [start, end] (inclusive)
fn cmax_tmax_in_window(profile: &Profile, start: f64, end: f64) -> (f64, f64) {
    let mut cmax = f64::NEG_INFINITY;
    let mut tmax = start;
    for (i, &t) in profile.times.iter().enumerate() {
        if t >= start && t <= end {
            let c = profile.concentrations[i];
            if c > cmax {
                cmax = c;
                tmax = t;
            }
        }
    }
    if cmax == f64::NEG_INFINITY {
        // No observations in window
        (0.0, start)
    } else {
        (cmax, tmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builder::SubjectBuilderExt;
    use crate::data::event::BLQRule;
    use crate::Subject;

    fn test_profile() -> Profile {
        let subject = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 5.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.25, 0)
            .build();
        let occ = &subject.occasions()[0];
        Profile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap()
    }

    #[test]
    fn test_analyze_basic() {
        let profile = test_profile();
        let options = NCAOptions::default();

        let result = analyze(
            &profile,
            None,
            Route::Extravascular,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

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

        let result = analyze(
            &profile,
            Some(100.0),
            Route::Extravascular,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        // Should have clearance if terminal phase estimated
        if result.terminal.is_some() {
            assert!(result.clearance.is_some());
        }
        // Exposure params are always present
        assert!(result.exposure.auc_last > 0.0);
    }

    #[test]
    fn test_analyze_iv_bolus() {
        let profile = test_profile();
        let options = NCAOptions::default();

        let result = analyze(
            &profile,
            Some(100.0),
            Route::IVBolus,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(matches!(result.route_params, Some(RouteParams::IVBolus(_))));
    }

    #[test]
    fn test_analyze_iv_infusion() {
        let profile = test_profile();
        let options = NCAOptions::default();

        let result = analyze(
            &profile,
            Some(100.0),
            Route::IVInfusion,
            Some(1.0),
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(matches!(
            result.route_params,
            Some(RouteParams::IVInfusion(_))
        ));
        if let Some(RouteParams::IVInfusion(ref inf)) = result.route_params {
            assert_eq!(inf.infusion_duration, 1.0);
        }
    }

    #[test]
    fn test_analyze_steady_state() {
        let profile = test_profile();
        let options = NCAOptions::default().with_tau(12.0);

        let result = analyze(
            &profile,
            Some(100.0),
            Route::Extravascular,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.steady_state.is_some());
        let ss = result.steady_state.unwrap();
        assert_eq!(ss.tau, 12.0);
        assert!(ss.auc_tau > 0.0);
    }

    #[test]
    fn test_analyze_multi_dose() {
        let profile = test_profile(); // times: 0,1,2,4,8,12,24 concs: 0,10,8,6,3,1.5,0.5
        let options = NCAOptions::default().with_dose_times(vec![0.0, 8.0]);

        let result = analyze(
            &profile,
            Some(100.0),
            Route::Extravascular,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.multi_dose.is_some());
        let md = result.multi_dose.unwrap();
        assert_eq!(md.dose_times.len(), 2);
        assert_eq!(md.auc_intervals.len(), 2);
        assert_eq!(md.cmax_intervals.len(), 2);
        assert_eq!(md.tmax_intervals.len(), 2);

        // First interval [0, 8]: Cmax should be 10 at t=1
        assert_eq!(md.cmax_intervals[0], 10.0);
        assert_eq!(md.tmax_intervals[0], 1.0);

        // Second interval [8, 24]: Cmax should be 2.0 at t=8
        assert_eq!(md.cmax_intervals[1], 2.0);
        assert_eq!(md.tmax_intervals[1], 8.0);

        // AUC intervals should be positive and sum ≈ AUC_last
        assert!(md.auc_intervals[0] > 0.0);
        assert!(md.auc_intervals[1] > 0.0);
        let auc_sum: f64 = md.auc_intervals.iter().sum();
        assert!((auc_sum - result.exposure.auc_last).abs() / result.exposure.auc_last < 0.01);
    }

    #[test]
    fn test_analyze_no_multi_dose_by_default() {
        let profile = test_profile();
        let options = NCAOptions::default();

        let result = analyze(
            &profile,
            Some(100.0),
            Route::Extravascular,
            None,
            &options,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.multi_dose.is_none());
    }
}
