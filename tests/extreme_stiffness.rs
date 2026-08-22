//! Extreme-stiffness regression tests for the ODE event loop.
//!
//! These target the failure modes seen in population fits (PMcore): sudden
//! RHS changes at infusion boundaries, lagged boluses, and steep covariate
//! changes drive the adaptive step size toward zero. diffsol also shrinks the
//! step to land exactly on stops that sit a few dozen ULPs apart and swallows
//! the resulting `StepSizeTooSmall`, after which every later step-size update
//! fails. The event loop must survive all of this — for every solver — by
//! restarting in place, and must return a descriptive error (not hang) when a
//! problem is genuinely beyond the selected solver.

use pharmsol::prelude::*;

/// Tolerances for analytical comparisons. The absolute component stays just
/// above the default solver `atol` without allowing a milliscale error when
/// the analytical result is zero.
const PRED_RELATIVE_TOLERANCE: f64 = 1e-3;
const PRED_ABSOLUTE_TOLERANCE: f64 = 2e-4;

fn implicit_solvers() -> [(&'static str, OdeSolver); 3] {
    [
        ("BDF", OdeSolver::Bdf),
        ("TRBDF2", OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
        ("ESDIRK34", OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
    ]
}

fn all_solvers() -> [(&'static str, OdeSolver); 4] {
    [
        ("BDF", OdeSolver::Bdf),
        ("TRBDF2", OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
        ("ESDIRK34", OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
        ("TSIT45", OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
    ]
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    assert!(
        actual.is_finite(),
        "{label}: prediction is not finite (expected {expected:.6e})"
    );
    let allowed_error = PRED_ABSOLUTE_TOLERANCE.max(PRED_RELATIVE_TOLERANCE * expected.abs());
    assert!(
        (actual - expected).abs() <= allowed_error,
        "{label}: prediction {actual:.6e} differs from analytical {expected:.6e} \
         by more than {allowed_error:.3e}"
    );
}

fn predictions_for(
    label: &str,
    model: &equation::ODE,
    subject: &Subject,
    parameters: &[(&str, f64)],
) -> Vec<f64> {
    let parameters = Parameters::with_model(model, parameters.iter().copied())
        .unwrap_or_else(|error| panic!("{label}: parameters should validate: {error}"));
    let predictions = model
        .estimate_predictions_dense(subject, parameters.as_slice())
        .unwrap_or_else(|error| panic!("{label}: simulation failed: {error}"));
    predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .collect()
}

// ---------------------------------------------------------------------------
// Stiff infusion boundary vs the analytical one-compartment solution.
// ---------------------------------------------------------------------------

fn infusion_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = rateiv[0] - ke * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("stiff_infusion")
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .routes([equation::Route::infusion("iv").to_state("central")]),
    )
    .expect("stiff infusion metadata should validate")
}

/// Unit-height infusion: rate `ke` over [0, 1] makes the steady state 1.0.
fn stiff_infusion_subject(observation_times: &[f64], ke: f64) -> Subject {
    let mut builder = Subject::builder("stiff_infusion").infusion(0.0, ke, "iv", 1.0);
    for &time in observation_times {
        builder = builder.missing_observation(time, "cp");
    }
    builder.build()
}

fn exact_unit_infusion(ke: f64, t: f64) -> f64 {
    if t <= 1.0 {
        1.0 - (-ke * t).exp()
    } else {
        (1.0 - (-ke).exp()) * (-ke * (t - 1.0)).exp()
    }
}

/// Observations straddling the infusion end at t = 1, with post-boundary
/// offsets scaled to the elimination timescale so the fast transient itself
/// is sampled.
fn boundary_observation_times(ke: f64) -> Vec<f64> {
    vec![
        0.5,
        1.0,
        1.0 + 0.5 / ke,
        1.0 + 1.0 / ke,
        1.0 + 2.0 / ke,
        1.0 + 5.0 / ke,
        2.0,
    ]
}

fn assert_stiff_infusion_boundary(label: &str, solver: OdeSolver, ke: f64) {
    let times = boundary_observation_times(ke);
    let subject = stiff_infusion_subject(&times, ke);
    let model = infusion_model(solver);
    let actual = predictions_for(label, &model, &subject, &[("ke", ke)]);
    assert_eq!(actual.len(), times.len(), "{label}: prediction count");
    for (time, actual) in times.iter().zip(actual) {
        let expected = exact_unit_infusion(ke, *time);
        assert_close(&format!("{label} at t = {time}"), actual, expected);
    }
}

#[test]
fn stiff_infusion_boundary_matches_analytical_implicit() {
    for (name, solver) in implicit_solvers() {
        for ke in [1e2, 1e4, 1e6, 1e8] {
            assert_stiff_infusion_boundary(&format!("{name} ke={ke:.0e}"), solver.clone(), ke);
        }
    }
}

#[test]
fn stiff_infusion_boundary_matches_analytical_explicit() {
    // Explicit RK is stability-limited; keep the stiffness within what it can
    // integrate in reasonable time.
    for ke in [1e2, 1e4] {
        assert_stiff_infusion_boundary(
            &format!("TSIT45 ke={ke:.0e}"),
            OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45),
            ke,
        );
    }
}

// ---------------------------------------------------------------------------
// Step size crushed by near-coincident stop times.
//
// diffsol shrinks the step to land exactly on a close stop and ignores the
// `StepSizeTooSmall` this raises; once the step sits below half the minimum,
// every later step-size update (even growth, capped at 2x) fails. These used
// to abort the simulation a few steps after the close pair.
// ---------------------------------------------------------------------------

#[test]
fn ulp_scale_gap_between_observations_recovers() {
    // Two observations 3e-14 apart at t = 0.1: too far apart for diffsol to
    // report the second stop as already reached, close enough to crush the
    // step size below recovery.
    let close = 0.1_f64 + 3e-14;
    assert!(close > 0.1, "gap must be representable");
    let ke = 0.5;
    let times = [0.1, close, 2.0];

    for (name, solver) in all_solvers() {
        let label = format!("{name} ulp-gap observations");
        let subject = stiff_infusion_subject(&times, ke);
        let model = infusion_model(solver);
        let actual = predictions_for(&label, &model, &subject, &[("ke", ke)]);
        assert_eq!(actual.len(), times.len(), "{label}: prediction count");
        for (time, actual) in times.iter().zip(actual) {
            let expected = exact_unit_infusion(ke, *time);
            assert_close(&format!("{label} at t = {time}"), actual, expected);
        }
    }
}

#[test]
fn ulp_scale_gap_between_observation_and_infusion_end_recovers() {
    // The infusion ends 3e-14 after an observation, so the crushed step is
    // carried into an infusion-boundary restart instead of a plain segment.
    let duration = 0.1_f64 + 3e-14;
    let ke = 0.5;
    let times = [0.1, 2.0];

    for (name, solver) in all_solvers() {
        let label = format!("{name} ulp-gap infusion end");
        let mut builder =
            Subject::builder("ulp_gap_infusion").infusion(0.0, ke * duration, "iv", duration);
        for &time in &times {
            builder = builder.missing_observation(time, "cp");
        }
        let subject = builder.build();
        let model = infusion_model(solver);
        let actual = predictions_for(&label, &model, &subject, &[("ke", ke)]);

        // Same unit-height model, infusion just ends at `duration` instead of 1.
        let exact = |t: f64| {
            if t <= duration {
                1.0 - (-ke * t).exp()
            } else {
                (1.0 - (-ke * duration).exp()) * (-ke * (t - duration)).exp()
            }
        };
        assert_eq!(actual.len(), times.len(), "{label}: prediction count");
        for (time, actual) in times.iter().zip(actual) {
            assert_close(&format!("{label} at t = {time}"), actual, exact(*time));
        }
    }
}

// ---------------------------------------------------------------------------
// Long free segment after a stiff infusion cut-off.
//
// With no observation near the boundary the restart begins with the large
// step size inherited from the smooth infusion phase, and the controller must
// reject its way down to the fast elimination timescale — the classic way to
// exhaust diffsol's error-test budget on very stiff parameter draws.
// ---------------------------------------------------------------------------

#[test]
fn stiff_cutoff_with_distant_observation_recovers() {
    for (name, solver) in implicit_solvers() {
        for ke in [1e6, 1e9] {
            let label = format!("{name} ke={ke:.0e} distant observation");
            // Rate ke over [0, 24] -> steady state 1.0; nothing stops the
            // solver between the boundary at 24 and the observation at 48.
            let subject = Subject::builder("stiff_cutoff")
                .infusion(0.0, ke * 24.0, "iv", 24.0)
                .missing_observation(48.0, "cp")
                .build();
            let model = infusion_model(solver.clone());
            let actual = predictions_for(&label, &model, &subject, &[("ke", ke)]);
            assert_eq!(actual.len(), 1, "{label}: prediction count");
            // Fully eliminated 24 time units after cut-off.
            assert_close(&label, actual[0], 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Steep covariate ramp in the middle of a segment (no event, no stop time).
// ---------------------------------------------------------------------------

fn covariate_clearance_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, _p, t, dx, b, _rateiv, cov| {
            fetch_cov!(cov, t, cl);
            dx[0] = b[0] - cl * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("covariate_clearance")
            .parameters(["dummy"])
            .states(["central"])
            .outputs(["cp"])
            .routes([equation::Route::bolus("dose")
                .to_state("central")
                .expect_explicit_input()]),
    )
    .expect("covariate clearance metadata should validate")
}

fn assert_covariate_ramp(label: &str, solver: OdeSolver, lambda: f64) {
    // Clearance ramps linearly from 1 to `lambda` over a window of width
    // 2/lambda starting at t = 5 — a near-discontinuous covariate change in
    // the middle of the [4, ...] segment. The integral of the ramp is
    // (1 + lambda)/lambda, so a bolus of exp(5 + (1 + lambda)/lambda) makes
    // the state exactly 1.0 at the end of the ramp.
    let ramp_end = 5.0 + 2.0 / lambda;
    let bolus = (5.0 + (1.0 + lambda) / lambda).exp();
    let times = [4.0, ramp_end + 1.0 / lambda, ramp_end + 3.0 / lambda];
    let expected = [bolus * (-4.0_f64).exp(), (-1.0_f64).exp(), (-3.0_f64).exp()];

    let subject = {
        let mut builder = Subject::builder("covariate_ramp")
            .bolus(0.0, bolus, "dose")
            .covariate("cl", 0.0, 1.0)
            .covariate("cl", 5.0, 1.0)
            .covariate("cl", ramp_end, lambda);
        for &time in &times {
            builder = builder.missing_observation(time, "cp");
        }
        builder.build()
    };

    let model = covariate_clearance_model(solver);
    let actual = predictions_for(label, &model, &subject, &[("dummy", 1.0)]);
    assert_eq!(actual.len(), times.len(), "{label}: prediction count");
    for ((time, actual), expected) in times.iter().zip(actual).zip(expected) {
        assert_close(&format!("{label} at t = {time}"), actual, expected);
    }
}

#[test]
fn steep_covariate_ramp_mid_segment_recovers() {
    for (name, solver) in all_solvers() {
        assert_covariate_ramp(&format!("{name} lambda=1e4"), solver, 1e4);
    }
    for (name, solver) in implicit_solvers() {
        assert_covariate_ramp(&format!("{name} lambda=1e6"), solver, 1e6);
    }
}

// ---------------------------------------------------------------------------
// Lagged bolus into a very stiff elimination.
// ---------------------------------------------------------------------------

fn lagged_bolus_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _tlag);
            dx[0] = b[0] - ke * x[0];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ke, tlag);
            lag! {0 => tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("lagged_bolus")
            .parameters(["ke", "tlag"])
            .states(["central"])
            .outputs(["cp"])
            .routes([equation::Route::bolus("dose")
                .to_state("central")
                .expect_explicit_input()]),
    )
    .expect("lagged bolus metadata should validate")
}

#[test]
fn lagged_bolus_into_stiff_elimination_matches_analytical() {
    let tlag = 0.5;
    let dose_time = 1.0;
    let arrival = dose_time + tlag;
    let amount = 100.0;

    for (name, solver) in implicit_solvers() {
        for ke in [1e4, 1e8] {
            let label = format!("{name} ke={ke:.0e} lagged bolus");
            let times = [
                1.0,
                arrival + 0.5 / ke,
                arrival + 1.0 / ke,
                arrival + 3.0 / ke,
                arrival + 10.0,
            ];
            let mut builder = Subject::builder("lagged_bolus").bolus(dose_time, amount, "dose");
            for &time in &times {
                builder = builder.missing_observation(time, "cp");
            }
            let subject = builder.build();
            let model = lagged_bolus_model(solver.clone());
            let actual = predictions_for(&label, &model, &subject, &[("ke", ke), ("tlag", tlag)]);
            assert_eq!(actual.len(), times.len(), "{label}: prediction count");
            for (time, actual) in times.iter().zip(actual) {
                let expected = if *time < arrival {
                    0.0
                } else {
                    amount * (-ke * (time - arrival)).exp()
                };
                assert_close(&format!("{label} at t = {time}"), actual, expected);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Michaelis-Menten depletion: stiffness switches on as the state crosses Km,
// far from any event. Every solver is compared with the independently derived
// analytical solution, not with another numerical solver.
// ---------------------------------------------------------------------------

fn michaelis_menten_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, vmax, km);
            dx[0] = rateiv[0] - vmax * x[0] / (km + x[0]);
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("michaelis_menten")
            .parameters(["vmax", "km"])
            .states(["central"])
            .outputs(["cp"])
            .routes([equation::Route::infusion("iv").to_state("central")]),
    )
    .expect("michaelis-menten metadata should validate")
}

fn michaelis_menten_infusion_elapsed(x: f64, vmax: f64, km: f64, rate: f64) -> f64 {
    let a = rate - vmax;
    let b = rate * km;
    let antiderivative = |state: f64| state / a - km * vmax / (a * a) * (a * state + b).ln();
    antiderivative(x) - antiderivative(0.0)
}

/// Independent analytical solution for the test trajectory. During infusion,
/// the monotone implicit integral is inverted by bisection. After infusion,
/// the exact depletion integral is
/// `elapsed = (x_start - x + km * ln(x_start / x)) / vmax`, which is inverted
/// by the same bracketed scalar solve. No ODE solver supplies this reference.
fn michaelis_menten_analytical_solution(vmax: f64, km: f64, time: f64) -> f64 {
    const INFUSION_RATE: f64 = 10.0;
    const INFUSION_END: f64 = 1.0;

    if time <= INFUSION_END {
        let mut lower = 0.0;
        let mut upper = INFUSION_RATE * time;
        for _ in 0..100 {
            let middle = 0.5 * (lower + upper);
            if michaelis_menten_infusion_elapsed(middle, vmax, km, INFUSION_RATE) < time {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        return 0.5 * (lower + upper);
    }

    let state_at_infusion_end = michaelis_menten_analytical_solution(vmax, km, INFUSION_END);
    let elapsed = time - INFUSION_END;
    let mut lower = 0.0;
    let mut upper = state_at_infusion_end;
    for _ in 0..100 {
        let middle = 0.5 * (lower + upper);
        let elapsed_from_middle = (state_at_infusion_end - middle
            + km * (state_at_infusion_end.ln() - middle.ln()))
            / vmax;
        if elapsed_from_middle > elapsed {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    0.5 * (lower + upper)
}

#[test]
fn michaelis_menten_depletion_agrees_across_solvers() {
    // Saturated elimination consumes the 10-unit infusion at ~1/unit time, so
    // the state crosses Km (1e-2) around t = 10 and the local stiffness jumps
    // from ~0 to vmax/km = 1e2 mid-segment. Solver tolerances are tightened so
    // agreement with the independent integral solution is meaningful through
    // the depletion corner.
    let parameters = [("vmax", 1.0), ("km", 1e-2)];
    let times = [0.5, 1.0, 5.0, 9.5, 9.9, 10.0, 10.05, 10.5, 11.0];
    let subject = {
        let mut builder = Subject::builder("michaelis_menten").infusion(0.0, 10.0, "iv", 1.0);
        for &time in &times {
            builder = builder.missing_observation(time, "cp");
        }
        builder.build()
    };
    let model_for = |solver: OdeSolver| michaelis_menten_model(solver).with_tolerances(1e-6, 1e-6);
    let expected = times
        .iter()
        .map(|&time| michaelis_menten_analytical_solution(1.0, 1e-2, time))
        .collect::<Vec<_>>();

    for (name, solver) in all_solvers() {
        let label = format!("{name} michaelis-menten");
        let actual = predictions_for(&label, &model_for(solver), &subject, &parameters);
        assert_eq!(actual.len(), expected.len(), "{label}: prediction count");
        for ((time, actual), expected) in times.iter().zip(&actual).zip(&expected) {
            assert_close(&format!("{label} at t = {time}"), *actual, *expected);
        }
    }
}

// ---------------------------------------------------------------------------
// Beyond rescue: a problem stiffer than the solver's hard minimum step must
// fail with a descriptive error, not hang or panic.
// ---------------------------------------------------------------------------

#[test]
fn impossibly_stiff_problem_returns_descriptive_error() {
    // ke = 1e12 needs accuracy steps below diffsol's minimum step size; no
    // number of restarts can integrate the transient.
    let ke = 1e12;
    let subject = Subject::builder("impossibly_stiff")
        .infusion(0.0, ke * 24.0, "iv", 24.0)
        .missing_observation(48.0, "cp")
        .build();
    let model = infusion_model(OdeSolver::Bdf);
    let parameters =
        Parameters::with_model(&model, [("ke", ke)]).expect("parameters should validate");

    let error = model
        .estimate_predictions_dense(&subject, parameters.as_slice())
        .expect_err("a problem beyond the minimum step size should fail");
    let message = error.to_string();
    let actionable_failure = (message.contains("step size collapsed")
        && message.contains("system is too stiff"))
        || (message.contains("error test failed")
            && message.contains("tolerances may be too tight or the parameters implausible"));
    assert!(
        actionable_failure
            && message.contains("subject `impossibly_stiff`")
            && message.contains("ke=1000000000000.0"),
        "error should identify the numerical failure and failing trajectory: {message}"
    );
}

#[test]
fn impossibly_stiff_explicit_problem_returns_descriptive_error() {
    const CHILD_MARKER: &str = "PHARMSOL_RUN_IMPOSSIBLY_STIFF_EXPLICIT_CHILD";
    if std::env::var_os(CHILD_MARKER).is_none() {
        let executable = std::env::current_exe().expect("resolve current test executable");
        let mut child = std::process::Command::new(executable)
            .args([
                "--exact",
                "impossibly_stiff_explicit_problem_returns_descriptive_error",
                "--nocapture",
            ])
            .env(CHILD_MARKER, "1")
            .spawn()
            .expect("spawn bounded explicit-stiffness test");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().expect("query child test status") {
                assert!(status.success(), "explicit-stiffness child test failed");
                return;
            }
            if std::time::Instant::now() >= deadline {
                match child.kill() {
                    Ok(()) => {
                        child.wait().expect("reap explicit-stiffness child test");
                        panic!(
                            "TSIT45 did not reject an impossible stiff problem within ten \
                             seconds; one extreme trajectory could stall a long-running job"
                        );
                    }
                    Err(kill_error) => {
                        let status = child
                            .wait()
                            .expect("reap explicit-stiffness child after exit race");
                        assert!(
                            status.success(),
                            "explicit-stiffness child failed while reaching the deadline \
                             ({kill_error})"
                        );
                        return;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    let ke = 1e12;
    let subject = Subject::builder("impossibly_stiff_explicit")
        .infusion(0.0, ke * 24.0, "iv", 24.0)
        .missing_observation(48.0, "cp")
        .build();
    let model = infusion_model(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let parameters =
        Parameters::with_model(&model, [("ke", ke)]).expect("parameters should validate");

    let error = model
        .estimate_predictions_dense(&subject, parameters.as_slice())
        .expect_err("an explicit solver should reject this impossible stiffness");
    let message = error.to_string();
    assert!(
        message.contains("did not recover") && message.contains("implicit"),
        "explicit-solver exhaustion should explain the failed recovery and suggest an implicit \
         method: {message}"
    );
}

fn close_gap_exponential_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| dx[0] = p[0] * x[0],
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[0] = 1.0,
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(0)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("close_gap_exponential")
            .parameters(["rate"])
            .states(["amount"])
            .outputs(["cp"]),
    )
    .expect("close-gap model metadata should validate")
}

fn absolute_time_rhs_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |_x, p, t, dx, _b, _rateiv, _cov| dx[0] = p[0] * t,
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[0] = 1.0,
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(0)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("absolute_time_rhs")
            .parameters(["rate"])
            .states(["amount"])
            .outputs(["cp"]),
    )
    .expect("absolute-time model metadata should validate")
}

fn absolute_time_linear_rhs_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, _b, _rateiv, _cov| dx[0] = p[0] * t * x[0],
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[0] = 1.0,
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(0)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("absolute_time_linear_rhs")
            .parameters(["rate"])
            .states(["amount"])
            .outputs(["cp"]),
    )
    .expect("absolute-time linear model metadata should validate")
}

fn rebase_absolute_time_rhs_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |_x, _p, t, dx, _b, rateiv, _cov| {
            dx[0] = rateiv[0];
            dx[1] = rateiv[0] * t;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| {
            y[0] = x[0];
            y[1] = x[1];
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(2)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("rebase_absolute_time_rhs")
            .states(["amount", "time_weighted_amount"])
            .outputs(["amount", "time_weighted_amount"])
            .routes([equation::Route::infusion("iv").to_state("amount")]),
    )
    .expect("rebase absolute-time model metadata should validate")
}

fn close_gap_bolus_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, bolus, _rateiv, _cov| dx[0] = bolus[0] - p[0] * x[0],
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("close_gap_bolus")
            .parameters(["rate"])
            .states(["amount"])
            .outputs(["cp"])
            .routes([equation::Route::bolus("dose")
                .to_state("amount")
                .expect_explicit_input()]),
    )
    .expect("close-gap bolus metadata should validate")
}

fn close_gap_failure(
    label: &str,
    result: Result<SubjectPredictions, PharmsolError>,
    start: f64,
    stop: f64,
    expected: f64,
) -> Option<String> {
    match result {
        Ok(predictions) => {
            let actual = predictions
                .predictions()
                .last()
                .expect("close-gap case should produce a prediction")
                .prediction();
            let relative_error = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
            (relative_error > 1.0e-3).then(|| {
                format!(
                    "{label}: solver silently skipped material dynamics: prediction \
                     {actual:.16e}, expected {expected:.16e}, relative error {relative_error:.3e}"
                )
            })
        }
        Err(error) => Some(format!(
            "{label}: solver must integrate the distinct interval from t = {start:.16e} to \
             t = {stop:.16e}, but returned: {error}"
        )),
    }
}

#[test]
fn close_distinct_stops_do_not_silently_skip_autonomous_dynamics() {
    let mut failures = Vec::new();
    for (solver_name, solver) in all_solvers() {
        for (case_name, start, stop, exponent) in [
            ("small-time-growth", 0.0, 3.0e-13, 0.3),
            ("small-time-decay", 0.0, 3.0e-13, -0.3),
            ("large-time-growth", 1.0e6, 1.0e6_f64.next_up(), 0.3),
            ("large-time-decay", 1.0e6, 1.0e6_f64.next_up(), -0.3),
        ] {
            let gap = stop - start;
            let rate = exponent / gap;
            let subject_id = format!("close-gap-{case_name}-{solver_name}");
            let subject = Subject::builder(&subject_id)
                .missing_observation(start, "cp")
                .missing_observation(stop, "cp")
                .build();
            let result = close_gap_exponential_model(solver.clone())
                .estimate_predictions_dense(&subject, &[rate]);
            if let Some(failure) = close_gap_failure(
                &format!("{solver_name} {case_name}"),
                result,
                start,
                stop,
                exponent.exp(),
            ) {
                failures.push(failure);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn large_initial_time_close_segment_preserves_absolute_model_time() {
    let expected_change = 0.3;
    let mut failures = Vec::new();

    for (case_name, start) in [
        ("large-positive", 1.0e6_f64),
        ("large-negative", -1.0e6_f64),
    ] {
        let stop = start.next_up();
        let integrated_time = 0.5 * (start + stop) * (stop - start);
        let rate = expected_change / integrated_time;
        let subject = Subject::builder(format!("large-initial-time-absolute-time-{case_name}"))
            .missing_observation(start, "cp")
            .missing_observation(stop, "cp")
            .build();

        for (solver_name, solver) in all_solvers() {
            let result =
                absolute_time_rhs_model(solver).estimate_predictions_dense(&subject, &[rate]);
            if let Some(failure) = close_gap_failure(
                &format!("{solver_name} {case_name} large-initial-time absolute-time RHS"),
                result,
                start,
                stop,
                1.0 + expected_change,
            ) {
                failures.push(failure);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn rebase_after_large_event_preserves_absolute_time_rhs() {
    let center = 1.0e12_f64;
    let stop = center.next_up();
    let gap = stop - center;
    let delivered = 0.3;
    let expected_time_weighted_amount = delivered * 0.5 * (center + stop);
    let subject = Subject::builder("rebase-after-large-event")
        .infusion(center, delivered, "iv", gap)
        .missing_observation(0.0, "amount")
        .missing_observation(center, "amount")
        .missing_observation(stop, "amount")
        .missing_observation(center, "time_weighted_amount")
        .missing_observation(stop, "time_weighted_amount")
        .build();

    for (solver_name, solver) in all_solvers() {
        let predictions = rebase_absolute_time_rhs_model(solver)
            .estimate_predictions_dense(&subject, &[])
            .unwrap_or_else(|error| panic!("{solver_name}: rebase simulation failed: {error}"));
        assert_eq!(
            predictions.predictions().len(),
            5,
            "{solver_name}: prediction count"
        );
        let value_at = |time: f64, outeq: usize| {
            predictions
                .predictions()
                .iter()
                .find(|prediction| prediction.time() == time && prediction.outeq() == outeq)
                .unwrap_or_else(|| {
                    panic!("{solver_name}: missing prediction at t = {time}, output {outeq}")
                })
                .prediction()
        };

        assert_close(
            &format!("{solver_name} amount at the large event"),
            value_at(center, 0),
            0.0,
        );
        assert_close(
            &format!("{solver_name} amount at the ULP-close event"),
            value_at(stop, 0),
            delivered,
        );
        assert_close(
            &format!("{solver_name} absolute-time RHS at the ULP-close event"),
            value_at(stop, 1),
            expected_time_weighted_amount,
        );
    }
}

#[test]
fn invalid_ode_schedules_return_actionable_errors() {
    let cases = [
        (
            "nonfinite event time",
            Subject::builder("nonfinite-event-time")
                .missing_observation(f64::NAN, "cp")
                .build(),
        ),
        (
            "nonfinite infusion amount",
            Subject::builder("nonfinite-infusion-amount")
                .infusion(0.0, f64::INFINITY, "iv", 1.0)
                .missing_observation(2.0, "cp")
                .build(),
        ),
        (
            "nonfinite infusion duration",
            Subject::builder("nonfinite-infusion-duration")
                .infusion(0.0, 1.0, "iv", f64::NAN)
                .missing_observation(2.0, "cp")
                .build(),
        ),
        (
            "unrepresentable infusion endpoint",
            Subject::builder("unrepresentable-infusion-endpoint")
                .infusion(f64::MAX, 1.0, "iv", 1.0)
                .missing_observation(f64::MAX, "cp")
                .build(),
        ),
        (
            "unrepresentable event gap",
            Subject::builder("unrepresentable-event-gap")
                .missing_observation(-f64::MAX, "cp")
                .missing_observation(f64::MAX, "cp")
                .build(),
        ),
    ];

    for (label, subject) in cases {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            infusion_model(OdeSolver::Bdf).estimate_predictions_dense(&subject, &[0.5])
        }))
        .unwrap_or_else(|_| panic!("{label}: invalid schedule must not panic"));
        let error = result.expect_err(&format!("{label}: invalid schedule must fail"));
        assert!(
            error.to_string().contains("invalid ODE event schedule"),
            "{label}: expected actionable schedule error, got {error}"
        );
    }
}

#[test]
fn finite_cancellation_schedule_preserves_absolute_callback_times() {
    let start = -192.0_f64;
    let stop = 0.35_f64;
    let rate = 1.0e-6_f64;
    let expected = (0.5_f64 * rate * (stop * stop - start * start)).exp();
    let subject = Subject::builder("finite-cancellation-schedule")
        .missing_observation(start, "cp")
        .missing_observation(stop, "cp")
        .build();

    for (solver_name, solver) in all_solvers() {
        let predictions = absolute_time_linear_rhs_model(solver)
            .estimate_predictions_dense(&subject, &[rate])
            .unwrap_or_else(|error| panic!("{solver_name}: finite cancellation failed: {error}"));
        assert_eq!(
            predictions.predictions().len(),
            2,
            "{solver_name}: prediction count"
        );
        assert_eq!(
            predictions.predictions()[1].time(),
            stop,
            "{solver_name}: absolute callback stop time"
        );
        assert_close(
            &format!("{solver_name}: absolute callback time integration"),
            predictions.predictions()[1].prediction(),
            expected,
        );
    }
}

#[test]
fn close_distinct_stop_after_bolus_does_not_skip_decay() {
    let start = 0.0;
    let stop = 3.0e-13;
    let exponent = 0.3;
    let rate = exponent / (stop - start);
    let mut failures = Vec::new();

    for (solver_name, solver) in all_solvers() {
        let subject_id = format!("close-gap-bolus-{solver_name}");
        let subject = Subject::builder(&subject_id)
            .bolus(start, 1.0, "dose")
            .missing_observation(start, "cp")
            .missing_observation(stop, "cp")
            .build();
        let result = close_gap_bolus_model(solver).estimate_predictions_dense(&subject, &[rate]);
        if let Some(failure) = close_gap_failure(
            &format!("{solver_name} close-gap bolus"),
            result,
            start,
            stop,
            (-exponent).exp(),
        ) {
            failures.push(failure);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn locf_subject() -> Subject {
    let mut subject = Subject::builder("locf-breakpoint")
        .covariate("cov_rate", 0.0, 1.0)
        .covariate("cov_rate", 5.0, 2.0)
        .missing_observation(0.0, "cp")
        .missing_observation(10.0, "cp")
        .build();
    for occasion in subject.occasions_mut() {
        assert!(occasion
            .covariates_mut()
            .set_covariate_fixed("cov_rate", true));
    }
    subject
}

fn locf_integral_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |_x, _p, t, dx, _b, _rateiv, covariates| {
            dx[0] = covariates
                .get_covariate("cov_rate")
                .expect("rate covariate")
                .interpolate(t)
                .expect("rate value");
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
    )
    .with_nstates(1)
    .with_ndrugs(0)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("locf_integral")
            .covariates([equation::Covariate::locf("cov_rate")])
            .states(["integral"])
            .outputs(["cp"]),
    )
    .expect("LOCF model metadata should validate")
}

#[test]
fn locf_rhs_change_matches_piecewise_analytical_integral() {
    let expected = 15.0;
    for (solver_name, solver) in all_solvers() {
        let predictions = locf_integral_model(solver)
            .estimate_predictions_dense(&locf_subject(), &[])
            .unwrap_or_else(|error| panic!("{solver_name} LOCF simulation failed: {error}"));
        let actual = predictions
            .predictions()
            .last()
            .expect("LOCF case should produce a prediction")
            .prediction();
        let absolute_error = (actual - expected).abs();
        assert!(
            absolute_error <= 5.0e-3,
            "{solver_name}: LOCF integral {actual:.16e}, expected {expected:.16e}, \
             absolute error {absolute_error:.3e}"
        );
    }
}
