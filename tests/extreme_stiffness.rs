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

/// Shared relative/absolute tolerance for analytical comparisons: well above
/// the solver tolerances (1e-4) but far below any qualitative difference.
const PRED_TOLERANCE: f64 = 1e-3;

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
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= PRED_TOLERANCE * scale,
        "{label}: prediction {actual:.6e} differs from analytical {expected:.6e} \
         by more than {PRED_TOLERANCE:.0e} (scale {scale:.3e})"
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
// far from any event. All solvers must agree with each other.
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

#[test]
fn michaelis_menten_depletion_agrees_across_solvers() {
    // Saturated elimination consumes the 10-unit infusion at ~1/unit time, so
    // the state crosses Km (1e-2) around t = 10 and the local stiffness jumps
    // from ~0 to vmax/km = 1e2 mid-segment. Solver tolerances are tightened so
    // cross-solver agreement is meaningful through the depletion corner.
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

    let reference = predictions_for(
        "BDF michaelis-menten",
        &model_for(OdeSolver::Bdf),
        &subject,
        &parameters,
    );
    assert_eq!(reference.len(), times.len());

    for (name, solver) in all_solvers() {
        let label = format!("{name} michaelis-menten");
        let actual = predictions_for(&label, &model_for(solver), &subject, &parameters);
        assert_eq!(actual.len(), reference.len(), "{label}: prediction count");
        for ((time, actual), expected) in times.iter().zip(&actual).zip(&reference) {
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
    assert!(
        message.contains("did not recover"),
        "error should mention the exhausted restarts: {message}"
    );
}
