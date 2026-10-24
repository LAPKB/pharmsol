use super::*;
use crate::simulator::equation::{AttachMetadataError, ModelMetadataError};
use crate::{fa, lag, Subject, SubjectBuilderExt};
use approx::assert_relative_eq;
use pharmsol_dsl::ModelKind;
use std::sync::atomic::{AtomicUsize, Ordering};

static PREDICTION_CACHE_DIFFEQ_CALLS: AtomicUsize = AtomicUsize::new(0);

fn simple_ode() -> ODE {
    ODE::new(
        |_x, _p, _t, _dx, _b, _rateiv, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, _y| {},
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
}

fn route_policy_subject() -> Subject {
    Subject::builder("route_policy")
        .bolus(0.0, 100.0, "oral")
        .infusion(0.0, 100.0, "iv", 1.0)
        .observation(1.0, 0.0, "cp")
        .build()
}

fn explicit_route_function(
    _x: &V,
    _p: &V,
    _t: f64,
    dx: &mut V,
    b: &V,
    rateiv: &V,
    _cov: &Covariates,
) {
    dx[0] = b[0] + rateiv[0];
}

fn injected_route_function(
    _x: &V,
    _p: &V,
    _t: f64,
    dx: &mut V,
    _b: &V,
    _rateiv: &V,
    _cov: &Covariates,
) {
    dx[0] = 0.0;
}

fn zero_lag(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
    std::collections::HashMap::new()
}

fn unit_fa(_p: &V, _t: f64, _cov: &Covariates) -> std::collections::HashMap<usize, f64> {
    std::collections::HashMap::new()
}

fn zero_init(_p: &V, _t: f64, _cov: &Covariates, _x: &mut V) {}

fn ten_init(_p: &V, _t: f64, _cov: &Covariates, x: &mut V) {
    x[0] = 10.0;
}

fn state_output(x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V) {
    y[0] = x[0];
}

fn infusion_rate_function(
    _x: &V,
    _p: &V,
    _t: f64,
    dx: &mut V,
    _b: &V,
    rateiv: &V,
    _cov: &Covariates,
) {
    dx[0] = rateiv[0];
}

fn run_infusions(subject: &Subject, solver: OdeSolver) -> Vec<f64> {
    let ode = ODE::new(
        infusion_rate_function,
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        super::super::metadata::new("infusion_rate_ode")
            .states(["central"])
            .outputs(["cp"])
            .route(super::super::Route::infusion("iv").to_state("central")),
    )
    .expect("metadata should validate");

    let predictions = ode
        .simulate_subject(subject, &crate::parameters::dense([]), None)
        .expect("infusion simulation should succeed")
        .0;

    predictions
        .predictions()
        .iter()
        .map(|p| p.prediction())
        .collect()
}

fn counting_function(_x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, _rateiv: &V, _cov: &Covariates) {
    PREDICTION_CACHE_DIFFEQ_CALLS.fetch_add(1, Ordering::SeqCst);
    dx[0] = 0.0;
}

#[test]
fn handwritten_ode_metadata_exposes_name_lookup() {
    let ode = simple_ode()
        .with_metadata(
            super::super::metadata::new("bimodal_ke")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata attachment should validate");
    let metadata = ode.metadata().expect("metadata exists");

    assert_eq!(ode.parameter_index("ke"), Some(0));
    assert_eq!(ode.parameter_index("v"), Some(1));
    assert_eq!(ode.state_index("central"), Some(0));
    assert!(metadata.route("iv").is_some());
    assert!(metadata.output("cp").is_some());
    assert_eq!(metadata.kind(), ModelKind::Ode);
}

#[test]
fn handwritten_ode_without_metadata_keeps_raw_path() {
    let ode = simple_ode();

    assert!(ode.metadata().is_none());
    assert_eq!(ode.state_index("central"), None);
}

#[test]
fn handwritten_ode_rejects_dimension_mismatches() {
    let error = simple_ode()
        .with_metadata(
            super::super::metadata::new("wrong_outputs")
                .parameters(["ke"])
                .states(["central"])
                .outputs(["cp", "auc"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect_err("output-count mismatches must fail");

    assert_eq!(
        error,
        AttachMetadataError::OutputCountMismatch {
            expected: 1,
            declared: 2,
        }
    );
}

#[test]
fn handwritten_ode_rejects_invalid_metadata() {
    let error = simple_ode()
        .with_metadata(
            super::super::metadata::new("missing_destination")
                .parameters(["ke"])
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv")),
        )
        .expect_err("invalid metadata must fail during attachment");

    assert_eq!(
        error,
        AttachMetadataError::Validation(ModelMetadataError::MissingRouteDestination {
            route: "iv".to_string(),
        })
    );
}

#[test]
fn handwritten_ode_defaults_to_explicit_route_vectors() {
    let ode = ODE::new(
        explicit_route_function,
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        super::super::metadata::new("explicit_routes")
            .states(["central"])
            .outputs(["cp"])
            .routes([
                super::super::Route::bolus("oral").to_state("central"),
                super::super::Route::infusion("iv").to_state("central"),
            ]),
    )
    .expect("metadata attachment should validate");

    let predictions = ode
        .simulate_subject(&route_policy_subject(), &crate::parameters::dense([]), None)
        .expect("simulation should succeed")
        .0;
    let metadata = ode.metadata().expect("metadata exists");

    assert_eq!(
        metadata.route("oral").map(|route| route.input_index()),
        Some(0)
    );
    assert_eq!(
        metadata.route("iv").map(|route| route.input_index()),
        Some(0)
    );
    assert_relative_eq!(
        predictions.predictions()[0].prediction(),
        200.0,
        epsilon = 1e-6
    );
}

#[test]
fn handwritten_ode_metadata_input_policy_is_descriptive_only() {
    let ode = ODE::new(
        injected_route_function,
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        super::super::metadata::new("injected_routes")
            .states(["central"])
            .outputs(["cp"])
            .routes([
                super::super::Route::bolus("oral")
                    .to_state("central")
                    .inject_input_to_destination(),
                super::super::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ]),
    )
    .expect("metadata attachment should validate");

    let predictions = ode
        .simulate_subject(&route_policy_subject(), &crate::parameters::dense([]), None)
        .expect("simulation should succeed")
        .0;

    assert_relative_eq!(
        predictions.predictions()[0].prediction(),
        0.0,
        epsilon = 1e-6
    );
}

#[test]
fn handwritten_ode_metadata_resolves_numeric_labels_by_exact_name() {
    let ode = ODE::new(
        explicit_route_function,
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        super::super::metadata::new("numeric_alias_ode")
            .states(["central"])
            .outputs(["1"])
            .route(super::super::Route::infusion("1").to_state("central")),
    )
    .expect("metadata attachment should validate");

    let numeric = Subject::builder("numeric")
        .infusion(0.0, 100.0, "1", 1.0)
        .observation(1.0, 0.0, "1")
        .build();
    let prediction = ode
        .simulate_subject(&numeric, &crate::parameters::dense([]), None)
        .expect("numeric labels should resolve by exact name")
        .0
        .predictions()[0]
        .prediction();
    assert!(prediction > 0.0);

    // `input_1` / `outeq_1` are different names, not aliases of `1`.
    for prefixed in [
        Subject::builder("prefixed_input")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "1")
            .build(),
        Subject::builder("prefixed_output")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build(),
    ] {
        assert!(ode
            .simulate_subject(&prefixed, &crate::parameters::dense([]), None)
            .is_err());
    }
}

#[test]
fn changing_dimensions_after_metadata_clears_route_metadata() {
    let ode = simple_ode()
        .with_metadata(
            super::super::metadata::new("bimodal_ke")
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata attachment should validate")
        .with_ndrugs(2);

    assert!(ode.metadata().is_none());
}

#[test]
fn handwritten_ode_runs_init_for_each_occasion() {
    let ode = ODE::new(
        injected_route_function,
        zero_lag,
        unit_fa,
        ten_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1);
    let subject = Subject::builder("per_occasion_init")
        .observation(0.0, 0.0, 0)
        .reset()
        .observation(0.0, 0.0, 0)
        .build();

    let predictions = ode
        .estimate_predictions(&subject, &crate::parameters::dense([]))
        .expect("predictions should succeed");
    assert_eq!(
        predictions
            .predictions()
            .iter()
            .map(|prediction| prediction.prediction())
            .collect::<Vec<_>>(),
        vec![10.0, 10.0]
    );
}

#[test]
fn handwritten_ode_estimate_predictions_uses_prediction_cache() {
    PREDICTION_CACHE_DIFFEQ_CALLS.store(0, Ordering::SeqCst);

    let ode = ODE::new(
        counting_function,
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1);
    let subject = Subject::builder("cached_predictions")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .build();

    let first = ode
        .estimate_predictions(&subject, &crate::parameters::dense([]))
        .expect("first prediction run should succeed");
    let first_calls = PREDICTION_CACHE_DIFFEQ_CALLS.load(Ordering::SeqCst);
    assert!(first_calls > 0);

    let second = ode
        .estimate_predictions(&subject, &crate::parameters::dense([]))
        .expect("second prediction run should succeed");
    let second_calls = PREDICTION_CACHE_DIFFEQ_CALLS.load(Ordering::SeqCst);

    assert_eq!(first.predictions().len(), second.predictions().len());
    assert_eq!(first_calls, second_calls);
}

#[test]
fn ode_infusions_short_dose_conserved_after_infusion() {
    let subject = Subject::builder("short_infusion")
        .infusion(0.0, 100.0, "iv", 0.1)
        .observation(0.5, 0.0, "cp")
        .build();

    let predictions = run_infusions(&subject, OdeSolver::Bdf);

    assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
}

#[test]
fn ode_infusions_observation_at_infusion_end_uses_active_left_rate() {
    let subject = Subject::builder("infusion_end_observation")
        .infusion(0.0, 100.0, "iv", 0.1)
        .observation(0.1, 0.0, "cp")
        .observation(0.5, 0.0, "cp")
        .build();

    let predictions = run_infusions(&subject, OdeSolver::Bdf);

    assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
    assert_relative_eq!(predictions[1], 100.0, max_relative = 1e-4);
}

#[test]
fn ode_infusions_dose_conservation_with_multiple_solvers() {
    // Genuinely short infusion (0.01 h), representative of the reported
    // failures; the dose must be fully conserved across all solver types.
    let subject = Subject::builder("short_infusion_multi_solver")
        .infusion(0.0, 100.0, "iv", 0.01)
        .observation(0.01, 0.0, "cp")
        .build();

    let solvers = [
        (OdeSolver::Bdf, "Bdf"),
        (OdeSolver::Sdirk(SdirkTableau::TrBdf2), "TrBdf2"),
        (OdeSolver::Sdirk(SdirkTableau::Esdirk34), "Esdirk34"),
        (OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45), "Tsit45"),
    ];

    for (solver, _label) in solvers {
        let predictions = run_infusions(&subject, solver);
        assert_relative_eq!(predictions[0], 100.0, max_relative = 1e-4);
    }
}

#[test]
fn ode_infusions_delayed_short_dose_with_bdf() {
    let subject = Subject::builder("short_infusion_delayed_start_bdf")
        .observation(0.0, 0.0, "cp")
        .infusion(0.5, 100.0, "iv", 0.01)
        .observation(0.52, 0.0, "cp")
        .build();

    let predictions = run_infusions(&subject, OdeSolver::Bdf);

    assert_relative_eq!(predictions[1], 100.0, max_relative = 1e-4);
}

#[test]
fn ode_infusions_back_to_back_discontinuities_conserve_dose() {
    let subject = Subject::builder("back_to_back_infusions")
        .infusion(0.0, 100.0, "iv", 0.5)
        .infusion(0.5, 100.0, "iv", 0.5)
        .observation(1.0, 0.0, "cp")
        .build();

    let predictions = run_infusions(&subject, OdeSolver::Bdf);

    assert_relative_eq!(predictions[0], 200.0, max_relative = 1e-4);
}

#[test]
fn ode_infusions_accepted_stop_before_event_keeps_integrating() {
    // The infusion ends exactly one ULP after the observation at t = 10.
    // After landing on the observation stop the solver is already within
    // diffsol's round-off of the end boundary, so `set_stop_time` reports
    // `StopTimeAtCurrentTime`, which confirms the boundary is reached.
    // The reached boundary is *before* the observation at
    // t = 20, so the event loop must keep integrating toward it; breaking
    // early would evaluate the observation with the state frozen at the
    // boundary and miss the exponential decay.
    let subject = Subject::builder("accepted_stop_before_event")
        .infusion(5.0, 100.0, "iv", 10.0_f64.next_up() - 5.0)
        .observation(10.0, 0.0, "cp")
        .observation(20.0, 0.0, "cp")
        .build();

    let ode = ODE::new(
        |x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, rateiv: &V, _cov: &Covariates| {
            dx[0] = rateiv[0] - 0.5 * x[0];
        },
        zero_lag,
        unit_fa,
        zero_init,
        state_output,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(OdeSolver::Bdf)
    .with_metadata(
        super::super::metadata::new("accepted_stop_ode")
            .states(["central"])
            .outputs(["cp"])
            .route(super::super::Route::infusion("iv").to_state("central")),
    )
    .expect("metadata should validate");

    let predictions = ode
        .simulate_subject(&subject, &crate::parameters::dense([]), None)
        .expect("infusion simulation should succeed")
        .0;

    // Dose delivered over [5, 10], then exponential decay for 10 h:
    // (100 * (1 - exp(-0.5 * 5)) / (0.5 * 5)) * exp(-0.5 * 10).
    let delivered = 100.0 * (1.0 - (-2.5_f64).exp()) / 2.5;
    let expected = delivered * (-5.0_f64).exp();
    let pred = predictions.predictions()[1].prediction();
    assert_relative_eq!(pred, expected, max_relative = 1e-3);
}

#[test]
fn dense_observations_a_few_ulps_from_event_time_do_not_error() {
    // Regression test for the 0.28.5 event loop: dense prediction grids
    // built with floating-point accumulation (e.g. a `t += 0.05` grid over
    // 0..24) place requested times a few ULPs away from event times. Here
    // the bolus at t = 12 meets observations ~16 ULPs on either side;
    // `set_stop_time` reports `StopTimeAtCurrentTime` for those stops and
    // the loop must treat them as reached instead of erroring.
    let ulp = 12.0f64.next_up() - 12.0;
    let subject = Subject::builder("dense_grid_near_event")
        .bolus(0.0, 200.0, 0)
        .bolus(12.0, 100.0, 0)
        .missing_observation(0.0, "cp")
        .missing_observation(12.0 - 16.0 * ulp, "cp")
        .missing_observation(12.0 + 16.0 * ulp, "cp")
        .missing_observation(24.0, "cp")
        .build();

    let solvers = [
        (OdeSolver::Bdf, "Bdf"),
        (OdeSolver::Sdirk(SdirkTableau::TrBdf2), "TrBdf2"),
        (OdeSolver::Sdirk(SdirkTableau::Esdirk34), "Esdirk34"),
        (OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45), "Tsit45"),
    ];

    for (solver, label) in solvers {
        let ode = ODE::new(
            |x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, _rateiv: &V, _cov: &Covariates| {
                dx[0] = -0.3 * x[0];
            },
            zero_lag,
            unit_fa,
            zero_init,
            state_output,
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_solver(solver)
        .with_metadata(
            super::super::metadata::new("dense_grid_near_event_ode")
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::bolus("0").to_state("central")),
        )
        .expect("metadata should validate");

        let predictions = ode
            .simulate_subject(&subject, &crate::parameters::dense([]), None)
            .unwrap_or_else(|error| panic!("{label}: dense grid near event failed: {error}"))
            .0;
        assert_eq!(predictions.predictions().len(), 4);
    }
}

// debug/script.R hybrid phage model (subject `1` of dat.csv), with the
// parameters hardcoded. A large infusion rate (1e9 over 1.25e-3 h) makes
// the first Newton solve after every infusion boundary fail once with
// diffsol's convergence heuristics and then recover via step-size
// reduction; over a long horizon those individually recovered failures
// accumulate in diffsol's absolute `number_of_nonlinear_solver_fails`
// counter and blow up the run unless the boundary restart avoids them.
fn hybrid_phage_diffeq(x: &V, _p: &V, _t: f64, dx: &mut V, _b: &V, rateiv: &V, _cov: &Covariates) {
    let eps = 1.0e-12_f64;
    let soft = |v: f64| 0.5 * (v + (v * v + eps * eps).sqrt());

    let kep = 20.799022436141968;
    let k12 = 3.611151695251465;
    let k21 = 0.20569434165954592;
    let kdep = 3.674600839614868;
    let kcl_air = 98.17452669143677;
    let kgr = 2.072104573249817;
    let kinf = 1.909232258796692e-6;
    let c50 = 427933.12072753906;
    let klysis = 0.8622971177101135;
    let burst = 1.591451644897461;
    let ksp = 4.387639760971069;
    let kdp = 0.09175211071968079;
    let kn = 1.147785520553589;
    let va = 12.829959392547607;

    let phage_air = soft(x[2]);
    let bacc_pos = soft(x[3]);
    let binf_pos = soft(x[4]);
    let bprot_pos = soft(x[5]);
    let tb = bacc_pos + binf_pos + bprot_pos;
    let bmax = 10_f64.powi(10);
    let cair = phage_air / va;
    let inf_eff = kinf * cair / (1.0 + cair / c50);

    dx[0] = -(kep + k12 + kdep) * x[0] + k21 * x[1] + rateiv[0];
    dx[1] = k12 * x[0] - k21 * x[1];
    dx[2] = kdep * x[0] - kcl_air * x[2] - inf_eff * bacc_pos + burst * klysis * binf_pos;
    dx[3] = kgr * bacc_pos * (1.0 - tb / bmax) - inf_eff * bacc_pos - ksp * x[3] + kdp * x[5]
        - kn * x[3];
    dx[4] = inf_eff * bacc_pos - klysis * x[4];
    dx[5] = ksp * x[3] - kdp * x[5];
}

fn hybrid_phage_init(_p: &V, _t: f64, _cov: &Covariates, x: &mut V) {
    x[3] = 3.0 * 10_f64.powf(5.5);
}

fn run_hybrid_phage_infusions(subject: &Subject) -> Vec<f64> {
    let ode = ODE::new(
        hybrid_phage_diffeq,
        zero_lag,
        unit_fa,
        hybrid_phage_init,
        state_output,
    )
    .with_nstates(6)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(OdeSolver::Bdf)
    .with_metadata(
        super::super::metadata::new("hybrid_phage_infusions")
            .states(["plasma", "peripheral", "airway", "bacc", "binf", "bprot"])
            .outputs(["cp"])
            .route(super::super::Route::infusion("iv").to_state("plasma")),
    )
    .expect("metadata should validate");

    let predictions = ode
        .simulate_subject(subject, &crate::parameters::dense([]), None)
        .expect("hybrid phage simulation should succeed")
        .0;

    predictions
        .predictions()
        .iter()
        .map(|p| p.prediction())
        .collect()
}

#[test]
fn ode_infusions_long_horizon_boundary_failures_do_not_accumulate() {
    // Exact finite-f64 reproduction of debug/dat.csv subject `1`: 103
    // short infusions and 42 observations (with duplicates). Without
    // boundary restarts, individually recovered Newton failures accumulate
    // past diffsol's 50-failure limit. `build()` orders events by time, so
    // the two tables reproduce the interleaved source schedule exactly.
    const INFUSION_DURATION: f64 = 0.00125;
    const INFUSIONS: [(f64, f64); 103] = [
        (0.0, 1e+09),
        (0.5, 1e+09),
        (1.0, 1e+09),
        (1.5, 1e+09),
        (2.0, 1e+09),
        (2.5, 1e+09),
        (3.0, 1e+09),
        (3.5, 1e+09),
        (4.0, 1e+09),
        (4.5, 1e+09),
        (5.0, 1e+09),
        (5.5, 1e+09),
        (6.0, 1e+09),
        (6.5, 1e+09),
        (7.0, 1e+09),
        (7.5, 1e+09),
        (8.49083333333333, 1e+09),
        (8.99291666666667, 1e+09),
        (9.49291666666667, 1e+09),
        (9.99291666666667, 1e+09),
        (10.4929166666667, 1e+09),
        (10.9929166666667, 1e+09),
        (11.4929166666667, 1e+09),
        (12.0145833333333, 3e+09),
        (13.0154166666667, 3e+09),
        (14.0154166666667, 3e+09),
        (15.0154166666667, 3e+09),
        (16.0154166666667, 3e+09),
        (17.0154166666667, 3e+09),
        (18.0154166666667, 3e+09),
        (19.0154166666667, 3e+09),
        (20.0491666666667, 3e+09),
        (21.0491666666667, 3e+09),
        (22.0491666666667, 3e+09),
        (23.0491666666667, 3e+09),
        (26.05125, 3e+09),
        (27.05125, 3e+09),
        (28.05125, 3e+09),
        (29.05125, 3e+09),
        (30.05125, 3e+09),
        (33.0604166666667, 3e+09),
        (34.0604166666667, 3e+09),
        (35.0604166666667, 3e+09),
        (36.0604166666667, 3e+09),
        (37.0604166666667, 3e+09),
        (40.0604166666667, 3e+09),
        (41.0604166666667, 3e+09),
        (42.0604166666667, 3e+09),
        (43.0604166666667, 3e+09),
        (44.0604166666667, 3e+09),
        (47.0604166666667, 3e+09),
        (48.0604166666667, 3e+09),
        (49.0604166666667, 3e+09),
        (50.0604166666667, 3e+09),
        (51.0604166666667, 3e+09),
        (54.0366666666667, 3e+09),
        (55.0366666666667, 3e+09),
        (56.1741666666667, 3e+09),
        (57.1741666666667, 3e+09),
        (58.1741666666667, 3e+09),
        (61.1758333333333, 3e+09),
        (62.1758333333333, 3e+09),
        (63.1758333333333, 3e+09),
        (64.1758333333333, 3e+09),
        (65.1758333333333, 3e+09),
        (68.1758333333333, 3e+09),
        (69.1758333333333, 3e+09),
        (70.1758333333333, 3e+09),
        (71.1758333333333, 3e+09),
        (72.1758333333333, 3e+09),
        (75.1758333333333, 3e+09),
        (76.1758333333333, 3e+09),
        (77.1758333333333, 3e+09),
        (78.1758333333333, 3e+09),
        (79.1758333333333, 3e+09),
        (82.1758333333333, 3e+09),
        (83.1758333333333, 3e+09),
        (84.1758333333333, 3e+09),
        (85.1758333333333, 3e+09),
        (86.1758333333333, 3e+09),
        (89.1758333333333, 3e+09),
        (90.1758333333333, 3e+09),
        (91.1758333333333, 3e+09),
        (92.1758333333333, 3e+09),
        (93.1758333333333, 3e+09),
        (96.0279166666667, 4.5e+09),
        (97.0279166666667, 4.5e+09),
        (98.0279166666667, 4.5e+09),
        (99.0279166666667, 4.5e+09),
        (100.027916666667, 4.5e+09),
        (103.027916666667, 4.5e+09),
        (104.027916666667, 4.5e+09),
        (105.027916666667, 4.5e+09),
        (106.027916666667, 4.5e+09),
        (107.027916666667, 4.5e+09),
        (110.027916666667, 4.5e+09),
        (111.027916666667, 4.5e+09),
        (112.027916666667, 4.5e+09),
        (113.027916666667, 4.5e+09),
        (114.027916666667, 4.5e+09),
        (117.05, 4.5e+09),
        (118.05, 4.5e+09),
        (119.05, 4.5e+09),
    ];
    const OBSERVATION_TIMES: [f64; 42] = [
        0.005,
        0.0179166666666667,
        0.0220833333333333,
        0.0220833333333333,
        0.0345833333333333,
        0.0383333333333333,
        0.0383333333333333,
        0.0845833333333333,
        0.18625,
        8.99166666666667,
        8.99583333333333,
        9.01041666666667,
        9.07416666666667,
        9.16666666666667,
        12.0395833333333,
        12.0395833333333,
        13.01375,
        13.0179166666667,
        13.1925,
        13.1925,
        13.1925,
        13.26875,
        21.0333333333333,
        21.055,
        21.0679166666667,
        33.0533333333333,
        33.06375,
        34.0375,
        34.0375,
        56.2166666666667,
        56.2166666666667,
        58.0154166666667,
        58.0154166666667,
        91.0595833333333,
        93.0741666666667,
        93.0741666666667,
        112.027083333333,
        112.027083333333,
        112.074166666667,
        112.074166666667,
        114.133333333333,
        114.133333333333,
    ];

    let mut builder = Subject::builder("long_horizon_short_infusions");
    for (time, amount) in INFUSIONS {
        builder = builder.infusion(time, amount, "iv", INFUSION_DURATION);
    }
    for time in OBSERVATION_TIMES {
        builder = builder.missing_observation(time, "cp");
    }
    let subject = builder.build();

    let predictions = run_hybrid_phage_infusions(&subject);

    // The simulation must complete and produce a finite, positive plasma
    // prediction (the crash is the regression this guards against).
    assert!(predictions[0].is_finite());
    assert!(predictions[0] > 0.0);
}
