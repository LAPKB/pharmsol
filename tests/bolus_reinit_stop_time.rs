//! Regression coverage for accepted stop times around state and RHS discontinuities.
//!
//! A solver restart can land a few ULPs from a requested event or infusion
//! boundary while diffsol still correctly reports that stop as reached. The
//! event loop must accept diffsol's `StopTimeAtCurrentTime`, integrate any
//! distinct residual instead of relabeling the clock, and restart with the
//! post-boundary RHS.

use pharmsol::prelude::*;

#[cfg(feature = "dsl")]
use pharmsol::Cache;

#[cfg(feature = "dsl")]
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};

const OBSERVATION_TIMES: [f64; 15] = [
    0.0,
    0.35,
    0.516666666666667,
    0.983333333333333,
    1.48333333333333,
    2.0,
    2.5,
    3.0,
    4.0,
    4.98333333333333,
    6.98333333333333,
    7.98333333333333,
    10.0,
    11.0,
    12.0,
];

fn solver_cases() -> [(&'static str, OdeSolver); 4] {
    [
        ("BDF", OdeSolver::Bdf),
        ("TSIT45", OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
        ("TRBDF2", OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
        ("ESDIRK34", OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
    ]
}

const PARAMETERS: [(&str, f64); 5] = [
    ("ka", 3.6156922578811646),
    ("cl0", 1.0289061069488525),
    ("vc0", 187.13204860687256),
    ("q0", 2.4602913856506348),
    ("vp0", 58.32162380218506),
];

const CRUSHED_STOP_PARAMETERS: [(&str, f64); 5] = [
    ("ka", 1.9966840744018555),
    ("cl0", 4.55677330493927),
    ("vc0", 8.071004152297974),
    ("q0", 2.822192907333374),
    ("vp0", 59.72854793071747),
];

const CRUSHED_STOP_OBSERVATION_TIMES: [f64; 6] = [
    0.0,
    0.3,
    0.566666666666667,
    1.06666666666667,
    1.53333333333333,
    2.0,
];

const LAGGED_STOP_PARAMETERS: [(&str, f64); 7] = [
    ("lag1", 2.963254451751709),
    ("ka", 2.2961074113845825),
    ("cl0", 2.888387441635132),
    ("vc0", 89.66886115074158),
    ("q0", 81.8881014585495),
    ("vp0", 2264.831554889679),
    ("fa1", 0.409921451807021),
];

const LAGGED_STOP_OBSERVATION_TIMES: [f64; 16] = [
    0.0,
    0.25,
    0.516666666666667,
    1.01666666666667,
    1.51666666666667,
    2.03333333333333,
    2.53333333333333,
    3.01666666666667,
    4.0,
    5.0,
    7.01666666666667,
    7.98333333333333,
    9.0,
    10.2666666666667,
    11.2666666666667,
    12.2666666666667,
];

fn subject_with_bolus_history() -> Subject {
    let mut builder = Subject::builder("g34");
    for dose_index in -16..=0 {
        builder = builder.bolus(f64::from(dose_index) * 12.0, 750.0, "input_1");
    }
    for time in OBSERVATION_TIMES {
        builder = builder.missing_observation(time, "outeq_1");
    }
    builder.build()
}

fn subject_with_crushed_endpoint_step() -> Subject {
    let mut builder = Subject::builder("g10");
    for dose_index in -16..=0 {
        builder = builder.bolus(f64::from(dose_index) * 12.0, 1000.0, "input_1");
    }
    for time in CRUSHED_STOP_OBSERVATION_TIMES {
        builder = builder.missing_observation(time, "outeq_1");
    }
    builder.build()
}

fn subject_with_lagged_absolute_stop_roundoff() -> Subject {
    let mut builder = Subject::builder("g28");
    for dose_index in -16..=0 {
        builder = builder.bolus(f64::from(dose_index) * 12.0, 750.0, "input_1");
    }
    for time in LAGGED_STOP_OBSERVATION_TIMES {
        builder = builder.missing_observation(time, "outeq_1");
    }
    builder.build()
}

fn closure_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ka, cl0, vc0, q0, vp0);
            let ke = cl0 / vc0;
            let k23 = q0 / vc0;
            let k32 = q0 / vp0;

            dx[0] = b[0] - x[0] * ka;
            dx[1] = rateiv[0] + x[0] * ka + x[2] * k32 - x[1] * (ke + k23);
            dx[2] = x[1] * k23 - x[2] * k32;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _cl0, vc0, _q0, _vp0);
            y[0] = x[1] / vc0;
        },
    )
    .with_nstates(3)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("bolus_reinit_stop_time")
            .parameters(["ka", "cl0", "vc0", "q0", "vp0"])
            .states(["x1", "x2", "x3"])
            .outputs(["outeq_1"])
            .routes([equation::Route::bolus("input_1")
                .to_state("x1")
                .expect_explicit_input()]),
    )
    .expect("regression model metadata should validate")
}

fn lagged_closure_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, _lag1, ka, cl0, vc0, q0, vp0, _fa1);
            let ke = cl0 / vc0;
            let k23 = q0 / vc0;
            let k32 = q0 / vp0;

            dx[0] = b[0] - x[0] * ka;
            dx[1] = rateiv[0] + x[0] * ka + x[2] * k32 - x[1] * (ke + k23);
            dx[2] = x[1] * k23 - x[2] * k32;
        },
        |p, _t, _cov| {
            fetch_params!(p, lag1, _ka, _cl0, _vc0, _q0, _vp0, _fa1);
            lag! { 0 => lag1 }
        },
        |p, _t, _cov| {
            fetch_params!(p, _lag1, _ka, _cl0, _vc0, _q0, _vp0, fa1);
            fa! { 0 => fa1 }
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _lag1, _ka, _cl0, vc0, _q0, _vp0, _fa1);
            y[0] = x[1] / vc0;
        },
    )
    .with_nstates(3)
    .with_ndrugs(1)
    .with_nout(1)
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("lagged_absolute_stop_roundoff")
            .parameters(["lag1", "ka", "cl0", "vc0", "q0", "vp0", "fa1"])
            .states(["x1", "x2", "x3"])
            .outputs(["outeq_1"])
            .routes([equation::Route::bolus("input_1")
                .to_state("x1")
                .expect_explicit_input()]),
    )
    .expect("lagged regression model metadata should validate")
}

#[test]
fn closure_solvers_accept_reached_stop_after_bolus_restarts(
) -> Result<(), Box<dyn std::error::Error>> {
    for (label, solver) in solver_cases() {
        let model = closure_model(solver);
        let parameters = Parameters::with_model(&model, PARAMETERS)?;
        let predictions = model
            .estimate_predictions_dense(&subject_with_bolus_history(), parameters.as_slice())
            .unwrap_or_else(|error| panic!("{label}: bolus-history simulation failed: {error}"));

        assert_eq!(predictions.predictions().len(), OBSERVATION_TIMES.len());
        assert!(predictions
            .predictions()
            .iter()
            .all(|prediction| prediction.prediction().is_finite()));
    }
    Ok(())
}

#[test]
fn bdf_restores_crushed_timestep_after_successful_stop() -> Result<(), Box<dyn std::error::Error>> {
    let subject = subject_with_crushed_endpoint_step();
    let bdf_model = closure_model(OdeSolver::Bdf);
    let bdf_parameters = Parameters::with_model(&bdf_model, CRUSHED_STOP_PARAMETERS)?;
    let bdf_predictions =
        bdf_model.estimate_predictions_dense(&subject, bdf_parameters.as_slice())?;

    let reference_model = closure_model(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let reference_parameters = Parameters::with_model(&reference_model, CRUSHED_STOP_PARAMETERS)?;
    let reference_predictions =
        reference_model.estimate_predictions_dense(&subject, reference_parameters.as_slice())?;

    assert_eq!(
        bdf_predictions.predictions().len(),
        CRUSHED_STOP_OBSERVATION_TIMES.len()
    );
    for (bdf, reference) in bdf_predictions
        .predictions()
        .iter()
        .zip(reference_predictions.predictions())
    {
        let bdf = bdf.prediction();
        let reference = reference.prediction();
        let scaled_error = (bdf - reference).abs() / reference.abs().max(1.0);
        assert!(
            scaled_error < 1.0e-3,
            "BDF prediction {bdf:.16e} differs from TSIT45 reference {reference:.16e} by {scaled_error:.3e}"
        );
    }
    Ok(())
}

#[test]
fn bdf_accepts_exact_absolute_stop_with_local_roundoff() -> Result<(), Box<dyn std::error::Error>> {
    let subject = subject_with_lagged_absolute_stop_roundoff();
    let bdf_model = lagged_closure_model(OdeSolver::Bdf);
    let bdf_parameters = Parameters::with_model(&bdf_model, LAGGED_STOP_PARAMETERS)?;
    let bdf_predictions =
        bdf_model.estimate_predictions_dense(&subject, bdf_parameters.as_slice())?;

    let reference_model = lagged_closure_model(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let reference_parameters = Parameters::with_model(&reference_model, LAGGED_STOP_PARAMETERS)?;
    let reference_predictions =
        reference_model.estimate_predictions_dense(&subject, reference_parameters.as_slice())?;

    assert_eq!(
        bdf_predictions.predictions().len(),
        LAGGED_STOP_OBSERVATION_TIMES.len()
    );
    for (bdf, reference) in bdf_predictions
        .predictions()
        .iter()
        .zip(reference_predictions.predictions())
    {
        let bdf = bdf.prediction();
        let reference = reference.prediction();
        let scaled_error = (bdf - reference).abs() / reference.abs().max(1.0);
        assert!(
            scaled_error < 1.0e-3,
            "BDF prediction {bdf:.16e} differs from TSIT45 reference {reference:.16e} by {scaled_error:.3e}"
        );
    }
    Ok(())
}

#[cfg(feature = "dsl")]
const DSL_MODEL: &str = r#"
name = bolus_reinit_stop_time
kind = ode
params = ka, cl0, vc0, q0, vp0
states = x1, x2, x3
outputs = outeq_1

bolus(input_1) -> x1
infusion(input_1) -> x2

cl = cl0
vc = vc0
q = q0
vp = vp0
ke = cl / vc
k23 = q / vc
k32 = q / vp

dx(x1) = -(x1 * ka)
dx(x2) = x1 * ka + (x3 * k32) - (x2 * (ke + k23))
dx(x3) = x2 * k23 - (x3 * k32)

out(outeq_1) = x2 / vc
"#;

fn infusion_boundary_subject() -> Subject {
    Subject::builder("accepted_infusion_boundary")
        .infusion(5.0, 100.0, "input_1", 10.0_f64.next_up() - 5.0)
        .missing_observation(10.0, "cp")
        .missing_observation(20.0, "cp")
        .build()
}

fn stepped_infusion_boundary_subject() -> Subject {
    let mut builder =
        Subject::builder("stepped_infusion_boundary").infusion(0.0, 100.0, "input_1", 12.0);
    for time in OBSERVATION_TIMES.into_iter().filter(|time| *time < 12.0) {
        builder = builder.missing_observation(time, "cp");
    }
    builder.missing_observation(20.0, "cp").build()
}

fn closure_infusion_model(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, _p, _t, dx, _b, rateiv, _cov| {
            dx[0] = rateiv[0] - 0.5 * x[0];
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
        equation::metadata::new("accepted_infusion_boundary")
            .states(["central"])
            .outputs(["cp"])
            .routes([equation::Route::infusion("input_1").to_state("central")]),
    )
    .expect("infusion metadata should validate")
}

fn accepted_boundary_expected() -> f64 {
    let delivered = 100.0 * (1.0 - (-2.5_f64).exp()) / 2.5;
    delivered * (-5.0_f64).exp()
}

fn stepped_boundary_expected() -> f64 {
    (100.0 / 12.0) / 0.5 * (1.0 - (-6.0_f64).exp()) * (-4.0_f64).exp()
}

fn assert_post_infusion_decay(
    label: &str,
    predictions: &SubjectPredictions,
    expected: f64,
    maximum_relative_error: f64,
) {
    assert!(!predictions.predictions().is_empty(), "{label}");
    let actual = predictions.predictions().last().unwrap().prediction();
    let relative_error = (actual - expected).abs() / expected;
    assert!(
        relative_error < maximum_relative_error,
        "{label}: post-infusion prediction {actual:.16e}, expected {expected:.16e}, relative error {relative_error:.3e}"
    );
}

#[test]
fn closure_solvers_restart_with_post_infusion_rhs() {
    for (label, solver) in solver_cases() {
        let model = closure_infusion_model(solver);
        for (scenario, subject, expected) in [
            (
                "accepted",
                infusion_boundary_subject(),
                accepted_boundary_expected(),
            ),
            (
                "stepped",
                stepped_infusion_boundary_subject(),
                stepped_boundary_expected(),
            ),
        ] {
            let predictions = model
                .estimate_predictions_dense(&subject, &[])
                .unwrap_or_else(|error| {
                    panic!("closure {label} {scenario} infusion failed: {error}")
                });
            let maximum_relative_error = if label == "TSIT45" { 1.0e-3 } else { 1.0e-2 };
            assert_post_infusion_decay(
                &format!("closure {label} {scenario}"),
                &predictions,
                expected,
                maximum_relative_error,
            );
        }
    }
}

#[cfg(feature = "dsl")]
const INFUSION_DSL_MODEL: &str = r#"
name = accepted_infusion_boundary
kind = ode
states = central
outputs = cp
infusion(input_1) -> central
dx(central) = -(0.5 * central)
out(cp) = central
"#;

#[test]
#[cfg(feature = "dsl")]
fn jit_solvers_restart_with_post_infusion_rhs() -> Result<(), Box<dyn std::error::Error>> {
    for (label, solver) in solver_cases() {
        let compiled = compile_module_source_to_runtime(
            INFUSION_DSL_MODEL,
            Some("accepted_infusion_boundary"),
            |_, _| {},
        )?;
        let model = match compiled {
            CompiledRuntimeModel::Ode(model) => {
                CompiledRuntimeModel::Ode(model.with_solver(solver))
            }
            _ => return Err("expected an ODE model".into()),
        };
        for (scenario, subject, expected) in [
            (
                "accepted",
                infusion_boundary_subject(),
                accepted_boundary_expected(),
            ),
            (
                "stepped",
                stepped_infusion_boundary_subject(),
                stepped_boundary_expected(),
            ),
        ] {
            let predictions = match &model {
                CompiledRuntimeModel::Ode(model) => model
                    .estimate_predictions_dense(&subject, &[])
                    .unwrap_or_else(|error| {
                        panic!("JIT {label} {scenario} infusion failed: {error}")
                    }),
                _ => unreachable!(),
            };
            let maximum_relative_error = if label == "TSIT45" { 1.0e-3 } else { 1.0e-2 };
            assert_post_infusion_decay(
                &format!("JIT {label} {scenario}"),
                &predictions,
                expected,
                maximum_relative_error,
            );
        }
    }
    Ok(())
}

fn material_short_infusion_subject() -> Subject {
    Subject::builder("material_short_infusion")
        .infusion(1.0, 100.0, "input_1", 1.0_f64.next_up() - 1.0)
        .missing_observation(2.0, "cp")
        .build()
}

fn material_short_infusion_expected() -> f64 {
    let duration = 1.0_f64.next_up() - 1.0;
    let elimination = 0.5;
    let state_at_end = 100.0 * -(-elimination * duration).exp_m1() / (elimination * duration);
    state_at_end * (-elimination * (2.0 - (1.0 + duration))).exp()
}

#[test]
fn closure_solvers_integrate_material_short_infusions() {
    for (label, solver) in solver_cases() {
        let model = closure_infusion_model(solver);
        let predictions = model
            .estimate_predictions_dense(&material_short_infusion_subject(), &[])
            .unwrap_or_else(|error| panic!("closure {label} short infusion failed: {error}"));
        assert_post_infusion_decay(
            &format!("closure {label} short infusion"),
            &predictions,
            material_short_infusion_expected(),
            1.0e-2,
        );
    }
}

#[test]
#[cfg(feature = "dsl")]
fn jit_solvers_integrate_material_short_infusions() -> Result<(), Box<dyn std::error::Error>> {
    for (label, solver) in solver_cases() {
        let compiled = compile_module_source_to_runtime(
            INFUSION_DSL_MODEL,
            Some("accepted_infusion_boundary"),
            |_, _| {},
        )?;
        let model = match compiled {
            CompiledRuntimeModel::Ode(model) => {
                CompiledRuntimeModel::Ode(model.with_solver(solver))
            }
            _ => return Err("expected an ODE model".into()),
        };
        let predictions = match &model {
            CompiledRuntimeModel::Ode(model) => model
                .estimate_predictions_dense(&material_short_infusion_subject(), &[])
                .unwrap_or_else(|error| panic!("JIT {label} short infusion failed: {error}")),
            _ => unreachable!(),
        };
        assert_post_infusion_decay(
            &format!("JIT {label} short infusion"),
            &predictions,
            material_short_infusion_expected(),
            1.0e-2,
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "dsl")]
fn jit_solvers_accept_reached_stop_after_bolus_restarts() -> Result<(), Box<dyn std::error::Error>>
{
    for (label, solver) in solver_cases() {
        let compiled =
            compile_module_source_to_runtime(DSL_MODEL, Some("bolus_reinit_stop_time"), |_, _| {})?;
        let model = match compiled {
            CompiledRuntimeModel::Ode(model) => {
                CompiledRuntimeModel::Ode(model.with_solver(solver))
            }
            _ => return Err("expected an ODE model".into()),
        };
        let parameters = Parameters::with_model(&model, PARAMETERS)?;

        let predictions = match &model {
            CompiledRuntimeModel::Ode(model) => model
                .estimate_predictions_dense(&subject_with_bolus_history(), parameters.as_slice())
                .unwrap_or_else(|error| panic!("{label}: JIT bolus-history failed: {error}")),
            _ => unreachable!(),
        };

        assert_eq!(predictions.predictions().len(), OBSERVATION_TIMES.len());
        assert!(predictions
            .predictions()
            .iter()
            .all(|prediction| prediction.prediction().is_finite()));
    }
    Ok(())
}

#[test]
#[cfg(feature = "dsl")]
fn jit_bdf_restores_crushed_timestep_after_successful_stop(
) -> Result<(), Box<dyn std::error::Error>> {
    let compiled =
        compile_module_source_to_runtime(DSL_MODEL, Some("bolus_reinit_stop_time"), |_, _| {})?;
    let model = match compiled {
        CompiledRuntimeModel::Ode(model) => {
            CompiledRuntimeModel::Ode(model.with_solver(OdeSolver::Bdf).disable_cache())
        }
        _ => return Err("expected an ODE model".into()),
    };
    let parameters = Parameters::with_model(&model, CRUSHED_STOP_PARAMETERS)?;
    let predictions = match &model {
        CompiledRuntimeModel::Ode(model) => model.estimate_predictions_dense(
            &subject_with_crushed_endpoint_step(),
            parameters.as_slice(),
        )?,
        _ => unreachable!(),
    };

    assert_eq!(
        predictions.predictions().len(),
        CRUSHED_STOP_OBSERVATION_TIMES.len()
    );
    assert!(predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.prediction().is_finite()));
    Ok(())
}

#[cfg(feature = "dsl")]
const CLOSE_GAP_DSL_MODEL: &str = r#"
name = close_gap_exponential
kind = ode
params = growth_rate
states = amount
outputs = cp
init(amount) = 1
dx(amount) = growth_rate * amount
out(cp) = amount
"#;

#[cfg(feature = "dsl")]
const ABSOLUTE_TIME_DSL_MODEL: &str = r#"
name = absolute_time_close_gap
kind = ode
params = slope
states = amount
outputs = cp
init(amount) = 1
dx(amount) = slope * time
out(cp) = amount
"#;

#[cfg(feature = "dsl")]
const PER_OCCASION_INIT_DSL_MODEL: &str = r#"
name = per_occasion_init
kind = ode
states = amount
outputs = cp
bolus(input_1) -> amount
init(amount) = 10
dx(amount) = 0
out(cp) = amount
"#;

#[cfg(feature = "dsl")]
const DERIVED_FA_DSL_MODEL: &str = r#"
name = derived_fa
kind = ode
params = scale
states = amount
outputs = cp
bolus(input_1) -> amount
fa(input_1) = dose_scale
dose_scale = scale
dx(amount) = 0
out(cp) = amount
"#;

#[cfg(feature = "dsl")]
const REBASE_ABSOLUTE_TIME_DSL_MODEL: &str = r#"
name = rebase_absolute_time_close_gap
kind = ode
states = amount, time_weighted_amount
outputs = cp, callback
infusion(input_1) -> amount
dx(amount) = 0
dx(time_weighted_amount) = rate(input_1) * time
out(cp) = amount
out(callback) = time_weighted_amount
"#;

#[cfg(feature = "dsl")]
const LOCF_DSL_MODEL: &str = r#"
name = locf_integral
kind = ode
covariates = cov_rate @locf
states = integral
outputs = cp
dx(integral) = cov_rate
out(cp) = integral
"#;

#[cfg(feature = "dsl")]
fn configured_runtime_ode(
    compiled: &CompiledRuntimeModel,
    solver: OdeSolver,
) -> CompiledRuntimeModel {
    match compiled.clone() {
        CompiledRuntimeModel::Ode(model) => {
            CompiledRuntimeModel::Ode(model.with_solver(solver).disable_cache())
        }
        _ => panic!("expected an ODE model"),
    }
}

#[cfg(feature = "dsl")]
fn assert_runtime_close_gaps(backend: &str, compiled: &CompiledRuntimeModel) {
    let mut failures = Vec::new();
    for (solver_name, solver) in solver_cases() {
        for (case_name, start, stop, exponent) in [
            ("small-time-growth", 0.0, 3.0e-13, 0.3),
            ("small-time-decay", 0.0, 3.0e-13, -0.3),
            ("large-time-growth", 1.0e6, 1.0e6_f64.next_up(), 0.3),
            ("large-time-decay", 1.0e6, 1.0e6_f64.next_up(), -0.3),
        ] {
            let gap = stop - start;
            let rate = exponent / gap;
            let subject_id = format!("{backend}-close-gap-{case_name}-{solver_name}");
            let subject = Subject::builder(&subject_id)
                .missing_observation(start, "cp")
                .missing_observation(stop, "cp")
                .build();
            let model = configured_runtime_ode(compiled, solver.clone());
            let result = match &model {
                CompiledRuntimeModel::Ode(model) => {
                    model.estimate_predictions_dense(&subject, &[rate])
                }
                _ => unreachable!(),
            };

            match result {
                Ok(predictions) => {
                    let actual = predictions
                        .predictions()
                        .last()
                        .expect("close-gap runtime case should produce a prediction")
                        .prediction();
                    let expected = exponent.exp();
                    let relative_error =
                        (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
                    if relative_error > 1.0e-3 {
                        failures.push(format!(
                            "{backend} {solver_name} {case_name}: solver silently skipped material \
                             dynamics: prediction {actual:.16e}, expected {expected:.16e}, relative \
                             error {relative_error:.3e}"
                        ));
                    }
                }
                Err(error) => failures.push(format!(
                    "{backend} {solver_name} {case_name}: solver must integrate the distinct \
                     interval from t = {start:.16e} to t = {stop:.16e}, but returned: {error}"
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "dsl")]
fn assert_runtime_large_initial_times_absolute_time(
    backend: &str,
    compiled: &CompiledRuntimeModel,
) {
    let expected_change = 0.3;
    let mut failures = Vec::new();
    for (solver_name, solver) in solver_cases() {
        for (case_name, start) in [
            ("large-positive", 1.0e6_f64),
            ("large-negative", -1.0e6_f64),
        ] {
            let stop = start.next_up();
            let integrated_time = 0.5 * (start + stop) * (stop - start);
            let rate = expected_change / integrated_time;
            let subject = Subject::builder(format!("{backend}-absolute-time-{case_name}"))
                .missing_observation(start, "cp")
                .missing_observation(stop, "cp")
                .build();
            let model = configured_runtime_ode(compiled, solver.clone());
            let result = match &model {
                CompiledRuntimeModel::Ode(model) => {
                    model.estimate_predictions_dense(&subject, &[rate])
                }
                _ => unreachable!(),
            };

            match result {
                Ok(predictions) => {
                    let actual = predictions
                        .predictions()
                        .last()
                        .expect("absolute-time runtime case should produce a prediction")
                        .prediction();
                    let relative_error =
                        (actual - (1.0 + expected_change)).abs() / (1.0 + expected_change);
                    if relative_error > 1.0e-3 {
                        failures.push(format!(
                            "{backend} {solver_name} {case_name}: absolute callback result {actual:.16e}, expected {:.16e}, relative error {relative_error:.3e}",
                            1.0 + expected_change
                        ));
                    }
                }
                Err(error) => failures.push(format!(
                    "{backend} {solver_name} {case_name}: absolute-time simulation failed: {error}"
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "dsl")]
fn assert_runtime_rebase_after_large_event(backend: &str, compiled: &CompiledRuntimeModel) {
    let center = 1.0e12_f64;
    let stop = center.next_up();
    let gap = stop - center;
    let delivered = 0.3;
    let expected_time_weighted_amount = delivered * 0.5 * (center + stop);
    let subject = Subject::builder(format!("{backend}-rebase-after-large-event"))
        .infusion(center, delivered, "input_1", gap)
        .missing_observation(0.0, "cp")
        .missing_observation(center, "cp")
        .missing_observation(stop, "cp")
        .missing_observation(center, "callback")
        .missing_observation(stop, "callback")
        .build();
    let mut failures = Vec::new();

    for (solver_name, solver) in solver_cases() {
        let model = configured_runtime_ode(compiled, solver);
        let result = match &model {
            CompiledRuntimeModel::Ode(model) => model.estimate_predictions_dense(&subject, &[]),
            _ => unreachable!(),
        };
        match result {
            Ok(predictions) => {
                if predictions.predictions().len() != 5 {
                    failures.push(format!(
                        "{backend} {solver_name}: expected five rebase observations, got {}",
                        predictions.predictions().len()
                    ));
                    continue;
                }
                let value_at = |time: f64, outeq: usize| {
                    predictions
                        .predictions()
                        .iter()
                        .find(|prediction| {
                            prediction.time() == time && prediction.outeq() == outeq
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "{backend} {solver_name}: missing prediction at t = {time}, output {outeq}"
                            )
                        })
                        .prediction()
                };
                let amount_at_center = value_at(center, 0);
                let amount_at_stop = value_at(stop, 0);
                let time_weighted_at_stop = value_at(stop, 1);
                let amount_error = (amount_at_stop - delivered).abs();
                let callback_error = (time_weighted_at_stop - expected_time_weighted_amount).abs();
                if amount_at_center.abs() > 5.0e-3
                    || amount_error > 5.0e-3
                    || callback_error > 1.0e-3 * expected_time_weighted_amount.abs()
                {
                    failures.push(format!(
                        "{backend} {solver_name}: rebase amount at center/stop [{amount_at_center:.16e}, {amount_at_stop:.16e}], expected stop {delivered:.16e}; time-weighted stop {time_weighted_at_stop:.16e}, expected {expected_time_weighted_amount:.16e}"
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "{backend} {solver_name}: rebase simulation failed: {error}"
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "dsl")]
fn runtime_locf_subject() -> Subject {
    let mut subject = Subject::builder("runtime-locf-breakpoint")
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

#[cfg(feature = "dsl")]
fn assert_runtime_locf_integral(backend: &str, compiled: &CompiledRuntimeModel) {
    let expected = 15.0;
    for (solver_name, solver) in solver_cases() {
        let model = configured_runtime_ode(compiled, solver);
        let predictions = match &model {
            CompiledRuntimeModel::Ode(model) => model
                .estimate_predictions_dense(&runtime_locf_subject(), &[])
                .unwrap_or_else(|error| {
                    panic!("{backend} {solver_name} LOCF simulation failed: {error}")
                }),
            _ => unreachable!(),
        };
        let actual = predictions
            .predictions()
            .last()
            .expect("runtime LOCF case should produce a prediction")
            .prediction();
        let absolute_error = (actual - expected).abs();
        assert!(
            absolute_error <= 5.0e-3,
            "{backend} {solver_name}: LOCF integral {actual:.16e}, expected {expected:.16e}, \
             absolute error {absolute_error:.3e}"
        );
    }
}

#[cfg(feature = "dsl")]
#[test]
fn jit_large_initial_times_preserve_absolute_time_callbacks(
) -> Result<(), Box<dyn std::error::Error>> {
    let compiled = compile_module_source_to_runtime(
        ABSOLUTE_TIME_DSL_MODEL,
        Some("absolute_time_close_gap"),
        |_, _| {},
    )?;
    assert_runtime_large_initial_times_absolute_time("JIT", &compiled);
    Ok(())
}

#[cfg(feature = "dsl")]
#[test]
fn jit_runs_user_init_before_each_occasion() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = compile_module_source_to_runtime(
        PER_OCCASION_INIT_DSL_MODEL,
        Some("per_occasion_init"),
        |_, _| {},
    )?;
    let subject = Subject::builder("per-occasion-init")
        .bolus(0.0, 0.0, "input_1")
        .missing_observation(0.0, "cp")
        .reset()
        .bolus(0.0, 0.0, "input_1")
        .missing_observation(0.0, "cp")
        .build();
    let predictions = match compiled {
        CompiledRuntimeModel::Ode(model) => model.estimate_predictions_dense(&subject, &[])?,
        _ => unreachable!("expected an ODE model"),
    };

    assert_eq!(
        predictions
            .predictions()
            .iter()
            .map(|prediction| prediction.prediction())
            .collect::<Vec<_>>(),
        vec![10.0, 10.0]
    );
    Ok(())
}

#[cfg(feature = "dsl")]
#[test]
fn jit_applies_derived_bioavailability() -> Result<(), Box<dyn std::error::Error>> {
    let compiled =
        compile_module_source_to_runtime(DERIVED_FA_DSL_MODEL, Some("derived_fa"), |_, _| {})?;
    let subject = Subject::builder("derived-fa")
        .bolus(0.0, 2.0, "input_1")
        .missing_observation(1.0, "cp")
        .build();
    let predictions = match compiled {
        CompiledRuntimeModel::Ode(model) => model.estimate_predictions_dense(&subject, &[3.0])?,
        _ => unreachable!("expected an ODE model"),
    };

    assert_eq!(predictions.predictions()[0].prediction(), 6.0);
    Ok(())
}

#[cfg(feature = "dsl")]
#[test]
fn jit_rebase_after_large_event_preserves_absolute_time_rhs(
) -> Result<(), Box<dyn std::error::Error>> {
    let compiled = compile_module_source_to_runtime(
        REBASE_ABSOLUTE_TIME_DSL_MODEL,
        Some("rebase_absolute_time_close_gap"),
        |_, _| {},
    )?;
    assert_runtime_rebase_after_large_event("JIT", &compiled);
    Ok(())
}

#[cfg(feature = "dsl")]
#[test]
fn jit_close_distinct_stops_do_not_silently_skip_dynamics() -> Result<(), Box<dyn std::error::Error>>
{
    let compiled = compile_module_source_to_runtime(
        CLOSE_GAP_DSL_MODEL,
        Some("close_gap_exponential"),
        |_, _| {},
    )?;
    assert_runtime_close_gaps("JIT", &compiled);
    Ok(())
}

#[cfg(feature = "dsl")]
#[test]
fn jit_locf_rhs_change_matches_piecewise_analytical_integral(
) -> Result<(), Box<dyn std::error::Error>> {
    let compiled =
        compile_module_source_to_runtime(LOCF_DSL_MODEL, Some("locf_integral"), |_, _| {})?;
    assert_runtime_locf_integral("JIT", &compiled);
    Ok(())
}
