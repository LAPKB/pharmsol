//! Regression coverage for accepted stop times around state and RHS discontinuities.
//!
//! A solver restart can land a few ULPs from a requested event or infusion
//! boundary while diffsol still correctly reports that stop as reached. The
//! event loop must accept diffsol's `StopTimeAtCurrentTime`, align the logical
//! state time with the accepted stop, and restart with the post-boundary RHS.

use pharmsol::prelude::*;

#[cfg(feature = "dsl-jit")]
use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeCompilationTarget,
};

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

#[cfg(feature = "dsl-jit")]
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

#[cfg(feature = "dsl-jit")]
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
#[cfg(feature = "dsl-jit")]
fn jit_solvers_restart_with_post_infusion_rhs() -> Result<(), Box<dyn std::error::Error>> {
    for (label, solver) in solver_cases() {
        let compiled = compile_module_source_to_runtime(
            INFUSION_DSL_MODEL,
            Some("accepted_infusion_boundary"),
            RuntimeCompilationTarget::Jit,
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

fn assert_material_infusion_error(label: &str, error: PharmsolError) {
    let message = error.to_string();
    assert!(
        message.contains("would skip infusion amount"),
        "{label}: unexpected error: {message}"
    );
}

#[test]
fn closure_solvers_reject_material_coalesced_infusions() {
    for (label, solver) in solver_cases() {
        let model = closure_infusion_model(solver);
        let error = model
            .estimate_predictions_dense(&material_short_infusion_subject(), &[])
            .unwrap_err();
        assert_material_infusion_error(&format!("closure {label}"), error);
    }
}

#[test]
#[cfg(feature = "dsl-jit")]
fn jit_solvers_reject_material_coalesced_infusions() -> Result<(), Box<dyn std::error::Error>> {
    for (label, solver) in solver_cases() {
        let compiled = compile_module_source_to_runtime(
            INFUSION_DSL_MODEL,
            Some("accepted_infusion_boundary"),
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )?;
        let model = match compiled {
            CompiledRuntimeModel::Ode(model) => {
                CompiledRuntimeModel::Ode(model.with_solver(solver))
            }
            _ => return Err("expected an ODE model".into()),
        };
        let error = match &model {
            CompiledRuntimeModel::Ode(model) => model
                .estimate_predictions_dense(&material_short_infusion_subject(), &[])
                .unwrap_err(),
            _ => unreachable!(),
        };
        assert_material_infusion_error(&format!("JIT {label}"), error);
    }
    Ok(())
}

#[test]
#[cfg(feature = "dsl-jit")]
fn jit_solvers_accept_reached_stop_after_bolus_restarts() -> Result<(), Box<dyn std::error::Error>>
{
    for (label, solver) in solver_cases() {
        let compiled = compile_module_source_to_runtime(
            DSL_MODEL,
            Some("bolus_reinit_stop_time"),
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )?;
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
