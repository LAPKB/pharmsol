//! Regression coverage for solver restarts after bolus state changes.
//!
//! A TSIT45 restart can land a few ULPs short of an observation time while
//! diffsol still correctly reports that the requested stop was reached. The
//! event loop must accept diffsol's `StopTimeAtCurrentTime` response rather than
//! requesting the same stop again and turning it into a simulation error.

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
fn closure_tsit45_accepts_reached_stop_after_bolus_restarts(
) -> Result<(), Box<dyn std::error::Error>> {
    let model = closure_model(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let parameters = Parameters::with_model(&model, PARAMETERS)?;
    let predictions =
        model.estimate_predictions_dense(&subject_with_bolus_history(), parameters.as_slice())?;

    assert_eq!(predictions.predictions().len(), OBSERVATION_TIMES.len());
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

#[test]
#[cfg(feature = "dsl-jit")]
fn jit_tsit45_accepts_reached_stop_after_bolus_restarts() -> Result<(), Box<dyn std::error::Error>>
{
    let compiled = compile_module_source_to_runtime(
        DSL_MODEL,
        Some("bolus_reinit_stop_time"),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?;
    let model = match compiled {
        CompiledRuntimeModel::Ode(model) => CompiledRuntimeModel::Ode(
            model.with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
        ),
        _ => return Err("expected an ODE model".into()),
    };
    let parameters = Parameters::with_model(&model, PARAMETERS)?;

    let predictions = match &model {
        CompiledRuntimeModel::Ode(model) => model
            .estimate_predictions_dense(&subject_with_bolus_history(), parameters.as_slice())?,
        _ => unreachable!(),
    };

    assert_eq!(predictions.predictions().len(), OBSERVATION_TIMES.len());
    Ok(())
}
