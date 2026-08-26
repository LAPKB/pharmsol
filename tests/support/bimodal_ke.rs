#![allow(dead_code)]

use std::error::Error;
use std::io;

use pharmsol::prelude::*;

pub const MODEL_NAME: &str = "bimodal_ke";
pub const OBSERVATION_TIMES: [f64; 7] = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
pub const SUPPORT_POINT: [f64; 2] = [1.2, 50.0];

pub const AUTHORING_DSL: &str = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

fn subject_for_indices(route_index: usize, output_index: usize) -> Subject {
    let mut builder = Subject::builder(MODEL_NAME).infusion(0.0, 500.0, route_index, 0.5);
    for time in OBSERVATION_TIMES {
        builder = builder.missing_observation(time, output_index);
    }
    builder.build()
}

fn subject_for_labels(route_label: &str, output_label: &str) -> Subject {
    let mut builder = Subject::builder(MODEL_NAME).infusion(0.0, 500.0, route_label, 0.5);
    for time in OBSERVATION_TIMES {
        builder = builder.missing_observation(time, output_label);
    }
    builder.build()
}

pub fn subject() -> Subject {
    subject_for_labels("iv", "cp")
}

#[cfg(feature = "dsl")]
pub fn subject_for_runtime_model(model: &pharmsol::dsl::CompiledRuntimeModel) -> Subject {
    let route_label = if model.info().routes.iter().any(|route| route.name == "iv") {
        "iv"
    } else if model
        .info()
        .routes
        .iter()
        .any(|route| route.name == "input_0")
    {
        "input_0"
    } else {
        panic!("bimodal_ke route is available");
    };
    assert!(
        model
            .info()
            .outputs
            .iter()
            .any(|output| output.name == "cp"),
        "cp output is available"
    );
    subject_for_labels(route_label, "cp")
}

pub fn reference_values() -> Result<Vec<f64>, Box<dyn Error>> {
    let model = equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new(MODEL_NAME)
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ),
    )
    .expect("bimodal_ke metadata should validate");

    let parameters =
        Parameters::with_model(&model, [("ke", SUPPORT_POINT[0]), ("v", SUPPORT_POINT[1])])
            .expect("bimodal_ke parameters should validate");

    let predictions = model.estimate_predictions(&subject(), &parameters)?;

    Ok(predictions.flat_predictions())
}

pub fn report_values(label: &str, actual: &[f64], tolerance: f64) -> Result<(), Box<dyn Error>> {
    let expected = reference_values()?;
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "{label}: expected {} predictions, got {}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    println!("{label}");
    println!(
        "{:<6} {:>14} {:>14} {:>14}",
        "t", "expected", "actual", "abs diff"
    );

    let mut max_abs_diff: f64 = 0.0;
    for ((time, expected_value), actual_value) in OBSERVATION_TIMES
        .iter()
        .zip(expected.iter())
        .zip(actual.iter())
    {
        let abs_diff = (expected_value - actual_value).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        println!(
            "{:<6.1} {:>14.6} {:>14.6} {:>14.6}",
            time, expected_value, actual_value, abs_diff
        );
        if abs_diff > tolerance {
            return Err(io::Error::other(format!(
                "{label}: prediction at t={time:.1} differed by {abs_diff:.6} (tolerance {tolerance:.6})"
            ))
            .into());
        }
    }

    println!("max abs diff: {:.6}\n", max_abs_diff);
    Ok(())
}

pub fn report_subject_predictions(
    label: &str,
    predictions: &SubjectPredictions,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let values = predictions.flat_predictions();
    report_values(label, &values, tolerance)
}

#[cfg(feature = "dsl")]
pub fn report_runtime_model(
    label: &str,
    model: &pharmsol::dsl::CompiledRuntimeModel,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let support =
        Parameters::with_model(model, [("ke", SUPPORT_POINT[0]), ("v", SUPPORT_POINT[1])])?;
    let predictions = model
        .estimate_predictions(&subject_for_runtime_model(model), &support)?
        .into_subject()
        .ok_or_else(|| io::Error::other(format!("{label}: expected subject predictions")))?;

    report_subject_predictions(label, &predictions, tolerance)
}

#[cfg(feature = "dsl")]
pub fn compile_runtime_jit_model() -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    Ok(pharmsol::dsl::compile_module_source_to_runtime(
        AUTHORING_DSL,
        Some(MODEL_NAME),
        pharmsol::dsl::RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?)
}
