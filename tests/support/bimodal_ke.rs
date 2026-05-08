#![allow(dead_code)]

use std::error::Error;
use std::io;
use std::path::PathBuf;

use pharmsol::prelude::*;
use tempfile::{tempdir, TempDir};

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

#[derive(Debug)]
pub struct ArtifactWorkspace {
    tempdir: TempDir,
}

impl ArtifactWorkspace {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            tempdir: tempdir()?,
        })
    }

    pub fn aot_output(&self, stem: &str) -> PathBuf {
        self.tempdir.path().join(format!("{stem}.pkm"))
    }

    pub fn build_root(&self, stem: &str) -> PathBuf {
        self.tempdir.path().join(stem)
    }
}

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
    subject_for_indices(0, 0)
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]
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
    let predictions = equation::ODE::new(
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
    .estimate_predictions(&subject(), &SUPPORT_POINT)?;

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

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]
pub fn report_runtime_model(
    label: &str,
    model: &pharmsol::dsl::CompiledRuntimeModel,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let predictions = model
        .estimate_predictions(&subject_for_runtime_model(model), &SUPPORT_POINT)?
        .into_subject()
        .ok_or_else(|| io::Error::other(format!("{label}: expected subject predictions")))?;

    report_subject_predictions(label, &predictions, tolerance)
}

#[cfg(feature = "dsl-jit")]
pub fn compile_runtime_jit_model() -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    Ok(pharmsol::dsl::compile_module_source_to_runtime(
        AUTHORING_DSL,
        Some(MODEL_NAME),
        pharmsol::dsl::RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?)
}

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
pub fn compile_runtime_native_aot_model(
    workspace: &ArtifactWorkspace,
) -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    Ok(pharmsol::dsl::compile_module_source_to_runtime(
        AUTHORING_DSL,
        Some(MODEL_NAME),
        pharmsol::dsl::RuntimeCompilationTarget::NativeAot(
            pharmsol::dsl::NativeAotCompileOptions::new(
                workspace.build_root("runtime-native-aot-build"),
            )
            .with_output(workspace.aot_output("bimodal-ke-runtime-native-aot")),
        ),
        |_, _| {},
    )?)
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_runtime_wasm_model() -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    Ok(pharmsol::dsl::compile_module_source_to_runtime(
        AUTHORING_DSL,
        Some(MODEL_NAME),
        pharmsol::dsl::RuntimeCompilationTarget::Wasm,
        |_, _| {},
    )?)
}

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
pub fn compile_direct_aot_model(
    workspace: &ArtifactWorkspace,
) -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    let artifact = pharmsol::dsl::compile_module_source_to_aot(
        AUTHORING_DSL,
        Some(MODEL_NAME),
        pharmsol::dsl::NativeAotCompileOptions::new(workspace.build_root("direct-aot-build"))
            .with_output(workspace.aot_output("bimodal-ke-direct-aot")),
        |_, _| {},
    )?;

    Ok(pharmsol::dsl::load_runtime_artifact(
        &artifact,
        pharmsol::dsl::RuntimeArtifactFormat::NativeAot,
    )?)
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_bytes_wasm_model() -> Result<pharmsol::dsl::CompiledRuntimeModel, Box<dyn Error>> {
    let bytes =
        pharmsol::dsl::compile_module_source_to_wasm_bytes(AUTHORING_DSL, Some(MODEL_NAME))?;
    Ok(pharmsol::dsl::load_runtime_wasm_bytes(&bytes)?)
}
