#![cfg(all(feature = "dsl-jit", feature = "dsl-aot", feature = "dsl-aot-load"))]

use approx::assert_relative_eq;
use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, NativeAotCompileOptions,
    RuntimeCompilationTarget,
};
use pharmsol::{get_e2, Parameters, Subject, SubjectBuilderExt};
use tempfile::TempDir;

const GET_E2_MODEL: &str = r#"
name = get_e2_backend_parity
kind = ode

params = u, v, alpha, h1, h2
states = central
outputs = cp

dx(central) = 0
out(cp) = get_e2(u, v, alpha, h1, h2)
"#;

fn subject() -> Subject {
    Subject::builder("get_e2_backend_parity")
        .missing_observation(0.0, "cp")
        .build()
}

fn prediction(model: &CompiledRuntimeModel, parameters: &Parameters) -> f64 {
    model
        .estimate_predictions(&subject(), parameters)
        .expect("get_e2 runtime prediction")
        .into_subject()
        .expect("ODE runtime returns subject predictions")
        .predictions()[0]
        .prediction()
}

fn compile_jit() -> Result<CompiledRuntimeModel, Box<dyn std::error::Error>> {
    Ok(compile_module_source_to_runtime(
        GET_E2_MODEL,
        Some("get_e2_backend_parity"),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?)
}

fn compile_aot(workspace: &TempDir) -> Result<CompiledRuntimeModel, Box<dyn std::error::Error>> {
    let output = workspace.path().join("get_e2_backend_parity.pkm");
    Ok(compile_module_source_to_runtime(
        GET_E2_MODEL,
        Some("get_e2_backend_parity"),
        RuntimeCompilationTarget::NativeAot(
            NativeAotCompileOptions::new(workspace.path().join("build")).with_output(output),
        ),
        |_, _| {},
    )?)
}

fn assert_vector(model: &CompiledRuntimeModel, values: &[(&str, f64)], expected: f64, label: &str) {
    let parameters =
        Parameters::with_model(model, values.iter().copied()).expect("valid get_e2 parameters");
    assert_relative_eq!(
        prediction(model, &parameters),
        expected,
        max_relative = 1e-10
    );
    assert_eq!(model.info().name, "get_e2_backend_parity", "{label}");
}

#[test]
fn direct_jit_and_native_aot_get_e2_values_are_identical() -> Result<(), Box<dyn std::error::Error>>
{
    let jit = compile_jit()?;
    let workspace = tempfile::tempdir()?;
    let aot = compile_aot(&workspace)?;

    let vectors = [
        (
            [
                ("u", 1.0),
                ("v", 1.0),
                ("alpha", -0.5),
                ("h1", 1.0),
                ("h2", 1.0),
            ],
            0.6,
        ),
        (
            [
                ("u", 0.25),
                ("v", 0.0),
                ("alpha", -3.0),
                ("h1", 2.0),
                ("h2", 1.0),
            ],
            1.0 / 3.0,
        ),
    ];

    for (values, expected) in vectors {
        let direct = get_e2(
            values[0].1,
            values[1].1,
            values[2].1,
            values[3].1,
            values[4].1,
        );
        assert_relative_eq!(direct, expected, max_relative = 1e-10);
        assert_vector(&jit, &values, direct, "JIT");
        assert_vector(&aot, &values, direct, "native AoT");
        let jit_parameters = Parameters::with_model(&jit, values).expect("valid JIT parameters");
        let aot_parameters = Parameters::with_model(&aot, values).expect("valid AoT parameters");
        assert_relative_eq!(
            prediction(&jit, &jit_parameters),
            prediction(&aot, &aot_parameters),
            max_relative = 1e-10
        );
    }

    Ok(())
}
