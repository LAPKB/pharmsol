#![cfg(all(feature = "dsl-jit", feature = "dsl-aot", feature = "dsl-aot-load"))]

use approx::assert_relative_eq;
use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, NativeAotCompileOptions,
    RuntimeCompilationTarget,
};
use pharmsol::{get_e2, get_e3, Parameters, Subject, SubjectBuilderExt};
use tempfile::TempDir;

const GET_E3_MODEL: &str = r#"
name = get_e3_backend_parity
kind = ode

params = a, b, c, alpha12, alpha13, alpha23, alpha123, h1, h2, h3
states = central
outputs = cp

dx(central) = 0
out(cp) = get_e3(a, b, c, alpha12, alpha13, alpha23, alpha123, h1, h2, h3)
"#;

fn subject() -> Subject {
    Subject::builder("get_e3_backend_parity")
        .missing_observation(0.0, "cp")
        .build()
}

fn prediction(
    model: &CompiledRuntimeModel,
    parameters: &Parameters,
) -> Result<f64, Box<dyn std::error::Error>> {
    let predictions = model.estimate_predictions(&subject(), parameters)?;
    let predictions = predictions
        .into_subject()
        .ok_or_else(|| std::io::Error::other("ODE runtime returned particle predictions"))?;
    Ok(predictions.predictions()[0].prediction())
}

fn compile_jit() -> Result<CompiledRuntimeModel, Box<dyn std::error::Error>> {
    Ok(compile_module_source_to_runtime(
        GET_E3_MODEL,
        Some("get_e3_backend_parity"),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?)
}

fn compile_aot(workspace: &TempDir) -> Result<CompiledRuntimeModel, Box<dyn std::error::Error>> {
    let output = workspace.path().join("get_e3_backend_parity.pkm");
    Ok(compile_module_source_to_runtime(
        GET_E3_MODEL,
        Some("get_e3_backend_parity"),
        RuntimeCompilationTarget::NativeAot(
            NativeAotCompileOptions::new(workspace.path().join("build")).with_output(output),
        ),
        |_, _| {},
    )?)
}

fn direct(values: &[(&str, f64); 10]) -> f64 {
    get_e3(
        values[0].1,
        values[1].1,
        values[2].1,
        values[3].1,
        values[4].1,
        values[5].1,
        values[6].1,
        values[7].1,
        values[8].1,
        values[9].1,
    )
}

fn assert_vector(
    model: &CompiledRuntimeModel,
    values: &[(&str, f64)],
    expected: f64,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let parameters = Parameters::with_model(model, values.iter().copied())?;
    assert_relative_eq!(
        prediction(model, &parameters)?,
        expected,
        max_relative = 1e-10
    );
    assert_eq!(model.info().name, "get_e3_backend_parity", "{label}");
    Ok(())
}

#[test]
fn direct_jit_and_native_aot_get_e3_values_are_identical() -> Result<(), Box<dyn std::error::Error>>
{
    let jit = compile_jit()?;
    let workspace = tempfile::tempdir()?;
    let aot = compile_aot(&workspace)?;

    let vectors = [
        [
            ("a", 1.0),
            ("b", 2.0),
            ("c", 0.5),
            ("alpha12", 0.2),
            ("alpha13", -0.1),
            ("alpha23", 0.3),
            ("alpha123", 0.4),
            ("h1", 1.0),
            ("h2", 1.0),
            ("h3", 1.0),
        ],
        [
            ("a", 1.0e-6),
            ("b", 1.25),
            ("c", 0.75),
            ("alpha12", 8.0),
            ("alpha13", -4.0),
            ("alpha23", -0.2),
            ("alpha123", 3.0),
            ("h1", 2.0),
            ("h2", 1.2),
            ("h3", 0.8),
        ],
        [
            ("a", 1.0),
            ("b", 1.0),
            ("c", 1.0),
            ("alpha12", -3.0),
            ("alpha13", 0.0),
            ("alpha23", 0.0),
            ("alpha123", 0.0),
            ("h1", 1.0),
            ("h2", 3.0),
            ("h3", 1.0),
        ],
    ];

    for values in vectors {
        let direct = direct(&values);
        assert!(direct.is_finite());
        assert_vector(&jit, &values, direct, "JIT")?;
        assert_vector(&aot, &values, direct, "native AoT")?;
        let jit_parameters = Parameters::with_model(&jit, values)?;
        let aot_parameters = Parameters::with_model(&aot, values)?;
        assert_relative_eq!(
            prediction(&jit, &jit_parameters)?,
            prediction(&aot, &aot_parameters)?,
            max_relative = 1e-10
        );
    }

    let reduction = vectors[1];
    assert_relative_eq!(
        direct(&reduction),
        get_e2(
            reduction[1].1,
            reduction[2].1,
            reduction[5].1,
            reduction[8].1,
            reduction[9].1,
        ),
        max_relative = 1e-12
    );

    let expected_antagonistic_xm = 0.430_159_709;
    assert_relative_eq!(
        direct(&vectors[2]),
        expected_antagonistic_xm / (1.0 + expected_antagonistic_xm),
        max_relative = 1e-6
    );

    Ok(())
}
