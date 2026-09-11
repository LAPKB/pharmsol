#![cfg(feature = "dsl")]

use approx::assert_relative_eq;
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};
use pharmsol::{estimate_effect_2, Parameters, Subject, SubjectBuilderExt};

const ESTIMATE_EFFECT_2_MODEL: &str = r#"
name = estimate_effect_2_backend_parity
kind = ode

params = u, v, alpha, h1, h2
states = central
outputs = cp

dx(central) = 0
out(cp) = estimate_effect_2(u, v, alpha, h1, h2)
"#;

fn subject() -> Subject {
    Subject::builder("estimate_effect_2_backend_parity")
        .missing_observation(0.0, "cp")
        .build()
}

fn prediction(model: &CompiledRuntimeModel, parameters: &Parameters) -> f64 {
    model
        .estimate_predictions(&subject(), parameters)
        .expect("estimate_effect_2 runtime prediction")
        .into_subject()
        .expect("ODE runtime returns subject predictions")
        .predictions()[0]
        .prediction()
}

fn compile_jit() -> Result<CompiledRuntimeModel, Box<dyn std::error::Error>> {
    Ok(compile_module_source_to_runtime(
        ESTIMATE_EFFECT_2_MODEL,
        Some("estimate_effect_2_backend_parity"),
        |_, _| {},
    )?)
}

fn assert_vector(model: &CompiledRuntimeModel, values: &[(&str, f64)], expected: f64, label: &str) {
    let parameters = Parameters::with_model(model, values.iter().copied())
        .expect("valid estimate_effect_2 parameters");
    assert_relative_eq!(
        prediction(model, &parameters),
        expected,
        max_relative = 1e-10
    );
    assert_eq!(
        model.info().name,
        "estimate_effect_2_backend_parity",
        "{label}"
    );
}

#[test]
fn direct_and_jit_estimate_effect_2_values_are_identical() -> Result<(), Box<dyn std::error::Error>>
{
    let jit = compile_jit()?;

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
        let direct = estimate_effect_2(
            values[0].1,
            values[1].1,
            values[2].1,
            values[3].1,
            values[4].1,
        );
        assert_relative_eq!(direct, expected, max_relative = 1e-10);
        assert_vector(&jit, &values, direct, "JIT");
    }

    Ok(())
}
