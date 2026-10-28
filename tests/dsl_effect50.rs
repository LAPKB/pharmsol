#![cfg(feature = "dsl")]

use approx::assert_relative_eq;
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};
use pharmsol::{effect50, Parameters, Subject, SubjectBuilderExt};

const EFFECT50_MODEL: &str = r#"
name = effect50_runtime
kind = ode

params = emin, emax, e50, h, x
states = central
outputs = effect

dx(central) = 0
out(effect) = effect50(emin, emax, e50, h, x)
"#;

fn prediction(model: &CompiledRuntimeModel, values: [(&str, f64); 5]) -> f64 {
    let subject = Subject::builder("effect50_runtime")
        .missing_observation(0.0, "effect")
        .build();
    let parameters = Parameters::with_model(model, values).expect("valid effect50 parameters");
    model
        .predict(&subject, &parameters)
        .expect("effect50 runtime prediction")
        .into_subject()
        .expect("ODE runtime returns subject predictions")
        .predictions()[0]
        .prediction()
}

#[test]
fn jit_effect50_matches_the_rust_implementation() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        compile_module_source_to_runtime(EFFECT50_MODEL, Some("effect50_runtime"), |_, _| {})?;

    let defaults = [
        ("emin", 0.0),
        ("emax", 1.0),
        ("e50", 0.5),
        ("h", 1.0),
        ("x", 0.5),
    ];
    assert_relative_eq!(prediction(&model, defaults), 0.5, max_relative = 1e-12);

    for values in [
        [
            ("emin", 2.0),
            ("emax", 10.0),
            ("e50", 3.0),
            ("h", 2.5),
            ("x", 3.0),
        ],
        [
            ("emin", 0.2),
            ("emax", 3.0),
            ("e50", 1.7),
            ("h", 2.3),
            ("x", 0.9),
        ],
        [
            ("emin", 1.0),
            ("emax", 0.0),
            ("e50", 0.5),
            ("h", 1.0),
            ("x", 0.0),
        ],
    ] {
        let expected = effect50(
            values[0].1,
            values[1].1,
            values[2].1,
            values[3].1,
            values[4].1,
        );
        assert_relative_eq!(prediction(&model, values), expected, max_relative = 1e-12);
    }

    Ok(())
}
