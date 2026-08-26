//! Regression coverage for numeric equality in the DSL runtime.

#![cfg(feature = "dsl")]

use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeCompilationTarget,
};
use pharmsol::{prelude::*, Parameters};

const MODEL_SOURCE: &str = r#"
name = numeric_equality
kind = ode

params = a, b, c
covariates = drug
states = central
outputs = e, non_integral, not_one

dx(central) = 0

out(e) = if (drug == 1) a else if (drug == 2) b else c
out(non_integral) = if (drug == 1.1) a else if (drug == 2.1) b else c
out(not_one) = if (drug != 1) a else b
"#;

fn output_value(model: &CompiledRuntimeModel, drug: f64, output: &str) -> f64 {
    let parameters = Parameters::with_model(model, [("a", 10.0), ("b", 20.0), ("c", 30.0)])
        .expect("valid named parameters");
    let subject = Subject::builder("numeric-equality")
        .covariate("drug", 0.0, drug)
        .missing_observation(0.0, output)
        .build();

    model
        .estimate_predictions(&subject, &parameters)
        .expect("predictions succeed")
        .into_subject()
        .expect("ODE predictions are subject predictions")
        .predictions()
        .first()
        .expect("one output prediction")
        .prediction()
}

fn assert_numeric_equality_results(model: &CompiledRuntimeModel) {
    for (drug, expected) in [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)] {
        assert_eq!(output_value(model, drug, "e"), expected, "drug={drug}");
    }

    for (drug, expected) in [
        (1.0, 30.0),
        (2.0, 30.0),
        (3.0, 30.0),
        (1.1, 10.0),
        (2.1, 20.0),
        (3.1, 30.0),
    ] {
        assert_eq!(
            output_value(model, drug, "non_integral"),
            expected,
            "drug={drug}"
        );
    }

    for (drug, expected) in [
        (1.0, 20.0),
        (2.0, 10.0),
        (3.0, 10.0),
        (1.1, 10.0),
        (2.1, 10.0),
        (3.1, 10.0),
    ] {
        assert_eq!(
            output_value(model, drug, "not_one"),
            expected,
            "drug={drug}"
        );
    }
}

#[cfg(feature = "dsl")]
#[test]
fn numeric_equality_selects_real_covariate_branches_under_jit() {
    let model = compile_module_source_to_runtime(
        MODEL_SOURCE,
        Some("numeric_equality"),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )
    .expect("compile numeric equality model with JIT");

    assert_numeric_equality_results(&model);
}
