//! Solver failures on compiled DSL models should explain themselves: the
//! message names the offending states and, when it applies, a rescaling hint.

#![cfg(feature = "dsl")]

use pharmsol::dsl::compile_module_source_to_runtime;
use pharmsol::{prelude::*, Parameters};

fn simulate(source: &str, name: &str) -> Result<(), String> {
    let model = compile_module_source_to_runtime(source, Some(name), |_, _| {}).unwrap();
    let support_point = Parameters::with_model(&model, [("k", 1.0)]).unwrap();
    let subject = Subject::builder("s1")
        .missing_observation(0.0, "y")
        .missing_observation(5.0, "y")
        .build();
    model
        .estimate_predictions(&subject, &support_point)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// `sqrt` of a negative number once `amount` undershoots 0.5 yields NaN.
#[test]
fn non_finite_rhs_is_reported_with_state_names() {
    let source = r#"
name = blowup
kind = ode

params = k
states = amount
outputs = y

dx(amount) = -k * sqrt(amount - 0.5)
init(amount) = 1
out(y) = amount
"#;
    let message = simulate(source, "blowup").expect_err("model must fail");
    assert!(message.contains("Diagnostics:"), "{message}");
    assert!(message.contains("damount/dt"), "{message}");
    assert!(message.contains("amount ="), "{message}");
}
