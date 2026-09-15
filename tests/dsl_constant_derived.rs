//! A derived assignment whose value folds to a compile-time constant must still
//! be materialised, because other statements read it by slot.

#![cfg(feature = "dsl")]

use pharmsol::dsl::compile_module_source_to_runtime;
use pharmsol::{prelude::*, Parameters};

fn prediction(
    model_source: &str,
    params: &[(&str, f64)],
) -> Result<f64, Box<dyn std::error::Error>> {
    let model = compile_module_source_to_runtime(model_source, Some("const_derived"), |_, _| {})?;
    let support_point = Parameters::with_model(&model, params.iter().copied())?;
    let subject = Subject::builder("const_derived")
        .missing_observation(0.0, "cp")
        .build();
    let predictions = model
        .estimate_predictions(&subject, &support_point)?
        .into_subject()
        .ok_or("expected subject predictions")?;
    Ok(predictions.predictions()[0].prediction())
}

const CONSTANT_DERIVED: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central, sink
outputs = cp

z1 = 1.0

init(central) = 2.0
init(sink) = 0.0

dx(central) = 0.0
dx(sink) = 0.0

out(cp) = z1
"#;

const FRACTIONAL_DERIVED: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central, sink
outputs = cp

z1 = 1.5

init(central) = 2.0
init(sink) = 0.0

dx(central) = 0.0
dx(sink) = 0.0

out(cp) = z1
"#;

const PARAMETER_DERIVED: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central, sink
outputs = cp

z1 = p1

init(central) = 2.0
init(sink) = 0.0

dx(central) = 0.0
dx(sink) = 0.0

out(cp) = z1
"#;

const MIN_MAX_CONSTANT_BRANCHES: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central, sink
outputs = cp

z1 = if (p1 > 0.0) 1.0 else 2.0

init(central) = 2.0
init(sink) = 0.0

dx(central) = 0.0
dx(sink) = 0.0

out(cp) = z1
"#;

#[test]
fn derived_value_from_a_parameter_is_read_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let value = prediction(PARAMETER_DERIVED, &[("p1", 4.25), ("p2", 1.0)])?;
    assert!((value - 4.25).abs() < 1e-9, "got {value}");
    Ok(())
}

#[test]
fn derived_value_from_a_constant_is_read_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let value = prediction(CONSTANT_DERIVED, &[("p1", 4.25), ("p2", 1.0)])?;
    assert!((value - 1.0).abs() < 1e-9, "got {value}");
    Ok(())
}

#[test]
fn derived_value_from_a_fractional_constant_is_read_correctly(
) -> Result<(), Box<dyn std::error::Error>> {
    let value = prediction(FRACTIONAL_DERIVED, &[("p1", 4.25), ("p2", 1.0)])?;
    assert!((value - 1.5).abs() < 1e-9, "got {value}");
    Ok(())
}

#[test]
fn constant_branch_conditional_is_read_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let value = prediction(MIN_MAX_CONSTANT_BRANCHES, &[("p1", 1.0), ("p2", 1.0)])?;
    assert!((value - 1.0).abs() < 1e-9, "got {value}");
    Ok(())
}
