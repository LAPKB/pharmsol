//! A derived assignment whose value folds to a compile-time constant must still
//! be materialised, because other statements read it by slot, and it must be
//! read back as a real whether the reader is an output or a derivative.

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

// The subject must carry an observation at the model start (`t = 0`) for the
// integration to be anchored there, matching the single-point helper above.
fn predictions_at(
    model_source: &str,
    params: &[(&str, f64)],
    times: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let model = compile_module_source_to_runtime(model_source, Some("const_derived"), |_, _| {})?;
    let support_point = Parameters::with_model(&model, params.iter().copied())?;
    let mut builder = Subject::builder("const_derived");
    for time in times {
        builder = builder.missing_observation(*time, "cp");
    }
    let subject = builder.build();
    let predictions = model
        .estimate_predictions(&subject, &support_point)?
        .into_subject()
        .ok_or("expected subject predictions")?;
    Ok(predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .collect())
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

// A whole-number derived value read by a derivative, not by an output: the
// rate constant must stay 1.0 rather than the bit pattern of 1.0.
const WHOLE_NUMBER_DERIVED_RATE: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central
outputs = cp

z1 = 1.0

init(central) = 2.0

dx(central) = -z1 * central

out(cp) = central
"#;

// The reported shape: whole-number derived branches feeding a conditional that
// scales a derivative.
const WHOLE_NUMBER_DERIVED_IN_CONDITIONAL: &str = r#"
name = const_derived
kind = ode

params = p1, p2
states = central, effector
outputs = cp

max_r = 1.0
min_r = 0.001
r = if (central > 0.0) max_r else min_r

init(central) = 2.0
init(effector) = 0.0

dx(central) = 0.0
dx(effector) = r * central - effector

out(cp) = effector
"#;

#[test]
fn whole_number_derived_scales_a_derivative_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let values = predictions_at(
        WHOLE_NUMBER_DERIVED_RATE,
        &[("p1", 4.25), ("p2", 1.0)],
        &[0.0, 1.0, 2.0],
    )?;
    for (value, time) in values.iter().zip([0.0f64, 1.0, 2.0]) {
        let expected = 2.0 * (-time).exp();
        assert!(
            (value - expected).abs() < 1e-4,
            "got {value}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn whole_number_derived_in_a_conditional_scales_a_derivative_correctly(
) -> Result<(), Box<dyn std::error::Error>> {
    let values = predictions_at(
        WHOLE_NUMBER_DERIVED_IN_CONDITIONAL,
        &[("p1", 4.25), ("p2", 1.0)],
        &[0.0, 1.0, 2.0],
    )?;
    // `central` is held at 2.0 and the conditional selects `max_r` (1.0), so
    // `effector(t) = 2 * (1 - exp(-t))`. A bit-pattern read would put `max_r`
    // at 4.607e18 and drive `effector` there instead.
    for (value, time) in values.iter().zip([0.0f64, 1.0, 2.0]) {
        let expected = 2.0 * (1.0 - (-time).exp());
        assert!(
            (value - expected).abs() < 1e-4,
            "got {value}, expected {expected}"
        );
    }
    Ok(())
}
