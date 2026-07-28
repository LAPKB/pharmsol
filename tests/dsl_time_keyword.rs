//! Verifies that the DSL's `t` keyword (and its `time` alias) resolves to the
//! current simulation time and flows correctly through analysis, execution
//! compilation, and the JIT / WASM backends.

#![cfg(any(feature = "dsl-jit", feature = "dsl-wasm"))]

use pharmsol::dsl::{compile_module_source_to_runtime, RuntimeCompilationTarget};
use pharmsol::{prelude::*, Parameters};

const MODEL_SOURCE: &str = r#"
name = time_probe
kind = ode

params = ke
states = central
outputs = cp, time_echo

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central
out(time_echo) = t
"#;

const MODEL_SOURCE_TIME_ALIAS: &str = r#"
name = time_probe
kind = ode

params = ke
states = central
outputs = cp, time_echo

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central
out(time_echo) = time
"#;

fn assert_time_echo_matches_observation_times(
    model: &pharmsol::dsl::CompiledRuntimeModel,
) -> Result<(), Box<dyn std::error::Error>> {
    let support_point = Parameters::with_model(model, [("ke", 1.0)])?;

    let subject = Subject::builder("time_probe")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(0.5, "time_echo")
        .missing_observation(1.0, "time_echo")
        .missing_observation(2.5, "time_echo")
        .missing_observation(4.0, "time_echo")
        .build();

    let predictions = model
        .estimate_predictions(&subject, &support_point)?
        .into_subject()
        .ok_or("expected subject predictions")?;

    let mut checked = 0;
    for prediction in predictions.predictions() {
        assert!(
            (prediction.time() - prediction.prediction()).abs() < 1e-6,
            "expected `t` to equal the observation time, got t={} prediction={}",
            prediction.time(),
            prediction.prediction()
        );
        checked += 1;
    }
    assert_eq!(
        checked, 4,
        "expected all four `time_echo` observations to be checked"
    );

    Ok(())
}

#[test]
#[cfg(feature = "dsl-jit")]
fn t_keyword_reflects_the_current_simulation_time_jit() -> Result<(), Box<dyn std::error::Error>> {
    let model = compile_module_source_to_runtime(
        MODEL_SOURCE,
        Some("time_probe"),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )?;
    assert_time_echo_matches_observation_times(&model)
}

#[test]
#[cfg(feature = "dsl-wasm")]
fn t_keyword_reflects_the_current_simulation_time_wasm() -> Result<(), Box<dyn std::error::Error>> {
    let model = compile_module_source_to_runtime(
        MODEL_SOURCE,
        Some("time_probe"),
        RuntimeCompilationTarget::Wasm,
        |_, _| {},
    )?;
    assert_time_echo_matches_observation_times(&model)
}
