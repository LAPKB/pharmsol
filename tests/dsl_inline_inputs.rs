#![cfg(feature = "dsl")]

//! Inline dose inputs: SDE drift written like ODE derivatives, and the
//! Pmetrics `b[n]` / `rateiv[n]` spellings, must simulate exactly like the
//! equivalent declared-route models.

use pharmsol::dsl::{self, RuntimePredictions};
use pharmsol::prelude::*;

const SDE_DECLARED: &str = r#"
name = sde_inputs
kind = sde
params = ka, ke, sigma_ke, v, tlag, f_oral
states = gut, central
outputs = cp
particles = 8

bolus(oral) -> gut
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral

dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
noise(central) = sigma_ke

out(cp) = central / v
"#;

const SDE_INLINE: &str = r#"
name = sde_inputs
kind = sde
params = ka, ke, sigma_ke, v, tlag, f_oral
states = gut, central
outputs = cp
particles = 8

lag(oral) = tlag

dx(gut) = f_oral * bolus(oral) - ka * gut
dx(central) = ka * gut - ke * central + infusion(iv)
noise(central) = sigma_ke

out(cp) = central / v
"#;

const ODE_NAMED: &str = r#"
name = ode_inputs
kind = ode
params = ka, ke, v
states = gut, central
outputs = cp

dx(gut) = bolus[1] - ka * gut
dx(central) = ka * gut - ke * central + infusion[1]

out(cp) = central / v
"#;

const ODE_PMETRICS: &str = r#"
name = ode_inputs
kind = ode
params = ka, ke, v
states = gut, central
outputs = cp

dx(gut) = b[1] - ka * gut
dx(central) = ka * gut - ke * central + rateiv[1]

out(cp) = central / v
"#;

fn compile(src: &str, name: &str) -> dsl::CompiledRuntimeModel {
    dsl::compile_module_source_to_runtime(src, Some(name), |_, _| {})
        .unwrap_or_else(|error| panic!("{name} should compile: {error}"))
}

fn subject(oral: &str, iv: &str) -> Subject {
    Subject::builder("inline-inputs")
        .bolus(0.0, 100.0, oral)
        .infusion(6.0, 60.0, iv, 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(6.5, "cp")
        .missing_observation(7.0, "cp")
        .missing_observation(8.0, "cp")
        .build()
}

fn predictions(
    model: &dsl::CompiledRuntimeModel,
    subject: &Subject,
    parameters: &[(&str, f64)],
) -> Vec<f64> {
    let parameters =
        Parameters::with_model(model, parameters.iter().copied()).expect("valid named parameters");
    match model
        .estimate_predictions(subject, &parameters)
        .expect("model should simulate")
    {
        RuntimePredictions::Subject(predictions) => predictions
            .predictions()
            .iter()
            .map(|prediction| prediction.prediction())
            .collect(),
        RuntimePredictions::Particles(particles) => particles
            .columns()
            .into_iter()
            .map(|column| {
                column
                    .iter()
                    .map(|prediction| prediction.prediction())
                    .sum::<f64>()
                    / column.len() as f64
            })
            .collect(),
    }
}

fn assert_same(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter().any(|value| *value > 0.1),
        "doses should reach the output: {left:?}"
    );
    for (left, right) in left.iter().zip(right) {
        assert!((left - right).abs() <= 1e-10, "{left} != {right}");
    }
}

#[test]
fn sde_inline_inputs_match_declared_routes() {
    let parameters = [
        ("ka", 1.1),
        ("ke", 0.2),
        ("sigma_ke", 0.0),
        ("v", 10.0),
        ("tlag", 0.25),
        ("f_oral", 0.8),
    ];
    let subject = subject("oral", "iv");
    let declared = predictions(&compile(SDE_DECLARED, "sde_inputs"), &subject, &parameters);
    let inline = predictions(&compile(SDE_INLINE, "sde_inputs"), &subject, &parameters);
    assert_same(&declared, &inline);
}

#[test]
fn pmetrics_input_aliases_match_bolus_and_infusion() {
    let parameters = [("ka", 1.1), ("ke", 0.2), ("v", 10.0)];
    let subject = subject("1", "1");
    let named = predictions(&compile(ODE_NAMED, "ode_inputs"), &subject, &parameters);
    let pmetrics = predictions(&compile(ODE_PMETRICS, "ode_inputs"), &subject, &parameters);
    assert_same(&named, &pmetrics);
}
