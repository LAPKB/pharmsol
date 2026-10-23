//! Regression coverage for infusion-rate injection in the authoring shorthand.
//!
//! Shorthand source is lowered to canonical form by inserting `rate(<route>)`
//! into the destination derivative. When the author already reads the rate the
//! insertion must be skipped, otherwise the infusion is counted twice.

#![cfg(feature = "dsl")]

use pharmsol::dsl::compile_module_source_to_runtime;
use pharmsol::{prelude::*, Parameters};

/// 100 mg infused over 1 h into a one-compartment model with `ke = 0.1`
/// reaches `R/ke * (1 - exp(-ke t))` at t = 1.
fn expected() -> f64 {
    let (rate, ke) = (100.0_f64, 0.1_f64);
    rate / ke * (1.0 - (-ke).exp())
}

fn predict(source: &str) -> f64 {
    let model = compile_module_source_to_runtime(source, None, |_kind, _message| {})
        .expect("model compiles");
    let parameters = Parameters::with_model(&model, [("ke", 0.1)]).expect("parameters bind");
    let subject = Subject::builder("infusion")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(1.0, "cp")
        .build();

    model
        .estimate_predictions(&subject, &parameters)
        .expect("predictions")
        .into_subject()
        .expect("subject predictions")
        .predictions()[0]
        .prediction()
}

#[test]
fn shorthand_implicit_rate_matches_closed_form() {
    let value = predict(
        r#"
name = m
kind = ode
params = ke
states = central
outputs = cp
infusion(iv) -> central
dx(central) = -ke * central
out(cp) = central
"#,
    );
    assert!((value - expected()).abs() < 1e-6, "got {value}");
}

#[test]
fn shorthand_explicit_rate_is_not_double_counted() {
    let value = predict(
        r#"
name = m
kind = ode
params = ke
states = central
outputs = cp
infusion(iv) -> central
dx(central) = rate(iv) - ke * central
out(cp) = central
"#,
    );
    assert!((value - expected()).abs() < 1e-6, "got {value}");
}

#[test]
fn shorthand_scaled_rate_uses_only_the_authored_term() {
    let value = predict(
        r#"
name = m
kind = ode
params = ke
states = central
outputs = cp
infusion(iv) -> central
dx(central) = 0.5 * rate(iv) - ke * central
out(cp) = central
"#,
    );
    assert!((value - expected() * 0.5).abs() < 1e-6, "got {value}");
}

#[test]
fn shorthand_rate_inside_nested_expression_disables_injection() {
    let value = predict(
        r#"
name = m
kind = ode
params = ke
states = central
outputs = cp
infusion(iv) -> central
dx(central) = max(rate(iv), 0.0) - ke * central
out(cp) = central
"#,
    );
    assert!((value - expected()).abs() < 1e-6, "got {value}");
}

#[test]
fn canonical_explicit_rate_matches_shorthand() {
    let value = predict(
        r#"
model m {
  kind ode
  parameters { ke }
  states { central }
  routes { iv -> central }
  dynamics { ddt(central) = rate(iv) - ke * central }
  outputs { cp = central }
}
"#,
    );
    assert!((value - expected()).abs() < 1e-6, "got {value}");
}
