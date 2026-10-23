//! Validates the symbolically derived state Jacobian against finite
//! differences of the compiled dynamics.
//!
//! This is the primary correctness gate for the differentiation pass: every
//! rule is exercised through real models, and any rule error shows up as a
//! mismatch against the numeric reference.

#![cfg(feature = "dsl")]

use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeOdeModel};
use pharmsol::Covariates;

fn ode_model(source: &str) -> RuntimeOdeModel {
    match compile_module_source_to_runtime(source, None, |_kind, _message| {})
        .expect("model compiles")
    {
        CompiledRuntimeModel::Ode(model) => model,
        other => panic!("expected an ODE model, got {other:?}"),
    }
}

/// Central difference of the compiled dynamics, row-major like the Jacobian.
fn numeric_jacobian(
    model: &RuntimeOdeModel,
    time: f64,
    state: &[f64],
    parameters: &[f64],
    covariates: &Covariates,
    route_inputs: &[f64],
) -> Vec<f64> {
    let states = state.len();
    let mut jacobian = vec![0.0; states * states];
    for column in 0..states {
        let step = 1e-6 * state[column].abs().max(1.0);
        let mut forward = state.to_vec();
        let mut backward = state.to_vec();
        forward[column] += step;
        backward[column] -= step;
        let high = model
            .state_derivative(time, &forward, parameters, covariates, route_inputs)
            .expect("forward derivative");
        let low = model
            .state_derivative(time, &backward, parameters, covariates, route_inputs)
            .expect("backward derivative");
        for row in 0..states {
            jacobian[row * states + column] = (high[row] - low[row]) / (2.0 * step);
        }
    }
    jacobian
}

fn assert_jacobian_matches(
    source: &str,
    parameters: &[f64],
    states: &[&[f64]],
    route_inputs: &[f64],
) {
    let model = ode_model(source);
    let covariates = Covariates::default();
    for state in states {
        let symbolic = model
            .state_jacobian(1.0, state, parameters, &covariates, route_inputs)
            .expect("jacobian evaluates")
            .expect("model has a symbolic jacobian");
        let numeric = numeric_jacobian(&model, 1.0, state, parameters, &covariates, route_inputs);
        assert_eq!(symbolic.len(), numeric.len());
        for (index, (lhs, rhs)) in symbolic.iter().zip(numeric.iter()).enumerate() {
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                (lhs - rhs).abs() / scale < 1e-5,
                "entry {index}: symbolic {lhs} vs numeric {rhs} at state {state:?}"
            );
        }
    }
}

#[test]
fn linear_two_compartment() {
    assert_jacobian_matches(
        r#"
name = two_cpt
kind = ode
params = ka, ke, kcp, kpc
states = gut, central, periph
outputs = cp
bolus(oral) -> gut
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central - kcp * central + kpc * periph
dx(periph) = kcp * central - kpc * periph
out(cp) = central
"#,
        &[1.1, 0.2, 0.4, 0.3],
        &[&[10.0, 5.0, 2.0], &[0.0, 0.0, 0.0], &[100.0, 0.5, 40.0]],
        &[0.0],
    );
}

#[test]
fn michaelis_menten_is_nonlinear_but_differentiable() {
    assert_jacobian_matches(
        r#"
name = mm
kind = ode
params = vmax, km, ka
states = gut, central
outputs = cp
bolus(oral) -> gut
dx(gut) = -ka * gut
dx(central) = ka * gut - vmax * central / (km + central)
out(cp) = central
"#,
        &[10.0, 5.0, 1.2],
        &[&[20.0, 3.0], &[1.0, 0.25], &[0.0, 50.0]],
        &[0.0],
    );
}

#[test]
fn products_quotients_and_powers() {
    assert_jacobian_matches(
        r#"
name = mixed
kind = ode
params = a, b, c
states = x, y
outputs = cp
dx(x) = -a * x * y + b * y / (c + x)
dx(y) = a * pow(x, 3.0) - b * sqrt(y) * x
out(cp) = x
"#,
        &[0.7, 1.3, 2.0],
        &[&[2.0, 3.0], &[0.5, 0.25], &[4.0, 9.0]],
        &[],
    );
}

#[test]
fn transcendental_intrinsics() {
    assert_jacobian_matches(
        r#"
name = transcendental
kind = ode
params = a, b
states = x, y
outputs = cp
dx(x) = -a * exp(-x) + b * ln(y)
dx(y) = a * sin(x) + b * cos(y) - tan(x * 0.1)
out(cp) = x
"#,
        &[0.9, 1.4],
        &[&[1.0, 2.0], &[0.25, 0.5], &[2.0, 4.0]],
        &[],
    );
}

#[test]
fn derived_values_and_covariates_are_constant_in_state() {
    assert_jacobian_matches(
        r#"
name = derived
kind = ode
params = cl, v, ka
derived = ke
states = gut, central
outputs = cp
bolus(oral) -> gut
ke = cl / v
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
out(cp) = central / v
"#,
        &[3.0, 30.0, 1.1],
        &[&[10.0, 4.0], &[0.0, 0.0]],
        &[0.0],
    );
}

#[test]
fn state_carrying_locals_are_differentiated() {
    assert_jacobian_matches(
        r#"
name = locals
kind = ode
params = ke, kt
states = a, b
outputs = cp
dx(a) = -ke * (a + b) * a
dx(b) = kt * (a + b) - ke * b
out(cp) = a
"#,
        &[0.3, 0.8],
        &[&[2.0, 5.0], &[0.1, 0.2]],
        &[],
    );
}

#[test]
fn infusion_rate_terms_do_not_appear_in_the_jacobian() {
    let model = ode_model(
        r#"
name = infusion
kind = ode
params = ke
states = central
outputs = cp
infusion(iv) -> central
dx(central) = -ke * central
out(cp) = central
"#,
    );
    let covariates = Covariates::default();
    let with_rate = model
        .state_jacobian(1.0, &[5.0], &[0.25], &covariates, &[100.0])
        .unwrap()
        .unwrap();
    let without_rate = model
        .state_jacobian(1.0, &[5.0], &[0.25], &covariates, &[0.0])
        .unwrap()
        .unwrap();
    assert_eq!(with_rate, vec![-0.25]);
    assert_eq!(without_rate, vec![-0.25]);
}

#[test]
fn state_dependent_branch_declines() {
    let model = ode_model(
        r#"
name = branchy
kind = ode
params = ke, threshold
states = central
outputs = cp
dx(central) = if (central > threshold) -ke * central else 0.0
out(cp) = central
"#,
    );
    let covariates = Covariates::default();
    assert!(model
        .state_jacobian(1.0, &[5.0], &[0.25, 1.0], &covariates, &[])
        .unwrap()
        .is_none());
}
