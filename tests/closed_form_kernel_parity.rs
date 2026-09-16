//! Checks that closed-form propagation reproduces the built-in analytical
//! kernels from plain `dx(...)` equations.
//!
//! These are the models the kernel library exists to serve. Reproducing them
//! from ordinary ODE source is what would let the library be retired.

#![cfg(feature = "dsl")]

use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeOdeModel};
use pharmsol::{prelude::*, Cache, Parameters};

fn ode_model(source: &str) -> RuntimeOdeModel {
    match compile_module_source_to_runtime(source, None, |_kind, _message| {})
        .expect("model compiles")
    {
        CompiledRuntimeModel::Ode(model) => model.disable_cache(),
        other => panic!("expected an ODE model, got {other:?}"),
    }
}

fn analytical_model(source: &str) -> pharmsol::dsl::RuntimeAnalyticalModel {
    match compile_module_source_to_runtime(source, None, |_kind, _message| {})
        .expect("model compiles")
    {
        CompiledRuntimeModel::Analytical(model) => model.disable_cache(),
        other => panic!("expected an analytical model, got {other:?}"),
    }
}

fn subject(route: &str) -> Subject {
    Subject::builder("kernel-parity")
        .bolus(0.0, 100.0, route)
        .bolus(12.0, 100.0, route)
        .infusion(4.0, 60.0, "iv", 3.0)
        .missing_observation(0.5, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(6.0, "cp")
        .missing_observation(13.0, "cp")
        .missing_observation(24.0, "cp")
        .build()
}

fn assert_matches_kernel(
    ode_source: &str,
    analytical_source: &str,
    values: &[(&str, f64)],
    route: &str,
) {
    let ode = ode_model(ode_source);
    let analytical = analytical_model(analytical_source);
    let subject = subject(route);
    assert!(
        ode.uses_closed_form_for(&subject),
        "the ODE form should qualify for closed-form propagation"
    );

    let ode_parameters = Parameters::with_model(&ode, values.to_vec()).expect("ode parameters");
    let analytical_parameters =
        Parameters::with_model(&analytical, values.to_vec()).expect("analytical parameters");

    let from_ode = ode
        .estimate_predictions(&subject, &ode_parameters)
        .expect("ode predictions")
        .flat_predictions()
        .to_vec();
    let from_kernel = analytical
        .estimate_predictions(&subject, &analytical_parameters)
        .expect("analytical predictions")
        .flat_predictions()
        .to_vec();

    assert_eq!(from_ode.len(), from_kernel.len());
    for (index, (lhs, rhs)) in from_ode.iter().zip(from_kernel.iter()).enumerate() {
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            (lhs - rhs).abs() / scale < 1e-9,
            "prediction {index}: closed form {lhs} vs kernel {rhs}"
        );
    }
}

#[test]
fn one_compartment() {
    assert_matches_kernel(
        r#"
name = one_cpt_ode
kind = ode
params = ke, v
states = central
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
dx(central) = -ke * central
out(cp) = central / v
"#,
        r#"
name = one_cpt_kernel
kind = analytical
params = ke, v
states = central
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
structure = one_compartment
out(cp) = central / v ~ continuous()
"#,
        &[("ke", 0.2), ("v", 15.0)],
        "iv",
    );
}

#[test]
fn one_compartment_with_absorption() {
    assert_matches_kernel(
        r#"
name = one_cpt_abs_ode
kind = ode
params = ka, ke, v
states = gut, central
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
out(cp) = central / v
"#,
        r#"
name = one_cpt_abs_kernel
kind = analytical
params = ka, ke, v
states = gut, central
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
structure = one_compartment_with_absorption
out(cp) = central / v ~ continuous()
"#,
        &[("ka", 1.3), ("ke", 0.2), ("v", 15.0)],
        "oral",
    );
}

#[test]
fn two_compartments() {
    assert_matches_kernel(
        r#"
name = two_cpt_ode
kind = ode
params = ke, kcp, kpc, v
states = central, periph
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
dx(central) = -(ke + kcp) * central + kpc * periph
dx(periph) = kcp * central - kpc * periph
out(cp) = central / v
"#,
        r#"
name = two_cpt_kernel
kind = analytical
params = ke, kcp, kpc, v
states = central, periph
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
structure = two_compartments
out(cp) = central / v ~ continuous()
"#,
        &[("ke", 0.2), ("kcp", 0.5), ("kpc", 0.3), ("v", 15.0)],
        "iv",
    );
}

#[test]
fn two_compartments_with_absorption() {
    assert_matches_kernel(
        r#"
name = two_cpt_abs_ode
kind = ode
params = ke, ka, kcp, kpc, v
states = gut, central, periph
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
dx(gut) = -ka * gut
dx(central) = ka * gut - (ke + kcp) * central + kpc * periph
dx(periph) = kcp * central - kpc * periph
out(cp) = central / v
"#,
        r#"
name = two_cpt_abs_kernel
kind = analytical
params = ke, ka, kcp, kpc, v
states = gut, central, periph
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
structure = two_compartments_with_absorption
out(cp) = central / v ~ continuous()
"#,
        &[
            ("ke", 0.2),
            ("ka", 1.3),
            ("kcp", 0.5),
            ("kpc", 0.3),
            ("v", 15.0),
        ],
        "oral",
    );
}

#[test]
fn three_compartments() {
    assert_matches_kernel(
        r#"
name = three_cpt_ode
kind = ode
params = k10, k12, k13, k21, k31, v
states = central, fast, slow
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
dx(central) = -(k10 + k12 + k13) * central + k21 * fast + k31 * slow
dx(fast) = k12 * central - k21 * fast
dx(slow) = k13 * central - k31 * slow
out(cp) = central / v
"#,
        r#"
name = three_cpt_kernel
kind = analytical
params = k10, k12, k13, k21, k31, v
states = central, fast, slow
outputs = cp
bolus(iv) -> central
infusion(iv) -> central
structure = three_compartments
out(cp) = central / v ~ continuous()
"#,
        &[
            ("k10", 0.2),
            ("k12", 0.5),
            ("k13", 0.1),
            ("k21", 0.3),
            ("k31", 0.05),
            ("v", 15.0),
        ],
        "iv",
    );
}
