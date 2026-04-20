//! Integration tests for the JIT-compiled model API.
//!
//! Validates that a Cranelift-compiled model produces the same predictions as
//! the closed-form analytical one-compartment IV-bolus solution to within
//! solver tolerance.

#![cfg(feature = "jit")]

use approx::assert_relative_eq;
use pharmsol::jit::Model;
use pharmsol::prelude::*;
use pharmsol::Predictions;

#[test]
fn one_compartment_iv_bolus_matches_analytical() {
    // dx/dt = -(CL/V) * x ; cp = x / V ; bolus 100 at t=0
    let ode = Model::new("1cmt-iv")
        .compartments(["central"])
        .params(["CL", "V"])
        .dxdt("central", "rateiv[0] - (CL / V) * central")
        .output("cp", "central / V")
        .compile()
        .expect("compile");

    let cl: f64 = 5.0;
    let v: f64 = 50.0;

    let mut builder = Subject::builder("p1").bolus(0.0, 100.0, 0);
    for &t in &[0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0] {
        builder = builder.observation(t, 0.0, 0);
    }
    let subject = builder.build();

    let (preds, _) = ode
        .simulate_subject(&subject, &[cl, v], None)
        .expect("simulate");

    let predictions = preds.get_predictions();
    assert_eq!(predictions.len(), 7);

    let ke = cl / v;
    for p in &predictions {
        let analytical = (100.0 / v) * (-ke * p.time()).exp();
        assert_relative_eq!(p.prediction(), analytical, max_relative = 1e-3);
    }
}

#[test]
fn one_compartment_iv_infusion_with_covariate() {
    // CLi = CL * (WT / 70)^0.75 ; one-compartment IV infusion.
    let ode = Model::new("1cmt-iv-allo")
        .compartments(["central"])
        .params(["CL", "V"])
        .covariates(["WT"])
        .dxdt(
            "central",
            "rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central",
        )
        .output("cp", "central / V")
        .compile()
        .expect("compile");

    let subject = Subject::builder("p1")
        .infusion(0.0, 100.0, 0, 1.0)
        .covariate("WT", 0.0, 70.0)
        .observation(2.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .build();

    let (preds, _) = ode
        .simulate_subject(&subject, &[5.0, 50.0], None)
        .expect("simulate");

    let v = preds.get_predictions();
    assert_eq!(v.len(), 2);
    // Sanity: predictions are positive and decreasing after infusion ends.
    assert!(v[0].prediction() > 0.0);
    assert!(v[1].prediction() > 0.0);
    assert!(v[1].prediction() < v[0].prediction());
}

#[test]
fn rejects_unknown_identifier() {
    let err = Model::new("bad")
        .compartments(["x"])
        .params(["k"])
        .dxdt("x", "-k * y") // y is undeclared
        .output("cp", "x")
        .compile()
        .expect_err("should fail");
    let msg = err.to_string();
    assert!(msg.contains("unresolved"), "got: {msg}");
}

#[test]
fn from_text_simulates_one_compartment_iv_bolus() {
    use approx::assert_relative_eq;

    let src = r#"
        name         = onecmt
        compartments = central
        params       = CL, V
        dxdt(central) = rateiv[0] - (CL / V) * central
        out(cp)       = central / V
    "#;
    let ode = Model::from_text(src)
        .expect("text parse")
        .compile()
        .expect("compile");

    let cl: f64 = 5.0;
    let v: f64 = 50.0;
    let mut b = Subject::builder("p1").bolus(0.0, 100.0, 0);
    for &t in &[0.5_f64, 1.0, 2.0, 4.0, 8.0] {
        b = b.observation(t, 0.0, 0);
    }
    let subject = b.build();

    let (preds, _) = ode
        .simulate_subject(&subject, &[cl, v], None)
        .expect("simulate");
    let ke = cl / v;
    for p in preds.get_predictions() {
        let analytical = (100.0 / v) * (-ke * p.time()).exp();
        assert_relative_eq!(p.prediction(), analytical, max_relative = 1e-3);
    }
}
