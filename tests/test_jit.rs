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

#[test]
fn let_bindings_share_subexpressions() {
    // Same model written two ways: with and without `let`. They must agree.
    let plain = Model::from_text(
        "
        name = m
        compartments = central
        params = CL, V
        covariates = WT
        dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / (V * (WT / 70.0))) * central
        out(cp) = central / (V * (WT / 70.0))
        ",
    )
    .unwrap()
    .compile()
    .unwrap();

    let withlet = Model::from_text(
        "
        name = m
        compartments = central
        params = CL, V
        covariates = WT
        let wt_ratio = WT / 70.0
        let v_scaled = V * wt_ratio
        let cl_scaled = CL * pow(wt_ratio, 0.75)
        dxdt(central) = rateiv[0] - (cl_scaled / v_scaled) * central
        out(cp) = central / v_scaled
        ",
    )
    .unwrap()
    .compile()
    .unwrap();

    let subject = Subject::builder("p1")
        .infusion(0.0, 100.0, 0, 0.5)
        .covariate("WT", 0.0, 80.0)
        .observation(1.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let (a, _) = plain.simulate_subject(&subject, &[5.0, 50.0], None).unwrap();
    let (b, _) = withlet
        .simulate_subject(&subject, &[5.0, 50.0], None)
        .unwrap();
    let av = a.get_predictions();
    let bv = b.get_predictions();
    assert_eq!(av.len(), bv.len());
    for (x, y) in av.iter().zip(bv.iter()) {
        assert_relative_eq!(x.prediction(), y.prediction(), max_relative = 1e-12);
    }
}

#[test]
fn init_sets_initial_state() {
    // No dose, just a non-zero initial state that decays.
    let ode = Model::from_text(
        "
        name = decay
        compartments = central
        params = CL, V
        init(central) = 200.0
        dxdt(central) = -(CL / V) * central
        out(cp) = central / V
        ",
    )
    .unwrap()
    .compile()
    .unwrap();

    let subject = Subject::builder("p1")
        .observation(0.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let cl = 5.0;
    let v = 50.0;
    let (preds, _) = ode.simulate_subject(&subject, &[cl, v], None).unwrap();
    let vp = preds.get_predictions();
    let ke = cl / v;
    for p in &vp {
        let analytical = (200.0 / v) * (-ke * p.time()).exp();
        assert_relative_eq!(p.prediction(), analytical, max_relative = 1e-3);
    }
}

#[test]
fn lag_shifts_bolus_in_time() {
    // Bolus at t=0 with lag = 2.0 should behave like a bolus at t=2.
    let ode = Model::from_text(
        "
        name = lag1
        compartments = central
        params = CL, V, TLAG
        lag(0) = TLAG
        dxdt(central) = -(CL / V) * central
        out(cp) = central / V
        ",
    )
    .unwrap()
    .compile()
    .unwrap();

    let cl = 5.0;
    let v = 50.0;
    let tlag = 2.0;

    let subject = Subject::builder("p1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0) // before lag-shifted dose -> 0
        .observation(3.0, 0.0, 0) // 1h after dose
        .observation(8.0, 0.0, 0) // 6h after dose
        .build();

    let (preds, _) = ode
        .simulate_subject(&subject, &[cl, v, tlag], None)
        .unwrap();
    let vp = preds.get_predictions();
    let ke = cl / v;

    // t=1 is before the lag-shifted dose at t=2
    assert!(vp[0].prediction().abs() < 1e-6, "got {}", vp[0].prediction());
    // t=3, t=8 -> dose has been at t=2
    let exp1 = (100.0 / v) * (-ke * (3.0 - tlag)).exp();
    let exp2 = (100.0 / v) * (-ke * (8.0 - tlag)).exp();
    assert_relative_eq!(vp[1].prediction(), exp1, max_relative = 1e-3);
    assert_relative_eq!(vp[2].prediction(), exp2, max_relative = 1e-3);
}

#[test]
fn fa_scales_bolus_amount() {
    // fa = 0.5 -> bolus of 100 acts like 50.
    let ode = Model::from_text(
        "
        name = fa1
        compartments = central
        params = CL, V, F
        fa(0) = F
        dxdt(central) = -(CL / V) * central
        out(cp) = central / V
        ",
    )
    .unwrap()
    .compile()
    .unwrap();

    let cl = 5.0;
    let v = 50.0;
    let f = 0.5;

    let subject = Subject::builder("p1")
        .bolus(0.0, 100.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .build();

    let (preds, _) = ode.simulate_subject(&subject, &[cl, v, f], None).unwrap();
    let ke = cl / v;
    for p in preds.get_predictions() {
        let analytical = (f * 100.0 / v) * (-ke * p.time()).exp();
        assert_relative_eq!(p.prediction(), analytical, max_relative = 1e-3);
    }
}

#[test]
fn rejects_let_cycle() {
    let err = Model::from_text(
        "
        name = bad
        compartments = c
        params = k
        let a = b
        let b = a
        dxdt(c) = -k * c + a
        out(y) = c
        ",
    )
    .unwrap()
    .compile()
    .expect_err("cycle should be rejected");
    assert!(err.to_string().contains("cycle"), "got: {err}");
}

#[test]
fn rejects_let_referencing_state_in_aux_context() {
    // `let` referencing a compartment is fine in dxdt/out, but using it from
    // init/lag/fa must fail because state isn't available there.
    let err = Model::from_text(
        "
        name = bad
        compartments = c
        params = k
        let illegal = c
        lag(0) = illegal
        dxdt(c) = -k * c
        out(y) = c
        ",
    )
    .unwrap()
    .compile()
    .expect_err("should fail");
    let msg = err.to_string();
    assert!(msg.contains("unresolved") || msg.contains("illegal"), "got: {msg}");
}
