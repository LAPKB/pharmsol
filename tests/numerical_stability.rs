use pharmsol::prelude::models::{
    one_compartment, one_compartment_with_absorption, two_compartments,
};
use pharmsol::*;

const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-2;

#[test]
fn infusion_vs_analytical_is_stable() {
    let subject = infusion_subject();
    let params = vec![0.1, 1.0];
    let (analytical, ode) = infusion_models();

    assert_models_agree("infusion", &analytical, &ode, &subject, &params);
}

#[test]
fn oral_absorption_tracks_reference() {
    let subject = absorption_subject();
    let params = vec![1.0, 0.1, 1.0];
    let (analytical, ode) = absorption_models();

    assert_models_agree("absorption", &analytical, &ode, &subject, &params);
}

#[test]
fn two_compartment_multi_dose_is_well_behaved() {
    let subject = two_compartment_subject();
    let params = vec![0.1, 3.0, 1.0, 1.0];
    let (analytical, ode) = two_compartment_models();

    assert_models_agree("two_compartment", &analytical, &ode, &subject, &params);
}

fn assert_models_agree(
    label: &str,
    analytical: &equation::Analytical,
    ode: &equation::ODE,
    subject: &Subject,
    params: &[f64],
) {
    let params_vec: Vec<f64> = params.to_vec();
    let analytical_predictions = analytical
        .estimate_predictions(subject, &params_vec)
        .expect("analytical predictions");
    let ode_predictions = ode
        .estimate_predictions(subject, &params_vec)
        .expect("ode predictions");

    let expected = analytical_predictions.flat_predictions();
    let actual = ode_predictions.flat_predictions();

    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: prediction vector length mismatch",
        label
    );

    for (idx, (&reference, &candidate)) in expected.iter().zip(actual.iter()).enumerate() {
        let abs_err = (reference - candidate).abs();
        let rel_err = abs_err / reference.abs().max(ABS_TOL);
        assert!(
            abs_err <= ABS_TOL || rel_err <= REL_TOL,
            "{}: prediction {} differs (ref={} vs cand={}, abs_err={}, rel_err={})",
            label,
            idx,
            reference,
            candidate,
            abs_err,
            rel_err
        );
    }
}

fn infusion_subject() -> Subject {
    let mut builder = Subject::builder("infusion_reference")
        .bolus(0.0, 100.0, 0)
        .infusion(24.0, 150.0, 0, 3.0);

    for &time in &[
        0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 25.0, 26.0, 27.0, 28.0, 32.0, 36.0,
    ] {
        builder = builder.missing_observation(time, 0);
    }

    builder.build()
}

fn infusion_models() -> (equation::Analytical, equation::ODE) {
    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    (analytical, ode)
}

fn absorption_subject() -> Subject {
    let mut builder = Subject::builder("absorption_reference")
        .bolus(0.0, 100.0, 1)
        .infusion(24.0, 150.0, 0, 3.0)
        .bolus(48.0, 100.0, 0);

    for &time in &[
        0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 25.0, 26.0, 27.0, 28.0, 32.0, 36.0, 48.0, 49.0, 50.0,
        52.0, 56.0, 60.0,
    ] {
        builder = builder.missing_observation(time, 0);
    }

    builder.build()
}

fn absorption_models() -> (equation::Analytical, equation::ODE) {
    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0] + b[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    (analytical, ode)
}

fn two_compartment_subject() -> Subject {
    let mut builder = Subject::builder("two_comp_reference")
        .bolus(0.0, 100.0, 0)
        .infusion(24.0, 150.0, 0, 3.0);

    for &time in &[
        0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 25.0, 26.0, 27.0, 28.0, 32.0, 36.0,
    ] {
        builder = builder.missing_observation(time, 0);
    }

    builder.build()
}

fn two_compartment_models() -> (equation::Analytical, equation::ODE) {
    let analytical = equation::Analytical::new(
        two_compartments,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _kcp, _kpc, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, kcp, kpc, _v);
            dx[0] = rateiv[0] - ke * x[0] - kcp * x[0] + kpc * x[1] + b[0];
            dx[1] = kcp * x[0] - kpc * x[1] + b[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _kcp, _kpc, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (2, 1),
    );

    (analytical, ode)
}
