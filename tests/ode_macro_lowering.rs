use approx::assert_relative_eq;
use pharmsol::prelude::*;

fn subject_for_route(input: usize) -> Subject {
    Subject::builder("macro-lowering")
        .infusion(0.0, 100.0, input, 1.0)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .build()
}

fn subject_for_shared_channel(input: usize) -> Subject {
    Subject::builder("macro-shared-channel")
        .bolus(0.0, 100.0, input)
        .infusion(6.0, 60.0, input, 2.0)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .missing_observation(6.5, 0)
        .missing_observation(7.0, 0)
        .missing_observation(8.0, 0)
        .build()
}

fn subject_for_covariates(input: usize) -> Subject {
    Subject::builder("macro-covariates")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .covariate("wt", 0.0, 70.0)
        .build()
}

fn injected_macro_ode() -> equation::ODE {
    ode! {
        name: "injected_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: {
            infusion(iv) -> central,
        },
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn injected_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = rateiv[0] - ke * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("injected_one_cpt")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten injected metadata should validate")
}

fn explicit_macro_ode() -> equation::ODE {
    ode! {
        name: "explicit_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: {
            infusion(iv) -> central,
        },
        diffeq: |x, _p, _t, dx, _bolus, rateiv, _cov| {
            dx[central] = rateiv[iv] - ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn explicit_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = rateiv[0] - ke * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("explicit_one_cpt")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ),
    )
    .expect("handwritten explicit metadata should validate")
}

fn shared_channel_macro_ode() -> equation::ODE {
    ode! {
        name: "shared_channel_one_cpt",
        params: [ka, ke, v, tlag, f_oral],
        states: [depot, central],
        outputs: [cp],
        routes: {
            bolus(oral) -> depot,
            infusion(iv) -> central,
        },
        diffeq: |x, _p, _t, dx, bolus, rateiv, _cov| {
            dx[depot] = bolus[oral] - ka * x[depot];
            dx[central] = ka * x[depot] + rateiv[iv] - ke * x[central];
        },
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn shared_channel_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, bolus, rateiv, _cov| {
            fetch_params!(p, ka, ke, _v, _tlag, _f_oral);
            dx[0] = bolus[0] - ka * x[0];
            dx[1] = ka * x[0] + rateiv[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("shared_channel_one_cpt")
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .expect_explicit_input(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("handwritten shared-channel metadata should validate")
}

fn covariate_macro_ode() -> equation::ODE {
    ode! {
        name: "covariate_one_cpt",
        params: [ka, ke, v],
        covariates: [wt],
        states: [gut, central],
        outputs: [cp],
        routes: {
            bolus(oral) -> gut,
        },
        diffeq: |x, _p, t, dx, cov| {
            fetch_cov!(cov, t, wt);
            let scaled_ke = ke * (wt / 70.0);
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - scaled_ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn covariate_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, bolus, _rateiv, cov| {
            fetch_cov!(cov, t, wt);
            fetch_params!(p, ka, ke, _v);
            let scaled_ke = ke * (wt / 70.0);
            dx[0] = bolus[0] - ka * x[0];
            dx[1] = ka * x[0] - scaled_ke * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("covariate_one_cpt")
            .parameters(["ka", "ke", "v"])
            .covariates([equation::Covariate::continuous("wt")])
            .states(["gut", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten covariate metadata should validate")
}

fn assert_prediction_match(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_relative_eq!(left, right, epsilon = 1e-10);
    }
}

#[test]
fn macro_injected_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = injected_macro_ode();
    let handwritten_ode = injected_handwritten_ode();
    let subject = subject_for_route(0);
    let support_point = [0.2, 10.0];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("iv"), Some(0));
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(0));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro injected model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten injected model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_explicit_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = explicit_macro_ode();
    let handwritten_ode = explicit_handwritten_ode();
    let subject = subject_for_route(0);
    let support_point = [0.2, 10.0];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("iv"), Some(0));
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(0));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro explicit model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten explicit model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_shared_channel_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = shared_channel_macro_ode();
    let handwritten_ode = shared_channel_handwritten_ode();
    let subject = subject_for_shared_channel(0);
    let support_point = [1.0, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("oral"), Some(0));
    assert_eq!(macro_ode.route_index("iv"), Some(0));
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.state_index("depot"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(1));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro shared-channel model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten shared-channel model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_covariate_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = covariate_macro_ode();
    let handwritten_ode = covariate_handwritten_ode();
    let subject = subject_for_covariates(0);
    let support_point = [1.0, 0.2, 10.0];
    let macro_metadata = macro_ode
        .metadata()
        .expect("macro covariate model should carry metadata");

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_metadata.covariates().len(), 1);
    assert_eq!(macro_ode.route_index("oral"), Some(0));
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.state_index("gut"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(1));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro covariate model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten covariate model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}
