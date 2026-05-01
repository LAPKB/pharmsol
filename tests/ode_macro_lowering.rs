use approx::assert_relative_eq;
use pharmsol::prelude::data::read_pmetrics;
use pharmsol::prelude::*;
use tempfile::NamedTempFile;

fn write_pmetrics_fixture(contents: &str) -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temporary Pmetrics fixture");
    std::fs::write(file.path(), contents).expect("write temporary Pmetrics fixture");
    file
}

fn subject_for_route(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("macro-lowering")
        .infusion(0.0, 100.0, input, 1.0)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn subject_for_shared_input() -> Subject {
    Subject::builder("macro-shared-input")
        .bolus(0.0, 100.0, "oral")
        .infusion(6.0, 60.0, "iv", 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(6.5, "cp")
        .missing_observation(7.0, "cp")
        .missing_observation(8.0, "cp")
        .build()
}

fn subject_for_covariates(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("macro-covariates")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .covariate("wt", 0.0, 70.0)
        .build()
}

fn subject_for_numeric_bolus_route(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("numeric-bolus-route")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn injected_macro_ode() -> equation::ODE {
    ode! {
        name: "injected_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
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

fn numeric_label_macro_ode() -> equation::ODE {
    ode! {
        name: "numeric_label_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [1],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
}

fn numeric_label_handwritten_ode() -> equation::ODE {
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
        equation::metadata::new("numeric_label_one_cpt")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["1"])
            .route(
                equation::Route::infusion("1")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten numeric-label metadata should validate")
}

fn shared_input_macro_ode() -> equation::ODE {
    ode! {
        name: "shared_input_one_cpt",
        params: [ka, ke, v, tlag, f_oral],
        states: [depot, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> depot,
            infusion(iv) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - ke * x[central];
        },
        lag: |_t| {
            lag! { oral => tlag }
        },
        fa: |_t| {
            fa! { oral => f_oral }
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn shared_input_handwritten_ode() -> equation::ODE {
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
        equation::metadata::new("shared_input_one_cpt")
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .inject_input_to_destination(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ]),
    )
    .expect("handwritten shared-input metadata should validate")
}

fn numeric_route_property_macro_ode() -> equation::ODE {
    ode! {
        name: "numeric_route_property_one_cpt",
        params: [ka, ke, v, tlag, f_oral],
        states: [depot, central],
        outputs: [1],
        routes: [
            bolus(1) -> depot,
        ],
        diffeq: |x, _t, dx| {
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - ke * x[central];
        },
        lag: |_t| {
            lag! { 1 => tlag }
        },
        fa: |_t| {
            fa! { 1 => f_oral }
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
}

fn numeric_route_property_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, bolus, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v, _tlag, _f_oral);
            dx[0] = bolus[0] - ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
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
        equation::metadata::new("numeric_route_property_one_cpt")
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["depot", "central"])
            .outputs(["1"])
            .route(
                equation::Route::bolus("1")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten numeric route-property metadata should validate")
}

fn mixed_output_labels_macro_ode() -> equation::ODE {
    ode! {
        name: "mixed_output_labels_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [cp, 0, 1],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[0] = 2.0 * x[central] / v;
            y[1] = 3.0 * x[central] / v;
        },
    }
}

fn mixed_output_labels_handwritten_ode() -> equation::ODE {
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
            y[1] = 2.0 * x[0] / v;
            y[2] = 3.0 * x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(3)
    .with_metadata(
        equation::metadata::new("mixed_output_labels_one_cpt")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp", "0", "1"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten mixed-output metadata should validate")
}

fn covariate_macro_ode() -> equation::ODE {
    ode! {
        name: "covariate_one_cpt",
        params: [ka, ke, v],
        covariates: [wt],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        diffeq: |x, _t, dx| {
            let scaled_ke = ke * (wt / 70.0);
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - scaled_ke * x[central];
        },
        out: |x, _t, y| {
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
    let subject = subject_for_route("iv", "cp");
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
fn macro_numeric_labels_lower_to_dense_slots() {
    let macro_ode = numeric_label_macro_ode();
    let handwritten_ode = numeric_label_handwritten_ode();
    let subject = subject_for_route("1", "1");
    let support_point = [0.2, 10.0];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("1"), Some(0));
    assert_eq!(macro_ode.output_index("1"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(0));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro numeric-label model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten numeric-label model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_shared_input_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = shared_input_macro_ode();
    let handwritten_ode = shared_input_handwritten_ode();
    let subject = subject_for_shared_input();
    let support_point = [1.0, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("oral"), Some(0));
    assert_eq!(macro_ode.route_index("iv"), Some(0));
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.state_index("depot"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(1));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro shared-input model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten shared-input model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_mixed_output_labels_lower_to_dense_slots() {
    let macro_ode = mixed_output_labels_macro_ode();
    let handwritten_ode = mixed_output_labels_handwritten_ode();
    let subject = Subject::builder("mixed-output-labels")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "0")
        .missing_observation(2.0, "1")
        .build();
    let support_point = [0.2, 10.0];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.output_index("cp"), Some(0));
    assert_eq!(macro_ode.output_index("0"), Some(1));
    assert_eq!(macro_ode.output_index("1"), Some(2));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro mixed-output model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten mixed-output model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_numeric_route_properties_lower_to_dense_slots() {
    let macro_ode = numeric_route_property_macro_ode();
    let handwritten_ode = numeric_route_property_handwritten_ode();
    let subject = subject_for_numeric_bolus_route("1", "1");
    let support_point = [1.0, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(macro_ode.route_index("1"), Some(0));
    assert_eq!(macro_ode.output_index("1"), Some(0));
    assert_eq!(macro_ode.state_index("depot"), Some(0));
    assert_eq!(macro_ode.state_index("central"), Some(1));

    let macro_predictions = macro_ode
        .estimate_predictions(&subject, &support_point)
        .expect("macro numeric route-property model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_ode
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten numeric route-property model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn macro_named_labels_resolve_from_pmetrics_ingestion() {
    let file = write_pmetrics_fixture(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\npt1,1,0,1,100,.,.,iv,.,.,.,.,.,.,.\npt1,0,0.5,.,.,.,.,.,.,cp,0,.,.,.,.\npt1,0,1.0,.,.,.,.,.,.,cp,0,.,.,.,.\npt1,0,2.0,.,.,.,.,.,.,cp,0,.,.,.,.\n",
    );

    let data =
        read_pmetrics(file.path().display().to_string()).expect("read named-label Pmetrics data");
    let subject = &data.subjects()[0];
    let support_point = [0.2, 10.0];

    let pmetrics_predictions = injected_macro_ode()
        .estimate_predictions(subject, &support_point)
        .expect("macro named-label model should simulate")
        .flat_predictions()
        .to_vec();
    let manual_predictions = injected_macro_ode()
        .estimate_predictions(&subject_for_route("iv", "cp"), &support_point)
        .expect("macro internal-index model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&pmetrics_predictions, &manual_predictions);
}

#[test]
fn macro_numeric_labels_resolve_from_pmetrics_ingestion() {
    let file = write_pmetrics_fixture(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\npt1,1,0,1,100,.,.,1,.,.,.,.,.,.,.\npt1,0,0.5,.,.,.,.,.,.,1,0,.,.,.,.\npt1,0,1.0,.,.,.,.,.,.,1,0,.,.,.,.\npt1,0,2.0,.,.,.,.,.,.,1,0,.,.,.,.\n",
    );

    let data =
        read_pmetrics(file.path().display().to_string()).expect("read numeric-label Pmetrics data");
    let subject = &data.subjects()[0];
    let support_point = [0.2, 10.0];

    let pmetrics_predictions = numeric_label_macro_ode()
        .estimate_predictions(subject, &support_point)
        .expect("macro numeric-label model should simulate")
        .flat_predictions()
        .to_vec();
    let manual_predictions = numeric_label_macro_ode()
        .estimate_predictions(&subject_for_route("1", "1"), &support_point)
        .expect("macro internal-index numeric-label model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&pmetrics_predictions, &manual_predictions);
}

#[test]
fn macro_covariate_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_ode = covariate_macro_ode();
    let handwritten_ode = covariate_handwritten_ode();
    let subject = subject_for_covariates("oral", "cp");
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
