use approx::assert_relative_eq;
use pharmsol::prelude::*;

fn infusion_subject(input: usize) -> Subject {
    Subject::builder("analytical-macro-iv")
        .infusion(0.0, 120.0, input, 1.0)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .build()
}

fn oral_subject(input: usize) -> Subject {
    Subject::builder("analytical-macro-oral")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .build()
}

fn shared_channel_subject(input: usize) -> Subject {
    Subject::builder("analytical-macro-shared")
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

fn macro_one_compartment() -> equation::Analytical {
    analytical! {
        name: "one_cpt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: {
            infusion(iv) -> central,
        },
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_one_compartment() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment,
        |_p, _t, _cov| {},
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
        equation::metadata::new("one_cpt_iv")
            .kind(equation::ModelKind::Analytical)
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(equation::Route::infusion("iv").to_state("central"))
            .analytical_kernel(equation::AnalyticalKernel::OneCompartment),
    )
    .expect("handwritten analytical metadata should validate")
}

fn macro_one_compartment_with_absorption() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs",
        params: [ka, ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        routes: {
            bolus(oral) -> gut,
        },
        structure: one_compartment_with_absorption,
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        init: |_p, _t, _cov, x| {
            x[gut] = 0.0;
            x[central] = 0.0;
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_one_compartment_with_absorption() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_abs")
            .kind(equation::ModelKind::Analytical)
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .with_lag()
                    .with_bioavailability(),
            )
            .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .expect("handwritten absorption metadata should validate")
}

fn macro_shared_channel_analytical() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs_shared",
        params: [ka, ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        routes: {
            bolus(oral) -> gut,
            infusion(iv) -> central,
        },
        structure: one_compartment_with_absorption,
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

fn handwritten_shared_channel_analytical() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |_p, _t, _cov| {},
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
        equation::metadata::new("one_cmt_abs_shared")
            .kind(equation::ModelKind::Analytical)
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv").to_state("central"),
            ])
            .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .expect("handwritten shared-channel analytical metadata should validate")
}

fn assert_prediction_match(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_relative_eq!(left, right, epsilon = 1e-10);
    }
}

#[test]
fn analytical_macro_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_model = macro_one_compartment();
    let handwritten_model = handwritten_one_compartment();
    let subject = infusion_subject(0);
    let support_point = [0.2, 10.0];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("iv"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("central"), Some(0));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn analytical_macro_supports_extra_parameters_and_named_route_bindings() {
    let macro_model = macro_one_compartment_with_absorption();
    let handwritten_model = handwritten_one_compartment_with_absorption();
    let subject = oral_subject(0);
    let support_point = [1.1, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("oral"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("gut"), Some(0));
    assert_eq!(
        macro_model
            .metadata()
            .expect("macro metadata exists")
            .analytical_kernel(),
        Some(equation::AnalyticalKernel::OneCompartmentWithAbsorption)
    );

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro absorption model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten absorption model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn analytical_macro_shared_channel_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_model = macro_shared_channel_analytical();
    let handwritten_model = handwritten_shared_channel_analytical();
    let subject = shared_channel_subject(0);
    let support_point = [1.1, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("oral"), Some(0));
    assert_eq!(macro_model.route_index("iv"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("gut"), Some(0));
    assert_eq!(macro_model.state_index("central"), Some(1));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro shared-channel analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten shared-channel analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}
