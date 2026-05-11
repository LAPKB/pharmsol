use approx::assert_relative_eq;
use pharmsol::prelude::*;

fn infusion_subject(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("analytical-macro-iv")
        .infusion(0.0, 120.0, input, 1.0)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn oral_subject(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("analytical-macro-oral")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn shared_input_subject() -> Subject {
    Subject::builder("analytical-macro-shared")
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

fn covariate_subject(oral: impl ToString, iv: impl ToString, cp: impl ToString) -> Subject {
    Subject::builder("analytical-macro-covariates")
        .bolus(1.0, 100.0, oral)
        .infusion(6.0, 140.0, iv, 2.0)
        .missing_observation(0.25, cp.to_string())
        .missing_observation(0.75, cp.to_string())
        .missing_observation(1.5, cp.to_string())
        .missing_observation(3.0, cp.to_string())
        .missing_observation(6.5, cp.to_string())
        .missing_observation(7.0, cp.to_string())
        .missing_observation(8.0, cp)
        .covariate("wt", 0.0, 68.0)
        .covariate("wt", 8.0, 74.0)
        .covariate("renal", 0.0, 95.0)
        .covariate("renal", 8.0, 72.0)
        .build()
}

fn macro_one_compartment() -> equation::Analytical {
    analytical! {
        name: "one_cpt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _t, y| {
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
        routes: [
            bolus(oral) -> gut,
        ],
        structure: one_compartment_with_absorption,
        lag: |_t| {
            lag! { oral => tlag }
        },
        fa: |_t| {
            fa! { oral => f_oral }
        },
        init: |_t, x| {
            x[gut] = 0.0;
            x[central] = 0.0;
        },
        out: |x, _t, y| {
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

fn macro_two_compartments_with_absorption_declared_order() -> equation::Analytical {
    analytical! {
        name: "two_cmt_abs_declared_order",
        params: [ka, ke, k12, k21, v],
        states: [gut, central, peripheral],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        structure: two_compartments_with_absorption,
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_two_compartments_with_absorption_declared_order() -> equation::Analytical {
    equation::Analytical::new(
        equation::two_compartments_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(3)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("two_cmt_abs_declared_order")
            .kind(equation::ModelKind::Analytical)
            .parameters(["ka", "ke", "k12", "k21", "v"])
            .states(["gut", "central", "peripheral"])
            .outputs(["cp"])
            .route(equation::Route::bolus("oral").to_state("gut"))
            .analytical_kernel(equation::AnalyticalKernel::TwoCompartmentsWithAbsorption),
    )
    .expect("handwritten reordered analytical metadata should validate")
}

fn macro_shared_input_analytical() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs_shared",
        params: [ka, ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        structure: one_compartment_with_absorption,
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

fn handwritten_shared_input_analytical() -> equation::Analytical {
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
    .expect("handwritten shared-input analytical metadata should validate")
}

fn macro_covariate_analytical() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs_covariates",
        params: [ka, ke, v, tlag, f_oral, base_gut, base_central, tvke],
        covariates: [wt, renal],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        structure: one_compartment_with_absorption,
        sec: |_t| {
            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            ke = tvke * wt_scale * renal_scale;
        },
        lag: |_t| {
            let lag_scale = (wt / 70.0).sqrt() * (90.0 / renal).powf(0.1);
            lag! { oral => tlag * lag_scale }
        },
        fa: |_t| {
            let fa_scale = (renal / 90.0).powf(0.1);
            fa! { oral => (f_oral * fa_scale).clamp(0.0, 1.0) }
        },
        init: |_t, x| {
            x[gut] = base_gut + 0.03 * wt;
            x[central] = base_central + 0.08 * renal;
        },
        out: |x, _t, y| {
            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[cp] = x[central] / adjusted_v;
        },
    }
}

fn handwritten_covariate_analytical() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |p, t, cov| {
            fetch_cov!(cov, t, wt, renal);

            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            p[1] = p[7] * wt_scale * renal_scale;
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _v,
                tlag,
                _f_oral,
                _base_gut,
                _base_central,
                _tvke
            );
            fetch_cov!(cov, t, wt, renal);

            let lag_scale = (wt / 70.0).sqrt() * (90.0 / renal).powf(0.1);
            lag! { 0 => tlag * lag_scale }
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _v,
                _tlag,
                f_oral,
                _base_gut,
                _base_central,
                _tvke
            );
            fetch_cov!(cov, t, wt, renal);

            let fa_scale = (renal / 90.0).powf(0.1);
            fa! { 0 => (f_oral * fa_scale).clamp(0.0, 1.0) }
        },
        |p, t, cov, x| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _v,
                _tlag,
                _f_oral,
                base_gut,
                base_central,
                _tvke
            );
            fetch_cov!(cov, t, wt, renal);

            x[0] = base_gut + 0.03 * wt;
            x[1] = base_central + 0.08 * renal;
        },
        |x, p, t, cov, y| {
            fetch_params!(
                p,
                _ka,
                _ke,
                v,
                _tlag,
                _f_oral,
                _base_gut,
                _base_central,
                _tvke
            );
            fetch_cov!(cov, t, wt, renal);

            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[0] = x[1] / adjusted_v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_abs_covariates")
            .kind(equation::ModelKind::Analytical)
            .parameters([
                "ka",
                "ke",
                "v",
                "tlag",
                "f_oral",
                "base_gut",
                "base_central",
                "tvke",
            ])
            .covariates([
                equation::Covariate::continuous("wt"),
                equation::Covariate::continuous("renal"),
            ])
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
    .expect("handwritten covariate analytical metadata should validate")
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
    let subject = infusion_subject("iv", "cp");
    let support_point = [0.2, 10.0];
    let macro_metadata = macro_model
        .metadata()
        .expect("macro analytical metadata exists");

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert!(macro_metadata.route("iv").is_some());
    assert!(macro_metadata.output("cp").is_some());
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
    let subject = oral_subject("oral", "cp");
    let support_point = [1.1, 0.2, 10.0, 0.25, 0.8];
    let macro_metadata = macro_model.metadata().expect("macro metadata exists");

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert!(macro_metadata.route("oral").is_some());
    assert!(macro_metadata.output("cp").is_some());
    assert_eq!(macro_model.state_index("gut"), Some(0));
    assert_eq!(
        macro_metadata.analytical_kernel(),
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
fn analytical_macro_supports_declared_parameter_order_for_built_in_structures() {
    let macro_model = macro_two_compartments_with_absorption_declared_order();
    let handwritten_model = handwritten_two_compartments_with_absorption_declared_order();
    let subject = oral_subject("oral", "cp");
    let support_point = [1.1, 0.2, 0.3, 0.15, 10.0];
    let macro_metadata = macro_model.metadata().expect("macro metadata exists");

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.parameter_index("ka"), Some(0));
    assert_eq!(macro_model.parameter_index("ke"), Some(1));
    assert_eq!(
        macro_metadata.analytical_kernel(),
        Some(equation::AnalyticalKernel::TwoCompartmentsWithAbsorption)
    );

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro reordered analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten reordered analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn analytical_macro_shared_input_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_model = macro_shared_input_analytical();
    let handwritten_model = handwritten_shared_input_analytical();
    let subject = shared_input_subject();
    let support_point = [1.1, 0.2, 10.0, 0.25, 0.8];
    let macro_metadata = macro_model.metadata().expect("macro metadata exists");

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert!(macro_metadata.route("oral").is_some());
    assert!(macro_metadata.route("iv").is_some());
    assert!(macro_metadata.output("cp").is_some());
    assert_eq!(macro_model.state_index("gut"), Some(0));
    assert_eq!(macro_model.state_index("central"), Some(1));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro shared-input analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten shared-input analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}

#[test]
fn analytical_macro_covariates_lower_to_handwritten_behavior() {
    let macro_model = macro_covariate_analytical();
    let handwritten_model = handwritten_covariate_analytical();

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());

    let subject = covariate_subject("oral", "iv", "cp");
    let support_point = [1.0, 0.16, 32.0, 0.5, 0.8, 3.0, 14.0, 0.16];

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro covariate analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten covariate analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_match(&macro_predictions, &handwritten_predictions);
}
