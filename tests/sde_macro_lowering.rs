use approx::assert_relative_eq;
use pharmsol::prelude::*;
use pharmsol::Predictions;

fn infusion_subject(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("sde-macro-iv")
        .infusion(0.0, 120.0, input, 1.0)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn oral_subject(input: impl ToString, outeq: impl ToString) -> Subject {
    Subject::builder("sde-macro-oral")
        .bolus(0.0, 100.0, input)
        .missing_observation(0.5, outeq.to_string())
        .missing_observation(1.0, outeq.to_string())
        .missing_observation(2.0, outeq)
        .build()
}

fn shared_channel_subject() -> Subject {
    Subject::builder("sde-macro-shared")
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
    Subject::builder("sde-macro-covariates")
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

fn prediction_means(predictions: &ndarray::Array2<Prediction>) -> Vec<f64> {
    predictions
        .get_predictions()
        .into_iter()
        .map(|prediction| prediction.prediction())
        .collect()
}

fn assert_prediction_match(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_relative_eq!(left, right, epsilon = 1e-10);
    }
}

fn macro_infusion_sde() -> equation::SDE {
    sde! {
        name: "one_cpt_sde",
        params: [ke, sigma_ke, v],
        states: [central],
        outputs: [cp],
        particles: 16,
        routes: {
            infusion(iv) -> central,
        },
        drift: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        diffusion: |sigma| {
            sigma[central] = sigma_ke;
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_infusion_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _sigma_ke, _v);
            dx[0] = rateiv[0] - ke * x[0];
        },
        |p, sigma| {
            fetch_params!(p, _ke, sigma_ke, _v);
            sigma[0] = sigma_ke;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _sigma_ke, v);
            y[0] = x[0] / v;
        },
        16,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cpt_sde")
            .kind(equation::ModelKind::Sde)
            .parameters(["ke", "sigma_ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            )
            .particles(16),
    )
    .expect("handwritten SDE metadata should validate")
}

fn macro_absorption_sde() -> equation::SDE {
    sde! {
        name: "one_cmt_abs_sde",
        params: [ka, ke, sigma_ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        particles: 8,
        routes: {
            bolus(oral) -> gut,
        },
        drift: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        diffusion: |sigma| {
            sigma[gut] = 0.0 * sigma_ke;
            sigma[central] = sigma_ke;
        },
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

fn handwritten_absorption_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _sigma_ke, _v, _tlag, _f_oral);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, sigma| {
            fetch_params!(p, _ka, _ke, sigma_ke, _v, _tlag, _f_oral);
            sigma[0] = 0.0 * sigma_ke;
            sigma[1] = sigma_ke;
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _sigma_ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
        8,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_abs_sde")
            .kind(equation::ModelKind::Sde)
            .parameters(["ka", "ke", "sigma_ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
            )
            .particles(8),
    )
    .expect("handwritten absorption SDE metadata should validate")
}

fn macro_shared_channel_sde() -> equation::SDE {
    sde! {
        name: "one_cmt_shared_sde",
        params: [ka, ke, sigma_ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        particles: 8,
        routes: {
            bolus(oral) -> gut,
            infusion(iv) -> central,
        },
        drift: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        diffusion: |sigma| {
            sigma[gut] = 0.0;
            sigma[central] = 0.0;
        },
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

fn handwritten_shared_channel_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ka, ke, _sigma_ke, _v, _tlag, _f_oral);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] + rateiv[0] - ke * x[1];
        },
        |_p, sigma| {
            sigma[0] = 0.0;
            sigma[1] = 0.0;
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _sigma_ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
        8,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_shared_sde")
            .kind(equation::ModelKind::Sde)
            .parameters(["ka", "ke", "sigma_ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ])
            .particles(8),
    )
    .expect("handwritten shared-channel SDE metadata should validate")
}

fn macro_covariate_sde() -> equation::SDE {
    sde! {
        name: "one_cmt_sde_covariates",
        params: [ka, ke, sigma_ke, v, tlag, f_oral, base_gut, base_central],
        covariates: [wt, renal],
        states: [gut, central],
        outputs: [cp],
        particles: 8,
        routes: {
            bolus(oral) -> gut,
            infusion(iv) -> central,
        },
        drift: |x, _t, dx| {
            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            let adjusted_ke = ke * wt_scale * renal_scale;

            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - adjusted_ke * x[central];
        },
        diffusion: |sigma| {
            sigma[gut] = 0.0 * sigma_ke;
            sigma[central] = 0.0 * sigma_ke;
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

fn handwritten_covariate_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(
                p,
                ka,
                ke,
                _sigma_ke,
                _v,
                _tlag,
                _f_oral,
                _base_gut,
                _base_central
            );
            fetch_cov!(cov, t, wt, renal);

            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            let adjusted_ke = ke * wt_scale * renal_scale;

            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] + rateiv[0] - adjusted_ke * x[1];
        },
        |p, sigma| {
            fetch_params!(
                p,
                _ka,
                _ke,
                sigma_ke,
                _v,
                _tlag,
                _f_oral,
                _base_gut,
                _base_central
            );
            sigma[0] = 0.0 * sigma_ke;
            sigma[1] = 0.0 * sigma_ke;
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _sigma_ke,
                _v,
                tlag,
                _f_oral,
                _base_gut,
                _base_central
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
                _sigma_ke,
                _v,
                _tlag,
                f_oral,
                _base_gut,
                _base_central
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
                _sigma_ke,
                _v,
                _tlag,
                _f_oral,
                base_gut,
                base_central
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
                _sigma_ke,
                v,
                _tlag,
                _f_oral,
                _base_gut,
                _base_central
            );
            fetch_cov!(cov, t, wt, renal);

            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[0] = x[1] / adjusted_v;
        },
        8,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_sde_covariates")
            .kind(equation::ModelKind::Sde)
            .parameters([
                "ka",
                "ke",
                "sigma_ke",
                "v",
                "tlag",
                "f_oral",
                "base_gut",
                "base_central",
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
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ])
            .particles(8),
    )
    .expect("handwritten covariate SDE metadata should validate")
}

#[test]
fn sde_macro_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_model = macro_infusion_sde();
    let handwritten_model = handwritten_infusion_sde();
    let subject = infusion_subject("iv", "cp");
    let support_point = [0.2, 0.0, 10.0];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("iv"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("central"), Some(0));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro SDE model should simulate");
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten SDE model should simulate");

    assert_prediction_match(
        &prediction_means(&macro_predictions),
        &prediction_means(&handwritten_predictions),
    );
}

#[test]
fn sde_macro_supports_lag_fa_init_and_named_sigma_bindings() {
    let macro_model = macro_absorption_sde();
    let handwritten_model = handwritten_absorption_sde();
    let subject = oral_subject("oral", "cp");
    let support_point = [1.1, 0.2, 0.0, 10.0, 0.25, 0.8];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("oral"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("gut"), Some(0));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro absorption SDE should simulate");
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten absorption SDE should simulate");

    assert_prediction_match(
        &prediction_means(&macro_predictions),
        &prediction_means(&handwritten_predictions),
    );
}

#[test]
fn sde_macro_shared_channel_lowering_matches_handwritten_metadata_and_predictions() {
    let macro_model = macro_shared_channel_sde();
    let handwritten_model = handwritten_shared_channel_sde();
    let subject = shared_channel_subject();
    let support_point = [1.1, 0.2, 0.0, 10.0, 0.25, 0.8];

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());
    assert_eq!(macro_model.route_index("oral"), Some(0));
    assert_eq!(macro_model.route_index("iv"), Some(0));
    assert_eq!(macro_model.output_index("cp"), Some(0));
    assert_eq!(macro_model.state_index("gut"), Some(0));
    assert_eq!(macro_model.state_index("central"), Some(1));

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro shared-channel SDE should simulate");
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten shared-channel SDE should simulate");

    assert_prediction_match(
        &prediction_means(&macro_predictions),
        &prediction_means(&handwritten_predictions),
    );
}

#[test]
fn sde_macro_covariates_lower_to_handwritten_behavior() {
    let macro_model = macro_covariate_sde();
    let handwritten_model = handwritten_covariate_sde();

    assert_eq!(macro_model.metadata(), handwritten_model.metadata());

    let subject = covariate_subject("oral", "iv", "cp");
    let support_point = [1.0, 0.16, 0.0, 32.0, 0.5, 0.8, 3.0, 14.0];

    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro covariate SDE should simulate");
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten covariate SDE should simulate");

    assert_prediction_match(
        &prediction_means(&macro_predictions),
        &prediction_means(&handwritten_predictions),
    );
}
