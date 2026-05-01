use pharmsol::prelude::*;

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn macro_ode_model() -> equation::ODE {
    ode! {
        name: "ode_full_feature_parity",
        params: [ka, ke, kcp, kpc, v, tlag, f_oral, base_depot, base_central, base_peripheral],
        covariates: [wt, renal],
        states: [depot, central, peripheral],
        outputs: [cp],
        routes: {
            bolus(oral) -> depot,
            bolus(load) -> central,
            infusion(iv) -> central,
        },
        diffeq: |x, _t, dx, bolus, rateiv| {
            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            let adjusted_ke = ke * wt_scale * renal_scale;
            let adjusted_kcp = kcp * (wt / 70.0).powf(0.25);

            dx[depot] = bolus[oral] - ka * x[depot];
            dx[central] = bolus[load] + ka * x[depot] + rateiv[iv]
                - (adjusted_ke + adjusted_kcp) * x[central]
                + kpc * x[peripheral];
            dx[peripheral] = adjusted_kcp * x[central] - kpc * x[peripheral];
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
            x[depot] = base_depot + 0.05 * wt;
            x[central] = base_central + 0.1 * renal;
            x[peripheral] = base_peripheral + 0.02 * wt;
        },
        out: |x, _t, y| {
            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[cp] = x[central] / adjusted_v;
        },
    }
}

fn handwritten_ode_model() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, bolus, rateiv, cov| {
            fetch_params!(
                p,
                ka,
                ke,
                kcp,
                kpc,
                _v,
                _tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            let adjusted_ke = ke * wt_scale * renal_scale;
            let adjusted_kcp = kcp * (wt / 70.0).powf(0.25);

            dx[0] = bolus[0] - ka * x[0];
            dx[1] =
                bolus[1] + ka * x[0] + rateiv[0] - (adjusted_ke + adjusted_kcp) * x[1] + kpc * x[2];
            dx[2] = adjusted_kcp * x[1] - kpc * x[2];
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                _v,
                tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
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
                _kcp,
                _kpc,
                _v,
                _tlag,
                f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
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
                _kcp,
                _kpc,
                _v,
                _tlag,
                _f_oral,
                base_depot,
                base_central,
                base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            x[0] = base_depot + 0.05 * wt;
            x[1] = base_central + 0.1 * renal;
            x[2] = base_peripheral + 0.02 * wt;
        },
        |x, p, t, cov, y| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                v,
                _tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[0] = x[1] / adjusted_v;
        },
    )
    .with_nstates(3)
    .with_ndrugs(2)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("ode_full_feature_parity")
            .parameters([
                "ka",
                "ke",
                "kcp",
                "kpc",
                "v",
                "tlag",
                "f_oral",
                "base_depot",
                "base_central",
                "base_peripheral",
            ])
            .covariates([
                equation::Covariate::continuous("wt"),
                equation::Covariate::continuous("renal"),
            ])
            .states(["depot", "central", "peripheral"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .expect_explicit_input(),
                equation::Route::bolus("load")
                    .to_state("central")
                    .expect_explicit_input(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("handwritten ODE metadata should validate")
}

fn build_ode_subject(oral: usize, load: usize, iv: usize, cp: usize) -> Subject {
    Subject::builder("macro-vs-handwritten-ode-full-features")
        .bolus(0.0, 80.0, load)
        .bolus(1.0, 120.0, oral)
        .infusion(6.0, 150.0, iv, 2.5)
        .missing_observation(0.25, cp)
        .missing_observation(0.75, cp)
        .missing_observation(1.5, cp)
        .missing_observation(3.0, cp)
        .missing_observation(6.5, cp)
        .missing_observation(7.0, cp)
        .missing_observation(8.0, cp)
        .missing_observation(12.0, cp)
        .covariate("wt", 0.0, 68.0)
        .covariate("wt", 8.0, 74.0)
        .covariate("renal", 0.0, 95.0)
        .covariate("renal", 8.0, 72.0)
        .build()
}

fn macro_analytical_model() -> equation::Analytical {
    analytical! {
        name: "analytical_full_feature_parity",
        params: [ka, ke, v, tlag, f_oral, base_gut, base_central, tvke],
        covariates: [wt, renal],
        states: [gut, central],
        outputs: [cp],
        routes: {
            bolus(oral) -> gut,
            bolus(load) -> central,
            infusion(iv) -> central,
        },
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

fn handwritten_analytical_model() -> equation::Analytical {
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
    .with_ndrugs(2)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("analytical_full_feature_parity")
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
                equation::Route::bolus("load").to_state("central"),
                equation::Route::infusion("iv").to_state("central"),
            ])
            .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .expect("handwritten analytical metadata should validate")
}

fn build_analytical_subject(oral: usize, load: usize, iv: usize, cp: usize) -> Subject {
    Subject::builder("macro-vs-handwritten-analytical-full-features")
        .bolus(0.0, 60.0, load)
        .bolus(1.0, 100.0, oral)
        .infusion(6.0, 140.0, iv, 2.0)
        .missing_observation(0.25, cp)
        .missing_observation(0.75, cp)
        .missing_observation(1.5, cp)
        .missing_observation(3.0, cp)
        .missing_observation(6.5, cp)
        .missing_observation(7.0, cp)
        .missing_observation(8.0, cp)
        .missing_observation(12.0, cp)
        .covariate("wt", 0.0, 68.0)
        .covariate("wt", 8.0, 74.0)
        .covariate("renal", 0.0, 95.0)
        .covariate("renal", 8.0, 72.0)
        .build()
}

#[test]
fn ode_full_feature_macro_matches_handwritten() -> Result<(), pharmsol::PharmsolError> {
    let macro_ode = macro_ode_model();
    let handwritten_ode = handwritten_ode_model();

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());

    let oral = macro_ode.route_index("oral").expect("oral route exists");
    let load = macro_ode.route_index("load").expect("load route exists");
    let iv = macro_ode.route_index("iv").expect("iv route exists");
    let cp = macro_ode.output_index("cp").expect("cp output exists");

    assert_eq!(oral, iv);
    assert_eq!(load, 1);
    assert_eq!(handwritten_ode.route_index("oral"), Some(oral));
    assert_eq!(handwritten_ode.route_index("load"), Some(load));
    assert_eq!(handwritten_ode.route_index("iv"), Some(iv));
    assert_eq!(handwritten_ode.output_index("cp"), Some(cp));

    let subject = build_ode_subject(oral, load, iv, cp);
    let params = [1.1, 0.18, 0.07, 0.04, 35.0, 0.6, 0.85, 4.0, 18.0, 9.0];

    let macro_predictions = macro_ode.estimate_predictions(&subject, &params)?;
    let handwritten_predictions = handwritten_ode.estimate_predictions(&subject, &params)?;

    let diff = max_abs_diff(
        &macro_predictions.flat_predictions(),
        &handwritten_predictions.flat_predictions(),
    );
    assert!(
        diff <= 1e-10,
        "macro and handwritten ODE predictions diverged: {diff:e}"
    );

    Ok(())
}

#[test]
fn analytical_full_feature_macro_matches_handwritten() -> Result<(), pharmsol::PharmsolError> {
    let macro_analytical = macro_analytical_model();
    let handwritten_analytical = handwritten_analytical_model();

    assert_eq!(
        macro_analytical.metadata(),
        handwritten_analytical.metadata()
    );

    let oral = macro_analytical
        .route_index("oral")
        .expect("oral route exists");
    let load = macro_analytical
        .route_index("load")
        .expect("load route exists");
    let iv = macro_analytical.route_index("iv").expect("iv route exists");
    let cp = macro_analytical
        .output_index("cp")
        .expect("cp output exists");

    assert_eq!(oral, iv);
    assert_eq!(load, 1);
    assert_eq!(handwritten_analytical.route_index("oral"), Some(oral));
    assert_eq!(handwritten_analytical.route_index("load"), Some(load));
    assert_eq!(handwritten_analytical.route_index("iv"), Some(iv));
    assert_eq!(handwritten_analytical.output_index("cp"), Some(cp));

    let subject = build_analytical_subject(oral, load, iv, cp);
    let params = [1.0, 0.16, 32.0, 0.5, 0.8, 3.0, 14.0, 0.16];

    let macro_predictions = macro_analytical.estimate_predictions(&subject, &params)?;
    let handwritten_predictions = handwritten_analytical.estimate_predictions(&subject, &params)?;

    let diff = max_abs_diff(
        &macro_predictions.flat_predictions(),
        &handwritten_predictions.flat_predictions(),
    );
    assert!(
        diff <= 1e-10,
        "macro and handwritten analytical predictions diverged: {diff:e}"
    );

    Ok(())
}
