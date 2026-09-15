//! Tests for label-addressed assay error models.
//!
//! NPAG optimizes gamma/lambda by iterating the error models. A model added by
//! label, like `.add("outeq_0", ...)` for data with outeq = 0, used to be
//! invisible to that iteration, so its factor was never optimized.

use pharmsol::prelude::*;
use pharmsol::{simulator::equation, ODE};

fn metadata_ode() -> ODE {
    ODE::new(
        |x, p, _t, dx, bolus, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = bolus[0] - ke * x[0];
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
        equation::metadata::new("label_keyed_error_model")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["outeq_0"])
            .route(equation::Route::bolus("iv").to_state("central")),
    )
    .expect("metadata should validate")
}

fn observed_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        // Raw outeq = 0 matches the declared `outeq_0` output.
        .observation(1.0, 2.0, 0)
        .observation(2.0, 1.5, 0)
        .observation(4.0, 0.8, 0)
        .build();
    Data::new(vec![subject])
}

#[test]
fn label_addressed_error_model_is_optimized_by_npag_style_loop() -> Result<(), PharmsolError> {
    let ode = metadata_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let parameters =
        Parameters::with_model(&ode, [("ke", 0.4), ("v", 20.0)]).expect("valid named parameters");

    let mut error_models = AssayErrorModels::new().add(
        "outeq_0",
        AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
    )?;

    // The same iteration NPAG does when it optimizes gamma/lambda.
    let optimizable: Vec<usize> = error_models
        .iter_mut()
        .filter_map(|(outeq, model)| model.optimize().then_some(outeq))
        .collect();
    assert_eq!(
        optimizable.len(),
        1,
        "a label-addressed model must be visible to the optimizer"
    );

    let before = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;
    let outeq = optimizable[0];
    let updated = error_models.factor(outeq)? * 2.0;
    error_models.set_factor(outeq, updated)?;
    let after = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;

    assert!(
        (before - after).abs() > f64::EPSILON,
        "updating lambda must change the likelihood"
    );
    Ok(())
}
