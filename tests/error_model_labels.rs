//! Error models are addressed by output label, end to end.
//!
//! NPAG optimizes gamma/lambda by iterating the error models and updating each
//! one it finds. A model added by label used to live in a separate store that
//! iteration and the accessors could not see, so its factor was never
//! optimized. Error models are now keyed by label only, and predictions keep
//! the label of the observation they came from.

use pharmsol::prelude::simulator::log_likelihood_batch;
use pharmsol::prelude::*;
use pharmsol::{ResidualErrorModel, ResidualErrorModels};

fn two_output_ode() -> equation::ODE {
    ode! {
        name: "two_output",
        params: [ke, v],
        states: [central],
        outputs: [cp, effect],
        diffeq: |x, _t, dx| {
            dx[central] = bolus[iv] - ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[effect] = 2.0 * x[central] / v;
        },
    }
}

fn observed_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .observation(1.0, 8.0, "effect")
        .observation(2.0, 7.0, "effect")
        .build();
    Data::new(vec![subject])
}

fn parameters(model: &equation::ODE) -> Parameters {
    Parameters::with_model(model, [("ke", 0.2), ("v", 20.0)]).expect("valid named parameters")
}

#[test]
fn label_keyed_error_models_are_optimized_by_npag_style_loop() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let parameters = parameters(&ode);

    let mut error_models = AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
        )?
        .add(
            "effect",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 1.0),
        )?;

    // The same iteration NPAG does when it optimizes gamma/lambda.
    let optimizable: Vec<OutputLabel> = error_models
        .iter_mut()
        .filter_map(|(label, model)| model.optimize().then(|| label.clone()))
        .collect();
    assert_eq!(
        optimizable,
        vec![OutputLabel::new("cp"), OutputLabel::new("effect")],
        "every label-keyed model must be visible to the optimizer"
    );

    let before = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;
    for label in &optimizable {
        let updated = error_models.factor(label)? * 2.0;
        error_models.set_factor(label, updated)?;
    }
    let after = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;

    assert_eq!(error_models.factor("cp")?, 1.0);
    assert_eq!(error_models.factor("effect")?, 2.0);
    assert!(
        (before - after).abs() > f64::EPSILON,
        "updating gamma/lambda must change the likelihood"
    );
    Ok(())
}

#[test]
fn error_models_bind_by_label_not_insertion_order() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let parameters = parameters(&ode);

    let cp = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5);
    let effect = AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 3.0);

    // Added in the opposite order to the declared outputs on purpose.
    let reversed = AssayErrorModels::new()
        .add("effect", effect.clone())?
        .add("cp", cp.clone())?;
    let declared = AssayErrorModels::new()
        .add("cp", cp)?
        .add("effect", effect)?;

    assert_eq!(reversed, declared);
    assert_eq!(
        ode.estimate_log_likelihood(subject, &parameters, &reversed)?,
        ode.estimate_log_likelihood(subject, &parameters, &declared)?,
    );
    Ok(())
}

#[test]
fn bound_and_label_likelihood_paths_agree() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let parameters = parameters(&ode);
    let error_models = AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
        )?
        .add(
            "effect",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 1.0),
        )?;

    // The simulator scores against error models bound to output slots; a
    // caller scores the same predictions by label.
    let bound = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;
    let by_label = ode
        .estimate_predictions(subject, &parameters)?
        .log_likelihood(&error_models)?;
    assert_eq!(bound, by_label);
    Ok(())
}

#[test]
fn predictions_keep_their_output_label() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let predictions = ode.estimate_predictions(subject, &parameters(&ode))?;

    let mut labels: Vec<&str> = predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.outeq().as_str())
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, vec!["cp", "cp", "effect", "effect"]);
    Ok(())
}

#[test]
fn error_model_for_undeclared_output_is_rejected() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let error_models = AssayErrorModels::new().add(
        "concentration",
        AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
    )?;

    let result = ode.estimate_log_likelihood(subject, &parameters(&ode), &error_models);
    assert!(
        matches!(&result, Err(error) if error.to_string().contains("concentration")),
        "an unknown label must be reported by name, got {result:?}"
    );
    Ok(())
}

#[test]
fn residual_error_models_score_by_label() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let parameters = parameters(&ode);

    let residual_error_models = ResidualErrorModels::new()
        .add("effect", ResidualErrorModel::proportional(0.1))
        .add("cp", ResidualErrorModel::constant(0.5));

    let batch = log_likelihood_batch(
        &ode,
        &data,
        &ndarray::Array2::from_shape_vec((1, 2), parameters.as_slice().to_vec()).unwrap(),
        &residual_error_models,
    )?;

    // The same score, computed by hand from labeled predictions.
    let predictions = ode.estimate_predictions(data.subjects()[0], &parameters)?;
    let expected = residual_error_models.total_log_likelihood(
        predictions.predictions().iter().filter_map(|prediction| {
            prediction
                .observation()
                .map(|obs| (prediction.outeq(), obs, prediction.prediction()))
        }),
    );

    assert!(expected.is_finite());
    assert_eq!(batch, vec![expected]);
    Ok(())
}
