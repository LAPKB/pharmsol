//! Error models are addressed by output label, end to end.
//!
//! NPAG optimizes gamma/lambda by iterating the error models and updating each
//! one it finds. A model added by label used to live in a separate store that
//! iteration and the accessors could not see, so its factor was never
//! optimized. Error models are now keyed by label only, and predictions keep
//! the label of the observation they came from.

use pharmsol::data::error_model::ErrorModelError;
use pharmsol::prelude::simulator::{log_likelihood_batch, log_likelihood_subject};
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

    let before = ode.log_likelihood(subject, &parameters, &error_models)?;
    for label in &optimizable {
        let updated = error_models.factor(label)? * 2.0;
        error_models.set_factor(label, updated)?;
    }
    let after = ode.log_likelihood(subject, &parameters, &error_models)?;

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
        ode.log_likelihood(subject, &parameters, &reversed)?,
        ode.log_likelihood(subject, &parameters, &declared)?,
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
    let bound = ode.log_likelihood(subject, &parameters, &error_models)?;
    let by_label = ode
        .predict(subject, &parameters)?
        .log_likelihood(&error_models)?;
    assert_eq!(bound, by_label);
    Ok(())
}

#[test]
fn predictions_keep_their_output_label() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let predictions = ode.predict(subject, &parameters(&ode))?;

    let mut labels: Vec<&str> = predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.outeq().as_str())
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, vec!["cp", "cp", "effect", "effect"]);

    // Turning a prediction back into an observation (e.g. to build simulated
    // data) keeps the label, so the observation resolves to the same output.
    for prediction in predictions.predictions() {
        assert_eq!(prediction.to_observation().outeq(), prediction.outeq());
    }
    Ok(())
}

#[test]
fn an_observed_output_without_an_error_model_is_reported_by_label() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];

    // `effect` is observed, but explicitly has no error model.
    let mut error_models = AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
        )?
        .add("effect", AssayErrorModel::None)?;

    let none_for_effect = |error: &ErrorModelError| matches!(error, ErrorModelError::NoneErrorModel(label) if label == "effect");

    match ode.log_likelihood(subject, &parameters(&ode), &error_models) {
        Err(PharmsolError::ErrorModelError(error)) if none_for_effect(&error) => {}
        other => panic!("expected NoneErrorModel(`effect`), got {other:?}"),
    }

    // An optimizer cannot read or move a factor that does not exist.
    assert!(matches!(error_models.factor("effect"), Err(e) if none_for_effect(&e)));
    assert!(matches!(error_models.set_factor("effect", 1.0), Err(e) if none_for_effect(&e)));
    assert_eq!(error_models.factor("cp")?, 0.5);
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

    let result = ode.log_likelihood(subject, &parameters(&ode), &error_models);
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
    let predictions = ode.predict(data.subjects()[0], &parameters)?;
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

#[test]
fn single_subject_residual_scoring_matches_the_batch() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = data.subjects()[0];
    let parameters = parameters(&ode);
    let parameter_matrix =
        ndarray::Array2::from_shape_vec((1, 2), parameters.as_slice().to_vec()).unwrap();

    let mut residual_error_models = ResidualErrorModels::new()
        .add("cp", ResidualErrorModel::constant(0.5))
        .add("effect", ResidualErrorModel::proportional(0.1));

    let single = log_likelihood_subject(&ode, subject, &parameters, &residual_error_models);
    let batch = log_likelihood_batch(&ode, &data, &parameter_matrix, &residual_error_models)?;
    assert!(single.is_finite());
    assert_eq!(vec![single], batch);

    // A parametric algorithm re-estimates sigma per output between iterations.
    for (_, model) in residual_error_models.iter_mut() {
        *model = model.with_updated_sigma(model.primary_parameter() * 2.0);
    }
    let updated = log_likelihood_subject(&ode, subject, &parameters, &residual_error_models);
    assert_ne!(updated, single);
    assert_eq!(
        vec![updated],
        log_likelihood_batch(&ode, &data, &parameter_matrix, &residual_error_models)?
    );

    // The single-subject entry point reports an undeclared label as -inf.
    let undeclared =
        ResidualErrorModels::new().add("concentration", ResidualErrorModel::constant(0.5));
    assert_eq!(
        log_likelihood_subject(&ode, subject, &parameters, &undeclared),
        f64::NEG_INFINITY
    );
    Ok(())
}

#[test]
fn error_models_serialize_as_a_label_map() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add(
        "cp",
        AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5),
    )?;

    let json = serde_json::to_value(&error_models).expect("serializes");
    assert!(json.get("cp").is_some(), "keyed by label, got {json}");

    let round_trip: AssayErrorModels = serde_json::from_value(json).expect("deserializes");
    assert_eq!(round_trip, error_models);

    let label: OutputLabel = serde_json::from_str("\"cp\"").expect("label deserializes");
    assert_eq!(serde_json::to_string(&label).unwrap(), "\"cp\"");
    Ok(())
}
