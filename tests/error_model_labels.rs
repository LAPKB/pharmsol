//! Tests for label-keyed assay error models.
//!
//! [`AssayErrorModels`] stores one model per output label and nothing else.
//! Labels are resolved to dense output slots exactly once, by the equation,
//! through the same resolver observation labels go through.

use pharmsol::prelude::*;
use pharmsol::{simulator::equation, Equation, ODE};

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

/// Two declared outputs, so a bound set always has two dense slots.
fn two_output_ode() -> ODE {
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
            y[1] = 2.0 * x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(2)
    .with_metadata(
        equation::metadata::new("two_output_error_model")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp", "effect"])
            .route(equation::Route::bolus("iv").to_state("central")),
    )
    .expect("metadata should validate")
}

fn two_output_parameters(ode: &ODE) -> Parameters {
    Parameters::with_model(ode, [("ke", 0.4), ("v", 20.0)]).expect("valid named parameters")
}

fn observed_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        // The data must spell the declared output name in full.
        .observation(1.0, 2.0, "outeq_0")
        .observation(2.0, 1.5, "outeq_0")
        .observation(4.0, 0.8, "outeq_0")
        .build();
    Data::new(vec![subject])
}

fn additive(lambda: f64) -> AssayErrorModel {
    AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), lambda)
}

#[test]
fn label_keyed_error_model_is_optimized_by_npag_style_loop() -> Result<(), PharmsolError> {
    let ode = metadata_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let parameters =
        Parameters::with_model(&ode, [("ke", 0.4), ("v", 20.0)]).expect("valid named parameters");

    let mut error_models = AssayErrorModels::new().add("outeq_0", additive(0.5))?;

    // The same iteration NPAG does when it optimizes gamma/lambda.
    let optimizable: Vec<OutputLabel> = error_models
        .iter()
        .filter(|(_, model)| model.optimize())
        .map(|(label, _)| label.clone())
        .collect();
    assert_eq!(
        optimizable.len(),
        1,
        "a label-keyed model must be visible to the optimizer"
    );

    let before = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;
    for label in &optimizable {
        error_models.set_factor(label, error_models.factor(label)? * 2.0)?;
    }
    let after = ode.estimate_log_likelihood(subject, &parameters, &error_models)?;

    assert!(
        (before - after).abs() > f64::EPSILON,
        "updating lambda must change the likelihood"
    );
    Ok(())
}

/// Regression: `AssayErrorModel::None` is a public variant, and the old dense
/// storage reused the first `None` slot for the next labelled model. Two labels
/// then aliased onto the same output and `len()` reported one model for two.
#[test]
fn a_none_model_does_not_steal_the_slot_of_another_label() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new()
        .add("cp", AssayErrorModel::None)?
        .add("effect", additive(0.5))?;

    assert_eq!(error_models.len(), 2, "both labels must be kept");

    let ode = two_output_ode();
    let parameters = two_output_parameters(&ode);

    // `effect` must resolve to its own output, not to `cp`'s slot.
    let effect_subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, "effect")
        .build();
    ode.estimate_log_likelihood(&effect_subject, &parameters, &error_models)?;

    // And `cp` still has no usable model, so an observation on it fails.
    let cp_subject = Subject::builder("2")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, "cp")
        .build();
    let err = ode
        .estimate_log_likelihood(&cp_subject, &parameters, &error_models)
        .expect_err("cp was given AssayErrorModel::None");
    assert!(
        err.to_string().contains("ErrorModel::None"),
        "expected a None-error-model failure, got: {err}"
    );
    Ok(())
}

/// Regression: a bound set is sized by the equation's output count, so an
/// output that was never given a model reports "no error model" rather than the
/// misleading "index is invalid" the short dense vector used to produce.
#[test]
fn an_output_without_a_model_reports_a_missing_model_not_a_bad_index() {
    let error_models = AssayErrorModels::new()
        .add("cp", additive(0.5))
        .expect("add cp");

    let ode = two_output_ode();
    let parameters = two_output_parameters(&ode);
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, "effect")
        .build();

    let err = ode
        .estimate_log_likelihood(&subject, &parameters, &error_models)
        .expect_err("effect has no error model");
    let message = err.to_string();
    assert!(
        message.contains("ErrorModel::None"),
        "expected a None-error-model failure, got: {message}"
    );
    assert!(
        !message.contains("is invalid"),
        "must not report an invalid index, got: {message}"
    );
}

/// Regression: error models resolve through the equation's own resolver, so an
/// unknown label reports which outputs are actually declared.
#[test]
fn an_unknown_label_lists_the_declared_outputs() {
    let error_models = AssayErrorModels::new()
        .add("plasma", additive(0.5))
        .expect("add plasma");

    let ode = two_output_ode();
    let parameters = two_output_parameters(&ode);
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, "cp")
        .build();

    let err = ode
        .estimate_log_likelihood(&subject, &parameters, &error_models)
        .expect_err("plasma is not a declared output");
    let message = err.to_string();
    assert!(message.contains("plasma"), "got: {message}");
    assert!(message.contains("cp"), "got: {message}");
    assert!(message.contains("effect"), "got: {message}");
}

/// Models bind by label, never by the order they were added in, and an update
/// made through the public type is visible after binding.
#[test]
fn models_bind_by_label_not_by_insertion_order() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let parameters = two_output_parameters(&ode);
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, "cp")
        .build();

    // Added in the opposite order to the declared outputs on purpose.
    let mut error_models = AssayErrorModels::new()
        .add(
            "effect",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 2.0),
        )?
        .add("cp", additive(1.0))?;

    let before = ode.estimate_log_likelihood(&subject, &parameters, &error_models)?;

    // Only `cp` is observed, so changing `effect` must not move the likelihood.
    error_models.set_factor("effect", 10.0)?;
    let unchanged = ode.estimate_log_likelihood(&subject, &parameters, &error_models)?;
    assert_eq!(before, unchanged);

    // Changing `cp` must.
    error_models.set_factor("cp", 5.0)?;
    let changed = ode.estimate_log_likelihood(&subject, &parameters, &error_models)?;
    assert!((before - changed).abs() > f64::EPSILON);
    Ok(())
}

/// Handwritten equations without metadata keep the numeric fallback, so
/// `.add(0, ..)` still resolves to the first output slot.
#[test]
fn numeric_labels_still_work_without_metadata() -> Result<(), PharmsolError> {
    let ode = ODE::new(
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
    .with_nout(1);

    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 2.0, 0)
        .build();
    let error_models = AssayErrorModels::new().add(0, additive(0.5))?;

    let ll = ode.estimate_log_likelihood_dense(&subject, &[0.4, 20.0], &error_models)?;
    assert!(ll.is_finite());
    Ok(())
}
