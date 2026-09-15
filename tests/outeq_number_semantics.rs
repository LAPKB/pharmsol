//! Numeric labels are ordinary labels, not OUTEQ/INPUT numbers.
//!
//! A bare numeric data label used to be aliased to a canonical `outeq_<n>` /
//! `input_<n>` declaration, so OUTEQ = 1 silently matched an output declared as
//! `outeq_1`. That aliasing is gone: a model that declares named outputs only
//! matches the exact declared name, and a numeric label is a hard error that
//! names the declared outputs.
//!
//! The metadata-less positional fallback is unaffected and still resolves a
//! numeric label to a dense slot, because there are no declared names to match.

use pharmsol::prelude::*;
use pharmsol::{simulator::equation, ODE};

fn single_output_ode() -> ODE {
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
        equation::metadata::new("outeq_one")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["outeq_1"]) // first and only output, so dense slot 0
            .route(equation::Route::bolus("iv").to_state("central")),
    )
    .expect("metadata should validate")
}

fn subject_with_output(outeq: &str) -> Subject {
    Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, outeq)
        .observation(2.0, 1.5, outeq)
        .build()
}

fn additive() -> AssayErrorModel {
    AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5)
}

fn log_likelihood(
    subject: &Subject,
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    let ode = single_output_ode();
    let parameters =
        Parameters::with_model(&ode, [("ke", 0.4), ("v", 20.0)]).expect("valid named parameters");
    ode.estimate_log_likelihood(subject, &parameters, error_models)
}

#[test]
fn the_declared_output_name_resolves() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add("outeq_1", additive())?;
    log_likelihood(&subject_with_output("outeq_1"), &error_models)?;
    Ok(())
}

/// A numeric OUTEQ column against a model with declared outputs is an error
/// that names the declared outputs and explains the migration.
#[test]
fn a_numeric_data_label_no_longer_aliases_to_the_declared_output() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add("outeq_1", additive())?;
    let message = log_likelihood(&subject_with_output("1"), &error_models)
        .expect_err("numeric OUTEQ must not alias to `outeq_1`")
        .to_string();

    assert!(message.contains("`1`"), "{message}");
    assert!(
        message.contains("available: outeq_1"),
        "the error must list the declared outputs: {message}"
    );
    assert!(
        message.contains("no longer matched"),
        "the error must explain the migration: {message}"
    );
    Ok(())
}

/// `.add(<number>, ..)` is likewise just a label, and it does not alias either:
/// a model that declares `outeq_1` has no output called `1`.
#[test]
fn a_numeric_error_model_label_no_longer_aliases_either() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add(1, additive())?;
    let message = log_likelihood(&subject_with_output("outeq_1"), &error_models)
        .expect_err("`add(1, ..)` must not alias to `outeq_1`")
        .to_string();

    assert!(
        message.contains("outeq_1"),
        "the error must name the unbound output: {message}"
    );
    Ok(())
}

/// Models without metadata have no declared names, so a numeric label still
/// falls back to the dense slot. That fallback is deliberately kept.
#[test]
fn metadata_less_models_still_resolve_numeric_labels_positionally() -> Result<(), PharmsolError> {
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
    let error_models = AssayErrorModels::new().add(0, additive())?;

    let ll = ode.estimate_log_likelihood_dense(&subject, &[0.4, 20.0], &error_models)?;
    assert!(ll.is_finite());
    Ok(())
}
