//! Numeric output labels mean OUTEQ numbers, not dense slots.
//!
//! Data with OUTEQ = 1 matches a declared output named `outeq_1`, and
//! `.add(1, ...)` resolves the same way. Before the fix the error model went to
//! dense slot 1 while the data resolved to slot 0, which is why Markus had to
//! write `.add(0, ...)` instead of `.add(1, ...)`.
//!
//! The label-addressed path is covered by
//! `error_model_labels.rs::label_addressed_error_model_is_optimized_by_npag_style_loop`.

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

// Pmetrics data with OUTEQ = 1.
fn data_with_outeq_one() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 2.0, 1)
        .observation(2.0, 1.5, 1)
        .build();
    Data::new(vec![subject])
}

fn additive() -> AssayErrorModel {
    AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.5)
}

fn log_likelihood(error_models: &AssayErrorModels) -> Result<f64, PharmsolError> {
    let ode = single_output_ode();
    let data = data_with_outeq_one();
    let subject = &data.subjects()[0];
    let parameters =
        Parameters::with_model(&ode, [("ke", 0.4), ("v", 20.0)]).expect("valid named parameters");
    ode.estimate_log_likelihood(subject, &parameters, error_models)
}

#[test]
fn label_addressed_error_model_matches_data_outeq_one() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add("outeq_1", additive())?;
    log_likelihood(&error_models)?;
    Ok(())
}

#[test]
fn numeric_add_should_use_the_outeq_number_not_the_dense_slot() -> Result<(), PharmsolError> {
    let error_models = AssayErrorModels::new().add(1, additive())?;
    let result = log_likelihood(&error_models);
    assert!(
        result.is_ok(),
        "data OUTEQ = 1 and add(1) should agree on the same output, got {result:?}"
    );
    Ok(())
}
