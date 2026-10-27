//! Numeric route and output labels are plain names.
//!
//! A model that declares output `1` is matched by data `OUTEQ=1`, by
//! `AssayErrorModels::new().add("1", ...)` and by
//! `ResidualErrorModels::new().add("1", ...)`, all by exact string match.
//! Numeric labels have no relationship to declaration order and are never
//! aliased to `input_<n>` / `outeq_<n>` or resolved by position.

use approx::assert_relative_eq;
use ndarray::Array2;
use pharmsol::prelude::simulator::log_likelihood_batch;
use pharmsol::prelude::*;
use pharmsol::{PharmsolError, Predictions, ResidualErrorModel, ResidualErrorModels};

const PMETRICS_HEADER: &str = "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3";

/// One subject: an infusion on `input`, three observations on `outeq`.
fn pmetrics_data(input: &str, outeq: &str) -> Data {
    let csv = format!(
        "{PMETRICS_HEADER}\n\
         pt1,1,0,1,100,.,.,{input},.,.,.,.,.,.,.\n\
         pt1,0,1,.,.,.,.,.,8.1,{outeq},0,.,.,.,.\n\
         pt1,0,2,.,.,.,.,.,6.9,{outeq},0,.,.,.,.\n\
         pt1,0,4,.,.,.,.,.,5.2,{outeq},0,.,.,.,.\n"
    );
    Data::from_pmetrics_csv_bytes(csv.as_bytes()).expect("Pmetrics data parses")
}

fn pmetrics_subject(input: &str, outeq: &str) -> Subject {
    pmetrics_data(input, outeq).subjects()[0].clone()
}

fn numeric_one_cpt() -> equation::ODE {
    ode! {
        name: "numeric_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [1],
        diffeq: |x, _t, dx| {
            dx[central] = infusion[1] - ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
}

fn named_one_cpt() -> equation::ODE {
    ode! {
        name: "named_one_cpt",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        diffeq: |x, _t, dx| {
            dx[central] = infusion[iv] - ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn assay_error_models(label: &str) -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            label,
            AssayErrorModel::additive(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 0.0),
        )
        .expect("label-first assay error model")
}

fn residual_error_models(label: &str) -> ResidualErrorModels {
    ResidualErrorModels::new().add(label, ResidualErrorModel::combined(0.2, 0.1))
}

fn one_cpt_parameters(model: &equation::ODE) -> Parameters {
    Parameters::with_model(model, [("ke", 0.2), ("v", 10.0)]).expect("valid parameters")
}

fn prediction_values(predictions: &impl Predictions) -> Vec<f64> {
    predictions
        .get_predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .collect()
}

#[test]
fn numeric_output_binds_error_models_from_pmetrics_data_like_a_named_output() {
    let numeric = numeric_one_cpt();
    let named = named_one_cpt();
    // `1.0` is how many tools export the numeric Pmetrics column; it is
    // normalized to the label `1` at parse time.
    let numeric_data = pmetrics_data("1", "1.0");
    let named_data = pmetrics_data("iv", "cp");

    let numeric_subject = &numeric_data.subjects()[0];
    let named_subject = &named_data.subjects()[0];

    let numeric_ll = numeric
        .log_likelihood(
            numeric_subject,
            &one_cpt_parameters(&numeric),
            &assay_error_models("1"),
        )
        .expect("assay error model for output `1` binds");
    let named_ll = named
        .log_likelihood(
            named_subject,
            &one_cpt_parameters(&named),
            &assay_error_models("cp"),
        )
        .expect("assay error model for output `cp` binds");
    assert!(numeric_ll.is_finite());
    assert_relative_eq!(numeric_ll, named_ll, max_relative = 1e-12);

    let parameters = Array2::from_shape_vec((1, 2), vec![0.2, 10.0]).unwrap();
    let numeric_residual = log_likelihood_batch(
        &numeric,
        &numeric_data,
        &parameters,
        &residual_error_models("1"),
    )
    .expect("residual error model for output `1` binds");
    let named_residual = log_likelihood_batch(
        &named,
        &named_data,
        &parameters,
        &residual_error_models("cp"),
    )
    .expect("residual error model for output `cp` binds");
    assert!(numeric_residual[0].is_finite());
    assert_relative_eq!(numeric_residual[0], named_residual[0], max_relative = 1e-12);
}

#[test]
fn error_models_for_undeclared_labels_fail_to_bind() {
    let model = numeric_one_cpt();
    let data = pmetrics_data("1", "1");
    let subject = data.subjects()[0];
    let parameters = one_cpt_parameters(&model);

    // `0` is not a position, and `outeq_1` / `01` are not aliases of `1`.
    for label in ["0", "2", "01", "outeq_1", "cp"] {
        let error = model
            .log_likelihood(subject, &parameters, &assay_error_models(label))
            .expect_err("undeclared assay error-model label must not bind");
        assert!(
            matches!(
                &error,
                PharmsolError::ErrorModelError(
                    pharmsol::data::error_model::ErrorModelError::UnknownOutputLabel(found)
                ) if found == label
            ),
            "unexpected error for `{label}`: {error:?}"
        );

        let parameters = Array2::from_shape_vec((1, 2), vec![0.2, 10.0]).unwrap();
        assert!(
            log_likelihood_batch(&model, &data, &parameters, &residual_error_models(label))
                .is_err(),
            "undeclared residual error-model label `{label}` must not bind"
        );
    }
}

#[test]
fn prefixed_looking_data_labels_are_not_aliases() {
    let model = numeric_one_cpt();
    let parameters = one_cpt_parameters(&model);

    for (input, outeq) in [("input_1", "1"), ("1", "outeq_1"), ("1", "0"), ("0", "1")] {
        let data = pmetrics_data(input, outeq);
        assert!(
            model.predict(data.subjects()[0], &parameters).is_err(),
            "INPUT={input}, OUTEQ={outeq} must not resolve"
        );
    }
}

fn output_two_only() -> equation::ODE {
    ode! {
        name: "output_two_only",
        params: [ke, v],
        states: [central],
        outputs: [2],
        diffeq: |x, _t, dx| {
            dx[central] = infusion[iv] - ke * x[central];
        },
        out: |x, _t, y| {
            y[2] = x[central] / v;
        },
    }
}

#[test]
fn non_contiguous_numeric_output_is_a_single_named_slot() {
    let model = output_two_only();
    let metadata = model.metadata().expect("macro attaches metadata");
    assert_eq!(model.nouteqs(), 1);
    assert_eq!(metadata.output_labels(), vec!["2".to_string()]);
    assert_eq!(metadata.output_for_label("2"), Some(0));

    let parameters = one_cpt_parameters(&model);
    let predictions = model
        .predict(&pmetrics_subject("iv", "2"), &parameters)
        .expect("OUTEQ=2 resolves to the output declared as `2`");
    assert!(prediction_values(&predictions)
        .iter()
        .all(|value| *value > 0.0));

    for outeq in ["1", "0"] {
        let error = model
            .predict(&pmetrics_subject("iv", outeq), &parameters)
            .expect_err("undeclared numeric output must fail");
        assert!(
            matches!(&error, PharmsolError::UnknownOutputLabel { label, .. } if label == outeq),
            "unexpected error for OUTEQ={outeq}: {error:?}"
        );
    }

    let error_models = assay_error_models("2");
    model
        .log_likelihood(&pmetrics_subject("iv", "2"), &parameters, &error_models)
        .expect("assay error model for output `2` binds to the only slot");
}

/// Numeric (`1`, `10`, `2`) and named labels produce identical simulations.
fn numeric_subject() -> Subject {
    Subject::builder("numeric")
        .bolus(0.0, 100.0, "1")
        .infusion(1.0, 50.0, "10", 1.0)
        .missing_observation(0.5, "2")
        .missing_observation(2.0, "2")
        .missing_observation(2.0, "10")
        .missing_observation(4.0, "10")
        .build()
}

fn named_subject() -> Subject {
    Subject::builder("named")
        .bolus(0.0, 100.0, "oral")
        .infusion(1.0, 50.0, "iv", 1.0)
        .missing_observation(0.5, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(2.0, "amount")
        .missing_observation(4.0, "amount")
        .build()
}

#[test]
fn ode_macro_numeric_labels_match_named_labels() {
    let numeric = ode! {
        name: "numeric_two_route",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [2, 10],
        diffeq: |x, _t, dx| {
            dx[gut] = bolus[1] - ka * x[gut];
            dx[central] = ka * x[gut] + infusion[10] - ke * x[central];
        },
        out: |x, _t, y| {
            y[2] = x[central] / v;
            y[10] = x[central];
        },
    };
    let named = ode! {
        name: "named_two_route",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [cp, amount],
        diffeq: |x, _t, dx| {
            dx[gut] = bolus[oral] - ka * x[gut];
            dx[central] = ka * x[gut] + infusion[iv] - ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[amount] = x[central];
        },
    };

    let metadata = numeric.metadata().expect("macro attaches metadata");
    assert_eq!(metadata.route_labels(), vec!["1", "10"]);
    assert_eq!(metadata.output_labels(), vec!["2", "10"]);

    let values = [("ka", 1.0), ("ke", 0.2), ("v", 10.0)];
    let numeric_predictions = numeric
        .predict(
            &numeric_subject(),
            &Parameters::with_model(&numeric, values).unwrap(),
        )
        .expect("numeric labels simulate");
    let named_predictions = named
        .predict(
            &named_subject(),
            &Parameters::with_model(&named, values).unwrap(),
        )
        .expect("named labels simulate");

    let numeric_values = prediction_values(&numeric_predictions);
    assert!(numeric_values.iter().all(|value| *value > 0.0));
    assert_eq!(numeric_values, prediction_values(&named_predictions));
}

#[test]
fn analytical_macro_numeric_labels_match_named_labels() {
    let numeric = analytical! {
        name: "numeric_analytical",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [2, 10],
        routes: [
            bolus(1) -> gut,
            infusion(10) -> central,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _t, y| {
            y[2] = x[central] / v;
            y[10] = x[central];
        },
    };
    let named = analytical! {
        name: "named_analytical",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [cp, amount],
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[amount] = x[central];
        },
    };

    let metadata = numeric.metadata().expect("macro attaches metadata");
    assert_eq!(metadata.route_labels(), vec!["1", "10"]);
    assert_eq!(metadata.output_labels(), vec!["2", "10"]);

    let values = [("ka", 1.0), ("ke", 0.2), ("v", 10.0)];
    let numeric_predictions = numeric
        .predict(
            &numeric_subject(),
            &Parameters::with_model(&numeric, values).unwrap(),
        )
        .expect("numeric labels simulate");
    let named_predictions = named
        .predict(
            &named_subject(),
            &Parameters::with_model(&named, values).unwrap(),
        )
        .expect("named labels simulate");

    let numeric_values = prediction_values(&numeric_predictions);
    assert!(numeric_values.iter().all(|value| *value > 0.0));
    assert_eq!(numeric_values, prediction_values(&named_predictions));
}

#[test]
fn sde_macro_numeric_labels_match_named_labels() {
    // Zero diffusion keeps the particles deterministic.
    let numeric = sde! {
        name: "numeric_sde",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [2, 10],
        particles: 4,
        routes: [
            bolus(1) -> gut,
            infusion(10) -> central,
        ],
        drift: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        diffusion: |sigma| {
            sigma[gut] = 0.0;
            sigma[central] = 0.0;
        },
        out: |x, _t, y| {
            y[2] = x[central] / v;
            y[10] = x[central];
        },
    };
    let named = sde! {
        name: "named_sde",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [cp, amount],
        particles: 4,
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        drift: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        diffusion: |sigma| {
            sigma[gut] = 0.0;
            sigma[central] = 0.0;
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[amount] = x[central];
        },
    };

    let metadata = numeric.metadata().expect("macro attaches metadata");
    assert_eq!(metadata.route_labels(), vec!["1", "10"]);
    assert_eq!(metadata.output_labels(), vec!["2", "10"]);

    let values = [("ka", 1.0), ("ke", 0.2), ("v", 10.0)];
    let numeric_predictions = numeric
        .predict(
            &numeric_subject(),
            &Parameters::with_model(&numeric, values).unwrap(),
        )
        .expect("numeric labels simulate");
    let named_predictions = named
        .predict(
            &named_subject(),
            &Parameters::with_model(&named, values).unwrap(),
        )
        .expect("named labels simulate");

    let numeric_values = prediction_values(&numeric_predictions);
    assert!(numeric_values.iter().all(|value| *value > 0.0));
    for (numeric, named) in numeric_values
        .iter()
        .zip(prediction_values(&named_predictions))
    {
        assert_relative_eq!(*numeric, named, max_relative = 1e-10);
    }
}

#[test]
fn prefixed_looking_names_are_ordinary_labels() {
    // `input_1` and `outeq_1` are just names; the output may even be called
    // `input_1`. They do not answer to `1`.
    let model = ode! {
        name: "prefixed_looking",
        params: [ke, v],
        states: [central],
        outputs: [input_1],
        diffeq: |x, _t, dx| {
            dx[central] = infusion[outeq_1] - ke * x[central];
        },
        out: |x, _t, y| {
            y[input_1] = x[central] / v;
        },
    };
    let parameters = Parameters::with_model(&model, [("ke", 0.2), ("v", 10.0)]).unwrap();

    model
        .predict(&pmetrics_subject("outeq_1", "input_1"), &parameters)
        .expect("prefixed-looking names resolve exactly");
    model
        .log_likelihood(
            &pmetrics_subject("outeq_1", "input_1"),
            &parameters,
            &assay_error_models("input_1"),
        )
        .expect("error model binds to the output named `input_1`");

    for (input, outeq) in [("1", "input_1"), ("outeq_1", "1")] {
        assert!(model
            .predict(&pmetrics_subject(input, outeq), &parameters)
            .is_err());
    }
}

#[cfg(feature = "dsl")]
mod dsl {
    use super::*;
    use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};

    fn runtime_ode(source: &str, name: &str) -> pharmsol::dsl::RuntimeOdeModel {
        match compile_module_source_to_runtime(source, Some(name), |_, _| {})
            .expect("DSL model compiles")
        {
            CompiledRuntimeModel::Ode(model) => model,
            _ => panic!("expected an ODE model"),
        }
    }

    const NUMERIC_DSL: &str = r#"
name = numeric_dsl
kind = ode
params = ke, v
states = central
outputs = 1
ddt(central) = infusion(1) - ke * central
out(1) = central / v
"#;

    const NAMED_DSL: &str = r#"
name = named_dsl
kind = ode
params = ke, v
states = central
outputs = cp
ddt(central) = infusion(iv) - ke * central
out(cp) = central / v
"#;

    #[test]
    fn dsl_numeric_output_binds_error_models_from_pmetrics_data() {
        let numeric = runtime_ode(NUMERIC_DSL, "numeric_dsl");
        let named = runtime_ode(NAMED_DSL, "named_dsl");
        let values = [("ke", 0.2), ("v", 10.0)];

        let numeric_ll = numeric
            .log_likelihood(
                &pmetrics_subject("1", "1"),
                &Parameters::with_model(&numeric, values).unwrap(),
                &assay_error_models("1"),
            )
            .expect("assay error model for DSL output `1` binds");
        let named_ll = named
            .log_likelihood(
                &pmetrics_subject("iv", "cp"),
                &Parameters::with_model(&named, values).unwrap(),
                &assay_error_models("cp"),
            )
            .expect("assay error model for DSL output `cp` binds");
        assert!(numeric_ll.is_finite());
        assert_relative_eq!(numeric_ll, named_ll, max_relative = 1e-12);

        assert!(numeric
            .log_likelihood(
                &pmetrics_subject("1", "1"),
                &Parameters::with_model(&numeric, values).unwrap(),
                &assay_error_models("outeq_1"),
            )
            .is_err());
        assert!(numeric
            .predict(
                &pmetrics_subject("input_1", "1"),
                &Parameters::with_model(&numeric, values).unwrap(),
            )
            .is_err());
    }
}
