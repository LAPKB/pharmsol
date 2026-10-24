//! Every way of scoring a subject against assay error models applies the same
//! up-front label check and gives the same likelihood.
//!
//! The simulator looks error models up by output label. An error model for a
//! label the equation does not declare is rejected before simulating, on every
//! entry point and for every backend, so a typo cannot silently drop an
//! output's error model.

use approx::assert_relative_eq;
use ndarray::Array2;
use pharmsol::data::error_model::ErrorModelError;
use pharmsol::prelude::simulator::log_likelihood_matrix;
use pharmsol::prelude::*;
use pharmsol::Cache;

/// One-compartment IV model data: an infusion on `iv`, two `cp` observations.
fn subject() -> Subject {
    Subject::builder("1")
        .infusion(0.0, 100.0, "iv", 1.0)
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .build()
}

const PARAMETERS: [(&str, f64); 2] = [("ke", 0.2), ("v", 20.0)];

fn cp_error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5),
        )
        .expect("cp error model")
}

fn undeclared_error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "concentration",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5),
        )
        .expect("concentration error model")
}

fn assert_unknown_label<T: std::fmt::Debug>(
    entry_point: &str,
    result: Result<T, PharmsolError>,
    label: &str,
) {
    match result {
        Err(PharmsolError::ErrorModelError(ErrorModelError::UnknownOutputLabel(found)))
            if found == label => {}
        other => panic!("{entry_point}: expected UnknownOutputLabel(`{label}`), got {other:?}"),
    }
}

/// Score the same subject through each public entry point and check that they
/// agree, then check that each one rejects an undeclared label.
fn assert_entry_points_agree(
    name: &str,
    equation: &impl Equation,
    parameters: Parameters,
) -> Result<(), PharmsolError> {
    let subject = subject();
    let data = Data::new(vec![subject.clone()]);
    let support_points =
        Array2::from_shape_vec((1, 2), parameters.as_slice().to_vec()).expect("one support point");
    let error_models = cp_error_models();

    let log_likelihood = equation.estimate_log_likelihood(&subject, &parameters, &error_models)?;
    assert!(log_likelihood.is_finite(), "{name}: {log_likelihood}");

    let (_, likelihood) = equation.simulate_subject(&subject, &parameters, Some(&error_models))?;
    let likelihood = likelihood.expect("simulate_subject scores when given error models");
    assert_relative_eq!(likelihood.ln(), log_likelihood, max_relative = 1e-10);

    let matrix = log_likelihood_matrix(equation, &data, &support_points, &error_models, false)?;
    assert_relative_eq!(matrix[[0, 0]], log_likelihood, max_relative = 1e-10);

    let undeclared = undeclared_error_models();
    assert_unknown_label(
        &format!("{name} estimate_log_likelihood"),
        equation.estimate_log_likelihood(&subject, &parameters, &undeclared),
        "concentration",
    );
    assert_unknown_label(
        &format!("{name} simulate_subject"),
        equation
            .simulate_subject(&subject, &parameters, Some(&undeclared))
            .map(|(_, likelihood)| likelihood),
        "concentration",
    );
    assert_unknown_label(
        &format!("{name} log_likelihood_matrix"),
        log_likelihood_matrix(equation, &data, &support_points, &undeclared, false),
        "concentration",
    );

    // Predictions without error models are still available.
    let (predictions, likelihood) = equation.simulate_subject(&subject, &parameters, None)?;
    assert!(likelihood.is_none());
    drop(predictions);
    Ok(())
}

fn one_cpt_ode() -> equation::ODE {
    ode! {
        name: "one_cpt_ode",
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

fn one_cpt_analytical() -> equation::Analytical {
    analytical! {
        name: "one_cpt_analytical",
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

fn one_cpt_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = rateiv[0] - ke * x[0];
        },
        // No diffusion, so every particle follows the same path and the
        // likelihood is deterministic.
        |_p, sigma| {
            sigma[0] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
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
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("SDE metadata validates")
}

#[test]
fn handwritten_ode_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
    {
        let model = one_cpt_ode();
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("ODE", &model, parameters)
    }
}

#[test]
fn analytical_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
    {
        let model = one_cpt_analytical();
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("Analytical", &model, parameters)
    }
}

#[test]
fn sde_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
    {
        let model = one_cpt_sde();
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("SDE", &model, parameters)
    }
}

/// The SDE likelihood cache keys on the error-model hash, so changing only an
/// error polynomial must not be answered from the cache.
fn assert_sde_cache_sees_polynomial_changes(
    name: &str,
    sde: &(impl Equation + Cache),
    parameters: Parameters,
) -> Result<(), PharmsolError> {
    let subject = subject();
    let mut error_models = cp_error_models();

    let first = sde.estimate_log_likelihood(&subject, &parameters, &error_models)?;
    // The likelihood cache is on by default: a repeat call is a cache hit.
    assert_eq!(
        sde.estimate_log_likelihood(&subject, &parameters, &error_models)?,
        first,
        "{name}: repeat call"
    );

    // Change only the polynomial; gamma/lambda and its fixed state stay put.
    error_models.set_errorpoly("cp", ErrorPoly::new(5.0, 0.0, 0.0, 0.0))?;
    let after = sde.estimate_log_likelihood(&subject, &parameters, &error_models)?;
    assert!(
        (after - first).abs() > 1e-6,
        "{name}: a polynomial-only change must not reuse the cached likelihood ({first} == {after})"
    );

    // The cache only saves work: without it the likelihood is the same.
    let uncached = sde.clone().disable_cache();
    assert_eq!(
        uncached.estimate_log_likelihood(&subject, &parameters, &error_models)?,
        after,
        "{name}: cached and uncached likelihoods differ"
    );
    Ok(())
}

#[test]
fn sde_likelihood_cache_sees_error_polynomial_changes() -> Result<(), PharmsolError> {
    let model = one_cpt_sde();
    let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
    assert_sde_cache_sees_polynomial_changes("SDE", &model, parameters)
}

/// A handwritten equation without metadata declares no output names. Its
/// outputs are labeled by position number, and only labels for outputs the
/// equation has are accepted.
#[test]
fn equations_without_metadata_accept_only_existing_output_numbers() -> Result<(), PharmsolError> {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            dx[0] = rateiv[0] - p[0] * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            y[0] = x[0] / p[1];
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1);
    let subject = Subject::builder("1")
        .infusion(0.0, 100.0, 0, 1.0)
        .observation(1.0, 4.0, 0)
        .build();
    let parameters = [0.2, 20.0];
    let error_model = AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5);

    let by_position = AssayErrorModels::new().add(0, error_model.clone())?;
    assert!(ode
        .estimate_log_likelihood(&subject, &parameters, &by_position)?
        .is_finite());

    // There is no second output, and an equation without metadata has no names.
    for label in ["1", "cp"] {
        let error_models = AssayErrorModels::new().add(label, error_model.clone())?;
        assert_unknown_label(
            &format!("label `{label}`"),
            ode.estimate_log_likelihood(&subject, &parameters, &error_models),
            label,
        );
    }
    Ok(())
}

#[cfg(feature = "dsl")]
mod dsl {
    use super::*;
    use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};

    fn compile(source: &str, name: &str) -> CompiledRuntimeModel {
        compile_module_source_to_runtime(source, Some(name), |_, _| {})
            .unwrap_or_else(|error| panic!("{name} compiles: {error}"))
    }

    const ODE_SOURCE: &str = r#"
name = one_cpt_ode
kind = ode

params = ke, v
states = central
outputs = cp

dx(central) = infusion(iv) - ke * central
out(cp) = central / v ~ continuous()
"#;

    const ANALYTICAL_SOURCE: &str = r#"
name = one_cpt_analytical
kind = analytical

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

structure = one_compartment

out(cp) = central / v ~ continuous()
"#;

    // No diffusion, so the likelihood is deterministic.
    const SDE_SOURCE: &str = r#"
name = one_cpt_sde
kind = sde

params = ke, v
states = central
particles = 16
outputs = cp


dx(central) = -ke * central + infusion(iv)
noise(central) = 0.0

out(cp) = central / v ~ continuous()
"#;

    #[test]
    fn dsl_ode_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Ode(model) = compile(ODE_SOURCE, "one_cpt_ode") else {
            panic!("expected an ODE model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("DSL ODE", &model, parameters)
    }

    #[test]
    fn dsl_analytical_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Analytical(model) =
            compile(ANALYTICAL_SOURCE, "one_cpt_analytical")
        else {
            panic!("expected an analytical model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("DSL analytical", &model, parameters)
    }

    #[test]
    fn dsl_sde_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Sde(model) = compile(SDE_SOURCE, "one_cpt_sde") else {
            panic!("expected an SDE model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("DSL SDE", &model, parameters)
    }

    #[test]
    fn dsl_sde_likelihood_cache_sees_error_polynomial_changes() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Sde(model) = compile(SDE_SOURCE, "one_cpt_sde") else {
            panic!("expected an SDE model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_sde_cache_sees_polynomial_changes("DSL SDE", &model, parameters)
    }
}
