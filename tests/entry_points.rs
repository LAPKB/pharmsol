//! Every way of simulating a subject gives the same predictions, and the SDE
//! particle filter applies the same up-front label check on each of its entry
//! points.
//!
//! `predict`, `predict_shared` and `map_predictions` must agree for every
//! backend. The SDE particle filter looks its observation model up by output
//! label; a model for a label the equation does not declare is rejected
//! before simulating, so a typo cannot silently drop an output.

#[path = "support/normal_observation.rs"]
mod normal_observation;

use approx::assert_relative_eq;
use ndarray::Array2;
use normal_observation::NormalObservation;
use pharmsol::prelude::simulator::map_predictions;
use pharmsol::prelude::*;
use pharmsol::simulator::equation::sde::SdeModel;
use pharmsol::{Cache, SDE};

/// One-compartment IV model data: an infusion on `iv`, two `cp` observations.
fn subject() -> Subject {
    Subject::builder("1")
        .infusion(0.0, 100.0, "iv", 1.0)
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .build()
}

const PARAMETERS: [(&str, f64); 2] = [("ke", 0.2), ("v", 20.0)];

fn prediction_values(predictions: &impl PointPredictions) -> Vec<f64> {
    predictions
        .point()
        .iter()
        .map(|prediction| prediction.prediction())
        .collect()
}

fn assert_unknown_label<T: std::fmt::Debug>(
    entry_point: &str,
    result: Result<T, PharmsolError>,
    expected: &str,
) {
    match result {
        Err(PharmsolError::UnknownOutputLabel { label, .. }) if label == expected => {}
        other => panic!("{entry_point}: expected UnknownOutputLabel(`{expected}`), got {other:?}"),
    }
}

/// Simulate the same subject through each public entry point and check that
/// they agree.
fn assert_entry_points_agree(
    name: &str,
    equation: &impl Equation,
    parameters: Parameters,
) -> Result<(), PharmsolError> {
    let subject = subject();
    let data = Data::new(vec![subject.clone()]);
    let support_points =
        Array2::from_shape_vec((1, 2), parameters.as_slice().to_vec()).expect("one support point");

    let predicted = prediction_values(&equation.predict(&subject, &parameters)?);
    assert_eq!(predicted.len(), 2, "{name}: one prediction per observation");
    assert!(
        predicted
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "{name}: {predicted:?}"
    );

    let shared = prediction_values(equation.predict_shared(&subject, &parameters)?.as_ref());
    assert_eq!(shared, predicted, "{name}: predict_shared");

    let mapped = map_predictions(equation, &data, &support_points, |_, predictions| {
        predictions.map(|predictions| prediction_values(predictions.as_ref()))
    })?;
    assert_eq!(mapped.dim(), (1, 1));
    let mapped = mapped[[0, 0]].as_ref().map_err(Clone::clone)?;
    assert_eq!(mapped, &predicted, "{name}: map_predictions");
    Ok(())
}

/// Score the same subject through each SDE filter entry point and check
/// that they agree, then check that each one rejects an undeclared label.
fn assert_filter_entry_points_agree<M: SdeModel>(
    name: &str,
    sde: &SDE<M>,
    parameters: Parameters,
) -> Result<(), PharmsolError> {
    let subject = subject();
    let observation_model = NormalObservation::new("cp", 0.5);

    let log_likelihood = sde.log_likelihood(&subject, &parameters, &observation_model)?;
    assert!(log_likelihood.is_finite(), "{name}: {log_likelihood}");

    let filtered = sde.filter(&subject, &parameters, &observation_model)?;
    assert_relative_eq!(
        filtered.log_likelihood,
        log_likelihood,
        max_relative = 1e-10
    );
    assert!(filtered.predictions.is_filtered());

    // Without diffusion every particle follows the free-running path, so the
    // filter's estimate is the plain log-likelihood of the predictions.
    let scored = observation_model.score(&sde.predict(&subject, &parameters)?)?;
    assert_relative_eq!(scored, log_likelihood, max_relative = 1e-10);

    let undeclared = NormalObservation::new("concentration", 0.5);
    assert_unknown_label(
        &format!("{name} log_likelihood"),
        sde.log_likelihood(&subject, &parameters, &undeclared),
        "concentration",
    );
    assert_unknown_label(
        &format!("{name} filter"),
        sde.filter(&subject, &parameters, &undeclared)
            .map(|filtered| filtered.log_likelihood),
        "concentration",
    );
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
fn handwritten_ode_entry_points_agree() -> Result<(), PharmsolError> {
    let model = one_cpt_ode();
    let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
    assert_entry_points_agree("ODE", &model, parameters)
}

#[test]
fn analytical_entry_points_agree() -> Result<(), PharmsolError> {
    let model = one_cpt_analytical();
    let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
    assert_entry_points_agree("Analytical", &model, parameters)
}

#[test]
fn sde_entry_points_agree_and_check_labels() -> Result<(), PharmsolError> {
    let model = one_cpt_sde();
    let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
    assert_entry_points_agree("SDE", &model, parameters.clone())?;
    assert_filter_entry_points_agree("SDE", &model, parameters)
}

/// The SDE likelihood cache keys on the observation model's cache key, so
/// changing only the model must not be answered from the cache.
fn assert_sde_cache_sees_observation_model_changes<M: SdeModel>(
    name: &str,
    sde: &SDE<M>,
    parameters: Parameters,
) -> Result<(), PharmsolError> {
    let subject = subject();
    let observation_model = NormalObservation::new("cp", 0.5);

    let first = sde.log_likelihood(&subject, &parameters, &observation_model)?;
    // The likelihood cache is on by default: a repeat call is a cache hit.
    assert_eq!(
        sde.log_likelihood(&subject, &parameters, &observation_model)?,
        first,
        "{name}: repeat call"
    );

    // Change only sigma.
    let changed = observation_model.with_sigma(5.0);
    let after = sde.log_likelihood(&subject, &parameters, &changed)?;
    assert!(
        (after - first).abs() > 1e-6,
        "{name}: a changed observation model must not reuse the cached likelihood ({first} == {after})"
    );

    // The cache only saves work: without it the likelihood is the same.
    let uncached = sde.clone().disable_cache();
    assert_eq!(
        uncached.log_likelihood(&subject, &parameters, &changed)?,
        after,
        "{name}: cached and uncached likelihoods differ"
    );
    Ok(())
}

#[test]
fn sde_likelihood_cache_sees_observation_model_changes() -> Result<(), PharmsolError> {
    let model = one_cpt_sde();
    let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
    assert_sde_cache_sees_observation_model_changes("SDE", &model, parameters)
}

/// A handwritten equation without metadata declares no output names. Its
/// outputs are labeled by position number, and only labels for outputs the
/// equation has are accepted by the particle filter.
#[test]
fn equations_without_metadata_accept_only_existing_output_numbers() -> Result<(), PharmsolError> {
    let sde = equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            dx[0] = rateiv[0] - p[0] * x[0];
        },
        |_p, sigma| {
            sigma[0] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            y[0] = x[0] / p[1];
        },
        16,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1);
    let subject = Subject::builder("1")
        .infusion(0.0, 100.0, 0, 1.0)
        .observation(1.0, 4.0, 0)
        .build();
    let parameters = [0.2, 20.0];

    let by_position = NormalObservation::new(0, 0.5);
    assert!(sde
        .log_likelihood(&subject, &parameters, &by_position)?
        .is_finite());

    // There is no second output, and an equation without metadata has no names.
    for label in ["1", "cp"] {
        assert_unknown_label(
            &format!("label `{label}`"),
            sde.log_likelihood(&subject, &parameters, &NormalObservation::new(label, 0.5)),
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
    fn dsl_ode_entry_points_agree() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Ode(model) = compile(ODE_SOURCE, "one_cpt_ode") else {
            panic!("expected an ODE model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_entry_points_agree("DSL ODE", &model, parameters)
    }

    #[test]
    fn dsl_analytical_entry_points_agree() -> Result<(), PharmsolError> {
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
        assert_entry_points_agree("DSL SDE", &model, parameters.clone())?;
        assert_filter_entry_points_agree("DSL SDE", &model, parameters)
    }

    #[test]
    fn dsl_sde_likelihood_cache_sees_observation_model_changes() -> Result<(), PharmsolError> {
        let CompiledRuntimeModel::Sde(model) = compile(SDE_SOURCE, "one_cpt_sde") else {
            panic!("expected an SDE model");
        };
        let parameters = Parameters::with_model(&model, PARAMETERS).expect("valid parameters");
        assert_sde_cache_sees_observation_model_changes("DSL SDE", &model, parameters)
    }
}
