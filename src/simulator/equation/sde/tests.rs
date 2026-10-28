use super::*;
use crate::simulator::equation::{self, Covariate, Route};
use crate::simulator::equation::{AttachMetadataError, ModelMetadataError};
use crate::simulator::Drift;
use crate::SubjectBuilderExt;
use crate::{fa, fetch_params, lag};
use crate::{OutputModels, ResidualErrorModel};
use pharmsol_dsl::ModelKind;

fn simple_sde() -> SDE {
    let drift = |x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
        dx[0] = rateiv[0] - x[0];
    };
    let diffusion = |_p: &V, g: &mut V| {
        g[0] = 1.0;
    };
    let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
    let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x[0] = 0.0;
    };
    let out = |x: &V, p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        fetch_params!(p, _ke, v);
        y[0] = x[0] / v;
    };

    SDE::new(drift, diffusion, lag, fa, init, out, 128)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
}

fn route_policy_sde(drift: Drift) -> SDE {
    let diffusion = |_p: &V, sigma: &mut V| {
        sigma.fill(0.0);
    };
    let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
    let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[1];
    };

    SDE::new(drift, diffusion, lag, fa, init, out, 16)
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
}

#[test]
fn handwritten_sde_metadata_exposes_name_lookup_and_particles() {
    let sde = simple_sde()
        .with_metadata(
            equation::metadata::new("one_cmt_sde")
                .parameters(["ke", "v"])
                .covariates([Covariate::continuous("wt")])
                .states(["central"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(128),
        )
        .expect("SDE metadata attachment should validate");

    let metadata = sde.metadata().expect("metadata exists");
    assert_eq!(metadata.kind(), ModelKind::Sde);
    assert_eq!(metadata.particles(), Some(128));
    assert_eq!(sde.parameter_index("ke"), Some(0));
    assert_eq!(sde.parameter_index("v"), Some(1));
    assert_eq!(sde.covariate_index("wt"), Some(0));
    assert_eq!(sde.state_index("central"), Some(0));
    assert!(metadata.route("iv").is_some());
    assert!(metadata.output("cp").is_some());
}

#[test]
fn handwritten_sde_metadata_resolves_numeric_labels_by_exact_name() {
    let drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
        dx.fill(0.0);
        dx[1] = rateiv[0];
    };
    let diffusion = |_p: &V, sigma: &mut V| {
        sigma.fill(0.0);
    };
    let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
    let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[1];
    };

    let sde = SDE::new(drift, diffusion, lag, fa, init, out, 16)
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            equation::metadata::new("numeric_alias_sde")
                .states(["depot", "central"])
                .outputs(["1"])
                .route(Route::infusion("1").to_state("central"))
                .particles(16),
        )
        .expect("SDE metadata attachment should validate");

    let numeric = Subject::builder("numeric")
        .infusion(0.0, 100.0, "1", 1.0)
        .observation(1.0, 0.0, "1")
        .build();
    let prediction = sde
        .predict(&numeric, &crate::parameters::dense([]))
        .expect("numeric labels should resolve by exact name")
        .values()[[0, 0]];
    assert!(prediction > 0.0);

    // `input_1` / `outeq_1` are different names, not aliases of `1`.
    for prefixed in [
        Subject::builder("prefixed_input")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "1")
            .build(),
        Subject::builder("prefixed_output")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build(),
    ] {
        assert!(sde
            .predict(&prefixed, &crate::parameters::dense([]))
            .is_err());
    }
}

#[test]
fn handwritten_sde_without_metadata_keeps_raw_path() {
    let sde = simple_sde();

    assert!(sde.metadata().is_none());
    assert_eq!(sde.parameter_index("ke"), None);
}

#[test]
fn handwritten_sde_rejects_dimension_mismatches() {
    let error = simple_sde()
        .with_metadata(
            equation::metadata::new("bad_sde")
                .parameters(["ke", "v"])
                .states(["central", "peripheral"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(128),
        )
        .expect_err("mismatched state metadata must fail");

    assert_eq!(
        error,
        AttachMetadataError::StateCountMismatch {
            expected: 1,
            declared: 2,
        }
    );
}

#[test]
fn handwritten_sde_rejects_particle_mismatch() {
    let error = simple_sde()
        .with_metadata(
            equation::metadata::new("particle_conflict")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(64),
        )
        .expect_err("mismatched SDE particles must fail");

    assert_eq!(
        error,
        AttachMetadataError::Validation(ModelMetadataError::ParticleCountConflict {
            declared: 64,
            fallback: 128,
        })
    );
}

#[test]
fn changing_dimensions_after_metadata_clears_sde_metadata() {
    let sde = simple_sde()
        .with_metadata(
            equation::metadata::new("one_cmt_sde")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(128),
        )
        .expect("metadata attachment should validate")
        .with_nout(2);

    assert!(sde.metadata().is_none());
}

#[test]
fn sde_metadata_input_policy_is_descriptive_only_for_bolus_routes() {
    let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
        dx.fill(0.0);
    };

    let explicit = route_policy_sde(zero_drift)
        .with_metadata(
            equation::metadata::new("explicit_bolus")
                .parameters(["theta"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(Route::bolus("oral").to_state("central"))
                .particles(16),
        )
        .expect("explicit metadata should validate");

    let injected = route_policy_sde(zero_drift)
        .with_metadata(
            equation::metadata::new("injected_bolus")
                .parameters(["theta"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(
                    Route::bolus("oral")
                        .to_state("central")
                        .inject_input_to_destination(),
                )
                .particles(16),
        )
        .expect("injected metadata should validate");

    let subject = Subject::builder("bolus_route")
        .bolus(0.0, 100.0, "oral")
        .missing_observation(0.1, "cp")
        .build();

    let explicit_predictions = explicit
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();
    let injected_predictions = injected
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    assert_eq!(explicit_predictions.values()[[0, 0]], 0.0);
    assert_eq!(injected_predictions.values()[[0, 0]], 0.0);
}

#[test]
fn sde_metadata_input_policy_does_not_change_explicit_rateiv_behavior() {
    let rateiv_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
        dx.fill(0.0);
        dx[1] = rateiv[0];
    };

    let explicit = route_policy_sde(rateiv_drift)
        .with_metadata(
            equation::metadata::new("explicit_infusion")
                .parameters(["theta"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(16),
        )
        .expect("explicit metadata should validate");

    let injected = route_policy_sde(rateiv_drift)
        .with_metadata(
            equation::metadata::new("injected_infusion")
                .parameters(["theta"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(
                    Route::infusion("iv")
                        .to_state("central")
                        .inject_input_to_destination(),
                )
                .particles(16),
        )
        .expect("injected metadata should validate");

    let subject = Subject::builder("infusion_route")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(1.0, "cp")
        .build();

    let explicit_predictions = explicit
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();
    let injected_predictions = injected
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    let explicit_prediction = explicit_predictions.values()[[0, 0]];
    let injected_prediction = injected_predictions.values()[[0, 0]];

    assert!(explicit_prediction > 0.0);
    assert!((injected_prediction - explicit_prediction).abs() < 1e-8);
}

#[test]
fn clearing_sde_metadata_preserves_raw_bolus_behavior() {
    let zero_drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, _rateiv: &V, _cov: &Covariates| {
        dx.fill(0.0);
    };

    let sde = route_policy_sde(zero_drift)
        .with_metadata(
            equation::metadata::new("injected_bolus")
                .parameters(["theta"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(
                    Route::bolus("oral")
                        .to_state("central")
                        .inject_input_to_destination(),
                )
                .particles(16),
        )
        .expect("injected metadata should validate")
        .with_nout(1);

    let subject = Subject::builder("bolus_route")
        .bolus(0.0, 100.0, 0)
        .missing_observation(0.1, 0)
        .build();

    let predictions = sde
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    assert!(sde.metadata().is_none());
    assert_eq!(predictions.values()[[0, 0]], 0.0);
}

fn values(predictions: &ParticlePredictions) -> Vec<f64> {
    predictions.values().iter().copied().collect()
}

fn filter_subject() -> Subject {
    Subject::builder("filter")
        .infusion(0.0, 100.0, 0, 1.0)
        .observation(0.5, 0.8, 0)
        .observation(1.0, 1.2, 0)
        .observation(2.0, 0.9, 0)
        .build()
}

fn filter_error_models() -> OutputModels {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    OutputModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.5, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

#[test]
fn particle_filter_is_deterministic() {
    let sde = simple_sde().disable_cache();
    let subject = filter_subject();
    let error_models = filter_error_models();
    let parameters = [1.0, 50.0];

    let first = sde
        .log_likelihood(&subject, &parameters, &error_models)
        .unwrap();
    let second = sde
        .log_likelihood(&subject, &parameters, &error_models)
        .unwrap();
    assert!(first.is_finite());
    assert_eq!(first, second);

    let predictions = sde.predict(&subject, &parameters).unwrap();
    let again = sde.predict(&subject, &parameters).unwrap();
    assert_eq!(predictions.dim(), (128, 3));
    assert_eq!(values(&predictions), values(&again));
}

#[test]
fn likelihood_matrix_uses_the_particle_filter() {
    use crate::simulator::likelihood::log_likelihood_matrix;
    use crate::Data;

    let sde = simple_sde();
    let subject = filter_subject();
    let error_models = filter_error_models();

    let direct = sde
        .log_likelihood(&subject, &[1.0, 50.0], &error_models)
        .unwrap();
    let matrix = log_likelihood_matrix(
        &sde.clone().disable_cache(),
        &Data::new(vec![subject]),
        &ndarray::array![[1.0, 50.0]],
        &error_models,
        false,
    )
    .unwrap();
    assert_eq!(matrix[[0, 0]], direct);
}

#[test]
fn seed_selects_independent_particle_paths() {
    let subject = filter_subject();
    let parameters = [1.0, 50.0];
    let default_seed = simple_sde().predict(&subject, &parameters).unwrap();
    let other_seed = simple_sde()
        .with_seed(7)
        .predict(&subject, &parameters)
        .unwrap();
    assert_ne!(values(&default_seed), values(&other_seed));
}

#[test]
fn particle_filter_preserves_finite_tail_log_likelihoods() {
    use crate::data::event::Observation;
    use crate::Censor;
    let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
    let values = vec![0.0; 16];
    let error_models = filter_error_models();
    let expected = error_models
        .log_density(observation.outeq(), 20.0, 0.0, Censor::None, None)
        .unwrap();
    assert!(expected.is_finite());
    assert_eq!(expected.exp(), 0.0);

    let filtered = filter(
        &observation,
        &values,
        &error_models,
        &mut StdRng::seed_from_u64(0),
    )
    .unwrap()
    .expect("finite particle log-likelihoods must not become zero weights");
    assert_eq!(filtered.log_mean, expected);
    assert_eq!(filtered.indices, (0..values.len()).collect::<Vec<_>>());
}

#[test]
fn far_observation_has_finite_log_likelihood() {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    let sde = simple_sde();
    let subject = Subject::builder("far").observation(1.0, 1.0e6, 0).build();
    let error_models = OutputModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(1.0e-3, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let log_likelihood = sde
        .log_likelihood(&subject, &[1.0, 50.0], &error_models)
        .unwrap();
    assert!(log_likelihood.is_finite());
}

#[test]
fn simulate_subject_returns_one_filtered_run() {
    let sde = simple_sde().disable_cache();
    let subject = filter_subject();
    let error_models = filter_error_models();
    let parameters = [1.0, 50.0];

    let simulation = sde
        .simulate(&subject, &parameters, Some(&error_models))
        .unwrap();
    let log_likelihood = sde
        .log_likelihood(&subject, &parameters, &error_models)
        .unwrap();
    let predictions = simulation.predictions;
    assert_eq!(predictions.dim(), (128, 3));
    assert_eq!(simulation.log_likelihood, Some(log_likelihood));

    // The particles are the filtered ones: identical to an unfiltered run up
    // to the first resampling, different after it.
    let unfiltered = sde.predict(&subject, &parameters).unwrap();
    assert!(predictions.is_filtered());
    assert!(!unfiltered.is_filtered());
    let column = |m: &ParticlePredictions, j: usize| -> Vec<f64> { m.values().column(j).to_vec() };
    assert_eq!(column(&predictions, 0), column(&unfiltered, 0));
    assert_ne!(column(&predictions, 2), column(&unfiltered, 2));

    // Each observation's weights are normalized.
    let weights = predictions.weights().unwrap();
    for column in weights.columns() {
        assert!((column.sum() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn particles_the_data_rules_out_get_zero_weight() {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    let sde = simple_sde().disable_cache();
    // So far from the particles that every per-particle log-density is -inf.
    let subject = Subject::builder("far")
        .observation(1.0, 1.0e6, 0)
        .observation(2.0, 1.0, 0)
        .build();
    let error_models = OutputModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(1.0e-200, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let log_likelihood = sde
        .log_likelihood(&subject, &[1.0, 50.0], &error_models)
        .unwrap();
    assert_eq!(log_likelihood, f64::NEG_INFINITY);

    // Predictions still cover every observation.
    let simulation = sde
        .simulate(&subject, &[1.0, 50.0], Some(&error_models))
        .unwrap();
    assert_eq!(simulation.predictions.dim(), (128, 2));
    assert_eq!(simulation.log_likelihood, Some(f64::NEG_INFINITY));
}

fn residual_error_models() -> OutputModels {
    OutputModels::new()
        .add(0, ResidualErrorModel::combined(0.2, 0.1))
        .unwrap()
}

/// An infusion with no elimination and no diffusion: every particle follows
/// the ODE, and Euler-Maruyama integrates its piecewise-linear solution
/// exactly.
fn deterministic_sde() -> SDE {
    let drift = |_x: &V, _p: &V, _t: f64, dx: &mut V, rateiv: &V, _cov: &Covariates| {
        dx[0] = rateiv[0];
    };
    let diffusion = |_p: &V, g: &mut V| {
        g[0] = 0.0;
    };
    let lag = |_p: &V, _t: f64, _cov: &Covariates| lag! {};
    let fa = |_p: &V, _t: f64, _cov: &Covariates| fa! {};
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x[0] = 0.0;
    };
    let out = |x: &V, p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        fetch_params!(p, _ke, v);
        y[0] = x[0] / v;
    };

    SDE::new(drift, diffusion, lag, fa, init, out, 32)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
}

/// The ODE that [`deterministic_sde`] discretizes.
fn equivalent_ode() -> crate::ODE {
    crate::ODE::new(
        |_x, _p, _t, dx, _b, rateiv, _cov| {
            dx[0] = rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
}

#[test]
fn residual_error_likelihood_is_a_deterministic_particle_filter() {
    let sde = simple_sde().disable_cache();
    let subject = filter_subject();
    let residual = residual_error_models();
    let parameters = [1.0, 50.0];

    let first = sde
        .log_likelihood(&subject, &parameters, &residual)
        .unwrap();
    let second = sde
        .log_likelihood(&subject, &parameters, &residual)
        .unwrap();
    assert!(first.is_finite());
    assert_eq!(first, second);

    // Another seed draws other paths, so another estimate.
    let reseeded = sde
        .clone()
        .with_seed(7)
        .log_likelihood(&subject, &parameters, &residual)
        .unwrap();
    assert_ne!(first, reseeded);

    // It is the filter's estimate, not the score of the mean particle.
    let mean_particle = sde
        .predict(&subject, &parameters)
        .unwrap()
        .point_summary()
        .log_likelihood(&residual)
        .unwrap();
    assert_ne!(first, mean_particle);

    // `simulate` runs the same filter.
    let simulation = sde
        .simulate(&subject, &parameters, Some(&residual))
        .unwrap();
    assert_eq!(simulation.log_likelihood, Some(first));
}

#[test]
fn residual_error_routes_through_the_particle_filter() {
    use crate::simulator::likelihood::{log_likelihood_batch, log_likelihood_subject};
    use crate::Data;

    let sde = simple_sde();
    let subject = filter_subject();
    let residual = residual_error_models();
    let parameters = [1.0, 50.0];
    let direct = sde
        .clone()
        .disable_cache()
        .log_likelihood(&subject, &parameters, &residual)
        .unwrap();

    let batch = log_likelihood_batch(
        &sde,
        &Data::new(vec![subject.clone()]),
        &ndarray::array![[1.0, 50.0]],
        &residual,
    )
    .unwrap();
    assert_eq!(batch, vec![direct]);

    let single = log_likelihood_subject(
        &sde,
        &subject,
        &crate::parameters::dense(parameters),
        &residual,
    );
    assert_eq!(single, direct);

    // The cache keys residual and assay error apart.
    let assay = filter_error_models();
    let cached_assay = sde.log_likelihood(&subject, &parameters, &assay).unwrap();
    let uncached_assay = sde
        .clone()
        .disable_cache()
        .log_likelihood(&subject, &parameters, &assay)
        .unwrap();
    assert_eq!(cached_assay, uncached_assay);
    assert_ne!(cached_assay, direct);
}

#[test]
fn zero_diffusion_likelihood_matches_the_ode() {
    let sde = deterministic_sde().disable_cache();
    let ode = equivalent_ode();
    // The infusion outlasts the observations, so the solution stays linear.
    let subject = Subject::builder("linear")
        .infusion(0.0, 300.0, 0, 3.0)
        .observation(0.5, 0.9, 0)
        .observation(1.0, 2.1, 0)
        .observation(2.0, 3.8, 0)
        .build();
    let parameters = [1.0, 50.0];

    let residual = residual_error_models();
    let assay = filter_error_models();
    let observation_models: [&dyn ObservationModel; 2] = [&residual, &assay];
    for observation_model in observation_models {
        let sde_log_likelihood = sde
            .log_likelihood(&subject, &parameters, observation_model)
            .unwrap();

        // Identical particles: the filter's estimate is exactly the score of
        // their (shared) prediction.
        let particles = sde.predict(&subject, &parameters).unwrap();
        let first = particles.values().row(0).to_owned();
        assert!(particles
            .values()
            .rows()
            .into_iter()
            .all(|row| row == first));
        let scored = particles
            .point_summary()
            .log_likelihood(observation_model)
            .unwrap();
        assert!((sde_log_likelihood - scored).abs() < 1e-12);

        // And that prediction is the ODE's.
        let ode_log_likelihood = ode
            .log_likelihood(&subject, &parameters, observation_model)
            .unwrap();
        assert!(
            (sde_log_likelihood - ode_log_likelihood).abs() < 1e-8,
            "SDE {sde_log_likelihood} vs ODE {ode_log_likelihood}"
        );
    }
}

#[test]
fn heavy_tails_score_an_outlier_above_a_normal() {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    use crate::{Family, OutputModel};

    let sde = simple_sde().disable_cache();
    let subject = Subject::builder("outlier")
        .observation(1.0, 50.0, 0)
        .build();
    let assay = AssayErrorModel::additive(ErrorPoly::new(1.0e-3, 0.0, 0.0, 0.0), 0.0);

    // Under a normal, an observation this far from every particle is
    // finite only because the filter works in log space.
    let normal = OutputModels::new().add(0, assay.clone()).unwrap();
    let normal_ll = sde.log_likelihood(&subject, &[1.0, 50.0], &normal).unwrap();
    assert!(normal_ll.is_finite());

    // A t-distribution's polynomial tails still weigh the particles.
    let student = OutputModels::new()
        .add(
            0,
            OutputModel::new(assay).with_family(Family::student_t(3.0).unwrap()),
        )
        .unwrap();
    let student_ll = sde
        .log_likelihood(&subject, &[1.0, 50.0], &student)
        .unwrap();
    assert!(student_ll.is_finite());
    assert!(student_ll > normal_ll);

    let simulation = sde
        .simulate(&subject, &[1.0, 50.0], Some(&student))
        .unwrap();
    assert_eq!(simulation.log_likelihood, Some(student_ll));
    let weights = simulation.predictions.weights().unwrap();
    assert!((weights.column(0).sum() - 1.0).abs() < 1e-12);
}
