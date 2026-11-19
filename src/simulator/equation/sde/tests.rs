use super::test_observation::NormalObservation;
use super::*;
use crate::data::event::OutputLabel;
use crate::simulator::equation::{self, Covariate, Route};
use crate::simulator::equation::{AttachMetadataError, ModelMetadataError};
use crate::simulator::Drift;
use crate::SubjectBuilderExt;
use crate::{fa, fetch_params, lag};
use crate::{Censor, ErrorPoly};
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

fn filter_observation_model() -> NormalObservation {
    NormalObservation::new(0, 0.5)
}

#[test]
fn particle_filter_is_deterministic() {
    let sde = simple_sde().disable_cache();
    let subject = filter_subject();
    let observation_model = filter_observation_model();
    let parameters = [1.0, 50.0];

    let first = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    let second = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    assert!(first.is_finite());
    assert_eq!(first, second);

    // Another seed draws other paths, so another estimate.
    let reseeded = sde
        .clone()
        .with_seed(7)
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    assert_ne!(first, reseeded);

    let predictions = sde.predict(&subject, &parameters).unwrap();
    let again = sde.predict(&subject, &parameters).unwrap();
    assert_eq!(predictions.dim(), (128, 3));
    assert_eq!(values(&predictions), values(&again));
}

#[test]
fn predict_shared_is_the_free_running_run() {
    let sde = simple_sde();
    let subject = filter_subject();
    let parameters = [1.0, 50.0];

    let shared = sde.predict_shared(&subject, &parameters).unwrap();
    let owned = sde.predict(&subject, &parameters).unwrap();
    assert!(!shared.is_filtered());
    assert_eq!(values(&shared), values(&owned));
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
    let observation = Observation::new(0.0, Some(20.0), 0, None, 0, Censor::None);
    let values = vec![0.0; 16];
    let observation_model = filter_observation_model();
    let expected = observation_model
        .log_density(observation.outeq(), 20.0, 0.0, Censor::None, None)
        .unwrap();
    assert!(expected.is_finite());
    assert_eq!(expected.exp(), 0.0);

    let filtered = weigh_and_resample(
        &observation,
        &values,
        &observation_model,
        &mut StdRng::seed_from_u64(0),
    )
    .unwrap()
    .expect("finite particle log-likelihoods must not become zero weights");
    assert_eq!(filtered.log_mean, expected);
    assert_eq!(filtered.indices, (0..values.len()).collect::<Vec<_>>());
}

/// Returns the prediction itself as the log-density, so a test can hand the
/// filter `Ok(NaN)` or `Ok(±inf)` without going through a real density.
struct RawDensity(Vec<OutputLabel>);

impl ObservationModel for RawDensity {
    fn log_density(
        &self,
        _outeq: &OutputLabel,
        _observation: f64,
        prediction: f64,
        _censoring: Censor,
        _errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        Ok(prediction)
    }

    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_> {
        Box::new(self.0.iter())
    }
}

#[test]
fn non_finite_densities_returned_as_ok_get_zero_weight() {
    use crate::data::event::Observation;
    let observation = Observation::new(0.0, Some(1.0), 0, None, 0, Censor::None);
    let model = RawDensity(vec![OutputLabel::from(0)]);
    let values = [f64::NAN, -1.0, f64::INFINITY, f64::NEG_INFINITY];

    let filtered = weigh_and_resample(&observation, &values, &model, &mut StdRng::seed_from_u64(0))
        .unwrap()
        .expect("one particle has a finite density");
    assert_eq!(filtered.weights, vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(filtered.indices, vec![1; values.len()]);
    assert_eq!(filtered.log_mean, -1.0 + (0.25f64).ln());

    let all_non_finite = [f64::NAN, f64::INFINITY];
    assert!(weigh_and_resample(
        &observation,
        &all_non_finite,
        &model,
        &mut StdRng::seed_from_u64(0),
    )
    .unwrap()
    .is_none());
}

#[test]
fn far_observation_has_finite_log_likelihood() {
    let sde = simple_sde();
    let subject = Subject::builder("far").observation(1.0, 1.0e6, 0).build();
    let observation_model = NormalObservation::new(0, 1.0e-3);

    let log_likelihood = sde
        .log_likelihood(&subject, &[1.0, 50.0], &observation_model)
        .unwrap();
    assert!(log_likelihood.is_finite());
}

#[test]
fn filter_returns_one_filtered_run() {
    let sde = simple_sde().disable_cache();
    let subject = filter_subject();
    let observation_model = filter_observation_model();
    let parameters = [1.0, 50.0];

    let filtered = sde
        .filter(&subject, &parameters, &observation_model)
        .unwrap();
    let log_likelihood = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    let predictions = filtered.predictions;
    assert_eq!(predictions.dim(), (128, 3));
    assert_eq!(filtered.log_likelihood, log_likelihood);

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

    // It is the filter's estimate, not the score of the mean particle.
    let mean_particle = observation_model
        .score(&unfiltered.point_summary())
        .unwrap();
    assert_ne!(log_likelihood, mean_particle);
}

#[test]
fn particles_the_data_rules_out_get_zero_weight() {
    let sde = simple_sde().disable_cache();
    // So far from the particles that every per-particle log-density is -inf.
    let subject = Subject::builder("far")
        .observation(1.0, 1.0e6, 0)
        .observation(2.0, 1.0, 0)
        .build();
    let observation_model = NormalObservation::new(0, 1.0e-200);

    let log_likelihood = sde
        .log_likelihood(&subject, &[1.0, 50.0], &observation_model)
        .unwrap();
    assert_eq!(log_likelihood, f64::NEG_INFINITY);

    // Predictions still cover every observation, with uniform weights once
    // nothing is filtered any more.
    let filtered = sde
        .filter(&subject, &[1.0, 50.0], &observation_model)
        .unwrap();
    assert_eq!(filtered.predictions.dim(), (128, 2));
    assert_eq!(filtered.log_likelihood, f64::NEG_INFINITY);
    let weights = filtered.predictions.weights().unwrap();
    assert!(weights.iter().all(|&weight| weight == 1.0 / 128.0));
}

#[test]
fn censored_observations_are_scored_by_the_observation_model() {
    let sde = simple_sde().disable_cache();
    let observation_model = filter_observation_model();
    let parameters = [1.0, 50.0];
    // The particles are near zero at t = 1, far below a limit of 5.
    let log_likelihood = |censoring| {
        let subject = Subject::builder("censored")
            .infusion(0.0, 100.0, 0, 1.0)
            .censored_observation(1.0, 5.0, 0, censoring)
            .build();
        sde.log_likelihood(&subject, &parameters, &observation_model)
            .unwrap()
    };
    let below = log_likelihood(Censor::BLOQ);
    let above = log_likelihood(Censor::ALOQ);
    let measured = log_likelihood(Censor::None);
    assert!(below > above);
    assert!(below > measured);
    assert!(below <= 0.0);
}

#[test]
fn the_cache_keys_observation_models_apart() {
    let sde = simple_sde();
    let subject = filter_subject();
    let parameters = [1.0, 50.0];
    let narrow = filter_observation_model();
    let wide = NormalObservation::new(0, 2.0);

    let uncached_narrow = sde
        .clone()
        .disable_cache()
        .log_likelihood(&subject, &parameters, &narrow)
        .unwrap();
    let cached_narrow = sde.log_likelihood(&subject, &parameters, &narrow).unwrap();
    let cached_wide = sde.log_likelihood(&subject, &parameters, &wide).unwrap();
    let uncached_wide = sde
        .clone()
        .disable_cache()
        .log_likelihood(&subject, &parameters, &wide)
        .unwrap();

    assert_eq!(cached_narrow, uncached_narrow);
    assert_eq!(cached_wide, uncached_wide);
    assert_ne!(cached_narrow, cached_wide);
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 2);
}

#[test]
fn models_without_a_cache_key_are_never_cached() {
    let sde = simple_sde();
    let subject = filter_subject();
    let parameters = [1.0, 50.0];
    let observation_model = filter_observation_model().uncached();
    assert_eq!(observation_model.cache_key(), None);

    let first = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    let second = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 0);

    // A keyed model with the same densities fills the cache.
    sde.log_likelihood(&subject, &parameters, &filter_observation_model())
        .unwrap();
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 1);
}

#[test]
fn observation_models_are_checked_before_simulating() {
    let sde = simple_sde()
        .with_metadata(
            equation::metadata::new("one_cmt_sde")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .route(Route::infusion("iv").to_state("central"))
                .particles(128),
        )
        .expect("metadata attachment should validate");
    let subject = Subject::builder("checked")
        .infusion(0.0, 100.0, "iv", 1.0)
        .observation(1.0, 1.0, "cp")
        .build();
    let parameters = [1.0, 50.0];

    assert!(sde
        .log_likelihood(&subject, &parameters, &NormalObservation::new("cp", 0.5))
        .is_ok());

    let unknown = NormalObservation::new("not_an_output", 0.5);
    for result in [
        sde.log_likelihood(&subject, &parameters, &unknown),
        sde.filter(&subject, &parameters, &unknown)
            .map(|filtered| filtered.log_likelihood),
    ] {
        assert!(matches!(
            result,
            Err(PharmsolError::UnknownOutputLabel { ref label, .. }) if label == "not_an_output"
        ));
    }

    // Without metadata, labels are dense output indices.
    let raw = simple_sde();
    let raw_subject = filter_subject();
    assert!(raw
        .log_likelihood(&raw_subject, &parameters, &NormalObservation::new(1, 0.5))
        .is_err());

    // The model's own settings are validated too.
    let invalid = NormalObservation::new("cp", -1.0);
    assert!(matches!(
        sde.log_likelihood(&subject, &parameters, &invalid),
        Err(PharmsolError::OtherError(_))
    ));

    // Trait objects work as observation models.
    let dynamic: &dyn ObservationModel = &NormalObservation::new("cp", 0.5);
    assert!(sde.filter(&subject, &parameters, dynamic).is_ok());
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

    for observation_model in [
        NormalObservation::new(0, 0.5),
        NormalObservation::new(0, 0.05),
    ] {
        let sde_log_likelihood = sde
            .log_likelihood(&subject, &parameters, &observation_model)
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
        let scored = observation_model.score(&particles.point_summary()).unwrap();
        assert!((sde_log_likelihood - scored).abs() < 1e-12);

        // And that prediction is the ODE's.
        let ode_log_likelihood = observation_model
            .score(&ode.predict(&subject, &parameters).unwrap())
            .unwrap();
        assert!(
            (sde_log_likelihood - ode_log_likelihood).abs() < 1e-8,
            "SDE {sde_log_likelihood} vs ODE {ode_log_likelihood}"
        );
    }
}

/// One `log_density` call: the label, observed value and polynomial the
/// filter passed.
type Seen = (OutputLabel, f64, Option<ErrorPoly>);

/// A normal observation model that records what the filter passes to
/// [`ObservationModel::log_density`]. Its sigma is the observation's
/// polynomial `c0` when there is one, and it rejects labels it does not
/// cover.
struct RecordingObservation {
    labels: Vec<OutputLabel>,
    sigma: f64,
    seen: std::sync::Mutex<Vec<Seen>>,
}

impl RecordingObservation {
    fn new(labels: impl IntoIterator<Item = usize>, sigma: f64) -> Self {
        Self {
            labels: labels.into_iter().map(OutputLabel::from).collect(),
            sigma,
            seen: Default::default(),
        }
    }

    /// The distinct calls seen so far, in first-seen order.
    fn seen(&self) -> Vec<Seen> {
        let mut distinct: Vec<Seen> = Vec::new();
        for call in self.seen.lock().unwrap().iter() {
            if !distinct.contains(call) {
                distinct.push(call.clone());
            }
        }
        distinct
    }

    /// The distinct observed values seen so far, in first-seen order.
    fn observed(&self) -> Vec<f64> {
        self.seen().into_iter().map(|(_, value, _)| value).collect()
    }

    fn clear(&self) {
        self.seen.lock().unwrap().clear();
    }
}

impl ObservationModel for RecordingObservation {
    fn log_density(
        &self,
        outeq: &OutputLabel,
        observation: f64,
        prediction: f64,
        _censoring: Censor,
        errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        if !self.labels.contains(outeq) {
            return Err(PharmsolError::OtherError(format!(
                "unexpected output {outeq}"
            )));
        }
        self.seen
            .lock()
            .unwrap()
            .push((outeq.clone(), observation, errorpoly));
        let sigma = errorpoly.map_or(self.sigma, |polynomial| polynomial.c0());
        let z = (observation - prediction) / sigma;
        let log_density = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln() - 0.5 * z * z;
        if log_density.is_finite() {
            Ok(log_density)
        } else {
            Err(PharmsolError::NonFiniteLikelihood(log_density))
        }
    }

    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_> {
        Box::new(self.labels.iter())
    }

    fn cache_key(&self) -> Option<u64> {
        Some(self.sigma.to_bits())
    }
}

/// [`simple_sde`] with a second output at twice the first.
fn two_output_sde() -> SDE {
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
        y[1] = 2.0 * x[0] / v;
    };

    SDE::new(drift, diffusion, lag, fa, init, out, 64)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(2)
}

#[test]
fn the_filter_passes_each_observations_own_label_and_polynomial() {
    let sde = two_output_sde().disable_cache();
    let polynomial = ErrorPoly::new(0.3, 0.0, 0.0, 0.0);
    let subject = Subject::builder("two outputs")
        .infusion(0.0, 100.0, 0, 1.0)
        .observation(0.5, 0.8, 0)
        .observation_with_error(1.0, 2.4, 1, polynomial, Censor::None)
        .observation(2.0, 0.9, 1)
        .build();
    let parameters = [1.0, 50.0];
    let observation_model = RecordingObservation::new([0, 1], 0.5);
    let expected: Vec<Seen> = vec![
        (OutputLabel::from(0), 0.8, None),
        (OutputLabel::from(1), 2.4, Some(polynomial)),
        (OutputLabel::from(1), 0.9, None),
    ];

    sde.log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    assert_eq!(observation_model.seen(), expected);

    observation_model.clear();
    sde.filter(&subject, &parameters, &observation_model)
        .unwrap();
    assert_eq!(observation_model.seen(), expected);

    // A model covering only the first output errors on the second.
    let first_only = RecordingObservation::new([0], 0.5);
    assert!(matches!(
        sde.log_likelihood(&subject, &parameters, &first_only),
        Err(PharmsolError::OtherError(_))
    ));
}

#[test]
fn the_observation_polynomial_reaches_the_cached_likelihood() {
    let sde = simple_sde();
    let parameters = [1.0, 50.0];
    let observation_model = RecordingObservation::new([0], 0.5);
    let subject = |c0: f64| {
        Subject::builder("polynomial")
            .infusion(0.0, 100.0, 0, 1.0)
            .observation(0.5, 0.8, 0)
            .observation_with_error(1.0, 1.2, 0, ErrorPoly::new(c0, 0.0, 0.0, 0.0), Censor::None)
            .build()
    };
    let (narrow, wide) = (subject(0.2), subject(2.0));

    let cached_narrow = sde
        .log_likelihood(&narrow, &parameters, &observation_model)
        .unwrap();
    let cached_wide = sde
        .log_likelihood(&wide, &parameters, &observation_model)
        .unwrap();
    assert_ne!(cached_narrow, cached_wide);
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 2);

    let uncached = sde.clone().disable_cache();
    for (subject, cached) in [(&narrow, cached_narrow), (&wide, cached_wide)] {
        assert_eq!(
            uncached
                .log_likelihood(subject, &parameters, &observation_model)
                .unwrap(),
            cached
        );
    }
}

#[test]
fn filter_never_fills_the_likelihood_cache() {
    let sde = simple_sde();
    let subject = filter_subject();
    let parameters = [1.0, 50.0];
    let observation_model = filter_observation_model();

    let filtered = sde
        .filter(&subject, &parameters, &observation_model)
        .unwrap();
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 0);

    let log_likelihood = sde
        .log_likelihood(&subject, &parameters, &observation_model)
        .unwrap();
    assert_eq!(sde.cache.as_ref().unwrap().entry_count(), 1);
    assert_eq!(filtered.log_likelihood, log_likelihood);
}

#[test]
fn missing_observations_are_not_scored() {
    let parameters = [1.0, 50.0];
    let subject = |missing: bool| {
        let builder = Subject::builder("missing")
            .infusion(0.0, 300.0, 0, 3.0)
            .observation(0.5, 0.9, 0);
        let builder = match missing {
            true => builder.missing_observation(1.0, 0),
            false => builder,
        };
        builder.observation(2.0, 3.8, 0).build()
    };

    // Identical particles, so leaving an observation out does not change
    // the other observations' estimates.
    let sde = deterministic_sde().disable_cache();
    let observation_model = RecordingObservation::new([0], 0.5);
    let with_missing = sde
        .log_likelihood(&subject(true), &parameters, &observation_model)
        .unwrap();
    assert_eq!(observation_model.observed(), vec![0.9, 3.8]);
    let without = sde
        .log_likelihood(&subject(false), &parameters, &observation_model)
        .unwrap();
    assert!((with_missing - without).abs() < 1e-12);

    // The filter keeps uniform weights at the missing observation.
    let filtered = simple_sde()
        .disable_cache()
        .filter(&subject(true), &parameters, &filter_observation_model())
        .unwrap();
    let weights = filtered.predictions.weights().unwrap();
    assert!(weights
        .column(1)
        .iter()
        .all(|&weight| weight == 1.0 / 128.0));
    assert!(weights
        .column(0)
        .iter()
        .any(|&weight| weight != 1.0 / 128.0));
}

#[test]
fn nothing_is_scored_after_an_observation_rules_out_every_particle() {
    let sde = simple_sde().disable_cache();
    let subject = Subject::builder("far")
        .observation(1.0, 1.0e6, 0)
        .observation(2.0, 1.0, 0)
        .build();
    let observation_model = RecordingObservation::new([0], 1.0e-200);

    let log_likelihood = sde
        .log_likelihood(&subject, &[1.0, 50.0], &observation_model)
        .unwrap();
    assert_eq!(log_likelihood, f64::NEG_INFINITY);
    assert_eq!(observation_model.observed(), vec![1.0e6]);

    // The filter keeps simulating to predict every observation, but scores
    // none after the first.
    observation_model.clear();
    let filtered = sde
        .filter(&subject, &[1.0, 50.0], &observation_model)
        .unwrap();
    assert_eq!(observation_model.observed(), vec![1.0e6]);
    assert_eq!(filtered.predictions.dim(), (128, 2));
    assert_eq!(filtered.log_likelihood, f64::NEG_INFINITY);
}
