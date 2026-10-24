use super::*;
use crate::simulator::equation::{self, Covariate, Route};
use crate::simulator::equation::{AttachMetadataError, ModelMetadataError};
use crate::simulator::Drift;
use crate::SubjectBuilderExt;
use crate::{fa, fetch_params, lag};
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
        .estimate_predictions(&numeric, &crate::parameters::dense([]))
        .expect("numeric labels should resolve by exact name")[[0, 0]]
    .prediction();
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
            .estimate_predictions(&prefixed, &crate::parameters::dense([]))
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
        .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
        .unwrap();
    let injected_predictions = injected
        .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    assert_eq!(explicit_predictions[[0, 0]].prediction(), 0.0);
    assert_eq!(injected_predictions[[0, 0]].prediction(), 0.0);
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
        .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
        .unwrap();
    let injected_predictions = injected
        .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    let explicit_prediction = explicit_predictions[[0, 0]].prediction();
    let injected_prediction = injected_predictions[[0, 0]].prediction();

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
        .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    assert!(sde.metadata().is_none());
    assert_eq!(predictions[[0, 0]].prediction(), 0.0);
}

fn values(predictions: &Array2<Prediction>) -> Vec<f64> {
    predictions.iter().map(Prediction::prediction).collect()
}

fn filter_subject() -> Subject {
    Subject::builder("filter")
        .infusion(0.0, 100.0, 0, 1.0)
        .observation(0.5, 0.8, 0)
        .observation(1.0, 1.2, 0)
        .observation(2.0, 0.9, 0)
        .build()
}

fn filter_error_models() -> AssayErrorModels {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    AssayErrorModels::new()
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
        .estimate_log_likelihood(&subject, &parameters, &error_models)
        .unwrap();
    let second = sde
        .estimate_log_likelihood(&subject, &parameters, &error_models)
        .unwrap();
    assert!(first.is_finite());
    assert_eq!(first, second);

    let predictions = sde.estimate_predictions(&subject, &parameters).unwrap();
    let again = sde.estimate_predictions(&subject, &parameters).unwrap();
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
        .estimate_log_likelihood(&subject, &[1.0, 50.0], &error_models)
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
    let default_seed = simple_sde()
        .estimate_predictions(&subject, &parameters)
        .unwrap();
    let other_seed = simple_sde()
        .with_seed(7)
        .estimate_predictions(&subject, &parameters)
        .unwrap();
    assert_ne!(values(&default_seed), values(&other_seed));
}

#[test]
fn impossible_observation_has_zero_likelihood() {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    let sde = simple_sde();
    let subject = Subject::builder("far").observation(1.0, 1.0e6, 0).build();
    let error_models = AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(1.0e-3, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let log_likelihood = sde
        .estimate_log_likelihood(&subject, &[1.0, 50.0], &error_models)
        .unwrap();
    assert_eq!(log_likelihood, f64::NEG_INFINITY);
}
