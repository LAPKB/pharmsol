use super::*;
use crate::simulator::equation::{AttachMetadataError, ModelMetadataError};
use crate::AssayErrorModels;
use crate::SubjectBuilderExt;
use approx::assert_relative_eq;
use diffsol::Vector;
use pharmsol_dsl::AnalyticalKernel;
use pharmsol_dsl::ModelKind;
use std::collections::HashMap;

pub(crate) enum SubjectInfo {
    InfusionDosing,
    OralInfusionDosage,
}
impl SubjectInfo {
    pub(crate) fn get_subject(&self) -> Subject {
        match self {
            SubjectInfo::InfusionDosing => Subject::builder("id1")
                .bolus(0.0, 100.0, 0)
                .infusion(24.0, 150.0, 0, 3.0)
                .missing_observation(0.0, 0)
                .missing_observation(1.0, 0)
                .missing_observation(2.0, 0)
                .missing_observation(4.0, 0)
                .missing_observation(8.0, 0)
                .missing_observation(12.0, 0)
                .missing_observation(24.0, 0)
                .missing_observation(25.0, 0)
                .missing_observation(26.0, 0)
                .missing_observation(27.0, 0)
                .missing_observation(28.0, 0)
                .missing_observation(32.0, 0)
                .missing_observation(36.0, 0)
                .build(),

            SubjectInfo::OralInfusionDosage => Subject::builder("id1")
                .bolus(0.0, 100.0, 1)
                .infusion(24.0, 150.0, 0, 3.0)
                .bolus(48.0, 100.0, 0)
                .missing_observation(0.0, 0)
                .missing_observation(1.0, 0)
                .missing_observation(2.0, 0)
                .missing_observation(4.0, 0)
                .missing_observation(8.0, 0)
                .missing_observation(12.0, 0)
                .missing_observation(24.0, 0)
                .missing_observation(25.0, 0)
                .missing_observation(26.0, 0)
                .missing_observation(27.0, 0)
                .missing_observation(28.0, 0)
                .missing_observation(32.0, 0)
                .missing_observation(36.0, 0)
                .missing_observation(48.0, 0)
                .missing_observation(49.0, 0)
                .missing_observation(50.0, 0)
                .missing_observation(52.0, 0)
                .missing_observation(56.0, 0)
                .missing_observation(60.0, 0)
                .build(),
        }
    }
}

#[test]
fn secondary_equations_accumulate_within_single_solve() {
    let eq = |x: &V, p: &V, dt: f64, _rateiv: &V, _cov: &Covariates| {
        let mut next = x.clone();
        next[0] += p[0] * dt;
        next
    };
    let seq_eq = |params: &mut V, _t: f64, _cov: &Covariates| {
        params[0] += 1.0;
    };
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[0];
    };

    let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);
    let subject = Subject::builder("seq")
        .bolus(0.0, 0.0, 0)
        .infusion(0.25, 1.0, 0, 0.25)
        .observation(1.0, 0.0, 0)
        .build();

    let predictions = analytical
        .predict(&subject, &crate::parameters::dense([1.0]))
        .unwrap();

    let value = predictions.predictions()[0].prediction();
    assert!((value - 2.5).abs() < 1e-12);
}

#[test]
fn infusion_inputs_match_state_dimension() {
    let eq = |x: &V, _p: &V, dt: f64, rateiv: &V, _cov: &Covariates| {
        let mut next = x.clone();
        next[0] += rateiv[3] * dt;
        next
    };
    let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[0];
    };

    let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out)
        .with_nstates(4)
        .with_ndrugs(4)
        .with_nout(1);
    let subject = Subject::builder("inf")
        .infusion(0.0, 4.0, 3, 1.0)
        .observation(1.0, 0.0, 0)
        .build();

    let predictions = analytical
        .predict(&subject, &crate::parameters::dense([0.0]))
        .unwrap();

    assert_eq!(predictions.predictions()[0].prediction(), 4.0);
}

#[test]
fn log_likelihood_matrix_reuses_cached_predictions() {
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    use crate::simulator::likelihood::log_likelihood_matrix;
    use crate::{Cache, Data};
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Counts calls into the structural kernel, i.e. real solves.
    static KERNEL_CALLS: AtomicUsize = AtomicUsize::new(0);

    let eq = |x: &V, p: &V, dt: f64, _rateiv: &V, _cov: &Covariates| {
        KERNEL_CALLS.fetch_add(1, Ordering::SeqCst);
        let mut next = x.clone();
        next[0] *= (-p[0] * dt).exp();
        next
    };
    let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[0] / p[1];
    };

    let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);
    let data = Data::new(vec![Subject::builder("cached")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 4.0, 0)
        .observation(2.0, 3.5, 0)
        .build()]);
    let support_points = array![[0.2, 20.0], [0.3, 25.0]];
    let error_models = AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 0.5),
        )
        .unwrap();

    let first =
        log_likelihood_matrix(&analytical, &data, &support_points, &error_models, false).unwrap();
    let after_first = KERNEL_CALLS.load(Ordering::SeqCst);
    assert!(after_first > 0, "the first matrix call must solve");
    assert_eq!(
        analytical.cache.as_ref().unwrap().entry_count(),
        support_points.nrows(),
        "the first matrix call must fill the prediction cache"
    );

    let second =
        log_likelihood_matrix(&analytical, &data, &support_points, &error_models, false).unwrap();
    assert_eq!(
        KERNEL_CALLS.load(Ordering::SeqCst),
        after_first,
        "the second matrix call on the same support points must be served from the cache"
    );
    assert_eq!(first, second);

    let parameters = crate::parameters::dense([0.2, 20.0]);
    analytical.predict(data.subjects()[0], &parameters).unwrap();
    assert_eq!(
        KERNEL_CALLS.load(Ordering::SeqCst),
        after_first,
        "predict must reuse the cached prediction"
    );

    // The cache only saves work: without it the matrix is the same.
    let uncached = analytical.disable_cache();
    let third =
        log_likelihood_matrix(&uncached, &data, &support_points, &error_models, false).unwrap();
    assert!(KERNEL_CALLS.load(Ordering::SeqCst) > after_first);
    assert_eq!(first, third);
}

fn simple_analytical() -> Analytical {
    let eq = |x: &V, _p: &V, _dt: f64, _rateiv: &V, _cov: &Covariates| x.clone();
    let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[0];
    };

    Analytical::new(eq, seq_eq, lag, fa, init, out)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
}

#[test]
fn handwritten_analytical_metadata_exposes_name_lookup() {
    let analytical = simple_analytical()
        .with_metadata(
            super::super::metadata::new("one_cmt_analytical")
                .parameters(["ke", "v"])
                .covariates([super::super::Covariate::continuous("wt")])
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata attachment should validate");
    let metadata = analytical.metadata().expect("metadata exists");

    assert_eq!(analytical.parameter_index("ke"), Some(0));
    assert_eq!(analytical.parameter_index("v"), Some(1));
    assert_eq!(analytical.covariate_index("wt"), Some(0));
    assert_eq!(analytical.state_index("central"), Some(0));
    assert!(metadata.route("iv").is_some());
    assert!(metadata.output("cp").is_some());
    assert_eq!(metadata.kind(), ModelKind::Analytical);
}

#[test]
fn handwritten_analytical_metadata_resolves_numeric_labels_by_exact_name() {
    let eq = |x: &V, _p: &V, dt: f64, rateiv: &V, _cov: &Covariates| {
        let mut next = x.clone();
        next[0] += rateiv[0] * dt;
        next
    };
    let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[0];
    };

    let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out)
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            super::super::metadata::new("numeric_alias_analytical")
                .states(["central"])
                .outputs(["1"])
                .route(super::super::Route::infusion("1").to_state("central")),
        )
        .expect("metadata attachment should validate");

    let numeric = Subject::builder("numeric")
        .infusion(0.0, 100.0, "1", 1.0)
        .observation(1.0, 0.0, "1")
        .build();
    let prediction = analytical
        .predict(&numeric, &crate::parameters::dense([]))
        .expect("numeric labels should resolve by exact name")
        .predictions()[0]
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
        assert!(analytical
            .predict(&prefixed, &crate::parameters::dense([]))
            .is_err());
    }
}

#[test]
fn handwritten_analytical_without_metadata_keeps_raw_path() {
    let analytical = simple_analytical();

    assert!(analytical.metadata().is_none());
    assert_eq!(analytical.state_index("central"), None);
}

#[test]
fn handwritten_analytical_rejects_dimension_mismatches() {
    let error = simple_analytical()
        .with_metadata(
            super::super::metadata::new("wrong_outputs")
                .parameters(["ke"])
                .states(["central"])
                .outputs(["cp", "auc"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect_err("output-count mismatches must fail");

    assert_eq!(
        error,
        AttachMetadataError::OutputCountMismatch {
            expected: 1,
            declared: 2,
        }
    );
}

#[test]
fn handwritten_analytical_rejects_particles_metadata() {
    let error = simple_analytical()
        .with_metadata(
            super::super::metadata::new("invalid_particles")
                .parameters(["ke"])
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central"))
                .particles(64),
        )
        .expect_err("analytical metadata must reject particles");

    assert_eq!(
        error,
        AttachMetadataError::Validation(ModelMetadataError::ParticlesNotAllowed {
            kind: ModelKind::Analytical,
        })
    );
}

#[test]
fn built_in_analytical_models_can_advertise_kernel_identity() {
    let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
    let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
    let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
        x.fill(0.0);
    };
    let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
        y[0] = x[1];
    };

    let analytical = Analytical::new(one_compartment_with_absorption, seq_eq, lag, fa, init, out)
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            super::super::metadata::new("one_cmt_abs")
                .parameters(["ka", "ke", "v"])
                .states(["gut", "central"])
                .outputs(["cp"])
                .routes([
                    super::super::Route::bolus("oral").to_state("gut"),
                    super::super::Route::infusion("iv").to_state("central"),
                ])
                .analytical_kernel(AnalyticalKernel::OneCompartmentWithAbsorption),
        )
        .expect("built-in analytical metadata should validate");

    assert_eq!(
        analytical
            .metadata()
            .expect("metadata exists")
            .analytical_kernel(),
        Some(AnalyticalKernel::OneCompartmentWithAbsorption)
    );
    let metadata = analytical.metadata().expect("metadata exists");
    assert_eq!(
        metadata.route("oral").map(|route| route.input_index()),
        Some(0)
    );
    assert_eq!(
        metadata.route("iv").map(|route| route.input_index()),
        Some(0)
    );
}

#[test]
fn changing_dimensions_after_metadata_clears_analytical_metadata() {
    let analytical = simple_analytical()
        .with_metadata(
            super::super::metadata::new("one_cmt_analytical")
                .states(["central"])
                .outputs(["cp"])
                .route(super::super::Route::infusion("iv").to_state("central")),
        )
        .expect("metadata attachment should validate")
        .with_ndrugs(2);

    assert!(analytical.metadata().is_none());
}

fn assert_pm_wrapper_matches_native(
    native: AnalyticalEq,
    wrapper: AnalyticalEq,
    compact_x: Vec<f64>,
    params: Vec<f64>,
    compact_rateiv: Vec<f64>,
) {
    let covariates = Covariates::new();
    let compact_x = V::from_vec(compact_x, NalgebraContext::new());
    let params = V::from_vec(params, NalgebraContext::new());
    let compact_rateiv = V::from_vec(compact_rateiv, NalgebraContext::new());

    let mut padded_x = vec![1234.0];
    padded_x.extend(compact_x.batch_as_slice(0).iter().copied());
    let padded_x = V::from_vec(padded_x, NalgebraContext::new());

    let mut padded_rateiv = vec![5678.0];
    padded_rateiv.extend(compact_rateiv.batch_as_slice(0).iter().copied());
    let padded_rateiv = V::from_vec(padded_rateiv, NalgebraContext::new());

    let native_output = native(&compact_x, &params, 1.5, &compact_rateiv, &covariates);
    let wrapped_output = wrapper(&padded_x, &params, 1.5, &padded_rateiv, &covariates);

    assert_eq!(wrapped_output[0], 0.0);
    assert_eq!(wrapped_output.len(), native_output.len() + 1);

    for (wrapped, native) in wrapped_output
        .batch_as_slice(0)
        .iter()
        .skip(1)
        .zip(native_output.batch_as_slice(0).iter())
    {
        assert_relative_eq!(*wrapped, *native, max_relative = 1e-10, epsilon = 1e-10);
    }
}

#[test]
fn pmetrics_wrappers_match_native_helpers() {
    assert_pm_wrapper_matches_native(
        one_compartment,
        pm_one_compartment,
        vec![100.0],
        vec![0.2],
        vec![5.0],
    );
    assert_pm_wrapper_matches_native(
        one_compartment_cl,
        pm_one_compartment_cl,
        vec![100.0],
        vec![0.2, 2.0],
        vec![5.0],
    );
    assert_pm_wrapper_matches_native(
        one_compartment_with_absorption,
        pm_one_compartment_with_absorption,
        vec![10.0, 20.0],
        vec![1.1, 0.2],
        vec![5.0],
    );
    assert_pm_wrapper_matches_native(
        one_compartment_cl_with_absorption,
        pm_one_compartment_cl_with_absorption,
        vec![10.0, 20.0],
        vec![1.1, 0.2, 2.0],
        vec![5.0],
    );
    assert_pm_wrapper_matches_native(
        two_compartments,
        pm_two_compartments,
        vec![100.0, 40.0],
        vec![0.1, 0.3, 0.2],
        vec![3.0],
    );
    assert_pm_wrapper_matches_native(
        two_compartments_cl,
        pm_two_compartments_cl,
        vec![100.0, 40.0],
        vec![0.1, 0.3, 1.0, 2.0],
        vec![3.0],
    );
    assert_pm_wrapper_matches_native(
        two_compartments_with_absorption,
        pm_two_compartments_with_absorption,
        vec![10.0, 100.0, 40.0],
        vec![0.1, 1.0, 0.3, 0.2],
        vec![3.0],
    );
    assert_pm_wrapper_matches_native(
        two_compartments_cl_with_absorption,
        pm_two_compartments_cl_with_absorption,
        vec![10.0, 100.0, 40.0],
        vec![1.0, 0.1, 0.3, 1.0, 2.0],
        vec![3.0],
    );
    assert_pm_wrapper_matches_native(
        three_compartments,
        pm_three_compartments,
        vec![100.0, 40.0, 20.0],
        vec![0.1, 3.0, 2.0, 1.0, 0.5],
        vec![2.0],
    );
    assert_pm_wrapper_matches_native(
        three_compartments_cl,
        pm_three_compartments_cl,
        vec![100.0, 40.0, 20.0],
        vec![0.1, 3.0, 2.0, 1.0, 3.0, 4.0],
        vec![2.0],
    );
    assert_pm_wrapper_matches_native(
        three_compartments_with_absorption,
        pm_three_compartments_with_absorption,
        vec![10.0, 100.0, 40.0, 20.0],
        vec![1.0, 0.1, 3.0, 2.0, 1.0, 0.5],
        vec![2.0],
    );
    assert_pm_wrapper_matches_native(
        three_compartments_cl_with_absorption,
        pm_three_compartments_cl_with_absorption,
        vec![10.0, 100.0, 40.0, 20.0],
        vec![1.0, 0.1, 3.0, 2.0, 1.0, 3.0, 4.0],
        vec![2.0],
    );
}
