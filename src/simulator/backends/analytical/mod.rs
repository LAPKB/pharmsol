pub mod one_compartment_cl_models;
pub mod one_compartment_models;
pub mod three_compartment_cl_models;
pub mod three_compartment_models;
pub mod two_compartment_cl_models;
pub mod two_compartment_models;

use crate::core::ModelInfo;
use diffsol::{NalgebraContext, Vector, VectorHost};
pub use one_compartment_cl_models::*;
pub use one_compartment_models::*;
use pharmsol_dsl::ModelKind;
use thiserror::Error;
pub use three_compartment_cl_models::*;
pub use three_compartment_models::*;
pub use two_compartment_cl_models::*;
pub use two_compartment_models::*;

use crate::simulator::backends::parameters_hash;

use crate::core::metadata::{ModelMetadata, ModelMetadataError, ValidatedModelMetadata};
use crate::data::error_model::AssayErrorModels;
use crate::simulator::cache::{BoundErrorModelCache, PredictionCache, DEFAULT_CACHE_SIZE};
use crate::simulator::likelihood::Prediction;
use crate::PharmsolError;
use crate::{data::Covariates, simulator::*, Observation, Subject};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AnalyticalMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("analytical model declares {declared} state metadata entries but model has {expected} states")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error("analytical model declares {declared} route metadata entries but model has {expected} inputs")]
    RouteCountMismatch { expected: usize, declared: usize },
    #[error("analytical model declares {declared} output metadata entries but model has {expected} outputs")]
    OutputCountMismatch { expected: usize, declared: usize },
}

/// Model equation using analytical solutions.
///
/// This implementation uses closed-form analytical solutions for the model
/// equations rather than numerical integration.
#[derive(Clone, Debug)]
pub struct Analytical {
    core: crate::core::ModelCore<PredictionCache>,
    eq: AnalyticalEq,
    seq_eq: SecEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
}

#[inline(always)]
pub(crate) fn compact_public_vector(vector: &V) -> V {
    V::from_vec(
        vector.as_slice().get(1..).unwrap_or(&[]).to_vec(),
        NalgebraContext,
    )
}

#[inline(always)]
pub(crate) fn pad_public_vector(vector: &V) -> V {
    let mut padded = Vec::with_capacity(vector.len() + 1);
    padded.push(0.0);
    padded.extend(vector.as_slice().iter().copied());
    V::from_vec(padded, NalgebraContext)
}

#[inline(always)]
pub(crate) fn wrap_pmetrics_analytical(
    x: &V,
    p: &V,
    t: T,
    rateiv: &V,
    cov: &Covariates,
    native: AnalyticalEq,
) -> V {
    let compact_x = compact_public_vector(x);
    let compact_rateiv = compact_public_vector(rateiv);
    let compact_output = native(&compact_x, p, t, &compact_rateiv, cov);
    pad_public_vector(&compact_output)
}

impl Analytical {
    pub fn new(eq: AnalyticalEq, seq_eq: SecEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Self {
            core: crate::core::ModelCore::new(Some(PredictionCache::new(DEFAULT_CACHE_SIZE))),
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
        }
    }

    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.core = self.core.with_nstates(nstates);
        self
    }

    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.core = self.core.with_ndrugs(ndrugs);
        self
    }

    pub fn with_nout(mut self, nout: usize) -> Self {
        self.core = self.core.with_nout(nout);
        self
    }

    pub fn with_metadata(
        mut self,
        metadata: ModelMetadata,
    ) -> Result<Self, AnalyticalMetadataError> {
        let validated = metadata
            .validate_for(ModelKind::Analytical)
            .map_err(AnalyticalMetadataError::Validation)?;
        validate_metadata_dimensions(&validated, &self.core.dims())?;
        self.core.set_metadata(validated);
        Ok(self)
    }

    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.core.metadata()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.core.metadata()?.state_index(name)
    }
}

fn validate_metadata_dimensions(
    metadata: &ValidatedModelMetadata,
    neqs: &Neqs,
) -> Result<(), AnalyticalMetadataError> {
    let declared_states = metadata.states().len();
    if declared_states != neqs.nstates {
        return Err(AnalyticalMetadataError::StateCountMismatch {
            expected: neqs.nstates,
            declared: declared_states,
        });
    }

    let declared_routes = metadata.route_input_count();
    if declared_routes != neqs.ndrugs {
        return Err(AnalyticalMetadataError::RouteCountMismatch {
            expected: neqs.ndrugs,
            declared: declared_routes,
        });
    }

    let declared_outputs = metadata.outputs().len();
    if declared_outputs != neqs.nout {
        return Err(AnalyticalMetadataError::OutputCountMismatch {
            expected: neqs.nout,
            declared: declared_outputs,
        });
    }

    Ok(())
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::Simulate;
    use crate::SubjectBuilderExt;
    use approx::assert_relative_eq;
    use diffsol::Vector;
    use pharmsol_dsl::AnalyticalKernel;
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
            .estimate_predictions(&subject, &crate::parameters::dense([1.0]))
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
            .estimate_predictions(&subject, &crate::parameters::dense([0.0]))
            .unwrap();

        assert_eq!(predictions.predictions()[0].prediction(), 4.0);
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
                crate::core::metadata::new("one_cmt_analytical")
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
    fn handwritten_analytical_metadata_resolves_raw_numeric_aliases_against_canonical_labels() {
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
                crate::core::metadata::new("numeric_alias_analytical")
                    .states(["central"])
                    .outputs(["outeq_1"])
                    .route(super::super::Route::infusion("input_1").to_state("central")),
            )
            .expect("metadata attachment should validate");

        let canonical = Subject::builder("canonical")
            .infusion(0.0, 100.0, "input_1", 1.0)
            .observation(1.0, 0.0, "outeq_1")
            .build();
        let aliased = Subject::builder("aliased")
            .infusion(0.0, 100.0, "1", 1.0)
            .observation(1.0, 0.0, "1")
            .build();

        let canonical_predictions = analytical
            .estimate_predictions(&canonical, &crate::parameters::dense([]))
            .expect("canonical labels should simulate");
        let aliased_predictions = analytical
            .estimate_predictions(&aliased, &crate::parameters::dense([]))
            .expect("raw numeric aliases should simulate");

        assert_relative_eq!(
            canonical_predictions.predictions()[0].prediction(),
            aliased_predictions.predictions()[0].prediction(),
            epsilon = 1e-10
        );
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
                crate::core::metadata::new("wrong_outputs")
                    .parameters(["ke"])
                    .states(["central"])
                    .outputs(["cp", "auc"])
                    .route(super::super::Route::infusion("iv").to_state("central")),
            )
            .expect_err("output-count mismatches must fail");

        assert_eq!(
            error,
            AnalyticalMetadataError::OutputCountMismatch {
                expected: 1,
                declared: 2,
            }
        );
    }

    #[test]
    fn handwritten_analytical_rejects_particles_metadata() {
        let error = simple_analytical()
            .with_metadata(
                crate::core::metadata::new("invalid_particles")
                    .parameters(["ke"])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::infusion("iv").to_state("central"))
                    .particles(64),
            )
            .expect_err("analytical metadata must reject particles");

        assert_eq!(
            error,
            AnalyticalMetadataError::Validation(ModelMetadataError::ParticlesNotAllowed {
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

        let analytical =
            Analytical::new(one_compartment_with_absorption, seq_eq, lag, fa, init, out)
                .with_nstates(2)
                .with_ndrugs(1)
                .with_nout(1)
                .with_metadata(
                    crate::core::metadata::new("one_cmt_abs")
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
                crate::core::metadata::new("one_cmt_analytical")
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
        let compact_x = V::from_vec(compact_x, NalgebraContext);
        let params = V::from_vec(params, NalgebraContext);
        let compact_rateiv = V::from_vec(compact_rateiv, NalgebraContext);

        let mut padded_x = vec![1234.0];
        padded_x.extend(compact_x.as_slice().iter().copied());
        let padded_x = V::from_vec(padded_x, NalgebraContext);

        let mut padded_rateiv = vec![5678.0];
        padded_rateiv.extend(compact_rateiv.as_slice().iter().copied());
        let padded_rateiv = V::from_vec(padded_rateiv, NalgebraContext);

        let native_output = native(&compact_x, &params, 1.5, &compact_rateiv, &covariates);
        let wrapped_output = wrapper(&padded_x, &params, 1.5, &padded_rateiv, &covariates);

        assert_eq!(wrapped_output[0], 0.0);
        assert_eq!(wrapped_output.len(), native_output.len() + 1);

        for (wrapped, native) in wrapped_output
            .as_slice()
            .iter()
            .skip(1)
            .zip(native_output.as_slice().iter())
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
}

// ── New core traits ─────────────────────────────────────────────────────────

impl crate::core::Solver for Analytical {
    type State = V;

    fn solve(
        &self,
        x: &mut Self::State,
        parameters: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        if ti == tf {
            return Ok(());
        }

        let mut ts = Vec::new();
        ts.push(ti);
        ts.push(tf);
        for inf in infusions {
            let t0 = inf.time();
            let t1 = t0 + inf.duration();
            if t0 > ti && t0 < tf {
                ts.push(t0)
            }
            if t1 > ti && t1 < tf {
                ts.push(t1)
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut current_t = ts[0];
        let mut parameters_v = V::from_vec(parameters.to_vec(), NalgebraContext);
        let mut rateiv = V::zeros(self.ndrugs(), NalgebraContext);

        for &next_t in &ts[1..] {
            rateiv.fill(0.0);
            for inf in infusions {
                let s = inf.time();
                let e = s + inf.duration();
                if current_t >= s && next_t <= e {
                    let input =
                        inf.input_index()
                            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                                label: inf.input().to_string(),
                            })?;
                    if input >= self.ndrugs() {
                        return Err(PharmsolError::InputOutOfRange {
                            input,
                            ndrugs: self.ndrugs(),
                        });
                    }
                    rateiv[input] += inf.amount() / inf.duration();
                }
            }

            (self.seq_eq)(&mut parameters_v, next_t, covariates);
            let dt = next_t - current_t;
            *x = (self.eq)(x, &parameters_v, dt, &rateiv, covariates);
            current_t = next_t;
        }

        Ok(())
    }

    fn process_observation(
        &self,
        x: &Self::State,
        parameters: &[f64],
        observation: &Observation,
        error_models: Option<&AssayErrorModels>,
        covariates: &Covariates,
    ) -> Result<(Prediction, Option<f64>), PharmsolError> {
        let mut y = V::zeros(self.nout(), NalgebraContext);
        (self.out)(
            x,
            &V::from_vec(parameters.to_vec(), NalgebraContext),
            observation.time(),
            covariates,
            &mut y,
        );
        let outeq = observation
            .outeq_index()
            .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                label: observation.outeq().to_string(),
            })?;
        let pred = observation.to_prediction(y[outeq], x.as_slice().to_vec());
        let lik = error_models
            .map(|em| pred.log_likelihood(em).map(f64::exp))
            .transpose()?;
        Ok((pred, lik))
    }

    fn initial_state(
        &self,
        parameters: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> V {
        let mut x = V::zeros(self.nstates(), NalgebraContext);
        if occasion_index == 0 {
            (self.init)(
                &V::from_vec(parameters.to_vec(), NalgebraContext),
                0.0,
                covariates,
                &mut x,
            );
        }
        x
    }
}

impl crate::core::ModelInfo for Analytical {
    fn nstates(&self) -> usize {
        self.core.nstates()
    }

    fn ndrugs(&self) -> usize {
        self.core.ndrugs()
    }

    fn nout(&self) -> usize {
        self.core.nout()
    }

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.core.metadata()
    }

    fn lag(&self) -> &Lag {
        &self.lag
    }

    fn fa(&self) -> &Fa {
        &self.fa
    }
}

impl crate::core::Caching for Analytical {
    fn prediction_cache(&self) -> Option<&PredictionCache> {
        self.core.cache()
    }

    fn error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.core.error_model_cache()
    }

    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.core = self.core.with_cache_capacity(PredictionCache::new(size));
        self
    }

    fn without_cache(mut self) -> Self {
        self.core = self.core.without_cache();
        self
    }

    fn clear_cache(&self) {
        self.core.clear_cache();
        if let Some(cache) = self.core.cache() {
            cache.invalidate_all();
        }
    }
}

impl crate::core::Simulate for Analytical {
    type Predictions = SubjectPredictions;

    fn simulate_subject(
        &self,
        subject: &Subject,
        params: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::Predictions, Option<f64>), PharmsolError> {
        if error_models.is_none() {
            if let Some(cache) = self.core.cache() {
                let key = (subject.hash(), parameters_hash(params));
                if let Some(cached) = cache.get(&key) {
                    return Ok((cached, None));
                }
            }
        }

        let result = crate::core::standard_event_loop::<Self, SubjectPredictions>(
            self,
            subject,
            params,
            error_models,
        )?;

        if error_models.is_none() {
            if let Some(cache) = self.core.cache() {
                let key = (subject.hash(), parameters_hash(params));
                cache.insert(key, result.0.clone());
            }
        }

        Ok(result)
    }

    fn kind() -> pharmsol_dsl::ModelKind {
        pharmsol_dsl::ModelKind::Analytical
    }
}
