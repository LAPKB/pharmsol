pub mod one_compartment_cl_models;
pub mod one_compartment_models;
pub mod three_compartment_cl_models;
pub mod three_compartment_models;
pub mod two_compartment_cl_models;
pub mod two_compartment_models;

use diffsol::{NalgebraContext, Vector, VectorHost};
pub use one_compartment_cl_models::*;
pub use one_compartment_models::*;
use pharmsol_dsl::ModelKind;
use thiserror::Error;
pub use three_compartment_cl_models::*;
pub use three_compartment_models::*;
pub use two_compartment_cl_models::*;
pub use two_compartment_models::*;

use super::spphash;

use super::{
    EqnKind, Equation, EquationPriv, EquationTypes, ModelMetadata, ModelMetadataError,
    ValidatedModelMetadata,
};
use crate::data::error_model::AssayErrorModels;
use crate::simulator::cache::{PredictionCache, DEFAULT_CACHE_SIZE};
use crate::PharmsolError;
use crate::{data::Covariates, simulator::*, Observation, Subject};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AnalyticalMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("analytical model declares {declared} state metadata entries but model has {expected} states")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error(
        "analytical model declares {declared} route metadata entries but model has {expected} input channels"
    )]
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
    eq: AnalyticalEq,
    seq_eq: SecEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    metadata: Option<ValidatedModelMetadata>,
    cache: Option<PredictionCache>,
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
    /// Create a new Analytical equation model with default Neqs (all sizes = 5).
    ///
    /// Use builder methods to configure dimensions:
    /// ```ignore
    /// Analytical::new(eq, seq_eq, lag, fa, init, out)
    ///     .with_nstates(2)
    ///     .with_ndrugs(1)
    ///     .with_nout(1)
    /// ```
    pub fn new(eq: AnalyticalEq, seq_eq: SecEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Self {
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            metadata: None,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    /// Set the number of state variables.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.neqs.nstates = nstates;
        self.invalidate_metadata();
        self
    }

    /// Set the number of drug input channels (size of bolus[] and rateiv[]).
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.neqs.ndrugs = ndrugs;
        self.invalidate_metadata();
        self
    }

    /// Set the number of output equations.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.neqs.nout = nout;
        self.invalidate_metadata();
        self
    }

    /// Attach validated handwritten-model metadata to this analytical model.
    pub fn with_metadata(
        mut self,
        metadata: ModelMetadata,
    ) -> Result<Self, AnalyticalMetadataError> {
        let metadata = metadata.validate_for(ModelKind::Analytical)?;
        validate_metadata_dimensions(&metadata, &self.neqs)?;
        self.metadata = Some(metadata);
        Ok(self)
    }

    /// Access the validated metadata attached to this analytical model, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.state_index(name)
    }

    pub fn route_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.route_index(name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.metadata()?.output_index(name)
    }

    fn invalidate_metadata(&mut self) {
        self.metadata = None;
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

    let declared_routes = metadata.route_channel_count();
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

impl super::Cache for Analytical {
    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

impl EquationTypes for Analytical {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for Analytical {
    // #[inline(always)]
    // fn get_init(&self) -> &Init {
    //     &self.init
    // }

    // #[inline(always)]
    // fn get_out(&self) -> &Out {
    //     &self.out
    // }

    // #[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.lag)(&V::from_vec(spp.to_owned())))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.fa)(&V::from_vec(spp.to_owned())))
    // }

    #[inline(always)]
    fn lag(&self) -> &Lag {
        &self.lag
    }

    #[inline(always)]
    fn fa(&self) -> &Fa {
        &self.fa
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.nstates
    }

    #[inline(always)]
    fn get_ndrugs(&self) -> usize {
        self.neqs.ndrugs
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.nout
    }
    #[inline(always)]
    fn solve(
        &self,
        x: &mut Self::S,
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        if ti == tf {
            return Ok(());
        }

        // 1) Build and sort event times
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

        // 2) March over each sub-interval
        let mut current_t = ts[0];
        let mut sp = V::from_vec(support_point.to_vec(), NalgebraContext);
        let mut rateiv = V::zeros(self.get_ndrugs(), NalgebraContext);

        for &next_t in &ts[1..] {
            // prepare support and infusion rate for [current_t .. next_t]
            rateiv.fill(0.0);
            for inf in infusions {
                let s = inf.time();
                let e = s + inf.duration();
                if current_t >= s && next_t <= e {
                    if inf.input() >= self.get_ndrugs() {
                        return Err(PharmsolError::InputOutOfRange {
                            input: inf.input(),
                            ndrugs: self.get_ndrugs(),
                        });
                    }
                    rateiv[inf.input()] += inf.amount() / inf.duration();
                }
            }

            // advance the support-point to next_t
            (self.seq_eq)(&mut sp, next_t, covariates);

            // advance state by dt
            let dt = next_t - current_t;
            *x = (self.eq)(x, &sp, dt, &rateiv, covariates);

            current_t = next_t;
        }

        Ok(())
    }

    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &[f64],
        observation: &Observation,
        error_models: Option<&AssayErrorModels>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.to_vec(), NalgebraContext),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_prediction(pred, x.as_slice().to_vec());
        if let Some(error_models) = error_models {
            likelihood.push(pred.log_likelihood(error_models)?.exp());
        }
        output.add_prediction(pred);
        Ok(())
    }
    #[inline(always)]
    fn initial_state(&self, spp: &[f64], covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates(), NalgebraContext);
        if occasion_index == 0 {
            (init)(
                &V::from_vec(spp.to_vec(), NalgebraContext),
                0.0,
                covariates,
                &mut x,
            );
        }
        x
    }
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
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

        let predictions = analytical.estimate_predictions(&subject, &[1.0]).unwrap();

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

        let predictions = analytical.estimate_predictions(&subject, &[0.0]).unwrap();

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
                super::super::metadata::new("one_cmt_analytical")
                    .parameters(["ke", "v"])
                    .covariates([super::super::Covariate::continuous("wt")])
                    .states(["central"])
                    .outputs(["cp"])
                    .route(super::super::Route::infusion("iv").to_state("central")),
            )
            .expect("metadata attachment should validate");

        assert_eq!(analytical.parameter_index("ke"), Some(0));
        assert_eq!(analytical.parameter_index("v"), Some(1));
        assert_eq!(analytical.covariate_index("wt"), Some(0));
        assert_eq!(analytical.state_index("central"), Some(0));
        assert_eq!(analytical.route_index("iv"), Some(0));
        assert_eq!(analytical.output_index("cp"), Some(0));
        assert_eq!(
            analytical.metadata().expect("metadata exists").kind(),
            ModelKind::Analytical
        );
    }

    #[test]
    fn handwritten_analytical_without_metadata_keeps_raw_path() {
        let analytical = simple_analytical();

        assert!(analytical.metadata().is_none());
        assert_eq!(analytical.state_index("central"), None);
        assert_eq!(analytical.route_index("iv"), None);
        assert_eq!(analytical.output_index("cp"), None);
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
        assert_eq!(analytical.route_index("oral"), Some(0));
        assert_eq!(analytical.route_index("iv"), Some(0));
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
        assert_eq!(analytical.route_index("iv"), None);
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
impl Equation for Analytical {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let ypred = _subject_predictions(self, subject, support_point)?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> EqnKind {
        EqnKind::Analytical
    }
}

#[inline(always)]
fn _subject_predictions(
    analytical: &Analytical,
    subject: &Subject,
    support_point: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = &analytical.cache {
        let key = (subject.hash(), spphash(support_point));
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = analytical.simulate_subject(subject, support_point, None)?.0;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        Ok(analytical.simulate_subject(subject, support_point, None)?.0)
    }
}

fn _estimate_likelihood(
    ode: &Analytical,
    subject: &Subject,
    support_point: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    let ypred = _subject_predictions(ode, subject, support_point)?;
    Ok(ypred.log_likelihood(error_models)?.exp())
}
