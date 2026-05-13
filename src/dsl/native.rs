use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, NalgebraContext, OdeBuilder,
    OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;
use ndarray::{concatenate, Array2, Axis};
use rayon::prelude::*;

#[cfg(feature = "dsl-jit")]
use cranelift_jit::JITModule;
#[cfg(feature = "dsl-aot-load")]
use libloading::Library;
use pharmsol_dsl::execution::KernelRole;
use pharmsol_dsl::{
    AnalyticalKernel, AnalyticalStructureInputKind, AnalyticalStructureInputPlan, RouteKind,
    NUMERIC_OUTPUT_PREFIX, NUMERIC_ROUTE_PREFIX,
};

pub use super::model_info::{
    NativeCovariateInfo, NativeModelInfo, NativeOutputInfo, NativeRouteInfo,
};
use crate::{
    data::error_model::AssayErrorModels,
    data::{Covariates, Infusion, InputLabel, OutputLabel},
    simulator::{
        cache::{
            BoundErrorModelCache, PredictionCache, SdeLikelihoodCache,
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE, DEFAULT_CACHE_SIZE,
        },
        equation::{
            ode::{closure_helpers::PMProblem, ExplicitRkTableau, OdeSolver, SdirkTableau},
            sde::simulate_sde_event_with,
            EqnKind, Equation, EquationPriv, EquationTypes, Predictions,
        },
        likelihood::{Prediction, SubjectPredictions},
        Fa, Lag, M, T, V,
    },
    Event, Observation, Occasion, Parameters, PharmsolError, Subject,
};

pub type DenseKernelFn = unsafe extern "C" fn(
    t: f64,
    states: *const f64,
    params: *const f64,
    covariates: *const f64,
    routes: *const f64,
    derived: *const f64,
    out: *mut f64,
);

const DEFAULT_ODE_RTOL: f64 = 1e-4;
const DEFAULT_ODE_ATOL: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeBackend {
    #[cfg(feature = "dsl-jit")]
    Jit,
    #[cfg(feature = "dsl-aot-load")]
    NativeAot,
    #[cfg(feature = "dsl-wasm")]
    Wasm,
}

pub(crate) trait KernelSession {
    #[allow(clippy::too_many_arguments)]
    unsafe fn invoke_raw(
        &mut self,
        role: KernelRole,
        time: f64,
        states: *const f64,
        params: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError>;
}

pub(crate) trait RuntimeArtifact: Send + Sync + std::fmt::Debug {
    fn backend(&self) -> RuntimeBackend;
    fn has_kernel(&self, role: KernelRole) -> bool;
    fn start_session(&self) -> Result<Box<dyn KernelSession + '_>, PharmsolError>;
}

#[allow(dead_code)]
enum NativeArtifactOwner {
    #[cfg(feature = "dsl-jit")]
    Jit(Box<JITModule>),
    #[cfg(feature = "dsl-aot-load")]
    Library(Library),
}

impl std::fmt::Debug for NativeArtifactOwner {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "dsl-jit")]
            Self::Jit(_) => _f.write_str("NativeArtifactOwner::Jit(..)"),
            #[cfg(feature = "dsl-aot-load")]
            Self::Library(_) => _f.write_str("NativeArtifactOwner::Library(..)"),
            #[cfg(not(any(feature = "dsl-jit", feature = "dsl-aot-load")))]
            _ => unreachable!(
                "native artifact owner should only exist for supported native backends"
            ),
        }
    }
}

pub struct NativeExecutionArtifact {
    pub model_name: String,
    pub derive: Option<DenseKernelFn>,
    pub dynamics: Option<DenseKernelFn>,
    pub outputs: DenseKernelFn,
    pub init: Option<DenseKernelFn>,
    pub drift: Option<DenseKernelFn>,
    pub diffusion: Option<DenseKernelFn>,
    pub route_lag: Option<DenseKernelFn>,
    pub route_bioavailability: Option<DenseKernelFn>,
    _owner: Option<NativeArtifactOwner>,
}

unsafe impl Send for NativeExecutionArtifact {}
unsafe impl Sync for NativeExecutionArtifact {}

impl std::fmt::Debug for NativeExecutionArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeExecutionArtifact")
            .field("model_name", &self.model_name)
            .field("derive", &self.derive.map(|ptr| ptr as *const ()))
            .field("dynamics", &self.dynamics.map(|ptr| ptr as *const ()))
            .field("outputs", &(self.outputs as *const ()))
            .field("init", &self.init.map(|ptr| ptr as *const ()))
            .field("drift", &self.drift.map(|ptr| ptr as *const ()))
            .field("diffusion", &self.diffusion.map(|ptr| ptr as *const ()))
            .field("route_lag", &self.route_lag.map(|ptr| ptr as *const ()))
            .field(
                "route_bioavailability",
                &self.route_bioavailability.map(|ptr| ptr as *const ()),
            )
            .finish()
    }
}

impl NativeExecutionArtifact {
    #[cfg(feature = "dsl-jit")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_jit_module(
        model_name: String,
        derive: Option<DenseKernelFn>,
        dynamics: Option<DenseKernelFn>,
        outputs: DenseKernelFn,
        init: Option<DenseKernelFn>,
        drift: Option<DenseKernelFn>,
        diffusion: Option<DenseKernelFn>,
        route_lag: Option<DenseKernelFn>,
        route_bioavailability: Option<DenseKernelFn>,
        module: JITModule,
    ) -> Self {
        Self {
            model_name,
            derive,
            dynamics,
            outputs,
            init,
            drift,
            diffusion,
            route_lag,
            route_bioavailability,
            _owner: Some(NativeArtifactOwner::Jit(Box::new(module))),
        }
    }

    #[cfg(feature = "dsl-aot-load")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_library(
        model_name: String,
        derive: Option<DenseKernelFn>,
        dynamics: Option<DenseKernelFn>,
        outputs: DenseKernelFn,
        init: Option<DenseKernelFn>,
        drift: Option<DenseKernelFn>,
        diffusion: Option<DenseKernelFn>,
        route_lag: Option<DenseKernelFn>,
        route_bioavailability: Option<DenseKernelFn>,
        library: Library,
    ) -> Self {
        Self {
            model_name,
            derive,
            dynamics,
            outputs,
            init,
            drift,
            diffusion,
            route_lag,
            route_bioavailability,
            _owner: Some(NativeArtifactOwner::Library(library)),
        }
    }
}

struct NativeKernelSession<'a> {
    artifact: &'a NativeExecutionArtifact,
}

impl KernelSession for NativeKernelSession<'_> {
    unsafe fn invoke_raw(
        &mut self,
        role: KernelRole,
        time: f64,
        states: *const f64,
        params: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError> {
        let kernel = match role {
            KernelRole::Derive => self.artifact.derive,
            KernelRole::Dynamics => self.artifact.dynamics,
            KernelRole::Outputs => Some(self.artifact.outputs),
            KernelRole::Init => self.artifact.init,
            KernelRole::Drift => self.artifact.drift,
            KernelRole::Diffusion => self.artifact.diffusion,
            KernelRole::RouteLag => self.artifact.route_lag,
            KernelRole::RouteBioavailability => self.artifact.route_bioavailability,
            KernelRole::Analytical => None,
        }
        .ok_or_else(|| {
            PharmsolError::OtherError(format!(
                "model `{}` does not provide a {:?} kernel",
                self.artifact.model_name, role
            ))
        })?;

        kernel(time, states, params, covariates, routes, derived, out);
        Ok(())
    }
}

impl RuntimeArtifact for NativeExecutionArtifact {
    fn backend(&self) -> RuntimeBackend {
        match &self._owner {
            #[cfg(feature = "dsl-jit")]
            Some(NativeArtifactOwner::Jit(_)) => RuntimeBackend::Jit,
            #[cfg(feature = "dsl-aot-load")]
            Some(NativeArtifactOwner::Library(_)) => RuntimeBackend::NativeAot,
            _ => unreachable!("native execution artifacts should always retain a supported owner"),
        }
    }

    fn has_kernel(&self, role: KernelRole) -> bool {
        match role {
            KernelRole::Derive => self.derive.is_some(),
            KernelRole::Dynamics => self.dynamics.is_some(),
            KernelRole::Outputs => true,
            KernelRole::Init => self.init.is_some(),
            KernelRole::Drift => self.drift.is_some(),
            KernelRole::Diffusion => self.diffusion.is_some(),
            KernelRole::RouteLag => self.route_lag.is_some(),
            KernelRole::RouteBioavailability => self.route_bioavailability.is_some(),
            KernelRole::Analytical => false,
        }
    }

    fn start_session(&self) -> Result<Box<dyn KernelSession + '_>, PharmsolError> {
        Ok(Box::new(NativeKernelSession { artifact: self }))
    }
}

#[derive(Clone, Debug)]
struct SharedNativeModel {
    info: Arc<NativeModelInfo>,
    route_semantics: Arc<RouteInputSemantics>,
    artifact: Arc<dyn RuntimeArtifact>,
}

#[derive(Clone, Debug)]
struct RouteInputSemantics {
    bolus_destinations: Vec<Option<usize>>,
    infusion_inputs: Vec<bool>,
    injected_infusion_destinations: Vec<Option<usize>>,
}

impl RouteInputSemantics {
    fn from_model_info(info: &NativeModelInfo) -> Self {
        let mut bolus_destinations = vec![None; info.route_len];
        let mut infusion_inputs = vec![false; info.route_len];
        let mut injected_infusion_destinations = vec![None; info.route_len];

        for route in &info.routes {
            match route.kind {
                Some(RouteKind::Bolus) => {
                    bolus_destinations[route.index] = Some(route.destination_offset);
                }
                Some(RouteKind::Infusion) => {
                    infusion_inputs[route.index] = true;
                    if route.inject_input_to_destination {
                        injected_infusion_destinations[route.index] =
                            Some(route.destination_offset);
                    }
                }
                None => {
                    bolus_destinations[route.index] = Some(route.destination_offset);
                    infusion_inputs[route.index] = true;
                    if route.inject_input_to_destination {
                        injected_infusion_destinations[route.index] =
                            Some(route.destination_offset);
                    }
                }
            }
        }

        Self {
            bolus_destinations,
            infusion_inputs,
            injected_infusion_destinations,
        }
    }

    fn supports_input(&self, input: usize, kind: RouteKind) -> bool {
        match kind {
            RouteKind::Bolus => self
                .bolus_destinations
                .get(input)
                .copied()
                .flatten()
                .is_some(),
            RouteKind::Infusion => self.infusion_inputs.get(input).copied().unwrap_or(false),
        }
    }

    fn bolus_destination(&self, input: usize) -> Option<usize> {
        self.bolus_destinations.get(input).copied().flatten()
    }
}

impl SharedNativeModel {
    fn new(info: NativeModelInfo, artifact: impl RuntimeArtifact + 'static) -> Self {
        Self {
            route_semantics: Arc::new(RouteInputSemantics::from_model_info(&info)),
            info: Arc::new(info),
            artifact: Arc::new(artifact),
        }
    }

    fn route_index(&self, name: &str) -> Option<usize> {
        self.info
            .routes
            .iter()
            .find(|route| route.name == name)
            .map(|route| route.index)
    }

    fn output_index(&self, name: &str) -> Option<usize> {
        self.info
            .outputs
            .iter()
            .find(|output| output.name == name)
            .map(|output| output.index)
    }

    fn metadata_route_index_for_label(&self, label: &str) -> Option<usize> {
        self.route_index(label).or_else(|| {
            canonical_numeric_alias(label, NUMERIC_ROUTE_PREFIX)
                .and_then(|alias| self.route_index(alias.as_str()))
        })
    }

    fn metadata_output_index_for_label(&self, label: &str) -> Option<usize> {
        self.output_index(label).or_else(|| {
            canonical_numeric_alias(label, NUMERIC_OUTPUT_PREFIX)
                .and_then(|alias| self.output_index(alias.as_str()))
        })
    }

    fn validate_support_point(&self, support_point: &[f64]) -> Result<(), PharmsolError> {
        if support_point.len() != self.info.parameters.len() {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` expects {} parameter value(s), got {}",
                self.info.name,
                self.info.parameters.len(),
                support_point.len()
            )));
        }
        Ok(())
    }

    fn validate_input(&self, input: usize) -> Result<(), PharmsolError> {
        if input >= self.info.route_len {
            return Err(PharmsolError::InputOutOfRange {
                input,
                ndrugs: self.info.route_len,
            });
        }
        Ok(())
    }

    fn validate_output(&self, outeq: usize) -> Result<(), PharmsolError> {
        if outeq >= self.info.output_len {
            return Err(PharmsolError::OuteqOutOfRange {
                outeq,
                nout: self.info.output_len,
            });
        }
        Ok(())
    }

    fn validate_input_for_kind(&self, input: usize, kind: RouteKind) -> Result<(), PharmsolError> {
        self.validate_input(input)?;
        if self.route_semantics.supports_input(input, kind) {
            return Ok(());
        }

        Err(PharmsolError::UnsupportedInputRouteKind { input, kind })
    }

    fn resolve_input_label(
        &self,
        label: &InputLabel,
        kind: RouteKind,
    ) -> Result<usize, PharmsolError> {
        let input = self
            .metadata_route_index_for_label(label.as_str())
            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                label: label.to_string(),
            })?;
        self.validate_input_for_kind(input, kind)?;
        Ok(input)
    }

    fn resolve_output_label(&self, label: &OutputLabel) -> Result<usize, PharmsolError> {
        self.metadata_output_index_for_label(label.as_str())
            .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                label: label.to_string(),
            })
    }

    fn resolve_events(&self, occasion: &Occasion) -> Result<Vec<Event>, PharmsolError> {
        let mut events = occasion.process_events(None, true);

        for event in events.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    let input = self.resolve_input_label(bolus.input(), RouteKind::Bolus)?;
                    bolus.set_input(input);
                }
                Event::Infusion(infusion) => {
                    let input = self.resolve_input_label(infusion.input(), RouteKind::Infusion)?;
                    infusion.set_input(input);
                }
                Event::Observation(observation) => {
                    let outeq = self.resolve_output_label(observation.outeq())?;
                    observation.set_outeq(outeq);
                }
            }
        }

        Ok(events)
    }

    fn fill_cov_buffer(&self, covariates: &Covariates, time: f64, buf: &mut [f64]) {
        for covariate in &self.info.covariates {
            buf[covariate.index] = match covariates.get_covariate(&covariate.name) {
                Some(values) => values.interpolate(time).unwrap_or(f64::NAN),
                None => f64::NAN,
            };
        }
    }

    fn apply_route_inputs_to_rates(&self, rates: &mut [f64], route_inputs: &[f64]) {
        for (input, destination) in self
            .route_semantics
            .injected_infusion_destinations
            .iter()
            .enumerate()
        {
            if let Some(destination) = destination {
                rates[*destination] += route_inputs[input];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refresh_derived(
        &self,
        session: &mut dyn KernelSession,
        time: f64,
        state: &[f64],
        support_point: &[f64],
        covariates: &Covariates,
        route_inputs: &[f64],
        derived: &mut [f64],
        cov_buf: &mut [f64],
    ) -> Result<(), PharmsolError> {
        self.fill_cov_buffer(covariates, time, cov_buf);
        if self.artifact.has_kernel(KernelRole::Derive) {
            unsafe {
                session.invoke_raw(
                    KernelRole::Derive,
                    time,
                    state.as_ptr(),
                    support_point.as_ptr(),
                    cov_buf.as_ptr(),
                    route_inputs.as_ptr(),
                    derived.as_ptr(),
                    derived.as_mut_ptr(),
                )?;
            }
        } else {
            derived.fill(0.0);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_outputs(
        &self,
        session: &mut dyn KernelSession,
        time: f64,
        state: &[f64],
        support_point: &[f64],
        route_inputs: &[f64],
        derived: &[f64],
        cov_buf: &[f64],
        out: &mut [f64],
    ) -> Result<(), PharmsolError> {
        unsafe {
            session.invoke_raw(
                KernelRole::Outputs,
                time,
                state.as_ptr(),
                support_point.as_ptr(),
                cov_buf.as_ptr(),
                route_inputs.as_ptr(),
                derived.as_ptr(),
                out.as_mut_ptr(),
            )?;
        }
        Ok(())
    }

    fn initial_state(
        &self,
        session: &mut dyn KernelSession,
        support_point: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Result<Vec<f64>, PharmsolError> {
        let mut state = vec![0.0; self.info.state_len];
        if occasion_index == 0 {
            let mut cov_buf = vec![0.0; self.info.covariates.len()];
            let routes = vec![0.0; self.info.route_len];
            let mut derived = vec![0.0; self.info.derived_len];
            self.refresh_derived(
                session,
                0.0,
                &state,
                support_point,
                covariates,
                &routes,
                &mut derived,
                &mut cov_buf,
            )?;
            if self.artifact.has_kernel(KernelRole::Init) {
                unsafe {
                    session.invoke_raw(
                        KernelRole::Init,
                        0.0,
                        state.as_ptr(),
                        support_point.as_ptr(),
                        cov_buf.as_ptr(),
                        routes.as_ptr(),
                        derived.as_ptr(),
                        state.as_mut_ptr(),
                    )?;
                }
            }
        }
        Ok(state)
    }

    fn apply_route_properties(
        &self,
        session: &mut dyn KernelSession,
        events: &mut [Event],
        covariates: &Covariates,
        support_point: &[f64],
    ) -> Result<(), PharmsolError> {
        if !self.artifact.has_kernel(KernelRole::RouteLag)
            && !self.artifact.has_kernel(KernelRole::RouteBioavailability)
        {
            return Ok(());
        }

        let mut lag_values = vec![0.0; self.info.route_len];
        let mut fa_values = vec![1.0; self.info.route_len];
        let zero_state = vec![0.0; self.info.state_len];
        let zero_routes = vec![0.0; self.info.route_len];
        let mut cov_buf = vec![0.0; self.info.covariates.len()];
        let mut derived = vec![0.0; self.info.derived_len];

        for event in events.iter_mut() {
            if let Event::Bolus(bolus) = event {
                let input =
                    bolus
                        .input_index()
                        .ok_or_else(|| PharmsolError::UnknownInputLabel {
                            label: bolus.input().to_string(),
                        })?;
                self.validate_input_for_kind(input, RouteKind::Bolus)?;

                if self.artifact.has_kernel(KernelRole::RouteLag) {
                    lag_values.fill(0.0);
                    self.refresh_derived(
                        session,
                        bolus.time(),
                        &zero_state,
                        support_point,
                        covariates,
                        &zero_routes,
                        &mut derived,
                        &mut cov_buf,
                    )?;
                    unsafe {
                        session.invoke_raw(
                            KernelRole::RouteLag,
                            bolus.time(),
                            zero_state.as_ptr(),
                            support_point.as_ptr(),
                            cov_buf.as_ptr(),
                            zero_routes.as_ptr(),
                            derived.as_ptr(),
                            lag_values.as_mut_ptr(),
                        )?;
                    }
                    let lag = lag_values[input];
                    if lag != 0.0 {
                        *bolus.mut_time() += lag;
                    }
                }

                if self.artifact.has_kernel(KernelRole::RouteBioavailability) {
                    fa_values.fill(1.0);
                    self.refresh_derived(
                        session,
                        bolus.time(),
                        &zero_state,
                        support_point,
                        covariates,
                        &zero_routes,
                        &mut derived,
                        &mut cov_buf,
                    )?;
                    unsafe {
                        session.invoke_raw(
                            KernelRole::RouteBioavailability,
                            bolus.time(),
                            zero_state.as_ptr(),
                            support_point.as_ptr(),
                            cov_buf.as_ptr(),
                            zero_routes.as_ptr(),
                            derived.as_ptr(),
                            fa_values.as_mut_ptr(),
                        )?;
                    }
                    let factor = fa_values[input];
                    if factor != 1.0 {
                        bolus.set_amount(bolus.amount() * factor);
                    }
                }
            }
        }

        sort_events(events);
        Ok(())
    }

    fn apply_bolus(
        &self,
        state: &mut [f64],
        input: usize,
        amount: f64,
    ) -> Result<(), PharmsolError> {
        self.validate_input_for_kind(input, RouteKind::Bolus)?;
        let destination = self.route_semantics.bolus_destination(input).ok_or(
            PharmsolError::UnsupportedInputRouteKind {
                input,
                kind: RouteKind::Bolus,
            },
        )?;
        state[destination] += amount;
        Ok(())
    }

    fn observation_prediction(
        &self,
        session: &mut dyn KernelSession,
        observation: &Observation,
        state: &[f64],
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
    ) -> Result<Prediction, PharmsolError> {
        let route_inputs = active_route_inputs(infusions, observation.time(), self.info.route_len);
        let mut cov_buf = vec![0.0; self.info.covariates.len()];
        let mut derived = vec![0.0; self.info.derived_len];
        let mut outputs = vec![0.0; self.info.output_len];
        self.refresh_derived(
            session,
            observation.time(),
            state,
            support_point,
            covariates,
            &route_inputs,
            &mut derived,
            &mut cov_buf,
        )?;
        self.write_outputs(
            session,
            observation.time(),
            state,
            support_point,
            &route_inputs,
            &derived,
            &cov_buf,
            &mut outputs,
        )?;
        let outeq = observation
            .outeq_index()
            .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                label: observation.outeq().to_string(),
            })?;
        self.validate_output(outeq)?;
        Ok(observation.to_prediction(outputs[outeq], state.to_vec()))
    }
}

#[derive(Clone, Debug)]
pub struct NativeOdeModel {
    shared: Arc<SharedNativeModel>,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
    cache: Option<PredictionCache>,
    error_model_cache: Option<BoundErrorModelCache>,
}

#[derive(Clone, Debug)]
pub struct NativeAnalyticalModel {
    shared: Arc<SharedNativeModel>,
    cache: Option<PredictionCache>,
    parameter_projection: AnalyticalStructureInputKind,
}

#[derive(Clone, Debug)]
pub struct NativeSdeModel {
    shared: Arc<SharedNativeModel>,
    nparticles: usize,
    cache: Option<SdeLikelihoodCache>,
}

#[derive(Clone, Debug)]
pub enum CompiledNativeModel {
    Ode(NativeOdeModel),
    Analytical(NativeAnalyticalModel),
    Sde(NativeSdeModel),
}

impl NativeOdeModel {
    pub(crate) fn new(info: NativeModelInfo, artifact: impl RuntimeArtifact + 'static) -> Self {
        Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)),
            solver: OdeSolver::default(),
            rtol: DEFAULT_ODE_RTOL,
            atol: DEFAULT_ODE_ATOL,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
            error_model_cache: Some(BoundErrorModelCache::new(
                DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
            )),
        }
    }

    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.estimate_predictions_dense(subject, parameters.as_slice())
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();
        let support_vector: V = DVector::from_vec(support_point.to_vec()).into();

        for occasion in subject.occasions() {
            let mut events = self.shared.resolve_events(occasion)?;
            let infusions = events
                .iter()
                .filter_map(|event| match event {
                    Event::Infusion(infusion) => Some(infusion.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let session = RefCell::new(self.shared.artifact.start_session()?);
            let mut route_session = session.borrow_mut();
            self.shared.apply_route_properties(
                &mut **route_session,
                events.as_mut_slice(),
                occasion.covariates(),
                support_point,
            )?;
            drop(route_session);

            let cov_buf = RefCell::new(vec![0.0; self.shared.info.covariates.len()]);
            let derived_buf = RefCell::new(vec![0.0; self.shared.info.derived_len]);
            let shared = Arc::clone(&self.shared);
            if !shared.artifact.has_kernel(KernelRole::Dynamics) {
                return Err(PharmsolError::OtherError(format!(
                    "model `{}` does not have a dynamics kernel",
                    shared.info.name
                )));
            }
            let kernel_error = RefCell::new(None::<PharmsolError>);

            let diffeq_session = &session;
            let diffeq_error = &kernel_error;
            let diffeq = move |x: &V,
                               p: &V,
                               t: f64,
                               dx: &mut V,
                               _bolus: &V,
                               rateiv: &V,
                               cov: &Covariates| {
                if diffeq_error.borrow().is_some() {
                    dx.as_mut_slice().fill(0.0);
                    return;
                }

                let mut cov_values = cov_buf.borrow_mut();
                let mut derived_values = derived_buf.borrow_mut();
                let mut session = diffeq_session.borrow_mut();
                if let Err(error) = shared.refresh_derived(
                    &mut **session,
                    t,
                    x.as_slice(),
                    p.as_slice(),
                    cov,
                    rateiv.as_slice(),
                    &mut derived_values,
                    &mut cov_values,
                ) {
                    *diffeq_error.borrow_mut() = Some(error);
                    dx.as_mut_slice().fill(0.0);
                    return;
                }

                if let Err(error) = unsafe {
                    session.invoke_raw(
                        KernelRole::Dynamics,
                        t,
                        x.as_slice().as_ptr(),
                        p.as_slice().as_ptr(),
                        cov_values.as_ptr(),
                        rateiv.as_slice().as_ptr(),
                        derived_values.as_ptr(),
                        dx.as_mut_slice().as_mut_ptr(),
                    )
                } {
                    *diffeq_error.borrow_mut() = Some(error);
                    dx.as_mut_slice().fill(0.0);
                } else {
                    shared.apply_route_inputs_to_rates(dx.as_mut_slice(), rateiv.as_slice());
                }
            };

            let initial_state = V::from_vec(
                {
                    let mut initial_session = session.borrow_mut();
                    self.shared.initial_state(
                        &mut **initial_session,
                        support_point,
                        occasion.covariates(),
                        occasion.index(),
                    )?
                },
                NalgebraContext,
            );
            let support_point_vec = support_point.to_vec();
            let problem = OdeBuilder::<M>::new()
                .atol(vec![self.atol])
                .rtol(self.rtol)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point_vec.clone())
                .build_from_eqn(PMProblem::with_params_v(
                    diffeq,
                    self.shared.info.state_len,
                    self.shared.info.route_len,
                    support_vector.clone(),
                    occasion.covariates(),
                    infusions.iter(),
                    initial_state,
                )?)?;

            macro_rules! run_solver {
                ($solver:expr) => {{
                    let mut solver = $solver?;
                    self.run_events(
                        &mut solver,
                        &events,
                        support_point,
                        occasion.covariates(),
                        infusions.as_slice(),
                        &mut output,
                        &session,
                        &kernel_error,
                    )?;
                }};
            }

            match &self.solver {
                OdeSolver::Bdf => run_solver!(problem.bdf::<diffsol::NalgebraLU<f64>>()),
                OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45) => run_solver!(problem.tsit45()),
                OdeSolver::Sdirk(SdirkTableau::TrBdf2) => {
                    run_solver!(problem.tr_bdf2::<diffsol::NalgebraLU<f64>>())
                }
                OdeSolver::Sdirk(SdirkTableau::Esdirk34) => {
                    run_solver!(problem.esdirk34::<diffsol::NalgebraLU<f64>>())
                }
            }

            if let Some(error) = kernel_error.into_inner() {
                return Err(error);
            }
        }

        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_events<'a, F, S>(
        &self,
        solver: &mut S,
        events: &[Event],
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        output: &mut SubjectPredictions,
        session: &RefCell<Box<dyn KernelSession + '_>>,
        kernel_error: &RefCell<Option<PharmsolError>>,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    {
        for (index, event) in events.iter().enumerate() {
            match event {
                Event::Bolus(bolus) => {
                    let input =
                        bolus
                            .input_index()
                            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
                            })?;
                    self.shared.apply_bolus(
                        solver.state_mut().y.as_mut_slice(),
                        input,
                        bolus.amount(),
                    )?;
                }
                Event::Infusion(_) => {}
                Event::Observation(observation) => {
                    if kernel_error.borrow().is_some() {
                        return Err(kernel_error.borrow_mut().take().unwrap());
                    }
                    let prediction = self.shared.observation_prediction(
                        &mut **session.borrow_mut(),
                        observation,
                        solver.state().y.as_slice(),
                        support_point,
                        covariates,
                        infusions,
                    )?;
                    output.add_prediction(prediction);
                }
            }

            if let Some(next_event) = events.get(index + 1) {
                if event.time() == next_event.time() {
                    continue;
                }

                match solver.set_stop_time(next_event.time()) {
                    Ok(_) => loop {
                        match solver.step() {
                            Ok(_) if kernel_error.borrow().is_some() => {
                                return Err(kernel_error.borrow_mut().take().unwrap());
                            }
                            Ok(OdeSolverStopReason::InternalTimestep) => continue,
                            Ok(OdeSolverStopReason::TstopReached) => break,
                            Err(diffsol::error::DiffsolError::OdeSolverError(
                                OdeSolverError::StepSizeTooSmall { time },
                            )) => {
                                return Err(PharmsolError::OtherError(format!(
                                    "ODE solver step size went to zero at t = {time:.4} (target t = {:.4}).",
                                    next_event.time()
                                )));
                            }
                            Err(_) | Ok(_) => {
                                return Err(PharmsolError::OtherError(
                                    "unexpected solver error".to_string(),
                                ));
                            }
                        }
                    },
                    Err(diffsol::error::DiffsolError::OdeSolverError(
                        OdeSolverError::StopTimeAtCurrentTime,
                    )) => continue,
                    Err(_) => {
                        return Err(PharmsolError::OtherError(
                            "unexpected solver error".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

fn runtime_no_lag(_: &V, _: T, _: &Covariates) -> HashMap<usize, T> {
    HashMap::new()
}

fn runtime_no_fa(_: &V, _: T, _: &Covariates) -> HashMap<usize, T> {
    HashMap::new()
}

#[inline(always)]
fn runtime_ode_predictions(
    model: &NativeOdeModel,
    subject: &Subject,
    support_point: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = &model.cache {
        let key = (
            subject.hash(),
            crate::simulator::equation::parameters_hash(support_point),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = model.estimate_predictions_dense(subject, support_point)?;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        model.estimate_predictions_dense(subject, support_point)
    }
}

impl crate::simulator::equation::Cache for NativeOdeModel {
    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
        if let Some(cache) = &self.error_model_cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self.error_model_cache = None;
        self
    }
}

impl EquationTypes for NativeOdeModel {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for NativeOdeModel {
    fn lag(&self) -> &Lag {
        &(runtime_no_lag as Lag)
    }

    fn fa(&self) -> &Fa {
        &(runtime_no_fa as Fa)
    }

    fn get_nstates(&self) -> usize {
        self.shared.info.state_len
    }

    fn get_ndrugs(&self) -> usize {
        self.shared.info.route_len
    }

    fn get_nouteqs(&self) -> usize {
        self.shared.info.output_len
    }

    fn metadata(&self) -> Option<&crate::ValidatedModelMetadata> {
        None
    }

    fn solve(
        &self,
        _state: &mut Self::S,
        _support_point: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve is not used for runtime ODE models")
    }

    fn process_observation(
        &self,
        _support_point: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation is not used for runtime ODE models")
    }

    fn initial_state(
        &self,
        _support_point: &[f64],
        _covariates: &Covariates,
        _occasion_index: usize,
    ) -> Self::S {
        V::zeros(self.shared.info.state_len, NalgebraContext)
    }
}

impl Equation for NativeOdeModel {
    fn bound_error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.error_model_cache.as_ref()
    }

    fn estimate_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        Ok(self
            .estimate_log_likelihood(subject, parameters, error_models)?
            .exp())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let predictions = runtime_ode_predictions(self, subject, parameters.as_slice())?;
        predictions.log_likelihood(&bound_error_models)
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters)
    }

    fn estimate_log_likelihood_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let predictions = runtime_ode_predictions(self, subject, parameters)?;
        predictions.log_likelihood(&bound_error_models)
    }

    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let bound_error_models = match error_models {
            Some(error_models) => Some(self.bind_error_models(error_models)?),
            None => None,
        };

        let predictions = runtime_ode_predictions(self, subject, parameters)?;
        let likelihood = match bound_error_models.as_ref() {
            Some(error_models) => Some(predictions.log_likelihood(error_models)?.exp()),
            None => None,
        };
        Ok((predictions, likelihood))
    }

    fn kind() -> EqnKind {
        EqnKind::ODE
    }

    fn assay_error_models(&self) -> AssayErrorModels {
        AssayErrorModels::with_output_names(
            self.info()
                .outputs
                .iter()
                .map(|output| output.name.as_str()),
        )
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters.as_slice())
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let support_point = parameters.as_slice();
        let bound_error_models = match error_models {
            Some(error_models) => Some(self.bind_error_models(error_models)?),
            None => None,
        };

        let predictions = runtime_ode_predictions(self, subject, support_point)?;
        let likelihood = match bound_error_models.as_ref() {
            Some(error_models) => Some(predictions.log_likelihood(error_models)?.exp()),
            None => None,
        };
        Ok((predictions, likelihood))
    }
}

impl NativeAnalyticalModel {
    pub(crate) fn new(
        info: NativeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let parameter_projection = build_analytical_parameter_projection(&info)?;
        Ok(Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)),
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
            parameter_projection,
        })
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<SubjectPredictions, PharmsolError> {
        let support_point = parameters.as_slice();
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();

        for occasion in subject.occasions() {
            let mut events = self.shared.resolve_events(occasion)?;
            let infusions = events
                .iter()
                .filter_map(|event| match event {
                    Event::Infusion(infusion) => Some(infusion.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut session = self.shared.artifact.start_session()?;
            self.shared.apply_route_properties(
                &mut *session,
                events.as_mut_slice(),
                occasion.covariates(),
                support_point,
            )?;

            let mut state = self.shared.initial_state(
                &mut *session,
                support_point,
                occasion.covariates(),
                occasion.index(),
            )?;

            for (index, event) in events.iter().enumerate() {
                match event {
                    Event::Bolus(bolus) => {
                        let input = bolus.input_index().ok_or_else(|| {
                            PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
                            }
                        })?;
                        self.shared.apply_bolus(&mut state, input, bolus.amount())?
                    }
                    Event::Infusion(_) => {}
                    Event::Observation(observation) => {
                        output.add_prediction(self.shared.observation_prediction(
                            &mut *session,
                            observation,
                            &state,
                            support_point,
                            occasion.covariates(),
                            infusions.as_slice(),
                        )?);
                    }
                }

                if let Some(next_event) = events.get(index + 1) {
                    self.solve_interval(
                        &mut *session,
                        &mut state,
                        support_point,
                        occasion.covariates(),
                        infusions.as_slice(),
                        event.time(),
                        next_event.time(),
                    )?;
                }
            }
        }

        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_interval(
        &self,
        session: &mut dyn KernelSession,
        state: &mut [f64],
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError> {
        if start_time == end_time {
            return Ok(());
        }

        let mut breakpoints = vec![start_time, end_time];
        for infusion in infusions {
            let begin = infusion.time();
            let finish = infusion.time() + infusion.duration();
            if begin > start_time && begin < end_time {
                breakpoints.push(begin);
            }
            if finish > start_time && finish < end_time {
                breakpoints.push(finish);
            }
        }
        breakpoints.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        breakpoints.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() < 1e-12);

        let mut current = breakpoints[0];
        let mut cov_buf = vec![0.0; self.shared.info.covariates.len()];
        let mut derived = vec![0.0; self.shared.info.derived_len];

        for next in breakpoints.iter().copied().skip(1) {
            let dt = next - current;
            let route_inputs =
                interval_route_inputs(infusions, current, next, self.shared.info.route_len);
            self.shared.refresh_derived(
                session,
                next,
                state,
                support_point,
                covariates,
                &route_inputs,
                &mut derived,
                &mut cov_buf,
            )?;
            let projected =
                project_analytical_parameters(&self.parameter_projection, support_point, &derived);
            let next_state = apply_analytical_kernel(
                self.shared.info.analytical.ok_or_else(|| {
                    PharmsolError::OtherError(format!(
                        "model `{}` does not declare an analytical kernel",
                        self.shared.info.name
                    ))
                })?,
                state,
                &projected,
                dt,
                &route_inputs,
                covariates,
            );
            state.copy_from_slice(next_state.as_slice());
            current = next;
        }

        Ok(())
    }
}

#[inline(always)]
fn runtime_analytical_predictions(
    model: &NativeAnalyticalModel,
    subject: &Subject,
    support_point: &[f64],
) -> Result<SubjectPredictions, PharmsolError> {
    if let Some(cache) = &model.cache {
        let key = (
            subject.hash(),
            crate::simulator::equation::spphash(support_point),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = model.estimate_predictions(subject, support_point)?;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        model.estimate_predictions(subject, support_point)
    }
}

impl crate::simulator::equation::Cache for NativeAnalyticalModel {
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

impl EquationTypes for NativeAnalyticalModel {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for NativeAnalyticalModel {
    fn lag(&self) -> &Lag {
        &(runtime_no_lag as Lag)
    }

    fn fa(&self) -> &Fa {
        &(runtime_no_fa as Fa)
    }

    fn get_nstates(&self) -> usize {
        self.shared.info.state_len
    }

    fn get_ndrugs(&self) -> usize {
        self.shared.info.route_len
    }

    fn get_nouteqs(&self) -> usize {
        self.shared.info.output_len
    }

    fn metadata(&self) -> Option<&crate::ValidatedModelMetadata> {
        None
    }

    fn solve(
        &self,
        _state: &mut Self::S,
        _support_point: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve is not used for runtime analytical models")
    }

    fn process_observation(
        &self,
        _support_point: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation is not used for runtime analytical models")
    }

    fn initial_state(
        &self,
        _support_point: &[f64],
        _covariates: &Covariates,
        _occasion_index: usize,
    ) -> Self::S {
        V::zeros(self.shared.info.state_len, NalgebraContext)
    }
}

impl Equation for NativeAnalyticalModel {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        Ok(self
            .estimate_log_likelihood(subject, support_point, error_models)?
            .exp())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        let predictions = runtime_analytical_predictions(self, subject, support_point)?;
        predictions.log_likelihood(&bound_error_models)
    }

    fn kind() -> EqnKind {
        EqnKind::Analytical
    }

    fn assay_error_models(&self) -> AssayErrorModels {
        AssayErrorModels::with_output_names(
            self.info()
                .outputs
                .iter()
                .map(|output| output.name.as_str()),
        )
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_analytical_predictions(self, subject, support_point)
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let bound_error_models = match error_models {
            Some(error_models) => Some(self.bind_error_models(error_models)?),
            None => None,
        };

        let predictions = runtime_analytical_predictions(self, subject, support_point)?;
        let likelihood = match bound_error_models.as_ref() {
            Some(error_models) => Some(predictions.log_likelihood(error_models)?.exp()),
            None => None,
        };
        Ok((predictions, likelihood))
    }
}

impl NativeSdeModel {
    pub(crate) fn new(info: NativeModelInfo, artifact: impl RuntimeArtifact + 'static) -> Self {
        let nparticles = info.particles.unwrap_or(1);
        Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)),
            nparticles,
            cache: Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    pub fn with_particles(mut self, nparticles: usize) -> Self {
        self.nparticles = nparticles;
        self
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Array2<Prediction>, PharmsolError> {
        let support_point = parameters.as_slice();
        self.shared.validate_support_point(support_point)?;
        let mut output = Array2::from_shape_fn((self.nparticles, 0), |_| Prediction::default());

        for occasion in subject.occasions() {
            let mut events = self.shared.resolve_events(occasion)?;
            let infusions = events
                .iter()
                .filter_map(|event| match event {
                    Event::Infusion(infusion) => Some(infusion.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut session = self.shared.artifact.start_session()?;
            self.shared.apply_route_properties(
                &mut *session,
                events.as_mut_slice(),
                occasion.covariates(),
                support_point,
            )?;

            let initial = self.shared.initial_state(
                &mut *session,
                support_point,
                occasion.covariates(),
                occasion.index(),
            )?;
            let mut particles = (0..self.nparticles)
                .map(|_| DVector::from_vec(initial.clone()))
                .collect::<Vec<_>>();

            for (index, event) in events.iter().enumerate() {
                match event {
                    Event::Bolus(bolus) => {
                        let input = bolus.input_index().ok_or_else(|| {
                            PharmsolError::UnknownInputLabel {
                                label: bolus.input().to_string(),
                            }
                        })?;
                        for particle in &mut particles {
                            self.shared.apply_bolus(
                                particle.as_mut_slice(),
                                input,
                                bolus.amount(),
                            )?;
                        }
                    }
                    Event::Infusion(_) => {}
                    Event::Observation(observation) => {
                        let mut column = Vec::with_capacity(self.nparticles);
                        for particle in &particles {
                            column.push(self.shared.observation_prediction(
                                &mut *session,
                                observation,
                                particle.as_slice(),
                                support_point,
                                occasion.covariates(),
                                infusions.as_slice(),
                            )?);
                        }
                        let column = Array2::from_shape_vec((self.nparticles, 1), column)?;
                        output = concatenate(Axis(1), &[output.view(), column.view()]).unwrap();
                    }
                }

                if let Some(next_event) = events.get(index + 1) {
                    self.solve_interval(
                        &mut particles,
                        support_point,
                        occasion.covariates(),
                        infusions.as_slice(),
                        event.time(),
                        next_event.time(),
                    )?;
                }
            }
        }

        Ok(output)
    }

    fn solve_interval(
        &self,
        particles: &mut [DVector<f64>],
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &[Infusion],
        start_time: f64,
        end_time: f64,
    ) -> Result<(), PharmsolError> {
        if start_time == end_time {
            return Ok(());
        }

        let shared = Arc::clone(&self.shared);
        let support = Arc::new(support_point.to_vec());
        let infusion_events = Arc::new(infusions.to_vec());
        let covariates = covariates.clone();
        if !shared.artifact.has_kernel(KernelRole::Drift) {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` does not have a drift kernel",
                shared.info.name
            )));
        }
        if !shared.artifact.has_kernel(KernelRole::Diffusion) {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` does not have a diffusion kernel",
                shared.info.name
            )));
        }

        particles
            .par_iter_mut()
            .try_for_each(|particle| -> Result<(), PharmsolError> {
                let shared = Arc::clone(&shared);
                let support = Arc::clone(&support);
                let infusions = Arc::clone(&infusion_events);
                let covariates = covariates.clone();
                let shared_for_diffusion = Arc::clone(&shared);
                let support_for_diffusion = Arc::clone(&support);
                let infusions_for_diffusion = Arc::clone(&infusions);
                let covariates_for_diffusion = covariates.clone();
                let route_len = shared.info.route_len;
                let cov_len = shared.info.covariates.len();
                let derived_len = shared.info.derived_len;
                let drift_state = particle.clone();
                let artifact = Arc::clone(&shared.artifact);
                let session = RefCell::new(artifact.start_session()?);
                let kernel_error = RefCell::new(None::<PharmsolError>);
                let drift_session = &session;
                let diffusion_session = &session;
                let drift_error = &kernel_error;
                let diffusion_error = &kernel_error;
                let next = simulate_sde_event_with(
                    move |time, state, out| {
                        if drift_error.borrow().is_some() {
                            out.fill(0.0);
                            return;
                        }
                        let route_inputs = active_route_inputs(&infusions, time, route_len);
                        let mut cov_buf = vec![0.0; cov_len];
                        let mut derived = vec![0.0; derived_len];
                        let mut session = drift_session.borrow_mut();
                        if let Err(error) = shared.refresh_derived(
                            &mut **session,
                            time,
                            state.as_slice(),
                            &support,
                            &covariates,
                            &route_inputs,
                            &mut derived,
                            &mut cov_buf,
                        ) {
                            *drift_error.borrow_mut() = Some(error);
                            out.fill(0.0);
                            return;
                        }
                        if let Err(error) = unsafe {
                            session.invoke_raw(
                                KernelRole::Drift,
                                time,
                                state.as_ptr(),
                                support.as_ptr(),
                                cov_buf.as_ptr(),
                                route_inputs.as_ptr(),
                                derived.as_ptr(),
                                out.as_mut_ptr(),
                            )
                        } {
                            *drift_error.borrow_mut() = Some(error);
                            out.fill(0.0);
                        } else {
                            shared.apply_route_inputs_to_rates(out.as_mut_slice(), &route_inputs);
                        }
                    },
                    move |time, state, out| {
                        if diffusion_error.borrow().is_some() {
                            out.fill(0.0);
                            return;
                        }
                        let route_inputs =
                            active_route_inputs(&infusions_for_diffusion, time, route_len);
                        let mut cov_buf = vec![0.0; cov_len];
                        let mut derived = vec![0.0; derived_len];
                        let mut session = diffusion_session.borrow_mut();
                        if let Err(error) = shared_for_diffusion.refresh_derived(
                            &mut **session,
                            time,
                            state.as_slice(),
                            &support_for_diffusion,
                            &covariates_for_diffusion,
                            &route_inputs,
                            &mut derived,
                            &mut cov_buf,
                        ) {
                            *diffusion_error.borrow_mut() = Some(error);
                            out.fill(0.0);
                            return;
                        }
                        if let Err(error) = unsafe {
                            session.invoke_raw(
                                KernelRole::Diffusion,
                                time,
                                state.as_ptr(),
                                support_for_diffusion.as_ptr(),
                                cov_buf.as_ptr(),
                                route_inputs.as_ptr(),
                                derived.as_ptr(),
                                out.as_mut_ptr(),
                            )
                        } {
                            *diffusion_error.borrow_mut() = Some(error);
                            out.fill(0.0);
                        }
                    },
                    drift_state,
                    start_time,
                    end_time,
                );
                if let Some(error) = kernel_error.into_inner() {
                    return Err(error);
                }
                *particle = next;
                Ok(())
            })?;

        Ok(())
    }
}

#[inline(always)]
fn runtime_sde_log_likelihood(
    model: &NativeSdeModel,
    subject: &Subject,
    support_point: &[f64],
    error_models: &AssayErrorModels,
) -> Result<f64, PharmsolError> {
    if let Some(cache) = &model.cache {
        let key = (
            subject.hash(),
            crate::simulator::equation::spphash(support_point),
            error_models.hash(),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let predictions = model.estimate_predictions(subject, support_point)?;
        let log_lik = predictions.log_likelihood(error_models)?;
        cache.insert(key, log_lik);
        Ok(log_lik)
    } else {
        let predictions = model.estimate_predictions(subject, support_point)?;
        predictions.log_likelihood(error_models)
    }
}

impl crate::simulator::equation::Cache for NativeSdeModel {
    fn with_cache_capacity(mut self, size: u64) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(size));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(SdeLikelihoodCache::new(DEFAULT_CACHE_SIZE));
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

impl EquationTypes for NativeSdeModel {
    type S = Vec<DVector<f64>>;
    type P = Array2<Prediction>;
}

impl EquationPriv for NativeSdeModel {
    fn lag(&self) -> &Lag {
        &(runtime_no_lag as Lag)
    }

    fn fa(&self) -> &Fa {
        &(runtime_no_fa as Fa)
    }

    fn get_nstates(&self) -> usize {
        self.shared.info.state_len
    }

    fn get_ndrugs(&self) -> usize {
        self.shared.info.route_len
    }

    fn get_nouteqs(&self) -> usize {
        self.shared.info.output_len
    }

    fn nparticles(&self) -> usize {
        self.nparticles
    }

    fn is_sde(&self) -> bool {
        true
    }

    fn metadata(&self) -> Option<&crate::ValidatedModelMetadata> {
        None
    }

    fn solve(
        &self,
        _state: &mut Self::S,
        _support_point: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve is not used for runtime SDE models")
    }

    fn process_observation(
        &self,
        _support_point: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation is not used for runtime SDE models")
    }

    fn initial_state(
        &self,
        _support_point: &[f64],
        _covariates: &Covariates,
        _occasion_index: usize,
    ) -> Self::S {
        vec![DVector::zeros(self.shared.info.state_len); self.nparticles]
    }
}

impl Equation for NativeSdeModel {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let log_lik = self.estimate_log_likelihood(subject, support_point, error_models)?;
        Ok(log_lik.exp())
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        let bound_error_models = self.bind_error_models(error_models)?;
        runtime_sde_log_likelihood(self, subject, support_point, &bound_error_models)
    }

    fn kind() -> EqnKind {
        EqnKind::SDE
    }

    fn assay_error_models(&self) -> AssayErrorModels {
        AssayErrorModels::with_output_names(
            self.info()
                .outputs
                .iter()
                .map(|output| output.name.as_str()),
        )
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        NativeSdeModel::estimate_predictions(self, subject, support_point)
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &[f64],
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let bound_error_models = match error_models {
            Some(error_models) => Some(self.bind_error_models(error_models)?),
            None => None,
        };

        let predictions = NativeSdeModel::estimate_predictions(self, subject, support_point)?;
        let likelihood = match bound_error_models.as_ref() {
            Some(error_models) => Some(predictions.log_likelihood(error_models)?.exp()),
            None => None,
        };
        Ok((predictions, likelihood))
    }
}

fn active_route_inputs(infusions: &[Infusion], time: f64, route_len: usize) -> Vec<f64> {
    let mut values = vec![0.0; route_len];
    for infusion in infusions {
        let input = infusion
            .input_index()
            .expect("resolved infusions should use numeric input labels");
        if input < route_len
            && time >= infusion.time()
            && time <= infusion.time() + infusion.duration()
        {
            values[input] += infusion.amount() / infusion.duration();
        }
    }
    values
}

fn interval_route_inputs(
    infusions: &[Infusion],
    start_time: f64,
    end_time: f64,
    route_len: usize,
) -> Vec<f64> {
    let mut values = vec![0.0; route_len];
    for infusion in infusions {
        let finish = infusion.time() + infusion.duration();
        let input = infusion
            .input_index()
            .expect("resolved infusions should use numeric input labels");
        if input < route_len && start_time >= infusion.time() && end_time <= finish {
            values[input] += infusion.amount() / infusion.duration();
        }
    }
    values
}

fn sort_events(events: &mut [Event]) {
    events.sort_by(|lhs, rhs| {
        fn order(event: &Event) -> u8 {
            match event {
                Event::Observation(_) => 1,
                Event::Bolus(_) => 2,
                Event::Infusion(_) => 3,
            }
        }

        match lhs.time().partial_cmp(&rhs.time()) {
            Some(std::cmp::Ordering::Equal) => order(lhs).cmp(&order(rhs)),
            Some(ordering) => ordering,
            None => std::cmp::Ordering::Equal,
        }
    });
}

fn canonical_numeric_alias(label: &str, prefix: &str) -> Option<String> {
    if label.is_empty() || !label.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(format!("{prefix}{label}"))
}

fn build_analytical_parameter_projection(
    info: &NativeModelInfo,
) -> Result<AnalyticalStructureInputKind, PharmsolError> {
    let kernel = info.analytical.ok_or_else(|| {
        PharmsolError::OtherError(format!(
            "model `{}` does not declare an analytical kernel",
            info.name
        ))
    })?;
    if info.derived.len() != info.derived_len {
        return Err(PharmsolError::OtherError(format!(
            "compiled analytical model `{}` has inconsistent derived metadata: {} declared name(s), {} derived slot(s)",
            info.name,
            info.derived.len(),
            info.derived_len
        )));
    }

    AnalyticalStructureInputPlan::for_kernel(
        kernel,
        info.parameters.iter().map(String::as_str),
        info.derived.iter().map(String::as_str),
    )
    .map(|plan| plan.kind().clone())
    .map_err(|error| {
        PharmsolError::OtherError(format!(
            "compiled analytical model `{}` has invalid structure inputs: {error}",
            info.name
        ))
    })
}

fn project_analytical_parameters(
    projection: &AnalyticalStructureInputKind,
    support_point: &[f64],
    derived: &[f64],
) -> V {
    match projection {
        AnalyticalStructureInputKind::AllPrimary { indices, identity } => {
            let values = if *identity {
                support_point[..indices.len()].to_vec()
            } else {
                indices.iter().map(|&index| support_point[index]).collect()
            };
            V::from_vec(values, NalgebraContext)
        }
        AnalyticalStructureInputKind::AllDerived { indices, identity } => {
            let values = if *identity {
                derived[..indices.len()].to_vec()
            } else {
                indices.iter().map(|&index| derived[index]).collect()
            };
            V::from_vec(values, NalgebraContext)
        }
        AnalyticalStructureInputKind::Mixed { bindings } => V::from_vec(
            bindings
                .iter()
                .map(|binding| match binding.source {
                    pharmsol_dsl::AnalyticalStructureInputSource::Primary => {
                        support_point[binding.index]
                    }
                    pharmsol_dsl::AnalyticalStructureInputSource::Derived => derived[binding.index],
                })
                .collect(),
            NalgebraContext,
        ),
    }
}

fn apply_analytical_kernel(
    kernel: AnalyticalKernel,
    state: &[f64],
    params: &V,
    dt: f64,
    route_inputs: &[f64],
    covariates: &Covariates,
) -> V {
    let state = V::from_vec(state.to_vec(), NalgebraContext);
    let route_inputs = V::from_vec(route_inputs.to_vec(), NalgebraContext);
    match kernel {
        AnalyticalKernel::OneCompartment => {
            crate::simulator::equation::analytical::one_compartment(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::OneCompartmentCl => {
            crate::simulator::equation::analytical::one_compartment_cl(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::OneCompartmentClWithAbsorption => {
            crate::simulator::equation::analytical::one_compartment_cl_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::OneCompartmentWithAbsorption => {
            crate::simulator::equation::analytical::one_compartment_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::TwoCompartments => {
            crate::simulator::equation::analytical::two_compartments(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::TwoCompartmentsCl => {
            crate::simulator::equation::analytical::two_compartments_cl(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::TwoCompartmentsClWithAbsorption => {
            crate::simulator::equation::analytical::two_compartments_cl_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::TwoCompartmentsWithAbsorption => {
            crate::simulator::equation::analytical::two_compartments_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::ThreeCompartments => {
            crate::simulator::equation::analytical::three_compartments(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::ThreeCompartmentsCl => {
            crate::simulator::equation::analytical::three_compartments_cl(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::ThreeCompartmentsClWithAbsorption => {
            crate::simulator::equation::analytical::three_compartments_cl_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
        AnalyticalKernel::ThreeCompartmentsWithAbsorption => {
            crate::simulator::equation::analytical::three_compartments_with_absorption(
                &state,
                params,
                dt,
                &route_inputs,
                covariates,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_analytical_parameter_projection, canonical_numeric_alias,
        project_analytical_parameters, KernelSession, NativeAnalyticalModel, NativeModelInfo,
        NativeOutputInfo, NativeRouteInfo, RuntimeArtifact, RuntimeBackend, SharedNativeModel,
        NUMERIC_OUTPUT_PREFIX, NUMERIC_ROUTE_PREFIX,
    };
    use crate::PharmsolError;
    use diffsol::VectorHost;
    use pharmsol_dsl::execution::KernelRole;
    use pharmsol_dsl::{AnalyticalKernel, AnalyticalStructureInputKind, ModelKind, RouteKind};

    #[derive(Debug)]
    struct DummyArtifact;

    impl RuntimeArtifact for DummyArtifact {
        fn backend(&self) -> RuntimeBackend {
            panic!("dummy artifact backend should not be used in tests")
        }

        fn has_kernel(&self, _role: KernelRole) -> bool {
            false
        }

        fn start_session(&self) -> Result<Box<dyn KernelSession + '_>, PharmsolError> {
            panic!("dummy artifact sessions should not be used in tests")
        }
    }

    fn bolus_only_shared_model() -> SharedNativeModel {
        SharedNativeModel::new(
            NativeModelInfo {
                name: "bolus_only".to_string(),
                kind: ModelKind::Ode,
                parameters: Vec::new(),
                derived: Vec::new(),
                covariates: Vec::new(),
                routes: vec![NativeRouteInfo {
                    name: "oral".to_string(),
                    declaration_index: 0,
                    index: 0,
                    kind: Some(RouteKind::Bolus),
                    destination_offset: 0,
                    inject_input_to_destination: false,
                }],
                outputs: vec![NativeOutputInfo {
                    name: "cp".to_string(),
                    index: 0,
                }],
                state_len: 1,
                derived_len: 0,
                output_len: 1,
                route_len: 1,
                analytical: None,
                particles: None,
            },
            DummyArtifact,
        )
    }

    fn analytical_model_info(
        parameters: &[&str],
        derived: &[&str],
        kernel: AnalyticalKernel,
    ) -> NativeModelInfo {
        NativeModelInfo {
            name: "analytical_projection".to_string(),
            kind: ModelKind::Analytical,
            parameters: parameters.iter().map(|name| (*name).to_string()).collect(),
            derived: derived.iter().map(|name| (*name).to_string()).collect(),
            covariates: Vec::new(),
            routes: Vec::new(),
            outputs: Vec::new(),
            state_len: kernel.state_count(),
            derived_len: derived.len(),
            output_len: 0,
            route_len: 0,
            analytical: Some(kernel),
            particles: None,
        }
    }

    fn analytical_projection_values(
        model: &NativeAnalyticalModel,
        support_point: &[f64],
        derived: &[f64],
    ) -> Vec<f64> {
        project_analytical_parameters(&model.parameter_projection, support_point, derived)
            .as_slice()
            .to_vec()
    }

    #[test]
    fn canonical_numeric_alias_maps_bare_numeric_labels_to_contextual_prefixes() {
        assert_eq!(
            canonical_numeric_alias("1", NUMERIC_ROUTE_PREFIX),
            Some("input_1".to_string())
        );
        assert_eq!(
            canonical_numeric_alias("10", NUMERIC_OUTPUT_PREFIX),
            Some("outeq_10".to_string())
        );
    }

    #[test]
    fn canonical_numeric_alias_ignores_symbolic_and_prefixed_labels() {
        assert_eq!(canonical_numeric_alias("iv", NUMERIC_ROUTE_PREFIX), None);
        assert_eq!(
            canonical_numeric_alias("input_1", NUMERIC_ROUTE_PREFIX),
            None
        );
        assert_eq!(
            canonical_numeric_alias("outeq_2", NUMERIC_OUTPUT_PREFIX),
            None
        );
    }

    #[test]
    fn validate_input_for_kind_reports_structured_route_kind_error() {
        let model = bolus_only_shared_model();

        let error = model
            .validate_input_for_kind(0, RouteKind::Infusion)
            .expect_err("bolus-only route should reject infusion usage");

        assert!(matches!(
            error,
            PharmsolError::UnsupportedInputRouteKind {
                input: 0,
                kind: RouteKind::Infusion,
            }
        ));
    }

    #[test]
    fn compiled_analytical_projection_uses_primary_identity_order() {
        let model = NativeAnalyticalModel::new(
            analytical_model_info(
                &["ka", "ke", "v"],
                &[],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model.parameter_projection,
            AnalyticalStructureInputKind::AllPrimary { identity: true, .. }
        ));
        assert_eq!(
            analytical_projection_values(&model, &[1.0, 0.15, 25.0], &[]),
            vec![1.0, 0.15]
        );
    }

    #[test]
    fn compiled_analytical_projection_reorders_all_derived_inputs() {
        let model = NativeAnalyticalModel::new(
            analytical_model_info(
                &["ke0", "v"],
                &["ke", "ka"],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model.parameter_projection,
            AnalyticalStructureInputKind::AllDerived {
                identity: false,
                ..
            }
        ));
        assert_eq!(
            analytical_projection_values(&model, &[0.15, 25.0], &[0.15, 1.0]),
            vec![1.0, 0.15]
        );
    }

    #[test]
    fn compiled_analytical_projection_gathers_mixed_primary_and_derived_inputs() {
        let model = NativeAnalyticalModel::new(
            analytical_model_info(
                &["ka", "v", "ke0"],
                &["ke"],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model.parameter_projection,
            AnalyticalStructureInputKind::Mixed { .. }
        ));
        assert_eq!(
            analytical_projection_values(&model, &[1.0, 25.0, 0.2], &[0.15]),
            vec![1.0, 0.15]
        );
    }

    #[test]
    fn compiled_analytical_projection_reports_missing_required_name_at_setup() {
        let error = NativeAnalyticalModel::new(
            analytical_model_info(
                &["ka", "v"],
                &[],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect_err("missing required name must fail at setup");

        let message = error.to_string();
        assert!(message.contains(
            "compiled analytical model `analytical_projection` has invalid structure inputs"
        ));
        assert!(message
            .contains("analytical structure `one_compartment_with_absorption` requires `ke`"));
        assert!(message.contains("declare it in `params` or `derived`"));
    }

    #[test]
    fn compiled_analytical_projection_reports_conflicting_primary_and_derived_name_at_setup() {
        let error = NativeAnalyticalModel::new(
            analytical_model_info(
                &["ka", "ke", "v"],
                &["ke"],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect_err("conflicting name must fail at setup");

        assert!(error.to_string().contains(
            "compiled analytical model `analytical_projection` has invalid structure inputs: `ke` is declared in both `params` and `derived`"
        ));
    }

    #[test]
    fn compiled_analytical_projection_rejects_inconsistent_derived_metadata() {
        let mut info = analytical_model_info(
            &["ka", "ke0", "v"],
            &["ke"],
            AnalyticalKernel::OneCompartmentWithAbsorption,
        );
        info.derived_len = 2;

        let error = build_analytical_parameter_projection(&info)
            .expect_err("inconsistent derived metadata must fail");

        assert!(error.to_string().contains(
            "compiled analytical model `analytical_projection` has inconsistent derived metadata"
        ));
    }
}
