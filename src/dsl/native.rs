use std::cell::RefCell;
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
use pharmsol_dsl::{AnalyticalKernel, RouteKind};

pub use super::model_info::{
    NativeCovariateInfo, NativeModelInfo, NativeOutputInfo, NativeRouteInfo,
};
use crate::{
    data::{Covariates, Infusion},
    simulator::{
        equation::{
            ode::{closure_helpers::PMProblem, ExplicitRkTableau, OdeSolver, SdirkTableau},
            sde::simulate_sde_event_with,
        },
        likelihood::{Prediction, SubjectPredictions},
        M, V,
    },
    Event, Observation, PharmsolError, Subject,
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

    fn validate_input_for_kind(&self, input: usize, kind: RouteKind) -> Result<(), PharmsolError> {
        self.validate_input(input)?;
        if self.route_semantics.supports_input(input, kind) {
            return Ok(());
        }

        Err(PharmsolError::OtherError(format!(
            "model `{}` does not declare a {:?} route for input channel {}",
            self.info.name, kind, input
        )))
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
                self.validate_input_for_kind(bolus.input(), RouteKind::Bolus)?;

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
                    let lag = lag_values[bolus.input()];
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
                    let factor = fa_values[bolus.input()];
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
        let destination = self
            .route_semantics
            .bolus_destination(input)
            .ok_or_else(|| {
                PharmsolError::OtherError(format!(
                    "model `{}` does not declare a bolus route for input channel {}",
                    self.info.name, input
                ))
            })?;
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
        if observation.outeq() >= outputs.len() {
            return Err(PharmsolError::OuteqOutOfRange {
                outeq: observation.outeq(),
                nout: outputs.len(),
            });
        }
        Ok(observation.to_prediction(outputs[observation.outeq()], state.to_vec()))
    }
}

#[derive(Clone, Debug)]
pub struct NativeOdeModel {
    shared: Arc<SharedNativeModel>,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
}

#[derive(Clone, Debug)]
pub struct NativeAnalyticalModel {
    shared: Arc<SharedNativeModel>,
}

#[derive(Clone, Debug)]
pub struct NativeSdeModel {
    shared: Arc<SharedNativeModel>,
    nparticles: usize,
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

    pub fn route_index(&self, name: &str) -> Option<usize> {
        self.shared.route_index(name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.shared.output_index(name)
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
        support_point: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();
        let support_vector: V = DVector::from_vec(support_point.to_vec()).into();

        for occasion in subject.occasions() {
            let infusion_refs = occasion.infusions_ref();
            let infusions = infusion_refs
                .iter()
                .map(|infusion| (*infusion).clone())
                .collect::<Vec<_>>();

            for infusion in &infusions {
                self.shared
                    .validate_input_for_kind(infusion.input(), RouteKind::Infusion)?;
            }

            let mut events = occasion.process_events(None, true);
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
            let problem = OdeBuilder::<M>::new()
                .atol(vec![self.atol])
                .rtol(self.rtol)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.to_vec())
                .build_from_eqn(PMProblem::with_params_v(
                    diffeq,
                    self.shared.info.state_len,
                    self.shared.info.route_len,
                    support_point.to_vec(),
                    support_vector.clone(),
                    occasion.covariates(),
                    infusion_refs.as_slice(),
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
                    self.shared.apply_bolus(
                        solver.state_mut().y.as_mut_slice(),
                        bolus.input(),
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

impl NativeAnalyticalModel {
    pub(crate) fn new(info: NativeModelInfo, artifact: impl RuntimeArtifact + 'static) -> Self {
        Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)),
        }
    }

    pub fn route_index(&self, name: &str) -> Option<usize> {
        self.shared.route_index(name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.shared.output_index(name)
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
        support_point: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();

        for occasion in subject.occasions() {
            let infusions = occasion
                .infusions_ref()
                .iter()
                .map(|infusion| (*infusion).clone())
                .collect::<Vec<_>>();

            for infusion in &infusions {
                self.shared
                    .validate_input_for_kind(infusion.input(), RouteKind::Infusion)?;
            }

            let mut events = occasion.process_events(None, true);
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
                        self.shared
                            .apply_bolus(&mut state, bolus.input(), bolus.amount())?
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
                project_analytical_parameters(&self.shared.info, support_point, &derived)?;
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

impl NativeSdeModel {
    pub(crate) fn new(info: NativeModelInfo, artifact: impl RuntimeArtifact + 'static) -> Self {
        let nparticles = info.particles.unwrap_or(1);
        Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)),
            nparticles,
        }
    }

    pub fn with_particles(mut self, nparticles: usize) -> Self {
        self.nparticles = nparticles;
        self
    }

    pub fn route_index(&self, name: &str) -> Option<usize> {
        self.shared.route_index(name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.shared.output_index(name)
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
        support_point: &[f64],
    ) -> Result<Array2<Prediction>, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = Array2::from_shape_fn((self.nparticles, 0), |_| Prediction::default());

        for occasion in subject.occasions() {
            let infusions = occasion
                .infusions_ref()
                .iter()
                .map(|infusion| (*infusion).clone())
                .collect::<Vec<_>>();

            for infusion in &infusions {
                self.shared
                    .validate_input_for_kind(infusion.input(), RouteKind::Infusion)?;
            }

            let mut events = occasion.process_events(None, true);
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
                        for particle in &mut particles {
                            self.shared.apply_bolus(
                                particle.as_mut_slice(),
                                bolus.input(),
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

fn active_route_inputs(infusions: &[Infusion], time: f64, route_len: usize) -> Vec<f64> {
    let mut values = vec![0.0; route_len];
    for infusion in infusions {
        if infusion.input() < route_len
            && time >= infusion.time()
            && time <= infusion.time() + infusion.duration()
        {
            values[infusion.input()] += infusion.amount() / infusion.duration();
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
        if infusion.input() < route_len && start_time >= infusion.time() && end_time <= finish {
            values[infusion.input()] += infusion.amount() / infusion.duration();
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

fn project_analytical_parameters(
    info: &NativeModelInfo,
    support_point: &[f64],
    derived: &[f64],
) -> Result<V, PharmsolError> {
    let kernel = info.analytical.ok_or_else(|| {
        PharmsolError::OtherError(format!(
            "model `{}` does not declare an analytical kernel",
            info.name
        ))
    })?;
    let arity = analytical_parameter_count(kernel);
    if support_point.len() < arity {
        return Err(PharmsolError::OtherError(format!(
            "analytical kernel for model `{}` requires {} parameter value(s), got {}",
            info.name,
            arity,
            support_point.len()
        )));
    }

    // Analytical authoring models can project kernel arguments through a derive
    // kernel by declaring exactly the built-in kernel arity in `derived`.
    if derived.len() == arity {
        return Ok(V::from_vec(derived.to_vec(), NalgebraContext));
    }

    Ok(V::from_vec(
        support_point[..arity].to_vec(),
        NalgebraContext,
    ))
}

fn analytical_parameter_count(kernel: AnalyticalKernel) -> usize {
    match kernel {
        AnalyticalKernel::OneCompartment => 1,
        AnalyticalKernel::OneCompartmentCl => 2,
        AnalyticalKernel::OneCompartmentClWithAbsorption => 3,
        AnalyticalKernel::OneCompartmentWithAbsorption => 2,
        AnalyticalKernel::TwoCompartments => 3,
        AnalyticalKernel::TwoCompartmentsCl => 4,
        AnalyticalKernel::TwoCompartmentsClWithAbsorption => 5,
        AnalyticalKernel::TwoCompartmentsWithAbsorption => 4,
        AnalyticalKernel::ThreeCompartments => 5,
        AnalyticalKernel::ThreeCompartmentsCl => 6,
        AnalyticalKernel::ThreeCompartmentsClWithAbsorption => 7,
        AnalyticalKernel::ThreeCompartmentsWithAbsorption => 6,
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
