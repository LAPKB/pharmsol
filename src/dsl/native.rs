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
use pharmsol_dsl::execution::ModelFunctionKind;
use pharmsol_dsl::{
    AnalyticalKernel, AnalyticalStructureInputKind, AnalyticalStructureInputPlan, ModelKind,
    RouteKind,
};

pub use super::model_info::{
    NativeCovariateInfo, NativeModelInfo, NativeOutputInfo, NativeRouteInfo, NativeStateInfo,
};
use crate::{
    data::{Covariates, Infusion, InputLabel, OutputLabel},
    simulator::{
        cache::{PredictionCache, DEFAULT_CACHE_SIZE},
        equation::{
            metadata::ValidatedRoute,
            ode::{closure_helpers::PMProblem, ExplicitRkTableau, OdeSolver, SdirkTableau},
            sde::{infusion_discontinuities, simulate_sde_event_with},
            EqnKind, Equation, EquationPriv, EquationTypes,
        },
        prediction::{Prediction, SubjectPredictions},
        Fa, Lag, M, T, V,
    },
    Event, Observation, Occasion, Parameters, PharmsolError, Subject, ValidatedModelMetadata,
};

pub type CompiledModelFunction = unsafe extern "C" fn(
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

pub(crate) trait FunctionSession {
    #[allow(clippy::too_many_arguments)]
    unsafe fn invoke_raw(
        &mut self,
        role: ModelFunctionKind,
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
    fn has_function(&self, role: ModelFunctionKind) -> bool;
    fn start_session(&self) -> Result<Box<dyn FunctionSession + '_>, PharmsolError>;
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
    pub derive: Option<CompiledModelFunction>,
    pub dynamics: Option<CompiledModelFunction>,
    pub outputs: CompiledModelFunction,
    pub init: Option<CompiledModelFunction>,
    pub drift: Option<CompiledModelFunction>,
    pub diffusion: Option<CompiledModelFunction>,
    pub route_lag: Option<CompiledModelFunction>,
    pub route_bioavailability: Option<CompiledModelFunction>,
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
        derive: Option<CompiledModelFunction>,
        dynamics: Option<CompiledModelFunction>,
        outputs: CompiledModelFunction,
        init: Option<CompiledModelFunction>,
        drift: Option<CompiledModelFunction>,
        diffusion: Option<CompiledModelFunction>,
        route_lag: Option<CompiledModelFunction>,
        route_bioavailability: Option<CompiledModelFunction>,
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
        derive: Option<CompiledModelFunction>,
        dynamics: Option<CompiledModelFunction>,
        outputs: CompiledModelFunction,
        init: Option<CompiledModelFunction>,
        drift: Option<CompiledModelFunction>,
        diffusion: Option<CompiledModelFunction>,
        route_lag: Option<CompiledModelFunction>,
        route_bioavailability: Option<CompiledModelFunction>,
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

struct NativeFunctionSession<'a> {
    artifact: &'a NativeExecutionArtifact,
}

impl FunctionSession for NativeFunctionSession<'_> {
    unsafe fn invoke_raw(
        &mut self,
        role: ModelFunctionKind,
        time: f64,
        states: *const f64,
        params: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError> {
        let function = match role {
            ModelFunctionKind::Derive => self.artifact.derive,
            ModelFunctionKind::Dynamics => self.artifact.dynamics,
            ModelFunctionKind::Outputs => Some(self.artifact.outputs),
            ModelFunctionKind::Init => self.artifact.init,
            ModelFunctionKind::Drift => self.artifact.drift,
            ModelFunctionKind::Diffusion => self.artifact.diffusion,
            ModelFunctionKind::RouteLag => self.artifact.route_lag,
            ModelFunctionKind::RouteBioavailability => self.artifact.route_bioavailability,
            ModelFunctionKind::Analytical => None,
        }
        .ok_or_else(|| {
            PharmsolError::OtherError(format!(
                "model `{}` does not provide a {:?} function",
                self.artifact.model_name, role
            ))
        })?;

        function(time, states, params, covariates, routes, derived, out);
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

    fn has_function(&self, role: ModelFunctionKind) -> bool {
        match role {
            ModelFunctionKind::Derive => self.derive.is_some(),
            ModelFunctionKind::Dynamics => self.dynamics.is_some(),
            ModelFunctionKind::Outputs => true,
            ModelFunctionKind::Init => self.init.is_some(),
            ModelFunctionKind::Drift => self.drift.is_some(),
            ModelFunctionKind::Diffusion => self.diffusion.is_some(),
            ModelFunctionKind::RouteLag => self.route_lag.is_some(),
            ModelFunctionKind::RouteBioavailability => self.route_bioavailability.is_some(),
            ModelFunctionKind::Analytical => false,
        }
    }

    fn start_session(&self) -> Result<Box<dyn FunctionSession + '_>, PharmsolError> {
        Ok(Box::new(NativeFunctionSession { artifact: self }))
    }
}

#[derive(Clone, Debug)]
struct SharedNativeModel {
    info: Arc<NativeModelInfo>,
    metadata: Arc<ValidatedModelMetadata>,
    route_semantics: Arc<RouteInputSemantics>,
    artifact: Arc<dyn RuntimeArtifact>,
}

fn sorted_dense_metadata<'a, T>(
    info: &NativeModelInfo,
    domain: &str,
    expected_len: usize,
    entries: &'a [T],
    index_of: impl Fn(&T) -> usize,
) -> Result<Vec<&'a T>, PharmsolError> {
    if entries.len() != expected_len {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "expected {expected_len} {domain} entr{} but found {}",
                if expected_len == 1 { "y" } else { "ies" },
                entries.len()
            ),
        });
    }

    let mut sorted = entries.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|entry| index_of(entry));
    for (expected, entry) in sorted.iter().enumerate() {
        let found = index_of(entry);
        if found != expected {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!(
                    "{domain} metadata must use dense 0-based indices; expected {expected}, found {found}"
                ),
            });
        }
    }

    Ok(sorted)
}

fn sorted_state_metadata(info: &NativeModelInfo) -> Result<Vec<&NativeStateInfo>, PharmsolError> {
    if info.state_len == 0 {
        if info.states.is_empty() {
            return Ok(Vec::new());
        }
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "expected no state metadata for an empty state buffer, found {} declaration(s)",
                info.states.len()
            ),
        });
    }

    if info.states.is_empty() {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "expected state metadata for {} state slot(s), found none",
                info.state_len
            ),
        });
    }

    let mut states = info.states.iter().collect::<Vec<_>>();
    states.sort_by_key(|state| state.offset);

    if states[0].offset != 0 {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "state metadata must start at offset 0; first declaration starts at {}",
                states[0].offset
            ),
        });
    }

    for window in states.windows(2) {
        let current = window[0];
        let next = window[1];
        if next.offset <= current.offset {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!(
                    "state metadata offsets must be strictly increasing; saw {} followed by {}",
                    current.offset, next.offset
                ),
            });
        }
    }

    let last_offset = states.last().expect("non-empty states").offset;
    if last_offset >= info.state_len {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "state metadata offset {} is out of range for state buffer length {}",
                last_offset, info.state_len
            ),
        });
    }

    Ok(states)
}

fn state_declaration_for_offset<'a>(
    info: &NativeModelInfo,
    states: &[&'a NativeStateInfo],
    offset: usize,
) -> Result<(usize, &'a NativeStateInfo), PharmsolError> {
    if offset >= info.state_len {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "state offset {} is out of range for state buffer length {}",
                offset, info.state_len
            ),
        });
    }

    let declaration_index = match states.binary_search_by_key(&offset, |state| state.offset) {
        Ok(index) => index,
        Err(0) => {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!("state offset {} precedes the first declared state", offset),
            });
        }
        Err(index) => index - 1,
    };

    Ok((declaration_index, states[declaration_index]))
}

fn runtime_model_metadata(info: &NativeModelInfo) -> Result<ValidatedModelMetadata, PharmsolError> {
    let states = sorted_state_metadata(info)?;
    let state_names = states
        .iter()
        .map(|state| state.name.clone())
        .collect::<Vec<_>>();

    let covariates = sorted_dense_metadata(
        info,
        "covariate",
        info.covariates.len(),
        &info.covariates,
        |covariate| covariate.index,
    )?;
    let routes = sorted_dense_metadata(
        info,
        "route declaration",
        info.routes.len(),
        &info.routes,
        |route| route.declaration_index,
    )?;
    let outputs =
        sorted_dense_metadata(info, "output", info.output_len, &info.outputs, |output| {
            output.index
        })?;

    let mut metadata = crate::simulator::equation::metadata::new(info.name.clone())
        .kind(info.kind)
        .parameters(info.parameters.iter().cloned())
        .covariates(covariates.into_iter().map(|covariate| {
            let mut declaration =
                crate::simulator::equation::metadata::Covariate::new(covariate.name.clone());
            if let Some(interpolation) = covariate.interpolation {
                declaration = declaration.with_interpolation(interpolation);
            }
            declaration
        }))
        .states(state_names.iter().cloned())
        .outputs(outputs.into_iter().map(|output| output.name.clone()));

    if let Some(function) = info.analytical {
        metadata = metadata.analytical_kernel(function);
    }

    if let Some(particles) = info.particles {
        metadata = metadata.particles(particles);
    }

    for route in &routes {
        let (_destination_index, destination_state) =
            state_declaration_for_offset(info, &states, route.destination_offset)?;
        let destination = destination_state.name.clone();
        if route.destination_name != destination {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!(
                    "route `{}` names destination `{}` but offset {} resolves to `{}`",
                    route.name, route.destination_name, route.destination_offset, destination
                ),
            });
        }
        // Structured-block DSL routes still lower without an explicit kind.
        // Treat them as declaration-ordered bolus routes for the shared
        // metadata surface while preserving the original runtime semantics
        // from `info.routes` below.
        let kind = route.kind.unwrap_or(RouteKind::Bolus);

        let mut declaration = match kind {
            RouteKind::Bolus => {
                crate::simulator::equation::metadata::Route::bolus(route.name.clone())
            }
            RouteKind::Infusion => {
                crate::simulator::equation::metadata::Route::infusion(route.name.clone())
            }
        }
        .to_state(destination);

        if route.has_lag {
            declaration = declaration.with_lag();
        }
        if route.has_bioavailability {
            declaration = declaration.with_bioavailability();
        }

        declaration = if route.inject_input_to_destination {
            declaration.inject_input_to_destination()
        } else {
            declaration.expect_explicit_input()
        };

        metadata = metadata.route(declaration);
    }

    let validated = match info.kind {
        ModelKind::Sde => {
            let particles = info
                .particles
                .ok_or_else(|| PharmsolError::InvalidMetadata {
                    model: info.name.clone(),
                    detail: "SDE models must declare a particle count".to_string(),
                })?;
            metadata.validate_for_with_particles(ModelKind::Sde, particles)
        }
        kind => metadata.validate_for(kind),
    }
    .map_err(|error| PharmsolError::InvalidMetadata {
        model: info.name.clone(),
        detail: error.to_string(),
    })?;

    if validated.route_input_count() != info.route_len {
        return Err(PharmsolError::InvalidMetadata {
            model: info.name.clone(),
            detail: format!(
                "route input count {} does not match declared route buffer length {}",
                validated.route_input_count(),
                info.route_len
            ),
        });
    }

    for route in routes {
        let (destination_index, _) =
            state_declaration_for_offset(info, &states, route.destination_offset)?;
        let validated_route = &validated.routes()[route.declaration_index];
        if validated_route.input_index() != route.index {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!(
                    "route `{}` uses input index {} but validated metadata resolves to {}",
                    route.name,
                    route.index,
                    validated_route.input_index()
                ),
            });
        }
        if validated_route.destination_index() != destination_index {
            return Err(PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: format!(
                    "route `{}` targets state declaration {} but validated metadata resolves to {}",
                    route.name,
                    destination_index,
                    validated_route.destination_index()
                ),
            });
        }
    }

    Ok(validated)
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
    fn with_info(&self, info: NativeModelInfo) -> Result<Self, PharmsolError> {
        let metadata = Arc::new(runtime_model_metadata(&info)?);
        let route_semantics = Arc::new(RouteInputSemantics::from_model_info(&info));
        Ok(Self {
            info: Arc::new(info),
            metadata,
            route_semantics,
            artifact: Arc::clone(&self.artifact),
        })
    }

    fn new(
        info: NativeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let artifact = Arc::new(artifact);
        let metadata = Arc::new(runtime_model_metadata(&info)?);
        let route_semantics = Arc::new(RouteInputSemantics::from_model_info(&info));
        Ok(Self {
            metadata,
            route_semantics,
            info: Arc::new(info),
            artifact,
        })
    }

    fn metadata(&self) -> &ValidatedModelMetadata {
        self.metadata.as_ref()
    }

    fn metadata_route_index_for_label(&self, label: &str) -> Option<usize> {
        self.metadata()
            .route_for_label(label)
            .map(ValidatedRoute::input_index)
    }

    fn metadata_output_index_for_label(&self, label: &str) -> Option<usize> {
        self.metadata().output_for_label(label)
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
            .ok_or_else(|| {
                PharmsolError::unknown_input_label(label.as_str(), &self.metadata().route_labels())
            })?;
        self.validate_input_for_kind(input, kind)?;
        Ok(input)
    }

    fn resolve_output_label(&self, label: &OutputLabel) -> Result<usize, PharmsolError> {
        self.metadata_output_index_for_label(label.as_str())
            .ok_or_else(|| {
                PharmsolError::unknown_output_label(
                    label.as_str(),
                    &self.metadata().output_labels(),
                )
            })
    }

    fn resolve_events(&self, occasion: &Occasion) -> Result<Vec<Event>, PharmsolError> {
        let mut events = occasion.process_events(None);

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
        session: &mut dyn FunctionSession,
        time: f64,
        state: &[f64],
        support_point: &[f64],
        covariates: &Covariates,
        route_inputs: &[f64],
        derived: &mut [f64],
        cov_buf: &mut [f64],
    ) -> Result<(), PharmsolError> {
        self.fill_cov_buffer(covariates, time, cov_buf);
        if self.artifact.has_function(ModelFunctionKind::Derive) {
            unsafe {
                session.invoke_raw(
                    ModelFunctionKind::Derive,
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
        session: &mut dyn FunctionSession,
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
                ModelFunctionKind::Outputs,
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
        session: &mut dyn FunctionSession,
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
            if self.artifact.has_function(ModelFunctionKind::Init) {
                unsafe {
                    session.invoke_raw(
                        ModelFunctionKind::Init,
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
        session: &mut dyn FunctionSession,
        events: &mut [Event],
        covariates: &Covariates,
        support_point: &[f64],
    ) -> Result<(), PharmsolError> {
        if !self.artifact.has_function(ModelFunctionKind::RouteLag)
            && !self
                .artifact
                .has_function(ModelFunctionKind::RouteBioavailability)
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
                let input = bolus.input_index().ok_or_else(|| {
                    PharmsolError::unknown_input_label(
                        bolus.input(),
                        &self.metadata().route_labels(),
                    )
                })?;
                self.validate_input_for_kind(input, RouteKind::Bolus)?;

                if self.artifact.has_function(ModelFunctionKind::RouteLag) {
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
                            ModelFunctionKind::RouteLag,
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

                if self
                    .artifact
                    .has_function(ModelFunctionKind::RouteBioavailability)
                {
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
                            ModelFunctionKind::RouteBioavailability,
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
        session: &mut dyn FunctionSession,
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
        let outeq = observation.outeq_index().ok_or_else(|| {
            PharmsolError::unknown_output_label(
                observation.outeq(),
                &self.metadata().output_labels(),
            )
        })?;
        self.validate_output(outeq)?;
        let label = OutputLabel::new(self.metadata().output_labels()[outeq]);
        Ok(observation.to_prediction(label, outputs[outeq]))
    }
}

#[derive(Clone, Debug)]
pub struct NativeOdeModel {
    shared: Arc<SharedNativeModel>,
    solver: OdeSolver,
    rtol: f64,
    atol: f64,
    cache: Option<PredictionCache>,
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
}

#[derive(Clone, Debug)]
pub enum CompiledNativeModel {
    Ode(NativeOdeModel),
    Analytical(NativeAnalyticalModel),
    Sde(NativeSdeModel),
}

impl CompiledNativeModel {
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        match self {
            Self::Ode(model) => model.metadata(),
            Self::Analytical(model) => model.metadata(),
            Self::Sde(model) => model.metadata(),
        }
    }
}

impl NativeOdeModel {
    pub(crate) fn new(
        info: NativeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        Ok(Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)?),
            solver: OdeSolver::default(),
            rtol: DEFAULT_ODE_RTOL,
            atol: DEFAULT_ODE_ATOL,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
        })
    }

    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.cache = self.cache.as_ref().map(PredictionCache::detached);
        self.solver = solver;
        self
    }

    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.cache = self.cache.as_ref().map(PredictionCache::detached);
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    /// Access the validated metadata attached to this compiled ODE model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.shared.metadata()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<SubjectPredictions, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters.as_slice())
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();
        output.set_id(subject.id());
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
            if !shared.artifact.has_function(ModelFunctionKind::Dynamics) {
                return Err(PharmsolError::OtherError(format!(
                    "model `{}` does not have a dynamics function",
                    shared.info.name
                )));
            }
            let function_error = RefCell::new(None::<PharmsolError>);

            let diffeq_session = &session;
            let diffeq_error = &function_error;
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
                        ModelFunctionKind::Dynamics,
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
                NalgebraContext::new(),
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
                        &function_error,
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

            if let Some(error) = function_error.into_inner() {
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
        session: &RefCell<Box<dyn FunctionSession + '_>>,
        function_error: &RefCell<Option<PharmsolError>>,
    ) -> Result<(), PharmsolError>
    where
        F: Fn(&V, &V, f64, &mut V, &V, &V, &Covariates) + 'a,
        S: OdeSolverMethod<'a, PMProblem<'a, F>>,
    {
        for (index, event) in events.iter().enumerate() {
            match event {
                Event::Bolus(bolus) => {
                    let input = bolus.input_index().ok_or_else(|| {
                        PharmsolError::unknown_input_label(
                            bolus.input(),
                            &self.shared.metadata().route_labels(),
                        )
                    })?;
                    self.shared.apply_bolus(
                        solver.state_mut().y.as_mut_slice(),
                        input,
                        bolus.amount(),
                    )?;
                }
                Event::Infusion(_) => {}
                Event::Observation(observation) => {
                    if function_error.borrow().is_some() {
                        return Err(function_error.borrow_mut().take().unwrap());
                    }
                    let prediction = self.shared.observation_prediction(
                        &mut **session.borrow_mut(),
                        observation,
                        solver.state().y.as_slice(),
                        support_point,
                        covariates,
                        infusions,
                    )?;
                    output.add_prediction(prediction, observation.occasion());
                }
            }

            if let Some(next_event) = events.get(index + 1) {
                if event.time() == next_event.time() {
                    continue;
                }

                match solver.set_stop_time(next_event.time()) {
                    Ok(_) => loop {
                        match solver.step() {
                            Ok(_) if function_error.borrow().is_some() => {
                                return Err(function_error.borrow_mut().take().unwrap());
                            }
                            Ok(OdeSolverStopReason::InternalTimestep) => continue,
                            Ok(OdeSolverStopReason::TstopReached) => break,
                            Ok(OdeSolverStopReason::RootFound(_, _)) => {
                                return Err(PharmsolError::OtherError(format!(
                                    "solver stopped at an unexpected root at t = {:.4} \
                                     (root finding is not configured)",
                                    next_event.time()
                                )));
                            }
                            Err(err) => {
                                return Err(PharmsolError::from_solver_error(
                                    err,
                                    next_event.time(),
                                ));
                            }
                        }
                    },
                    Err(diffsol::error::DiffsolError::OdeSolverError(
                        OdeSolverError::StopTimeAtCurrentTime,
                    )) => continue,
                    Err(err) => {
                        return Err(PharmsolError::from_solver_error(err, next_event.time()));
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
    let add_context = |e: PharmsolError| {
        e.with_subject_context(
            subject.id(),
            support_point,
            &model.metadata().parameter_names(),
        )
    };
    if let Some(cache) = &model.cache {
        let key = (
            subject.hash(),
            crate::simulator::equation::parameters_hash(support_point),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = model
            .estimate_predictions_dense(subject, support_point)
            .map_err(add_context)?;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        model
            .estimate_predictions_dense(subject, support_point)
            .map_err(add_context)
    }
}

impl crate::simulator::equation::Cache for NativeOdeModel {
    fn with_cache_capacity(mut self, size: usize) -> Self {
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
        Some(self.shared.metadata())
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
        _parameters: &[f64],
        _observation: &Observation,
        _time: f64,
        _covariates: &Covariates,
        _state: &mut Self::S,
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
        V::zeros(self.shared.info.state_len, NalgebraContext::new())
    }
}

impl Equation for NativeOdeModel {
    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters)
    }

    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters)
    }

    fn kind() -> EqnKind {
        EqnKind::ODE
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        runtime_ode_predictions(self, subject, parameters.as_slice())
    }
}

impl NativeAnalyticalModel {
    pub(crate) fn new(
        info: NativeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let parameter_projection = build_analytical_parameter_projection(&info)?;
        Ok(Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)?),
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
            parameter_projection,
        })
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    /// Access the validated metadata attached to this compiled analytical model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.shared.metadata()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<SubjectPredictions, PharmsolError> {
        runtime_analytical_predictions(self, subject, parameters.as_slice())
    }

    fn estimate_predictions_dense_uncached(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.shared.validate_support_point(support_point)?;
        let mut output = SubjectPredictions::default();
        output.set_id(subject.id());

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
                            PharmsolError::unknown_input_label(
                                bolus.input(),
                                &self.shared.metadata().route_labels(),
                            )
                        })?;
                        self.shared.apply_bolus(&mut state, input, bolus.amount())?
                    }
                    Event::Infusion(_) => {}
                    Event::Observation(observation) => {
                        output.add_prediction(
                            self.shared.observation_prediction(
                                &mut *session,
                                observation,
                                &state,
                                support_point,
                                occasion.covariates(),
                                infusions.as_slice(),
                            )?,
                            observation.occasion(),
                        );
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
        session: &mut dyn FunctionSession,
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
                        "model `{}` does not declare an analytical function",
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
    let add_context = |e: PharmsolError| {
        e.with_subject_context(
            subject.id(),
            support_point,
            &model.metadata().parameter_names(),
        )
    };
    if let Some(cache) = &model.cache {
        let key = (
            subject.hash(),
            crate::simulator::equation::parameters_hash(support_point),
        );
        if let Some(cached) = cache.get(&key) {
            return Ok(cached);
        }

        let result = model
            .estimate_predictions_dense_uncached(subject, support_point)
            .map_err(add_context)?;
        cache.insert(key, result.clone());
        Ok(result)
    } else {
        model
            .estimate_predictions_dense_uncached(subject, support_point)
            .map_err(add_context)
    }
}

impl crate::simulator::equation::Cache for NativeAnalyticalModel {
    fn with_cache_capacity(mut self, size: usize) -> Self {
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
        Some(self.shared.metadata())
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
        _parameters: &[f64],
        _observation: &Observation,
        _time: f64,
        _covariates: &Covariates,
        _state: &mut Self::S,
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
        V::zeros(self.shared.info.state_len, NalgebraContext::new())
    }
}

impl Equation for NativeAnalyticalModel {
    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_analytical_predictions(self, subject, parameters)
    }

    fn kind() -> EqnKind {
        EqnKind::Analytical
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        runtime_analytical_predictions(self, subject, parameters.as_slice())
    }

    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        runtime_analytical_predictions(self, subject, parameters)
    }
}

impl NativeSdeModel {
    pub(crate) fn new(
        info: NativeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let nparticles = info
            .particles
            .ok_or_else(|| PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: "SDE models must declare a particle count".to_string(),
            })?;
        Ok(Self {
            shared: Arc::new(SharedNativeModel::new(info, artifact)?),
            nparticles,
        })
    }

    pub fn with_particles(mut self, nparticles: usize) -> Self {
        if self.nparticles == nparticles {
            return self;
        }

        let mut info = self.shared.info.as_ref().clone();
        info.particles = Some(nparticles);
        self.shared = Arc::new(
            self.shared
                .with_info(info)
                .expect("compiled SDE metadata should stay valid after particle override"),
        );
        self.nparticles = nparticles;
        self
    }

    pub fn info(&self) -> &NativeModelInfo {
        self.shared.info.as_ref()
    }

    /// Access the validated metadata attached to this compiled SDE model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.shared.metadata()
    }

    pub fn backend(&self) -> RuntimeBackend {
        self.shared.artifact.backend()
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Array2<Prediction>, PharmsolError> {
        self.estimate_predictions_dense(subject, parameters.as_slice())
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<Array2<Prediction>, PharmsolError> {
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
                            PharmsolError::unknown_input_label(
                                bolus.input(),
                                &self.shared.metadata().route_labels(),
                            )
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
        let discontinuities = Arc::new(infusion_discontinuities(infusions, start_time, end_time));
        let covariates = covariates.clone();
        if !shared.artifact.has_function(ModelFunctionKind::Drift) {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` does not have a drift function",
                shared.info.name
            )));
        }
        if !shared.artifact.has_function(ModelFunctionKind::Diffusion) {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` does not have a diffusion function",
                shared.info.name
            )));
        }

        particles
            .par_iter_mut()
            .try_for_each(|particle| -> Result<(), PharmsolError> {
                let shared = Arc::clone(&shared);
                let support = Arc::clone(&support);
                let infusions = Arc::clone(&infusion_events);
                let discontinuities = Arc::clone(&discontinuities);
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
                let function_error = RefCell::new(None::<PharmsolError>);
                let drift_session = &session;
                let diffusion_session = &session;
                let drift_error = &function_error;
                let diffusion_error = &function_error;
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
                                ModelFunctionKind::Drift,
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
                                ModelFunctionKind::Diffusion,
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
                    &discontinuities,
                );
                if let Some(error) = function_error.into_inner() {
                    return Err(error);
                }
                *particle = next;
                Ok(())
            })?;

        Ok(())
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

    fn metadata(&self) -> Option<&crate::ValidatedModelMetadata> {
        Some(self.shared.metadata())
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
        _parameters: &[f64],
        _observation: &Observation,
        _time: f64,
        _covariates: &Covariates,
        _state: &mut Self::S,
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
    fn kind() -> EqnKind {
        EqnKind::SDE
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<Self::P, PharmsolError> {
        NativeSdeModel::estimate_predictions(self, subject, parameters)
    }

    fn estimate_predictions_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        NativeSdeModel::estimate_predictions_dense(self, subject, parameters)
    }

    fn simulate_subject_dense(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Self::P, PharmsolError> {
        NativeSdeModel::estimate_predictions_dense(self, subject, parameters)
    }
}

fn active_route_inputs(infusions: &[Infusion], time: f64, route_len: usize) -> Vec<f64> {
    let mut values = vec![0.0; route_len];
    for infusion in infusions {
        let input = infusion
            .input_index()
            .expect("resolved infusions should use numeric input labels");
        // Infusion activity is half-open `[start, end)`: the rate is delivered
        // up to but not including the finish time. Using `<=` here would keep
        // the rate active at the exact end time, which is also a segment
        // boundary during native SDE propagation, and would add an extra
        // end-time rate step that over-delivers the infused amount.
        if input < route_len
            && time >= infusion.time()
            && time < infusion.time() + infusion.duration()
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

fn build_analytical_parameter_projection(
    info: &NativeModelInfo,
) -> Result<AnalyticalStructureInputKind, PharmsolError> {
    let function = info.analytical.ok_or_else(|| {
        PharmsolError::OtherError(format!(
            "model `{}` does not declare an analytical function",
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
        function,
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
            V::from_vec(values, NalgebraContext::new())
        }
        AnalyticalStructureInputKind::AllDerived { indices, identity } => {
            let values = if *identity {
                derived[..indices.len()].to_vec()
            } else {
                indices.iter().map(|&index| derived[index]).collect()
            };
            V::from_vec(values, NalgebraContext::new())
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
            NalgebraContext::new(),
        ),
    }
}

fn apply_analytical_kernel(
    function: AnalyticalKernel,
    state: &[f64],
    params: &V,
    dt: f64,
    route_inputs: &[f64],
    covariates: &Covariates,
) -> V {
    let state = V::from_vec(state.to_vec(), NalgebraContext::new());
    let route_inputs = V::from_vec(route_inputs.to_vec(), NalgebraContext::new());
    match function {
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
        build_analytical_parameter_projection, project_analytical_parameters, FunctionSession,
        NativeAnalyticalModel, NativeCovariateInfo, NativeModelInfo, NativeOdeModel,
        NativeOutputInfo, NativeRouteInfo, NativeSdeModel, NativeStateInfo, RuntimeArtifact,
        RuntimeBackend, SharedNativeModel,
    };
    use super::{runtime_analytical_predictions, PredictionCache};
    #[cfg(any(
        feature = "dsl-jit",
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        all(
            feature = "dsl-wasm",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )
    ))]
    use super::{runtime_ode_predictions, DEFAULT_ODE_ATOL, DEFAULT_ODE_RTOL};
    #[cfg(any(
        feature = "dsl-jit",
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        all(
            feature = "dsl-wasm",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )
    ))]
    use crate::dsl::{CompiledRuntimeModel, RuntimePredictions};
    use crate::{
        data::builder::SubjectBuilderExt,
        prelude::SubjectPredictions,
        simulator::equation::{Cache, Equation},
        Parameters, PharmsolError, Subject,
    };
    use diffsol::VectorHost;
    use pharmsol_dsl::execution::ModelFunctionKind;
    use pharmsol_dsl::{
        AnalyticalKernel, AnalyticalStructureInputKind, CovariateInterpolation, ModelKind,
        RouteKind,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct DummyArtifact;

    impl RuntimeArtifact for DummyArtifact {
        fn backend(&self) -> RuntimeBackend {
            panic!("dummy artifact backend should not be used in tests")
        }

        fn has_function(&self, _role: ModelFunctionKind) -> bool {
            false
        }

        fn start_session(&self) -> Result<Box<dyn FunctionSession + '_>, PharmsolError> {
            panic!("dummy artifact sessions should not be used in tests")
        }
    }

    #[derive(Debug)]
    struct CountingAnalyticalArtifact {
        session_count: Arc<AtomicUsize>,
    }

    impl RuntimeArtifact for CountingAnalyticalArtifact {
        fn backend(&self) -> RuntimeBackend {
            panic!("counting analytical artifact backend should not be used in tests")
        }

        fn has_function(&self, role: ModelFunctionKind) -> bool {
            role == ModelFunctionKind::Outputs
        }

        fn start_session(&self) -> Result<Box<dyn FunctionSession + '_>, PharmsolError> {
            self.session_count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingAnalyticalSession))
        }
    }

    struct CountingAnalyticalSession;

    impl FunctionSession for CountingAnalyticalSession {
        unsafe fn invoke_raw(
            &mut self,
            role: ModelFunctionKind,
            _time: f64,
            states: *const f64,
            _params: *const f64,
            _covariates: *const f64,
            _routes: *const f64,
            _derived: *const f64,
            out: *mut f64,
        ) -> Result<(), PharmsolError> {
            if role != ModelFunctionKind::Outputs {
                return Err(PharmsolError::OtherError(format!(
                    "unexpected kernel role {role:?} in analytical cache test"
                )));
            }
            unsafe { *out = *states };
            Ok(())
        }
    }

    fn counting_analytical_model() -> (NativeAnalyticalModel, Arc<AtomicUsize>) {
        let session_count = Arc::new(AtomicUsize::new(0));
        let model = NativeAnalyticalModel::new(
            NativeModelInfo {
                name: "analytical_cache".to_string(),
                kind: ModelKind::Analytical,
                parameters: vec!["ke".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![NativeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: Vec::new(),
                outputs: vec![NativeOutputInfo {
                    name: "cp".to_string(),
                    index: 0,
                }],
                state_len: 1,
                derived_len: 0,
                output_len: 1,
                route_len: 0,
                analytical: Some(AnalyticalKernel::OneCompartment),
                particles: None,
            },
            CountingAnalyticalArtifact {
                session_count: Arc::clone(&session_count),
            },
        )
        .expect("counting analytical model should build");
        (model, session_count)
    }

    fn analytical_cache_subject() -> Subject {
        Subject::builder("analytical_cache")
            .missing_observation(0.0, "cp")
            .build()
    }

    fn analytical_cache_parameters(model: &NativeAnalyticalModel) -> Parameters {
        Parameters::with_model(model, [("ke", 0.1)])
            .expect("analytical cache parameters should validate")
    }

    fn bolus_only_shared_model() -> SharedNativeModel {
        SharedNativeModel::new(
            NativeModelInfo {
                name: "bolus_only".to_string(),
                kind: ModelKind::Ode,
                parameters: Vec::new(),
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![NativeStateInfo {
                    name: "gut".to_string(),
                    offset: 0,
                }],
                routes: vec![NativeRouteInfo {
                    name: "oral".to_string(),
                    declaration_index: 0,
                    index: 0,
                    kind: Some(RouteKind::Bolus),
                    destination_offset: 0,
                    destination_name: "gut".to_string(),
                    has_lag: false,
                    has_bioavailability: false,
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
        .expect("bolus-only metadata should build")
    }

    fn analytical_model_info(
        parameters: &[&str],
        derived: &[&str],
        function: AnalyticalKernel,
    ) -> NativeModelInfo {
        NativeModelInfo {
            name: "analytical_projection".to_string(),
            kind: ModelKind::Analytical,
            parameters: parameters.iter().map(|name| (*name).to_string()).collect(),
            derived: derived.iter().map(|name| (*name).to_string()).collect(),
            covariates: Vec::new(),
            states: (0..function.state_count())
                .map(|offset| NativeStateInfo {
                    name: format!("state_{offset}"),
                    offset,
                })
                .collect(),
            routes: Vec::new(),
            outputs: Vec::new(),
            state_len: function.state_count(),
            derived_len: derived.len(),
            output_len: 0,
            route_len: 0,
            analytical: Some(function),
            particles: None,
        }
    }

    #[test]
    fn runtime_ode_models_expose_validated_metadata_for_declared_routes() {
        let model = NativeOdeModel::new(
            NativeModelInfo {
                name: "runtime_metadata".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string(), "v".to_string()],
                derived: Vec::new(),
                covariates: vec![NativeCovariateInfo {
                    name: "wt".to_string(),
                    index: 0,
                    interpolation: Some(CovariateInterpolation::Linear),
                }],
                states: vec![NativeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: vec![NativeRouteInfo {
                    name: "iv".to_string(),
                    declaration_index: 0,
                    index: 0,
                    kind: Some(RouteKind::Infusion),
                    destination_offset: 0,
                    destination_name: "central".to_string(),
                    has_lag: false,
                    has_bioavailability: false,
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
        .expect("runtime ODE metadata should build");

        let metadata = model.metadata();
        assert_eq!(metadata.parameter_index("ke"), Some(0));
        assert_eq!(
            metadata.covariate("wt").unwrap().interpolation(),
            Some(CovariateInterpolation::Linear)
        );
        assert_eq!(metadata.route("iv").unwrap().destination(), "central");
        assert_eq!(metadata.output("cp").unwrap().name(), "cp");

        let compiled = super::CompiledNativeModel::Ode(model.clone());
        assert_eq!(
            compiled.metadata().route("iv").unwrap().destination(),
            "central"
        );
    }

    #[test]
    fn runtime_ode_models_map_array_state_offsets_to_declarations() {
        let model = NativeOdeModel::new(
            NativeModelInfo {
                name: "array_state_runtime_metadata".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string(), "v".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![
                    NativeStateInfo {
                        name: "transit".to_string(),
                        offset: 0,
                    },
                    NativeStateInfo {
                        name: "central".to_string(),
                        offset: 4,
                    },
                ],
                routes: vec![
                    NativeRouteInfo {
                        name: "oral".to_string(),
                        declaration_index: 0,
                        index: 0,
                        kind: Some(RouteKind::Bolus),
                        destination_offset: 0,
                        destination_name: "transit".to_string(),
                        has_lag: false,
                        has_bioavailability: false,
                        inject_input_to_destination: false,
                    },
                    NativeRouteInfo {
                        name: "iv".to_string(),
                        declaration_index: 1,
                        index: 0,
                        kind: Some(RouteKind::Infusion),
                        destination_offset: 4,
                        destination_name: "central".to_string(),
                        has_lag: false,
                        has_bioavailability: false,
                        inject_input_to_destination: false,
                    },
                ],
                outputs: vec![NativeOutputInfo {
                    name: "cp".to_string(),
                    index: 0,
                }],
                state_len: 5,
                derived_len: 0,
                output_len: 1,
                route_len: 1,
                analytical: None,
                particles: None,
            },
            DummyArtifact,
        )
        .expect("array-state runtime metadata should build");

        let metadata = model.metadata();
        assert_eq!(metadata.states()[0].name(), "transit");
        assert_eq!(metadata.states()[1].name(), "central");
        assert_eq!(metadata.route_input_count(), 1);
        assert_eq!(metadata.route("oral").unwrap().destination(), "transit");
        assert_eq!(metadata.route("oral").unwrap().destination_index(), 0);
        assert_eq!(metadata.route("iv").unwrap().destination(), "central");
        assert_eq!(metadata.route("iv").unwrap().destination_index(), 1);
    }

    #[test]
    fn runtime_ode_model_setup_rejects_invalid_route_destination_metadata() {
        let error = NativeOdeModel::new(
            NativeModelInfo {
                name: "runtime_metadata_invalid_destination".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![NativeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: vec![NativeRouteInfo {
                    name: "iv".to_string(),
                    declaration_index: 0,
                    index: 0,
                    kind: Some(RouteKind::Infusion),
                    destination_offset: 1,
                    destination_name: "central".to_string(),
                    has_lag: false,
                    has_bioavailability: false,
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
        .expect_err("invalid route destination metadata must fail at setup");

        assert!(error.to_string().contains(
            "Compiled model `runtime_metadata_invalid_destination` has invalid runtime metadata"
        ));
        assert!(error
            .to_string()
            .contains("state offset 1 is out of range for state buffer length 1"));
    }

    #[test]
    fn runtime_sde_with_particles_updates_metadata_and_info() {
        let model = NativeSdeModel::new(
            NativeModelInfo {
                name: "runtime_sde_particles".to_string(),
                kind: ModelKind::Sde,
                parameters: vec!["ke".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![NativeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: Vec::new(),
                outputs: vec![NativeOutputInfo {
                    name: "cp".to_string(),
                    index: 0,
                }],
                state_len: 1,
                derived_len: 0,
                output_len: 1,
                route_len: 0,
                analytical: None,
                particles: Some(32),
            },
            DummyArtifact,
        )
        .expect("runtime SDE metadata should build")
        .with_particles(64);

        assert_eq!(model.info().particles, Some(64));
        assert_eq!(model.metadata().particles(), Some(64));
    }

    /// Test-only runtime artifact that models a single central compartment with
    /// zero intrinsic drift and zero diffusion. The infused rate reaches the
    /// state purely through `apply_route_inputs_to_rates`, so the delivered mass
    /// equals the exact time integral of the active infusion rate.
    #[derive(Debug)]
    struct SdeInfusionDeliveryArtifact;

    impl RuntimeArtifact for SdeInfusionDeliveryArtifact {
        fn backend(&self) -> RuntimeBackend {
            panic!("infusion-delivery test artifact backend should not be used")
        }

        fn has_function(&self, role: ModelFunctionKind) -> bool {
            matches!(
                role,
                ModelFunctionKind::Drift | ModelFunctionKind::Diffusion | ModelFunctionKind::Outputs
            )
        }

        fn start_session(&self) -> Result<Box<dyn FunctionSession + '_>, PharmsolError> {
            Ok(Box::new(SdeInfusionDeliverySession))
        }
    }

    struct SdeInfusionDeliverySession;

    impl FunctionSession for SdeInfusionDeliverySession {
        unsafe fn invoke_raw(
            &mut self,
            role: ModelFunctionKind,
            _time: f64,
            states: *const f64,
            _params: *const f64,
            _covariates: *const f64,
            _routes: *const f64,
            _derived: *const f64,
            out: *mut f64,
        ) -> Result<(), PharmsolError> {
            match role {
                // Intrinsic drift is zero; the injected infusion rate is added
                // by `apply_route_inputs_to_rates` after this kernel returns.
                ModelFunctionKind::Drift => unsafe { *out = 0.0 },
                // Deterministic (noise-free) diffusion keeps delivered mass exact.
                ModelFunctionKind::Diffusion => unsafe { *out = 0.0 },
                // Single-state passthrough output: cp == central.
                ModelFunctionKind::Outputs => unsafe { *out = *states },
                other => {
                    return Err(PharmsolError::OtherError(format!(
                        "unexpected kernel role {other:?} in SDE infusion-delivery test"
                    )))
                }
            }
            Ok(())
        }
    }

    /// Regression: an infusion whose finish time lands strictly between two
    /// observations must deliver its exact declared amount. The infusion end is
    /// also a native SDE segment boundary, so a closed `[start, end]` activity
    /// window (or an extra end-time rate step) would over-deliver mass.
    #[test]
    fn native_sde_infusion_ending_between_observations_delivers_exact_amount() {
        use crate::data::builder::SubjectBuilderExt;

        let model = NativeSdeModel::new(
            NativeModelInfo {
                name: "sde_infusion_delivery".to_string(),
                kind: ModelKind::Sde,
                parameters: Vec::new(),
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![NativeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: vec![NativeRouteInfo {
                    name: "iv".to_string(),
                    declaration_index: 0,
                    index: 0,
                    kind: Some(RouteKind::Infusion),
                    destination_offset: 0,
                    destination_name: "central".to_string(),
                    has_lag: false,
                    has_bioavailability: false,
                    inject_input_to_destination: true,
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
                particles: Some(2),
            },
            SdeInfusionDeliveryArtifact,
        )
        .expect("SDE infusion-delivery metadata should build");

        // Infusion delivers 20 units over [1, 3] (rate 10/unit-time). The finish
        // time (t = 3) sits strictly between the observations at t = 2 and t = 4.
        let subject = crate::Subject::builder("sde_infusion_delivery")
            .infusion(1.0, 20.0, "iv", 2.0)
            .missing_observation(0.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let predictions = model
            .estimate_predictions_dense(&subject, &[])
            .expect("noise-free SDE infusion model should simulate");

        assert_eq!(predictions.dim(), (2, 3));

        for particle in 0..predictions.nrows() {
            // Before the infusion begins nothing has been delivered.
            assert!(
                predictions[(particle, 0)].prediction().abs() < 1e-9,
                "particle {particle} should start empty, got {}",
                predictions[(particle, 0)].prediction()
            );
            // Halfway through the infusion exactly half the mass is delivered.
            assert!(
                (predictions[(particle, 1)].prediction() - 10.0).abs() < 1e-9,
                "particle {particle} should hold 10 units mid-infusion, got {}",
                predictions[(particle, 1)].prediction()
            );
            // After the infusion ends between observations the full declared
            // amount is delivered and no extra end-time rate step is added.
            assert!(
                (predictions[(particle, 2)].prediction() - 20.0).abs() < 1e-9,
                "particle {particle} should hold the full 20 units, got {}",
                predictions[(particle, 2)].prediction()
            );
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

    #[cfg(any(
        feature = "dsl-jit",
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        all(
            feature = "dsl-wasm",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )
    ))]
    fn cached_runtime_ode_model() -> NativeOdeModel {
        NativeOdeModel {
            shared: Arc::new(bolus_only_shared_model()),
            solver: Default::default(),
            rtol: DEFAULT_ODE_RTOL,
            atol: DEFAULT_ODE_ATOL,
            cache: Some(PredictionCache::new(1)),
        }
    }

    #[cfg(any(
        feature = "dsl-jit",
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        all(
            feature = "dsl-wasm",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )
    ))]
    fn cached_runtime_subject() -> Subject {
        Subject::builder("runtime_cached_prediction")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .build()
    }

    #[test]
    fn compiled_analytical_predictions_consistently_obey_cache_controls() {
        let (model, session_count) = counting_analytical_model();
        let subject = analytical_cache_subject();
        let parameters = analytical_cache_parameters(&model);

        model
            .estimate_predictions(&subject, &parameters)
            .expect("first analytical prediction should compute");
        model
            .estimate_predictions(&subject, &parameters)
            .expect("repeated inherent prediction should hit the cache");
        Equation::estimate_predictions_dense(&model, &subject, parameters.as_slice())
            .expect("dense prediction should hit the same cache");
        Equation::estimate_predictions(&model, &subject, &parameters)
            .expect("Equation prediction should hit the same cache");
        Equation::simulate_subject_dense(&model, &subject, parameters.as_slice())
            .expect("deterministic simulation should hit the same cache");
        runtime_analytical_predictions(&model, &subject, parameters.as_slice())
            .expect("runtime helper should hit the same cache without recursion");
        assert_eq!(session_count.load(Ordering::SeqCst), 1);

        model.clear_cache();
        model
            .estimate_predictions(&subject, &parameters)
            .expect("prediction after clear should recompute");
        assert_eq!(session_count.load(Ordering::SeqCst), 2);

        let resized = model.clone().with_cache_capacity(1);
        resized
            .estimate_predictions(&subject, &parameters)
            .expect("replacement cache should start empty");
        resized
            .estimate_predictions(&subject, &parameters)
            .expect("replacement cache should retain its entry");
        assert_eq!(session_count.load(Ordering::SeqCst), 3);
        Equation::estimate_predictions_dense(&resized, &subject, &[0.2])
            .expect("a second key should compute");
        Equation::estimate_predictions_dense(&resized, &subject, parameters.as_slice())
            .expect("capacity-one cache should evict the first key");
        assert_eq!(session_count.load(Ordering::SeqCst), 5);

        let reenabled = model.clone().enable_cache();
        reenabled
            .estimate_predictions(&subject, &parameters)
            .expect("reenabled cache should start empty");
        reenabled
            .estimate_predictions(&subject, &parameters)
            .expect("reenabled cache should retain its entry");
        assert_eq!(session_count.load(Ordering::SeqCst), 6);

        let uncached = model.disable_cache();
        uncached
            .estimate_predictions(&subject, &parameters)
            .expect("disabled cache should compute");
        uncached
            .estimate_predictions(&subject, &parameters)
            .expect("disabled cache should recompute");
        assert_eq!(session_count.load(Ordering::SeqCst), 8);
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

    #[cfg(any(
        feature = "dsl-jit",
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        all(
            feature = "dsl-wasm",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )
    ))]
    #[test]
    fn compiled_runtime_ode_predictions_use_prefilled_cache() {
        let model = cached_runtime_ode_model();
        let compiled = CompiledRuntimeModel::Ode(model.clone());
        let subject = cached_runtime_subject();
        let parameters = Parameters::with_model(&compiled, std::iter::empty::<(&str, f64)>())
            .expect("empty parameter list should validate");
        let expected = SubjectPredictions::default();
        let key = (
            subject.hash(),
            crate::simulator::equation::parameters_hash(parameters.as_slice()),
        );

        model
            .cache
            .as_ref()
            .expect("cache should be enabled")
            .insert(key, expected.clone());

        let actual = compiled
            .estimate_predictions(&subject, &parameters)
            .expect("compiled runtime should use cached prediction");

        match actual {
            RuntimePredictions::Subject(predictions) => {
                assert_eq!(predictions.flat_predictions(), expected.flat_predictions());
            }
            RuntimePredictions::Particles(_) => {
                panic!("ODE runtime model should return subject predictions")
            }
        }

        let direct = runtime_ode_predictions(&model, &subject, parameters.as_slice())
            .expect("direct native helper should return cached prediction");
        assert_eq!(direct.flat_predictions(), expected.flat_predictions());
    }
}
