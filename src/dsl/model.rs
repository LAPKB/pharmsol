//! Compiled DSL models.
//!
//! [`JitModel`] adapts the functions in a [`RuntimeArtifact`] to the model
//! traits of every equation family, so compiled models run through the same
//! drivers as handwritten ones.

use std::sync::Arc;

use diffsol::{NalgebraContext, Vector};
use pharmsol_dsl::execution::ModelFunctionKind;
use pharmsol_dsl::{
    AnalyticalStructureInputKind, AnalyticalStructureInputPlan, ModelKind, RouteKind,
};

use super::artifact::RuntimeArtifact;
use super::model_info::{RuntimeModelInfo, RuntimeStateInfo};
use crate::{
    data::Covariates,
    simulator::{
        equation::{
            analytical::kernel_function, Analytical, AnalyticalModel, Model, OdeModel, SdeModel,
            ODE, SDE,
        },
        AnalyticalEq, V,
    },
    PharmsolError, ValidatedModelMetadata,
};

fn sorted_dense_metadata<'a, T>(
    info: &RuntimeModelInfo,
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

fn sorted_state_metadata(info: &RuntimeModelInfo) -> Result<Vec<&RuntimeStateInfo>, PharmsolError> {
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
    info: &RuntimeModelInfo,
    states: &[&'a RuntimeStateInfo],
    offset: usize,
) -> Result<(usize, &'a RuntimeStateInfo), PharmsolError> {
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

fn runtime_model_metadata(
    info: &RuntimeModelInfo,
) -> Result<ValidatedModelMetadata, PharmsolError> {
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

        if route.kind.is_none() {
            declaration = declaration.accept_either_kind();
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

/// Where compiled routes deliver their doses, by dense input index.
#[derive(Clone, Debug)]
struct RouteInputSemantics {
    /// Destination state offset of each bolus input.
    bolus_destinations: Vec<Option<usize>>,
    /// Destination state offset of each infusion input whose rate the
    /// dynamics do not read explicitly.
    injected_infusion_destinations: Vec<Option<usize>>,
}

impl RouteInputSemantics {
    fn from_model_info(info: &RuntimeModelInfo) -> Self {
        let mut bolus_destinations = vec![None; info.route_len];
        let mut injected_infusion_destinations = vec![None; info.route_len];

        for route in &info.routes {
            let bolus = route.kind != Some(RouteKind::Infusion);
            let infusion = route.kind != Some(RouteKind::Bolus);
            if bolus {
                bolus_destinations[route.index] = Some(route.destination_offset);
            }
            if infusion && route.inject_input_to_destination {
                injected_infusion_destinations[route.index] = Some(route.destination_offset);
            }
        }

        Self {
            bolus_destinations,
            injected_infusion_destinations,
        }
    }

    fn bolus_destination(&self, input: usize) -> Option<usize> {
        self.bolus_destinations.get(input).copied().flatten()
    }
}

/// A DSL model compiled to native code.
///
/// Implements the model traits of every family; [`RuntimeOdeModel`],
/// [`RuntimeAnalyticalModel`] and [`RuntimeSdeModel`] pair it with a driver.
/// Cloning is cheap: the compiled code and metadata are shared.
#[derive(Clone, Debug)]
pub struct JitModel {
    info: Arc<RuntimeModelInfo>,
    metadata: Arc<ValidatedModelMetadata>,
    route_semantics: Arc<RouteInputSemantics>,
    analytical: Option<Arc<AnalyticalPlan>>,
    artifact: Arc<dyn RuntimeArtifact>,
}

/// Scratch buffers for one [`JitModel`] occasion.
#[derive(Debug)]
pub struct JitWorkspace {
    covariates: Vec<f64>,
    derived: Vec<f64>,
    route_values: Vec<f64>,
    zero_state: Vec<f64>,
    zero_routes: Vec<f64>,
}

impl JitModel {
    pub(crate) fn new(
        info: RuntimeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        Self::from_parts(info, Arc::new(artifact))
    }

    fn from_parts(
        info: RuntimeModelInfo,
        artifact: Arc<dyn RuntimeArtifact>,
    ) -> Result<Self, PharmsolError> {
        let metadata = Arc::new(runtime_model_metadata(&info)?);
        let route_semantics = Arc::new(RouteInputSemantics::from_model_info(&info));
        let analytical = match info.analytical {
            Some(kernel) => Some(Arc::new(AnalyticalPlan {
                kernel: kernel_function(kernel),
                projection: build_analytical_parameter_projection(&info)?,
            })),
            None => None,
        };
        Ok(Self {
            info: Arc::new(info),
            metadata,
            route_semantics,
            analytical,
            artifact,
        })
    }

    /// The same compiled code with different model info.
    fn with_info(&self, info: RuntimeModelInfo) -> Result<Self, PharmsolError> {
        Self::from_parts(info, Arc::clone(&self.artifact))
    }

    /// Public names and buffer sizes of the compiled model.
    pub fn info(&self) -> &RuntimeModelInfo {
        &self.info
    }

    fn validated_metadata(&self) -> &ValidatedModelMetadata {
        &self.metadata
    }

    fn has_function(&self, role: ModelFunctionKind) -> bool {
        self.artifact.has_function(role)
    }

    /// Call one compiled function.
    ///
    /// `derived` may alias `out` (the derive function updates it in place),
    /// which is why this takes pointers rather than slices.
    #[allow(clippy::too_many_arguments)]
    unsafe fn call(
        &self,
        role: ModelFunctionKind,
        time: f64,
        state: *const f64,
        p: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError> {
        unsafe {
            self.artifact
                .invoke_raw(role, time, state, p, covariates, routes, derived, out)
        }
    }

    /// Reject a buffer shorter than the compiled code will read or write.
    ///
    /// Compiled functions take raw pointers, so this is what keeps the safe
    /// [`Model`] methods sound when they are called directly.
    fn check_len(&self, buffer: &str, len: usize, required: usize) -> Result<(), PharmsolError> {
        if len < required {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` needs a {buffer} buffer of length {required}, got {len}",
                self.info.name
            )));
        }
        Ok(())
    }

    /// Check a workspace against this model; one from another model may be
    /// too small. The zero buffers are taken out while in use and checked as
    /// the state and route arguments instead.
    fn check_workspace(&self, ws: &JitWorkspace) -> Result<(), PharmsolError> {
        let info = &self.info;
        self.check_len("covariate", ws.covariates.len(), info.covariates.len())?;
        self.check_len("derived", ws.derived.len(), info.derived_len)?;
        self.check_len("route value", ws.route_values.len(), info.route_len)
    }

    /// Interpolate covariates and run the derive function, if any, into
    /// `ws.derived`.
    ///
    /// Every compiled call goes through here first, so this also checks the
    /// workspace and the state, parameter and route buffers.
    fn refresh_derived(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        state: &[f64],
        p: &[f64],
        cov: &Covariates,
        routes: &[f64],
    ) -> Result<(), PharmsolError> {
        self.check_workspace(ws)?;
        self.check_len("state", state.len(), self.info.state_len)?;
        self.check_len("parameter", p.len(), self.info.parameters.len())?;
        self.check_len("route", routes.len(), self.info.route_len)?;
        for covariate in &self.info.covariates {
            ws.covariates[covariate.index] = match cov.get_covariate(&covariate.name) {
                Some(values) => values.interpolate(time).unwrap_or(f64::NAN),
                None => f64::NAN,
            };
        }
        if !self.has_function(ModelFunctionKind::Derive) {
            ws.derived.fill(0.0);
            return Ok(());
        }
        unsafe {
            self.call(
                ModelFunctionKind::Derive,
                time,
                state.as_ptr(),
                p.as_ptr(),
                ws.covariates.as_ptr(),
                routes.as_ptr(),
                ws.derived.as_ptr(),
                ws.derived.as_mut_ptr(),
            )
        }
    }

    /// Evaluate a per-route property (lag or bioavailability) for `input`.
    #[allow(clippy::too_many_arguments)]
    fn route_value(
        &self,
        ws: &mut JitWorkspace,
        role: ModelFunctionKind,
        default: f64,
        input: usize,
        time: f64,
        p: &[f64],
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        if !self.has_function(role) {
            return Ok(default);
        }
        // Route properties see an empty state and no active infusions.
        let zero_state = std::mem::take(&mut ws.zero_state);
        let zero_routes = std::mem::take(&mut ws.zero_routes);
        let result = self
            .refresh_derived(ws, time, &zero_state, p, cov, &zero_routes)
            .and_then(|()| {
                ws.route_values.fill(default);
                unsafe {
                    self.call(
                        role,
                        time,
                        zero_state.as_ptr(),
                        p.as_ptr(),
                        ws.covariates.as_ptr(),
                        zero_routes.as_ptr(),
                        ws.derived.as_ptr(),
                        ws.route_values.as_mut_ptr(),
                    )
                }
            });
        ws.zero_state = zero_state;
        ws.zero_routes = zero_routes;
        result.map(|()| ws.route_values[input])
    }

    /// Add the rate of each injected infusion route to its destination.
    fn inject_route_rates(&self, dx: &mut [f64], rates: &[f64]) {
        for (input, destination) in self
            .route_semantics
            .injected_infusion_destinations
            .iter()
            .enumerate()
        {
            if let Some(destination) = destination {
                dx[*destination] += rates[input];
            }
        }
    }

    /// Evaluate a state-derivative function (dynamics, drift or diffusion).
    #[allow(clippy::too_many_arguments)]
    fn derivative(
        &self,
        ws: &mut JitWorkspace,
        role: ModelFunctionKind,
        time: f64,
        x: &[f64],
        p: &[f64],
        cov: &Covariates,
        rates: &[f64],
        out: &mut [f64],
    ) -> Result<(), PharmsolError> {
        let out_len = match role {
            ModelFunctionKind::Outputs => self.info.output_len,
            _ => self.info.state_len,
        };
        self.check_len("output", out.len(), out_len)?;
        self.refresh_derived(ws, time, x, p, cov, rates)?;
        unsafe {
            self.call(
                role,
                time,
                x.as_ptr(),
                p.as_ptr(),
                ws.covariates.as_ptr(),
                rates.as_ptr(),
                ws.derived.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

impl Model for JitModel {
    type Workspace = JitWorkspace;

    fn workspace(&self) -> JitWorkspace {
        JitWorkspace {
            covariates: vec![0.0; self.info.covariates.len()],
            derived: vec![0.0; self.info.derived_len],
            route_values: vec![0.0; self.info.route_len],
            zero_state: vec![0.0; self.info.state_len],
            zero_routes: vec![0.0; self.info.route_len],
        }
    }

    fn nstates(&self) -> usize {
        self.info.state_len
    }

    fn ninputs(&self) -> usize {
        self.info.route_len
    }

    fn noutputs(&self) -> usize {
        self.info.output_len
    }

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        Some(&self.metadata)
    }

    fn check_parameters(&self, parameters: &[f64]) -> Result<(), PharmsolError> {
        if parameters.len() != self.info.parameters.len() {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` expects {} parameter value(s), got {}",
                self.info.name,
                self.info.parameters.len(),
                parameters.len()
            )));
        }
        Ok(())
    }

    fn lag(
        &self,
        ws: &mut JitWorkspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        self.route_value(
            ws,
            ModelFunctionKind::RouteLag,
            0.0,
            input,
            time,
            p.batch_as_slice(0),
            cov,
        )
    }

    fn bioavailability(
        &self,
        ws: &mut JitWorkspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        self.route_value(
            ws,
            ModelFunctionKind::RouteBioavailability,
            1.0,
            input,
            time,
            p.batch_as_slice(0),
            cov,
        )
    }

    fn init(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        p: &V,
        cov: &Covariates,
        x: &mut V,
    ) -> Result<(), PharmsolError> {
        let zero_routes = std::mem::take(&mut ws.zero_routes);
        let result = self
            .refresh_derived(
                ws,
                time,
                x.batch_as_slice(0),
                p.batch_as_slice(0),
                cov,
                &zero_routes,
            )
            .and_then(|()| {
                if !self.has_function(ModelFunctionKind::Init) {
                    return Ok(());
                }
                let x = x.batch_as_mut_slice(0);
                unsafe {
                    self.call(
                        ModelFunctionKind::Init,
                        time,
                        x.as_ptr(),
                        p.batch_as_slice(0).as_ptr(),
                        ws.covariates.as_ptr(),
                        zero_routes.as_ptr(),
                        ws.derived.as_ptr(),
                        x.as_mut_ptr(),
                    )
                }
            });
        ws.zero_routes = zero_routes;
        result
    }

    fn bolus(
        &self,
        _ws: &mut JitWorkspace,
        _time: f64,
        x: &mut V,
        _p: &V,
        _cov: &Covariates,
        input: usize,
        amount: f64,
    ) -> Result<(), PharmsolError> {
        let destination = self.route_semantics.bolus_destination(input).ok_or(
            PharmsolError::UnsupportedInputRouteKind {
                input,
                kind: RouteKind::Bolus,
            },
        )?;
        x[destination] += amount;
        Ok(())
    }

    fn outputs(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        y: &mut V,
    ) -> Result<(), PharmsolError> {
        self.derivative(
            ws,
            ModelFunctionKind::Outputs,
            time,
            x.batch_as_slice(0),
            p.batch_as_slice(0),
            cov,
            rates.batch_as_slice(0),
            y.batch_as_mut_slice(0),
        )
    }
}

impl OdeModel for JitModel {
    fn rhs(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError> {
        self.derivative(
            ws,
            ModelFunctionKind::Dynamics,
            time,
            x.batch_as_slice(0),
            p.batch_as_slice(0),
            cov,
            rates.batch_as_slice(0),
            dx.batch_as_mut_slice(0),
        )?;
        self.inject_route_rates(dx.batch_as_mut_slice(0), rates.batch_as_slice(0));
        Ok(())
    }
}

/// A DSL ODE model.
pub type RuntimeOdeModel = ODE<JitModel>;

impl ODE<JitModel> {
    pub(crate) fn compiled(
        info: RuntimeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        Ok(ODE::from_model(JitModel::new(info, artifact)?))
    }

    /// Public names and buffer sizes of the compiled model.
    pub fn info(&self) -> &RuntimeModelInfo {
        self.model().info()
    }

    /// The validated metadata of the compiled model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.model().validated_metadata()
    }
}

/// The closed-form kernel of a compiled analytical model and how to gather
/// its structural parameters.
#[derive(Debug)]
struct AnalyticalPlan {
    kernel: AnalyticalEq,
    projection: AnalyticalStructureInputKind,
}

impl AnalyticalModel for JitModel {
    fn step(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        dt: f64,
        x: &mut V,
        p: &mut V,
        cov: &Covariates,
        rates: &V,
    ) -> Result<(), PharmsolError> {
        let plan = self.analytical.as_ref().ok_or_else(|| {
            PharmsolError::OtherError(format!(
                "model `{}` does not declare an analytical function",
                self.info.name
            ))
        })?;
        self.refresh_derived(
            ws,
            time,
            x.batch_as_slice(0),
            p.batch_as_slice(0),
            cov,
            rates.batch_as_slice(0),
        )?;
        let structural =
            project_analytical_parameters(&plan.projection, p.batch_as_slice(0), &ws.derived);
        *x = (plan.kernel)(x, &structural, dt, rates, cov);
        Ok(())
    }
}

/// A DSL analytical model.
pub type RuntimeAnalyticalModel = Analytical<JitModel>;

impl Analytical<JitModel> {
    pub(crate) fn compiled(
        info: RuntimeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let model = JitModel::new(info, artifact)?;
        if model.analytical.is_none() {
            return Err(PharmsolError::OtherError(format!(
                "model `{}` does not declare an analytical function",
                model.info.name
            )));
        }
        Ok(Analytical::from_model(model))
    }

    /// Public names and buffer sizes of the compiled model.
    pub fn info(&self) -> &RuntimeModelInfo {
        self.model().info()
    }

    /// The validated metadata of the compiled model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.model().validated_metadata()
    }
}

impl SdeModel for JitModel {
    fn drift(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError> {
        self.derivative(
            ws,
            ModelFunctionKind::Drift,
            time,
            x.batch_as_slice(0),
            p.batch_as_slice(0),
            cov,
            rates.batch_as_slice(0),
            dx.batch_as_mut_slice(0),
        )?;
        self.inject_route_rates(dx.batch_as_mut_slice(0), rates.batch_as_slice(0));
        Ok(())
    }

    fn diffusion(
        &self,
        ws: &mut JitWorkspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        sigma: &mut V,
    ) -> Result<(), PharmsolError> {
        self.derivative(
            ws,
            ModelFunctionKind::Diffusion,
            time,
            x.batch_as_slice(0),
            p.batch_as_slice(0),
            cov,
            rates.batch_as_slice(0),
            sigma.batch_as_mut_slice(0),
        )
    }
}

/// A DSL SDE model.
pub type RuntimeSdeModel = SDE<JitModel>;

impl SDE<JitModel> {
    pub(crate) fn compiled(
        info: RuntimeModelInfo,
        artifact: impl RuntimeArtifact + 'static,
    ) -> Result<Self, PharmsolError> {
        let nparticles = info
            .particles
            .ok_or_else(|| PharmsolError::InvalidMetadata {
                model: info.name.clone(),
                detail: "SDE models must declare a particle count".to_string(),
            })?;
        Ok(SDE::from_model(JitModel::new(info, artifact)?, nparticles))
    }

    /// Use a different particle count. The model metadata follows.
    pub fn with_particles(self, nparticles: usize) -> Self {
        if self.nparticles() == nparticles {
            return self;
        }

        let mut info = self.info().clone();
        info.particles = Some(nparticles);
        let model = self
            .model()
            .with_info(info)
            .expect("compiled SDE metadata should stay valid after particle override");
        self.with_model(model, nparticles)
    }

    /// Public names and buffer sizes of the compiled model.
    pub fn info(&self) -> &RuntimeModelInfo {
        self.model().info()
    }

    /// The validated metadata of the compiled model.
    pub fn metadata(&self) -> &ValidatedModelMetadata {
        self.model().validated_metadata()
    }
}

fn build_analytical_parameter_projection(
    info: &RuntimeModelInfo,
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

#[cfg(test)]
mod tests {
    use super::{
        build_analytical_parameter_projection, project_analytical_parameters, JitModel,
        RuntimeAnalyticalModel, RuntimeArtifact, RuntimeModelInfo, RuntimeOdeModel,
        RuntimeSdeModel, RuntimeStateInfo,
    };
    use crate::dsl::model_info::RuntimeRouteInfo;
    use crate::dsl::model_info::{RuntimeCovariateInfo, RuntimeOutputInfo};
    use crate::simulator::equation::model::resolve_input;
    use crate::simulator::equation::Cache;
    use crate::simulator::likelihood::log_likelihood_matrix;
    use crate::InputLabel;
    use crate::PharmsolError;
    use crate::{
        data::builder::SubjectBuilderExt,
        dsl::{CompiledRuntimeModel, RuntimePredictions},
        prelude::SubjectPredictions,
        Parameters, Subject,
    };
    use crate::{Data, Equation, OutputModels};
    use ndarray::array;
    use pharmsol_dsl::execution::ModelFunctionKind;
    use pharmsol_dsl::{
        AnalyticalKernel, AnalyticalStructureInputKind, CovariateInterpolation, ModelKind,
        RouteKind,
    };

    #[derive(Debug)]
    struct DummyArtifact;

    impl RuntimeArtifact for DummyArtifact {
        fn has_function(&self, _role: ModelFunctionKind) -> bool {
            false
        }

        unsafe fn invoke_raw(
            &self,
            _role: ModelFunctionKind,
            _time: f64,
            _states: *const f64,
            _params: *const f64,
            _covariates: *const f64,
            _routes: *const f64,
            _derived: *const f64,
            _out: *mut f64,
        ) -> Result<(), PharmsolError> {
            panic!("dummy artifact functions should not be called in tests")
        }
    }

    fn bolus_only_model() -> JitModel {
        JitModel::new(
            RuntimeModelInfo {
                name: "bolus_only".to_string(),
                kind: ModelKind::Ode,
                parameters: Vec::new(),
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![RuntimeStateInfo {
                    name: "gut".to_string(),
                    offset: 0,
                }],
                routes: vec![RuntimeRouteInfo {
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
                outputs: vec![RuntimeOutputInfo {
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
    ) -> RuntimeModelInfo {
        RuntimeModelInfo {
            name: "analytical_projection".to_string(),
            kind: ModelKind::Analytical,
            parameters: parameters.iter().map(|name| (*name).to_string()).collect(),
            derived: derived.iter().map(|name| (*name).to_string()).collect(),
            covariates: Vec::new(),
            states: (0..function.state_count())
                .map(|offset| RuntimeStateInfo {
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
        let model = RuntimeOdeModel::compiled(
            RuntimeModelInfo {
                name: "runtime_metadata".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string(), "v".to_string()],
                derived: Vec::new(),
                covariates: vec![RuntimeCovariateInfo {
                    name: "wt".to_string(),
                    index: 0,
                    interpolation: Some(CovariateInterpolation::Linear),
                }],
                states: vec![RuntimeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: vec![RuntimeRouteInfo {
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
                outputs: vec![RuntimeOutputInfo {
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

        let compiled = CompiledRuntimeModel::Ode(model.clone());
        assert_eq!(
            compiled.metadata().route("iv").unwrap().destination(),
            "central"
        );
    }

    #[test]
    fn runtime_ode_models_map_array_state_offsets_to_declarations() {
        let model = RuntimeOdeModel::compiled(
            RuntimeModelInfo {
                name: "array_state_runtime_metadata".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string(), "v".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![
                    RuntimeStateInfo {
                        name: "transit".to_string(),
                        offset: 0,
                    },
                    RuntimeStateInfo {
                        name: "central".to_string(),
                        offset: 4,
                    },
                ],
                routes: vec![
                    RuntimeRouteInfo {
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
                    RuntimeRouteInfo {
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
                outputs: vec![RuntimeOutputInfo {
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
        let error = RuntimeOdeModel::compiled(
            RuntimeModelInfo {
                name: "runtime_metadata_invalid_destination".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![RuntimeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: vec![RuntimeRouteInfo {
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
                outputs: vec![RuntimeOutputInfo {
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
        let model = RuntimeSdeModel::compiled(
            RuntimeModelInfo {
                name: "runtime_sde_particles".to_string(),
                kind: ModelKind::Sde,
                parameters: vec!["ke".to_string()],
                derived: Vec::new(),
                covariates: Vec::new(),
                states: vec![RuntimeStateInfo {
                    name: "central".to_string(),
                    offset: 0,
                }],
                routes: Vec::new(),
                outputs: vec![RuntimeOutputInfo {
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

    fn analytical_projection_values(
        model: &RuntimeAnalyticalModel,
        support_point: &[f64],
        derived: &[f64],
    ) -> Vec<f64> {
        let plan = model.model().analytical.as_ref().expect("analytical plan");
        project_analytical_parameters(&plan.projection, support_point, derived)
            .batch_as_slice(0)
            .to_vec()
    }

    fn cached_runtime_ode_model() -> RuntimeOdeModel {
        RuntimeOdeModel::from_model(bolus_only_model()).with_cache_capacity(1)
    }

    fn cached_runtime_subject() -> Subject {
        Subject::builder("runtime_cached_prediction")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .build()
    }

    #[test]
    fn infusion_on_bolus_route_reports_structured_route_kind_error() {
        let model = bolus_only_model();

        let error = resolve_input(&model, &InputLabel::new("oral"), RouteKind::Infusion)
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
        let model = RuntimeAnalyticalModel::compiled(
            analytical_model_info(
                &["ka", "ke", "v"],
                &[],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model
                .model()
                .analytical
                .as_ref()
                .expect("analytical plan")
                .projection,
            AnalyticalStructureInputKind::AllPrimary { identity: true, .. }
        ));
        assert_eq!(
            analytical_projection_values(&model, &[1.0, 0.15, 25.0], &[]),
            vec![1.0, 0.15]
        );
    }

    #[test]
    fn compiled_analytical_projection_reorders_all_derived_inputs() {
        let model = RuntimeAnalyticalModel::compiled(
            analytical_model_info(
                &["ke0", "v"],
                &["ke", "ka"],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model
                .model()
                .analytical
                .as_ref()
                .expect("analytical plan")
                .projection,
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
        let model = RuntimeAnalyticalModel::compiled(
            analytical_model_info(
                &["ka", "v", "ke0"],
                &["ke"],
                AnalyticalKernel::OneCompartmentWithAbsorption,
            ),
            DummyArtifact,
        )
        .expect("analytical model builds");

        assert!(matches!(
            model
                .model()
                .analytical
                .as_ref()
                .expect("analytical plan")
                .projection,
            AnalyticalStructureInputKind::Mixed { .. }
        ));
        assert_eq!(
            analytical_projection_values(&model, &[1.0, 25.0, 0.2], &[0.15]),
            vec![1.0, 0.15]
        );
    }

    #[test]
    fn compiled_analytical_projection_reports_missing_required_name_at_setup() {
        let error = RuntimeAnalyticalModel::compiled(
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
        let error = RuntimeAnalyticalModel::compiled(
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

    #[test]
    fn runtime_analytical_dense_entry_points_use_prefilled_cache() {
        let model = RuntimeAnalyticalModel::compiled(
            analytical_model_info(&["ke", "v"], &[], AnalyticalKernel::OneCompartment),
            DummyArtifact,
        )
        .expect("analytical model builds")
        .with_cache_capacity(1);
        // DummyArtifact panics if a function is called, so any real solve fails the test.
        let subject = Subject::builder("runtime_cached_analytical").build();
        let support_point = [0.2, 20.0];
        let expected = SubjectPredictions::default();
        model
            .prediction_cache()
            .expect("cache should be enabled")
            .insert(
                model.prediction_key(&subject, &support_point),
                std::sync::Arc::new(expected.clone()),
            );

        let dense = Equation::predict(&model, &subject, &support_point)
            .expect("dense predictions should come from the cache");
        assert_eq!(dense.flat_predictions(), expected.flat_predictions());

        let parameters =
            Parameters::with_model(&model, [("ke", 0.2), ("v", 20.0)]).expect("valid parameters");
        let predictions = Equation::predict(&model, &subject, &parameters)
            .expect("predictions should come from the cache");
        assert_eq!(predictions.flat_predictions(), expected.flat_predictions());

        let matrix = log_likelihood_matrix(
            &model,
            &Data::new(vec![subject]),
            &array![[0.2, 20.0]],
            &OutputModels::new(),
            false,
        )
        .expect("matrix should be served from the cache");
        assert_eq!(matrix[[0, 0]], 0.0);
    }

    #[test]
    fn compiled_runtime_ode_predictions_use_prefilled_cache() {
        let model = cached_runtime_ode_model();
        let compiled = CompiledRuntimeModel::Ode(model.clone());
        let subject = cached_runtime_subject();
        let parameters = Parameters::with_model(&compiled, std::iter::empty::<(&str, f64)>())
            .expect("empty parameter list should validate");
        let expected = SubjectPredictions::default();
        let key = model.prediction_key(&subject, &parameters);

        model
            .prediction_cache()
            .expect("cache should be enabled")
            .insert(key, std::sync::Arc::new(expected.clone()));

        let actual = compiled
            .predict(&subject, &parameters)
            .expect("compiled runtime should use cached prediction");

        match actual {
            RuntimePredictions::Subject(predictions) => {
                assert_eq!(predictions.flat_predictions(), expected.flat_predictions());
            }
            RuntimePredictions::Particles(_) => {
                panic!("ODE runtime model should return subject predictions")
            }
        }

        let direct = model
            .predict(&subject, &parameters)
            .expect("the equation should return the cached prediction");
        assert_eq!(direct.flat_predictions(), expected.flat_predictions());
    }

    /// Calling the safe `Model` methods directly with undersized buffers must
    /// be an error, not an out-of-bounds access in compiled code. The dummy
    /// artifact panics if it is ever reached.
    #[test]
    fn direct_model_calls_reject_undersized_buffers() {
        use crate::data::Covariates;
        use crate::simulator::equation::{Model, OdeModel};
        use crate::simulator::V;
        use diffsol::{NalgebraContext, Vector};

        let model = bolus_only_model();
        let vec = |len: usize| V::from_vec(vec![0.0; len], NalgebraContext::new());
        let cov = Covariates::new();
        let message = |result: Result<(), PharmsolError>| {
            result
                .expect_err("an undersized buffer must be rejected")
                .to_string()
        };

        let mut ws = model.workspace();
        let mut y = vec(0);
        assert!(
            message(model.outputs(&mut ws, 0.0, &vec(1), &vec(0), &cov, &vec(1), &mut y))
                .contains("output buffer")
        );

        let mut dx = vec(1);
        assert!(
            message(model.rhs(&mut ws, 0.0, &vec(0), &vec(0), &cov, &vec(1), &mut dx))
                .contains("state buffer")
        );
        assert!(
            message(model.rhs(&mut ws, 0.0, &vec(1), &vec(0), &cov, &vec(0), &mut dx))
                .contains("route buffer")
        );

        let mut foreign = super::JitWorkspace {
            covariates: Vec::new(),
            derived: Vec::new(),
            route_values: Vec::new(),
            zero_state: Vec::new(),
            zero_routes: Vec::new(),
        };
        let mut y = vec(1);
        assert!(
            message(model.outputs(&mut foreign, 0.0, &vec(1), &vec(0), &cov, &vec(1), &mut y))
                .contains("buffer of length")
        );
    }
}
