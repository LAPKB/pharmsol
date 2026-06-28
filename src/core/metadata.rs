//! Metadata builders and validated metadata views for handwritten models.
//!
//! Use this module when a handwritten [`crate::ODE`], [`crate::Analytical`], or
//! [`crate::SDE`] model should expose the same public names that appear in data
//! rows, subject builders, or parsed files.
//!
//! Metadata gives names to parameters, covariates, states, routes, and outputs.
//! After validation, the execution layer can resolve public labels such as
//! `"iv"` and `"cp"` against those declarations before simulation.
//!
//! Without metadata, handwritten models fall back to numeric labels. With
//! metadata, labels are matched by name.
//!
//! # Example
//!
//! ```rust
//! use pharmsol::{metadata, ModelKind};
//!
//! let metadata = metadata::new("one_cmt")
//!     .kind(ModelKind::Ode)
//!     .parameters(["cl", "v"])
//!     .states(["central"])
//!     .outputs(["cp"])
//!     .route(metadata::Route::infusion("iv").to_state("central"))
//!     .validate()
//!     .unwrap();
//!
//! assert_eq!(metadata.name(), "one_cmt");
//! assert_eq!(metadata.route("iv").unwrap().destination(), "central");
//! assert!(metadata.output("cp").is_some());
//! ```

use pharmsol_dsl::{AnalyticalKernel, CovariateInterpolation, ModelKind};
use std::fmt;
use thiserror::Error;

/// Shorthand for [`ModelMetadata::new`].
pub fn new(name: impl Into<String>) -> ModelMetadata {
    ModelMetadata::new(name)
}

/// Validation errors for handwritten model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelMetadataError {
    #[error("model kind is required for metadata validation")]
    MissingModelKind,
    #[error("metadata declares kind `{declared:?}` but validation requested `{requested:?}`")]
    ModelKindConflict {
        declared: ModelKind,
        requested: ModelKind,
    },
    #[error("duplicate {domain} name `{name}`")]
    DuplicateName { domain: NameDomain, name: String },
    #[error("route `{route}` must declare a destination state")]
    MissingRouteDestination { route: String },
    #[error("route `{route}` targets unknown state `{destination}`")]
    UnknownRouteDestination { route: String, destination: String },
    #[error("infusion route `{route}` cannot declare lag")]
    InfusionLagNotAllowed { route: String },
    #[error("infusion route `{route}` cannot declare bioavailability")]
    InfusionBioavailabilityNotAllowed { route: String },
    #[error("{kind:?} metadata cannot declare particles")]
    ParticlesNotAllowed { kind: ModelKind },
    #[error("Sde metadata requires particles")]
    MissingParticles,
    #[error(
        "metadata declares {declared} particle(s) but validation provided {fallback} fallback particle(s)"
    )]
    ParticleCountConflict { declared: usize, fallback: usize },
    #[error("{kind:?} metadata cannot declare an analytical kernel")]
    AnalyticalKernelNotAllowed { kind: ModelKind },
}

/// Name domain used in duplicate-name validation messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameDomain {
    Parameter,
    Covariate,
    State,
    Route,
    Output,
}

impl fmt::Display for NameDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let domain = match self {
            Self::Parameter => "parameter",
            Self::Covariate => "covariate",
            Self::State => "state",
            Self::Route => "route",
            Self::Output => "output",
        };
        f.write_str(domain)
    }
}

/// Validated metadata view used by the execution layer.
///
/// This type is what handwritten equation builders store after metadata has
/// passed validation. It provides stable lookup helpers from public names to the
/// dense indices used during execution.
///
/// Route lookups expose two different indices:
/// - [`ValidatedModelMetadata::route_declaration_index`] is the route position in
///   declaration order.
/// - [`ValidatedModelMetadata::route_index`] is the dense execution input index
///   for that route kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedModelMetadata {
    name: String,
    kind: ModelKind,
    parameters: Vec<Parameter>,
    covariates: Vec<Covariate>,
    states: Vec<State>,
    routes: Vec<ValidatedRoute>,
    route_input_count: usize,
    outputs: Vec<Output>,
    particles: Option<usize>,
    analytical: Option<AnalyticalKernel>,
}

impl ValidatedModelMetadata {
    /// Get the public model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the validated model family.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    pub fn covariates(&self) -> &[Covariate] {
        &self.covariates
    }

    pub fn states(&self) -> &[State] {
        &self.states
    }

    pub fn routes(&self) -> &[ValidatedRoute] {
        &self.routes
    }

    /// Get the number of dense execution input slots needed for routes.
    ///
    /// This is the maximum of the bolus-route count and infusion-route count.
    pub fn route_input_count(&self) -> usize {
        self.route_input_count
    }

    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    pub fn particles(&self) -> Option<usize> {
        self.particles
    }

    pub fn analytical_kernel(&self) -> Option<AnalyticalKernel> {
        self.analytical
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.parameters
            .iter()
            .position(|parameter| parameter.name() == name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.covariates
            .iter()
            .position(|covariate| covariate.name() == name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|state| state.name() == name)
    }

    /// Look up a route by public name and return its declaration-order index.
    pub fn route_declaration_index(&self, name: &str) -> Option<usize> {
        self.routes.iter().position(|route| route.name() == name)
    }

    /// Look up an output by public name and return its dense output index.
    pub(crate) fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|output| output.name() == name)
    }

    pub fn parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameter_index(name)
            .map(|index| &self.parameters[index])
    }

    pub fn covariate(&self, name: &str) -> Option<&Covariate> {
        self.covariate_index(name)
            .map(|index| &self.covariates[index])
    }

    pub fn state(&self, name: &str) -> Option<&State> {
        self.state_index(name).map(|index| &self.states[index])
    }

    pub fn route(&self, name: &str) -> Option<&ValidatedRoute> {
        self.route_declaration_index(name)
            .map(|index| &self.routes[index])
    }

    pub fn output(&self, name: &str) -> Option<&Output> {
        self.output_index(name).map(|index| &self.outputs[index])
    }
}

/// One validated route declaration with resolved execution details.
///
/// A validated route keeps both the declaration-order index and the dense input
/// index used during execution. Those values can differ from each other when a
/// model mixes bolus and infusion routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedRoute {
    name: String,
    kind: RouteKind,
    declaration_index: usize,
    input_index: usize,
    destination: String,
    destination_index: usize,
    has_lag: bool,
    has_bioavailability: bool,
    input_policy: Option<RouteInputPolicy>,
}

impl ValidatedRoute {
    /// Get the public route name used for label matching.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> RouteKind {
        self.kind
    }

    /// Get the declaration-order index for this route.
    pub fn declaration_index(&self) -> usize {
        self.declaration_index
    }

    /// Get the dense execution input index for this route kind.
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Get the destination state name.
    pub fn destination(&self) -> &str {
        &self.destination
    }

    /// Get the destination state index in model order.
    pub fn destination_index(&self) -> usize {
        self.destination_index
    }

    pub fn has_lag(&self) -> bool {
        self.has_lag
    }

    pub fn has_bioavailability(&self) -> bool {
        self.has_bioavailability
    }

    pub fn input_policy(&self) -> Option<RouteInputPolicy> {
        self.input_policy
    }
}

/// Builder for handwritten model metadata.
///
/// Use [`ModelMetadata`] to declare the public names that should be attached to
/// a handwritten equation. After validation, the resulting metadata can be
/// attached to handwritten [`crate::ODE`], [`crate::Analytical`], and
/// [`crate::SDE`] models through their `with_metadata(...)` methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    name: String,
    kind: Option<ModelKind>,
    parameters: Vec<Parameter>,
    covariates: Vec<Covariate>,
    states: Vec<State>,
    routes: Vec<Route>,
    outputs: Vec<Output>,
    particles: Option<usize>,
    analytical: Option<AnalyticalKernel>,
}

impl ModelMetadata {
    /// Create a new metadata builder with a model name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: None,
            parameters: Vec::new(),
            covariates: Vec::new(),
            states: Vec::new(),
            routes: Vec::new(),
            outputs: Vec::new(),
            particles: None,
            analytical: None,
        }
    }

    /// Set the model kind explicitly.
    pub fn kind(mut self, kind: ModelKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Replace the ordered parameter list.
    pub fn parameters<I, S>(mut self, parameters: I) -> Self
    where
        I: IntoIterator<Item = S>,
        Parameter: From<S>,
    {
        self.parameters = parameters.into_iter().map(Parameter::from).collect();
        self
    }

    /// Replace the ordered covariate list.
    pub fn covariates<I>(mut self, covariates: I) -> Self
    where
        I: IntoIterator<Item = Covariate>,
    {
        self.covariates = covariates.into_iter().collect();
        self
    }

    /// Replace the ordered state list.
    pub fn states<I, S>(mut self, states: I) -> Self
    where
        I: IntoIterator<Item = S>,
        State: From<S>,
    {
        self.states = states.into_iter().map(State::from).collect();
        self
    }

    /// Add one route declaration.
    pub fn route(mut self, route: Route) -> Self {
        self.routes.push(route);
        self
    }

    /// Extend with multiple route declarations.
    pub fn routes<I>(mut self, routes: I) -> Self
    where
        I: IntoIterator<Item = Route>,
    {
        self.routes.extend(routes);
        self
    }

    /// Replace the ordered output list.
    pub fn outputs<I, S>(mut self, outputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        Output: From<S>,
    {
        self.outputs = outputs.into_iter().map(Output::from).collect();
        self
    }

    /// Set the particle count for stochastic models.
    pub fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self
    }

    /// Set the analytical kernel identity for built-in analytical models.
    pub fn analytical_kernel(mut self, analytical: AnalyticalKernel) -> Self {
        self.analytical = Some(analytical);
        self
    }

    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the explicit model kind, if already declared.
    pub fn kind_decl(&self) -> Option<ModelKind> {
        self.kind
    }

    /// Get the ordered parameter metadata.
    pub fn parameters_decl(&self) -> &[Parameter] {
        &self.parameters
    }

    /// Get the ordered covariate metadata.
    pub fn covariates_decl(&self) -> &[Covariate] {
        &self.covariates
    }

    /// Get the ordered state metadata.
    pub fn states_decl(&self) -> &[State] {
        &self.states
    }

    /// Get the ordered route metadata.
    pub fn routes_decl(&self) -> &[Route] {
        &self.routes
    }

    /// Get the ordered output metadata.
    pub fn outputs_decl(&self) -> &[Output] {
        &self.outputs
    }

    /// Get the declared particle count.
    pub fn particles_decl(&self) -> Option<usize> {
        self.particles
    }

    /// Get the declared analytical kernel identity.
    pub fn analytical_kernel_decl(&self) -> Option<AnalyticalKernel> {
        self.analytical
    }

    /// Validate this metadata using its declared kind.
    ///
    /// Use this when the metadata itself already declares whether the model is
    /// ODE, analytical, or SDE.
    pub fn validate(self) -> Result<ValidatedModelMetadata, ModelMetadataError> {
        self.validate_internal(None, None)
    }

    /// Validate this metadata for a specific model kind.
    ///
    /// Use this when the equation type determines the model family and you want
    /// validation to enforce that family explicitly.
    pub fn validate_for(
        self,
        kind: ModelKind,
    ) -> Result<ValidatedModelMetadata, ModelMetadataError> {
        self.validate_internal(Some(kind), None)
    }

    /// Validate this metadata for a specific model kind, using a fallback
    /// particle count when the metadata itself does not declare one.
    pub fn validate_for_with_particles(
        self,
        kind: ModelKind,
        fallback_particles: usize,
    ) -> Result<ValidatedModelMetadata, ModelMetadataError> {
        self.validate_internal(Some(kind), Some(fallback_particles))
    }

    fn validate_internal(
        self,
        requested_kind: Option<ModelKind>,
        fallback_particles: Option<usize>,
    ) -> Result<ValidatedModelMetadata, ModelMetadataError> {
        let kind = resolve_kind(self.kind, requested_kind)?;
        validate_unique_names(&self.parameters, NameDomain::Parameter, Parameter::name)?;
        validate_unique_names(&self.covariates, NameDomain::Covariate, Covariate::name)?;
        validate_unique_names(&self.states, NameDomain::State, State::name)?;
        validate_unique_names(&self.routes, NameDomain::Route, Route::name)?;
        validate_unique_names(&self.outputs, NameDomain::Output, Output::name)?;

        let particles = resolve_particles(kind, self.particles, fallback_particles)?;
        validate_kind_specific_fields(kind, self.analytical, particles)?;

        let (routes, route_input_count) = validate_routes(self.routes, &self.states)?;

        Ok(ValidatedModelMetadata {
            name: self.name,
            kind,
            parameters: self.parameters,
            covariates: self.covariates,
            states: self.states,
            routes,
            route_input_count,
            outputs: self.outputs,
            particles,
            analytical: self.analytical,
        })
    }
}

/// One named parameter in model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    name: String,
}

impl Parameter {
    /// Create a named parameter declaration.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<S> From<S> for Parameter
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

/// One named covariate plus interpolation semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Covariate {
    name: String,
    interpolation: Option<CovariateInterpolation>,
}

impl Covariate {
    /// Create a named covariate without an explicit interpolation policy.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            interpolation: None,
        }
    }

    /// Create a continuous covariate that uses linear interpolation.
    pub fn continuous(name: impl Into<String>) -> Self {
        Self::new(name).with_interpolation(CovariateInterpolation::Linear)
    }

    /// Create a covariate that uses last-observation-carried-forward semantics.
    pub fn locf(name: impl Into<String>) -> Self {
        Self::new(name).with_interpolation(CovariateInterpolation::Locf)
    }

    /// Set the interpolation policy explicitly.
    pub fn with_interpolation(mut self, interpolation: CovariateInterpolation) -> Self {
        self.interpolation = Some(interpolation);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn interpolation(&self) -> Option<CovariateInterpolation> {
        self.interpolation
    }
}

/// One named state in model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    name: String,
}

impl State {
    /// Create a named state declaration.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<S> From<S> for State
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

/// One named output in model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    name: String,
}

impl Output {
    /// Create a named output declaration.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<S> From<S> for Output
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

/// Route declaration kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteKind {
    /// Instantaneous dose input.
    Bolus,
    /// Dose input over a duration.
    Infusion,
}

/// How route inputs should be interpreted by the execution layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteInputPolicy {
    /// Inject the resolved input directly into the declared destination state.
    InjectToDestination,
    /// Expect the low-level execution path to provide an explicit input vector.
    ExplicitInputVector,
}

/// One named route declaration.
///
/// Route names are the public labels matched against dose events such as `iv`
/// or `oral`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    name: String,
    kind: RouteKind,
    destination: Option<String>,
    has_lag: bool,
    has_bioavailability: bool,
    input_policy: Option<RouteInputPolicy>,
}

impl Route {
    /// Create a named bolus route declaration.
    pub fn bolus(name: impl Into<String>) -> Self {
        Self::new(name, RouteKind::Bolus)
    }

    /// Create a named infusion route declaration.
    pub fn infusion(name: impl Into<String>) -> Self {
        Self::new(name, RouteKind::Infusion)
    }

    /// Create a route declaration with an explicit kind.
    pub fn new(name: impl Into<String>, kind: RouteKind) -> Self {
        Self {
            name: name.into(),
            kind,
            destination: None,
            has_lag: false,
            has_bioavailability: false,
            input_policy: None,
        }
    }

    /// Declare which state this route targets.
    pub fn to_state(mut self, destination: impl Into<String>) -> Self {
        self.destination = Some(destination.into());
        self
    }

    /// Mark this route as supporting lag handling.
    pub fn with_lag(mut self) -> Self {
        self.has_lag = true;
        self
    }

    /// Mark this route as supporting bioavailability handling.
    pub fn with_bioavailability(mut self) -> Self {
        self.has_bioavailability = true;
        self
    }

    /// Request direct injection into the destination state at execution time.
    pub fn inject_input_to_destination(mut self) -> Self {
        self.input_policy = Some(RouteInputPolicy::InjectToDestination);
        self
    }

    /// Request an explicit low-level input vector at execution time.
    pub fn expect_explicit_input(mut self) -> Self {
        self.input_policy = Some(RouteInputPolicy::ExplicitInputVector);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> RouteKind {
        self.kind
    }

    pub fn destination(&self) -> Option<&str> {
        self.destination.as_deref()
    }

    pub fn has_lag(&self) -> bool {
        self.has_lag
    }

    pub fn has_bioavailability(&self) -> bool {
        self.has_bioavailability
    }

    pub fn input_policy(&self) -> Option<RouteInputPolicy> {
        self.input_policy
    }
}

fn resolve_kind(
    declared_kind: Option<ModelKind>,
    requested_kind: Option<ModelKind>,
) -> Result<ModelKind, ModelMetadataError> {
    match (declared_kind, requested_kind) {
        (Some(declared), Some(requested)) if declared != requested => {
            Err(ModelMetadataError::ModelKindConflict {
                declared,
                requested,
            })
        }
        (Some(declared), _) => Ok(declared),
        (None, Some(requested)) => Ok(requested),
        (None, None) => Err(ModelMetadataError::MissingModelKind),
    }
}

fn resolve_particles(
    kind: ModelKind,
    declared_particles: Option<usize>,
    fallback_particles: Option<usize>,
) -> Result<Option<usize>, ModelMetadataError> {
    let particles = match (declared_particles, fallback_particles) {
        (Some(declared), Some(fallback)) if declared != fallback => {
            return Err(ModelMetadataError::ParticleCountConflict { declared, fallback });
        }
        (Some(declared), _) => Some(declared),
        (None, Some(fallback)) => Some(fallback),
        (None, None) => None,
    };

    match kind {
        ModelKind::Ode | ModelKind::Analytical if particles.is_some() => {
            Err(ModelMetadataError::ParticlesNotAllowed { kind })
        }
        ModelKind::Sde if particles.is_none() => Err(ModelMetadataError::MissingParticles),
        _ => Ok(particles),
    }
}

fn validate_kind_specific_fields(
    kind: ModelKind,
    analytical: Option<AnalyticalKernel>,
    particles: Option<usize>,
) -> Result<(), ModelMetadataError> {
    match kind {
        ModelKind::Ode => {
            if analytical.is_some() {
                return Err(ModelMetadataError::AnalyticalKernelNotAllowed { kind });
            }
            if particles.is_some() {
                return Err(ModelMetadataError::ParticlesNotAllowed { kind });
            }
        }
        ModelKind::Analytical => {
            if particles.is_some() {
                return Err(ModelMetadataError::ParticlesNotAllowed { kind });
            }
        }
        ModelKind::Sde => {
            if analytical.is_some() {
                return Err(ModelMetadataError::AnalyticalKernelNotAllowed { kind });
            }
        }
    }
    Ok(())
}

fn validate_unique_names<T>(
    values: &[T],
    domain: NameDomain,
    name_of: impl Fn(&T) -> &str,
) -> Result<(), ModelMetadataError> {
    let mut names = std::collections::HashSet::with_capacity(values.len());
    for value in values {
        let name = name_of(value);
        if !names.insert(name) {
            return Err(ModelMetadataError::DuplicateName {
                domain,
                name: name.to_string(),
            });
        }
    }
    Ok(())
}

fn validate_routes(
    routes: Vec<Route>,
    states: &[State],
) -> Result<(Vec<ValidatedRoute>, usize), ModelMetadataError> {
    let mut bolus_inputs = 0;
    let mut infusion_inputs = 0;
    let mut validated_routes = Vec::with_capacity(routes.len());

    for (declaration_index, route) in routes.into_iter().enumerate() {
        let input_index = match route.kind {
            RouteKind::Bolus => {
                let index = bolus_inputs;
                bolus_inputs += 1;
                index
            }
            RouteKind::Infusion => {
                let index = infusion_inputs;
                infusion_inputs += 1;
                index
            }
        };

        validated_routes.push(validate_route(
            route,
            declaration_index,
            input_index,
            states,
        )?);
    }

    Ok((validated_routes, bolus_inputs.max(infusion_inputs)))
}

fn validate_route(
    route: Route,
    declaration_index: usize,
    input_index: usize,
    states: &[State],
) -> Result<ValidatedRoute, ModelMetadataError> {
    if route.kind == RouteKind::Infusion && route.has_lag {
        return Err(ModelMetadataError::InfusionLagNotAllowed {
            route: route.name.clone(),
        });
    }

    if route.kind == RouteKind::Infusion && route.has_bioavailability {
        return Err(ModelMetadataError::InfusionBioavailabilityNotAllowed {
            route: route.name.clone(),
        });
    }

    let destination =
        route
            .destination
            .clone()
            .ok_or_else(|| ModelMetadataError::MissingRouteDestination {
                route: route.name.clone(),
            })?;
    let destination_index = states
        .iter()
        .position(|state| state.name() == destination)
        .ok_or_else(|| ModelMetadataError::UnknownRouteDestination {
            route: route.name.clone(),
            destination: destination.clone(),
        })?;

    Ok(ValidatedRoute {
        name: route.name,
        kind: route.kind,
        declaration_index,
        input_index,
        destination,
        destination_index,
        has_lag: route.has_lag,
        has_bioavailability: route.has_bioavailability,
        input_policy: route.input_policy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_ode_metadata_shape() {
        let metadata = new("bimodal_ke")
            .kind(ModelKind::Ode)
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"));

        assert_eq!(metadata.name(), "bimodal_ke");
        assert_eq!(metadata.kind_decl(), Some(ModelKind::Ode));
        assert_eq!(metadata.parameters_decl()[0].name(), "ke");
        assert_eq!(metadata.parameters_decl()[1].name(), "v");
        assert_eq!(metadata.states_decl()[0].name(), "central");
        assert_eq!(metadata.outputs_decl()[0].name(), "cp");
        assert_eq!(metadata.routes_decl()[0].name(), "iv");
        assert_eq!(metadata.routes_decl()[0].kind(), RouteKind::Infusion);
        assert_eq!(metadata.routes_decl()[0].destination(), Some("central"));
    }

    #[test]
    fn builds_analytical_metadata_shape() {
        let metadata = new("one_cmt_abs")
            .kind(ModelKind::Analytical)
            .parameters(["ka", "ke", "v"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .route(Route::bolus("oral").to_state("gut").with_bioavailability())
            .route(Route::infusion("iv").to_state("central"))
            .analytical_kernel(AnalyticalKernel::OneCompartmentWithAbsorption);

        assert_eq!(metadata.kind_decl(), Some(ModelKind::Analytical));
        assert_eq!(metadata.states_decl()[0].name(), "gut");
        assert_eq!(metadata.states_decl()[1].name(), "central");
        assert_eq!(metadata.routes_decl()[0].kind(), RouteKind::Bolus);
        assert!(metadata.routes_decl()[0].has_bioavailability());
        assert_eq!(
            metadata.analytical_kernel_decl(),
            Some(AnalyticalKernel::OneCompartmentWithAbsorption)
        );
    }

    #[test]
    fn builds_sde_metadata_shape() {
        let metadata = new("one_cmt_sde")
            .kind(ModelKind::Sde)
            .parameters(["ke", "sigma", "v"])
            .covariates([Covariate::continuous("wt"), Covariate::locf("age")])
            .states(["central"])
            .outputs(["cp"])
            .route(
                Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            )
            .particles(128);

        assert_eq!(metadata.kind_decl(), Some(ModelKind::Sde));
        assert_eq!(metadata.covariates_decl()[0].name(), "wt");
        assert_eq!(
            metadata.covariates_decl()[0].interpolation(),
            Some(CovariateInterpolation::Linear)
        );
        assert_eq!(metadata.covariates_decl()[1].name(), "age");
        assert_eq!(
            metadata.covariates_decl()[1].interpolation(),
            Some(CovariateInterpolation::Locf)
        );
        assert_eq!(metadata.particles_decl(), Some(128));
        assert_eq!(
            metadata.routes_decl()[0].input_policy(),
            Some(RouteInputPolicy::InjectToDestination)
        );
    }

    #[test]
    fn validates_metadata_and_exposes_lookup_helpers() {
        let metadata = new("bimodal_ke")
            .kind(ModelKind::Ode)
            .parameters(["ke", "v"])
            .covariates([Covariate::continuous("wt")])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate()
            .expect("metadata should validate");

        assert_eq!(metadata.parameter_index("ke"), Some(0));
        assert_eq!(metadata.parameter_index("v"), Some(1));
        assert_eq!(metadata.covariate_index("wt"), Some(0));
        assert_eq!(metadata.state_index("central"), Some(0));
        assert!(metadata.route("iv").is_some());
        assert_eq!(metadata.route_declaration_index("iv"), Some(0));
        assert_eq!(metadata.route_input_count(), 1);
        assert_eq!(metadata.output_index("cp"), Some(0));
        assert_eq!(
            metadata.route("iv").expect("route exists").destination(),
            "central"
        );
        assert_eq!(
            metadata
                .route("iv")
                .expect("route exists")
                .declaration_index(),
            0
        );
        assert_eq!(metadata.route("iv").expect("route exists").input_index(), 0);
        assert_eq!(
            metadata
                .route("iv")
                .expect("route exists")
                .destination_index(),
            0
        );
    }

    #[test]
    fn duplicate_names_fail_validation() {
        let error = new("dup_params")
            .kind(ModelKind::Ode)
            .parameters(["ke", "ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate()
            .expect_err("duplicate parameters must fail");

        assert_eq!(
            error,
            ModelMetadataError::DuplicateName {
                domain: NameDomain::Parameter,
                name: "ke".to_string(),
            }
        );
    }

    #[test]
    fn missing_route_destination_fails_validation() {
        let error = new("missing_route_destination")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv"))
            .validate()
            .expect_err("route destination is required");

        assert_eq!(
            error,
            ModelMetadataError::MissingRouteDestination {
                route: "iv".to_string(),
            }
        );
    }

    #[test]
    fn unknown_route_destination_fails_validation() {
        let error = new("unknown_route_destination")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("peripheral"))
            .validate()
            .expect_err("unknown destinations must fail");

        assert_eq!(
            error,
            ModelMetadataError::UnknownRouteDestination {
                route: "iv".to_string(),
                destination: "peripheral".to_string(),
            }
        );
    }

    #[test]
    fn shared_input_routes_preserve_declaration_and_input_identity() {
        let metadata = new("shared_input")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .routes([
                Route::bolus("oral").to_state("gut"),
                Route::infusion("iv").to_state("central"),
            ])
            .validate()
            .expect("shared-input metadata should validate");

        assert_eq!(metadata.routes().len(), 2);
        assert_eq!(metadata.route_input_count(), 1);
        assert_eq!(metadata.route_declaration_index("oral"), Some(0));
        assert_eq!(metadata.route_declaration_index("iv"), Some(1));
        assert_eq!(metadata.route("oral").expect("oral route").input_index(), 0);
        assert_eq!(metadata.route("iv").expect("iv route").input_index(), 0);
        assert_eq!(
            metadata
                .route("oral")
                .expect("oral route")
                .declaration_index(),
            0
        );
        assert_eq!(
            metadata.route("iv").expect("iv route").declaration_index(),
            1
        );
    }

    #[test]
    fn infusion_routes_reject_lag_and_bioavailability() {
        let lag_error = new("infusion_lag")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central").with_lag())
            .validate()
            .expect_err("infusion lag must fail");

        assert_eq!(
            lag_error,
            ModelMetadataError::InfusionLagNotAllowed {
                route: "iv".to_string(),
            }
        );

        let fa_error = new("infusion_fa")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                Route::infusion("iv")
                    .to_state("central")
                    .with_bioavailability(),
            )
            .validate()
            .expect_err("infusion bioavailability must fail");

        assert_eq!(
            fa_error,
            ModelMetadataError::InfusionBioavailabilityNotAllowed {
                route: "iv".to_string(),
            }
        );
    }

    #[test]
    fn validate_requires_or_accepts_a_kind() {
        let error = new("kind_required")
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate()
            .expect_err("kindless metadata needs explicit validation kind");

        assert_eq!(error, ModelMetadataError::MissingModelKind);

        let validated = new("kind_override")
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate_for(ModelKind::Ode)
            .expect("caller-provided kind should validate");

        assert_eq!(validated.kind(), ModelKind::Ode);
    }

    #[test]
    fn conflicting_kinds_fail_validation() {
        let error = new("kind_conflict")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate_for(ModelKind::Sde)
            .expect_err("conflicting kinds must fail");

        assert_eq!(
            error,
            ModelMetadataError::ModelKindConflict {
                declared: ModelKind::Ode,
                requested: ModelKind::Sde,
            }
        );
    }

    #[test]
    fn particles_are_rejected_for_ode_and_analytical() {
        let ode_error = new("ode_particles")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .particles(64)
            .validate()
            .expect_err("ODE metadata cannot declare particles");

        assert_eq!(
            ode_error,
            ModelMetadataError::ParticlesNotAllowed {
                kind: ModelKind::Ode,
            }
        );

        let analytical_error = new("analytical_particles")
            .kind(ModelKind::Analytical)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .particles(64)
            .validate()
            .expect_err("Analytical metadata cannot declare particles");

        assert_eq!(
            analytical_error,
            ModelMetadataError::ParticlesNotAllowed {
                kind: ModelKind::Analytical,
            }
        );
    }

    #[test]
    fn analytical_kernel_is_limited_to_analytical_models() {
        let error = new("ode_kernel")
            .kind(ModelKind::Ode)
            .parameters(["ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .analytical_kernel(AnalyticalKernel::OneCompartment)
            .validate()
            .expect_err("ODE metadata cannot declare an analytical kernel");

        assert_eq!(
            error,
            ModelMetadataError::AnalyticalKernelNotAllowed {
                kind: ModelKind::Ode,
            }
        );
    }

    #[test]
    fn sde_requires_particles_or_a_fallback_count() {
        let error = new("sde_missing_particles")
            .kind(ModelKind::Sde)
            .parameters(["ke", "sigma"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate()
            .expect_err("SDE metadata requires particles");

        assert_eq!(error, ModelMetadataError::MissingParticles);

        let validated = new("sde_fallback_particles")
            .parameters(["ke", "sigma"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .validate_for_with_particles(ModelKind::Sde, 128)
            .expect("fallback particle count should satisfy SDE validation");

        assert_eq!(validated.kind(), ModelKind::Sde);
        assert_eq!(validated.particles(), Some(128));
    }

    #[test]
    fn conflicting_particle_counts_fail_validation() {
        let error = new("sde_particle_conflict")
            .parameters(["ke", "sigma"])
            .states(["central"])
            .outputs(["cp"])
            .route(Route::infusion("iv").to_state("central"))
            .particles(64)
            .validate_for_with_particles(ModelKind::Sde, 128)
            .expect_err("mismatched particle counts must fail");

        assert_eq!(
            error,
            ModelMetadataError::ParticleCountConflict {
                declared: 64,
                fallback: 128,
            }
        );
    }
}
