//! Raw functions produced by the JIT compiler.

use std::panic::{catch_unwind, AssertUnwindSafe};

use cranelift_jit::JITModule;
use pharmsol_dsl::execution::ModelFunctionKind;

use crate::PharmsolError;

/// Host callback ABI for the runtime-only two-site effect function.
pub type EstimateEffect2Callback = extern "C" fn(f64, f64, f64, f64, f64) -> f64;

/// Host callback ABI for the runtime-only three-site effect function.
pub type EstimateEffect3Callback =
    extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64;

/// ABI for JIT kernels.
pub type CompiledModelFunction = unsafe extern "C" fn(
    t: f64,
    states: *const f64,
    params: *const f64,
    covariates: *const f64,
    routes: *const f64,
    derived: *const f64,
    out: *mut f64,
    estimate_effect_2_callback: EstimateEffect2Callback,
    estimate_effect_3_callback: EstimateEffect3Callback,
);

/// The host callback never permits a Rust panic to cross the C ABI.
pub(crate) extern "C" fn estimate_effect_2_callback(
    u: f64,
    v: f64,
    alpha: f64,
    h1: f64,
    h2: f64,
) -> f64 {
    catch_unwind(AssertUnwindSafe(|| {
        crate::estimate_effect_2(u, v, alpha, h1, h2)
    }))
    .unwrap_or(f64::NAN)
}

#[allow(clippy::too_many_arguments)]
pub(crate) extern "C" fn estimate_effect_3_callback(
    a: f64,
    b: f64,
    c: f64,
    alpha12: f64,
    alpha13: f64,
    alpha23: f64,
    alpha123: f64,
    h1: f64,
    h2: f64,
    h3: f64,
) -> f64 {
    catch_unwind(AssertUnwindSafe(|| {
        crate::estimate_effect_3(a, b, c, alpha12, alpha13, alpha23, alpha123, h1, h2, h3)
    }))
    .unwrap_or(f64::NAN)
}

/// The compiled functions of one model.
///
/// Implemented by [`RuntimeExecutionArtifact`], and by stubs in tests.
pub(crate) trait RuntimeArtifact: Send + Sync + std::fmt::Debug {
    fn has_function(&self, role: ModelFunctionKind) -> bool;

    /// Call the compiled function for `role`.
    ///
    /// # Safety
    /// Every pointer must address a buffer of the length the compiled model
    /// expects for that argument.
    #[allow(clippy::too_many_arguments)]
    unsafe fn invoke_raw(
        &self,
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

enum RuntimeArtifactOwner {
    /// Held only to keep the JIT-allocated code alive for the artifact's lifetime.
    Jit(#[allow(dead_code)] Box<JITModule>),
}

impl std::fmt::Debug for RuntimeArtifactOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Jit(_) => f.write_str("RuntimeArtifactOwner::Jit(..)"),
        }
    }
}

pub struct RuntimeExecutionArtifact {
    pub model_name: String,
    pub derive: Option<CompiledModelFunction>,
    pub dynamics: Option<CompiledModelFunction>,
    pub outputs: CompiledModelFunction,
    pub init: Option<CompiledModelFunction>,
    pub drift: Option<CompiledModelFunction>,
    pub diffusion: Option<CompiledModelFunction>,
    pub route_lag: Option<CompiledModelFunction>,
    pub route_bioavailability: Option<CompiledModelFunction>,
    _owner: RuntimeArtifactOwner,
}

unsafe impl Send for RuntimeExecutionArtifact {}
unsafe impl Sync for RuntimeExecutionArtifact {}

impl std::fmt::Debug for RuntimeExecutionArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeExecutionArtifact")
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

impl RuntimeExecutionArtifact {
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
            _owner: RuntimeArtifactOwner::Jit(Box::new(module)),
        }
    }
}

impl RuntimeArtifact for RuntimeExecutionArtifact {
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

    unsafe fn invoke_raw(
        &self,
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
            ModelFunctionKind::Derive => self.derive,
            ModelFunctionKind::Dynamics => self.dynamics,
            ModelFunctionKind::Outputs => Some(self.outputs),
            ModelFunctionKind::Init => self.init,
            ModelFunctionKind::Drift => self.drift,
            ModelFunctionKind::Diffusion => self.diffusion,
            ModelFunctionKind::RouteLag => self.route_lag,
            ModelFunctionKind::RouteBioavailability => self.route_bioavailability,
            ModelFunctionKind::Analytical => None,
        }
        .ok_or_else(|| {
            PharmsolError::OtherError(format!(
                "model `{}` does not provide a {:?} function",
                self.model_name, role
            ))
        })?;

        let estimate_effect_2: EstimateEffect2Callback = estimate_effect_2_callback;
        let estimate_effect_3: EstimateEffect3Callback = estimate_effect_3_callback;
        unsafe {
            function(
                time,
                states,
                params,
                covariates,
                routes,
                derived,
                out,
                estimate_effect_2,
                estimate_effect_3,
            );
        }
        Ok(())
    }
}
