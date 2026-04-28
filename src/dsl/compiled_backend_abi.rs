use serde::{Deserialize, Serialize};

use super::model_info::NativeModelInfo;
use pharmsol_dsl::execution::{ExecutionModel, KernelRole};

#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const API_VERSION_SYMBOL: &str = "pharmsol_dsl_api_version";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const MODEL_INFO_JSON_PTR_SYMBOL: &str = "pharmsol_dsl_model_info_json_ptr";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const MODEL_INFO_JSON_LEN_SYMBOL: &str = "pharmsol_dsl_model_info_json_len";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const DERIVE_SYMBOL: &str = "pharmsol_dsl_kernel_derive";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const DYNAMICS_SYMBOL: &str = "pharmsol_dsl_kernel_dynamics";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const OUTPUTS_SYMBOL: &str = "pharmsol_dsl_kernel_outputs";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const INIT_SYMBOL: &str = "pharmsol_dsl_kernel_init";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const DRIFT_SYMBOL: &str = "pharmsol_dsl_kernel_drift";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const DIFFUSION_SYMBOL: &str = "pharmsol_dsl_kernel_diffusion";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const ROUTE_LAG_SYMBOL: &str = "pharmsol_dsl_kernel_route_lag";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const ROUTE_BIOAVAILABILITY_SYMBOL: &str = "pharmsol_dsl_kernel_route_bioavailability";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const ALLOC_F64_BUFFER_SYMBOL: &str = "pharmsol_dsl_alloc_f64_buffer";
#[cfg(any(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm",
    feature = "dsl-wasm-compile"
))]
pub const FREE_F64_BUFFER_SYMBOL: &str = "pharmsol_dsl_free_f64_buffer";

#[cfg(any(test, feature = "dsl-wasm-compile"))]
pub const JS_KERNEL_EXPORTS: [(&str, &str); 8] = [
    ("derive", DERIVE_SYMBOL),
    ("dynamics", DYNAMICS_SYMBOL),
    ("outputs", OUTPUTS_SYMBOL),
    ("init", INIT_SYMBOL),
    ("drift", DRIFT_SYMBOL),
    ("diffusion", DIFFUSION_SYMBOL),
    ("route_lag", ROUTE_LAG_SYMBOL),
    ("route_bioavailability", ROUTE_BIOAVAILABILITY_SYMBOL),
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledKernelAvailability {
    pub derive: bool,
    pub dynamics: bool,
    pub outputs: bool,
    pub init: bool,
    pub drift: bool,
    pub diffusion: bool,
    pub route_lag: bool,
    pub route_bioavailability: bool,
}

impl CompiledKernelAvailability {
    pub fn from_execution_model(model: &ExecutionModel) -> Self {
        let mut availability = Self::default();
        for kernel in &model.kernels {
            match kernel.role {
                KernelRole::Derive => availability.derive = true,
                KernelRole::Dynamics => availability.dynamics = true,
                KernelRole::Outputs => availability.outputs = true,
                KernelRole::Init => availability.init = true,
                KernelRole::Drift => availability.drift = true,
                KernelRole::Diffusion => availability.diffusion = true,
                KernelRole::RouteLag => availability.route_lag = true,
                KernelRole::RouteBioavailability => availability.route_bioavailability = true,
                KernelRole::Analytical => {}
            }
        }
        availability
    }

    pub fn has(self, role: KernelRole) -> bool {
        match role {
            KernelRole::Derive => self.derive,
            KernelRole::Dynamics => self.dynamics,
            KernelRole::Outputs => self.outputs,
            KernelRole::Init => self.init,
            KernelRole::Drift => self.drift,
            KernelRole::Diffusion => self.diffusion,
            KernelRole::RouteLag => self.route_lag,
            KernelRole::RouteBioavailability => self.route_bioavailability,
            KernelRole::Analytical => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledModelInfoEnvelope {
    pub abi_version: u32,
    pub model: NativeModelInfo,
    pub kernels: CompiledKernelAvailability,
}

#[cfg(any(test, feature = "dsl-aot", feature = "dsl-wasm-compile"))]
pub fn compiled_model_info_envelope(
    model: &ExecutionModel,
    abi_version: u32,
) -> CompiledModelInfoEnvelope {
    CompiledModelInfoEnvelope {
        abi_version,
        model: NativeModelInfo::from_execution_model(model),
        kernels: CompiledKernelAvailability::from_execution_model(model),
    }
}

#[cfg(any(test, feature = "dsl-aot", feature = "dsl-wasm-compile"))]
pub fn encode_compiled_model_info(
    model: &ExecutionModel,
    abi_version: u32,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&compiled_model_info_envelope(model, abi_version))
}

#[cfg(any(
    test,
    feature = "dsl-aot-load",
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
pub fn decode_compiled_model_info(
    bytes: &[u8],
) -> Result<CompiledModelInfoEnvelope, serde_json::Error> {
    serde_json::from_slice(bytes)
}

#[cfg(any(test, feature = "dsl-aot", feature = "dsl-wasm-compile"))]
pub fn compiled_kernel_symbol(role: KernelRole) -> Option<&'static str> {
    match role {
        KernelRole::Derive => Some(DERIVE_SYMBOL),
        KernelRole::Dynamics => Some(DYNAMICS_SYMBOL),
        KernelRole::Outputs => Some(OUTPUTS_SYMBOL),
        KernelRole::Init => Some(INIT_SYMBOL),
        KernelRole::Drift => Some(DRIFT_SYMBOL),
        KernelRole::Diffusion => Some(DIFFUSION_SYMBOL),
        KernelRole::RouteLag => Some(ROUTE_LAG_SYMBOL),
        KernelRole::RouteBioavailability => Some(ROUTE_BIOAVAILABILITY_SYMBOL),
        KernelRole::Analytical => None,
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputBufferBinding {
    States,
    Derived,
    Scratch,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputBufferPlan {
    pub binding: OutputBufferBinding,
    pub len: usize,
    pub zero_before_call: bool,
}

#[cfg(test)]
pub fn output_buffer_plan(
    info: &NativeModelInfo,
    role: KernelRole,
    aliases_states: bool,
    aliases_derived: bool,
) -> OutputBufferPlan {
    let binding = if aliases_states {
        OutputBufferBinding::States
    } else if aliases_derived {
        OutputBufferBinding::Derived
    } else {
        OutputBufferBinding::Scratch
    };

    OutputBufferPlan {
        binding,
        len: kernel_output_len(info, role),
        zero_before_call: matches!(binding, OutputBufferBinding::Scratch),
    }
}

#[cfg(test)]
fn kernel_output_len(info: &NativeModelInfo, role: KernelRole) -> usize {
    match role {
        KernelRole::Derive => info.derived_len,
        KernelRole::Dynamics | KernelRole::Init | KernelRole::Drift | KernelRole::Diffusion => {
            info.state_len
        }
        KernelRole::Outputs => info.output_len,
        KernelRole::RouteLag | KernelRole::RouteBioavailability => info.route_len,
        KernelRole::Analytical => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_info::{NativeCovariateInfo, NativeOutputInfo, NativeRouteInfo};
    use super::*;
    use pharmsol_dsl::ModelKind;

    #[test]
    fn compiled_backend_symbol_names_are_frozen() {
        assert_eq!(API_VERSION_SYMBOL, "pharmsol_dsl_api_version");
        assert_eq!(
            MODEL_INFO_JSON_PTR_SYMBOL,
            "pharmsol_dsl_model_info_json_ptr"
        );
        assert_eq!(
            MODEL_INFO_JSON_LEN_SYMBOL,
            "pharmsol_dsl_model_info_json_len"
        );
        assert_eq!(DERIVE_SYMBOL, "pharmsol_dsl_kernel_derive");
        assert_eq!(DYNAMICS_SYMBOL, "pharmsol_dsl_kernel_dynamics");
        assert_eq!(OUTPUTS_SYMBOL, "pharmsol_dsl_kernel_outputs");
        assert_eq!(INIT_SYMBOL, "pharmsol_dsl_kernel_init");
        assert_eq!(DRIFT_SYMBOL, "pharmsol_dsl_kernel_drift");
        assert_eq!(DIFFUSION_SYMBOL, "pharmsol_dsl_kernel_diffusion");
        assert_eq!(ROUTE_LAG_SYMBOL, "pharmsol_dsl_kernel_route_lag");
        assert_eq!(
            ROUTE_BIOAVAILABILITY_SYMBOL,
            "pharmsol_dsl_kernel_route_bioavailability"
        );
        assert_eq!(ALLOC_F64_BUFFER_SYMBOL, "pharmsol_dsl_alloc_f64_buffer");
        assert_eq!(FREE_F64_BUFFER_SYMBOL, "pharmsol_dsl_free_f64_buffer");
        assert_eq!(JS_KERNEL_EXPORTS[0], ("derive", DERIVE_SYMBOL));
        assert_eq!(
            JS_KERNEL_EXPORTS[7],
            ("route_bioavailability", ROUTE_BIOAVAILABILITY_SYMBOL)
        );
    }

    #[test]
    fn compiled_model_info_round_trips_kernel_availability_and_dimensions() {
        let envelope = CompiledModelInfoEnvelope {
            abi_version: 7,
            model: NativeModelInfo {
                name: "example".to_string(),
                kind: ModelKind::Ode,
                parameters: vec!["ke".to_string(), "v".to_string()],
                covariates: vec![NativeCovariateInfo {
                    name: "wt".to_string(),
                    index: 0,
                }],
                routes: vec![NativeRouteInfo {
                    name: "iv".to_string(),
                    index: 0,
                    destination_offset: 1,
                    inject_input_to_destination: true,
                }],
                outputs: vec![NativeOutputInfo {
                    name: "cp".to_string(),
                    index: 0,
                }],
                state_len: 2,
                derived_len: 3,
                output_len: 1,
                route_len: 1,
                analytical: None,
                particles: Some(32),
            },
            kernels: CompiledKernelAvailability {
                derive: true,
                dynamics: true,
                outputs: true,
                init: true,
                drift: false,
                diffusion: false,
                route_lag: true,
                route_bioavailability: false,
            },
        };

        let json = serde_json::to_vec(&envelope).expect("serialize envelope");
        let decoded = decode_compiled_model_info(&json).expect("decode envelope");
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn output_buffer_plan_tracks_aliasing_and_zeroing_rules() {
        let info = NativeModelInfo {
            name: "example".to_string(),
            kind: ModelKind::Ode,
            parameters: vec![],
            covariates: vec![],
            routes: vec![],
            outputs: vec![],
            state_len: 2,
            derived_len: 3,
            output_len: 4,
            route_len: 1,
            analytical: None,
            particles: None,
        };

        let scratch = output_buffer_plan(&info, KernelRole::Diffusion, false, false);
        assert_eq!(scratch.binding, OutputBufferBinding::Scratch);
        assert_eq!(scratch.len, 2);
        assert!(scratch.zero_before_call);

        let states = output_buffer_plan(&info, KernelRole::Dynamics, true, false);
        assert_eq!(states.binding, OutputBufferBinding::States);
        assert_eq!(states.len, 2);
        assert!(!states.zero_before_call);

        let derived = output_buffer_plan(&info, KernelRole::Derive, false, true);
        assert_eq!(derived.binding, OutputBufferBinding::Derived);
        assert_eq!(derived.len, 3);
        assert!(!derived.zero_before_call);
    }
}
