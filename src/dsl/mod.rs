//! Public DSL facade for pharmsol.
//!
//! The backend-neutral frontend is being extracted into `pharmsol-dsl`.
//! Frontend syntax, diagnostics, semantic analysis, and lowering now come
//! from `pharmsol-dsl`, while runtime and backend compilation entrypoints
//! remain owned by `pharmsol`.

#[cfg(any(feature = "dsl-aot", feature = "dsl-aot-load"))]
mod aot;
mod compiled_backend_abi;
#[cfg(feature = "dsl-jit")]
mod jit;
mod model_info;
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load", feature = "dsl-wasm"))]
mod native;
#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]
mod runtime;
#[cfg(feature = "dsl-aot")]
mod rust_backend;
#[cfg(feature = "dsl-wasm")]
mod wasm;
#[cfg(feature = "dsl-wasm-compile")]
mod wasm_compile;
#[cfg(feature = "dsl-wasm-compile")]
mod wasm_direct_emitter;

#[cfg(feature = "dsl-aot")]
pub use aot::{
    compile_module_source_to_aot, export_execution_model_to_aot, AotError, NativeAotCompileOptions,
    NativeAotTarget, AOT_API_VERSION,
};
#[cfg(feature = "dsl-aot-load")]
pub use aot::{load_aot_model, read_aot_model_info};
#[cfg(all(not(feature = "dsl-aot"), feature = "dsl-aot-load"))]
pub use aot::{AotError, AOT_API_VERSION};
pub use pharmsol_dsl::*;
pub use compiled_backend_abi::{CompiledKernelAvailability, CompiledModelInfoEnvelope};
#[cfg(feature = "dsl-jit")]
pub use jit::{
    compile_analytical_model_to_jit, compile_execution_artifact, compile_execution_model_to_jit,
    compile_ode_model_to_jit, compile_sde_model_to_jit, CompiledJitModel, JitAnalyticalModel,
    JitCompileError, JitExecutionArtifact, JitOdeModel, JitSdeModel,
};
pub use model_info::{NativeCovariateInfo, NativeModelInfo, NativeOutputInfo, NativeRouteInfo};
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load", feature = "dsl-wasm"))]
pub use native::{
    CompiledNativeModel, DenseKernelFn, NativeAnalyticalModel, NativeExecutionArtifact,
    NativeOdeModel, NativeSdeModel, RuntimeBackend,
};
#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]
pub use runtime::{
    compile_execution_model_to_runtime, compile_module_source_to_runtime, load_runtime_artifact,
    CompiledRuntimeModel, RuntimeAnalyticalModel, RuntimeArtifactFormat, RuntimeCompilationTarget,
    RuntimeCovariateInfo, RuntimeError, RuntimeModelInfo, RuntimeOdeModel, RuntimeOutputInfo,
    RuntimePredictions, RuntimeRouteInfo, RuntimeSdeModel,
};
#[cfg(feature = "dsl-wasm")]
pub use runtime::{
    compile_execution_model_to_runtime_wasm, compile_module_source_to_runtime_wasm,
    load_runtime_wasm_bytes,
};
#[cfg(feature = "dsl-wasm")]
pub use wasm::{read_wasm_model_info, read_wasm_model_info_bytes};
#[cfg(feature = "dsl-wasm-compile")]
pub use wasm_compile::{
    browser_loader_source, compile_execution_model_to_wasm_bytes,
    compile_execution_model_to_wasm_module, compile_module_source_to_wasm_bytes,
    compile_module_source_to_wasm_module, CompiledWasmModule, WasmCompileCache, WasmError,
    DEFAULT_WASM_COMPILE_CACHE_CAPACITY, WASM_API_VERSION,
};
