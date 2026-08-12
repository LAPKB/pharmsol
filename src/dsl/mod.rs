//! Public DSL facade for pharmsol.
//!
//! Use this module when you want to work with pharmsol models as source text
//! and stay inside the main crate for the full workflow: parse DSL source,
//! inspect diagnostics, compile to the execution model and then to a runtime
//! backend, load saved artifacts, and run predictions.
//!
//! Use the `pharmsol-dsl` crate directly only when you need the source-to-execution
//! compiler as an engineering API. That crate owns parsing, diagnostics,
//! analysis, and compilation to the execution model. This module re-exports
//! that stable compiler surface and adds the backend-specific entrypoints
//! that stay owned by `pharmsol`.
//!
//! Main entrypoints:
//!
//! - [`parse_model`], [`parse_module`], [`analyze_model`], and
//!   [`analyze_module`] for source-level validation and inspection.
//! - [`compile_analyzed_model`] and [`compile_analyzed_module`] for compiling
//!   analyzed models into the ready-to-run form used by the runtime backends.
//! - [`compile_module_source_to_runtime`] and [`compile_execution_model_to_runtime`]
//!   for the one-stop compile-and-run path.
//! - [`load_runtime_artifact`] and [`load_aot_model`] for loading saved
//!   artifacts back into a model you can execute.
//!
//! Common workflow choices:
//!
//! - Compiler only: parse, analyze, and compile to the execution model when
//!   you need diagnostics, authoring tools, or your own backend.
//! - In-process execution: compile straight to [`RuntimeCompilationTarget`] and
//!   keep everything inside the current process.
//! - Native artifact shipping: export a native AoT artifact, then load it later
//!   on a compatible host.
//!
//! Feature map:
//!
//! - `dsl-core`: enables this facade and the compiler re-exports from
//!   `pharmsol-dsl`.
//! - `dsl-jit`: enables in-process JIT compilation through
//!   [`compile_module_source_to_runtime`] with
//!   [`RuntimeCompilationTarget::Jit`], plus the lower-level JIT compile
//!   entrypoints.
//! - `dsl-aot`: enables native ahead-of-time artifact export through
//!   [`compile_module_source_to_aot`] and [`export_execution_model_to_aot`].
//! - `dsl-aot-load`: enables native AoT artifact loading through
//!   [`load_aot_model`] and [`read_aot_model_info`].
//!
//! Smallest compile-to-runtime example:
//!
//! This example requires `dsl-jit`.
//!
//! ```rust,no_run
//! use pharmsol::dsl::{compile_module_source_to_runtime, RuntimeCompilationTarget};
//!
//! let source = r#"
//! name = bimodal_ke
//! kind = ode
//!
//! params = ke, v
//! states = central
//! outputs = cp
//!
//! infusion(iv) -> central
//!
//! dx(central) = -ke * central
//! out(cp) = central / v
//! "#;
//!
//! let model = compile_module_source_to_runtime(
//!     source,
//!     Some("bimodal_ke"),
//!     RuntimeCompilationTarget::Jit,
//!     |_, _| {},
//! )?;
//!
//! # let _ = model;
//! # Ok::<(), pharmsol::dsl::RuntimeError>(())
//! ```
//!
//! For just the source-to-execution compiler without backend selection, use
//! `pharmsol-dsl`. For a complete runtime path inside the main crate, stay in
//! [`pharmsol::dsl`](self).

#[cfg(any(feature = "dsl-aot", feature = "dsl-aot-load"))]
mod aot;
mod compiled_backend_abi;
#[cfg(feature = "dsl-jit")]
mod jit;
mod model_info;
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load"))]
mod native;
#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load")
))]
mod runtime;
#[cfg(feature = "dsl-aot")]
mod rust_backend;

#[cfg(feature = "dsl-aot")]
pub use aot::{
    compile_module_source_to_aot, export_execution_model_to_aot, AotError, NativeAotCompileOptions,
    NativeAotTarget, AOT_API_VERSION,
};
#[cfg(feature = "dsl-aot-load")]
pub use aot::{load_aot_model, read_aot_model_info};
#[cfg(all(not(feature = "dsl-aot"), feature = "dsl-aot-load"))]
pub use aot::{AotError, AOT_API_VERSION};
pub use compiled_backend_abi::{CompiledFunctionAvailability, CompiledModelInfoEnvelope};
#[cfg(feature = "dsl-jit")]
pub use jit::{
    compile_analytical_model_to_jit, compile_execution_artifact, compile_execution_model_to_jit,
    compile_ode_model_to_jit, compile_sde_model_to_jit, CompiledJitModel, JitAnalyticalModel,
    JitCompileError, JitExecutionArtifact, JitOdeModel, JitSdeModel,
};
pub use model_info::{NativeCovariateInfo, NativeModelInfo, NativeOutputInfo, NativeRouteInfo};
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load"))]
pub use native::{
    CompiledModelFunction, CompiledNativeModel, NativeAnalyticalModel, NativeExecutionArtifact,
    NativeOdeModel, NativeSdeModel, RuntimeBackend,
};
pub use pharmsol_dsl::*;
#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load")
))]
pub use runtime::{
    compile_execution_model_to_runtime, compile_module_source_to_runtime, load_runtime_artifact,
    CompiledRuntimeModel, RuntimeAnalyticalModel, RuntimeArtifactFormat, RuntimeCompilationTarget,
    RuntimeCovariateInfo, RuntimeError, RuntimeModelInfo, RuntimeOdeModel, RuntimeOutputInfo,
    RuntimePredictions, RuntimeRouteInfo, RuntimeSdeModel, RuntimeStateInfo,
};
