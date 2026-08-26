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
//!
//! Common workflow choices:
//!
//! - Compiler only: parse, analyze, and compile to the execution model when
//!   you need diagnostics, authoring tools, or your own backend.
//! - In-process execution: compile straight to [`CompiledRuntimeModel`] and
//!   keep everything inside the current process.
//!
//! Feature map:
//!
//! - `dsl`: enables this facade, the compiler re-exports from `pharmsol-dsl`,
//!   and in-process JIT compilation through
//!   [`compile_module_source_to_runtime`], plus the lower-level JIT compile
//!   entrypoints.
//!
//! Smallest compile-to-runtime example:
//!
//! ```rust,no_run
//! use pharmsol::dsl::compile_module_source_to_runtime;
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

mod backend;
mod jit;
mod model_info;
mod runtime;

pub use backend::{
    CompiledModelFunction, RuntimeAnalyticalModel, RuntimeExecutionArtifact, RuntimeOdeModel,
    RuntimeSdeModel,
};
pub use jit::{
    compile_analytical_model_to_jit, compile_execution_artifact, compile_execution_model_to_jit,
    compile_ode_model_to_jit, compile_sde_model_to_jit, JitCompileError,
};
pub use model_info::{
    RuntimeCovariateInfo, RuntimeModelInfo, RuntimeOutputInfo, RuntimeRouteInfo, RuntimeStateInfo,
};
pub use pharmsol_dsl::*;
pub use runtime::{
    compile_execution_model_to_runtime, compile_module_source_to_runtime, CompiledRuntimeModel,
    RuntimeError, RuntimePredictions,
};
