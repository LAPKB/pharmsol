//! Backend-neutral frontend for the Proposal 2 model DSL.
//!
//! This module owns source parsing, syntax diagnostics, and the concrete AST
//! for structured-block models. It is intentionally independent from JIT,
//! ahead-of-time native export, and WASM emission so later slices can lower the
//! same parsed model into a shared semantic IR.

#[cfg(any(feature = "dsl-aot", feature = "dsl-aot-load"))]
mod aot;
mod ast;
mod authoring;
mod compiled_backend_abi;
mod diagnostic;
mod execution;
mod ir;
#[cfg(feature = "dsl-jit")]
mod jit;
mod lexer;
mod model_info;
#[cfg(any(feature = "dsl-jit", feature = "dsl-aot-load", feature = "dsl-wasm"))]
mod native;
mod parser;
#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]
mod runtime;
#[cfg(feature = "dsl-aot")]
mod rust_backend;
mod semantic;
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
pub use ast::*;
pub use compiled_backend_abi::{CompiledKernelAvailability, CompiledModelInfoEnvelope};
pub use diagnostic::{
    Applicability, Diagnostic, DiagnosticCode, DiagnosticLabel, DiagnosticLabelKind,
    DiagnosticPhase, DiagnosticReport, DiagnosticReportEdit, DiagnosticReportEntry,
    DiagnosticReportLabel, DiagnosticReportSource, DiagnosticReportSpan,
    DiagnosticReportSuggestion, DiagnosticSeverity, DiagnosticSuggestion, ParseError, Span,
    TextEdit, DSL_BACKEND_GENERIC, DSL_LOWERING_GENERIC, DSL_PARSE_GENERIC, DSL_SEMANTIC_GENERIC,
};
pub use execution::{
    lower_typed_model, lower_typed_module, ExecutionModel, ExecutionModule, LoweringError,
};
pub use ir::*;
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
pub use parser::{parse_model, parse_module};
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
pub use semantic::{analyze_model, analyze_module, SemanticError};
#[cfg(feature = "dsl-wasm")]
pub use wasm::{
    read_wasm_model_info, read_wasm_model_info_bytes,
};
#[cfg(feature = "dsl-wasm-compile")]
pub use wasm_compile::{
    browser_loader_source, compile_execution_model_to_wasm_bytes,
    compile_execution_model_to_wasm_module, compile_module_source_to_wasm_bytes,
    compile_module_source_to_wasm_module, CompiledWasmModule, WasmCompileCache, WasmError,
    DEFAULT_WASM_COMPILE_CACHE_CAPACITY, WASM_API_VERSION,
};
