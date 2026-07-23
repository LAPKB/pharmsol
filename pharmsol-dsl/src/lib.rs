//! Compiles pharmsol DSL model source into a ready-to-run form.
//!
//! This crate reads model source text written in the pharmsol DSL and turns
//! it into an [`ExecutionModel`]: a fully checked, ready-to-run model that the
//! simulation backends in the main `pharmsol` crate execute directly.
//!
//! Use this crate when you need to work with model source as data:
//!
//! - parse DSL text into a syntax tree
//! - inspect spans and diagnostics
//! - check names, types, and model structure
//! - compile a validated model into its execution form
//!
//! Do not use this crate for JIT compilation, native ahead-of-time export or
//! load, WASM runtime loading, or `Subject`-based prediction helpers. Those
//! workflows stay in `pharmsol::dsl` in the main `pharmsol` crate.
//!
//! Main entrypoints:
//!
//! - [`compile_model`] and [`compile_module`] run the whole pipeline in one
//!   call, failing with the unified [`DslError`].
//! - [`parse_model`] and [`parse_module`] turn DSL source text into the
//!   syntax tree in [`syntax`].
//! - [`analyze_model`] and [`analyze_module`] check names, types, and model
//!   structure, producing the analyzed model in [`analysis`].
//! - [`compile_analyzed_model`] and [`compile_analyzed_module`] compile an
//!   analyzed model into the ready-to-run form in [`execution`].
//!
//! The pipeline is intentionally simple:
//!
//! 1. Parse source text into syntax.
//! 2. Analyze the syntax into a checked model.
//! 3. Compile the checked model into an [`ExecutionModel`] or
//!    [`ExecutionModule`].
//!
//! This crate accepts both canonical `model { ... }` source and the authoring
//! shorthand used by the `pharmsol` examples. The returned diagnostics carry
//! source spans, rendered messages, and structured data for editor or UI use.
//!
//! Main modules:
//!
//! - [`syntax`] for the syntax tree.
//! - [`analysis`] for the analyzed, fully checked model.
//! - [`diagnostic`] for spans, diagnostic codes, and rendered reports.
//! - [`execution`] for the ready-to-run model shared by the JIT, AoT, and
//!   WASM backends.
//!
//! Smallest one-shot example:
//!
//! ```rust
//! use pharmsol_dsl::compile_model;
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
//! let execution = compile_model(source).expect("model compiles");
//!
//! assert_eq!(execution.name, "bimodal_ke");
//! assert_eq!(execution.metadata.routes.len(), 1);
//! assert_eq!(execution.metadata.outputs.len(), 1);
//! ```
//!
//! The same pipeline staged, for callers that need the intermediate
//! representations:
//!
//! ```rust
//! use pharmsol_dsl::{analyze_model, compile_analyzed_model, parse_model};
//!
//! # let source = "name = m\nkind = ode\nstates = central\nddt(central) = 0\nout(cp) = central";
//! let syntax = parse_model(source).expect("source parses");
//! let analyzed = analyze_model(&syntax).expect("model is valid");
//! let execution = compile_analyzed_model(&analyzed).expect("model compiles");
//! # assert_eq!(execution.name, "m");
//! ```
//!
//! Errors from any stage carry source spans and render as annotated reports
//! when printed:
//!
//! ```rust
//! use pharmsol_dsl::compile_model;
//!
//! let error = compile_model("name = m\nstates = central\nout(cp) = missing_state").unwrap_err();
//! let report = error.to_string();
//! assert!(report.contains("error[DSL2000]"));
//! assert!(report.contains("unknown identifier `missing_state`"));
//! ```
//!
//! If you are building an authoring tool, custom compiler, or diagnostics UI,
//! stay in this crate. If you want a complete source-to-runtime workflow,
//! switch to `pharmsol::dsl` in the main crate.

pub mod analysis;
mod analyze;
mod authoring;
pub mod diagnostic;
pub mod execution;
mod lexer;
mod name_match;
mod parser;
mod pipeline;
pub mod syntax;
#[cfg(test)]
mod test_fixtures;

/// Canonical prefix for numeric route labels such as `input_1`.
pub const NUMERIC_ROUTE_PREFIX: &str = "input_";
/// Canonical prefix for numeric output labels such as `outeq_1`.
pub const NUMERIC_OUTPUT_PREFIX: &str = "outeq_";
/// Upper bound for compile-time-resolved sizes: state array sizes, route
/// destination indices, and particle counts. Keeps compiled models within a
/// comfortably allocatable state space.
pub const MAX_CONST_USIZE: usize = 1_048_576;
pub(crate) const RATE_FUNCTION_NAME: &str = "rate";

pub use analysis::*;
pub use analyze::{analyze_model, analyze_module, AnalysisError};
pub use diagnostic::*;
pub use execution::{
    compile_analyzed_model, compile_analyzed_module, CompileError, ExecutionModel, ExecutionModule,
};
pub use parser::{parse_model, parse_module, MAX_NESTING_DEPTH};
pub use pipeline::{compile_model, compile_module, DslError};
pub use syntax::*;
