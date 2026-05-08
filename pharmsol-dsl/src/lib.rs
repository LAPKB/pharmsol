//! Backend-neutral frontend crate for the pharmsol DSL.
//!
//! Use this crate when you need the DSL frontend as an engineering API: parse
//! model source, inspect diagnostics, analyze names and types, and lower a
//! validated model into the execution representation that backends consume.
//!
//! Do not use this crate when you already know you want JIT compilation,
//! native AoT artifacts, WASM artifacts, or `Subject`-based prediction
//! helpers. Those runtime-facing workflows stay in the main `pharmsol` crate
//! under `pharmsol::dsl`.
//!
//! Main entrypoints:
//!
//! - [`parse_model`] and [`parse_module`] for turning DSL source text into the
//!   syntax tree in [`ast`].
//! - [`analyze_model`] and [`analyze_module`] for semantic validation and the
//!   typed IR in [`ir`].
//! - [`lower_typed_model`] and [`lower_typed_module`] for lowering typed models
//!   into the execution representation in [`execution`].
//!
//! The frontend pipeline is intentionally simple:
//!
//! 1. Parse source text into syntax.
//! 2. Analyze the syntax into a typed model.
//! 3. Lower the typed model into an [`ExecutionModel`] or [`ExecutionModule`].
//!
//! This crate accepts both canonical `model { ... }` source and the authoring
//! shorthand used by the `pharmsol` examples. The returned diagnostics carry
//! source spans, rendered messages, and structured data for editor or UI use.
//!
//! Main modules:
//!
//! - [`ast`] for syntax-level nodes.
//! - [`diagnostic`] for spans, diagnostic codes, and rendered reports.
//! - [`ir`] for the typed intermediate representation.
//! - [`execution`] for the lowered execution model shared by JIT, AoT, and
//!   WASM backends.
//!
//! Smallest parse-analyze-lower example:
//!
//! ```rust
//! use pharmsol_dsl::{analyze_model, lower_typed_model, parse_model};
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
//! let syntax = parse_model(source).expect("model parses");
//! let typed = analyze_model(&syntax).expect("model analyzes");
//! let execution = lower_typed_model(&typed).expect("model lowers");
//!
//! assert_eq!(execution.name, "bimodal_ke");
//! assert_eq!(execution.metadata.routes.len(), 1);
//! assert_eq!(execution.metadata.outputs.len(), 1);
//! ```
//!
//! If you are building an authoring tool, custom compiler, or diagnostics UI,
//! stay in this crate. If you want a complete source-to-runtime workflow,
//! switch to `pharmsol::dsl` in the main crate.

pub mod ast;
mod authoring;
pub mod diagnostic;
pub mod execution;
pub mod ir;
mod lexer;
mod parser;
mod semantic;
#[cfg(test)]
mod test_fixtures;

pub use ast::*;
pub use diagnostic::*;
pub use execution::{
    lower_typed_model, lower_typed_module, ExecutionModel, ExecutionModule, LoweringError,
};
pub use ir::*;
pub use parser::{parse_model, parse_module};
pub use semantic::{analyze_model, analyze_module, SemanticError};
