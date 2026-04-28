//! Backend-neutral frontend crate for the pharmsol DSL.
//!
//! This crate owns parsing, diagnostics, authoring desugaring, semantic
//! analysis, and execution lowering for DSL modules. `pharmsol::dsl`
//! re-exports the stable runtime-facing surface in the main crate.

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
