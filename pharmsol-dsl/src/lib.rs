//! Backend-neutral frontend crate for the pharmsol Proposal 2 DSL.
//!
//! Slice 2 moves the parsing frontend here on top of the shared frontend data
//! modules already extracted in Slice 1:
//!
//! - AST and model syntax types
//! - diagnostic and report types
//! - typed semantic IR
//! - lexical analysis
//! - canonical parse entrypoints
//! - authoring desugaring used by the parser
//! - semantic analysis and diagnostics
//!
//! Execution lowering now also lives here, while `pharmsol::dsl` continues to
//! re-export the stable runtime-facing surface during the migration.

mod authoring;
pub mod ast;
pub mod diagnostic;
pub mod execution;
pub mod ir;
mod lexer;
mod parser;
mod semantic;

pub use ast::*;
pub use diagnostic::*;
pub use execution::{ExecutionModel, ExecutionModule, LoweringError, lower_typed_model, lower_typed_module};
pub use ir::*;
pub use parser::{parse_model, parse_module};
pub use semantic::{analyze_model, analyze_module, SemanticError};
