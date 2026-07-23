//! One-shot source-to-execution pipeline and its unified error type.
//!
//! [`compile_model`] and [`compile_module`] run the full pipeline
//! (parse, analyze, compile) in a single call. Failures are reported as
//! [`DslError`], which unifies the phase-specific error types returned by the
//! individual stages behind one `std::error::Error` implementation.

use std::fmt;
use std::sync::Arc;

use crate::analyze::{analyze_model, analyze_module, AnalysisError};
use crate::diagnostic::{Diagnostic, DiagnosticPhase, DiagnosticReport, ParseError};
use crate::execution::{
    compile_analyzed_model, compile_analyzed_module, CompileError, ExecutionModel, ExecutionModule,
};
use crate::parser::{parse_model, parse_module};

/// Error produced by any stage of the DSL pipeline.
///
/// Use [`DslError::phase`] to learn which stage failed and
/// [`DslError::diagnostics`] for the structured diagnostics. When source text
/// is attached — always the case for errors returned by [`compile_model`] and
/// [`compile_module`] — printing the error renders the annotated report:
///
/// ```rust
/// use pharmsol_dsl::compile_model;
///
/// let error = compile_model("model broken { kind ode").unwrap_err();
/// assert!(error.to_string().contains("error[DSL1000]"));
/// assert_eq!(error.phase(), pharmsol_dsl::DiagnosticPhase::Parse);
/// ```
#[derive(Clone, thiserror::Error)]
pub enum DslError {
    /// Parsing the source text failed.
    #[error(transparent)]
    Parse(#[from] ParseError),
    /// Analyzing the model failed.
    #[error(transparent)]
    Analysis(#[from] AnalysisError),
    /// Compiling to the execution model failed.
    #[error(transparent)]
    Compile(#[from] CompileError),
}

impl DslError {
    /// The pipeline phase that produced this error.
    pub fn phase(&self) -> DiagnosticPhase {
        match self {
            Self::Parse(_) => DiagnosticPhase::Parse,
            Self::Analysis(_) => DiagnosticPhase::Analysis,
            Self::Compile(_) => DiagnosticPhase::Compile,
        }
    }

    /// All diagnostics carried by this error, in source order where known.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match self {
            Self::Parse(error) => error.diagnostics(),
            Self::Analysis(error) => std::slice::from_ref(error.diagnostic()),
            Self::Compile(error) => std::slice::from_ref(error.diagnostic()),
        }
    }

    /// Attaches source text so `Display` renders the annotated report.
    pub fn with_source(self, source: impl Into<Arc<str>>) -> Self {
        let source = source.into();
        match self {
            Self::Parse(error) => Self::Parse(error.with_source(source)),
            Self::Analysis(error) => Self::Analysis(error.with_source(source)),
            Self::Compile(error) => Self::Compile(error.with_source(source)),
        }
    }

    /// Source text attached to this error, if any.
    pub fn source(&self) -> Option<&str> {
        match self {
            Self::Parse(error) => error.source(),
            Self::Analysis(error) => error.source(),
            Self::Compile(error) => error.source(),
        }
    }

    /// Renders the diagnostics against `src` without attaching it.
    pub fn render(&self, src: &str) -> String {
        match self {
            Self::Parse(error) => error.render(src),
            Self::Analysis(error) => error.render(src),
            Self::Compile(error) => error.render(src),
        }
    }

    /// Structured, JSON-serializable report for editors and tooling.
    pub fn diagnostic_report(&self, source_name: impl Into<String>) -> DiagnosticReport {
        let source_name = source_name.into();
        match self {
            Self::Parse(error) => error.diagnostic_report(source_name),
            Self::Analysis(error) => error.diagnostic_report(source_name),
            Self::Compile(error) => error.diagnostic_report(source_name),
        }
    }
}

impl fmt::Debug for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Parses, analyzes, and compiles the module in `src` in one call.
///
/// This is the one-shot equivalent of [`parse_module`], [`analyze_module`],
/// and [`compile_analyzed_module`] in sequence. Errors carry the source text, so
/// printing them renders the annotated report.
pub fn compile_module(src: &str) -> Result<ExecutionModule, DslError> {
    let parsed = parse_module(src)?;
    let analyzed = analyze_module(&parsed).map_err(|error| error.with_source(src))?;
    compile_analyzed_module(&analyzed).map_err(|error| error.with_source(src).into())
}

/// Parses, analyzes, and compiles the single model in `src` in one call.
///
/// Like [`compile_module`], but requires the source to contain exactly one
/// model, mirroring [`parse_model`].
pub fn compile_model(src: &str) -> Result<ExecutionModel, DslError> {
    let parsed = parse_model(src)?;
    let analyzed = analyze_model(&parsed).map_err(|error| error.with_source(src))?;
    compile_analyzed_model(&analyzed).map_err(|error| error.with_source(src).into())
}
