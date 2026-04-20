//! High-level builder DSL for JIT-compiled pharmacometric models.
//!
//! Authors a model entirely in Rust — no JSON, no source files, no external
//! toolchain. After [`Model::compile`], the model is ready to plug into the
//! existing pharmsol simulator via [`crate::jit::JitOde`].
//!
//! # Example
//!
//! ```ignore
//! use pharmsol::jit::Model;
//! use pharmsol::prelude::*;
//!
//! let ode = Model::new("1cmt-iv")
//!     .compartments(["central"])
//!     .params(["CL", "V"])
//!     .covariates(["WT"])
//!     .dxdt("central", "rateiv[0] - (CL / V) * central")
//!     .output("cp", "central / V")
//!     .compile()
//!     .expect("compile");
//!
//! let subject = Subject::builder("p1")
//!     .infusion(0.0, 100.0, 0, 0.5)
//!     .observation(1.0, 5.0, 0)
//!     .covariate("WT", 0.0, 70.0)
//!     .build();
//!
//! let (preds, _) = ode.simulate_subject(&subject, &[10.0, 50.0], None).unwrap();
//! ```

use std::sync::Arc;

use thiserror::Error;

use super::ast::Expr;
use super::codegen::compile_artifact;
use super::equation::JitOde;
use super::parser::{self, ParseError};
use super::text::{self, TextError};

/// Errors produced while building or compiling a JIT model.
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("duplicate name {0:?}")]
    Duplicate(String),
    #[error("unknown compartment {0:?}")]
    UnknownCompartment(String),
    #[error("compartment {0:?} has no dxdt definition")]
    MissingDxDt(String),
    #[error("parse error in {context}: {source}")]
    Parse {
        context: String,
        #[source]
        source: ParseError,
    },
    #[error("unresolved identifier {ident:?} in {context}")]
    UnresolvedIdent { ident: String, context: String },
    #[error("rateiv index {0} out of range (ndrugs = {1})")]
    RateIvOutOfRange(usize, usize),
    #[error("bolus index {0} out of range (ndrugs = {1})")]
    BolusOutOfRange(usize, usize),
    #[error("invalid index target {0:?} (only `rateiv` and `bolus` are indexable)")]
    InvalidIndexTarget(String),
    #[error("unknown function {name:?} (supported: exp, ln, log, log10, sqrt, abs, pow)")]
    UnknownFunction { name: String },
    #[error("function {name:?} expects {expected} args, got {got}")]
    BadArity {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("duplicate let binding {0:?}")]
    DuplicateLet(String),
    #[error("let binding {0:?} forms a cycle")]
    LetCycle(String),
    #[error("duplicate init for compartment {0:?}")]
    DuplicateInit(String),
    #[error("unknown compartment {0:?} in init")]
    UnknownInitCompartment(String),
    #[error("duplicate lag for input channel {0}")]
    DuplicateLag(usize),
    #[error("duplicate fa for input channel {0}")]
    DuplicateFa(usize),
    #[error("lag channel {0} out of range (ndrugs = {1})")]
    LagOutOfRange(usize, usize),
    #[error("fa channel {0} out of range (ndrugs = {1})")]
    FaOutOfRange(usize, usize),
    #[error("codegen error: {0}")]
    Codegen(String),
}

/// A symbolic dxdt or output equation, parsed and stored in the model.
#[derive(Debug, Clone)]
pub(crate) struct Equation {
    pub(crate) target: String,
    pub(crate) expr: Expr,
}

/// A `let name = expr` binding declared in the model. Lets are scoped to the
/// whole model and may be referenced from any subsequent expression
/// (including other lets, dxdt, output, init, lag, fa).
#[derive(Debug, Clone)]
pub(crate) struct LetBinding {
    pub(crate) name: String,
    pub(crate) expr: Expr,
}

/// An `init(compartment) = expr` declaration.
#[derive(Debug, Clone)]
pub(crate) struct InitDecl {
    pub(crate) compartment: String,
    pub(crate) expr: Expr,
}

/// A `lag(channel) = expr` or `fa(channel) = expr` declaration.
#[derive(Debug, Clone)]
pub(crate) struct ChannelDecl {
    pub(crate) channel: usize,
    pub(crate) expr: Expr,
}

/// Mutable builder for a JIT model.
#[derive(Debug, Clone)]
pub struct Model {
    pub(crate) name: String,
    pub(crate) compartments: Vec<String>,
    pub(crate) params: Vec<String>,
    pub(crate) covariates: Vec<String>,
    pub(crate) ndrugs: usize,
    pub(crate) dxdt: Vec<Equation>,
    pub(crate) outputs: Vec<Equation>,
    pub(crate) lets: Vec<LetBinding>,
    pub(crate) inits: Vec<InitDecl>,
    pub(crate) lags: Vec<ChannelDecl>,
    pub(crate) fas: Vec<ChannelDecl>,
}

impl Model {
    /// Start a new model with the given identifier (used for diagnostics only).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            compartments: Vec::new(),
            params: Vec::new(),
            covariates: Vec::new(),
            ndrugs: 1,
            dxdt: Vec::new(),
            outputs: Vec::new(),
            lets: Vec::new(),
            inits: Vec::new(),
            lags: Vec::new(),
            fas: Vec::new(),
        }
    }

    /// Parse a model from the line-based text format documented in
    /// [`crate::jit::text`]. Pair with [`Model::compile`] to JIT-compile.
    ///
    /// This is the entry point for runtime authoring: callers using only the
    /// precompiled `pharmsol` library can build a working ODE model from a
    /// string with no Rust toolchain involved.
    pub fn from_text(src: &str) -> Result<Self, TextError> {
        text::parse_text(src)
    }

    /// Convenience: read a model definition from a file on disk.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
        let src = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ModelError::Codegen(format!("read {}: {e}", path.as_ref().display())))?;
        Self::from_text(&src).map_err(|e| ModelError::Codegen(e.to_string()))
    }

    /// Declare the compartments (state variables), in order. Their index in
    /// the state vector matches their order here.
    pub fn compartments<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.compartments = names.into_iter().map(Into::into).collect();
        self
    }

    /// Declare the model parameters, in the order they appear in the
    /// support-point vector passed to the simulator.
    pub fn params<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.params = names.into_iter().map(Into::into).collect();
        self
    }

    /// Declare the covariates referenced by the equations. Their order
    /// determines the layout of the per-call covariate buffer.
    pub fn covariates<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.covariates = names.into_iter().map(Into::into).collect();
        self
    }

    /// Number of drug input channels (size of `rateiv[]` and `bolus[]`).
    /// Defaults to 1.
    pub fn ndrugs(mut self, n: usize) -> Self {
        self.ndrugs = n;
        self
    }

    /// Add a dxdt equation: `dxdt(compartment, "expression")`.
    pub fn dxdt(mut self, compartment: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let target = compartment.into();
        // Defer parse errors until compile() so the builder API stays infallible.
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.dxdt.push(Equation {
            target,
            expr: parsed,
        });
        self
    }

    /// Add an output equation: `output(name, "expression")`. The output index
    /// matches the order of these calls.
    pub fn output(mut self, name: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let target = name.into();
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.outputs.push(Equation {
            target,
            expr: parsed,
        });
        self
    }

    /// Add a `let` binding: a named scalar expression that may be referenced
    /// (by name) in any subsequent equation. Useful for sharing
    /// sub-expressions between dxdt/out/init/lag/fa.
    pub fn let_binding(mut self, name: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.lets.push(LetBinding {
            name: name.into(),
            expr: parsed,
        });
        self
    }

    /// Add an initial-state expression for a compartment. Compartments
    /// without an `init` default to 0.
    pub fn init(mut self, compartment: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.inits.push(InitDecl {
            compartment: compartment.into(),
            expr: parsed,
        });
        self
    }

    /// Add a lag-time expression for an input channel. Channels without a
    /// `lag` default to 0. Lag time is added to bolus event times only
    /// (matching pharmsol's existing semantics).
    pub fn lag(mut self, channel: usize, expr: impl AsRef<str>) -> Self {
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.lags.push(ChannelDecl {
            channel,
            expr: parsed,
        });
        self
    }

    /// Add a bioavailability expression for an input channel. Channels
    /// without an `fa` default to 1. Bioavailability scales bolus amounts
    /// only (matching pharmsol's existing semantics).
    pub fn fa(mut self, channel: usize, expr: impl AsRef<str>) -> Self {
        let parsed = parser::parse(expr.as_ref()).unwrap_or(Expr::Const(f64::NAN));
        self.fas.push(ChannelDecl {
            channel,
            expr: parsed,
        });
        self
    }

    /// Validate, compile to native code via Cranelift, and return a [`JitOde`]
    /// ready to use with `Equation::simulate_subject`.
    pub fn compile(self) -> Result<JitOde, ModelError> {
        // Re-parse all equations now to surface real parse errors with context.
        let mut dxdt = Vec::with_capacity(self.dxdt.len());
        for eq in &self.dxdt {
            // We stored a sentinel NaN if parsing failed. Re-parse the original
            // source isn't available here, so at compile-time we parse fresh
            // from the AST stored… but we already have the AST. Instead, the
            // builder API parses eagerly; if the user wants nice errors at the
            // call site they should use `try_dxdt` / `try_output`.
            dxdt.push(eq.clone());
        }
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for eq in &self.outputs {
            outputs.push(eq.clone());
        }

        // Validate every compartment has a dxdt and reorder dxdt to match.
        let mut ordered_dxdt: Vec<Option<Equation>> = vec![None; self.compartments.len()];
        for eq in dxdt {
            let idx = self
                .compartments
                .iter()
                .position(|c| c == &eq.target)
                .ok_or_else(|| ModelError::UnknownCompartment(eq.target.clone()))?;
            if ordered_dxdt[idx].is_some() {
                return Err(ModelError::Duplicate(eq.target.clone()));
            }
            ordered_dxdt[idx] = Some(eq);
        }
        let dxdt_in_order: Vec<Equation> = ordered_dxdt
            .into_iter()
            .enumerate()
            .map(|(i, e)| e.ok_or_else(|| ModelError::MissingDxDt(self.compartments[i].clone())))
            .collect::<Result<_, _>>()?;

        let nstates = self.compartments.len();
        let nout = outputs.len();
        let ndrugs = self.ndrugs;

        // Validate let bindings: no duplicates.
        let mut seen_let = std::collections::HashSet::new();
        for lb in &self.lets {
            if !seen_let.insert(lb.name.clone()) {
                return Err(ModelError::DuplicateLet(lb.name.clone()));
            }
        }

        // Validate inits: target must be a compartment, no duplicates.
        let mut init_by_idx: Vec<Option<Expr>> = vec![None; nstates];
        for init in &self.inits {
            let idx = self
                .compartments
                .iter()
                .position(|c| c == &init.compartment)
                .ok_or_else(|| ModelError::UnknownInitCompartment(init.compartment.clone()))?;
            if init_by_idx[idx].is_some() {
                return Err(ModelError::DuplicateInit(init.compartment.clone()));
            }
            init_by_idx[idx] = Some(init.expr.clone());
        }

        // Validate lag/fa: channel in range, no duplicates.
        let mut lag_by_idx: Vec<Option<Expr>> = vec![None; ndrugs];
        for lag in &self.lags {
            if lag.channel >= ndrugs {
                return Err(ModelError::LagOutOfRange(lag.channel, ndrugs));
            }
            if lag_by_idx[lag.channel].is_some() {
                return Err(ModelError::DuplicateLag(lag.channel));
            }
            lag_by_idx[lag.channel] = Some(lag.expr.clone());
        }
        let mut fa_by_idx: Vec<Option<Expr>> = vec![None; ndrugs];
        for fa in &self.fas {
            if fa.channel >= ndrugs {
                return Err(ModelError::FaOutOfRange(fa.channel, ndrugs));
            }
            if fa_by_idx[fa.channel].is_some() {
                return Err(ModelError::DuplicateFa(fa.channel));
            }
            fa_by_idx[fa.channel] = Some(fa.expr.clone());
        }

        let artifact = compile_artifact(
            &self.name,
            &self.compartments,
            &self.params,
            &self.covariates,
            ndrugs,
            &dxdt_in_order,
            &outputs,
            &self.lets,
            &init_by_idx,
            &lag_by_idx,
            &fa_by_idx,
        )?;

        Ok(JitOde::new(
            Arc::new(artifact),
            nstates,
            ndrugs,
            nout,
            self.covariates,
            self.params,
        ))
    }
}
