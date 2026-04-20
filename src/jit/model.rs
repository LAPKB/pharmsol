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

use super::ast::Expr;
use super::codegen::compile_artifact;
use super::equation::JitOde;
use super::parser::{self, ParseError};
use super::text::{self, TextError};

/// All function names recognised by the JIT (used for did-you-mean hints).
pub(crate) const KNOWN_FUNCTIONS: &[&str] =
    &["exp", "ln", "log", "log10", "log2", "sqrt", "abs", "pow"];

/// Suggest the closest match from `candidates` to `target` (Levenshtein <= 2).
pub(crate) fn did_you_mean<'a, I, S>(target: &str, candidates: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str> + 'a,
{
    let mut best: Option<(usize, String)> = None;
    for c in candidates {
        let s = c.as_ref();
        if s == target {
            continue;
        }
        let d = levenshtein(target, s);
        if best.as_ref().map_or(true, |(bd, _)| d < *bd) {
            best = Some((d, s.to_string()));
        }
    }
    best.and_then(|(d, s)| {
        let limit = if target.len() <= 3 { 1 } else { 2 };
        if d <= limit {
            Some(s)
        } else {
            None
        }
    })
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca.eq_ignore_ascii_case(cb) { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Errors produced while building or compiling a JIT model.
#[derive(Debug)]
pub enum ModelError {
    /// Two declarations target the same compartment / output / let name.
    Duplicate(String),
    /// `dxdt(name)` or `init(name)` referenced a compartment not declared in `compartments = ...`.
    UnknownCompartment(String),
    /// A compartment was declared but no `dxdt(name) = ...` was provided.
    MissingDxDt(String),
    /// An expression could not be parsed. The original source is included for context.
    Parse {
        context: String,
        src: String,
        source: ParseError,
    },
    /// A name in an expression doesn't match any param, covariate, compartment, let, or `t`.
    UnresolvedIdent {
        ident: String,
        context: String,
        suggestion: Option<String>,
        available: Vec<String>,
    },
    /// `rateiv[i]` referenced a channel not within `0..ndrugs`.
    RateIvOutOfRange(usize, usize),
    /// `bolus[i]` referenced a channel not within `0..ndrugs`.
    BolusOutOfRange(usize, usize),
    /// Indexing was used on something other than `rateiv` / `bolus`.
    InvalidIndexTarget(String),
    /// A function call used a name that isn't a known builtin.
    UnknownFunction {
        name: String,
        suggestion: Option<String>,
    },
    /// A function call had the wrong number of arguments.
    BadArity {
        name: String,
        expected: usize,
        got: usize,
    },
    DuplicateLet(String),
    LetCycle(String),
    DuplicateInit(String),
    UnknownInitCompartment(String),
    DuplicateLag(usize),
    DuplicateFa(usize),
    LagOutOfRange(usize, usize),
    FaOutOfRange(usize, usize),
    /// User explicitly declared `ndrugs(N)` but expressions reference channels beyond it.
    NdrugsTooSmall {
        declared: usize,
        implied: usize,
        sites: Vec<String>,
    },
    /// Backend (Cranelift) error while emitting code.
    Codegen(String),
    /// A user-declared name collides with another user name (across kinds) or
    /// shadows a reserved/builtin identifier.
    NameCollision {
        name: String,
        kind: &'static str,
        reason: String,
    },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Duplicate(n) => write!(f, "duplicate name {n:?}"),
            ModelError::UnknownCompartment(n) => write!(f, "unknown compartment {n:?}"),
            ModelError::MissingDxDt(n) => write!(f, "compartment {n:?} has no dxdt definition"),
            ModelError::Parse {
                context,
                src,
                source,
            } => {
                write!(
                    f,
                    "parse error in {context}\n  source : {src}\n  detail : {source}"
                )
            }
            ModelError::UnresolvedIdent {
                ident,
                context,
                suggestion,
                available,
            } => {
                write!(f, "unresolved identifier `{ident}` in {context}")?;
                if let Some(s) = suggestion {
                    write!(f, "\n  hint   : did you mean `{s}`?")?;
                }
                if !available.is_empty() {
                    write!(f, "\n  in scope: {}", available.join(", "))?;
                }
                Ok(())
            }
            ModelError::RateIvOutOfRange(i, n) => {
                write!(f, "rateiv[{i}] is out of range (ndrugs = {n})")
            }
            ModelError::BolusOutOfRange(i, n) => {
                write!(f, "bolus[{i}] is out of range (ndrugs = {n})")
            }
            ModelError::InvalidIndexTarget(t) => write!(
                f,
                "invalid index target {t:?} (only `rateiv` and `bolus` are indexable)"
            ),
            ModelError::UnknownFunction { name, suggestion } => {
                write!(f, "unknown function `{name}`")?;
                if let Some(s) = suggestion {
                    write!(f, " (did you mean `{s}`?)")?;
                }
                write!(f, "\n  supported: {}", KNOWN_FUNCTIONS.join(", "))
            }
            ModelError::BadArity {
                name,
                expected,
                got,
            } => write!(f, "function `{name}` expects {expected} arg(s), got {got}"),
            ModelError::DuplicateLet(n) => write!(f, "duplicate `let` binding {n:?}"),
            ModelError::LetCycle(n) => write!(f, "let binding {n:?} forms a dependency cycle"),
            ModelError::DuplicateInit(n) => write!(f, "duplicate init for compartment {n:?}"),
            ModelError::UnknownInitCompartment(n) => {
                write!(f, "unknown compartment {n:?} in init declaration")
            }
            ModelError::DuplicateLag(c) => write!(f, "duplicate lag for input channel {c}"),
            ModelError::DuplicateFa(c) => write!(f, "duplicate fa  for input channel {c}"),
            ModelError::LagOutOfRange(c, n) => {
                write!(f, "lag channel {c} out of range (ndrugs = {n})")
            }
            ModelError::FaOutOfRange(c, n) => {
                write!(f, "fa  channel {c} out of range (ndrugs = {n})")
            }
            ModelError::NdrugsTooSmall {
                declared,
                implied,
                sites,
            } => {
                write!(
                    f,
                    "declared ndrugs = {declared}, but expressions reference up to channel {} (need ndrugs >= {implied})",
                    implied - 1
                )?;
                if !sites.is_empty() {
                    write!(f, "\n  uses   : {}", sites.join(", "))?;
                }
                Ok(())
            }
            ModelError::Codegen(s) => write!(f, "codegen error: {s}"),
            ModelError::NameCollision { name, kind, reason } => {
                write!(f, "{kind} name {name:?} is invalid: {reason}")
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A symbolic dxdt or output equation, parsed and stored in the model.
#[derive(Debug, Clone)]
pub(crate) struct Equation {
    pub(crate) target: String,
    pub(crate) expr: Expr,
    pub(crate) src: String,
    pub(crate) parse_err: Option<ParseError>,
}

/// A `let name = expr` binding declared in the model. Lets are scoped to the
/// whole model and may be referenced from any subsequent expression
/// (including other lets, dxdt, output, init, lag, fa).
#[derive(Debug, Clone)]
pub(crate) struct LetBinding {
    pub(crate) name: String,
    pub(crate) expr: Expr,
    pub(crate) src: String,
    pub(crate) parse_err: Option<ParseError>,
}

/// An `init(compartment) = expr` declaration.
#[derive(Debug, Clone)]
pub(crate) struct InitDecl {
    pub(crate) compartment: String,
    pub(crate) expr: Expr,
    pub(crate) src: String,
    pub(crate) parse_err: Option<ParseError>,
}

/// A `lag(channel) = expr` or `fa(channel) = expr` declaration.
#[derive(Debug, Clone)]
pub(crate) struct ChannelDecl {
    pub(crate) channel: usize,
    pub(crate) expr: Expr,
    pub(crate) src: String,
    pub(crate) parse_err: Option<ParseError>,
}

/// Mutable builder for a JIT model.
#[derive(Debug, Clone)]
pub struct Model {
    pub(crate) name: String,
    pub(crate) compartments: Vec<String>,
    pub(crate) params: Vec<String>,
    pub(crate) covariates: Vec<String>,
    /// Explicit user setting; `None` means "infer from expressions".
    pub(crate) ndrugs: Option<usize>,
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
            ndrugs: None,
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
    /// If never called, the value is auto-inferred from the maximum index
    /// referenced by `rateiv[N]`, `bolus[N]`, `lag(N)`, and `fa(N)`
    /// (defaulting to 1 if none are used).
    pub fn ndrugs(mut self, n: usize) -> Self {
        self.ndrugs = Some(n);
        self
    }

    /// Add a dxdt equation: `dxdt(compartment, "expression")`.
    pub fn dxdt(mut self, compartment: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let target = compartment.into();
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.dxdt.push(Equation {
            target,
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Add an output equation: `output(name, "expression")`. The output index
    /// matches the order of these calls.
    pub fn output(mut self, name: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let target = name.into();
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.outputs.push(Equation {
            target,
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Add a `let` binding: a named scalar expression that may be referenced
    /// (by name) in any subsequent equation. Useful for sharing
    /// sub-expressions between dxdt/out/init/lag/fa.
    pub fn let_binding(mut self, name: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.lets.push(LetBinding {
            name: name.into(),
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Add an initial-state expression for a compartment. Compartments
    /// without an `init` default to 0.
    pub fn init(mut self, compartment: impl Into<String>, expr: impl AsRef<str>) -> Self {
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.inits.push(InitDecl {
            compartment: compartment.into(),
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Add a lag-time expression for an input channel. Channels without a
    /// `lag` default to 0. Lag time is added to bolus event times only
    /// (matching pharmsol's existing semantics).
    pub fn lag(mut self, channel: usize, expr: impl AsRef<str>) -> Self {
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.lags.push(ChannelDecl {
            channel,
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Add a bioavailability expression for an input channel. Channels
    /// without an `fa` default to 1. Bioavailability scales bolus amounts
    /// only (matching pharmsol's existing semantics).
    pub fn fa(mut self, channel: usize, expr: impl AsRef<str>) -> Self {
        let src = expr.as_ref().to_string();
        let (expr, parse_err) = parse_or_sentinel(&src);
        self.fas.push(ChannelDecl {
            channel,
            expr,
            src,
            parse_err,
        });
        self
    }

    /// Validate, compile to native code via Cranelift, and return a [`JitOde`]
    /// ready to use with `Equation::simulate_subject`.
    pub fn compile(self) -> Result<JitOde, ModelError> {
        // 1. Surface any deferred parse errors first, with full source context.
        for eq in &self.dxdt {
            if let Some(e) = &eq.parse_err {
                return Err(ModelError::Parse {
                    context: format!("dxdt({})", eq.target),
                    src: eq.src.clone(),
                    source: e.clone(),
                });
            }
        }
        for eq in &self.outputs {
            if let Some(e) = &eq.parse_err {
                return Err(ModelError::Parse {
                    context: format!("out({})", eq.target),
                    src: eq.src.clone(),
                    source: e.clone(),
                });
            }
        }
        for lb in &self.lets {
            if let Some(e) = &lb.parse_err {
                return Err(ModelError::Parse {
                    context: format!("let {}", lb.name),
                    src: lb.src.clone(),
                    source: e.clone(),
                });
            }
        }
        for d in &self.inits {
            if let Some(e) = &d.parse_err {
                return Err(ModelError::Parse {
                    context: format!("init({})", d.compartment),
                    src: d.src.clone(),
                    source: e.clone(),
                });
            }
        }
        for d in &self.lags {
            if let Some(e) = &d.parse_err {
                return Err(ModelError::Parse {
                    context: format!("lag({})", d.channel),
                    src: d.src.clone(),
                    source: e.clone(),
                });
            }
        }
        for d in &self.fas {
            if let Some(e) = &d.parse_err {
                return Err(ModelError::Parse {
                    context: format!("fa({})", d.channel),
                    src: d.src.clone(),
                    source: e.clone(),
                });
            }
        }

        // 1b. Validate naming: no duplicates within a kind, no collisions
        //     between kinds, and no shadowing of reserved identifiers.
        validate_names(
            &self.compartments,
            &self.params,
            &self.covariates,
            &self.outputs,
            &self.lets,
        )?;

        // 2. Validate every compartment has a dxdt and reorder dxdt to match.
        let mut ordered_dxdt: Vec<Option<Equation>> = vec![None; self.compartments.len()];
        for eq in self.dxdt.iter().cloned() {
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
        let outputs = self.outputs.clone();
        let nout = outputs.len();

        // 3. Infer ndrugs from expression usage and lag/fa channels.
        let (implied_ndrugs, sites) = infer_ndrugs(
            &dxdt_in_order,
            &outputs,
            &self.lets,
            &self.inits,
            &self.lags,
            &self.fas,
        );
        let ndrugs = match self.ndrugs {
            Some(declared) => {
                if declared < implied_ndrugs {
                    return Err(ModelError::NdrugsTooSmall {
                        declared,
                        implied: implied_ndrugs,
                        sites,
                    });
                }
                declared
            }
            None => implied_ndrugs.max(1),
        };

        // 4. Validate let bindings: no duplicates.
        let mut seen_let = std::collections::HashSet::new();
        for lb in &self.lets {
            if !seen_let.insert(lb.name.clone()) {
                return Err(ModelError::DuplicateLet(lb.name.clone()));
            }
        }

        // 5. Validate inits: target must be a compartment, no duplicates.
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

        // 6. Validate lag/fa: channel in range, no duplicates.
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
            self.compartments,
            outputs.iter().map(|e| e.target.clone()).collect(),
            self.covariates,
            self.params,
        ))
    }
}

/// Parse `src` into an `Expr`, returning a NaN sentinel paired with the parse
/// error if parsing failed (so the builder API stays infallible and the error
/// can be surfaced with full source context inside `compile()`).
fn parse_or_sentinel(src: &str) -> (Expr, Option<ParseError>) {
    match parser::parse(src) {
        Ok(e) => (e, None),
        Err(e) => (Expr::Const(f64::NAN), Some(e)),
    }
}

/// Reserved identifiers that may not be used as a compartment, param,
/// covariate, output, or `let` name. These are recognised by the expression
/// language and shadowing them in the user namespace would produce confusing
/// or incorrect code.
const RESERVED_NAMES: &[&str] = &["t", "rateiv", "bolus"];

fn is_reserved(name: &str) -> bool {
    RESERVED_NAMES.contains(&name) || KNOWN_FUNCTIONS.contains(&name)
}

/// Validate the user-declared names: no duplicates within a kind, no
/// shadowing of reserved/builtin identifiers, and no collisions across kinds
/// that share a namespace inside expressions (compartments, params,
/// covariates, lets all resolve from the same identifier scope).
///
/// Output names live in their own namespace (they are not referenced by the
/// expression language) so they only need to be unique among themselves.
fn validate_names(
    compartments: &[String],
    params: &[String],
    covariates: &[String],
    outputs: &[Equation],
    lets: &[LetBinding],
) -> Result<(), ModelError> {
    // 1. No name in any kind may shadow a reserved identifier.
    for (kind, list) in [
        ("compartment", compartments),
        ("param", params),
        ("covariate", covariates),
    ] {
        for n in list {
            if is_reserved(n) {
                return Err(ModelError::NameCollision {
                    name: n.clone(),
                    kind,
                    reason: format!(
                        "shadows reserved identifier (reserved: {}; builtins: {})",
                        RESERVED_NAMES.join(", "),
                        KNOWN_FUNCTIONS.join(", "),
                    ),
                });
            }
        }
    }
    for lb in lets {
        if is_reserved(&lb.name) {
            return Err(ModelError::NameCollision {
                name: lb.name.clone(),
                kind: "let",
                reason: format!(
                    "shadows reserved identifier (reserved: {}; builtins: {})",
                    RESERVED_NAMES.join(", "),
                    KNOWN_FUNCTIONS.join(", "),
                ),
            });
        }
    }
    for eq in outputs {
        if is_reserved(&eq.target) {
            return Err(ModelError::NameCollision {
                name: eq.target.clone(),
                kind: "output",
                reason: format!(
                    "shadows reserved identifier (reserved: {}; builtins: {})",
                    RESERVED_NAMES.join(", "),
                    KNOWN_FUNCTIONS.join(", "),
                ),
            });
        }
    }

    // 2. Duplicates within a kind.
    fn dup_check(kind: &'static str, names: &[String]) -> Result<(), ModelError> {
        let mut seen = std::collections::HashSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(ModelError::NameCollision {
                    name: n.clone(),
                    kind,
                    reason: format!("declared more than once in {kind}s"),
                });
            }
        }
        Ok(())
    }
    dup_check("compartment", compartments)?;
    dup_check("param", params)?;
    dup_check("covariate", covariates)?;
    let output_names: Vec<String> = outputs.iter().map(|e| e.target.clone()).collect();
    dup_check("output", &output_names)?;
    let let_names: Vec<String> = lets.iter().map(|lb| lb.name.clone()).collect();
    dup_check("let", &let_names)?;

    // 3. Cross-kind collisions inside the expression namespace.
    //    Compartments, params, covariates, and lets all resolve from the same
    //    identifier scope (see `codegen::validate`). A name appearing in two
    //    of these would silently bind to one and produce a wrong model.
    let mut origin: std::collections::HashMap<&str, &'static str> =
        std::collections::HashMap::new();
    for (kind, list) in [
        ("compartment", compartments),
        ("param", params),
        ("covariate", covariates),
    ] {
        for n in list {
            if let Some(prev) = origin.insert(n.as_str(), kind) {
                if prev != kind {
                    return Err(ModelError::NameCollision {
                        name: n.clone(),
                        kind,
                        reason: format!("also declared as a {prev}"),
                    });
                }
            }
        }
    }
    for lb in lets {
        if let Some(prev) = origin.insert(lb.name.as_str(), "let") {
            if prev != "let" {
                return Err(ModelError::NameCollision {
                    name: lb.name.clone(),
                    kind: "let",
                    reason: format!("also declared as a {prev}"),
                });
            }
        }
    }

    Ok(())
}

/// Walk every expression in the model and find the maximum drug-channel index
/// that's referenced. Returns `(implied_ndrugs, sites)` where `implied_ndrugs`
/// is `max_index + 1` (or 0 if nothing references a channel) and `sites` is a
/// human-readable list of where the references appear (used in error reports
/// when the user-declared ndrugs is too small).
fn infer_ndrugs(
    dxdt: &[Equation],
    outputs: &[Equation],
    lets: &[LetBinding],
    inits: &[InitDecl],
    lags: &[ChannelDecl],
    fas: &[ChannelDecl],
) -> (usize, Vec<String>) {
    let mut max_idx: Option<usize> = None;
    let mut sites: Vec<String> = Vec::new();

    fn visit_expr(context: &str, e: &Expr, max_idx: &mut Option<usize>, sites: &mut Vec<String>) {
        scan_drug_indices(e, &mut |kind, i| {
            if max_idx.map_or(true, |m| i > m) {
                *max_idx = Some(i);
            }
            sites.push(format!("{kind}[{i}] in {context}"));
        });
    }

    for eq in dxdt {
        visit_expr(
            &format!("dxdt({})", eq.target),
            &eq.expr,
            &mut max_idx,
            &mut sites,
        );
    }
    for eq in outputs {
        visit_expr(
            &format!("out({})", eq.target),
            &eq.expr,
            &mut max_idx,
            &mut sites,
        );
    }
    for lb in lets {
        visit_expr(
            &format!("let {}", lb.name),
            &lb.expr,
            &mut max_idx,
            &mut sites,
        );
    }
    for d in inits {
        visit_expr(
            &format!("init({})", d.compartment),
            &d.expr,
            &mut max_idx,
            &mut sites,
        );
    }
    for d in lags {
        visit_expr(
            &format!("lag({})", d.channel),
            &d.expr,
            &mut max_idx,
            &mut sites,
        );
        if max_idx.map_or(true, |m| d.channel > m) {
            max_idx = Some(d.channel);
        }
        sites.push(format!("lag({})", d.channel));
    }
    for d in fas {
        visit_expr(
            &format!("fa({})", d.channel),
            &d.expr,
            &mut max_idx,
            &mut sites,
        );
        if max_idx.map_or(true, |m| d.channel > m) {
            max_idx = Some(d.channel);
        }
        sites.push(format!("fa({})", d.channel));
    }

    let implied = max_idx.map(|m| m + 1).unwrap_or(0);
    (implied, sites)
}

/// Recursively walk an expression, invoking `visit(kind, idx)` for every
/// `rateiv[idx]` and `bolus[idx]` reference encountered.
fn scan_drug_indices<F: FnMut(&'static str, usize)>(e: &Expr, visit: &mut F) {
    match e {
        Expr::Const(_) | Expr::Ident(_) => {}
        Expr::Index(name, idx) => match name.as_str() {
            "rateiv" => visit("rateiv", *idx),
            "bolus" => visit("bolus", *idx),
            _ => {}
        },
        Expr::Neg(inner) => scan_drug_indices(inner, visit),
        Expr::Bin(_, l, r) => {
            scan_drug_indices(l, visit);
            scan_drug_indices(r, visit);
        }
        Expr::Call(_, args) => {
            for a in args {
                scan_drug_indices(a, visit);
            }
        }
    }
}
