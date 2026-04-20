//! JIT-compiled pharmacometric models.
//!
//! Build a model in Rust with the [`Model`] builder, call [`Model::compile`],
//! and use the returned [`JitOde`] anywhere an [`crate::Equation`] is expected.
//!
//! No source files, no `cargo`/`rustc`, no JSON. Equations are written as
//! short string expressions over named compartments, parameters, and
//! covariates and are compiled to native code by Cranelift in tens of
//! milliseconds. Per-step evaluation runs at native speed (no interpretation,
//! no trampolines beyond a single indirect call).
//!
//! # Example
//!
//! ```ignore
//! use pharmsol::jit::Model;
//! use pharmsol::prelude::*;
//!
//! let ode = Model::new("1cmt-iv-bolus")
//!     .compartments(["central"])
//!     .params(["CL", "V"])
//!     .dxdt("central", "rateiv[0] - (CL / V) * central")
//!     .output("cp", "central / V")
//!     .compile()?;
//!
//! let subject = Subject::builder("p1")
//!     .bolus(0.0, 100.0, 0)
//!     .observation(1.0, 0.0, 0)
//!     .build();
//!
//! let (preds, _) = ode.simulate_subject(&subject, &[10.0, 50.0], None)?;
//! # Ok::<(), pharmsol::PharmsolError>(())
//! ```
//!
//! # Supported expression syntax
//!
//! - Numeric literals (`1.5e-2`, `42`)
//! - Identifiers: parameter names, compartment names, covariate names, and `t`
//! - Indexing: `rateiv[i]` (infusion rates) and `bolus[i]` (always 0 inside
//!   expressions; pharmsol applies bolus state changes itself)
//! - Operators: `+ - * /` (left assoc) and `^` (right assoc, calls `pow`)
//! - Functions: `exp`, `ln`, `log` (alias of `ln`), `log10`, `sqrt`, `abs`,
//!   `pow(a, b)`

mod ast;
mod codegen;
mod equation;
mod model;
mod parser;
pub mod text;

pub use equation::JitOde;
pub use model::{Model, ModelError};
pub use parser::ParseError;
pub use text::TextError;
