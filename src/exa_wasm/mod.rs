//! WASM-compatible `exa` alternative.
//!
//! This module contains a small IR emitter and an interpreter that can run
//! user-defined models in WASM hosts without requiring cargo compilation or
//! dynamic library loading. It's gated under the `exa-wasm` cargo feature.

pub mod build;
pub mod interpreter;

pub use build::emit_ir;
pub use interpreter::{load_ir_ode, ode_for_id, unregister_model};
