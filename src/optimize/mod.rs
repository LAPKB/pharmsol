//! Optimizer-oriented helpers for pharmacometric workflows.
//!
//! This module provides optimization utilities built on [`argmin`]:
//!
//! - [`effect`] — Find maximum effects for two- and three-site Greco response
//!   surfaces via Nelder‑Mead optimization in log‑space.
//! - [`parameters`] — Nelder‑Mead parameter refinement for an [`Equation`]
//!   against a [`Data`] set and [`AssayErrorModels`].

pub mod effect;
pub mod parameters;
