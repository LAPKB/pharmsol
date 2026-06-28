//! Optimizer-oriented helpers for pharmacometric workflows.
//!
//! This module provides optimization utilities built on [`argmin`]:
//!
//! - [`effect`] — Find the maximum effect (`E2`) for dual-site PD models
//!   via Nelder‑Mead optimization in log‑space.
//! - [`parameters`] — Nelder‑Mead parameter refinement for a [`Simulate`] model
//!   against a [`Data`] set and [`AssayErrorModels`].

pub mod effect;
pub mod parameters;
