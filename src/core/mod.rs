//! Core traits for the pharmsol simulation framework.
//!
//! This module holds the foundational traits that separate concerns:
//!
//! - [`Solver`] — *How* to advance state through time (backend authors implement this).
//! - [`ModelInfo`] — *What* the model's structure is (dimensions, metadata, lag/fa).
//! - [`Caching`] — *Where* prediction and error-model caches live.
//! - [`Simulate`] — *User-facing* prediction and likelihood API. Anything that
//!   implements `Solver + ModelInfo + Caching` can become `Simulate`.
//! - [`State`] — Low-level state vector that can receive bolus doses.
//! - [`Predictions`] — Prediction containers with log-likelihood computation.
//!
//! The free function [`standard_event_loop`] provides the default simulation
//! driver — iterate events, apply boluses, track infusions, compute observations,
//! and advance time via [`Solver::solve`]. Backends that use batch integration
//! (e.g. diffsol-based ODE) set [`Solver::is_batch`] to `true` and implement
//! `Simulate::simulate_subject` themselves.

pub mod caching;
pub mod metadata;
pub mod model_core;
pub mod model_info;
pub mod predictions;
pub mod simulate;
pub mod solver;
pub mod state;

pub use caching::Caching;
pub use model_core::ModelCore;
pub use model_info::ModelInfo;
pub use predictions::Predictions;
pub use simulate::{standard_event_loop, PredictionsContainer, Simulate};
pub use solver::Solver;
pub use state::State;
