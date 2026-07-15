//! Data structures for building pharmacometric input data.
//!
//! Use this module when you need to describe what happened to each subject:
//! doses, infusions, observations, covariates, and occasion boundaries.
//!
//! This module is the input side of `pharmsol`. It is where you assemble
//! subjects and datasets before simulation, estimation, or NCA. It is not where
//! you define model equations or choose a backend. For those workflows, move to
//! [`crate::simulator`], [`crate::nca`], or the feature-gated `pharmsol::dsl`
//! surface.
//!
//! # Start Here
//!
//! Most users only need three entrypoints first:
//!
//! - [`Subject`] for one individual and their full schedule.
//! - [`Data`] for a dataset containing many subjects.
//! - `Subject::builder` for the smallest fluent API to create doses,
//!   observations, and covariates in Rust.
//!
//! The main supporting types are:
//!
//! - [`Occasion`] for repeated periods within one subject.
//! - [`Event`], [`Bolus`], [`Infusion`], and [`Observation`] for explicit
//!   event-level control.
//! - [`Covariate`] and [`Covariates`] for time-varying subject characteristics.
//! - [`ErrorPoly`] for the transport-neutral Pmetrics C0-C3 values attached to
//!   observations.
//! - [`ObservationError`] for invalid or insufficient observation data during
//!   profile construction and related preprocessing.
//!
//! # Choose A Data Input Path
//!
//! - Use `Subject::builder` when you are authoring a schedule directly in Rust.
//! - Use [`row::DataRow`] and [`row::DataRowBuilder`] when your source data is
//!   already row-shaped in memory.
//! - Use [`parser::read_pmetrics`] when you are loading a Pmetrics-style file
//!   from disk.
//! - Use [`Event`] variants directly when you already have validated event
//!   records and need lower-level control than the builder offers.
//!
//! # Label Semantics
//!
//! Dosing inputs and observation outputs use public labels.
//!
//! - The `input` on [`Bolus`] and [`Infusion`] is the route or input label that
//!   will be matched against the model.
//! - The `outeq` on [`Observation`] is the output label that identifies which
//!   model output the observation belongs to.
//! - Prefer stable names such as `"depot"`, `"central"`, `"iv"`, or `"cp"`.
//! - If you pass a number, it is still treated as a public label string. Use
//!   numeric values only when your model intentionally declares numeric labels.
//!
//! [`Occasion`] indices are different: they are integer period markers used to
//! separate repeated dosing blocks within one subject.
//!
//! # Observation ErrorPoly Data
//!
//! Observations can carry a transport-neutral [`ErrorPoly`] DTO (the Pmetrics
//! C0-C3 columns). pharmsol stores and round-trips these values verbatim.
//!
//! # Example
//!
//! ```rust
//! use pharmsol::*;
//!
//! let subject = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, "depot")
//!     .observation(1.0, 12.3, "cp")
//!     .missing_observation(2.0, "cp")
//!     .covariate("weight", 0.0, 70.0)
//!     .build();
//!
//! let data = Data::new(vec![subject]);
//!
//! assert_eq!(data.subjects().len(), 1);
//! ```

pub mod auc;
pub mod builder;
pub mod covariate;
pub mod error_poly;
pub mod event;
pub mod observation_error;
pub mod parser;
pub mod row;
pub mod structs;
pub use crate::nca::{MetricsError, ObservationMetrics};
pub use covariate::*;
pub use error_poly::ErrorPoly;
pub use event::*;
pub use observation_error::ObservationError;
pub use structs::{Data, Occasion, Subject};
