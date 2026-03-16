//! Data structures and utilities for pharmacometric modeling
//!
//! This module provides types for representing pharmacokinetic/pharmacodynamic data,
//! including subjects, dosing events, observations, and covariates. It also includes
//! utilities for reading and manipulating this data.
//!
//! # Key Components
//!
//! - **Events**: Dosing events (bolus, infusion) and observations
//! - **Covariates**: Time-varying subject characteristics
//! - **Subjects**: Collections of events and covariates for a single individual
//! - **Data**: Collections of subjects, representing a complete dataset
//! - **Error Models**: Two types for different algorithm families:
//!   - [`ErrorModel`]: Observation-based (assay error) for non-parametric algorithms
//!   - [`ResidualErrorModel`]: Prediction-based (residual error) for parametric algorithms
//!
//! # Examples
//!
//! Creating a subject with the builder pattern:
//!
//! ```rust
//! use pharmsol::*;
//!
//! let subject = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, 0)
//!     .observation(1.0, 10.5, 0)
//!     .observation(2.0, 8.2, 0)
//!     .covariate("weight", 0.0, 70.0)
//!     .build();
//! ```

pub mod builder;
pub mod covariate;
pub mod error_model;
pub mod event;
pub mod parser;
pub mod residual_error;
pub mod row;
pub mod structs;
pub use covariate::*;
pub use error_model::*;
pub use event::*;
pub use residual_error::*;
pub use structs::{Data, Occasion, Subject};
