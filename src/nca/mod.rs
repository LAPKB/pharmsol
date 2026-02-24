//! Non-Compartmental Analysis (NCA) for pharmacokinetic data
//!
//! This module provides a clean, powerful API for calculating standard NCA parameters
//! from concentration-time data. It integrates seamlessly with pharmsol's data structures
//! ([`crate::Subject`], [`crate::Occasion`]).
//!
//! # Design Philosophy
//!
//! - **Simple**: Single entry point via `.nca()` method on data structures
//! - **Powerful**: Full support for all standard NCA parameters
//! - **Data-aware**: Doses and routes are auto-detected from the data
//! - **Configurable**: Analysis options via [`NCAOptions`]
//!
//! # Key Parameters
//!
//! | Parameter | Description |
//! |-----------|-------------|
//! | Cmax | Maximum observed concentration |
//! | Tmax | Time of maximum concentration |
//! | Clast | Last measurable concentration (> 0) |
//! | Tlast | Time of last measurable concentration |
//! | AUClast | Area under curve from 0 to Tlast |
//! | AUCinf | AUC extrapolated to infinity |
//! | λz | Terminal elimination rate constant |
//! | t½ | Terminal half-life (ln(2)/λz) |
//! | CL/F | Apparent clearance |
//! | Vz/F | Apparent volume of distribution |
//! | MRT | Mean residence time |
//!
//! # Usage
//!
//! NCA is performed by calling `.nca()` on a `Subject`. Dose and route
//! information are automatically detected from the dose events in the data.
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//! use pharmsol::nca::NCAOptions;
//!
//! // Build subject with dose and observation events
//! let subject = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, 0)  // 100 mg oral dose
//!     .observation(1.0, 10.0, 0)
//!     .observation(2.0, 8.0, 0)
//!     .observation(4.0, 4.0, 0)
//!     .build();
//!
//! // Perform NCA with default options
//! let result = subject.nca(&NCAOptions::default()).expect("NCA failed");
//!
//! println!("Cmax: {:.2}", result.exposure.cmax);
//! println!("AUClast: {:.2}", result.exposure.auc_last);
//! ```
//!
//! # Steady-State Analysis
//!
//! ```rust,ignore
//! use pharmsol::nca::NCAOptions;
//!
//! // Configure for steady-state with 12h dosing interval
//! let options = NCAOptions::default().with_tau(12.0);
//! let result = subject.nca(&options).unwrap();
//!
//! if let Some(ref ss) = result.steady_state {
//!     println!("Cavg: {:.2}", ss.cavg);
//!     println!("Fluctuation: {:.1}%", ss.fluctuation);
//! }
//! ```
//!
//! # Population Analysis
//!
//! ```rust,ignore
//! use pharmsol::nca::{NCAOptions, NCA, NCAPopulation};
//!
//! // All occasions flat
//! let all_results = data.nca_all(&options);
//!
//! // Grouped by subject (includes error isolation)
//! let grouped = data.nca_grouped(&options);
//! for subj in &grouped {
//!     println!("{}: {} ok, {} errors",
//!         subj.subject_id,
//!         subj.successes().len(),
//!         subj.errors().len());
//! }
//! ```

// Internal modules
mod analyze;
mod calc;
mod error;
pub mod summary;
mod traits;
mod types;

// Feature modules
pub mod bioavailability;
pub mod sparse;
pub mod superposition;

#[cfg(test)]
mod tests;

// Crate-internal re-exports
// (traits.rs accesses analyze::analyze and calc::tlag_from_raw directly)

// Public API
pub use bioavailability::{
    bioavailability, bioequivalence, compare, metabolite_parent_ratio, BioavailabilityResult,
    BioequivalenceResult,
};

pub use error::NCAError;
pub use sparse::{sparse_auc, sparse_auc_from_data, SparsePKResult};
pub use summary::{nca_to_csv, summarize, ParameterSummary, PopulationSummary};
pub use superposition::{Superposition, SuperpositionResult};
pub use traits::{NCAPopulation, SubjectNCAResult, NCA};
pub use types::{
    C0Method, ClearanceParams, ExposureParams, IVBolusParams, IVInfusionParams, LambdaZMethod,
    LambdaZOptions, MultiDoseParams, NCAOptions, NCAResult, Quality, RegressionStats, RouteParams,
    Severity, SteadyStateParams, TerminalParams, Warning,
};

// Re-export shared types (backwards compatible)
pub use crate::data::event::{AUCMethod, BLQRule, Route};
