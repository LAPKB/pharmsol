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
//! let results = subject.nca(&NCAOptions::default(), 0);
//! let result = results[0].as_ref().expect("NCA failed");
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
//! let results = subject.nca(&options, 0);
//!
//! if let Some(ref ss) = results[0].as_ref().unwrap().steady_state {
//!     println!("Cavg: {:.2}", ss.cavg);
//!     println!("Fluctuation: {:.1}%", ss.fluctuation);
//! }
//! ```

// Internal modules
mod analyze;
mod calc;
mod error;
mod profile;
mod types;

#[cfg(test)]
mod tests;

// Crate-internal re-exports (for data/structs.rs)
pub(crate) use analyze::{analyze_arrays, DoseContext};

// Public API
pub use error::NCAError;
pub use types::{
    AUCMethod, BLQRule, C0Method, ClastType, ClearanceParams, ExposureParams, IVBolusParams,
    IVInfusionParams, LambdaZMethod, LambdaZOptions, NCAOptions, NCAResult, Quality,
    RegressionStats, Route, SteadyStateParams, TerminalParams, Warning,
};
