//! Non-Compartmental Analysis (NCA) for pharmacokinetic data
//!
//! This module provides a clean, powerful API for calculating standard NCA parameters
//! from concentration-time data. It integrates seamlessly with pharmsol's data structures
//! ([`Data`], [`Subject`], [`Occasion`]).
//!
//! # Design Philosophy
//!
//! - **Simple**: Single entry point via `nca()` method on data structures
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
//! NCA is performed by calling `nca()` on data structures. Dose and route
//! information are automatically detected from the dose events in the data.
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//! use pharmsol::nca::NCAOptions;
//!
//! // Load data with dose and observation events
//! let data = read_pmetrics("study.csv")?;
//!
//! // Analyze all subjects with default options
//! let results = data.nca(&NCAOptions::default())?;
//!
//! // Or analyze a single subject
//! let subject_results = data.get_subject("001")?.nca(&NCAOptions::default())?;
//! ```
//!
//! # Steady-State Analysis
//!
//! ```rust,ignore
//! use pharmsol::nca::NCAOptions;
//!
//! // Configure for steady-state with 12h dosing interval
//! let options = NCAOptions::default().with_tau(12.0);
//! let result = subject.nca(&options)?;
//!
//! if let Some(ss) = &result[0].steady_state {
//!     println!("Cavg: {:.2}", ss.cavg);
//!     println!("Fluctuation: {:.1}%", ss.fluctuation);
//! }
//! ```

// Internal modules
pub mod analyze;
mod calc;
mod error;
mod profile;
mod types;

#[cfg(test)]
mod tests;

// Public API
pub use analyze::DoseContext;
pub use error::NCAError;
pub use types::{
    AUCMethod, BLQRule, ClearanceParams, ExposureParams, ExtravascularParams,
    IVBolusParams, IVInfusionParams, LambdaZMethod, LambdaZOptions, NCAOptions, NCAResult,
    Quality, RegressionStats, Route, SteadyStateParams, TerminalParams, Warning,
};

use profile::Profile;

/// Perform NCA analysis from raw time and concentration arrays
///
/// This is the primary entry point for NCA analysis when working with
/// raw data arrays rather than pharmsol data structures.
///
/// # Arguments
///
/// * `times` - Time points (must be monotonically increasing)
/// * `concentrations` - Concentration values at each time point
/// * `options` - Analysis configuration
///
/// # Returns
///
/// Complete [`NCAResult`] with all computable parameters, or [`NCAError`] if analysis fails.
///
/// # Example
///
/// ```rust
/// use pharmsol::nca::{NCAOptions, nca_from_arrays};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
/// let concs = vec![0.0, 8.5, 6.2, 3.1, 1.2, 0.4];
///
/// // Without dose - basic exposure metrics only
/// let result = nca_from_arrays(&times, &concs, None, &NCAOptions::default()).unwrap();
/// assert!(result.exposure.cmax > 0.0);
/// ```
pub fn nca_from_arrays(
    times: &[f64],
    concentrations: &[f64],
    dose: Option<&DoseContext>,
    options: &NCAOptions,
) -> Result<NCAResult, NCAError> {
    let profile = Profile::from_arrays(times, concentrations, options.loq, options.blq_rule)?;
    analyze::analyze(&profile, dose, options)
}
