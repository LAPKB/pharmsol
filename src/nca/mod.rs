//! Non-Compartmental Analysis (NCA) for pharmacokinetic data
//!
//! This module provides functions for calculating standard NCA parameters from
//! concentration-time data. It is designed to work directly with pharmsol's
//! data structures ([`Subject`], [`Occasion`]) and also provides lower-level
//! functions for use with raw time/concentration arrays (useful for PMcore's bestdose).
//!
//! # Overview
//!
//! NCA is a model-independent approach to pharmacokinetic analysis that calculates
//! exposure metrics directly from observed concentration-time data without assuming
//! a specific compartmental model.
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
//!
//! # AUC Calculation Methods
//!
//! Two methods are supported:
//!
//! - **Linear Trapezoidal**: Simple average of adjacent concentrations
//!   ```text
//!   AUC = Σ (C[i] + C[i-1])/2 × (t[i] - t[i-1])
//!   ```
//!
//! - **Linear Up/Log Down** (recommended): Linear for ascending concentrations,
//!   log-linear for descending. This is the industry standard.
//!   ```text
//!   Ascending:  AUC = (C[i] + C[i-1])/2 × Δt
//!   Descending: AUC = (C[i-1] - C[i]) × Δt / ln(C[i-1]/C[i])
//!   ```
//!
//! # Examples
//!
//! ## Direct calculation from arrays (PMcore-compatible)
//!
//! ```rust
//! use pharmsol::nca::{auc_last, AUCMethod};
//!
//! let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
//! let concs = vec![0.0, 8.5, 6.2, 3.1, 1.2, 0.4];
//!
//! let auc = auc_last(&times, &concs, AUCMethod::LinUpLogDown);
//! println!("AUClast = {:.2}", auc);
//! ```
//!
//! ## Full NCA on a Subject
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//! use pharmsol::nca::{NCAOptions, NCAResult};
//!
//! let subject = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, 0)
//!     .observation(1.0, 8.5, 0)
//!     .observation(2.0, 6.2, 0)
//!     .observation(4.0, 3.1, 0)
//!     .build();
//!
//! let options = NCAOptions::default();
//! let result = subject.nca(&options);
//!
//! println!("Cmax: {:.2}", result.cmax);
//! println!("AUClast: {:.2}", result.auc_last);
//! ```
//!
//! # Module Structure
//!
//! - [`auc`]: AUC calculation functions (trapezoidal integration)
//! - [`params`]: Primary NCA parameters (Cmax, Tmax, Clast, Tlast)
//! - [`terminal`]: Terminal phase analysis (λz, half-life)
//! - [`results`]: Result structures and options

pub mod auc;
pub mod params;
pub mod results;
pub mod terminal;

// Re-export commonly used items
pub use auc::{
    auc_all, auc_at_times, auc_cumulative, auc_interval, auc_last, auc_segment, AUCMethod,
};
pub use params::{cav, peak_trough_ratio, percent_fluctuation, swing};
pub use params::{clast_tlast, cmax_tmax, cmin_tmin, ClastTlast, CmaxTmax, CminTmin};
pub use results::{calculate_nca, AdministrationRoute, BLQRule, NCAOptions, NCAResult};
pub use terminal::{
    auc_inf, auc_inf_pred, auc_percent_extrap, aumc_last, aumc_segment, c0_iv_bolus, clearance,
    ka_extravascular, lambda_z, lambda_z_auto, mrt, tlag_extravascular, vd_iv_bolus, vss_iv, vz,
    LambdaZMethod, LambdaZOptions, LambdaZResult, RegressionWeight,
};
