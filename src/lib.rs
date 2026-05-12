//! `pharmsol` is a Rust library for pharmacometric work.
//!
//! You can use it to:
//!
//! - build PK/PD datasets from dose and observation events
//! - simulate analytical, ODE, and SDE models
//! - run non-compartmental analysis (NCA)
//! - compile and run models from the pharmsol DSL when the DSL features are enabled
//!
//! Most users start in one of these places:
//!
//! - [`prelude`] for the common types, traits, and macros
//! - [`data`] to build subjects, occasions, events, and covariates
//! - [`simulator`] to define models and generate predictions
//! - [`nca`] to calculate NCA metrics from the same data structures
//! - [`optimize`] for optimizer-oriented workflows
//!
//! The DSL runtime surface is feature-gated. When you enable `dsl-core`, the
//! `pharmsol::dsl` module adds parsing, analysis, lowering, compile, and runtime
//! entrypoints for models written as DSL source text.
//!
//! ## Quick Start
//!
//! This example shows the smallest full workflow: define a model, build a
//! subject, and generate predictions.
//!
//! ```rust
//! use pharmsol::{Parameters, prelude::*};
//!
//! let model = analytical! {
//!     name: "one_cmt_iv",
//!     params: [ke, v],
//!     states: [central],
//!     outputs: [cp],
//!     routes: [
//!         infusion(iv) -> central,
//!     ],
//!     structure: one_compartment,
//!     out: |x, _p, _t, _cov, y| {
//!         y[cp] = x[central] / v;
//!     },
//! };
//!
//! let subject = Subject::builder("patient_001")
//!     .infusion(0.0, 500.0, "iv", 0.5)
//!     .missing_observation(0.5, "cp")
//!     .missing_observation(1.0, "cp")
//!     .build();
//!
//! let parameters = Parameters::with_model(&model, [("ke", 1.022), ("v", 194.0)])
//!     .expect("valid named parameters");
//! let predictions = model.estimate_predictions(&subject, &parameters)?;
//! assert_eq!(predictions.flat_predictions().len(), 2);
//! # Ok::<(), pharmsol::PharmsolError>(())
//! ```
//!
//! For metadata-backed models, prefer [`Parameters::with_model`] for one
//! support point and [`ParameterOrder::with_model`] for repeated dense batches.
//! Raw slices and dense matrices remain the low-level execution substrate, and
//! handwritten models without attached parameter metadata cannot validate named
//! input.
//!
//! ## Choose A Workflow
//!
//! Use this guide when you are deciding where to start.
//!
//! | Task | Start Here | Notes |
//! | --- | --- | --- |
//! | Build subject data | [`data`] or [`prelude`] | Best when you already know dose times, labels, and observations. |
//! | Simulate a model written in Rust | [`simulator`] or [`prelude`] | Supports analytical, ODE, and SDE models. |
//! | Run NCA | [`nca`] or [`prelude`] | Reuses the same `Subject`, `Occasion`, and `Data` types. |
//! | Use optimization helpers | [`optimize`] | Intended for advanced workflows. |
//! | Parse or compile DSL source | `pharmsol::dsl` | Requires one or more DSL features. |
//!
//! ## Feature Guide
//!
//! Core simulation and NCA APIs do not need extra crate features on native
//! targets.
//!
//! DSL work is feature-gated:
//!
//! - `dsl-core`: exposes the `pharmsol::dsl` facade and frontend types
//! - `dsl-jit`: adds in-process JIT compilation
//! - `dsl-aot`: adds native ahead-of-time artifact compilation
//! - `dsl-aot-load`: adds native artifact loading
//! - `dsl-wasm-compile`: adds WASM artifact generation
//! - `dsl-wasm`: adds WASM runtime loading and execution
//!
//! ## Labels And Indices
//!
//! Public data APIs use route labels and output labels such as `"iv"`,
//! `"oral"`, and `"cp"`.
//!
//! Use labels in builders and parsed data unless you are deliberately working
//! with dense internal indices from a lower-level API.
//!
//! ## Platform Notes
//!
//! The main `data`, `simulator`, `nca`, and `optimize` modules are documented
//! for native targets. Some surfaces are not built on `wasm32-unknown-unknown`.
//! The DSL runtime also has feature-specific platform limits.
//!
//! ## Next Stops
//!
//! - Start with [`prelude`] if you want one import for the common workflow.
//! - Open [`data`] if you need to construct subjects or parse input files.
//! - Open [`simulator`] if you need predictions from analytical, ODE, or SDE models.
//! - Open [`nca`] if you need exposure and terminal metrics.
//! - Use `pharmsol::dsl` if the model comes from source text instead of Rust code.

#[cfg(feature = "dsl-aot")]
mod build_support;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod data;
#[cfg(feature = "dsl-core")]
pub mod dsl;
pub mod error;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod nca;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod optimize;
mod parameter_order;
mod parameters;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod simulator;
#[cfg(all(
    test,
    any(
        feature = "dsl-jit",
        all(feature = "dsl-wasm", feature = "dsl-wasm-compile")
    )
))]
mod test_fixtures;

//extension traits
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::data::builder::SubjectBuilderExt;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::data::Interpolation::*;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::data::*;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::optimize::effect::get_e2;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::optimize::parameters::ParameterOptimizer;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::simulator::equation::analytical::*;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::simulator::equation::metadata;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use crate::simulator::equation::{
    self,
    ode::{ExplicitRkTableau, OdeSolver, SdirkTableau},
    Analytical, AnalyticalKernel, Cache, Equation, ModelKind, ModelMetadata, ModelMetadataError,
    NameDomain, Predictions, RouteInputPolicy, RouteKind, State, ValidatedModelMetadata, ODE, SDE,
};
pub use error::PharmsolError;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use nalgebra::dmatrix;
pub use parameters::{ParameterError, ParameterOrder, Parameters};
pub use pharmsol_macros::{analytical, ode, sde};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use std::collections::HashMap;

/// Common imports for the main pharmsol workflow.
///
/// Use the prelude when you want one import that covers the common public API:
///
/// - subject and dataset types
/// - subject builders and events
/// - simulation types and prediction results
/// - NCA traits and option types
/// - declaration-first macros such as [`crate::ode`] and [`crate::analytical`]
///
/// This is the fastest way to get started with examples, scripts, and small
/// applications.
///
/// If you need a narrower import surface, use the modules directly instead.
///
/// # Example
/// ```rust
/// use pharmsol::prelude::*;
///
/// let subject = Subject::builder("patient_001")
///     .infusion(0.0, 100.0, "iv", 1.0)
///     .missing_observation(1.0, "cp")
///     .build();
///
/// assert_eq!(subject.id(), "patient_001");
/// ```
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod prelude {

    pub use crate::Parameters;
    // Re-export error type
    pub use crate::error::PharmsolError;

    // Data submodule for organized access and backward compatibility
    pub mod data {
        pub use crate::data::{
            error_model::{AssayErrorModel, AssayErrorModels},
            parser::{read_pmetrics, DataRow, DataRowBuilder},
            residual_error::{ResidualErrorModel, ResidualErrorModels},
            Covariates, Data, Event, Occasion, Subject,
        };
    }

    // Direct data re-exports for convenience
    pub use crate::data::{
        builder::SubjectBuilderExt,
        error_model::{AssayErrorModel, AssayErrorModels, ErrorPoly},
        event::{AUCMethod, BLQRule, Route},
        Covariates, Data, Event, Interpolation, Occasion, Subject,
    };

    // NCA extension traits (provides .nca(), .nca_all(), etc. on data types)
    pub use crate::nca::NCA;
    pub use crate::nca::{MetricsError, ObservationMetrics};
    pub use crate::nca::{NCAOptions, NCAPopulation, SubjectNCAResult};

    // AUC primitives for direct use on raw arrays
    pub use crate::data::auc::{auc, auc_interval, aumc, interpolate_linear};

    #[allow(deprecated)]
    // Simulator submodule for organized access to simulation types.
    pub mod simulator {
        pub use crate::simulator::{
            cache::{self, PredictionCache, SdeLikelihoodCache, DEFAULT_CACHE_SIZE},
            equation,
            equation::Equation,
            likelihood::{
                log_likelihood_batch, log_likelihood_matrix, log_likelihood_subject, log_psi, psi,
                PopulationPredictions, Prediction, SubjectPredictions,
            },
        };
    }

    // Direct simulator re-exports for convenience
    pub use crate::simulator::{
        cache::{PredictionCache, SdeLikelihoodCache, DEFAULT_CACHE_SIZE},
        equation::{
            self,
            ode::{ExplicitRkTableau, OdeSolver, SdirkTableau},
            Equation,
        },
        likelihood::{Prediction, SubjectPredictions},
    };

    // Analytical model functions
    pub use crate::simulator::equation::analytical::{
        one_compartment, one_compartment_cl, one_compartment_cl_with_absorption,
        one_compartment_with_absorption, three_compartments, three_compartments_with_absorption,
        two_compartments, two_compartments_cl, two_compartments_cl_with_absorption,
        two_compartments_with_absorption,
    };

    /// Models submodule for organized access to analytical model functions
    pub mod models {
        pub use crate::simulator::equation::analytical::{
            one_compartment, one_compartment_cl, one_compartment_cl_with_absorption,
            one_compartment_with_absorption, three_compartments,
            three_compartments_with_absorption, two_compartments, two_compartments_cl,
            two_compartments_cl_with_absorption, two_compartments_with_absorption,
        };
    }

    // Re-export macros (they are exported at crate root via #[macro_export])
    #[doc(inline)]
    pub use crate::analytical;
    #[doc(inline)]
    pub use crate::fa;
    #[doc(inline)]
    pub use crate::fetch_cov;
    #[doc(inline)]
    pub use crate::fetch_params;
    #[doc(inline)]
    pub use crate::lag;
    #[doc(inline)]
    pub use crate::ode;
    #[doc(inline)]
    pub use crate::sde;
}

#[macro_export]
macro_rules! fetch_params {
    ($p:expr, $($name:ident),*) => {
        let p = $p;
        let mut idx = 0;
        $(
            #[allow(unused_mut)]
            let mut $name = p[idx];
            idx += 1;
        )*
        let _ = idx; // Consume idx to avoid unused_assignments warning
    };
}

#[macro_export]
macro_rules! fetch_cov {
    ($cov:expr, $t:expr, $($name:ident),*) => {
        $(
            let $name = match $cov.get_covariate(stringify!($name)) {
                Some(cov) => cov.interpolate($t).unwrap(),
                None => panic!("Covariate {} not found", stringify!($name)),
            };

        )*
    };
}

#[macro_export]
macro_rules! lag {
    ($($k:expr => $v:expr),* $(,)?) => {{
        core::convert::From::from([$(($k, $v),)*])
    }};
}

#[macro_export]
macro_rules! fa {
    ($($k:expr => $v:expr),* $(,)?) => {{
        core::convert::From::from([$(($k, $v),)*])
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fetch_params_macro() {
        // Test basic parameter fetching
        let params = vec![1.0, 2.5, 3.7];

        fetch_params!(params, ka, ke, v);

        assert_eq!(ka, 1.0);
        assert_eq!(ke, 2.5);
        assert_eq!(v, 3.7);
    }
}
