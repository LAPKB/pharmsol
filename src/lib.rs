// Allow the crate to reference itself as `::pharmsol` (used by proc macro output)
extern crate self as pharmsol;

pub mod data;
pub mod error;
#[cfg(feature = "exa")]
pub mod exa;
pub mod optimize;
pub mod simulator;

// Re-export the pk_model proc macro
pub use pharmsol_macros::pk_model;

//extension traits
pub use crate::data::builder::SubjectBuilderExt;
pub use crate::data::Interpolation::*;
pub use crate::data::*;
pub use crate::equation::*;
pub use crate::optimize::effect::get_e2;
pub use crate::optimize::spp::SppOptimizer;
pub use crate::simulator::equation::{self, ODE};
pub use error::PharmsolError;
#[cfg(feature = "exa")]
pub use exa::*;
pub use nalgebra::dmatrix;
pub use std::collections::HashMap;

/// Prelude module that re-exports all commonly used types and traits.
///
/// Use `use pharmsol::prelude::*;` to import everything needed for basic
/// pharmacometric modeling.
///
/// # Example
/// ```rust
/// use pharmsol::prelude::*;
///
/// let subject = Subject::builder("patient_001")
///     .bolus(0.0, 100.0, 0)
///     .observation(1.0, 10.5, 0)
///     .build();
/// ```
pub mod prelude {
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
        Covariates, Data, Event, Interpolation, Occasion, Subject,
    };

    // Simulator submodule for internal use and advanced users
    pub mod simulator {
        pub use crate::simulator::{
            equation,
            equation::Equation,
            likelihood::{
                log_likelihood_batch, log_likelihood_matrix, log_likelihood_subject,
                LikelihoodMatrixOptions, PopulationPredictions, Prediction, SubjectPredictions,
            },
        };

        // Deprecated re-exports for backward compatibility
        #[allow(deprecated)]
        pub use crate::simulator::likelihood::{log_psi, psi};
    }

    // Direct simulator re-exports for convenience
    pub use crate::simulator::{
        equation::{self, Equation},
        likelihood::{Prediction, SubjectPredictions},
    };

    // Analytical model functions
    pub use crate::simulator::equation::analytical::{
        one_compartment, one_compartment_with_absorption, three_compartments,
        three_compartments_with_absorption, two_compartments, two_compartments_with_absorption,
    };

    /// Models submodule for organized access to analytical model functions
    pub mod models {
        pub use crate::simulator::equation::analytical::{
            one_compartment, one_compartment_with_absorption, three_compartments,
            three_compartments_with_absorption, two_compartments, two_compartments_with_absorption,
        };
    }

    // Re-export macros (they are exported at crate root via #[macro_export])
    #[doc(inline)]
    pub use crate::fa;
    #[doc(inline)]
    pub use crate::fetch_cov;
    #[doc(inline)]
    pub use crate::fetch_params;
    #[doc(inline)]
    pub use crate::lag;
    #[doc(inline)]
    pub use crate::pk_model;
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

    #[test]
    fn test_pk_model_basic() {
        use crate::prelude::*;

        let subject = Subject::builder("id1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .build();

        // Clean declarative DSL — no closures needed
        let ode_dsl = pk_model! {
            params: (ke, v),
            diffeq: {
                dx[0] = -ke * x[0];
            },
            out: {
                y[0] = x[0] / v;
            },
            neqs: (1, 1),
        };

        // Manual version for comparison
        let ode_manual = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let params = vec![0.1, 50.0];
        let pred_dsl = ode_dsl.estimate_predictions(&subject, &params).unwrap();
        let pred_manual = ode_manual.estimate_predictions(&subject, &params).unwrap();

        let dsl_preds = pred_dsl.flat_predictions();
        let manual_preds = pred_manual.flat_predictions();

        assert_eq!(dsl_preds.len(), manual_preds.len());
        for (d, m) in dsl_preds.iter().zip(manual_preds.iter()) {
            assert!(
                (d - m).abs() < 1e-10,
                "Predictions differ: DSL={}, Manual={}",
                d,
                m
            );
        }
    }

    #[test]
    fn test_pk_model_with_lag() {
        use crate::prelude::*;

        let subject = Subject::builder("id1")
            .bolus(0.0, 100.0, 0)
            .observation(0.5, 0.0, 0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .build();

        let ode = pk_model! {
            params: (ka, ke, tlag, v),
            lag: { 0 => tlag },
            diffeq: {
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            out: {
                y[0] = x[1] / v;
            },
            neqs: (2, 1),
        };

        let params = vec![1.0, 0.1, 0.5, 50.0];
        let predictions = ode.estimate_predictions(&subject, &params).unwrap();
        let preds = predictions.flat_predictions();

        assert_eq!(preds.len(), 3);
        // With tlag=0.5, bolus arrives at t=0.5, so prediction at t=0.5 should be ~0
        assert!(
            preds[0].abs() < 0.01,
            "At t=0.5 (lag time), pred should be ~0"
        );
        // At t=1.0 and t=2.0, should have positive predictions
        assert!(preds[1] > 0.0, "At t=1.0, pred should be positive");
        assert!(preds[2] > 0.0, "At t=2.0, pred should be positive");
    }

    #[test]
    fn test_pk_model_with_covariates() {
        use crate::prelude::*;

        let subject = Subject::builder("id1")
            .bolus(0.0, 100.0, 0)
            .covariate("wt", 0.0, 70.0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .build();

        let ode = pk_model! {
            params: (cl, v),
            covariates: (wt),
            diffeq: {
                let ke = cl * (wt / 70.0).powf(0.75) / (v * wt / 70.0);
                dx[0] = -ke * x[0];
            },
            out: {
                let v_adj = v * wt / 70.0;
                y[0] = x[0] / v_adj;
            },
            neqs: (1, 1),
        };

        let params = vec![5.0, 50.0];
        let predictions = ode.estimate_predictions(&subject, &params).unwrap();
        let preds = predictions.flat_predictions();

        assert_eq!(preds.len(), 2);
        assert!(preds[0] > 0.0, "Should have positive predictions");
        assert!(preds[1] > 0.0, "Should have positive predictions");
        assert!(preds[0] > preds[1], "Concentrations should decrease");
    }

    #[test]
    fn test_pk_model_with_fa() {
        use crate::prelude::*;

        let subject = Subject::builder("id1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 0.0, 0)
            .build();

        // With 50% bioavailability
        let ode_half = pk_model! {
            params: (ke, v, bio),
            fa: { 0 => bio },
            diffeq: {
                dx[0] = -ke * x[0];
            },
            out: {
                y[0] = x[0] / v;
            },
            neqs: (1, 1),
        };

        // Without bioavailability adjustment (full dose)
        let ode_full = pk_model! {
            params: (ke, v, _bio),
            diffeq: {
                dx[0] = -ke * x[0];
            },
            out: {
                y[0] = x[0] / v;
            },
            neqs: (1, 1),
        };

        let params = vec![0.1, 50.0, 0.5];
        let pred_half = ode_half.estimate_predictions(&subject, &params).unwrap();
        let pred_full = ode_full.estimate_predictions(&subject, &params).unwrap();

        let half_val = pred_half.flat_predictions()[0];
        let full_val = pred_full.flat_predictions()[0];

        // With 50% bioavailability, prediction should be ~half
        assert!(
            (half_val / full_val - 0.5).abs() < 0.01,
            "50% bioavailability should produce ~half the concentration"
        );
    }

    #[test]
    fn test_pk_model_with_init() {
        use crate::prelude::*;

        // Need an event at t=0 so the ODE solver starts from t=0
        // (initial_time() is the earliest event time)
        let subject = Subject::builder("id1")
            .bolus(0.0, 0.0, 0) // zero-amount bolus to anchor t0=0
            .observation(1.0, 0.0, 0)
            .build();

        let ode = pk_model! {
            params: (ke, v, init_amount),
            init: {
                x[0] = init_amount;
            },
            diffeq: {
                dx[0] = -ke * x[0];
            },
            out: {
                y[0] = x[0] / v;
            },
            neqs: (1, 1),
        };

        let params = vec![0.1, 50.0, 100.0];
        let predictions = ode.estimate_predictions(&subject, &params).unwrap();
        let preds = predictions.flat_predictions();

        assert_eq!(preds.len(), 1);
        // Initial amount 100, ke=0.1, v=50, at t=1: C = 100*exp(-0.1)/50
        let expected = 100.0 * (-0.1_f64).exp() / 50.0;
        assert!(
            (preds[0] - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            preds[0]
        );
    }

    #[test]
    fn test_pk_model_with_infusion() {
        use crate::prelude::*;

        let subject = Subject::builder("id1")
            .infusion(0.0, 100.0, 0, 1.0) // 100mg over 1hr
            .observation(0.5, 0.0, 0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .build();

        let ode = pk_model! {
            params: (ke, v),
            diffeq: {
                dx[0] = -ke * x[0] + rateiv[0];
            },
            out: {
                y[0] = x[0] / v;
            },
            neqs: (1, 1),
        };

        let params = vec![0.1, 50.0];
        let predictions = ode.estimate_predictions(&subject, &params).unwrap();
        let preds = predictions.flat_predictions();

        assert_eq!(preds.len(), 3);
        // During infusion (t=0.5), concentration rising
        assert!(preds[0] > 0.0);
        // At end of infusion (t=1.0), peak
        assert!(preds[1] > preds[0]);
        // After infusion (t=2.0), declining
        assert!(preds[2] < preds[1]);
    }
}
