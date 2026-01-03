pub mod data;
pub mod error;
#[cfg(feature = "exa")]
pub mod exa;
pub mod optimize;
pub mod simulator;

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
            builder::SubjectBuilderExt,
            error_model::{ErrorModel, ErrorModels, ErrorPoly},
            parser::read_pmetrics,
            Covariates, Data, Event, Interpolation, Occasion, Subject,
        };
    }

    // Direct data re-exports for convenience
    pub use crate::data::{
        builder::SubjectBuilderExt,
        error_model::{ErrorModel, ErrorModels, ErrorPoly},
        Covariates, Data, Event, Interpolation, Occasion, Subject,
    };

    // Simulator submodule for internal use and advanced users
    pub mod simulator {
        pub use crate::simulator::{
            equation,
            equation::Equation,
            likelihood::{psi, PopulationPredictions, Prediction, SubjectPredictions},
        };
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
