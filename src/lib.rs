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

pub mod prelude {
    pub mod data {
        pub use crate::data::{
            error_model::ErrorModels, parser::read_pmetrics, Covariates, Data, Event, Occasion,
            Subject,
        };
    }
    pub mod simulator {
        pub use crate::simulator::{
            equation,
            equation::Equation,
            likelihood::{psi, PopulationPredictions, Prediction, SubjectPredictions},
        };
    }
    pub mod models {
        pub use crate::simulator::equation::analytical::one_compartment;
        pub use crate::simulator::equation::analytical::one_compartment_with_absorption;
        pub use crate::simulator::equation::analytical::three_compartments;
        pub use crate::simulator::equation::analytical::three_compartments_with_absorption;
        pub use crate::simulator::equation::analytical::two_compartments;
        pub use crate::simulator::equation::analytical::two_compartments_with_absorption;
    }

    //extension traits
    pub use crate::data::builder::SubjectBuilderExt;
    pub use crate::data::Interpolation::*;
    pub use crate::data::*;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
