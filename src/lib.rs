pub mod data;
pub mod simulator;

//extension traits
pub use crate::data::builder::SubjectBuilderExt;
pub use crate::data::InterpolationMethod::*;
pub use crate::data::*;

pub mod prelude {
    pub mod data {
        pub use crate::data::{
            error_model::{ErrorModel, ErrorType},
            parse_pmetrics::read_pmetrics,
            Covariates, Data, Event, Occasion, Subject,
        };
    }
    pub mod simulator {
        pub use crate::simulator::{
            likelihood::{
                get_population_predictions, pf_psi, PopulationPredictions, Prediction,
                SubjectPredictions,
            },
            Equation,
        };
    }
    pub mod models {
        pub use crate::simulator::analytical::one_compartment;
        pub use crate::simulator::analytical::one_compartment_with_absorption;
        pub use crate::simulator::analytical::two_compartments;
        pub use crate::simulator::analytical::two_compartments_with_absorption;
    }
    //extension traits
    pub use crate::data::builder::SubjectBuilderExt;
    pub use crate::data::InterpolationMethod::*;
    pub use crate::data::*;
    //traits
    pub use crate::simulator::fitting::{EstimateTheta, OptimalSupportPoint};

    #[macro_export]
    macro_rules! fetch_params {
        ($p:expr, $($name:ident),*) => {
            let p = $p;
            let mut idx = 0;
            $(
                let $name = p[idx];
                idx += 1;
            )*
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
