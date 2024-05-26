mod data;
mod simulator;

pub mod prelude {
    pub mod data {
        pub use crate::data::{
            error_model::{ErrorModel, ErrorType},
            parse_pmetrics::{read_pmetrics, write_pmetrics_observations},
            Covariates, Data, Event, Occasion, Subject,
        };
    }
    pub mod simulator {
        pub use crate::simulator::{
            get_population_predictions, likelihood::PopulationPredictions, Equation,
        };
    }
    pub mod models {
        pub use crate::simulator::analytical::one_compartment;
        pub use crate::simulator::analytical::one_compartment_with_absorption;
        pub use crate::simulator::analytical::two_compartments;
        pub use crate::simulator::analytical::two_compartments_with_absorption;
    }
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
                let $name = $cov.get_covariate(stringify!($name)).unwrap().interpolate($t).unwrap();
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
