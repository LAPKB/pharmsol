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
pub use crate::simulator::equation::analytical::{Analytical, AnalyticalBuilder};
pub use crate::simulator::equation::ode::ODEBuilder;
pub use crate::simulator::equation::sde::{SDEBuilder, SDE};
pub use crate::simulator::equation::{self, Missing, Provided, ODE};
pub use crate::simulator::Neqs;
pub use error::PharmsolError;
#[cfg(feature = "exa")]
pub use exa::*;
pub use nalgebra::dmatrix;
pub use std::collections::HashMap;

// Re-export derive macros
pub use pharmsol_macros::{Covariates, Params};

// Re-export paste for macro identifier concatenation
pub use paste;

// Re-export diffsol traits needed by macros
pub use diffsol::vector::VectorHost as __VectorHost;

// ============================================================================
// Traits for typed parameters and covariates
// ============================================================================

/// Trait for typed parameter structs.
///
/// Implement this trait (or use `#[derive(Params)]`) to enable
/// compile-time checked parameter access in model equations.
///
/// # Example
///
/// ```ignore
/// #[derive(Params)]
/// struct Pk {
///     ke: f64,
///     v: f64,
/// }
/// ```
pub trait Params: Sized {
    /// Returns the names of all parameter fields
    fn field_names() -> &'static [&'static str];

    /// Returns the number of parameters
    fn num_params() -> usize;

    /// Create from a slice of values (in field order)
    fn from_slice(values: &[f64]) -> Self;

    /// Convert to a vector of values (in field order)
    fn to_vec(&self) -> Vec<f64>;
}

/// Trait for typed covariate structs.
///
/// Implement this trait (or use `#[derive(Covariates)]`) to enable
/// compile-time checked covariate access with automatic interpolation.
///
/// # Example
///
/// ```ignore
/// #[derive(Covariates)]
/// struct Cov {
///     wt: f64,
///     age: f64,
/// }
/// ```
pub trait CovariateSet: Sized {
    /// Returns the names of all covariate fields
    fn field_names() -> &'static [&'static str];

    /// Returns the number of covariates
    fn num_covariates() -> usize;

    /// Create from covariates data, interpolated at time t
    fn from_covariates(cov: &Covariates, t: f64) -> Result<Self, PharmsolError>;
}

/// Unit type for models without typed covariates
impl CovariateSet for () {
    fn field_names() -> &'static [&'static str] {
        &[]
    }
    fn num_covariates() -> usize {
        0
    }
    fn from_covariates(_cov: &Covariates, _t: f64) -> Result<Self, PharmsolError> {
        Ok(())
    }
}

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
            equation::analytical::{Analytical, AnalyticalBuilder},
            equation::ode::ODEBuilder,
            equation::sde::{SDEBuilder, SDE},
            equation::Equation,
            likelihood::{log_psi, psi, PopulationPredictions, Prediction, SubjectPredictions},
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

    // Re-export traits and derive macros for ergonomic API
    pub use crate::{CovariateSet, Params};
    pub use pharmsol_macros::Covariates;
    // Note: Params derive macro is re-exported from crate root

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

    // ========================================================================
    // New ergonomic macros for typed model equations
    // ========================================================================

    /// Macro for defining differential equations with cleaner syntax.
    ///
    /// This macro reduces boilerplate by letting you specify only the variables
    /// and parameters you need.
    ///
    /// # Syntax Options
    ///
    /// ## With typed params struct (ORDER-INDEPENDENT - recommended):
    /// ```ignore
    /// diffeq!(Pk |x, dx, rateiv| {
    ///     // All fields from Pk struct are available: ke, v, etc.
    ///     dx[0] = -ke * x[0] + rateiv[0];
    /// })
    /// ```
    ///
    /// ## With positional params (order matters):
    /// ```ignore
    /// diffeq!(|x, dx, rateiv| params: [ke, v] => {
    ///     dx[0] = -ke * x[0] + rateiv[0];
    /// })
    /// ```
    ///
    /// # Available Variables to Request
    ///
    /// You can request any of these in the `|...|` list:
    /// - `x` - Current state vector (read-only)
    /// - `dx` - Derivative vector (write to this)
    /// - `t` - Current time
    /// - `rateiv` - IV infusion rates
    /// - `cov` - Covariates (use with `fetch_cov!`)
    /// - `bolus` - Bolus amounts
    /// - `p` - Raw parameter vector (if needed)
    #[macro_export]
    macro_rules! diffeq {
        // NEW: Typed params struct syntax - parameter order doesn't matter!
        // The struct type must have #[derive(Params)] and be in scope
        // Usage: diffeq!(Pk as pk |x, dx, rateiv| { dx[0] = -pk.ke * x[0]; })
        ($params_type:ident as $pk:ident |$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _dx: &mut $crate::simulator::V,
             _bolus: &$crate::simulator::V,
             _rateiv: &$crate::simulator::V,
             _cov: &$crate::Covariates|
            {
                use $crate::__VectorHost;

                // Make variables accessible through user-specified names
                $(let $var = $crate::__diffeq_var!($var, _x, _p, _t, _dx, _bolus, _rateiv, _cov);)*

                // Convert to typed struct - user accesses via $pk.field
                let $pk: $params_type = <$params_type as $crate::Params>::from_slice(_p.as_slice());

                $body
            }
        };
        // Destructuring variant: diffeq!(Pk { ke, v } |x, dx, rateiv| { ... })
        ($params_type:ident { $($field:ident),* $(,)? } |$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _dx: &mut $crate::simulator::V,
             _bolus: &$crate::simulator::V,
             _rateiv: &$crate::simulator::V,
             _cov: &$crate::Covariates|
            {
                use $crate::__VectorHost;

                // Make variables accessible through user-specified names
                $(let $var = $crate::__diffeq_var!($var, _x, _p, _t, _dx, _bolus, _rateiv, _cov);)*

                // Convert to typed struct and destructure inline
                let $params_type { $($field),*, .. } = <$params_type as $crate::Params>::from_slice(_p.as_slice());

                $body
            }
        };
        // Positional params syntax (original - order matters)
        (|$($var:ident),* $(,)?| params: [$($param:ident),* $(,)?] => $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _dx: &mut $crate::simulator::V,
             _bolus: &$crate::simulator::V,
             _rateiv: &$crate::simulator::V,
             _cov: &$crate::Covariates|
            {
                // Make variables accessible through user-specified names
                $(let $var = $crate::__diffeq_var!($var, _x, _p, _t, _dx, _bolus, _rateiv, _cov);)*

                // Extract parameters by position
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        // Variant without params extraction
        (|$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _dx: &mut $crate::simulator::V,
             _bolus: &$crate::simulator::V,
             _rateiv: &$crate::simulator::V,
             _cov: &$crate::Covariates|
            {
                // Make variables accessible through user-specified names
                $(let $var = $crate::__diffeq_var!($var, _x, _p, _t, _dx, _bolus, _rateiv, _cov);)*

                $body
            }
        };
    }

    // Helper macro to map user variable names to internal ones
    #[macro_export]
    #[doc(hidden)]
    macro_rules! __diffeq_var {
        (x, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $x
        };
        (p, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $p
        };
        (t, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $t
        };
        (dx, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $dx
        };
        (bolus, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $bolus
        };
        (rateiv, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $rateiv
        };
        (cov, $x:expr, $p:expr, $t:expr, $dx:expr, $bolus:expr, $rateiv:expr, $cov:expr) => {
            $cov
        };
    }

    /// Macro for defining output equations with cleaner syntax.
    ///
    /// # Syntax Options
    ///
    /// ## With typed params struct (ORDER-INDEPENDENT - recommended):
    /// ```ignore
    /// out!(Pk |x, y| {
    ///     // All fields from Pk struct are available: ke, v, etc.
    ///     y[0] = x[0] / v;
    /// })
    /// ```
    ///
    /// ## With positional params (order matters):
    /// ```ignore
    /// out!(|x, y| params: [ke, v] => {
    ///     y[0] = x[0] / v;
    /// })
    /// ```
    ///
    /// # Available Variables to Request
    ///
    /// - `x` - Current state vector (read-only)
    /// - `y` - Output vector (write to this)
    /// - `t` - Current time
    /// - `cov` - Covariates (use with `fetch_cov!`)
    /// - `p` - Raw parameter vector (if needed)
    #[macro_export]
    macro_rules! out {
        // NEW: Typed params struct with alias - parameter order doesn't matter!
        // Usage: out!(Pk as pk |x, y| { y[0] = x[0] / pk.v; })
        ($params_type:ident as $pk:ident |$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _y: &mut $crate::simulator::V|
            {
                use $crate::__VectorHost;

                // Make variables accessible through user-specified names
                $(let $var = $crate::__out_var!($var, _x, _p, _t, _cov, _y);)*

                // Convert to typed struct - user accesses via $pk.field
                let $pk: $params_type = <$params_type as $crate::Params>::from_slice(_p.as_slice());

                $body
            }
        };
        // Destructuring variant: out!(Pk { ke, v } |x, y| { ... })
        ($params_type:ident { $($field:ident),* $(,)? } |$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _y: &mut $crate::simulator::V|
            {
                use $crate::__VectorHost;

                // Make variables accessible through user-specified names
                $(let $var = $crate::__out_var!($var, _x, _p, _t, _cov, _y);)*

                // Convert to typed struct and destructure inline
                let $params_type { $($field),*, .. } = <$params_type as $crate::Params>::from_slice(_p.as_slice());

                $body
            }
        };
        // Positional params syntax (original - order matters)
        (|$($var:ident),* $(,)?| params: [$($param:ident),* $(,)?] => $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _y: &mut $crate::simulator::V|
            {
                // Make variables accessible through user-specified names
                $(let $var = $crate::__out_var!($var, _x, _p, _t, _cov, _y);)*

                // Extract parameters by position
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        // Variant without params extraction
        (|$($var:ident),* $(,)?| $body:block) => {
            |_x: &$crate::simulator::V,
             _p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _y: &mut $crate::simulator::V|
            {
                // Make variables accessible through user-specified names
                $(let $var = $crate::__out_var!($var, _x, _p, _t, _cov, _y);)*

                $body
            }
        };
    }

    // Helper macro to map user variable names to internal ones
    #[macro_export]
    #[doc(hidden)]
    macro_rules! __out_var {
        (x, $x:expr, $p:expr, $t:expr, $cov:expr, $y:expr) => {
            $x
        };
        (p, $x:expr, $p:expr, $t:expr, $cov:expr, $y:expr) => {
            $p
        };
        (t, $x:expr, $p:expr, $t:expr, $cov:expr, $y:expr) => {
            $t
        };
        (cov, $x:expr, $p:expr, $t:expr, $cov:expr, $y:expr) => {
            $cov
        };
        (y, $x:expr, $p:expr, $t:expr, $cov:expr, $y:expr) => {
            $y
        };
    }

    /// Macro for defining lag time functions with cleaner syntax.
    ///
    /// # Syntax
    ///
    /// ```ignore
    /// lag_fn!(|cov| params: [tlag] => {
    ///     lag![0 => tlag]
    /// })
    /// ```
    ///
    /// # Available Variables to Request
    /// - `cov` - Covariates
    /// - `t` - Current time  
    /// - `p` - Raw parameter vector (if needed)
    #[macro_export]
    macro_rules! lag_fn {
        (|$($var:ident),* $(,)?| params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                $(let $var = $crate::__lag_fa_var!($var, _p, _t, _cov);)*
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        (params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        ($body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                let _ = (&_p, &_t, &_cov);

                $body
            }
        };
    }

    // Helper macro for lag_fn and fa_fn variable mapping
    #[macro_export]
    #[doc(hidden)]
    macro_rules! __lag_fa_var {
        (p, $p:expr, $t:expr, $cov:expr) => {
            $p
        };
        (t, $p:expr, $t:expr, $cov:expr) => {
            $t
        };
        (cov, $p:expr, $t:expr, $cov:expr) => {
            $cov
        };
    }

    /// Macro for defining bioavailability functions with cleaner syntax.
    ///
    /// # Syntax
    ///
    /// ```ignore
    /// fa_fn!(|cov| params: [bio] => {
    ///     fa![0 => bio]
    /// })
    /// ```
    ///
    /// # Available Variables to Request
    /// - `cov` - Covariates
    /// - `t` - Current time
    /// - `p` - Raw parameter vector (if needed)
    #[macro_export]
    macro_rules! fa_fn {
        (|$($var:ident),* $(,)?| params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                $(let $var = $crate::__lag_fa_var!($var, _p, _t, _cov);)*
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        (params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        ($body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates| -> std::collections::HashMap<usize, $crate::simulator::T>
            {
                let _ = (&_p, &_t, &_cov);

                $body
            }
        };
    }

    /// Macro for defining initial state functions with cleaner syntax.
    ///
    /// # Syntax
    ///
    /// ```ignore
    /// init_fn!(|x, t, cov| params: [dose, v] => {
    ///     x[0] = dose / v;
    /// })
    /// ```
    ///
    /// # Available Variables to Request
    /// - `x` - State vector to initialize (write to this)
    /// - `t` - Current time
    /// - `cov` - Covariates
    /// - `p` - Raw parameter vector (if needed)
    #[macro_export]
    macro_rules! init_fn {
        (|$($var:ident),* $(,)?| params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _x: &mut $crate::simulator::V|
            {
                $(let $var = $crate::__init_var!($var, _p, _t, _cov, _x);)*
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        (params: [$($param:ident),* $(,)?] => $body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _x: &mut $crate::simulator::V|
            {
                $crate::fetch_params!(_p, $($param),*);

                $body
            }
        };
        ($body:block) => {
            |_p: &$crate::simulator::V,
             _t: $crate::simulator::T,
             _cov: &$crate::Covariates,
             _x: &mut $crate::simulator::V|
            {
                let _ = (&_p, &_t, &_cov, &_x);

                $body
            }
        };
    }

    // Helper macro for init_fn variable mapping
    #[macro_export]
    #[doc(hidden)]
    macro_rules! __init_var {
        (p, $p:expr, $t:expr, $cov:expr, $x:expr) => {
            $p
        };
        (t, $p:expr, $t:expr, $cov:expr, $x:expr) => {
            $t
        };
        (cov, $p:expr, $t:expr, $cov:expr, $x:expr) => {
            $cov
        };
        (x, $p:expr, $t:expr, $cov:expr, $x:expr) => {
            $x
        };
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
