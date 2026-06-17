//! Solver backends (ODE, Analytical, SDE) and their shared infrastructure.
//!
//! For the user-facing simulation API, see [`crate::core::Simulate`].
//! For the solver interface that backend authors implement, see [`crate::core::Solver`].

pub mod analytical;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use ode::*;
pub use pharmsol_dsl::{AnalyticalKernel, ModelKind};
pub use sde::*;

// Re-export metadata types for convenience (canonical home is crate::core::metadata)
pub use crate::core::metadata::{
    Covariate, NameDomain, Route, RouteInputPolicy, RouteKind, ValidatedModelMetadata,
};

/// Hash parameter vectors to a u64 for cache key generation.
#[inline(always)]
pub(crate) fn parameters_hash(parameters: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    for &value in parameters {
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}
