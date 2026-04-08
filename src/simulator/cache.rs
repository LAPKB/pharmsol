//! Cache mechanisms for [Equation]s
//!
//! This module provides lightweight cache wrappers that can be embedded
//! directly in equation structs ([`ODE`], [`Analytical`], [`SDE`]).
//! Each equation instance can optionally own a cache; cloning the equation
//! produces a shallow clone that shares the same cache data.
//!
//! # Example
//! ```ignore
//! use pharmsol::*;
//!
//! // No caching (default):
//! let ode = ODE::new(diffeq, lag, fa, init, out);
//!
//! // Enable caching with default size:
//! let ode = ODE::new(diffeq, lag, fa, init, out).with_default_cache();
//!
//! // Enable caching with custom size:
//! let ode = ODE::new(diffeq, lag, fa, init, out).with_cache(50_000);
//! ```

use std::fmt;

use moka::sync::Cache;

use crate::simulator::likelihood::SubjectPredictions;

/// Default maximum number of entries per cache.
pub const DEFAULT_CACHE_SIZE: u64 = 100_000;

/// Cache key: (subject_hash, support_point_hash)
pub(crate) type PredictionKey = (u64, u64);

/// Cache key for SDE: (subject_hash, support_point_hash, error_model_hash)
pub(crate) type SdeKey = (u64, u64, u64);

/// Thread-safe LRU cache for subject predictions.
///
/// Used by [`ODE`](crate::ODE) and [`Analytical`](crate::simulator::equation::Analytical)
/// to avoid recomputing predictions for the same (subject, parameters) pair.
///
/// `Clone` produces a shallow clone that shares the same underlying cache data,
/// so cloned equations share cache hits.
#[derive(Clone)]
pub struct PredictionCache(Cache<PredictionKey, SubjectPredictions>);

impl PredictionCache {
    /// Create a new prediction cache with a given maximum number of entries.
    pub fn new(size: u64) -> Self {
        Self(Cache::new(size))
    }

    /// Look up a cached prediction.
    #[inline]
    pub fn get(&self, key: &PredictionKey) -> Option<SubjectPredictions> {
        self.0.get(key)
    }

    /// Insert a prediction into the cache.
    #[inline]
    pub fn insert(&self, key: PredictionKey, value: SubjectPredictions) {
        self.0.insert(key, value);
    }

    /// Remove all entries from the cache.
    pub fn invalidate_all(&self) {
        self.0.invalidate_all();
    }

    /// Return the number of entries currently in the cache.
    pub fn entry_count(&self) -> u64 {
        self.0.entry_count()
    }
}

impl fmt::Debug for PredictionCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredictionCache")
            .field("entry_count", &self.0.entry_count())
            .finish()
    }
}

/// Cache for SDE likelihood values.
///
/// SDEs do not produce subject predictions that can be cached, but
/// the likelihood values for a given subject and parameters can still be cached.
///
/// Note that the use of a cache for SDEs
///
/// `Clone` produces a shallow clone that shares the same underlying cache data.
#[derive(Clone)]
pub struct SdeLikelihoodCache(Cache<SdeKey, f64>);

impl SdeLikelihoodCache {
    /// Create a new SDE likelihood cache with the given maximum number of entries.
    pub fn new(size: u64) -> Self {
        Self(Cache::new(size))
    }

    /// Look up a cached likelihood value.
    #[inline]
    pub fn get(&self, key: &SdeKey) -> Option<f64> {
        self.0.get(key)
    }

    /// Insert a likelihood value into the cache.
    #[inline]
    pub fn insert(&self, key: SdeKey, value: f64) {
        self.0.insert(key, value);
    }

    /// Remove all entries from the cache.
    pub fn invalidate_all(&self) {
        self.0.invalidate_all();
    }

    /// Return the number of entries currently in the cache.
    pub fn entry_count(&self) -> u64 {
        self.0.entry_count()
    }
}

impl fmt::Debug for SdeLikelihoodCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SdeLikelihoodCache")
            .field("entry_count", &self.0.entry_count())
            .finish()
    }
}
