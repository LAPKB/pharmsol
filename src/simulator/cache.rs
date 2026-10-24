//! Cache mechanisms for [Equation](crate::Equation)s
//!
//! This module provides lightweight cache wrappers that can be embedded
//! directly in equation structs ([`ODE`], [`Analytical`], [`SDE`]).
//! Each equation instance owns a cache by default; cloning the equation
//! produces a shallow clone that shares the same cache data.
//!
//! # Example
//! ```ignore
//! use pharmsol::*;
//!
//! // Caching is enabled by default (100,000 entries):
//! let ode = ODE::new(diffeq, lag, fa, init, out);
//!
//! // Custom cache capacity:
//! let ode = ODE::new(diffeq, lag, fa, init, out).with_cache_capacity(50_000);
//!
//! // Disable caching:
//! let ode = ODE::new(diffeq, lag, fa, init, out).disable_cache();
//! ```

use std::hash::Hash;
use std::{fmt, sync::Arc};

use quick_cache::sync::Cache;

use crate::simulator::likelihood::SubjectPredictions;

/// Default maximum number of entries per cache.
pub const DEFAULT_CACHE_SIZE: usize = 100_000;

/// Cache key: (subject hash, parameters hash combined with solver settings)
pub(crate) type PredictionKey = (u64, u64);

/// Cache key for SDE: (subject hash combined with seed and particle count,
/// parameters hash, error-model hash)
pub(crate) type SdeKey = (u64, u64, u64);

/// Thread-safe bounded cache of simulation results.
///
/// `Clone` produces a shallow clone that shares the same underlying cache data,
/// so cloned equations share cache hits.
pub struct SimulationCache<K, V>(Arc<Cache<K, V>>);

/// Subject predictions of [`ODE`](crate::ODE) and
/// [`Analytical`](crate::simulator::equation::Analytical) models.
pub type PredictionCache = SimulationCache<PredictionKey, Arc<SubjectPredictions>>;

/// Particle-filter log-likelihoods of [`SDE`](crate::SDE) models.
pub type SdeLikelihoodCache = SimulationCache<SdeKey, f64>;

impl<K: Eq + Hash + Clone, V: Clone> SimulationCache<K, V> {
    /// Create a cache with a given maximum number of entries.
    pub fn new(size: usize) -> Self {
        Self(Arc::new(Cache::new(size)))
    }

    /// Look up a cached value.
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        self.0.get(key)
    }

    /// Insert a value into the cache.
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        self.0.insert(key, value);
    }

    /// Return the cached value, or compute and cache it.
    ///
    /// Concurrent callers missing on the same key wait for one computation.
    /// Errors are returned and not cached.
    pub fn get_or_try_insert<E>(
        &self,
        key: K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        self.0.get_or_insert_with(&key, compute)
    }

    /// Remove all entries from the cache.
    pub fn invalidate_all(&self) {
        self.0.clear();
    }

    /// Return the number of entries currently in the cache.
    pub fn entry_count(&self) -> usize {
        self.0.len()
    }
}

impl<K, V> Clone for SimulationCache<K, V> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<K: Eq + Hash + Clone, V: Clone> fmt::Debug for SimulationCache<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimulationCache")
            .field("entry_count", &self.entry_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_cache_miss_returns_none() {
        let cache = PredictionCache::new(10);
        assert!(cache.get(&(1, 2)).is_none());
    }

    #[test]
    fn prediction_cache_hit_returns_value() {
        let cache = PredictionCache::new(10);
        let key: PredictionKey = (42, 99);
        let preds = Arc::new(SubjectPredictions::default());
        cache.insert(key, preds);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn prediction_cache_entry_count() {
        let cache = PredictionCache::new(10);
        assert_eq!(cache.entry_count(), 0);
        cache.insert((1, 1), Arc::new(SubjectPredictions::default()));
        cache.insert((2, 2), Arc::new(SubjectPredictions::default()));
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn prediction_cache_invalidate_all_clears_entries() {
        let cache = PredictionCache::new(10);
        cache.insert((1, 1), Arc::new(SubjectPredictions::default()));
        cache.insert((2, 2), Arc::new(SubjectPredictions::default()));
        assert_eq!(cache.entry_count(), 2);

        cache.invalidate_all();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&(1, 1)).is_none());
    }

    #[test]
    fn prediction_cache_overwrite_same_key() {
        let cache = PredictionCache::new(10);
        let key: PredictionKey = (1, 1);
        cache.insert(key, Arc::new(SubjectPredictions::default()));
        cache.insert(key, Arc::new(SubjectPredictions::default()));
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn prediction_cache_clone_shares_data() {
        let cache = PredictionCache::new(10);
        cache.insert((1, 1), Arc::new(SubjectPredictions::default()));
        let clone = cache.clone();
        // Clone sees existing entry
        assert!(clone.get(&(1, 1)).is_some());
        // Insert via clone is visible through original
        clone.insert((2, 2), Arc::new(SubjectPredictions::default()));
        assert!(cache.get(&(2, 2)).is_some());
    }

    #[test]
    fn prediction_cache_debug_format() {
        let cache = PredictionCache::new(10);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("SimulationCache"));
        assert!(dbg.contains("entry_count"));
    }

    #[test]
    fn sde_cache_miss_returns_none() {
        let cache = SdeLikelihoodCache::new(10);
        assert!(cache.get(&(1, 2, 3)).is_none());
    }

    #[test]
    fn sde_cache_hit_returns_value() {
        let cache = SdeLikelihoodCache::new(10);
        let key: SdeKey = (10, 20, 30);
        cache.insert(key, -42.5);
        assert_eq!(cache.get(&key), Some(-42.5));
    }

    #[test]
    fn sde_cache_entry_count() {
        let cache = SdeLikelihoodCache::new(10);
        cache.insert((1, 1, 1), 0.0);
        cache.insert((2, 2, 2), 1.0);
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn sde_cache_invalidate_all_clears_entries() {
        let cache = SdeLikelihoodCache::new(10);
        cache.insert((1, 1, 1), 0.0);
        cache.insert((2, 2, 2), 1.0);
        assert_eq!(cache.entry_count(), 2);

        cache.invalidate_all();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&(1, 1, 1)).is_none());
    }

    #[test]
    fn sde_cache_overwrite_same_key() {
        let cache = SdeLikelihoodCache::new(10);
        let key: SdeKey = (1, 1, 1);
        cache.insert(key, 1.0);
        cache.insert(key, 2.0);
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.get(&key), Some(2.0));
    }

    #[test]
    fn sde_cache_clone_shares_data() {
        let cache = SdeLikelihoodCache::new(10);
        cache.insert((1, 1, 1), 5.0);
        let clone = cache.clone();
        assert_eq!(clone.get(&(1, 1, 1)), Some(5.0));
        clone.insert((2, 2, 2), 10.0);
        assert_eq!(cache.get(&(2, 2, 2)), Some(10.0));
    }

    #[test]
    fn sde_cache_debug_format() {
        let cache = SdeLikelihoodCache::new(10);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("SimulationCache"));
        assert!(dbg.contains("entry_count"));
    }
}

#[cfg(test)]
mod get_or_try_insert_tests {
    use super::*;

    #[test]
    fn computes_once_and_does_not_cache_errors() {
        let cache = SdeLikelihoodCache::new(10);
        let failed: Result<f64, &str> = cache.get_or_try_insert((1, 1, 1), || Err("boom"));
        assert!(failed.is_err());
        assert_eq!(cache.entry_count(), 0);

        let first: Result<f64, &str> = cache.get_or_try_insert((1, 1, 1), || Ok(3.0));
        let second: Result<f64, &str> =
            cache.get_or_try_insert((1, 1, 1), || panic!("should be cached"));
        assert_eq!(first, Ok(3.0));
        assert_eq!(second, Ok(3.0));
    }
}
