//! Cache mechanisms for [Equation]s
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
//! let ode = ODE::new(diffeq, lag, fa, init, out).enable_cache_with_capacity(50_000);
//!
//! // Disable caching:
//! let ode = ODE::new(diffeq, lag, fa, init, out).disable_cache();
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
/// Note that the use of a cache could be counterproductive for SDEs, as this removes the
/// stochastic nature of the likelihood evaluation. However, it can be useful for
/// producing a deterministic likelihood for an otherwise stochastic process.
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
        let preds = SubjectPredictions::default();
        cache.insert(key, preds.clone());
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn prediction_cache_entry_count() {
        let cache = PredictionCache::new(10);
        assert_eq!(cache.entry_count(), 0);
        cache.insert((1, 1), SubjectPredictions::default());
        cache.insert((2, 2), SubjectPredictions::default());
        // moka may need a short sync before entry_count updates
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn prediction_cache_invalidate_all_clears_entries() {
        let cache = PredictionCache::new(10);
        cache.insert((1, 1), SubjectPredictions::default());
        cache.insert((2, 2), SubjectPredictions::default());
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 2);

        cache.invalidate_all();
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&(1, 1)).is_none());
    }

    #[test]
    fn prediction_cache_overwrite_same_key() {
        let cache = PredictionCache::new(10);
        let key: PredictionKey = (1, 1);
        cache.insert(key, SubjectPredictions::default());
        cache.insert(key, SubjectPredictions::default());
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn prediction_cache_clone_shares_data() {
        let cache = PredictionCache::new(10);
        cache.insert((1, 1), SubjectPredictions::default());
        let clone = cache.clone();
        // Clone sees existing entry
        assert!(clone.get(&(1, 1)).is_some());
        // Insert via clone is visible through original
        clone.insert((2, 2), SubjectPredictions::default());
        assert!(cache.get(&(2, 2)).is_some());
    }

    #[test]
    fn prediction_cache_debug_format() {
        let cache = PredictionCache::new(10);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("PredictionCache"));
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
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn sde_cache_invalidate_all_clears_entries() {
        let cache = SdeLikelihoodCache::new(10);
        cache.insert((1, 1, 1), 0.0);
        cache.insert((2, 2, 2), 1.0);
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 2);

        cache.invalidate_all();
        cache.0.run_pending_tasks();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&(1, 1, 1)).is_none());
    }

    #[test]
    fn sde_cache_overwrite_same_key() {
        let cache = SdeLikelihoodCache::new(10);
        let key: SdeKey = (1, 1, 1);
        cache.insert(key, 1.0);
        cache.insert(key, 2.0);
        cache.0.run_pending_tasks();
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
        assert!(dbg.contains("SdeLikelihoodCache"));
        assert!(dbg.contains("entry_count"));
    }
}
