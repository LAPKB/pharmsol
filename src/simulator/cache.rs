//! Cache mechanisms for deterministic [`Equation`](crate::Equation) models.
//!
//! This module provides lightweight cache wrappers that can be embedded
//! directly in [`ODE`](crate::ODE) and
//! [`Analytical`](crate::simulator::equation::Analytical) models.
//! Each deterministic equation instance owns a cache by default; cloning the equation
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

use std::{fmt, sync::Arc};

use quick_cache::sync::Cache;

use crate::simulator::prediction::SubjectPredictions;

/// Default maximum number of entries per cache.
pub const DEFAULT_CACHE_SIZE: usize = 100_000;

/// Cache key: (subject_hash, parameters_hash)
pub(crate) type PredictionKey = (u64, u64);

/// Thread-safe bounded cache for subject predictions.
///
/// Used by [`ODE`](crate::ODE) and [`Analytical`](crate::simulator::equation::Analytical)
/// to avoid recomputing predictions for the same (subject, parameters) pair.
///
/// `Clone` produces a shallow clone that shares the same underlying cache data,
/// so cloned equations share cache hits.
#[derive(Clone)]
pub struct PredictionCache {
    inner: Arc<Cache<PredictionKey, SubjectPredictions>>,
    capacity: usize,
}

impl PredictionCache {
    /// Create a new prediction cache with a given maximum number of entries.
    pub fn new(size: usize) -> Self {
        Self {
            inner: Arc::new(Cache::new(size)),
            capacity: size,
        }
    }

    /// Create an empty cache with the same capacity but no shared entries.
    pub(crate) fn detached(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Look up a cached prediction.
    #[inline]
    pub fn get(&self, key: &PredictionKey) -> Option<SubjectPredictions> {
        self.inner.get(key)
    }

    /// Insert a prediction into the cache.
    #[inline]
    pub fn insert(&self, key: PredictionKey, value: SubjectPredictions) {
        self.inner.insert(key, value);
    }

    /// Remove all entries from the cache.
    pub fn invalidate_all(&self) {
        self.inner.clear();
    }

    /// Return the number of entries currently in the cache.
    pub fn entry_count(&self) -> usize {
        self.inner.len()
    }
}

impl fmt::Debug for PredictionCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredictionCache")
            .field("entry_count", &self.entry_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Censor, ErrorPoly, Subject, SubjectBuilderExt};

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
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn prediction_cache_invalidate_all_clears_entries() {
        let cache = PredictionCache::new(10);
        cache.insert((1, 1), SubjectPredictions::default());
        cache.insert((2, 2), SubjectPredictions::default());
        assert_eq!(cache.entry_count(), 2);

        cache.invalidate_all();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&(1, 1)).is_none());
    }

    #[test]
    fn prediction_cache_overwrite_same_key() {
        let cache = PredictionCache::new(10);
        let key: PredictionKey = (1, 1);
        cache.insert(key, SubjectPredictions::default());
        cache.insert(key, SubjectPredictions::default());
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
    fn observation_metadata_changes_do_not_reuse_cached_predictions() {
        let cache = PredictionCache::new(10);
        let baseline = Subject::builder("cache-subject")
            .observation_with_error(
                1.0,
                5.0,
                "cp",
                ErrorPoly::new(0.1, 0.2, 0.3, 0.4),
                Censor::None,
            )
            .build();
        let changed = Subject::builder("cache-subject")
            .observation_with_error(
                1.0,
                5.0,
                "cp",
                ErrorPoly::new(0.1, 0.2, 0.3, 1.4),
                Censor::None,
            )
            .build();
        let baseline_key = (baseline.hash(), 99);
        let changed_key = (changed.hash(), 99);

        cache.insert(baseline_key, SubjectPredictions::default());

        assert!(cache.get(&baseline_key).is_some());
        assert!(cache.get(&changed_key).is_none());
    }

    #[test]
    fn prediction_cache_debug_format() {
        let cache = PredictionCache::new(10);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("PredictionCache"));
        assert!(dbg.contains("entry_count"));
    }
}
