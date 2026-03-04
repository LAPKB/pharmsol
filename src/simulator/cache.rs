//! Global cache for predictions
//!
//! This module provides a configurable, concurrent LRU cache for prediction results.
//! The supports enabling/disabling caching at runtime and adjusting cache size.
//!
//! The cache can be cleared between runs, which is useful to avoid stale data and bad hits.

//!
//! # Example
//! ```ignore
//! use pharmsol::simulator::cache::{configure_cache, CacheSettings};
//!
//! // Use a smaller cache
//! configure_cache(CacheSettings::with_size(10_000));
//!
//! // Disable caching entirely
//! configure_cache(CacheSettings::disabled());
//! ```

use std::sync::{
    atomic::{AtomicBool, Ordering},
    LazyLock, RwLock,
};

use moka::sync::Cache;

use crate::{simulator::likelihood::SubjectPredictions, PharmsolError};

/// Default maximum number of entries per cache.
pub const DEFAULT_CACHE_SIZE: u64 = 100_000;

static CACHE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Settings for the prediction cache.
#[derive(Debug, Clone)]
pub struct CacheSettings {
    /// Whether caching is enabled.
    pub enabled: bool,
    /// Maximum number of entries per equation type.
    pub size: u64,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: DEFAULT_CACHE_SIZE,
        }
    }
}

impl CacheSettings {
    /// Create settings with caching disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create settings with a custom cache size.
    pub fn with_size(size: u64) -> Self {
        Self {
            enabled: true,
            size,
        }
    }
}

/// Apply new cache settings globally.
///
/// This replaces all caches with new instances of the given size
/// and updates the enabled flag. Any cached entries are discarded.
pub fn configure_cache(settings: CacheSettings) -> Result<(), PharmsolError> {
    CACHE_ENABLED.store(settings.enabled, Ordering::Relaxed);

    let size = settings.size;

    // Replace each cache with a fresh instance of the requested size.
    {
        let mut c = ana_cache_lock_write()?;
        *c = Cache::new(size);
    }
    {
        let mut c = ode_cache_lock_write()?;
        *c = Cache::new(size);
    }
    {
        let mut c = sde_cache_lock_write()?;
        *c = Cache::new(size);
    }

    Ok(())
}

/// Clear all prediction caches without changing settings.
pub fn reset_caches() -> Result<(), PharmsolError> {
    ana_cache_lock_read()?.invalidate_all();
    ode_cache_lock_read()?.invalidate_all();
    sde_cache_lock_read()?.invalidate_all();
    Ok(())
}

/// Disable caching entirely and clear all caches.
pub fn disable_cache() -> Result<(), PharmsolError> {
    CACHE_ENABLED.store(false, Ordering::Relaxed);
    reset_caches()
}

/// Enable caching (uses existing size settings).
pub fn enable_cache() {
    CACHE_ENABLED.store(true, Ordering::Relaxed);
}

/// Returns `true` if caching is currently enabled.
#[inline(always)]
pub fn cache_enabled() -> bool {
    CACHE_ENABLED.load(Ordering::Relaxed)
}

/// Get the current cache settings.
pub fn cache_settings() -> Result<CacheSettings, PharmsolError> {
    let size = ana_cache_lock_read()?.policy().max_capacity().unwrap_or(0);
    Ok(CacheSettings {
        enabled: cache_enabled(),
        size,
    })
}

// ---------------------------------------------------------------------------
// Per-equation-type caches
// ---------------------------------------------------------------------------

/// Cache key: (subject_id_hash, support_point_hash)
pub(crate) type PredictionKey = (u64, u64);

/// Cache key for SDE: (subject_id_hash, support_point_hash, error_model_hash)
pub(crate) type SdeKey = (u64, u64, u64);

// The caches use RwLock so that the hot path (read lock) allows full moka
// concurrency, while resize (write lock) is exclusive but rare.

static ANA_CACHE: LazyLock<RwLock<Cache<PredictionKey, SubjectPredictions>>> =
    LazyLock::new(|| RwLock::new(Cache::new(DEFAULT_CACHE_SIZE)));

static ODE_CACHE: LazyLock<RwLock<Cache<PredictionKey, SubjectPredictions>>> =
    LazyLock::new(|| RwLock::new(Cache::new(DEFAULT_CACHE_SIZE)));

static SDE_CACHE: LazyLock<RwLock<Cache<SdeKey, f64>>> =
    LazyLock::new(|| RwLock::new(Cache::new(DEFAULT_CACHE_SIZE)));

/// Wrapper for lock errors
fn lock_err(context: &str) -> PharmsolError {
    PharmsolError::OtherError(format!("Failed to lock {context} cache"))
}

// -- Analytical --

pub(crate) fn ana_cache_lock_read() -> Result<
    std::sync::RwLockReadGuard<'static, Cache<PredictionKey, SubjectPredictions>>,
    PharmsolError,
> {
    ANA_CACHE.read().map_err(|_| lock_err("analytical"))
}

fn ana_cache_lock_write() -> Result<
    std::sync::RwLockWriteGuard<'static, Cache<PredictionKey, SubjectPredictions>>,
    PharmsolError,
> {
    ANA_CACHE.write().map_err(|_| lock_err("analytical"))
}

// -- ODE --

pub(crate) fn ode_cache_lock_read() -> Result<
    std::sync::RwLockReadGuard<'static, Cache<PredictionKey, SubjectPredictions>>,
    PharmsolError,
> {
    ODE_CACHE.read().map_err(|_| lock_err("ODE"))
}

fn ode_cache_lock_write() -> Result<
    std::sync::RwLockWriteGuard<'static, Cache<PredictionKey, SubjectPredictions>>,
    PharmsolError,
> {
    ODE_CACHE.write().map_err(|_| lock_err("ODE"))
}

// -- SDE --

pub(crate) fn sde_cache_lock_read(
) -> Result<std::sync::RwLockReadGuard<'static, Cache<SdeKey, f64>>, PharmsolError> {
    SDE_CACHE.read().map_err(|_| lock_err("SDE"))
}

fn sde_cache_lock_write(
) -> Result<std::sync::RwLockWriteGuard<'static, Cache<SdeKey, f64>>, PharmsolError> {
    SDE_CACHE.write().map_err(|_| lock_err("SDE"))
}
