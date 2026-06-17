use crate::simulator::cache::{BoundErrorModelCache, PredictionCache};

/// Cache management for simulation results.
///
/// Implementors own optional prediction and error-model caches.
/// The `Clone` impl typically produces shallow copies that share cache data.
pub trait Caching: Sized {
    /// Access the prediction cache, if enabled.
    fn prediction_cache(&self) -> Option<&PredictionCache>;

    /// Access the bound error-model cache, if enabled.
    fn error_model_cache(&self) -> Option<&BoundErrorModelCache>;

    /// Set the prediction cache capacity. Replaces any existing cache.
    fn with_cache_capacity(self, size: u64) -> Self;

    /// Disable prediction caching entirely.
    fn without_cache(self) -> Self;

    /// Clear all cached entries (prediction + error-model).
    fn clear_cache(&self);
}

// We intentionally do NOT put bind_error_models here because it needs ModelInfo
// metadata. It lives as a free function in `simulate` instead.
