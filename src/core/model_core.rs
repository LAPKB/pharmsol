use crate::simulator::cache::{BoundErrorModelCache, DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE};
use crate::simulator::Neqs;
use crate::ValidatedModelMetadata;

/// Shared model infrastructure: dimensions, metadata, and cache.
///
/// Each backend (ODE, Analytical, SDE) wraps a `ModelCore` with its
/// backend-specific fields (closure functions, solver config, etc.).
///
/// `C` is the prediction-cache type: [`PredictionCache`] for deterministic
/// backends, [`SdeLikelihoodCache`] for stochastic backends.
#[derive(Clone, Debug)]
pub struct ModelCore<C> {
    dims: Neqs,
    metadata: Option<ValidatedModelMetadata>,
    cache: Option<C>,
    error_model_cache: Option<BoundErrorModelCache>,
}

impl<C: Clone> ModelCore<C> {
    /// Create a new `ModelCore` with default dimensions (all 5) and
    /// an optional prediction cache.
    pub fn new(cache: Option<C>) -> Self {
        Self {
            dims: Neqs::default(),
            metadata: None,
            cache,
            error_model_cache: Some(BoundErrorModelCache::new(
                DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
            )),
        }
    }

    // ── Dimensions ──────────────────────────────────────────────────────

    /// Current dimension configuration.
    pub fn dims(&self) -> Neqs {
        self.dims
    }

    /// Number of state variables.
    pub fn nstates(&self) -> usize {
        self.dims.nstates
    }

    /// Number of drug input routes.
    pub fn ndrugs(&self) -> usize {
        self.dims.ndrugs
    }

    /// Number of output equations.
    pub fn nout(&self) -> usize {
        self.dims.nout
    }

    /// Set the number of state variables. Invalidates metadata.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.dims.nstates = nstates;
        self.invalidate();
        self
    }

    /// Set the number of drug inputs. Invalidates metadata.
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.dims.ndrugs = ndrugs;
        self.invalidate();
        self
    }

    /// Set the number of output equations. Invalidates metadata.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.dims.nout = nout;
        self.invalidate();
        self
    }

    // ── Metadata ────────────────────────────────────────────────────────

    /// Attached validated metadata, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    /// Attach validated metadata. The caller is responsible for
    /// dimension validation and backend-specific error handling.
    pub fn set_metadata(&mut self, metadata: ValidatedModelMetadata) {
        self.metadata = Some(metadata);
    }

    // ── Caches ──────────────────────────────────────────────────────────

    /// Prediction cache, if enabled.
    pub fn cache(&self) -> Option<&C> {
        self.cache.as_ref()
    }

    /// Bound error-model cache, if enabled.
    pub fn error_model_cache(&self) -> Option<&BoundErrorModelCache> {
        self.error_model_cache.as_ref()
    }

    /// Set the prediction cache capacity. Replaces any existing cache.
    pub fn with_cache_capacity(mut self, cache: C) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Disable prediction caching.
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self.error_model_cache = None;
        self
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        // Prediction cache clearing is type-specific and handled by
        // the backend's Caching impl.
        if let Some(cache) = &self.error_model_cache {
            cache.invalidate_all();
        }
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn invalidate(&mut self) {
        self.metadata = None;
        self.error_model_cache = Some(BoundErrorModelCache::new(
            DEFAULT_BOUND_ERROR_MODEL_CACHE_SIZE,
        ));
    }
}
