use crate::data::Subject;
use crate::simulator::likelihood::SubjectPredictions;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of entries in the cache
const CACHE_SIZE: usize = 10000;

/// Percentage of entries to evict when the cache is full
const EVICTION_PERCENTAGE: f64 = 0.10; // 10%
#[derive(Clone, Debug, PartialEq, Hash)]
struct CacheKey {
    subject: SubjectHash,
    support_point: SupportPointHash,
}

impl Eq for CacheKey {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SubjectHash(u64);

impl SubjectHash {
    fn new(subject: &Subject) -> Self {
        SubjectHash(subject.hash())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SupportPointHash(u64);

impl SupportPointHash {
    fn new(support_point: &Vec<f64>) -> Self {
        let mut hasher = DefaultHasher::new();
        for value in support_point {
            value.to_bits().hash(&mut hasher);
        }
        SupportPointHash(hasher.finish())
    }
}

#[derive(Clone)]
struct CacheEntry {
    predictions: SubjectPredictions,
    timestamp: u64,
}

lazy_static! {
    static ref CACHE: DashMap<CacheKey, CacheEntry> = DashMap::with_capacity(CACHE_SIZE);
    static ref CURRENT_SIZE: AtomicUsize = AtomicUsize::new(0);
}

/// Get the current timestamp in seconds since the Unix epoch
fn current_timestamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

/// Evict entries from the cache if needed to make room for new entries
fn evict_if_needed() {
    

    let current_size = CURRENT_SIZE.load(Ordering::SeqCst);
    if current_size > CACHE_SIZE {
        let num_to_evict = ((current_size as f64) * EVICTION_PERCENTAGE).ceil() as usize;

        let mut oldest_entries = CACHE.iter()
            .map(|entry| (entry.key().clone(), entry.value().timestamp))
            .collect::<Vec<_>>();

        oldest_entries.sort_by_key(|&(_, timestamp)| timestamp);

        for (key, _) in oldest_entries.iter().take(num_to_evict) {
            if CACHE.remove(key).is_some() {
                CURRENT_SIZE.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

/// Get the predictions for a subject and support point from the cache
pub(crate) fn get_entry(subject: &Subject, support_point: &Vec<f64>) -> Option<SubjectPredictions> {
    let cache_key = CacheKey {
        subject: SubjectHash::new(subject),
        support_point: SupportPointHash::new(support_point),
    };

    if let Some(existing_entry) = CACHE.get(&cache_key) {
        // Update timestamp
        let mut entry = existing_entry.clone();
        entry.timestamp = current_timestamp();
        CACHE.insert(cache_key.clone(), entry);
        return Some(existing_entry.predictions.clone());
    }

    None
}

/// Insert the predictions for a subject and support point into the cache
pub(crate) fn insert_entry(
    subject: &Subject,
    support_point: &Vec<f64>,
    predictions: SubjectPredictions,
) {
    let cache_key = CacheKey {
        subject: SubjectHash::new(subject),
        support_point: SupportPointHash::new(support_point),
    };

    let cache_entry = CacheEntry {
        predictions,
        timestamp: current_timestamp(),
    };

    CACHE.insert(cache_key.clone(), cache_entry);
    let old_size = CURRENT_SIZE.fetch_add(1, Ordering::SeqCst);

    if old_size >= CACHE_SIZE {
        evict_if_needed();
    }
}
