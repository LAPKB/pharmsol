use crate::data::Subject;
use crate::simulator::likelihood::SubjectPredictions;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const CACHE_SIZE: usize = 10000;

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
        // Hash each element in the Vec
        for value in support_point {
            value.to_bits().hash(&mut hasher);
        }
        // Get the resulting hash
        SupportPointHash(hasher.finish())
    }
}

lazy_static! {
    static ref CACHE: DashMap<CacheKey, SubjectPredictions> = DashMap::with_capacity(CACHE_SIZE);
}

pub(crate) fn get_entry(subject: &Subject, support_point: &Vec<f64>) -> Option<SubjectPredictions> {
    let cache_key = CacheKey {
        subject: SubjectHash::new(subject),
        support_point: SupportPointHash::new(support_point),
    };

    // Check if the key already exists
    CACHE
        .get(&cache_key)
        .map(|existing_entry| existing_entry.clone())
}

pub(crate) fn insert_entry(
    subject: &Subject,
    support_point: &Vec<f64>,
    predictions: SubjectPredictions,
) {
    let cache_key = CacheKey {
        subject: SubjectHash::new(subject),
        support_point: SupportPointHash::new(support_point),
    };

    // Insert the new entry
    CACHE.insert(cache_key, predictions);
}
