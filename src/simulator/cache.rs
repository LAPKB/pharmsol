use crate::data::Subject;
use crate::simulator::likelihood::SubjectPredictions;
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct Cache {
    map: DashMap<CacheKey, SubjectPredictions>,
}

impl Cache {
    pub fn new(capacity: usize) -> Self {
        Cache {
            map: DashMap::with_capacity(capacity),
        }
    }

    pub fn get_entry(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
    ) -> Option<SubjectPredictions> {
        let cache_key = CacheKey {
            subject: SubjectHash::new(subject),
            support_point: SupportPointHash::new(support_point),
        };
        self.map.get(&cache_key).map(|entry| entry.clone())
    }

    pub fn insert_entry(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        predictions: SubjectPredictions,
    ) {
        let cache_key = CacheKey {
            subject: SubjectHash::new(subject),
            support_point: SupportPointHash::new(support_point),
        };
        self.map.insert(cache_key, predictions);
    }

    // Optional: Add more methods, like clearing the cache, resizing, etc.
    pub fn clear(&self) {
        self.map.clear();
    }

    pub fn resize(&mut self, new_size: usize) {
        self.map = DashMap::with_capacity(new_size);
    }
}

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
