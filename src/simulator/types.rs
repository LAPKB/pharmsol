use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct Theta {
    matrix: Array2<f64>,
    parameters: Vec<String>,
}

impl Theta {
    pub fn new(matrix: Array2<f64>, parameters: Vec<String>) -> Self {
        Self { matrix, parameters }
    }

    pub fn len(&self) -> usize {
        self.matrix.nrows()
    }

    pub fn get(&self, i: usize) -> SupportPoint {
        let mut map = BTreeMap::new();
        for (j, parameter) in self.parameters.iter().enumerate() {
            map.insert(parameter.clone(), self.matrix[[i, j]]);
        }
        SupportPoint::new(map)
    }
}

#[derive(Clone, Default, Debug)]
pub struct SupportPoint {
    map: BTreeMap<String, f64>,
}
#[macro_export]
macro_rules! support_point {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = BTreeMap::new();
        $(
            map.insert($key.to_string(), $value);
        )*
        SupportPoint::new(map)
    }};
}
#[macro_export]
macro_rules! fetch_params {
    ($p:expr, $($param:ident),*) => {
        $(
            let $param = match $p.get(stringify!($param).to_lowercase().as_str()){
                Some(value) => value,
                None => panic!("Primary parameter {} is not defined", stringify!($param)),
            };
        )*
    };
}

impl SupportPoint {
    /// Create a new, empty support point
    pub fn new(map: BTreeMap<String, f64>) -> Self {
        Self { map }
    }

    pub fn from_vec(vec: Vec<f64>, parameters: Vec<impl Into<String>>) -> Self {
        assert!(vec.len() == parameters.len());
        let parameters: Vec<String> = parameters.into_iter().map(|s| s.into()).collect();
        let mut map = BTreeMap::new();
        for (i, parameter) in parameters.iter().enumerate() {
            map.insert(parameter.clone(), vec[i]);
        }
        Self { map }
    }

    /// Get the value of a parameter in the support point
    pub fn get(&self, key: &str) -> Option<f64> {
        self.map.get(key).copied()
    }

    /// Insert a new parameter into the support point
    pub fn insert(&mut self, key: String, value: f64) {
        self.map.insert(key, value);
    }

    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        self.map.iter().for_each(|(_, &value)| {
            let bits = value.to_bits();
            bits.hash(&mut hasher);
        });

        hasher.finish()
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut f64> {
        self.map.get_mut(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    // THIS DOES NOT GUARANTEE ORDER, MIGHT FAIL!
    pub fn to_vec(&self) -> Vec<f64> {
        self.map.values().copied().collect()
    }

    pub fn parameters(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}
