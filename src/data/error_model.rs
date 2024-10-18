use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The error model is used to determine the standard deviation of the observations
///
/// Two types of error models are currently supported: proportional ([ErrorType::Prop]) and additive ([ErrorType::Add])
///
/// They are defined as follows:
/// ```text
/// Proportional: SD = (c0 + c1 * y + c2 * y^2 + c3 * y^3) * gamma)
/// Additive    : SD = sqrt((c0 + c1 * y + c2 * y^2 + c3 * y^3)^2 + gl^2)
/// ```
///
///
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ErrorModel {
    Additive(f64),
    Proportional(f64),
}

impl ErrorModel {
    // A method to get the value from the enum
    pub fn get_scalar(&self) -> f64 {
        match self {
            ErrorModel::Additive(value) => *value,
            ErrorModel::Proportional(value) => *value,
        }
    }
    pub fn set_scalar(&mut self, new_value: f64) {
        match self {
            ErrorModel::Additive(ref mut value) => *value = new_value,
            ErrorModel::Proportional(ref mut value) => *value = new_value,
        }
    }
}

/// The error polynomial represents the coefficients of the polynomial that is used to determine the analytical noise of the observations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AssayPolynomial {
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl AssayPolynomial {
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    pub fn get_polynomial(&self) -> (f64, f64, f64, f64) {
        (self.c0, self.c1, self.c2, self.c3)
    }
}

impl Default for AssayPolynomial {
    fn default() -> Self {
        Self {
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }
}

/// Stores the (general) assay polynomial for each output equation
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorMap {
    map: HashMap<usize, AssayPolynomial>,
}

impl ErrorMap {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
    pub fn insert(&mut self, key: usize, value: AssayPolynomial) {
        self.map.insert(key, value);
    }
    pub fn get(&self, key: &usize) -> Option<&AssayPolynomial> {
        self.map.get(&key)
    }
}

impl Default for ErrorMap {
    fn default() -> Self {
        Self::new()
    }
}
