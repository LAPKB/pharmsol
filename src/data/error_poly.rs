//! Data-only transport for per-observation Pmetrics C0-C3 values.

use serde::{Deserialize, Serialize};

/// Four floating-point coefficients attached to an observation.
///
/// pharmsol transports these values verbatim and does not interpret them.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct ErrorPoly {
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl ErrorPoly {
    /// Create a coefficient tuple from C0-C3.
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Get all four coefficients.
    pub fn coefficients(&self) -> (f64, f64, f64, f64) {
        (self.c0, self.c1, self.c2, self.c3)
    }

    /// Get C0.
    pub fn c0(&self) -> f64 {
        self.c0
    }

    /// Get C1.
    pub fn c1(&self) -> f64 {
        self.c1
    }

    /// Get C2.
    pub fn c2(&self) -> f64 {
        self.c2
    }

    /// Get C3.
    pub fn c3(&self) -> f64 {
        self.c3
    }

    /// Replace all four coefficients.
    pub fn set_coefficients(&mut self, c0: f64, c1: f64, c2: f64, c3: f64) {
        self.c0 = c0;
        self.c1 = c1;
        self.c2 = c2;
        self.c3 = c3;
    }
}
