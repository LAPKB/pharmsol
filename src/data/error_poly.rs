//! The per-observation assay error polynomial.
//!
//! [`ErrorPoly`] is plain data: the coefficients of an assay's standard
//! deviation as a function of the observed value. Observations carry one when
//! the data set provides it (the `C0`..`C3` columns of a Pmetrics file), and
//! predictions pass it on unchanged. pharmsol does not evaluate it; error
//! models in downstream crates do.

use serde::{Deserialize, Serialize};

/// Error polynomial coefficients of an observation's assay.
///
/// This struct holds the coefficients for a polynomial used to model
/// the error in pharmacometric analyses. It represents the error associated with quantification
/// of e.g. the drug concentration in a biological sample, such as blood or plasma.
/// More simply, it is the error associated with the observed value.
/// The polynomial is defined as:
///
/// ```text
/// error = c0 + c1 * observation + c2 * observation^2 + c3 * observation^3
/// ```
///
/// where `c0`, `c1`, `c2`, and `c3` are the coefficients of the polynomial.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct ErrorPoly {
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

/// Leading coefficients, lowest degree first; the rest are zero.
///
/// ```rust
/// use pharmsol::ErrorPoly;
///
/// assert_eq!(ErrorPoly::from([0.0, 0.1]), ErrorPoly::new(0.0, 0.1, 0.0, 0.0));
/// ```
impl<const N: usize> From<[f64; N]> for ErrorPoly {
    fn from(coefficients: [f64; N]) -> Self {
        const {
            assert!(
                N >= 1 && N <= 4,
                "an error polynomial has one to four coefficients"
            )
        };
        let mut c = [0.0; 4];
        c[..N].copy_from_slice(&coefficients);
        Self::new(c[0], c[1], c[2], c[3])
    }
}

impl ErrorPoly {
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Get the coefficients of the error polynomial
    pub fn coefficients(&self) -> (f64, f64, f64, f64) {
        (self.c0, self.c1, self.c2, self.c3)
    }

    pub fn c0(&self) -> f64 {
        self.c0
    }
    pub fn c1(&self) -> f64 {
        self.c1
    }
    pub fn c2(&self) -> f64 {
        self.c2
    }
    pub fn c3(&self) -> f64 {
        self.c3
    }

    /// Set the coefficients of the error polynomial
    pub fn set_coefficients(&mut self, c0: f64, c1: f64, c2: f64, c3: f64) {
        self.c0 = c0;
        self.c1 = c1;
        self.c2 = c2;
        self.c3 = c3;
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorPoly;

    #[test]
    fn short_arrays_fill_the_leading_coefficients() {
        assert_eq!(ErrorPoly::from([1.0]), ErrorPoly::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(
            ErrorPoly::from([1.0, 2.0, 3.0, 4.0]).coefficients(),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn coefficients_can_be_read_and_replaced() {
        let mut poly = ErrorPoly::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            (poly.c0(), poly.c1(), poly.c2(), poly.c3()),
            (1.0, 2.0, 3.0, 4.0)
        );
        poly.set_coefficients(5.0, 6.0, 7.0, 8.0);
        assert_eq!(poly.coefficients(), (5.0, 6.0, 7.0, 8.0));
    }
}
