//! Matrix exponential via scaling and squaring with a Padé approximant.
//!
//! Follows Higham, "The Scaling and Squaring Method for the Matrix Exponential
//! Revisited" (SIAM J. Matrix Anal. Appl. 26(4), 2005): pick the lowest Padé
//! order whose backward error bound covers `||A||_1`, or scale by a power of
//! two and square back up.
//!
//! Used to propagate linear time-invariant dynamics exactly over an interval.
//! Unlike hand-derived compartment solutions it has no trouble with repeated
//! eigenvalues and needs no matrix inverse.

use nalgebra::DMatrix;

/// Padé orders and the largest `||A||_1` each one covers.
const THETA: [(usize, f64); 5] = [
    (3, 1.495_585_217_958_292e-2),
    (5, 2.539_398_330_063_23e-1),
    (7, 9.504_178_996_162_932e-1),
    (9, 2.097_847_961_257_068e0),
    (13, 5.371_920_351_148_152e0),
];

/// Padé numerator coefficients, indexed by order.
fn pade_coefficients(order: usize) -> &'static [f64] {
    match order {
        3 => &[120.0, 60.0, 12.0, 1.0],
        5 => &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        7 => &[
            17_297_280.0,
            8_648_640.0,
            1_995_840.0,
            277_200.0,
            25_200.0,
            1512.0,
            56.0,
            1.0,
        ],
        9 => &[
            17_643_225_600.0,
            8_821_612_800.0,
            2_075_673_600.0,
            302_702_400.0,
            30_270_240.0,
            2_162_160.0,
            110_880.0,
            3960.0,
            90.0,
            1.0,
        ],
        _ => &[
            64_764_752_532_480_000.0,
            32_382_376_266_240_000.0,
            7_771_770_303_897_600.0,
            1_187_353_796_428_800.0,
            129_060_195_264_000.0,
            10_559_470_521_600.0,
            670_442_572_800.0,
            33_522_128_640.0,
            1_323_241_920.0,
            40_840_800.0,
            960_960.0,
            16_380.0,
            182.0,
            1.0,
        ],
    }
}

fn one_norm(matrix: &DMatrix<f64>) -> f64 {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[(row, column)].abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

/// `exp(matrix)` for a square matrix. Returns `None` when the input is not
/// finite or the Padé solve is singular.
pub fn expm(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = matrix.nrows();
    debug_assert_eq!(n, matrix.ncols());
    if n == 0 {
        return Some(matrix.clone());
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let norm = one_norm(matrix);
    let (order, squarings) = match THETA.iter().find(|(_, theta)| norm <= *theta) {
        Some((order, _)) => (*order, 0usize),
        None => {
            let theta13 = THETA[4].1;
            let squarings = ((norm / theta13).log2().ceil().max(0.0)) as usize;
            (13, squarings)
        }
    };

    let scaled = matrix / 2f64.powi(squarings as i32);
    let (numerator, denominator) = pade(&scaled, order);
    let mut result = denominator.lu().solve(&numerator)?;
    for _ in 0..squarings {
        result = &result * &result;
    }
    Some(result)
}

/// Build the Padé numerator `U + V` and denominator `V - U`.
fn pade(matrix: &DMatrix<f64>, order: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = matrix.nrows();
    let identity = DMatrix::<f64>::identity(n, n);
    let coefficients = pade_coefficients(order);

    let squared = matrix * matrix;
    let mut powers = vec![identity.clone(), squared.clone()];
    while powers.len() * 2 <= order + 1 {
        let next = powers.last().unwrap() * &squared;
        powers.push(next);
    }

    // Odd terms carry a leading `A`, even terms do not.
    let mut odd = DMatrix::<f64>::zeros(n, n);
    let mut even = DMatrix::<f64>::zeros(n, n);
    for (index, coefficient) in coefficients.iter().enumerate() {
        let power = index / 2;
        let Some(term) = powers.get(power) else {
            continue;
        };
        if index % 2 == 0 {
            even += term * *coefficient;
        } else {
            odd += term * *coefficient;
        }
    }
    let odd = matrix * odd;

    (&even + &odd, &even - &odd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn diagonal_matches_scalar_exponentials() {
        let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![-0.5, 1.25, 0.0]));
        let result = expm(&matrix).expect("expm");
        assert_relative_eq!(result[(0, 0)], (-0.5f64).exp(), epsilon = 1e-12);
        assert_relative_eq!(result[(1, 1)], 1.25f64.exp(), epsilon = 1e-12);
        assert_relative_eq!(result[(2, 2)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result[(0, 1)], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn nilpotent_matches_truncated_series() {
        // [[0, 1], [0, 0]] squares to zero, so exp(A) = I + A exactly.
        let matrix = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let result = expm(&matrix).expect("expm");
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-14);
        assert_relative_eq!(result[(0, 1)], 1.0, epsilon = 1e-14);
        assert_relative_eq!(result[(1, 1)], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn repeated_eigenvalues_are_handled() {
        // ka == ke defeats the closed-form one-compartment absorption solution
        // but is unremarkable here.
        let k = 0.7;
        let matrix = DMatrix::from_row_slice(2, 2, &[-k, 0.0, k, -k]);
        let t = 2.0;
        let result = expm(&(matrix * t)).expect("expm");
        assert_relative_eq!(result[(0, 0)], (-k * t).exp(), epsilon = 1e-12);
        assert_relative_eq!(result[(1, 0)], k * t * (-k * t).exp(), epsilon = 1e-12);
        assert_relative_eq!(result[(1, 1)], (-k * t).exp(), epsilon = 1e-12);
    }

    #[test]
    fn large_norm_uses_scaling_and_squaring() {
        let matrix = DMatrix::from_row_slice(2, 2, &[-40.0, 0.0, 40.0, -1.0]);
        let result = expm(&matrix).expect("expm");
        assert_relative_eq!(result[(0, 0)], (-40.0f64).exp(), epsilon = 1e-14);
        // Two-compartment transfer: k/(k - l) * (exp(-l) - exp(-k)).
        let expected = 40.0 / (40.0 - 1.0) * ((-1.0f64).exp() - (-40.0f64).exp());
        assert_relative_eq!(result[(1, 0)], expected, epsilon = 1e-12);
    }

    #[test]
    fn rejects_non_finite_input() {
        let matrix = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 0.0]);
        assert!(expm(&matrix).is_none());
    }
}
