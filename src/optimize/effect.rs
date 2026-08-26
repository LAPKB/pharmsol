//! Maximum-effect (`E2`) optimization for dual-site pharmacodynamic models.
//!
//! The central entry point is [`get_e2`], which computes the maximum achievable
//! effect for a model with two binding sites. The canonical equation is solved
//! in positive `M` space through a one-dimensional Nelder-Mead optimization,
//! with the historical FINDM0-style estimate used to recover from a poor first
//! optimization result.

use std::panic::{catch_unwind, AssertUnwindSafe};

use argmin::{
    core::{CostFunction, Executor, TerminationReason, TerminationStatus},
    solver::neldermead::NelderMead,
};

const RESIDUAL_COST_TOLERANCE: f64 = 1.0e-10;
const INVALID_COST: f64 = 1.0e100;

#[derive(Debug, Clone)]
struct BestM0 {
    u: f64,
    v: f64,
    w: f64,
    h1: f64,
    h2: f64,
    xx: f64,
}

#[derive(Debug, Clone, Copy)]
struct OptimizationResult {
    xm: f64,
    cost: f64,
    converged: bool,
}

/// We'll optimize over `y = ln(M)`, so `Param = f64` (the log of `M`).
impl CostFunction for BestM0 {
    type Param = f64;
    type Output = f64;

    fn cost(&self, y: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let xm = y.exp();
        let Some(residual) = self.residual(xm) else {
            // Returning a finite penalty keeps argmin total for invalid trial
            // points while still allowing the caller to reject the result.
            return Ok(INVALID_COST);
        };

        let cost = residual * residual;
        if cost.is_finite() {
            Ok(cost)
        } else {
            Ok(INVALID_COST)
        }
    }
}

impl BestM0 {
    fn residual(&self, xm: f64) -> Option<f64> {
        if !(xm.is_finite() && xm > 0.0) {
            return None;
        }

        let term = |coefficient: f64, exponent: f64| {
            if coefficient == 0.0 {
                Some(0.0)
            } else {
                let denominator = xm.powf(exponent);
                let value = coefficient / denominator;
                value.is_finite().then_some(value)
            }
        };

        let t1 = term(self.u, self.h1)?;
        let t2 = term(self.v, self.h2)?;
        let t3 = term(self.w, self.xx)?;
        let residual = 1.0 - t1 - t2 - t3;
        residual.is_finite().then_some(residual)
    }

    /// Start and step are in log-space (`ln(M)`). Optimizer panics and missing
    /// best parameters are converted into ordinary errors so no failure can
    /// escape the public scalar API.
    fn get_best(&self, start_log: f64, step_log: f64) -> Result<OptimizationResult, BestM0Error> {
        let run = catch_unwind(AssertUnwindSafe(|| {
            if !start_log.is_finite() {
                return Err(BestM0Error::OptimizationError(
                    "optimizer start is not finite".to_string(),
                ));
            }

            let second = start_log + step_log;
            let initial_simplex = if !second.is_finite() || (second - start_log).abs() < 1.0e-12 {
                vec![start_log, start_log + 0.1_f64]
            } else {
                vec![start_log, second]
            };

            let solver = NelderMead::new(initial_simplex)
                .with_sd_tolerance(1.0e-8)
                .map_err(|error| {
                    BestM0Error::OptimizationError(format!(
                        "failed setting the Nelder-Mead tolerance: {error}"
                    ))
                })?;

            let result = Executor::new(self.clone(), solver)
                .configure(|state| state.max_iters(1000))
                .run()
                .map_err(|error| {
                    BestM0Error::OptimizationError(format!(
                        "failed running the Nelder-Mead solver: {error}"
                    ))
                })?;

            let converged = matches!(
                &result.state.termination_status,
                TerminationStatus::Terminated(TerminationReason::SolverConverged)
            );

            let Some(best_log) = result.state.best_param else {
                return Err(BestM0Error::OptimizationError(
                    "Nelder-Mead returned no best parameter".to_string(),
                ));
            };
            let xm = best_log.exp();
            if !(xm.is_finite() && xm > 0.0) {
                return Err(BestM0Error::OptimizationError(
                    "Nelder-Mead returned an invalid best parameter".to_string(),
                ));
            }

            let cost = self.residual(xm).map(|residual| residual * residual);
            let Some(cost) = cost.filter(|cost| cost.is_finite()) else {
                return Err(BestM0Error::OptimizationError(
                    "Nelder-Mead returned a non-finite residual".to_string(),
                ));
            };

            Ok(OptimizationResult {
                xm,
                cost,
                converged,
            })
        }));

        match run {
            Ok(result) => result,
            Err(_) => Err(BestM0Error::OptimizationError(
                "Nelder-Mead panicked while optimizing".to_string(),
            )),
        }
    }
}

#[derive(thiserror::Error, Debug)]
enum BestM0Error {
    #[error("optimization error: {0}")]
    OptimizationError(String),
}

/// Historical FINDM0-style estimate used to seed a second optimizer run.
fn find_m0(u_final: f64, v: f64, alpha: f64, h1: f64, h2: f64) -> f64 {
    if !(u_final.is_finite()
        && v.is_finite()
        && alpha.is_finite()
        && h1.is_finite()
        && h2.is_finite()
        && u_final >= 0.0
        && v >= 0.0
        && h1 > 0.0
        && h2 > 0.0)
    {
        return -1.0;
    }

    let noint = 1000;
    let del_a = u_final / (noint as f64);
    let mut xm = if v > 0.0 { v.powf(1.0 / h2) } else { 1.0 };
    let mut a = 0.0;
    let hh = (h1 + h2) / 2.0;

    for int in 1..=noint {
        if !(xm.is_finite() && xm > 0.0) {
            return -1.0;
        }

        let xm_h1 = xm.powf(h1);
        let xm_h2 = xm.powf(h2);
        let xm_hh = xm.powf(hh);
        let xm_h1_plus_one = xm.powf(h1 + 1.0);
        let xm_h2_plus_one = xm.powf(h2 + 1.0);
        let xm_hh_plus_one = xm.powf(hh + 1.0);
        if [
            xm_h1,
            xm_h2,
            xm_hh,
            xm_h1_plus_one,
            xm_h2_plus_one,
            xm_hh_plus_one,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value == 0.0)
        {
            return -1.0;
        }

        let top = 1.0 / xm_h1 + alpha * v / xm_hh;
        let b1 = a * h1 / xm_h1_plus_one;
        let b2 = v * h2 / xm_h2_plus_one;
        let b3 = alpha * a * v * hh / xm_hh_plus_one;
        let denominator = b1 + b2 + b3;
        if !(denominator.is_finite() && denominator != 0.0) {
            return -1.0;
        }

        let xmp = top / denominator;
        xm += xmp * del_a;
        if !(xm.is_finite() && xm > 0.0) {
            return -1.0;
        }
        a = del_a * (int as f64);
    }

    xm
}

#[inline]
fn effect_from_xm(xm: f64) -> f64 {
    if xm.is_finite() && xm >= 0.0 {
        xm / (xm + 1.0)
    } else {
        f64::NAN
    }
}

fn single_site_xm(coefficient: f64, hill: f64) -> Option<f64> {
    if coefficient == 0.0 {
        return Some(0.0);
    }
    let xm = (coefficient.ln() / hill).exp();
    (xm.is_finite() && xm > 0.0).then_some(xm)
}

fn select_lower_residual(
    first: Option<OptimizationResult>,
    second: Option<OptimizationResult>,
) -> Option<OptimizationResult> {
    [first, second]
        .into_iter()
        .flatten()
        .filter(|candidate| {
            candidate.xm.is_finite() && candidate.xm > 0.0 && candidate.cost.is_finite()
        })
        .min_by(|left, right| {
            left.cost
                .partial_cmp(&right.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Computes the effect metric for a dual-site pharmacodynamic model.
///
/// The canonical equation solved by this function is
///
/// ```text
/// r(M) = 1 - u/M^h1 - v/M^h2
///        - (alpha*u*v)/M^((h1+h2)/2)
/// ```
///
/// The returned value is `M / (1 + M)`. The interaction coefficient is
/// calculated internally as `w = alpha * u * v`, matching the Drusano and
/// mod120 model definitions. A finite negative `alpha` is valid and represents
/// antagonism.
///
/// The first Nelder-Mead result is accepted only when its squared residual is
/// at most `1e-10`, matching the Fortran `VALMIN` threshold. Otherwise the
/// historical FINDM0-style estimate seeds a second solve, and the candidate
/// with the lower squared residual is selected. If no exact positive root
/// exists, the historical behavior is retained: the best finite least-squares
/// candidate is returned. Any optimizer failure is handled as a scalar
/// fallback rather than a panic.
///
/// # Arguments
///
/// * `u` - Coefficient for the first binding site.
/// * `v` - Coefficient for the second binding site.
/// * `alpha` - Finite interaction coefficient, including negative antagonistic values.
/// * `h1` - Hill exponent for the first site; must be positive and finite.
/// * `h2` - Hill exponent for the second site; must be positive and finite.
///
/// # Returns
///
/// The effect `M / (1 + M)`. Exact zero coefficients use their closed-form
/// single-site solution. Malformed inputs and cases without a meaningful finite
/// scalar return `NaN`.
///
/// # Example
///
/// ```
/// use pharmsol::get_e2;
///
/// // Single-site model: M = u^(1/h1) = 1.
/// let e2 = get_e2(1.0, 0.0, 0.5, 1.0, 2.0);
/// assert!((e2 - 0.5).abs() < 1e-10);
///
/// // Equal exponents have a closed-form combined coefficient.
/// let e2 = get_e2(1.0, 1.0, -0.5, 1.0, 1.0);
/// assert!((e2 - 0.6).abs() < 1e-6);
/// ```
pub fn get_e2(u: f64, v: f64, alpha: f64, h1: f64, h2: f64) -> f64 {
    if !(u.is_finite()
        && v.is_finite()
        && alpha.is_finite()
        && h1.is_finite()
        && h2.is_finite()
        && u >= 0.0
        && v >= 0.0
        && h1 > 0.0
        && h2 > 0.0)
    {
        return f64::NAN;
    }

    let w = alpha * u * v;
    if !w.is_finite() {
        return f64::NAN;
    }

    if u == 0.0 && v == 0.0 {
        return 0.0;
    }
    if v == 0.0 {
        return single_site_xm(u, h1).map_or(f64::NAN, effect_from_xm);
    }
    if u == 0.0 {
        return single_site_xm(v, h2).map_or(f64::NAN, effect_from_xm);
    }

    let xx = (h1 + h2) / 2.0;
    if !xx.is_finite() {
        return f64::NAN;
    }
    let objective = BestM0 {
        u,
        v,
        w,
        h1,
        h2,
        xx,
    };

    let xm_guess = v.powf(1.0 / h2).max(u.powf(1.0 / h1)).max(1.0e-12);
    if !(xm_guess.is_finite() && xm_guess > 0.0) {
        return f64::NAN;
    }

    let first = objective.get_best(xm_guess.ln(), 0.1).ok();

    // A nominal convergence status is not sufficient: argmin can converge to
    // a flat, non-root minimum for antagonistic or badly scaled inputs.
    if let Some(candidate) = first {
        if candidate.converged && candidate.cost <= RESIDUAL_COST_TOLERANCE {
            return effect_from_xm(candidate.xm);
        }
        // Preserve the useful historical behavior for a very small residual
        // even when the optimizer's status was not marked converged.
        if candidate.cost <= RESIDUAL_COST_TOLERANCE {
            return effect_from_xm(candidate.xm);
        }
    }

    let fallback_xm = find_m0(u, v, alpha, h1, h2);
    let second = if fallback_xm.is_finite() && fallback_xm > 0.0 {
        objective.get_best(fallback_xm.ln(), 0.1).ok()
    } else {
        None
    };

    if let Some(candidate) = select_lower_residual(first, second) {
        return effect_from_xm(candidate.xm);
    }

    // FINDM0 itself is still a meaningful historical scalar when the second
    // optimizer cannot be started. Do not turn that recoverable case into NaN.
    if fallback_xm.is_finite() && fallback_xm > 0.0 {
        return effect_from_xm(fallback_xm);
    }

    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_residual(u: f64, v: f64, alpha: f64, h1: f64, h2: f64, xm: f64) -> f64 {
        1.0 - u / xm.powf(h1) - v / xm.powf(h2) - (alpha * u * v) / xm.powf((h1 + h2) / 2.0)
    }

    #[test]
    fn single_site_vectors_match_closed_form() {
        assert!((get_e2(1.0, 0.0, 7.0, 1.0, 2.0) - 0.5).abs() < 1.0e-12);
        let expected = 2.0_f64.sqrt() / (1.0 + 2.0_f64.sqrt());
        assert!((get_e2(0.0, 2.0, -3.0, 1.0, 2.0) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn equal_hill_exponents_use_the_canonical_interaction_term() {
        let u = 1.0;
        let v = 1.0;
        let alpha = -0.5;
        let expected_m = u + v + alpha * u * v;
        let expected = expected_m / (1.0 + expected_m);
        let actual = get_e2(u, v, alpha, 1.0, 1.0);
        assert!((actual - expected).abs() < 1.0e-6);
    }

    #[test]
    fn antagonistic_alpha_still_solves_a_finite_root() {
        let u = 1.0;
        let v = 1.0;
        let alpha = -1.5;
        let h1 = 1.0;
        let h2 = 1.0;
        let xm = 0.5;
        assert!(canonical_residual(u, v, alpha, h1, h2, xm).abs() < 1.0e-12);
        assert!((get_e2(u, v, alpha, h1, h2) - xm / (1.0 + xm)).abs() < 1.0e-6);
    }

    #[test]
    fn optimizer_candidate_cost_is_the_canonical_squared_residual() {
        let objective = BestM0 {
            u: 1.0,
            v: 1.0,
            w: -0.5,
            h1: 1.0,
            h2: 1.0,
            xx: 1.0,
        };
        let root = 1.5_f64;
        let cost = objective.cost(&root.ln()).expect("finite cost");
        assert!(cost < RESIDUAL_COST_TOLERANCE);
        assert!(
            (objective.residual(root).expect("finite residual")
                - canonical_residual(1.0, 1.0, -0.5, 1.0, 1.0, root))
            .abs()
                < 1.0e-15
        );
    }

    #[test]
    fn no_positive_root_returns_the_historical_best_candidate() {
        let effect = get_e2(1.0, 1.0, -3.0, 1.0, 1.0);
        assert!(effect.is_finite() && (0.0..=1.0).contains(&effect));

        let xm = effect / (1.0 - effect);
        let residual = canonical_residual(1.0, 1.0, -3.0, 1.0, 1.0, xm);
        assert!(residual * residual > RESIDUAL_COST_TOLERANCE);
    }

    #[test]
    fn malformed_inputs_are_total_and_return_nan() {
        for args in [
            (f64::NAN, 1.0, 0.0, 1.0, 1.0),
            (1.0, f64::INFINITY, 0.0, 1.0, 1.0),
            (1.0, 1.0, f64::NAN, 1.0, 1.0),
            (1.0, 1.0, 0.0, 0.0, 1.0),
            (-1.0, 1.0, 0.0, 1.0, 1.0),
        ] {
            let result =
                std::panic::catch_unwind(|| get_e2(args.0, args.1, args.2, args.3, args.4));
            assert!(result.is_ok());
            assert!(result.expect("call did not panic").is_nan());
        }
    }
}
