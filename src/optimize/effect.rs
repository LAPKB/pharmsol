//! Maximum-effect optimization for multi-site pharmacodynamic models.
//!
//! [`get_e2`] solves the canonical two-site Drusano/Greco equation, while
//! [`get_e3`] implements the three-site extension described by Snyder et al.
//! (PMID 10722511). Both equations are solved in positive `M` space. The
//! two-site path retains its historical Nelder-Mead and FINDM0 behavior; the
//! three-site path combines root bracketing with multi-start Nelder-Mead so
//! signed interactions cannot trap the solve at a non-root local minimum.

use std::panic::{catch_unwind, AssertUnwindSafe};

use argmin::{
    core::{CostFunction, Executor, TerminationReason, TerminationStatus},
    solver::neldermead::NelderMead,
};

const RESIDUAL_COST_TOLERANCE: f64 = 1.0e-10;
const HISTORICAL_NELDER_MEAD_TOLERANCE: f64 = 1.0e-8;
const E3_NELDER_MEAD_TOLERANCE: f64 = 1.0e-24;
const E3_ROOT_COST_TOLERANCE: f64 = 1.0e-24;
const INVALID_COST: f64 = 1.0e100;
const NEGLIGIBLE_EXPOSURE: f64 = 1.0e-5;
const MAX_RESIDUAL_TERMS: usize = 7;
const E3_LOG_SEARCH_MARGIN: f64 = 32.0;
const E3_LOG_SEARCH_STEP: f64 = 0.25;
const E3_MAX_SEARCH_INTERVALS: usize = 8192;

#[derive(Debug, Clone, Copy)]
struct ResidualTerm {
    coefficient: f64,
    exponent: f64,
}

impl ResidualTerm {
    const ZERO: Self = Self {
        coefficient: 0.0,
        exponent: 1.0,
    };

    const fn new(coefficient: f64, exponent: f64) -> Self {
        Self {
            coefficient,
            exponent,
        }
    }
}

#[derive(Debug, Clone)]
struct BestM0 {
    terms: [ResidualTerm; MAX_RESIDUAL_TERMS],
    term_count: usize,
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
    fn from_terms<const N: usize>(terms: [ResidualTerm; N]) -> Self {
        assert!(N <= MAX_RESIDUAL_TERMS);
        let mut padded = [ResidualTerm::ZERO; MAX_RESIDUAL_TERMS];
        padded[..N].copy_from_slice(&terms);
        Self {
            terms: padded,
            term_count: N,
        }
    }

    fn residual(&self, xm: f64) -> Option<f64> {
        if !(xm.is_finite() && xm > 0.0) {
            return None;
        }

        let mut residual = 1.0;
        for term in &self.terms[..self.term_count] {
            if term.coefficient == 0.0 {
                continue;
            }
            let denominator = xm.powf(term.exponent);
            let value = term.coefficient / denominator;
            if !value.is_finite() {
                return None;
            }
            residual -= value;
        }
        residual.is_finite().then_some(residual)
    }

    /// Start and step are in log-space (`ln(M)`). This entry point preserves
    /// the historical E2 Nelder-Mead tolerance.
    fn get_best(&self, start_log: f64, step_log: f64) -> Result<OptimizationResult, BestM0Error> {
        self.get_best_with_tolerance(start_log, step_log, HISTORICAL_NELDER_MEAD_TOLERANCE)
    }

    /// Optimizer panics and missing best parameters are converted into ordinary
    /// errors so no failure can escape either public scalar API.
    fn get_best_with_tolerance(
        &self,
        start_log: f64,
        step_log: f64,
        tolerance: f64,
    ) -> Result<OptimizationResult, BestM0Error> {
        let run = catch_unwind(AssertUnwindSafe(|| {
            if !start_log.is_finite() || !(tolerance.is_finite() && tolerance > 0.0) {
                return Err(BestM0Error::OptimizationError(
                    "optimizer start or tolerance is invalid".to_string(),
                ));
            }

            let second = start_log + step_log;
            let initial_simplex = if !second.is_finite() || (second - start_log).abs() < 1.0e-12 {
                vec![start_log, start_log + 0.1_f64]
            } else {
                vec![start_log, second]
            };

            let solver = NelderMead::new(initial_simplex)
                .with_sd_tolerance(tolerance)
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

    fn candidate_at_log(&self, log_xm: f64) -> Option<(OptimizationResult, f64)> {
        let xm = log_xm.exp();
        let residual = self.residual(xm)?;
        let cost = residual * residual;
        cost.is_finite().then_some((
            OptimizationResult {
                xm,
                cost,
                converged: false,
            },
            residual,
        ))
    }

    fn bisect_root(
        &self,
        mut left_log: f64,
        mut left_residual: f64,
        mut right_log: f64,
        right_residual: f64,
    ) -> Option<OptimizationResult> {
        let mut best = select_lower_residual(
            self.candidate_at_log(left_log).map(|value| value.0),
            self.candidate_at_log(right_log).map(|value| value.0),
        );
        if left_residual == 0.0 || right_residual == 0.0 {
            return best;
        }
        if left_residual.is_sign_negative() == right_residual.is_sign_negative() {
            return best;
        }

        for _ in 0..128 {
            let middle_log = left_log + (right_log - left_log) / 2.0;
            let (mut candidate, middle_residual) = self.candidate_at_log(middle_log)?;
            candidate.converged = true;
            best = select_lower_residual(best, Some(candidate));
            if candidate.cost <= E3_ROOT_COST_TOLERANCE {
                return Some(candidate);
            }

            if left_residual.is_sign_negative() == middle_residual.is_sign_negative() {
                left_log = middle_log;
                left_residual = middle_residual;
            } else {
                right_log = middle_log;
            }
        }
        best
    }

    /// Search characteristic concentration scales across the one-dimensional
    /// E3 objective, avoiding non-root local minima from signed interactions.
    fn get_global_best(&self, initial_log: f64) -> Result<OptimizationResult, BestM0Error> {
        if !initial_log.is_finite() {
            return Err(BestM0Error::OptimizationError(
                "optimizer start is not finite".to_string(),
            ));
        }

        let mut anchors = vec![initial_log];
        let mut grouped_terms: Vec<ResidualTerm> = Vec::new();
        for term in &self.terms[..self.term_count] {
            if term.coefficient == 0.0 {
                continue;
            }
            let characteristic_log = term.coefficient.abs().ln() / term.exponent;
            if characteristic_log.is_finite() {
                anchors.push(characteristic_log);
            }
            if let Some(group) = grouped_terms
                .iter_mut()
                .find(|group| group.exponent.to_bits() == term.exponent.to_bits())
            {
                group.coefficient += term.coefficient;
            } else {
                grouped_terms.push(*term);
            }
        }
        for term in grouped_terms {
            if term.coefficient != 0.0 {
                let characteristic_log = term.coefficient.abs().ln() / term.exponent;
                if characteristic_log.is_finite() {
                    anchors.push(characteristic_log);
                }
            }
        }

        let min_anchor = anchors.iter().copied().fold(f64::INFINITY, f64::min);
        let max_anchor = anchors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lower = (min_anchor - E3_LOG_SEARCH_MARGIN).max(-700.0);
        let upper = (max_anchor + E3_LOG_SEARCH_MARGIN).min(700.0);
        if !(lower.is_finite() && upper.is_finite() && lower < upper) {
            return Err(BestM0Error::OptimizationError(
                "could not construct a finite E3 search interval".to_string(),
            ));
        }

        let interval_count = (((upper - lower) / E3_LOG_SEARCH_STEP).ceil() as usize)
            .clamp(1, E3_MAX_SEARCH_INTERVALS);
        let step = (upper - lower) / interval_count as f64;
        let mut samples = Vec::with_capacity(interval_count + 1);
        let mut best = None;
        for index in 0..=interval_count {
            let log_xm = if index == interval_count {
                upper
            } else {
                lower + index as f64 * step
            };
            let sample = self.candidate_at_log(log_xm);
            if let Some((candidate, _)) = sample {
                best = select_lower_residual(best, Some(candidate));
            }
            samples.push(sample.map(|(candidate, residual)| (log_xm, candidate, residual)));
        }

        for pair in samples.windows(2) {
            let (Some((left_log, _, left_residual)), Some((right_log, _, right_residual))) =
                (pair[0], pair[1])
            else {
                continue;
            };
            if left_residual == 0.0
                || right_residual == 0.0
                || left_residual.is_sign_negative() != right_residual.is_sign_negative()
            {
                best = select_lower_residual(
                    best,
                    self.bisect_root(left_log, left_residual, right_log, right_residual),
                );
                if best.is_some_and(|candidate| candidate.cost <= E3_ROOT_COST_TOLERANCE) {
                    return best.ok_or_else(|| {
                        BestM0Error::OptimizationError("E3 root selection failed".to_string())
                    });
                }
            }
        }

        let mut seeds = anchors;
        if let Some(candidate) = best {
            seeds.push(candidate.xm.ln());
        }
        for window in samples.windows(3) {
            let (Some((_, left, _)), Some((middle_log, middle, _)), Some((_, right, _))) =
                (window[0], window[1], window[2])
            else {
                continue;
            };
            if middle.cost <= left.cost && middle.cost <= right.cost {
                seeds.push(middle_log);
            }
        }
        seeds.retain(|seed| seed.is_finite());
        seeds.sort_by(|left, right| left.total_cmp(right));
        seeds.dedup_by(|left, right| (*left - *right).abs() < 1.0e-8);

        for seed in seeds.into_iter().take(32) {
            let candidate = self
                .get_best_with_tolerance(seed, 0.1, E3_NELDER_MEAD_TOLERANCE)
                .ok();
            best = select_lower_residual(best, candidate);
        }

        best.ok_or_else(|| {
            BestM0Error::OptimizationError("E3 search found no finite candidate".to_string())
        })
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
/// * `h1` - Positive finite exponent `H1`, the reciprocal of the first conventional Hill coefficient.
/// * `h2` - Positive finite exponent `H2`, the reciprocal of the second conventional Hill coefficient.
///
/// # Returns
///
/// The effect `M / (1 + M)`. Two coefficients below `1e-5` return zero, as in
/// the historical boundary condition, and exact zero coefficients use their
/// closed-form single-site solution. Malformed inputs and cases without a
/// meaningful finite scalar return `NaN`.
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

    if u < NEGLIGIBLE_EXPOSURE && v < NEGLIGIBLE_EXPOSURE {
        return 0.0;
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
    let objective = BestM0::from_terms([
        ResidualTerm::new(u, h1),
        ResidualTerm::new(v, h2),
        ResidualTerm::new(w, xx),
    ]);

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

/// Computes the effect metric for the three-site Greco response-surface model.
///
/// This is the three-drug extension published by Snyder et al. (PMID 10722511).
/// With positive `M`, the canonical residual is
///
/// ```text
/// r(M) = 1 - a/M^h1 - b/M^h2 - c/M^h3
///        - (alpha12*a*b)/M^((h1+h2)/2)
///        - (alpha13*a*c)/M^((h1+h3)/2)
///        - (alpha23*b*c)/M^((h2+h3)/2)
///        - (alpha123*a*b*c)/M^((h1+h2+h3)/3)
/// ```
///
/// The function brackets positive roots across characteristic concentration
/// scales and uses multi-start Nelder-Mead for any remaining least-squares
/// minima, then returns `M / (1 + M)`.
/// Each normalized exposure below `1e-5` is treated as zero. Consequently, the
/// model reduces exactly to [`get_e2`] when one site is absent and to the
/// corresponding closed-form single-site equation when two sites are absent.
/// Finite negative interaction parameters are valid and represent antagonism.
///
/// The exponent arguments are `H_i = 1 / m_i`, where `m_i` are the conventional
/// Hill coefficients used in the source publication. Pairwise and three-way
/// interaction exponents are therefore the arithmetic means of these reciprocal
/// Hill coefficients.
///
/// # Arguments
///
/// * `a`, `b`, `c` - Nonnegative finite normalized drug exposures.
/// * `alpha12`, `alpha13`, `alpha23` - Finite pairwise interaction coefficients.
/// * `alpha123` - Finite three-way interaction coefficient.
/// * `h1`, `h2`, `h3` - Positive finite reciprocal Hill exponents `H1`, `H2`, and `H3`.
///
/// # Returns
///
/// The effect `M / (1 + M)`. Malformed inputs and search failures without a
/// meaningful finite scalar return `NaN`. If no exact positive root exists, the
/// best finite least-squares candidate found by the bounded log-space search is
/// returned.
///
/// # Example
///
/// ```
/// use pharmsol::get_e3;
///
/// // With all exponents equal to one, M is the sum of all seven coefficients.
/// let effect = get_e3(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
/// assert!((effect - 0.8).abs() < 1e-6);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn get_e3(
    a: f64,
    b: f64,
    c: f64,
    alpha12: f64,
    alpha13: f64,
    alpha23: f64,
    alpha123: f64,
    h1: f64,
    h2: f64,
    h3: f64,
) -> f64 {
    if ![a, b, c, alpha12, alpha13, alpha23, alpha123, h1, h2, h3]
        .into_iter()
        .all(f64::is_finite)
        || a < 0.0
        || b < 0.0
        || c < 0.0
        || h1 <= 0.0
        || h2 <= 0.0
        || h3 <= 0.0
    {
        return f64::NAN;
    }

    let a = if a < NEGLIGIBLE_EXPOSURE { 0.0 } else { a };
    let b = if b < NEGLIGIBLE_EXPOSURE { 0.0 } else { b };
    let c = if c < NEGLIGIBLE_EXPOSURE { 0.0 } else { c };

    match (a == 0.0, b == 0.0, c == 0.0) {
        (true, true, true) => return 0.0,
        (false, true, true) => return single_site_xm(a, h1).map_or(f64::NAN, effect_from_xm),
        (true, false, true) => return single_site_xm(b, h2).map_or(f64::NAN, effect_from_xm),
        (true, true, false) => return single_site_xm(c, h3).map_or(f64::NAN, effect_from_xm),
        (false, false, true) => return get_e2(a, b, alpha12, h1, h2),
        (false, true, false) => return get_e2(a, c, alpha13, h1, h3),
        (true, false, false) => return get_e2(b, c, alpha23, h2, h3),
        (false, false, false) => {}
    }

    let h12 = (h1 + h2) / 2.0;
    let h13 = (h1 + h3) / 2.0;
    let h23 = (h2 + h3) / 2.0;
    let h123 = (h1 + h2 + h3) / 3.0;
    let w12 = alpha12 * a * b;
    let w13 = alpha13 * a * c;
    let w23 = alpha23 * b * c;
    let w123 = alpha123 * a * b * c;
    if ![h12, h13, h23, h123, w12, w13, w23, w123]
        .into_iter()
        .all(f64::is_finite)
    {
        return f64::NAN;
    }

    let objective = BestM0::from_terms([
        ResidualTerm::new(a, h1),
        ResidualTerm::new(b, h2),
        ResidualTerm::new(c, h3),
        ResidualTerm::new(w12, h12),
        ResidualTerm::new(w13, h13),
        ResidualTerm::new(w23, h23),
        ResidualTerm::new(w123, h123),
    ]);

    let Some(a_xm) = single_site_xm(a, h1) else {
        return f64::NAN;
    };
    let Some(b_xm) = single_site_xm(b, h2) else {
        return f64::NAN;
    };
    let Some(c_xm) = single_site_xm(c, h3) else {
        return f64::NAN;
    };
    let xm_guess = a_xm.max(b_xm).max(c_xm).max(1.0e-12);

    objective
        .get_global_best(xm_guess.ln())
        .ok()
        .map_or(f64::NAN, |candidate| effect_from_xm(candidate.xm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_residual(u: f64, v: f64, alpha: f64, h1: f64, h2: f64, xm: f64) -> f64 {
        1.0 - u / xm.powf(h1) - v / xm.powf(h2) - (alpha * u * v) / xm.powf((h1 + h2) / 2.0)
    }

    fn canonical_e3_residual(
        exposures: [f64; 3],
        pairwise_alpha: [f64; 3],
        alpha123: f64,
        exponents: [f64; 3],
        xm: f64,
    ) -> f64 {
        let [a, b, c] = exposures;
        let [alpha12, alpha13, alpha23] = pairwise_alpha;
        let [h1, h2, h3] = exponents;
        1.0 - a / xm.powf(h1)
            - b / xm.powf(h2)
            - c / xm.powf(h3)
            - alpha12 * a * b / xm.powf((h1 + h2) / 2.0)
            - alpha13 * a * c / xm.powf((h1 + h3) / 2.0)
            - alpha23 * b * c / xm.powf((h2 + h3) / 2.0)
            - alpha123 * a * b * c / xm.powf((h1 + h2 + h3) / 3.0)
    }

    #[test]
    fn single_site_vectors_match_closed_form() {
        assert!((get_e2(1.0, 0.0, 7.0, 1.0, 2.0) - 0.5).abs() < 1.0e-12);
        let expected = 2.0_f64.sqrt() / (1.0 + 2.0_f64.sqrt());
        assert!((get_e2(0.0, 2.0, -3.0, 1.0, 2.0) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn dual_exposure_boundary_matches_the_historical_threshold() {
        assert_eq!(get_e2(0.9e-5, 0.9e-5, f64::MAX, 1.0, 1.0), 0.0);
    }

    #[test]
    fn get_e2_preserves_the_historical_optimizer_result() {
        assert!((get_e2(1.0, 1.0, -0.5, 1.0, 1.0) - 0.6).abs() < 1.0e-10);
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
        let objective = BestM0::from_terms([
            ResidualTerm::new(1.0, 1.0),
            ResidualTerm::new(1.0, 1.0),
            ResidualTerm::new(-0.5, 1.0),
        ]);
        let root = 1.5_f64;
        let cost = objective.cost(&root.ln()).unwrap_or(INVALID_COST);
        assert!(cost < RESIDUAL_COST_TOLERANCE);
        let residual = objective.residual(root).unwrap_or(f64::NAN);
        assert!((residual - canonical_residual(1.0, 1.0, -0.5, 1.0, 1.0, root)).abs() < 1.0e-15);
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
    fn get_e3_equal_exponents_match_the_closed_form() {
        let (a, b, c) = (1.0, 2.0, 0.5);
        let (alpha12, alpha13, alpha23, alpha123) = (0.2, -0.1, 0.3, 0.4);
        let expected_m =
            a + b + c + alpha12 * a * b + alpha13 * a * c + alpha23 * b * c + alpha123 * a * b * c;
        let expected = expected_m / (1.0 + expected_m);
        let actual = get_e3(a, b, c, alpha12, alpha13, alpha23, alpha123, 1.0, 1.0, 1.0);
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn get_e3_uses_reciprocal_hill_averages_from_the_published_equation() {
        let exposures = [0.8, 1.1, 0.6];
        let pairwise_alpha = [0.25, -0.1, 0.4];
        let alpha123 = 0.15;
        let exponents = [0.75, 1.4, 2.1];
        let effect = get_e3(
            exposures[0],
            exposures[1],
            exposures[2],
            pairwise_alpha[0],
            pairwise_alpha[1],
            pairwise_alpha[2],
            alpha123,
            exponents[0],
            exponents[1],
            exponents[2],
        );
        assert!(effect.is_finite() && (0.0..1.0).contains(&effect));
        let xm = effect / (1.0 - effect);
        let residual = canonical_e3_residual(exposures, pairwise_alpha, alpha123, exponents, xm);
        assert!(
            residual * residual <= RESIDUAL_COST_TOLERANCE,
            "effect={effect}, xm={xm}, residual={residual}"
        );
    }

    #[test]
    fn get_e3_finds_the_global_antagonistic_root() {
        // Multiplying the residual by M^3 gives the strictly increasing cubic
        // M^3 - 2M^2 + 3M - 1, which has one positive root near 0.43016.
        let effect = get_e3(1.0, 1.0, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0);
        assert!(effect.is_finite() && (0.0..1.0).contains(&effect));
        let xm = effect / (1.0 - effect);
        let residual =
            canonical_e3_residual([1.0, 1.0, 1.0], [-3.0, 0.0, 0.0], 0.0, [1.0, 3.0, 1.0], xm);
        assert!(
            (xm - 0.430_159_709).abs() < 1.0e-6,
            "xm={xm}, residual={residual}"
        );
        assert!(residual * residual <= E3_ROOT_COST_TOLERANCE * 10.0);
    }

    #[test]
    fn get_e3_reduces_to_the_canonical_lower_order_models() {
        let expected_bc = get_e2(1.25, 0.75, -0.2, 1.2, 0.8);
        assert_eq!(
            get_e3(1.0e-6, 1.25, 0.75, 8.0, -4.0, -0.2, 3.0, 2.0, 1.2, 0.8,),
            expected_bc
        );

        let expected_ac = get_e2(0.8, 1.4, 0.3, 1.1, 0.9);
        assert_eq!(
            get_e3(0.8, 1.0e-6, 1.4, -2.0, 0.3, 4.0, -1.0, 1.1, 2.0, 0.9,),
            expected_ac
        );
        let expected_ab = get_e2(0.8, 1.4, -0.3, 1.1, 0.9);
        assert_eq!(
            get_e3(0.8, 1.4, 1.0e-6, -0.3, 2.0, -4.0, 1.0, 1.1, 0.9, 2.0,),
            expected_ab
        );

        let expected_a_xm = 2.0_f64.sqrt();
        let expected_a = expected_a_xm / (1.0 + expected_a_xm);
        assert!(
            (get_e3(2.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5) - expected_a).abs() < 1.0e-15
        );
        assert_eq!(
            get_e3(0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5),
            0.0
        );
    }

    #[test]
    fn get_e3_is_invariant_to_site_permutation() {
        let original = get_e3(0.8, 1.1, 0.6, 0.25, -0.1, 0.4, 0.15, 0.75, 1.4, 2.1);
        let swapped_ab = get_e3(1.1, 0.8, 0.6, 0.25, 0.4, -0.1, 0.15, 1.4, 0.75, 2.1);
        assert!((original - swapped_ab).abs() < 1.0e-8);
    }

    #[test]
    fn get_e3_malformed_inputs_are_total_and_return_nan() {
        let cases = [
            [f64::NAN, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, f64::INFINITY, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        for args in cases {
            let result = std::panic::catch_unwind(|| {
                get_e3(
                    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                    args[8], args[9],
                )
            });
            assert!(result.is_ok());
            assert!(result.expect("call did not panic").is_nan());
        }
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
