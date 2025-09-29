use argmin::{
    core::{CostFunction, Executor, TerminationReason, TerminationStatus},
    solver::neldermead::NelderMead,
};

#[derive(Debug, Clone)]
struct BestM0 {
    a: f64,
    b: f64,
    w: f64,
    h1: f64,
    h2: f64,
    xx: f64,
}

/// We'll optimize over y = ln(xm), so Param = f64 (the log of xm)
impl CostFunction for BestM0 {
    type Param = f64; // this is ln(xm)
    type Output = f64;

    fn cost(&self, y: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // compute xm from log-parameter
        let xm = y.exp();

        // guard: xm must be > 0 and finite
        if !(xm.is_finite() && xm > 0.0) {
            // return a very large cost instead of NaN
            return Ok(1.0e100);
        }

        // guard a,b,w positive/negative combinations: powf with positive base fine
        let t1 = if self.a == 0.0 {
            0.0
        } else {
            self.a / xm.powf(self.h1)
        };
        let t2 = if self.b == 0.0 {
            0.0
        } else {
            self.b / xm.powf(self.h2)
        };
        let t3 = if self.w == 0.0 {
            0.0
        } else {
            self.w / xm.powf(self.xx)
        };

        // If any term is NaN or infinite, treat as bad point
        if !t1.is_finite() || !t2.is_finite() || !t3.is_finite() {
            return Ok(1.0e100);
        }

        let val = (1.0 - t1 - t2 - t3).powi(2);
        if !val.is_finite() {
            return Ok(1.0e100);
        }

        Ok(val)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BestM0Error {
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

impl BestM0 {
    /// start and step are in log-space (ln(x))
    fn get_best(&self, start_log: f64, step_log: f64) -> Result<(f64, f64, bool), BestM0Error> {
        // Build a simplex with two log-parameters, both finite and distinct
        let second = start_log + step_log;
        // if step pushed us to invalid values, choose a small positive step
        let initial_simplex = if !(second.is_finite()) || (second - start_log).abs() < 1e-12 {
            vec![start_log, start_log + 0.1_f64] // 0.1 in log-space ~ 10% change
        } else {
            vec![start_log, second]
        };

        let solver = NelderMead::new(initial_simplex)
            .with_sd_tolerance(1e-8)
            .map_err(|e| {
                BestM0Error::OptimizationError(format!(
                    "Failed setting up the tolerance of the NelderMead solver: {}",
                    e
                ))
            })?;

        let res = Executor::new(self.clone(), solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .map_err(|e| {
                BestM0Error::OptimizationError(format!(
                    "Failed creating the configuration for the NelderMead solver: {}",
                    e
                ))
            })?;

        let converged = match &res.state.termination_status {
            TerminationStatus::Terminated(reason) => {
                matches!(reason, TerminationReason::SolverConverged)
            }
            _ => false,
        };

        // best_param is ln(xm). Convert back to xm
        let best_log = res.state.best_param.unwrap();
        let xm = best_log.exp();

        Ok((xm, res.state.best_cost, converged))
    }
}

/// find_m0 left largely as-is, but consider returning Result<f64>
/// Keep in mind it expects a,b,h1,h2 in valid ranges
fn find_m0(afinal: f64, b: f64, alpha: f64, h1: f64, h2: f64) -> f64 {
    let noint = 1000;
    let del_a = afinal / (noint as f64);
    // initial guess; must be positive
    let mut xm = if b > 0.0 { b.powf(1.0 / h2) } else { 1.0 };
    let mut a = 0.0;
    let hh = (h1 + h2) / 2.0;

    for int in 1..=noint {
        // safe guards: avoid dividing by zero
        if xm <= 0.0 || xm.is_nan() || !xm.is_finite() {
            return -1.0;
        }

        let top = 1.0 / xm.powf(h1) + alpha * b / xm.powf(hh);
        let b1 = a * h1 / xm.powf(h1 + 1.0);
        let b2 = b * h2 / xm.powf(h2 + 1.0);
        let b3 = alpha * a * b * hh / xm.powf(hh + 1.0);

        let denom = b1 + b2 + b3;
        if denom == 0.0 || !denom.is_finite() {
            return -1.0;
        }

        let xmp = top / denom;
        xm = xm + xmp * del_a;

        if !(xm.is_finite() && xm > 0.0) {
            return -1.0;
        }

        a = del_a * (int as f64);
    }

    xm
}

pub fn get_e2(a: f64, b: f64, w: f64, h1: f64, h2: f64, alpha_s: f64) -> f64 {
    // trivial cases
    if a.abs() < 1.0e-12 && b.abs() < 1.0e-12 {
        return 0.0;
    }

    // precompute
    let xx = (h1 + h2) / 2.0;
    let bm0 = BestM0 {
        a,
        b,
        w,
        h1,
        h2,
        xx,
    };

    // if one coefficient negative/zero, return simple closed-form estimate
    if b <= 0.0 && a > 0.0 {
        let xm0best = a.powf(1.0 / h1);
        return xm0best / (xm0best + 1.0);
    }
    if a <= 0.0 && b > 0.0 {
        let xm0best = b.powf(1.0 / h2);
        return xm0best / (xm0best + 1.0);
    }

    // both positive: do optimization in log-space
    // choose a safe initial guess > 0
    let xm_guess = if b > 0.0 {
        b.powf(1.0 / h2)
    } else if a > 0.0 {
        a.powf(1.0 / h1)
    } else {
        1.0
    };
    let start_log = xm_guess.max(1e-12).ln();
    let step_log = 0.1_f64; // ~10% step in xm

    // first optimization from small start
    match bm0.get_best(start_log, step_log) {
        Ok((xm0best1, valmin1, conv1)) => {
            if !conv1 {
                // we still keep the answer if cost is tiny
                if valmin1 < 1e-10 {
                    return xm0best1 / (xm0best1 + 1.0);
                }
                // fallback to iterative estimator
                let xm0est = find_m0(a, b, alpha_s, h1, h2);
                if xm0est < 0.0 {
                    return xm0best1 / (xm0best1 + 1.0);
                }
                // refine from bg estimate:
                let start_log2 = xm0est.ln();
                if let Ok((xm0best2, valmin2, conv2)) = bm0.get_best(start_log2, 0.1) {
                    if conv2 && valmin2 < valmin1 {
                        return xm0best2 / (xm0best2 + 1.0);
                    } else {
                        return xm0best1 / (xm0best1 + 1.0);
                    }
                } else {
                    return xm0best1 / (xm0best1 + 1.0);
                }
            } else {
                return xm0best1 / (xm0best1 + 1.0);
            }
        }
        Err(_) => {
            // if optimizer failed, fallback to numerical estimator
            let xm0est = find_m0(a, b, alpha_s, h1, h2);
            if xm0est > 0.0 {
                return xm0est / (xm0est + 1.0);
            } else {
                // last resort: simple closed form (if possible)
                if a > 0.0 {
                    let xm0best = a.powf(1.0 / h1);
                    return xm0best / (xm0best + 1.0);
                }
                if b > 0.0 {
                    let xm0best = b.powf(1.0 / h2);
                    return xm0best / (xm0best + 1.0);
                }
                return 0.0;
            }
        }
    }
}
