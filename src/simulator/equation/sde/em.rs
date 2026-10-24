//! Adaptive Euler-Maruyama integration.

use diffsol::{NalgebraContext, Vector, VectorHost};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::simulator::V;
use crate::PharmsolError;

const RTOL: f64 = 1e-2;
const ATOL: f64 = 1e-2;
const MAX_STEP: f64 = 0.1;
const MIN_STEP: f64 = 1e-6;
const SAFETY: f64 = 0.9;

/// Buffers for [`advance`], sized to the state.
pub(crate) struct EmScratch {
    full_step: V,
    half_steps: V,
    drift: V,
    diffusion: V,
}

impl EmScratch {
    pub(crate) fn new(nstates: usize) -> Self {
        let ctx = NalgebraContext::new();
        Self {
            full_step: V::zeros(nstates, ctx),
            half_steps: V::zeros(nstates, ctx),
            drift: V::zeros(nstates, ctx),
            diffusion: V::zeros(nstates, ctx),
        }
    }
}

/// Advance `x` from `t0` to `tf` with adaptive Euler-Maruyama steps.
///
/// Each attempt compares one full step with two half steps and accepts the
/// half-step result when the scaled difference is at most one. Steps never
/// cross `tf`, and a step at the minimum size is always accepted.
///
/// `drift` and `diffusion` write the coefficients at `(t, x)` into their
/// output, which is zeroed first.
pub(crate) fn advance<D, G, R>(
    drift: &mut D,
    diffusion: &mut G,
    x: &mut V,
    t0: f64,
    tf: f64,
    rng: &mut R,
    scratch: &mut EmScratch,
) -> Result<(), PharmsolError>
where
    D: FnMut(f64, &V, &mut V) -> Result<(), PharmsolError>,
    G: FnMut(f64, &V, &mut V) -> Result<(), PharmsolError>,
    R: Rng + ?Sized,
{
    let EmScratch {
        full_step,
        half_steps,
        drift: drift_term,
        diffusion: diffusion_term,
    } = scratch;
    let mut t = t0;
    let mut dt = MAX_STEP;

    while t < tf {
        dt = dt.min(tf - t);
        let half = dt / 2.0;

        full_step.copy_from(x);
        let mut terms = (&mut *drift_term, &mut *diffusion_term);
        step(drift, diffusion, t, dt, full_step, rng, &mut terms)?;
        half_steps.copy_from(x);
        step(drift, diffusion, t, half, half_steps, rng, &mut terms)?;
        step(
            drift,
            diffusion,
            t + half,
            half,
            half_steps,
            rng,
            &mut terms,
        )?;

        let error = scaled_error(x, full_step, half_steps);
        if error <= 1.0 || dt <= MIN_STEP {
            t += dt;
            x.copy_from(half_steps);
        }
        dt = next_step(dt, error);
    }
    Ok(())
}

/// One Euler-Maruyama step of size `dt` from `t`, in place.
fn step<D, G, R>(
    drift: &mut D,
    diffusion: &mut G,
    t: f64,
    dt: f64,
    x: &mut V,
    rng: &mut R,
    (drift_term, diffusion_term): &mut (&mut V, &mut V),
) -> Result<(), PharmsolError>
where
    D: FnMut(f64, &V, &mut V) -> Result<(), PharmsolError>,
    G: FnMut(f64, &V, &mut V) -> Result<(), PharmsolError>,
    R: Rng + ?Sized,
{
    drift_term.fill(0.0);
    drift(t, x, drift_term)?;
    diffusion_term.fill(0.0);
    diffusion(t, x, diffusion_term)?;

    let sqrt_dt = dt.sqrt();
    let x = x.as_mut_slice();
    for i in 0..x.len() {
        let noise: f64 = StandardNormal.sample(rng);
        x[i] += drift_term[i] * dt + diffusion_term[i] * noise * sqrt_dt;
    }
    Ok(())
}

/// Largest component-wise difference between two approximations, scaled by
/// the tolerance at the starting state.
fn scaled_error(start: &V, full_step: &V, half_steps: &V) -> f64 {
    let mut error = 0.0f64;
    for i in 0..start.len() {
        let tolerance = ATOL + RTOL * start[i].abs();
        error = error.max((full_step[i] - half_steps[i]).abs() / tolerance);
    }
    error
}

fn next_step(dt: f64, error: f64) -> f64 {
    (dt * SAFETY * (1.0 / error).powf(0.5)).clamp(MIN_STEP, MAX_STEP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn decay(_t: f64, x: &V, dx: &mut V) -> Result<(), PharmsolError> {
        dx[0] = -x[0];
        Ok(())
    }

    fn no_noise(_t: f64, _x: &V, _sigma: &mut V) -> Result<(), PharmsolError> {
        Ok(())
    }

    #[test]
    fn short_intervals_stop_at_the_end_time() {
        let mut x = V::from_vec(vec![1.0], NalgebraContext::new());
        let mut scratch = EmScratch::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        advance(
            &mut decay,
            &mut no_noise,
            &mut x,
            0.0,
            0.01,
            &mut rng,
            &mut scratch,
        )
        .unwrap();
        // One clamped Euler step of 0.01 (two halves of 0.005).
        let expected = (1.0 - 0.005) * (1.0 - 0.005);
        assert!((x[0] - expected).abs() < 1e-12, "{}", x[0]);
    }

    #[test]
    fn errors_from_the_model_propagate() {
        let mut x = V::from_vec(vec![1.0], NalgebraContext::new());
        let mut scratch = EmScratch::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let mut failing =
            |_t: f64, _x: &V, _dx: &mut V| Err(PharmsolError::OtherError("drift".into()));
        let result = advance(
            &mut failing,
            &mut no_noise,
            &mut x,
            0.0,
            1.0,
            &mut rng,
            &mut scratch,
        );
        assert!(result.is_err());
    }
}
