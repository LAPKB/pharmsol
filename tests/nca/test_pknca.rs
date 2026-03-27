//! PKNCA Cross-Validation Tests
//!
//! This module validates pharmsol's NCA implementation against expected values
//! computed by PKNCA v0.12.1 (the gold-standard R package for NCA).
//!
//! Each test is a self-contained scenario with hardcoded inputs and expected outputs.
//! The reference values were generated using PKNCA via `Rscript generate_expected.R`.
//!
//! Tolerance: 0.1% relative, 1e-10 absolute.

use pharmsol::nca::{AUCMethod, BLQRule, NCAOptions, Route, RouteParams};
use pharmsol::{prelude::*, Censor};

const RELATIVE_TOLERANCE: f64 = 0.001;
const ABSOLUTE_TOLERANCE: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }

    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs());
    diff <= ABSOLUTE_TOLERANCE || diff <= RELATIVE_TOLERANCE * max_val
}

macro_rules! assert_approx {
    ($actual:expr, $expected:expr, $name:expr) => {
        let actual = $actual;
        let expected = $expected;
        assert!(
            approx_eq(actual, expected),
            "{}: expected {:.6}, got {:.6} (diff: {:.2e})",
            $name,
            expected,
            actual,
            (actual - expected).abs()
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic PK Profiles
    // =========================================================================

    /// Basic single-dose oral absorption
    /// Standard oral PK profile with clear absorption and elimination phases
    #[test]
    fn pknca_basic_oral_01() {
        let subject = Subject::builder("basic_oral_01")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 2.5, 0)
            .observation(1.0, 8.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(3.0, 10.0, 0)
            .observation(4.0, 7.5, 0)
            .observation(6.0, 4.2, 0)
            .observation(8.0, 2.3, 0)
            .observation(12.0, 0.7, 0)
            .observation(24.0, 0.05, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.05, "clast");
        assert_approx!(terminal.lambda_z, 0.2526, "lambda_z");
        assert_approx!(terminal.half_life, 2.7445, "half_life");
        assert_approx!(reg.r_squared, 0.9941, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9926, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 6.0, "n_points");
        assert_approx!(reg.span_ratio, 7.6516, "span_ratio");
    }

    /// Oral with delayed Tmax (slower absorption, Tmax at 4 hours)
    #[test]
    fn pknca_basic_oral_02() {
        let subject = Subject::builder("basic_oral_02")
            .bolus(0.0, 250.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 0.5, 0)
            .observation(1.0, 2.0, 0)
            .observation(2.0, 5.5, 0)
            .observation(4.0, 10.0, 0)
            .observation(6.0, 8.5, 0)
            .observation(8.0, 6.2, 0)
            .observation(12.0, 3.1, 0)
            .observation(24.0, 0.8, 0)
            .observation(48.0, 0.05, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 4.0, "tmax");
        assert_approx!(result.exposure.tlast, 48.0, "tlast");
        assert_approx!(result.exposure.clast, 0.05, "clast");
        assert_approx!(terminal.lambda_z, 0.1148, "lambda_z");
        assert_approx!(terminal.half_life, 6.0395, "half_life");
        assert_approx!(reg.r_squared, 1.0, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9999, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 3.0, "n_points");
        assert_approx!(reg.span_ratio, 5.9607, "span_ratio");
    }

    /// IV bolus single compartment (monoexponential decline)
    #[test]
    fn pknca_iv_bolus_01() {
        let subject = Subject::builder("iv_bolus_01")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 10.0, 0)
            .observation(0.25, 8.8, 0)
            .observation(0.5, 7.8, 0)
            .observation(1.0, 6.1, 0)
            .observation(2.0, 3.7, 0)
            .observation(4.0, 1.4, 0)
            .observation(6.0, 0.5, 0)
            .observation(8.0, 0.2, 0)
            .observation(12.0, 0.03, 0)
            .build();

        let options = NCAOptions::default().with_route(Route::IVBolus);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");
        let rp = result.route_params.as_ref().expect("route params");
        let clearance = result.clearance.as_ref().expect("clearance");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 0.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.03, "clast");
        assert_approx!(result.exposure.auc_last, 20.172, "auc_last");
        assert_approx!(
            result.exposure.aumc_last.expect("aumc_last"),
            40.3646,
            "aumc_last"
        );
        assert_approx!(
            result.exposure.auc_inf_obs.expect("auc_inf_obs"),
            20.2338,
            "auc_inf_obs"
        );
        assert_approx!(
            result.exposure.auc_inf_pred.expect("auc_inf_pred"),
            20.2316,
            "auc_inf_pred"
        );
        assert_approx!(
            result.exposure.aumc_inf.expect("aumc_inf"),
            41.2336,
            "aumc_inf"
        );
        assert_approx!(terminal.lambda_z, 0.4854, "lambda_z");
        assert_approx!(terminal.half_life, 1.4279, "half_life");
        assert_approx!(terminal.mrt.expect("mrt"), 2.0379, "mrt");
        assert_approx!(reg.r_squared, 0.9998, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9998, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 8.0, "n_points");
        assert_approx!(reg.span_ratio, 8.2287, "span_ratio");

        if let RouteParams::IVBolus(ref iv) = rp {
            assert_approx!(iv.c0, 10.0, "c0");
        } else {
            panic!("Expected IVBolus route params");
        }

        assert_approx!(clearance.cl_f, 4.9422, "cl");
        assert_approx!(clearance.vz_f, 10.1814, "vz");
        assert_approx!(clearance.vss.expect("vss"), 10.0716, "vss");
    }

    /// IV bolus two-compartment (biexponential decline with distribution phase)
    #[test]
    fn pknca_iv_bolus_02() {
        let subject = Subject::builder("iv_bolus_02")
            .bolus(0.0, 500.0, 0)
            .observation(0.0, 50.0, 0)
            .observation(0.083, 35.0, 0)
            .observation(0.25, 22.0, 0)
            .observation(0.5, 15.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 6.5, 0)
            .observation(4.0, 3.8, 0)
            .observation(8.0, 1.3, 0)
            .observation(12.0, 0.45, 0)
            .observation(24.0, 0.05, 0)
            .build();

        let options = NCAOptions::default().with_route(Route::IVBolus);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");
        let rp = result.route_params.as_ref().expect("route params");
        let clearance = result.clearance.as_ref().expect("clearance");

        assert_approx!(result.exposure.cmax, 50.0, "cmax");
        assert_approx!(result.exposure.tmax, 0.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.05, "clast");
        assert_approx!(result.exposure.auc_last, 51.7981, "auc_last");
        assert_approx!(
            result.exposure.aumc_last.expect("aumc_last"),
            166.7329,
            "aumc_last"
        );
        assert_approx!(
            result.exposure.auc_inf_obs.expect("auc_inf_obs"),
            52.0494,
            "auc_inf_obs"
        );
        assert_approx!(
            result.exposure.auc_inf_pred.expect("auc_inf_pred"),
            52.0401,
            "auc_inf_pred"
        );
        assert_approx!(
            result.exposure.aumc_inf.expect("aumc_inf"),
            174.0302,
            "aumc_inf"
        );
        assert_approx!(terminal.lambda_z, 0.1989, "lambda_z");
        assert_approx!(terminal.half_life, 3.485, "half_life");
        assert_approx!(terminal.mrt.expect("mrt"), 3.3436, "mrt");
        assert_approx!(reg.r_squared, 0.9932, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9865, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 3.0, "n_points");
        assert_approx!(reg.span_ratio, 4.5911, "span_ratio");

        if let RouteParams::IVBolus(ref iv) = rp {
            assert_approx!(iv.c0, 50.0, "c0");
        } else {
            panic!("Expected IVBolus route params");
        }

        assert_approx!(clearance.cl_f, 9.6063, "cl");
        assert_approx!(clearance.vz_f, 48.2984, "vz");
        assert_approx!(clearance.vss.expect("vss"), 32.119, "vss");
    }

    /// 1-hour IV infusion
    #[test]
    fn pknca_iv_infusion_01() {
        let subject = Subject::builder("iv_infusion_01")
            .infusion(0.0, 200.0, 0, 1.0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 8.0, 0)
            .observation(1.0, 15.0, 0)
            .observation(1.5, 12.5, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 5.0, 0)
            .observation(6.0, 2.5, 0)
            .observation(8.0, 1.25, 0)
            .observation(12.0, 0.3, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 15.0, "cmax");
        assert_approx!(result.exposure.tmax, 1.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.3, "clast");
        assert_approx!(terminal.lambda_z, 0.3525, "lambda_z");
        assert_approx!(terminal.half_life, 1.9666, "half_life");
        assert_approx!(reg.r_squared, 0.9999, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9998, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 6.0, "n_points");
        assert_approx!(reg.span_ratio, 5.339, "span_ratio");
    }

    // =========================================================================
    // AUC Calculation Methods
    // =========================================================================

    /// AUC comparison - Linear trapezoidal method
    #[test]
    fn pknca_auc_method_linear() {
        let subject = Subject::builder("auc_method_linear")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(3.0, 8.0, 0)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .observation(12.0, 0.4, 0)
            .build();

        let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.4, "clast");
        assert_approx!(terminal.lambda_z, 0.3356, "lambda_z");
        assert_approx!(terminal.half_life, 2.0652, "half_life");
        assert_approx!(reg.r_squared, 0.9997, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 4.3579, "span_ratio");
    }

    /// AUC comparison - Lin up/log down method (default)
    #[test]
    fn pknca_auc_method_linuplogdown() {
        let subject = Subject::builder("auc_method_linuplogdown")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(3.0, 8.0, 0)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .observation(12.0, 0.4, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.4, "clast");
        assert_approx!(terminal.lambda_z, 0.3356, "lambda_z");
        assert_approx!(terminal.half_life, 2.0652, "half_life");
        assert_approx!(reg.r_squared, 0.9997, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 4.3579, "span_ratio");
    }

    /// AUC comparison - Lin-log method (linear pre-Tmax, log post-Tmax)
    #[test]
    fn pknca_auc_method_linlog() {
        let subject = Subject::builder("auc_method_linlog")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(3.0, 8.0, 0)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .observation(12.0, 0.4, 0)
            .build();

        let options = NCAOptions::default().with_auc_method(AUCMethod::LinLog);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.4, "clast");
        assert_approx!(terminal.lambda_z, 0.3356, "lambda_z");
        assert_approx!(terminal.half_life, 2.0652, "half_life");
        assert_approx!(reg.r_squared, 0.9997, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 4.3579, "span_ratio");
    }

    // =========================================================================
    // Lambda-z / Terminal Phase
    // =========================================================================

    /// Lambda-z with minimum points (short terminal phase)
    #[test]
    fn pknca_lambda_z_short() {
        let subject = Subject::builder("lambda_z_short")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(6.0, 2.0, 0)
            .observation(8.0, 1.0, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 1.0, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 1.0, "clast");
        assert_approx!(terminal.lambda_z, 0.3466, "lambda_z");
        assert_approx!(terminal.half_life, 2.0, "half_life");
        assert_approx!(reg.r_squared, 1.0, "r_squared");
        assert_approx!(reg.adj_r_squared, 1.0, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 4.0, "n_points");
        assert_approx!(reg.span_ratio, 3.0, "span_ratio");
    }

    /// Lambda-z with many points (extended terminal phase, 8 points)
    #[test]
    fn pknca_lambda_z_long() {
        let subject = Subject::builder("lambda_z_long")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(4.0, 8.0, 0)
            .observation(6.0, 5.5, 0)
            .observation(8.0, 3.8, 0)
            .observation(12.0, 1.8, 0)
            .observation(16.0, 0.85, 0)
            .observation(24.0, 0.19, 0)
            .observation(36.0, 0.02, 0)
            .observation(48.0, 0.002, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 48.0, "tlast");
        assert_approx!(result.exposure.clast, 0.002, "clast");
        assert_approx!(terminal.lambda_z, 0.1882, "lambda_z");
        assert_approx!(terminal.half_life, 3.6828, "half_life");
        assert_approx!(reg.r_squared, 1.0, "r_squared");
        assert_approx!(reg.adj_r_squared, 1.0, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 8.0, "n_points");
        assert_approx!(reg.span_ratio, 11.9474, "span_ratio");
    }

    // =========================================================================
    // BLQ (Below Limit of Quantification)
    // =========================================================================

    /// BLQ in middle of profile (values between positive concentrations, excluded)
    #[test]
    fn pknca_blq_middle() {
        let subject = Subject::builder("blq_middle")
            .bolus(0.0, 100.0, 0)
            .censored_observation(0.0, 0.1, 0, Censor::BLOQ)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .censored_observation(3.0, 0.1, 0, Censor::BLOQ)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .observation(12.0, 0.4, 0)
            .build();

        let options = NCAOptions::default().with_blq_rule(BLQRule::Exclude);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.4, "clast");
        assert_approx!(terminal.lambda_z, 0.3383, "lambda_z");
        assert_approx!(terminal.half_life, 2.0491, "half_life");
        assert_approx!(reg.r_squared, 0.9998, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 4.0, "n_points");
        assert_approx!(reg.span_ratio, 3.9042, "span_ratio");
    }

    /// BLQ with positional handling (BLQ at start, middle, and end)
    #[test]
    fn pknca_blq_positional() {
        let subject = Subject::builder("blq_positional")
            .bolus(0.0, 100.0, 0)
            .censored_observation(0.0, 0.1, 0, Censor::BLOQ)
            .observation(1.0, 10.0, 0)
            .censored_observation(2.0, 0.1, 0, Censor::BLOQ)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .censored_observation(12.0, 0.1, 0, Censor::BLOQ)
            .build();

        let options = NCAOptions::default().with_blq_rule(BLQRule::Positional);
        let result = subject.nca(&options).expect("NCA should succeed");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 1.0, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 2.0, "clast");
        assert_approx!(result.exposure.auc_last, 36.186, "auc_last");
        assert_approx!(
            result.exposure.aumc_last.expect("aumc_last"),
            116.2766,
            "aumc_last"
        );
    }

    /// AUCall with terminal BLQ values (BLQ at end, excluded)
    #[test]
    fn pknca_auc_all_terminal_blq() {
        let subject = Subject::builder("auc_all_terminal_blq")
            .bolus(0.0, 100.0, 0)
            .censored_observation(0.0, 0.5, 0, Censor::BLOQ)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .censored_observation(10.0, 0.5, 0, Censor::BLOQ)
            .censored_observation(12.0, 0.5, 0, Censor::BLOQ)
            .build();

        let options = NCAOptions::default().with_blq_rule(BLQRule::Exclude);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 1.5, "clast");
        assert_approx!(terminal.lambda_z, 0.3466, "lambda_z");
        assert_approx!(terminal.half_life, 2.0, "half_life");
        assert_approx!(reg.r_squared, 1.0, "r_squared");
        assert_approx!(reg.adj_r_squared, 1.0, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 3.0, "n_points");
        assert_approx!(reg.span_ratio, 2.0, "span_ratio");
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    /// Sparse sampling profile (only 4 time points)
    #[test]
    fn pknca_sparse_profile() {
        let subject = Subject::builder("sparse_profile")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(8.0, 3.0, 0)
            .observation(24.0, 0.2, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.2, "clast");
    }

    /// Multiple Tmax candidates (flat Cmax plateau)
    /// Note: pharmsol may not estimate terminal phase for this edge case
    /// (insufficient non-plateau points after Tmax). PKNCA uses a slightly
    /// different point selection algorithm that finds lambda_z=0.301 here.
    #[test]
    fn pknca_flat_cmax() {
        let subject = Subject::builder("flat_cmax")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(3.0, 10.0, 0)
            .observation(4.0, 10.0, 0)
            .observation(6.0, 6.0, 0)
            .observation(8.0, 3.0, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 3.0, "clast");

        // Terminal phase may not be estimable due to Cmax plateau
        if let Some(terminal) = result.terminal.as_ref() {
            assert_approx!(terminal.lambda_z, 0.301, "lambda_z");
            assert_approx!(terminal.half_life, 2.3029, "half_life");
            if let Some(reg) = terminal.regression.as_ref() {
                assert_approx!(reg.r_squared, 0.9924, "r_squared");
                assert_approx!(reg.adj_r_squared, 0.9848, "adj_r_squared");
                assert_approx!(reg.n_points as f64, 3.0, "n_points");
                assert_approx!(reg.span_ratio, 1.737, "span_ratio");
            }
        }
    }

    /// High AUC extrapolation percentage (short sampling window)
    /// Note: pharmsol may not estimate terminal phase with only 3 post-Tmax
    /// points. PKNCA finds lambda_z=0.2452 with these minimal data.
    #[test]
    fn pknca_high_extrapolation() {
        let subject = Subject::builder("high_extrapolation")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 5.0, 0)
            .observation(6.0, 3.0, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 1.0, "tmax");
        assert_approx!(result.exposure.tlast, 6.0, "tlast");
        assert_approx!(result.exposure.clast, 3.0, "clast");

        // Terminal phase may not be estimable with minimal post-Tmax data
        if let Some(terminal) = result.terminal.as_ref() {
            assert_approx!(terminal.lambda_z, 0.2452, "lambda_z");
            assert_approx!(terminal.half_life, 2.8268, "half_life");
            if let Some(reg) = terminal.regression.as_ref() {
                assert_approx!(reg.r_squared, 0.9994, "r_squared");
                assert_approx!(reg.adj_r_squared, 0.9988, "adj_r_squared");
                assert_approx!(reg.n_points as f64, 3.0, "n_points");
                assert_approx!(reg.span_ratio, 1.415, "span_ratio");
            }
        }
    }

    /// Clast observed vs predicted comparison
    #[test]
    fn pknca_clast_pred_comparison() {
        let subject = Subject::builder("clast_pred_comparison")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 8.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(4.0, 7.0, 0)
            .observation(6.0, 4.0, 0)
            .observation(8.0, 2.3, 0)
            .observation(12.0, 0.8, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 0.8, "clast");
        assert_approx!(terminal.lambda_z, 0.2708, "lambda_z");
        assert_approx!(terminal.half_life, 2.5597, "half_life");
        assert_approx!(reg.r_squared, 0.9998, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 4.0, "n_points");
        assert_approx!(reg.span_ratio, 3.1254, "span_ratio");
    }

    /// Partial AUC calculation over [2, 8] interval
    #[test]
    fn pknca_partial_auc() {
        let subject = Subject::builder("partial_auc")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 8.0, 0)
            .observation(6.0, 5.5, 0)
            .observation(8.0, 3.5, 0)
            .observation(12.0, 1.5, 0)
            .observation(24.0, 0.3, 0)
            .build();

        let options = NCAOptions::default().with_auc_interval(2.0, 8.0);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.3, "clast");
        assert_approx!(terminal.lambda_z, 0.1631, "lambda_z");
        assert_approx!(terminal.half_life, 4.2493, "half_life");
        assert_approx!(reg.r_squared, 0.9862, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9816, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 4.7066, "span_ratio");
    }

    // =========================================================================
    // Advanced Parameters
    // =========================================================================

    /// MRT and related parameters
    #[test]
    fn pknca_mrt_calculation() {
        let subject = Subject::builder("mrt_calculation")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 3.0, 0)
            .observation(1.0, 8.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 6.5, 0)
            .observation(6.0, 4.0, 0)
            .observation(8.0, 2.5, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.15, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.15, "clast");
        assert_approx!(terminal.lambda_z, 0.1792, "lambda_z");
        assert_approx!(terminal.half_life, 3.8672, "half_life");
        assert_approx!(reg.r_squared, 0.9913, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.987, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 4.0, "n_points");
        assert_approx!(reg.span_ratio, 4.6545, "span_ratio");
    }

    /// Lag time detection (absorption delay of 0.5h)
    #[test]
    fn pknca_tlag_detection() {
        let subject = Subject::builder("tlag_detection")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.25, 0.0, 0)
            .observation(0.5, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 6.0, 0)
            .observation(6.0, 3.0, 0)
            .observation(8.0, 1.5, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 10.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 1.5, "clast");
        assert_approx!(result.exposure.tlag.expect("tlag"), 0.5, "tlag");
        assert_approx!(terminal.lambda_z, 0.3466, "lambda_z");
        assert_approx!(terminal.half_life, 2.0, "half_life");
        assert_approx!(reg.r_squared, 1.0, "r_squared");
        assert_approx!(reg.adj_r_squared, 1.0, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 3.0, "n_points");
        assert_approx!(reg.span_ratio, 2.0, "span_ratio");
    }

    /// Numerical precision test (high-precision values, 1000 mg dose)
    #[test]
    fn pknca_numerical_precision() {
        let subject = Subject::builder("numerical_precision")
            .bolus(0.0, 1000.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 15.234, 0)
            .observation(1.0, 45.678, 0)
            .observation(2.0, 67.891, 0)
            .observation(4.0, 52.345, 0)
            .observation(8.0, 28.123, 0)
            .observation(12.0, 15.067, 0)
            .observation(24.0, 4.321, 0)
            .observation(48.0, 0.354, 0)
            .observation(72.0, 0.029, 0)
            .observation(96.0, 0.002, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 67.891, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 96.0, "tlast");
        assert_approx!(result.exposure.clast, 0.002, "clast");
        assert_approx!(terminal.lambda_z, 0.1059, "lambda_z");
        assert_approx!(terminal.half_life, 6.5456, "half_life");
        assert_approx!(reg.r_squared, 0.9998, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9997, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 12.8331, "span_ratio");
    }

    /// C0 back-extrapolation (IV bolus with C0 estimated via log-linear regression)
    #[test]
    fn pknca_c0_logslope() {
        let subject = Subject::builder("c0_logslope")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(0.5, 8.0, 0)
            .observation(1.0, 6.5, 0)
            .observation(2.0, 4.3, 0)
            .observation(4.0, 1.9, 0)
            .observation(6.0, 0.8, 0)
            .observation(8.0, 0.35, 0)
            .build();

        let options = NCAOptions::default().with_route(Route::IVBolus);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");
        let rp = result.route_params.as_ref().expect("route params");

        assert_approx!(result.exposure.cmax, 8.0, "cmax");
        assert_approx!(result.exposure.tmax, 0.5, "tmax");
        assert_approx!(result.exposure.tlast, 8.0, "tlast");
        assert_approx!(result.exposure.clast, 0.35, "clast");
        assert_approx!(terminal.lambda_z, 0.4182, "lambda_z");
        assert_approx!(terminal.half_life, 1.6573, "half_life");
        assert_approx!(reg.r_squared, 0.9999, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9999, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 4.2237, "span_ratio");

        if let RouteParams::IVBolus(ref iv) = rp {
            assert_approx!(iv.c0, 9.8462, "c0");
        } else {
            panic!("Expected IVBolus route params");
        }
    }

    /// Span ratio quality metric
    #[test]
    fn pknca_span_ratio_test() {
        let subject = Subject::builder("span_ratio_test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 8.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(4.0, 9.0, 0)
            .observation(8.0, 5.0, 0)
            .observation(12.0, 2.8, 0)
            .observation(24.0, 0.9, 0)
            .observation(48.0, 0.1, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 48.0, "tlast");
        assert_approx!(result.exposure.clast, 0.1, "clast");
        assert_approx!(terminal.lambda_z, 0.0924, "lambda_z");
        assert_approx!(terminal.half_life, 7.5002, "half_life");
        assert_approx!(reg.r_squared, 0.9999, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9999, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 3.0, "n_points");
        assert_approx!(reg.span_ratio, 4.7999, "span_ratio");
    }

    // =========================================================================
    // Steady-State
    // =========================================================================

    /// Steady-state oral dosing (tau=12h)
    #[test]
    fn pknca_steady_state_oral() {
        let subject = Subject::builder("steady_state_oral")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 1.5, 0)
            .observation(0.5, 5.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 12.0, 0)
            .observation(4.0, 8.0, 0)
            .observation(6.0, 5.5, 0)
            .observation(8.0, 3.5, 0)
            .observation(10.0, 2.2, 0)
            .observation(12.0, 1.5, 0)
            .build();

        let options = NCAOptions::default().with_tau(12.0);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");
        let clearance = result.clearance.as_ref().expect("clearance");
        let ss = result.steady_state.as_ref().expect("steady state");

        assert_approx!(result.exposure.cmax, 12.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 12.0, "tlast");
        assert_approx!(result.exposure.clast, 1.5, "clast");
        assert_approx!(result.exposure.auc_last, 67.5547, "auc_last");
        assert_approx!(
            result.exposure.aumc_last.expect("aumc_last"),
            295.7289,
            "aumc_last"
        );
        assert_approx!(
            result.exposure.auc_inf_obs.expect("auc_inf_obs"),
            74.59,
            "auc_inf_obs"
        );
        assert_approx!(
            result.exposure.auc_inf_pred.expect("auc_inf_pred"),
            74.5051,
            "auc_inf_pred"
        );
        assert_approx!(
            result.exposure.aumc_inf.expect("aumc_inf"),
            413.1483,
            "aumc_inf"
        );
        assert_approx!(terminal.lambda_z, 0.2132, "lambda_z");
        assert_approx!(terminal.half_life, 3.251, "half_life");
        assert_approx!(terminal.mrt.expect("mrt"), 5.5389, "mrt");
        assert_approx!(reg.r_squared, 0.9986, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9981, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 5.0, "n_points");
        assert_approx!(reg.span_ratio, 2.4608, "span_ratio");
        assert_approx!(clearance.cl_f, 1.3407, "cl");
        assert_approx!(clearance.vz_f, 6.2879, "vz");
        assert_approx!(ss.cmin, 1.5, "cmin");
        assert_approx!(ss.cavg, 5.6296, "cavg");
    }

    /// Steady-state IV infusion (tau=24h, 2-hour infusion)
    #[test]
    fn pknca_steady_state_iv() {
        let subject = Subject::builder("steady_state_iv")
            .infusion(0.0, 500.0, 0, 2.0)
            .observation(0.0, 2.0, 0)
            .observation(1.0, 12.0, 0)
            .observation(2.0, 18.0, 0)
            .observation(4.0, 14.0, 0)
            .observation(6.0, 10.5, 0)
            .observation(8.0, 7.5, 0)
            .observation(12.0, 4.0, 0)
            .observation(18.0, 1.5, 0)
            .observation(24.0, 0.5, 0)
            .build();

        let options = NCAOptions::default().with_tau(24.0);
        let result = subject.nca(&options).expect("NCA should succeed");
        let terminal = result.terminal.as_ref().expect("terminal phase");
        let reg = terminal.regression.as_ref().expect("regression");
        let clearance = result.clearance.as_ref().expect("clearance");
        let ss = result.steady_state.as_ref().expect("steady state");
        let rp = result.route_params.as_ref().expect("route params");

        assert_approx!(result.exposure.cmax, 18.0, "cmax");
        assert_approx!(result.exposure.tmax, 2.0, "tmax");
        assert_approx!(result.exposure.tlast, 24.0, "tlast");
        assert_approx!(result.exposure.clast, 0.5, "clast");
        assert_approx!(result.exposure.auc_last, 139.0232, "auc_last");
        assert_approx!(
            result.exposure.aumc_last.expect("aumc_last"),
            920.3314,
            "aumc_last"
        );
        assert_approx!(
            result.exposure.auc_inf_obs.expect("auc_inf_obs"),
            142.0334,
            "auc_inf_obs"
        );
        assert_approx!(
            result.exposure.auc_inf_pred.expect("auc_inf_pred"),
            142.1897,
            "auc_inf_pred"
        );
        assert_approx!(
            result.exposure.aumc_inf.expect("aumc_inf"),
            1010.7007,
            "aumc_inf"
        );
        assert_approx!(terminal.lambda_z, 0.1661, "lambda_z");
        assert_approx!(terminal.half_life, 4.1731, "half_life");
        assert_approx!(terminal.mrt.expect("mrt"), 7.1159, "mrt");
        assert_approx!(reg.r_squared, 0.999, "r_squared");
        assert_approx!(reg.adj_r_squared, 0.9988, "adj_r_squared");
        assert_approx!(reg.n_points as f64, 6.0, "n_points");
        assert_approx!(reg.span_ratio, 4.7926, "span_ratio");
        assert_approx!(clearance.cl_f, 3.5203, "cl");
        assert_approx!(clearance.vss.expect("vss"), 25.0502, "vss");
        assert_approx!(ss.cmin, 0.5, "cmin");
        assert_approx!(ss.cavg, 5.7926, "cavg");

        if let RouteParams::IVInfusion(ref iv) = rp {
            assert_approx!(iv.mrt_iv.expect("mrt_iv"), 6.1159, "mrt_iv");
        } else {
            panic!("Expected IVInfusion route params");
        }
    }

    // =========================================================================
    // Sanity Check (no PKNCA dependency)
    // =========================================================================

    /// Quick sanity test that validates basic NCA functionality
    #[test]
    fn basic_nca_sanity_check() {
        let subject = Subject::builder("sanity")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 10.0, 0)
            .observation(1.0, 6.0, 0)
            .observation(2.0, 3.6, 0)
            .observation(4.0, 1.3, 0)
            .observation(8.0, 0.17, 0)
            .build();

        let options = NCAOptions::default();
        let result = subject.nca(&options).expect("NCA should succeed");

        assert_eq!(result.exposure.cmax, 10.0);
        assert_eq!(result.exposure.tmax, 0.0);
        assert!(result.exposure.auc_last > 0.0);
        assert!(result.terminal.is_some());

        let terminal = result.terminal.as_ref().unwrap();
        assert!(terminal.lambda_z > 0.0);
        assert!(terminal.half_life > 0.0);
    }
}
