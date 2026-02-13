//! Bioavailability and cross-comparison NCA functions
//!
//! Computes bioavailability (F) from crossover study designs where the same
//! subject receives both test and reference formulations (or IV vs oral).
//!
//! F = (AUC_test / Dose_test) / (AUC_ref / Dose_ref)
//!
//! For population-level bioequivalence assessment, [`bioequivalence()`] computes
//! the geometric mean ratio (GMR) and confidence interval from paired results.

use super::types::NCAResult;

/// Result of a bioavailability comparison
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BioavailabilityResult {
    /// Bioavailability ratio (F) based on AUCinf
    pub f_auc_inf: Option<f64>,
    /// Bioavailability ratio (F) based on AUClast
    pub f_auc_last: f64,
    /// Test AUCinf (dose-normalized)
    pub test_auc_inf_dn: Option<f64>,
    /// Reference AUCinf (dose-normalized)
    pub ref_auc_inf_dn: Option<f64>,
    /// Test AUClast (dose-normalized)
    pub test_auc_last_dn: f64,
    /// Reference AUClast (dose-normalized)
    pub ref_auc_last_dn: f64,
}

/// Calculate bioavailability (F) from two NCA results (e.g., test vs reference)
///
/// This is typically used in crossover bioequivalence studies:
/// - **F from AUCinf**: `(AUCinf_test / Dose_test) / (AUCinf_ref / Dose_ref)`
/// - **F from AUClast**: `(AUClast_test / Dose_test) / (AUClast_ref / Dose_ref)`
///
/// Both results must have dose information for meaningful computation.
///
/// # Arguments
/// * `test` - NCA result for the test formulation (or extravascular administration)
/// * `reference` - NCA result for the reference formulation (or IV administration)
///
/// # Returns
/// `None` if either result lacks dose information (dose = 0 or missing)
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::nca::{bioavailability, NCAOptions, NCA};
///
/// let oral_result = oral_subject.nca(&NCAOptions::default())?;
/// let iv_result = iv_subject.nca(&NCAOptions::default())?;
///
/// if let Some(f) = bioavailability(&oral_result, &iv_result) {
///     println!("Absolute bioavailability: {:.1}%", f.f_auc_inf.unwrap_or(f.f_auc_last) * 100.0);
/// }
/// ```
pub fn bioavailability(test: &NCAResult, reference: &NCAResult) -> Option<BioavailabilityResult> {
    let test_dose = test.dose_amount.filter(|&d| d > 0.0)?;
    let ref_dose = reference.dose_amount.filter(|&d| d > 0.0)?;

    let test_auc_last_dn = test.exposure.auc_last / test_dose;
    let ref_auc_last_dn = reference.exposure.auc_last / ref_dose;

    let f_auc_last = if ref_auc_last_dn > 0.0 {
        test_auc_last_dn / ref_auc_last_dn
    } else {
        f64::NAN
    };

    let (f_auc_inf, test_auc_inf_dn, ref_auc_inf_dn) =
        match (test.exposure.auc_inf_obs, reference.exposure.auc_inf_obs) {
            (Some(test_auc_inf), Some(ref_auc_inf)) => {
                let test_dn = test_auc_inf / test_dose;
                let ref_dn = ref_auc_inf / ref_dose;
                let f = if ref_dn > 0.0 {
                    test_dn / ref_dn
                } else {
                    f64::NAN
                };
                (Some(f), Some(test_dn), Some(ref_dn))
            }
            _ => (None, None, None),
        };

    Some(BioavailabilityResult {
        f_auc_inf,
        f_auc_last,
        test_auc_inf_dn,
        ref_auc_inf_dn,
        test_auc_last_dn,
        ref_auc_last_dn,
    })
}

/// Population-level bioequivalence assessment result
///
/// Contains geometric mean ratios and confidence intervals for both
/// AUClast and AUCinf endpoints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BioequivalenceResult {
    /// Number of evaluable pairs
    pub n: usize,
    /// Geometric mean ratio (AUClast, dose-normalized)
    pub gmr_auc_last: f64,
    /// Lower bound of CI for AUClast GMR
    pub ci_lower_auc_last: f64,
    /// Upper bound of CI for AUClast GMR
    pub ci_upper_auc_last: f64,
    /// Geometric mean ratio (AUCinf, dose-normalized) — None if not all pairs have AUCinf
    pub gmr_auc_inf: Option<f64>,
    /// Lower bound of CI for AUCinf GMR
    pub ci_lower_auc_inf: Option<f64>,
    /// Upper bound of CI for AUCinf GMR
    pub ci_upper_auc_inf: Option<f64>,
    /// Confidence level used (e.g. 0.90)
    pub ci_level: f64,
    /// Individual F values per pair (AUClast)
    pub individual_f: Vec<f64>,
}

/// Compute population-level bioequivalence from paired NCA results
///
/// Takes a slice of `(test, reference)` NCA result pairs — typically one pair
/// per subject from a crossover design. Computes:
/// - Per-pair F values via [`bioavailability()`]
/// - Geometric mean ratio: `exp(mean(ln(F_i)))`
/// - Confidence interval: `exp(mean ± t_{α/2,n-1} × SE)` on log scale
///
/// # Arguments
/// * `pairs` - Slice of (test, reference) NCA result pairs
/// * `ci_level` - Confidence level, e.g. 0.90 for 90% CI (standard for BE)
///
/// # Returns
/// `None` if fewer than 2 evaluable pairs or all F values are non-positive
///
/// # Example
/// ```rust,ignore
/// use pharmsol::nca::bioavailability::{bioequivalence, BioequivalenceResult};
///
/// let pairs: Vec<(NCAResult, NCAResult)> = subjects.iter()
///     .map(|s| (s.test_result.clone(), s.ref_result.clone()))
///     .collect();
///
/// if let Some(be) = bioequivalence(&pairs, 0.90) {
///     println!("GMR: {:.4}, 90% CI: [{:.4}, {:.4}]",
///         be.gmr_auc_last, be.ci_lower_auc_last, be.ci_upper_auc_last);
/// }
/// ```
pub fn bioequivalence(
    pairs: &[(NCAResult, NCAResult)],
    ci_level: f64,
) -> Option<BioequivalenceResult> {
    // Compute individual F values
    let f_values: Vec<f64> = pairs
        .iter()
        .filter_map(|(test, reference)| {
            bioavailability(test, reference).map(|r| r.f_auc_last)
        })
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();

    let n = f_values.len();
    if n < 2 {
        return None;
    }

    // Log-transform for GMR calculation
    let ln_f: Vec<f64> = f_values.iter().map(|f| f.ln()).collect();
    let mean_ln = ln_f.iter().sum::<f64>() / n as f64;
    let var_ln = ln_f.iter().map(|x| (x - mean_ln).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se_ln = (var_ln / n as f64).sqrt();

    // t critical value approximation (two-tailed)
    let alpha = 1.0 - ci_level;
    let t_crit = t_quantile(1.0 - alpha / 2.0, (n - 1) as f64);

    let gmr_auc_last = mean_ln.exp();
    let ci_lower_auc_last = (mean_ln - t_crit * se_ln).exp();
    let ci_upper_auc_last = (mean_ln + t_crit * se_ln).exp();

    // Same for AUCinf if all pairs have it
    let f_inf_values: Vec<f64> = pairs
        .iter()
        .filter_map(|(test, reference)| {
            bioavailability(test, reference).and_then(|r| r.f_auc_inf)
        })
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();

    let (gmr_auc_inf, ci_lower_auc_inf, ci_upper_auc_inf) = if f_inf_values.len() >= 2 {
        let n_inf = f_inf_values.len();
        let ln_f_inf: Vec<f64> = f_inf_values.iter().map(|f| f.ln()).collect();
        let mean_ln_inf = ln_f_inf.iter().sum::<f64>() / n_inf as f64;
        let var_ln_inf = ln_f_inf
            .iter()
            .map(|x| (x - mean_ln_inf).powi(2))
            .sum::<f64>()
            / (n_inf - 1) as f64;
        let se_ln_inf = (var_ln_inf / n_inf as f64).sqrt();
        let t_crit_inf = t_quantile(1.0 - alpha / 2.0, (n_inf - 1) as f64);

        (
            Some(mean_ln_inf.exp()),
            Some((mean_ln_inf - t_crit_inf * se_ln_inf).exp()),
            Some((mean_ln_inf + t_crit_inf * se_ln_inf).exp()),
        )
    } else {
        (None, None, None)
    };

    Some(BioequivalenceResult {
        n,
        gmr_auc_last,
        ci_lower_auc_last,
        ci_upper_auc_last,
        gmr_auc_inf,
        ci_lower_auc_inf,
        ci_upper_auc_inf,
        ci_level,
        individual_f: f_values,
    })
}

/// Approximate t-distribution quantile using the Abramowitz & Stegun formula
/// Student's t-distribution quantile via `statrs`
fn t_quantile(p: f64, df: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    StudentsT::new(0.0, 1.0, df).unwrap().inverse_cdf(p)
}

/// Compute metabolite-to-parent ratios from paired NCA results
///
/// Returns a HashMap with ratio names → values:
/// - `"auc_last_ratio"`: AUClast(metabolite) / AUClast(parent)
/// - `"auc_inf_ratio"`:  AUCinf(metabolite) / AUCinf(parent) (if both available)
/// - `"cmax_ratio"`:     Cmax(metabolite) / Cmax(parent)
///
/// # Arguments
/// * `parent` - NCA result for the parent compound
/// * `metabolite` - NCA result for the metabolite
///
/// # Example
/// ```rust,ignore
/// use pharmsol::nca::{metabolite_parent_ratio, NCAOptions, NCA};
///
/// let parent_result = subject.nca(&NCAOptions::default())?;
/// let metabolite_result = subject.nca(&NCAOptions::default().with_outeq(1))?;
/// let ratios = metabolite_parent_ratio(&parent_result, &metabolite_result);
/// println!("AUC ratio: {:.2}", ratios["auc_last_ratio"]);
/// ```
pub fn metabolite_parent_ratio(
    parent: &NCAResult,
    metabolite: &NCAResult,
) -> std::collections::HashMap<&'static str, f64> {
    let mut ratios = std::collections::HashMap::new();

    // AUClast ratio
    if parent.exposure.auc_last > 0.0 {
        ratios.insert(
            "auc_last_ratio",
            metabolite.exposure.auc_last / parent.exposure.auc_last,
        );
    }

    // AUCinf ratio (if both available)
    if let (Some(m_inf), Some(p_inf)) = (
        metabolite.exposure.auc_inf_obs,
        parent.exposure.auc_inf_obs,
    ) {
        if p_inf > 0.0 {
            ratios.insert("auc_inf_ratio", m_inf / p_inf);
        }
    }

    // Cmax ratio
    if parent.exposure.cmax > 0.0 {
        ratios.insert(
            "cmax_ratio",
            metabolite.exposure.cmax / parent.exposure.cmax,
        );
    }

    ratios
}

/// Compare two NCA results and return ratios (test/reference) for key parameters
///
/// Returns a HashMap with parameter names → ratio values. Uses `to_params()`
/// internally and computes test/reference for every parameter present in both.
///
/// # Arguments
/// * `test` - NCA result for the test condition
/// * `reference` - NCA result for the reference condition
///
/// # Example
/// ```rust,ignore
/// use pharmsol::nca::{compare, NCAOptions, NCA};
///
/// let ratios = compare(&test_result, &reference_result);
/// println!("AUC ratio: {:.3}", ratios["auc_last"]);
/// println!("Cmax ratio: {:.3}", ratios["cmax"]);
/// ```
pub fn compare(
    test: &NCAResult,
    reference: &NCAResult,
) -> std::collections::HashMap<&'static str, f64> {
    let test_params = test.to_params();
    let ref_params = reference.to_params();
    let mut ratios = std::collections::HashMap::new();

    for (&name, &ref_val) in &ref_params {
        if ref_val.abs() > f64::EPSILON {
            if let Some(&test_val) = test_params.get(name) {
                ratios.insert(name, test_val / ref_val);
            }
        }
    }

    ratios
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builder::SubjectBuilderExt;
    use crate::nca::{NCAOptions, NCA};
    use crate::Subject;

    #[test]
    fn test_bioavailability_basic() {
        // Oral: lower exposure, same dose
        let oral = Subject::builder("oral")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.25, 0)
            .build();

        // IV: higher exposure, same dose
        let iv = Subject::builder("iv")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 20.0, 0)
            .observation(1.0, 15.0, 0)
            .observation(2.0, 10.0, 0)
            .observation(4.0, 5.0, 0)
            .observation(8.0, 2.5, 0)
            .observation(12.0, 1.25, 0)
            .observation(24.0, 0.3, 0)
            .build();

        let opts = NCAOptions::default();
        let oral_result = oral.nca(&opts).unwrap();
        let iv_result = iv.nca(&opts).unwrap();

        let f = bioavailability(&oral_result, &iv_result).unwrap();
        assert!(f.f_auc_last > 0.0 && f.f_auc_last < 1.0, "F should be < 1 (lower oral exposure)");
        // F from AUClast is AUClast_oral / AUClast_iv (same dose)
        let expected = oral_result.exposure.auc_last / iv_result.exposure.auc_last;
        assert!((f.f_auc_last - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bioavailability_no_dose() {
        let subject = Subject::builder("no_dose")
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build();

        let opts = NCAOptions::default();
        let result = subject.nca(&opts).unwrap();

        assert!(bioavailability(&result, &result).is_none());
    }

    #[test]
    fn test_t_quantile_accuracy() {
        // Known t-distribution quantiles at p=0.975
        // (two-sided 95% critical values)
        let cases = [
            (5.0, 2.5706),
            (10.0, 2.2281),
            (30.0, 2.0423),
            (120.0, 1.9799),
        ];
        for (df, expected) in cases {
            let got = t_quantile(0.975, df);
            assert!(
                (got - expected).abs() < 0.001,
                "t(0.975, df={df}): got {got:.4}, expected {expected:.4}"
            );
        }
    }

    #[test]
    fn test_metabolite_parent_ratio() {
        // Parent: higher exposure
        let parent = Subject::builder("parent")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 20.0, 0)
            .observation(2.0, 15.0, 0)
            .observation(4.0, 8.0, 0)
            .observation(8.0, 4.0, 0)
            .observation(12.0, 2.0, 0)
            .observation(24.0, 0.5, 0)
            .build();

        // Metabolite: lower exposure
        let metabolite = Subject::builder("metabolite")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 5.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.25, 0)
            .build();

        let opts = NCAOptions::default();
        let p = parent.nca(&opts).unwrap();
        let m = metabolite.nca(&opts).unwrap();

        let ratios = metabolite_parent_ratio(&p, &m);
        assert!(ratios.contains_key("auc_last_ratio"));
        assert!(ratios.contains_key("cmax_ratio"));
        // Metabolite has lower Cmax, so ratio < 1
        assert!(*ratios.get("cmax_ratio").unwrap() < 1.0);
        assert!(*ratios.get("auc_last_ratio").unwrap() < 1.0);
    }

    #[test]
    fn test_compare() {
        let test_subj = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.25, 0)
            .build();

        let ref_subj = Subject::builder("ref")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .observation(24.0, 0.25, 0)
            .build();

        let opts = NCAOptions::default();
        let test_r = test_subj.nca(&opts).unwrap();
        let ref_r = ref_subj.nca(&opts).unwrap();

        let ratios = compare(&test_r, &ref_r);
        // Same data → all ratios should be ~1.0
        for (&name, &ratio) in &ratios {
            assert!(
                (ratio - 1.0).abs() < 1e-10,
                "ratio for {name} should be 1.0, got {ratio}"
            );
        }
        assert!(ratios.contains_key("cmax"));
        assert!(ratios.contains_key("auc_last"));
    }
}
