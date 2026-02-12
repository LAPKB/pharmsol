//! Bioavailability and cross-comparison NCA functions
//!
//! Computes bioavailability (F) from crossover study designs where the same
//! subject receives both test and reference formulations (or IV vs oral).
//!
//! F = (AUC_test / Dose_test) / (AUC_ref / Dose_ref)

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
/// let oral_result = oral_subject.nca_first(&NCAOptions::default(), 0)?;
/// let iv_result = iv_subject.nca_first(&NCAOptions::default(), 0)?;
///
/// if let Some(f) = bioavailability(&oral_result, &iv_result) {
///     println!("Absolute bioavailability: {:.1}%", f.f_auc_inf.unwrap_or(f.f_auc_last) * 100.0);
/// }
/// ```
pub fn bioavailability(test: &NCAResult, reference: &NCAResult) -> Option<BioavailabilityResult> {
    let test_dose = test.dose.as_ref().filter(|d| d.amount > 0.0)?.amount;
    let ref_dose = reference.dose.as_ref().filter(|d| d.amount > 0.0)?.amount;

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
        let oral_result = oral.nca_first(&opts, 0).unwrap();
        let iv_result = iv.nca_first(&opts, 0).unwrap();

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
        let result = subject.nca_first(&opts, 0).unwrap();

        assert!(bioavailability(&result, &result).is_none());
    }
}
