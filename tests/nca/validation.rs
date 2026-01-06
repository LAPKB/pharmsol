//! Validation framework for NCA algorithms
//!
//! This module provides utilities for validating NCA calculations against
//! reference implementations (PKanalix, etc.) and known correct results.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a validation dataset with expected results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDataset {
    pub name: String,
    pub description: String,
    pub reference_tool: String,
    pub date_generated: String,
    pub subjects: Vec<SubjectValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectValidation {
    pub id: String,
    pub data: SubjectData,
    pub settings: AnalysisSettings,
    pub expected_parameters: HashMap<String, ExpectedParameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectData {
    pub times: Vec<f64>,
    pub concentrations: Vec<f64>,
    pub dose: f64,
    pub route: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSettings {
    pub lambda_z_method: String,
    pub lambda_z_range: Option<(f64, f64)>,
    pub auc_method: String,
    pub dose: f64,
    pub route: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedParameter {
    pub value: f64,
    pub unit: String,
    pub tolerance: f64,                  // Absolute tolerance
    pub relative_tolerance: Option<f64>, // Relative tolerance (%)
}

#[derive(Debug)]
pub struct ValidationResult {
    pub subject_id: String,
    pub parameter: String,
    pub expected: f64,
    pub actual: f64,
    pub difference: f64,
    pub percent_diff: f64,
    pub passed: bool,
    pub tolerance: f64,
}

impl ValidationResult {
    pub fn new(
        subject_id: String,
        parameter: String,
        expected: f64,
        actual: f64,
        tolerance: f64,
        relative_tolerance: Option<f64>,
    ) -> Self {
        let difference = actual - expected;
        let percent_diff = if expected != 0.0 {
            (difference / expected) * 100.0
        } else {
            0.0
        };

        // Check both absolute and relative tolerance
        let passed = if let Some(rel_tol) = relative_tolerance {
            difference.abs() <= tolerance || percent_diff.abs() <= rel_tol
        } else {
            difference.abs() <= tolerance
        };

        Self {
            subject_id,
            parameter,
            expected,
            actual,
            difference,
            percent_diff,
            passed,
            tolerance,
        }
    }
}

/// Load a validation dataset from JSON
pub fn load_validation_dataset(
    path: &str,
) -> Result<ValidationDataset, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let dataset: ValidationDataset = serde_json::from_str(&content)?;
    Ok(dataset)
}

/// Compare calculated results with expected values
pub fn compare_results(
    subject_id: &str,
    expected: &HashMap<String, ExpectedParameter>,
    actual: &HashMap<String, f64>,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();

    for (param, exp) in expected {
        if let Some(&actual_value) = actual.get(param) {
            let result = ValidationResult::new(
                subject_id.to_string(),
                param.clone(),
                exp.value,
                actual_value,
                exp.tolerance,
                exp.relative_tolerance,
            );
            results.push(result);
        }
    }

    results
}

/// Generate a validation report
pub fn generate_report(results: &[ValidationResult]) -> String {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let mut report = String::new();
    report.push_str(&format!("Validation Report\n"));
    report.push_str(&format!("=================\n\n"));
    report.push_str(&format!("Total tests: {}\n", total));
    report.push_str(&format!(
        "Passed: {} ({:.1}%)\n",
        passed,
        (passed as f64 / total as f64) * 100.0
    ));
    report.push_str(&format!(
        "Failed: {} ({:.1}%)\n\n",
        failed,
        (failed as f64 / total as f64) * 100.0
    ));

    if failed > 0 {
        report.push_str("Failed Tests:\n");
        report.push_str("-------------\n");
        for result in results.iter().filter(|r| !r.passed) {
            report.push_str(&format!(
                "  {} [{}]: Expected={:.6}, Actual={:.6}, Diff={:.6} ({:.2}%), Tolerance={:.6}\n",
                result.subject_id,
                result.parameter,
                result.expected,
                result.actual,
                result.difference,
                result.percent_diff,
                result.tolerance
            ));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_absolute_tolerance() {
        let result = ValidationResult::new(
            "001".to_string(),
            "AUC_last".to_string(),
            100.0,
            100.05,
            0.1,
            None,
        );

        assert!(result.passed);
        assert_eq!(result.difference, 0.05);
        assert!((result.percent_diff - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_validation_result_relative_tolerance() {
        let result = ValidationResult::new(
            "001".to_string(),
            "AUC_last".to_string(),
            100.0,
            100.2,
            0.05,      // Absolute tolerance (would fail)
            Some(0.5), // Relative tolerance 0.5% (should pass)
        );

        assert!(result.passed);
        assert_eq!(result.difference, 0.2);
        assert!((result.percent_diff - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_validation_result_fails() {
        let result = ValidationResult::new(
            "001".to_string(),
            "AUC_last".to_string(),
            100.0,
            102.0,
            0.1,
            Some(0.5),
        );

        assert!(!result.passed);
        assert_eq!(result.difference, 2.0);
        assert!((result.percent_diff - 2.0).abs() < 1e-10);
    }
}
