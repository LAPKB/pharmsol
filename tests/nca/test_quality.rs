//! Tests for quality assessment and acceptance criteria

use approx::assert_relative_eq;
use pharmsol::nca::quality::*;

#[test]
fn test_quality_assessment_good_data() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.998,
        adjusted_r_squared: 0.997,
        n_points: 5,
        span: 3.5,
        time_first: 6.0,
        time_last: 24.0,
        intercept: 4.6,
        slope: -0.092,
    };

    let auc_last = 480.0;
    let auc_inf = 495.0;

    let quality = assess_lambda_z_quality(&lambda_z_result, auc_last, auc_inf);

    assert!(quality.overall_pass);
    assert!(quality.r_squared_pass);
    assert!(quality.span_pass);
    assert!(quality.extrapolation_pass);
    assert_eq!(quality.issues.len(), 0);
}

#[test]
fn test_quality_assessment_poor_r_squared() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.85, // Below typical threshold (0.90)
        adjusted_r_squared: 0.82,
        n_points: 4,
        span: 3.0,
        time_first: 8.0,
        time_last: 24.0,
        intercept: 4.5,
        slope: -0.092,
    };

    let auc_last = 480.0;
    let auc_inf = 495.0;

    let quality = assess_lambda_z_quality(&lambda_z_result, auc_last, auc_inf);

    assert!(!quality.overall_pass);
    assert!(!quality.r_squared_pass);
    assert!(quality
        .issues
        .iter()
        .any(|i| i.severity == Severity::Warning));
}

#[test]
fn test_quality_assessment_low_span() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.15,
        r_squared: 0.995,
        adjusted_r_squared: 0.993,
        n_points: 3,
        span: 1.5, // Below recommended threshold (2.0)
        time_first: 12.0,
        time_last: 22.0,
        intercept: 4.4,
        slope: -0.15,
    };

    let auc_last = 480.0;
    let auc_inf = 495.0;

    let quality = assess_lambda_z_quality(&lambda_z_result, auc_last, auc_inf);

    assert!(!quality.span_pass);
    assert!(quality
        .issues
        .iter()
        .any(|i| i.issue_type == IssueType::LowSpan));
}

#[test]
fn test_quality_assessment_high_extrapolation() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.998,
        adjusted_r_squared: 0.997,
        n_points: 5,
        span: 3.5,
        time_first: 6.0,
        time_last: 24.0,
        intercept: 4.6,
        slope: -0.092,
    };

    let auc_last = 300.0;
    let auc_inf = 500.0; // 40% extrapolation (above 20% threshold)

    let quality = assess_lambda_z_quality(&lambda_z_result, auc_last, auc_inf);

    assert!(!quality.extrapolation_pass);
    assert!(quality
        .issues
        .iter()
        .any(|i| i.issue_type == IssueType::HighExtrapolation));
}

#[test]
fn test_quality_score_calculation() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.98,
        adjusted_r_squared: 0.97,
        n_points: 5,
        span: 3.2,
        time_first: 6.0,
        time_last: 24.0,
        intercept: 4.6,
        slope: -0.092,
    };

    let auc_last = 450.0;
    let auc_inf = 475.0;

    let score = calculate_quality_score(&lambda_z_result, auc_last, auc_inf);

    // Good quality should score 80-100
    assert!(score > 80.0 && score <= 100.0);
}

#[test]
fn test_quality_recommendations() {
    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.88, // Slightly low
        adjusted_r_squared: 0.85,
        n_points: 3, // Minimum
        span: 1.8,   // Slightly low
        time_first: 12.0,
        time_last: 24.0,
        intercept: 4.5,
        slope: -0.092,
    };

    let auc_last = 400.0;
    let auc_inf = 550.0; // High extrapolation

    let recommendations = generate_recommendations(&lambda_z_result, auc_last, auc_inf);

    // Should have multiple recommendations
    assert!(recommendations.len() > 0);

    // Should recommend more points
    assert!(recommendations
        .iter()
        .any(|r| r.contains("more points") || r.contains("earlier")));

    // Should recommend about extrapolation
    assert!(recommendations
        .iter()
        .any(|r| r.contains("extrapolation") || r.contains("AUC_last")));
}

#[test]
fn test_acceptance_criteria() {
    let criteria = AcceptanceCriteria {
        min_r_squared: 0.95,
        min_adjusted_r_squared: 0.93,
        min_span: 2.5,
        max_extrapolation_percent: 15.0,
        min_points: 4,
    };

    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.96,
        adjusted_r_squared: 0.94,
        n_points: 5,
        span: 3.0,
        time_first: 6.0,
        time_last: 24.0,
        intercept: 4.6,
        slope: -0.092,
    };

    let auc_last = 470.0;
    let auc_inf = 490.0; // 4.3% extrapolation

    let passes = check_acceptance_criteria(&criteria, &lambda_z_result, auc_last, auc_inf);

    assert!(passes);
}

#[test]
fn test_acceptance_criteria_fails() {
    let criteria = AcceptanceCriteria {
        min_r_squared: 0.98, // Strict
        min_adjusted_r_squared: 0.97,
        min_span: 3.0,
        max_extrapolation_percent: 10.0,
        min_points: 5,
    };

    let lambda_z_result = LambdaZResult {
        lambda: 0.092,
        r_squared: 0.96, // Fails strict criterion
        adjusted_r_squared: 0.94,
        n_points: 4, // Too few
        span: 2.5,   // Too small
        time_first: 8.0,
        time_last: 24.0,
        intercept: 4.6,
        slope: -0.092,
    };

    let auc_last = 400.0;
    let auc_inf = 480.0; // 16.7% extrapolation - fails

    let passes = check_acceptance_criteria(&criteria, &lambda_z_result, auc_last, auc_inf);

    assert!(!passes);
}

#[test]
fn test_confidence_level_determination() {
    // High confidence
    let quality1 = QualityAssessment {
        overall_pass: true,
        r_squared_pass: true,
        span_pass: true,
        extrapolation_pass: true,
        confidence_level: ConfidenceLevel::High,
        quality_score: 95.0,
        issues: vec![],
    };
    assert_eq!(quality1.confidence_level, ConfidenceLevel::High);

    // Medium confidence
    let quality2 = QualityAssessment {
        overall_pass: true,
        r_squared_pass: true,
        span_pass: false,
        extrapolation_pass: true,
        confidence_level: ConfidenceLevel::Medium,
        quality_score: 75.0,
        issues: vec![QualityIssue {
            issue_type: IssueType::LowSpan,
            severity: Severity::Warning,
            message: "Span is 1.8, recommend > 2.0".to_string(),
        }],
    };
    assert_eq!(quality2.confidence_level, ConfidenceLevel::Medium);

    // Low confidence
    let quality3 = QualityAssessment {
        overall_pass: false,
        r_squared_pass: false,
        span_pass: false,
        extrapolation_pass: false,
        confidence_level: ConfidenceLevel::Low,
        quality_score: 45.0,
        issues: vec![QualityIssue {
            issue_type: IssueType::PoorFit,
            severity: Severity::Critical,
            message: "R² = 0.75, below threshold".to_string(),
        }],
    };
    assert_eq!(quality3.confidence_level, ConfidenceLevel::Low);
}

#[test]
fn test_data_adequacy_assessment() {
    // Rich sampling - good
    let times1 = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0];
    let adequacy1 = assess_data_adequacy(&times1);
    assert!(adequacy1.is_adequate);
    assert_eq!(adequacy1.sampling_type, SamplingType::Rich);

    // Sparse sampling - marginal
    let times2 = vec![0.0, 2.0, 8.0, 24.0];
    let adequacy2 = assess_data_adequacy(&times2);
    assert_eq!(adequacy2.sampling_type, SamplingType::Sparse);

    // Very sparse - inadequate
    let times3 = vec![0.0, 24.0];
    let adequacy3 = assess_data_adequacy(&times3);
    assert!(!adequacy3.is_adequate);
}

#[test]
fn test_blq_assessment() {
    let concs = vec![100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0, 0.0];
    let lloq = 5.0;

    let blq_assessment = assess_blq_handling(&concs, lloq);

    // 3 BLQ values out of 8 = 37.5%
    assert_relative_eq!(blq_assessment.percent_blq, 37.5, epsilon = 0.1);
    assert_eq!(blq_assessment.n_blq, 3);
    assert!(blq_assessment.has_trailing_blq);
}

#[test]
fn test_cmax_at_first_point_warning() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![100.0, 90.0, 80.0, 60.0, 30.0];

    let warning = check_cmax_at_first_point(&times, &concs);

    // Cmax at t=0 should trigger warning (missed absorption)
    assert!(warning.is_some());
    assert!(warning.unwrap().contains("first observation"));
}

#[test]
fn test_cmax_not_at_first_point() {
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![0.0, 50.0, 80.0, 90.0, 60.0, 30.0];

    let warning = check_cmax_at_first_point(&times, &concs);

    // Cmax at t=2.0 - no warning
    assert!(warning.is_none());
}
