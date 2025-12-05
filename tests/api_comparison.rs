//! Tests to verify that the old tuple-based API and new builder API produce identical results.
//!
//! This ensures backward compatibility while validating the new type-state builder pattern.
//! The new builder API enforces required fields at compile time and provides sensible defaults
//! for optional fields (lag, fa, init).

use pharmsol::prelude::models::{one_compartment, one_compartment_with_absorption};
use pharmsol::*;

const TOLERANCE: f64 = 1e-12;

/// Helper to assert that predictions from two models are identical
fn assert_predictions_match<E1: Equation, E2: Equation>(
    label: &str,
    model1: &E1,
    model2: &E2,
    subject: &Subject,
    params: &[f64],
) {
    let params_vec: Vec<f64> = params.to_vec();

    let pred1 = model1
        .estimate_predictions(subject, &params_vec)
        .expect("model1 predictions should succeed");
    let pred2 = model2
        .estimate_predictions(subject, &params_vec)
        .expect("model2 predictions should succeed");

    let preds1 = pred1.get_predictions();
    let preds2 = pred2.get_predictions();

    assert_eq!(
        preds1.len(),
        preds2.len(),
        "{}: prediction count mismatch ({} vs {})",
        label,
        preds1.len(),
        preds2.len()
    );

    for (idx, (p1, p2)) in preds1.iter().zip(preds2.iter()).enumerate() {
        let diff = (p1.prediction() - p2.prediction()).abs();
        assert!(
            diff < TOLERANCE,
            "{}: prediction {} differs (old={:.15}, new={:.15}, diff={:.2e})",
            label,
            idx,
            p1.prediction(),
            p2.prediction(),
            diff
        );
    }
}

// =============================================================================
// ODE API COMPARISON TESTS
// =============================================================================

#[test]
fn ode_builder_matches_tuple_api_one_compartment() {
    let subject = Subject::builder("ode_comparison")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // Old API with tuple
    let ode_old = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // New builder API - minimal version (only required fields)
    let ode_new = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "ode_one_compartment",
        &ode_old,
        &ode_new,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn ode_builder_matches_tuple_api_two_compartment() {
    let subject = Subject::builder("ode_two_comp")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // Old API with tuple
    let ode_old = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );

    // New builder API - minimal version
    let ode_new = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        })
        .nstates(2)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "ode_two_compartment",
        &ode_old,
        &ode_new,
        &subject,
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn ode_builder_with_neqs_struct_matches_tuple() {
    let subject = Subject::builder("ode_neqs_struct")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .build();

    // Old API with tuple
    let ode_old = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // New builder API with Neqs struct
    let ode_new = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .neqs(Neqs::new(1, 1))
        .build();

    assert_predictions_match(
        "ode_neqs_struct",
        &ode_old,
        &ode_new,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn ode_new_accepts_neqs_struct() {
    let subject = Subject::builder("ode_new_neqs")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // Old API with tuple
    let ode_tuple = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // Old API with Neqs struct (new feature!)
    let ode_neqs = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        Neqs::new(1, 1),
    );

    assert_predictions_match(
        "ode_new_with_neqs",
        &ode_tuple,
        &ode_neqs,
        &subject,
        &[0.1, 50.0],
    );
}

// =============================================================================
// ANALYTICAL API COMPARISON TESTS
// =============================================================================

#[test]
fn analytical_builder_matches_tuple_api() {
    let subject = Subject::builder("analytical_comparison")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // Old API with tuple
    let analytical_old = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // New builder API - minimal version
    let analytical_new = equation::Analytical::builder()
        .eq(one_compartment)
        .seq_eq(|_p, _t, _cov| {})
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "analytical_one_compartment",
        &analytical_old,
        &analytical_new,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn analytical_builder_matches_tuple_api_with_absorption() {
    let subject = Subject::builder("analytical_absorption")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // Old API with tuple
    let analytical_old = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );

    // New builder API - minimal version
    let analytical_new = equation::Analytical::builder()
        .eq(one_compartment_with_absorption)
        .seq_eq(|_p, _t, _cov| {})
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        })
        .nstates(2)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "analytical_with_absorption",
        &analytical_old,
        &analytical_new,
        &subject,
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn analytical_new_accepts_neqs_struct() {
    let subject = Subject::builder("analytical_neqs")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // Old API with tuple
    let analytical_tuple = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // Old API with Neqs struct
    let analytical_neqs = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        Neqs::new(1, 1),
    );

    assert_predictions_match(
        "analytical_new_with_neqs",
        &analytical_tuple,
        &analytical_neqs,
        &subject,
        &[0.1, 50.0],
    );
}

// =============================================================================
// TYPE-STATE BUILDER - MINIMAL API TESTS
// =============================================================================

#[test]
fn ode_builder_minimal_matches_full_explicit() {
    let subject = Subject::builder("minimal_vs_full")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // Minimal builder (only required fields)
    let ode_minimal = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    // Full builder (all fields explicit, using defaults)
    let ode_full = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "minimal_vs_full",
        &ode_minimal,
        &ode_full,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn analytical_builder_minimal_matches_full_explicit() {
    let subject = Subject::builder("analytical_minimal_vs_full")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // Minimal builder (only required fields)
    let analytical_minimal = equation::Analytical::builder()
        .eq(one_compartment)
        .seq_eq(|_p, _t, _cov| {})
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    // Full builder (all fields explicit)
    let analytical_full = equation::Analytical::builder()
        .eq(one_compartment)
        .seq_eq(|_p, _t, _cov| {})
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    assert_predictions_match(
        "analytical_minimal_vs_full",
        &analytical_minimal,
        &analytical_full,
        &subject,
        &[0.1, 50.0],
    );
}

// =============================================================================
// NEQS STRUCT TESTS
// =============================================================================

#[test]
fn neqs_struct_conversion_from_tuple() {
    let neqs: Neqs = (2, 3).into();
    assert_eq!(neqs.nstates, 2);
    assert_eq!(neqs.nouteqs, 3);
}

#[test]
fn neqs_struct_conversion_to_tuple() {
    let neqs = Neqs::new(4, 5);
    let tuple: (usize, usize) = neqs.into();
    assert_eq!(tuple, (4, 5));
}

#[test]
fn neqs_struct_new() {
    let neqs = Neqs::new(1, 2);
    assert_eq!(neqs.nstates, 1);
    assert_eq!(neqs.nouteqs, 2);
}

// =============================================================================
// LIKELIHOOD COMPARISON TESTS
// =============================================================================

#[test]
fn likelihood_matches_between_apis() {
    let subject = Subject::builder("likelihood_comparison")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.8, 0)
        .observation(2.0, 1.6, 0)
        .observation(4.0, 1.3, 0)
        .observation(8.0, 0.8, 0)
        .build();

    let error_models = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let params = vec![0.1, 50.0];

    // ODE: Old API
    let ode_old = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // ODE: New API - minimal version
    let ode_new = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    let ll_old = ode_old
        .estimate_likelihood(&subject, &params, &error_models, false)
        .expect("old likelihood");

    let ll_new = ode_new
        .estimate_likelihood(&subject, &params, &error_models, false)
        .expect("new likelihood");

    let diff = (ll_old - ll_new).abs();
    assert!(
        diff < TOLERANCE,
        "Likelihoods should match: old={:.15}, new={:.15}, diff={:.2e}",
        ll_old,
        ll_new,
        diff
    );
}
