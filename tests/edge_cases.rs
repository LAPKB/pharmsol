//! Unusual models and subjects: each must give a result or an error, never a
//! panic or a hang, and every equation family must agree on what is invalid.

use std::collections::HashMap;

use ndarray::array;
use pharmsol::prelude::simulator::log_likelihood_matrix;
use pharmsol::prelude::*;
use pharmsol::{equation, Analytical, Cache, Predictions, ODE, SDE};

fn ode() -> ODE {
    ODE::new(
        |x, p, _t, dx, b, r, _c| {
            dx[0] = -p[0] * x[0] + r[0] + b[0];
        },
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c, _x| {},
        |x, p, _t, _c, y| {
            y[0] = x[0] / p[1];
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
}

fn analytical(ndrugs: usize) -> Analytical {
    Analytical::new(
        equation::one_compartment,
        |_p, _t, _c| {},
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c, _x| {},
        |x, p, _t, _c, y| {
            y[0] = x[0] / p[1];
        },
    )
    .with_nstates(1)
    .with_ndrugs(ndrugs)
    .with_nout(1)
}

/// Diffusion `sigma` is the third parameter.
fn sde(nparticles: usize) -> SDE {
    SDE::new(
        |x, p, _t, dx, r, _c| {
            dx[0] = -p[0] * x[0] + r[0];
        },
        |p, d| {
            d[0] = p[2];
        },
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c, _x| {},
        |x, p, _t, _c, y| {
            y[0] = x[0] / p[1];
        },
        nparticles,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
}

const P: [f64; 3] = [0.5, 10.0, 0.0];

fn error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

fn assert_err_contains<T: std::fmt::Debug>(result: Result<T, PharmsolError>, needle: &str) {
    let error = result.expect_err("expected an error");
    assert!(
        error.to_string().contains(needle),
        "`{error}` does not mention `{needle}`"
    );
}

#[test]
fn zero_duration_infusions_are_rejected_by_every_family() {
    let subject = Subject::builder("s")
        .infusion(0.0, 100.0, 0, 0.0)
        .observation(1.0, 1.0, 0)
        .build();
    assert_err_contains(ode().estimate_predictions(&subject, &P), "positive");
    assert_err_contains(analytical(1).estimate_predictions(&subject, &P), "positive");
    assert_err_contains(sde(4).estimate_predictions(&subject, &P), "positive");
}

#[test]
fn non_finite_lag_is_rejected_by_analytical_models_too() {
    let subject = Subject::builder("s")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.0, 0)
        .build();
    let analytical = Analytical::new(
        equation::one_compartment,
        |_p, _t, _c| {},
        |_p, _t, _c| HashMap::from([(0usize, f64::NAN)]),
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c, _x| {},
        |x, p, _t, _c, y| {
            y[0] = x[0] / p[1];
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1);
    assert_err_contains(analytical.estimate_predictions(&subject, &P), "not finite");
}

#[test]
fn out_of_range_labels_are_errors_not_panics() {
    let bad_input = Subject::builder("s")
        .bolus(0.0, 100.0, 5)
        .observation(1.0, 1.0, 0)
        .build();
    let bad_output = Subject::builder("s")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.0, 3)
        .build();
    for subject in [&bad_input, &bad_output] {
        assert!(ode().estimate_predictions(subject, &P).is_err());
        assert!(analytical(1).estimate_predictions(subject, &P).is_err());
        assert!(sde(4).estimate_predictions(subject, &P).is_err());
    }
}

#[test]
fn bolus_into_a_missing_state_is_an_error() {
    // Two inputs but one state: input 1 has nowhere to land.
    let subject = Subject::builder("s")
        .bolus(0.0, 100.0, 1)
        .observation(1.0, 1.0, 0)
        .build();
    assert_err_contains(
        analytical(2).estimate_predictions(&subject, &P),
        "no state to land in",
    );
}

#[test]
fn too_few_parameters_for_declared_metadata_is_an_error() {
    let ode = ode()
        .with_metadata(
            equation::metadata::new("one_cmt")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .route(equation::Route::bolus("iv").to_state("central")),
        )
        .unwrap();
    let subject = Subject::builder("s")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 1.0, "cp")
        .build();
    assert_err_contains(ode.estimate_predictions(&subject, &[0.5]), "declares 2");
}

#[test]
fn a_model_without_states_still_produces_outputs() {
    let constant = ODE::new(
        |_x, _p, _t, _dx, _b, _r, _c| {},
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c| HashMap::new(),
        |_p, _t, _c, _x| {},
        |_x, _p, t, _c, y| {
            y[0] = 2.0 * t;
        },
    )
    .with_nstates(0)
    .with_ndrugs(0)
    .with_nout(1);
    let subject = Subject::builder("s")
        .observation(1.0, 1.0, 0)
        .observation(3.0, 1.0, 0)
        .build();
    let predictions = constant.estimate_predictions(&subject, &[]).unwrap();
    let values: Vec<f64> = predictions
        .predictions()
        .iter()
        .map(|p| p.prediction())
        .collect();
    assert_eq!(values, vec![2.0, 6.0]);
}

#[test]
fn an_sde_without_particles_is_an_error() {
    let subject = Subject::builder("s")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.0, 0)
        .build();
    let model = sde(0);
    assert_err_contains(model.estimate_predictions(&subject, &P), "particle");
    assert_err_contains(
        model.estimate_log_likelihood(&subject, &P, &error_models()),
        "particle",
    );
}

#[test]
fn sde_steps_never_cross_a_tiny_interval() {
    let subject = Subject::builder("s")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.0, 0)
        .observation(1.0 + 1e-12, 1.0, 0)
        .build();
    let predictions = sde(4).estimate_predictions(&subject, &P).unwrap();
    let means = predictions.get_predictions();
    assert!((means[0].prediction() - means[1].prediction()).abs() < 1e-9);
}

#[test]
fn sde_likelihood_matrix_with_duplicate_subjects_and_points_completes() {
    // Identical subjects and support points share cache keys while the
    // particle filter runs parallel work; this must neither deadlock nor
    // disagree with the direct computation.
    let subject = Subject::builder("dup")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 8.0, 0)
        .observation(1.0, 6.0, 0)
        .build();
    let data = Data::new(vec![subject.clone(); 6]);
    let points = array![[0.5, 10.0, 0.3], [0.5, 10.0, 0.3], [0.4, 10.0, 0.3]];
    let model = sde(256);

    let matrix = log_likelihood_matrix(&model, &data, &points, &error_models(), false).unwrap();
    let direct = model
        .clone()
        .disable_cache()
        .estimate_log_likelihood(&subject, &[0.5, 10.0, 0.3], &error_models())
        .unwrap();
    for row in 0..6 {
        assert_eq!(matrix[[row, 0]], direct);
        assert_eq!(matrix[[row, 1]], direct);
    }
}
