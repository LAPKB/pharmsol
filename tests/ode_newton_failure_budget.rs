//! The Newton-failure budget is per event-to-event interval.
//!
//! diffsol counts Newton failures over a solver's whole lifetime, and one
//! solver integrates a whole occasion, so long dosing histories used to abort
//! on failures that each interval had recovered from.

use pharmsol::{prelude::*, Parameters};

fn implicit_solvers() -> [(&'static str, OdeSolver); 3] {
    [
        ("BDF", OdeSolver::Bdf),
        ("TRBDF2", OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
        ("ESDIRK34", OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
    ]
}

fn predict(
    model: &equation::ODE,
    subject: &Subject,
    parameters: &[(&str, f64)],
) -> Result<Vec<f64>, PharmsolError> {
    let parameters = Parameters::with_model(model, parameters.iter().copied())
        .expect("parameters should validate");
    model
        .predict(subject, parameters.as_slice())
        .map(|predictions| predictions.flat_predictions())
}

fn failure(model: &equation::ODE, subject: &Subject, parameters: &[(&str, f64)]) -> String {
    predict(model, subject, parameters)
        .expect_err("the solver should give up")
        .to_string()
}

/// One compartment with Michaelis-Menten elimination and a tiny Km: mildly
/// stiff, so the solver recovers from an occasional Newton failure.
fn michaelis_menten() -> equation::ODE {
    ode! {
        name: "michaelis_menten",
        params: [vmax, km, v],
        states: [central],
        outputs: [cp],
        diffeq: |x, _t, dx| {
            dx[central] = bolus[dose] - vmax * x[central] / (km + x[central]);
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

const MICHAELIS_MENTEN: [(&str, f64); 3] = [("vmax", 200.0), ("km", 0.01), ("v", 5.0)];

/// 500 every 12 h for 60 doses, observed every 0.25 h.
fn long_dosing_history() -> Subject {
    Subject::builder("long_history")
        .bolus(0.0, 500.0, "dose")
        .repeat(59, 12.0)
        .missing_observation(0.25, "cp")
        .repeat(2879, 0.25)
        .build()
}

#[test]
fn newton_failures_do_not_accumulate_across_intervals() {
    let subject = long_dosing_history();
    let reference = predict(
        &michaelis_menten().with_tolerances(1e-8, 1e-10),
        &subject,
        &MICHAELIS_MENTEN,
    )
    .expect("tight-tolerance reference should solve");

    for (name, solver) in implicit_solvers() {
        let predictions = predict(
            &michaelis_menten().with_solver(solver),
            &subject,
            &MICHAELIS_MENTEN,
        )
        .unwrap_or_else(|error| panic!("{name}: 60 doses should solve: {error}"));
        assert_eq!(predictions.len(), reference.len(), "{name}");
        for (index, (actual, expected)) in predictions.iter().zip(&reference).enumerate() {
            // 1e-4 of the peak concentration (100), or 0.1% of the value.
            let allowed = 1e-2_f64.max(1e-3 * expected.abs());
            assert!(
                (actual - expected).abs() <= allowed,
                "{name}: prediction {index} is {actual}, reference {expected}"
            );
        }
    }
}

/// After 700 h of recovered failures (several hundred over the occasion), a
/// covariate makes Km tiny and one interval fails on its own. The failure is
/// counted and reported for that interval alone.
#[test]
fn failure_after_a_long_history_reports_its_own_interval() {
    let model = ode! {
        name: "michaelis_menten_switch",
        params: [vmax, km, v],
        covariates: [km_scale],
        states: [central],
        outputs: [cp],
        diffeq: |x, _t, dx| {
            dx[central] = bolus[dose] - vmax * x[central] / (km * km_scale + x[central].abs());
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    };
    let subject = Subject::builder("late_failure")
        .covariate("km_scale", 0.0, 1.0)
        .covariate("km_scale", 700.0, 1.0)
        .covariate("km_scale", 701.0, 1e-10)
        .bolus(0.0, 500.0, "dose")
        .repeat(59, 12.0)
        .missing_observation(0.25, "cp")
        .repeat(2879, 0.25)
        .build();

    for (name, solver) in implicit_solvers() {
        let message = failure(
            &model.clone().with_solver(solver),
            &subject,
            &MICHAELIS_MENTEN,
        );
        assert!(
            message.contains("Newton solver failed 51 times at t = 70"),
            "{name}: {message}"
        );
    }
}

#[test]
fn fast_binding_tmdd_survives_many_doses_under_tr_bdf2() {
    let model = ode! {
        name: "tmdd",
        params: [kel, kon, koff, kint, ksyn, kdeg, v],
        states: [ligand, receptor, complex],
        outputs: [cfree],
        init: |_t, x| {
            x[receptor] = ksyn / kdeg;
        },
        diffeq: |x, _t, dx| {
            let binding = kon * x[ligand] * x[receptor] - koff * x[complex];
            dx[ligand] = bolus[dose] - kel * x[ligand] - binding;
            dx[receptor] = ksyn - kdeg * x[receptor] - binding;
            dx[complex] = binding - kint * x[complex];
        },
        out: |x, _t, y| {
            y[cfree] = x[ligand] / v;
        },
    };
    let subject = Subject::builder("tmdd")
        .bolus(0.0, 100.0, "dose")
        .repeat(59, 12.0)
        .missing_observation(0.25, "cfree")
        .repeat(2879, 0.25)
        .build();
    for kon in [10.0, 1e3, 1e4] {
        let parameters = [
            ("kel", 0.1),
            ("kon", kon),
            ("koff", 0.01),
            ("kint", 0.05),
            ("ksyn", 1.0),
            ("kdeg", 0.1),
            ("v", 5.0),
        ];
        let predictions = predict(
            &model
                .clone()
                .with_solver(OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
            &subject,
            &parameters,
        )
        .unwrap_or_else(|error| panic!("kon = {kon}: 60 doses should solve: {error}"));
        assert!(predictions.iter().all(|value| value.is_finite()));
    }
}
