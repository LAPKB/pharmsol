//! Closed-form propagation of linear models, checked against the numeric
//! integrator and against hand-derived solutions.
//!
//! Models whose dynamics are linear with coefficients that hold still between
//! integration boundaries are advanced with a matrix exponential instead of an
//! ODE solver. These tests pin three things:
//!
//! - which models qualify, and why the others do not;
//! - that the closed form agrees with the numeric path within solver tolerance;
//! - that it matches exact solutions where one is known.

#![cfg(feature = "dsl")]

use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeOdeModel};
use pharmsol::{prelude::*, Cache, Parameters};

fn ode_model(source: &str) -> RuntimeOdeModel {
    match compile_module_source_to_runtime(source, None, |_kind, _message| {})
        .expect("model compiles")
    {
        CompiledRuntimeModel::Ode(model) => model,
        other => panic!("expected an ODE model, got {other:?}"),
    }
}

fn predictions(model: &RuntimeOdeModel, subject: &Subject, values: &[(&str, f64)]) -> Vec<f64> {
    let parameters = Parameters::with_model(model, values.to_vec()).expect("parameters bind");
    model
        .estimate_predictions(subject, &parameters)
        .expect("predictions")
        .flat_predictions()
        .to_vec()
}

/// Run a model through both paths and require agreement within `tolerance`.
///
/// The numeric reference is run at far tighter tolerances than the default
/// (1e-4), so this asserts that the closed form is the value the solver
/// converges towards rather than merely close to its default output.
fn assert_paths_agree(
    source: &str,
    subject: &Subject,
    values: &[(&str, f64)],
    tolerance: f64,
) -> Vec<f64> {
    let closed_form = ode_model(source).disable_cache();
    let numeric = ode_model(source)
        .force_numeric_solver()
        .with_tolerances(1e-11, 1e-11)
        .disable_cache();
    assert!(
        closed_form.uses_closed_form_for(subject),
        "model should qualify for closed-form propagation"
    );
    assert!(!numeric.uses_closed_form_for(subject));

    let from_closed_form = predictions(&closed_form, subject, values);
    let from_numeric = predictions(&numeric, subject, values);
    assert_eq!(from_closed_form.len(), from_numeric.len());
    for (index, (lhs, rhs)) in from_closed_form.iter().zip(from_numeric.iter()).enumerate() {
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            (lhs - rhs).abs() / scale < tolerance,
            "prediction {index}: closed form {lhs} vs numeric {rhs}"
        );
    }
    from_closed_form
}

const ONE_COMPARTMENT_ORAL: &str = r#"
name = one_cpt_oral
kind = ode
params = ka, ke, v
states = gut, central
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
out(cp) = central / v
"#;

const TWO_COMPARTMENT: &str = r#"
name = two_cpt
kind = ode
params = ka, ke, kcp, kpc, v
states = gut, central, periph
outputs = cp
bolus(oral) -> gut
infusion(iv) -> central
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central - kcp * central + kpc * periph
dx(periph) = kcp * central - kpc * periph
out(cp) = central / v
"#;

const MICHAELIS_MENTEN: &str = r#"
name = mm
kind = ode
params = vmax, km, v
states = central
outputs = cp
bolus(iv) -> central
dx(central) = -vmax * central / (km + central)
out(cp) = central / v
"#;

const LINEAR_COVARIATE: &str = r#"
name = linear_cov
kind = ode
params = ke0, v
covariates = wt@linear
derived = ke
states = central
outputs = cp
bolus(iv) -> central
ke = ke0 * pow(wt / 70.0, 0.75)
dx(central) = -ke * central
out(cp) = central / v
"#;

const LOCF_COVARIATE: &str = r#"
name = locf_cov
kind = ode
params = ke0, v
covariates = wt@locf
derived = ke
states = central
outputs = cp
bolus(iv) -> central
ke = ke0 * pow(wt / 70.0, 0.75)
dx(central) = -ke * central
out(cp) = central / v
"#;

const EXPLICIT_TIME: &str = r#"
name = time_dependent
kind = ode
params = ke
states = central
outputs = cp
bolus(iv) -> central
dx(central) = -ke * central * (1.0 + 0.1 * t)
out(cp) = central
"#;

fn oral_subject() -> Subject {
    Subject::builder("oral")
        .bolus(0.0, 100.0, "oral")
        .bolus(12.0, 100.0, "oral")
        .infusion(4.0, 50.0, "iv", 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(5.0, "cp")
        .missing_observation(8.0, "cp")
        .missing_observation(13.0, "cp")
        .missing_observation(24.0, "cp")
        .build()
}

#[test]
fn one_compartment_oral_qualifies_and_agrees_with_the_solver() {
    assert_paths_agree(
        ONE_COMPARTMENT_ORAL,
        &oral_subject(),
        &[("ka", 1.1), ("ke", 0.15), ("v", 20.0)],
        1e-6,
    );
}

#[test]
fn two_compartment_qualifies_and_agrees_with_the_solver() {
    assert_paths_agree(
        TWO_COMPARTMENT,
        &oral_subject(),
        &[
            ("ka", 1.1),
            ("ke", 0.15),
            ("kcp", 0.4),
            ("kpc", 0.25),
            ("v", 20.0),
        ],
        1e-6,
    );
}

#[test]
fn locf_covariates_qualify_and_switch_at_their_knots() {
    let mut subject = Subject::builder("locf")
        .bolus(0.0, 100.0, "iv")
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 6.0, 100.0)
        .missing_observation(3.0, "cp")
        .missing_observation(9.0, "cp")
        .missing_observation(18.0, "cp")
        .build();
    for occasion in subject.occasions_mut() {
        occasion.covariates_mut().set_covariate_fixed("wt", true);
    }
    assert_paths_agree(LOCF_COVARIATE, &subject, &[("ke0", 0.2), ("v", 10.0)], 1e-6);
}

#[test]
fn closed_form_matches_the_exact_one_compartment_solution() {
    let model = ode_model(ONE_COMPARTMENT_ORAL).disable_cache();
    let subject = Subject::builder("exact")
        .bolus(0.0, 100.0, "oral")
        .missing_observation(3.0, "cp")
        .build();
    let values = [("ka", 1.1), ("ke", 0.15), ("v", 20.0)];
    let actual = predictions(&model, &subject, &values)[0];

    let (ka, ke, v, dose, t) = (1.1_f64, 0.15_f64, 20.0_f64, 100.0_f64, 3.0_f64);
    let expected = dose * ka / (v * (ka - ke)) * ((-ke * t).exp() - (-ka * t).exp());
    assert!(
        (actual - expected).abs() < 1e-12,
        "closed form {actual} vs exact {expected}"
    );
}

#[test]
fn repeated_eigenvalues_are_not_a_special_case() {
    // ka == ke makes the hand-derived absorption solution divide by zero; the
    // matrix exponential is indifferent.
    let model = ode_model(ONE_COMPARTMENT_ORAL).disable_cache();
    let subject = Subject::builder("degenerate")
        .bolus(0.0, 100.0, "oral")
        .missing_observation(3.0, "cp")
        .build();
    let k = 0.4_f64;
    let actual = predictions(&model, &subject, &[("ka", k), ("ke", k), ("v", 20.0)])[0];

    let (dose, v, t) = (100.0_f64, 20.0_f64, 3.0_f64);
    let expected = dose * k * t * (-k * t).exp() / v;
    assert!(
        (actual - expected).abs() < 1e-12,
        "closed form {actual} vs exact {expected}"
    );
}

#[test]
fn nonlinear_dynamics_fall_back_to_the_solver() {
    let model = ode_model(MICHAELIS_MENTEN);
    assert!(!model.uses_closed_form_for(&oral_subject()));
    assert!(!model.info().is_linear_time_invariant());
}

#[test]
fn continuously_interpolated_covariate_data_falls_back_to_the_solver() {
    // The model declares `@linear`, but interpolation is settled by the data,
    // so the requirement is checked against the subject.
    let model = ode_model(LINEAR_COVARIATE);
    let interpolated = Subject::builder("interpolated")
        .bolus(0.0, 100.0, "iv")
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 6.0, 100.0)
        .missing_observation(3.0, "cp")
        .build();
    assert!(
        !model.uses_closed_form_for(&interpolated),
        "a linearly interpolated covariate makes the coefficients vary within an interval"
    );
}

#[test]
fn carry_forward_covariate_data_qualifies_even_when_declared_linear() {
    let model = ode_model(LINEAR_COVARIATE);
    let mut carried = Subject::builder("carried")
        .bolus(0.0, 100.0, "iv")
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 6.0, 100.0)
        .missing_observation(3.0, "cp")
        .build();
    for occasion in carried.occasions_mut() {
        occasion.covariates_mut().set_covariate_fixed("wt", true);
    }
    assert!(model.uses_closed_form_for(&carried));
}

#[test]
fn explicit_time_dependence_falls_back_to_the_solver() {
    let model = ode_model(EXPLICIT_TIME);
    assert!(!model.uses_closed_form_for(&oral_subject()));
    assert!(!model.info().is_linear_time_invariant());
}

#[test]
fn solver_class_is_reported_for_inspection() {
    use pharmsol::dsl::SolverClass;
    assert!(matches!(
        ode_model(ONE_COMPARTMENT_ORAL).info().solver_class,
        SolverClass::LinearTimeInvariant { .. }
    ));
    assert!(matches!(
        ode_model(MICHAELIS_MENTEN).info().solver_class,
        SolverClass::Numeric { .. }
    ));
    assert_eq!(
        ode_model(LOCF_COVARIATE)
            .info()
            .closed_form_covariate_requirements(),
        ["wt"]
    );
}

#[test]
fn compile_time_declines_explain_themselves() {
    let explanation = ode_model(MICHAELIS_MENTEN).info().solver_explanation();
    assert!(
        explanation.contains("integrated numerically") && explanation.contains("nonlinear"),
        "{explanation}"
    );

    let explanation = ode_model(EXPLICIT_TIME).info().solver_explanation();
    assert!(
        explanation.contains("`t` appears in the dynamics"),
        "{explanation}"
    );

    let explanation = ode_model(ONE_COMPARTMENT_ORAL).info().solver_explanation();
    assert!(explanation.contains("closed form"), "{explanation}");

    // A model that qualifies only for some data says so up front.
    let explanation = ode_model(LOCF_COVARIATE).info().solver_explanation();
    assert!(
        explanation.contains("carry-forward") && explanation.contains("`wt`"),
        "{explanation}"
    );
}

#[test]
fn data_dependent_declines_name_the_covariate_and_subject() {
    let model = ode_model(LINEAR_COVARIATE);
    let interpolated = Subject::builder("patient-7")
        .bolus(0.0, 100.0, "iv")
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 6.0, 100.0)
        .missing_observation(3.0, "cp")
        .build();
    let decline = model
        .closed_form_decline_for(&interpolated)
        .expect("interpolated covariate data declines");
    assert!(decline.contains("`wt`"), "{decline}");
    assert!(decline.contains("patient-7"), "{decline}");

    let mut carried = interpolated.clone();
    for occasion in carried.occasions_mut() {
        occasion.covariates_mut().set_covariate_fixed("wt", true);
    }
    assert_eq!(model.closed_form_decline_for(&carried), None);
}

#[test]
fn compilation_reports_the_chosen_solver() {
    use std::sync::{Arc, Mutex};

    let events: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    compile_module_source_to_runtime(MICHAELIS_MENTEN, None, move |kind, message| {
        sink.lock().unwrap().push((kind, message));
    })
    .expect("model compiles");

    let events = events.lock().unwrap();
    let solver = events
        .iter()
        .find(|(kind, _)| kind == "solver")
        .expect("compilation emits a solver event");
    assert!(solver.1.contains("integrated numerically"), "{}", solver.1);
}

#[test]
fn the_solver_notice_points_at_the_offending_equation() {
    use std::sync::{Arc, Mutex};

    let events: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    compile_module_source_to_runtime(MICHAELIS_MENTEN, None, move |kind, message| {
        sink.lock().unwrap().push((kind, message));
    })
    .expect("model compiles");

    let events = events.lock().unwrap();
    let rendered = &events
        .iter()
        .find(|(kind, _)| kind == "solver")
        .expect("solver event")
        .1;
    // A note rather than an error, pointing at the equation responsible.
    assert!(rendered.starts_with("note[DSL5000]"), "{rendered}");
    assert!(rendered.contains("--> line 8"), "{rendered}");
    assert!(
        rendered.contains("dx(central) = -vmax * central / (km + central)"),
        "{rendered}"
    );
}

#[test]
fn a_time_dependent_decline_points_at_the_time_reference() {
    use std::sync::{Arc, Mutex};

    let events: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    compile_module_source_to_runtime(EXPLICIT_TIME, None, move |kind, message| {
        sink.lock().unwrap().push((kind, message));
    })
    .expect("model compiles");

    let events = events.lock().unwrap();
    let rendered = &events
        .iter()
        .find(|(kind, _)| kind == "solver")
        .expect("solver event")
        .1;
    assert!(rendered.contains("not time-invariant"), "{rendered}");
    assert!(
        rendered.contains("dx(central) = -ke * central * (1.0 + 0.1 * t)"),
        "{rendered}"
    );
}

#[test]
fn closed_form_can_be_required_instead_of_assumed() {
    // A model that qualifies is unaffected.
    let linear = ode_model(ONE_COMPARTMENT_ORAL).require_closed_form();
    let parameters =
        Parameters::with_model(&linear, vec![("ka", 1.1), ("ke", 0.15), ("v", 20.0)]).unwrap();
    assert!(linear
        .estimate_predictions(&oral_subject(), &parameters)
        .is_ok());

    // A model that does not qualify fails instead of silently slowing down.
    let nonlinear = ode_model(MICHAELIS_MENTEN).require_closed_form();
    let parameters =
        Parameters::with_model(&nonlinear, vec![("vmax", 10.0), ("km", 5.0), ("v", 20.0)]).unwrap();
    let subject = Subject::builder("mm")
        .bolus(0.0, 100.0, "iv")
        .missing_observation(3.0, "cp")
        .build();
    let error = nonlinear
        .estimate_predictions(&subject, &parameters)
        .expect_err("requiring closed form should fail for a nonlinear model");
    let message = error.to_string();
    assert!(
        message.contains("requires closed-form propagation"),
        "{message}"
    );
    assert!(message.contains("nonlinear"), "{message}");
}

#[test]
fn requiring_closed_form_also_catches_data_that_disqualifies_a_linear_model() {
    let model = ode_model(LINEAR_COVARIATE).require_closed_form();
    let interpolated = Subject::builder("patient-7")
        .bolus(0.0, 100.0, "iv")
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 6.0, 100.0)
        .missing_observation(3.0, "cp")
        .build();
    let parameters = Parameters::with_model(&model, vec![("ke0", 0.2), ("v", 10.0)]).unwrap();

    let error = model
        .estimate_predictions(&interpolated, &parameters)
        .expect_err("interpolated covariate data should fail the requirement");
    let message = error.to_string();
    assert!(message.contains("patient-7"), "{message}");
    assert!(message.contains("`wt`"), "{message}");
}
