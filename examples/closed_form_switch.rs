//! Shows which models are propagated in closed form, which fall back to a
//! numeric integrator and why, and how the two paths compare.
//!
//! Run with:
//! cargo run --release --example closed_form_switch --features dsl

#[cfg(feature = "dsl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeOdeModel};
    use pharmsol::{prelude::*, Cache, Parameters};

    fn compile(source: &str) -> Result<RuntimeOdeModel, Box<dyn std::error::Error>> {
        match compile_module_source_to_runtime(source, None, |_kind, _message| {})? {
            CompiledRuntimeModel::Ode(model) => Ok(model),
            other => Err(format!("expected an ODE model, got {other:?}").into()),
        }
    }

    let one_compartment_oral = r#"
name = one_cpt_oral
kind = ode
params = ka, ke, v
states = gut, central
outputs = cp
bolus(oral) -> gut
dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
out(cp) = central / v
"#;

    let three_compartment = r#"
name = three_cpt
kind = ode
params = ka, k10, k12, k21, k13, k31, v
states = gut, central, fast, slow
outputs = cp
bolus(oral) -> gut
dx(gut) = -ka * gut
dx(central) = ka * gut - (k10 + k12 + k13) * central + k21 * fast + k31 * slow
dx(fast) = k12 * central - k21 * fast
dx(slow) = k13 * central - k31 * slow
out(cp) = central / v
"#;

    let michaelis_menten = r#"
name = mm
kind = ode
params = ka, vmax, km, v
states = gut, central
outputs = cp
bolus(oral) -> gut
dx(gut) = -ka * gut
dx(central) = ka * gut - vmax * central / (km + central)
out(cp) = central / v
"#;

    let time_varying = r#"
name = circadian
kind = ode
params = ke, v
states = central
outputs = cp
bolus(iv) -> central
dx(central) = -ke * (1.0 + 0.3 * sin(t)) * central
out(cp) = central / v
"#;

    let mut builder = Subject::builder("demo");
    for dose in 0..7 {
        let time = dose as f64 * 12.0;
        builder = builder
            .bolus(time, 100.0, "oral")
            .missing_observation(time + 1.0, "cp")
            .missing_observation(time + 4.0, "cp")
            .missing_observation(time + 11.5, "cp");
    }
    let subject = builder.build();

    println!("{:<18} {:>12}   why", "model", "solver");
    println!("{}", "-".repeat(78));
    for (name, source) in [
        ("one_cpt_oral", one_compartment_oral),
        ("three_cpt", three_compartment),
        ("michaelis_menten", michaelis_menten),
        ("circadian", time_varying),
    ] {
        let model = compile(source)?;
        let closed_form = model.uses_closed_form_for(&subject);
        let solver = if closed_form { "closed form" } else { "numeric" };
        let why = if closed_form {
            "linear, coefficients constant between events".to_string()
        } else {
            // The compiler keeps the reason it declined, rendered against the
            // model source with a span.
            let compiled = pharmsol_dsl::compile_model(source)?;
            match pharmsol_dsl::classify_linear_time_invariant(&compiled) {
                Ok(_) => "-".to_string(),
                Err(decline) => decline.reason,
            }
        };
        println!("{name:<18} {solver:>12}   {why}");
    }

    println!("\nOne-compartment oral, 7 doses over 72 h, 200 repeated solves:");
    let closed_form = compile(one_compartment_oral)?.disable_cache();
    let numeric = compile(one_compartment_oral)?
        .force_numeric_solver()
        .disable_cache();
    let values = [("ka", 1.1), ("ke", 0.15), ("v", 20.0)];

    for (label, model) in [("closed form", &closed_form), ("numeric   ", &numeric)] {
        let parameters = Parameters::with_model(model, values)?;
        let start = Instant::now();
        let mut last = 0.0;
        for _ in 0..200 {
            let predictions = model.estimate_predictions(&subject, &parameters)?;
            last = predictions.flat_predictions()[0];
        }
        let elapsed = start.elapsed();
        println!(
            "  {label}  {:>8.1} us/solve   first prediction {last:.12}",
            elapsed.as_secs_f64() * 1e6 / 200.0
        );
    }

    // ka == ke is a removable singularity that the classical closed-form
    // absorption solution divides straight through.
    println!("\nDegenerate ka == ke = 0.4:");
    let degenerate = [("ka", 0.4), ("ke", 0.4), ("v", 20.0)];
    let parameters = Parameters::with_model(&closed_form, degenerate)?;
    let single = Subject::builder("degenerate")
        .bolus(0.0, 100.0, "oral")
        .missing_observation(3.0, "cp")
        .build();
    let actual = closed_form
        .estimate_predictions(&single, &parameters)?
        .flat_predictions()[0];
    let exact = 100.0 * 0.4 * 3.0 * (-0.4f64 * 3.0).exp() / 20.0;
    println!("  closed form {actual:.12}");
    println!("  exact       {exact:.12}");

    Ok(())
}

#[cfg(not(feature = "dsl"))]
fn main() {
    eprintln!("run with --features dsl");
}
