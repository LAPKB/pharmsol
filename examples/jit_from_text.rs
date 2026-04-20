//! End-to-end example: text definition → JIT-compiled model → simulation.
//!
//! Demonstrates the pure-runtime path: only a precompiled `pharmsol` library
//! is required. The model is read from a string at runtime, JIT-compiled with
//! Cranelift, and simulated. No `rustc`, no source files (besides this demo
//! driver), no recompilation step.
//!
//! Run with:
//! ```bash
//! cargo run --features jit --example jit_from_text
//! ```

#[cfg(not(feature = "jit"))]
fn main() {
    eprintln!("Re-run with `--features jit`.");
}

#[cfg(feature = "jit")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pharmsol::jit::Model;
    use pharmsol::prelude::*;
    use pharmsol::Predictions;

    // The model definition is just a string. In a real workflow it would
    // come from a file, an HTTP response, an R `character` vector, etc.
    let src = r#"
        # One-compartment IV with allometric scaling on CL
        name         = onecmt-allo
        compartments = central
        params       = CL, V
        covariates   = WT
        ndrugs       = 1

        dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central
        out(cp)       = central / V
    "#;

    // Text → Model → JIT-compiled native code.
    let ode = Model::from_text(src)?.compile()?;

    // Build a subject with a 100 mg infusion over 0.5 h, body weight 80 kg.
    let subject = Subject::builder("p1")
        .infusion(0.0, 100.0, 0, 0.5)
        .covariate("WT", 0.0, 80.0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    // Run.
    let support_point = [5.0_f64, 50.0]; // CL, V
    let (preds, _) = ode.simulate_subject(&subject, &support_point, None)?;

    println!(" t       cp");
    for p in preds.get_predictions() {
        println!("{:5.1}   {:>8.4}", p.time(), p.prediction());
    }
    Ok(())
}
