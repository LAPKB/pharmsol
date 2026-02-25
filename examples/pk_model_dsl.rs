/// PK Model DSL Example
///
/// Demonstrates how to define pharmacokinetic ODE models using the `pk_model!`
/// macro, which provides a clean declarative syntax that eliminates boilerplate.
///
/// Compare this with `ode_readme.rs` and `two_compartment.rs` for the manual approach.

fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::prelude::*;

    // ========================================================================
    // Example 1: Simple one-compartment IV bolus model
    // ========================================================================
    println!("=== Example 1: One-compartment IV bolus ===\n");

    let subject = Subject::builder("patient_001")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // The pk_model! macro handles all the wiring:
    // - Parameters are declared once and available in all blocks by name
    // - x, dx, y, t, rateiv are automatically available — no closures needed
    // - Optional sections (lag, fa, init) default to no-ops
    let ode = pk_model! {
        params: (ke, v),
        diffeq: {
            dx[0] = -ke * x[0];
        },
        out: {
            y[0] = x[0] / v;
        },
        neqs: (1, 1),
    };

    let params = vec![0.1, 50.0]; // ke=0.1 hr⁻¹, V=50 L
    let predictions = ode.estimate_predictions(&subject, &params)?;

    println!("  Parameters: ke={}, V={} L", params[0], params[1]);
    println!("  {:>10}  {:>12}", "Time (hr)", "Conc (mg/L)");
    for pred in predictions.predictions() {
        println!("  {:>10.1}  {:>12.4}", pred.time(), pred.prediction());
    }

    // ========================================================================
    // Example 2: Two-compartment model with oral absorption and lag time
    // ========================================================================
    println!("\n=== Example 2: Two-compartment oral with lag ===\n");

    let subject = Subject::builder("patient_002")
        .bolus(0.0, 500.0, 0) // Oral dose to absorption compartment
        .repeat(2, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let ode = pk_model! {
        params: (ka, ke, kcp, kpc, tlag, v),
        lag: { 0 => tlag },
        diffeq: {
            // Compartment 0: absorption (gut)
            dx[0] = -ka * x[0];
            // Compartment 1: central
            dx[1] = ka * x[0] - ke * x[1] - kcp * x[1] + kpc * x[2];
            // Compartment 2: peripheral
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        out: {
            y[0] = x[1] / v;
        },
        neqs: (3, 1),
    };

    let params = vec![1.5, 0.1, 0.3, 0.2, 0.5, 50.0];
    let predictions = ode.estimate_predictions(&subject, &params)?;

    println!(
        "  Parameters: ka={}, ke={}, kcp={}, kpc={}, tlag={}, V={} L",
        params[0], params[1], params[2], params[3], params[4], params[5]
    );
    println!("  {:>10}  {:>12}", "Time (hr)", "Conc (mg/L)");
    for pred in predictions.predictions() {
        println!("  {:>10.1}  {:>12.4}", pred.time(), pred.prediction());
    }

    // ========================================================================
    // Example 3: One-compartment IV infusion with weight-based covariate
    // ========================================================================
    println!("\n=== Example 3: IV infusion with weight scaling ===\n");

    let subject = Subject::builder("patient_003")
        .infusion(0.0, 500.0, 0, 1.0) // 500 mg over 1 hour
        .covariate("wt", 0.0, 85.0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let ode = pk_model! {
        params: (cl, v),
        covariates: (wt),
        diffeq: {
            // Allometric scaling: CL ~ wt^0.75, V ~ wt^1.0
            let cl_scaled = cl * (wt / 70.0).powf(0.75);
            let v_scaled = v * (wt / 70.0);
            let ke = cl_scaled / v_scaled;
            dx[0] = -ke * x[0] + rateiv[0];
        },
        out: {
            let v_scaled = v * (wt / 70.0);
            y[0] = x[0] / v_scaled;
        },
        neqs: (1, 1),
    };

    let params = vec![5.0, 50.0]; // CL=5 L/hr, V=50 L (at 70 kg reference)
    let predictions = ode.estimate_predictions(&subject, &params)?;

    println!(
        "  Parameters: CL={} L/hr, V={} L (reference 70 kg)",
        params[0], params[1]
    );
    println!("  Weight: 85 kg");
    println!("  {:>10}  {:>12}", "Time (hr)", "Conc (mg/L)");
    for pred in predictions.predictions() {
        println!("  {:>10.1}  {:>12.4}", pred.time(), pred.prediction());
    }

    // ========================================================================
    // Example 4: Model with bioavailability and initial conditions
    // ========================================================================
    println!("\n=== Example 4: With bioavailability and initial conditions ===\n");

    let subject = Subject::builder("patient_004")
        .bolus(0.0, 200.0, 0) // Oral dose
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let ode = pk_model! {
        params: (ka, ke, v, bio),
        fa: { 0 => bio },
        diffeq: {
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        out: {
            y[0] = x[1] / v;
        },
        neqs: (2, 1),
    };

    let params = vec![1.0, 0.1, 50.0, 0.6]; // 60% bioavailability
    let predictions = ode.estimate_predictions(&subject, &params)?;

    println!(
        "  Parameters: ka={}, ke={}, V={} L, F={}",
        params[0], params[1], params[2], params[3]
    );
    println!("  {:>10}  {:>12}", "Time (hr)", "Conc (mg/L)");
    for pred in predictions.predictions() {
        println!("  {:>10.1}  {:>12.4}", pred.time(), pred.prediction());
    }

    println!("\nAll examples completed successfully!");
    Ok(())
}
