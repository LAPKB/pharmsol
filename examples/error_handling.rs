/// Example demonstrating error propagation from model closures.
///
/// All user-supplied closures (diffeq, out, lag, fa, init, seceq) return `Result`,
/// enabling graceful error messages instead of panics when something goes wrong.
///
/// This example shows:
/// 1. A correct model that runs successfully
/// 2. A model with too few parameters, triggering a bounds-check error
/// 3. A model with a missing covariate, triggering a covariate lookup error
fn main() {
    use pharmsol::prelude::*;

    // Create a simple subject with one dose and some observations
    let subject = Subject::builder("patient_1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.8, 0)
        .observation(2.0, 1.5, 0)
        .observation(4.0, 0.9, 0)
        .observation(8.0, 0.3, 0)
        .build();

    // =========================================================================
    // 1. Correct model — runs successfully
    // =========================================================================
    println!("=== Correct model ===");

    let correct_model = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // ke = 0.1, v = 50
    match correct_model.estimate_predictions(&subject, &vec![0.1, 50.0]) {
        Ok(predictions) => {
            println!("Predictions: {:?}", predictions.flat_predictions());
        }
        Err(e) => {
            println!("Unexpected error: {e}");
        }
    }

    // =========================================================================
    // 2. Faulty model — requests 3 parameters but only 2 are supplied
    // =========================================================================
    println!("\n=== Model with too few parameters ===");

    let faulty_model = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            // BUG: this model asks for 3 parameters (ke, v, extra)
            // but we will only supply 2 when calling estimate_predictions
            fetch_params!(p, ke, _v, _extra);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // Only supply 2 parameters, but diffeq expects 3
    match faulty_model.estimate_predictions(&subject, &vec![0.1, 50.0]) {
        Ok(_) => {
            println!("Model ran (unexpectedly).");
        }
        Err(e) => {
            // This will print a descriptive error instead of panicking
            println!("Caught error: {e}");
        }
    }

    // =========================================================================
    // 3. Model referencing a covariate that doesn't exist on the subject
    // =========================================================================
    println!("\n=== Model with missing covariate ===");

    let cov_model = equation::ODE::new(
        |x, p, t, dx, b, rateiv, cov| {
            fetch_params!(p, ke_ref, _v);
            // BUG: subject has no "wt" covariate defined
            fetch_cov!(cov, t, wt);
            let ke = ke_ref * (wt / 70.0_f64).powf(0.75);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    match cov_model.estimate_predictions(&subject, &vec![0.1, 50.0]) {
        Ok(_) => {
            println!("Model ran (unexpectedly).");
        }
        Err(e) => {
            println!("Caught error: {e}");
        }
    }

    println!("\nAll scenarios handled gracefully — no panics!");
}
