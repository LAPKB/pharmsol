/// Two-Compartment Pharmacokinetic Model Example
///
/// This example demonstrates how to implement a two-compartment pharmacokinetic model
/// with weight-based covariate scaling using pharmsol.
///
/// The two-compartment model describes drug distribution between:
/// - Central compartment (x[0]): where drug enters and is eliminated
/// - Peripheral compartment (x[1]): a tissue compartment in equilibrium with central
///
/// Model equations:
///   dx[0]/dt = -Ke * x[0] - Kcp * x[0] + Kpc * x[1] + RateIV
///   dx[1]/dt =  Kcp * x[0] - Kpc * x[1]
///
/// Where:
///   Ke  = CL / V   (elimination rate constant)
///   Kcp = Q / V    (central to peripheral rate constant)
///   Kpc = Q / Vp   (peripheral to central rate constant)

fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::prelude::*;

    // Create a subject using the builder pattern
    let subject = Subject::builder("subject_001")
        // An infusion of 500 mg over 0.5 hours (1000 mg/hr rate)
        .infusion(0.0, 500.0, 0, 0.5)
        // Weight covariate at baseline (85 kg reference weight)
        .covariate("wt", 0.0, 70.0)
        // Observations at various time points (concentration in mg/L)
        .observation(0.5, 8.5, 0)
        .observation(1.0, 6.2, 0)
        .observation(2.0, 4.1, 0)
        .observation(4.0, 2.3, 0)
        .observation(6.0, 1.5, 0)
        .observation(8.0, 1.1, 0)
        .observation(12.0, 0.7, 0)
        // Missing observation to force prediction at this time point
        .missing_observation(24.0, 0)
        .build();

    // Define the two-compartment ODE model
    let ode = equation::ODE::new(
        // Primary differential equation block
        |x, p, t, dx, _b, rateiv, cov| {
            // Fetch the (possibly interpolated) weight covariate at time t
            fetch_cov!(cov, t, wt);

            // Fetch parameters from the parameter vector
            // CL: Clearance (L/hr), V: Central volume (L)
            // Vp: Peripheral volume (L), Q: Inter-compartmental clearance (L/hr)
            fetch_params!(p, cl, v, vp, q);

            // Weight-based allometric scaling
            // Reference weight is 85 kg
            let wt_ratio = wt / 85.0;
            // Volumes scale linearly with weight
            let v_scaled = v * wt_ratio;
            let vp_scaled = vp * wt_ratio;
            // Clearances scale with weight^0.75 (allometric exponent)
            let cl_scaled = cl * wt_ratio.powf(0.75);
            let q_scaled = q * wt_ratio.powf(0.75);

            // Calculate rate constants
            let ke = cl_scaled / v_scaled; // Elimination rate constant
            let kcp = q_scaled / v_scaled; // Central to peripheral rate constant
            let kpc = q_scaled / vp_scaled; // Peripheral to central rate constant

            // Two-compartment model differential equations
            // Central compartment: elimination + distribution + infusion input
            dx[0] = -ke * x[0] - kcp * x[0] + kpc * x[1] + rateiv[0];
            // Peripheral compartment: distribution equilibrium
            dx[1] = kcp * x[0] - kpc * x[1];
            Ok(())
        },
        // Lag time block (no lag in this model)
        |_p, _t, _cov| Ok(lag! {}),
        // Bioavailability block (100% for IV, so not needed)
        |_p, _t, _cov| Ok(fa! {}),
        // Secondary equations block (not used here)
        |_p, _t, _cov, _x| Ok(()),
        // Output equation block - calculates observed concentration
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt);
            fetch_params!(p, _cl, v, _vp, _q);

            // Calculate scaled volume for concentration
            let wt_ratio = wt / 85.0;
            let v_scaled = v * wt_ratio;

            // Concentration = Amount / Volume
            y[0] = x[0] / v_scaled;
            Ok(())
        },
        // Model dimensions: (number of compartments, number of outputs)
        (2, 1),
    );

    // Define parameter values
    // Note: order must match the fetch_params! macro order
    let cl = 5.0; // Clearance (L/hr)
    let v = 50.0; // Central volume of distribution (L)
    let vp = 100.0; // Peripheral volume of distribution (L)
    let q = 10.0; // Inter-compartmental clearance (L/hr)
    let params = vec![cl, v, vp, q];

    // Compute predictions
    let predictions = ode.estimate_predictions(&subject, &params)?;

    // Print results in a formatted table
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║          Two-Compartment Model Predictions                 ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Parameters:                                               ║");
    println!(
        "║    CL = {:.1} L/hr, V = {:.1} L, Vp = {:.1} L, Q = {:.1} L/hr    ║",
        cl, v, vp, q
    );
    println!("║    Weight = 70 kg (scaled from 85 kg reference)            ║");
    println!("╠═════════════╦══════════════════════════════════════════════╣");
    println!("║  Time (hr)  ║  Predicted Concentration                     ║");
    println!("╠═════════════╬══════════════════════════════════════════════╣");

    for pred in predictions.predictions() {
        println!(
            "║ {:>10.2}  ║ {:>18.4} mg/L                       ║",
            pred.time(),
            pred.prediction()
        );
    }
    println!("╚═════════════╩══════════════════════════════════════════════╝\n");

    Ok(())
}
