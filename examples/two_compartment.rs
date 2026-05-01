/// Two-Compartment Pharmacokinetic Model Example
///
/// This example demonstrates how to implement a two-compartment pharmacokinetic model
/// with weight-based covariate scaling using pharmsol.
///
/// It uses the declaration-first `ode!` surface so the route, covariate,
/// state, and output metadata stay aligned with the generated execution path.
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

    let ode = ode! {
        name: "two_cmt_wt",
        params: [cl, v, vp, q],
        covariates: [wt],
        states: [central, peripheral],
        outputs: [cp],
        routes: {
            infusion(iv) -> central,
        },
        diffeq: |x, _t, dx| {
            // CL: Clearance (L/hr), V: Central volume (L)
            // Vp: Peripheral volume (L), Q: Inter-compartmental clearance (L/hr)
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
            // Central compartment: elimination + distribution
            dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
            // Peripheral compartment: distribution equilibrium
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        // Output equation block - calculates observed concentration
        out: |x, _t, y| {
            // Calculate scaled volume for concentration
            let wt_ratio = wt / 85.0;
            let v_scaled = v * wt_ratio;

            // Concentration = Amount / Volume
            y[cp] = x[central] / v_scaled;
        },
    };

    let iv = ode.route_index("iv").expect("iv route exists");
    let cp = ode.output_index("cp").expect("cp output exists");

    // Create a subject using metadata-backed route and output names instead of
    // hard-coded numeric indices.
    let subject = Subject::builder("subject_001")
        .infusion(0.0, 500.0, iv, 0.5)
        .covariate("wt", 0.0, 70.0)
        .observation(0.5, 8.5, cp)
        .observation(1.0, 6.2, cp)
        .observation(2.0, 4.1, cp)
        .observation(4.0, 2.3, cp)
        .observation(6.0, 1.5, cp)
        .observation(8.0, 1.1, cp)
        .observation(12.0, 0.7, cp)
        .missing_observation(24.0, cp)
        .build();

    // Define parameter values
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
