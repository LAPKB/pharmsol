//! NCA (Non-Compartmental Analysis) Example
//!
//! This example demonstrates the NCA capabilities of pharmsol.
//!
//! Run with: `cargo run --example nca`

use pharmsol::nca::{BLQRule, DoseContext, NCAOptions, nca_from_arrays};
use pharmsol::prelude::*;

fn main() {
    println!("=== pharmsol NCA Example ===\n");

    // Example 1: Basic NCA from arrays
    basic_nca_example();

    // Example 2: IV Bolus analysis
    iv_bolus_example();

    // Example 3: Oral (extravascular) analysis
    oral_example();

    // Example 4: Steady-state analysis
    steady_state_example();

    // Example 5: NCA on Subject data
    subject_nca_example();
}

/// Basic NCA analysis from time-concentration arrays
fn basic_nca_example() {
    println!("--- Basic NCA Example ---\n");

    // Typical single-dose PK profile
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.25];

    // Default options (LinUpLogDown AUC, exclude BLQ)
    let options = NCAOptions::default();
    let dose = DoseContext::bolus(100.0, true); // 100 mg extravascular

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).expect("NCA analysis failed");

    println!("Exposure Parameters:");
    println!("  Cmax:     {:.2}", result.exposure.cmax);
    println!("  Tmax:     {:.2} h", result.exposure.tmax);
    println!("  Clast:    {:.3}", result.exposure.clast);
    println!("  Tlast:    {:.1} h", result.exposure.tlast);
    println!("  AUClast:  {:.2}", result.exposure.auc_last);

    if let Some(ref term) = result.terminal {
        println!("\nTerminal Phase:");
        println!("  Lambda-z: {:.4} h⁻¹", term.lambda_z);
        println!("  Half-life: {:.2} h", term.half_life);
        if let Some(mrt) = term.mrt {
            println!("  MRT:      {:.2} h", mrt);
        }
    }

    if let Some(ref cl) = result.clearance {
        println!("\nClearance Parameters:");
        println!("  CL/F:    {:.2} L/h", cl.cl_f);
        println!("  Vz/F:    {:.2} L", cl.vz_f);
    }

    println!("\nQuality: {:?}\n", result.quality.warnings);
}

/// IV Bolus analysis with C0 back-extrapolation
fn iv_bolus_example() {
    println!("--- IV Bolus Example ---\n");

    // IV bolus profile (high initial concentration)
    let times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![95.0, 82.0, 61.0, 34.0, 10.0, 3.0, 0.9];

    let options = NCAOptions::default();
    let dose = DoseContext::bolus(500.0, false); // 500 mg IV bolus

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).expect("NCA analysis failed");

    println!("Exposure:");
    println!("  Cmax:     {:.1}", result.exposure.cmax);
    println!("  AUClast:  {:.1}", result.exposure.auc_last);

    if let Some(ref bolus) = result.iv_bolus {
        println!("\nIV Bolus Parameters:");
        println!("  C0 (back-extrap): {:.1}", bolus.c0);
        println!("  Vd:               {:.1} L", bolus.vd);
        if let Some(vss) = bolus.vss {
            println!("  Vss:              {:.1} L", vss);
        }
    }

    println!();
}

/// Oral (extravascular) analysis
fn oral_example() {
    println!("--- Oral (Extravascular) Example ---\n");

    // Oral absorption profile with lag time
    let times = vec![0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 0.0, 2.0, 8.0, 10.0, 9.0, 5.0, 2.0, 0.8, 0.1];

    let options = NCAOptions::default().with_blq(0.05, BLQRule::Exclude);
    let dose = DoseContext::bolus(200.0, true); // 200 mg oral

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).expect("NCA analysis failed");

    println!("Exposure:");
    println!("  Cmax:     {:.1}", result.exposure.cmax);
    println!("  Tmax:     {:.2} h", result.exposure.tmax);
    println!("  AUClast:  {:.1}", result.exposure.auc_last);

    if let Some(ref oral) = result.extravascular {
        if let Some(tlag) = oral.tlag {
            println!("\nExtravascular Parameters:");
            println!("  Tlag:     {:.2} h", tlag);
        }
    }

    println!();
}

/// Steady-state analysis
fn steady_state_example() {
    println!("--- Steady-State Example ---\n");

    // Steady-state profile (Q12H dosing)
    let times = vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
    let concs = vec![5.0, 15.0, 12.0, 8.0, 6.0, 5.5, 5.0];

    let options = NCAOptions::default().with_tau(12.0); // 12-hour dosing interval
    let dose = DoseContext::bolus(100.0, true); // 100 mg oral

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).expect("NCA analysis failed");

    println!("Exposure:");
    println!("  Cmax:     {:.1}", result.exposure.cmax);
    println!("  AUClast:  {:.1}", result.exposure.auc_last);

    if let Some(ref ss) = result.steady_state {
        println!("\nSteady-State Parameters (tau = {} h):", ss.tau);
        println!("  AUCtau:       {:.1}", ss.auc_tau);
        println!("  Cmin:         {:.1}", ss.cmin);
        println!("  Cmax,ss:      {:.1}", ss.cmax_ss);
        println!("  Cavg:         {:.2}", ss.cavg);
        println!("  Fluctuation:  {:.1}%", ss.fluctuation);
        println!("  Swing:        {:.2}", ss.swing);
    }

    println!();
}

/// NCA analysis using pharmsol's Subject data structure
fn subject_nca_example() {
    println!("--- Subject NCA Example ---\n");

    // Build a subject with dose and observations
    let subject = Subject::builder("patient_001")
        .bolus(0.0, 100.0, 0) // 100 mg dose to depot
        .observation(0.5, 5.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 0.25, 0)
        .build();

    // NCA automatically detects dose and route from events
    let options = NCAOptions::default();

    // Analyze all occasions
    let results = subject.nca(&options, 0);

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(r) => {
                println!("Occasion {} Results:", i);
                println!("  Subject: {:?}", r.subject_id);
                println!("  Cmax: {:.2}", r.exposure.cmax);
                println!("  Tmax: {:.2} h", r.exposure.tmax);
                println!("  AUClast: {:.2}", r.exposure.auc_last);

                if let Some(ref term) = r.terminal {
                    println!("  Half-life: {:.2} h", term.half_life);
                }

                if let Some(ref cl) = r.clearance {
                    println!("  CL/F: {:.2} L/h", cl.cl_f);
                }

                println!();
            }
            Err(e) => {
                println!("Occasion {} failed: {:?}\n", i, e);
            }
        }
    }

    // Display full result
    if let Some(Ok(result)) = results.first() {
        println!("Full Result Display:\n{}", result);
    }
}
