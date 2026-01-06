//! NCA (Non-Compartmental Analysis) Example
//!
//! This example demonstrates the NCA capabilities of pharmsol.
//!
//! Run with: `cargo run --example nca`

use pharmsol::nca::{BLQRule, NCAOptions};
use pharmsol::prelude::*;

fn main() {
    println!("=== pharmsol NCA Example ===\n");

    // Example 1: Basic oral PK analysis
    basic_oral_example();

    // Example 2: IV Bolus analysis
    iv_bolus_example();

    // Example 3: IV Infusion analysis
    iv_infusion_example();

    // Example 4: Steady-state analysis
    steady_state_example();

    // Example 5: BLQ handling
    blq_handling_example();
}

/// Basic oral PK NCA analysis
fn basic_oral_example() {
    println!("--- Basic Oral PK Example ---\n");

    // Build subject with oral dose and observations
    let subject = Subject::builder("patient_001")
        .bolus(0.0, 100.0, 0) // 100 mg oral dose (input 0 = depot)
        .observation(0.0, 0.0, 0)
        .observation(0.5, 5.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 0.25, 0)
        .build();

    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().expect("NCA analysis failed");

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

    // Build subject with IV bolus (input 1 = central compartment)
    let subject = Subject::builder("iv_patient")
        .bolus(0.0, 500.0, 1) // 500 mg IV bolus
        .observation(0.25, 95.0, 0)
        .observation(0.5, 82.0, 0)
        .observation(1.0, 61.0, 0)
        .observation(2.0, 34.0, 0)
        .observation(4.0, 10.0, 0)
        .observation(8.0, 3.0, 0)
        .observation(12.0, 0.9, 0)
        .build();

    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().expect("NCA analysis failed");

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

/// IV Infusion analysis
fn iv_infusion_example() {
    println!("--- IV Infusion Example ---\n");

    // Build subject with IV infusion
    let subject = Subject::builder("infusion_patient")
        .infusion(0.0, 100.0, 1, 0.5) // 100 mg over 0.5h to central
        .observation(0.0, 0.0, 0)
        .observation(0.5, 15.0, 0)
        .observation(1.0, 12.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 1.5, 0)
        .observation(12.0, 0.5, 0)
        .build();

    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().expect("NCA analysis failed");

    println!("Exposure:");
    println!("  Cmax:     {:.1}", result.exposure.cmax);
    println!("  Tmax:     {:.2} h", result.exposure.tmax);
    println!("  AUClast:  {:.1}", result.exposure.auc_last);

    if let Some(ref infusion) = result.iv_infusion {
        println!("\nIV Infusion Parameters:");
        println!("  Infusion duration: {:.2} h", infusion.infusion_duration);
        if let Some(mrt_iv) = infusion.mrt_iv {
            println!("  MRT (corrected):   {:.2} h", mrt_iv);
        }
    }

    println!();
}

/// Steady-state analysis
fn steady_state_example() {
    println!("--- Steady-State Example ---\n");

    // Build subject at steady-state (Q12H dosing)
    let subject = Subject::builder("ss_patient")
        .bolus(0.0, 100.0, 0) // 100 mg oral
        .observation(0.0, 5.0, 0)
        .observation(1.0, 15.0, 0)
        .observation(2.0, 12.0, 0)
        .observation(4.0, 8.0, 0)
        .observation(6.0, 6.0, 0)
        .observation(8.0, 5.5, 0)
        .observation(12.0, 5.0, 0)
        .build();

    let options = NCAOptions::default().with_tau(12.0); // 12-hour dosing interval
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().expect("NCA analysis failed");

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

/// BLQ handling demonstration
fn blq_handling_example() {
    println!("--- BLQ Handling Example ---\n");

    // Build subject with BLQ observations
    let subject = Subject::builder("blq_patient")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 0.5, 0)
        .observation(24.0, 0.02, 0) // Below LOQ of 0.05
        .build();

    // With BLQ exclusion
    let options_exclude = NCAOptions::default().with_blq(0.05, BLQRule::Exclude);
    let results_exclude = subject.nca(&options_exclude, 0);
    let result_exclude = results_exclude[0].as_ref().unwrap();

    // With BLQ = 0
    let options_zero = NCAOptions::default().with_blq(0.05, BLQRule::Zero);
    let results_zero = subject.nca(&options_zero, 0);
    let result_zero = results_zero[0].as_ref().unwrap();

    println!("BLQ Handling Comparison (LOQ = 0.05):");
    println!("\n  Exclude BLQ:");
    println!("    Tlast:   {:.1} h", result_exclude.exposure.tlast);
    println!("    AUClast: {:.2}", result_exclude.exposure.auc_last);

    println!("\n  BLQ = 0:");
    println!("    Tlast:   {:.1} h", result_zero.exposure.tlast);
    println!("    AUClast: {:.2}", result_zero.exposure.auc_last);

    println!();

    // Full result display
    println!("--- Full Result Display ---\n");
    println!("{}", result_exclude);
}
