//! NCA (Non-Compartmental Analysis) Example
//!
//! This example demonstrates the NCA capabilities of pharmsol.
//!
//! Run with: `cargo run --example nca`

use pharmsol::nca::{summarize, BLQRule, NCAOptions, RouteParams};
use pharmsol::prelude::*;
use pharmsol::Censor;

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

    // Example 6: Population summary
    population_summary_example();
}

/// Basic oral PK NCA analysis
fn basic_oral_example() {
    println!("--- Basic Oral PK Example ---\n");

    // Build subject with oral dose using the bolus_ev() alias
    let subject = Subject::builder("patient_001")
        .bolus_ev(0.0, 100.0) // 100 mg oral dose (depot compartment)
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

    // nca_first() is a convenience that returns the first occasion's result directly
    let result = subject.nca_first(&options, 0).expect("NCA analysis failed");

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

    // Build subject with IV bolus using bolus_iv() alias
    let subject = Subject::builder("iv_patient")
        .bolus_iv(0.0, 500.0) // 500 mg IV bolus (central compartment)
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

    if let Some(RouteParams::IVBolus(ref bolus)) = result.route_params {
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

    // Build subject with IV infusion using infusion_iv() alias
    let subject = Subject::builder("infusion_patient")
        .infusion_iv(0.0, 100.0, 0.5) // 100 mg over 0.5h to central
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

    if let Some(RouteParams::IVInfusion(ref infusion)) = result.route_params {
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
        .bolus_ev(0.0, 100.0) // 100 mg oral
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

    // Build subject with BLQ observations marked using Censor::BLOQ
    // This is the proper way to indicate BLQ samples - the censoring
    // information is stored with each observation, not determined
    // retroactively by a numeric threshold.
    let subject = Subject::builder("blq_patient")
        .bolus_ev(0.0, 100.0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 0.5, 0)
        // The last observation is BLQ - mark it with Censor::BLOQ
        // The value (0.02) represents the LOQ threshold
        .censored_observation(24.0, 0.02, 0, Censor::BLOQ)
        .build();

    // With BLQ exclusion - BLOQ-marked samples are excluded
    let options_exclude = NCAOptions::default().with_blq_rule(BLQRule::Exclude);
    let results_exclude = subject.nca(&options_exclude, 0);
    let result_exclude = results_exclude[0].as_ref().unwrap();

    // With BLQ = 0 - BLOQ-marked samples are set to zero
    let options_zero = NCAOptions::default().with_blq_rule(BLQRule::Zero);
    let results_zero = subject.nca(&options_zero, 0);
    let result_zero = results_zero[0].as_ref().unwrap();

    // With LOQ/2 - BLOQ-marked samples are set to LOQ/2 (0.02/2 = 0.01)
    let options_loq2 = NCAOptions::default().with_blq_rule(BLQRule::LoqOver2);
    let results_loq2 = subject.nca(&options_loq2, 0);
    let result_loq2 = results_loq2[0].as_ref().unwrap();

    println!("BLQ Handling Comparison (using Censor::BLOQ marking):");
    println!("\n  Exclude BLQ:");
    println!("    Tlast:   {:.1} h", result_exclude.exposure.tlast);
    println!("    AUClast: {:.2}", result_exclude.exposure.auc_last);

    println!("\n  BLQ = 0:");
    println!("    Tlast:   {:.1} h", result_zero.exposure.tlast);
    println!("    AUClast: {:.2}", result_zero.exposure.auc_last);

    println!("\n  BLQ = LOQ/2:");
    println!("    Tlast:   {:.1} h", result_loq2.exposure.tlast);
    println!("    AUClast: {:.2}", result_loq2.exposure.auc_last);

    println!();

    // Full result display
    println!("--- Full Result Display ---\n");
    println!("{}", result_exclude);
}

/// Population summary statistics
fn population_summary_example() {
    println!("--- Population Summary Example ---\n");

    // Build a small population dataset
    let subjects = vec![
        Subject::builder("subj_01")
            .bolus_ev(0.0, 100.0)
            .observation(0.5, 4.0, 0)
            .observation(1.0, 9.0, 0)
            .observation(2.0, 7.0, 0)
            .observation(4.0, 3.5, 0)
            .observation(8.0, 1.5, 0)
            .observation(24.0, 0.2, 0)
            .build(),
        Subject::builder("subj_02")
            .bolus_ev(0.0, 100.0)
            .observation(0.5, 5.5, 0)
            .observation(1.0, 12.0, 0)
            .observation(2.0, 9.0, 0)
            .observation(4.0, 5.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(24.0, 0.3, 0)
            .build(),
        Subject::builder("subj_03")
            .bolus_ev(0.0, 100.0)
            .observation(0.5, 3.0, 0)
            .observation(1.0, 8.0, 0)
            .observation(2.0, 6.5, 0)
            .observation(4.0, 3.0, 0)
            .observation(8.0, 1.0, 0)
            .observation(24.0, 0.1, 0)
            .build(),
    ];

    let options = NCAOptions::default();

    // Collect successful NCA results
    let results: Vec<_> = subjects
        .iter()
        .filter_map(|s| s.nca_first(&options, 0).ok())
        .collect();

    // Compute population summary
    let summary = summarize(&results);
    println!(
        "Population: {} subjects\n",
        summary.n_subjects
    );

    for stats in &summary.parameters {
        println!(
            "  {:<12} mean={:>8.2}  CV%={:>6.1}  [{:.2} - {:.2}]",
            stats.name, stats.mean, stats.cv_pct, stats.min, stats.max
        );
    }

    // Demonstrate to_row() for CSV-like output
    println!("\n--- Individual Results (to_row headers) ---\n");
    if let Some(first) = results.first() {
        let row = first.to_row();
        let headers: Vec<&str> = row.iter().map(|(k, _)| *k).collect();
        println!("  Columns: {:?}", &headers[..headers.len().min(8)]);
        println!("  ...(and {} more)", headers.len().saturating_sub(8));
    }

    println!();
}
