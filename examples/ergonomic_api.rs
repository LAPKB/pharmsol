//! Example demonstrating the new ergonomic API with typed parameters and covariates.
//!
//! This example shows the cleanest possible API for defining PK/PD models:
//!
//! - `#[derive(Params)]` for compile-time checked parameters
//! - `#[derive(Covariates)]` for compile-time checked covariates  
//! - `diffeq!` and `out!` macros for clean model equations
//!
//! All parameter and covariate fields are automatically available as local variables
//! within the model equations - no manual unpacking required!

use pharmsol::prelude::*;
use pharmsol::*;

// ============================================================================
// Define typed parameters - compile-time checked!
// ============================================================================

/// One-compartment model parameters
#[derive(Params, Debug, Clone)]
struct Pk {
    /// Elimination rate constant (1/h)
    ke: f64,
    /// Volume of distribution (L)
    v: f64,
}

/// Two-compartment model parameters with covariate effects
#[derive(Params, Debug, Clone)]
struct PkCov {
    /// Elimination rate constant (1/h)
    ke: f64,
    /// Volume of distribution (L)  
    v: f64,
    /// Weight effect on clearance
    theta_wt: f64,
}

// ============================================================================
// Define typed covariates - compile-time checked!
// ============================================================================

/// Patient covariates
#[derive(Covariates, Debug, Clone)]
struct Cov {
    /// Body weight (kg)
    wt: f64,
}

fn main() {
    println!("=== Ergonomic API Demo ===\n");

    // ========================================================================
    // Example 1: Simple one-compartment model without covariates
    // ========================================================================
    
    println!("--- Example 1: Simple One-Compartment Model ---");
    
    let subject = Subject::builder("patient_1")
        .infusion(0.0, 500.0, 0, 0.5)  // 500mg infusion over 0.5h
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // Old verbose API (still works for backward compatibility)
    let ode_old = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // NEW ergonomic API with macros!
    // Use |vars| params: [names] => { body } syntax
    let ode_new = equation::ODE::builder()
        .diffeq(diffeq!(|x, dx, rateiv| params: [ke, _v] => {
            // Variables x, dx, rateiv are requested in |...|
            // Parameters ke, _v are extracted automatically
            dx[0] = -ke * x[0] + rateiv[0];
        }))
        .out(out!(|x, y| params: [_ke, v] => {
            // Variables x, y are requested
            // Parameters _ke, v are extracted automatically
            y[0] = x[0] / v;
        }))
        .nstates(1)
        .nouteqs(1)
        .build();

    let params = vec![1.02282724609375, 194.51904296875]; // ke, v

    let pred_old = ode_old.estimate_predictions(&subject, &params).unwrap();
    let pred_new = ode_new.estimate_predictions(&subject, &params).unwrap();

    println!("Old API predictions:");
    for p in pred_old.flat_predictions() {
        print!("{:.6} ", p);
    }
    println!("\n");

    println!("New API predictions:");
    for p in pred_new.flat_predictions() {
        print!("{:.6} ", p);
    }
    println!("\n");

    // Verify they match
    let match_result = pred_old.flat_predictions()
        .iter()
        .zip(pred_new.flat_predictions().iter())
        .all(|(a, b)| (a - b).abs() < 1e-12);
    
    println!("Results match: {} ✓\n", match_result);

    // ========================================================================
    // Example 2: Using the Params struct for type-safe parameter creation
    // ========================================================================
    
    println!("--- Example 2: Type-Safe Parameter Creation ---");
    
    // Create params using the struct
    let pk = Pk { ke: 1.02282724609375, v: 194.51904296875 };
    println!("Pk struct: {:?}", pk);
    
    // Convert to Vec for use with existing API
    let params_vec = pk.to_vec();
    println!("As Vec: {:?}", params_vec);
    
    // Can also create from HashMap
    let params_map: std::collections::HashMap<&str, f64> = 
        [("ke", 1.02), ("v", 194.5)].into_iter().collect();
    let pk_from_map: Pk = params_map.into();
    println!("From HashMap: {:?}\n", pk_from_map);

    // ========================================================================
    // Example 3: Model with covariates
    // ========================================================================
    
    println!("--- Example 3: Model with Covariates ---");
    
    // Subject with weight covariate
    let subject_with_cov = Subject::builder("patient_2")
        .infusion(0.0, 500.0, 0, 0.5)
        .covariate("wt", 0.0, 70.0)  // 70 kg at time 0
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let ode_with_cov = equation::ODE::builder()
        .diffeq(diffeq!(|x, dx, rateiv, cov, t| params: [ke, _v, theta_wt] => {
            fetch_cov!(cov, t, wt);
            let cl = ke * (wt / 70.0_f64).powf(theta_wt);
            dx[0] = -cl * x[0] + rateiv[0];
        }))
        .out(out!(|x, y| params: [_ke, v, _theta_wt] => {
            y[0] = x[0] / v;
        }))
        .nstates(1)
        .nouteqs(1)
        .build();

    let params_with_theta = vec![1.02, 194.5, 0.75]; // ke, v, theta_wt
    
    let pred_cov = ode_with_cov.estimate_predictions(&subject_with_cov, &params_with_theta).unwrap();
    
    println!("Predictions with covariate model:");
    for p in pred_cov.flat_predictions() {
        print!("{:.6} ", p);
    }
    println!("\n");

    // ========================================================================
    // Summary: API Comparison
    // ========================================================================
    
    println!("=== API Comparison ===\n");
    
    println!("OLD API (verbose closure with 7 arguments):");
    println!("  |x, p, _t, dx, _b, rateiv, _cov| {{");
    println!("      fetch_params!(p, ke, _v);");
    println!("      dx[0] = -ke * x[0] + rateiv[0];");
    println!("  }}");
    println!();
    
    println!("NEW API (diffeq! macro - specify only what you need):");
    println!("  diffeq!(|x, dx, rateiv| params: [ke, _v] => {{");
    println!("      dx[0] = -ke * x[0] + rateiv[0];");
    println!("  }})");
    println!();

    println!("BENEFITS:");
    println!("  ✓ Only specify variables you actually use");
    println!("  ✓ No need to remember closure argument order");
    println!("  ✓ Automatic parameter extraction with params: [...]");
    println!("  ✓ Params struct enables type-safe parameter creation");
    println!("  ✓ Covariates struct enables compile-time covariate checking");
}
