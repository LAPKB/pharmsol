use pharmsol::prelude::*;

fn main() {
    // Demonstrate the new covariate functionality
    println!("=== Covariate Refactoring Demo ===\n");

    // 1. Create covariates directly with observations
    let mut covariates = Covariates::new();
    
    println!("1. Adding observations directly to covariates:");
    covariates.add_observation("weight", 0.0, 70.0, false);
    covariates.add_observation("weight", 12.0, 72.0, false);
    covariates.add_observation("weight", 24.0, 75.0, false);
    covariates.add_observation("age", 0.0, 35.0, true); // Fixed covariate
    
    // Test interpolation
    println!("Weight at time 6.0: {:.1} kg", 
             covariates.get_covariate_mut("weight").unwrap().interpolate(6.0).unwrap());
    println!("Weight at time 18.0: {:.1} kg", 
             covariates.get_covariate_mut("weight").unwrap().interpolate(18.0).unwrap());
    println!("Age at time 100.0: {:.1} years", 
             covariates.get_covariate_mut("age").unwrap().interpolate(100.0).unwrap());
    
    // 2. Update observations dynamically
    println!("\n2. Updating observations dynamically:");
    println!("Updating weight at time 12.0 from 72.0 to 74.0 kg");
    covariates.update_observation("weight", 12.0, 74.0);
    
    println!("Weight at time 6.0 after update: {:.1} kg", 
             covariates.get_covariate_mut("weight").unwrap().interpolate(6.0).unwrap());
    println!("Weight at time 18.0 after update: {:.1} kg", 
             covariates.get_covariate_mut("weight").unwrap().interpolate(18.0).unwrap());
    
    // 3. Update interpolation methods for individual segments
    println!("\n3. Updating interpolation methods:");
    println!("Changing first segment to carry forward (constant value)");
    covariates.update_covariate_segment("weight", 6.0, Interpolation::CarryForward { value: 71.0 });
    
    println!("Weight at time 6.0 with carry forward: {:.1} kg", 
             covariates.get_covariate_mut("weight").unwrap().interpolate(6.0).unwrap());
    
    // 4. Demonstrate Pmetrics parsing
    println!("\n4. Parsing from Pmetrics format:");
    let mut pmetrics_data = std::collections::HashMap::new();
    pmetrics_data.insert("bmi".to_string(), vec![(0.0, Some(25.0)), (30.0, Some(24.5))]);
    pmetrics_data.insert("sex!".to_string(), vec![(0.0, Some(1.0))]); // Fixed covariate (! suffix)
    
    let pmetrics_covariates = Covariates::from_pmetrics_observations(&pmetrics_data);
    println!("BMI at time 15.0: {:.1}", 
             pmetrics_covariates.get_covariate("bmi").unwrap().interpolate_immutable(15.0).unwrap());
    println!("Sex at time 50.0 (fixed): {:.0}", 
             pmetrics_covariates.get_covariate("sex").unwrap().interpolate_immutable(50.0).unwrap());
    
    println!("\n=== Demo Complete ===");
}
