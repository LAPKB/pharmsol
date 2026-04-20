//! JIT-compiled one-compartment model.
//!
//! Run with:
//! ```bash
//! cargo run --features jit --example jit_one_compartment
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

    let ode = Model::new("1cmt-iv-allo")
        .compartments(["central"])
        .params(["CL", "V"])
        .covariates(["WT"])
        .dxdt(
            "central",
            "rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central",
        )
        .output("cp", "central / V")
        .compile()?;

    let subject = Subject::builder("p1")
        .infusion(0.0, 100.0, 0, 0.5)
        .covariate("WT", 0.0, 80.0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let (preds, _) = ode.simulate_subject(&subject, &[5.0, 50.0], None)?;

    println!(" t      cp");
    for p in preds.get_predictions() {
        println!("{:5.1}   {:>8.4}", p.time(), p.prediction());
    }
    Ok(())
}
