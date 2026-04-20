//! Benchmark: native Rust JIT call timing.
//!
//! Compiles the same one-compartment model used by the R benchmark
//! (`pharmsolr/inst/examples/bench.R`) and measures per-iteration time
//! of `simulate_subject`.
//!
//! Run with:
//! ```bash
//! cargo run --release --features jit --example jit_bench
//! ```

#[cfg(not(feature = "jit"))]
fn main() {
    eprintln!("Re-run with `--features jit`.");
}

#[cfg(feature = "jit")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pharmsol::jit::Model;
    use pharmsol::prelude::*;
    use std::time::Instant;

    let text = "
        name         = onecmt-allo
        compartments = central
        params       = CL, V
        covariates   = WT
        dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central
        out(cp)       = central / V
    ";

    // Compile time
    let t0 = Instant::now();
    let ode = Model::from_text(text)?.compile()?;
    let compile_ns = t0.elapsed().as_nanos();
    println!("compile: {:.3} ms", compile_ns as f64 / 1.0e6);

    let subject = Subject::builder("p1")
        .infusion(0.0, 100.0, 0, 0.5)
        .covariate("WT", 0.0, 80.0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();
    let params = [5.0, 50.0];

    // Warm-up
    for _ in 0..50 {
        let _ = ode.simulate_subject(&subject, &params, None)?;
    }

    let n: usize = std::env::var("BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let t0 = Instant::now();
    for _ in 0..n {
        let _ = ode.simulate_subject(&subject, &params, None)?;
    }
    let total = t0.elapsed();
    println!(
        "simulate_subject (reused Subject):  n={n}  total={:.3} s  per_call={:.3} us  ({:.0} calls/s)",
        total.as_secs_f64(),
        total.as_nanos() as f64 / n as f64 / 1.0e3,
        n as f64 / total.as_secs_f64()
    );

    let t0 = Instant::now();
    for _ in 0..n {
        let s = Subject::builder("p1")
            .infusion(0.0, 100.0, 0, 0.5)
            .covariate("WT", 0.0, 80.0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .observation(4.0, 0.0, 0)
            .observation(8.0, 0.0, 0)
            .observation(12.0, 0.0, 0)
            .build();
        let _ = ode.simulate_subject(&s, &params, None)?;
    }
    let total = t0.elapsed();
    println!(
        "simulate_subject (rebuild Subject): n={n}  total={:.3} s  per_call={:.3} us  ({:.0} calls/s)",
        total.as_secs_f64(),
        total.as_nanos() as f64 / n as f64 / 1.0e3,
        n as f64 / total.as_secs_f64()
    );

    Ok(())
}
