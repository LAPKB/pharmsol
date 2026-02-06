use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pharmsol::*;
use std::hint::black_box;

/// Subject for IV bolus models (one-compartment and two-compartment IV)
fn example_subject_iv() -> Subject {
    Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build()
}

/// Subject for oral absorption models (dose into depot compartment)
fn example_subject_oral() -> Subject {
    Subject::builder("1")
        .bolus(0.0, 500.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build()
}

// =============================================================================
// One-compartment IV models
// =============================================================================

fn one_compartment_iv_ode(subject: &Subject, params: &Vec<f64>) {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
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
    black_box(ode.estimate_predictions(subject, params).unwrap());
}

fn one_compartment_iv_analytical(subject: &Subject, params: &Vec<f64>) {
    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
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
    black_box(analytical.estimate_predictions(subject, params).unwrap());
}

// =============================================================================
// One-compartment with oral absorption models
// =============================================================================

fn one_compartment_oral_ode(subject: &Subject, params: &Vec<f64>) {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );
    black_box(ode.estimate_predictions(subject, params).unwrap());
}

fn one_compartment_oral_analytical(subject: &Subject, params: &Vec<f64>) {
    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );
    black_box(analytical.estimate_predictions(subject, params).unwrap());
}

// =============================================================================
// Two-compartment IV models
// =============================================================================

fn two_compartment_iv_ode(subject: &Subject, params: &Vec<f64>) {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, k12, k21, _v);
            dx[0] = -ke * x[0] - k12 * x[0] + k21 * x[1] + rateiv[0];
            dx[1] = k12 * x[0] - k21 * x[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _k12, _k21, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (2, 1),
    );
    black_box(ode.estimate_predictions(subject, params).unwrap());
}

fn two_compartment_iv_analytical(subject: &Subject, params: &Vec<f64>) {
    let analytical = equation::Analytical::new(
        two_compartments,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _k12, _k21, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (2, 1),
    );
    black_box(analytical.estimate_predictions(subject, params).unwrap());
}

// =============================================================================
// Two-compartment with oral absorption models
// =============================================================================

fn two_compartment_oral_ode(subject: &Subject, params: &Vec<f64>) {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, k12, k21, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] - k12 * x[1] + k21 * x[2];
            dx[2] = k12 * x[1] - k21 * x[2];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (3, 1),
    );
    black_box(ode.estimate_predictions(subject, params).unwrap());
}

fn two_compartment_oral_analytical(subject: &Subject, params: &Vec<f64>) {
    let analytical = equation::Analytical::new(
        two_compartments_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (3, 1),
    );
    black_box(analytical.estimate_predictions(subject, params).unwrap());
}

// =============================================================================
// Benchmark
// =============================================================================

fn analytical_vs_ode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Analytical vs ODE");

    let subject_iv = example_subject_iv();
    let subject_oral = example_subject_oral();

    // Parameter vectors
    let params_1cmt_iv = vec![0.1, 50.0]; // ke, v
    let params_1cmt_oral = vec![1.0, 0.1, 50.0]; // ka, ke, v
    let params_2cmt_iv = vec![0.1, 0.3, 0.2, 50.0]; // ke, k12, k21, v
    let params_2cmt_oral = vec![1.0, 0.1, 0.3, 0.2, 50.0]; // ka, ke, k12, k21, v

    // One-compartment IV
    group.bench_with_input(
        BenchmarkId::new("One-compartment IV", "ODE"),
        &(&subject_iv, &params_1cmt_iv),
        |b, (s, p)| b.iter(|| one_compartment_iv_ode(s, p)),
    );
    group.bench_with_input(
        BenchmarkId::new("One-compartment IV", "Analytical"),
        &(&subject_iv, &params_1cmt_iv),
        |b, (s, p)| b.iter(|| one_compartment_iv_analytical(s, p)),
    );

    // One-compartment with absorption
    group.bench_with_input(
        BenchmarkId::new("One-compartment oral", "ODE"),
        &(&subject_oral, &params_1cmt_oral),
        |b, (s, p)| b.iter(|| one_compartment_oral_ode(s, p)),
    );
    group.bench_with_input(
        BenchmarkId::new("One-compartment oral", "Analytical"),
        &(&subject_oral, &params_1cmt_oral),
        |b, (s, p)| b.iter(|| one_compartment_oral_analytical(s, p)),
    );

    // Two-compartment IV
    group.bench_with_input(
        BenchmarkId::new("Two-compartment IV", "ODE"),
        &(&subject_iv, &params_2cmt_iv),
        |b, (s, p)| b.iter(|| two_compartment_iv_ode(s, p)),
    );
    group.bench_with_input(
        BenchmarkId::new("Two-compartment IV", "Analytical"),
        &(&subject_iv, &params_2cmt_iv),
        |b, (s, p)| b.iter(|| two_compartment_iv_analytical(s, p)),
    );

    // Two-compartment with absorption
    group.bench_with_input(
        BenchmarkId::new("Two-compartment oral", "ODE"),
        &(&subject_oral, &params_2cmt_oral),
        |b, (s, p)| b.iter(|| two_compartment_oral_ode(s, p)),
    );
    group.bench_with_input(
        BenchmarkId::new("Two-compartment oral", "Analytical"),
        &(&subject_oral, &params_2cmt_oral),
        |b, (s, p)| b.iter(|| two_compartment_oral_analytical(s, p)),
    );

    group.finish();
}

criterion_group!(benches, analytical_vs_ode_benchmark);
criterion_main!(benches);
