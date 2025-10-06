use criterion::{criterion_group, criterion_main, Criterion};
use pharmsol::*;
use std::hint::black_box;

fn example_subject() -> Subject {
    Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(3.0, 0.1, 0)
        .observation(6.0, 0.4, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 1.1, 0)
        .build()
}

fn example_subject_covariates() -> Subject {
    Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .covariate("iron", 0.0, 10.0)
        .observation(3.0, 0.1, 0)
        .covariate("iron", 3.0, 40.0)
        .observation(6.0, 0.4, 0)
        .covariate("iron", 6.0, 40.0)
        .observation(12.0, 1.0, 0)
        .covariate("iron", 12.0, 50.0)
        .observation(24.0, 1.1, 0)
        .covariate("iron", 24.0, 100.0)
        .build()
}

fn one_compartment() {
    let subject = example_subject();
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _tlag, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    black_box(
        ode.estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 70.0])
            .unwrap(),
    );
}

fn one_compartment_covariates() {
    let subject = example_subject_covariates();
    let ode = equation::ODE::new(
        |x, p, t, dx, _b, _rateiv, cov| {
            fetch_cov!(cov, t, iron);
            fetch_params!(p, ka, ke, _tlag, _v);

            ke = ke * iron.powf(0.75) / 50.0;
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    black_box(
        ode.estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 70.0])
            .unwrap(),
    );
}

fn two_compartment() {
    let subject = example_subject();
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, k12, k21, _tlag, _v);
            dx[0] = -ka * x[0] - k12 * x[0] + k21 * x[1];
            dx[1] = k12 * x[0] - k21 * x[1] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _k12, _k21, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    black_box(
        ode.estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 0.04, 0.08, 70.0])
            .unwrap(),
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("one_compartment", |b| b.iter(one_compartment));
    c.bench_function("one_compartment_covariates", |b| {
        b.iter(one_compartment_covariates)
    });
    c.bench_function("two_compartment", |b| b.iter(two_compartment));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
