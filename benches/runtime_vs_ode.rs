use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pharmsol::*;
use std::hint::black_box;

fn subject_no_covariates() -> Subject {
    Subject::builder("runtime-vs-ode-no-cov")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build()
}

fn subject_with_covariates() -> Subject {
    Subject::builder("runtime-vs-ode-cov")
        .infusion(0.0, 500.0, 0, 0.5)
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 1.0, 84.0)
        .covariate("wt", 4.0, 80.0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build()
}

fn runtime_vs_regular_ode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RuntimeODE vs ODE");

    let subject_plain = subject_no_covariates();
    let params_plain = vec![0.8, 200.0];

    let regular_plain = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            let ke = p[0];
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| HashMap::new(),
        |_p, _t, _cov| HashMap::new(),
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            let v = p[1];
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    let runtime_plain = RuntimeODE::from_model(RuntimeOdeModel {
        states: vec!["A".to_string()],
        parameters: vec!["ke".to_string(), "v".to_string()],
        outputs: vec!["cp".to_string()],
        derivatives: vec!["-ke * A + rateiv[0]".to_string()],
        output_equations: vec!["A / v".to_string()],
        covariates: vec![],
        init: vec![],
    })
    .unwrap();

    group.bench_with_input(
        BenchmarkId::new("One-compartment no cov", "ODE"),
        &(),
        |b, _| {
            b.iter(|| {
                black_box(
                    regular_plain
                        .estimate_predictions(black_box(&subject_plain), black_box(&params_plain))
                        .unwrap(),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("One-compartment no cov", "RuntimeODE"),
        &(),
        |b, _| {
            b.iter(|| {
                black_box(
                    runtime_plain
                        .estimate_predictions(black_box(&subject_plain), black_box(&params_plain))
                        .unwrap(),
                )
            })
        },
    );

    let subject_cov = subject_with_covariates();
    let params_cov = vec![0.8, 200.0];

    let regular_cov = equation::ODE::new(
        |x, p, t, dx, _b, rateiv, cov| {
            let ke = p[0];
            let wt = cov.get_covariate("wt").unwrap().interpolate(t).unwrap();
            dx[0] = -(ke * wt / 70.0) * x[0] + rateiv[0];
        },
        |_p, _t, _cov| HashMap::new(),
        |_p, _t, _cov| HashMap::new(),
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            let v = p[1];
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    let runtime_cov = RuntimeODE::from_model(RuntimeOdeModel {
        states: vec!["A".to_string()],
        parameters: vec!["ke".to_string(), "v".to_string()],
        outputs: vec!["cp".to_string()],
        derivatives: vec!["-(ke * wt / 70.0) * A + rateiv[0]".to_string()],
        output_equations: vec!["A / v".to_string()],
        covariates: vec!["wt".to_string()],
        init: vec![],
    })
    .unwrap();

    group.bench_with_input(
        BenchmarkId::new("One-compartment cov", "ODE"),
        &(),
        |b, _| {
            b.iter(|| {
                black_box(
                    regular_cov
                        .estimate_predictions(black_box(&subject_cov), black_box(&params_cov))
                        .unwrap(),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("One-compartment cov", "RuntimeODE"),
        &(),
        |b, _| {
            b.iter(|| {
                black_box(
                    runtime_cov
                        .estimate_predictions(black_box(&subject_cov), black_box(&params_cov))
                        .unwrap(),
                )
            })
        },
    );

    group.finish();
}

criterion_group!(benches, runtime_vs_regular_ode_benchmark);
criterion_main!(benches);
