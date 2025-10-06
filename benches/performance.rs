use criterion::{criterion_group, criterion_main, Criterion};
use pharmsol::*;
use std::hint::black_box;

fn example_subject() -> Subject {
    Subject::builder("id1")
        .bolus(0.0, 100.0, 0)
        .repeat(2, 0.5)
        .observation(0.5, 0.1, 0)
        .observation(1.0, 0.4, 0)
        .observation(2.0, 1.0, 0)
        .observation(2.5, 1.1, 0)
        .covariate("wt", 0.0, 80.0)
        .covariate("wt", 1.0, 83.0)
        .covariate("age", 0.0, 25.0)
        .build()
}

fn readme(n: usize) {
    let subject = example_subject();
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            // fetch_cov!(cov, t, _wt, _age);
            fetch_params!(p, ka, ke, _tlag, _v);
            //Struct
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
    for _ in 0..n {
        let op = ode
            .estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 70.0])
            .unwrap();
        black_box(op);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("readme 20", |b| b.iter(|| readme(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
