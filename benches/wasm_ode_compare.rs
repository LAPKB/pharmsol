use criterion::{criterion_group, criterion_main, Criterion};
use pharmsol::*;
use std::hint::black_box;

fn example_subject() -> Subject {
    Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 1.645776, 0)
        .observation(1.0, 1.216442, 0)
        .observation(2.0, 0.4622729, 0)
        .observation(3.0, 0.1697458, 0)
        .observation(4.0, 0.06382178, 0)
        .observation(6.0, 0.009099384, 0)
        .observation(8.0, 0.001017932, 0)
        .missing_observation(12.0, 0)
        .build()
}

fn regular_ode_predictions(c: &mut Criterion) {
    let subject = example_subject();
    let ode = equation::ODE::new(
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
    let params = vec![1.02282724609375, 194.51904296875];

    c.bench_function("regular_ode_predictions", |b| {
        b.iter(|| {
            black_box(ode.estimate_predictions(&subject, &params).unwrap());
        })
    });
}

fn wasm_ir_ode_predictions(c: &mut Criterion) {
    let subject = example_subject();

    // Setup WASM IR model
    let test_dir = std::env::current_dir().expect("Failed to get current directory");
    let ir_path = test_dir.join("test_model_ir_bench.pkm");

    let _ir_file = exa_wasm::build::emit_ir::<equation::ODE>(
        "|x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        }"
        .to_string(),
        None,
        None,
        Some("|p, _t, _cov, x| { }".to_string()),
        Some(
            "|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        }"
            .to_string(),
        ),
        Some(ir_path.clone()),
        vec!["ke".to_string(), "v".to_string()],
    )
    .expect("emit_ir failed");

    let (wasm_ode, _meta, _id) =
        exa_wasm::interpreter::load_ir_ode(ir_path.clone()).expect("load_ir_ode failed");

    let params = vec![1.02282724609375, 194.51904296875];

    c.bench_function("wasm_ir_ode_predictions", |b| {
        b.iter(|| {
            black_box(wasm_ode.estimate_predictions(&subject, &params).unwrap());
        })
    });

    // Clean up
    std::fs::remove_file(ir_path).ok();
}

fn regular_ode_likelihood(c: &mut Criterion) {
    let subject = example_subject();
    let ode = equation::ODE::new(
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
    let params = vec![1.02282724609375, 194.51904296875];
    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();

    c.bench_function("regular_ode_likelihood", |b| {
        b.iter(|| {
            black_box(
                ode.estimate_likelihood(&subject, &params, &ems, false)
                    .unwrap(),
            );
        })
    });
}

fn wasm_ir_ode_likelihood(c: &mut Criterion) {
    let subject = example_subject();

    // Setup WASM IR model
    let test_dir = std::env::current_dir().expect("Failed to get current directory");
    let ir_path = test_dir.join("test_model_ir_bench_ll.pkm");

    let _ir_file = exa_wasm::build::emit_ir::<equation::ODE>(
        "|x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        }"
        .to_string(),
        None,
        None,
        Some("|p, _t, _cov, x| { }".to_string()),
        Some(
            "|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        }"
            .to_string(),
        ),
        Some(ir_path.clone()),
        vec!["ke".to_string(), "v".to_string()],
    )
    .expect("emit_ir failed");

    let (wasm_ode, _meta, _id) =
        exa_wasm::interpreter::load_ir_ode(ir_path.clone()).expect("load_ir_ode failed");

    let params = vec![1.02282724609375, 194.51904296875];
    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();

    c.bench_function("wasm_ir_ode_likelihood", |b| {
        b.iter(|| {
            black_box(
                wasm_ode
                    .estimate_likelihood(&subject, &params, &ems, false)
                    .unwrap(),
            );
        })
    });

    // Clean up
    std::fs::remove_file(ir_path).ok();
}

fn criterion_benchmark(c: &mut Criterion) {
    regular_ode_predictions(c);
    wasm_ir_ode_predictions(c);
    regular_ode_likelihood(c);
    wasm_ir_ode_likelihood(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
