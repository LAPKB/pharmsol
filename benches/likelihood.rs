use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use ndarray::Array2;
use pharmsol::prelude::simulator::{log_likelihood_batch, log_likelihood_matrix};
use pharmsol::prelude::*;
use pharmsol::{ResidualErrorModel, ResidualErrorModels, ODE};
use std::hint::black_box;
use std::time::Duration;

fn benchmark_equation() -> ODE {
    equation::ODE::new(
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
    )
    .with_nstates(2)
    .with_nout(1)
}

fn benchmark_data(n_subjects: usize) -> Data {
    let subjects = (0..n_subjects)
        .map(|index| {
            let dose = 100.0 + index as f64;
            let offset = index as f64 * 0.02;
            Subject::builder(format!("subject_{index}"))
                .bolus(0.0, dose, 0)
                .observation(0.5, 4.2 + offset, 0)
                .observation(1.0, 5.9 + offset, 0)
                .observation(2.0, 7.4 + offset, 0)
                .observation(4.0, 6.8 + offset, 0)
                .observation(8.0, 4.7 + offset, 0)
                .observation(12.0, 3.3 + offset, 0)
                .build()
        })
        .collect();

    Data::new(subjects)
}

fn support_points(n_support_points: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_support_points, 4), |(row, column)| match column {
        0 => 0.55 + row as f64 * 0.002,
        1 => 0.08 + row as f64 * 0.0004,
        2 => 0.05,
        3 => 18.0 + row as f64 * 0.15,
        _ => unreachable!(),
    })
}

fn batch_parameters(n_subjects: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_subjects, 4), |(row, column)| match column {
        0 => 0.60 + row as f64 * 0.001,
        1 => 0.09 + row as f64 * 0.0002,
        2 => 0.05,
        3 => 20.0 + row as f64 * 0.05,
        _ => unreachable!(),
    })
}

fn assay_error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

fn residual_error_models() -> ResidualErrorModels {
    ResidualErrorModels::new().add(0, ResidualErrorModel::constant(0.2))
}

fn criterion_benchmark(c: &mut Criterion) {
    let assay_error_models = assay_error_models();
    let residual_error_models = residual_error_models();

    let matrix_data = benchmark_data(64);
    let matrix_theta = support_points(128);
    let matrix_equation = benchmark_equation();
    let _ = log_likelihood_matrix(
        &matrix_equation,
        &matrix_data,
        &matrix_theta,
        &assay_error_models,
        false,
    )
    .unwrap();

    let batch_data = benchmark_data(256);
    let batch_theta = batch_parameters(256);
    let batch_equation = benchmark_equation();
    let _ = log_likelihood_batch(
        &batch_equation,
        &batch_data,
        &batch_theta,
        &residual_error_models,
    )
    .unwrap();

    let mut matrix_group = c.benchmark_group("population/likelihood-matrix");

    matrix_group.sampling_mode(SamplingMode::Flat);
    matrix_group.sample_size(10);
    matrix_group.measurement_time(Duration::from_secs(30));
    matrix_group.throughput(Throughput::Elements((64 * 128) as u64));
    matrix_group.bench_function("ode-hot-64x128", |b| {
        b.iter(|| {
            black_box(log_likelihood_matrix(
                black_box(&matrix_equation),
                black_box(&matrix_data),
                black_box(&matrix_theta),
                black_box(&assay_error_models),
                false,
            ))
            .unwrap()
        })
    });
    matrix_group.finish();

    let mut batch_group = c.benchmark_group("population/likelihood-batch");

    batch_group.sampling_mode(SamplingMode::Flat);
    batch_group.throughput(Throughput::Elements(256));
    batch_group.bench_function("ode-hot-256", |b| {
        b.iter(|| {
            black_box(log_likelihood_batch(
                black_box(&batch_equation),
                black_box(&batch_data),
                black_box(&batch_theta),
                black_box(&residual_error_models),
            ))
            .unwrap()
        })
    });
    batch_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
