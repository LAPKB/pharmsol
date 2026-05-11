use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use ndarray::Array2;
use pharmsol::prelude::simulator::{log_likelihood_batch, log_likelihood_matrix};
use pharmsol::prelude::*;
use pharmsol::{Cache, ResidualErrorModel, ResidualErrorModels, ODE};
use std::hint::black_box;
use std::time::Duration;

fn example_equation() -> ODE {
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

fn example_data(n_subjects: usize) -> Data {
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

fn example_subject() -> Subject {
    example_data(1)
        .get_subject("subject_0")
        .expect("example subject exists")
        .clone()
}

fn example_params() -> [f64; 4] {
    [0.60, 0.09, 0.05, 20.0]
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

    let subject = example_subject();
    let params = example_params();
    let equation_cold = example_equation().disable_cache();
    let equation_hot = example_equation();
    let _ = equation_hot
        .estimate_predictions(&subject, params.as_slice())
        .unwrap();
    let predictions = equation_cold
        .estimate_predictions(&subject, params.as_slice())
        .unwrap();

    let mut breakdown_group = c.benchmark_group("ode/runtime-breakdown");
    breakdown_group.sample_size(10);
    breakdown_group.warm_up_time(Duration::from_millis(250));
    breakdown_group.measurement_time(Duration::from_secs(1));
    breakdown_group.bench_function("predict-cold", |b| {
        b.iter(|| {
            black_box(
                equation_cold
                    .estimate_predictions(&subject, params.as_slice())
                    .unwrap(),
            )
        })
    });
    breakdown_group.bench_function("predict-hot", |b| {
        b.iter(|| {
            black_box(
                equation_hot
                    .estimate_predictions(&subject, params.as_slice())
                    .unwrap(),
            )
        })
    });
    breakdown_group.bench_function("score-only", |b| {
        b.iter(|| black_box(predictions.log_likelihood(&assay_error_models)).unwrap())
    });
    breakdown_group.bench_function("loglik-cold", |b| {
        b.iter(|| {
            black_box(
                equation_cold
                    .estimate_log_likelihood(&subject, params.as_slice(), &assay_error_models)
                    .unwrap(),
            )
        })
    });
    breakdown_group.bench_function("loglik-hot", |b| {
        b.iter(|| {
            black_box(
                equation_hot
                    .estimate_log_likelihood(&subject, params.as_slice(), &assay_error_models)
                    .unwrap(),
            )
        })
    });
    breakdown_group.finish();

    let mut matrix_group = c.benchmark_group("likelihood/matrix");
    matrix_group.sampling_mode(SamplingMode::Flat);
    matrix_group.sample_size(10);
    matrix_group.warm_up_time(Duration::from_millis(250));
    matrix_group.measurement_time(Duration::from_secs(2));

    for (n_subjects, n_support_points) in [(16usize, 32usize), (64usize, 128usize)] {
        let case = (example_data(n_subjects), support_points(n_support_points));
        let equation_cold = example_equation().disable_cache();
        let equation_hot = example_equation();
        let _ = log_likelihood_matrix(&equation_hot, &case.0, &case.1, &assay_error_models, false)
            .unwrap();

        matrix_group.throughput(Throughput::Elements((n_subjects * n_support_points) as u64));
        matrix_group.bench_with_input(
            BenchmarkId::new("cold", format!("{n_subjects}x{n_support_points}")),
            &case,
            |b, (data, theta)| {
                b.iter(|| {
                    black_box(log_likelihood_matrix(
                        &equation_cold,
                        data,
                        theta,
                        &assay_error_models,
                        false,
                    ))
                    .unwrap()
                })
            },
        );
        matrix_group.bench_with_input(
            BenchmarkId::new("hot", format!("{n_subjects}x{n_support_points}")),
            &case,
            |b, (data, theta)| {
                b.iter(|| {
                    black_box(log_likelihood_matrix(
                        &equation_hot,
                        data,
                        theta,
                        &assay_error_models,
                        false,
                    ))
                    .unwrap()
                })
            },
        );
    }
    matrix_group.finish();

    let mut batch_group = c.benchmark_group("likelihood/batch");
    batch_group.sample_size(10);
    batch_group.warm_up_time(Duration::from_millis(250));
    batch_group.measurement_time(Duration::from_secs(1));

    for n_subjects in [16usize, 64usize, 256usize] {
        let data = example_data(n_subjects);
        let parameters = batch_parameters(n_subjects);
        let equation_cold = example_equation().disable_cache();
        let equation_hot = example_equation();
        let _ = log_likelihood_batch(&equation_hot, &data, &parameters, &residual_error_models)
            .unwrap();

        batch_group.throughput(Throughput::Elements(n_subjects as u64));
        batch_group.bench_with_input(BenchmarkId::new("cold", n_subjects), &data, |b, data| {
            b.iter(|| {
                black_box(log_likelihood_batch(
                    &equation_cold,
                    data,
                    &parameters,
                    &residual_error_models,
                ))
                .unwrap()
            })
        });
        batch_group.bench_with_input(BenchmarkId::new("hot", n_subjects), &data, |b, data| {
            b.iter(|| {
                black_box(log_likelihood_batch(
                    &equation_hot,
                    data,
                    &parameters,
                    &residual_error_models,
                ))
                .unwrap()
            })
        });
    }
    batch_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
