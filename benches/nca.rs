use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use pharmsol::nca::{NCAOptions, NCA};
use pharmsol::prelude::*;
use std::hint::black_box;

fn typical_oral_subject(id: impl Into<String>, scale: f64) -> Subject {
    Subject::builder(id.into())
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(0.25, 2.5 * scale, 0)
        .observation(0.5, 5.0 * scale, 0)
        .observation(1.0, 8.0 * scale, 0)
        .observation(2.0, 10.0 * scale, 0)
        .observation(4.0, 7.5 * scale, 0)
        .observation(6.0, 5.0 * scale, 0)
        .observation(8.0, 3.5 * scale, 0)
        .observation(12.0, 1.5 * scale, 0)
        .observation(16.0, 0.8 * scale, 0)
        .observation(24.0, 0.2 * scale, 0)
        .observation(36.0, 0.05 * scale, 0)
        .build()
}

fn build_population(n: usize) -> Data {
    let subjects: Vec<Subject> = (0..n)
        .map(|i| typical_oral_subject(format!("subj_{i}"), 1.0 + (i as f64 % 7.0) * 0.05))
        .collect();
    Data::new(subjects)
}

fn criterion_benchmark(c: &mut Criterion) {
    let data = build_population(128);
    let opts = NCAOptions::default();

    let mut group = c.benchmark_group("nca");

    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(128));
    group.bench_function("population-128", |b| {
        b.iter(|| black_box(black_box(&data).nca_all(black_box(&opts))))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
