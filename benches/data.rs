use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pharmsol::prelude::*;
use pharmsol::SubjectBuilderExt; // Ensure this trait is in scope

fn subject_builder_benchmark(c: &mut Criterion) {
    // Simple case, few observations
    c.bench_function("SubjectBuilder simple", |b| {
        b.iter(|| {
            let subject = Subject::builder("strongman")
                .bolus(0.0, 100.0, 0)
                .observation(3.0, 100.0, 0)
                .observation(4.0, 200.0, 0)
                .observation(5.0, 300.0, 0)
                .build();
            black_box(subject); // Prevent compiler optimizations
        })
    });

    // Same as above with covariates
    c.bench_function("SubjectBuilder covariates", |b| {
        b.iter(|| {
            let subject = Subject::builder("strongman")
                .bolus(0.0, 100.0, 0)
                .observation(3.0, 100.0, 0)
                .observation(4.0, 200.0, 0)
                .observation(5.0, 300.0, 0)
                .observation(12.0, 300.0, 0)
                .covariate("iron", 0.0, 100.0)
                .covariate("iron", 12.0, 50.0)
                .build();
            black_box(subject); // Prevent compiler optimizations
        })
    });
}

criterion_group!(benches, subject_builder_benchmark);
criterion_main!(benches);
