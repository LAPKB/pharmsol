use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pharmsol::prelude::*;
use pharmsol::nca::{lambda_z_candidates, NCAOptions};
use std::hint::black_box;

/// Build a typical PK subject with 12 time points (oral dose)
fn typical_oral_subject(id: &str) -> Subject {
    Subject::builder(id)
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(0.25, 2.5, 0)
        .observation(0.5, 5.0, 0)
        .observation(1.0, 8.0, 0)
        .observation(2.0, 10.0, 0)
        .observation(4.0, 7.5, 0)
        .observation(6.0, 5.0, 0)
        .observation(8.0, 3.5, 0)
        .observation(12.0, 1.5, 0)
        .observation(16.0, 0.8, 0)
        .observation(24.0, 0.2, 0)
        .observation(36.0, 0.05, 0)
        .build()
}

/// Build a population of n subjects with slight variation
fn build_population(n: usize) -> Data {
    let subjects: Vec<Subject> = (0..n)
        .map(|i| {
            let scale = 1.0 + (i as f64 % 7.0) * 0.05; // slight variation
            Subject::builder(&format!("subj_{}", i))
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
        })
        .collect();
    Data::new(subjects)
}

fn bench_single_subject_nca(c: &mut Criterion) {
    let subject = typical_oral_subject("bench_subj");
    let opts = NCAOptions::default();

    c.bench_function("nca_single_subject", |b| {
        b.iter(|| {
            let result = black_box(&subject).nca(black_box(&opts), 0);
            black_box(result);
        });
    });
}

fn bench_population_nca(c: &mut Criterion) {
    let mut group = c.benchmark_group("nca_population");

    for size in [10, 100, 500] {
        let data = build_population(size);
        let opts = NCAOptions::default();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let results = black_box(&data).nca(black_box(&opts), 0);
                black_box(results);
            });
        });
    }

    group.finish();
}

fn bench_lambda_z_candidates(c: &mut Criterion) {
    use pharmsol::data::observation::ObservationProfile;
    use pharmsol::nca::LambdaZOptions;
    use pharmsol::data::event::{AUCMethod, BLQRule};

    let subject = typical_oral_subject("bench_subj");
    let occ = &subject.occasions()[0];
    let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();
    let lz_opts = LambdaZOptions::default();

    // Get AUClast for the candidate scoring
    let auc_results = subject.auc(0, &AUCMethod::Linear, &BLQRule::Exclude);
    let auc_last = auc_results[0].as_ref().copied().unwrap_or(50.0);

    c.bench_function("nca_lambda_z_candidates", |b| {
        b.iter(|| {
            let candidates =
                lambda_z_candidates(black_box(&profile), black_box(&lz_opts), black_box(auc_last));
            black_box(candidates);
        });
    });
}

fn bench_observation_metrics(c: &mut Criterion) {
    use pharmsol::data::event::{AUCMethod, BLQRule};

    let subject = typical_oral_subject("bench_subj");

    c.bench_function("nca_auc_cmax_metrics", |b| {
        b.iter(|| {
            let auc = black_box(&subject).auc(0, &AUCMethod::Linear, &BLQRule::Exclude);
            let cmax = black_box(&subject).cmax(0, &BLQRule::Exclude);
            black_box((auc, cmax));
        });
    });
}

criterion_group!(
    benches,
    bench_single_subject_nca,
    bench_population_nca,
    bench_lambda_z_candidates,
    bench_observation_metrics,
);
criterion_main!(benches);
