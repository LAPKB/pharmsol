use criterion::{criterion_group, criterion_main, Criterion};
use pharmsol::prelude::*;
use pharmsol::SubjectBuilderExt; // Ensure this trait is in scope
use std::hint::black_box;

fn subject_builder_benchmark(c: &mut Criterion) {
    // Simple subject creation
    c.bench_function("SubjectBuilder simple", |b| {
        b.iter(|| {
            let subject = Subject::builder("patient1")
                .bolus(0.0, 100.0, 0)
                .observation(3.0, 100.0, 0)
                .observation(4.0, 200.0, 0)
                .observation(5.0, 300.0, 0)
                .build();
            black_box(subject);
        })
    });

    // Subject with covariates
    c.bench_function("SubjectBuilder with covariates", |b| {
        b.iter(|| {
            let subject = Subject::builder("patient1")
                .bolus(0.0, 100.0, 0)
                .observation(3.0, 100.0, 0)
                .observation(4.0, 200.0, 0)
                .observation(5.0, 300.0, 0)
                .observation(12.0, 300.0, 0)
                .covariate("age", 0.0, 30.0)
                .covariate("weight", 0.0, 70.0)
                .build();
            black_box(subject);
        })
    });

    // Multi-occasion subject
    c.bench_function("SubjectBuilder multi-occasion", |b| {
        b.iter(|| {
            let subject = Subject::builder("patient1")
                .bolus(0.0, 100.0, 0)
                .observation(1.0, 50.0, 0)
                .observation(6.0, 25.0, 0)
                .covariate("age", 0.0, 30.0)
                .reset()
                .bolus(24.0, 120.0, 0)
                .observation(25.0, 55.0, 0)
                .observation(30.0, 30.0, 0)
                .covariate("age", 24.0, 30.0)
                .build();
            black_box(subject);
        })
    });
}

fn data_expand_benchmark(c: &mut Criterion) {
    // Create datasets for expansion testing
    let simple_data = {
        let subject = Subject::builder("patient1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 50.0, 0)
            .observation(6.0, 25.0, 0)
            .observation(12.0, 15.0, 0)
            .build();
        Data::new(vec![subject])
    };

    let complex_data = {
        let subjects = (1..=10)
            .map(|i| {
                Subject::builder(&format!("patient{}", i))
                    .bolus(0.0, 100.0 * i as f64, 0)
                    .observation(1.0, 50.0, 0)
                    .observation(2.0, 45.0, 1) // Different output equation
                    .observation(6.0, 25.0, 0)
                    .observation(24.0, 8.0, 1)
                    .infusion(48.0, 200.0, 0, 2.0)
                    .observation(50.0, 30.0, 0)
                    .build()
            })
            .collect();
        Data::new(subjects)
    };

    // Test different expansion intervals
    c.bench_function("Data expand simple (1h intervals)", |b| {
        b.iter(|| {
            let expanded = simple_data.expand(1.0, 0.0);
            black_box(expanded);
        })
    });

    c.bench_function("Data expand complex (1h intervals)", |b| {
        b.iter(|| {
            let expanded = complex_data.expand(1.0, 0.0);
            black_box(expanded);
        })
    });

    c.bench_function("Data expand with additional time", |b| {
        b.iter(|| {
            let expanded = complex_data.expand(1.0, 24.0);
            black_box(expanded);
        })
    });
}

fn dose_modification_benchmark(c: &mut Criterion) {
    let create_dosing_data = || {
        let subjects = (1..=5)
            .map(|i| {
                Subject::builder(&format!("patient{}", i))
                    .bolus(0.0, 100.0, 0)
                    .bolus(12.0, 100.0, 0)
                    .bolus(24.0, 100.0, 0)
                    .infusion(48.0, 200.0, 0, 2.0)
                    .infusion(72.0, 250.0, 0, 3.0)
                    .observation(1.0, 50.0, 0)
                    .observation(25.0, 45.0, 0)
                    .observation(50.0, 30.0, 0)
                    .build()
            })
            .collect();
        Data::new(subjects)
    };

    c.bench_function("Modify all bolus doses", |b| {
        b.iter(|| {
            let mut data = create_dosing_data();
            for subject in data.iter_mut() {
                for occasion in subject.occasions_iter_mut() {
                    for event in occasion.events_iter_mut() {
                        if let Event::Bolus(bolus) = event {
                            bolus.set_amount(bolus.amount() * 2.0);
                        }
                    }
                }
            }
            black_box(data);
        })
    });

    c.bench_function("Modify all infusion doses", |b| {
        b.iter(|| {
            let mut data = create_dosing_data();
            for subject in data.iter_mut() {
                for occasion in subject.occasions_iter_mut() {
                    for event in occasion.events_iter_mut() {
                        if let Event::Infusion(infusion) = event {
                            infusion.set_amount(infusion.amount() * 1.5);
                            infusion.set_duration(infusion.duration() * 0.8);
                        }
                    }
                }
            }
            black_box(data);
        })
    });

    c.bench_function("Conditional dose modification", |b| {
        b.iter(|| {
            let mut data = create_dosing_data();
            for subject in data.iter_mut() {
                for occasion in subject.occasions_iter_mut() {
                    for event in occasion.events_iter_mut() {
                        match event {
                            Event::Bolus(bolus) if bolus.time() >= 24.0 => {
                                bolus.set_amount(bolus.amount() * 1.2);
                            }
                            Event::Infusion(infusion) if infusion.time() >= 48.0 => {
                                infusion.set_amount(infusion.amount() * 0.9);
                            }
                            _ => {}
                        }
                    }
                }
            }
            black_box(data);
        })
    });
}

fn data_operations_benchmark(c: &mut Criterion) {
    // Combined benchmark for data filtering and large dataset creation
    let large_data = {
        let subjects = (1..=100)
            .map(|i| {
                Subject::builder(&format!("patient{:03}", i))
                    .bolus(0.0, 100.0, 0)
                    .observation(1.0, 50.0, 0)
                    .observation(6.0, 25.0, 0)
                    .observation(12.0, 15.0, 0)
                    .covariate("age", 0.0, 20.0 + (i as f64 % 50.0))
                    .covariate("weight", 0.0, 60.0 + (i as f64 % 40.0))
                    .build()
            })
            .collect();
        Data::new(subjects)
    };

    c.bench_function("Create large dataset (100 subjects)", |b| {
        b.iter(|| {
            let subjects: Vec<Subject> = (1..=100)
                .map(|i| {
                    Subject::builder(&format!("patient{:03}", i))
                        .bolus(0.0, 100.0, 0)
                        .observation(0.5, 80.0, 0)
                        .observation(1.0, 60.0, 0)
                        .observation(2.0, 45.0, 0)
                        .observation(4.0, 30.0, 0)
                        .observation(8.0, 20.0, 0)
                        .observation(12.0, 15.0, 0)
                        .observation(24.0, 8.0, 0)
                        .covariate("age", 0.0, 25.0 + (i as f64 % 50.0))
                        .covariate("weight", 0.0, 60.0 + (i as f64 % 40.0))
                        .build()
                })
                .collect();
            let data = Data::new(subjects);
            black_box(data);
        })
    });

    c.bench_function("Filter include subjects", |b| {
        b.iter(|| {
            let include_ids: Vec<String> = (1..=20).map(|i| format!("patient{:03}", i)).collect();
            let filtered = large_data.filter_include(&include_ids);
            black_box(filtered);
        })
    });

    c.bench_function("Filter exclude subjects", |b| {
        b.iter(|| {
            let exclude_ids: Vec<String> = (80..=100).map(|i| format!("patient{:03}", i)).collect();
            let filtered = large_data.filter_exclude(exclude_ids);
            black_box(filtered);
        })
    });
}

criterion_group!(
    benches,
    subject_builder_benchmark,
    data_expand_benchmark,
    dose_modification_benchmark,
    data_operations_benchmark
);
criterion_main!(benches);
