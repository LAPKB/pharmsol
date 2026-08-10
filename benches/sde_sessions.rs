//! Focused stochastic simulation benchmarks without deterministic cache modes.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use pharmsol::prelude::*;
use pharmsol::Parameters;
use rand::{rngs::StdRng, SeedableRng};

mod common;
use common::{handwritten_sde, named_params, subject_for_predictions, SolverKind, Workload};

fn sde_group(c: &mut Criterion) {
    let workload = Workload::Short;
    let model = handwritten_sde(workload);
    let subject = subject_for_predictions(workload);
    let parameters = Parameters::with_model(&model, named_params(workload, SolverKind::Sde))
        .expect("SDE benchmark parameters should validate");

    let mut group = c.benchmark_group("sde/simulation");
    group.sample_size(10);

    group.bench_function("standard-prediction", |b| {
        b.iter(|| {
            black_box(
                model
                    .estimate_predictions(black_box(&subject), black_box(&parameters))
                    .unwrap(),
            )
        })
    });

    group.bench_function("session-retain", |b| {
        let mut seed = 1_u64;
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(seed);
            seed = seed.wrapping_add(1);
            let mut session = model
                .particle_session(&subject, &parameters, 16, &mut rng)
                .unwrap();
            while session.next_observation().unwrap().is_some() {
                session.retain_particles().unwrap();
            }
            black_box(session.particle_count())
        })
    });

    group.bench_function("session-select-ancestors", |b| {
        let ancestors = (0..16).rev().collect::<Vec<_>>();
        let mut seed = 10_000_u64;
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(seed);
            seed = seed.wrapping_add(1);
            let mut session = model
                .particle_session(&subject, &parameters, 16, &mut rng)
                .unwrap();
            while session.next_observation().unwrap().is_some() {
                session.select_ancestors(&ancestors).unwrap();
            }
            black_box(session.particle_count())
        })
    });

    group.finish();
}

criterion_group!(sde_sessions, sde_group);
criterion_main!(sde_sessions);
