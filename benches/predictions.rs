use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use pharmsol::prelude::*;
use pharmsol::{Analytical, Cache, ODE};
use std::hint::black_box;

/// Oral acetaminophen dose with measurements in plasma
fn acetaminophen_oral_subject() -> Subject {
    Subject::builder("bench_acetaminophen_oral")
        .bolus(0.0, 500.0, "acetaminophen_oral")
        .missing_observation(0.25, "plasma")
        .missing_observation(0.5, "plasma")
        .missing_observation(1.0, "plasma")
        .missing_observation(2.0, "plasma")
        .missing_observation(4.0, "plasma")
        .missing_observation(8.0, "plasma")
        .missing_observation(12.0, "plasma")
        .missing_observation(24.0, "plasma")
        .build()
}

/// Intravenous acetaminophen infusion with measurements in plasma
fn acetaminophen_iv_subject() -> Subject {
    Subject::builder("bench_acetaminophen_iv")
        .infusion(0.0, 500.0, "acetaminophen_iv", 0.5)
        .missing_observation(0.5, "plasma")
        .missing_observation(1.0, "plasma")
        .missing_observation(2.0, "plasma")
        .missing_observation(4.0, "plasma")
        .missing_observation(8.0, "plasma")
        .missing_observation(12.0, "plasma")
        .missing_observation(24.0, "plasma")
        .build()
}

fn analytical_one_cpt() -> Analytical {
    analytical! {
        name: "one_cmt_acetaminophen_oral",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [plasma],
        routes: [
            bolus(acetaminophen_oral) -> gut,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _p, _t, _cov, y| {
            y[plasma] = x[central] / v;
        },
    }
}

/// Two-compartment intravenous model
fn analytical_two_cpt() -> Analytical {
    analytical! {
        name: "two_cmt_acetaminophen_iv",
        params: [ke, kcp, kpc, v],
        states: [central, peripheral],
        outputs: [plasma],
        routes: [
            infusion(acetaminophen_iv) -> central,
        ],
        structure: two_compartments,
        out: |x, _p, _t, _cov, y| {
            y[plasma] = x[central] / v;
        },
    }
}

fn ode_one_cpt() -> ODE {
    ode! {
        name: "one_cmt_acetaminophen_oral",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [plasma],
        routes: [
            bolus(acetaminophen_oral) -> gut,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[plasma] = x[central] / v;
        },
    }
}

/// Two-compartment model with infusion
fn ode_two_cpt() -> ODE {
    ode! {
        name: "two_cmt_acetaminophen_iv",
        params: [cl, v, vp, q],
        states: [central, peripheral],
        outputs: [plasma],
        routes: [
            infusion(acetaminophen_iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            let ke = cl / v;
            let kcp = q / v;
            let kpc = q / vp;
            dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        out: |x, _p, _t, _cov, y| {
            y[plasma] = x[central] / v;
        },
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    // One-compartment oral acetaminophen
    let acetaminophen_oral = acetaminophen_oral_subject();
    let acetaminophen_oral_params = [1.0, 0.1, 50.0];
    let analytical_1 = analytical_one_cpt();
    let ode_1_cold = ode_one_cpt().disable_cache();
    let ode_1_hot = ode_one_cpt();
    let _ = ode_1_hot
        .estimate_predictions(&acetaminophen_oral, acetaminophen_oral_params.as_slice())
        .unwrap();

    // Two-compartment intravenous acetaminophen
    let acetaminophen_iv = acetaminophen_iv_subject();
    let analytical_2_params = [0.1, 0.05, 0.04, 50.0];
    let ode_2_params = [5.0, 50.0, 100.0, 10.0];
    let analytical_2 = analytical_two_cpt();
    let ode_2_cold = ode_two_cpt().disable_cache();
    let ode_2_hot = ode_two_cpt();
    let _ = ode_2_hot
        .estimate_predictions(&acetaminophen_iv, ode_2_params.as_slice())
        .unwrap();

    let mut group = c.benchmark_group("core/predictions");

    // Per-iteration cost is in the sub-microsecond to tens-of-microseconds
    // range, so Criterion's defaults (100 samples, 5 s measurement) collect
    // plenty of iterations. The only override is flat sampling, which keeps
    // per-iteration timings comparable across runs instead of using the
    // default linear ramp.
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("analytical/one-compartment-acetaminophen-oral", |b| {
        b.iter(|| {
            black_box(
                analytical_1
                    .estimate_predictions(
                        black_box(&acetaminophen_oral),
                        black_box(acetaminophen_oral_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("ode/one-compartment-acetaminophen-oral-cold", |b| {
        b.iter(|| {
            black_box(
                ode_1_cold
                    .estimate_predictions(
                        black_box(&acetaminophen_oral),
                        black_box(acetaminophen_oral_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("ode/one-compartment-acetaminophen-oral-hot", |b| {
        b.iter(|| {
            black_box(
                ode_1_hot
                    .estimate_predictions(
                        black_box(&acetaminophen_oral),
                        black_box(acetaminophen_oral_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("analytical/two-compartment-acetaminophen-iv", |b| {
        b.iter(|| {
            black_box(
                analytical_2
                    .estimate_predictions(
                        black_box(&acetaminophen_iv),
                        black_box(analytical_2_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("ode/two-compartment-acetaminophen-iv-cold", |b| {
        b.iter(|| {
            black_box(
                ode_2_cold
                    .estimate_predictions(
                        black_box(&acetaminophen_iv),
                        black_box(ode_2_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("ode/two-compartment-acetaminophen-iv-hot", |b| {
        b.iter(|| {
            black_box(
                ode_2_hot
                    .estimate_predictions(
                        black_box(&acetaminophen_iv),
                        black_box(ode_2_params.as_slice()),
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
