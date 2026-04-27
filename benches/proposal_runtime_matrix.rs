#[path = "../tests/support/proposal_runtime_corpus.rs"]
mod proposal_runtime_corpus;

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
fn proposal_runtime_matrix_benchmark(c: &mut Criterion) {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{BatchSize, BenchmarkId, SamplingMode};
    use pharmsol::dsl::WasmCompileCache;
    use proposal_runtime_corpus as corpus;
    use proposal_runtime_corpus::{ArtifactWorkspace, CorpusCase};

    let cases = [CorpusCase::Ode, CorpusCase::Analytical, CorpusCase::Sde];

    let mut compile_group = c.benchmark_group("proposal2/runtime-compile");
    compile_group.sampling_mode(SamplingMode::Flat);
    compile_group.sample_size(10);
    compile_group.warm_up_time(Duration::from_millis(500));
    compile_group.measurement_time(Duration::from_secs(1));

    for case in cases {
        compile_group.bench_with_input(BenchmarkId::new("jit", case.label()), &case, |b, case| {
            b.iter(|| black_box(corpus::compile_runtime_jit_model(*case).unwrap()))
        });
        compile_group.bench_with_input(
            BenchmarkId::new("native-aot", case.label()),
            &case,
            |b, case| {
                b.iter_batched(
                    || ArtifactWorkspace::new().unwrap(),
                    |workspace| {
                        black_box(
                            corpus::compile_runtime_native_aot_model(*case, &workspace).unwrap(),
                        )
                    },
                    BatchSize::PerIteration,
                )
            },
        );
        compile_group.bench_with_input(BenchmarkId::new("wasm", case.label()), &case, |b, case| {
            b.iter(|| black_box(corpus::compile_runtime_wasm_model(*case).unwrap()))
        });
        compile_group.bench_with_input(
            BenchmarkId::new("wasm-module", case.label()),
            &case,
            |b, case| b.iter(|| black_box(corpus::compile_wasm_module(*case).unwrap())),
        );

        let cache = WasmCompileCache::default();
        corpus::compile_wasm_module_with_cache(case, &cache).unwrap();
        compile_group.bench_function(BenchmarkId::new("wasm-module-cached", case.label()), |b| {
            b.iter(|| black_box(corpus::compile_wasm_module_with_cache(case, &cache).unwrap()))
        });
    }
    compile_group.finish();

    let mut runtime_group = c.benchmark_group("proposal2/runtime-predict");
    runtime_group.sampling_mode(SamplingMode::Flat);
    runtime_group.sample_size(10);
    runtime_group.warm_up_time(Duration::from_millis(500));
    runtime_group.measurement_time(Duration::from_secs(1));

    for case in cases {
        let jit = corpus::compile_runtime_jit_model(case).unwrap();
        runtime_group.bench_function(BenchmarkId::new("jit", case.label()), |b| {
            b.iter(|| black_box(corpus::estimate_runtime_predictions(case, &jit).unwrap()))
        });

        let aot_workspace = ArtifactWorkspace::new().unwrap();
        let aot = corpus::compile_runtime_native_aot_model(case, &aot_workspace).unwrap();
        runtime_group.bench_function(BenchmarkId::new("native-aot", case.label()), |b| {
            b.iter(|| black_box(corpus::estimate_runtime_predictions(case, &aot).unwrap()))
        });

        let wasm = corpus::compile_runtime_wasm_model(case).unwrap();
        runtime_group.bench_function(BenchmarkId::new("wasm", case.label()), |b| {
            b.iter(|| black_box(corpus::estimate_runtime_predictions(case, &wasm).unwrap()))
        });
    }
    runtime_group.finish();
}

#[cfg(not(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
)))]
fn proposal_runtime_matrix_benchmark(_: &mut Criterion) {}

criterion_group!(benches, proposal_runtime_matrix_benchmark);
criterion_main!(benches);
