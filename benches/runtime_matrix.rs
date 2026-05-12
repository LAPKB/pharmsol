use criterion::{criterion_group, criterion_main, Criterion};

mod self_contained {
    use std::error::Error;
    use std::io;
    use std::path::PathBuf;

    use pharmsol::dsl::{
        self, CompiledRuntimeModel, CompiledWasmModule, RuntimeCompilationTarget,
        RuntimePredictions, WasmCompileCache,
    };
    use pharmsol::{Subject, SubjectBuilderExt};
    use tempfile::{tempdir, TempDir};

    const ODE_SOURCE: &str = r#"
name = one_cmt_acetaminophen_oral_iv
kind = ode

params = ka, cl, v, tlag, f_oral
covariates = wt@linear
states = depot, central
derived = cl_i, ke
outputs = plasma

bolus(acetaminophen_oral) -> depot
infusion(acetaminophen_iv) -> central

lag(acetaminophen_oral) = tlag
fa(acetaminophen_oral) = f_oral

cl_i = cl * pow(wt / 70.0, 0.75)
ke = cl_i / v

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(plasma) = central / v ~ continuous()
"#;

    const ANALYTICAL_SOURCE: &str = r#"
name = one_cmt_acetaminophen_oral
kind = analytical

params = ka, ke, v, tlag, f_oral
states = depot, central
outputs = plasma

bolus(acetaminophen_oral) -> depot

lag(acetaminophen_oral) = tlag
fa(acetaminophen_oral) = f_oral

structure = one_compartment_with_absorption

out(plasma) = central / v ~ continuous()
"#;

    const SDE_SOURCE: &str = r#"
name = acetaminophen_iv_sde
kind = sde

params = ka, ke0, kcp, kpc, vol, ske
covariates = wt@locf
states = depot, central, peripheral, ke_latent
particles = 16
outputs = plasma

bolus(acetaminophen_iv) -> depot

init(ke_latent) = ke0

dx(depot) = -ka * depot
dx(central) = ka * depot - (ke_latent + kcp) * central + kpc * peripheral
dx(peripheral) = kcp * central - kpc * peripheral
dx(ke_latent) = -ke_latent + ke0

noise(ke_latent) = ske

out(plasma) = central / (vol * wt) ~ continuous()
"#;

    const SDE_PARTICLE_COUNT: usize = 16;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CorpusCase {
        Ode,
        Analytical,
        Sde,
    }

    impl CorpusCase {
        pub fn label(self) -> &'static str {
            match self {
                Self::Ode => "dsl-ode-one_cmt_acetaminophen_oral_iv",
                Self::Analytical => "dsl-analytical-one_cmt_acetaminophen_oral",
                Self::Sde => "dsl-sde-acetaminophen_iv_sde",
            }
        }

        fn model_name(self) -> &'static str {
            match self {
                Self::Ode => "one_cmt_acetaminophen_oral_iv",
                Self::Analytical => "one_cmt_acetaminophen_oral",
                Self::Sde => "acetaminophen_iv_sde",
            }
        }

        fn source(self) -> &'static str {
            match self {
                Self::Ode => ODE_SOURCE,
                Self::Analytical => ANALYTICAL_SOURCE,
                Self::Sde => SDE_SOURCE,
            }
        }

        fn support_point(self) -> &'static [f64] {
            match self {
                Self::Ode => &[1.2, 5.0, 40.0, 0.5, 0.8],
                Self::Analytical => &[1.0, 0.15, 25.0, 0.5, 0.8],
                Self::Sde => &[1.1, 0.2, 0.12, 0.08, 15.0, 0.0],
            }
        }

        fn runtime_subject(self, model: &CompiledRuntimeModel) -> Result<Subject, Box<dyn Error>> {
            model
                .info()
                .outputs
                .iter()
                .find(|output| output.name == "plasma")
                .ok_or_else(|| {
                    io::Error::other(format!("{}: missing plasma output", self.label()))
                })?;

            let subject = match self {
                Self::Ode => {
                    model
                        .info()
                        .routes
                        .iter()
                        .find(|route| route.name == "acetaminophen_oral")
                        .ok_or_else(|| {
                            io::Error::other(format!(
                                "{}: missing acetaminophen_oral route",
                                self.label()
                            ))
                        })?;
                    model
                        .info()
                        .routes
                        .iter()
                        .find(|route| route.name == "acetaminophen_iv")
                        .ok_or_else(|| {
                            io::Error::other(format!(
                                "{}: missing acetaminophen_iv route",
                                self.label()
                            ))
                        })?;
                    Subject::builder(self.label())
                        .covariate("wt", 0.0, 70.0)
                        .bolus(0.0, 120.0, "acetaminophen_oral")
                        .infusion(6.0, 60.0, "acetaminophen_iv", 2.0)
                        .missing_observation(0.5, "plasma")
                        .missing_observation(1.0, "plasma")
                        .missing_observation(2.0, "plasma")
                        .missing_observation(6.0, "plasma")
                        .missing_observation(7.0, "plasma")
                        .missing_observation(9.0, "plasma")
                        .build()
                }
                Self::Analytical => {
                    model
                        .info()
                        .routes
                        .iter()
                        .find(|route| route.name == "acetaminophen_oral")
                        .ok_or_else(|| {
                            io::Error::other(format!(
                                "{}: missing acetaminophen_oral route",
                                self.label()
                            ))
                        })?;
                    Subject::builder(self.label())
                        .bolus(0.0, 100.0, "acetaminophen_oral")
                        .missing_observation(0.5, "plasma")
                        .missing_observation(1.0, "plasma")
                        .missing_observation(2.0, "plasma")
                        .missing_observation(4.0, "plasma")
                        .build()
                }
                Self::Sde => {
                    model
                        .info()
                        .routes
                        .iter()
                        .find(|route| route.name == "acetaminophen_iv")
                        .ok_or_else(|| {
                            io::Error::other(format!(
                                "{}: missing acetaminophen_iv route",
                                self.label()
                            ))
                        })?;
                    Subject::builder(self.label())
                        .covariate("wt", 0.0, 70.0)
                        .bolus(0.0, 80.0, "acetaminophen_iv")
                        .missing_observation(0.5, "plasma")
                        .missing_observation(1.0, "plasma")
                        .missing_observation(2.0, "plasma")
                        .missing_observation(4.0, "plasma")
                        .build()
                }
            };

            Ok(subject)
        }
    }

    #[derive(Debug)]
    pub struct ArtifactWorkspace {
        tempdir: TempDir,
    }

    impl ArtifactWorkspace {
        pub fn new() -> Result<Self, Box<dyn Error>> {
            Ok(Self {
                tempdir: tempdir()?,
            })
        }

        fn aot_output(&self, stem: &str) -> PathBuf {
            self.tempdir.path().join(format!("{stem}.pkm"))
        }

        fn build_root(&self, stem: &str) -> PathBuf {
            self.tempdir.path().join(stem)
        }
    }

    fn adjust_runtime_model(case: CorpusCase, model: CompiledRuntimeModel) -> CompiledRuntimeModel {
        match (case, model) {
            (CorpusCase::Sde, CompiledRuntimeModel::Sde(model)) => {
                CompiledRuntimeModel::Sde(model.with_particles(SDE_PARTICLE_COUNT))
            }
            (_, model) => model,
        }
    }

    pub fn compile_runtime_jit_model(
        case: CorpusCase,
    ) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
        Ok(adjust_runtime_model(
            case,
            dsl::compile_module_source_to_runtime(
                case.source(),
                Some(case.model_name()),
                RuntimeCompilationTarget::Jit,
                |_, _| {},
            )?,
        ))
    }

    pub fn compile_runtime_native_aot_model(
        case: CorpusCase,
        workspace: &ArtifactWorkspace,
    ) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
        Ok(adjust_runtime_model(
            case,
            dsl::compile_module_source_to_runtime(
                case.source(),
                Some(case.model_name()),
                RuntimeCompilationTarget::NativeAot(
                    dsl::NativeAotCompileOptions::new(
                        workspace.build_root(&format!("{}-runtime-aot-build", case.label())),
                    )
                    .with_output(workspace.aot_output(&format!("{}-runtime-aot", case.label()))),
                ),
                |_, _| {},
            )?,
        ))
    }

    pub fn compile_runtime_wasm_model(
        case: CorpusCase,
    ) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
        Ok(adjust_runtime_model(
            case,
            dsl::compile_module_source_to_runtime_wasm(case.source(), Some(case.model_name()))?,
        ))
    }

    pub fn compile_wasm_module(case: CorpusCase) -> Result<CompiledWasmModule, Box<dyn Error>> {
        Ok(dsl::compile_module_source_to_wasm_module(
            case.source(),
            Some(case.model_name()),
        )?)
    }

    pub fn compile_wasm_module_with_cache(
        case: CorpusCase,
        cache: &WasmCompileCache,
    ) -> Result<CompiledWasmModule, Box<dyn Error>> {
        Ok(cache.compile_module_source_to_wasm_module(case.source(), Some(case.model_name()))?)
    }

    pub fn estimate_runtime_predictions(
        case: CorpusCase,
        model: &CompiledRuntimeModel,
    ) -> Result<RuntimePredictions, Box<dyn Error>> {
        Ok(model.estimate_predictions(&case.runtime_subject(model)?, case.support_point())?)
    }
}

fn runtime_matrix_benchmark(c: &mut Criterion) {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{BatchSize, BenchmarkId, SamplingMode};
    use pharmsol::dsl::WasmCompileCache;
    use self_contained as corpus;
    use self_contained::{ArtifactWorkspace, CorpusCase};

    let cases = [CorpusCase::Ode, CorpusCase::Analytical, CorpusCase::Sde];

    let mut compile_group = c.benchmark_group("dsl/runtime-compile");
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

    let mut runtime_group = c.benchmark_group("dsl/runtime-predict");
    runtime_group.sampling_mode(SamplingMode::Flat);

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

criterion_group!(benches, runtime_matrix_benchmark);
criterion_main!(benches);
