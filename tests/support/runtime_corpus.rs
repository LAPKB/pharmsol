#![allow(dead_code)]
#![cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    feature = "dsl-wasm"
))]

use std::error::Error;
use std::io;
use std::path::PathBuf;

use diffsol::Vector;
use ndarray::Array2;
use pharmsol::dsl::{self, CompiledRuntimeModel, RuntimeCompilationTarget, RuntimePredictions};
use pharmsol::prelude::{
    one_compartment_with_absorption, Equation, Prediction, SubjectPredictions,
};
use pharmsol::{equation, fa, fetch_cov, fetch_params, lag, Parameters, Subject, SubjectBuilderExt, SDE};
use tempfile::{tempdir, TempDir};

const ODE_SOURCE: &str = r#"
name = one_cmt_oral_iv
kind = ode

params = ka, cl, v, tlag, f_oral
covariates = wt@linear
states = depot, central
derived = cl_i, ke
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central

lag(oral) = tlag
fa(oral) = f_oral

cl_i = cl * pow(wt / 70.0, 0.75)
ke = cl_i / v

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(cp) = central / v ~ continuous()
"#;

const ODE_FULL_SOURCE: &str = r#"
name = ode_full_feature_parity
kind = ode

params = ka, ke, kcp, kpc, v, tlag, f_oral, base_depot, base_central, base_peripheral
covariates = wt@linear, renal@linear
derived = adjusted_ke, adjusted_kcp, adjusted_v
states = depot, central, peripheral
outputs = cp

bolus(oral) -> depot
bolus(load) -> central
infusion(iv) -> central

lag(oral) = tlag * sqrt(wt / 70.0) * pow(90.0 / renal, 0.1)
fa(oral) = min(max(f_oral * pow(renal / 90.0, 0.1), 0.0), 1.0)

adjusted_ke = ke * pow(wt / 70.0, 0.75) * pow(renal / 90.0, 0.25)
adjusted_kcp = kcp * pow(wt / 70.0, 0.25)
adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0))

dx(depot) = -ka * depot
dx(central) = ka * depot - (adjusted_ke + adjusted_kcp) * central + kpc * peripheral
dx(peripheral) = adjusted_kcp * central - kpc * peripheral

init(depot) = base_depot + 0.05 * wt
init(central) = base_central + 0.1 * renal
init(peripheral) = base_peripheral + 0.02 * wt

out(cp) = central / adjusted_v ~ continuous()
"#;

const ANALYTICAL_SOURCE: &str = r#"
name = one_cmt_abs
kind = analytical

params = ka, ke, v, tlag, f_oral
states = depot, central
outputs = cp

bolus(oral) -> depot

lag(oral) = tlag
fa(oral) = f_oral

structure = one_compartment_with_absorption

out(cp) = central / v ~ continuous()
"#;

const ANALYTICAL_FULL_SOURCE: &str = r#"
name = analytical_full_feature_parity
kind = analytical

params = ka, ke, v, tlag, f_oral, base_gut, base_central
covariates = wt@linear, renal@linear
derived = adjusted_v
states = gut, central
outputs = cp

bolus(oral) -> gut
bolus(load) -> central
infusion(iv) -> central

lag(oral) = tlag * sqrt(wt / 70.0) * pow(90.0 / renal, 0.1)
fa(oral) = min(max(f_oral * pow(renal / 90.0, 0.1), 0.0), 1.0)

adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0))

structure = one_compartment_with_absorption

init(gut) = base_gut + 0.03 * wt
init(central) = base_central + 0.08 * renal

out(cp) = central / adjusted_v ~ continuous()
"#;

const SDE_SOURCE: &str = r#"
name = vanco_sde
kind = sde

params = ka, ke0, kcp, kpc, vol, ske
covariates = wt@locf
states = depot, central, peripheral, ke_latent
particles = 16
outputs = cp

bolus(oral) -> depot

init(ke_latent) = ke0

dx(depot) = -ka * depot
dx(central) = ka * depot - (ke_latent + kcp) * central + kpc * peripheral
dx(peripheral) = kcp * central - kpc * peripheral
dx(ke_latent) = -ke_latent + ke0

noise(ke_latent) = ske

out(cp) = central / (vol * wt) ~ continuous()
"#;

pub const SDE_PARTICLE_COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusCase {
    Ode,
    OdeFull,
    Analytical,
    AnalyticalFull,
    Sde,
}

impl CorpusCase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ode => "dsl-ode-one_cmt_oral_iv",
            Self::OdeFull => "dsl-ode-full-feature-parity",
            Self::Analytical => "dsl-analytical-one_cmt_abs",
            Self::AnalyticalFull => "dsl-analytical-full-feature-parity",
            Self::Sde => "dsl-sde-vanco_sde",
        }
    }

    pub fn model_name(self) -> &'static str {
        match self {
            Self::Ode => "one_cmt_oral_iv",
            Self::OdeFull => "ode_full_feature_parity",
            Self::Analytical => "one_cmt_abs",
            Self::AnalyticalFull => "analytical_full_feature_parity",
            Self::Sde => "vanco_sde",
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::Ode => ODE_SOURCE,
            Self::OdeFull => ODE_FULL_SOURCE,
            Self::Analytical => ANALYTICAL_SOURCE,
            Self::AnalyticalFull => ANALYTICAL_FULL_SOURCE,
            Self::Sde => SDE_SOURCE,
        }
    }

    pub fn tolerance(self) -> f64 {
        match self {
            Self::Ode => 1e-4,
            Self::OdeFull => 1e-4,
            Self::Analytical => 1e-8,
            Self::AnalyticalFull => 1e-8,
            Self::Sde => 1e-4,
        }
    }

    pub fn support_point(self) -> &'static [f64] {
        match self {
            Self::Ode => &[1.2, 5.0, 40.0, 0.5, 0.8],
            Self::OdeFull => &[1.1, 0.18, 0.07, 0.04, 35.0, 0.6, 0.85, 4.0, 18.0, 9.0],
            Self::Analytical => &[1.0, 0.15, 25.0, 0.5, 0.8],
            Self::AnalyticalFull => &[1.0, 0.16, 32.0, 0.5, 0.8, 3.0, 14.0],
            Self::Sde => &[1.1, 0.2, 0.12, 0.08, 15.0, 0.0],
        }
    }

    fn runtime_subject(self, model: &CompiledRuntimeModel) -> Result<Subject, Box<dyn Error>> {
        model
            .info()
            .outputs
            .iter()
            .find(|output| output.name == "cp")
            .ok_or_else(|| io::Error::other(format!("{}: missing cp output", self.label())))?;

        let subject = match self {
            Self::Ode => {
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "oral")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing oral route", self.label()))
                    })?;
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "iv")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing iv route", self.label()))
                    })?;
                Subject::builder(self.label())
                    .covariate("wt", 0.0, 70.0)
                    .bolus(0.0, 120.0, "oral")
                    .infusion(6.0, 60.0, "iv", 2.0)
                    .missing_observation(0.5, "cp")
                    .missing_observation(1.0, "cp")
                    .missing_observation(2.0, "cp")
                    .missing_observation(6.0, "cp")
                    .missing_observation(7.0, "cp")
                    .missing_observation(9.0, "cp")
                    .build()
            }
            Self::OdeFull => {
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "oral")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing oral route", self.label()))
                    })?;
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "load")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing load route", self.label()))
                    })?;
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "iv")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing iv route", self.label()))
                    })?;
                Subject::builder(self.label())
                    .bolus(0.0, 80.0, "load")
                    .bolus(1.0, 120.0, "oral")
                    .infusion(6.0, 150.0, "iv", 2.5)
                    .missing_observation(0.25, "cp")
                    .missing_observation(0.75, "cp")
                    .missing_observation(1.5, "cp")
                    .missing_observation(3.0, "cp")
                    .missing_observation(6.5, "cp")
                    .missing_observation(7.0, "cp")
                    .missing_observation(8.0, "cp")
                    .missing_observation(12.0, "cp")
                    .covariate("wt", 0.0, 68.0)
                    .covariate("wt", 8.0, 74.0)
                    .covariate("renal", 0.0, 95.0)
                    .covariate("renal", 8.0, 72.0)
                    .build()
            }
            Self::Analytical => {
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "oral")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing oral route", self.label()))
                    })?;
                Subject::builder(self.label())
                    .bolus(0.0, 100.0, "oral")
                    .missing_observation(0.5, "cp")
                    .missing_observation(1.0, "cp")
                    .missing_observation(2.0, "cp")
                    .missing_observation(4.0, "cp")
                    .build()
            }
            Self::AnalyticalFull => {
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "oral")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing oral route", self.label()))
                    })?;
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "load")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing load route", self.label()))
                    })?;
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "iv")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing iv route", self.label()))
                    })?;
                Subject::builder(self.label())
                    .bolus(0.0, 60.0, "load")
                    .bolus(1.0, 100.0, "oral")
                    .infusion(6.0, 140.0, "iv", 2.0)
                    .missing_observation(0.25, "cp")
                    .missing_observation(0.75, "cp")
                    .missing_observation(1.5, "cp")
                    .missing_observation(3.0, "cp")
                    .missing_observation(6.5, "cp")
                    .missing_observation(7.0, "cp")
                    .missing_observation(8.0, "cp")
                    .missing_observation(12.0, "cp")
                    .covariate("wt", 0.0, 68.0)
                    .covariate("wt", 8.0, 74.0)
                    .covariate("renal", 0.0, 95.0)
                    .covariate("renal", 8.0, 72.0)
                    .build()
            }
            Self::Sde => {
                model
                    .info()
                    .routes
                    .iter()
                    .find(|route| route.name == "oral")
                    .ok_or_else(|| {
                        io::Error::other(format!("{}: missing oral route", self.label()))
                    })?;
                Subject::builder(self.label())
                    .covariate("wt", 0.0, 70.0)
                    .bolus(0.0, 80.0, "oral")
                    .missing_observation(0.5, "cp")
                    .missing_observation(1.0, "cp")
                    .missing_observation(2.0, "cp")
                    .missing_observation(4.0, "cp")
                    .build()
            }
        };

        Ok(subject)
    }

    fn reference_predictions(self) -> Result<ExpectedPredictions, Box<dyn Error>> {
        match self {
            Self::Ode => Ok(ExpectedPredictions::Subject(reference_ode_predictions()?)),
            Self::OdeFull => Ok(ExpectedPredictions::Subject(
                reference_ode_full_predictions()?,
            )),
            Self::Analytical => Ok(ExpectedPredictions::Subject(
                reference_analytical_predictions()?,
            )),
            Self::AnalyticalFull => Ok(ExpectedPredictions::Subject(
                reference_analytical_full_predictions()?,
            )),
            Self::Sde => Ok(ExpectedPredictions::Particles(reference_sde_predictions()?)),
        }
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

    fn wasm_output(&self, stem: &str) -> PathBuf {
        self.tempdir.path().join(format!("{stem}.wasm"))
    }

    fn build_root(&self, stem: &str) -> PathBuf {
        self.tempdir.path().join(stem)
    }
}

enum ExpectedPredictions {
    Subject(SubjectPredictions),
    Particles(Array2<Prediction>),
}

fn adjust_runtime_model(case: CorpusCase, model: CompiledRuntimeModel) -> CompiledRuntimeModel {
    match (case, model) {
        (CorpusCase::Sde, CompiledRuntimeModel::Sde(model)) => {
            CompiledRuntimeModel::Sde(model.with_particles(SDE_PARTICLE_COUNT))
        }
        (_, model) => model,
    }
}

#[cfg(feature = "dsl-jit")]
pub fn compile_runtime_jit_model(case: CorpusCase) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
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

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
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

#[cfg(feature = "dsl-wasm")]
pub fn compile_runtime_wasm_model(
    case: CorpusCase,
) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
    Ok(adjust_runtime_model(
        case,
        dsl::compile_module_source_to_runtime_wasm(case.source(), Some(case.model_name()))?,
    ))
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_wasm_module(case: CorpusCase) -> Result<dsl::CompiledWasmModule, Box<dyn Error>> {
    Ok(dsl::compile_module_source_to_wasm_module(
        case.source(),
        Some(case.model_name()),
    )?)
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_wasm_runtime_from_bytes(
    case: CorpusCase,
) -> Result<CompiledRuntimeModel, Box<dyn Error>> {
    let parsed = dsl::parse_module(case.source())?;
    let typed = dsl::analyze_module(&parsed)?;
    let model = typed
        .models
        .iter()
        .find(|model| model.name == case.model_name())
        .ok_or_else(|| io::Error::other(format!("{}: missing model in source", case.label())))?;
    let execution = dsl::lower_typed_model(model)?;
    let bytes = dsl::compile_execution_model_to_wasm_bytes(&execution)?;
    Ok(adjust_runtime_model(
        case,
        dsl::load_runtime_wasm_bytes(&bytes)?,
    ))
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_wasm_module_with_cache(
    case: CorpusCase,
    cache: &dsl::WasmCompileCache,
) -> Result<dsl::CompiledWasmModule, Box<dyn Error>> {
    Ok(cache.compile_module_source_to_wasm_module(case.source(), Some(case.model_name()))?)
}

pub fn assert_runtime_model_matches_reference(
    case: CorpusCase,
    backend_label: &str,
    model: &CompiledRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let actual = estimate_runtime_predictions(case, model)?;
    let expected = case.reference_predictions()?;

    match (&actual, &expected) {
        (RuntimePredictions::Subject(actual), ExpectedPredictions::Subject(expected)) => {
            compare_subject_predictions(case, backend_label, actual, expected)
        }
        (RuntimePredictions::Particles(actual), ExpectedPredictions::Particles(expected)) => {
            compare_particle_predictions(case, backend_label, actual, expected)
        }
        (RuntimePredictions::Subject(_), ExpectedPredictions::Particles(_))
        | (RuntimePredictions::Particles(_), ExpectedPredictions::Subject(_)) => {
            Err(io::Error::other(format!(
                "{} [{}]: runtime prediction kind did not match reference kind",
                case.label(),
                backend_label
            ))
            .into())
        }
    }
}

pub fn assert_runtime_models_match_each_other(
    case: CorpusCase,
    left_label: &str,
    left: &CompiledRuntimeModel,
    right_label: &str,
    right: &CompiledRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let left_predictions = estimate_runtime_predictions(case, left)?;
    let right_predictions = estimate_runtime_predictions(case, right)?;

    match (&left_predictions, &right_predictions) {
        (RuntimePredictions::Subject(left), RuntimePredictions::Subject(right)) => {
            compare_subject_predictions_pairwise(case, left_label, left, right_label, right)
        }
        (RuntimePredictions::Particles(left), RuntimePredictions::Particles(right)) => {
            compare_particle_predictions_pairwise(case, left_label, left, right_label, right)
        }
        (RuntimePredictions::Subject(_), RuntimePredictions::Particles(_))
        | (RuntimePredictions::Particles(_), RuntimePredictions::Subject(_)) => {
            Err(io::Error::other(format!(
                "{} [{} vs {}]: runtime prediction kind mismatch",
                case.label(),
                left_label,
                right_label
            ))
            .into())
        }
    }
}

pub fn estimate_runtime_predictions(
    case: CorpusCase,
    model: &CompiledRuntimeModel,
) -> Result<RuntimePredictions, Box<dyn Error>> {
    let parameters = Parameters::dense(case.support_point().to_vec());
    Ok(model.estimate_predictions(&case.runtime_subject(model)?, &parameters)?)
}

fn compare_subject_predictions(
    case: CorpusCase,
    backend_label: &str,
    actual: &SubjectPredictions,
    expected: &SubjectPredictions,
) -> Result<(), Box<dyn Error>> {
    let actual_values = actual.flat_predictions();
    let expected_values = expected.flat_predictions();

    if actual_values.len() != expected_values.len() {
        return Err(io::Error::other(format!(
            "{} [{}]: expected {} subject predictions, got {}",
            case.label(),
            backend_label,
            expected_values.len(),
            actual_values.len()
        ))
        .into());
    }

    for (index, (actual_value, expected_value)) in
        actual_values.iter().zip(expected_values.iter()).enumerate()
    {
        let abs_diff = (actual_value - expected_value).abs();
        if abs_diff > case.tolerance() {
            return Err(io::Error::other(format!(
                "{} [{}]: prediction {} differed by {:.6} (expected {:.6}, actual {:.6}, tolerance {:.6})",
                case.label(),
                backend_label,
                index,
                abs_diff,
                expected_value,
                actual_value,
                case.tolerance()
            ))
            .into());
        }
    }

    Ok(())
}

fn compare_subject_predictions_pairwise(
    case: CorpusCase,
    left_label: &str,
    left: &SubjectPredictions,
    right_label: &str,
    right: &SubjectPredictions,
) -> Result<(), Box<dyn Error>> {
    let left_values = left.flat_predictions();
    let right_values = right.flat_predictions();

    if left_values.len() != right_values.len() {
        return Err(io::Error::other(format!(
            "{} [{} vs {}]: prediction length mismatch ({} vs {})",
            case.label(),
            left_label,
            right_label,
            left_values.len(),
            right_values.len()
        ))
        .into());
    }

    for (index, (left_value, right_value)) in
        left_values.iter().zip(right_values.iter()).enumerate()
    {
        let abs_diff = (left_value - right_value).abs();
        if abs_diff > case.tolerance() {
            return Err(io::Error::other(format!(
                "{} [{} vs {}]: prediction {} differed by {:.6} (left {:.6}, right {:.6}, tolerance {:.6})",
                case.label(),
                left_label,
                right_label,
                index,
                abs_diff,
                left_value,
                right_value,
                case.tolerance()
            ))
            .into());
        }
    }

    Ok(())
}

fn compare_particle_predictions(
    case: CorpusCase,
    backend_label: &str,
    actual: &Array2<Prediction>,
    expected: &Array2<Prediction>,
) -> Result<(), Box<dyn Error>> {
    if actual.dim() != expected.dim() {
        return Err(io::Error::other(format!(
            "{} [{}]: expected particle matrix {:?}, got {:?}",
            case.label(),
            backend_label,
            expected.dim(),
            actual.dim()
        ))
        .into());
    }

    for row in 0..actual.nrows() {
        for col in 0..actual.ncols() {
            let actual_prediction = &actual[(row, col)];
            let expected_prediction = &expected[(row, col)];
            let abs_diff =
                (actual_prediction.prediction() - expected_prediction.prediction()).abs();
            if abs_diff > case.tolerance() {
                return Err(io::Error::other(format!(
                    "{} [{}]: particle ({row}, {col}) differed by {:.6} (expected {:.6}, actual {:.6}, tolerance {:.6})",
                    case.label(),
                    backend_label,
                    abs_diff,
                    expected_prediction.prediction(),
                    actual_prediction.prediction(),
                    case.tolerance()
                ))
                .into());
            }
        }
    }

    Ok(())
}

fn compare_particle_predictions_pairwise(
    case: CorpusCase,
    left_label: &str,
    left: &Array2<Prediction>,
    right_label: &str,
    right: &Array2<Prediction>,
) -> Result<(), Box<dyn Error>> {
    if left.dim() != right.dim() {
        return Err(io::Error::other(format!(
            "{} [{} vs {}]: particle matrix mismatch {:?} vs {:?}",
            case.label(),
            left_label,
            right_label,
            left.dim(),
            right.dim()
        ))
        .into());
    }

    for row in 0..left.nrows() {
        for col in 0..left.ncols() {
            let left_prediction = &left[(row, col)];
            let right_prediction = &right[(row, col)];
            let abs_diff = (left_prediction.prediction() - right_prediction.prediction()).abs();
            if abs_diff > case.tolerance() {
                return Err(io::Error::other(format!(
                    "{} [{} vs {}]: particle ({row}, {col}) differed by {:.6} (left {:.6}, right {:.6}, tolerance {:.6})",
                    case.label(),
                    left_label,
                    right_label,
                    abs_diff,
                    left_prediction.prediction(),
                    right_prediction.prediction(),
                    case.tolerance()
                ))
                .into());
            }
        }
    }

    Ok(())
}

fn reference_ode_predictions() -> Result<SubjectPredictions, Box<dyn Error>> {
    Ok(equation::ODE::new(
        |x, p, t, dx, bolus, rateiv, cov| {
            fetch_cov!(cov, t, wt);
            fetch_params!(p, ka, cl, v, _tlag, _f_oral);

            let cl_i = cl * (wt / 70.0).powf(0.75);
            let v_i = if wt > 120.0 { v * 1.15 } else { v };
            let ke = cl_i / v_i;

            dx[0] = -ka * x[0] + bolus[0];
            dx[1] = ka * x[0] - ke * x[1] + bolus[1] + rateiv[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _cl, _v, tlag, f_oral);
            lag! {0 => tlag}
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _cl, _v, _tlag, f_oral);
            fa! {0 => f_oral}
        },
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            let wt = cov
                .get_covariate("wt")
                .map(|values| values.interpolate(t).unwrap())
                .unwrap();
            fetch_params!(p, _ka, _cl, v, _tlag, _f_oral);
            let v_i = if wt > 120.0 { v * 1.15 } else { v };
            y[0] = x[1] / v_i;
        },
    )
    .with_nstates(2)
    .with_ndrugs(2)
    .with_nout(1)
    .estimate_predictions(
        &Subject::builder(CorpusCase::Ode.label())
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, 0)
            .infusion(6.0, 60.0, 1, 2.0)
            .missing_observation(0.5, 0)
            .missing_observation(1.0, 0)
            .missing_observation(2.0, 0)
            .missing_observation(6.0, 0)
            .missing_observation(7.0, 0)
            .missing_observation(9.0, 0)
            .build(),
        &Parameters::dense(CorpusCase::Ode.support_point().to_vec()),
    )?)
}

fn reference_ode_full_predictions() -> Result<SubjectPredictions, Box<dyn Error>> {
    Ok(equation::ODE::new(
        |x, p, t, dx, bolus, rateiv, cov| {
            fetch_params!(
                p,
                ka,
                ke,
                kcp,
                kpc,
                _v,
                _tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let wt_scale = (wt / 70.0).powf(0.75);
            let renal_scale = (renal / 90.0).powf(0.25);
            let adjusted_ke = ke * wt_scale * renal_scale;
            let adjusted_kcp = kcp * (wt / 70.0).powf(0.25);

            dx[0] = bolus[0] - ka * x[0];
            dx[1] =
                bolus[1] + ka * x[0] + rateiv[0] - (adjusted_ke + adjusted_kcp) * x[1] + kpc * x[2];
            dx[2] = adjusted_kcp * x[1] - kpc * x[2];
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                _v,
                tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let lag_scale = (wt / 70.0).sqrt() * (90.0 / renal).powf(0.1);
            lag! { 0 => tlag * lag_scale }
        },
        |p, t, cov| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                _v,
                _tlag,
                f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let fa_scale = (renal / 90.0).powf(0.1);
            fa! { 0 => (f_oral * fa_scale).clamp(0.0, 1.0) }
        },
        |p, t, cov, x| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                _v,
                _tlag,
                _f_oral,
                base_depot,
                base_central,
                base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            x[0] = base_depot + 0.05 * wt;
            x[1] = base_central + 0.1 * renal;
            x[2] = base_peripheral + 0.02 * wt;
        },
        |x, p, t, cov, y| {
            fetch_params!(
                p,
                _ka,
                _ke,
                _kcp,
                _kpc,
                v,
                _tlag,
                _f_oral,
                _base_depot,
                _base_central,
                _base_peripheral
            );
            fetch_cov!(cov, t, wt, renal);

            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[0] = x[1] / adjusted_v;
        },
    )
    .with_nstates(3)
    .with_ndrugs(2)
    .with_nout(1)
    .estimate_predictions(
        &Subject::builder(CorpusCase::OdeFull.label())
            .bolus(0.0, 80.0, 1)
            .bolus(1.0, 120.0, 0)
            .infusion(6.0, 150.0, 0, 2.5)
            .missing_observation(0.25, 0)
            .missing_observation(0.75, 0)
            .missing_observation(1.5, 0)
            .missing_observation(3.0, 0)
            .missing_observation(6.5, 0)
            .missing_observation(7.0, 0)
            .missing_observation(8.0, 0)
            .missing_observation(12.0, 0)
            .covariate("wt", 0.0, 68.0)
            .covariate("wt", 8.0, 74.0)
            .covariate("renal", 0.0, 95.0)
            .covariate("renal", 8.0, 72.0)
            .build(),
        &Parameters::dense(CorpusCase::OdeFull.support_point().to_vec()),
    )?)
}

fn reference_analytical_predictions() -> Result<SubjectPredictions, Box<dyn Error>> {
    Ok(equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! {0 => tlag}
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! {0 => f_oral}
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .estimate_predictions(
        &Subject::builder(CorpusCase::Analytical.label())
            .bolus(0.0, 100.0, 0)
            .missing_observation(0.5, 0)
            .missing_observation(1.0, 0)
            .missing_observation(2.0, 0)
            .missing_observation(4.0, 0)
            .build(),
        &Parameters::dense(CorpusCase::Analytical.support_point().to_vec()),
    )?)
}

fn reference_analytical_full_predictions() -> Result<SubjectPredictions, Box<dyn Error>> {
    Ok(equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p, t, cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral, _base_gut, _base_central);
            fetch_cov!(cov, t, wt, renal);

            let lag_scale = (wt / 70.0).sqrt() * (90.0 / renal).powf(0.1);
            lag! { 0 => tlag * lag_scale }
        },
        |p, t, cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral, _base_gut, _base_central);
            fetch_cov!(cov, t, wt, renal);

            let fa_scale = (renal / 90.0).powf(0.1);
            fa! { 0 => (f_oral * fa_scale).clamp(0.0, 1.0) }
        },
        |p, t, cov, x| {
            fetch_params!(p, _ka, _ke, _v, _tlag, _f_oral, base_gut, base_central);
            fetch_cov!(cov, t, wt, renal);

            x[0] = base_gut + 0.03 * wt;
            x[1] = base_central + 0.08 * renal;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral, _base_gut, _base_central);
            fetch_cov!(cov, t, wt, renal);

            let adjusted_v = v * (wt / 70.0) * (1.0 + 0.001 * (renal - 90.0));
            y[0] = x[1] / adjusted_v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(2)
    .with_nout(1)
    .estimate_predictions(
        &Subject::builder(CorpusCase::AnalyticalFull.label())
            .bolus(0.0, 60.0, 1)
            .bolus(1.0, 100.0, 0)
            .infusion(6.0, 140.0, 0, 2.0)
            .missing_observation(0.25, 0)
            .missing_observation(0.75, 0)
            .missing_observation(1.5, 0)
            .missing_observation(3.0, 0)
            .missing_observation(6.5, 0)
            .missing_observation(7.0, 0)
            .missing_observation(8.0, 0)
            .missing_observation(12.0, 0)
            .covariate("wt", 0.0, 68.0)
            .covariate("wt", 8.0, 74.0)
            .covariate("renal", 0.0, 95.0)
            .covariate("renal", 8.0, 72.0)
            .build(),
        &Parameters::dense(CorpusCase::AnalyticalFull.support_point().to_vec()),
    )?)
}

fn reference_sde_predictions() -> Result<Array2<Prediction>, Box<dyn Error>> {
    Ok(SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke0, kcp, kpc, _vol, _ske);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (x[3] + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
            dx[3] = -x[3] + ke0;
        },
        |p, sigma| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, _vol, ske);
            sigma.fill(0.0);
            sigma[3] = ske;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _kcp, _kpc, _vol, _ske);
            x[3] = ke0;
        },
        |x, p, t, cov, y| {
            let wt = cov
                .get_covariate("wt")
                .map(|values| values.interpolate(t).unwrap())
                .unwrap();
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol, _ske);
            y[0] = x[1] / (vol * wt);
        },
        SDE_PARTICLE_COUNT,
    )
    .with_nstates(4)
    .with_ndrugs(1)
    .with_nout(1)
    .estimate_predictions(
        &Subject::builder(CorpusCase::Sde.label())
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 80.0, 0)
            .missing_observation(0.5, 0)
            .missing_observation(1.0, 0)
            .missing_observation(2.0, 0)
            .missing_observation(4.0, 0)
            .build(),
        &Parameters::dense(CorpusCase::Sde.support_point().to_vec()),
    )?)
}
