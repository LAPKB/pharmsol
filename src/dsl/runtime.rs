//! Unified runtime entrypoints for DSL-backed models.
//!
//! Use this module when you already know you want an executable model and need
//! one surface for compile and prediction workflows.
//!
//! Use [`super::jit`] directly only when you need the lower-level compile
//! entrypoints or the raw execution artifact.
//!
//! Main entrypoints:
//!
//! - [`compile_module_source_to_runtime`] for the one-stop source-to-runtime
//!   path.
//! - [`compile_execution_model_to_runtime`] when you already have an
//!   [`ExecutionModel`](pharmsol_dsl::ExecutionModel).
//! - [`CompiledRuntimeModel::estimate_predictions`] for kind-agnostic
//!   execution against a [`Subject`](crate::Subject).
//!
//! Smallest compile-and-run example:
//!
//! ```rust,no_run
//! use pharmsol::dsl::compile_module_source_to_runtime;
//! use pharmsol::{Parameters, prelude::*};
//!
//! let source = r#"
//! name = bimodal_ke
//! kind = ode
//!
//! params = ke, v
//! states = central
//! outputs = cp
//!
//! infusion(iv) -> central
//!
//! dx(central) = -ke * central
//! out(cp) = central / v
//! "#;
//!
//! let model = compile_module_source_to_runtime(
//!     source,
//!     Some("bimodal_ke"),
//!     |_, _| {},
//! )?;
//!
//! let subject = Subject::builder("patient_001")
//!     .infusion(0.0, 500.0, "iv", 0.5)
//!     .missing_observation(0.5, "cp")
//!     .missing_observation(1.0, "cp")
//!     .missing_observation(2.0, "cp")
//!     .build();
//!
//! let parameters = Parameters::with_model(&model, [("ke", 1.2), ("v", 50.0)])
//!     .expect("valid named parameters");
//! let predictions = model.estimate_predictions(&subject, &parameters)?;
//! assert!(predictions.as_subject().is_some());
//! # Ok::<(), pharmsol::dsl::RuntimeError>(())
//! ```

use std::fmt;

use ndarray::Array2;
use thiserror::Error;

use super::backend::{RuntimeAnalyticalModel, RuntimeOdeModel, RuntimeSdeModel};
use super::jit::{compile_execution_model_to_jit, JitCompileError};
use super::model_info::RuntimeModelInfo;
use crate::{
    simulator::likelihood::{Prediction, SubjectPredictions},
    Parameters, PharmsolError, Subject, ValidatedModelMetadata,
};
use pharmsol_dsl::{
    analyze_module, compile_analyzed_model, parse_module, AnalysisError, CompileError, Diagnostic,
    DiagnosticReport, ExecutionModel, ModelKind, ParseError,
};

/// Backend-neutral prediction output from a compiled runtime model.
///
/// ODE and analytical models return subject predictions. SDE models return the
/// particle matrix used by the stochastic workflow.
#[derive(Clone, Debug)]
pub enum RuntimePredictions {
    Subject(SubjectPredictions),
    Particles(Array2<Prediction>),
}

impl RuntimePredictions {
    pub fn as_subject(&self) -> Option<&SubjectPredictions> {
        match self {
            Self::Subject(predictions) => Some(predictions),
            Self::Particles(_) => None,
        }
    }

    pub fn as_particles(&self) -> Option<&Array2<Prediction>> {
        match self {
            Self::Particles(predictions) => Some(predictions),
            Self::Subject(_) => None,
        }
    }

    pub fn into_subject(self) -> Option<SubjectPredictions> {
        match self {
            Self::Subject(predictions) => Some(predictions),
            Self::Particles(_) => None,
        }
    }

    pub fn into_particles(self) -> Option<Array2<Prediction>> {
        match self {
            Self::Particles(predictions) => Some(predictions),
            Self::Subject(_) => None,
        }
    }
}

/// Executable runtime model returned by the compile entrypoints.
///
/// This type hides the model kind and keeps the prediction entrypoint the
/// same across ODE, analytical, and SDE models.
#[derive(Clone, Debug)]
pub enum CompiledRuntimeModel {
    Ode(RuntimeOdeModel),
    Analytical(RuntimeAnalyticalModel),
    Sde(RuntimeSdeModel),
}

impl CompiledRuntimeModel {
    pub fn info(&self) -> &RuntimeModelInfo {
        match self {
            Self::Ode(model) => model.info(),
            Self::Analytical(model) => model.info(),
            Self::Sde(model) => model.info(),
        }
    }

    pub fn kind(&self) -> ModelKind {
        self.info().kind
    }

    pub fn metadata(&self) -> &ValidatedModelMetadata {
        match self {
            Self::Ode(model) => model.metadata(),
            Self::Analytical(model) => model.metadata(),
            Self::Sde(model) => model.metadata(),
        }
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &Parameters,
    ) -> Result<RuntimePredictions, RuntimeError> {
        Ok(match self {
            Self::Ode(model) => {
                RuntimePredictions::Subject(model.estimate_predictions(subject, parameters)?)
            }
            Self::Analytical(model) => {
                RuntimePredictions::Subject(model.estimate_predictions(subject, parameters)?)
            }
            Self::Sde(model) => {
                RuntimePredictions::Particles(model.estimate_predictions(subject, parameters)?)
            }
        })
    }
}

/// Errors produced while parsing, lowering, compiling, or executing a runtime
/// DSL model.
#[derive(Error)]
pub enum RuntimeError {
    #[error("failed to parse DSL source: {0}")]
    Parse(#[source] ParseError),
    #[error("failed to analyze DSL source: {0}")]
    Semantic(#[source] AnalysisError),
    #[error("failed to lower DSL model: {0}")]
    Lowering(#[source] CompileError),
    #[error("{0}")]
    ModelSelection(String),
    #[error(transparent)]
    Jit(#[from] JitCompileError),
    #[error(transparent)]
    Runtime(#[from] PharmsolError),
}

impl RuntimeError {
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        match self {
            Self::Parse(error) => Some(error.diagnostic()),
            Self::Semantic(error) => Some(error.diagnostic()),
            Self::Lowering(error) => Some(error.diagnostic()),
            Self::Jit(error) => Some(error.diagnostic()),
            _ => None,
        }
    }

    pub fn render_diagnostic(&self, src: &str) -> Option<String> {
        self.diagnostic().map(|diagnostic| diagnostic.render(src))
    }

    pub fn diagnostic_report(&self, source_name: impl Into<String>) -> Option<DiagnosticReport> {
        let source_name = source_name.into();
        match self {
            Self::Parse(error) => Some(error.diagnostic_report(source_name)),
            Self::Semantic(error) => Some(error.diagnostic_report(source_name)),
            Self::Lowering(error) => Some(error.diagnostic_report(source_name)),
            Self::Jit(error) => Some(error.diagnostic_report(source_name)),
            _ => None,
        }
    }
}

impl fmt::Debug for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => fmt::Display::fmt(error, f),
            Self::Semantic(error) => fmt::Display::fmt(error, f),
            Self::Lowering(error) => fmt::Display::fmt(error, f),
            Self::Jit(error) => fmt::Display::fmt(error, f),
            _ => fmt::Display::fmt(self, f),
        }
    }
}

/// Parse, analyze, lower, compile, and return a runtime model in one step.
///
/// Use this when your input is DSL source text and you want the shortest path
/// from source to predictions.
pub fn compile_module_source_to_runtime(
    source: &str,
    model_name: Option<&str>,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    let parsed =
        parse_module(source).map_err(|error| RuntimeError::Parse(error.with_source(source)))?;
    let analyzed = analyze_module(&parsed)
        .map_err(|error| RuntimeError::Semantic(error.with_source(source)))?;

    let model = match model_name {
        Some(name) => analyzed
            .models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| {
                RuntimeError::ModelSelection(format!("model `{name}` not found in module"))
            })?,
        None if analyzed.models.len() == 1 => &analyzed.models[0],
        None => {
            return Err(RuntimeError::ModelSelection(
                "module contains multiple models; pass an explicit model name".to_string(),
            ))
        }
    };

    let execution = compile_analyzed_model(model)
        .map_err(|error| RuntimeError::Lowering(error.with_source(source)))?;
    compile_with_solver_notice(&execution, Some(source), event_callback).map_err(|error| {
        if let RuntimeError::Jit(error) = error {
            return RuntimeError::Jit(error.with_source(source));
        }
        error
    })
}

/// Compile a compiled execution model to an executable runtime model.
///
/// Use this when you already own the frontend pipeline and only need the final
/// backend step.
pub fn compile_execution_model_to_runtime(
    model: &ExecutionModel,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    compile_with_solver_notice(model, None, event_callback)
}

/// Compile and report the chosen solver through `event_callback` under the
/// `"solver"` kind.
///
/// With `source` available the notice is rendered against it and points at the
/// equation that forced the decision; otherwise it is a plain one-liner.
fn compile_with_solver_notice(
    model: &ExecutionModel,
    source: Option<&str>,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    event_callback(
        "started".into(),
        format!("Compiling jit model `{}`", model.name),
    );
    let compiled = compile_execution_model_to_jit(model)?;
    if matches!(compiled, CompiledRuntimeModel::Ode(_)) {
        let info = compiled.info();
        let message = match source {
            Some(source) => info.solver_diagnostic().render(source),
            None => info.solver_explanation(),
        };
        event_callback("solver".into(), message);
    }
    event_callback(
        "finished".into(),
        format!("Compiled jit model `{}`", model.name),
    );
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::compile_sde_model_to_jit;
    use crate::test_fixtures::STRUCTURED_BLOCK_CORPUS;
    use crate::PharmsolError;
    use crate::SubjectBuilderExt;
    use pharmsol_dsl::{DiagnosticPhase, RouteKind, DSL_BACKEND_GENERIC, DSL_PARSE_GENERIC};

    const MULTI_DIGIT_OUTPUT_ORDER_RUNTIME_DSL: &str = r#"
name = multi_digit_output_runtime
kind = ode

params = ke, v
states = central
outputs = outeq_2, outeq_10, outeq_11

infusion(iv) -> central

dx(central) = -ke * central

out(outeq_10) = central / v ~ continuous()
out(outeq_2) = central / v ~ continuous()
out(outeq_11) = central / v ~ continuous()
"#;

    const NUMERIC_ROUTE_LABELS_RUNTIME_DSL: &str = r#"
name = prefixed_numeric_route_runtime
kind = ode

params = ke, v
states = central
outputs = cp

bolus(input_10) -> central
bolus(input_11) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

    const SHARED_NUMERIC_ROUTE_OUTPUT_LABEL_RUNTIME_DSL: &str = r#"
name = prefixed_numeric_route_output_runtime
kind = ode

params = ke, v
states = central
outputs = outeq_1

infusion(input_1) -> central

dx(central) = -ke * central

out(outeq_1) = central / v ~ continuous()
"#;

    const UNDECLARED_NUMERIC_OUTPUT_LABEL_RUNTIME_DSL: &str = r#"
name = undeclared_numeric_output_runtime
kind = ode

params = ke, v
states = central
outputs = a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10

infusion(iv) -> central

dx(central) = -ke * central

out(a0) = central / v ~ continuous()
out(a1) = central / v ~ continuous()
out(a2) = central / v ~ continuous()
out(a3) = central / v ~ continuous()
out(a4) = central / v ~ continuous()
out(a5) = central / v ~ continuous()
out(a6) = central / v ~ continuous()
out(a7) = central / v ~ continuous()
out(a8) = central / v ~ continuous()
out(a9) = central / v ~ continuous()
out(a10) = central / v ~ continuous()
"#;

    const UNDECLARED_NUMERIC_INPUT_LABEL_RUNTIME_DSL: &str = r#"
name = undeclared_numeric_input_runtime
kind = ode

params = ke, v
states = central
outputs = cp

bolus(r0) -> central
bolus(r1) -> central
bolus(r2) -> central
bolus(r3) -> central
bolus(r4) -> central
bolus(r5) -> central
bolus(r6) -> central
bolus(r7) -> central
bolus(r8) -> central
bolus(r9) -> central
bolus(r10) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

    fn corpus_source() -> &'static str {
        STRUCTURED_BLOCK_CORPUS
    }

    fn corpus_model(name: &str) -> ExecutionModel {
        let parsed = pharmsol_dsl::parse_module(corpus_source()).expect("parse corpus module");
        let analyzed = pharmsol_dsl::analyze_module(&parsed).expect("analyze corpus module");
        let model = analyzed
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model present in corpus module");
        pharmsol_dsl::compile_analyzed_model(model).expect("lower corpus model")
    }

    fn ode_subject() -> Subject {
        Subject::builder("ode")
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

    fn subject_values(predictions: &RuntimePredictions) -> Vec<f64> {
        predictions
            .as_subject()
            .expect("subject predictions")
            .predictions()
            .iter()
            .map(|prediction| prediction.prediction())
            .collect()
    }

    fn compile_runtime_model(source: &str, model_name: &str) -> CompiledRuntimeModel {
        compile_module_source_to_runtime(source, Some(model_name), |_, _| {})
            .expect("compile jit runtime model")
    }

    fn compiled_route_input_index(model: &CompiledRuntimeModel, name: &str) -> Option<usize> {
        model
            .info()
            .routes
            .iter()
            .find(|route| route.name == name)
            .map(|route| route.index)
    }

    fn compiled_output_slot_index(model: &CompiledRuntimeModel, name: &str) -> Option<usize> {
        model
            .info()
            .outputs
            .iter()
            .find(|output| output.name == name)
            .map(|output| output.index)
    }

    fn numeric_route_subject() -> Subject {
        Subject::builder("numeric-route-runtime")
            .bolus(0.0, 120.0, "input_10")
            .bolus(1.0, 80.0, "input_11")
            .missing_observation(0.5, "cp")
            .missing_observation(1.5, "cp")
            .build()
    }

    fn numeric_route_alias_subject() -> Subject {
        Subject::builder("numeric-route-runtime-alias")
            .bolus(0.0, 120.0, "10")
            .bolus(1.0, 80.0, "11")
            .missing_observation(0.5, "cp")
            .missing_observation(1.5, "cp")
            .build()
    }

    fn shared_numeric_route_output_subject() -> Subject {
        Subject::builder("prefixed-numeric-route-output-runtime")
            .infusion(0.0, 120.0, "input_1", 1.0)
            .missing_observation(0.5, "outeq_1")
            .missing_observation(1.5, "outeq_1")
            .build()
    }

    fn shared_numeric_route_output_alias_subject() -> Subject {
        Subject::builder("raw-numeric-route-output-runtime")
            .infusion(0.0, 120.0, "1", 1.0)
            .missing_observation(0.5, "1")
            .missing_observation(1.5, "1")
            .build()
    }

    fn mismatched_route_kind_subject() -> Subject {
        Subject::builder("mismatched-route-kind-runtime")
            .infusion(0.0, 120.0, "10", 1.0)
            .missing_observation(0.5, "cp")
            .build()
    }

    fn assert_unknown_output_label(
        model: &CompiledRuntimeModel,
        subject: &Subject,
        support: &Parameters,
        expected_label: &str,
    ) {
        let error = model
            .estimate_predictions(subject, support)
            .expect_err("undeclared numeric output label should fail");

        assert!(matches!(
            error,
            RuntimeError::Runtime(PharmsolError::UnknownOutputLabel { label, .. }) if label == expected_label
        ));
    }

    fn assert_unknown_input_label(
        model: &CompiledRuntimeModel,
        subject: &Subject,
        support: &Parameters,
        expected_label: &str,
    ) {
        let error = model
            .estimate_predictions(subject, support)
            .expect_err("undeclared numeric input label should fail");

        assert!(matches!(
            error,
            RuntimeError::Runtime(PharmsolError::UnknownInputLabel { label, .. }) if label == expected_label
        ));
    }

    fn assert_unsupported_input_route_kind(
        model: &CompiledRuntimeModel,
        subject: &Subject,
        support: &Parameters,
        expected_input: usize,
        expected_kind: RouteKind,
    ) {
        let error = model
            .estimate_predictions(subject, support)
            .expect_err("mismatched route kind should fail");

        match error {
            RuntimeError::Runtime(PharmsolError::UnsupportedInputRouteKind { input, kind })
                if input == expected_input && kind == expected_kind => {}
            other => panic!(
                "expected UnsupportedInputRouteKind {{ input: {expected_input}, kind: {:?} }}, got {other:?}",
                expected_kind
            ),
        }
    }

    #[test]
    fn runtime_jit_matches_ode_predictions() {
        let jit = compile_runtime_model(corpus_source(), "one_cmt_oral_iv");
        assert_eq!(jit.info().name, "one_cmt_oral_iv");
        assert_eq!(
            jit.info().parameters,
            vec!["ka", "cl", "v", "tlag", "f_oral"]
        );
        let support = Parameters::with_model(
            &jit,
            [
                ("ka", 1.2),
                ("cl", 5.0),
                ("v", 40.0),
                ("tlag", 0.5),
                ("f_oral", 0.8),
            ],
        )
        .expect("valid named parameters");

        assert!(compiled_route_input_index(&jit, "oral").is_some());
        assert!(compiled_route_input_index(&jit, "iv").is_some());
        assert_eq!(compiled_output_slot_index(&jit, "cp"), Some(0));
        let subject = ode_subject();

        let jit_values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        assert_eq!(jit_values.len(), 6);
        assert!(jit_values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn runtime_jit_kindless_routes_accept_both_input_kinds() {
        let jit = compile_runtime_model(corpus_source(), "one_cmt_oral_iv");
        let support = Parameters::with_model(
            &jit,
            [
                ("ka", 1.2),
                ("cl", 5.0),
                ("v", 40.0),
                ("tlag", 0.5),
                ("f_oral", 0.8),
            ],
        )
        .expect("valid named parameters");

        // Canonical `model {}` routes carry no kind and keep their declaration
        // ordinals. A future collapse of `None` to `Some(Bolus)` anywhere in
        // the lowering pipeline must fail here, close to its source.
        let routes = &jit.info().routes;
        let oral = routes
            .iter()
            .find(|route| route.name == "oral")
            .expect("oral route");
        let iv = routes
            .iter()
            .find(|route| route.name == "iv")
            .expect("iv route");
        assert_eq!(oral.kind, None, "oral route kind collapsed");
        assert_eq!(iv.kind, None, "iv route kind collapsed");
        assert_eq!(oral.index, 0);
        assert_eq!(iv.index, 1);

        // A kindless route is usable as either input kind: bolus and infusion
        // events both resolve through the same declaration, in both the
        // natural and the cross-kind directions.
        let natural_bolus = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, "oral")
            .missing_observation(1.0, "cp")
            .build();
        let natural_infusion = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .infusion(0.0, 60.0, "iv", 2.0)
            .missing_observation(1.0, "cp")
            .build();
        let cross_bolus = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, "iv")
            .missing_observation(1.0, "cp")
            .build();
        let cross_infusion = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .infusion(0.0, 60.0, "oral", 2.0)
            .missing_observation(1.0, "cp")
            .build();

        jit.estimate_predictions(&natural_bolus, &support)
            .expect("bolus oral resolves on kindless route");
        jit.estimate_predictions(&natural_infusion, &support)
            .expect("infusion iv resolves on kindless route");
        jit.estimate_predictions(&cross_bolus, &support)
            .expect("bolus iv resolves on kindless route");
        jit.estimate_predictions(&cross_infusion, &support)
            .expect("infusion oral resolves on kindless route");
    }

    #[test]
    fn runtime_jit_preserves_array_state_metadata() {
        let model = compile_module_source_to_runtime(
            corpus_source(),
            Some("transit_absorption"),
            |_, _| {},
        )
        .expect("compile jit runtime model");

        let metadata = model.metadata();
        assert_eq!(metadata.states()[0].name(), "transit");
        assert_eq!(metadata.states()[1].name(), "central");
        assert_eq!(metadata.route("oral").unwrap().destination(), "transit");
        assert_eq!(metadata.route("oral").unwrap().destination_index(), 0);

        assert_eq!(model.info().state_len, 5);
        assert_eq!(model.info().states[0].offset, 0);
        assert_eq!(model.info().states[1].offset, 4);
    }

    #[test]
    fn runtime_jit_reports_route_kind_mismatch() {
        let subject = mismatched_route_kind_subject();

        let jit = compile_runtime_model(
            NUMERIC_ROUTE_LABELS_RUNTIME_DSL,
            "prefixed_numeric_route_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");
        let expected_input =
            compiled_route_input_index(&jit, "input_10").expect("input_10 route index");

        assert_unsupported_input_route_kind(
            &jit,
            &subject,
            &support,
            expected_input,
            RouteKind::Infusion,
        );
    }

    #[test]
    fn runtime_jit_preserves_multi_digit_output_label_order() {
        let jit = compile_runtime_model(
            MULTI_DIGIT_OUTPUT_ORDER_RUNTIME_DSL,
            "multi_digit_output_runtime",
        );

        assert_eq!(compiled_output_slot_index(&jit, "outeq_2"), Some(0));
        assert_eq!(compiled_output_slot_index(&jit, "outeq_10"), Some(1));
        assert_eq!(compiled_output_slot_index(&jit, "outeq_11"), Some(2));
    }

    #[test]
    fn runtime_jit_supports_prefixed_multi_digit_numeric_route_labels() {
        let jit = compile_runtime_model(
            NUMERIC_ROUTE_LABELS_RUNTIME_DSL,
            "prefixed_numeric_route_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");

        assert_eq!(compiled_route_input_index(&jit, "input_10"), Some(0));
        assert_eq!(compiled_route_input_index(&jit, "input_11"), Some(1));

        let subject = numeric_route_subject();

        let values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn runtime_jit_resolves_raw_numeric_route_labels_against_prefixed_metadata() {
        let jit = compile_runtime_model(
            NUMERIC_ROUTE_LABELS_RUNTIME_DSL,
            "prefixed_numeric_route_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");

        let subject = numeric_route_alias_subject();

        let values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn runtime_jit_supports_prefixed_numeric_route_and_output_labels() {
        let jit = compile_runtime_model(
            SHARED_NUMERIC_ROUTE_OUTPUT_LABEL_RUNTIME_DSL,
            "prefixed_numeric_route_output_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");

        assert_eq!(compiled_route_input_index(&jit, "input_1"), Some(0));
        assert_eq!(compiled_output_slot_index(&jit, "outeq_1"), Some(0));

        let subject = shared_numeric_route_output_subject();

        let values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn runtime_jit_resolves_shared_raw_numeric_route_and_output_aliases() {
        let jit = compile_runtime_model(
            SHARED_NUMERIC_ROUTE_OUTPUT_LABEL_RUNTIME_DSL,
            "prefixed_numeric_route_output_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");

        let subject = shared_numeric_route_output_alias_subject();

        let values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn runtime_jit_rejects_undeclared_numeric_output_labels() {
        let jit = compile_runtime_model(
            UNDECLARED_NUMERIC_OUTPUT_LABEL_RUNTIME_DSL,
            "undeclared_numeric_output_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");
        let subject = Subject::builder("runtime-undeclared-numeric-output")
            .infusion(0.0, 100.0, "iv", 1.0)
            .missing_observation(0.5, "10")
            .build();

        assert_unknown_output_label(&jit, &subject, &support, "10");
    }

    #[test]
    fn runtime_jit_rejects_undeclared_numeric_input_labels() {
        let jit = compile_runtime_model(
            UNDECLARED_NUMERIC_INPUT_LABEL_RUNTIME_DSL,
            "undeclared_numeric_input_runtime",
        );
        let support = Parameters::with_model(&jit, [("ke", 0.2), ("v", 10.0)])
            .expect("valid named parameters");
        let subject = Subject::builder("runtime-undeclared-numeric-input")
            .bolus(0.0, 100.0, "10")
            .missing_observation(0.5, "cp")
            .build();

        assert_unknown_input_label(&jit, &subject, &support, "10");
    }

    #[test]
    fn runtime_compile_preserves_parse_diagnostic_structure() {
        let source = "model broken { kind ode outputs { cp = 1 + } }";
        let error = compile_module_source_to_runtime(source, None, |_, _| {})
            .expect_err("invalid DSL should fail before runtime compilation");

        let diagnostic = error
            .diagnostic()
            .expect("runtime should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Parse);
        assert_eq!(diagnostic.code, DSL_PARSE_GENERIC);
        assert!(diagnostic.message.contains("expected expression"));
        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL1000]"), "{}", rendered);
        assert!(rendered.contains("expected expression"), "{}", rendered);
        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL1000]"), "{}", debugged);
        assert!(debugged.contains("expected expression"), "{}", debugged);
        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "inline.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL1000");
        assert_eq!(report.diagnostics[0].labels[0].span.start_line, Some(1));
        assert!(report
            .to_json()
            .expect("serialize report")
            .contains("\"name\":\"inline.dsl\""),);
    }

    #[test]
    fn runtime_exposes_jit_backend_diagnostic_structure() {
        let source = corpus_source();
        let model = corpus_model("one_cmt_oral_iv");
        let error = RuntimeError::from(
            compile_sde_model_to_jit(&model)
                .expect_err("ODE model should not compile through the SDE JIT entrypoint")
                .with_source(source),
        );

        let diagnostic = error
            .diagnostic()
            .expect("runtime should expose jit diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Backend);
        assert_eq!(diagnostic.code, DSL_BACKEND_GENERIC);
        assert!(diagnostic.message.contains("not an SDE model"));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered backend diagnostic");
        assert!(rendered.contains("error[DSL4000]"), "{}", rendered);
        assert!(rendered.contains("not an SDE model"), "{}", rendered);

        let report = error
            .diagnostic_report("model.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "model.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL4000");
        assert_eq!(report.diagnostics[0].phase, "backend");
        assert!(report.diagnostics[0].labels[0].span.start_line.is_some());

        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL4000]"), "{}", debugged);
    }
}
