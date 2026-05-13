use std::ops::Deref;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use ndarray::Array2;
use thiserror::Error;

use crate::parameter_order::{ParameterOrderError, ParameterOrderPlan};

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
use crate::dsl::{CompiledRuntimeModel, RuntimeAnalyticalModel, RuntimeSdeModel};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::Equation;

/// Errors produced while validating named parameter input against model order.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ParameterError {
    #[error("named parameter ingress requires parameter metadata")]
    MissingMetadata,
    #[error("unknown parameter `{name}`")]
    UnknownParameter { name: String },
    #[error("duplicate parameter `{name}`")]
    DuplicateParameter { name: String },
    #[error("missing required parameter(s): {names}")]
    MissingParameters { names: String },
    #[error("parameter order expects {expected} value(s), got {got}")]
    WidthMismatch { expected: usize, got: usize },
}

impl From<ParameterOrderError> for ParameterError {
    fn from(value: ParameterOrderError) -> Self {
        match value {
            ParameterOrderError::MissingMetadata => Self::MissingMetadata,
            ParameterOrderError::UnknownParameter { name } => Self::UnknownParameter { name },
            ParameterOrderError::DuplicateParameter { name } => Self::DuplicateParameter { name },
            ParameterOrderError::MissingParameters { names } => Self::MissingParameters {
                names: names.join(", "),
            },
            ParameterOrderError::WidthMismatch { expected, got } => {
                Self::WidthMismatch { expected, got }
            }
        }
    }
}

/// Thin owned dense parameter storage for one support point.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameters(Vec<f64>);

/// Reusable validated external parameter order for dense batch workflows.
///
/// Downstream crates with their own dense matrix types can validate source names
/// once with `with_model(...)`, then apply `permutation()` to their local row or
/// matrix storage during setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterOrder {
    plan: ParameterOrderPlan,
}

#[cfg(test)]
pub(crate) fn dense<V>(values: V) -> Parameters
where
    V: Into<Vec<f64>>,
{
    Parameters(values.into())
}

impl Parameters {
    /// Build a dense parameter vector from named values using model metadata.
    #[allow(private_bounds)]
    pub fn with_model<M, S, N>(model: &M, named_parameters: S) -> Result<Self, ParameterError>
    where
        M: NamedParameterModel,
        S: IntoIterator<Item = (N, f64)>,
        N: AsRef<str>,
    {
        let mut source_names = Vec::new();
        let mut source_values = Vec::new();

        for (name, value) in named_parameters {
            source_names.push(name.as_ref().to_string());
            source_values.push(value);
        }

        let plan = model.parameter_order_plan(source_names.iter().map(String::as_str))?;
        let dense_values = plan.reorder_values(&source_values)?;
        Ok(Self(dense_values))
    }

    /// Borrow the dense model-order values.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Consume the container and return the dense model-order values.
    pub fn into_inner(self) -> Vec<f64> {
        self.0
    }
}

impl ParameterOrder {
    /// Validate one external source order against model metadata.
    #[allow(private_bounds)]
    pub fn with_model<M, S, N>(model: &M, source_names: S) -> Result<Self, ParameterError>
    where
        M: NamedParameterModel,
        S: IntoIterator<Item = N>,
        N: AsRef<str>,
    {
        let plan = model.parameter_order_plan(source_names)?;
        Ok(Self { plan })
    }

    /// Reorder one dense support point from source order into model order.
    pub fn values(&self, source_values: &[f64]) -> Result<Vec<f64>, ParameterError> {
        self.plan
            .reorder_values(source_values)
            .map_err(ParameterError::from)
    }

    /// Reorder a dense support-point matrix whose rows are support points.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub fn matrix(&self, source_values: Array2<f64>) -> Result<Array2<f64>, ParameterError> {
        if source_values.ncols() != self.width() {
            return Err(ParameterError::WidthMismatch {
                expected: self.width(),
                got: source_values.ncols(),
            });
        }

        if self.is_identity() {
            return Ok(source_values);
        }

        let mut dense_values = Array2::default(source_values.raw_dim());
        for (model_column, source_column) in self.permutation().iter().copied().enumerate() {
            dense_values
                .column_mut(model_column)
                .assign(&source_values.column(source_column));
        }

        Ok(dense_values)
    }

    /// Borrow the model-order permutation where each element is the source index
    /// for that model-order slot.
    ///
    /// This is the stable low-level contract intended for downstream crates that
    /// keep their own dense matrix storage.
    pub fn permutation(&self) -> &[usize] {
        self.plan.permutation()
    }

    /// Return the expected dense width.
    pub fn width(&self) -> usize {
        self.plan.width()
    }

    /// Return whether the validated source order already matches model order.
    pub fn is_identity(&self) -> bool {
        self.plan.is_identity()
    }
}

impl AsRef<[f64]> for Parameters {
    fn as_ref(&self) -> &[f64] {
        self.as_slice()
    }
}

impl Deref for Parameters {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

trait NamedParameterModel {
    fn parameter_order_plan<S>(
        &self,
        source_names: S,
    ) -> Result<ParameterOrderPlan, ParameterError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>;
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<M> NamedParameterModel for M
where
    M: Equation,
{
    fn parameter_order_plan<S>(&self, source_names: S) -> Result<ParameterOrderPlan, ParameterError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        ParameterOrderPlan::from_metadata(self.metadata(), source_names)
            .map_err(ParameterError::from)
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl NamedParameterModel for CompiledRuntimeModel {
    fn parameter_order_plan<S>(&self, source_names: S) -> Result<ParameterOrderPlan, ParameterError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        ParameterOrderPlan::from_runtime_info(self.info(), source_names)
            .map_err(ParameterError::from)
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl NamedParameterModel for RuntimeAnalyticalModel {
    fn parameter_order_plan<S>(&self, source_names: S) -> Result<ParameterOrderPlan, ParameterError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        ParameterOrderPlan::from_runtime_info(self.info(), source_names)
            .map_err(ParameterError::from)
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl NamedParameterModel for RuntimeSdeModel {
    fn parameter_order_plan<S>(&self, source_names: S) -> Result<ParameterOrderPlan, ParameterError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        ParameterOrderPlan::from_runtime_info(self.info(), source_names)
            .map_err(ParameterError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::{ParameterError, ParameterOrder, Parameters};

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    use ndarray::array;

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    use crate::{fa, lag, metadata, Equation, ModelKind, Subject, SubjectBuilderExt, ODE};

    #[cfg(feature = "dsl-jit")]
    use crate::dsl::{compile_module_source_to_runtime, RuntimeCompilationTarget};

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn metadata_backed_ode() -> ODE {
        ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                dx[0] = -p[1] * x[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, x| {
                x[0] = 0.0;
            },
            |x, p, _t, _cov, y| {
                y[0] = x[0] / p[0];
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            metadata::new("named_parameter_ode")
                .kind(ModelKind::Ode)
                .parameters(["v", "ke"])
                .states(["central"])
                .outputs(["cp"])
                .route(metadata::Route::bolus("iv").to_state("central")),
        )
        .expect("attach metadata")
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn simple_subject() -> Subject {
        Subject::builder("named-parameters")
            .bolus(0.0, 100.0, "iv")
            .missing_observation(1.0, "cp")
            .build()
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn builds_dense_parameters_for_metadata_backed_ode() {
        let ode = metadata_backed_ode();
        let subject = simple_subject();

        let parameters = Parameters::with_model(&ode, [("ke", 0.5), ("v", 10.0)]).unwrap();
        let predictions = ode.estimate_predictions(&subject, &parameters).unwrap();

        assert_eq!(parameters.as_slice(), &[10.0, 0.5]);
        assert_eq!(predictions.predictions().len(), 1);
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn rejects_named_parameters_without_metadata() {
        let ode = ODE::new(
            |_x, _p, _t, _dx, _b, _rateiv, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |_x, _p, _t, _cov, _y| {},
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);

        let error = Parameters::with_model(&ode, [("ke", 0.5)]).unwrap_err();

        assert_eq!(error, ParameterError::MissingMetadata);
        assert_eq!(
            error.to_string(),
            "named parameter ingress requires parameter metadata"
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn reports_public_missing_parameter_error() {
        let ode = metadata_backed_ode();

        let error = Parameters::with_model(&ode, [("ke", 0.5)]).unwrap_err();

        assert_eq!(
            error,
            ParameterError::MissingParameters {
                names: "v".to_string(),
            }
        );
        assert_eq!(error.to_string(), "missing required parameter(s): v");
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn builds_identity_batch_order() {
        let ode = metadata_backed_ode();
        let order = ParameterOrder::with_model(&ode, ["v", "ke"]).unwrap();
        let theta = array![[10.0, 0.5], [20.0, 0.7]];

        assert!(order.is_identity());
        assert_eq!(order.permutation(), &[0, 1]);
        assert_eq!(order.width(), 2);
        assert_eq!(order.values(&[10.0, 0.5]).unwrap(), vec![10.0, 0.5]);
        assert_eq!(order.matrix(theta.clone()).unwrap(), theta);
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn reorders_dense_rows_and_matrices() {
        let ode = metadata_backed_ode();
        let order = ParameterOrder::with_model(&ode, ["ke", "v"]).unwrap();
        let theta = array![[0.5, 10.0], [0.7, 20.0]];

        assert!(!order.is_identity());
        assert_eq!(order.permutation(), &[1, 0]);
        assert_eq!(order.values(&[0.5, 10.0]).unwrap(), vec![10.0, 0.5]);
        assert_eq!(
            order.matrix(theta).unwrap(),
            array![[10.0, 0.5], [20.0, 0.7]]
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn rejects_wrong_width_batch_matrix() {
        let ode = metadata_backed_ode();
        let order = ParameterOrder::with_model(&ode, ["ke", "v"]).unwrap();
        let error = order.matrix(array![[0.5], [0.7]]).unwrap_err();

        assert_eq!(
            error,
            ParameterError::WidthMismatch {
                expected: 2,
                got: 1,
            }
        );
        assert_eq!(
            error.to_string(),
            "parameter order expects 2 value(s), got 1"
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn downstream_dense_theta_can_apply_parameter_order_permutation() {
        struct DenseTheta {
            names: Vec<String>,
            rows: Vec<Vec<f64>>,
        }

        impl DenseTheta {
            fn reordered(&self, order: &ParameterOrder) -> Self {
                let rows = self
                    .rows
                    .iter()
                    .map(|row| {
                        order
                            .permutation()
                            .iter()
                            .map(|&source_index| row[source_index])
                            .collect()
                    })
                    .collect();

                Self {
                    names: vec!["v".to_string(), "ke".to_string()],
                    rows,
                }
            }
        }

        let ode = metadata_backed_ode();
        let source_theta = DenseTheta {
            names: vec!["ke".to_string(), "v".to_string()],
            rows: vec![vec![0.5, 10.0], vec![0.7, 20.0]],
        };
        let order = ParameterOrder::with_model(&ode, source_theta.names.iter().map(String::as_str))
            .unwrap();

        let dense_theta = source_theta.reordered(&order);

        assert_eq!(order.permutation(), &[1, 0]);
        assert_eq!(dense_theta.names, vec!["v", "ke"]);
        assert_eq!(dense_theta.rows, vec![vec![10.0, 0.5], vec![20.0, 0.7]]);
    }

    #[cfg(feature = "dsl-jit")]
    #[test]
    fn builds_dense_parameters_for_compiled_runtime_model() {
        const SIMPLE_RUNTIME_DSL: &str = r#"
name = named_runtime
kind = ode

params = ke, v
states = central
outputs = cp

bolus(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

        let model = compile_module_source_to_runtime(
            SIMPLE_RUNTIME_DSL,
            Some("named_runtime"),
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )
        .expect("compile runtime model");

        let parameters = Parameters::with_model(&model, [("v", 50.0), ("ke", 1.2)]).unwrap();

        assert_eq!(parameters.as_slice(), &[1.2, 50.0]);
    }

    #[cfg(feature = "dsl-jit")]
    #[test]
    fn builds_batch_order_for_compiled_runtime_model() {
        const SIMPLE_RUNTIME_DSL: &str = r#"
name = named_runtime
kind = ode

params = ke, v
states = central
outputs = cp

bolus(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

        let model = compile_module_source_to_runtime(
            SIMPLE_RUNTIME_DSL,
            Some("named_runtime"),
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )
        .expect("compile runtime model");
        let order = ParameterOrder::with_model(&model, ["v", "ke"]).unwrap();

        assert_eq!(order.permutation(), &[1, 0]);
        assert_eq!(order.values(&[50.0, 1.2]).unwrap(), vec![1.2, 50.0]);
    }
}
