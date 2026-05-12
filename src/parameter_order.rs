#![allow(dead_code)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[cfg(feature = "dsl-core")]
use crate::dsl::NativeModelInfo;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::simulator::equation::ValidatedModelMetadata;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParameterOrderPlan {
    permutation: Vec<usize>,
    width: usize,
    identity: bool,
}

impl ParameterOrderPlan {
    pub(crate) fn from_names<M, S>(
        model_names: M,
        source_names: S,
    ) -> Result<Self, ParameterOrderError>
    where
        M: IntoIterator,
        M::Item: AsRef<str>,
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        let model_names = model_names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();
        let mut model_index_by_name = HashMap::with_capacity(model_names.len());
        for (index, name) in model_names.iter().enumerate() {
            model_index_by_name.insert(name.as_str(), index);
        }

        let mut permutation = vec![usize::MAX; model_names.len()];
        let mut width = 0;

        for source_name in source_names {
            let source_name = source_name.as_ref();
            let Some(&model_index) = model_index_by_name.get(source_name) else {
                return Err(ParameterOrderError::UnknownParameter {
                    name: source_name.to_string(),
                });
            };

            if permutation[model_index] != usize::MAX {
                return Err(ParameterOrderError::DuplicateParameter {
                    name: source_name.to_string(),
                });
            }

            permutation[model_index] = width;
            width += 1;
        }

        let missing = model_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                if permutation[index] == usize::MAX {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ParameterOrderError::MissingParameters { names: missing });
        }

        let identity = permutation
            .iter()
            .enumerate()
            .all(|(model_index, source_index)| model_index == *source_index);

        Ok(Self {
            permutation,
            width,
            identity,
        })
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub(crate) fn from_metadata<S>(
        metadata: Option<&ValidatedModelMetadata>,
        source_names: S,
    ) -> Result<Self, ParameterOrderError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        let Some(metadata) = metadata else {
            return Err(ParameterOrderError::MissingMetadata);
        };

        Self::from_names(
            metadata
                .parameters()
                .iter()
                .map(|parameter| parameter.name()),
            source_names,
        )
    }

    #[cfg(feature = "dsl-core")]
    pub(crate) fn from_runtime_info<S>(
        info: &NativeModelInfo,
        source_names: S,
    ) -> Result<Self, ParameterOrderError>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        Self::from_names(info.parameters.iter().map(String::as_str), source_names)
    }

    pub(crate) fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }

    pub(crate) fn reorder_values(
        &self,
        source_values: &[f64],
    ) -> Result<Vec<f64>, ParameterOrderError> {
        if source_values.len() != self.width {
            return Err(ParameterOrderError::WidthMismatch {
                expected: self.width,
                got: source_values.len(),
            });
        }

        if self.identity {
            return Ok(source_values.to_vec());
        }

        Ok(self
            .permutation
            .iter()
            .map(|source_index| source_values[*source_index])
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParameterOrderError {
    MissingMetadata,
    UnknownParameter { name: String },
    DuplicateParameter { name: String },
    MissingParameters { names: Vec<String> },
    WidthMismatch { expected: usize, got: usize },
}

impl fmt::Display for ParameterOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMetadata => {
                f.write_str("named parameter ingress requires parameter metadata")
            }
            Self::UnknownParameter { name } => write!(f, "unknown parameter `{name}`"),
            Self::DuplicateParameter { name } => write!(f, "duplicate parameter `{name}`"),
            Self::MissingParameters { names } => {
                write!(f, "missing required parameter(s): {}", names.join(", "))
            }
            Self::WidthMismatch { expected, got } => {
                write!(f, "parameter order expects {expected} value(s), got {got}")
            }
        }
    }
}

impl Error for ParameterOrderError {}

#[cfg(test)]
mod tests {
    use super::{ParameterOrderError, ParameterOrderPlan};

    #[cfg(feature = "dsl-core")]
    use crate::dsl::NativeModelInfo;
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    use crate::{metadata, ModelKind};
    #[cfg(feature = "dsl-core")]
    use pharmsol_dsl::ModelKind as DslModelKind;

    #[test]
    fn builds_identity_permutation() {
        let plan = ParameterOrderPlan::from_names(["ka", "ke"], ["ka", "ke"]).unwrap();

        assert_eq!(plan.permutation(), &[0, 1]);
        assert_eq!(plan.width(), 2);
        assert!(plan.is_identity());
        assert_eq!(plan.reorder_values(&[0.1, 0.3]).unwrap(), vec![0.1, 0.3]);
    }

    #[test]
    fn builds_reordered_permutation() {
        let plan = ParameterOrderPlan::from_names(["ka", "ke"], ["ke", "ka"]).unwrap();

        assert_eq!(plan.permutation(), &[1, 0]);
        assert_eq!(plan.width(), 2);
        assert!(!plan.is_identity());
        assert_eq!(plan.reorder_values(&[0.3, 0.1]).unwrap(), vec![0.1, 0.3]);
    }

    #[test]
    fn rejects_unknown_parameter() {
        let error = ParameterOrderPlan::from_names(["ka", "ke"], ["ka", "kel"]).unwrap_err();

        assert_eq!(
            error,
            ParameterOrderError::UnknownParameter {
                name: "kel".to_string(),
            }
        );
        assert_eq!(error.to_string(), "unknown parameter `kel`");
    }

    #[test]
    fn rejects_duplicate_parameter() {
        let error = ParameterOrderPlan::from_names(["ka", "ke"], ["ka", "ka"]).unwrap_err();

        assert_eq!(
            error,
            ParameterOrderError::DuplicateParameter {
                name: "ka".to_string(),
            }
        );
        assert_eq!(error.to_string(), "duplicate parameter `ka`");
    }

    #[test]
    fn reports_all_missing_parameters_in_model_order() {
        let error = ParameterOrderPlan::from_names(["ka", "ke", "v"], ["v"]).unwrap_err();

        assert_eq!(
            error,
            ParameterOrderError::MissingParameters {
                names: vec!["ka".to_string(), "ke".to_string()],
            }
        );
        assert_eq!(error.to_string(), "missing required parameter(s): ka, ke");
    }

    #[test]
    fn rejects_width_mismatch_when_reordering_values() {
        let plan = ParameterOrderPlan::from_names(["ka", "ke"], ["ke", "ka"]).unwrap();
        let error = plan.reorder_values(&[0.3]).unwrap_err();

        assert_eq!(
            error,
            ParameterOrderError::WidthMismatch {
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
    fn metadata_wrapper_requires_metadata() {
        let error = ParameterOrderPlan::from_metadata(None, ["ka", "ke"]).unwrap_err();

        assert_eq!(error, ParameterOrderError::MissingMetadata);
        assert_eq!(
            error.to_string(),
            "named parameter ingress requires parameter metadata"
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[test]
    fn metadata_wrapper_uses_declared_parameter_order() {
        let metadata = metadata::new("one_cmt")
            .kind(ModelKind::Ode)
            .parameters(["ka", "ke"])
            .states(["central"])
            .outputs(["cp"])
            .route(metadata::Route::infusion("iv").to_state("central"))
            .validate()
            .unwrap();

        let plan = ParameterOrderPlan::from_metadata(Some(&metadata), ["ke", "ka"]).unwrap();

        assert_eq!(plan.permutation(), &[1, 0]);
        assert_eq!(plan.reorder_values(&[0.3, 0.1]).unwrap(), vec![0.1, 0.3]);
    }

    #[cfg(feature = "dsl-core")]
    #[test]
    fn runtime_info_wrapper_uses_declared_parameter_order() {
        let info = NativeModelInfo {
            name: "one_cmt".to_string(),
            kind: DslModelKind::Ode,
            parameters: vec!["ka".to_string(), "ke".to_string()],
            derived: Vec::new(),
            covariates: Vec::new(),
            routes: Vec::new(),
            outputs: Vec::new(),
            state_len: 0,
            derived_len: 0,
            output_len: 0,
            route_len: 0,
            analytical: None,
            particles: None,
        };

        let plan = ParameterOrderPlan::from_runtime_info(&info, ["ke", "ka"]).unwrap();

        assert_eq!(plan.permutation(), &[1, 0]);
        assert_eq!(plan.reorder_values(&[0.3, 0.1]).unwrap(), vec![0.1, 0.3]);
    }
}
