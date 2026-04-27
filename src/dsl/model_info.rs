use serde::{Deserialize, Serialize};

use super::{AnalyticalKernel, ExecutionModel, ModelKind};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeModelInfo {
    pub name: String,
    pub kind: ModelKind,
    pub parameters: Vec<String>,
    pub covariates: Vec<NativeCovariateInfo>,
    pub routes: Vec<NativeRouteInfo>,
    pub outputs: Vec<NativeOutputInfo>,
    pub state_len: usize,
    pub derived_len: usize,
    pub output_len: usize,
    pub route_len: usize,
    pub analytical: Option<AnalyticalKernel>,
    pub particles: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCovariateInfo {
    pub name: String,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRouteInfo {
    pub name: String,
    pub index: usize,
    pub destination_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOutputInfo {
    pub name: String,
    pub index: usize,
}

impl NativeModelInfo {
    pub fn from_execution_model(model: &ExecutionModel) -> Self {
        Self {
            name: model.name.clone(),
            kind: model.kind,
            parameters: model
                .metadata
                .parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect(),
            covariates: model
                .metadata
                .covariates
                .iter()
                .map(|covariate| NativeCovariateInfo {
                    name: covariate.name.clone(),
                    index: covariate.index,
                })
                .collect(),
            routes: model
                .metadata
                .routes
                .iter()
                .map(|route| NativeRouteInfo {
                    name: route.name.clone(),
                    index: route.index,
                    destination_offset: route.destination.state_offset,
                })
                .collect(),
            outputs: model
                .metadata
                .outputs
                .iter()
                .map(|output| NativeOutputInfo {
                    name: output.name.clone(),
                    index: output.index,
                })
                .collect(),
            state_len: model.abi.state_buffer.len,
            derived_len: model.abi.derived_buffer.len,
            output_len: model.abi.output_buffer.len,
            route_len: model.abi.route_buffer.len,
            analytical: model.metadata.analytical,
            particles: model.metadata.particles,
        }
    }
}