use serde::{Deserialize, Serialize};

use pharmsol_dsl::execution::{
    ExecutionExpr, ExecutionExprKind, ExecutionLoad, ExecutionStmt, ExecutionStmtKind,
    ExecutionModel, KernelImplementation, KernelRole,
};
use pharmsol_dsl::{AnalyticalKernel, ModelKind};

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
    pub inject_input_to_destination: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOutputInfo {
    pub name: String,
    pub index: usize,
}

impl NativeModelInfo {
    pub fn from_execution_model(model: &ExecutionModel) -> Self {
        let explicit_route_input_usage = explicit_route_input_usage(model);
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
                    inject_input_to_destination: !explicit_route_input_usage
                        .get(route.index)
                        .copied()
                        .unwrap_or(false),
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

fn explicit_route_input_usage(model: &ExecutionModel) -> Vec<bool> {
    let Some(kernel) = (match model.kind {
        ModelKind::Ode => model.kernel(KernelRole::Dynamics),
        ModelKind::Sde => model.kernel(KernelRole::Drift),
        ModelKind::Analytical => None,
    }) else {
        return vec![false; model.metadata.routes.len()];
    };

    let mut usage = vec![false; model.metadata.routes.len()];
    if let KernelImplementation::Statements(program) = &kernel.implementation {
        mark_route_inputs_in_statements(&program.body.statements, &mut usage);
    }
    usage
}

fn mark_route_inputs_in_statements(statements: &[ExecutionStmt], usage: &mut [bool]) {
    for statement in statements {
        match &statement.kind {
            ExecutionStmtKind::Let(let_stmt) => {
                mark_route_inputs_in_expr(&let_stmt.value, usage);
            }
            ExecutionStmtKind::Assign(assign_stmt) => {
                mark_route_inputs_in_expr(&assign_stmt.value, usage);
            }
            ExecutionStmtKind::If(if_stmt) => {
                mark_route_inputs_in_expr(&if_stmt.condition, usage);
                mark_route_inputs_in_statements(&if_stmt.then_branch, usage);
                if let Some(else_branch) = &if_stmt.else_branch {
                    mark_route_inputs_in_statements(else_branch, usage);
                }
            }
            ExecutionStmtKind::For(for_stmt) => {
                mark_route_inputs_in_expr(&for_stmt.range.start, usage);
                mark_route_inputs_in_expr(&for_stmt.range.end, usage);
                mark_route_inputs_in_statements(&for_stmt.body, usage);
            }
        }
    }
}

fn mark_route_inputs_in_expr(expr: &ExecutionExpr, usage: &mut [bool]) {
    match &expr.kind {
        ExecutionExprKind::Literal(_) => {}
        ExecutionExprKind::Load(ExecutionLoad::RouteInput(index)) => {
            if let Some(slot) = usage.get_mut(*index) {
                *slot = true;
            }
        }
        ExecutionExprKind::Load(_) => {}
        ExecutionExprKind::Unary { expr, .. } => mark_route_inputs_in_expr(expr, usage),
        ExecutionExprKind::Binary { lhs, rhs, .. } => {
            mark_route_inputs_in_expr(lhs, usage);
            mark_route_inputs_in_expr(rhs, usage);
        }
        ExecutionExprKind::Call { args, .. } => {
            for arg in args {
                mark_route_inputs_in_expr(arg, usage);
            }
        }
    }
}
