use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use pharmsol_dsl::execution::{
    ExecutionExpr, ExecutionExprKind, ExecutionLoad, ExecutionModel, ExecutionStmt,
    ExecutionStmtKind, KernelImplementation, KernelRole,
};
use pharmsol_dsl::{AnalyticalKernel, ModelKind, RouteKind};

/// Public metadata extracted from a compiled backend model.
///
/// This is the shared inspection surface returned by the native AoT, WASM, and
/// runtime loaders. It keeps public labels and buffer sizes available without
/// exposing backend-specific kernel details.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeModelInfo {
    /// Public model name.
    pub name: String,
    /// High-level model family.
    pub kind: ModelKind,
    /// Parameter names in support-point order.
    pub parameters: Vec<String>,
    /// Derived value names in dense runtime order.
    #[serde(default)]
    pub derived: Vec<String>,
    /// Declared covariates and their dense runtime indices.
    pub covariates: Vec<NativeCovariateInfo>,
    /// Declared routes together with declaration-order and dense runtime indices.
    pub routes: Vec<NativeRouteInfo>,
    /// Declared outputs and their dense runtime indices.
    pub outputs: Vec<NativeOutputInfo>,
    /// Length of the state buffer used during execution.
    pub state_len: usize,
    /// Length of the derived-value buffer used during execution.
    pub derived_len: usize,
    /// Length of the output buffer used during execution.
    pub output_len: usize,
    /// Length of the dense route-input buffer used during execution.
    pub route_len: usize,
    /// Built-in analytical structure metadata when the compiled model is
    /// analytical.
    pub analytical: Option<AnalyticalKernel>,
    /// Particle count when the compiled model is stochastic.
    pub particles: Option<usize>,
}

/// Metadata for one compiled covariate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCovariateInfo {
    /// Public covariate name.
    pub name: String,
    /// Dense runtime covariate index.
    pub index: usize,
}

/// Metadata for one compiled route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRouteInfo {
    /// Public route label.
    pub name: String,
    /// Route position in declaration order.
    #[serde(default)]
    pub declaration_index: usize,
    /// Dense runtime route-input index.
    pub index: usize,
    /// Coarse route kind when declared in metadata.
    #[serde(default)]
    pub kind: Option<RouteKind>,
    /// Dense destination state offset used by compiled kernels.
    pub destination_offset: usize,
    /// Whether the compiled backend injects the route input into the destination
    /// state automatically when the model does not read the route input
    /// explicitly.
    pub inject_input_to_destination: bool,
}

/// Metadata for one compiled output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOutputInfo {
    /// Public output label.
    pub name: String,
    /// Dense runtime output index.
    pub index: usize,
}

impl NativeModelInfo {
    /// Build public compiled-model metadata from a lowered execution model.
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
            derived: model
                .metadata
                .derived
                .iter()
                .map(|derived| derived.name.clone())
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
                    declaration_index: route.declaration_index,
                    index: route.index,
                    kind: route.kind,
                    destination_offset: route.destination.state_offset,
                    inject_input_to_destination: !explicit_route_input_usage
                        .get(route.declaration_index)
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
    let declaration_slots = model
        .metadata
        .routes
        .iter()
        .map(|route| (route.symbol, route.declaration_index))
        .collect::<BTreeMap<_, _>>();
    let Some(kernel) = (match model.kind {
        ModelKind::Ode => model.kernel(KernelRole::Dynamics),
        ModelKind::Sde => model.kernel(KernelRole::Drift),
        ModelKind::Analytical => None,
    }) else {
        return vec![false; model.metadata.routes.len()];
    };

    let mut usage = vec![false; model.metadata.routes.len()];
    if let KernelImplementation::Statements(program) = &kernel.implementation {
        mark_route_inputs_in_statements(&program.body.statements, &declaration_slots, &mut usage);
    }
    usage
}

fn mark_route_inputs_in_statements(
    statements: &[ExecutionStmt],
    declaration_slots: &BTreeMap<usize, usize>,
    usage: &mut [bool],
) {
    for statement in statements {
        match &statement.kind {
            ExecutionStmtKind::Let(let_stmt) => {
                mark_route_inputs_in_expr(&let_stmt.value, declaration_slots, usage);
            }
            ExecutionStmtKind::Assign(assign_stmt) => {
                mark_route_inputs_in_expr(&assign_stmt.value, declaration_slots, usage);
            }
            ExecutionStmtKind::If(if_stmt) => {
                mark_route_inputs_in_expr(&if_stmt.condition, declaration_slots, usage);
                mark_route_inputs_in_statements(&if_stmt.then_branch, declaration_slots, usage);
                if let Some(else_branch) = &if_stmt.else_branch {
                    mark_route_inputs_in_statements(else_branch, declaration_slots, usage);
                }
            }
            ExecutionStmtKind::For(for_stmt) => {
                mark_route_inputs_in_expr(&for_stmt.range.start, declaration_slots, usage);
                mark_route_inputs_in_expr(&for_stmt.range.end, declaration_slots, usage);
                mark_route_inputs_in_statements(&for_stmt.body, declaration_slots, usage);
            }
        }
    }
}

fn mark_route_inputs_in_expr(
    expr: &ExecutionExpr,
    declaration_slots: &BTreeMap<usize, usize>,
    usage: &mut [bool],
) {
    match &expr.kind {
        ExecutionExprKind::Literal(_) => {}
        ExecutionExprKind::Load(ExecutionLoad::RouteInput { route, .. }) => {
            if let Some(slot) = declaration_slots
                .get(route)
                .and_then(|index| usage.get_mut(*index))
            {
                *slot = true;
            }
        }
        ExecutionExprKind::Load(_) => {}
        ExecutionExprKind::Unary { expr, .. } => {
            mark_route_inputs_in_expr(expr, declaration_slots, usage)
        }
        ExecutionExprKind::Binary { lhs, rhs, .. } => {
            mark_route_inputs_in_expr(lhs, declaration_slots, usage);
            mark_route_inputs_in_expr(rhs, declaration_slots, usage);
        }
        ExecutionExprKind::Call { args, .. } => {
            for arg in args {
                mark_route_inputs_in_expr(arg, declaration_slots, usage);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pharmsol_dsl::{analyze_model, lower_typed_model, parse_model};

    fn load_model_info(src: &str) -> NativeModelInfo {
        let model = parse_model(src).expect("model parses");
        let typed = analyze_model(&model).expect("model analyzes");
        let lowered = lower_typed_model(&typed).expect("model lowers");
        NativeModelInfo::from_execution_model(&lowered)
    }

    #[test]
    fn declaration_first_routes_inject_by_default() {
        let info = load_model_info(
            r#"
model implicit_route_injection {
    kind ode
    states { central }
    routes { iv -> central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = central
    }
}
"#,
        );

        assert_eq!(info.routes.len(), 1);
        assert!(info.routes[0].inject_input_to_destination);
    }

    #[test]
    fn explicit_rate_usage_disables_automatic_injection() {
        let info = load_model_info(
            r#"
model explicit_route_usage {
    kind ode
    states { central }
    routes { iv -> central }
    dynamics {
        ddt(central) = rate(iv)
    }
    outputs {
        cp = central
    }
}
"#,
        );

        assert_eq!(info.routes.len(), 1);
        assert!(!info.routes[0].inject_input_to_destination);
    }

    #[test]
    fn authoring_shared_input_routes_keep_declaration_specific_injection() {
        let info = load_model_info(
            r#"
name = shared_authoring
kind = ode

params = ka, ke, v
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(cp) = central / v ~ continuous()
"#,
        );

        assert_eq!(info.route_len, 1);
        assert_eq!(info.routes.len(), 2);
        assert_eq!(info.routes[0].kind, Some(RouteKind::Bolus));
        assert_eq!(info.routes[1].kind, Some(RouteKind::Infusion));
        assert_eq!(info.routes[0].index, 0);
        assert_eq!(info.routes[1].index, 0);
        assert!(info.routes[0].inject_input_to_destination);
        assert!(!info.routes[1].inject_input_to_destination);
    }

    #[test]
    fn native_model_info_preserves_canonical_numeric_channel_names() {
        let info = load_model_info(
            r#"
name = canonical_numeric_channels
kind = ode

params = ke, v
states = depot, central
outputs = cp, outeq_2

bolus(input_10) -> depot
infusion(iv) -> central

dx(depot) = -ke * depot
dx(central) = rate(input_10) - ke * central

out(cp) = central / v
out(outeq_2) = depot / v
"#,
        );

        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.name.as_str())
                .collect::<Vec<_>>(),
            vec!["input_10", "iv"]
        );
        assert_eq!(
            info.outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cp", "outeq_2"]
        );
    }

    #[test]
    fn native_model_info_preserves_derived_names_for_analytical_models() {
        let info = load_model_info(
            r#"
name = analytical_derived_names
kind = analytical

params = ka, cl, v
derived = ke
states = depot, central
outputs = cp

bolus(oral) -> depot

ke = cl / v

structure = one_compartment_with_absorption

out(cp) = central / v ~ continuous()
"#,
        );

        assert_eq!(info.parameters, vec!["ka", "cl", "v"]);
        assert_eq!(info.derived, vec!["ke"]);
        assert_eq!(
            info.analytical,
            Some(AnalyticalKernel::OneCompartmentWithAbsorption)
        );
    }
}
