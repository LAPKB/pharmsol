use pharmsol_dsl::{analyze_model, compile_analyzed_model, parse_model, parse_module};

#[test]
fn output_annotation_is_optional() {
    let annotated = r#"
name = optional_output_annotation
kind = ode
states = central
ddt(central) = 0
out(cp) = central ~ continuous()
"#;
    let plain = r#"
name = optional_output_annotation
kind = ode
states = central
ddt(central) = 0
out(cp) = central
"#;

    let annotated = parse_module(annotated).expect("annotated authoring model parses");
    let plain = parse_module(plain).expect("plain authoring model parses");

    assert_eq!(annotated.to_string(), plain.to_string());
}

#[test]
fn dx_and_ddt_compile_equivalently() {
    let dx_src = r#"
name = derivative_alias
kind = ode
params = ke
states = central
dx(central) = -ke * central
out(cp) = central
"#;
    let ddt_src = r#"
name = derivative_alias
kind = ode
params = ke
states = central
ddt(central) = -ke * central
out(cp) = central
"#;

    let dx_model = parse_module(dx_src).expect("dx authoring model parses");
    let ddt_model = parse_module(ddt_src).expect("ddt authoring model parses");

    assert_eq!(dx_model.to_string(), ddt_model.to_string());
}

#[test]
fn rejects_out_target_not_in_declared_outputs() {
    let src = r#"
name = bimodal_ke
kind = ode
params = ke, v
states = central
outputs = cpa
infusion(iv) -> central
ddt(central) = -ke * central
out(cp) = central / v ~ continuous()
"#;

    let err = parse_model(src).expect_err("undeclared output target must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains("output `cp` is not declared in `outputs = ...`"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("declared output `cpa` is here"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: did you mean `cpa`?"),
        "{}",
        rendered
    );
    assert!(
        err.diagnostic().suggestions.iter().any(|suggestion| {
            suggestion.message.contains("did you mean `cpa`?")
                && suggestion
                    .edits
                    .iter()
                    .any(|edit| edit.replacement == "cpa")
        }),
        "{}",
        rendered
    );
}

#[test]
fn rejects_out_target_not_in_declared_outputs_when_declared_later() {
    let src = r#"
name = bimodal_ke
kind = ode
params = ke, v
states = central
out(cp) = central / v ~ continuous()
outputs = cpa
ddt(central) = -ke * central
"#;

    let err = parse_model(src).expect_err("late outputs declaration must still validate targets");
    let rendered = err.render(src);

    assert!(
        rendered.contains("output `cp` is not declared in `outputs = ...`"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: did you mean `cpa`?"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_declared_output_without_assignment() {
    let src = r#"
name = bimodal_ke
kind = ode
params = ke, v
states = central
outputs = cp, cpa
ddt(central) = -ke * central
out(cp) = central / v
"#;

    let err = parse_model(src).expect_err("unassigned declared output must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains("output `cpa` is declared in `outputs = ...` but never assigned"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_unknown_output_annotation_name() {
    let src = r#"
name = bimodal_ke
kind = ode
states = central
ddt(central) = 0
out(cp) = central ~ continous()
"#;

    let err = parse_model(src).expect_err("unknown output annotation must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains("expected the output annotation `continuous()`"),
        "{}",
        rendered
    );
}

#[test]
fn mixed_named_and_prefixed_numeric_output_labels_compile_and_round_trip() {
    let src = r#"
name = mixed_output_labels
kind = ode
params = ke, v
states = central
outputs = cp, outeq_0, outeq_1
infusion(iv) -> central
ddt(central) = -ke * central
out(cp) = central / v
out(outeq_0) = 2 * central / v
out(outeq_1) = 3 * central / v
"#;

    let module = parse_module(src).expect("mixed output labels should parse in authoring DSL");
    let model = module
        .models
        .first()
        .expect("authoring DSL should produce one model");
    let analyzed = analyze_model(model).expect("mixed output labels should analyze");
    let compiled = compile_analyzed_model(&analyzed).expect("mixed output labels should compile");

    assert_eq!(
        compiled
            .metadata
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["cp", "outeq_0", "outeq_1"]
    );
    assert_eq!(
        compiled
            .metadata
            .outputs
            .iter()
            .map(|output| output.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let rendered = module.to_string();
    let reparsed = parse_module(&rendered).expect("rendered mixed-output model should reparse");

    assert_eq!(rendered, reparsed.to_string());
}

#[test]
fn prefixed_numeric_route_and_output_labels_compile_and_round_trip() {
    let src = r#"
name = prefixed_numeric_route_output_labels
kind = ode
params = ke, v
states = central
outputs = outeq_1
infusion(input_1) -> central
ddt(central) = -ke * central
out(outeq_1) = central / v
"#;

    let module = parse_module(src).expect("prefixed numeric route/output labels should parse");
    let model = module
        .models
        .first()
        .expect("authoring DSL should produce one model");
    let analyzed =
        analyze_model(model).expect("prefixed numeric route/output labels should analyze");
    let compiled = compile_analyzed_model(&analyzed)
        .expect("prefixed numeric route/output labels should compile");

    assert_eq!(
        compiled
            .metadata
            .routes
            .iter()
            .map(|route| route.name.as_str())
            .collect::<Vec<_>>(),
        vec!["input_1"]
    );
    assert_eq!(
        compiled
            .metadata
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["outeq_1"]
    );

    let rendered = module.to_string();
    let reparsed = parse_module(&rendered).expect("rendered shared-label model should reparse");

    assert_eq!(rendered, reparsed.to_string());
}

#[test]
fn rejects_authoring_bare_numeric_output_declarations() {
    let src = r#"
name = numeric_outputs
kind = ode
states = central
outputs = 1, 2
ddt(central) = 0
out(1) = central
"#;

    let err = parse_model(src).expect_err("bare numeric output declarations must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "bare numeric output labels are not allowed in the DSL; use `outeq_1` instead"
        ),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: use `outeq_1`"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_authoring_bare_numeric_route_labels() {
    let src = r#"
name = numeric_routes
kind = ode
states = central
outputs = cp
infusion(1) -> central
ddt(central) = 0
out(cp) = central
"#;

    let err = parse_model(src).expect_err("bare numeric route labels must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "bare numeric route labels are not allowed in the DSL; use `input_1` instead"
        ),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: use `input_1`"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_structured_bare_numeric_output_targets() {
    let src = r#"
model numeric_output_target {
    kind ode
    states { central }
    outputs {
        1 = central
    }
}
"#;

    let model = parse_model(src).expect("structured model parses");
    let err = analyze_model(&model).expect_err("bare numeric output target must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "bare numeric output labels are not allowed in the DSL; use `outeq_1` instead"
        ),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: use `outeq_1`"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_structured_bare_numeric_route_labels() {
    let src = r#"
model numeric_route_label {
    kind ode
    states { central }
    routes {
        1 -> central
    }
    outputs {
        cp = central
    }
}
"#;

    let model = parse_model(src).expect("structured model parses");
    let err = analyze_model(&model).expect_err("bare numeric route label must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "bare numeric route labels are not allowed in the DSL; use `input_1` instead"
        ),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: use `input_1`"),
        "{}",
        rendered
    );
}

fn collect_route_input_indices(
    expr: &pharmsol_dsl::execution::ExecutionExpr,
    indices: &mut Vec<usize>,
) {
    use pharmsol_dsl::execution::{ExecutionExprKind, ExecutionLoad};
    match &expr.kind {
        ExecutionExprKind::Load(ExecutionLoad::RouteInput { index, .. }) => indices.push(*index),
        ExecutionExprKind::Unary { expr, .. } => collect_route_input_indices(expr, indices),
        ExecutionExprKind::Binary { lhs, rhs, .. } => {
            collect_route_input_indices(lhs, indices);
            collect_route_input_indices(rhs, indices);
        }
        ExecutionExprKind::Call { args, .. } => {
            for arg in args {
                collect_route_input_indices(arg, indices);
            }
        }
        _ => {}
    }
}

fn collect_stmt_route_input_indices(
    statements: &[pharmsol_dsl::execution::ExecutionStmt],
    indices: &mut Vec<usize>,
) {
    use pharmsol_dsl::execution::ExecutionStmtKind;
    for statement in statements {
        match &statement.kind {
            ExecutionStmtKind::Let(let_stmt) => {
                collect_route_input_indices(&let_stmt.value, indices);
            }
            ExecutionStmtKind::Assign(assign_stmt) => {
                collect_route_input_indices(&assign_stmt.value, indices);
            }
            ExecutionStmtKind::If(if_stmt) => {
                collect_route_input_indices(&if_stmt.condition, indices);
                collect_stmt_route_input_indices(&if_stmt.then_branch, indices);
                if let Some(else_branch) = &if_stmt.else_branch {
                    collect_stmt_route_input_indices(else_branch, indices);
                }
            }
            ExecutionStmtKind::For(for_stmt) => {
                collect_route_input_indices(&for_stmt.range.start, indices);
                collect_route_input_indices(&for_stmt.range.end, indices);
                collect_stmt_route_input_indices(&for_stmt.body, indices);
            }
        }
    }
}

#[test]
fn shared_label_bolus_and_infusion_compile_with_per_kind_slots() {
    let src = r#"
name = shared_label_authoring
kind = ode
params = ke, v, tlag
states = central
outputs = cp
bolus(input_1) -> central
infusion(input_1) -> central
lag(input_1) = tlag
ddt(central) = -ke * central
out(cp) = central / v
"#;

    let model = parse_model(src).expect("shared-label authoring model parses");
    let analyzed = analyze_model(&model).expect("shared-label authoring model analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("shared-label authoring model compiles");

    let routes = &compiled.metadata.routes;
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().all(|route| route.name == "input_1"));
    let bolus = routes
        .iter()
        .find(|route| route.kind == Some(pharmsol_dsl::RouteKind::Bolus))
        .expect("bolus route");
    let infusion = routes
        .iter()
        .find(|route| route.kind == Some(pharmsol_dsl::RouteKind::Infusion))
        .expect("infusion route");
    assert_eq!(bolus.index, 0);
    assert_eq!(infusion.index, 0);
    assert!(bolus.has_lag);
    assert!(!infusion.has_lag);

    // The injected `rate(input_1)` term must read the infusion input slot.
    let mut indices = Vec::new();
    if let Some(function) = compiled.function(pharmsol_dsl::execution::ModelFunctionKind::Dynamics)
    {
        if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
            collect_stmt_route_input_indices(&program.body.statements, &mut indices);
        }
    }
    assert_eq!(indices, vec![0]);
}

#[test]
fn rejects_shared_label_within_same_kind() {
    let src = r#"
name = duplicate_bolus
kind = ode
states = central
outputs = cp
bolus(input_1) -> central
bolus(input_1) -> central
ddt(central) = 0
out(cp) = central
"#;

    let err = parse_model(src).expect_err("duplicate bolus routes must fail");
    assert!(
        err.render(src).contains("duplicate route `input_1`"),
        "{}",
        err.render(src)
    );
}

#[test]
fn rejects_duplicate_route_names_in_canonical_models() {
    let src = r#"
model dup_routes {
    kind ode
    states { central }
    routes { input_1 -> central, input_1 -> central }
    dynamics { ddt(central) = 0 }
    outputs { cp = central }
}
"#;

    let model = parse_model(src).expect("canonical model parses");
    let err = analyze_model(&model).expect_err("duplicate canonical route names must fail");
    assert!(
        err.render(src).contains("duplicate route `input_1`"),
        "{}",
        err.render(src)
    );
}

#[test]
fn canonical_shared_label_routes_compile_with_per_kind_slots() {
    let src = r#"
model dosing {
    kind ode
    parameters { ke, v }
    states { central }
    routes {
        bolus input_1 -> central,
        infusion input_1 -> central
    }
    dynamics {
        ddt(central) = -ke * central
    }
    outputs {
        cp = central / v
    }
}
"#;

    let module = parse_module(src).expect("canonical shared-label model parses");
    let model = module.models.first().expect("one model");
    let analyzed = analyze_model(model).expect("canonical shared-label model analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("canonical shared-label model compiles");

    let routes = &compiled.metadata.routes;
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().all(|route| route.name == "input_1"));
    let bolus = routes
        .iter()
        .find(|route| route.kind == Some(pharmsol_dsl::RouteKind::Bolus))
        .expect("bolus route");
    let infusion = routes
        .iter()
        .find(|route| route.kind == Some(pharmsol_dsl::RouteKind::Infusion))
        .expect("infusion route");
    assert_eq!(bolus.index, 0);
    assert_eq!(infusion.index, 0);

    // Rendering preserves route kinds and reparses identically.
    let rendered = module.to_string();
    assert!(rendered.contains("bolus input_1 -> central"), "{rendered}");
    assert!(
        rendered.contains("infusion input_1 -> central"),
        "{rendered}"
    );
    let reparsed = parse_module(&rendered).expect("rendered canonical model reparses");
    assert_eq!(rendered, reparsed.to_string());
}

#[test]
fn canonical_route_kind_keyword_is_optional_and_unambiguous() {
    // `bolus -> gut` is a kind-less route whose label is `bolus`.
    let src = r#"
model kind_label {
    kind ode
    states { gut }
    routes { bolus -> gut }
    dynamics { ddt(gut) = 0 }
    outputs { cp = gut }
}
"#;

    let model = parse_model(src).expect("kind-looking label parses");
    let analyzed = analyze_model(&model).expect("kind-looking label analyzes");
    let compiled = compile_analyzed_model(&analyzed).expect("kind-looking label compiles");

    let route = &compiled.metadata.routes[0];
    assert_eq!(route.name, "bolus");
    assert_eq!(route.kind, None);
    assert_eq!(route.index, 0);
}

#[test]
fn rejects_mixed_kind_less_and_kinded_duplicate_route_names() {
    let src = r#"
model mixed_kind {
    kind ode
    states { central }
    routes {
        input_1 -> central,
        bolus input_1 -> central
    }
    dynamics { ddt(central) = 0 }
    outputs { cp = central }
}
"#;

    let model = parse_model(src).expect("mixed-kind model parses");
    let err = analyze_model(&model).expect_err("kind-less and kinded duplicates must fail");
    assert!(
        err.render(src).contains("duplicate route `input_1`"),
        "{}",
        err.render(src)
    );
}

#[test]
fn rejects_rate_numeric_literals_with_prefixed_guidance() {
    let src = r#"
model numeric_rate_arg {
    kind ode
    states { central }
    routes { input_5 -> central }
    dynamics {
        ddt(central) = rate(5)
    }
    outputs {
        cp = central
    }
}
"#;

    let model = parse_model(src).expect("structured model parses");
    let err = analyze_model(&model).expect_err("bare numeric rate argument must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "bare numeric route labels are not allowed in the DSL; use `input_5` instead"
        ),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: use `input_5`"),
        "{}",
        rendered
    );
}

#[test]
fn rejects_wrong_prefix_labels_in_authored_dsl() {
    let src = r#"
name = wrong_prefix_route
kind = ode
states = central
outputs = cp
infusion(outeq_1) -> central
ddt(central) = 0
out(cp) = central
"#;

    let err = parse_model(src).expect_err("wrong-prefix route labels must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "`outeq_1` is an output label and cannot be used as a route; use `input_1` here"
        ),
        "{}",
        rendered
    );

    let src = r#"
name = wrong_prefix_output
kind = ode
states = central
outputs = cp
infusion(iv) -> central
ddt(central) = 0
out(input_1) = central
"#;

    let err = parse_model(src).expect_err("wrong-prefix output labels must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains(
            "`input_1` is a route label and cannot be used as an output; use `outeq_1` here"
        ),
        "{}",
        rendered
    );
}

#[test]
fn route_labels_still_collide_with_scalar_symbol_names() {
    let src = r#"
name = route_state_collision
kind = ode
params = ke
states = central, iv
outputs = cp
infusion(iv) -> central
ddt(central) = -ke * central
ddt(iv) = 0
out(cp) = central
"#;

    let model = parse_model(src).expect("route/state collision model parses");
    let err = analyze_model(&model).expect_err("route label should still collide with state name");
    let rendered = err.render(src);

    assert!(
        rendered.contains("symbol name `iv` collides with existing `iv`"),
        "{}",
        rendered
    );
}

#[test]
fn unknown_route_destination_state_suggests_declared_state() {
    let src = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> centrale

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

    let model = parse_model(src).expect("authoring model parses");
    let err = analyze_model(&model).expect_err("unknown route destination state must fail");
    let rendered = err.render(src);

    assert!(
        rendered.contains("unknown state `centrale`"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("state `central` declared here"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("suggestion: did you mean `central`?"),
        "{}",
        rendered
    );
    assert!(
        err.diagnostic().suggestions.iter().any(|suggestion| {
            suggestion.message.contains("did you mean `central`?")
                && suggestion
                    .edits
                    .iter()
                    .any(|edit| edit.replacement == "central")
        }),
        "{}",
        rendered
    );
}
