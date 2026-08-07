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
fn rejects_duplicate_kind_reappearing_after_shared_label() {
    // `bolus, infusion, bolus` under one label must be rejected even though
    // the parser's kind-keyed uniqueness cannot see it (the canonical parser
    // has no duplicate check; the analyzer tracks every kind seen per label).
    let src = r#"
model triple {
    kind ode
    states { central }
    routes {
        bolus input_1 -> central,
        infusion input_1 -> central,
        bolus input_1 -> central
    }
    dynamics { ddt(central) = 0 }
    outputs { cp = central }
}
"#;

    let model = parse_model(src).expect("triple-route model parses");
    let err = analyze_model(&model).expect_err("reappearing same-kind route must fail");
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

fn collect_if_branch_lens(
    statements: &[pharmsol_dsl::execution::ExecutionStmt],
    out: &mut Vec<(usize, Option<usize>)>,
) {
    use pharmsol_dsl::execution::ExecutionStmtKind;
    for statement in statements {
        match &statement.kind {
            ExecutionStmtKind::If(if_stmt) => {
                out.push((
                    if_stmt.then_branch.len(),
                    if_stmt.else_branch.as_ref().map(|branch| branch.len()),
                ));
                collect_if_branch_lens(&if_stmt.then_branch, out);
                if let Some(else_branch) = &if_stmt.else_branch {
                    collect_if_branch_lens(else_branch, out);
                }
            }
            ExecutionStmtKind::For(for_stmt) => {
                collect_if_branch_lens(&for_stmt.body, out);
            }
            _ => {}
        }
    }
}

#[test]
fn statement_level_if_with_braces_parses_and_lowers() {
    let src = r#"
name = braced_if_statement
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    x = 1
} else {
    x = 2
}
"#;
    let model = parse_model(src).expect("braced if authoring model parses");
    let analyzed = analyze_model(&model).expect("braced if model analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("braced if model compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("dynamics function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))], "one if with one stmt per branch");
}

#[test]
fn else_if_chain_lowers_to_nested_if() {
    let src = r#"
name = else_if_chain
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 1) {
    x = 1
} else if (ke > 0.5) {
    x = 2
} else {
    x = 3
}
"#;
    let model = parse_model(src).expect("else-if chain parses");
    let analyzed = analyze_model(&model).expect("else-if chain analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("else-if chain compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("dynamics function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    // outer if: then 1, else 1 (nested if); nested if: then 1, else 1
    assert_eq!(
        shapes,
        vec![(1, Some(1)), (1, Some(1))],
        "outer if nests an inner if in its else branch"
    );
}

#[test]
fn single_line_braced_if_expression_lowers() {
    let src = r#"
name = braced_if_expression
kind = ode
states = central
params = ke, v
ddt(central) = -ke * central
out(cp) = if (ke > 0.5) { central / v } else { central / (2 * v) }
"#;
    let model = parse_model(src).expect("braced if-expression parses");
    let analyzed = analyze_model(&model).expect("braced if-expression analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("braced if-expression compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Outputs)
        .expect("outputs function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))], "if-expression lowers to if");
}

#[test]
fn statement_level_if_requires_braces() {
    let src = r#"
name = unbraced_if
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5)
    out(cp) = central
"#;
    let model = parse_model(src).expect_err("unbraced if must be rejected");
    let rendered = model.render(src);
    assert!(
        rendered.contains("expected `{` to open `if`/`else` body"),
        "{}",
        rendered
    );
}

#[test]
fn unclosed_if_brace_is_rejected() {
    let src = r#"
name = unclosed_if
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    out(cp) = central
"#;
    let model = parse_model(src).expect_err("unclosed brace must be rejected");
    let rendered = model.render(src);
    assert!(
        rendered.contains("unclosed `{` in `if`/`else` body"),
        "{}",
        rendered
    );
}

#[test]
fn single_line_statement_level_if_parses() {
    let src = r#"
name = single_line_if
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) { x = 1 } else { x = 2 }
"#;
    let model = parse_model(src).expect("single-line statement if parses");
    let analyzed = analyze_model(&model).expect("single-line statement if analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("single-line statement if compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("derive function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))]);
}

#[test]
fn multi_line_if_expression_in_assignment_lowers() {
    let src = r#"
name = multiline_if_expression
kind = ode
states = central
params = ke, v
ddt(central) = -ke * central
out(cp) = if (ke > 0.5) {
    central / v
} else {
    central / (2 * v)
}
"#;
    let model = parse_model(src).expect("multi-line if-expression parses");
    let analyzed = analyze_model(&model).expect("multi-line if-expression analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("multi-line if-expression compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Outputs)
        .expect("outputs function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))]);
}

#[test]
fn comment_line_inside_multiline_if_block_does_not_truncate() {
    let src = r#"
name = commented_if_block
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    # keep this comment; it must not swallow the rest of the block
    x = 1
} else {
    x = 2
}
"#;
    let model = parse_model(src).expect("commented if block parses");
    let analyzed = analyze_model(&model).expect("commented if block analyzes");
    let compiled =
        compile_analyzed_model(&analyzed).expect("commented if block compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("derive function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))]);
}

#[test]
fn nested_braced_if_expression_associates_outer_else() {
    let src = r#"
name = nested_braced_if_expression
kind = ode
states = central
params = ka, kb
ddt(central) = -ka * central
out(cp) = if (ka > 1) { if (kb > 0.5) { 1 } else { 2 } } else { 3 }
"#;
    let model = parse_model(src).expect("nested braced if-expression parses");
    let analyzed = analyze_model(&model).expect("nested braced if-expression analyzes");
    let compiled = compile_analyzed_model(&analyzed)
        .expect("nested braced if-expression compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Outputs)
        .expect("outputs function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(
        shapes,
        vec![(1, Some(1)), (1, Some(1))],
        "outer if wraps the inner if; both have one stmt per branch"
    );
}

#[test]
fn trailing_tokens_after_braced_else_body_are_rejected() {
    let src = r#"
name = trailing_else_tokens
kind = ode
states = central
params = ke
ddt(central) = -ke * central
out(cp) = if (ke > 0.5) { 1 } else { 2 } + 3
"#;
    let err = parse_model(src).expect_err("trailing tokens after else body must fail");
    let rendered = err.render(src);
    assert!(
        rendered.contains("unexpected tokens after `if`/`else` expression"),
        "{}",
        rendered
    );
}

#[test]
fn else_on_own_line_or_after_trivia_lines_parses() {
    let variants = [
        "} else {",
        "}\nelse {",
        "}\n\nelse {",
        "}\n# explanation\nelse {",
    ];
    for (i, variant) in variants.iter().enumerate() {
        let src = format!(
            r#"
name = trivia_else_{i}
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {{
    x = 1
{variant}
    x = 2
}}
"#
        );
        let model = parse_model(&src).unwrap_or_else(|err| {
            panic!("variant {variant:?} failed: {}", err.render(&src))
        });
        let analyzed = analyze_model(&model).expect("model analyzes");
        let compiled = compile_analyzed_model(&analyzed).expect("model compiles");

        let mut shapes = Vec::new();
        let function = compiled
            .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
            .expect("dynamics function");
        if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
            collect_if_branch_lens(&program.body.statements, &mut shapes);
        }
        assert_eq!(shapes, vec![(1, Some(1))], "variant {variant:?}");
    }
}

#[test]
fn else_if_chain_with_blank_and_comment_lines_parses() {
    let src = r#"
name = chain_with_trivia
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 1) {
    x = 1
}

else if (ke > 0.5) {
    x = 2
}

# a comment sitting between the chain links
else {
    x = 3
}
"#;
    let model = parse_model(src).expect("else-if chain with trivia parses");
    let analyzed = analyze_model(&model).expect("model analyzes");
    let compiled = compile_analyzed_model(&analyzed).expect("model compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("dynamics function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(
        shapes,
        vec![(1, Some(1)), (1, Some(1))],
        "outer if nests an inner if in its else branch"
    );
}

#[test]
fn nested_statement_if_else_across_lines_parses() {
    let variants = [
        "} else {",
        "}\nelse {",
        "}\n\nelse {",
        "}\n    # explanation\n    else {",
    ];
    for (i, variant) in variants.iter().enumerate() {
        let src = format!(
            r#"
name = nested_if_else_{i}
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {{
    if (ke > 1) {{
        x = 1
    {variant}
        x = 2
    }}
}}
"#
        );
        let model = parse_model(&src).unwrap_or_else(|err| {
            panic!("variant {variant:?} failed: {}", err.render(&src))
        });
        let analyzed = analyze_model(&model).expect("model analyzes");
        let compiled = compile_analyzed_model(&analyzed).expect("model compiles");

        let mut shapes = Vec::new();
        let function = compiled
            .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
            .expect("dynamics function");
        if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
            collect_if_branch_lens(&program.body.statements, &mut shapes);
        }
        assert_eq!(
            shapes,
            vec![(1, None), (1, Some(1))],
            "outer if holds the nested if; nested if has one stmt per branch (variant {variant:?})"
        );
    }
}

#[test]
fn inline_comments_in_statement_if_branches_parse() {
    let src = r#"
name = inline_comments_if
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    x = 1 # then-branch comment
} else {
    x = 2 # else-branch comment
} # trailing comment after the closing brace
"#;
    let model = parse_model(src).expect("inline comments in if branches parse");
    let analyzed = analyze_model(&model).expect("model analyzes");
    let compiled = compile_analyzed_model(&analyzed).expect("model compiles");

    let mut shapes = Vec::new();
    let function = compiled
        .function(pharmsol_dsl::execution::ModelFunctionKind::Derive)
        .expect("dynamics function");
    if let pharmsol_dsl::execution::FunctionBody::Statements(program) = &function.body {
        collect_if_branch_lens(&program.body.statements, &mut shapes);
    }
    assert_eq!(shapes, vec![(1, Some(1))]);
}

#[test]
fn trailing_tokens_after_statement_else_body_are_rejected() {
    let src = r#"
name = trailing_statement_else_tokens
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    x = 1
} else {
    x = 2
} garbage
"#;
    let err = parse_model(src).expect_err("trailing tokens after statement else body must fail");
    let rendered = err.render(src);
    assert!(
        rendered.contains("unexpected tokens after `if`/`else` statement"),
        "{}",
        rendered
    );
}

#[test]
fn diagnostic_positions_survive_earlier_inline_comments() {
    let src = r#"
name = diag_positions
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    x = 1 # a long inline comment that used to shift later spans: pad pad pad pad pad pad pad
    y = 
}
"#;
    let err = parse_model(src).expect_err("empty rhs inside if body must be rejected");
    let rendered = err.render(src);
    assert!(
        rendered.contains("expected `name = <expression>`"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("--> line 9, column 5"),
        "diagnostic must point at the `y =` line, got:\n{}",
        rendered
    );
}

#[test]
fn call_style_assignment_inside_statement_if_is_rejected() {
    let src = r#"
name = call_style_if_body
kind = ode
states = central
params = ke
ddt(central) = -ke * central
if (ke > 0.5) {
    ddt(central) = -2 * ke * central
}
"#;
    let err = parse_model(src).expect_err("call-style assignment in if body must be rejected");
    let rendered = err.render(src);
    assert!(
        rendered.contains("only plain-variable assignments"),
        "{}",
        rendered
    );
    assert!(
        rendered.contains("ddt(central) = if (cond) <a> else <b>"),
        "{}",
        rendered
    );
}
