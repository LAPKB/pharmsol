//! Hardening and API tests for the DSL frontend: error quality, pathological
//! inputs, constant evaluation edges, and the unified pipeline surface.

use pharmsol_dsl::{
    analyze_model, compile_model, compile_module, lower_typed_model, parse_model, ConstValue,
    Diagnostic, DiagnosticCode, DiagnosticPhase, DiagnosticReport, DslError, ParseError,
    SemanticError, Span, DSL_PARSE_GENERIC, DSL_SEMANTIC_GENERIC, MAX_CONST_USIZE,
    MAX_NESTING_DEPTH,
};

fn ode_skeleton(body: &str) -> String {
    format!(
        "model m {{ kind ode states {{ central }} dynamics {{ {body} }} outputs {{ cp = central }} }}"
    )
}

fn ode_with_constants(constants: &str) -> String {
    format!(
        "model m {{ kind ode constants {{ {constants} }} states {{ central }} dynamics {{ ddt(central) = 0 }} outputs {{ cp = central }} }}"
    )
}

// ---------------------------------------------------------------------------
// Number literals
// ---------------------------------------------------------------------------

#[test]
fn rejects_number_literals_that_overflow_to_infinity() {
    for src in [
        ode_skeleton("ddt(central) = 1e999"),
        ode_skeleton("ddt(central) = -1e999"),
        "name = m\nkind = ode\nstates = central\nddt(central) = 1e999\nout(cp) = central"
            .to_string(),
    ] {
        let err = parse_model(&src).expect_err("overflowing literal must be rejected");
        let rendered = err.render(&src);
        assert!(rendered.contains("overflows to infinity"), "{}", rendered);
    }
}

#[test]
fn accepts_large_but_finite_number_literals() {
    let src = ode_skeleton("ddt(central) = 1e308");
    let model = parse_model(&src).expect("finite literal parses");
    let typed = analyze_model(&model).expect("finite literal analyzes");
    assert!(matches!(
        typed.constants.as_slice(),
        [] // no constants; the literal lives in the dynamics block
    ));
}

// ---------------------------------------------------------------------------
// Recursion depth
// ---------------------------------------------------------------------------

fn nested_expr(open: &str, close: &str, depth: usize) -> String {
    let mut expr = String::new();
    for _ in 0..depth {
        expr.push_str(open);
    }
    expr.push('1');
    for _ in 0..depth {
        expr.push_str(close);
    }
    expr
}

#[test]
fn rejects_deeply_nested_expressions_without_crashing() {
    let parens = ode_skeleton(&format!("ddt(central) = {}", nested_expr("(", ")", 10_000)));
    let err = parse_model(&parens).expect_err("deep parens must fail cleanly");
    assert_eq!(err.diagnostics().len(), 1, "{}", err.render(&parens));
    assert!(err.diagnostics()[0].message.contains("nested too deeply"));

    let unary = ode_skeleton(&format!("ddt(central) = {}1", "-".repeat(10_000)));
    let err = parse_model(&unary).expect_err("deep unary chain must fail cleanly");
    assert!(err.diagnostics()[0].message.contains("nested too deeply"));

    let caret = ode_skeleton(&format!("ddt(central) = 1{}", "^1".repeat(5_000)));
    let err = parse_model(&caret).expect_err("deep caret chain must fail cleanly");
    assert!(err.diagnostics()[0].message.contains("nested too deeply"));
}

#[test]
fn rejects_deeply_nested_statements_without_crashing() {
    let mut body = String::new();
    for _ in 0..1_000 {
        body.push_str("if true { ");
    }
    body.push_str("ddt(central) = 1");
    for _ in 0..1_000 {
        body.push_str(" }");
    }
    let src = ode_skeleton(&body);
    let err = parse_model(&src).expect_err("deep statement nesting must fail cleanly");
    assert!(
        err.diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("nested too deeply")),
        "{}",
        err.render(&src)
    );

    let mut body = String::from("if true { ddt(central) = 1 }");
    for _ in 0..1_000 {
        body.push_str(" else if true { ddt(central) = 1 }");
    }
    let src = ode_skeleton(&body);
    let err = parse_model(&src).expect_err("deep else-if chain must fail cleanly");
    assert!(
        err.diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("nested too deeply")),
        "{}",
        err.render(&src)
    );
}

#[test]
fn rejects_deeply_nested_authoring_conditionals_without_crashing() {
    let mut rhs = String::from("if (true) 1");
    for _ in 0..1_000 {
        rhs.push_str(" else if (true) 1");
    }
    rhs.push_str(" else 2");
    let src = format!("states = central\nddt(central) = 0\nout(cp) = {rhs}\n");
    let err = parse_model(&src).expect_err("deep authoring conditionals must fail cleanly");
    let rendered = err.render(&src);
    assert!(rendered.contains("nested too deeply"), "{}", rendered);
}

#[test]
fn moderate_nesting_still_parses() {
    let parens = ode_skeleton(&format!("ddt(central) = {}", nested_expr("(", ")", 64)));
    parse_model(&parens).expect("64 levels of parens parse");

    let mut body = String::new();
    for _ in 0..100 {
        body.push_str("if true { ");
    }
    body.push_str("ddt(central) = 1");
    for _ in 0..100 {
        body.push_str(" }");
    }
    parse_model(&ode_skeleton(&body)).expect("100 nested if statements parse");
}

// ---------------------------------------------------------------------------
// Constant evaluation
// ---------------------------------------------------------------------------

fn analyzed_constant(src: &str, name: &str) -> ConstValue {
    let model = parse_model(src).expect("model parses");
    let typed = analyze_model(&model).expect("model analyzes");
    typed
        .constants
        .iter()
        .find(|constant| {
            typed
                .symbols
                .iter()
                .any(|symbol| symbol.id == constant.symbol && symbol.name == name)
        })
        .unwrap_or_else(|| panic!("constant `{name}` exists"))
        .value
        .clone()
}

#[test]
fn integer_constant_overflow_degrades_to_real_instead_of_panicking() {
    for constants in ["x = 4611686018427387904 * 2", "x = 9223372036854775807 + 1"] {
        assert_eq!(
            analyzed_constant(&ode_with_constants(constants), "x"),
            ConstValue::Real(9.223372036854776e18),
            "{constants}"
        );
    }
    assert_eq!(
        analyzed_constant(&ode_with_constants("x = -9223372036854775807 - 2"), "x"),
        ConstValue::Real(-9.223372036854776e18)
    );
}

#[test]
fn negating_i64_min_degrades_to_real_instead_of_panicking() {
    let src = ode_with_constants("a = -9223372036854775807 - 1, b = -a");
    assert_eq!(
        analyzed_constant(&src, "b"),
        ConstValue::Real(9.223372036854776e18)
    );
}

#[test]
fn constant_two_to_the_63_stays_real() {
    let src = ode_with_constants("x = 9223372036854775808");
    assert_eq!(
        analyzed_constant(&src, "x"),
        ConstValue::Real(9.223372036854776e18)
    );
}

#[test]
fn compile_time_calls_enforce_intrinsic_arity() {
    for (call, message) in [
        (
            "pow(2, 3, 4)",
            "function `pow` expects 2 argument(s), got 3",
        ),
        ("max(1)", "function `max` expects 2 argument(s), got 1"),
    ] {
        let src = ode_with_constants(&format!("x = {call}"));
        let model = parse_model(&src).expect("model parses");
        let err = analyze_model(&model).expect_err("arity mismatch must fail");
        let rendered = err.render(&src);
        assert!(rendered.contains(message), "{}", rendered);
    }
}

#[test]
fn state_array_size_rejects_non_representable_integer() {
    let src = "model m { kind ode states { x[9223372036854775808], central } dynamics { ddt(central) = 0 } outputs { cp = central } }";
    let model = parse_model(src).expect("model parses");
    let err = analyze_model(&model).expect_err("2^63 state size must fail");
    let rendered = err.render(src);
    assert!(
        rendered.contains("state array size must be an integer constant"),
        "{}",
        rendered
    );
}

#[test]
fn compile_time_sizes_are_capped() {
    let src = format!(
        "model m {{ kind ode states {{ x[{}], central }} dynamics {{ ddt(central) = 0 }} outputs {{ cp = central }} }}",
        MAX_CONST_USIZE + 1
    );
    let model = parse_model(&src).expect("model parses");
    let err = analyze_model(&model).expect_err("oversized state array must fail");
    let rendered = err.render(&src);
    assert!(
        rendered.contains(&format!(
            "state array size exceeds the maximum supported value of {MAX_CONST_USIZE}"
        )),
        "{}",
        rendered
    );

    let src = format!(
        "model m {{ kind sde states {{ central }} particles {} drift {{ ddt(central) = 0 }} diffusion {{ noise(central) = 0 }} outputs {{ cp = central }} }}",
        MAX_CONST_USIZE + 1
    );
    let model = parse_model(&src).expect("model parses");
    let err = analyze_model(&model).expect_err("oversized particle count must fail");
    let rendered = err.render(&src);
    assert!(
        rendered.contains(&format!(
            "particles exceeds the maximum supported value of {MAX_CONST_USIZE}"
        )),
        "{}",
        rendered
    );

    let src = format!(
        "model m {{ kind ode states {{ x[{MAX_CONST_USIZE}], central }} dynamics {{ ddt(x[0]) = 0, ddt(central) = 0 }} outputs {{ cp = central }} }}"
    );
    let model = parse_model(&src).expect("model parses");
    analyze_model(&model).expect("state array at the cap still analyzes");
}

// ---------------------------------------------------------------------------
// Unified pipeline and DslError
// ---------------------------------------------------------------------------

const VALID_MODEL: &str = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central
out(cp) = central / v
"#;

#[test]
fn compile_model_matches_the_staged_pipeline() {
    let staged = lower_typed_model(
        &analyze_model(&parse_model(VALID_MODEL).expect("parses")).expect("analyzes"),
    )
    .expect("lowers");
    let one_shot = compile_model(VALID_MODEL).expect("compiles");

    assert_eq!(one_shot.name, staged.name);
    assert_eq!(one_shot.metadata.routes.len(), 1);
    assert_eq!(one_shot.metadata.outputs.len(), 1);
}

#[test]
fn compile_module_lowers_every_model() {
    let src = "model a { kind ode states { central } dynamics { ddt(central) = 0 } outputs { cp = central } }\nmodel b { kind ode states { central } dynamics { ddt(central) = 0 } outputs { cp = central } }";
    let module = compile_module(src).expect("module compiles");
    assert_eq!(module.models.len(), 2);
}

#[test]
fn dsl_error_reports_phase_and_renders_source() {
    let err = compile_model("model broken { kind ode").unwrap_err();
    assert!(matches!(err, DslError::Parse(_)));
    assert_eq!(err.phase(), DiagnosticPhase::Parse);
    assert!(!err.diagnostics().is_empty());
    assert!(err.source().is_some());
    assert!(err.to_string().contains("error[DSL1000]"));
    assert!(format!("{err:?}").contains("error[DSL1000]"));

    let src = ode_skeleton("ddt(central) = mystery");
    let err = compile_model(&src).unwrap_err();
    assert!(matches!(err, DslError::Semantic(_)));
    assert_eq!(err.phase(), DiagnosticPhase::Semantic);
    assert!(err.to_string().contains("error[DSL2000]"));
    assert!(err.to_string().contains("unknown identifier `mystery`"));

    let src = "name = m\nkind = ode\nstates = central\ninfusion(iv) -> central\nlag(iv) = 0.5\nddt(central) = 0\nout(cp) = central\n";
    let err = compile_model(src).unwrap_err();
    assert!(matches!(err, DslError::Lowering(_)));
    assert_eq!(err.phase(), DiagnosticPhase::Lowering);
    let rendered = err.to_string();
    assert!(
        rendered.contains("does not allow `lag` on infusion route `iv`"),
        "{}",
        rendered
    );
}

#[test]
fn dsl_error_builds_structured_reports() {
    let err = compile_model("model broken { kind ode").unwrap_err();
    let report: DiagnosticReport = err.diagnostic_report("broken.dsl");
    let json = report.to_json().expect("report serializes");
    assert!(json.contains("broken.dsl"));
    assert!(json.contains("DSL1000"));

    let src = ode_skeleton("ddt(central) = mystery");
    let err = compile_model(&src).unwrap_err();
    assert!(err.render(&src).contains("error[DSL2000]"));
    let report = err.diagnostic_report("model.dsl");
    assert_eq!(report.diagnostics.len(), 1);
}

#[test]
fn dsl_error_with_source_renders_after_the_fact() {
    let err = parse_model("model broken { kind ode").unwrap_err();
    let err = DslError::from(err);
    let rendered = err.with_source("model broken { kind ode").to_string();
    assert!(rendered.contains("error[DSL1000]"));
}

// ---------------------------------------------------------------------------
// Console formatting
// ---------------------------------------------------------------------------

#[test]
fn display_without_source_is_compact_and_coded() {
    let err = ParseError::new("expected `->`", Span::new(3, 5));
    assert_eq!(
        err.to_string(),
        "error[DSL1000]: expected `->` (at bytes 3..5)"
    );

    let first = Diagnostic::error(
        DSL_PARSE_GENERIC,
        DiagnosticPhase::Parse,
        "first problem",
        Span::new(0, 1),
    );
    let second = Diagnostic::error(
        DSL_PARSE_GENERIC,
        DiagnosticPhase::Parse,
        "second problem",
        Span::new(2, 3),
    );
    let err = ParseError::from_diagnostics(vec![first, second]);
    assert_eq!(
        err.to_string(),
        "error[DSL1000]: first problem (at bytes 0..1) (+1 more error)"
    );

    let err = SemanticError::new("unknown identifier `ke`", Span::new(10, 12));
    assert_eq!(
        err.to_string(),
        "error[DSL2000]: unknown identifier `ke` (at bytes 10..12)"
    );
}

#[test]
fn render_does_not_repeat_source_for_labels_on_the_same_line() {
    let src = "model m { kind ode constants { x = 1, x = 2 } states { central } dynamics { ddt(central) = x } outputs { cp = central } }";
    let model = parse_model(src).expect("model parses");
    let err = analyze_model(&model).expect_err("duplicate constant must fail");
    let rendered = err.render(src);
    assert!(rendered.contains("duplicate constant `x`"), "{}", rendered);
    assert_eq!(
        rendered.matches("constants { x = 1, x = 2 }").count(),
        1,
        "same-line labels share one source excerpt:\n{}",
        rendered
    );
}

#[test]
fn malformed_inputs_fail_with_errors_not_panics() {
    let cases = [
        "",
        "model",
        "model {",
        "model m {",
        "model m { kind }",
        "model m { kind ode",
        "states = ",
        "states = central\nddt(central) = ",
        "states = central\nddt(central) = 1 +\n",
        "bolus(iv) -> ",
        "x = if (true) 1",
        "x = f(",
        "params = ke, ",
        "kind = invalid",
        "model m { kind ode states { x[] } }",
        "model m { kind ode routes { oral -> } }",
        "model m { kind ode outputs { 18446744073709551616 = central } }",
    ];
    for src in cases {
        assert!(
            parse_model(src).is_err(),
            "input should fail cleanly: {src:?}"
        );
    }
}

#[test]
fn route_destination_index_is_bounds_checked() {
    let src = "model m { kind ode states { transit[4], central } routes { oral -> transit[9] } dynamics { ddt(transit[0]) = 0, ddt(central) = 0 } outputs { cp = central } }";
    let err = compile_model(src).unwrap_err();
    assert!(matches!(err, DslError::Lowering(_)));
    let rendered = err.render(src);
    assert!(
        rendered.contains("indexes element 9, but state length is 4"),
        "{}",
        rendered
    );
}

#[test]
fn diagnostic_codes_are_stable_and_documented() {
    assert_eq!(DSL_PARSE_GENERIC, DiagnosticCode::new("DSL1000"));
    assert_eq!(DSL_SEMANTIC_GENERIC, DiagnosticCode::new("DSL2000"));

    let src = ode_skeleton(&format!("ddt(central) = {}", nested_expr("(", ")", 1_000)));
    let err = parse_model(&src).expect_err("deep nesting must fail");
    assert!(
        err.diagnostics()[0]
            .message
            .contains(&format!("maximum nesting depth is {MAX_NESTING_DEPTH}")),
        "{}",
        err.diagnostics()[0].message
    );
}
