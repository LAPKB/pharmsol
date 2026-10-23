use pharmsol_dsl::{compile_model, parse_module};

const ONE_LINE: &str = r#"
name = layout
kind = ode
params = ke, v
states = central
outputs = cp
dx(central) = -ke * central + infusion(iv)
out(cp) = central / v
"#;

fn parses_like_one_line(src: &str) {
    let expected = parse_module(ONE_LINE).expect("reference model parses");
    let actual = parse_module(src).unwrap_or_else(|error| panic!("{}", error.render(src)));
    assert_eq!(actual.to_string(), expected.to_string());
}

fn render_error(src: &str) -> String {
    compile_model(src)
        .expect_err("model should be rejected")
        .to_string()
}

#[test]
fn expression_continues_after_trailing_operator_on_indented_line() {
    parses_like_one_line(
        r#"
name = layout
kind = ode
params = ke, v
states = central
outputs = cp
dx(central) = -ke * central +
    infusion(iv)
out(cp) = central / v
"#,
    );
}

#[test]
fn expression_continues_on_line_starting_with_operator() {
    parses_like_one_line(
        r#"
name = layout
kind = ode
params = ke, v
states = central
outputs = cp
dx(central) = -ke * central
    + infusion(iv)  # comments are allowed between continuation lines
out(cp) = central / v
"#,
    );
}

#[test]
fn expression_continues_inside_open_parentheses() {
    parses_like_one_line(
        r#"
name = layout
kind = ode
params = ke, v
states = central
outputs = cp
dx(central) = (
    -ke * central
    + infusion(iv)
)
out(cp) = central / v
"#,
    );
}

#[test]
fn declaration_list_continues_after_trailing_comma() {
    parses_like_one_line(
        r#"
name = layout
kind = ode
params = ke,
    v
states = central
outputs = cp
dx(central) = -ke * central + infusion(iv)
out(cp) = central / v
"#,
    );
}

#[test]
fn semicolons_separate_statements_on_one_line() {
    parses_like_one_line(
        r#"
name = layout; kind = ode
params = ke, v; states = central; outputs = cp
dx(central) = -ke * central + infusion(iv); out(cp) = central / v
"#,
    );
}

#[test]
fn unclosed_group_is_reported_on_its_own_line() {
    let report = render_error(
        r#"name = m
states = central
dx(central) = -(central
out(cp) = central
"#,
    );
    assert!(report.contains("line 3"), "{report}");
    assert!(report.contains("expected `)`"), "{report}");
}

#[test]
fn duplicate_derivative_is_rejected() {
    let report = render_error(
        r#"name = m
params = ke
states = central
dx(central) = -ke * central
dx(central) = 0
out(cp) = central
"#,
    );
    assert!(
        report.contains("`dx(central)` is assigned more than once"),
        "{report}"
    );
    assert!(report.contains("line 5"), "{report}");
    assert!(report.contains("first assigned here"), "{report}");
}

#[test]
fn dx_and_ddt_for_the_same_state_are_duplicates() {
    let report = render_error(
        r#"name = m
states = central
dx(central) = 0
ddt(central) = 1
out(cp) = central
"#,
    );
    assert!(report.contains("assigned more than once"), "{report}");
}

#[test]
fn duplicate_output_across_both_spellings_is_rejected() {
    let report = render_error(
        r#"name = m
states = central
outputs = cp
dx(central) = 0
out(cp) = central
cp = 2 * central
"#,
    );
    assert!(
        report.contains("`out(cp)` is assigned more than once"),
        "{report}"
    );
}

#[test]
fn missing_state_equation_points_at_every_unassigned_state() {
    let report = render_error(
        r#"name = m
params = ke
states = a, b, c
dx(a) = -ke * a
out(cp) = a
"#,
    );
    assert!(
        report.contains("dynamics block does not assign `b`, `c`"),
        "{report}"
    );
    assert!(report.contains("line 3"), "{report}");
    assert!(report.contains("dx(b) = 0"), "{report}");
}

#[test]
fn misspelled_model_kind_suggests_the_closest_kind() {
    let report = render_error("name = m\nkind = analytic\nstates = central\nout(cp) = central\n");
    assert!(report.contains("did you mean `analytical`?"), "{report}");
}

#[test]
fn misspelled_declaration_header_is_named() {
    let report = render_error(
        r#"name = m
paramz = ke, v
states = central
dx(central) = -ke * central
out(cp) = central / v
"#,
    );
    assert!(
        report.contains("`paramz` is not a declaration header"),
        "{report}"
    );
    assert!(report.contains("did you mean `params`?"), "{report}");
}

#[test]
fn double_equals_statement_suggests_single_equals() {
    let report = render_error(
        r#"name = m
states = central
dx(central) == 0
out(cp) = central
"#,
    );
    assert!(
        report.contains("expected `=` in equation, found `==`"),
        "{report}"
    );
    assert!(report.contains("use `=` to assign"), "{report}");
}

#[test]
fn double_star_suggests_caret() {
    let report =
        render_error("name = m\nparams = ke\nstates = c\ndx(c) = -ke * c**2\nout(cp) = c\n");
    assert!(
        report.contains("`**` is not an operator; powers use `^`"),
        "{report}"
    );
}
