#![cfg(feature = "dsl-core")]

use pharmsol::dsl::{analyze_model, parse_model, parse_module};

#[test]
fn output_annotation_is_optional() {
    let annotated = r#"
model = optional_output_annotation
kind = ode
states = central
ddt(central) = 0
out(cp) = central ~ continuous()
"#;
    let plain = r#"
model = optional_output_annotation
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
fn dx_and_ddt_lower_equivalently() {
    let dx_src = r#"
model = derivative_alias
kind = ode
params = ke
states = central
dx(central) = -ke * central
out(cp) = central
"#;
    let ddt_src = r#"
model = derivative_alias
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
model = bimodal_ke
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
        err.diagnostic()
            .suggestions
            .iter()
            .any(
                |suggestion| suggestion.message.contains("did you mean `cpa`?")
                    && suggestion
                        .edits
                        .iter()
                        .any(|edit| edit.replacement == "cpa")
            ),
        "{}",
        rendered
    );
}

#[test]
fn rejects_out_target_not_in_declared_outputs_when_declared_later() {
    let src = r#"
model = bimodal_ke
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
model = bimodal_ke
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
model = bimodal_ke
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
fn unknown_route_destination_state_suggests_declared_state() {
    let src = r#"
model = bimodal_ke
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
        err.diagnostic()
            .suggestions
            .iter()
            .any(
                |suggestion| suggestion.message.contains("did you mean `central`?")
                    && suggestion
                        .edits
                        .iter()
                        .any(|edit| edit.replacement == "central")
            ),
        "{}",
        rendered
    );
}
