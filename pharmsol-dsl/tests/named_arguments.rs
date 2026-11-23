use pharmsol_dsl::{compile_model, parse_module};

const SHORTHAND: &str = r#"
name = named_effect
kind = ode

params = ke, v, ec50, gamma
states = central
outputs = cp, effect

dx(central) = infusion(iv) - ke * central
out(cp) = central / v
out(effect) = effect50(
    max = 100,
    e50 = ec50,
    h = gamma, x = central / v
)
"#;

const CANONICAL: &str = r#"
model named_effect {
    kind ode
    parameters { ke, v, ec50, gamma }
    states { central }
    dynamics {
        ddt(central) = -ke * central
    }
    outputs {
        cp = central / v
        effect = effect50(
            max = 100,
            e50 = ec50,
            h = gamma,
            x = central / v
        )
    }
}
"#;

#[test]
fn named_arguments_may_span_lines_in_both_syntaxes() {
    let shorthand = compile_model(SHORTHAND).expect("shorthand compiles");
    assert_eq!(shorthand.metadata.outputs.len(), 2);

    let module = parse_module(CANONICAL).expect("canonical source parses");
    let printed = module.to_string();
    assert!(
        printed.contains("effect50(max = 100, e50 = ec50, h = gamma, x = (central / v))"),
        "{printed}"
    );
}

#[test]
fn printed_named_arguments_parse_back_to_the_same_call() {
    let module = parse_module(CANONICAL).expect("canonical source parses");
    let printed = module.to_string();
    let reparsed = parse_module(&printed).expect("printed source parses");
    assert_eq!(reparsed.to_string(), printed);
}

#[test]
fn unclosed_call_before_an_assignment_reports_the_next_statement() {
    let source = r#"
model unclosed {
    kind ode
    parameters { v }
    states { central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = max(central,
        other = v
    }
}
"#;
    let error = compile_model(source).expect_err("unclosed call must fail");
    let rendered = error.to_string();
    assert!(
        rendered.contains("expected expression after `,`"),
        "{rendered}"
    );
    assert!(
        rendered.contains("next statement starts here"),
        "{rendered}"
    );
}
