use pharmsol_dsl::{compile_model, parse_module, RouteKind};

fn route_summary(src: &str) -> Vec<(String, Option<RouteKind>, String)> {
    let model = compile_model(src).unwrap_or_else(|error| panic!("{error}"));
    model
        .metadata
        .routes
        .iter()
        .map(|route| {
            (
                route.name.clone(),
                route.kind,
                route.destination.state_name.clone(),
            )
        })
        .collect()
}

#[test]
fn pmetrics_aliases_infer_numeric_routes() {
    let routes = route_summary(
        r#"
params = ka, ke, v
states = depot, central
dx(depot) = b[1] - ka * depot
dx(central) = ka * depot + rateiv[1] - ke * central
out(cp) = central / v
"#,
    );
    assert_eq!(
        routes,
        vec![
            ("1".to_string(), Some(RouteKind::Bolus), "depot".to_string()),
            (
                "1".to_string(),
                Some(RouteKind::Infusion),
                "central".to_string()
            ),
        ]
    );
}

#[test]
fn pmetrics_aliases_parse_like_canonical_inputs() {
    let alias = parse_module(
        "params = ke\nstates = central\ndx(central) = 2 * b[1] + rateiv[2] - ke * central\nout(cp) = central\n",
    )
    .expect("alias model parses");
    let canonical = parse_module(
        "params = ke\nstates = central\ndx(central) = 2 * bolus[1] + infusion[2] - ke * central\nout(cp) = central\n",
    )
    .expect("canonical model parses");
    assert_eq!(alias.to_string(), canonical.to_string());
}

#[test]
fn state_array_named_b_is_not_an_input_alias() {
    let routes = route_summary(
        r#"
params = ke
states = b[2]
dx(b[0]) = bolus(iv) - ke * b[0]
dx(b[1]) = ke * b[0]
out(cp) = b[1]
"#,
    );
    assert_eq!(
        routes,
        vec![("iv".to_string(), Some(RouteKind::Bolus), "b".to_string())]
    );
}

#[test]
fn sde_inline_inputs_lower_like_declared_routes() {
    let declared = parse_module(
        r#"
kind = sde
params = ka, ke, v, s, f
states = depot, central
particles = 16
bolus(oral) -> depot
infusion(iv) -> central
fa(oral) = f
dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central
noise(central) = s
out(cp) = central / v
"#,
    )
    .expect("declared SDE parses");
    let inline = parse_module(
        r#"
kind = sde
params = ka, ke, v, s, f
states = depot, central
particles = 16
dx(depot) = -ka * depot + f * bolus(oral)
dx(central) = ka * depot - ke * central + infusion(iv)
noise(central) = s
out(cp) = central / v
"#,
    )
    .expect("inline SDE parses");
    assert_eq!(inline.to_string(), declared.to_string());
}

#[test]
fn sde_cannot_mix_declared_routes_and_inline_inputs() {
    let error = compile_model(
        r#"
kind = sde
params = ke, s
states = central
particles = 4
bolus(iv) -> central
dx(central) = -ke * central + infusion(iv)
noise(central) = s
out(cp) = central
"#,
    )
    .expect_err("mixing route styles should fail")
    .to_string();
    assert!(
        error.contains("inputs cannot also appear in `dx(...)`"),
        "{error}"
    );
}
