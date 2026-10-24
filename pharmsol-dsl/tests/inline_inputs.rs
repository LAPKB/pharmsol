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
fn sde_inline_inputs_infer_routes_and_bioavailability() {
    let model = compile_model(
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
    .unwrap_or_else(|error| panic!("{error}"));
    let routes: Vec<_> = model
        .metadata
        .routes
        .iter()
        .map(|route| {
            (
                route.name.as_str(),
                route.kind,
                route.destination.state_name.as_str(),
                route.has_bioavailability,
            )
        })
        .collect();
    assert_eq!(
        routes,
        vec![
            ("oral", Some(RouteKind::Bolus), "depot", true),
            ("iv", Some(RouteKind::Infusion), "central", false),
        ]
    );
}

#[test]
fn sde_route_declarations_are_rejected() {
    let error = compile_model(
        r#"
kind = sde
params = ke, s
states = central
particles = 4
infusion(iv) -> central
dx(central) = -ke * central
noise(central) = s
out(cp) = central
"#,
    )
    .expect_err("SDE route declarations are removed")
    .to_string();
    assert!(
        error.contains("SDE inputs belong in the derivative RHS"),
        "{error}"
    );
}

#[test]
fn dose_input_outside_derivative_is_named() {
    let error =
        compile_model("params = ke\nstates = c\nx = b[1]\ndx(c) = x - ke * c\nout(cp) = c\n")
            .expect_err("bolus outside dx should fail")
            .to_string();
    assert!(
        error.contains("dose input `b` can only be used in a `dx(...)` equation"),
        "{error}"
    );
}

#[test]
fn infusion_outside_derivative_points_to_rate() {
    let error = compile_model(
        "params = ke\nstates = c\ndx(c) = rateiv[1] - ke * c\nout(cp) = c + rateiv[1]\n",
    )
    .expect_err("infusion outside dx should fail")
    .to_string();
    assert!(error.contains("use `rate(1)`"), "{error}");
    compile_model("params = ke\nstates = c\ndx(c) = rateiv[1] - ke * c\nout(cp) = c + rate(1)\n")
        .expect("the suggested `rate(1)` compiles");
}

#[test]
fn bolus_scaled_by_state_is_explained() {
    let error = compile_model("params = ke\nstates = c\ndx(c) = b[1] * c - ke * c\nout(cp) = c\n")
        .expect_err("state-scaled bolus should fail")
        .to_string();
    assert!(
        error.contains(
            "a bolus can only be scaled by values that do not depend on states; `c` is a state"
        ),
        "{error}"
    );
}

#[test]
fn analytical_model_without_routes_is_rejected() {
    let error = compile_model(
        "kind = analytical\nparams = ke, v\nstates = central\nstructure = one_compartment\nout(cp) = central / v\n",
    )
    .expect_err("route-less analytical model should fail")
    .to_string();
    assert!(
        error.contains("declares no routes, so doses in the data would be ignored"),
        "{error}"
    );
}

#[test]
fn analytical_model_with_only_initial_conditions_needs_no_routes() {
    compile_model(
        "kind = analytical\nparams = ke, v, c0\nstates = central\nstructure = one_compartment\ninit(central) = c0\nout(cp) = central / v\n",
    )
    .expect("init-driven analytical model compiles");
}

#[test]
fn empty_canonical_routes_block_does_not_count() {
    let model = |blocks: &str| {
        format!(
            "model m {{\n  kind analytical\n  parameters {{ ke, v }}\n  states {{ central }}\n  {blocks}\n  analytical {{ structure = one_compartment }}\n  outputs {{ cp = central / v }}\n}}\n"
        )
    };
    let error = compile_model(&model("routes {}"))
        .expect_err("an empty routes block gives doses nowhere to go")
        .to_string();
    assert!(error.contains("declares no routes"), "{error}");
    // An empty `init {}` is already rejected by the parser.
    assert!(compile_model(&model("init {}")).is_err());
}
