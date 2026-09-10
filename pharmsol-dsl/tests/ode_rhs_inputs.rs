use pharmsol_dsl::{analyze_model, parse_model, parse_module};

fn model(equation: &str) -> String {
    let gut = if equation.contains("dx(gut)") {
        ""
    } else {
        "dx(gut) = 0"
    };
    let central = if equation.contains("dx(central)") {
        ""
    } else {
        "dx(central) = 0"
    };
    format!("name = rhs_inputs\nkind = ode\nparams = f, v, ke\ncovariates = wt\nstates = gut, central\nscale = f * wt / 70\n{equation}\n{gut}\n{central}\nout(cp) = central / v\n")
}

#[test]
fn rhs_inputs_lower_to_existing_routes_and_rate() {
    let source = model("lag(oral) = 1\ndx(gut) = bolus(oral) * scale - ke * gut\ndx(central) = infusion(iv) * v + ke * gut");
    let parsed = parse_model(&source).unwrap();
    analyze_model(&parsed).unwrap();
    let normalized = parse_module(&source).unwrap().to_string();
    assert!(normalized.contains("rate(iv)"));
    assert!(normalized.contains("bioavailability"));
    assert!(!normalized.contains("bolus(oral) *"));
}

#[test]
fn brackets_and_derived_bolus_scales_are_supported() {
    let source =
        model("dx(gut) = scale * bolus[oral] * f - ke * gut\ndx(central) = infusion[iv] * v");
    analyze_model(&parse_model(&source).unwrap()).unwrap();
}

#[test]
fn shared_bolus_and_infusion_label_is_supported() {
    let source = model("dx(gut) = bolus(input_1) * scale\ndx(central) = infusion(input_1) * v");
    analyze_model(&parse_model(&source).unwrap()).unwrap();
}

#[test]
fn removed_ode_authoring_forms_fail_clearly() {
    for (equation, expected) in [
        ("bolus(oral) -> gut\ndx(gut) = 0", "ODE inputs belong"),
        (
            "infusion(iv) -> central\ndx(central) = 0",
            "ODE inputs belong",
        ),
        (
            "fa(oral) = f\ndx(gut) = bolus(oral)",
            "no longer accept `fa(input)`",
        ),
    ] {
        assert!(parse_model(&model(equation))
            .unwrap_err()
            .to_string()
            .contains(expected));
    }
}

#[test]
fn unsupported_bolus_uses_are_not_silently_changed() {
    for equation in [
        "dx(gut) = -bolus(oral)",
        "dx(gut) = 0 - bolus(oral)",
        "dx(gut) = bolus(oral)^2",
        "dx(gut) = bolus(oral) * bolus(oral)",
        "dx(gut) = bolus(oral)\ndx(central) = bolus(oral)",
        "dx(gut) = bolus(oral) + bolus(oral)",
    ] {
        assert!(parse_model(&model(equation)).is_err(), "{equation}");
    }
}

#[test]
fn infusion_rates_can_be_used_by_multiple_derivatives() {
    let source = model("dx(gut) = infusion(iv)\ndx(central) = infusion(iv) * time");
    let parsed = parse_module(&source).unwrap();
    let analyzed = analyze_model(&parsed.models[0]).unwrap();
    let compiled = pharmsol_dsl::compile_analyzed_model(&analyzed).unwrap();
    assert_eq!(compiled.metadata.routes.len(), 1);
    assert_eq!(parsed.to_string().matches("rate(iv)").count(), 2);
}

#[test]
fn conditional_infusions_do_not_get_an_extra_rate() {
    let source = model("dx(central) = if (f > 0) infusion(iv) * f else infusion(iv) * v");
    let parsed = parse_module(&source).unwrap();
    analyze_model(&parsed.models[0]).unwrap();
    assert_eq!(parsed.to_string().matches("rate(iv)").count(), 2);
}

#[test]
fn conditional_bolus_scaling_belongs_in_a_derived_value() {
    let source =
        model("oral_scale = if (wt > 70) f else f / 2\ndx(gut) = bolus(oral) * oral_scale");
    analyze_model(&parse_model(&source).unwrap()).unwrap();
    let unsupported = model("dx(gut) = if (f > 0) bolus(oral) else 0");
    assert!(parse_model(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("conditional bolus scaling"));
}

#[test]
fn bolus_scales_retain_event_safety_checks() {
    for equation in [
        "dx(gut) = bolus(oral) * central",
        "unsafe_scale = central * f\ndx(gut) = bolus(oral) * unsafe_scale",
        "unsafe_scale = rate(iv) * f\ndx(gut) = bolus(oral) * unsafe_scale\ndx(central) = infusion(iv)",
    ] {
        let parsed = parse_model(&model(equation)).unwrap();
        assert!(analyze_model(&parsed).is_err(), "{equation}");
    }
}
