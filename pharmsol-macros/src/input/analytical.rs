//! `analytical!` input.

use proc_macro2::TokenStream as TokenStream2;
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream},
    ExprClosure, Ident, LitStr, Token,
};

use super::{
    missing_required_field, parse_ident_list, parse_route_list, parse_symbolic_index_list, set_once,
};
use crate::crate_path::{parse_crate_marker, resolve_crate_path};
use crate::kernel::resolve_analytical_structure;
use crate::symbols::{symbolic_index_idents, OdeRouteDecl, SymbolicIndex};
use crate::validate::{
    extract_route_property_routes, validate_analytical_derive_contract,
    validate_analytical_named_binding_compatibility, validate_analytical_structure_inputs,
    validate_route_property_kinds, validate_routes, validate_unique_idents,
    validate_unique_symbolic_indices, AnalyticalBindingClosures, CommonBindingClosures,
    NamedBindingSets,
};

const MACRO_LABEL: &str = "built-in `analytical!`";

pub(crate) struct AnalyticalInput {
    pub(crate) name: LitStr,
    pub(crate) krate: TokenStream2,
    pub(crate) params: Vec<Ident>,
    pub(crate) derived: Vec<Ident>,
    pub(crate) covariates: Vec<Ident>,
    pub(crate) states: Vec<Ident>,
    pub(crate) outputs: Vec<SymbolicIndex>,
    pub(crate) routes: Vec<OdeRouteDecl>,
    pub(crate) structure: Ident,
    pub(crate) derive: Option<ExprClosure>,
    pub(crate) lag: Option<ExprClosure>,
    pub(crate) fa: Option<ExprClosure>,
    pub(crate) init: Option<ExprClosure>,
    pub(crate) out: ExprClosure,
}

impl Parse for AnalyticalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forwarded_krate = parse_crate_marker(input)?;
        let mut name = None;
        let mut krate = None;
        let mut params = None;
        let mut derived = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut structure = None;
        let mut derive = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.call(Ident::parse_any)?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once(&mut name, input.parse()?, &key, "name", "analytical!")?,
                "crate" => set_once(
                    &mut krate,
                    input.parse::<LitStr>()?,
                    &key,
                    "crate",
                    "analytical!",
                )?,
                "params" => set_once(
                    &mut params,
                    parse_ident_list(input)?,
                    &key,
                    "params",
                    "analytical!",
                )?,
                "derived" => set_once(
                    &mut derived,
                    parse_ident_list(input)?,
                    &key,
                    "derived",
                    "analytical!",
                )?,
                "covariates" => set_once(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                    "analytical!",
                )?,
                "states" => set_once(
                    &mut states,
                    parse_ident_list(input)?,
                    &key,
                    "states",
                    "analytical!",
                )?,
                "outputs" => set_once(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                    "analytical!",
                )?,
                "routes" => set_once(
                    &mut routes,
                    parse_route_list(input)?,
                    &key,
                    "routes",
                    "analytical!",
                )?,
                "structure" => set_once(
                    &mut structure,
                    input.parse()?,
                    &key,
                    "structure",
                    "analytical!",
                )?,
                "derive" => set_once(&mut derive, input.parse()?, &key, "derive", "analytical!")?,
                "sec" => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        "built-in `analytical!` no longer supports `sec`; use `derived: [...]` plus `derive: ...`",
                    ));
                }
                "lag" => set_once(&mut lag, input.parse()?, &key, "lag", "analytical!")?,
                "fa" => set_once(&mut fa, input.parse()?, &key, "fa", "analytical!")?,
                "init" => set_once(&mut init, input.parse()?, &key, "init", "analytical!")?,
                "out" => set_once(&mut out, input.parse()?, &key, "out", "analytical!")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, crate, params, derived, covariates, states, outputs, routes, structure, derive, lag, fa, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| missing_required_field("name", MACRO_LABEL))?;
        let krate = resolve_crate_path(krate, forwarded_krate)?;
        let params = params.ok_or_else(|| missing_required_field("params", MACRO_LABEL))?;
        let derived = derived.unwrap_or_default();
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_field("states", MACRO_LABEL))?;
        let outputs = outputs.ok_or_else(|| missing_required_field("outputs", MACRO_LABEL))?;
        let routes = routes.ok_or_else(|| missing_required_field("routes", MACRO_LABEL))?;
        let structure =
            structure.ok_or_else(|| missing_required_field("structure", MACRO_LABEL))?;
        let out = out.ok_or_else(|| missing_required_field("out", MACRO_LABEL))?;

        validate_unique_idents("covariate", &covariates, "analytical!")?;
        validate_unique_idents("state", &states, "analytical!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "analytical!")?;
        validate_routes(&routes, &states, "analytical!")?;

        let function_spec = resolve_analytical_structure(&structure)?;
        validate_analytical_structure_inputs(
            &structure,
            function_spec.function,
            &params,
            &derived,
        )?;
        if states.len() != function_spec.state_count {
            return Err(syn::Error::new_spanned(
                &structure,
                format!(
                    "analytical structure `{}` expects {} state value(s), but `states` declares {}",
                    structure,
                    function_spec.state_count,
                    states.len()
                ),
            ));
        }

        validate_analytical_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &derived,
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            AnalyticalBindingClosures {
                derive: derive.as_ref(),
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        validate_analytical_derive_contract(
            function_spec.function,
            &params,
            &derived,
            &covariates,
            derive.as_ref(),
        )?;

        if let Some(lag) = lag.as_ref() {
            let lag_routes = extract_route_property_routes(MACRO_LABEL, "lag", lag, &routes)?;
            validate_route_property_kinds(MACRO_LABEL, "lag", &routes, &lag_routes)?;
        }

        if let Some(fa) = fa.as_ref() {
            let fa_routes = extract_route_property_routes(MACRO_LABEL, "fa", fa, &routes)?;
            validate_route_property_kinds(MACRO_LABEL, "fa", &routes, &fa_routes)?;
        }

        Ok(Self {
            name,
            krate,
            params,
            derived,
            covariates,
            states,
            outputs,
            routes,
            structure,
            derive,
            lag,
            fa,
            init,
            out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_accepts_extra_parameters_beyond_kernel_arity() {
        let input = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v, tlag, tvke], derived: [ke], covariates: [wt, renal], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = tvke; }, out: |x, p, t, cov, y| {}",
        )
        .expect("extra declared parameters should be allowed");

        assert_eq!(input.params.len(), 5);
        assert_eq!(input.derived.len(), 1);
        assert_eq!(input.covariates.len(), 2);
        assert!(input.derive.is_some());
        assert_eq!(input.states.len(), 2);
    }

    #[test]
    fn analytical_rejects_legacy_sec_with_migration_message() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, sec: |_t| { ke = 1.0; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("legacy sec must fail");

        assert!(error
            .to_string()
            .contains("no longer supports `sec`; use `derived: [...]` plus `derive: ...`"));
    }

    #[test]
    fn analytical_allows_shared_label_across_bolus_and_infusion_routes() {
        let input = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(input_1) -> gut, infusion(input_1) -> central], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .expect("bolus and infusion sharing a label must parse");

        assert_eq!(input.routes.len(), 2);
    }

    #[test]
    fn analytical_rejects_shared_label_within_same_kind() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(input_1) -> gut, bolus(input_1) -> central], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("duplicate bolus routes must fail");

        assert!(error
            .to_string()
            .contains("duplicate route `input_1` in declaration-first `analytical!`"));
    }

    #[test]
    fn analytical_rejects_unknown_structure() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], structure: mystery, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown analytical structure must fail");

        assert!(error
            .to_string()
            .contains("unknown analytical structure `mystery`"));
    }

    #[test]
    fn analytical_rejects_missing_required_structure_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing required structure name must fail");

        assert!(error.to_string().contains("requires `ka`"));
    }

    #[test]
    fn analytical_rejects_overlap_between_params_and_derived() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = 1.0; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("overlap must fail");

        assert!(error
            .to_string()
            .contains("`ke` is declared in both `params` and `derived`"));
    }

    #[test]
    fn analytical_rejects_invalid_derive_target() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke0 = 1.0; ke = 0.1; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("invalid derive target must fail");

        assert!(error
            .to_string()
            .contains("`derive` cannot assign to `ke0`"));
    }

    #[test]
    fn analytical_rejects_if_only_assignment_for_required_derived_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], covariates: [wt], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { if wt > 70.0 { ke = ke0; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("bare if must fail");

        assert!(error
            .to_string()
            .contains("not definitely assigned on every path"));
    }

    #[test]
    fn analytical_accepts_if_else_assignment_for_required_derived_name() {
        syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], covariates: [wt], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { if wt > 70.0 { ke = ke0; } else { ke = ke0 * 0.5; } }, out: |x, p, t, cov, y| {}",
        )
        .expect("if / else should establish derived assignment");
    }

    #[test]
    fn analytical_rejects_loop_only_assignment_for_required_derived_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { for i in 0..1 { let _ = i; ke = ke0; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("loop-only assignment must fail");

        assert!(error
            .to_string()
            .contains("not definitely assigned on every path"));
    }

    #[test]
    fn analytical_accepts_initial_assignment_followed_by_loop_updates() {
        syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = ke0; for i in 0..2 { let _ = i; ke = ke + 1.0; } }, out: |x, p, t, cov, y| {}",
        )
        .expect("initial assignment plus loop updates should pass");
    }

    #[test]
    fn analytical_rejects_invalid_derive_target_in_assignment_rhs() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = { ke0 = 1.0; 0.1 }; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("invalid assignment in the right-hand side must fail");

        assert!(error
            .to_string()
            .contains("`derive` cannot assign to `ke0`"));
    }

    #[test]
    fn analytical_rejects_invalid_derive_target_in_if_condition() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { if { ke0 = 1.0; true } { ke = 0.1; } else { ke = 0.2; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("invalid assignment in an `if` condition must fail");

        assert!(error
            .to_string()
            .contains("`derive` cannot assign to `ke0`"));
    }

    #[test]
    fn analytical_rejects_invalid_derive_target_in_for_iterator() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = 0.1; for i in { ke0 = 1.0; 0..2 } { let _ = i; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("invalid assignment in a `for` iterator must fail");

        assert!(error
            .to_string()
            .contains("`derive` cannot assign to `ke0`"));
    }

    #[test]
    fn analytical_accepts_derived_assignment_in_assignment_rhs() {
        syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke, kb], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = { kb = ke0; kb * 2.0 }; }, out: |x, p, t, cov, y| {}",
        )
        .expect("assignments in the right-hand side should count as definite");
    }

    #[test]
    fn analytical_rejects_unknown_route_property_binding() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, lag: |_p, _t, _cov| { lag! { iv => 1.0 } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown lag route must fail");

        assert!(error
            .to_string()
            .contains("route `iv` in `lag!` is not declared in the `routes` section"));
    }

    #[test]
    fn analytical_rejects_infusion_lag_binding() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke, v, tlag], states: [central], outputs: [cp], routes: [infusion(iv) -> central], structure: one_compartment, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("infusion lag must fail");

        assert!(error
            .to_string()
            .contains("built-in `analytical!` does not allow `lag` on infusion route `iv`"));
    }
}
