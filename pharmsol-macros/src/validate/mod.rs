//! Compile-time validation of declaration-first macro input.

mod derive;
mod naming;
mod route_property;

pub(crate) use derive::{
    validate_analytical_derive_contract, validate_analytical_structure_inputs,
};
pub(crate) use naming::{
    validate_analytical_named_binding_compatibility, validate_named_binding_compatibility,
    validate_sde_named_binding_compatibility, AnalyticalBindingClosures, CommonBindingClosures,
    NamedBindingSets, OdeBindingClosures, SdeBindingClosures,
};
pub(crate) use route_property::{extract_route_property_routes, validate_route_property_kinds};

use std::collections::HashSet;
use syn::{ExprClosure, Ident};

use crate::analysis::{closure_param_ident, closure_param_names, ClosureBodyUsage};
use crate::symbols::{route_input_idents, OdeRouteDecl, SymbolicIndex};

pub(crate) fn validate_unique_idents(
    kind: &str,
    idents: &[Ident],
    macro_name: &str,
) -> syn::Result<()> {
    let mut seen = HashSet::new();
    for ident in idents {
        let name = ident.to_string();
        if !seen.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                ident,
                format!("duplicate {kind} `{name}` in declaration-first `{macro_name}`"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_unique_symbolic_indices(
    kind: &str,
    labels: &[SymbolicIndex],
    macro_name: &str,
) -> syn::Result<()> {
    let mut seen = HashSet::new();
    for label in labels {
        let name = label.name();
        if !seen.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                label,
                format!("duplicate {kind} `{name}` in declaration-first `{macro_name}`"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_routes(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    macro_name: &str,
) -> syn::Result<()> {
    let known_states = states.iter().map(Ident::to_string).collect::<HashSet<_>>();
    let mut seen_routes = HashSet::new();

    for route in routes {
        let route_name = route.input.name();
        // Route labels are unique per kind: a bolus and an infusion may share
        // a label (one drug given by either route) and resolve to separate
        // per-kind input slots, but two routes of the same kind may not.
        if !seen_routes.insert((route.kind, route_name.clone())) {
            return Err(syn::Error::new_spanned(
                &route.input,
                format!("duplicate route `{route_name}` in declaration-first `{macro_name}`"),
            ));
        }

        if !known_states.contains(&route.destination.to_string()) {
            return Err(syn::Error::new_spanned(
                &route.destination,
                format!(
                    "route destination `{}` is not declared in the `states` section",
                    route.destination
                ),
            ));
        }
    }

    Ok(())
}

pub(crate) fn validate_ode_diffeq_uses_automatic_injection(
    diffeq: &ExprClosure,
    routes: &[OdeRouteDecl],
) -> syn::Result<()> {
    match closure_param_names(diffeq).len() {
        3 => Ok(()),
        5 => {
            let usage = ClosureBodyUsage::analyze(diffeq.body.as_ref());
            let route_inputs = route_input_idents(routes);
            let fourth_param = closure_param_ident(diffeq, 3);
            let fifth_param = closure_param_ident(diffeq, 4);
            let mentions_route_inputs = route_inputs.iter().any(|route| usage.mentions(route));
            let indexes_fifth_param = fifth_param.as_ref().is_some_and(|ident| usage.indexes(ident));
            let reads_fourth_param_as_input = fourth_param
                .as_ref()
                .is_some_and(|ident| usage.indexes(ident) && !usage.assigns_index(ident));

            if mentions_route_inputs || indexes_fifth_param || reads_fourth_param_as_input {
                Err(syn::Error::new_spanned(
                    diffeq,
                    "declaration-first `ode!` only supports automatic route injection in `diffeq`; use either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx| and remove manual `bolus[...]` / `rateiv[...]` terms",
                ))
            } else {
                Ok(())
            }
        }
        _ => Err(syn::Error::new_spanned(
            diffeq,
            "declaration-first `ode!` only supports automatic route injection in `diffeq`; use either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|",
        )),
    }
}
