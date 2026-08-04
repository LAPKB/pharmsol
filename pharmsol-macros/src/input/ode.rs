//! `ode!` input.

use proc_macro2::{Span, TokenStream as TokenStream2};
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream},
    ExprClosure, Ident, LitStr, Token,
};

use super::{
    missing_required_field, parse_ident_list, parse_route_list, parse_symbolic_index_list, set_once,
};
use crate::crate_path::{parse_crate_marker, resolve_crate_path};
use crate::symbols::{symbolic_index_idents, OdeRouteDecl, SymbolicIndex};
use crate::validate::{
    validate_named_binding_compatibility, validate_ode_diffeq_uses_automatic_injection,
    validate_routes, validate_unique_idents, validate_unique_symbolic_indices,
    CommonBindingClosures, NamedBindingSets, OdeBindingClosures,
};

const MACRO_LABEL: &str = "declaration-first `ode!`";

pub(crate) struct OdeInput {
    pub(crate) name: LitStr,
    pub(crate) krate: TokenStream2,
    pub(crate) params: Vec<Ident>,
    pub(crate) covariates: Vec<Ident>,
    pub(crate) states: Vec<Ident>,
    pub(crate) outputs: Vec<SymbolicIndex>,
    pub(crate) routes: Vec<OdeRouteDecl>,
    pub(crate) diffeq: ExprClosure,
    pub(crate) lag: Option<ExprClosure>,
    pub(crate) fa: Option<ExprClosure>,
    pub(crate) init: Option<ExprClosure>,
    pub(crate) out: ExprClosure,
}

impl Parse for OdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forwarded_krate = parse_crate_marker(input)?;
        let mut name = None;
        let mut krate = None;
        let mut params = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut diffeq = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.call(Ident::parse_any)?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once(&mut name, input.parse()?, &key, "name", "ode!")?,
                "crate" => set_once(&mut krate, input.parse::<LitStr>()?, &key, "crate", "ode!")?,
                "params" => set_once(
                    &mut params,
                    parse_ident_list(input)?,
                    &key,
                    "params",
                    "ode!",
                )?,
                "covariates" => set_once(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                    "ode!",
                )?,
                "states" => set_once(
                    &mut states,
                    parse_ident_list(input)?,
                    &key,
                    "states",
                    "ode!",
                )?,
                "outputs" => set_once(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                    "ode!",
                )?,
                "routes" => set_once(
                    &mut routes,
                    parse_route_list(input)?,
                    &key,
                    "routes",
                    "ode!",
                )?,
                "diffeq" => set_once(&mut diffeq, input.parse()?, &key, "diffeq", "ode!")?,
                "lag" => set_once(&mut lag, input.parse()?, &key, "lag", "ode!")?,
                "fa" => set_once(&mut fa, input.parse()?, &key, "fa", "ode!")?,
                "init" => set_once(&mut init, input.parse()?, &key, "init", "ode!")?,
                "out" => set_once(&mut out, input.parse()?, &key, "out", "ode!")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, crate, params, covariates, states, outputs, routes, diffeq, lag, fa, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "declaration-first `ode!` requires `name`, `params`, `states`, `outputs`, and `routes`; the old inferred-dimensions form has been removed",
            )
        })?;
        let krate = resolve_crate_path(krate, forwarded_krate)?;
        let params = params.ok_or_else(|| missing_required_field("params", MACRO_LABEL))?;
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_field("states", MACRO_LABEL))?;
        let outputs = outputs.ok_or_else(|| missing_required_field("outputs", MACRO_LABEL))?;
        let routes = routes.ok_or_else(|| missing_required_field("routes", MACRO_LABEL))?;
        let diffeq = diffeq.ok_or_else(|| missing_required_field("diffeq", MACRO_LABEL))?;
        let out = out.ok_or_else(|| missing_required_field("out", MACRO_LABEL))?;
        validate_ode_diffeq_uses_automatic_injection(&diffeq, &routes)?;

        validate_unique_idents("parameter", &params, "ode!")?;
        validate_unique_idents("covariate", &covariates, "ode!")?;
        validate_unique_idents("state", &states, "ode!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "ode!")?;
        validate_routes(&routes, &states, "ode!")?;
        validate_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &[],
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            OdeBindingClosures {
                diffeq: &diffeq,
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        Ok(Self {
            name,
            krate,
            params,
            covariates,
            states,
            outputs,
            routes,
            diffeq,
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
    use crate::symbols::{dense_index_len, ode_route_input_bindings};

    #[test]
    fn crate_key_overrides_the_emitted_path() {
        let input = syn::parse_str::<OdeInput>(
            "name: \"demo\", crate: \"pmcore::pharmsol\", params: [ke], covariates: [wt], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| { dx[central] = -ke * x[central] * wt; }, out: |x, p, t, cov, y| { y[cp] = x[central]; }",
        )
        .expect("`crate` key must parse");

        assert_eq!(input.krate.to_string(), ":: pmcore :: pharmsol");
    }

    #[test]
    fn forwarded_marker_sets_the_emitted_path() {
        let input = syn::parse_str::<OdeInput>(
            "@pharmsol_crate(::reexporter::pharmsol) name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("forwarded marker must parse");

        assert_eq!(input.krate.to_string(), ":: reexporter :: pharmsol");
    }

    #[test]
    fn crate_key_wins_over_the_forwarded_marker() {
        let input = syn::parse_str::<OdeInput>(
            "@pharmsol_crate(::reexporter::pharmsol) name: \"demo\", crate: \"my_vendor::pharmsol\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("both crate sources must parse");

        assert_eq!(input.krate.to_string(), ":: my_vendor :: pharmsol");
    }

    #[test]
    fn crate_key_rejects_generic_arguments() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", crate: \"pmcore::pharmsol<T>\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("generic arguments must fail");

        assert!(error
            .to_string()
            .contains("plain module path without generic arguments"));
    }

    #[test]
    fn rejects_removed_legacy_form() {
        let error = syn::parse_str::<OdeInput>(
            "diffeq: |x, p, t, dx, b, rateiv, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("legacy macro form must fail");

        assert!(error
            .to_string()
            .contains("requires `name`, `params`, `states`, `outputs`, and `routes`"));
        assert!(error
            .to_string()
            .contains("old inferred-dimensions form has been removed"));
    }

    #[test]
    fn validates_route_destinations() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> peripheral], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown route destination must fail");

        assert!(error
            .to_string()
            .contains("route destination `peripheral` is not declared in the `states` section"));
    }

    #[test]
    fn rejects_named_binding_collisions() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [central, v], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("parameter/state binding collisions must fail");

        assert!(error
            .to_string()
            .contains("named parameter binding `central` conflicts with named state binding"));
    }

    #[test]
    fn ode_route_bindings_share_inputs_by_kind_local_ordinal() {
        let input = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ka, ke, v], states: [depot, central], outputs: [cp], routes: [bolus(oral) -> depot, infusion(iv) -> central, bolus(sc) -> depot], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("declaration-first ode input should parse");

        let bindings = ode_route_input_bindings(&input.routes);

        assert_eq!(dense_index_len(&bindings), 2);
        assert_eq!(bindings[0].0.name(), "oral");
        assert_eq!(bindings[0].1, 0);
        assert_eq!(bindings[1].0.name(), "iv");
        assert_eq!(bindings[1].1, 0);
        assert_eq!(bindings[2].0.name(), "sc");
        assert_eq!(bindings[2].1, 1);
    }

    #[test]
    fn rejects_braced_route_lists() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("braced route lists must fail");

        assert!(error
            .to_string()
            .contains("declaration-first macro `routes` must use `[...]`, not `{...}`"));
    }
}
