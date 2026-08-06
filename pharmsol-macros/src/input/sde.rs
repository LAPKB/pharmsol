//! `sde!` input.

use proc_macro2::TokenStream as TokenStream2;
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream},
    Expr, ExprClosure, Ident, LitStr, Token,
};

use super::{
    missing_required_field, parse_ident_list, parse_route_list, parse_symbolic_index_list, set_once,
};
use crate::crate_path::{parse_crate_marker, resolve_crate_path};
use crate::symbols::{symbolic_index_idents, OdeRouteDecl, SymbolicIndex};
use crate::validate::{
    extract_route_property_routes, validate_route_property_kinds, validate_routes,
    validate_sde_named_binding_compatibility, validate_unique_idents,
    validate_unique_symbolic_indices, CommonBindingClosures, NamedBindingSets, SdeBindingClosures,
};

const MACRO_LABEL: &str = "declaration-first `sde!`";

pub(crate) struct SdeInput {
    pub(crate) name: LitStr,
    pub(crate) krate: TokenStream2,
    pub(crate) params: Vec<Ident>,
    pub(crate) covariates: Vec<Ident>,
    pub(crate) states: Vec<Ident>,
    pub(crate) outputs: Vec<SymbolicIndex>,
    pub(crate) routes: Vec<OdeRouteDecl>,
    pub(crate) particles: Expr,
    pub(crate) drift: ExprClosure,
    pub(crate) diffusion: ExprClosure,
    pub(crate) lag: Option<ExprClosure>,
    pub(crate) fa: Option<ExprClosure>,
    pub(crate) init: Option<ExprClosure>,
    pub(crate) out: ExprClosure,
}

impl Parse for SdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forwarded_krate = parse_crate_marker(input)?;
        let mut name = None;
        let mut krate = None;
        let mut params = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut particles = None;
        let mut drift = None;
        let mut diffusion = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.call(Ident::parse_any)?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once(&mut name, input.parse()?, &key, "name", "sde!")?,
                "crate" => set_once(&mut krate, input.parse::<LitStr>()?, &key, "crate", "sde!")?,
                "params" => set_once(
                    &mut params,
                    parse_ident_list(input)?,
                    &key,
                    "params",
                    "sde!",
                )?,
                "covariates" => set_once(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                    "sde!",
                )?,
                "states" => set_once(
                    &mut states,
                    parse_ident_list(input)?,
                    &key,
                    "states",
                    "sde!",
                )?,
                "outputs" => set_once(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                    "sde!",
                )?,
                "routes" => set_once(
                    &mut routes,
                    parse_route_list(input)?,
                    &key,
                    "routes",
                    "sde!",
                )?,
                "particles" => set_once(&mut particles, input.parse()?, &key, "particles", "sde!")?,
                "drift" => set_once(&mut drift, input.parse()?, &key, "drift", "sde!")?,
                "diffusion" => set_once(&mut diffusion, input.parse()?, &key, "diffusion", "sde!")?,
                "lag" => set_once(&mut lag, input.parse()?, &key, "lag", "sde!")?,
                "fa" => set_once(&mut fa, input.parse()?, &key, "fa", "sde!")?,
                "init" => set_once(&mut init, input.parse()?, &key, "init", "sde!")?,
                "out" => set_once(&mut out, input.parse()?, &key, "out", "sde!")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, crate, params, covariates, states, outputs, routes, particles, drift, diffusion, lag, fa, init, out"
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
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_field("states", MACRO_LABEL))?;
        let outputs = outputs.ok_or_else(|| missing_required_field("outputs", MACRO_LABEL))?;
        let routes = routes.ok_or_else(|| missing_required_field("routes", MACRO_LABEL))?;
        let particles =
            particles.ok_or_else(|| missing_required_field("particles", MACRO_LABEL))?;
        let drift = drift.ok_or_else(|| missing_required_field("drift", MACRO_LABEL))?;
        let diffusion =
            diffusion.ok_or_else(|| missing_required_field("diffusion", MACRO_LABEL))?;
        let out = out.ok_or_else(|| missing_required_field("out", MACRO_LABEL))?;

        validate_unique_idents("parameter", &params, "sde!")?;
        validate_unique_idents("covariate", &covariates, "sde!")?;
        validate_unique_idents("state", &states, "sde!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "sde!")?;
        validate_routes(&routes, &states, "sde!")?;
        validate_sde_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &[],
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            SdeBindingClosures {
                drift: &drift,
                diffusion: &diffusion,
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
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
            covariates,
            states,
            outputs,
            routes,
            particles,
            drift,
            diffusion,
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
    fn sde_requires_particles() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, theta], states: [central], outputs: [cp], routes: [infusion(iv) -> central], drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing particles must fail");

        assert!(error
            .to_string()
            .contains("missing required field `particles` in declaration-first `sde!`"));
    }

    #[test]
    fn sde_rejects_unknown_route_property_binding() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, sigma_ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { oral => 1.0 } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown lag route must fail");

        assert!(error
            .to_string()
            .contains("route `oral` in `lag!` is not declared in the `routes` section"));
    }

    #[test]
    fn sde_rejects_infusion_lag_binding() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, sigma_ke, tlag], states: [central], outputs: [cp], routes: [infusion(iv) -> central], particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("infusion lag must fail");

        assert!(error
            .to_string()
            .contains("declaration-first `sde!` does not allow `lag` on infusion route `iv`"));
    }

    #[test]
    fn sde_allows_shared_label_across_bolus_and_infusion_routes() {
        let input = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, v], states: [central], outputs: [cp], particles: 16, routes: [bolus(input_1) -> central, infusion(input_1) -> central], drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("bolus and infusion sharing a label must parse");

        assert_eq!(input.routes.len(), 2);
    }

    #[test]
    fn sde_rejects_shared_label_within_same_kind() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, v], states: [central], outputs: [cp], particles: 16, routes: [bolus(input_1) -> central, bolus(input_1) -> central], drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("duplicate bolus routes must fail");

        assert!(error
            .to_string()
            .contains("duplicate route `input_1` in declaration-first `sde!`"));
    }
}
