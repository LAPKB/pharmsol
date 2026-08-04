//! `sde!` expansion.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashSet;
use syn::{ExprClosure, Ident};

use super::{covariate_metadata, empty_init, empty_route_map, route_destination_index};
use crate::analysis::{
    closure_param_ident, generated_ident, IndexRewriteTarget, NumericLabelRewriter,
};
use crate::bindings::{
    generate_covariate_bindings, generate_index_consts, generate_mapped_index_consts,
    generate_parameter_bindings, generate_supported_input_aliases,
};
use crate::crate_path::rewrite_crate_paths;
use crate::input::SdeInput;
use crate::symbols::{
    dense_index_len, ode_route_input_bindings, symbolic_index_bindings,
    symbolic_numeric_binding_map, OdeRouteDecl, OdeRouteKind, SymbolicIndex,
};
use crate::validate::{extract_route_property_routes, validate_route_property_kinds};

const MACRO_LABEL: &str = "declaration-first `sde!`";

pub(crate) fn expand(input: SdeInput) -> syn::Result<TokenStream2> {
    let route_bindings = ode_route_input_bindings(&input.routes);

    let lag_routes = route_property_routes(input.lag.as_ref(), "lag", &input.routes)?;
    let fa_routes = route_property_routes(input.fa.as_ref(), "fa", &input.routes)?;

    let drift = expand_drift(
        &input.drift,
        &input.params,
        &input.covariates,
        &input.states,
        &input.routes,
        &route_bindings,
    )?;

    let diffusion = expand_diffusion(&input.diffusion, &input.params, &input.states)?;

    let lag = match input.lag.as_ref() {
        Some(closure) => expand_route_map(
            "lag",
            closure,
            &input.params,
            &input.covariates,
            &route_bindings,
        )?,
        None => empty_route_map(),
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => expand_route_map(
            "fa",
            closure,
            &input.params,
            &input.covariates,
            &route_bindings,
        )?,
        None => empty_route_map(),
    };

    let init = match input.init.as_ref() {
        Some(closure) => expand_init(closure, &input.params, &input.covariates, &input.states)?,
        None => empty_init(),
    };

    let out = expand_out(
        &input.out,
        &input.params,
        &input.covariates,
        &input.states,
        &input.outputs,
    )?;

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let states = &input.states;
    let outputs = &input.outputs;
    let particles = &input.particles;
    let routes = expand_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let bolus_mappings =
        expand_injected_bolus_mappings(&input.routes, &input.states, &route_bindings);
    let covariates = covariate_metadata(&input.covariates);

    let expanded = quote! {{
        let __pharmsol_particles: usize = #particles;
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Sde)
            .parameters([#(stringify!(#params)),*])
            #covariates
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .particles(__pharmsol_particles);

        ::pharmsol::equation::SDE::new(
            #drift,
            #diffusion,
            #lag,
            #fa,
            #init,
            #out,
            __pharmsol_particles,
        )
        .with_nstates(#nstates)
        .with_ndrugs(#ndrugs)
        .with_nout(#nout)
        #bolus_mappings
        .with_metadata(__pharmsol_metadata)
        .expect("declaration-first `sde!` generated invalid metadata")
    }};

    Ok(rewrite_crate_paths(expanded, &input.krate))
}

fn route_property_routes(
    closure: Option<&ExprClosure>,
    label: &str,
    routes: &[OdeRouteDecl],
) -> syn::Result<HashSet<String>> {
    let Some(closure) = closure else {
        return Ok(HashSet::new());
    };

    let property_routes = extract_route_property_routes(MACRO_LABEL, label, closure, routes)?;
    validate_route_property_kinds(MACRO_LABEL, label, routes, &property_routes)?;
    Ok(property_routes)
}

fn expand_drift(
    drift: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    states: &[Ident],
    routes: &[OdeRouteDecl],
    route_bindings: &[(SymbolicIndex, usize)],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let dx = generated_ident("__pharmsol_dx");
    let rateiv = generated_ident("__pharmsol_rateiv");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [x.clone(), p.clone(), t.clone(), dx.clone(), cov.clone()];
    let reduced_inputs = [x.clone(), t.clone(), dx.clone()];
    let input_aliases = generate_supported_input_aliases(
        drift,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `drift` to have either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, drift, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, drift, &cov, &t);
    let body = &drift.body;
    let dx_binding = if drift.inputs.len() == full_inputs.len() {
        closure_param_ident(drift, 3).unwrap_or_else(|| dx.clone())
    } else {
        closure_param_ident(drift, 2).unwrap_or_else(|| dx.clone())
    };
    let rate_terms =
        expand_injected_rate_terms(routes, states, route_bindings, &dx_binding, &rateiv);

    Ok(quote! {{
        let __pharmsol_drift: fn(
            &::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            f64,
            &mut ::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            &::pharmsol::data::Covariates,
        ) = |#x: &::pharmsol::simulator::V,
             #p: &::pharmsol::simulator::V,
             #t: f64,
             #dx: &mut ::pharmsol::simulator::V,
             #rateiv: &::pharmsol::simulator::V,
             #cov: &::pharmsol::data::Covariates| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #covariate_bindings
            #body
            #rate_terms
        };
        __pharmsol_drift
    }})
}

fn expand_diffusion(
    diffusion: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let sigma = generated_ident("__pharmsol_sigma");
    let full_inputs = [p.clone(), sigma.clone()];
    let reduced_inputs = [sigma.clone()];
    let input_aliases = generate_supported_input_aliases(
        diffusion,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `diffusion` to have either 2 parameters: |p, sigma| or 1 parameter: |sigma|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, diffusion, &p);
    let body = &diffusion.body;

    Ok(quote! {{
        let __pharmsol_diffusion: fn(
            &::pharmsol::simulator::V,
            &mut ::pharmsol::simulator::V,
        ) = |#p: &::pharmsol::simulator::V,
             #sigma: &mut ::pharmsol::simulator::V| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #body
        };
        __pharmsol_diffusion
    }})
}

fn expand_route_map(
    label: &str,
    closure: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
) -> syn::Result<TokenStream2> {
    let route_consts = generate_mapped_index_consts(route_bindings);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [p.clone(), t.clone(), cov.clone()];
    let reduced_inputs = [t.clone()];
    let input_aliases = generate_supported_input_aliases(
        closure,
        &[&full_inputs, &reduced_inputs],
        &format!(
            "declaration-first `sde!` requires `{label}` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|"
        ),
    )?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, closure, &cov, &t);
    let body = NumericLabelRewriter::rewrite(
        closure.body.as_ref(),
        Vec::new(),
        Some(symbolic_numeric_binding_map(route_bindings)),
    );

    Ok(quote! {{
        let __pharmsol_route_map: fn(
            &::pharmsol::simulator::V,
            f64,
            &::pharmsol::data::Covariates,
        ) -> ::std::collections::HashMap<usize, f64> = |#p: &::pharmsol::simulator::V,
             #t: f64,
             #cov: &::pharmsol::data::Covariates| {
            #input_aliases
            #route_consts
            #parameter_bindings
            #covariate_bindings
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_init(
    init: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    states: &[Ident],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let x = generated_ident("__pharmsol_x");
    let full_inputs = [p.clone(), t.clone(), cov.clone(), x.clone()];
    let reduced_inputs = [t.clone(), x.clone()];
    let input_aliases = generate_supported_input_aliases(
        init,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `init` to have either 4 parameters: |p, t, cov, x| or 2 parameters: |t, x|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, init, &cov, &t);
    let body = &init.body;

    Ok(quote! {{
        let __pharmsol_init: fn(
            &::pharmsol::simulator::V,
            f64,
            &::pharmsol::data::Covariates,
            &mut ::pharmsol::simulator::V,
        ) = |#p: &::pharmsol::simulator::V,
             #t: f64,
             #cov: &::pharmsol::data::Covariates,
             #x: &mut ::pharmsol::simulator::V| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #covariate_bindings
            #body
        };
        __pharmsol_init
    }})
}

fn expand_out(
    out: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    states: &[Ident],
    outputs: &[SymbolicIndex],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let output_bindings = symbolic_index_bindings(outputs);
    let output_consts = generate_mapped_index_consts(&output_bindings);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let y = generated_ident("__pharmsol_y");
    let full_inputs = [x.clone(), p.clone(), t.clone(), cov.clone(), y.clone()];
    let reduced_inputs = [x.clone(), t.clone(), y.clone()];
    let input_aliases = generate_supported_input_aliases(
        out,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `out` to have either 5 parameters: |x, p, t, cov, y| or 3 parameters: |x, t, y|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, out, &cov, &t);
    let y_binding = if out.inputs.len() == full_inputs.len() {
        closure_param_ident(out, 4).unwrap_or_else(|| y.clone())
    } else {
        closure_param_ident(out, 2).unwrap_or_else(|| y.clone())
    };
    let body = NumericLabelRewriter::rewrite(
        out.body.as_ref(),
        vec![IndexRewriteTarget::new(
            y_binding,
            symbolic_numeric_binding_map(&output_bindings),
        )],
        None,
    );

    Ok(quote! {{
        let __pharmsol_out: fn(
            &::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            f64,
            &::pharmsol::data::Covariates,
            &mut ::pharmsol::simulator::V,
        ) = |#x: &::pharmsol::simulator::V,
             #p: &::pharmsol::simulator::V,
             #t: f64,
             #cov: &::pharmsol::data::Covariates,
             #y: &mut ::pharmsol::simulator::V| {
            #input_aliases
            #state_consts
            #output_consts
            #parameter_bindings
            #covariate_bindings
            #body
        };
        __pharmsol_out
    }})
}

fn expand_route_metadata(
    routes: &[OdeRouteDecl],
    lag_routes: &HashSet<String>,
    fa_routes: &HashSet<String>,
) -> Vec<TokenStream2> {
    routes
        .iter()
        .map(|route| {
            let input = &route.input;
            let destination = &route.destination;
            let route_name = route.input.name();
            let route_builder = match route.kind {
                OdeRouteKind::Bolus => {
                    quote! { ::pharmsol::equation::Route::bolus(stringify!(#input)) }
                }
                OdeRouteKind::Infusion => {
                    quote! { ::pharmsol::equation::Route::infusion(stringify!(#input)) }
                }
            };
            let lag_flag = if lag_routes.contains(&route_name) {
                quote! { .with_lag() }
            } else {
                quote! {}
            };
            let fa_flag = if fa_routes.contains(&route_name) {
                quote! { .with_bioavailability() }
            } else {
                quote! {}
            };

            quote! {
                #route_builder
                    .to_state(stringify!(#destination))
                    .inject_input_to_destination()
                    #lag_flag
                    #fa_flag
            }
        })
        .collect()
}

fn expand_injected_rate_terms(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
    dx: &Ident,
    rateiv: &Ident,
) -> TokenStream2 {
    let terms = routes
        .iter()
        .zip(route_bindings.iter())
        .filter_map(|(route, (_, input_index))| match route.kind {
            OdeRouteKind::Bolus => None,
            OdeRouteKind::Infusion => {
                let destination = route_destination_index(route, states);
                Some(quote! {
                    #dx[#destination] += #rateiv[#input_index];
                })
            }
        });

    quote! {
        #(#terms)*
    }
}

fn expand_injected_bolus_mappings(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
) -> TokenStream2 {
    let mut destinations = vec![quote! { None }; dense_index_len(route_bindings)];

    for (route, (_, input_index)) in routes.iter().zip(route_bindings.iter()) {
        if let OdeRouteKind::Bolus = route.kind {
            let destination = route_destination_index(route, states);
            destinations[*input_index] = quote! { Some(#destination) };
        }
    }

    quote! {
        .with_injected_bolus_inputs(&[#(#destinations),*])
    }
}
