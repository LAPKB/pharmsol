//! `analytical!` expansion.

use pharmsol_dsl::{AnalyticalStructureInputKind, AnalyticalStructureInputSource};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::HashSet;
use syn::{ExprClosure, Ident};

use super::{covariate_metadata, empty_init, empty_route_map};
use crate::analysis::{
    closure_param_ident, generated_ident, IndexRewriteTarget, NumericLabelRewriter,
};
use crate::bindings::{
    generate_covariate_bindings, generate_derived_bindings, generate_index_consts,
    generate_mapped_index_consts, generate_parameter_bindings, generate_supported_input_aliases,
};
use crate::crate_path::rewrite_crate_paths;
use crate::input::AnalyticalInput;
use crate::kernel::resolve_analytical_structure;
use crate::symbols::{
    bolus_route_input_bindings, dense_index_len, ode_route_input_bindings, symbolic_index_bindings,
    symbolic_numeric_binding_map, OdeRouteDecl, OdeRouteKind, SymbolicIndex,
};
use crate::validate::{
    extract_route_property_routes, validate_analytical_structure_inputs,
    validate_route_property_kinds,
};

const MACRO_LABEL: &str = "built-in `analytical!`";

pub(crate) fn expand(input: AnalyticalInput) -> syn::Result<TokenStream2> {
    let route_bindings = ode_route_input_bindings(&input.routes);
    let bolus_route_bindings = bolus_route_input_bindings(&input.routes);
    let function_spec = resolve_analytical_structure(&input.structure)?;
    let projection = validate_analytical_structure_inputs(
        &input.structure,
        function_spec.function,
        &input.params,
        &input.derived,
    )?;

    let lag_routes = route_property_routes(input.lag.as_ref(), "lag", &input.routes)?;
    let fa_routes = route_property_routes(input.fa.as_ref(), "fa", &input.routes)?;

    let derive = expand_derive(
        input.derive.as_ref(),
        &input.params,
        &input.covariates,
        &input.derived,
    )?;
    let eq = expand_runtime(&function_spec.runtime_path, projection.kind());

    let out = expand_out(
        &input.out,
        &input.params,
        &input.derived,
        &input.covariates,
        &input.states,
        &input.outputs,
    )?;

    let lag = match input.lag.as_ref() {
        Some(closure) => expand_route_map(
            "lag",
            closure,
            &input.params,
            &input.derived,
            &input.covariates,
            &bolus_route_bindings,
        )?,
        None => empty_route_map(),
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => expand_route_map(
            "fa",
            closure,
            &input.params,
            &input.derived,
            &input.covariates,
            &bolus_route_bindings,
        )?,
        None => empty_route_map(),
    };

    let init = match input.init.as_ref() {
        Some(closure) => expand_init(
            closure,
            &input.params,
            &input.derived,
            &input.covariates,
            &input.states,
        )?,
        None => empty_init(),
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();
    let name = &input.name;
    let params = &input.params;
    let states = &input.states;
    let outputs = &input.outputs;
    let routes = expand_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let metadata_function = function_spec.metadata_function;
    let covariates = covariate_metadata(&input.covariates);

    let expanded = quote! {{
        #derive
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Analytical)
            .parameters([#(stringify!(#params)),*])
            #covariates
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .analytical_kernel(#metadata_function);

        ::pharmsol::equation::Analytical::new(
            #eq,
            |_, _, _| {},
            #lag,
            #fa,
            #init,
            #out,
        )
        .with_nstates(#nstates)
        .with_ndrugs(#ndrugs)
        .with_nout(#nout)
        .with_metadata(__pharmsol_metadata)
        .expect("built-in `analytical!` generated invalid metadata")
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

fn expand_derive(
    derive: Option<&ExprClosure>,
    params: &[Ident],
    covariates: &[Ident],
    derived: &[Ident],
) -> syn::Result<TokenStream2> {
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let derived_len = syn::LitInt::new(&derived.len().to_string(), Span::call_site());

    if let Some(derive) = derive {
        let full_inputs = [p.clone(), t.clone(), cov.clone()];
        let reduced_inputs = [t.clone()];
        let input_aliases = generate_supported_input_aliases(
            derive,
            &[&full_inputs, &reduced_inputs],
            "built-in `analytical!` requires `derive` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|",
        )?;
        let parameter_bindings = generate_parameter_bindings(params, derive, &p);
        let covariate_bindings = generate_covariate_bindings(covariates, derive, &cov, &t);
        let derived_decls = derived.iter().map(|ident| {
            quote! {
                #[allow(unused_mut)]
                let mut #ident: f64;
            }
        });
        let body = &derive.body;

        Ok(quote! {
            fn __pharmsol_derive(
                #p: &::pharmsol::simulator::V,
                #t: f64,
                #cov: &::pharmsol::data::Covariates,
            ) -> [f64; #derived_len] {
                #input_aliases
                #parameter_bindings
                #covariate_bindings
                #(#derived_decls)*
                #body
                [#(#derived),*]
            }
        })
    } else {
        let zeros = derived.iter().map(|_| quote! { 0.0 });
        Ok(quote! {
            fn __pharmsol_derive(
                _: &::pharmsol::simulator::V,
                _: f64,
                _: &::pharmsol::data::Covariates,
            ) -> [f64; #derived_len] {
                [#(#zeros),*]
            }
        })
    }
}

fn expand_runtime(
    runtime_path: &TokenStream2,
    projection: &AnalyticalStructureInputKind,
) -> TokenStream2 {
    match projection {
        AnalyticalStructureInputKind::AllPrimary { identity: true, .. } => runtime_path.clone(),
        AnalyticalStructureInputKind::AllPrimary { indices, .. } => {
            let projected = indices.iter().map(|index| quote! { __pharmsol_p[#index] });
            quote! {{
                let __pharmsol_eq: fn(
                    &::pharmsol::simulator::V,
                    &::pharmsol::simulator::V,
                    f64,
                    &::pharmsol::simulator::V,
                    &::pharmsol::data::Covariates,
                ) -> ::pharmsol::simulator::V = |
                    __pharmsol_x: &::pharmsol::simulator::V,
                    __pharmsol_p: &::pharmsol::simulator::V,
                    __pharmsol_t: f64,
                    __pharmsol_rateiv: &::pharmsol::simulator::V,
                    __pharmsol_cov: &::pharmsol::data::Covariates,
                | {
                    let __pharmsol_projected = ::pharmsol::__macro_support::vector_from_values(vec![#(#projected),*]);
                    #runtime_path(__pharmsol_x, &__pharmsol_projected, __pharmsol_t, __pharmsol_rateiv, __pharmsol_cov)
                };
                __pharmsol_eq
            }}
        }
        AnalyticalStructureInputKind::AllDerived { indices, .. } => {
            let projected = indices
                .iter()
                .map(|index| quote! { __pharmsol_derived[#index] });
            quote! {{
                let __pharmsol_eq: fn(
                    &::pharmsol::simulator::V,
                    &::pharmsol::simulator::V,
                    f64,
                    &::pharmsol::simulator::V,
                    &::pharmsol::data::Covariates,
                ) -> ::pharmsol::simulator::V = |
                    __pharmsol_x: &::pharmsol::simulator::V,
                    __pharmsol_p: &::pharmsol::simulator::V,
                    __pharmsol_t: f64,
                    __pharmsol_rateiv: &::pharmsol::simulator::V,
                    __pharmsol_cov: &::pharmsol::data::Covariates,
                | {
                    let __pharmsol_derived = __pharmsol_derive(__pharmsol_p, __pharmsol_t, __pharmsol_cov);
                    let __pharmsol_projected = ::pharmsol::__macro_support::vector_from_values(vec![#(#projected),*]);
                    #runtime_path(__pharmsol_x, &__pharmsol_projected, __pharmsol_t, __pharmsol_rateiv, __pharmsol_cov)
                };
                __pharmsol_eq
            }}
        }
        AnalyticalStructureInputKind::Mixed { bindings } => {
            let projected = bindings.iter().map(|binding| match binding.source {
                AnalyticalStructureInputSource::Primary => {
                    let index = binding.index;
                    quote! { __pharmsol_p[#index] }
                }
                AnalyticalStructureInputSource::Derived => {
                    let index = binding.index;
                    quote! { __pharmsol_derived[#index] }
                }
            });
            quote! {{
                let __pharmsol_eq: fn(
                    &::pharmsol::simulator::V,
                    &::pharmsol::simulator::V,
                    f64,
                    &::pharmsol::simulator::V,
                    &::pharmsol::data::Covariates,
                ) -> ::pharmsol::simulator::V = |
                    __pharmsol_x: &::pharmsol::simulator::V,
                    __pharmsol_p: &::pharmsol::simulator::V,
                    __pharmsol_t: f64,
                    __pharmsol_rateiv: &::pharmsol::simulator::V,
                    __pharmsol_cov: &::pharmsol::data::Covariates,
                | {
                    let __pharmsol_derived = __pharmsol_derive(__pharmsol_p, __pharmsol_t, __pharmsol_cov);
                    let __pharmsol_projected = ::pharmsol::__macro_support::vector_from_values(vec![#(#projected),*]);
                    #runtime_path(__pharmsol_x, &__pharmsol_projected, __pharmsol_t, __pharmsol_rateiv, __pharmsol_cov)
                };
                __pharmsol_eq
            }}
        }
    }
}

fn expand_route_map(
    label: &str,
    closure: &ExprClosure,
    params: &[Ident],
    derived: &[Ident],
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
            "built-in `analytical!` requires `{label}` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|"
        ),
    )?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
    let derived_values = generated_ident("__pharmsol_derived");
    let derived_bindings = generate_derived_bindings(derived, closure, &derived_values);
    let derive_values = if derived_bindings.is_empty() {
        quote! {}
    } else {
        quote! {
            let #derived_values = __pharmsol_derive(#p, #t, #cov);
        }
    };
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
            #derive_values
            #derived_bindings
            #covariate_bindings
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_init(
    init: &ExprClosure,
    params: &[Ident],
    derived: &[Ident],
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
        "built-in `analytical!` requires `init` to have either 4 parameters: |p, t, cov, x| or 2 parameters: |t, x|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
    let derived_values = generated_ident("__pharmsol_derived");
    let derived_bindings = generate_derived_bindings(derived, init, &derived_values);
    let derive_values = if derived_bindings.is_empty() {
        quote! {}
    } else {
        quote! {
            let #derived_values = __pharmsol_derive(#p, #t, #cov);
        }
    };
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
            #derive_values
            #derived_bindings
            #covariate_bindings
            #body
        };
        __pharmsol_init
    }})
}

fn expand_out(
    out: &ExprClosure,
    params: &[Ident],
    derived: &[Ident],
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
        "built-in `analytical!` requires `out` to have either 5 parameters: |x, p, t, cov, y| or 3 parameters: |x, t, y|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
    let derived_values = generated_ident("__pharmsol_derived");
    let derived_bindings = generate_derived_bindings(derived, out, &derived_values);
    let derive_values = if derived_bindings.is_empty() {
        quote! {}
    } else {
        quote! {
            let #derived_values = __pharmsol_derive(#p, #t, #cov);
        }
    };
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
            #derive_values
            #derived_bindings
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
            // Lag and bioavailability are bolus-only; when a bolus and an
            // infusion share a label, the property binds to the bolus route.
            let bolus_route = matches!(route.kind, OdeRouteKind::Bolus);
            let lag_flag = if bolus_route && lag_routes.contains(&route_name) {
                quote! { .with_lag() }
            } else {
                quote! {}
            };
            let fa_flag = if bolus_route && fa_routes.contains(&route_name) {
                quote! { .with_bioavailability() }
            } else {
                quote! {}
            };

            quote! {
                #route_builder
                    .to_state(stringify!(#destination))
                    #lag_flag
                    #fa_flag
            }
        })
        .collect()
}
