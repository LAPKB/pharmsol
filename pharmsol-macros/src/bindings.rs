//! Generation of the `let` bindings and index constants injected into every
//! expanded closure.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{ExprClosure, Ident};

use crate::analysis::ClosureBodyUsage;
use crate::symbols::SymbolicIndex;

fn generate_closure_input_aliases(
    closure: &ExprClosure,
    internal_names: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure.inputs.len() != internal_names.len() {
        return Err(syn::Error::new_spanned(
            closure,
            "internal named binding generation error: closure arity mismatch",
        ));
    }

    let aliases =
        closure
            .inputs
            .iter()
            .zip(internal_names.iter())
            .map(|(pattern, internal_name)| {
                quote! {
                    let #pattern = #internal_name;
                }
            });

    Ok(quote! {
        #(#aliases)*
    })
}

/// Aliases the user's closure parameters onto the generated ones, picking the
/// first supported arity.
pub(crate) fn generate_supported_input_aliases(
    closure: &ExprClosure,
    supported_internal_names: &[&[Ident]],
    error_message: &str,
) -> syn::Result<TokenStream2> {
    for internal_names in supported_internal_names {
        if closure.inputs.len() == internal_names.len() {
            return generate_closure_input_aliases(closure, internal_names);
        }
    }

    Err(syn::Error::new_spanned(closure, error_message))
}

pub(crate) fn generate_parameter_bindings(
    params: &[Ident],
    closure: &ExprClosure,
    parameter_vector: &Ident,
) -> TokenStream2 {
    let usage = ClosureBodyUsage::analyze(closure.body.as_ref());
    let bindings = params
        .iter()
        .enumerate()
        .filter(|(_, ident)| usage.uses(ident))
        .map(|(index, ident)| {
            quote! {
                #[allow(unused_variables)]
                let #ident = #parameter_vector[#index];
            }
        });

    quote! {
        #(#bindings)*
    }
}

pub(crate) fn generate_derived_bindings(
    derived: &[Ident],
    closure: &ExprClosure,
    derived_values: &Ident,
) -> TokenStream2 {
    let usage = ClosureBodyUsage::analyze(closure.body.as_ref());
    let bindings = derived
        .iter()
        .enumerate()
        .filter(|(_, ident)| usage.uses(ident))
        .map(|(index, ident)| {
            quote! {
                #[allow(unused_variables)]
                let #ident = #derived_values[#index];
            }
        });

    quote! {
        #(#bindings)*
    }
}

pub(crate) fn generate_covariate_bindings(
    covariates: &[Ident],
    closure: &ExprClosure,
    covariate_map: &Ident,
    time: &Ident,
) -> TokenStream2 {
    let usage = ClosureBodyUsage::analyze(closure.body.as_ref());
    let used_covariates = covariates
        .iter()
        .filter(|ident| usage.uses(ident))
        .collect::<Vec<_>>();

    if used_covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            ::pharmsol::fetch_cov!(#covariate_map, #time, #(#used_covariates),*);
        }
    }
}

pub(crate) fn generate_index_consts(idents: &[Ident]) -> TokenStream2 {
    let bindings = idents.iter().enumerate().map(|(index, ident)| {
        quote! {
            #[allow(non_upper_case_globals, dead_code)]
            const #ident: usize = #index;
        }
    });

    quote! {
        #(#bindings)*
    }
}

pub(crate) fn generate_mapped_index_consts(bindings: &[(SymbolicIndex, usize)]) -> TokenStream2 {
    let bindings = bindings.iter().filter_map(|(label, index)| {
        label.ident().map(|ident| {
            quote! {
                #[allow(non_upper_case_globals, dead_code)]
                const #ident: usize = #index;
            }
        })
    });

    quote! {
        #(#bindings)*
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::generated_ident;

    #[test]
    fn generated_parameter_bindings_only_include_referenced_locals_in_hot_closures() {
        let params = vec![generated_ident("ke"), generated_ident("v")];
        let closure = syn::parse_str::<ExprClosure>(
            "|x, _p, _t, dx, _cov| { dx[central] = -ke * x[central]; }",
        )
        .expect("closure should parse");

        let bindings =
            generate_parameter_bindings(&params, &closure, &generated_ident("__pharmsol_p"))
                .to_string();

        assert!(
            bindings.contains("let ke = __pharmsol_p [0usize] ;")
                || bindings.contains("let ke = __pharmsol_p [ 0 ] ;")
        );
        assert!(!bindings.contains("let v ="));
    }

    #[test]
    fn generated_parameter_bindings_fall_back_to_all_params_for_stmt_macros() {
        let params = vec![generated_ident("ka"), generated_ident("tlag")];
        let closure = syn::parse_str::<ExprClosure>("|_p, _t, _cov| { lag! { oral => tlag } }")
            .expect("closure should parse");

        let bindings =
            generate_parameter_bindings(&params, &closure, &generated_ident("__pharmsol_p"))
                .to_string();

        assert!(bindings.contains("let ka ="));
        assert!(bindings.contains("let tlag ="));
    }
}
