//! Token generation for the three declaration-first macros.

pub(crate) mod analytical;
pub(crate) mod ode;
pub(crate) mod sde;

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;

use crate::symbols::OdeRouteDecl;

fn route_destination_index(route: &OdeRouteDecl, states: &[Ident]) -> usize {
    states
        .iter()
        .position(|state| state == &route.destination)
        .expect("validated route destination should exist")
}

fn covariate_metadata(covariates: &[Ident]) -> TokenStream2 {
    if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    }
}

fn empty_route_map() -> TokenStream2 {
    quote! { |_, _, _| ::std::collections::HashMap::new() }
}

fn empty_init() -> TokenStream2 {
    quote! { |_, _, _, _| {} }
}
