//! Declaration-first labels and route declarations shared by all three macros.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::ToTokens;
use std::collections::HashMap;
use syn::{
    parse::{Parse, ParseStream},
    token, Expr, Ident, LitInt, Token,
};

/// A `states` / `outputs` / route label, written either as a name or as a
/// numeric index.
#[derive(Clone)]
pub(crate) enum SymbolicIndex {
    Ident(Ident),
    Int(LitInt),
}

impl SymbolicIndex {
    pub(crate) fn name(&self) -> String {
        match self {
            Self::Ident(ident) => ident.to_string(),
            Self::Int(lit) => lit.base10_digits().to_string(),
        }
    }

    pub(crate) fn ident(&self) -> Option<&Ident> {
        match self {
            Self::Ident(ident) => Some(ident),
            Self::Int(_) => None,
        }
    }

    pub(crate) fn numeric_value(&self) -> Option<usize> {
        match self {
            Self::Ident(_) => None,
            Self::Int(lit) => Some(
                lit.base10_parse::<usize>()
                    .expect("validated numeric label should fit usize"),
            ),
        }
    }

    pub(crate) fn numeric(value: usize) -> Self {
        Self::Int(LitInt::new(&value.to_string(), Span::call_site()))
    }
}

impl Parse for SymbolicIndex {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(LitInt) {
            let lit: LitInt = input.parse()?;
            lit.base10_parse::<usize>().map_err(|_| {
                syn::Error::new_spanned(
                    &lit,
                    "numeric declaration-first labels must be non-negative base-10 integers that fit in usize",
                )
            })?;
            Ok(Self::Int(lit))
        } else {
            Ok(Self::Ident(input.parse()?))
        }
    }
}

impl ToTokens for SymbolicIndex {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::Ident(ident) => ident.to_tokens(tokens),
            Self::Int(lit) => lit.to_tokens(tokens),
        }
    }
}

impl std::fmt::Display for SymbolicIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

pub(crate) struct OdeRouteDecl {
    pub(crate) kind: OdeRouteKind,
    pub(crate) input: SymbolicIndex,
    pub(crate) destination: Ident,
}

#[derive(Clone, Copy)]
pub(crate) enum OdeRouteKind {
    Bolus,
    Infusion,
}

impl Parse for OdeRouteDecl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kind_ident: Ident = input.parse()?;
        let kind = match kind_ident.to_string().as_str() {
            "bolus" => OdeRouteKind::Bolus,
            "infusion" => OdeRouteKind::Infusion,
            other => {
                return Err(syn::Error::new_spanned(
                    &kind_ident,
                    format!("unknown route kind `{other}`, expected `bolus` or `infusion`"),
                ));
            }
        };

        let content;
        syn::parenthesized!(content in input);
        let route_input: SymbolicIndex = content.parse()?;
        if !content.is_empty() {
            return Err(content.error("expected a single route input name inside `(...)`"));
        }

        if !input.peek(Token![->]) {
            return Err(
                input.error("expected `->` followed by a destination state in route declaration")
            );
        }
        input.parse::<Token![->]>()?;
        let destination: Ident = input.parse()?;

        if input.peek(token::Brace) {
            return Err(
                input.error("route properties are not supported in declaration-first `ode!` yet")
            );
        }

        Ok(Self {
            kind,
            input: route_input,
            destination,
        })
    }
}

/// A single `route => value` entry inside a `lag! { ... }` / `fa! { ... }` body.
pub(crate) struct RoutePropertyEntry {
    pub(crate) route: SymbolicIndex,
    pub(crate) value: Expr,
}

impl Parse for RoutePropertyEntry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let route: SymbolicIndex = input.parse()?;
        input.parse::<Token![=>]>()?;
        let value: Expr = input.parse()?;
        Ok(Self { route, value })
    }
}

pub(crate) fn symbolic_index_idents(labels: &[SymbolicIndex]) -> Vec<Ident> {
    labels
        .iter()
        .filter_map(|label| label.ident().cloned())
        .collect()
}

pub(crate) fn symbolic_index_bindings(labels: &[SymbolicIndex]) -> Vec<(SymbolicIndex, usize)> {
    labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, label)| (label, index))
        .collect()
}

pub(crate) fn symbolic_numeric_binding_map(
    bindings: &[(SymbolicIndex, usize)],
) -> HashMap<usize, usize> {
    bindings
        .iter()
        .filter_map(|(label, index)| label.numeric_value().map(|value| (value, *index)))
        .collect()
}

pub(crate) fn route_input_idents(routes: &[OdeRouteDecl]) -> Vec<Ident> {
    routes
        .iter()
        .filter_map(|route| route.input.ident().cloned())
        .collect()
}

pub(crate) fn route_input_names(routes: &[OdeRouteDecl]) -> Vec<String> {
    routes.iter().map(|route| route.input.name()).collect()
}

/// Assigns each route an input slot, numbered per route kind so bolus and
/// infusion inputs share the same ordinal space.
pub(crate) fn ode_route_input_bindings(routes: &[OdeRouteDecl]) -> Vec<(SymbolicIndex, usize)> {
    let mut next_bolus_index = 0usize;
    let mut next_infusion_index = 0usize;

    routes
        .iter()
        .map(|route| {
            let index = match route.kind {
                OdeRouteKind::Bolus => {
                    let index = next_bolus_index;
                    next_bolus_index += 1;
                    index
                }
                OdeRouteKind::Infusion => {
                    let index = next_infusion_index;
                    next_infusion_index += 1;
                    index
                }
            };
            (route.input.clone(), index)
        })
        .collect()
}

pub(crate) fn dense_index_len(bindings: &[(SymbolicIndex, usize)]) -> usize {
    bindings
        .iter()
        .map(|(_, index)| index + 1)
        .max()
        .unwrap_or(0)
}
