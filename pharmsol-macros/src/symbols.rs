//! Declaration-first labels and route declarations shared by all three macros.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::ToTokens;
use std::collections::HashMap;
use syn::{
    parse::{Parse, ParseStream},
    token, Expr, Ident, LitInt, Token,
};

/// A `states` / `outputs` / route label, written either as a name or as a
/// numeric literal.
///
/// A numeric label is a plain name whose text is the literal (`1`, `10`):
/// it has no relationship to declaration order or dense slots.
#[derive(Clone)]
pub(crate) enum SymbolicIndex {
    Ident(Ident),
    Int(LitInt),
}

impl SymbolicIndex {
    pub(crate) fn name(&self) -> String {
        match self {
            Self::Ident(ident) => ident.to_string(),
            Self::Int(lit) => lit.to_string(),
        }
    }

    pub(crate) fn ident(&self) -> Option<&Ident> {
        match self {
            Self::Ident(ident) => Some(ident),
            Self::Int(_) => None,
        }
    }

    /// The literal text of a numeric label (`"1"` for `1`, `"01"` for `01`).
    pub(crate) fn numeric_text(&self) -> Option<String> {
        match self {
            Self::Ident(_) => None,
            Self::Int(lit) => Some(lit.to_string()),
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
            if !is_plain_decimal(&lit.to_string()) {
                return Err(syn::Error::new_spanned(
                    &lit,
                    "numeric declaration-first labels must be plain decimal digits such as `1` or `10`",
                ));
            }
            Ok(Self::Int(lit))
        } else {
            Ok(Self::Ident(input.parse()?))
        }
    }
}

/// Whether a literal's source text is only ASCII digits (no suffix, sign,
/// radix prefix or underscores).
pub(crate) fn is_plain_decimal(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

/// Maps each numeric label's literal text to the dense slot its declaration
/// was assigned.
pub(crate) fn symbolic_numeric_binding_map(
    bindings: &[(SymbolicIndex, usize)],
) -> HashMap<String, usize> {
    bindings
        .iter()
        .filter_map(|(label, index)| label.numeric_text().map(|text| (text, *index)))
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

/// Bolus-route input bindings only, numbered in declaration order.
///
/// Lag and bioavailability are bolus-only route properties, so property
/// closures resolve route names against bolus inputs. When a bolus and an
/// infusion share a label (one drug given by both routes), this keeps the
/// generated `const` bindings and numeric rewrites unambiguous.
pub(crate) fn bolus_route_input_bindings(routes: &[OdeRouteDecl]) -> Vec<(SymbolicIndex, usize)> {
    let mut next_bolus_index = 0usize;

    routes
        .iter()
        .filter(|route| matches!(route.kind, OdeRouteKind::Bolus))
        .map(|route| {
            let index = next_bolus_index;
            next_bolus_index += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_labels_keep_their_literal_text() {
        let labels = ["1", "01", "10"]
            .map(|src| syn::parse_str::<SymbolicIndex>(src).expect("numeric label parses"));
        assert_eq!(
            labels.each_ref().map(SymbolicIndex::name),
            ["1", "01", "10"]
        );

        let map = symbolic_numeric_binding_map(&symbolic_index_bindings(&labels));
        assert_eq!(map.get("1"), Some(&0));
        assert_eq!(map.get("01"), Some(&1));
        assert_eq!(map.get("10"), Some(&2));
    }

    #[test]
    fn rejects_non_decimal_numeric_labels() {
        for src in ["1usize", "0x1", "1_0"] {
            assert!(syn::parse_str::<SymbolicIndex>(src).is_err(), "{src}");
        }
    }
}
