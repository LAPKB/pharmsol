//! Parsing of the declaration-first macro bodies.

mod analytical;
mod ode;
mod sde;

pub(crate) use analytical::AnalyticalInput;
pub(crate) use ode::OdeInput;
pub(crate) use sde::SdeInput;

use syn::{parse::ParseStream, punctuated::Punctuated, token, Ident, Token};

use crate::symbols::{OdeRouteDecl, SymbolicIndex};

fn missing_required_field(name: &str, macro_label: &str) -> syn::Error {
    syn::Error::new(
        proc_macro2::Span::call_site(),
        format!("missing required field `{name}` in {macro_label}"),
    )
}

fn set_once<T>(
    slot: &mut Option<T>,
    value: T,
    key: &Ident,
    name: &str,
    macro_name: &str,
) -> syn::Result<()> {
    if slot.is_some() {
        Err(syn::Error::new_spanned(
            key,
            format!("duplicate field `{name}` in `{macro_name}`"),
        ))
    } else {
        *slot = Some(value);
        Ok(())
    }
}

fn parse_ident_list(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    syn::bracketed!(content in input);
    Ok(Punctuated::<Ident, Token![,]>::parse_terminated(&content)?
        .into_iter()
        .collect())
}

fn parse_symbolic_index_list(input: ParseStream) -> syn::Result<Vec<SymbolicIndex>> {
    let content;
    syn::bracketed!(content in input);
    Ok(
        Punctuated::<SymbolicIndex, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect(),
    )
}

fn parse_route_list(input: ParseStream) -> syn::Result<Vec<OdeRouteDecl>> {
    if input.peek(token::Brace) {
        return Err(input.error("declaration-first macro `routes` must use `[...]`, not `{...}`"));
    }

    if !input.peek(token::Bracket) {
        return Err(
            input.error("expected a bracketed route list like `routes: [infusion(iv) -> central]`")
        );
    }

    let content;
    syn::bracketed!(content in input);
    Ok(
        Punctuated::<OdeRouteDecl, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect(),
    )
}
