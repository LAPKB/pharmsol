//! Crate path resolution.
//!
//! Every `quote!` block in this crate emits the placeholder path `::pharmsol`.
//! Just before a macro returns, [`rewrite_crate_paths`] replaces that
//! placeholder with the path resolved for the current invocation, so nested
//! macro calls (`::pharmsol::fetch_cov!`) are rewritten alongside plain type
//! paths.
//!
//! A leading `::` only resolves against the extern prelude, which is populated
//! from direct dependencies, so it breaks for crates that receive `pharmsol`
//! transitively. The `macro_rules!` wrappers in `pharmsol` therefore forward
//! `$crate` through the `@pharmsol_crate(...)` marker below, which resolves
//! correctly no matter how the caller reached the crate.

use proc_macro2::{Group, Spacing, Span, TokenStream as TokenStream2, TokenTree};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{parse::ParseStream, Ident, LitStr, Path, PathArguments, Token};

mod kw {
    syn::custom_keyword!(pharmsol_crate);
}

/// Consumes the `@pharmsol_crate($crate)` prefix injected by the `macro_rules!`
/// wrappers in `pharmsol`.
pub(crate) fn parse_crate_marker(input: ParseStream) -> syn::Result<Option<TokenStream2>> {
    if !(input.peek(Token![@]) && input.peek2(kw::pharmsol_crate)) {
        return Ok(None);
    }

    input.parse::<Token![@]>()?;
    input.parse::<kw::pharmsol_crate>()?;
    let marker;
    syn::parenthesized!(marker in input);
    let path = marker.parse::<TokenStream2>()?;
    if input.peek(Token![,]) {
        input.parse::<Token![,]>()?;
    }

    Ok(Some(path))
}

/// Resolves the crate path for one macro invocation.
///
/// An explicit `crate: "..."` key wins, then the `$crate` forwarded by the
/// wrapper macros, then detection from the caller's `Cargo.toml`.
pub(crate) fn resolve_crate_path(
    explicit: Option<LitStr>,
    forwarded: Option<TokenStream2>,
) -> syn::Result<TokenStream2> {
    if let Some(literal) = explicit {
        let path: Path = literal.parse()?;
        if let Some(segment) = path
            .segments
            .iter()
            .find(|segment| !matches!(segment.arguments, PathArguments::None))
        {
            return Err(syn::Error::new_spanned(
                segment,
                "`crate` must be a plain module path without generic arguments",
            ));
        }
        return Ok(absolute_crate_path(path));
    }

    Ok(forwarded.unwrap_or_else(detect_crate_path))
}

/// Anchors a user-supplied path so it resolves the same way from any module.
///
/// `pmcore::pharmsol` becomes `::pmcore::pharmsol`, while `crate::…`,
/// `self::…`, `super::…`, and already-absolute paths are left untouched.
fn absolute_crate_path(path: Path) -> TokenStream2 {
    let anchored = path.leading_colon.is_some()
        || path.segments.first().is_some_and(|segment| {
            let first = segment.ident.to_string();
            matches!(first.as_str(), "crate" | "self" | "super")
        });

    if anchored {
        quote! { #path }
    } else {
        quote! { ::#path }
    }
}

/// Detects how the downstream crate refers to `pharmsol`, honouring renamed
/// dependencies (`pharmsol = { package = "…" }`).
fn detect_crate_path() -> TokenStream2 {
    match crate_name("pharmsol") {
        Ok(FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, Span::call_site());
            quote! { ::#ident }
        }
        // `Itself` covers pharmsol's own tests, examples, benches, and doctests,
        // which all link the library through the extern prelude. `pharmsol`
        // itself declares `extern crate self as pharmsol;` so the same path also
        // resolves from inside the library.
        Ok(FoundCrate::Itself) | Err(_) => quote! { ::pharmsol },
    }
}

/// Replaces every `::pharmsol` placeholder prefix in `tokens` with `krate`.
pub(crate) fn rewrite_crate_paths(tokens: TokenStream2, krate: &TokenStream2) -> TokenStream2 {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut out = TokenStream2::new();
    let mut index = 0;

    while index < trees.len() {
        if is_crate_placeholder(&trees[index..]) {
            out.extend(krate.clone());
            index += 3;
            continue;
        }

        match &trees[index] {
            TokenTree::Group(group) => {
                let mut rewritten = Group::new(
                    group.delimiter(),
                    rewrite_crate_paths(group.stream(), krate),
                );
                rewritten.set_span(group.span());
                out.extend([TokenTree::Group(rewritten)]);
            }
            other => out.extend([other.clone()]),
        }

        index += 1;
    }

    out
}

fn is_crate_placeholder(trees: &[TokenTree]) -> bool {
    let [TokenTree::Punct(first), TokenTree::Punct(second), TokenTree::Ident(ident), ..] = trees
    else {
        return false;
    };

    first.as_char() == ':'
        && first.spacing() == Spacing::Joint
        && second.as_char() == ':'
        && ident == "pharmsol"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_crate_placeholder_including_nested_macro_calls() {
        let krate = quote! { ::pmcore::pharmsol };
        let tokens = quote! {
            {
                ::pharmsol::fetch_cov!(cov, t, wt);
                let x: &::pharmsol::simulator::V = ::pharmsol::equation::Route::bolus("oral");
            }
        };

        let rewritten = rewrite_crate_paths(tokens, &krate).to_string();

        assert!(rewritten.contains(":: pmcore :: pharmsol :: fetch_cov !"));
        assert!(rewritten.contains(":: pmcore :: pharmsol :: simulator :: V"));
        assert!(rewritten.contains(":: pmcore :: pharmsol :: equation :: Route"));
    }

    #[test]
    fn rewrites_crate_placeholder_for_renamed_dependencies() {
        let krate = quote! { ::ps };
        let tokens = quote! { ::pharmsol::simulator::V };

        assert_eq!(
            rewrite_crate_paths(tokens, &krate).to_string(),
            ":: ps :: simulator :: V"
        );
    }

    #[test]
    fn crate_key_leaves_relative_anchors_untouched() {
        for (written, expected) in [
            ("crate::vendored::pharmsol", "crate :: vendored :: pharmsol"),
            ("::pharmsol", ":: pharmsol"),
            ("self::reexports::pharmsol", "self :: reexports :: pharmsol"),
        ] {
            let path = syn::parse_str::<Path>(written).expect("valid path");
            assert_eq!(absolute_crate_path(path).to_string(), expected);
        }
    }
}
