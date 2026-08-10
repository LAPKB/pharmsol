//! Validation of the `lag! { ... }` / `fa! { ... }` route-property blocks.

use std::collections::HashSet;
use syn::{parse::Parser, punctuated::Punctuated, Expr, ExprClosure, Stmt, Token};

use quote::ToTokens;

use crate::symbols::{route_input_names, OdeRouteDecl, OdeRouteKind, RoutePropertyEntry};

fn route_property_error<T: ToTokens>(macro_name: &str, label: &str, node: T) -> syn::Error {
    syn::Error::new_spanned(
        node,
        format!(
            "{macro_name} requires `{label}` to return `{label}! {{ ... }}` so route-property metadata can be synthesized"
        ),
    )
}

fn find_terminal_macro_invocation(
    macro_name: &str,
    label: &str,
    closure: &ExprClosure,
) -> syn::Result<syn::Macro> {
    match closure.body.as_ref() {
        Expr::Macro(expr_macro) if expr_macro.mac.path.is_ident(label) => {
            Ok(expr_macro.mac.clone())
        }
        Expr::Macro(expr_macro) => Err(route_property_error(macro_name, label, expr_macro)),
        Expr::Block(expr_block) => {
            for stmt in expr_block.block.stmts.iter().rev() {
                match stmt {
                    Stmt::Expr(Expr::Macro(expr_macro), _)
                        if expr_macro.mac.path.is_ident(label) =>
                    {
                        return Ok(expr_macro.mac.clone());
                    }
                    Stmt::Expr(Expr::Macro(expr_macro), _) => {
                        return Err(route_property_error(macro_name, label, expr_macro));
                    }
                    Stmt::Expr(other, _) => {
                        return Err(route_property_error(macro_name, label, other));
                    }
                    Stmt::Macro(stmt_macro) if stmt_macro.mac.path.is_ident(label) => {
                        return Ok(stmt_macro.mac.clone());
                    }
                    Stmt::Macro(stmt_macro) => {
                        return Err(route_property_error(macro_name, label, stmt_macro));
                    }
                    _ => continue,
                }
            }

            Err(route_property_error(macro_name, label, expr_block))
        }
        other => Err(route_property_error(macro_name, label, other)),
    }
}

/// Returns the routes named by a `lag!` / `fa!` block, rejecting unknown or
/// duplicated entries.
pub(crate) fn extract_route_property_routes(
    macro_name: &str,
    label: &str,
    closure: &ExprClosure,
    routes: &[OdeRouteDecl],
) -> syn::Result<HashSet<String>> {
    let macro_expr = find_terminal_macro_invocation(macro_name, label, closure)?;
    let entries = Punctuated::<RoutePropertyEntry, Token![,]>::parse_terminated
        .parse2(macro_expr.tokens.clone())?;
    let known_routes = route_input_names(routes)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();

    for entry in entries {
        let route_name = entry.route.name();
        if !known_routes.contains(&route_name) {
            return Err(syn::Error::new_spanned(
                &entry.route,
                format!(
                    "route `{route_name}` in `{label}!` is not declared in the `routes` section"
                ),
            ));
        }
        if !seen.insert(route_name.clone()) {
            return Err(syn::Error::new_spanned(
                &entry.route,
                format!("duplicate route `{route_name}` in `{label}!`"),
            ));
        }
        let _ = entry.value;
    }

    Ok(seen)
}

pub(crate) fn validate_route_property_kinds(
    macro_name: &str,
    label: &str,
    routes: &[OdeRouteDecl],
    property_routes: &HashSet<String>,
) -> syn::Result<()> {
    for route in routes {
        if property_routes.contains(&route.input.name())
            && matches!(route.kind, OdeRouteKind::Infusion)
            && !routes.iter().any(|other| {
                matches!(other.kind, OdeRouteKind::Bolus)
                    && other.input.name() == route.input.name()
            })
        {
            return Err(syn::Error::new_spanned(
                &route.input,
                format!(
                    "{macro_name} does not allow `{label}` on infusion route `{}`",
                    route.input
                ),
            ));
        }
    }

    Ok(())
}
