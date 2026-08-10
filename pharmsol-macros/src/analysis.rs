//! Inspection and rewriting of user-written closure bodies.

use proc_macro2::Span;
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::{
    parse::Parser, punctuated::Punctuated, visit::Visit, visit_mut::VisitMut, Expr, ExprClosure,
    Ident, Lit, LitInt, Pat, Token,
};

use crate::symbols::{RoutePropertyEntry, SymbolicIndex};

fn param_name(pat: &Pat) -> String {
    match pat {
        Pat::Ident(p) => p.ident.to_string(),
        _ => String::new(),
    }
}

pub(crate) fn closure_param_names(c: &ExprClosure) -> Vec<String> {
    c.inputs.iter().map(param_name).collect()
}

pub(crate) fn closure_param_ident(c: &ExprClosure, index: usize) -> Option<Ident> {
    c.inputs.get(index).and_then(|pat| match pat {
        Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
        _ => None,
    })
}

pub(crate) fn generated_ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

/// Which names a closure body mentions, so only the referenced bindings are
/// materialized in the expansion.
#[derive(Default)]
pub(crate) struct ClosureBodyUsage {
    idents: HashSet<String>,
    indexed_idents: HashSet<String>,
    assigned_indexed_idents: HashSet<String>,
    contains_macro: bool,
}

impl ClosureBodyUsage {
    pub(crate) fn analyze(expr: &Expr) -> Self {
        let mut usage = Self::default();
        usage.visit_expr(expr);
        usage
    }

    pub(crate) fn uses(&self, ident: &Ident) -> bool {
        self.contains_macro || self.idents.contains(&ident.to_string())
    }

    pub(crate) fn mentions(&self, ident: &Ident) -> bool {
        self.idents.contains(&ident.to_string())
    }

    pub(crate) fn indexes(&self, ident: &Ident) -> bool {
        self.indexed_idents.contains(&ident.to_string())
    }

    pub(crate) fn assigns_index(&self, ident: &Ident) -> bool {
        self.assigned_indexed_idents.contains(&ident.to_string())
    }
}

impl<'ast> Visit<'ast> for ClosureBodyUsage {
    fn visit_expr_path(&mut self, expr_path: &'ast syn::ExprPath) {
        if expr_path.qself.is_none()
            && expr_path.path.leading_colon.is_none()
            && expr_path.path.segments.len() == 1
        {
            self.idents
                .insert(expr_path.path.segments[0].ident.to_string());
        }

        syn::visit::visit_expr_path(self, expr_path);
    }

    fn visit_expr_macro(&mut self, expr_macro: &'ast syn::ExprMacro) {
        self.contains_macro = true;
        syn::visit::visit_expr_macro(self, expr_macro);
    }

    fn visit_stmt_macro(&mut self, stmt_macro: &'ast syn::StmtMacro) {
        self.contains_macro = true;
        syn::visit::visit_stmt_macro(self, stmt_macro);
    }

    fn visit_expr_index(&mut self, expr_index: &'ast syn::ExprIndex) {
        if let Expr::Path(expr_path) = expr_index.expr.as_ref() {
            if expr_path.qself.is_none()
                && expr_path.path.leading_colon.is_none()
                && expr_path.path.segments.len() == 1
            {
                self.indexed_idents
                    .insert(expr_path.path.segments[0].ident.to_string());
            }
        }

        syn::visit::visit_expr_index(self, expr_index);
    }

    fn visit_expr_assign(&mut self, expr_assign: &'ast syn::ExprAssign) {
        if let Expr::Index(expr_index) = expr_assign.left.as_ref() {
            if let Expr::Path(expr_path) = expr_index.expr.as_ref() {
                if expr_path.qself.is_none()
                    && expr_path.path.leading_colon.is_none()
                    && expr_path.path.segments.len() == 1
                {
                    self.assigned_indexed_idents
                        .insert(expr_path.path.segments[0].ident.to_string());
                }
            }
        }

        syn::visit::visit_expr_assign(self, expr_assign);
    }
}

pub(crate) struct IndexRewriteTarget {
    container: Ident,
    labels: HashMap<usize, usize>,
}

impl IndexRewriteTarget {
    pub(crate) fn new(container: Ident, labels: HashMap<usize, usize>) -> Self {
        Self { container, labels }
    }
}

/// Maps user-visible numeric labels onto the internal dense indices used by the
/// generated closures.
pub(crate) struct NumericLabelRewriter {
    index_targets: Vec<IndexRewriteTarget>,
    route_labels: Option<HashMap<usize, usize>>,
}

impl NumericLabelRewriter {
    pub(crate) fn rewrite(
        expr: &Expr,
        index_targets: Vec<IndexRewriteTarget>,
        route_labels: Option<HashMap<usize, usize>>,
    ) -> Expr {
        let mut rewritten = expr.clone();
        let mut rewriter = Self {
            index_targets,
            route_labels,
        };
        rewriter.visit_expr_mut(&mut rewritten);
        rewritten
    }

    fn target_labels(&self, path: &syn::ExprPath) -> Option<&HashMap<usize, usize>> {
        if path.qself.is_some()
            || path.path.leading_colon.is_some()
            || path.path.segments.len() != 1
        {
            return None;
        }

        let ident = &path.path.segments[0].ident;
        self.index_targets
            .iter()
            .find(|target| target.container == *ident)
            .map(|target| &target.labels)
    }

    fn rewrite_route_macro(&self, mac: &mut syn::Macro) {
        let Some(route_labels) = self.route_labels.as_ref() else {
            return;
        };
        if !(mac.path.is_ident("lag") || mac.path.is_ident("fa")) {
            return;
        }

        let Ok(entries) = Punctuated::<RoutePropertyEntry, Token![,]>::parse_terminated
            .parse2(mac.tokens.clone())
        else {
            return;
        };

        let entries = entries.into_iter().map(|mut entry| {
            if let Some(value) = entry.route.numeric_value() {
                if let Some(internal_index) = route_labels.get(&value) {
                    entry.route = SymbolicIndex::numeric(*internal_index);
                }
            }
            entry
        });

        let tokens = entries.map(|entry| {
            let route = entry.route;
            let value = entry.value;
            quote! { #route => #value }
        });
        mac.tokens = quote! { #(#tokens),* };
    }
}

impl VisitMut for NumericLabelRewriter {
    fn visit_expr_index_mut(&mut self, expr_index: &mut syn::ExprIndex) {
        syn::visit_mut::visit_expr_index_mut(self, expr_index);

        let Expr::Path(expr_path) = expr_index.expr.as_ref() else {
            return;
        };
        let Some(labels) = self.target_labels(expr_path) else {
            return;
        };
        let Expr::Lit(expr_lit) = expr_index.index.as_ref() else {
            return;
        };
        let Lit::Int(lit) = &expr_lit.lit else {
            return;
        };
        let Ok(external_index) = lit.base10_parse::<usize>() else {
            return;
        };
        let Some(internal_index) = labels.get(&external_index) else {
            return;
        };

        *expr_index.index = Expr::Lit(syn::ExprLit {
            attrs: Vec::new(),
            lit: Lit::Int(LitInt::new(&internal_index.to_string(), lit.span())),
        });
    }

    fn visit_expr_macro_mut(&mut self, expr_macro: &mut syn::ExprMacro) {
        self.rewrite_route_macro(&mut expr_macro.mac);
        syn::visit_mut::visit_expr_macro_mut(self, expr_macro);
    }

    fn visit_stmt_macro_mut(&mut self, stmt_macro: &mut syn::StmtMacro) {
        self.rewrite_route_macro(&mut stmt_macro.mac);
        syn::visit_mut::visit_stmt_macro_mut(self, stmt_macro);
    }
}
