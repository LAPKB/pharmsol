//! Validation of the `derived` / `derive` contract for built-in analytical
//! structures.

use pharmsol_dsl::{
    AnalyticalKernel as ResolverAnalyticalKernel, AnalyticalStructureInputKind,
    AnalyticalStructureInputPlan, AnalyticalStructureInputSource,
};
use proc_macro2::Span;
use std::collections::HashSet;
use syn::{visit::Visit, Expr, ExprClosure, Ident, Pat, Stmt};

use crate::analysis::generated_ident;
use crate::bindings::generate_supported_input_aliases;

fn analytical_error_span<'a>(names: &'a [Ident], target: &str) -> Option<&'a Ident> {
    names.iter().find(|ident| *ident == target)
}

/// Resolves how the kernel's required inputs are sourced from `params` and
/// `derived`, translating resolver errors into spanned diagnostics.
pub(crate) fn validate_analytical_structure_inputs(
    structure: &Ident,
    function: ResolverAnalyticalKernel,
    params: &[Ident],
    derived: &[Ident],
) -> syn::Result<AnalyticalStructureInputPlan> {
    let primary_names = params.iter().map(Ident::to_string).collect::<Vec<_>>();
    let derived_names = derived.iter().map(Ident::to_string).collect::<Vec<_>>();
    AnalyticalStructureInputPlan::for_kernel(function, &primary_names, &derived_names).map_err(
        |error| match error {
            pharmsol_dsl::AnalyticalStructureInputError::DuplicatePrimary { name } => {
                let span = analytical_error_span(params, &name).unwrap_or(structure);
                syn::Error::new_spanned(span, format!("duplicate primary parameter `{name}`"))
            }
            pharmsol_dsl::AnalyticalStructureInputError::DuplicateDerived { name } => {
                let span = analytical_error_span(derived, &name).unwrap_or(structure);
                syn::Error::new_spanned(span, format!("duplicate derived parameter `{name}`"))
            }
            pharmsol_dsl::AnalyticalStructureInputError::ConflictingName { name } => {
                let span = analytical_error_span(derived, &name)
                    .or_else(|| analytical_error_span(params, &name))
                    .unwrap_or(structure);
                syn::Error::new_spanned(
                    span,
                    format!("`{name}` is declared in both `params` and `derived`"),
                )
            }
            pharmsol_dsl::AnalyticalStructureInputError::MissingRequiredName {
                structure,
                name,
                suggestion,
            } => {
                let message = if let Some(candidate) = suggestion {
                    format!(
                        "analytical structure `{structure}` requires `{name}`; did you mean `{candidate}`? declare it in `params: [...]` or `derived: [...]`"
                    )
                } else {
                    format!(
                        "analytical structure `{structure}` requires `{name}`; declare it in `params: [...]` or `derived: [...]`"
                    )
                };
                syn::Error::new_spanned(structure, message)
            }
        },
    )
}

#[derive(Clone)]
struct DeriveValidationContext {
    params: HashSet<String>,
    covariates: HashSet<String>,
    derived: HashSet<String>,
}

impl DeriveValidationContext {
    fn new(params: &[Ident], covariates: &[Ident], derived: &[Ident]) -> Self {
        Self {
            params: params.iter().map(Ident::to_string).collect(),
            covariates: covariates.iter().map(Ident::to_string).collect(),
            derived: derived.iter().map(Ident::to_string).collect(),
        }
    }

    fn invalid_target_error(&self, ident: &Ident) -> syn::Error {
        let name = ident.to_string();
        let message = if self.params.contains(&name) {
            format!(
                "`derive` cannot assign to `{name}`; only names declared in `derived: [...]` are valid derive targets"
            )
        } else if self.covariates.contains(&name) {
            format!(
                "`derive` cannot assign to covariate `{name}`; only names declared in `derived: [...]` are valid derive targets"
            )
        } else {
            format!(
                "`derive` cannot assign to `{name}`; declare it in `derived: [...]` before assigning to it"
            )
        };
        syn::Error::new_spanned(ident, message)
    }
}

fn bound_local_names(pat: &Pat) -> Vec<String> {
    struct BoundNames {
        names: Vec<String>,
    }

    impl<'ast> Visit<'ast> for BoundNames {
        fn visit_pat_ident(&mut self, pat_ident: &'ast syn::PatIdent) {
            self.names.push(pat_ident.ident.to_string());
        }
    }

    let mut bound = BoundNames { names: Vec::new() };
    bound.visit_pat(pat);
    bound.names
}

fn analyze_derive_block(
    block: &syn::Block,
    context: &DeriveValidationContext,
    locals: &mut HashSet<String>,
    assigned: &HashSet<String>,
) -> syn::Result<HashSet<String>> {
    let mut assigned_now = assigned.clone();
    for stmt in &block.stmts {
        assigned_now = analyze_derive_stmt(stmt, context, locals, &assigned_now)?;
    }
    Ok(assigned_now)
}

fn analyze_derive_stmt(
    stmt: &Stmt,
    context: &DeriveValidationContext,
    locals: &mut HashSet<String>,
    assigned: &HashSet<String>,
) -> syn::Result<HashSet<String>> {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                let _ = analyze_derive_expr(&init.expr, context, &mut locals.clone(), assigned)?;
            }
            for name in bound_local_names(&local.pat) {
                locals.insert(name);
            }
            Ok(assigned.clone())
        }
        Stmt::Expr(expr, _) => analyze_derive_expr(expr, context, locals, assigned),
        Stmt::Macro(stmt_macro) => Err(syn::Error::new_spanned(
            stmt_macro,
            "`derive` only supports assignments, `if`, `if` / `else`, `for`, and local `let` bindings",
        )),
        _ => Ok(assigned.clone()),
    }
}

fn analyze_derive_expr(
    expr: &Expr,
    context: &DeriveValidationContext,
    locals: &mut HashSet<String>,
    assigned: &HashSet<String>,
) -> syn::Result<HashSet<String>> {
    match expr {
        Expr::Assign(assign) => {
            if let Expr::Path(path) = assign.left.as_ref() {
                if path.qself.is_none()
                    && path.path.leading_colon.is_none()
                    && path.path.segments.len() == 1
                {
                    let ident = &path.path.segments[0].ident;
                    let name = ident.to_string();
                    if context.derived.contains(&name) {
                        let mut next = assigned.clone();
                        next.insert(name);
                        return Ok(next);
                    }
                    if locals.contains(&name) {
                        return Ok(assigned.clone());
                    }
                    return Err(context.invalid_target_error(ident));
                }
            }
            Err(syn::Error::new_spanned(
                &assign.left,
                "`derive` assignments must target a name declared in `derived: [...]`",
            ))
        }
        Expr::If(expr_if) => {
            let mut then_locals = locals.clone();
            let then_assigned =
                analyze_derive_block(&expr_if.then_branch, context, &mut then_locals, assigned)?;

            if let Some((_, else_branch)) = &expr_if.else_branch {
                let mut else_locals = locals.clone();
                let else_assigned =
                    analyze_derive_expr(else_branch, context, &mut else_locals, assigned)?;
                Ok(then_assigned
                    .intersection(&else_assigned)
                    .cloned()
                    .collect::<HashSet<_>>())
            } else {
                Ok(assigned.clone())
            }
        }
        Expr::ForLoop(expr_for) => {
            let mut loop_locals = locals.clone();
            for name in bound_local_names(&expr_for.pat) {
                loop_locals.insert(name);
            }
            let _ = analyze_derive_block(&expr_for.body, context, &mut loop_locals, assigned)?;
            Ok(assigned.clone())
        }
        Expr::Block(expr_block) => {
            analyze_derive_block(&expr_block.block, context, locals, assigned)
        }
        Expr::While(expr_while) => Err(syn::Error::new_spanned(
            expr_while,
            "`derive` does not support `while`; use straight-line code, `if`, `if` / `else`, or `for`",
        )),
        Expr::Loop(expr_loop) => Err(syn::Error::new_spanned(
            expr_loop,
            "`derive` does not support `loop`; use straight-line code, `if`, `if` / `else`, or `for`",
        )),
        Expr::Match(expr_match) => Err(syn::Error::new_spanned(
            expr_match,
            "`derive` does not support `match`; use straight-line code, `if`, `if` / `else`, or `for`",
        )),
        _ => Ok(assigned.clone()),
    }
}

/// Checks that every declared derived parameter is definitely assigned before
/// the analytical kernel reads it.
pub(crate) fn validate_analytical_derive_contract(
    function: ResolverAnalyticalKernel,
    params: &[Ident],
    derived: &[Ident],
    covariates: &[Ident],
    derive: Option<&ExprClosure>,
) -> syn::Result<()> {
    if derived.is_empty() {
        if let Some(derive) = derive {
            return Err(syn::Error::new_spanned(
                derive,
                "built-in `analytical!` `derive` requires `derived: [...]`",
            ));
        }
        return Ok(());
    }

    let derive = derive.ok_or_else(|| {
        syn::Error::new_spanned(
            &derived[0],
            "built-in `analytical!` declares `derived: [...]` but is missing `derive: ...`",
        )
    })?;

    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [p, t.clone(), cov];
    let reduced_inputs = [t];
    generate_supported_input_aliases(
        derive,
        &[&full_inputs, &reduced_inputs],
        "built-in `analytical!` requires `derive` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|",
    )?;

    let context = DeriveValidationContext::new(params, covariates, derived);
    let mut locals = HashSet::new();
    let assigned = match derive.body.as_ref() {
        Expr::Block(expr_block) => {
            analyze_derive_block(&expr_block.block, &context, &mut locals, &HashSet::new())?
        }
        expr => analyze_derive_expr(expr, &context, &mut locals, &HashSet::new())?,
    };

    let required_derived = match validate_analytical_structure_inputs(
        &Ident::new(function.name(), Span::call_site()),
        function,
        params,
        derived,
    ) {
        Ok(plan) => match plan.kind() {
            AnalyticalStructureInputKind::AllPrimary { .. } => HashSet::new(),
            AnalyticalStructureInputKind::AllDerived { indices, .. } => indices
                .iter()
                .map(|index| derived[*index].to_string())
                .collect::<HashSet<_>>(),
            AnalyticalStructureInputKind::Mixed { bindings } => bindings
                .iter()
                .filter_map(|binding| match binding.source {
                    AnalyticalStructureInputSource::Primary => None,
                    AnalyticalStructureInputSource::Derived => {
                        Some(derived[binding.index].to_string())
                    }
                })
                .collect::<HashSet<_>>(),
        },
        Err(_) => HashSet::new(),
    };

    for ident in derived {
        let name = ident.to_string();
        if !assigned.contains(&name) {
            let message = if required_derived.contains(&name) {
                format!(
                    "derived parameter `{name}` is not definitely assigned on every path before analytical structure `{}` uses it",
                    function.name()
                )
            } else {
                format!(
                    "derived parameter `{name}` is declared in `derived: [...]` but is not definitely assigned in `derive`"
                )
            };
            return Err(syn::Error::new_spanned(ident, message));
        }
    }

    Ok(())
}
