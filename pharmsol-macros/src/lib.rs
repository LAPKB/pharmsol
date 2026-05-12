//! Procedural macros for [`pharmsol`](https://crates.io/crates/pharmsol).
//!
//! This crate is not intended to be used directly. Use the re-exports from the
//! `pharmsol` crate instead.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use pharmsol_dsl::{
    AnalyticalKernel as ResolverAnalyticalKernel, AnalyticalStructureInputKind,
    AnalyticalStructureInputPlan, AnalyticalStructureInputSource,
};
use quote::{quote, ToTokens};
use std::collections::{HashMap, HashSet};
use syn::{
    parse::{Parse, ParseStream, Parser},
    punctuated::Punctuated,
    token,
    visit::Visit,
    visit_mut::VisitMut,
    Expr, ExprClosure, Ident, Lit, LitInt, LitStr, Pat, Stmt, Token,
};

// ---------------------------------------------------------------------------
// Macro input parsing
// ---------------------------------------------------------------------------

struct OdeInput {
    name: LitStr,
    params: Vec<Ident>,
    covariates: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<SymbolicIndex>,
    routes: Vec<OdeRouteDecl>,
    diffeq: ExprClosure,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

struct AnalyticalInput {
    name: LitStr,
    params: Vec<Ident>,
    derived: Vec<Ident>,
    covariates: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<SymbolicIndex>,
    routes: Vec<OdeRouteDecl>,
    structure: Ident,
    derive: Option<ExprClosure>,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

struct SdeInput {
    name: LitStr,
    params: Vec<Ident>,
    covariates: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<SymbolicIndex>,
    routes: Vec<OdeRouteDecl>,
    particles: Expr,
    drift: ExprClosure,
    diffusion: ExprClosure,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

struct OdeRouteDecl {
    kind: OdeRouteKind,
    input: SymbolicIndex,
    destination: Ident,
}

#[derive(Clone, Copy)]
enum OdeRouteKind {
    Bolus,
    Infusion,
}

struct AnalyticalKernelSpec {
    kernel: ResolverAnalyticalKernel,
    runtime_path: TokenStream2,
    metadata_kernel: TokenStream2,
    state_count: usize,
}

struct RoutePropertyEntry {
    route: SymbolicIndex,
    value: Expr,
}

#[derive(Clone)]
enum SymbolicIndex {
    Ident(Ident),
    Int(LitInt),
}

impl SymbolicIndex {
    fn name(&self) -> String {
        match self {
            Self::Ident(ident) => ident.to_string(),
            Self::Int(lit) => lit.base10_digits().to_string(),
        }
    }

    fn ident(&self) -> Option<&Ident> {
        match self {
            Self::Ident(ident) => Some(ident),
            Self::Int(_) => None,
        }
    }

    fn numeric_value(&self) -> Option<usize> {
        match self {
            Self::Ident(_) => None,
            Self::Int(lit) => Some(
                lit.base10_parse::<usize>()
                    .expect("validated numeric label should fit usize"),
            ),
        }
    }

    fn numeric(value: usize) -> Self {
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

impl Parse for OdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut params = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut diffeq = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once_ode(&mut name, input.parse()?, &key, "name")?,
                "params" => set_once_ode(&mut params, parse_ident_list(input)?, &key, "params")?,
                "covariates" => set_once_ode(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                )?,
                "states" => set_once_ode(&mut states, parse_ident_list(input)?, &key, "states")?,
                "outputs" => set_once_ode(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                )?,
                "routes" => set_once_ode(&mut routes, parse_route_list(input)?, &key, "routes")?,
                "diffeq" => set_once_ode(&mut diffeq, input.parse()?, &key, "diffeq")?,
                "lag" => set_once_ode(&mut lag, input.parse()?, &key, "lag")?,
                "fa" => set_once_ode(&mut fa, input.parse()?, &key, "fa")?,
                "init" => set_once_ode(&mut init, input.parse()?, &key, "init")?,
                "out" => set_once_ode(&mut out, input.parse()?, &key, "out")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, params, covariates, states, outputs, routes, diffeq, lag, fa, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "declaration-first `ode!` requires `name`, `params`, `states`, `outputs`, and `routes`; the old inferred-dimensions form has been removed",
            )
        })?;
        let params = params.ok_or_else(|| missing_required_ode_field("params"))?;
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_ode_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_ode_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_ode_field("routes"))?;
        let diffeq = diffeq.ok_or_else(|| missing_required_ode_field("diffeq"))?;
        let out = out.ok_or_else(|| missing_required_ode_field("out"))?;
        validate_ode_diffeq_uses_automatic_injection(&diffeq, &routes)?;

        validate_unique_idents("parameter", &params, "ode!")?;
        validate_unique_idents("covariate", &covariates, "ode!")?;
        validate_unique_idents("state", &states, "ode!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "ode!")?;
        validate_routes(&routes, &states, "ode!")?;
        validate_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &[],
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            OdeBindingClosures {
                diffeq: &diffeq,
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        Ok(Self {
            name,
            params,
            covariates,
            states,
            outputs,
            routes,
            diffeq,
            lag,
            fa,
            init,
            out,
        })
    }
}

impl Parse for RoutePropertyEntry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let route: SymbolicIndex = input.parse()?;
        input.parse::<Token![=>]>()?;
        let value: Expr = input.parse()?;
        Ok(Self { route, value })
    }
}

impl Parse for AnalyticalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut params = None;
        let mut derived = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut structure = None;
        let mut derive = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once_analytical(&mut name, input.parse()?, &key, "name")?,
                "params" => {
                    set_once_analytical(&mut params, parse_ident_list(input)?, &key, "params")?
                }
                "derived" => set_once_analytical(
                    &mut derived,
                    parse_ident_list(input)?,
                    &key,
                    "derived",
                )?,
                "covariates" => set_once_analytical(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                )?,
                "states" => {
                    set_once_analytical(&mut states, parse_ident_list(input)?, &key, "states")?
                }
                "outputs" => set_once_analytical(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                )?,
                "routes" => {
                    set_once_analytical(&mut routes, parse_route_list(input)?, &key, "routes")?
                }
                "structure" => {
                    set_once_analytical(&mut structure, input.parse()?, &key, "structure")?
                }
                "derive" => {
                    set_once_analytical(&mut derive, input.parse()?, &key, "derive")?
                }
                "sec" => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        "built-in `analytical!` no longer supports `sec`; use `derived: [...]` plus `derive: ...`",
                    ));
                }
                "lag" => set_once_analytical(&mut lag, input.parse()?, &key, "lag")?,
                "fa" => set_once_analytical(&mut fa, input.parse()?, &key, "fa")?,
                "init" => set_once_analytical(&mut init, input.parse()?, &key, "init")?,
                "out" => set_once_analytical(&mut out, input.parse()?, &key, "out")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, params, derived, covariates, states, outputs, routes, structure, derive, lag, fa, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| missing_required_analytical_field("name"))?;
        let params = params.ok_or_else(|| missing_required_analytical_field("params"))?;
        let derived = derived.unwrap_or_default();
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_analytical_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_analytical_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_analytical_field("routes"))?;
        let structure = structure.ok_or_else(|| missing_required_analytical_field("structure"))?;
        let out = out.ok_or_else(|| missing_required_analytical_field("out"))?;

        validate_unique_idents("covariate", &covariates, "analytical!")?;
        validate_unique_idents("state", &states, "analytical!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "analytical!")?;
        validate_routes(&routes, &states, "analytical!")?;

        let kernel_spec = resolve_analytical_structure(&structure)?;
        validate_analytical_structure_inputs(&structure, kernel_spec.kernel, &params, &derived)?;
        if states.len() != kernel_spec.state_count {
            return Err(syn::Error::new_spanned(
                &structure,
                format!(
                    "analytical structure `{}` expects {} state value(s), but `states` declares {}",
                    structure,
                    kernel_spec.state_count,
                    states.len()
                ),
            ));
        }

        validate_analytical_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &derived,
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            AnalyticalBindingClosures {
                derive: derive.as_ref(),
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        validate_analytical_derive_contract(
            kernel_spec.kernel,
            &params,
            &derived,
            &covariates,
            derive.as_ref(),
        )?;

        if let Some(lag) = lag.as_ref() {
            let lag_routes =
                extract_route_property_routes("built-in `analytical!`", "lag", lag, &routes)?;
            validate_route_property_kinds("built-in `analytical!`", "lag", &routes, &lag_routes)?;
        }

        if let Some(fa) = fa.as_ref() {
            let fa_routes =
                extract_route_property_routes("built-in `analytical!`", "fa", fa, &routes)?;
            validate_route_property_kinds("built-in `analytical!`", "fa", &routes, &fa_routes)?;
        }

        Ok(Self {
            name,
            params,
            derived,
            covariates,
            states,
            outputs,
            routes,
            structure,
            derive,
            lag,
            fa,
            init,
            out,
        })
    }
}

impl Parse for SdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut params = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut particles = None;
        let mut drift = None;
        let mut diffusion = None;
        let mut lag = None;
        let mut fa = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once_sde(&mut name, input.parse()?, &key, "name")?,
                "params" => set_once_sde(&mut params, parse_ident_list(input)?, &key, "params")?,
                "covariates" => set_once_sde(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                )?,
                "states" => set_once_sde(&mut states, parse_ident_list(input)?, &key, "states")?,
                "outputs" => set_once_sde(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                )?,
                "routes" => set_once_sde(&mut routes, parse_route_list(input)?, &key, "routes")?,
                "particles" => set_once_sde(&mut particles, input.parse()?, &key, "particles")?,
                "drift" => set_once_sde(&mut drift, input.parse()?, &key, "drift")?,
                "diffusion" => set_once_sde(&mut diffusion, input.parse()?, &key, "diffusion")?,
                "lag" => set_once_sde(&mut lag, input.parse()?, &key, "lag")?,
                "fa" => set_once_sde(&mut fa, input.parse()?, &key, "fa")?,
                "init" => set_once_sde(&mut init, input.parse()?, &key, "init")?,
                "out" => set_once_sde(&mut out, input.parse()?, &key, "out")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, params, covariates, states, outputs, routes, particles, drift, diffusion, lag, fa, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| missing_required_sde_field("name"))?;
        let params = params.ok_or_else(|| missing_required_sde_field("params"))?;
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_sde_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_sde_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_sde_field("routes"))?;
        let particles = particles.ok_or_else(|| missing_required_sde_field("particles"))?;
        let drift = drift.ok_or_else(|| missing_required_sde_field("drift"))?;
        let diffusion = diffusion.ok_or_else(|| missing_required_sde_field("diffusion"))?;
        let out = out.ok_or_else(|| missing_required_sde_field("out"))?;

        validate_unique_idents("parameter", &params, "sde!")?;
        validate_unique_idents("covariate", &covariates, "sde!")?;
        validate_unique_idents("state", &states, "sde!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "sde!")?;
        validate_routes(&routes, &states, "sde!")?;
        validate_sde_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &[],
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            SdeBindingClosures {
                drift: &drift,
                diffusion: &diffusion,
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: fa.as_ref(),
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        if let Some(lag) = lag.as_ref() {
            let lag_routes =
                extract_route_property_routes("declaration-first `sde!`", "lag", lag, &routes)?;
            validate_route_property_kinds("declaration-first `sde!`", "lag", &routes, &lag_routes)?;
        }

        if let Some(fa) = fa.as_ref() {
            let fa_routes =
                extract_route_property_routes("declaration-first `sde!`", "fa", fa, &routes)?;
            validate_route_property_kinds("declaration-first `sde!`", "fa", &routes, &fa_routes)?;
        }

        Ok(Self {
            name,
            params,
            covariates,
            states,
            outputs,
            routes,
            particles,
            drift,
            diffusion,
            lag,
            fa,
            init,
            out,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn missing_required_ode_field(name: &str) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!("missing required field `{name}` in declaration-first `ode!`"),
    )
}

fn missing_required_analytical_field(name: &str) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!("missing required field `{name}` in built-in `analytical!`"),
    )
}

fn missing_required_sde_field(name: &str) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!("missing required field `{name}` in declaration-first `sde!`"),
    )
}

fn set_once_ode<T>(slot: &mut Option<T>, value: T, key: &Ident, name: &str) -> syn::Result<()> {
    if slot.is_some() {
        Err(syn::Error::new_spanned(
            key,
            format!("duplicate field `{name}` in `ode!`"),
        ))
    } else {
        *slot = Some(value);
        Ok(())
    }
}

fn set_once_analytical<T>(
    slot: &mut Option<T>,
    value: T,
    key: &Ident,
    name: &str,
) -> syn::Result<()> {
    if slot.is_some() {
        Err(syn::Error::new_spanned(
            key,
            format!("duplicate field `{name}` in `analytical!`"),
        ))
    } else {
        *slot = Some(value);
        Ok(())
    }
}

fn set_once_sde<T>(slot: &mut Option<T>, value: T, key: &Ident, name: &str) -> syn::Result<()> {
    if slot.is_some() {
        Err(syn::Error::new_spanned(
            key,
            format!("duplicate field `{name}` in `sde!`"),
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

fn param_name(pat: &Pat) -> String {
    match pat {
        Pat::Ident(p) => p.ident.to_string(),
        _ => String::new(),
    }
}

fn closure_param_names(c: &ExprClosure) -> Vec<String> {
    c.inputs.iter().map(param_name).collect()
}

fn closure_param_ident(c: &ExprClosure, index: usize) -> Option<Ident> {
    c.inputs.get(index).and_then(|pat| match pat {
        Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
        _ => None,
    })
}

fn generated_ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

fn symbolic_index_idents(labels: &[SymbolicIndex]) -> Vec<Ident> {
    labels
        .iter()
        .filter_map(|label| label.ident().cloned())
        .collect()
}

fn symbolic_index_bindings(labels: &[SymbolicIndex]) -> Vec<(SymbolicIndex, usize)> {
    labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, label)| (label, index))
        .collect()
}

fn symbolic_numeric_binding_map(bindings: &[(SymbolicIndex, usize)]) -> HashMap<usize, usize> {
    bindings
        .iter()
        .filter_map(|(label, index)| label.numeric_value().map(|value| (value, *index)))
        .collect()
}

#[derive(Default)]
struct ClosureBodyUsage {
    idents: HashSet<String>,
    indexed_idents: HashSet<String>,
    assigned_indexed_idents: HashSet<String>,
    contains_macro: bool,
}

impl ClosureBodyUsage {
    fn analyze(expr: &Expr) -> Self {
        let mut usage = Self::default();
        usage.visit_expr(expr);
        usage
    }

    fn uses(&self, ident: &Ident) -> bool {
        self.contains_macro || self.idents.contains(&ident.to_string())
    }

    fn mentions(&self, ident: &Ident) -> bool {
        self.idents.contains(&ident.to_string())
    }

    fn indexes(&self, ident: &Ident) -> bool {
        self.indexed_idents.contains(&ident.to_string())
    }

    fn assigns_index(&self, ident: &Ident) -> bool {
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

struct IndexRewriteTarget {
    container: Ident,
    labels: HashMap<usize, usize>,
}

impl IndexRewriteTarget {
    fn new(container: Ident, labels: HashMap<usize, usize>) -> Self {
        Self { container, labels }
    }
}

struct NumericLabelRewriter {
    index_targets: Vec<IndexRewriteTarget>,
    route_labels: Option<HashMap<usize, usize>>,
}

impl NumericLabelRewriter {
    fn rewrite(
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

fn generate_supported_input_aliases(
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

fn generate_parameter_bindings(
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

fn generate_derived_bindings(
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

fn generate_covariate_bindings(
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

fn analytical_error_span<'a>(names: &'a [Ident], target: &str) -> Option<&'a Ident> {
    names.iter().find(|ident| ident.to_string() == target)
}

fn validate_analytical_structure_inputs(
    structure: &Ident,
    kernel: ResolverAnalyticalKernel,
    params: &[Ident],
    derived: &[Ident],
) -> syn::Result<AnalyticalStructureInputPlan> {
    let primary_names = params.iter().map(Ident::to_string).collect::<Vec<_>>();
    let derived_names = derived.iter().map(Ident::to_string).collect::<Vec<_>>();
    AnalyticalStructureInputPlan::for_kernel(kernel, &primary_names, &derived_names).map_err(
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
            let then_assigned = analyze_derive_block(
                &expr_if.then_branch,
                context,
                &mut then_locals,
                assigned,
            )?;

            if let Some((_, else_branch)) = &expr_if.else_branch {
                let mut else_locals = locals.clone();
                let else_assigned = analyze_derive_expr(
                    else_branch,
                    context,
                    &mut else_locals,
                    assigned,
                )?;
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
        Expr::Block(expr_block) => analyze_derive_block(&expr_block.block, context, locals, assigned),
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

fn validate_analytical_derive_contract(
    kernel: ResolverAnalyticalKernel,
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
        &Ident::new(kernel.name(), Span::call_site()),
        kernel,
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
                    AnalyticalStructureInputSource::Derived => Some(derived[binding.index].to_string()),
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
                    kernel.name()
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

fn validate_ode_diffeq_uses_automatic_injection(
    diffeq: &ExprClosure,
    routes: &[OdeRouteDecl],
) -> syn::Result<()> {
    match closure_param_names(diffeq).len() {
        3 => Ok(()),
        5 => {
            let usage = ClosureBodyUsage::analyze(diffeq.body.as_ref());
            let route_inputs = route_input_idents(routes);
            let fourth_param = closure_param_ident(diffeq, 3);
            let fifth_param = closure_param_ident(diffeq, 4);
            let mentions_route_inputs = route_inputs.iter().any(|route| usage.mentions(route));
            let indexes_fifth_param = fifth_param.as_ref().is_some_and(|ident| usage.indexes(ident));
            let reads_fourth_param_as_input = fourth_param
                .as_ref()
                .is_some_and(|ident| usage.indexes(ident) && !usage.assigns_index(ident));

            if mentions_route_inputs || indexes_fifth_param || reads_fourth_param_as_input {
                Err(syn::Error::new_spanned(
                    diffeq,
                    "declaration-first `ode!` only supports automatic route injection in `diffeq`; use either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx| and remove manual `bolus[...]` / `rateiv[...]` terms",
                ))
            } else {
                Ok(())
            }
        }
        _ => Err(syn::Error::new_spanned(
            diffeq,
            "declaration-first `ode!` only supports automatic route injection in `diffeq`; use either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|",
        )),
    }
}

fn route_input_idents(routes: &[OdeRouteDecl]) -> Vec<Ident> {
    routes
        .iter()
        .filter_map(|route| route.input.ident().cloned())
        .collect()
}

fn route_input_names(routes: &[OdeRouteDecl]) -> Vec<String> {
    routes.iter().map(|route| route.input.name()).collect()
}

fn ode_route_input_bindings(routes: &[OdeRouteDecl]) -> Vec<(SymbolicIndex, usize)> {
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

fn dense_index_len(bindings: &[(SymbolicIndex, usize)]) -> usize {
    bindings
        .iter()
        .map(|(_, index)| index + 1)
        .max()
        .unwrap_or(0)
}

fn validate_binding_conflicts(
    left_label: &str,
    left: &[Ident],
    right_label: &str,
    right: &[Ident],
    context: &str,
) -> syn::Result<()> {
    let right_names = right.iter().map(Ident::to_string).collect::<HashSet<_>>();

    for ident in left {
        let name = ident.to_string();
        if right_names.contains(&name) {
            return Err(syn::Error::new_spanned(
                ident,
                format!(
                    "named {left_label} binding `{name}` conflicts with named {right_label} binding in {context}"
                ),
            ));
        }
    }

    Ok(())
}

fn validate_closure_param_conflicts(
    closure_label: &str,
    closure: &ExprClosure,
    bindings: &[Ident],
    binding_label: &str,
) -> syn::Result<()> {
    let parameter_names = closure_param_names(closure)
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<HashSet<_>>();

    for ident in bindings {
        let name = ident.to_string();
        if parameter_names.contains(&name) {
            return Err(syn::Error::new_spanned(
                ident,
                format!(
                    "named {binding_label} binding `{name}` conflicts with `{closure_label}` closure parameter `{name}`"
                ),
            ));
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct NamedBindingSets<'a> {
    params: &'a [Ident],
    derived: &'a [Ident],
    covariates: &'a [Ident],
    states: &'a [Ident],
    outputs: &'a [Ident],
    routes: &'a [OdeRouteDecl],
}

#[derive(Clone, Copy)]
struct CommonBindingClosures<'a> {
    lag: Option<&'a ExprClosure>,
    fa: Option<&'a ExprClosure>,
    init: Option<&'a ExprClosure>,
    out: &'a ExprClosure,
}

#[derive(Clone, Copy)]
struct AnalyticalBindingClosures<'a> {
    derive: Option<&'a ExprClosure>,
    common: CommonBindingClosures<'a>,
}

#[derive(Clone, Copy)]
struct OdeBindingClosures<'a> {
    diffeq: &'a ExprClosure,
    common: CommonBindingClosures<'a>,
}

#[derive(Clone, Copy)]
struct SdeBindingClosures<'a> {
    drift: &'a ExprClosure,
    diffusion: &'a ExprClosure,
    common: CommonBindingClosures<'a>,
}

fn validate_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: OdeBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived: _,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let OdeBindingClosures {
        diffeq,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "declaration-first `ode!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`diffeq` and `out` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`out` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`out` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "declaration-first `ode!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "declaration-first `ode!` named binding generation",
    )?;

    validate_closure_param_conflicts("diffeq", diffeq, params, "parameter")?;
    validate_closure_param_conflicts("diffeq", diffeq, covariates, "covariate")?;
    validate_closure_param_conflicts("diffeq", diffeq, states, "state")?;

    if let Some(lag) = lag {
        validate_binding_conflicts(
            "covariate",
            covariates,
            "route",
            &route_inputs,
            "`lag` named binding generation",
        )?;
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_binding_conflicts(
            "covariate",
            covariates,
            "route",
            &route_inputs,
            "`fa` named binding generation",
        )?;
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

fn validate_analytical_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: AnalyticalBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let AnalyticalBindingClosures {
        derive,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "covariate",
        covariates,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "output",
        outputs,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;

    if let Some(derive) = derive {
        validate_closure_param_conflicts("derive", derive, params, "parameter")?;
        validate_closure_param_conflicts("derive", derive, derived, "derived parameter")?;
        validate_closure_param_conflicts("derive", derive, covariates, "covariate")?;
    }

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, derived, "derived parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, derived, "derived parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, derived, "derived parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, derived, "derived parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

fn validate_sde_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: SdeBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived: _,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let SdeBindingClosures {
        drift,
        diffusion,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "output",
        outputs,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;

    validate_closure_param_conflicts("drift", drift, params, "parameter")?;
    validate_closure_param_conflicts("drift", drift, covariates, "covariate")?;
    validate_closure_param_conflicts("drift", drift, states, "state")?;
    validate_closure_param_conflicts("diffusion", diffusion, params, "parameter")?;
    validate_closure_param_conflicts("diffusion", diffusion, states, "state")?;

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

fn generate_index_consts(idents: &[Ident]) -> TokenStream2 {
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

fn generate_mapped_index_consts(bindings: &[(SymbolicIndex, usize)]) -> TokenStream2 {
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

fn expand_out(
    out: &ExprClosure,
    params: &[Ident],
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
        "declaration-first `ode!` requires `out` to have either 5 parameters: |x, p, t, cov, y| or 3 parameters: |x, t, y|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_out
    }})
}

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

fn extract_route_property_routes(
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

fn validate_route_property_kinds(
    macro_name: &str,
    label: &str,
    routes: &[OdeRouteDecl],
    property_routes: &HashSet<String>,
) -> syn::Result<()> {
    for route in routes {
        if property_routes.contains(&route.input.name())
            && matches!(route.kind, OdeRouteKind::Infusion)
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

fn expand_ode_route_map(
    label: &str,
    closure: &ExprClosure,
    params: &[Ident],
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
            "declaration-first `ode!` requires `{label}` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|"
        ),
    )?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_ode_init(
    init: &ExprClosure,
    params: &[Ident],
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
        "declaration-first `ode!` requires `init` to have either 4 parameters: |p, t, cov, x| or 2 parameters: |t, x|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_init
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
            let lag_flag = if lag_routes.contains(&route_name) {
                quote! { .with_lag() }
            } else {
                quote! {}
            };
            let fa_flag = if fa_routes.contains(&route_name) {
                quote! { .with_bioavailability() }
            } else {
                quote! {}
            };

            quote! {
                #route_builder
                    .to_state(stringify!(#destination))
                    #lag_flag
                    #fa_flag
                    .inject_input_to_destination()
            }
        })
        .collect()
}

fn expand_analytical_route_metadata(
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
            let lag_flag = if lag_routes.contains(&route_name) {
                quote! { .with_lag() }
            } else {
                quote! {}
            };
            let fa_flag = if fa_routes.contains(&route_name) {
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

fn expand_sde_route_metadata(
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
            let lag_flag = if lag_routes.contains(&route_name) {
                quote! { .with_lag() }
            } else {
                quote! {}
            };
            let fa_flag = if fa_routes.contains(&route_name) {
                quote! { .with_bioavailability() }
            } else {
                quote! {}
            };

            quote! {
                #route_builder
                    .to_state(stringify!(#destination))
                    .inject_input_to_destination()
                    #lag_flag
                    #fa_flag
            }
        })
        .collect()
}

fn route_destination_index(route: &OdeRouteDecl, states: &[Ident]) -> usize {
    states
        .iter()
        .position(|state| state == &route.destination)
        .expect("validated route destination should exist")
}

fn expand_injected_ode_route_terms(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
    dx: &Ident,
    bolus: &Ident,
    rateiv: &Ident,
) -> TokenStream2 {
    let terms = routes
        .iter()
        .zip(route_bindings.iter())
        .map(|(route, (_, input_index))| {
            let destination = route_destination_index(route, states);
            match route.kind {
                OdeRouteKind::Bolus => quote! {
                    #dx[#destination] += #bolus[#input_index];
                },
                OdeRouteKind::Infusion => quote! {
                    #dx[#destination] += #rateiv[#input_index];
                },
            }
        });

    quote! {
        #(#terms)*
    }
}

fn expand_injected_sde_rate_terms(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
    dx: &Ident,
    rateiv: &Ident,
) -> TokenStream2 {
    let terms = routes
        .iter()
        .zip(route_bindings.iter())
        .filter_map(|(route, (_, input_index))| match route.kind {
            OdeRouteKind::Bolus => None,
            OdeRouteKind::Infusion => {
                let destination = route_destination_index(route, states);
                Some(quote! {
                    #dx[#destination] += #rateiv[#input_index];
                })
            }
        });

    quote! {
        #(#terms)*
    }
}

fn expand_injected_sde_bolus_mappings(
    routes: &[OdeRouteDecl],
    states: &[Ident],
    route_bindings: &[(SymbolicIndex, usize)],
) -> TokenStream2 {
    let mut destinations = vec![quote! { None }; dense_index_len(route_bindings)];

    for (route, (_, input_index)) in routes.iter().zip(route_bindings.iter()) {
        if let OdeRouteKind::Bolus = route.kind {
            let destination = route_destination_index(route, states);
            destinations[*input_index] = quote! { Some(#destination) };
        }
    }

    quote! {
        .with_injected_bolus_inputs(&[#(#destinations),*])
    }
}

fn validate_unique_idents(kind: &str, idents: &[Ident], macro_name: &str) -> syn::Result<()> {
    let mut seen = HashSet::new();
    for ident in idents {
        let name = ident.to_string();
        if !seen.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                ident,
                format!("duplicate {kind} `{name}` in declaration-first `{macro_name}`"),
            ));
        }
    }
    Ok(())
}

fn validate_unique_symbolic_indices(
    kind: &str,
    labels: &[SymbolicIndex],
    macro_name: &str,
) -> syn::Result<()> {
    let mut seen = HashSet::new();
    for label in labels {
        let name = label.name();
        if !seen.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                label,
                format!("duplicate {kind} `{name}` in declaration-first `{macro_name}`"),
            ));
        }
    }
    Ok(())
}

fn validate_routes(routes: &[OdeRouteDecl], states: &[Ident], macro_name: &str) -> syn::Result<()> {
    let known_states = states.iter().map(Ident::to_string).collect::<HashSet<_>>();
    let mut seen_routes = HashSet::new();

    for route in routes {
        let route_name = route.input.name();
        if !seen_routes.insert(route_name.clone()) {
            return Err(syn::Error::new_spanned(
                &route.input,
                format!("duplicate route `{route_name}` in declaration-first `{macro_name}`"),
            ));
        }

        if !known_states.contains(&route.destination.to_string()) {
            return Err(syn::Error::new_spanned(
                &route.destination,
                format!(
                    "route destination `{}` is not declared in the `states` section",
                    route.destination
                ),
            ));
        }
    }

    Ok(())
}

fn expand_diffeq(
    diffeq: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    states: &[Ident],
    routes: &[OdeRouteDecl],
    route_bindings: &[(SymbolicIndex, usize)],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let dx = generated_ident("__pharmsol_dx");
    let bolus = generated_ident("__pharmsol_bolus");
    let rateiv = generated_ident("__pharmsol_rateiv");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [x.clone(), p.clone(), t.clone(), dx.clone(), cov.clone()];
    let reduced_inputs = [x.clone(), t.clone(), dx.clone()];
    let input_aliases = generate_supported_input_aliases(
        diffeq,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `ode!` injected-route `diffeq` requires either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, diffeq, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, diffeq, &cov, &t);
    let body = &diffeq.body;
    let dx_binding = if diffeq.inputs.len() == full_inputs.len() {
        closure_param_ident(diffeq, 3).unwrap_or_else(|| dx.clone())
    } else {
        closure_param_ident(diffeq, 2).unwrap_or_else(|| dx.clone())
    };
    let route_terms = expand_injected_ode_route_terms(
        routes,
        states,
        route_bindings,
        &dx_binding,
        &bolus,
        &rateiv,
    );

    Ok(quote! {{
        let __pharmsol_diffeq: fn(
            &::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            f64,
            &mut ::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            &::pharmsol::data::Covariates,
        ) = |#x: &::pharmsol::simulator::V,
             #p: &::pharmsol::simulator::V,
             #t: f64,
             #dx: &mut ::pharmsol::simulator::V,
             #bolus: &::pharmsol::simulator::V,
             #rateiv: &::pharmsol::simulator::V,
             #cov: &::pharmsol::data::Covariates| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #covariate_bindings
            #body
            #route_terms
        };
        __pharmsol_diffeq
    }})
}

fn resolve_analytical_structure(structure: &Ident) -> syn::Result<AnalyticalKernelSpec> {
    let structure_name = structure.to_string();
    let (kernel, runtime_path, metadata_kernel, state_count) = match structure_name.as_str() {
        "one_compartment" => (
            ResolverAnalyticalKernel::OneCompartment,
            quote! { ::pharmsol::equation::one_compartment },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartment },
            1,
        ),
        "one_compartment_cl" => (
            ResolverAnalyticalKernel::OneCompartmentCl,
            quote! { ::pharmsol::equation::one_compartment_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentCl },
            1,
        ),
        "one_compartment_cl_with_absorption" => (
            ResolverAnalyticalKernel::OneCompartmentClWithAbsorption,
            quote! { ::pharmsol::equation::one_compartment_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentClWithAbsorption },
            2,
        ),
        "one_compartment_with_absorption" => (
            ResolverAnalyticalKernel::OneCompartmentWithAbsorption,
            quote! { ::pharmsol::equation::one_compartment_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentWithAbsorption },
            2,
        ),
        "two_compartments" => (
            ResolverAnalyticalKernel::TwoCompartments,
            quote! { ::pharmsol::equation::two_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartments },
            2,
        ),
        "two_compartments_cl" => (
            ResolverAnalyticalKernel::TwoCompartmentsCl,
            quote! { ::pharmsol::equation::two_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsCl },
            2,
        ),
        "two_compartments_cl_with_absorption" => (
            ResolverAnalyticalKernel::TwoCompartmentsClWithAbsorption,
            quote! { ::pharmsol::equation::two_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsClWithAbsorption },
            3,
        ),
        "two_compartments_with_absorption" => (
            ResolverAnalyticalKernel::TwoCompartmentsWithAbsorption,
            quote! { ::pharmsol::equation::two_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsWithAbsorption },
            3,
        ),
        "three_compartments" => (
            ResolverAnalyticalKernel::ThreeCompartments,
            quote! { ::pharmsol::equation::three_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartments },
            3,
        ),
        "three_compartments_cl" => (
            ResolverAnalyticalKernel::ThreeCompartmentsCl,
            quote! { ::pharmsol::equation::three_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsCl },
            3,
        ),
        "three_compartments_cl_with_absorption" => (
            ResolverAnalyticalKernel::ThreeCompartmentsClWithAbsorption,
            quote! { ::pharmsol::equation::three_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsClWithAbsorption },
            4,
        ),
        "three_compartments_with_absorption" => (
            ResolverAnalyticalKernel::ThreeCompartmentsWithAbsorption,
            quote! { ::pharmsol::equation::three_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsWithAbsorption },
            4,
        ),
        _ => {
            return Err(syn::Error::new_spanned(
                structure,
                format!("unknown analytical structure `{structure_name}`"),
            ));
        }
    };

    Ok(AnalyticalKernelSpec {
        kernel,
        runtime_path,
        metadata_kernel,
        state_count,
    })
}

fn expand_analytical_route_map(
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

fn expand_analytical_derive(
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

fn expand_analytical_runtime(
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
            let projected = indices.iter().map(|index| quote! { __pharmsol_derived[#index] });
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

fn expand_analytical_init(
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

fn expand_analytical_out(
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

fn expand_sde_drift(
    drift: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
    states: &[Ident],
    routes: &[OdeRouteDecl],
    route_bindings: &[(SymbolicIndex, usize)],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let dx = generated_ident("__pharmsol_dx");
    let rateiv = generated_ident("__pharmsol_rateiv");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [x.clone(), p.clone(), t.clone(), dx.clone(), cov.clone()];
    let reduced_inputs = [x.clone(), t.clone(), dx.clone()];
    let input_aliases = generate_supported_input_aliases(
        drift,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `drift` to have either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, drift, &p);
    let covariate_bindings = generate_covariate_bindings(covariates, drift, &cov, &t);
    let body = &drift.body;
    let dx_binding = if drift.inputs.len() == full_inputs.len() {
        closure_param_ident(drift, 3).unwrap_or_else(|| dx.clone())
    } else {
        closure_param_ident(drift, 2).unwrap_or_else(|| dx.clone())
    };
    let rate_terms =
        expand_injected_sde_rate_terms(routes, states, route_bindings, &dx_binding, &rateiv);

    Ok(quote! {{
        let __pharmsol_drift: fn(
            &::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            f64,
            &mut ::pharmsol::simulator::V,
            &::pharmsol::simulator::V,
            &::pharmsol::data::Covariates,
        ) = |#x: &::pharmsol::simulator::V,
             #p: &::pharmsol::simulator::V,
             #t: f64,
             #dx: &mut ::pharmsol::simulator::V,
             #rateiv: &::pharmsol::simulator::V,
             #cov: &::pharmsol::data::Covariates| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #covariate_bindings
            #body
            #rate_terms
        };
        __pharmsol_drift
    }})
}

fn expand_sde_diffusion(
    diffusion: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let sigma = generated_ident("__pharmsol_sigma");
    let full_inputs = [p.clone(), sigma.clone()];
    let reduced_inputs = [sigma.clone()];
    let input_aliases = generate_supported_input_aliases(
        diffusion,
        &[&full_inputs, &reduced_inputs],
        "declaration-first `sde!` requires `diffusion` to have either 2 parameters: |p, sigma| or 1 parameter: |sigma|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, diffusion, &p);
    let body = &diffusion.body;

    Ok(quote! {{
        let __pharmsol_diffusion: fn(
            &::pharmsol::simulator::V,
            &mut ::pharmsol::simulator::V,
        ) = |#p: &::pharmsol::simulator::V,
             #sigma: &mut ::pharmsol::simulator::V| {
            #input_aliases
            #state_consts
            #parameter_bindings
            #body
        };
        __pharmsol_diffusion
    }})
}

fn expand_sde_route_map(
    label: &str,
    closure: &ExprClosure,
    params: &[Ident],
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
            "declaration-first `sde!` requires `{label}` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|"
        ),
    )?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_sde_init(
    init: &ExprClosure,
    params: &[Ident],
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
        "declaration-first `sde!` requires `init` to have either 4 parameters: |p, t, cov, x| or 2 parameters: |t, x|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_init
    }})
}

fn expand_sde_out(
    out: &ExprClosure,
    params: &[Ident],
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
        "declaration-first `sde!` requires `out` to have either 5 parameters: |x, p, t, cov, y| or 3 parameters: |x, t, y|",
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
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
            #covariate_bindings
            #body
        };
        __pharmsol_out
    }})
}

// ---------------------------------------------------------------------------
// Proc macros
// ---------------------------------------------------------------------------

#[proc_macro]
pub fn ode(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as OdeInput);

    let route_bindings = ode_route_input_bindings(&input.routes);

    let lag_routes = match input.lag.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "declaration-first `ode!`",
            "lag",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "declaration-first `ode!`",
                    "lag",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let fa_routes = match input.fa.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "declaration-first `ode!`",
            "fa",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "declaration-first `ode!`",
                    "fa",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let diffeq = match expand_diffeq(
        &input.diffeq,
        &input.params,
        &input.covariates,
        &input.states,
        &input.routes,
        &route_bindings,
    ) {
        Ok(diffeq) => diffeq,
        Err(error) => return error.to_compile_error().into(),
    };

    let out = match expand_out(
        &input.out,
        &input.params,
        &input.covariates,
        &input.states,
        &input.outputs,
    ) {
        Ok(out) => out,
        Err(error) => return error.to_compile_error().into(),
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let covariates = &input.covariates;
    let states = &input.states;
    let outputs = &input.outputs;
    let routes = expand_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let covariate_metadata = if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    };

    let lag = match input.lag.as_ref() {
        Some(closure) => match expand_ode_route_map(
            "lag",
            closure,
            &input.params,
            &input.covariates,
            &route_bindings,
        ) {
            Ok(lag) => lag,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_ode_route_map(
                "fa",
                closure,
                &input.params,
                &input.covariates,
                &route_bindings,
            ) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = match input.init.as_ref() {
        Some(closure) => {
            match expand_ode_init(closure, &input.params, &input.covariates, &input.states) {
                Ok(init) => init,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _, _| {} },
    };

    quote! {{
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .parameters([#(stringify!(#params)),*])
            #covariate_metadata
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*;

        ::pharmsol::equation::ODE::new(
            #diffeq,
            #lag,
            #fa,
            #init,
            #out,
        )
        .with_nstates(#nstates)
        .with_ndrugs(#ndrugs)
        .with_nout(#nout)
        .with_metadata(__pharmsol_metadata)
        .expect("declaration-first `ode!` generated invalid metadata")
    }}
    .into()
}

#[proc_macro]
pub fn analytical(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as AnalyticalInput);
    let route_bindings = ode_route_input_bindings(&input.routes);

    let kernel_spec = match resolve_analytical_structure(&input.structure) {
        Ok(spec) => spec,
        Err(error) => return error.to_compile_error().into(),
    };
    let projection = match validate_analytical_structure_inputs(
        &input.structure,
        kernel_spec.kernel,
        &input.params,
        &input.derived,
    ) {
        Ok(plan) => plan,
        Err(error) => return error.to_compile_error().into(),
    };

    let lag_routes = match input.lag.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "built-in `analytical!`",
            "lag",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "built-in `analytical!`",
                    "lag",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let fa_routes = match input.fa.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "built-in `analytical!`",
            "fa",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "built-in `analytical!`",
                    "fa",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let derive = match expand_analytical_derive(
        input.derive.as_ref(),
        &input.params,
        &input.covariates,
        &input.derived,
    ) {
        Ok(derive) => derive,
        Err(error) => return error.to_compile_error().into(),
    };
    let eq = expand_analytical_runtime(&kernel_spec.runtime_path, projection.kind());

    let out = match expand_analytical_out(
        &input.out,
        &input.params,
        &input.derived,
        &input.covariates,
        &input.states,
        &input.outputs,
    ) {
        Ok(out) => out,
        Err(error) => return error.to_compile_error().into(),
    };

    let lag = match input.lag.as_ref() {
        Some(closure) => {
            match expand_analytical_route_map(
                "lag",
                closure,
                &input.params,
                &input.derived,
                &input.covariates,
                &route_bindings,
            ) {
                Ok(lag) => lag,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_analytical_route_map(
                "fa",
                closure,
                &input.params,
                &input.derived,
                &input.covariates,
                &route_bindings,
            ) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = match input.init.as_ref() {
        Some(closure) => {
            match expand_analytical_init(
                closure,
                &input.params,
                &input.derived,
                &input.covariates,
                &input.states,
            ) {
                Ok(init) => init,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _, _| {} },
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let covariates = &input.covariates;
    let states = &input.states;
    let outputs = &input.outputs;
    let routes = expand_analytical_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let metadata_kernel = kernel_spec.metadata_kernel;
    let covariate_metadata = if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    };

    quote! {{
        #derive
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Analytical)
            .parameters([#(stringify!(#params)),*])
            #covariate_metadata
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .analytical_kernel(#metadata_kernel);

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
    }}
    .into()
}

#[proc_macro]
pub fn sde(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as SdeInput);
    let route_bindings = ode_route_input_bindings(&input.routes);

    let lag_routes = match input.lag.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "declaration-first `sde!`",
            "lag",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "declaration-first `sde!`",
                    "lag",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let fa_routes = match input.fa.as_ref() {
        Some(closure) => match extract_route_property_routes(
            "declaration-first `sde!`",
            "fa",
            closure,
            &input.routes,
        ) {
            Ok(routes) => {
                if let Err(error) = validate_route_property_kinds(
                    "declaration-first `sde!`",
                    "fa",
                    &input.routes,
                    &routes,
                ) {
                    return error.to_compile_error().into();
                }
                routes
            }
            Err(error) => return error.to_compile_error().into(),
        },
        None => HashSet::new(),
    };

    let drift = match expand_sde_drift(
        &input.drift,
        &input.params,
        &input.covariates,
        &input.states,
        &input.routes,
        &route_bindings,
    ) {
        Ok(drift) => drift,
        Err(error) => return error.to_compile_error().into(),
    };

    let diffusion = match expand_sde_diffusion(&input.diffusion, &input.params, &input.states) {
        Ok(diffusion) => diffusion,
        Err(error) => return error.to_compile_error().into(),
    };

    let lag = match input.lag.as_ref() {
        Some(closure) => match expand_sde_route_map(
            "lag",
            closure,
            &input.params,
            &input.covariates,
            &route_bindings,
        ) {
            Ok(lag) => lag,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_sde_route_map(
                "fa",
                closure,
                &input.params,
                &input.covariates,
                &route_bindings,
            ) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = match input.init.as_ref() {
        Some(closure) => {
            match expand_sde_init(closure, &input.params, &input.covariates, &input.states) {
                Ok(init) => init,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _, _| {} },
    };

    let out = match expand_sde_out(
        &input.out,
        &input.params,
        &input.covariates,
        &input.states,
        &input.outputs,
    ) {
        Ok(out) => out,
        Err(error) => return error.to_compile_error().into(),
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let covariates = &input.covariates;
    let states = &input.states;
    let outputs = &input.outputs;
    let particles = &input.particles;
    let routes = expand_sde_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let bolus_mappings =
        expand_injected_sde_bolus_mappings(&input.routes, &input.states, &route_bindings);
    let covariate_metadata = if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    };

    quote! {{
        let __pharmsol_particles: usize = #particles;
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Sde)
            .parameters([#(stringify!(#params)),*])
            #covariate_metadata
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .particles(__pharmsol_particles);

        ::pharmsol::equation::SDE::new(
            #drift,
            #diffusion,
            #lag,
            #fa,
            #init,
            #out,
            __pharmsol_particles,
        )
        .with_nstates(#nstates)
        .with_ndrugs(#ndrugs)
        .with_nout(#nout)
        #bolus_mappings
        .with_metadata(__pharmsol_metadata)
        .expect("declaration-first `sde!` generated invalid metadata")
    }}
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_removed_legacy_form() {
        let error = syn::parse_str::<OdeInput>(
            "diffeq: |x, p, t, dx, b, rateiv, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("legacy macro form must fail");

        assert!(error
            .to_string()
            .contains("requires `name`, `params`, `states`, `outputs`, and `routes`"));
        assert!(error
            .to_string()
            .contains("old inferred-dimensions form has been removed"));
    }

    #[test]
    fn validates_route_destinations() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> peripheral], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown route destination must fail");

        assert!(error
            .to_string()
            .contains("route destination `peripheral` is not declared in the `states` section"));
    }

    #[test]
    fn rejects_named_binding_collisions() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [central, v], states: [central], outputs: [cp], routes: [infusion(iv) -> central], diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("parameter/state binding collisions must fail");

        assert!(error
            .to_string()
            .contains("named parameter binding `central` conflicts with named state binding"));
    }

    #[test]
    fn ode_route_bindings_share_inputs_by_kind_local_ordinal() {
        let input = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ka, ke, v], states: [depot, central], outputs: [cp], routes: [bolus(oral) -> depot, infusion(iv) -> central, bolus(sc) -> depot], diffeq: |x, p, t, dx, b, rateiv, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("declaration-first ode input should parse");

        let bindings = ode_route_input_bindings(&input.routes);

        assert_eq!(dense_index_len(&bindings), 2);
        assert_eq!(bindings[0].0.name(), "oral");
        assert_eq!(bindings[0].1, 0);
        assert_eq!(bindings[1].0.name(), "iv");
        assert_eq!(bindings[1].1, 0);
        assert_eq!(bindings[2].0.name(), "sc");
        assert_eq!(bindings[2].1, 1);
    }

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

    #[test]
    fn analytical_accepts_extra_parameters_beyond_kernel_arity() {
        let input = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v, tlag, tvke], derived: [ke], covariates: [wt, renal], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = tvke; }, out: |x, p, t, cov, y| {}",
        )
        .expect("extra declared parameters should be allowed");

        assert_eq!(input.params.len(), 5);
        assert_eq!(input.derived.len(), 1);
        assert_eq!(input.covariates.len(), 2);
        assert!(input.derive.is_some());
        assert_eq!(input.states.len(), 2);
    }

    #[test]
    fn analytical_rejects_legacy_sec_with_migration_message() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, sec: |_t| { ke = 1.0; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("legacy sec must fail");

        assert!(error
            .to_string()
            .contains("no longer supports `sec`; use `derived: [...]` plus `derive: ...`"));
    }

    #[test]
    fn analytical_rejects_unknown_structure() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], structure: mystery, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown analytical structure must fail");

        assert!(error
            .to_string()
            .contains("unknown analytical structure `mystery`"));
    }

    #[test]
    fn analytical_rejects_missing_required_structure_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing required structure name must fail");

        assert!(error
            .to_string()
            .contains("requires `ka`"));
    }

    #[test]
    fn analytical_rejects_overlap_between_params_and_derived() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = 1.0; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("overlap must fail");

        assert!(error
            .to_string()
            .contains("`ke` is declared in both `params` and `derived`"));
    }

    #[test]
    fn analytical_rejects_invalid_derive_target() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke0 = 1.0; ke = 0.1; }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("invalid derive target must fail");

        assert!(error.to_string().contains("`derive` cannot assign to `ke0`"));
    }

    #[test]
    fn analytical_rejects_if_only_assignment_for_required_derived_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], covariates: [wt], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { if wt > 70.0 { ke = ke0; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("bare if must fail");

        assert!(error
            .to_string()
            .contains("not definitely assigned on every path"));
    }

    #[test]
    fn analytical_accepts_if_else_assignment_for_required_derived_name() {
        syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], covariates: [wt], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { if wt > 70.0 { ke = ke0; } else { ke = ke0 * 0.5; } }, out: |x, p, t, cov, y| {}",
        )
        .expect("if / else should establish derived assignment");
    }

    #[test]
    fn analytical_rejects_loop_only_assignment_for_required_derived_name() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { for i in 0..1 { let _ = i; ke = ke0; } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("loop-only assignment must fail");

        assert!(error
            .to_string()
            .contains("not definitely assigned on every path"));
    }

    #[test]
    fn analytical_accepts_initial_assignment_followed_by_loop_updates() {
        syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke0, v], derived: [ke], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, derive: |_t| { ke = ke0; for i in 0..2 { let _ = i; ke = ke + 1.0; } }, out: |x, p, t, cov, y| {}",
        )
        .expect("initial assignment plus loop updates should pass");
    }

    #[test]
    fn analytical_rejects_unknown_route_property_binding() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, lag: |_p, _t, _cov| { lag! { iv => 1.0 } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown lag route must fail");

        assert!(error
            .to_string()
            .contains("route `iv` in `lag!` is not declared in the `routes` section"));
    }

    #[test]
    fn analytical_rejects_infusion_lag_binding() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke, v, tlag], states: [central], outputs: [cp], routes: [infusion(iv) -> central], structure: one_compartment, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("infusion lag must fail");

        assert!(error
            .to_string()
            .contains("built-in `analytical!` does not allow `lag` on infusion route `iv`"));
    }

    #[test]
    fn sde_requires_particles() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, theta], states: [central], outputs: [cp], routes: [infusion(iv) -> central], drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing particles must fail");

        assert!(error
            .to_string()
            .contains("missing required field `particles` in declaration-first `sde!`"));
    }

    #[test]
    fn sde_rejects_unknown_route_property_binding() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, sigma_ke], states: [central], outputs: [cp], routes: [infusion(iv) -> central], particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { oral => 1.0 } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown lag route must fail");

        assert!(error
            .to_string()
            .contains("route `oral` in `lag!` is not declared in the `routes` section"));
    }

    #[test]
    fn sde_rejects_infusion_lag_binding() {
        let error = syn::parse_str::<SdeInput>(
            "name: \"demo\", params: [ke, sigma_ke, tlag], states: [central], outputs: [cp], routes: [infusion(iv) -> central], particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("infusion lag must fail");

        assert!(error
            .to_string()
            .contains("declaration-first `sde!` does not allow `lag` on infusion route `iv`"));
    }

    #[test]
    fn rejects_braced_route_lists() {
        let error = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("braced route lists must fail");

        assert!(error
            .to_string()
            .contains("declaration-first macro `routes` must use `[...]`, not `{...}`"));
    }
}
