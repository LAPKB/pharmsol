//! Procedural macros for [`pharmsol`](https://crates.io/crates/pharmsol).
//!
//! This crate is not intended to be used directly. Use the re-exports from the
//! `pharmsol` crate instead.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use std::collections::HashSet;
use syn::{
    parse::{Parse, ParseStream, Parser},
    punctuated::Punctuated,
    token,
    visit::Visit,
    Expr, ExprClosure, Ident, LitStr, Pat, Stmt, Token,
};

// ---------------------------------------------------------------------------
// Macro input parsing
// ---------------------------------------------------------------------------

struct OdeInput {
    name: LitStr,
    params: Vec<Ident>,
    covariates: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<Ident>,
    routes: Vec<OdeRouteDecl>,
    diffeq_mode: OdeDiffeqMode,
    diffeq: ExprClosure,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

struct AnalyticalInput {
    name: LitStr,
    params: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<Ident>,
    routes: Vec<OdeRouteDecl>,
    structure: Ident,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

struct SdeInput {
    name: LitStr,
    params: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<Ident>,
    routes: Vec<OdeRouteDecl>,
    particles: Expr,
    drift: ExprClosure,
    diffusion: ExprClosure,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OdeDiffeqMode {
    InjectedRouteInputs,
    ExplicitRouteVectors,
}

struct OdeRouteDecl {
    kind: OdeRouteKind,
    input: Ident,
    destination: Ident,
}

#[derive(Clone, Copy)]
enum OdeRouteKind {
    Bolus,
    Infusion,
}

struct AnalyticalKernelSpec {
    runtime_path: TokenStream2,
    metadata_kernel: TokenStream2,
    parameter_arity: usize,
    state_count: usize,
}

struct RoutePropertyEntry {
    route: Ident,
    value: Expr,
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
        let route_input: Ident = content.parse()?;
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
                "outputs" => set_once_ode(&mut outputs, parse_ident_list(input)?, &key, "outputs")?,
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
        let diffeq_mode = classify_diffeq_mode(&diffeq)?;

        validate_unique_idents("parameter", &params, "ode!")?;
        validate_unique_idents("covariate", &covariates, "ode!")?;
        validate_unique_idents("state", &states, "ode!")?;
        validate_unique_idents("output", &outputs, "ode!")?;
        validate_routes(&routes, &states, "ode!")?;
        validate_named_binding_compatibility(
            &params,
            &states,
            &outputs,
            &routes,
            &diffeq,
            &out,
            diffeq_mode,
        )?;

        Ok(Self {
            name,
            params,
            covariates,
            states,
            outputs,
            routes,
            diffeq_mode,
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
        let route: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let value: Expr = input.parse()?;
        Ok(Self { route, value })
    }
}

impl Parse for AnalyticalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut params = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut structure = None;
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
                "states" => {
                    set_once_analytical(&mut states, parse_ident_list(input)?, &key, "states")?
                }
                "outputs" => {
                    set_once_analytical(&mut outputs, parse_ident_list(input)?, &key, "outputs")?
                }
                "routes" => {
                    set_once_analytical(&mut routes, parse_route_list(input)?, &key, "routes")?
                }
                "structure" => {
                    set_once_analytical(&mut structure, input.parse()?, &key, "structure")?
                }
                "lag" => set_once_analytical(&mut lag, input.parse()?, &key, "lag")?,
                "fa" => set_once_analytical(&mut fa, input.parse()?, &key, "fa")?,
                "init" => set_once_analytical(&mut init, input.parse()?, &key, "init")?,
                "out" => set_once_analytical(&mut out, input.parse()?, &key, "out")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, params, states, outputs, routes, structure, lag, fa, init, out"
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
        let states = states.ok_or_else(|| missing_required_analytical_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_analytical_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_analytical_field("routes"))?;
        let structure = structure.ok_or_else(|| missing_required_analytical_field("structure"))?;
        let out = out.ok_or_else(|| missing_required_analytical_field("out"))?;

        validate_unique_idents("parameter", &params, "analytical!")?;
        validate_unique_idents("state", &states, "analytical!")?;
        validate_unique_idents("output", &outputs, "analytical!")?;
        validate_routes(&routes, &states, "analytical!")?;

        let kernel_spec = resolve_analytical_structure(&structure)?;
        if params.len() < kernel_spec.parameter_arity {
            return Err(syn::Error::new_spanned(
                &structure,
                format!(
                    "analytical structure `{}` requires at least {} parameter value(s), but `params` declares {}",
                    structure, kernel_spec.parameter_arity, params.len()
                ),
            ));
        }
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
            &params,
            &states,
            &outputs,
            &routes,
            lag.as_ref(),
            fa.as_ref(),
            init.as_ref(),
            &out,
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
            states,
            outputs,
            routes,
            structure,
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
                "states" => set_once_sde(&mut states, parse_ident_list(input)?, &key, "states")?,
                "outputs" => set_once_sde(&mut outputs, parse_ident_list(input)?, &key, "outputs")?,
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
                            "unknown field `{other}`, expected one of: name, params, states, outputs, routes, particles, drift, diffusion, lag, fa, init, out"
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
        let states = states.ok_or_else(|| missing_required_sde_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_sde_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_sde_field("routes"))?;
        let particles = particles.ok_or_else(|| missing_required_sde_field("particles"))?;
        let drift = drift.ok_or_else(|| missing_required_sde_field("drift"))?;
        let diffusion = diffusion.ok_or_else(|| missing_required_sde_field("diffusion"))?;
        let out = out.ok_or_else(|| missing_required_sde_field("out"))?;

        validate_unique_idents("parameter", &params, "sde!")?;
        validate_unique_idents("state", &states, "sde!")?;
        validate_unique_idents("output", &outputs, "sde!")?;
        validate_routes(&routes, &states, "sde!")?;
        validate_sde_named_binding_compatibility(
            &params,
            &states,
            &outputs,
            &routes,
            &drift,
            &diffusion,
            lag.as_ref(),
            fa.as_ref(),
            init.as_ref(),
            &out,
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

fn parse_route_list(input: ParseStream) -> syn::Result<Vec<OdeRouteDecl>> {
    let content;
    syn::braced!(content in input);
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

#[derive(Default)]
struct ClosureBodyUsage {
    idents: HashSet<String>,
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

fn classify_diffeq_mode(diffeq: &ExprClosure) -> syn::Result<OdeDiffeqMode> {
    match closure_param_names(diffeq).len() {
        5 => Ok(OdeDiffeqMode::InjectedRouteInputs),
        7 => Ok(OdeDiffeqMode::ExplicitRouteVectors),
        _ => Err(syn::Error::new_spanned(
            diffeq,
            "declaration-first `ode!` requires `diffeq` to have either 5 parameters: |x, p, t, dx, cov| or 7 parameters: |x, p, t, dx, bolus, rateiv, cov|",
        )),
    }
}

fn route_input_idents(routes: &[OdeRouteDecl]) -> Vec<Ident> {
    routes.iter().map(|route| route.input.clone()).collect()
}

fn ode_route_channel_bindings(routes: &[OdeRouteDecl]) -> Vec<(Ident, usize)> {
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

fn dense_index_len(bindings: &[(Ident, usize)]) -> usize {
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

fn validate_named_binding_compatibility(
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
    routes: &[OdeRouteDecl],
    diffeq: &ExprClosure,
    out: &ExprClosure,
    diffeq_mode: OdeDiffeqMode,
) -> syn::Result<()> {
    let route_inputs = route_input_idents(routes);

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

    validate_closure_param_conflicts("diffeq", diffeq, params, "parameter")?;
    validate_closure_param_conflicts("diffeq", diffeq, states, "state")?;
    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    if diffeq_mode == OdeDiffeqMode::ExplicitRouteVectors {
        validate_binding_conflicts(
            "parameter",
            params,
            "route",
            &route_inputs,
            "`diffeq` named binding generation",
        )?;
        validate_binding_conflicts(
            "state",
            states,
            "route",
            &route_inputs,
            "`diffeq` named binding generation",
        )?;
        validate_closure_param_conflicts("diffeq", diffeq, &route_inputs, "route")?;
    }

    Ok(())
}

fn validate_analytical_named_binding_compatibility(
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
    routes: &[OdeRouteDecl],
    lag: Option<&ExprClosure>,
    fa: Option<&ExprClosure>,
    init: Option<&ExprClosure>,
    out: &ExprClosure,
) -> syn::Result<()> {
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
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
        "parameter",
        params,
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

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

fn validate_sde_named_binding_compatibility(
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
    routes: &[OdeRouteDecl],
    drift: &ExprClosure,
    diffusion: &ExprClosure,
    lag: Option<&ExprClosure>,
    fa: Option<&ExprClosure>,
    init: Option<&ExprClosure>,
    out: &ExprClosure,
) -> syn::Result<()> {
    let route_inputs = route_input_idents(routes);

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
    validate_closure_param_conflicts("drift", drift, states, "state")?;
    validate_closure_param_conflicts("diffusion", diffusion, params, "parameter")?;
    validate_closure_param_conflicts("diffusion", diffusion, states, "state")?;

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
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

fn generate_mapped_index_consts(bindings: &[(Ident, usize)]) -> TokenStream2 {
    let bindings = bindings.iter().map(|(ident, index)| {
        quote! {
            #[allow(non_upper_case_globals, dead_code)]
            const #ident: usize = #index;
        }
    });

    quote! {
        #(#bindings)*
    }
}

fn expand_out(
    out: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure_param_names(out).len() != 5 {
        return Err(syn::Error::new_spanned(
            out,
            "declaration-first `ode!` requires `out` to have 5 parameters: |x, p, t, cov, y|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let output_consts = generate_index_consts(outputs);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let y = generated_ident("__pharmsol_y");
    let input_aliases = generate_closure_input_aliases(
        out,
        &[x.clone(), p.clone(), t.clone(), cov.clone(), y.clone()],
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
    let body = &out.body;

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
    let known_routes = route_input_idents(routes)
        .into_iter()
        .map(|route| route.to_string())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();

    for entry in entries {
        let route_name = entry.route.to_string();
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
        if property_routes.contains(&route.input.to_string())
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
    route_bindings: &[(Ident, usize)],
) -> syn::Result<TokenStream2> {
    if closure_param_names(closure).len() != 3 {
        return Err(syn::Error::new_spanned(
            closure,
            format!(
                "declaration-first `ode!` requires `{label}` to have 3 parameters: |p, t, cov|"
            ),
        ));
    }

    let route_consts = generate_mapped_index_consts(route_bindings);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let input_aliases =
        generate_closure_input_aliases(closure, &[p.clone(), t.clone(), cov.clone()])?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
    let body = &closure.body;

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
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_route_metadata(
    routes: &[OdeRouteDecl],
    diffeq_mode: OdeDiffeqMode,
    lag_routes: &HashSet<String>,
    fa_routes: &HashSet<String>,
) -> Vec<TokenStream2> {
    routes
        .iter()
        .map(|route| {
            let input = &route.input;
            let destination = &route.destination;
            let route_name = route.input.to_string();
            let route_builder = match route.kind {
                OdeRouteKind::Bolus => {
                    quote! { ::pharmsol::equation::Route::bolus(stringify!(#input)) }
                }
                OdeRouteKind::Infusion => {
                    quote! { ::pharmsol::equation::Route::infusion(stringify!(#input)) }
                }
            };
            let input_policy = match diffeq_mode {
                OdeDiffeqMode::InjectedRouteInputs => quote! { .inject_input_to_destination() },
                OdeDiffeqMode::ExplicitRouteVectors => quote! { .expect_explicit_input() },
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
                    #input_policy
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
            let route_name = route.input.to_string();
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
            let route_name = route.input.to_string();
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
    route_bindings: &[(Ident, usize)],
    dx: &Ident,
    bolus: &Ident,
    rateiv: &Ident,
) -> TokenStream2 {
    let terms = routes
        .iter()
        .zip(route_bindings.iter())
        .map(|(route, (_, channel_index))| {
            let destination = route_destination_index(route, states);
            match route.kind {
                OdeRouteKind::Bolus => quote! {
                    #dx[#destination] += #bolus[#channel_index];
                },
                OdeRouteKind::Infusion => quote! {
                    #dx[#destination] += #rateiv[#channel_index];
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
    route_bindings: &[(Ident, usize)],
    dx: &Ident,
    rateiv: &Ident,
) -> TokenStream2 {
    let terms =
        routes
            .iter()
            .zip(route_bindings.iter())
            .filter_map(|(route, (_, channel_index))| match route.kind {
                OdeRouteKind::Bolus => None,
                OdeRouteKind::Infusion => {
                    let destination = route_destination_index(route, states);
                    Some(quote! {
                        #dx[#destination] += #rateiv[#channel_index];
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
    route_bindings: &[(Ident, usize)],
) -> TokenStream2 {
    let mut destinations = vec![quote! { None }; dense_index_len(route_bindings)];

    for (route, (_, channel_index)) in routes.iter().zip(route_bindings.iter()) {
        if let OdeRouteKind::Bolus = route.kind {
            let destination = route_destination_index(route, states);
            destinations[*channel_index] = quote! { Some(#destination) };
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

fn validate_routes(routes: &[OdeRouteDecl], states: &[Ident], macro_name: &str) -> syn::Result<()> {
    let known_states = states.iter().map(Ident::to_string).collect::<HashSet<_>>();
    let mut seen_routes = HashSet::new();

    for route in routes {
        let route_name = route.input.to_string();
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
    states: &[Ident],
    routes: &[OdeRouteDecl],
    route_bindings: &[(Ident, usize)],
    diffeq_mode: OdeDiffeqMode,
) -> syn::Result<TokenStream2> {
    let state_consts = generate_index_consts(states);

    match diffeq_mode {
        OdeDiffeqMode::ExplicitRouteVectors => {
            let route_consts = generate_mapped_index_consts(route_bindings);
            let x = generated_ident("__pharmsol_x");
            let p = generated_ident("__pharmsol_p");
            let t = generated_ident("__pharmsol_t");
            let dx = generated_ident("__pharmsol_dx");
            let bolus = generated_ident("__pharmsol_bolus");
            let rateiv = generated_ident("__pharmsol_rateiv");
            let cov = generated_ident("__pharmsol_cov");
            let input_aliases = generate_closure_input_aliases(
                diffeq,
                &[
                    x.clone(),
                    p.clone(),
                    t.clone(),
                    dx.clone(),
                    bolus.clone(),
                    rateiv.clone(),
                    cov.clone(),
                ],
            )?;
            let parameter_bindings = generate_parameter_bindings(params, diffeq, &p);
            let body = &diffeq.body;

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
                    #route_consts
                    #parameter_bindings
                    #body
                };
                __pharmsol_diffeq
            }})
        }
        OdeDiffeqMode::InjectedRouteInputs => {
            let x = generated_ident("__pharmsol_x");
            let p = generated_ident("__pharmsol_p");
            let t = generated_ident("__pharmsol_t");
            let dx = generated_ident("__pharmsol_dx");
            let bolus = generated_ident("__pharmsol_bolus");
            let rateiv = generated_ident("__pharmsol_rateiv");
            let cov = generated_ident("__pharmsol_cov");
            let input_aliases = generate_closure_input_aliases(
                diffeq,
                &[x.clone(), p.clone(), t.clone(), dx.clone(), cov.clone()],
            )?;
            let parameter_bindings = generate_parameter_bindings(params, diffeq, &p);
            let body = &diffeq.body;
            let dx_binding = closure_param_ident(diffeq, 3).unwrap_or_else(|| dx.clone());
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
                    #body
                    #route_terms
                };
                __pharmsol_diffeq
            }})
        }
    }
}

fn resolve_analytical_structure(structure: &Ident) -> syn::Result<AnalyticalKernelSpec> {
    let structure_name = structure.to_string();
    let (runtime_path, metadata_kernel, parameter_arity, state_count) = match structure_name
        .as_str()
    {
        "one_compartment" => (
            quote! { ::pharmsol::equation::one_compartment },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartment },
            1,
            1,
        ),
        "one_compartment_cl" => (
            quote! { ::pharmsol::equation::one_compartment_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentCl },
            2,
            1,
        ),
        "one_compartment_cl_with_absorption" => (
            quote! { ::pharmsol::equation::one_compartment_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentClWithAbsorption },
            3,
            2,
        ),
        "one_compartment_with_absorption" => (
            quote! { ::pharmsol::equation::one_compartment_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentWithAbsorption },
            2,
            2,
        ),
        "two_compartments" => (
            quote! { ::pharmsol::equation::two_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartments },
            3,
            2,
        ),
        "two_compartments_cl" => (
            quote! { ::pharmsol::equation::two_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsCl },
            4,
            2,
        ),
        "two_compartments_cl_with_absorption" => (
            quote! { ::pharmsol::equation::two_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsClWithAbsorption },
            5,
            3,
        ),
        "two_compartments_with_absorption" => (
            quote! { ::pharmsol::equation::two_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsWithAbsorption },
            4,
            3,
        ),
        "three_compartments" => (
            quote! { ::pharmsol::equation::three_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartments },
            5,
            3,
        ),
        "three_compartments_cl" => (
            quote! { ::pharmsol::equation::three_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsCl },
            6,
            3,
        ),
        "three_compartments_cl_with_absorption" => (
            quote! { ::pharmsol::equation::three_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsClWithAbsorption },
            7,
            4,
        ),
        "three_compartments_with_absorption" => (
            quote! { ::pharmsol::equation::three_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsWithAbsorption },
            6,
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
        runtime_path,
        metadata_kernel,
        parameter_arity,
        state_count,
    })
}

fn expand_analytical_route_map(
    label: &str,
    closure: &ExprClosure,
    params: &[Ident],
    route_bindings: &[(Ident, usize)],
) -> syn::Result<TokenStream2> {
    if closure_param_names(closure).len() != 3 {
        return Err(syn::Error::new_spanned(
            closure,
            format!("built-in `analytical!` requires `{label}` to have 3 parameters: |p, t, cov|"),
        ));
    }

    let route_consts = generate_mapped_index_consts(route_bindings);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let input_aliases =
        generate_closure_input_aliases(closure, &[p.clone(), t.clone(), cov.clone()])?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
    let body = &closure.body;

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
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_analytical_init(
    init: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure_param_names(init).len() != 4 {
        return Err(syn::Error::new_spanned(
            init,
            "built-in `analytical!` requires `init` to have 4 parameters: |p, t, cov, x|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let x = generated_ident("__pharmsol_x");
    let input_aliases =
        generate_closure_input_aliases(init, &[p.clone(), t.clone(), cov.clone(), x.clone()])?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
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
            #body
        };
        __pharmsol_init
    }})
}

fn expand_analytical_out(
    out: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure_param_names(out).len() != 5 {
        return Err(syn::Error::new_spanned(
            out,
            "built-in `analytical!` requires `out` to have 5 parameters: |x, p, t, cov, y|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let output_consts = generate_index_consts(outputs);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let y = generated_ident("__pharmsol_y");
    let input_aliases = generate_closure_input_aliases(
        out,
        &[x.clone(), p.clone(), t.clone(), cov.clone(), y.clone()],
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
    let body = &out.body;

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
            #body
        };
        __pharmsol_out
    }})
}

fn expand_sde_drift(
    drift: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
    routes: &[OdeRouteDecl],
    route_bindings: &[(Ident, usize)],
) -> syn::Result<TokenStream2> {
    if closure_param_names(drift).len() != 5 {
        return Err(syn::Error::new_spanned(
            drift,
            "declaration-first `sde!` requires `drift` to have 5 parameters: |x, p, t, dx, cov|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let dx = generated_ident("__pharmsol_dx");
    let rateiv = generated_ident("__pharmsol_rateiv");
    let cov = generated_ident("__pharmsol_cov");
    let input_aliases = generate_closure_input_aliases(
        drift,
        &[x.clone(), p.clone(), t.clone(), dx.clone(), cov.clone()],
    )?;
    let parameter_bindings = generate_parameter_bindings(params, drift, &p);
    let body = &drift.body;
    let dx_binding = closure_param_ident(drift, 3).unwrap_or_else(|| dx.clone());
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
    if closure_param_names(diffusion).len() != 2 {
        return Err(syn::Error::new_spanned(
            diffusion,
            "declaration-first `sde!` requires `diffusion` to have 2 parameters: |p, sigma|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let sigma = generated_ident("__pharmsol_sigma");
    let input_aliases = generate_closure_input_aliases(diffusion, &[p.clone(), sigma.clone()])?;
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
    route_bindings: &[(Ident, usize)],
) -> syn::Result<TokenStream2> {
    if closure_param_names(closure).len() != 3 {
        return Err(syn::Error::new_spanned(
            closure,
            format!(
                "declaration-first `sde!` requires `{label}` to have 3 parameters: |p, t, cov|"
            ),
        ));
    }

    let route_consts = generate_mapped_index_consts(route_bindings);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let input_aliases =
        generate_closure_input_aliases(closure, &[p.clone(), t.clone(), cov.clone()])?;
    let parameter_bindings = generate_parameter_bindings(params, closure, &p);
    let body = &closure.body;

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
            #body
        };
        __pharmsol_route_map
    }})
}

fn expand_sde_init(
    init: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure_param_names(init).len() != 4 {
        return Err(syn::Error::new_spanned(
            init,
            "declaration-first `sde!` requires `init` to have 4 parameters: |p, t, cov, x|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let x = generated_ident("__pharmsol_x");
    let input_aliases =
        generate_closure_input_aliases(init, &[p.clone(), t.clone(), cov.clone(), x.clone()])?;
    let parameter_bindings = generate_parameter_bindings(params, init, &p);
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
            #body
        };
        __pharmsol_init
    }})
}

fn expand_sde_out(
    out: &ExprClosure,
    params: &[Ident],
    states: &[Ident],
    outputs: &[Ident],
) -> syn::Result<TokenStream2> {
    if closure_param_names(out).len() != 5 {
        return Err(syn::Error::new_spanned(
            out,
            "declaration-first `sde!` requires `out` to have 5 parameters: |x, p, t, cov, y|",
        ));
    }

    let state_consts = generate_index_consts(states);
    let output_consts = generate_index_consts(outputs);
    let x = generated_ident("__pharmsol_x");
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let y = generated_ident("__pharmsol_y");
    let input_aliases = generate_closure_input_aliases(
        out,
        &[x.clone(), p.clone(), t.clone(), cov.clone(), y.clone()],
    )?;
    let parameter_bindings = generate_parameter_bindings(params, out, &p);
    let body = &out.body;

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

    let route_bindings = ode_route_channel_bindings(&input.routes);

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
        &input.states,
        &input.routes,
        &route_bindings,
        input.diffeq_mode,
    ) {
        Ok(diffeq) => diffeq,
        Err(error) => return error.to_compile_error().into(),
    };

    let out = match expand_out(&input.out, &input.params, &input.states, &input.outputs) {
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
    let routes = expand_route_metadata(&input.routes, input.diffeq_mode, &lag_routes, &fa_routes);
    let covariate_metadata = if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    };

    let lag = match input.lag.as_ref() {
        Some(closure) => match expand_ode_route_map("lag", closure, &input.params, &route_bindings)
        {
            Ok(lag) => lag,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_ode_route_map("fa", closure, &input.params, &route_bindings) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = input
        .init
        .as_ref()
        .map_or_else(|| quote! { |_, _, _, _| {} }, |closure| quote! { #closure });

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
    let route_bindings = ode_route_channel_bindings(&input.routes);

    let kernel_spec = match resolve_analytical_structure(&input.structure) {
        Ok(spec) => spec,
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

    let out = match expand_analytical_out(&input.out, &input.params, &input.states, &input.outputs)
    {
        Ok(out) => out,
        Err(error) => return error.to_compile_error().into(),
    };

    let lag = match input.lag.as_ref() {
        Some(closure) => {
            match expand_analytical_route_map("lag", closure, &input.params, &route_bindings) {
                Ok(lag) => lag,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_analytical_route_map("fa", closure, &input.params, &route_bindings) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = match input.init.as_ref() {
        Some(closure) => match expand_analytical_init(closure, &input.params, &input.states) {
            Ok(init) => init,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _, _| {} },
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let states = &input.states;
    let outputs = &input.outputs;
    let routes = expand_analytical_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let runtime_path = kernel_spec.runtime_path;
    let metadata_kernel = kernel_spec.metadata_kernel;

    quote! {{
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Analytical)
            .parameters([#(stringify!(#params)),*])
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .analytical_kernel(#metadata_kernel);

        ::pharmsol::equation::Analytical::new(
            #runtime_path,
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
    let route_bindings = ode_route_channel_bindings(&input.routes);

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
        Some(closure) => match expand_sde_route_map("lag", closure, &input.params, &route_bindings)
        {
            Ok(lag) => lag,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let fa = match input.fa.as_ref() {
        Some(closure) => {
            match expand_sde_route_map("fa", closure, &input.params, &route_bindings) {
                Ok(fa) => fa,
                Err(error) => return error.to_compile_error().into(),
            }
        }
        None => quote! { |_, _, _| ::std::collections::HashMap::new() },
    };

    let init = match input.init.as_ref() {
        Some(closure) => match expand_sde_init(closure, &input.params, &input.states) {
            Ok(init) => init,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _, _| {} },
    };

    let out = match expand_sde_out(&input.out, &input.params, &input.states, &input.outputs) {
        Ok(out) => out,
        Err(error) => return error.to_compile_error().into(),
    };

    let nstates = input.states.len();
    let ndrugs = dense_index_len(&route_bindings);
    let nout = input.outputs.len();

    let name = &input.name;
    let params = &input.params;
    let states = &input.states;
    let outputs = &input.outputs;
    let particles = &input.particles;
    let routes = expand_sde_route_metadata(&input.routes, &lag_routes, &fa_routes);
    let bolus_mappings =
        expand_injected_sde_bolus_mappings(&input.routes, &input.states, &route_bindings);

    quote! {{
        let __pharmsol_particles: usize = #particles;
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Sde)
            .parameters([#(stringify!(#params)),*])
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
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: { infusion(iv) -> peripheral }, diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
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
            "name: \"demo\", params: [central, v], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, diffeq: |x, p, t, dx, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("parameter/state binding collisions must fail");

        assert!(error
            .to_string()
            .contains("named parameter binding `central` conflicts with named state binding"));
    }

    #[test]
    fn ode_route_bindings_share_channels_by_kind_local_ordinal() {
        let input = syn::parse_str::<OdeInput>(
            "name: \"demo\", params: [ka, ke, v], states: [depot, central], outputs: [cp], routes: { bolus(oral) -> depot, infusion(iv) -> central, bolus(sc) -> depot }, diffeq: |x, p, t, dx, b, rateiv, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .expect("declaration-first ode input should parse");

        let bindings = ode_route_channel_bindings(&input.routes);

        assert_eq!(dense_index_len(&bindings), 2);
        assert_eq!(bindings[0].0.to_string(), "oral");
        assert_eq!(bindings[0].1, 0);
        assert_eq!(bindings[1].0.to_string(), "iv");
        assert_eq!(bindings[1].1, 0);
        assert_eq!(bindings[2].0.to_string(), "sc");
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
            "name: \"demo\", params: [ka, ke, v, tlag], states: [gut, central], outputs: [cp], routes: { bolus(oral) -> gut }, structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .expect("extra declared parameters should be allowed");

        assert_eq!(input.params.len(), 4);
        assert_eq!(input.states.len(), 2);
    }

    #[test]
    fn analytical_rejects_unknown_structure() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, structure: mystery, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("unknown analytical structure must fail");

        assert!(error
            .to_string()
            .contains("unknown analytical structure `mystery`"));
    }

    #[test]
    fn analytical_rejects_insufficient_kernel_parameters() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke], states: [gut, central], outputs: [cp], routes: { bolus(oral) -> gut }, structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("insufficient kernel parameters must fail");

        assert!(error
            .to_string()
            .contains("requires at least 2 parameter value(s)"));
    }

    #[test]
    fn analytical_rejects_unknown_route_property_binding() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, v], states: [gut, central], outputs: [cp], routes: { bolus(oral) -> gut }, structure: one_compartment_with_absorption, lag: |_p, _t, _cov| { lag! { iv => 1.0 } }, out: |x, p, t, cov, y| {}",
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
            "name: \"demo\", params: [ke, v, tlag], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, structure: one_compartment, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
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
            "name: \"demo\", params: [ke, theta], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, out: |x, p, t, cov, y| {}",
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
            "name: \"demo\", params: [ke, sigma_ke], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { oral => 1.0 } }, out: |x, p, t, cov, y| {}",
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
            "name: \"demo\", params: [ke, sigma_ke, tlag], states: [central], outputs: [cp], routes: { infusion(iv) -> central }, particles: 16, drift: |x, p, t, dx, cov| {}, diffusion: |p, sigma| {}, lag: |_p, _t, _cov| { lag! { iv => tlag } }, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("infusion lag must fail");

        assert!(error
            .to_string()
            .contains("declaration-first `sde!` does not allow `lag` on infusion route `iv`"));
    }
}
