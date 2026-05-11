//! Procedural macros for [`pharmsol`](https://crates.io/crates/pharmsol).
//!
//! This crate is not intended to be used directly. Use the re-exports from the
//! `pharmsol` crate instead.

use pharmsol_dsl::AnalyticalKernel;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
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
    covariates: Vec<Ident>,
    states: Vec<Ident>,
    outputs: Vec<SymbolicIndex>,
    routes: Vec<OdeRouteDecl>,
    structure: Ident,
    sec: Option<ExprClosure>,
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
    structure_kind: AnalyticalKernel,
    runtime_path: TokenStream2,
    metadata_kernel: TokenStream2,
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
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut routes = None;
        let mut structure = None;
        let mut sec = None;
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
                "sec" => set_once_analytical(&mut sec, input.parse()?, &key, "sec")?,
                "lag" => set_once_analytical(&mut lag, input.parse()?, &key, "lag")?,
                "fa" => set_once_analytical(&mut fa, input.parse()?, &key, "fa")?,
                "init" => set_once_analytical(&mut init, input.parse()?, &key, "init")?,
                "out" => set_once_analytical(&mut out, input.parse()?, &key, "out")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, params, covariates, states, outputs, routes, structure, sec, lag, fa, init, out"
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
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_analytical_field("states"))?;
        let outputs = outputs.ok_or_else(|| missing_required_analytical_field("outputs"))?;
        let routes = routes.ok_or_else(|| missing_required_analytical_field("routes"))?;
        let structure = structure.ok_or_else(|| missing_required_analytical_field("structure"))?;
        let out = out.ok_or_else(|| missing_required_analytical_field("out"))?;

        validate_unique_idents("parameter", &params, "analytical!")?;
        validate_unique_idents("covariate", &covariates, "analytical!")?;
        validate_unique_idents("state", &states, "analytical!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "analytical!")?;
        validate_routes(&routes, &states, "analytical!")?;

        let kernel_spec = resolve_analytical_structure(&structure)?;
        validate_analytical_structure_requirements(
            &structure,
            &params,
            &states,
            kernel_spec.structure_kind,
        )?;

        validate_analytical_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            AnalyticalBindingClosures {
                sec: sec.as_ref(),
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
            covariates,
            states,
            outputs,
            routes,
            structure,
            sec,
            lag,
            fa,
            init,
            out,
        })
    }
}

fn validate_analytical_structure_requirements(
    structure: &Ident,
    params: &[Ident],
    states: &[Ident],
    structure_kind: AnalyticalKernel,
) -> syn::Result<()> {
    let structure_name = structure_kind.name();
    if states.len() != structure_kind.state_count() {
        return Err(syn::Error::new_spanned(
            structure,
            format!(
                "analytical structure `{structure_name}` expects {} state value(s), but `states` declares {}",
                structure_kind.state_count(),
                states.len()
            ),
        ));
    }

    let declared_params = params
        .iter()
        .map(|param| param.to_string())
        .collect::<Vec<_>>();
    for required_name in structure_kind.required_parameter_names() {
        if !declared_params.iter().any(|param| param == required_name) {
            return Err(missing_required_analytical_parameter_error(
                structure,
                structure_kind,
                required_name,
                &declared_params,
            ));
        }
    }

    Ok(())
}

fn missing_required_analytical_parameter_error(
    structure: &Ident,
    structure_kind: AnalyticalKernel,
    parameter: &'static str,
    declared_params: &[String],
) -> syn::Error {
    let structure_name = structure_kind.name();

    if let Some(suggested_parameter) =
        best_analytical_parameter_suggestion(parameter, structure_kind, declared_params)
    {
        syn::Error::new_spanned(
            structure,
            format!(
                "analytical structure `{structure_name}` requires parameter `{parameter}`; did you mean `{suggested_parameter}`?"
            ),
        )
    } else {
        syn::Error::new_spanned(
            structure,
            format!(
                "analytical structure `{structure_name}` requires parameter `{parameter}`; declare it in `params: {}`",
                suggested_analytical_parameter_declaration(structure_kind, declared_params)
            ),
        )
    }
}

fn suggested_analytical_parameter_declaration(
    structure_kind: AnalyticalKernel,
    declared_params: &[String],
) -> String {
    let required_names = structure_kind.required_parameter_names();
    let mut declaration = required_names
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();

    for declared_param in declared_params {
        if !required_names.contains(&declared_param.as_str()) {
            declaration.push(declared_param.clone());
        }
    }

    format!("[{}]", declaration.join(", "))
}

fn best_analytical_parameter_suggestion(
    needle: &str,
    structure_kind: AnalyticalKernel,
    declared_params: &[String],
) -> Option<String> {
    let original_needle = needle;
    let needle = needle.to_ascii_lowercase();
    let required_names = structure_kind.required_parameter_names();
    let mut best: Option<((usize, usize, usize), &str)> = None;
    let mut tied = false;

    for declared_param in declared_params {
        let candidate = declared_param.as_str();
        if candidate == original_needle || required_names.contains(&candidate) {
            continue;
        }

        let lookup = candidate.to_ascii_lowercase();
        let distance = if is_single_adjacent_transposition(&needle, &lookup) {
            1
        } else {
            edit_distance(&needle, &lookup)
        };
        let prefix = common_prefix_len(&needle, &lookup);
        if !is_high_confidence_match(&needle, &lookup, distance, prefix) {
            continue;
        }

        let score = (
            distance,
            usize::MAX - prefix,
            needle.len().abs_diff(lookup.len()),
        );
        match &best {
            None => {
                best = Some((score, candidate));
                tied = false;
            }
            Some((best_score, _)) if score < *best_score => {
                best = Some((score, candidate));
                tied = false;
            }
            Some((best_score, _)) if score == *best_score => tied = true,
            _ => {}
        }
    }

    if tied {
        None
    } else {
        best.map(|(_, candidate)| candidate.to_string())
    }
}

fn is_high_confidence_match(needle: &str, candidate: &str, distance: usize, prefix: usize) -> bool {
    let max_len = needle.len().max(candidate.len());
    let max_distance = match max_len {
        0..=4 => 1,
        5..=8 => 2,
        _ => 3,
    };

    distance <= max_distance && (prefix > 0 || distance <= 1)
}

fn common_prefix_len(lhs: &str, rhs: &str) -> usize {
    lhs.chars()
        .zip(rhs.chars())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count()
}

fn is_single_adjacent_transposition(lhs: &str, rhs: &str) -> bool {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.len() != rhs.len() {
        return false;
    }

    let differing = lhs
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter_map(|(index, (lhs, rhs))| (lhs != rhs).then_some(index))
        .collect::<Vec<_>>();

    if differing.len() != 2 || differing[1] != differing[0] + 1 {
        return false;
    }

    let first = differing[0];
    lhs[first] == rhs[first + 1] && lhs[first + 1] == rhs[first]
}

fn edit_distance(lhs: &str, rhs: &str) -> usize {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.is_empty() {
        return rhs.len();
    }
    if rhs.is_empty() {
        return lhs.len();
    }

    let mut previous: Vec<usize> = (0..=rhs.len()).collect();
    let mut current = vec![0; rhs.len() + 1];

    for (lhs_index, lhs_char) in lhs.iter().enumerate() {
        current[0] = lhs_index + 1;
        for (rhs_index, rhs_char) in rhs.iter().enumerate() {
            let substitution_cost = usize::from(lhs_char != rhs_char);
            current[rhs_index + 1] = (current[rhs_index] + 1)
                .min(previous[rhs_index + 1] + 1)
                .min(previous[rhs_index] + substitution_cost);
        }
        previous.clone_from_slice(&current);
    }

    previous[rhs.len()]
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

fn generate_mutable_parameter_bindings(
    params: &[Ident],
    closure: &ExprClosure,
    parameter_vector: &Ident,
) -> (TokenStream2, TokenStream2) {
    let usage = ClosureBodyUsage::analyze(closure.body.as_ref());
    let used_params = params
        .iter()
        .enumerate()
        .filter(|(_, ident)| usage.uses(ident))
        .collect::<Vec<_>>();

    let bindings = used_params.iter().map(|(index, ident)| {
        quote! {
            #[allow(unused_mut, unused_variables)]
            let mut #ident = #parameter_vector[#index];
        }
    });
    let writebacks = used_params.iter().map(|(index, ident)| {
        quote! {
            #parameter_vector[#index] = #ident;
        }
    });

    (quote! { #(#bindings)* }, quote! { #(#writebacks)* })
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
    sec: Option<&'a ExprClosure>,
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
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let AnalyticalBindingClosures {
        sec,
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

    if let Some(sec) = sec {
        validate_closure_param_conflicts("sec", sec, params, "parameter")?;
        validate_closure_param_conflicts("sec", sec, covariates, "covariate")?;
    }

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

fn validate_sde_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: SdeBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
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
    let structure_kind = AnalyticalKernel::from_name(&structure_name).ok_or_else(|| {
        syn::Error::new_spanned(
            structure,
            format!("unknown analytical structure `{structure_name}`"),
        )
    })?;
    let (runtime_path, metadata_kernel) = match structure_kind {
        AnalyticalKernel::OneCompartment => (
            quote! { ::pharmsol::equation::one_compartment },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartment },
        ),
        AnalyticalKernel::OneCompartmentCl => (
            quote! { ::pharmsol::equation::one_compartment_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentCl },
        ),
        AnalyticalKernel::OneCompartmentClWithAbsorption => (
            quote! { ::pharmsol::equation::one_compartment_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentClWithAbsorption },
        ),
        AnalyticalKernel::OneCompartmentWithAbsorption => (
            quote! { ::pharmsol::equation::one_compartment_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::OneCompartmentWithAbsorption },
        ),
        AnalyticalKernel::TwoCompartments => (
            quote! { ::pharmsol::equation::two_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartments },
        ),
        AnalyticalKernel::TwoCompartmentsCl => (
            quote! { ::pharmsol::equation::two_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsCl },
        ),
        AnalyticalKernel::TwoCompartmentsClWithAbsorption => (
            quote! { ::pharmsol::equation::two_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsClWithAbsorption },
        ),
        AnalyticalKernel::TwoCompartmentsWithAbsorption => (
            quote! { ::pharmsol::equation::two_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::TwoCompartmentsWithAbsorption },
        ),
        AnalyticalKernel::ThreeCompartments => (
            quote! { ::pharmsol::equation::three_compartments },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartments },
        ),
        AnalyticalKernel::ThreeCompartmentsCl => (
            quote! { ::pharmsol::equation::three_compartments_cl },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsCl },
        ),
        AnalyticalKernel::ThreeCompartmentsClWithAbsorption => (
            quote! { ::pharmsol::equation::three_compartments_cl_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsClWithAbsorption },
        ),
        AnalyticalKernel::ThreeCompartmentsWithAbsorption => (
            quote! { ::pharmsol::equation::three_compartments_with_absorption },
            quote! { ::pharmsol::equation::AnalyticalKernel::ThreeCompartmentsWithAbsorption },
        ),
    };

    Ok(AnalyticalKernelSpec {
        structure_kind,
        runtime_path,
        metadata_kernel,
    })
}

fn expand_analytical_route_map(
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
            "built-in `analytical!` requires `{label}` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|"
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

fn expand_analytical_sec(
    sec: &ExprClosure,
    params: &[Ident],
    covariates: &[Ident],
) -> syn::Result<TokenStream2> {
    let p = generated_ident("__pharmsol_p");
    let t = generated_ident("__pharmsol_t");
    let cov = generated_ident("__pharmsol_cov");
    let full_inputs = [p.clone(), t.clone(), cov.clone()];
    let reduced_inputs = [t.clone()];
    let input_aliases = generate_supported_input_aliases(
        sec,
        &[&full_inputs, &reduced_inputs],
        "built-in `analytical!` requires `sec` to have either 3 parameters: |p, t, cov| or 1 parameter: |t|",
    )?;
    let parameter_vector = if sec.inputs.len() == full_inputs.len() {
        closure_param_ident(sec, 0).unwrap_or_else(|| p.clone())
    } else {
        p.clone()
    };
    let (parameter_bindings, parameter_writebacks) =
        generate_mutable_parameter_bindings(params, sec, &parameter_vector);
    let covariate_bindings = generate_covariate_bindings(covariates, sec, &cov, &t);
    let body = &sec.body;

    Ok(quote! {{
        let __pharmsol_sec: fn(
            &mut ::pharmsol::simulator::V,
            f64,
            &::pharmsol::data::Covariates,
        ) = |#p: &mut ::pharmsol::simulator::V,
             #t: f64,
             #cov: &::pharmsol::data::Covariates| {
            #input_aliases
            #parameter_bindings
            #covariate_bindings
            #body
            #parameter_writebacks
        };
        __pharmsol_sec
    }})
}

fn expand_analytical_init(
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
        "built-in `analytical!` requires `init` to have either 4 parameters: |p, t, cov, x| or 2 parameters: |t, x|",
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

fn expand_analytical_out(
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
        "built-in `analytical!` requires `out` to have either 5 parameters: |x, p, t, cov, y| or 3 parameters: |x, t, y|",
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

    let sec = match input.sec.as_ref() {
        Some(closure) => match expand_analytical_sec(closure, &input.params, &input.covariates) {
            Ok(sec) => sec,
            Err(error) => return error.to_compile_error().into(),
        },
        None => quote! { |_, _, _| {} },
    };

    let out = match expand_analytical_out(
        &input.out,
        &input.params,
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
            match expand_analytical_init(closure, &input.params, &input.covariates, &input.states) {
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
    let runtime_path = kernel_spec.runtime_path;
    let metadata_kernel = kernel_spec.metadata_kernel;
    let covariate_metadata = if covariates.is_empty() {
        quote! {}
    } else {
        quote! {
            .covariates([#(::pharmsol::equation::Covariate::continuous(stringify!(#covariates))),*])
        }
    };

    quote! {{
        let __pharmsol_metadata = ::pharmsol::equation::metadata::new(#name)
            .kind(::pharmsol::equation::ModelKind::Analytical)
            .parameters([#(stringify!(#params)),*])
            #covariate_metadata
            .states([#(stringify!(#states)),*])
            .outputs([#(stringify!(#outputs)),*])
            #(.route(#routes))*
            .analytical_kernel(#metadata_kernel);

        ::pharmsol::equation::Analytical::new(
            #runtime_path,
            #sec,
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
            "name: \"demo\", params: [ka, ke, v, tlag, tvke], covariates: [wt, renal], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, sec: |_t| { ke = tvke; }, out: |x, p, t, cov, y| {}",
        )
        .expect("extra declared parameters should be allowed");

        assert_eq!(input.params.len(), 5);
        assert_eq!(input.covariates.len(), 2);
        assert!(input.sec.is_some());
        assert_eq!(input.states.len(), 2);
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
    fn analytical_accepts_declared_parameter_order_different_from_structure_order() {
        let input = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, ke, k12, k21, v], states: [gut, central, peripheral], outputs: [cp], routes: [bolus(oral) -> gut], structure: two_compartments_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .expect("declared-name parameter order should be accepted");

        assert_eq!(input.params[0].to_string(), "ka");
        assert_eq!(input.params[1].to_string(), "ke");
        assert_eq!(input.states.len(), 3);
    }

    #[test]
    fn analytical_rejects_missing_required_parameter_with_suggestion() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ka, kel, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing required parameter must fail");

        assert!(error
            .to_string()
            .contains("requires parameter `ke`; did you mean `kel`?"));
    }

    #[test]
    fn analytical_rejects_missing_required_parameter_without_suggestion() {
        let error = syn::parse_str::<AnalyticalInput>(
            "name: \"demo\", params: [ke, v], states: [gut, central], outputs: [cp], routes: [bolus(oral) -> gut], structure: one_compartment_with_absorption, out: |x, p, t, cov, y| {}",
        )
        .err()
        .expect("missing required parameter must fail");

        assert!(error
            .to_string()
            .contains("requires parameter `ka`; declare it in `params: [ka, ke, v]`"));
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
