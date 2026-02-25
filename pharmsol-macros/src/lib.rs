//! Procedural macros for the pharmsol pharmacometric modeling library.
//!
//! This crate provides the `pk_model!` proc macro that enables a clean,
//! declarative DSL for defining pharmacokinetic ODE models.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    braced, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Expr, Ident, Result, Token,
};

/// A single lag or fa mapping entry: `compartment_index => expression`
struct LagFaEntry {
    key: Expr,
    value: Expr,
}

/// The fully parsed DSL input for `pk_model!`.
struct PkModelInput {
    params: Vec<Ident>,
    covariates: Vec<Ident>,
    lag_entries: Vec<LagFaEntry>,
    fa_entries: Vec<LagFaEntry>,
    init_body: Option<TokenStream2>,
    diffeq_body: TokenStream2,
    out_body: TokenStream2,
    nstates: Expr,
    nout: Expr,
}

/// Check if the next token is an identifier matching `keyword`.
fn peek_keyword(input: ParseStream, keyword: &str) -> bool {
    let fork = input.fork();
    fork.parse::<Ident>()
        .map(|i| i == keyword)
        .unwrap_or(false)
}

/// Parse the contents of a `{ key => value, ... }` block for lag/fa sections.
fn parse_lag_fa_entries(input: ParseStream) -> Result<Vec<LagFaEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let key: Expr = content.parse()?;
        let _: Token![=>] = content.parse()?;
        let value: Expr = content.parse()?;
        entries.push(LagFaEntry { key, value });
        if content.peek(Token![,]) {
            let _: Token![,] = content.parse()?;
        }
    }
    Ok(entries)
}

impl Parse for PkModelInput {
    fn parse(input: ParseStream) -> Result<Self> {
        // --- params: (ident, ident, ...) ---
        let kw: Ident = input.parse()?;
        if kw != "params" {
            return Err(syn::Error::new(kw.span(), "expected 'params'"));
        }
        let _: Token![:] = input.parse()?;
        let params_paren;
        parenthesized!(params_paren in input);
        let params = Punctuated::<Ident, Token![,]>::parse_terminated(&params_paren)?;
        let params: Vec<Ident> = params.into_iter().collect();
        let _: Token![,] = input.parse()?;

        // --- Optional: covariates: (ident, ident, ...) ---
        let mut covariates = Vec::new();
        if peek_keyword(input, "covariates") {
            let _: Ident = input.parse()?;
            let _: Token![:] = input.parse()?;
            let cov_paren;
            parenthesized!(cov_paren in input);
            let covs = Punctuated::<Ident, Token![,]>::parse_terminated(&cov_paren)?;
            covariates = covs.into_iter().collect();
            let _: Token![,] = input.parse()?;
        }

        // --- Optional: lag: { key => value, ... } ---
        let mut lag_entries = Vec::new();
        if peek_keyword(input, "lag") {
            let _: Ident = input.parse()?;
            let _: Token![:] = input.parse()?;
            lag_entries = parse_lag_fa_entries(input)?;
            let _: Token![,] = input.parse()?;
        }

        // --- Optional: fa: { key => value, ... } ---
        let mut fa_entries = Vec::new();
        if peek_keyword(input, "fa") {
            let _: Ident = input.parse()?;
            let _: Token![:] = input.parse()?;
            fa_entries = parse_lag_fa_entries(input)?;
            let _: Token![,] = input.parse()?;
        }

        // --- Optional: init: { ... } ---
        let mut init_body = None;
        if peek_keyword(input, "init") {
            let _: Ident = input.parse()?;
            let _: Token![:] = input.parse()?;
            let content;
            braced!(content in input);
            init_body = Some(content.parse()?);
            let _: Token![,] = input.parse()?;
        }

        // --- Required: diffeq: { ... } ---
        let diffeq_kw: Ident = input.parse()?;
        if diffeq_kw != "diffeq" {
            return Err(syn::Error::new(
                diffeq_kw.span(),
                format!("expected 'diffeq', found '{}'", diffeq_kw),
            ));
        }
        let _: Token![:] = input.parse()?;
        let diffeq_content;
        braced!(diffeq_content in input);
        let diffeq_body: TokenStream2 = diffeq_content.parse()?;
        let _: Token![,] = input.parse()?;

        // --- Required: out: { ... } ---
        let out_kw: Ident = input.parse()?;
        if out_kw != "out" {
            return Err(syn::Error::new(
                out_kw.span(),
                format!("expected 'out', found '{}'", out_kw),
            ));
        }
        let _: Token![:] = input.parse()?;
        let out_content;
        braced!(out_content in input);
        let out_body: TokenStream2 = out_content.parse()?;
        let _: Token![,] = input.parse()?;

        // --- Required: neqs: (nstates, nout) ---
        let neqs_kw: Ident = input.parse()?;
        if neqs_kw != "neqs" {
            return Err(syn::Error::new(
                neqs_kw.span(),
                format!("expected 'neqs', found '{}'", neqs_kw),
            ));
        }
        let _: Token![:] = input.parse()?;
        let neqs_paren;
        parenthesized!(neqs_paren in input);
        let nstates: Expr = neqs_paren.parse()?;
        let _: Token![,] = neqs_paren.parse()?;
        let nout: Expr = neqs_paren.parse()?;

        // Optional trailing comma
        if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
        }

        Ok(PkModelInput {
            params,
            covariates,
            lag_entries,
            fa_entries,
            init_body,
            diffeq_body,
            out_body,
            nstates,
            nout,
        })
    }
}

/// Generate the `ODE::new(...)` expression from the parsed model definition.
fn generate_ode(model: PkModelInput) -> TokenStream2 {
    let params = &model.params;
    let covariates = &model.covariates;

    // fetch_params! is identical in every closure
    let fetch_params = quote! {
        ::pharmsol::fetch_params!(__pk_p, #(#params),*);
    };

    // fetch_cov! uses `t` (the closure parameter exposed to user code)
    let fetch_covs = if !covariates.is_empty() {
        quote! { ::pharmsol::fetch_cov!(__pk_cov, t, #(#covariates),*); }
    } else {
        quote! {}
    };

    // --- Lag HashMap ---
    let lag_body = if model.lag_entries.is_empty() {
        quote! { ::core::convert::From::from([]) }
    } else {
        let entries: Vec<_> = model
            .lag_entries
            .iter()
            .map(|e| {
                let k = &e.key;
                let v = &e.value;
                quote! { (#k, #v) }
            })
            .collect();
        quote! { ::core::convert::From::from([#(#entries),*]) }
    };

    // --- Fa HashMap ---
    let fa_body = if model.fa_entries.is_empty() {
        quote! { ::core::convert::From::from([]) }
    } else {
        let entries: Vec<_> = model
            .fa_entries
            .iter()
            .map(|e| {
                let k = &e.key;
                let v = &e.value;
                quote! { (#k, #v) }
            })
            .collect();
        quote! { ::core::convert::From::from([#(#entries),*]) }
    };

    // --- Init body ---
    let init_code = model
        .init_body
        .as_ref()
        .map_or_else(|| quote! {}, |body| quote! { #body });

    let diffeq_body = &model.diffeq_body;
    let out_body = &model.out_body;
    let nstates = &model.nstates;
    let nout = &model.nout;

    quote! {
        ::pharmsol::simulator::equation::ODE::new(
            // DiffEq: fn(&V, &V, T, &mut V, &V, &V, &Covariates)
            |x, __pk_p, t, dx, bolus, rateiv, __pk_cov| {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                {
                    #fetch_params
                    #fetch_covs
                    #diffeq_body
                }
            },
            // Lag: fn(&V, T, &Covariates) -> HashMap<usize, T>
            |__pk_p, t, __pk_cov| {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                {
                    #fetch_params
                    #fetch_covs
                    #lag_body
                }
            },
            // Fa: fn(&V, T, &Covariates) -> HashMap<usize, T>
            |__pk_p, t, __pk_cov| {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                {
                    #fetch_params
                    #fetch_covs
                    #fa_body
                }
            },
            // Init: fn(&V, T, &Covariates, &mut V)
            |__pk_p, t, __pk_cov, x| {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                {
                    #fetch_params
                    #fetch_covs
                    #init_code
                }
            },
            // Out: fn(&V, &V, T, &Covariates, &mut V)
            |x, __pk_p, t, __pk_cov, y| {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                {
                    #fetch_params
                    #fetch_covs
                    #out_body
                }
            },
            (#nstates, #nout),
        )
    }
}

/// Define a pharmacokinetic ODE model using a clean, declarative syntax.
///
/// This macro generates an `equation::ODE` by automatically wiring up the
/// differential equations, output equations, and optional components (lag times,
/// bioavailability, initial conditions, covariates).
///
/// Parameters are declared once and automatically available by name in all blocks.
/// Covariates (when declared) are automatically interpolated at the current time `t`.
///
/// # Sections
///
/// | Section | Required | Description |
/// |---|---|---|
/// | `params` | **Yes** | Named model parameters (positional from support point vector) |
/// | `covariates` | No | Named covariates (auto-interpolated at time `t`) |
/// | `lag` | No | Lag times per compartment: `{ 0 => tlag }` |
/// | `fa` | No | Bioavailability per compartment: `{ 0 => 0.8 }` |
/// | `init` | No | Initial state: write to `x[i]` |
/// | `diffeq` | **Yes** | ODE right-hand side: write to `dx[i]` |
/// | `out` | **Yes** | Output equations: write to `y[i]` |
/// | `neqs` | **Yes** | `(number_of_states, number_of_outputs)` |
///
/// # Built-in variables
///
/// These variables are automatically available in each block:
///
/// | Variable | Type | Available in | Description |
/// |---|---|---|---|
/// | `x` | state vector | `diffeq`, `out`, `init` | Compartment amounts |
/// | `dx` | derivatives | `diffeq` | Rate of change for each compartment |
/// | `y` | output vector | `out` | Predicted observations |
/// | `t` | `f64` | all blocks | Current time |
/// | `rateiv` | vector | `diffeq` | Current infusion rates per compartment |
/// | `bolus` | vector | `diffeq` | Current bolus amounts per compartment |
/// | *params* | `f64` | all blocks | Each declared parameter by name |
/// | *covariates* | `f64` | all blocks | Each declared covariate by name |
///
/// # Examples
///
/// ## One-compartment IV bolus
/// ```rust
/// use pharmsol::prelude::*;
///
/// let ode = pk_model! {
///     params: (ke, v),
///     diffeq: {
///         dx[0] = -ke * x[0];
///     },
///     out: {
///         y[0] = x[0] / v;
///     },
///     neqs: (1, 1),
/// };
/// ```
///
/// ## Two-compartment with absorption and lag
/// ```rust
/// use pharmsol::prelude::*;
///
/// let ode = pk_model! {
///     params: (ka, ke, tlag, v),
///     lag: { 0 => tlag },
///     diffeq: {
///         dx[0] = -ka * x[0];
///         dx[1] = ka * x[0] - ke * x[1];
///     },
///     out: {
///         y[0] = x[1] / v;
///     },
///     neqs: (2, 1),
/// };
/// ```
///
/// ## With covariates and infusion
/// ```rust
/// use pharmsol::prelude::*;
///
/// let ode = pk_model! {
///     params: (cl, v),
///     covariates: (wt),
///     diffeq: {
///         let ke = cl * (wt / 70.0).powf(0.75) / (v * wt / 70.0);
///         dx[0] = -ke * x[0] + rateiv[0];
///     },
///     out: {
///         let v_adj = v * wt / 70.0;
///         y[0] = x[0] / v_adj;
///     },
///     neqs: (1, 1),
/// };
/// ```
#[proc_macro]
pub fn pk_model(input: TokenStream) -> TokenStream {
    let model = syn::parse_macro_input!(input as PkModelInput);
    generate_ode(model).into()
}
