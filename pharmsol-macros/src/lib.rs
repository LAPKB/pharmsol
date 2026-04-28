//! Procedural macros for [`pharmsol`](https://crates.io/crates/pharmsol).
//!
//! This crate is not intended to be used directly. Use the re-exports from the
//! `pharmsol` crate instead.

use proc_macro::TokenStream;
use proc_macro2::TokenTree;
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    ExprClosure, Ident, Pat, Token,
};

// ---------------------------------------------------------------------------
// Macro input parsing
// ---------------------------------------------------------------------------

struct OdeInput {
    diffeq: ExprClosure,
    lag: Option<ExprClosure>,
    fa: Option<ExprClosure>,
    init: Option<ExprClosure>,
    out: ExprClosure,
}

impl Parse for OdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut diffeq = None;
        let mut lag = None;
        let mut fa_val = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let closure: ExprClosure = input.parse()?;

            match key.to_string().as_str() {
                "diffeq" => diffeq = Some(closure),
                "lag" => lag = Some(closure),
                "fa" => fa_val = Some(closure),
                "init" => init = Some(closure),
                "out" => out = Some(closure),
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!("unknown field `{other}`, expected: diffeq, lag, fa, init, out"),
                    ));
                }
            }

            // optional trailing comma
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(OdeInput {
            diffeq: diffeq.ok_or_else(|| input.error("missing required field `diffeq`"))?,
            lag,
            fa: fa_val,
            init,
            out: out.ok_or_else(|| input.error("missing required field `out`"))?,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the identifier string for a closure parameter (empty for wildcards).
fn param_name(pat: &Pat) -> String {
    match pat {
        Pat::Ident(p) => p.ident.to_string(),
        _ => String::new(),
    }
}

fn closure_param_names(c: &ExprClosure) -> Vec<String> {
    c.inputs.iter().map(param_name).collect()
}

/// Recursively scan `tokens` for `ident[literal_int]` patterns where the
/// ident matches one of `names`.  Returns the maximum literal integer found.
fn max_literal_index(tokens: proc_macro2::TokenStream, names: &[&str]) -> Option<usize> {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let mut best: Option<usize> = None;

    for (i, tt) in tts.iter().enumerate() {
        match tt {
            TokenTree::Ident(ident) => {
                let s = ident.to_string();
                if names.contains(&s.as_str()) {
                    if let Some(TokenTree::Group(g)) = tts.get(i + 1) {
                        if g.delimiter() == proc_macro2::Delimiter::Bracket {
                            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                            if inner.len() == 1 {
                                if let TokenTree::Literal(lit) = &inner[0] {
                                    if let Ok(n) = lit.to_string().parse::<usize>() {
                                        best = Some(best.map_or(n, |m: usize| m.max(n)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // recurse into brace / paren groups (bracket groups are indexing, handled above)
            TokenTree::Group(g)
                if matches!(
                    g.delimiter(),
                    proc_macro2::Delimiter::Brace | proc_macro2::Delimiter::Parenthesis
                ) =>
            {
                if let Some(n) = max_literal_index(g.stream(), names) {
                    best = Some(best.map_or(n, |m: usize| m.max(n)));
                }
            }
            _ => {}
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Proc macro
// ---------------------------------------------------------------------------

/// Build an `equation::ODE` while **inferring** `nstates`, `ndrugs` and
/// `nout` from the maximum literal bracket-indices used in the closures.
///
/// # Fields (any order, comma-separated)
///
/// | Field    | Required | Signature                                      |
/// |----------|----------|-------------------------------------------------|
/// | `diffeq` | **yes**  | `\|x, p, t, dx, bolus, rateiv, cov\| { … }`   |
/// | `out`    | **yes**  | `\|x, p, t, cov, y\| { … }`                   |
/// | `init`   | no       | `\|p, t, cov, x\| { … }`                      |
/// | `lag`    | no       | `\|p, t, cov\| lag! { … }`                    |
/// | `fa`     | no       | `\|p, t, cov\| fa! { … }`                     |
///
/// # Inference rules
///
/// * **nstates** = max literal index of the state / derivative vectors + 1
/// * **ndrugs**  = max literal index of bolus / rateiv vectors + 1
/// * **nout**    = max literal index of the output vector + 1
///
/// Parameter names are taken from the closure signatures so you can name them
/// however you like.  Only **literal** integer indices (e.g. `x[2]`) are
/// detected; computed indices require manual `.with_nstates()` etc.
///
/// # Example
///
/// ```ignore
/// use pharmsol::prelude::*;
///
/// let ode = ode! {
///     diffeq: |x, p, _t, dx, b, rateiv, _cov| {
///         fetch_params!(p, ke, kcp, kpc, _v);
///         dx[0] = rateiv[0] + b[0] - ke * x[0] - kcp * x[0] + kpc * x[1];
///         dx[1] = kcp * x[0] - kpc * x[1];
///     },
///     out: |x, p, _t, _cov, y| {
///         fetch_params!(p, _ke, _kcp, _kpc, v);
///         y[0] = x[0] / v;
///     },
/// };
/// // Inferred: nstates=2, ndrugs=1, nout=1
/// ```
#[proc_macro]
pub fn ode(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as OdeInput);

    // ── Validate parameter counts ────────────────────────────────
    let de_params = closure_param_names(&input.diffeq);
    if de_params.len() != 7 {
        return syn::Error::new_spanned(
            &input.diffeq,
            "diffeq closure must have 7 parameters: |x, p, t, dx, bolus, rateiv, cov|",
        )
        .to_compile_error()
        .into();
    }

    let out_params = closure_param_names(&input.out);
    if out_params.len() != 5 {
        return syn::Error::new_spanned(
            &input.out,
            "out closure must have 5 parameters: |x, p, t, cov, y|",
        )
        .to_compile_error()
        .into();
    }

    // ── Collect names by role ────────────────────────────────────
    //  diffeq positions: 0=x  3=dx  4=bolus  5=rateiv
    //  out    positions: 0=x  4=y
    //  init   positions: 3=x
    let mut state_names: Vec<String> = vec![
        de_params[0].clone(),
        de_params[3].clone(),
        out_params[0].clone(),
    ];
    if let Some(ref ic) = input.init {
        let ip = closure_param_names(ic);
        if ip.len() >= 4 {
            state_names.push(ip[3].clone());
        }
    }
    state_names.sort();
    state_names.dedup();

    let drug_names = [de_params[4].clone(), de_params[5].clone()];
    let output_names = [out_params[4].clone()];

    // filter empties (from wildcard `_` params)
    let state_refs: Vec<&str> = state_names
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let drug_refs: Vec<&str> = drug_names
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let output_refs: Vec<&str> = output_names
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();

    // ── Scan closure bodies ──────────────────────────────────────
    let de_tokens = input.diffeq.body.to_token_stream();
    let out_tokens = input.out.body.to_token_stream();
    let init_tokens = input.init.as_ref().map(|c| c.body.to_token_stream());

    let max_state = [
        max_literal_index(de_tokens.clone(), &state_refs),
        max_literal_index(out_tokens.clone(), &state_refs),
        init_tokens.and_then(|t| max_literal_index(t, &state_refs)),
    ]
    .into_iter()
    .flatten()
    .max();

    let max_drug = max_literal_index(de_tokens, &drug_refs);
    let max_out = max_literal_index(out_tokens, &output_refs);

    let nstates = max_state.map_or(1, |n| n + 1);
    let ndrugs = max_drug.map_or(1, |n| n + 1);
    let nout = max_out.map_or(1, |n| n + 1);

    // ── Generate output ──────────────────────────────────────────
    let diffeq = &input.diffeq;
    let out = &input.out;

    let lag = input.lag.as_ref().map_or_else(
        || quote! { |_, _, _| ::std::collections::HashMap::new() },
        |c| quote! { #c },
    );

    let fa = input.fa.as_ref().map_or_else(
        || quote! { |_, _, _| ::std::collections::HashMap::new() },
        |c| quote! { #c },
    );

    let init = input
        .init
        .as_ref()
        .map_or_else(|| quote! { |_, _, _, _| {} }, |c| quote! { #c });

    quote! {
        equation::ODE::new(
            #diffeq,
            #lag,
            #fa,
            #init,
            #out,
        )
        .with_nstates(#nstates)
        .with_ndrugs(#ndrugs)
        .with_nout(#nout)
    }
    .into()
}
