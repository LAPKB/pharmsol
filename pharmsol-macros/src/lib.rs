//! Procedural macros for [`pharmsol`](https://crates.io/crates/pharmsol).
//!
//! This crate is not intended to be used directly. Use the re-exports from the
//! `pharmsol` crate instead.

mod analysis;
mod bindings;
mod crate_path;
mod expand;
mod input;
mod kernel;
mod symbols;
mod validate;

use proc_macro::TokenStream;

use crate::input::{AnalyticalInput, OdeInput, SdeInput};

/// Implementation macro behind `pharmsol::ode!`. Call that instead: it forwards
/// `$crate` so the expansion resolves from any crate.
#[doc(hidden)]
#[proc_macro]
pub fn ode(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as OdeInput);

    match expand::ode::expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Implementation macro behind `pharmsol::analytical!`. Call that instead: it
/// forwards `$crate` so the expansion resolves from any crate.
#[doc(hidden)]
#[proc_macro]
pub fn analytical(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as AnalyticalInput);

    match expand::analytical::expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Implementation macro behind `pharmsol::sde!`. Call that instead: it forwards
/// `$crate` so the expansion resolves from any crate.
#[doc(hidden)]
#[proc_macro]
pub fn sde(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as SdeInput);

    match expand::sde::expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
