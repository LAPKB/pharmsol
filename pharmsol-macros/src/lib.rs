//! Procedural macros for pharmsol
//!
//! This crate provides derive macros for ergonomic PK/PD model definitions:
//!
//! - `#[derive(Params)]` - Define model parameters with named access
//! - `#[derive(Covariates)]` - Define covariates with automatic interpolation
//!
//! # Example
//!
//! ```ignore
//! use pharmsol::prelude::*;
//!
//! #[derive(Params)]
//! struct Pk {
//!     ke: f64,
//!     v: f64,
//! }
//!
//! #[derive(Covariates)]
//! struct Cov {
//!     wt: f64,
//!     age: f64,
//! }
//!
//! let ode = ODE::<Pk, Cov>::builder()
//!     .diffeq(diffeq! {
//!         dx[0] = -ke * x[0] + rateiv[0];
//!     })
//!     .out(out! {
//!         y[0] = x[0] / v;
//!     })
//!     .nstates(1)
//!     .nouteqs(1)
//!     .build();
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derive macro for model parameters.
///
/// Generates:
/// - `Params` trait implementation
/// - `From<HashMap<String, f64>>` implementation
/// - `From<&[f64]>` implementation (using field order)
/// - A helper macro `{StructName}_params!` for destructuring
///
/// # Example
///
/// ```ignore
/// #[derive(Params)]
/// struct Pk {
///     ke: f64,
///     v: f64,
/// }
///
/// // Can create from HashMap
/// let params: Pk = HashMap::from([("ke", 1.0), ("v", 100.0)]).into();
///
/// // Or from slice (uses field order)
/// let params: Pk = [1.0, 100.0].as_slice().into();
/// ```
#[proc_macro_derive(Params)]
pub fn derive_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Params can only be derived for structs with named fields"),
        },
        _ => panic!("Params can only be derived for structs"),
    };

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_strings: Vec<_> = field_names.iter().map(|n| n.to_string()).collect();
    let field_indices: Vec<_> = (0..field_names.len()).collect();
    let num_fields = field_names.len();

    // Generate the macro name for this struct
    let macro_name = format_ident!("{}_params", name);
    let destructure_macro_name = format_ident!("__destructure_{}", name);

    let expanded = quote! {
        impl pharmsol::Params for #name {
            fn field_names() -> &'static [&'static str] {
                &[#(#field_strings),*]
            }

            fn num_params() -> usize {
                #num_fields
            }

            fn from_slice(values: &[f64]) -> Self {
                assert_eq!(values.len(), #num_fields,
                    "Expected {} parameters, got {}", #num_fields, values.len());
                Self {
                    #(#field_names: values[#field_indices]),*
                }
            }

            fn to_vec(&self) -> Vec<f64> {
                vec![#(self.#field_names),*]
            }
        }

        impl From<std::collections::HashMap<String, f64>> for #name {
            fn from(map: std::collections::HashMap<String, f64>) -> Self {
                Self {
                    #(#field_names: *map.get(#field_strings)
                        .unwrap_or_else(|| panic!("Missing parameter: {}", #field_strings))),*
                }
            }
        }

        impl From<std::collections::HashMap<&str, f64>> for #name {
            fn from(map: std::collections::HashMap<&str, f64>) -> Self {
                Self {
                    #(#field_names: *map.get(#field_strings)
                        .unwrap_or_else(|| panic!("Missing parameter: {}", #field_strings))),*
                }
            }
        }

        impl From<&[f64]> for #name {
            fn from(slice: &[f64]) -> Self {
                <#name as pharmsol::Params>::from_slice(slice)
            }
        }

        impl From<Vec<f64>> for #name {
            fn from(vec: Vec<f64>) -> Self {
                <#name as pharmsol::Params>::from_slice(&vec)
            }
        }

        /// Macro to destructure parameters into local variables (legacy)
        #[macro_export]
        macro_rules! #macro_name {
            ($p:expr) => {
                let #name { #(#field_names),* } = $p;
            };
        }

        /// Internal macro for destructuring params - used by diffeq!/out! macros
        #[macro_export]
        #[doc(hidden)]
        macro_rules! #destructure_macro_name {
            ($p:expr) => {
                #[allow(unused_variables)]
                let #name { #(#field_names),* } = $p;
            };
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for covariates.
///
/// Generates:
/// - `Covariates` trait implementation
/// - Methods to extract and interpolate covariate values at a given time
/// - A helper macro `{StructName}_cov!` for destructuring
///
/// # Example
///
/// ```ignore
/// #[derive(Covariates)]
/// struct Cov {
///     wt: f64,
///     age: f64,
/// }
/// ```
#[proc_macro_derive(Covariates)]
pub fn derive_covariates(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Covariates can only be derived for structs with named fields"),
        },
        _ => panic!("Covariates can only be derived for structs"),
    };

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_strings: Vec<_> = field_names.iter().map(|n| n.to_string()).collect();
    let num_fields = field_names.len();

    // Generate the macro name for this struct
    let macro_name = format_ident!("{}_cov", name);

    let expanded = quote! {
        impl pharmsol::CovariateSet for #name {
            fn field_names() -> &'static [&'static str] {
                &[#(#field_strings),*]
            }

            fn num_covariates() -> usize {
                #num_fields
            }

            fn from_covariates(cov: &pharmsol::Covariates, t: f64) -> Result<Self, pharmsol::PharmsolError> {
                Ok(Self {
                    #(#field_names: cov.get_covariate(#field_strings)
                        .ok_or_else(|| pharmsol::PharmsolError::CovariateNotFound(#field_strings.to_string()))?
                        .interpolate(t)?),*
                })
            }
        }

        /// Macro to destructure covariates into local variables
        #[macro_export]
        macro_rules! #macro_name {
            ($c:expr) => {
                let #name { #(#field_names),* } = $c;
            };
        }
    };

    TokenStream::from(expanded)
}
