//! Lookup table mapping `structure: ...` names onto the built-in analytical
//! kernels.

use pharmsol_dsl::AnalyticalKernel as ResolverAnalyticalKernel;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;

pub(crate) struct AnalyticalKernelSpec {
    pub(crate) function: ResolverAnalyticalKernel,
    pub(crate) runtime_path: TokenStream2,
    pub(crate) metadata_function: TokenStream2,
    pub(crate) state_count: usize,
}

pub(crate) fn resolve_analytical_structure(structure: &Ident) -> syn::Result<AnalyticalKernelSpec> {
    let structure_name = structure.to_string();
    let (function, runtime_path, metadata_function, state_count) = match structure_name.as_str() {
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
        function,
        runtime_path,
        metadata_function,
        state_count,
    })
}
