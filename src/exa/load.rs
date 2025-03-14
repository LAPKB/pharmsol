use libloading::{Library, Symbol};
use std::path::PathBuf;

use crate::{Meta, ODE};

/// Loads an ODE (Ordinary Differential Equation) and its metadata from a given library.
///
/// # Safety
///
/// This function is unsafe because it involves FFI (Foreign Function Interface) calls
/// and raw pointer manipulation.
///
/// # Arguments
///
/// * `lib` - A reference to the library from which to load the equation and metadata.
///
/// # Returns
///
/// A tuple containing the ODE and its associated metadata.
pub unsafe fn load_ode(lib: &Library) -> (ODE, Meta) {
    let create_eqn: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"create_eqn_ptr").expect("Failed to load symbol") };
    let eqn_ptr = unsafe { create_eqn() };

    let create_meta: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"metadata_ptr").expect("Failed to load symbol") };
    let meta_ptr = unsafe { create_meta() };

    unsafe {
        (
            (&*(eqn_ptr as *mut ODE)).clone(),
            (&*(meta_ptr as *mut Meta)).clone(),
        )
    }
}

/// Retrieves the model parameters from a dynamically loaded model.
///
/// # Arguments
///
/// * `model_path` - Path to the compiled model library file.
///
/// # Returns
///
/// A vector of strings representing the model parameters.
pub unsafe fn model_parameters(model_path: PathBuf) -> Vec<String> {
    let lib = unsafe { Library::new(model_path).expect("Failed to load library") };
    let (_, meta) = unsafe { load_ode(&lib) };
    meta.get_params().clone()
}
