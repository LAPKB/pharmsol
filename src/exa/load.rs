use libloading::{Library, Symbol};
use std::path::PathBuf;

use crate::{EqnKind, Equation, Meta};

pub unsafe fn get_kind(model_path: PathBuf) -> EqnKind {
    let lib = unsafe { Library::new(model_path).expect("Failed to load library") };
    let get_kind: Symbol<unsafe extern "C" fn() -> EqnKind> =
        unsafe { lib.get(b"equation_kind").expect("Failed to load symbol") };
    let kind = unsafe { get_kind() };
    kind.clone()
}

/// Loads an ODE (Ordinary Differential Equation) and its metadata from a given library.
///
/// # Safety
///
/// This function is unsafe because:
/// - It involves FFI (Foreign Function Interface) calls and raw pointer manipulation.
/// - The returned `ODE` and `Meta` instances will become invalid if the `Library` is dropped.
///   The caller must ensure that the `Library` remains alive as long as the `ODE` and `Meta`
///   are being used.
///
/// # Arguments
///
/// * `model_path` - Path to the compiled model library file.
///
/// # Returns
///
/// A tuple containing the `Library` and a pair of (`ODE`, `Meta`). The `Library` must be kept
/// in scope as long as the `ODE` and `Meta` are used.
pub unsafe fn load<E: Equation>(model_path: PathBuf) -> (Library, (E, Meta)) {
    let lib = unsafe { Library::new(model_path).expect("Failed to load library") };

    let create_eqn: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"create_eqn_ptr").expect("Failed to load symbol") };
    let eqn_ptr = unsafe { create_eqn() };

    let create_meta: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"metadata_ptr").expect("Failed to load symbol") };
    let meta_ptr = unsafe { create_meta() };

    unsafe {
        (
            lib,
            (
                (&*(eqn_ptr as *mut E)).clone(),
                (&*(meta_ptr as *mut Meta)).clone(),
            ),
        )
    }
}

/// Retrieves the model parameters from a dynamically loaded model.
///
/// # Safety
///
/// This function is unsafe for the same reasons as `load_ode`. It internally calls `load_ode`
/// which performs FFI calls and pointer manipulation.
///
/// # Arguments
///
/// * `model_path` - Path to the compiled model library file.
///
/// # Returns
///
/// A vector of strings representing the model parameters.
pub unsafe fn model_parameters<E: Equation>(model_path: PathBuf) -> Vec<String> {
    let (_, (_, meta)) = unsafe { load::<E>(model_path) };
    meta.get_params().clone()
}
