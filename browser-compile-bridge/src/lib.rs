use std::cell::RefCell;
use std::slice;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use pharmsol::dsl::WasmCompileCache;
use serde_json::json;

thread_local! {
    static LAST_RESULT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static COMPILE_CACHE: WasmCompileCache = WasmCompileCache::default();
}

fn write_result(value: serde_json::Value) {
    let bytes = serde_json::to_vec(&value).unwrap_or_else(|error| {
        format!(
            "{{\"ok\":false,\"message\":\"failed to serialize browser compile bridge response: {}\"}}",
            error
        )
        .into_bytes()
    });
    LAST_RESULT.with(|slot| {
        *slot.borrow_mut() = bytes;
    });
}

unsafe fn read_utf8(ptr: *const u8, len: usize) -> Result<String, String> {
    if len == 0 {
        return Ok(String::new());
    }
    if ptr.is_null() {
        return Err("received a null pointer for a non-empty UTF-8 input".to_string());
    }
    let bytes = slice::from_raw_parts(ptr, len);
    std::str::from_utf8(bytes)
        .map(|value| value.to_owned())
        .map_err(|error| format!("invalid UTF-8 input: {error}"))
}

fn normalize_model_name(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn cache_entry_count() -> usize {
    COMPILE_CACHE.with(|cache| cache.entry_count())
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }

    let mut bytes = Vec::<u8>::with_capacity(len);
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    drop(Vec::from_raw_parts(ptr, len, len));
}

#[no_mangle]
pub unsafe extern "C" fn compile_model(
    source_ptr: *const u8,
    source_len: usize,
    model_name_ptr: *const u8,
    model_name_len: usize,
) -> i32 {
    let source = match read_utf8(source_ptr, source_len) {
        Ok(source) => source,
        Err(message) => {
            write_result(json!({
                "ok": false,
                "message": message,
            }));
            return 1;
        }
    };

    let model_name = match read_utf8(model_name_ptr, model_name_len) {
        Ok(model_name) => normalize_model_name(model_name),
        Err(message) => {
            write_result(json!({
                "ok": false,
                "message": message,
            }));
            return 1;
        }
    };

    match COMPILE_CACHE.with(|cache| {
        cache.compile_module_source_to_wasm_module(&source, model_name.as_deref())
    }) {
        Ok(compiled) => {
            write_result(json!({
                "ok": true,
                "loaderSource": compiled.browser_loader_source,
                "wasmBase64": STANDARD.encode(&compiled.wasm_bytes),
                "cacheEntries": cache_entry_count(),
                "metadata": {
                    "abiVersion": compiled.metadata.abi_version,
                    "model": compiled.metadata.model,
                    "kernels": compiled.metadata.kernels,
                }
            }));
            0
        }
        Err(error) => {
            write_result(json!({
                "ok": false,
                "message": error.to_string(),
                "debug": format!("{error:?}"),
                "cacheEntries": cache_entry_count(),
                "diagnosticReport": error.diagnostic_report("inline.dsl"),
            }));
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn result_ptr() -> *const u8 {
    LAST_RESULT.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn result_len() -> usize {
    LAST_RESULT.with(|slot| slot.borrow().len())
}