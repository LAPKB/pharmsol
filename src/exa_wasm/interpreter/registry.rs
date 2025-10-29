use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::exa_wasm::interpreter::ast::Expr;

#[derive(Clone, Debug)]
pub struct RegistryEntry {
    pub dx: HashMap<usize, Expr>,
    pub out: HashMap<usize, Expr>,
    pub init: HashMap<usize, Expr>,
    pub lag: HashMap<usize, Expr>,
    pub fa: HashMap<usize, Expr>,
    pub pmap: HashMap<String, usize>,
    pub nstates: usize,
    pub _nouteqs: usize,
}

static EXPR_REGISTRY: Lazy<Mutex<HashMap<usize, RegistryEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static NEXT_EXPR_ID: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(1));

thread_local! {
    static CURRENT_EXPR_ID: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
    static LAST_RUNTIME_ERROR: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

pub fn set_current_expr_id(id: Option<usize>) -> Option<usize> {
    let prev = CURRENT_EXPR_ID.with(|c| {
        let p = c.get();
        c.set(id);
        p
    });
    prev
}

pub fn current_expr_id() -> Option<usize> {
    CURRENT_EXPR_ID.with(|c| c.get())
}

pub fn set_runtime_error(msg: String) {
    LAST_RUNTIME_ERROR.with(|c| {
        *c.borrow_mut() = Some(msg);
    });
}

pub fn take_runtime_error() -> Option<String> {
    LAST_RUNTIME_ERROR.with(|c| c.borrow_mut().take())
}

pub fn register_entry(entry: RegistryEntry) -> usize {
    let id = NEXT_EXPR_ID.fetch_add(1, Ordering::SeqCst);
    let mut guard = EXPR_REGISTRY.lock().unwrap();
    guard.insert(id, entry);
    id
}

pub fn unregister_model(id: usize) {
    let mut guard = EXPR_REGISTRY.lock().unwrap();
    guard.remove(&id);
}

pub fn get_entry(id: usize) -> Option<RegistryEntry> {
    let guard = EXPR_REGISTRY.lock().unwrap();
    guard.get(&id).cloned()
}

pub fn ode_for_id(id: usize) -> Option<crate::simulator::equation::ODE> {
    if let Some(entry) = get_entry(id) {
        let nstates = entry.nstates;
        let nouteqs = entry._nouteqs;
        let ode = crate::simulator::equation::ODE::with_registry_id(
            crate::exa_wasm::interpreter::dispatch::diffeq_dispatch,
            crate::exa_wasm::interpreter::dispatch::lag_dispatch,
            crate::exa_wasm::interpreter::dispatch::fa_dispatch,
            crate::exa_wasm::interpreter::dispatch::init_dispatch,
            crate::exa_wasm::interpreter::dispatch::out_dispatch,
            (nstates, nouteqs),
            Some(id),
        );
        Some(ode)
    } else {
        None
    }
}
