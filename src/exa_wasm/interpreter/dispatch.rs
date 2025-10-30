use diffsol::Vector;
use std::collections::HashMap;

use crate::exa_wasm::interpreter::registry;

fn current_id() -> Option<usize> {
    registry::current_expr_id()
}

pub fn diffeq_dispatch(
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    dx: &mut crate::simulator::V,
    _bolus: crate::simulator::V,
    rateiv: crate::simulator::V,
    _cov: &crate::data::Covariates,
) {
    if let Some(id) = current_id() {
        if let Some(entry) = registry::get_entry(id) {
            // execute prelude assignments in order, storing values in locals
            let mut locals: HashMap<String, f64> = HashMap::new();
            for (name, expr) in entry.prelude.iter() {
                let val = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    x,
                    p,
                    &rateiv,
                    Some(&locals),
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                locals.insert(name.clone(), val.as_number());
            }
            // debug: print locals to stderr to verify prelude execution
            if !locals.is_empty() {
                // eprintln!("[exa_wasm prelude locals] {:?}", locals);
            }
            // execute statement ASTs which may assign to dx indices or locals
            let mut assign_closure = |name: &str, idx: usize, val: f64| match name {
                "dx" => {
                    if idx < dx.len() {
                        dx[idx] = val;
                    } else {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "index out of bounds 'dx'[{}] (nstates={})",
                            idx,
                            dx.len()
                        ));
                    }
                }
                _ => {
                    crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                        "unsupported indexed assignment '{}' in diffeq",
                        name
                    ));
                }
            };
            for st in entry.diffeq_stmts.iter() {
                crate::exa_wasm::interpreter::eval::eval_stmt(
                    st,
                    x,
                    p,
                    _t,
                    &rateiv,
                    &mut locals,
                    Some(&entry.pmap),
                    Some(_cov),
                    &mut assign_closure,
                );
            }
        }
    }
}

pub fn out_dispatch(
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
    y: &mut crate::simulator::V,
) {
    let tmp = crate::simulator::V::zeros(1, diffsol::NalgebraContext);
    if let Some(id) = current_id() {
        if let Some(entry) = registry::get_entry(id) {
            // execute out statements, allowing writes to y[]
            let mut assign = |name: &str, idx: usize, val: f64| match name {
                "y" => {
                    if idx < y.len() {
                        y[idx] = val;
                    } else {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "index out of bounds 'y'[{}] (nouteqs={})",
                            idx,
                            y.len()
                        ));
                    }
                }
                _ => {
                    crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                        "unsupported indexed assignment '{}' in out",
                        name
                    ));
                }
            };
            for st in entry.out_stmts.iter() {
                crate::exa_wasm::interpreter::eval::eval_stmt(
                    st,
                    x,
                    p,
                    _t,
                    &tmp,
                    &mut std::collections::HashMap::new(),
                    Some(&entry.pmap),
                    Some(_cov),
                    &mut assign,
                );
            }
        }
    }
}

pub fn lag_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
) -> std::collections::HashMap<usize, crate::simulator::T> {
    let mut out: std::collections::HashMap<usize, crate::simulator::T> =
        std::collections::HashMap::new();
    if let Some(id) = current_id() {
        if let Some(entry) = registry::get_entry(id) {
            let zero_x = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            for (i, expr) in entry.lag.iter() {
                let v = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    &zero_x,
                    p,
                    &zero_rate,
                    None,
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                out.insert(*i, v.as_number());
            }
        }
    }
    out
}

pub fn fa_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
) -> std::collections::HashMap<usize, crate::simulator::T> {
    let mut out: std::collections::HashMap<usize, crate::simulator::T> =
        std::collections::HashMap::new();
    if let Some(id) = current_id() {
        if let Some(entry) = registry::get_entry(id) {
            let zero_x = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            for (i, expr) in entry.fa.iter() {
                let v = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    &zero_x,
                    p,
                    &zero_rate,
                    None,
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                out.insert(*i, v.as_number());
            }
        }
    }
    out
}

pub fn init_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    cov: &crate::data::Covariates,
    x: &mut crate::simulator::V,
) {
    if let Some(id) = current_id() {
        if let Some(entry) = registry::get_entry(id) {
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            // execute init statements which may assign to x[] or locals
            let mut assign = |name: &str, idx: usize, val: f64| match name {
                "x" => {
                    if idx < x.len() {
                        x[idx] = val;
                    } else {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "index out of bounds 'x'[{}] (nstates={})",
                            idx,
                            x.len()
                        ));
                    }
                }
                _ => {
                    crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                        "unsupported indexed assignment '{}' in init",
                        name
                    ));
                }
            };
            for st in entry.init_stmts.iter() {
                // use zeros for rateiv parameter
                crate::exa_wasm::interpreter::eval::eval_stmt(
                    st,
                    &crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext),
                    p,
                    _t,
                    &zero_rate,
                    &mut std::collections::HashMap::new(),
                    Some(&entry.pmap),
                    Some(cov),
                    &mut assign,
                );
            }
        }
    }
}
