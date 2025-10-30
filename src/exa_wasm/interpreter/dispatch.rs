use diffsol::Vector;
use diffsol::VectorHost;
use std::collections::HashMap;

use crate::exa_wasm::interpreter::registry;
use crate::exa_wasm::interpreter::vm;
use crate::exa_wasm::interpreter::eval;

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
            // prepare locals vector: use emitted locals ordering if present,
            // otherwise fall back to building slots from prelude ordering.
            let mut locals_vec: Vec<f64> = vec![0.0; entry.locals.len()];
            let mut local_index: HashMap<String, usize> = HashMap::new();
            if !entry.locals.is_empty() {
                for (i, n) in entry.locals.iter().enumerate() {
                    local_index.insert(n.clone(), i);
                }
            }
            // evaluate prelude into a temporary map then populate locals_vec
            let mut temp_locals: HashMap<String, f64> = HashMap::new();
            for (name, expr) in entry.prelude.iter() {
                let val = eval::eval_expr(
                    expr,
                    x,
                    p,
                    &rateiv,
                    Some(&temp_locals),
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                temp_locals.insert(name.clone(), val.as_number());
            }
            // populate locals_vec from temp_locals using emitted locals ordering
            if !entry.locals.is_empty() {
                for (name, &idx) in local_index.iter() {
                    if let Some(v) = temp_locals.get(name) {
                        locals_vec[idx] = *v;
                    }
                }
            } else {
                // no emitted locals ordering: create slots for prelude in insertion order
                let mut i = 0usize;
                for (name, _) in entry.prelude.iter() {
                    local_index.insert(name.clone(), i);
                    if let Some(v) = temp_locals.get(name) {
                        if i >= locals_vec.len() {
                            locals_vec.push(*v);
                        } else {
                            locals_vec[i] = *v;
                        }
                    }
                    i += 1;
                }
            }
            // debug: locals are in `locals_vec` and `local_index`
            // If emitted bytecode exists for diffeq, prefer executing it
            if !entry.bytecode_diffeq.is_empty() {
                // builtin dispatch closure: translate f64 args -> eval::Value and call eval::eval_call
                let builtins_dispatch = |name: &str, args: &[f64]| -> f64 {
                    let vals: Vec<eval::Value> = args.iter().map(|a| eval::Value::Number(*a)).collect();
                    eval::eval_call(name, &vals).as_number()
                };
                // assignment closure maps VM stores to simulator vectors
                let mut assign = |name: &str, idx: usize, val: f64| match name {
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
                    "x" | "y" => {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "write to '{}' not allowed in diffeq bytecode",
                            name
                        ));
                    }
                    _ => {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "unsupported indexed assignment '{}' in diffeq",
                            name
                        ));
                    }
                };
                for (_i, code) in entry.bytecode_diffeq.iter() {
                    let mut locals_mut = locals_vec.clone();
                    vm::run_bytecode_full(
                        code.as_slice(),
                        x.as_slice(),
                        p.as_slice(),
                        rateiv.as_slice(),
                        _t,
                        &mut locals_mut,
                        &entry.funcs,
                        &builtins_dispatch,
                        |n, i, v| assign(n, i, v),
                    );
                }
            } else {
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
                // convert locals_vec into a HashMap for eval_stmt
                let mut locals_map: HashMap<String, f64> = HashMap::new();
                for (name, &idx) in local_index.iter() {
                    if idx < locals_vec.len() {
                        locals_map.insert(name.clone(), locals_vec[idx]);
                    }
                }
                for st in entry.diffeq_stmts.iter() {
                    crate::exa_wasm::interpreter::eval::eval_stmt(
                        st,
                        x,
                        p,
                        _t,
                        &rateiv,
                        &mut locals_map,
                        Some(&entry.pmap),
                        Some(_cov),
                        &mut assign_closure,
                    );
                }
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
            // prepare locals vector for out bytecode (use emitted locals ordering)
            let mut locals_vec: Vec<f64> = vec![0.0; entry.locals.len()];
            let mut local_index: HashMap<String, usize> = HashMap::new();
            if !entry.locals.is_empty() {
                for (i, n) in entry.locals.iter().enumerate() {
                    local_index.insert(n.clone(), i);
                }
            }
            // evaluate prelude into temporary map and populate locals_vec
            let mut temp_locals: HashMap<String, f64> = HashMap::new();
            for (name, expr) in entry.prelude.iter() {
                let val = eval::eval_expr(
                    expr,
                    x,
                    p,
                    &tmp,
                    Some(&temp_locals),
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                temp_locals.insert(name.clone(), val.as_number());
            }
            for (name, &idx) in local_index.iter() {
                if let Some(v) = temp_locals.get(name) {
                    locals_vec[idx] = *v;
                }
            }

            if !entry.bytecode_out.is_empty() {
                let builtins_dispatch = |name: &str, args: &[f64]| -> f64 {
                    let vals: Vec<eval::Value> = args.iter().map(|a| eval::Value::Number(*a)).collect();
                    eval::eval_call(name, &vals).as_number()
                };
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
                for (_i, code) in entry.bytecode_out.iter() {
                    let mut locals_mut = locals_vec.clone();
                    vm::run_bytecode_full(
                        code.as_slice(),
                        x.as_slice(),
                        p.as_slice(),
                        tmp.as_slice(),
                        _t,
                        &mut locals_mut,
                        &entry.funcs,
                        &builtins_dispatch,
                        |n, i, v| assign(n, i, v),
                    );
                }
            } else {
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
            // prepare locals vector for init bytecode (use emitted locals ordering)
            let mut locals_vec: Vec<f64> = vec![0.0; entry.locals.len()];
            let mut local_index: HashMap<String, usize> = HashMap::new();
            if !entry.locals.is_empty() {
                for (i, n) in entry.locals.iter().enumerate() {
                    local_index.insert(n.clone(), i);
                }
            }
            let mut temp_locals: HashMap<String, f64> = HashMap::new();
            for (name, expr) in entry.prelude.iter() {
                let val = eval::eval_expr(
                    expr,
                    &crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext),
                    p,
                    &zero_rate,
                    Some(&temp_locals),
                    Some(&entry.pmap),
                    Some(_t),
                    Some(cov),
                );
                temp_locals.insert(name.clone(), val.as_number());
            }
            for (name, &idx) in local_index.iter() {
                if let Some(v) = temp_locals.get(name) {
                    locals_vec[idx] = *v;
                }
            }

            if !entry.bytecode_init.is_empty() {
                let builtins_dispatch = |name: &str, args: &[f64]| -> f64 {
                    let vals: Vec<eval::Value> = args.iter().map(|a| eval::Value::Number(*a)).collect();
                    eval::eval_call(name, &vals).as_number()
                };
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
                for (_i, code) in entry.bytecode_init.iter() {
                    let mut locals_mut = locals_vec.clone();
                    vm::run_bytecode_full(
                        code.as_slice(),
                        &crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext).as_slice(),
                        p.as_slice(),
                        zero_rate.as_slice(),
                        _t,
                        &mut locals_mut,
                        &entry.funcs,
                        &builtins_dispatch,
                        |n, i, v| assign(n, i, v),
                    );
                }
            } else {
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
}
