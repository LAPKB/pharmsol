use diffsol::Vector;

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
            for (i, expr) in entry.dx.iter() {
                let val = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    x,
                    p,
                    &rateiv,
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                dx[*i] = val;
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
            for (i, expr) in entry.out.iter() {
                let val = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    x,
                    p,
                    &tmp,
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                y[*i] = val;
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
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                out.insert(*i, v);
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
                    Some(&entry.pmap),
                    Some(_t),
                    Some(_cov),
                );
                out.insert(*i, v);
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
            for (i, expr) in entry.init.iter() {
                let v = crate::exa_wasm::interpreter::eval::eval_expr(
                    expr,
                    &crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext),
                    p,
                    &zero_rate,
                    Some(&entry.pmap),
                    Some(_t),
                    Some(cov),
                );
                x[*i] = v;
            }
        }
    }
}
