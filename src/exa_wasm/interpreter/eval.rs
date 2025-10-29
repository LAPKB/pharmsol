use crate::data::Covariates;
use crate::exa_wasm::interpreter::ast::Expr;
use crate::simulator::T;
use crate::simulator::V;
use std::collections::HashMap;

// Evaluator extracted from mod.rs. Uses super::set_runtime_error to report
// runtime problems so the parent module can expose them to the simulator.
fn eval_call(name: &str, args: &[f64]) -> f64 {
    match name {
        "exp" => args.get(0).cloned().unwrap_or(0.0).exp(),
        "if" => {
            let cond = args.get(0).cloned().unwrap_or(0.0);
            if cond != 0.0 {
                args.get(1).cloned().unwrap_or(0.0)
            } else {
                args.get(2).cloned().unwrap_or(0.0)
            }
        }
        "ln" | "log" => args.get(0).cloned().unwrap_or(0.0).ln(),
        "log10" => args.get(0).cloned().unwrap_or(0.0).log10(),
        "log2" => args.get(0).cloned().unwrap_or(0.0).log2(),
        "sqrt" => args.get(0).cloned().unwrap_or(0.0).sqrt(),
        "pow" | "powf" => {
            let a = args.get(0).cloned().unwrap_or(0.0);
            let b = args.get(1).cloned().unwrap_or(0.0);
            a.powf(b)
        }
        "min" => {
            let a = args.get(0).cloned().unwrap_or(0.0);
            let b = args.get(1).cloned().unwrap_or(0.0);
            a.min(b)
        }
        "max" => {
            let a = args.get(0).cloned().unwrap_or(0.0);
            let b = args.get(1).cloned().unwrap_or(0.0);
            a.max(b)
        }
        "abs" => args.get(0).cloned().unwrap_or(0.0).abs(),
        "floor" => args.get(0).cloned().unwrap_or(0.0).floor(),
        "ceil" => args.get(0).cloned().unwrap_or(0.0).ceil(),
        "round" => args.get(0).cloned().unwrap_or(0.0).round(),
        "sin" => args.get(0).cloned().unwrap_or(0.0).sin(),
        "cos" => args.get(0).cloned().unwrap_or(0.0).cos(),
        "tan" => args.get(0).cloned().unwrap_or(0.0).tan(),
        _ => 0.0,
    }
}

fn eval_expr(
    expr: &Expr,
    x: &V,
    p: &V,
    rateiv: &V,
    pmap: Option<&HashMap<String, usize>>,
    t: Option<T>,
    cov: Option<&Covariates>,
) -> f64 {
    use crate::exa_wasm::interpreter::set_runtime_error;

    match expr {
        Expr::Number(v) => *v,
        Expr::Ident(name) => {
            if name.starts_with('_') {
                return 0.0;
            }
            if let Some(map) = pmap {
                if let Some(idx) = map.get(name) {
                    return p[*idx];
                }
            }
            if name == "t" {
                return t.unwrap_or(0.0);
            }
            if let Some(covariates) = cov {
                if let Some(covariate) = covariates.get_covariate(name) {
                    if let Some(time) = t {
                        if let Ok(v) = covariate.interpolate(time) {
                            return v;
                        }
                    }
                }
            }
            set_runtime_error(format!("unknown identifier '{}'", name));
            0.0
        }
        Expr::Indexed(name, idx_expr) => {
            let idxf = eval_expr(idx_expr, x, p, rateiv, pmap, t, cov);
            if !idxf.is_finite() || idxf.is_sign_negative() {
                set_runtime_error(format!(
                    "invalid index expression for '{}' -> {}",
                    name, idxf
                ));
                return 0.0;
            }
            let idx = idxf as usize;
            match name.as_str() {
                "x" => {
                    if idx < x.len() {
                        x[idx]
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'x'[{}] (nstates={})",
                            idx,
                            x.len()
                        ));
                        0.0
                    }
                }
                "p" | "params" => {
                    if idx < p.len() {
                        p[idx]
                    } else {
                        set_runtime_error(format!(
                            "parameter index out of bounds '{}'[{}] (nparams={})",
                            name,
                            idx,
                            p.len()
                        ));
                        0.0
                    }
                }
                "rateiv" => {
                    if idx < rateiv.len() {
                        rateiv[idx]
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'rateiv'[{}] (len={})",
                            idx,
                            rateiv.len()
                        ));
                        0.0
                    }
                }
                _ => {
                    set_runtime_error(format!("unknown indexed symbol '{}'", name));
                    0.0
                }
            }
        }
        Expr::UnaryOp { op, rhs } => {
            let v = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
            match op.as_str() {
                "-" => -v,
                "!" => {
                    if v == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => v,
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = eval_expr(lhs, x, p, rateiv, pmap, t, cov);
            match op.as_str() {
                "&&" => {
                    if a == 0.0 {
                        return 0.0;
                    }
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    if b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "||" => {
                    if a != 0.0 {
                        return 1.0;
                    }
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    if b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => {
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    match op.as_str() {
                        "+" => a + b,
                        "-" => a - b,
                        "*" => a * b,
                        "/" => a / b,
                        "^" => a.powf(b),
                        "<" => {
                            if a < b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ">" => {
                            if a > b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "<=" => {
                            if a <= b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ">=" => {
                            if a >= b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "==" => {
                            if a == b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "!=" => {
                            if a != b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => a,
                    }
                }
            }
        }
        Expr::Call { name, args } => {
            let mut avals: Vec<f64> = Vec::new();
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            if res == 0.0 {
                if !matches!(
                    name.as_str(),
                    "min"
                        | "max"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "ln"
                        | "log"
                        | "log10"
                        | "log2"
                        | "pow"
                        | "powf"
                ) {
                    set_runtime_error(format!("unknown function '{}()', returned 0.0", name));
                }
            }
            res
        }
        Expr::Ternary {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = eval_expr(cond, x, p, rateiv, pmap, t, cov);
            if c != 0.0 {
                eval_expr(then_branch, x, p, rateiv, pmap, t, cov)
            } else {
                eval_expr(else_branch, x, p, rateiv, pmap, t, cov)
            }
        }
        Expr::MethodCall {
            receiver,
            name,
            args,
        } => {
            let recv = eval_expr(receiver, x, p, rateiv, pmap, t, cov);
            let mut avals: Vec<f64> = Vec::new();
            avals.push(recv);
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            if res == 0.0 {
                if !matches!(
                    name.as_str(),
                    "min"
                        | "max"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "ln"
                        | "log"
                        | "log10"
                        | "log2"
                        | "pow"
                        | "powf"
                ) {
                    set_runtime_error(format!("unknown method '{}', returned 0.0", name));
                }
            }
            res
        }
    }
}

pub(crate) use eval_call;
pub(crate) use eval_expr;
