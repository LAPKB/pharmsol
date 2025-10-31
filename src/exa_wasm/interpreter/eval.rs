use diffsol::Vector;

use crate::data::Covariates;
use crate::exa_wasm::interpreter::ast::Expr;
use crate::exa_wasm::interpreter::builtins;
use crate::simulator::T;
use crate::simulator::V;
use std::collections::HashMap;

// runtime value type
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Bool(bool),
}

impl Value {
    pub fn as_number(&self) -> f64 {
        match self {
            Value::Number(n) => *n,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
        }
    }
}

// Evaluator extracted from mod.rs. Uses super::set_runtime_error to report
// runtime problems so the parent module can expose them to the simulator.
pub(crate) fn eval_call(name: &str, args: &[Value]) -> Value {
    use Value::Number;
    // runtime arity and known-function checks using centralized builtins table
    if let Some(range) = builtins::arg_count_range(name) {
        if !range.contains(&args.len()) {
            crate::exa_wasm::interpreter::set_runtime_error(format!(
                "builtin '{}' called with wrong arity: got {}, expected {:?}",
                name,
                args.len(),
                range
            ));
            return Number(0.0);
        }
    } else {
        // if arg_count_range returns None, it's unknown to our builtin table
        if !builtins::is_known_function(name) {
            crate::exa_wasm::interpreter::set_runtime_error(format!(
                "unknown function '{}', not present in builtins table",
                name
            ));
            return Number(0.0);
        }
    }

    match name {
        "exp" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .exp(),
        ),
        "if" => {
            let cond = args.get(0).cloned().unwrap_or(Number(0.0));
            if cond.as_bool() {
                args.get(1).cloned().unwrap_or(Number(0.0))
            } else {
                args.get(2).cloned().unwrap_or(Number(0.0))
            }
        }
        "ln" | "log" => Number(args.get(0).cloned().unwrap_or(Number(0.0)).as_number().ln()),
        "log10" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .log10(),
        ),
        "log2" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .log2(),
        ),
        "sqrt" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .sqrt(),
        ),
        "pow" | "powf" => {
            let a = args.get(0).cloned().unwrap_or(Number(0.0)).as_number();
            let b = args.get(1).cloned().unwrap_or(Number(0.0)).as_number();
            Number(a.powf(b))
        }
        "min" => {
            let a = args.get(0).cloned().unwrap_or(Number(0.0)).as_number();
            let b = args.get(1).cloned().unwrap_or(Number(0.0)).as_number();
            Number(a.min(b))
        }
        "max" => {
            let a = args.get(0).cloned().unwrap_or(Number(0.0)).as_number();
            let b = args.get(1).cloned().unwrap_or(Number(0.0)).as_number();
            Number(a.max(b))
        }
        "abs" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .abs(),
        ),
        "floor" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .floor(),
        ),
        "ceil" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .ceil(),
        ),
        "round" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .round(),
        ),
        "sin" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .sin(),
        ),
        "cos" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .cos(),
        ),
        "tan" => Number(
            args.get(0)
                .cloned()
                .unwrap_or(Number(0.0))
                .as_number()
                .tan(),
        ),
        _ => {
            // Unknown function: report a runtime error so callers/users
            // can detect mistakes (typos, missing builtins) instead of
            // silently receiving 0.0 which hides problems.
            crate::exa_wasm::interpreter::set_runtime_error(format!("unknown function '{}'", name));
            Number(0.0)
        }
    }
}

pub(crate) fn eval_expr(
    expr: &Expr,
    x: &V,
    p: &V,
    rateiv: &V,
    locals: Option<&HashMap<String, f64>>,
    pmap: Option<&HashMap<String, usize>>,
    t: Option<T>,
    cov: Option<&Covariates>,
) -> Value {
    use crate::exa_wasm::interpreter::set_runtime_error;

    match expr {
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Number(v) => Value::Number(*v),
        Expr::Ident(name) => {
            if name.starts_with('_') {
                return Value::Number(0.0);
            }
            // local variables defined by prelude take precedence
            if let Some(loc) = locals {
                if let Some(v) = loc.get(name) {
                    return Value::Number(*v);
                }
            }
            if let Some(map) = pmap {
                if let Some(idx) = map.get(name) {
                    let val = p[*idx];
                    return Value::Number(val);
                }
            }
            if name == "t" {
                let val = t.unwrap_or(0.0);
                return Value::Number(val);
            }
            if let Some(covariates) = cov {
                if let Some(covariate) = covariates.get_covariate(name) {
                    if let Some(time) = t {
                        if let Ok(v) = covariate.interpolate(time) {
                            return Value::Number(v);
                        }
                    }
                }
            }
            set_runtime_error(format!("unknown identifier '{}'", name));
            Value::Number(0.0)
        }
        Expr::Param(idx) => {
            let i = *idx;
            if i < p.len() {
                Value::Number(p[i])
            } else {
                set_runtime_error(format!(
                    "parameter index out of bounds p[{}] (nparams={})",
                    i,
                    p.len()
                ));
                Value::Number(0.0)
            }
        }
        Expr::Indexed(name, idx_expr) => {
            let idxv = eval_expr(idx_expr, x, p, rateiv, locals, pmap, t, cov);
            let idxf = idxv.as_number();
            if !idxf.is_finite() || idxf.is_sign_negative() {
                set_runtime_error(format!(
                    "invalid index expression for '{}' -> {}",
                    name, idxf
                ));
                return Value::Number(0.0);
            }
            let idx = idxf as usize;
            match name.as_str() {
                "x" => {
                    if idx < x.len() {
                        Value::Number(x[idx])
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'x'[{}] (nstates={})",
                            idx,
                            x.len()
                        ));
                        Value::Number(0.0)
                    }
                }
                "p" | "params" => {
                    if idx < p.len() {
                        Value::Number(p[idx])
                    } else {
                        set_runtime_error(format!(
                            "parameter index out of bounds '{}'[{}] (nparams={})",
                            name,
                            idx,
                            p.len()
                        ));
                        Value::Number(0.0)
                    }
                }
                "rateiv" => {
                    if idx < rateiv.len() {
                        Value::Number(rateiv[idx])
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'rateiv'[{}] (len={})",
                            idx,
                            rateiv.len()
                        ));
                        Value::Number(0.0)
                    }
                }
                _ => {
                    set_runtime_error(format!("unknown indexed symbol '{}'", name));
                    Value::Number(0.0)
                }
            }
        }
        Expr::UnaryOp { op, rhs } => {
            let v = eval_expr(rhs, x, p, rateiv, locals, pmap, t, cov);
            match op.as_str() {
                "-" => Value::Number(-v.as_number()),
                "!" => Value::Bool(!v.as_bool()),
                _ => v,
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            match op.as_str() {
                "&&" => {
                    let a = eval_expr(lhs, x, p, rateiv, locals, pmap, t, cov);
                    if !a.as_bool() {
                        return Value::Bool(false);
                    }
                    let b = eval_expr(rhs, x, p, rateiv, locals, pmap, t, cov);
                    Value::Bool(b.as_bool())
                }
                "||" => {
                    let a = eval_expr(lhs, x, p, rateiv, locals, pmap, t, cov);
                    if a.as_bool() {
                        return Value::Bool(true);
                    }
                    let b = eval_expr(rhs, x, p, rateiv, locals, pmap, t, cov);
                    Value::Bool(b.as_bool())
                }
                _ => {
                    let a = eval_expr(lhs, x, p, rateiv, locals, pmap, t, cov);
                    let b = eval_expr(rhs, x, p, rateiv, locals, pmap, t, cov);
                    match op.as_str() {
                        "+" => Value::Number(a.as_number() + b.as_number()),
                        "-" => Value::Number(a.as_number() - b.as_number()),
                        "*" => Value::Number(a.as_number() * b.as_number()),
                        "/" => Value::Number(a.as_number() / b.as_number()),
                        "^" => Value::Number(a.as_number().powf(b.as_number())),
                        "<" => Value::Bool(a.as_number() < b.as_number()),
                        ">" => Value::Bool(a.as_number() > b.as_number()),
                        "<=" => Value::Bool(a.as_number() <= b.as_number()),
                        ">=" => Value::Bool(a.as_number() >= b.as_number()),
                        "==" => {
                            // equality for numbers and bools via coercion
                            match (a, b) {
                                (Value::Bool(aa), Value::Bool(bb)) => Value::Bool(aa == bb),
                                (aa, bb) => Value::Bool(aa.as_number() == bb.as_number()),
                            }
                        }
                        "!=" => match (a, b) {
                            (Value::Bool(aa), Value::Bool(bb)) => Value::Bool(aa != bb),
                            (aa, bb) => Value::Bool(aa.as_number() != bb.as_number()),
                        },
                        _ => a,
                    }
                }
            }
        }
        Expr::Call { name, args } => {
            let mut avals: Vec<Value> = Vec::new();
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, locals, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            // warn if unknown function returned Number(0.0)? Keep legacy behavior minimal
            res
        }
        Expr::Ternary {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = eval_expr(cond, x, p, rateiv, locals, pmap, t, cov);
            if c.as_bool() {
                eval_expr(then_branch, x, p, rateiv, locals, pmap, t, cov)
            } else {
                eval_expr(else_branch, x, p, rateiv, locals, pmap, t, cov)
            }
        }
        Expr::MethodCall {
            receiver,
            name,
            args,
        } => {
            let recv = eval_expr(receiver, x, p, rateiv, locals, pmap, t, cov);
            let mut avals: Vec<Value> = Vec::new();
            avals.push(recv);
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, locals, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            res
        }
    }
}

// functions are exported as `pub(crate)` above for use by parent module

pub(crate) fn eval_stmt<FAssign>(
    stmt: &crate::exa_wasm::interpreter::ast::Stmt,
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    t: crate::simulator::T,
    rateiv: &crate::simulator::V,
    locals: &mut std::collections::HashMap<String, f64>,
    pmap: Option<&std::collections::HashMap<String, usize>>,
    cov: Option<&crate::data::Covariates>,
    assign_indexed: &mut FAssign,
) where
    FAssign: FnMut(&str, usize, f64),
{
    use crate::exa_wasm::interpreter::ast::{Lhs, Stmt};

    match stmt {
        Stmt::Expr(e) => {
            let _ = eval_expr(e, x, p, rateiv, Some(&*locals), pmap, Some(t), cov);
        }
        Stmt::Assign(lhs, rhs) => {
            // evaluate rhs
            let val = eval_expr(rhs, x, p, rateiv, Some(&*locals), pmap, Some(t), cov);
            match lhs {
                Lhs::Ident(name) => {
                    locals.insert(name.clone(), val.as_number());
                }
                Lhs::Indexed(name, idx_expr) => {
                    let idxv =
                        eval_expr(idx_expr, x, p, rateiv, Some(&*locals), pmap, Some(t), cov);
                    let idxf = idxv.as_number();
                    if !idxf.is_finite() || idxf.is_sign_negative() {
                        crate::exa_wasm::interpreter::registry::set_runtime_error(format!(
                            "invalid index expression for '{}' -> {}",
                            name, idxf
                        ));
                        return;
                    }
                    let idx = idxf as usize;
                    // delegate actual assignment to the provided closure
                    assign_indexed(name.as_str(), idx, val.as_number());
                }
            }
        }
        Stmt::Block(v) => {
            for s in v.iter() {
                eval_stmt(s, x, p, t, rateiv, locals, pmap, cov, assign_indexed);
            }
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = eval_expr(cond, x, p, rateiv, Some(&*locals), pmap, Some(t), cov);
            if c.as_bool() {
                eval_stmt(
                    then_branch,
                    x,
                    p,
                    t,
                    rateiv,
                    locals,
                    pmap,
                    cov,
                    assign_indexed,
                );
            } else if let Some(eb) = else_branch {
                eval_stmt(eb, x, p, t, rateiv, locals, pmap, cov, assign_indexed);
            }
        }
    }
}
