use crate::exa_wasm::interpreter::ast::{Expr, Stmt};
use std::collections::HashMap;

// Loader helper utilities used by `loader.rs`. These functions implement a
// conservative extraction and validation surface that mirrors the prior inline
// implementations in `loader.rs` so they can be reused and unit-tested.

// Loader helper utilities extracted from the large `load_ir_ode` function.

// ongoing refactor can wire them into `loader.rs` incrementally.

/// Rewrite parameter identifier `Ident(name)` nodes in a parsed statement
/// vector into `Expr::Param(index)` nodes using the provided `pmap`.
// NOTE: textual rewriting of params in statement vectors was previously
// provided as a helper. The emitter now emits rewritten ASTs (Param nodes)
// directly, and the runtime loader consumes pre-parsed ASTs. This helper
// was removed as part of removing fragile textual fallbacks.

/// Return the body text inside the first top-level pair of braces.
/// Example: given `|t, y| { ... }` returns Some("...") or None.
pub fn extract_closure_body(src: &str) -> Option<String> {
    if let Some(lb_pos) = src.find('{') {
        let bytes = src.as_bytes();
        let mut depth: isize = 0;
        let mut i = lb_pos;
        while i < bytes.len() {
            match bytes[i] as char {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        // return inner text between lb_pos and i
                        let inner = &src[lb_pos + 1..i];
                        return Some(inner.to_string());
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
    None
}
// The textual extraction helpers (macro-stripping, prelude scanning and
// textual `fetch_*` extraction) have been removed. The emitter now emits
// structured `fetch_params` and `fetch_cov` fields in the IR and rewrites
// parameter identifiers into `Expr::Param` nodes before emission. The
// runtime loader consumes the structured IR and no longer attempts to scan
// raw closure text at runtime.

/// Lightweight validator stubs (moved out of loader.rs so the loader can
/// call into a shared place). These can be expanded to perform expression
/// and statement validations that previously lived inside load_ir_ode.
pub fn validate_expr(
    expr: &Expr,
    pmap: &HashMap<String, usize>,
    nstates: usize,
    nparams: usize,
    errors: &mut Vec<String>,
) {
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::Ident(name) => {
            if name == "t" {
                return;
            }
            if pmap.contains_key(name) {
                return;
            }
            errors.push(format!("unknown identifier '{}'", name));
        }
        Expr::Param(_) => {
            // param by index is valid
        }
        Expr::Indexed(name, idx_expr) => match &**idx_expr {
            Expr::Number(n) => {
                let idx = *n as usize;
                match name.as_str() {
                    "x" | "rateiv" => {
                        if idx >= nstates {
                            errors.push(format!(
                                "index out of bounds '{}'[{}] (nstates={})",
                                name, idx, nstates
                            ));
                        }
                    }
                    "p" | "params" => {
                        if idx >= nparams {
                            errors.push(format!(
                                "parameter index out of bounds '{}'[{}] (nparams={})",
                                name, idx, nparams
                            ));
                        }
                    }
                    "y" => {}
                    _ => {
                        errors.push(format!("unknown indexed symbol '{}'", name));
                    }
                }
            }
            other => validate_expr(other, pmap, nstates, nparams, errors),
        },
        Expr::UnaryOp { rhs, .. } => validate_expr(rhs, pmap, nstates, nparams, errors),
        Expr::BinaryOp { lhs, rhs, .. } => {
            validate_expr(lhs, pmap, nstates, nparams, errors);
            validate_expr(rhs, pmap, nstates, nparams, errors);
        }
        Expr::Call { name: _, args } => {
            for a in args.iter() {
                validate_expr(a, pmap, nstates, nparams, errors);
            }
        }
        Expr::MethodCall {
            receiver,
            name: _,
            args,
        } => {
            validate_expr(receiver, pmap, nstates, nparams, errors);
            for a in args.iter() {
                validate_expr(a, pmap, nstates, nparams, errors);
            }
        }
        Expr::Ternary {
            cond,
            then_branch,
            else_branch,
        } => {
            validate_expr(cond, pmap, nstates, nparams, errors);
            validate_expr(then_branch, pmap, nstates, nparams, errors);
            validate_expr(else_branch, pmap, nstates, nparams, errors);
        }
    }
}

pub fn validate_prelude_expr(
    expr: &Expr,
    pmap: &HashMap<String, usize>,
    known_locals: &std::collections::HashSet<String>,
    nstates: usize,
    nparams: usize,
    errors: &mut Vec<String>,
) {
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::Ident(name) => {
            if name == "t" {
                return;
            }
            if known_locals.contains(name) {
                return;
            }
            if pmap.contains_key(name) {
                return;
            }
            errors.push(format!("unknown identifier '{}' in prelude", name));
        }
        Expr::Param(_) => {}
        Expr::Indexed(name, idx_expr) => match &**idx_expr {
            Expr::Number(n) => {
                let idx = *n as usize;
                match name.as_str() {
                    "x" | "rateiv" => {
                        if idx >= nstates {
                            errors.push(format!(
                                "index out of bounds '{}'[{}] (nstates={})",
                                name, idx, nstates
                            ));
                        }
                    }
                    "p" | "params" => {
                        if idx >= nparams {
                            errors.push(format!(
                                "parameter index out of bounds '{}'[{}] (nparams={})",
                                name, idx, nparams
                            ));
                        }
                    }
                    "y" => {}
                    _ => {
                        errors.push(format!("unknown indexed symbol '{}'", name));
                    }
                }
            }
            other => validate_prelude_expr(other, pmap, known_locals, nstates, nparams, errors),
        },
        Expr::UnaryOp { rhs, .. } => {
            validate_prelude_expr(rhs, pmap, known_locals, nstates, nparams, errors)
        }
        Expr::BinaryOp { lhs, rhs, .. } => {
            validate_prelude_expr(lhs, pmap, known_locals, nstates, nparams, errors);
            validate_prelude_expr(rhs, pmap, known_locals, nstates, nparams, errors);
        }
        Expr::Call { name: _, args } => {
            for a in args.iter() {
                validate_prelude_expr(a, pmap, known_locals, nstates, nparams, errors);
            }
        }
        Expr::MethodCall {
            receiver,
            name: _,
            args,
        } => {
            validate_prelude_expr(receiver, pmap, known_locals, nstates, nparams, errors);
            for a in args.iter() {
                validate_prelude_expr(a, pmap, known_locals, nstates, nparams, errors);
            }
        }
        Expr::Ternary {
            cond,
            then_branch,
            else_branch,
        } => {
            validate_prelude_expr(cond, pmap, known_locals, nstates, nparams, errors);
            validate_prelude_expr(then_branch, pmap, known_locals, nstates, nparams, errors);
            validate_prelude_expr(else_branch, pmap, known_locals, nstates, nparams, errors);
        }
    }
}

pub fn validate_stmt(
    st: &Stmt,
    pmap: &HashMap<String, usize>,
    nstates: usize,
    nparams: usize,
    errors: &mut Vec<String>,
) {
    use crate::exa_wasm::interpreter::ast::{Lhs, Stmt};
    match st {
        Stmt::Expr(e) => validate_expr(e, pmap, nstates, nparams, errors),
        Stmt::Assign(lhs, rhs) => {
            validate_expr(rhs, pmap, nstates, nparams, errors);
            if let Lhs::Indexed(_, idx_expr) = lhs {
                validate_expr(idx_expr, pmap, nstates, nparams, errors);
            }
        }
        Stmt::Block(v) => {
            for s in v.iter() {
                validate_stmt(s, pmap, nstates, nparams, errors);
            }
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            validate_expr(cond, pmap, nstates, nparams, errors);
            validate_stmt(then_branch, pmap, nstates, nparams, errors);
            if let Some(eb) = else_branch {
                validate_stmt(eb, pmap, nstates, nparams, errors);
            }
        }
    }
}

pub fn collect_max_index(
    stmts: &Vec<crate::exa_wasm::interpreter::ast::Stmt>,
    _name: &str,
) -> Option<usize> {
    let mut max: Option<usize> = None;
    fn visit(s: &crate::exa_wasm::interpreter::ast::Stmt, max: &mut Option<usize>) {
        use crate::exa_wasm::interpreter::ast::Lhs;
        match s {
            crate::exa_wasm::interpreter::ast::Stmt::Assign(lhs, _) => {
                if let Lhs::Indexed(_nm, idx_expr) = lhs {
                    if let crate::exa_wasm::interpreter::ast::Expr::Number(nn) = &**idx_expr {
                        let idx = *nn as usize;
                        match max {
                            Some(m) if *m < idx => *max = Some(idx),
                            None => *max = Some(idx),
                            _ => {}
                        }
                    }
                }
            }
            crate::exa_wasm::interpreter::ast::Stmt::Block(v) => {
                for ss in v.iter() {
                    visit(ss, max);
                }
            }
            crate::exa_wasm::interpreter::ast::Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                visit(then_branch, max);
                if let Some(eb) = else_branch {
                    visit(eb, max);
                }
            }
            crate::exa_wasm::interpreter::ast::Stmt::Expr(_) => {}
        }
    }
    for s in stmts.iter() {
        visit(s, &mut max);
    }
    max
}
