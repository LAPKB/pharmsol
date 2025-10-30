use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

use serde::Deserialize;

use crate::exa_wasm::interpreter::ast::Expr;
use crate::exa_wasm::interpreter::parser::{tokenize, Parser};
use crate::exa_wasm::interpreter::registry;
use crate::exa_wasm::interpreter::typecheck;

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct IrFile {
    ir_version: Option<String>,
    kind: Option<String>,
    params: Option<Vec<String>>,
    model_text: Option<String>,
    diffeq: Option<String>,
    lag: Option<String>,
    fa: Option<String>,
    init: Option<String>,
    out: Option<String>,
    lag_map: Option<std::collections::HashMap<usize, String>>,
    fa_map: Option<std::collections::HashMap<usize, String>>,
    // optional pre-parsed ASTs emitted by `emit_ir`
    diffeq_ast: Option<Vec<crate::exa_wasm::interpreter::ast::Stmt>>,
    out_ast: Option<Vec<crate::exa_wasm::interpreter::ast::Stmt>>,
    init_ast: Option<Vec<crate::exa_wasm::interpreter::ast::Stmt>>,
}

pub fn load_ir_ode(
    ir_path: PathBuf,
) -> Result<
    (
        crate::simulator::equation::ODE,
        crate::simulator::equation::Meta,
        usize,
    ),
    io::Error,
> {
    let contents = fs::read_to_string(&ir_path)?;
    let ir: IrFile = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("serde_json: {}", e)))?;

    let params = ir.params.unwrap_or_default();
    let meta = crate::simulator::equation::Meta::new(params.iter().map(|s| s.as_str()).collect());

    let mut pmap = std::collections::HashMap::new();
    for (i, name) in params.iter().enumerate() {
        pmap.insert(name.clone(), i);
    }

    let diffeq_text = ir
        .diffeq
        .clone()
        .unwrap_or_else(|| ir.model_text.clone().unwrap_or_default());
    let out_text = ir.out.clone().unwrap_or_default();
    let init_text = ir.init.clone().unwrap_or_default();
    let lag_text = ir.lag.clone().unwrap_or_default();
    let fa_text = ir.fa.clone().unwrap_or_default();

    let mut dx_map: HashMap<usize, Expr> = HashMap::new();
    let mut out_map: HashMap<usize, Expr> = HashMap::new();
    let mut init_map: HashMap<usize, Expr> = HashMap::new();
    let mut lag_map: HashMap<usize, Expr> = HashMap::new();
    let mut fa_map: HashMap<usize, Expr> = HashMap::new();
    let mut prelude: Vec<(String, Expr)> = Vec::new();
    // statement vectors (full statement ASTs parsed from closures)
    let mut diffeq_stmts: Vec<crate::exa_wasm::interpreter::ast::Stmt> = Vec::new();
    let mut out_stmts: Vec<crate::exa_wasm::interpreter::ast::Stmt> = Vec::new();
    let mut init_stmts: Vec<crate::exa_wasm::interpreter::ast::Stmt> = Vec::new();

    let mut parse_errors: Vec<String> = Vec::new();

    // Extract top-level assignments like `dx[i] = expr;` from the closure body.
    // Only statements at the first brace nesting level (depth == 1) are
    // considered top-level; assignments inside nested blocks (e.g. inside
    // `if { ... }`) will be ignored. This avoids accidentally extracting
    // conditional assignments that should not be treated as unconditional
    // runtime equations.
    // extract_all_assign delegated to loader_helpers

    // Prefer a pre-parsed AST emitted by the IR emitter when available.
    // This allows us to skip textual parsing/fallbacks at runtime.
    if let Some(ast) = ir.diffeq_ast.clone() {
        // ensure the AST types are valid
        if let Err(e) = typecheck::check_statements(&ast) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("type errors in diffeq AST in IR: {:?}", e),
            ));
        }
        diffeq_stmts = ast;
    }

    // Prefer structural parsing of the closure body using the new statement
    // parser when no pre-parsed AST is provided. This is more robust than
    // substring scanning and allows us to convert top-level `if` statements
    // into conditional RHS expressions. closure extraction and macro-stripping
    // delegated to loader_helpers

    // boolean literals are parsed by the tokenizer (Token::Bool). No normalization needed.

    if let Some(body) =
        crate::exa_wasm::interpreter::loader_helpers::extract_closure_body(&diffeq_text)
    {
        let mut cleaned = body.clone();
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_params!",
        );
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_param!",
        );
        cleaned =
            crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(&cleaned, "fetch_cov!");

        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(mut stmts) = p.parse_statements() {
            // rewrite param identifiers into Param(index) nodes for faster lookup
            fn rewrite_params_in_expr(
                e: &mut crate::exa_wasm::interpreter::ast::Expr,
                pmap: &HashMap<String, usize>,
            ) {
                use crate::exa_wasm::interpreter::ast::*;
                match e {
                    Expr::Ident(name) => {
                        if let Some(idx) = pmap.get(name) {
                            *e = Expr::Param(*idx);
                        }
                    }
                    Expr::Indexed(_, idx_expr) => rewrite_params_in_expr(idx_expr, pmap),
                    Expr::UnaryOp { rhs, .. } => rewrite_params_in_expr(rhs, pmap),
                    Expr::BinaryOp { lhs, rhs, .. } => {
                        rewrite_params_in_expr(lhs, pmap);
                        rewrite_params_in_expr(rhs, pmap);
                    }
                    Expr::Call { args, .. } => {
                        for a in args.iter_mut() {
                            rewrite_params_in_expr(a, pmap);
                        }
                    }
                    Expr::MethodCall { receiver, args, .. } => {
                        rewrite_params_in_expr(receiver, pmap);
                        for a in args.iter_mut() {
                            rewrite_params_in_expr(a, pmap);
                        }
                    }
                    Expr::Ternary {
                        cond,
                        then_branch,
                        else_branch,
                    } => {
                        rewrite_params_in_expr(cond, pmap);
                        rewrite_params_in_expr(then_branch, pmap);
                        rewrite_params_in_expr(else_branch, pmap);
                    }
                    _ => {}
                }
            }
            fn rewrite_params_in_stmt(
                s: &mut crate::exa_wasm::interpreter::ast::Stmt,
                pmap: &HashMap<String, usize>,
            ) {
                use crate::exa_wasm::interpreter::ast::*;
                match s {
                    Stmt::Expr(e) => rewrite_params_in_expr(e, pmap),
                    Stmt::Assign(lhs, rhs) => {
                        if let Lhs::Indexed(_, idx_expr) = lhs {
                            rewrite_params_in_expr(idx_expr, pmap);
                        }
                        rewrite_params_in_expr(rhs, pmap);
                    }
                    Stmt::Block(v) => {
                        for ss in v.iter_mut() {
                            rewrite_params_in_stmt(ss, pmap);
                        }
                    }
                    Stmt::If {
                        cond,
                        then_branch,
                        else_branch,
                    } => {
                        rewrite_params_in_expr(cond, pmap);
                        rewrite_params_in_stmt(then_branch, pmap);
                        if let Some(eb) = else_branch {
                            rewrite_params_in_stmt(eb, pmap);
                        }
                    }
                }
            }

            for st in stmts.iter_mut() {
                rewrite_params_in_stmt(st, &pmap);
            }

            // run a lightweight type-check pass and reject obviously bad IR
            if let Err(e) = typecheck::check_statements(&stmts) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("type errors in diffeq closure: {:?}", e),
                ));
            }
            // keep the parsed statements for later execution
            diffeq_stmts = stmts;
        } else {
            // fallback: extract dx[...] assignments into synthetic Assign stmts
            for (i, rhs) in crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(
                &diffeq_text,
                "dx[",
            ) {
                let toks = tokenize(&rhs);
                let mut p = Parser::new(toks);
                let res = p.parse_expr_result();
                match res {
                    Ok(expr) => {
                        dx_map.insert(i, expr.clone());
                    }
                    Err(e) => {
                        parse_errors
                            .push(format!("failed to parse dx[{}] RHS='{}' : {}", i, rhs, e));
                    }
                }
            }
            // convert dx_map into simple Assign statements
            for (i, expr) in dx_map.iter() {
                let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                    "dx".to_string(),
                    Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
                );
                diffeq_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                    lhs,
                    expr.clone(),
                ));
            }
        }
    } else {
        // no closure body: attempt substring scan fallback
        for (i, rhs) in
            crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(&diffeq_text, "dx[")
        {
            let toks = tokenize(&rhs);
            let mut p = Parser::new(toks);
            let res = p.parse_expr_result();
            match res {
                Ok(expr) => {
                    dx_map.insert(i, expr.clone());
                }
                Err(e) => {
                    parse_errors.push(format!("failed to parse dx[{}] RHS='{}' : {}", i, rhs, e));
                }
            }
        }
        for (i, expr) in dx_map.iter() {
            let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                "dx".to_string(),
                Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
            );
            diffeq_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                lhs,
                expr.clone(),
            ));
        }
    }

    // extract non-indexed assignments like `ke = ke + 0.5;` from diffeq prelude
    for (name, rhs) in crate::exa_wasm::interpreter::loader_helpers::extract_prelude(&diffeq_text) {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        match p.parse_expr_result() {
            Ok(expr) => prelude.push((name, expr)),
            Err(e) => parse_errors.push(format!(
                "failed to parse prelude assignment '{} = {}' : {}",
                name, rhs, e
            )),
        }
    }
    if !prelude.is_empty() {
        eprintln!(
            "[loader] parsed prelude assignments: {:?}",
            prelude.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
        );
    }
    // If the IR includes a pre-parsed out AST, use it.
    if let Some(ast) = ir.out_ast.clone() {
        if let Err(e) = typecheck::check_statements(&ast) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("type errors in out AST in IR: {:?}", e),
            ));
        }
        out_stmts = ast;
    }

    // parse out closure into statements (fall back to extraction)
    if let Some(body) =
        crate::exa_wasm::interpreter::loader_helpers::extract_closure_body(&out_text)
    {
        let mut cleaned = body.clone();
        // strip macros
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_params!",
        );
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_param!",
        );
        cleaned =
            crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(&cleaned, "fetch_cov!");
        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(mut stmts) = p.parse_statements() {
            // rewrite params into Param(index)
            fn rewrite_params_in_expr(
                e: &mut crate::exa_wasm::interpreter::ast::Expr,
                pmap: &HashMap<String, usize>,
            ) {
                use crate::exa_wasm::interpreter::ast::*;
                match e {
                    Expr::Ident(name) => {
                        if let Some(idx) = pmap.get(name) {
                            *e = Expr::Param(*idx);
                        }
                    }
                    Expr::Indexed(_, idx_expr) => rewrite_params_in_expr(idx_expr, pmap),
                    Expr::UnaryOp { rhs, .. } => rewrite_params_in_expr(rhs, pmap),
                    Expr::BinaryOp { lhs, rhs, .. } => {
                        rewrite_params_in_expr(lhs, pmap);
                        rewrite_params_in_expr(rhs, pmap);
                    }
                    Expr::Call { args, .. } => {
                        for a in args.iter_mut() {
                            rewrite_params_in_expr(a, pmap);
                        }
                    }
                    Expr::MethodCall { receiver, args, .. } => {
                        rewrite_params_in_expr(receiver, pmap);
                        for a in args.iter_mut() {
                            rewrite_params_in_expr(a, pmap);
                        }
                    }
                    Expr::Ternary {
                        cond,
                        then_branch,
                        else_branch,
                    } => {
                        rewrite_params_in_expr(cond, pmap);
                        rewrite_params_in_expr(then_branch, pmap);
                        rewrite_params_in_expr(else_branch, pmap);
                    }
                    _ => {}
                }
            }
            fn rewrite_params_in_stmt(
                s: &mut crate::exa_wasm::interpreter::ast::Stmt,
                pmap: &HashMap<String, usize>,
            ) {
                use crate::exa_wasm::interpreter::ast::*;
                match s {
                    Stmt::Expr(e) => rewrite_params_in_expr(e, pmap),
                    Stmt::Assign(lhs, rhs) => {
                        if let Lhs::Indexed(_, idx_expr) = lhs {
                            rewrite_params_in_expr(idx_expr, pmap);
                        }
                        rewrite_params_in_expr(rhs, pmap);
                    }
                    Stmt::Block(v) => {
                        for ss in v.iter_mut() {
                            rewrite_params_in_stmt(ss, pmap);
                        }
                    }
                    Stmt::If {
                        cond,
                        then_branch,
                        else_branch,
                    } => {
                        rewrite_params_in_expr(cond, pmap);
                        rewrite_params_in_stmt(then_branch, pmap);
                        if let Some(eb) = else_branch {
                            rewrite_params_in_stmt(eb, pmap);
                        }
                    }
                }
            }

            for st in stmts.iter_mut() {
                rewrite_params_in_stmt(st, &pmap);
            }

            if let Err(e) = typecheck::check_statements(&stmts) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("type errors in out closure: {:?}", e),
                ));
            }
            out_stmts = stmts;
        } else {
            for (i, rhs) in
                crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(&out_text, "y[")
            {
                let toks = tokenize(&rhs);
                let mut p = Parser::new(toks);
                let res = p.parse_expr_result();
                match res {
                    Ok(expr) => {
                        out_map.insert(i, expr);
                    }
                    Err(e) => {
                        parse_errors
                            .push(format!("failed to parse y[{}] RHS='{}' : {}", i, rhs, e));
                    }
                }
            }
            for (i, expr) in out_map.iter() {
                let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                    "y".to_string(),
                    Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
                );
                out_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                    lhs,
                    expr.clone(),
                ));
            }
        }
    } else {
        for (i, rhs) in
            crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(&out_text, "y[")
        {
            let toks = tokenize(&rhs);
            let mut p = Parser::new(toks);
            let res = p.parse_expr_result();
            match res {
                Ok(expr) => {
                    out_map.insert(i, expr);
                }
                Err(e) => {
                    parse_errors.push(format!("failed to parse y[{}] RHS='{}' : {}", i, rhs, e));
                }
            }
        }
        for (i, expr) in out_map.iter() {
            let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                "y".to_string(),
                Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
            );
            out_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                lhs,
                expr.clone(),
            ));
        }
    }

    // If the IR includes a pre-parsed init AST, use it.
    if let Some(ast) = ir.init_ast.clone() {
        if let Err(e) = typecheck::check_statements(&ast) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("type errors in init AST in IR: {:?}", e),
            ));
        }
        init_stmts = ast;
    }

    // parse init closure into statements
    if let Some(body) =
        crate::exa_wasm::interpreter::loader_helpers::extract_closure_body(&init_text)
    {
        let mut cleaned = body.clone();
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_params!",
        );
        cleaned = crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(
            &cleaned,
            "fetch_param!",
        );
        cleaned =
            crate::exa_wasm::interpreter::loader_helpers::strip_macro_calls(&cleaned, "fetch_cov!");
        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(mut stmts) = p.parse_statements() {
            for st in stmts.iter_mut() {
                // reuse the same rewrite helpers as above
                fn rewrite_params_in_expr(
                    e: &mut crate::exa_wasm::interpreter::ast::Expr,
                    pmap: &HashMap<String, usize>,
                ) {
                    use crate::exa_wasm::interpreter::ast::*;
                    match e {
                        Expr::Ident(name) => {
                            if let Some(idx) = pmap.get(name) {
                                *e = Expr::Param(*idx);
                            }
                        }
                        Expr::Indexed(_, idx_expr) => rewrite_params_in_expr(idx_expr, pmap),
                        Expr::UnaryOp { rhs, .. } => rewrite_params_in_expr(rhs, pmap),
                        Expr::BinaryOp { lhs, rhs, .. } => {
                            rewrite_params_in_expr(lhs, pmap);
                            rewrite_params_in_expr(rhs, pmap);
                        }
                        Expr::Call { args, .. } => {
                            for a in args.iter_mut() {
                                rewrite_params_in_expr(a, pmap);
                            }
                        }
                        Expr::MethodCall { receiver, args, .. } => {
                            rewrite_params_in_expr(receiver, pmap);
                            for a in args.iter_mut() {
                                rewrite_params_in_expr(a, pmap);
                            }
                        }
                        Expr::Ternary {
                            cond,
                            then_branch,
                            else_branch,
                        } => {
                            rewrite_params_in_expr(cond, pmap);
                            rewrite_params_in_expr(then_branch, pmap);
                            rewrite_params_in_expr(else_branch, pmap);
                        }
                        _ => {}
                    }
                }
                fn rewrite_params_in_stmt(
                    s: &mut crate::exa_wasm::interpreter::ast::Stmt,
                    pmap: &HashMap<String, usize>,
                ) {
                    use crate::exa_wasm::interpreter::ast::*;
                    match s {
                        Stmt::Expr(e) => rewrite_params_in_expr(e, pmap),
                        Stmt::Assign(lhs, rhs) => {
                            if let Lhs::Indexed(_, idx_expr) = lhs {
                                rewrite_params_in_expr(idx_expr, pmap);
                            }
                            rewrite_params_in_expr(rhs, pmap);
                        }
                        Stmt::Block(v) => {
                            for ss in v.iter_mut() {
                                rewrite_params_in_stmt(ss, pmap);
                            }
                        }
                        Stmt::If {
                            cond,
                            then_branch,
                            else_branch,
                        } => {
                            rewrite_params_in_expr(cond, pmap);
                            rewrite_params_in_stmt(then_branch, pmap);
                            if let Some(eb) = else_branch {
                                rewrite_params_in_stmt(eb, pmap);
                            }
                        }
                    }
                }
                rewrite_params_in_stmt(st, &pmap);
            }

            if let Err(e) = typecheck::check_statements(&stmts) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("type errors in init closure: {:?}", e),
                ));
            }
            init_stmts = stmts;
        } else {
            for (i, rhs) in
                crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(&init_text, "x[")
            {
                let toks = tokenize(&rhs);
                let mut p = Parser::new(toks);
                let res = p.parse_expr_result();
                match res {
                    Ok(expr) => {
                        init_map.insert(i, expr);
                    }
                    Err(e) => {
                        parse_errors.push(format!(
                            "failed to parse init x[{}] RHS='{}' : {}",
                            i, rhs, e
                        ));
                    }
                }
            }
            for (i, expr) in init_map.iter() {
                let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                    "x".to_string(),
                    Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
                );
                init_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                    lhs,
                    expr.clone(),
                ));
            }
        }
    } else {
        for (i, rhs) in
            crate::exa_wasm::interpreter::loader_helpers::extract_all_assign(&init_text, "x[")
        {
            let toks = tokenize(&rhs);
            let mut p = Parser::new(toks);
            let res = p.parse_expr_result();
            match res {
                Ok(expr) => {
                    init_map.insert(i, expr);
                }
                Err(e) => {
                    parse_errors.push(format!(
                        "failed to parse init x[{}] RHS='{}' : {}",
                        i, rhs, e
                    ));
                }
            }
        }
        for (i, expr) in init_map.iter() {
            let lhs = crate::exa_wasm::interpreter::ast::Lhs::Indexed(
                "x".to_string(),
                Box::new(crate::exa_wasm::interpreter::ast::Expr::Number(*i as f64)),
            );
            init_stmts.push(crate::exa_wasm::interpreter::ast::Stmt::Assign(
                lhs,
                expr.clone(),
            ));
        }
    }

    if let Some(lmap) = ir.lag_map.clone() {
        for (i, rhs) in lmap.into_iter() {
            let toks = tokenize(&rhs);
            let mut p = Parser::new(toks);
            match p.parse_expr_result() {
                Ok(expr) => {
                    lag_map.insert(i, expr);
                }
                Err(e) => {
                    parse_errors.push(format!(
                        "failed to parse lag! entry {} => '{}' : {}",
                        i, rhs, e
                    ));
                }
            }
        }
    } else {
        if !lag_text.trim().is_empty() {
            parse_errors.push("IR missing structured `lag_map` field; textual `lag!{}` parsing is no longer supported at runtime".to_string());
        }
    }
    if let Some(fmap) = ir.fa_map.clone() {
        for (i, rhs) in fmap.into_iter() {
            let toks = tokenize(&rhs);
            let mut p = Parser::new(toks);
            match p.parse_expr_result() {
                Ok(expr) => {
                    fa_map.insert(i, expr);
                }
                Err(e) => {
                    parse_errors.push(format!(
                        "failed to parse fa! entry {} => '{}' : {}",
                        i, rhs, e
                    ));
                }
            }
        }
    } else {
        if !fa_text.trim().is_empty() {
            parse_errors.push("IR missing structured `fa_map` field; textual `fa!{}` parsing is no longer supported at runtime".to_string());
        }
    }

    // fetch_params / fetch_cov helpers delegated to loader_helpers

    let mut fetch_macro_bodies: Vec<String> = Vec::new();
    fetch_macro_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_params(&diffeq_text));
    fetch_macro_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_params(&out_text));
    fetch_macro_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_params(&init_text));

    for body in fetch_macro_bodies.iter() {
        let parts: Vec<String> = body
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '"' || c == '\''))
            .map(|s| s.to_string())
            .collect();
        if parts.is_empty() {
            parse_errors.push(format!("empty fetch_params! macro body: '{}'", body));
            continue;
        }
        for name in parts.iter().skip(1) {
            if name.starts_with('_') {
                continue;
            }
            if !params.iter().any(|p| p == name) {
                parse_errors.push(format!(
                    "fetch_params! references unknown parameter '{}' not present in IR params {:?}",
                    name, params
                ));
            }
        }
    }

    let mut fetch_cov_bodies: Vec<String> = Vec::new();
    fetch_cov_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_cov(&diffeq_text));
    fetch_cov_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_cov(&out_text));
    fetch_cov_bodies
        .extend(crate::exa_wasm::interpreter::loader_helpers::extract_fetch_cov(&init_text));

    for body in fetch_cov_bodies.iter() {
        let parts: Vec<String> = body
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '"' || c == '\''))
            .map(|s| s.to_string())
            .collect();
        if parts.len() < 3 {
            parse_errors.push(format!(
                "fetch_cov! macro expects at least (cov, t, name...), got '{}'",
                body
            ));
            continue;
        }
        let cov_var = parts[0].clone();
        if cov_var.is_empty() || !cov_var.chars().next().unwrap().is_ascii_alphabetic() {
            parse_errors.push(format!(
                "invalid first argument '{}' in fetch_cov! macro",
                cov_var
            ));
        }
        let _tvar = parts[1].clone();
        if _tvar.is_empty() {
            parse_errors.push(format!(
                "invalid time argument '{}' in fetch_cov! macro",
                _tvar
            ));
        }
        for name in parts.iter().skip(2) {
            if name.is_empty() {
                parse_errors.push(format!(
                    "empty covariate name in fetch_cov! macro body '{}'",
                    body
                ));
            }
            if !name.starts_with('_')
                && !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                parse_errors.push(format!(
                    "invalid covariate identifier '{}' in fetch_cov! macro",
                    name
                ));
            }
        }
    }

    if diffeq_stmts.is_empty() {
        parse_errors.push(
            "no dx[...] assignments found in diffeq; emit_ir must populate dx entries in the IR"
                .to_string(),
        );
    }

    if !parse_errors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("parse errors: {}", parse_errors.join("; ")),
        ));
    }

    // expression validation delegated to loader_helpers

    // Determine number of states and output eqs from parsed assignments
    let max_dx =
        crate::exa_wasm::interpreter::loader_helpers::collect_max_index(&diffeq_stmts, "dx")
            .unwrap_or_else(|| dx_map.keys().copied().max().unwrap_or(0));
    let max_y = crate::exa_wasm::interpreter::loader_helpers::collect_max_index(&out_stmts, "y")
        .unwrap_or_else(|| out_map.keys().copied().max().unwrap_or(0));
    let nstates = max_dx + 1;
    let nouteqs = max_y + 1;

    let nparams = params.len();
    // Prelude and statement validation delegated to loader_helpers

    for s in diffeq_stmts.iter() {
        crate::exa_wasm::interpreter::loader_helpers::validate_stmt(
            s,
            &pmap,
            nstates,
            nparams,
            &mut parse_errors,
        );
    }
    for s in out_stmts.iter() {
        crate::exa_wasm::interpreter::loader_helpers::validate_stmt(
            s,
            &pmap,
            nstates,
            nparams,
            &mut parse_errors,
        );
    }
    for s in init_stmts.iter() {
        crate::exa_wasm::interpreter::loader_helpers::validate_stmt(
            s,
            &pmap,
            nstates,
            nparams,
            &mut parse_errors,
        );
    }
    for (_i, expr) in lag_map.iter() {
        crate::exa_wasm::interpreter::loader_helpers::validate_expr(
            expr,
            &pmap,
            nstates,
            nparams,
            &mut parse_errors,
        );
    }
    for (_i, expr) in fa_map.iter() {
        crate::exa_wasm::interpreter::loader_helpers::validate_expr(
            expr,
            &pmap,
            nstates,
            nparams,
            &mut parse_errors,
        );
    }

    // validate prelude ordering: each prelude RHS may reference params or earlier locals
    {
        let mut known: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (name, expr) in prelude.iter() {
            crate::exa_wasm::interpreter::loader_helpers::validate_prelude_expr(
                expr,
                &pmap,
                &known,
                nstates,
                nparams,
                &mut parse_errors,
            );
            known.insert(name.clone());
        }
    }

    if !parse_errors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("parse errors: {}", parse_errors.join("; ")),
        ));
    }

    let entry = registry::RegistryEntry {
        diffeq_stmts,
        out_stmts,
        init_stmts,
        lag: lag_map,
        fa: fa_map,
        prelude,
        pmap: pmap.clone(),
        nstates,
        _nouteqs: nouteqs,
    };

    let id = registry::register_entry(entry);

    let ode = crate::simulator::equation::ODE::with_registry_id(
        crate::exa_wasm::interpreter::dispatch::diffeq_dispatch,
        crate::exa_wasm::interpreter::dispatch::lag_dispatch,
        crate::exa_wasm::interpreter::dispatch::fa_dispatch,
        crate::exa_wasm::interpreter::dispatch::init_dispatch,
        crate::exa_wasm::interpreter::dispatch::out_dispatch,
        (nstates, nouteqs),
        Some(id),
    );
    Ok((ode, meta, id))
}

#[cfg(test)]
mod tests {
    use crate::exa_wasm::interpreter::ast::{Expr, Lhs, Stmt};
    use crate::exa_wasm::interpreter::parser::{tokenize, Parser};

    // simple extractor for the inner closure body used in tests
    fn extract_body(src: &str) -> String {
        let lb = src.find('{').expect("no '{' found");
        let rb = src.rfind('}').expect("no '}' found");
        src[lb + 1..rb].to_string()
    }

    fn extract_and_parse(src: &str) -> Vec<Stmt> {
        let mut cleaned = extract_body(src);
        // normalize booleans for parser (tests don't include macros)
        cleaned = cleaned.replace("true", "1.0").replace("false", "0.0");
        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        p.parse_statements().expect("parse_statements failed")
    }

    fn contains_dx_assign(stmt: &Stmt, idx_expected: usize) -> bool {
        match stmt {
            Stmt::Assign(lhs, _rhs) => match lhs {
                Lhs::Indexed(name, idx_expr) => {
                    if name == "dx" {
                        if let Expr::Number(n) = &**idx_expr {
                            return (*n as usize) == idx_expected;
                        }
                    }
                    false
                }
                _ => false,
            },
            Stmt::Block(v) => v.iter().any(|s| contains_dx_assign(s, idx_expected)),
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                contains_dx_assign(then_branch, idx_expected)
                    || else_branch
                        .as_ref()
                        .map(|b| contains_dx_assign(b, idx_expected))
                        .unwrap_or(false)
            }
            Stmt::Expr(_) => false,
        }
    }

    #[test]
    fn test_if_true_parsed_cond_is_one_and_assign_present() {
        let src = "|x, p, _t, dx, rateiv, _cov| { if true { dx[0] = -ke * x[0]; } }";
        let stmts = extract_and_parse(src);
        assert!(!stmts.is_empty());
        let mut found = false;
        for st in stmts.iter() {
            if let Stmt::If {
                cond, then_branch, ..
            } = st
            {
                if let Expr::Number(n) = cond {
                    assert_eq!(*n, 1.0f64);
                } else {
                    panic!("cond not normalized to number for 'true'");
                }
                assert!(contains_dx_assign(then_branch, 0));
                found = true;
                break;
            }
        }
        assert!(found, "No If statement found in parsed stmts");
    }

    #[test]
    fn test_if_false_parsed_cond_is_zero_and_assign_present() {
        let src = "|x, p, _t, dx, rateiv, _cov| { if false { dx[0] = -ke * x[0]; } }";
        let stmts = extract_and_parse(src);
        assert!(!stmts.is_empty());
        let mut found = false;
        for st in stmts.iter() {
            if let Stmt::If {
                cond, then_branch, ..
            } = st
            {
                if let Expr::Number(n) = cond {
                    assert_eq!(*n, 0.0f64);
                } else {
                    panic!("cond not normalized to number for 'false'");
                }
                // parser still preserves the assignment in the then-branch
                assert!(contains_dx_assign(then_branch, 0));
                found = true;
                break;
            }
        }
        assert!(found, "No If statement found in parsed stmts");
    }
}
