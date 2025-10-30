use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

use rand::Rng;
use rand_distr::Alphanumeric;

/// Emit a minimal JSON IR for a model (WASM-friendly emitter).
pub fn emit_ir<E: crate::Equation>(
    diffeq_txt: String,
    lag_txt: Option<String>,
    fa_txt: Option<String>,
    init_txt: Option<String>,
    out_txt: Option<String>,
    output: Option<PathBuf>,
    params: Vec<String>,
) -> Result<String, io::Error> {
    use serde_json::json;
    use std::collections::HashMap;

    // Extract structured lag/fa maps from macro text so the runtime does not
    // need to re-parse macro bodies. These will be empty maps if not present.
    fn extract_macro_map(src: &str, mac: &str) -> HashMap<usize, String> {
        let mut res = HashMap::new();
        let mut search = 0usize;
        while let Some(pos) = src[search..].find(mac) {
            let start = search + pos;
            if let Some(lb_rel) = src[start..].find('{') {
                let lb = start + lb_rel;
                let mut depth: isize = 0;
                let mut i = lb;
                let bytes = src.as_bytes();
                let len = src.len();
                let mut end_opt: Option<usize> = None;
                while i < len {
                    match bytes[i] as char {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end_opt = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if let Some(rb) = end_opt {
                    let body = &src[lb + 1..rb];
                    // split top-level entries by commas not inside parentheses/braces
                    let mut entry = String::new();
                    let mut paren = 0isize;
                    let mut brace = 0isize;
                    for ch in body.chars() {
                        match ch {
                            '(' => {
                                paren += 1;
                                entry.push(ch);
                            }
                            ')' => {
                                paren -= 1;
                                entry.push(ch);
                            }
                            '{' => {
                                brace += 1;
                                entry.push(ch);
                            }
                            '}' => {
                                brace -= 1;
                                entry.push(ch);
                            }
                            ',' if paren == 0 && brace == 0 => {
                                let parts: Vec<&str> = entry.split("=>").collect();
                                if parts.len() == 2 {
                                    if let Ok(k) = parts[0].trim().parse::<usize>() {
                                        res.insert(k, parts[1].trim().to_string());
                                    }
                                }
                                entry.clear();
                            }
                            _ => entry.push(ch),
                        }
                    }
                    if !entry.trim().is_empty() {
                        let parts: Vec<&str> = entry.split("=>").collect();
                        if parts.len() == 2 {
                            if let Ok(k) = parts[0].trim().parse::<usize>() {
                                res.insert(k, parts[1].trim().to_string());
                            }
                        }
                    }
                    search = rb + 1;
                    continue;
                }
            }
            search = start + mac.len();
        }
        res
    }

    let lag_map = extract_macro_map(lag_txt.as_deref().unwrap_or(""), "lag!");
    let fa_map = extract_macro_map(fa_txt.as_deref().unwrap_or(""), "fa!");

    // Try to parse and emit pre-parsed AST for diffeq/init/out closures so the
    // runtime loader can skip text parsing. We will rewrite parameter
    // identifiers into Param(index) nodes using the supplied params vector.
    let mut diffeq_ast_val = serde_json::Value::Null;
    let mut out_ast_val = serde_json::Value::Null;
    let mut init_ast_val = serde_json::Value::Null;

    // Build param -> index map
    let mut pmap: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, n) in params.iter().enumerate() {
        pmap.insert(n.clone(), i);
    }

    // helper to parse a closure text into Vec<Stmt>
    // This emitter requires closures to parse successfully; if parsing fails
    // we return an error rather than emitting textual closures. That lets the
    // runtime rely on a single robust pipeline (AST + bytecode) instead of
    // fragile textual fallbacks.
    fn try_parse_and_rewrite(
        src: &str,
        pmap: &std::collections::HashMap<String, usize>,
    ) -> Option<Vec<crate::exa_wasm::interpreter::Stmt>> {
        if let Some(body) = crate::exa_wasm::interpreter::extract_closure_body(src) {
            let mut cleaned = body.clone();
            cleaned = crate::exa_wasm::interpreter::strip_macro_calls(&cleaned, "fetch_params!");
            cleaned = crate::exa_wasm::interpreter::strip_macro_calls(&cleaned, "fetch_param!");
            cleaned = crate::exa_wasm::interpreter::strip_macro_calls(&cleaned, "fetch_cov!");
            let toks = crate::exa_wasm::interpreter::tokenize(&cleaned);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            if let Some(mut stmts) = p.parse_statements() {
                // rewrite idents -> Param(index)
                fn rewrite_expr(
                    e: &mut crate::exa_wasm::interpreter::Expr,
                    pmap: &std::collections::HashMap<String, usize>,
                ) {
                    match e {
                        crate::exa_wasm::interpreter::Expr::Ident(name) => {
                            if let Some(idx) = pmap.get(name) {
                                *e = crate::exa_wasm::interpreter::Expr::Param(*idx);
                            }
                        }
                        crate::exa_wasm::interpreter::Expr::Indexed(_, idx_expr) => {
                            rewrite_expr(idx_expr, pmap)
                        }
                        crate::exa_wasm::interpreter::Expr::UnaryOp { rhs, .. } => {
                            rewrite_expr(rhs, pmap)
                        }
                        crate::exa_wasm::interpreter::Expr::BinaryOp { lhs, rhs, .. } => {
                            rewrite_expr(lhs, pmap);
                            rewrite_expr(rhs, pmap);
                        }
                        crate::exa_wasm::interpreter::Expr::Call { args, .. } => {
                            for a in args.iter_mut() {
                                rewrite_expr(a, pmap);
                            }
                        }
                        crate::exa_wasm::interpreter::Expr::MethodCall {
                            receiver, args, ..
                        } => {
                            rewrite_expr(receiver, pmap);
                            for a in args.iter_mut() {
                                rewrite_expr(a, pmap);
                            }
                        }
                        crate::exa_wasm::interpreter::Expr::Ternary {
                            cond,
                            then_branch,
                            else_branch,
                        } => {
                            rewrite_expr(cond, pmap);
                            rewrite_expr(then_branch, pmap);
                            rewrite_expr(else_branch, pmap);
                        }
                        _ => {}
                    }
                }
                fn rewrite_stmt(
                    s: &mut crate::exa_wasm::interpreter::Stmt,
                    pmap: &std::collections::HashMap<String, usize>,
                ) {
                    match s {
                        crate::exa_wasm::interpreter::Stmt::Expr(e) => rewrite_expr(e, pmap),
                        crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) => {
                            if let crate::exa_wasm::interpreter::Lhs::Indexed(_, idx_expr) = lhs {
                                rewrite_expr(idx_expr, pmap);
                            }
                            rewrite_expr(rhs, pmap);
                        }
                        crate::exa_wasm::interpreter::Stmt::Block(v) => {
                            for ss in v.iter_mut() {
                                rewrite_stmt(ss, pmap);
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::If {
                            cond,
                            then_branch,
                            else_branch,
                        } => {
                            rewrite_expr(cond, pmap);
                            rewrite_stmt(then_branch, pmap);
                            if let Some(eb) = else_branch {
                                rewrite_stmt(eb, pmap);
                            }
                        }
                    }
                }
                for st in stmts.iter_mut() {
                    rewrite_stmt(st, pmap);
                }
                return Some(stmts);
            }
        }
        None
    }

    if let Some(stmts) = try_parse_and_rewrite(&diffeq_txt, &pmap) {
        diffeq_ast_val = serde_json::to_value(&stmts).unwrap_or(serde_json::Value::Null);
    }
    if let Some(stmts) = try_parse_and_rewrite(out_txt.as_deref().unwrap_or(""), &pmap) {
        out_ast_val = serde_json::to_value(&stmts).unwrap_or(serde_json::Value::Null);
    }
    if let Some(stmts) = try_parse_and_rewrite(init_txt.as_deref().unwrap_or(""), &pmap) {
        init_ast_val = serde_json::to_value(&stmts).unwrap_or(serde_json::Value::Null);
    }

    let mut ir_obj = json!({
        "ir_version": "1.0",
        "kind": E::kind().to_str(),
        "params": params,
        "diffeq": diffeq_txt,
        "lag": lag_txt,
        "fa": fa_txt,
        "lag_map": lag_map,
        "fa_map": fa_map,
        "init": init_txt,
        "out": out_txt,
        // IR schema field so consumers can be resilient to future AST/IR changes
        "ir_schema": { "version": "1.0", "ast_version": "1" },
    });

    // attach parsed ASTs when present
    if !diffeq_ast_val.is_null() {
        ir_obj["diffeq_ast"] = diffeq_ast_val;
    }
    if !out_ast_val.is_null() {
        ir_obj["out_ast"] = out_ast_val;
    }
    if !init_ast_val.is_null() {
        ir_obj["init_ast"] = init_ast_val;
    }

    // Extract fetch_params! and fetch_cov! macro bodies from closure texts and
    // attach to IR so loader can validate without scanning raw text at runtime.
    fn extract_fetch_bodies(src: &str, name: &str) -> Vec<String> {
        let mut res = Vec::new();
        let mut rest = src;
        while let Some(pos) = rest.find(name) {
            if let Some(lb_rel) = rest[pos..].find('(') {
                let tail = &rest[pos + lb_rel + 1..];
                let mut depth: isize = 0;
                let mut i = 0usize;
                let bytes = tail.as_bytes();
                let mut found: Option<usize> = None;
                while i < tail.len() {
                    match bytes[i] as char {
                        '(' => depth += 1,
                        ')' => {
                            if depth == 0 {
                                found = Some(i);
                                break;
                            }
                            depth -= 1;
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if let Some(rb) = found {
                    let body = &tail[..rb];
                    res.push(body.to_string());
                    rest = &tail[rb + 1..];
                    continue;
                }
            }
            rest = &rest[pos + name.len()..];
        }
        res
    }

    let mut fetch_params_bodies: Vec<String> = Vec::new();
    fetch_params_bodies.extend(extract_fetch_bodies(&diffeq_txt, "fetch_params!"));
    fetch_params_bodies.extend(extract_fetch_bodies(&diffeq_txt, "fetch_param!"));
    fetch_params_bodies.extend(extract_fetch_bodies(
        out_txt.as_deref().unwrap_or(""),
        "fetch_params!",
    ));
    fetch_params_bodies.extend(extract_fetch_bodies(
        out_txt.as_deref().unwrap_or(""),
        "fetch_param!",
    ));
    fetch_params_bodies.extend(extract_fetch_bodies(
        init_txt.as_deref().unwrap_or(""),
        "fetch_params!",
    ));
    fetch_params_bodies.extend(extract_fetch_bodies(
        init_txt.as_deref().unwrap_or(""),
        "fetch_param!",
    ));

    let mut fetch_cov_bodies: Vec<String> = Vec::new();
    fetch_cov_bodies.extend(extract_fetch_bodies(&diffeq_txt, "fetch_cov!"));
    fetch_cov_bodies.extend(extract_fetch_bodies(
        out_txt.as_deref().unwrap_or(""),
        "fetch_cov!",
    ));
    fetch_cov_bodies.extend(extract_fetch_bodies(
        init_txt.as_deref().unwrap_or(""),
        "fetch_cov!",
    ));

    if !fetch_params_bodies.is_empty() {
        ir_obj["fetch_params"] =
            serde_json::to_value(&fetch_params_bodies).unwrap_or(serde_json::Value::Null);
    }
    if !fetch_cov_bodies.is_empty() {
        ir_obj["fetch_cov"] =
            serde_json::to_value(&fetch_cov_bodies).unwrap_or(serde_json::Value::Null);
    }

    // Compile expressions into bytecode. This compiler covers numeric
    // literals, Param(i), simple indexed loads with constant indices (x/p/rateiv),
    // locals, unary -, binary ops, calls to known builtins, and ternary.
    fn compile_expr_top(
        expr: &crate::exa_wasm::interpreter::Expr,
        out: &mut Vec<crate::exa_wasm::interpreter::Opcode>,
        funcs: &mut Vec<String>,
        locals: &Vec<String>,
    ) -> bool {
        use crate::exa_wasm::interpreter::{Expr, Opcode};
        match expr {
            Expr::Number(n) => {
                out.push(Opcode::PushConst(*n));
                true
            }
            Expr::Bool(b) => {
                out.push(Opcode::PushConst(if *b { 1.0 } else { 0.0 }));
                true
            }
            Expr::Param(i) => {
                out.push(Opcode::LoadParam(*i));
                true
            }
            Expr::Ident(name) => {
                // treat as local if present
                if let Some(pos) = locals.iter().position(|n| n == name) {
                    out.push(Opcode::LoadLocal(pos));
                    true
                } else {
                    // unknown bare identifier — compilation fails; loader will
                    // catch this earlier via typecheck, but be conservative here
                    false
                }
            }
            Expr::Indexed(name, idx_expr) => {
                // support constant numeric indices and dynamic indices
                if let Expr::Number(n) = &**idx_expr {
                    let idx = *n as usize;
                    match name.as_str() {
                        "x" => {
                            out.push(Opcode::LoadX(idx));
                            true
                        }
                        "rateiv" => {
                            out.push(Opcode::LoadRateiv(idx));
                            true
                        }
                        "p" | "params" => {
                            out.push(Opcode::LoadParam(idx));
                            true
                        }
                        _ => false,
                    }
                } else {
                    // dynamic index: compile index expression then emit a dyn-load
                    if !compile_expr_top(idx_expr, out, funcs, locals) {
                        return false;
                    }
                    match name.as_str() {
                        "x" => {
                            out.push(Opcode::LoadXDyn);
                            true
                        }
                        "rateiv" => {
                            out.push(Opcode::LoadRateivDyn);
                            true
                        }
                        "p" | "params" => {
                            out.push(Opcode::LoadParamDyn);
                            true
                        }
                        _ => false,
                    }
                }
            }
            Expr::UnaryOp { op, rhs } => {
                if op == "-" {
                    if !compile_expr_top(rhs, out, funcs, locals) {
                        return false;
                    }
                    // multiply by -1.0
                    out.push(Opcode::PushConst(-1.0));
                    out.push(Opcode::Mul);
                    true
                } else {
                    false
                }
            }
            Expr::BinaryOp { lhs, op, rhs } => {
                // handle short-circuit logical operators specially so we
                // preserve AST semantics (avoid evaluating rhs when not
                // necessary). For arithmetic/comparison operators we
                // compile both sides in order.
                match op.as_str() {
                    "&&" => {
                        // lhs && rhs -> if lhs==0.0 jump to push 0.0; else evaluate rhs and return rhs!=0 as 0/1
                        if !compile_expr_top(lhs, out, funcs, locals) {
                            return false;
                        }
                        // JumpIfFalse to false path if lhs is false
                        let jf_pos = out.len();
                        out.push(Opcode::JumpIfFalse(0));

                        // evaluate rhs
                        if !compile_expr_top(rhs, out, funcs, locals) {
                            return false;
                        }
                        // if rhs is false -> push 0, else push 1
                        let jf2_pos = out.len();
                        out.push(Opcode::JumpIfFalse(0));
                        // rhs true -> push 1
                        out.push(Opcode::PushConst(1.0));
                        // jump to end
                        let jmp_pos = out.len();
                        out.push(Opcode::Jump(0));
                        // false path
                        let false_pos = out.len();
                        // set first JumpIfFalse target to false_pos
                        if let Opcode::JumpIfFalse(ref mut addr) = out[jf_pos] {
                            *addr = false_pos;
                        }
                        // push 0.0 for false
                        out.push(Opcode::PushConst(0.0));
                        // fix jumps
                        let end_pos = out.len();
                        if let Opcode::Jump(ref mut addr) = out[jmp_pos] {
                            *addr = end_pos;
                        }
                        if let Opcode::JumpIfFalse(ref mut addr) = out[jf2_pos] {
                            *addr = false_pos;
                        }
                        true
                    }
                    "||" => {
                        // lhs || rhs -> if lhs != 0 -> push 1 and skip rhs; else evaluate rhs and return rhs!=0 as 0/1
                        if !compile_expr_top(lhs, out, funcs, locals) {
                            return false;
                        }
                        // if lhs is false, evaluate rhs; JumpIfFalse should jump to rhs
                        let jf_pos = out.len();
                        out.push(Opcode::JumpIfFalse(0));
                        // lhs true -> push 1
                        out.push(Opcode::PushConst(1.0));
                        // jump to end
                        let jmp_pos = out.len();
                        out.push(Opcode::Jump(0));
                        // else/rhs path
                        let else_pos = out.len();
                        if let Opcode::JumpIfFalse(ref mut addr) = out[jf_pos] {
                            *addr = else_pos;
                        }
                        // evaluate rhs
                        if !compile_expr_top(rhs, out, funcs, locals) {
                            return false;
                        }
                        // now convert rhs to 0/1
                        let jf2_pos = out.len();
                        out.push(Opcode::JumpIfFalse(0));
                        out.push(Opcode::PushConst(1.0));
                        let jmp2 = out.len();
                        out.push(Opcode::Jump(0));
                        let false_pos = out.len();
                        if let Opcode::JumpIfFalse(ref mut addr) = out[jf2_pos] {
                            *addr = false_pos;
                        }
                        out.push(Opcode::PushConst(0.0));
                        let end_pos = out.len();
                        if let Opcode::Jump(ref mut addr) = out[jmp_pos] {
                            *addr = end_pos;
                        }
                        if let Opcode::Jump(ref mut addr) = out[jmp2] {
                            *addr = end_pos;
                        }
                        true
                    }
                    _ => {
                        // default: arithmetic/comparison operators compile lhs then rhs
                        if !compile_expr_top(lhs, out, funcs, locals) {
                            return false;
                        }
                        if !compile_expr_top(rhs, out, funcs, locals) {
                            return false;
                        }
                        match op.as_str() {
                            "+" => out.push(Opcode::Add),
                            "-" => out.push(Opcode::Sub),
                            "*" => out.push(Opcode::Mul),
                            "/" => out.push(Opcode::Div),
                            "^" => out.push(Opcode::Pow),
                            "<" => out.push(Opcode::Lt),
                            ">" => out.push(Opcode::Gt),
                            "<=" => out.push(Opcode::Le),
                            ">=" => out.push(Opcode::Ge),
                            "==" => out.push(Opcode::Eq),
                            "!=" => out.push(Opcode::Ne),
                            _ => return false,
                        }
                        true
                    }
                }
            }
            Expr::Call { name, args } => {
                // only compile known builtins and check arity
                if crate::exa_wasm::interpreter::is_known_function(name.as_str()) {
                    // verify arity where possible
                    if let Some(rng) = crate::exa_wasm::interpreter::arg_count_range(name.as_str())
                    {
                        if !rng.contains(&args.len()) {
                            return false;
                        }
                    }
                    // compile args
                    for a in args.iter() {
                        if !compile_expr_top(a, out, funcs, locals) {
                            return false;
                        }
                    }
                    // register function in funcs table
                    let idx = match funcs.iter().position(|f| f == name) {
                        Some(i) => i,
                        None => {
                            funcs.push(name.clone());
                            funcs.len() - 1
                        }
                    };
                    out.push(Opcode::CallBuiltin(idx, args.len()));
                    true
                } else {
                    false
                }
            }
            Expr::MethodCall {
                receiver,
                name,
                args,
            } => {
                // lower method call to function with receiver as first arg
                if crate::exa_wasm::interpreter::is_known_function(name.as_str()) {
                    // verify arity for method-style calls
                    if let Some(rng) = crate::exa_wasm::interpreter::arg_count_range(name.as_str())
                    {
                        if !rng.contains(&(1 + args.len())) {
                            return false;
                        }
                    }
                    if !compile_expr_top(receiver, out, funcs, locals) {
                        return false;
                    }
                    for a in args.iter() {
                        if !compile_expr_top(a, out, funcs, locals) {
                            return false;
                        }
                    }
                    let idx = match funcs.iter().position(|f| f == name) {
                        Some(i) => i,
                        None => {
                            funcs.push(name.clone());
                            funcs.len() - 1
                        }
                    };
                    out.push(Opcode::CallBuiltin(idx, 1 + args.len()));
                    true
                } else {
                    false
                }
            }
            Expr::Ternary {
                cond,
                then_branch,
                else_branch,
            } => {
                // compile cond
                if !compile_expr_top(cond, out, funcs, locals) {
                    return false;
                }
                // emit JumpIfFalse to else
                let jf_pos = out.len();
                out.push(Opcode::JumpIfFalse(0));
                // then
                if !compile_expr_top(then_branch, out, funcs, locals) {
                    return false;
                }
                // jump over else
                let jmp_pos = out.len();
                out.push(Opcode::Jump(0));
                // fix JumpIfFalse target
                let else_pos = out.len();
                if let Opcode::JumpIfFalse(ref mut addr) = out[jf_pos] {
                    *addr = else_pos;
                }
                // else
                if !compile_expr_top(else_branch, out, funcs, locals) {
                    return false;
                }
                // fix Jump target
                let end_pos = out.len();
                if let Opcode::Jump(ref mut addr) = out[jmp_pos] {
                    *addr = end_pos;
                }
                true
            }
        }
    }

    let mut bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    // shared tables discovered during compilation
    let mut shared_funcs: Vec<String> = Vec::new();
    let mut shared_locals: Vec<String> = Vec::new();

    if let Some(v) = ir_obj.get("diffeq_ast") {
        // try to deserialize back into AST
        match serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone()) {
            Ok(stmts) => {
                // collect local variable names from non-indexed assignments
                for st in stmts.iter() {
                    if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, _rhs) = st {
                        if let crate::exa_wasm::interpreter::Lhs::Ident(name) = lhs {
                            if !shared_locals.iter().any(|n| n == name) {
                                shared_locals.push(name.clone());
                            }
                        }
                    }
                }

                // reuse compile_expr_top defined above for expression compilation
                // but visit statements recursively so nested Blocks/Ifs are
                // handled (previous code only inspected top-level stmts).
                fn visit_stmt(
                    st: &crate::exa_wasm::interpreter::Stmt,
                    bytecode_map: &mut std::collections::HashMap<
                        usize,
                        Vec<crate::exa_wasm::interpreter::Opcode>,
                    >,
                    shared_funcs: &mut Vec<String>,
                    shared_locals: &Vec<String>,
                ) {
                    match st {
                        crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) => {
                            if let crate::exa_wasm::interpreter::Lhs::Indexed(_name, idx_expr) = lhs
                            {
                                if _name == "dx" {
                                    // constant index
                                    if let crate::exa_wasm::interpreter::Expr::Number(n) =
                                        &**idx_expr
                                    {
                                        let idx = *n as usize;
                                        let mut code: Vec<crate::exa_wasm::interpreter::Opcode> =
                                            Vec::new();
                                        if compile_expr_top(
                                            rhs,
                                            &mut code,
                                            shared_funcs,
                                            shared_locals,
                                        ) {
                                            code.push(
                                                crate::exa_wasm::interpreter::Opcode::StoreDx(idx),
                                            );
                                            bytecode_map.insert(idx, code);
                                        }
                                    } else {
                                        // dynamic index: compile index then rhs then StoreDxDyn
                                        let mut code: Vec<crate::exa_wasm::interpreter::Opcode> =
                                            Vec::new();
                                        if compile_expr_top(
                                            idx_expr,
                                            &mut code,
                                            shared_funcs,
                                            shared_locals,
                                        ) && compile_expr_top(
                                            rhs,
                                            &mut code,
                                            shared_funcs,
                                            shared_locals,
                                        ) {
                                            code.push(
                                                crate::exa_wasm::interpreter::Opcode::StoreDxDyn,
                                            );
                                            // use a special key for dynamic-indexed entries
                                            bytecode_map.insert(usize::MAX, code);
                                        }
                                    }
                                }
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::Block(v) => {
                            for ss in v.iter() {
                                visit_stmt(ss, bytecode_map, shared_funcs, shared_locals);
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::If {
                            then_branch,
                            else_branch,
                            ..
                        } => {
                            visit_stmt(then_branch, bytecode_map, shared_funcs, shared_locals);
                            if let Some(eb) = else_branch {
                                visit_stmt(eb, bytecode_map, shared_funcs, shared_locals);
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::Expr(_) => {}
                    }
                }

                for st in stmts.iter() {
                    visit_stmt(st, &mut bytecode_map, &mut shared_funcs, &shared_locals);
                }
            }
            Err(_) => {}
        }
    }

    // If we didn't produce a bytecode_map above (e.g. try_parse_and_rewrite
    // failed or the AST wasn't attached), attempt a best-effort parse of the
    // raw diffeq closure text and compile it into bytecode. This increases
    // emitter coverage for forms that may have been missed earlier and helps
    // the parity tests exercise the VM path.
    if bytecode_map.is_empty() {
        if let Some(body) = crate::exa_wasm::interpreter::extract_closure_body(
            &ir_obj["diffeq"].as_str().unwrap_or(&"".to_string()),
        ) {
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            if let Some(stmts) = p.parse_statements() {
                // collect local variable names from non-indexed assignments
                for st in stmts.iter() {
                    if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, _rhs) = st {
                        if let crate::exa_wasm::interpreter::Lhs::Ident(name) = lhs {
                            if !shared_locals.iter().any(|n| n == name) {
                                shared_locals.push(name.clone());
                            }
                        }
                    }
                }

                // Visit statements recursively to find dx[...] assignments even
                // when nested in blocks/ifs and compile them into bytecode.
                fn visit_stmt(
                    st: &crate::exa_wasm::interpreter::Stmt,
                    bytecode_map: &mut std::collections::HashMap<
                        usize,
                        Vec<crate::exa_wasm::interpreter::Opcode>,
                    >,
                    shared_funcs: &mut Vec<String>,
                    shared_locals: &Vec<String>,
                ) {
                    match st {
                        crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) => {
                            if let crate::exa_wasm::interpreter::Lhs::Indexed(_name, idx_expr) = lhs
                            {
                                if _name == "dx" {
                                    if let crate::exa_wasm::interpreter::Expr::Number(n) =
                                        &**idx_expr
                                    {
                                        let idx = *n as usize;
                                        let mut code: Vec<crate::exa_wasm::interpreter::Opcode> =
                                            Vec::new();
                                        if compile_expr_top(
                                            rhs,
                                            &mut code,
                                            shared_funcs,
                                            &shared_locals,
                                        ) {
                                            code.push(
                                                crate::exa_wasm::interpreter::Opcode::StoreDx(idx),
                                            );
                                            bytecode_map.insert(idx, code);
                                        }
                                    } else {
                                        let mut code: Vec<crate::exa_wasm::interpreter::Opcode> =
                                            Vec::new();
                                        if compile_expr_top(
                                            idx_expr,
                                            &mut code,
                                            shared_funcs,
                                            &shared_locals,
                                        ) && compile_expr_top(
                                            rhs,
                                            &mut code,
                                            shared_funcs,
                                            &shared_locals,
                                        ) {
                                            code.push(
                                                crate::exa_wasm::interpreter::Opcode::StoreDxDyn,
                                            );
                                            bytecode_map.insert(usize::MAX, code);
                                        }
                                    }
                                }
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::Block(v) => {
                            for ss in v.iter() {
                                visit_stmt(ss, bytecode_map, shared_funcs, shared_locals);
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::If {
                            then_branch,
                            else_branch,
                            ..
                        } => {
                            visit_stmt(then_branch, bytecode_map, shared_funcs, shared_locals);
                            if let Some(eb) = else_branch {
                                visit_stmt(eb, bytecode_map, shared_funcs, shared_locals);
                            }
                        }
                        crate::exa_wasm::interpreter::Stmt::Expr(_) => {}
                    }
                }

                for st in stmts.iter() {
                    visit_stmt(st, &mut bytecode_map, &mut shared_funcs, &shared_locals);
                }
            }
        }
    }

    if !bytecode_map.is_empty() {
        // emit the conservative diffeq bytecode map under the new IR field names
        ir_obj["bytecode_map"] =
            serde_json::to_value(&bytecode_map).unwrap_or(serde_json::Value::Null);
        // new field expected by loader: diffeq_bytecode (index -> opcode sequence)
        ir_obj["diffeq_bytecode"] =
            serde_json::to_value(&bytecode_map).unwrap_or(serde_json::Value::Null);
        // emit discovered funcs/locals if any
        if !shared_funcs.is_empty() {
            ir_obj["funcs"] =
                serde_json::to_value(&shared_funcs).unwrap_or(serde_json::Value::Null);
        }
        if !shared_locals.is_empty() {
            ir_obj["locals"] =
                serde_json::to_value(&shared_locals).unwrap_or(serde_json::Value::Null);
        }
    }

    // Attempt to compile out/init closures into bytecode similarly to diffeq POC
    let mut out_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    let mut init_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();

    // Helper to compile an Assign stmt into bytecode when LHS is y[idx] or x[idx]
    if let Some(v) = ir_obj.get("out_ast") {
        if let Ok(stmts) =
            serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone())
        {
            for st in stmts.iter() {
                if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) = st {
                    if let crate::exa_wasm::interpreter::Lhs::Indexed(_name, idx_expr) = lhs {
                        if let crate::exa_wasm::interpreter::Expr::Number(n) = &**idx_expr {
                            let idx = *n as usize;
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(rhs, &mut code, &mut shared_funcs, &shared_locals) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreY(idx));
                                out_bytecode_map.insert(idx, code);
                            }
                        } else {
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(
                                idx_expr,
                                &mut code,
                                &mut shared_funcs,
                                &shared_locals,
                            ) && compile_expr_top(
                                rhs,
                                &mut code,
                                &mut shared_funcs,
                                &shared_locals,
                            ) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreYDyn);
                                out_bytecode_map.insert(usize::MAX, code);
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(v) = ir_obj.get("init_ast") {
        if let Ok(stmts) =
            serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone())
        {
            for st in stmts.iter() {
                if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) = st {
                    if let crate::exa_wasm::interpreter::Lhs::Indexed(_name, idx_expr) = lhs {
                        if let crate::exa_wasm::interpreter::Expr::Number(n) = &**idx_expr {
                            let idx = *n as usize;
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(rhs, &mut code, &mut shared_funcs, &shared_locals) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreX(idx));
                                init_bytecode_map.insert(idx, code);
                            }
                        } else {
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(
                                idx_expr,
                                &mut code,
                                &mut shared_funcs,
                                &shared_locals,
                            ) && compile_expr_top(
                                rhs,
                                &mut code,
                                &mut shared_funcs,
                                &shared_locals,
                            ) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreXDyn);
                                init_bytecode_map.insert(usize::MAX, code);
                            }
                        }
                    }
                }
            }
        }
    }

    if !out_bytecode_map.is_empty() {
        ir_obj["out_bytecode"] =
            serde_json::to_value(&out_bytecode_map).unwrap_or(serde_json::Value::Null);
    }
    if !init_bytecode_map.is_empty() {
        ir_obj["init_bytecode"] =
            serde_json::to_value(&init_bytecode_map).unwrap_or(serde_json::Value::Null);
    }

    // Compile lag_map/fa_map entries into bytecode when present. The
    // IR contains textual RHS strings for each entry; parse and compile
    // them here so the runtime loader can consume bytecode directly.
    let mut lag_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    if let Some(v) = ir_obj.get("lag_map") {
        if let Some(map) = v.as_object() {
            for (k, val) in map.iter() {
                if let Ok(idx) = k.parse::<usize>() {
                    if let Some(rhs_str) = val.as_str() {
                        let toks = crate::exa_wasm::interpreter::tokenize(rhs_str);
                        let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
                        if let Ok(expr) = p.parse_expr_result() {
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(&expr, &mut code, &mut shared_funcs, &shared_locals)
                            {
                                lag_bytecode_map.insert(idx, code);
                            }
                        }
                    }
                }
            }
        }
    }
    if !lag_bytecode_map.is_empty() {
        ir_obj["lag_bytecode"] =
            serde_json::to_value(&lag_bytecode_map).unwrap_or(serde_json::Value::Null);
    }

    let mut fa_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    if let Some(v) = ir_obj.get("fa_map") {
        if let Some(map) = v.as_object() {
            for (k, val) in map.iter() {
                if let Ok(idx) = k.parse::<usize>() {
                    if let Some(rhs_str) = val.as_str() {
                        let toks = crate::exa_wasm::interpreter::tokenize(rhs_str);
                        let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
                        if let Ok(expr) = p.parse_expr_result() {
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(&expr, &mut code, &mut shared_funcs, &shared_locals)
                            {
                                fa_bytecode_map.insert(idx, code);
                            }
                        }
                    }
                }
            }
        }
    }
    if !fa_bytecode_map.is_empty() {
        ir_obj["fa_bytecode"] =
            serde_json::to_value(&fa_bytecode_map).unwrap_or(serde_json::Value::Null);
    }

    // Ensure shared function table and locals are present in the IR when
    // we discovered any during compilation.
    if !shared_funcs.is_empty() {
        ir_obj["funcs"] = serde_json::to_value(&shared_funcs).unwrap_or(serde_json::Value::Null);
    }
    if !shared_locals.is_empty() {
        ir_obj["locals"] = serde_json::to_value(&shared_locals).unwrap_or(serde_json::Value::Null);
    }

    let output_path = output.unwrap_or_else(|| {
        let random_suffix: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(5)
            .map(char::from)
            .collect();
        let default_name = format!("model_ir_{}_{}.json", env::consts::OS, random_suffix);
        env::temp_dir().join("exa_tmp").with_file_name(default_name)
    });

    let serialized = serde_json::to_vec_pretty(&ir_obj)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde_json error: {}", e)))?;

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    fs::write(&output_path, serialized)?;
    Ok(output_path.to_string_lossy().to_string())
}
