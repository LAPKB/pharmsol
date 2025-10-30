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

    // Attempt to compile a tiny bytecode for simple dx assignments found in
    // the parsed diffeq AST. This is a conservative, best-effort POC: only
    // compile assignments where the LHS is `dx[const]` and RHS contains
    // numeric constants, Params and binary ops (+ - * / ^).
    // small expression compiler reused for diffeq/out/init compilation
    fn compile_expr_top(
        expr: &crate::exa_wasm::interpreter::Expr,
        out: &mut Vec<crate::exa_wasm::interpreter::Opcode>,
    ) -> bool {
        match expr {
            crate::exa_wasm::interpreter::Expr::Number(n) => {
                out.push(crate::exa_wasm::interpreter::Opcode::PushConst(*n));
                true
            }
            crate::exa_wasm::interpreter::Expr::Param(i) => {
                out.push(crate::exa_wasm::interpreter::Opcode::LoadParam(*i));
                true
            }
            crate::exa_wasm::interpreter::Expr::BinaryOp { lhs, op, rhs } => {
                if !compile_expr_top(lhs, out) {
                    return false;
                }
                if !compile_expr_top(rhs, out) {
                    return false;
                }
                match op.as_str() {
                    "+" => out.push(crate::exa_wasm::interpreter::Opcode::Add),
                    "-" => out.push(crate::exa_wasm::interpreter::Opcode::Sub),
                    "*" => out.push(crate::exa_wasm::interpreter::Opcode::Mul),
                    "/" => out.push(crate::exa_wasm::interpreter::Opcode::Div),
                    "^" => out.push(crate::exa_wasm::interpreter::Opcode::Pow),
                    _ => return false,
                }
                true
            }
            _ => false,
        }
    }

    let mut bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    if let Some(v) = ir_obj.get("diffeq_ast") {
        // try to deserialize back into AST
        match serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone()) {
            Ok(stmts) => {
                // reuse compile_expr_top defined above for expression compilation

                for st in stmts.iter() {
                    if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) = st {
                        if let crate::exa_wasm::interpreter::Lhs::Indexed(name, idx_expr) = lhs {
                            if name == "dx" {
                                // only constant numeric index supported in POC
                                match &**idx_expr {
                                    crate::exa_wasm::interpreter::Expr::Number(n) => {
                                        let idx = *n as usize;
                                        let mut code: Vec<crate::exa_wasm::interpreter::Opcode> =
                                            Vec::new();
                                        if compile_expr_top(rhs, &mut code) {
                                            code.push(
                                                crate::exa_wasm::interpreter::Opcode::StoreDx(idx),
                                            );
                                            bytecode_map.insert(idx, code);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }

    if !bytecode_map.is_empty() {
        // emit the conservative diffeq bytecode map under the new IR field names
        ir_obj["bytecode_map"] =
            serde_json::to_value(&bytecode_map).unwrap_or(serde_json::Value::Null);
        // new field expected by loader: diffeq_bytecode (index -> opcode sequence)
        ir_obj["diffeq_bytecode"] =
            serde_json::to_value(&bytecode_map).unwrap_or(serde_json::Value::Null);
        // emit empty funcs/locals placeholders for now
        ir_obj["funcs"] = serde_json::to_value(Vec::<String>::new()).unwrap();
        ir_obj["locals"] = serde_json::to_value(Vec::<String>::new()).unwrap();
    }

    // Attempt to compile out/init closures into bytecode similarly to diffeq POC
    let mut out_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();
    let mut init_bytecode_map: HashMap<usize, Vec<crate::exa_wasm::interpreter::Opcode>> =
        HashMap::new();

    // Helper to compile an Assign stmt into bytecode when LHS is y[idx] or x[idx]
    if let Some(v) = ir_obj.get("out_ast") {
        if let Ok(stmts) = serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone()) {
            for st in stmts.iter() {
                if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) = st {
                    if let crate::exa_wasm::interpreter::Lhs::Indexed(name, idx_expr) = lhs {
                        if let crate::exa_wasm::interpreter::Expr::Number(n) = &**idx_expr {
                            let idx = *n as usize;
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(rhs, &mut code) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreY(idx));
                                out_bytecode_map.insert(idx, code);
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(v) = ir_obj.get("init_ast") {
        if let Ok(stmts) = serde_json::from_value::<Vec<crate::exa_wasm::interpreter::Stmt>>(v.clone()) {
            for st in stmts.iter() {
                if let crate::exa_wasm::interpreter::Stmt::Assign(lhs, rhs) = st {
                    if let crate::exa_wasm::interpreter::Lhs::Indexed(name, idx_expr) = lhs {
                        if let crate::exa_wasm::interpreter::Expr::Number(n) = &**idx_expr {
                            let idx = *n as usize;
                            let mut code: Vec<crate::exa_wasm::interpreter::Opcode> = Vec::new();
                            if compile_expr_top(rhs, &mut code) {
                                code.push(crate::exa_wasm::interpreter::Opcode::StoreX(idx));
                                init_bytecode_map.insert(idx, code);
                            }
                        }
                    }
                }
            }
        }
    }

    if !out_bytecode_map.is_empty() {
        ir_obj["out_bytecode"] = serde_json::to_value(&out_bytecode_map).unwrap_or(serde_json::Value::Null);
    }
    if !init_bytecode_map.is_empty() {
        ir_obj["init_bytecode"] = serde_json::to_value(&init_bytecode_map).unwrap_or(serde_json::Value::Null);
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
