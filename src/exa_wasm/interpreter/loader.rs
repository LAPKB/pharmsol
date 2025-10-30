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
    fn extract_all_assign(src: &str, lhs_prefix: &str) -> Vec<(usize, String)> {
        let mut res = Vec::new();

        let mut brace_depth: isize = 0;
        let mut paren_depth: isize = 0;
        let mut stmt = String::new();

        // Helper: scan a collected top-level statement and extract any
        // `lhs_prefix` assignments that occur at brace nesting level 1.
        fn scan_stmt_collect(s: &str, lhs_prefix: &str, res: &mut Vec<(usize, String)>) {
            let mut depth: isize = 0;
            let bytes = s.as_bytes();
            let mut i: usize = 0;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch == '{' {
                    depth += 1;
                    i += 1;
                    continue;
                }
                if ch == '}' {
                    if depth > 0 {
                        depth -= 1;
                    }
                    i += 1;
                    continue;
                }
                // only consider matches when at depth == 1
                if depth == 1 {
                    if s[i..].starts_with(lhs_prefix) {
                        let after = &s[i + lhs_prefix.len()..];
                        if let Some(rb) = after.find(']') {
                            let idx_str = &after[..rb];
                            if let Ok(idx) = idx_str.trim().parse::<usize>() {
                                if let Some(eqpos) = after.find('=') {
                                    // find semicolon after eqpos
                                    if let Some(semi) = after[eqpos + 1..].find(';') {
                                        let rhs =
                                            after[eqpos + 1..eqpos + 1 + semi].trim().to_string();
                                        res.push((idx, rhs));
                                    }
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
        }

        for ch in src.chars() {
            match ch {
                '{' => {
                    brace_depth += 1;
                    if brace_depth >= 1 {
                        stmt.push(ch);
                    }
                }
                '}' => {
                    if brace_depth > 0 {
                        brace_depth -= 1;
                    }
                    if brace_depth >= 1 {
                        stmt.push(ch);
                        // If we've just closed an inner block and returned to
                        // the top-level closure body (depth == 1), treat the
                        // collected text as a complete top-level statement
                        // (this covers `if { ... }` without a trailing
                        // semicolon).
                        if paren_depth == 0 && brace_depth == 1 {
                            let s = stmt.trim();
                            if !s.is_empty() {
                                let s_trim = s.trim_start();
                                let s_work = if s_trim.starts_with('{') {
                                    s_trim[1..].trim_start()
                                } else {
                                    s_trim
                                };
                                if s_work.starts_with("if") {
                                    if let Some(lb_rel2) = s_work.find('{') {
                                        let lb2 = lb_rel2;
                                        let mut depth3: isize = 0;
                                        let bytes3 = s_work.as_bytes();
                                        let mut jj = lb2;
                                        let mut rb2_opt: Option<usize> = None;
                                        while jj < bytes3.len() {
                                            let ch3 = bytes3[jj] as char;
                                            if ch3 == '{' {
                                                depth3 += 1;
                                            } else if ch3 == '}' {
                                                depth3 -= 1;
                                                if depth3 == 0 {
                                                    rb2_opt = Some(jj);
                                                    break;
                                                }
                                            }
                                            jj += 1;
                                        }
                                        if let Some(rb2) = rb2_opt {
                                            let cond_txt_raw = &s_work
                                                [2..s_work.find('{').unwrap_or(s_work.len())];
                                            let mut cond_txt = cond_txt_raw.trim().to_string();
                                            if cond_txt.eq_ignore_ascii_case("true") {
                                                cond_txt = "1.0".to_string();
                                            } else if cond_txt.eq_ignore_ascii_case("false") {
                                                cond_txt = "0.0".to_string();
                                            }
                                            let inner_block = &s_work[lb2 + 1..rb2];
                                            let mut kk = 0usize;
                                            let inner_b = inner_block.as_bytes();
                                            while kk < inner_b.len() {
                                                if inner_block[kk..].starts_with(lhs_prefix) {
                                                    let after3 =
                                                        &inner_block[kk + lhs_prefix.len()..];
                                                    if let Some(rb3) = after3.find(']') {
                                                        let idx_str3 = &after3[..rb3];
                                                        if let Ok(idx3) =
                                                            idx_str3.trim().parse::<usize>()
                                                        {
                                                            if let Some(eqpos3) = after3.find('=') {
                                                                if let Some(semi3) =
                                                                    after3[eqpos3 + 1..].find(';')
                                                                {
                                                                    let rhs3 = after3[eqpos3 + 1
                                                                        ..eqpos3 + 1 + semi3]
                                                                        .trim();
                                                                    let tern3 = format!(
                                                                        "({}) ? ({}) : 0.0",
                                                                        cond_txt, rhs3
                                                                    );
                                                                    res.push((idx3, tern3));
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if let Some(next_semi3) =
                                                        inner_block[kk..].find(';')
                                                    {
                                                        kk += next_semi3 + 1;
                                                        continue;
                                                    } else {
                                                        break;
                                                    }
                                                }
                                                kk += 1;
                                            }
                                        }
                                    }
                                } else {
                                    scan_stmt_collect(s, lhs_prefix, &mut res);
                                }
                            }
                            stmt.clear();
                        }
                    }
                }
                '(' => {
                    paren_depth += 1;
                    if brace_depth >= 1 {
                        stmt.push(ch);
                    }
                }
                ')' => {
                    if paren_depth > 0 {
                        paren_depth -= 1;
                    }
                    if brace_depth >= 1 {
                        stmt.push(ch);
                    }
                }
                ';' => {
                    if brace_depth >= 1 {
                        // Treat statements finished at top-level inside the
                        // closure body (brace_depth == 1, not inside
                        // parentheses) as candidates for assignment
                        // extraction. Nested semicolons are kept inside the
                        // collected statement text.
                        if paren_depth == 0 && brace_depth == 1 {
                            // include the delimiter so downstream scanners can find ';'
                            stmt.push(';');
                            let s = stmt.trim();
                            if !s.is_empty() {
                                let s_trim = s.trim_start();
                                // allow an optional leading '{' (we collected it earlier)
                                let s_work = if s_trim.starts_with('{') {
                                    s_trim[1..].trim_start()
                                } else {
                                    s_trim
                                };
                                if s_work.starts_with("if") {
                                    // Handle top-level `if` statement: extract
                                    // condition and inner block, convert inner
                                    // `dx[...] = rhs;` assignments into
                                    // ternary RHS strings `cond ? rhs : 0.0`.
                                    if let Some(lb_rel2) = s_work.find('{') {
                                        let lb2 = lb_rel2;
                                        // find matching '}' within s_work
                                        let mut depth3: isize = 0;
                                        let bytes3 = s_work.as_bytes();
                                        let mut jj = lb2;
                                        let mut rb2_opt: Option<usize> = None;
                                        while jj < bytes3.len() {
                                            let ch3 = bytes3[jj] as char;
                                            if ch3 == '{' {
                                                depth3 += 1;
                                            } else if ch3 == '}' {
                                                depth3 -= 1;
                                                if depth3 == 0 {
                                                    rb2_opt = Some(jj);
                                                    break;
                                                }
                                            }
                                            jj += 1;
                                        }
                                        if let Some(rb2) = rb2_opt {
                                            let cond_txt_raw = &s_work
                                                [2..s_work.find('{').unwrap_or(s_work.len())];
                                            let mut cond_txt = cond_txt_raw.trim().to_string();
                                            if cond_txt.eq_ignore_ascii_case("true") {
                                                cond_txt = "1.0".to_string();
                                            } else if cond_txt.eq_ignore_ascii_case("false") {
                                                cond_txt = "0.0".to_string();
                                            }
                                            let inner_block = &s_work[lb2 + 1..rb2];
                                            // scan inner_block for lhs_prefix occurrences
                                            let mut kk = 0usize;
                                            let inner_b = inner_block.as_bytes();
                                            while kk < inner_b.len() {
                                                if inner_block[kk..].starts_with(lhs_prefix) {
                                                    let after3 =
                                                        &inner_block[kk + lhs_prefix.len()..];
                                                    if let Some(rb3) = after3.find(']') {
                                                        let idx_str3 = &after3[..rb3];
                                                        if let Ok(idx3) =
                                                            idx_str3.trim().parse::<usize>()
                                                        {
                                                            if let Some(eqpos3) = after3.find('=') {
                                                                if let Some(semi3) =
                                                                    after3[eqpos3 + 1..].find(';')
                                                                {
                                                                    let rhs3 = after3[eqpos3 + 1
                                                                        ..eqpos3 + 1 + semi3]
                                                                        .trim();
                                                                    let tern3 = format!(
                                                                        "({}) ? ({}) : 0.0",
                                                                        cond_txt, rhs3
                                                                    );
                                                                    res.push((idx3, tern3));
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if let Some(next_semi3) =
                                                        inner_block[kk..].find(';')
                                                    {
                                                        kk += next_semi3 + 1;
                                                        continue;
                                                    } else {
                                                        break;
                                                    }
                                                }
                                                kk += 1;
                                            }
                                        }
                                    }
                                } else {
                                    scan_stmt_collect(s, lhs_prefix, &mut res);
                                }
                            }
                            stmt.clear();
                            continue;
                        } else {
                            // nested semicolon -> keep it inside stmt
                            stmt.push(';');
                            continue;
                        }
                    } else {
                        // semicolon outside the closure body: ignore
                        stmt.clear();
                        continue;
                    }
                }
                _ => {
                    if brace_depth >= 1 {
                        stmt.push(ch);
                    }
                }
            }
        }

        // handle final stmt without trailing semicolon (scan depth-aware)
        let s = stmt.trim();
        if !s.is_empty() {
            scan_stmt_collect(s, lhs_prefix, &mut res);
        }

        res
    }

    // Prefer structural parsing of the closure body using the new statement
    // parser. This is more robust than substring scanning and allows us to
    // convert top-level `if` statements into conditional RHS expressions.
    fn extract_closure_body(src: &str) -> Option<String> {
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
                            // return inner text between first '{' and matching '}'
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

    // helper to strip macro calls like `fetch_params!(...)` from a text
    fn strip_macro_calls(s: &str, name: &str) -> String {
        let mut out = String::new();
        let mut i = 0usize;
        while i < s.len() {
            if s[i..].starts_with(name) {
                if let Some(lb_rel) = s[i..].find('(') {
                    let lb = i + lb_rel;
                    let mut depth: isize = 0;
                    let mut j = lb;
                    let mut found = None;
                    while j < s.len() {
                        match s.as_bytes()[j] as char {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    found = Some(j);
                                    break;
                                }
                            }
                            _ => {}
                        }
                        j += 1;
                    }
                    if let Some(rb) = found {
                        let mut k = rb + 1;
                        while k < s.len() && s.as_bytes()[k].is_ascii_whitespace() {
                            k += 1;
                        }
                        if k < s.len() && s.as_bytes()[k] as char == ';' {
                            i = k + 1;
                            continue;
                        }
                        i = rb + 1;
                        continue;
                    }
                }
            }
            out.push(s.as_bytes()[i] as char);
            i += 1;
        }
        out
    }

    // boolean literals are parsed by the tokenizer (Token::Bool). No normalization needed.

    if let Some(body) = extract_closure_body(&diffeq_text) {
        let mut cleaned = body.clone();
        cleaned = strip_macro_calls(&cleaned, "fetch_params!");
        cleaned = strip_macro_calls(&cleaned, "fetch_param!");
        cleaned = strip_macro_calls(&cleaned, "fetch_cov!");

        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(stmts) = p.parse_statements() {
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
            for (i, rhs) in extract_all_assign(&diffeq_text, "dx[") {
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
        for (i, rhs) in extract_all_assign(&diffeq_text, "dx[") {
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
    fn extract_prelude(src: &str) -> Vec<(String, String)> {
        let mut res = Vec::new();
        // remove single-line comments to avoid mixing comment text with assignments
        let cleaned = src
            .lines()
            .map(|l| match l.find("//") {
                Some(pos) => &l[..pos],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        for part in cleaned.split(';') {
            let s = part.trim();
            if s.is_empty() {
                continue;
            }
            if let Some(eqpos) = s.find('=') {
                let lhs = s[..eqpos].trim();
                let rhs = s[eqpos + 1..].trim();
                // ensure lhs is a simple identifier (no brackets, single token)
                if !lhs.contains('[')
                    && lhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                    && lhs
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_alphabetic())
                        .unwrap_or(false)
                {
                    res.push((lhs.to_string(), rhs.to_string()));
                }
            }
        }
        res
    }

    for (name, rhs) in extract_prelude(&diffeq_text) {
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
    // parse out closure into statements (fall back to extraction)
    if let Some(body) = extract_closure_body(&out_text) {
        let mut cleaned = body.clone();
        // strip macros
        cleaned = strip_macro_calls(&cleaned, "fetch_params!");
        cleaned = strip_macro_calls(&cleaned, "fetch_param!");
        cleaned = strip_macro_calls(&cleaned, "fetch_cov!");
        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(stmts) = p.parse_statements() {
            if let Err(e) = typecheck::check_statements(&stmts) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("type errors in out closure: {:?}", e),
                ));
            }
            out_stmts = stmts;
        } else {
            for (i, rhs) in extract_all_assign(&out_text, "y[") {
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
        for (i, rhs) in extract_all_assign(&out_text, "y[") {
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

    // parse init closure into statements
    if let Some(body) = extract_closure_body(&init_text) {
        let mut cleaned = body.clone();
        cleaned = strip_macro_calls(&cleaned, "fetch_params!");
        cleaned = strip_macro_calls(&cleaned, "fetch_param!");
        cleaned = strip_macro_calls(&cleaned, "fetch_cov!");
        let toks = tokenize(&cleaned);
        let mut p = Parser::new(toks);
        if let Some(stmts) = p.parse_statements() {
            if let Err(e) = typecheck::check_statements(&stmts) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("type errors in init closure: {:?}", e),
                ));
            }
            init_stmts = stmts;
        } else {
            for (i, rhs) in extract_all_assign(&init_text, "x[") {
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
        for (i, rhs) in extract_all_assign(&init_text, "x[") {
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

    // fetch_params / fetch_cov validation (copied from prior implementation)
    fn extract_fetch_params(src: &str) -> Vec<String> {
        let mut res = Vec::new();
        let mut rest = src;
        while let Some(pos) = rest.find("fetch_params!") {
            if let Some(lb) = rest[pos..].find('(') {
                let tail = &rest[pos + lb + 1..];
                if let Some(rb) = tail.find(')') {
                    let body = &tail[..rb];
                    res.push(body.to_string());
                    rest = &tail[rb + 1..];
                    continue;
                }
            }
            rest = &rest[pos + "fetch_params!".len()..];
        }
        // also catch common typo `fetch_param!`
        rest = src;
        while let Some(pos) = rest.find("fetch_param!") {
            if let Some(lb) = rest[pos..].find('(') {
                // find matching ')' allowing nested parentheses
                let mut i = pos + lb + 1;
                let mut depth = 0isize;
                let bytes = rest.as_bytes();
                let mut found = None;
                while i < rest.len() {
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
                    let body = &rest[pos + lb + 1..rb];
                    res.push(body.to_string());
                    rest = &rest[rb + 1..];
                    continue;
                }
            }
            rest = &rest[pos + "fetch_param!".len()..];
        }
        res
    }

    let mut fetch_macro_bodies: Vec<String> = Vec::new();
    fetch_macro_bodies.extend(extract_fetch_params(&diffeq_text));
    fetch_macro_bodies.extend(extract_fetch_params(&out_text));
    fetch_macro_bodies.extend(extract_fetch_params(&init_text));

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

    fn extract_fetch_cov(src: &str) -> Vec<String> {
        let mut res = Vec::new();
        let mut rest = src;
        while let Some(pos) = rest.find("fetch_cov!") {
            if let Some(lb) = rest[pos..].find('(') {
                let tail = &rest[pos + lb + 1..];
                if let Some(rb) = tail.find(')') {
                    let body = &tail[..rb];
                    res.push(body.to_string());
                    rest = &tail[rb + 1..];
                    continue;
                }
            }
            rest = &rest[pos + "fetch_cov!".len()..];
        }
        res
    }

    let mut fetch_cov_bodies: Vec<String> = Vec::new();
    fetch_cov_bodies.extend(extract_fetch_cov(&diffeq_text));
    fetch_cov_bodies.extend(extract_fetch_cov(&out_text));
    fetch_cov_bodies.extend(extract_fetch_cov(&init_text));

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

    // Validate expressions (copied from prior implementation)
    fn validate_expr(
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
                other => {
                    validate_expr(other, pmap, nstates, nparams, errors);
                }
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

    // Determine number of states and output eqs from parsed assignments
    fn collect_max_index(
        stmts: &Vec<crate::exa_wasm::interpreter::ast::Stmt>,
        name: &str,
    ) -> Option<usize> {
        let mut max: Option<usize> = None;
        fn visit(s: &crate::exa_wasm::interpreter::ast::Stmt, name: &str, max: &mut Option<usize>) {
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
                        visit(ss, name, max);
                    }
                }
                crate::exa_wasm::interpreter::ast::Stmt::If {
                    cond: _,
                    then_branch,
                    else_branch,
                } => {
                    visit(then_branch, name, max);
                    if let Some(eb) = else_branch {
                        visit(eb, name, max);
                    }
                }
                crate::exa_wasm::interpreter::ast::Stmt::Expr(_) => {}
            }
        }
        for s in stmts.iter() {
            visit(s, name, &mut max);
        }
        max
    }

    let max_dx = collect_max_index(&diffeq_stmts, "dx")
        .unwrap_or_else(|| dx_map.keys().copied().max().unwrap_or(0));
    let max_y = collect_max_index(&out_stmts, "y")
        .unwrap_or_else(|| out_map.keys().copied().max().unwrap_or(0));
    let nstates = max_dx + 1;
    let nouteqs = max_y + 1;

    let nparams = params.len();
    // validate prelude: ensure references are to params, t, or previously defined prelude names
    fn validate_prelude_expr(
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
    // Walk statement ASTs and validate embedded expressions
    fn validate_stmt(
        st: &crate::exa_wasm::interpreter::ast::Stmt,
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

    for s in diffeq_stmts.iter() {
        validate_stmt(s, &pmap, nstates, nparams, &mut parse_errors);
    }
    for s in out_stmts.iter() {
        validate_stmt(s, &pmap, nstates, nparams, &mut parse_errors);
    }
    for s in init_stmts.iter() {
        validate_stmt(s, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in lag_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in fa_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }

    // validate prelude ordering: each prelude RHS may reference params or earlier locals
    {
        let mut known: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (name, expr) in prelude.iter() {
            validate_prelude_expr(expr, &pmap, &known, nstates, nparams, &mut parse_errors);
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
