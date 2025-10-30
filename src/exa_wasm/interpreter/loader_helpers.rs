use crate::exa_wasm::interpreter::ast::{Expr, Stmt};
use std::collections::HashMap;

// Loader helper utilities used by `loader.rs`. These functions implement a
// conservative extraction and validation surface that mirrors the prior inline
// implementations in `loader.rs` so they can be reused and unit-tested.

// Loader helper utilities extracted from the large `load_ir_ode` function.

// ongoing refactor can wire them into `loader.rs` incrementally.

/// Extract top-level assignments like `dx[i] = expr;` from an IR closure
/// body string. The function attempts to only collect assignments that
/// live at the first brace nesting level (i.e. direct children of the
/// closure body). For simple top-level `if cond { dx[i] = rhs; }`
/// constructs the helper will convert those into a ternary-style RHS
/// string `cond ? rhs : 0.0` and return it as if it were a direct
/// assignment.
pub fn extract_all_assign(src: &str, lhs_prefix: &str) -> Vec<(usize, String)> {
    let mut res = Vec::new();

    let mut brace_depth: isize = 0;
    let mut paren_depth: isize = 0;
    let mut stmt = String::new();

    // scan a collected statement for direct lhs_prefix assignments
    fn scan_stmt_collect(s: &str, lhs_prefix: &str, res: &mut Vec<(usize, String)>) {
        let bytes = s.as_bytes();
        let mut i: usize = 0;
        while i < bytes.len() {
            let ch = bytes[i] as char;
            if ch == '{' || ch == '}' {
                i += 1;
                continue;
            }
            if let Some(rel) = s[i..].find(lhs_prefix) {
                let pos = i + rel;
                let after = &s[pos + lhs_prefix.len()..];
                if let Some(rb) = after.find(']') {
                    let idx_str = &after[..rb];
                    if let Ok(idx) = idx_str.trim().parse::<usize>() {
                        if let Some(eqpos) = after.find('=') {
                            if let Some(semi) = after[eqpos + 1..].find(';') {
                                let rhs = after[eqpos + 1..eqpos + 1 + semi].trim().to_string();
                                res.push((idx, rhs));
                            }
                        }
                    }
                    i = pos + lhs_prefix.len() + rb + 1;
                    continue;
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
                                if let Some(lb_rel) = s_work.find('{') {
                                    // find matching '}' for this inner block
                                    let mut depth3: isize = 0;
                                    let bytes3 = s_work.as_bytes();
                                    let mut jj = lb_rel;
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
                                        let cond_txt_raw =
                                            &s_work[2..s_work.find('{').unwrap_or(s_work.len())];
                                        let mut cond_txt = cond_txt_raw.trim().to_string();
                                        if cond_txt.eq_ignore_ascii_case("true") {
                                            cond_txt = "1.0".to_string();
                                        } else if cond_txt.eq_ignore_ascii_case("false") {
                                            cond_txt = "0.0".to_string();
                                        }
                                        let inner_block = &s_work[lb_rel + 1..rb2];
                                        // collect assignments inside inner_block
                                        let mut kk = 0usize;
                                        while kk < inner_block.len() {
                                            if inner_block[kk..].starts_with(lhs_prefix) {
                                                let after3 = &inner_block[kk + lhs_prefix.len()..];
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
                    if paren_depth == 0 && brace_depth == 1 {
                        stmt.push(';');
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
                                        let cond_txt_raw =
                                            &s_work[2..s_work.find('{').unwrap_or(s_work.len())];
                                        let mut cond_txt = cond_txt_raw.trim().to_string();
                                        if cond_txt.eq_ignore_ascii_case("true") {
                                            cond_txt = "1.0".to_string();
                                        } else if cond_txt.eq_ignore_ascii_case("false") {
                                            cond_txt = "0.0".to_string();
                                        }
                                        let inner_block = &s_work[lb2 + 1..rb2];
                                        let mut kk = 0usize;
                                        while kk < inner_block.len() {
                                            if inner_block[kk..].starts_with(lhs_prefix) {
                                                let after3 = &inner_block[kk + lhs_prefix.len()..];
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
                        stmt.push(';');
                        continue;
                    }
                } else {
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

    // final stmt without trailing semicolon
    let s = stmt.trim();
    if !s.is_empty() {
        // reuse scan helper to catch any trailing assignment
        scan_stmt_collect(s, lhs_prefix, &mut res);
    }

    res
}

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

/// Strip simple macro invocations we don't want to see at parse-time.
/// Currently this is a no-op placeholder so the refactor can progressively
/// adopt specific macro-stripping behaviour later.
pub fn strip_macro_calls(s: &str, name: &str) -> String {
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

/// Extract prelude assignments (simple var defs) from the closure body.
/// This is a conservative scanner that returns raw assignment strings.
pub fn extract_prelude(src: &str) -> Vec<(String, String)> {
    let mut res = Vec::new();
    // remove single-line comments
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

/// Extract `fetch` style param mappings. Stubbed: returns an empty map.
pub fn extract_fetch_params(src: &str) -> Vec<String> {
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

/// Extract covariate fetch mappings. Stubbed: returns an empty map.
pub fn extract_fetch_cov(src: &str) -> Vec<String> {
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
