use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

use serde::Deserialize;

use crate::exa_wasm::interpreter::ast::Expr;
use crate::exa_wasm::interpreter::parser::{tokenize, Parser};
use crate::exa_wasm::interpreter::registry;

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

    let mut parse_errors: Vec<String> = Vec::new();

    fn extract_all_assign(src: &str, lhs_prefix: &str) -> Vec<(usize, String)> {
        let mut res = Vec::new();
        let mut rest = src;
        while let Some(pos) = rest.find(lhs_prefix) {
            let after = &rest[pos + lhs_prefix.len()..];
            if let Some(rb) = after.find(']') {
                let idx_str = &after[..rb];
                if let Ok(idx) = idx_str.trim().parse::<usize>() {
                    if let Some(eqpos) = after.find('=') {
                        let tail = &after[eqpos + 1..];
                        if let Some(semi) = tail.find(';') {
                            let rhs = tail[..semi].trim().to_string();
                            res.push((idx, rhs));
                            rest = &tail[semi + 1..];
                            continue;
                        }
                    }
                }
            }
            rest = &rest[pos + lhs_prefix.len()..];
        }
        res
    }

    for (i, rhs) in extract_all_assign(&diffeq_text, "dx[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        let res = p.parse_expr_result();
        match res {
            Ok(expr) => {
                dx_map.insert(i, expr);
            }
            Err(e) => {
                parse_errors.push(format!("failed to parse dx[{}] RHS='{}' : {}", i, rhs, e));
            }
        }
    }
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

    if dx_map.is_empty() {
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
    let max_dx = dx_map.keys().copied().max().unwrap_or(0);
    let max_y = out_map.keys().copied().max().unwrap_or(0);
    let nstates = max_dx + 1;
    let nouteqs = max_y + 1;

    let nparams = params.len();
    for (_i, expr) in dx_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in out_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in init_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in lag_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }
    for (_i, expr) in fa_map.iter() {
        validate_expr(expr, &pmap, nstates, nparams, &mut parse_errors);
    }

    if !parse_errors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("parse errors: {}", parse_errors.join("; ")),
        ));
    }

    let entry = registry::RegistryEntry {
        dx: dx_map,
        out: out_map,
        init: init_map,
        lag: lag_map,
        fa: fa_map,
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
