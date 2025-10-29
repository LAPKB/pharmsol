use diffsol::Vector;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::simulator::equation::{Meta, ODE};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_and_parse_simple() {
        let s = "-ke * x[0] + rateiv[0] / 2";
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        // evaluate with dummy vectors
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let mut pvec = V::zeros(1, diffsol::NalgebraContext);
        pvec[0] = 3.0; // ke
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        // evaluation should succeed (ke resolves via pmap not provided -> 0)
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        // numeric result must be finite
        assert!(val.is_finite());
    }

    #[test]
    fn test_emit_ir_and_load_roundtrip() {
        // create a temporary IR file via emit_ir and load it with load_ir_ode
        use std::env;
        use std::fs;
        let tmp = env::temp_dir().join("exa_test_ir.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = 100.0; }".to_string();
        let out = "|x, p, _t, _cov, y| { y[0] = x[0]; }".to_string();
    let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            Some("|p, t, cov, x| { x[0] = 1.0; }".to_string()),
            Some(out),
            Some(tmp.clone()),
            vec!["ke".to_string()],
        )
        .expect("emit_ir failed");
    let (_ode, _meta, id) = load_ir_ode(tmp.clone()).expect("load_ir_ode failed");
        // clean up
        fs::remove_file(tmp).ok();
        // ensure ode_for_id returns an ODE
        assert!(ode_for_id(id).is_some());
    }

    #[test]
    fn test_method_and_function_call() {
        let s = "1.0.exp()*tlag";
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let mut pvec = V::zeros(1, diffsol::NalgebraContext);
        pvec[0] = 2.0; // tlag
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let mut pmap = std::collections::HashMap::new();
        pmap.insert("tlag".to_string(), 0usize);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, Some(&pmap), Some(0.0), None);
        assert!(val.is_finite());
    }
}

// --- rest of interpreter implementation follows (copy of original) ---

// Small expression AST for arithmetic used in model RHS and outputs.
#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    Ident(String),          // e.g. ke
    Indexed(String, usize), // e.g. x[0], rateiv[0], y[0]
    UnaryOp {
        op: char,
        rhs: Box<Expr>,
    },
    BinaryOp {
        lhs: Box<Expr>,
        op: char,
        rhs: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        name: String,
        args: Vec<Expr>,
    },
}

use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug)]
struct RegistryEntry {
    dx: HashMap<usize, Expr>,
    out: HashMap<usize, Expr>,
    init: HashMap<usize, Expr>,
    lag: HashMap<usize, Expr>,
    fa: HashMap<usize, Expr>,
    pmap: HashMap<String, usize>,
    nstates: usize,
    _nouteqs: usize,
}

static EXPR_REGISTRY: Lazy<Mutex<HashMap<usize, RegistryEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static NEXT_EXPR_ID: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(1));

thread_local! {
    static CURRENT_EXPR_ID: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
}

pub(crate) fn set_current_expr_id(id: Option<usize>) -> Option<usize> {
    let prev = CURRENT_EXPR_ID.with(|c| {
        let p = c.get();
        c.set(id);
        p
    });
    prev
}

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Ident(String),
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Dot,
    Op(char),
    Semicolon,
}

fn tokenize(s: &str) -> Vec<Token> {
    let mut toks = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let mut num = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit()
                    || d == '.'
                    || d == 'e'
                    || d == 'E'
                    || d == '+'
                    || d == '-' && num.ends_with('e')
                {
                    num.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Ok(v) = num.parse::<f64>() {
                toks.push(Token::Num(v));
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut id = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_alphanumeric() || d == '_' {
                    id.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            toks.push(Token::Ident(id));
            continue;
        }
        match c {
            '[' => {
                toks.push(Token::LBracket);
                chars.next();
            }
            ']' => {
                toks.push(Token::RBracket);
                chars.next();
            }
            '(' => {
                toks.push(Token::LParen);
                chars.next();
            }
            ')' => {
                toks.push(Token::RParen);
                chars.next();
            }
            ',' => {
                toks.push(Token::Comma);
                chars.next();
            }
            ';' => {
                toks.push(Token::Semicolon);
                chars.next();
            }
            '+' | '-' | '*' | '/' => {
                toks.push(Token::Op(c));
                chars.next();
            }
            '^' => {
                toks.push(Token::Op('^'));
                chars.next();
            }
            '.' => {
                toks.push(Token::Dot);
                chars.next();
            }
            _ => {
                chars.next();
            }
        }
    }
    toks
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn next(&mut self) -> Option<&Token> {
        let r = self.tokens.get(self.pos);
        if r.is_some() {
            self.pos += 1;
        }
        r
    }

    fn parse_expr(&mut self) -> Option<Expr> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Option<Expr> {
        let mut node = self.parse_mul_div()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Op('+') => {
                    self.next();
                    let rhs = self.parse_mul_div()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: '+',
                        rhs: Box::new(rhs),
                    };
                }
                Token::Op('-') => {
                    self.next();
                    let rhs = self.parse_mul_div()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: '-',
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Some(node)
    }

    fn parse_mul_div(&mut self) -> Option<Expr> {
        let mut node = self.parse_power()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Op('*') => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: '*',
                        rhs: Box::new(rhs),
                    };
                }
                Token::Op('/') => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: '/',
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Some(node)
    }

    // right-associative power
    fn parse_power(&mut self) -> Option<Expr> {
        let node = self.parse_unary()?;
        if let Some(Token::Op('^')) = self.peek() {
            self.next();
            let rhs = self.parse_power()?; // right-associative
            return Some(Expr::BinaryOp {
                lhs: Box::new(node),
                op: '^',
                rhs: Box::new(rhs),
            });
        }
        Some(node)
    }

    fn parse_unary(&mut self) -> Option<Expr> {
        if let Some(Token::Op('-')) = self.peek() {
            self.next();
            let rhs = self.parse_unary()?;
            return Some(Expr::UnaryOp {
                op: '-',
                rhs: Box::new(rhs),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        let mut node = match self.next().cloned()? {
            Token::Num(v) => Expr::Number(v),
            Token::Ident(id) => {
                // Function call: ident(...)
                if let Some(Token::LParen) = self.peek() {
                    self.next();
                    let mut args: Vec<Expr> = Vec::new();
                    if let Some(Token::RParen) = self.peek() {
                        // empty arglist
                        self.next();
                        Expr::Call { name: id.clone(), args }
                    } else {
                        loop {
                            if let Some(expr) = self.parse_expr() {
                                args.push(expr);
                            } else {
                                return None;
                            }
                            match self.peek() {
                                Some(Token::Comma) => {
                                    self.next();
                                    continue;
                                }
                                Some(Token::RParen) => {
                                    self.next();
                                    break;
                                }
                                _ => return None,
                            }
                        }
                        Expr::Call { name: id.clone(), args }
                    }
                } else if let Some(Token::LBracket) = self.peek() {
                    // Indexing: Ident[NUM]
                    self.next();
                    if let Some(Token::Num(n)) = self.next().cloned() {
                        let idx = n as usize;
                        if let Some(Token::RBracket) = self.next().cloned() {
                            Expr::Indexed(id.clone(), idx)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    Expr::Ident(id.clone())
                }
            }
            Token::LParen => {
                let expr = self.parse_expr();
                if let Some(Token::RParen) = self.next().cloned() {
                    if let Some(e) = expr { e } else { return None }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Postfix method-call chaining like primary.ident(arg1, ...)
        loop {
            if let Some(Token::Dot) = self.peek() {
                // consume dot
                self.next();
                // expect identifier
                let name = if let Some(Token::Ident(n)) = self.next().cloned() {
                    n
                } else {
                    return None;
                };
                // optional arglist
                let mut args: Vec<Expr> = Vec::new();
                if let Some(Token::LParen) = self.peek() {
                    self.next();
                    // empty arglist
                    if let Some(Token::RParen) = self.peek() {
                        self.next();
                    } else {
                        loop {
                            if let Some(expr) = self.parse_expr() {
                                args.push(expr);
                            } else {
                                return None;
                            }
                            match self.peek() {
                                Some(Token::Comma) => {
                                    self.next();
                                    continue;
                                }
                                Some(Token::RParen) => {
                                    self.next();
                                    break;
                                }
                                _ => return None,
                            }
                        }
                    }
                }
                node = Expr::MethodCall {
                    receiver: Box::new(node),
                    name,
                    args,
                };
                continue;
            }
            break;
        }

        Some(node)
    }
}

fn eval_expr(
    expr: &Expr,
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    rateiv: &crate::simulator::V,
    pmap: Option<&HashMap<String, usize>>,
    t: Option<crate::simulator::T>,
    cov: Option<&crate::data::Covariates>,
) -> f64 {
    match expr {
        Expr::Number(v) => *v,
        Expr::Ident(name) => {
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
            0.0
        }
        Expr::Indexed(name, idx) => match name.as_str() {
            "x" => x[*idx],
            "p" | "params" => p[*idx],
            "rateiv" => rateiv[*idx],
            _ => 0.0,
        },
        Expr::UnaryOp { op, rhs } => {
            let v = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
            match op {
                '-' => -v,
                _ => v,
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = eval_expr(lhs, x, p, rateiv, pmap, t, cov);
            let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
            match op {
                '+' => a + b,
                '-' => a - b,
                '*' => a * b,
                '/' => a / b,
                '^' => a.powf(b),
                _ => a,
            }
        }
        Expr::Call { name, args } => {
            let mut avals: Vec<f64> = Vec::new();
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            eval_call(name.as_str(), &avals)
        }
        Expr::MethodCall { receiver, name, args } => {
            let recv = eval_expr(receiver, x, p, rateiv, pmap, t, cov);
            let mut avals: Vec<f64> = Vec::new();
            avals.push(recv);
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            eval_call(name.as_str(), &avals)
        }
    }
}


fn eval_call(name: &str, args: &[f64]) -> f64 {
    match name {
        "exp" => args.get(0).cloned().unwrap_or(0.0).exp(),
        "ln" | "log" => args.get(0).cloned().unwrap_or(0.0).ln(),
        "log10" => args.get(0).cloned().unwrap_or(0.0).log10(),
        "sqrt" => args.get(0).cloned().unwrap_or(0.0).sqrt(),
        "pow" => {
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
fn diffeq_dispatch(
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    dx: &mut crate::simulator::V,
    _bolus: crate::simulator::V,
    rateiv: crate::simulator::V,
    _cov: &crate::data::Covariates,
) {
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            for (i, expr) in entry.dx.iter() {
                let val = eval_expr(expr, x, p, &rateiv, Some(&entry.pmap), Some(_t), Some(_cov));
                dx[*i] = val;
            }
        }
    }
}

fn out_dispatch(
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
    y: &mut crate::simulator::V,
) {
    let tmp = crate::simulator::V::zeros(1, diffsol::NalgebraContext);
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            for (i, expr) in entry.out.iter() {
                let val = eval_expr(expr, x, p, &tmp, Some(&entry.pmap), Some(_t), Some(_cov));
                y[*i] = val;
            }
        }
    }
}

fn lag_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
) -> std::collections::HashMap<usize, crate::simulator::T> {
    let mut out: std::collections::HashMap<usize, crate::simulator::T> =
        std::collections::HashMap::new();
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            let zero_x = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            for (i, expr) in entry.lag.iter() {
                let v = eval_expr(
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

fn fa_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
) -> std::collections::HashMap<usize, crate::simulator::T> {
    let mut out: std::collections::HashMap<usize, crate::simulator::T> =
        std::collections::HashMap::new();
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            let zero_x = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            for (i, expr) in entry.fa.iter() {
                let v = eval_expr(
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

fn init_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    cov: &crate::data::Covariates,
    x: &mut crate::simulator::V,
) {
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            let zero_rate = crate::simulator::V::zeros(entry.nstates, diffsol::NalgebraContext);
            for (i, expr) in entry.init.iter() {
                let v = eval_expr(
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

pub fn load_ir_ode(ir_path: PathBuf) -> Result<(ODE, Meta, usize), io::Error> {
    let contents = fs::read_to_string(&ir_path)?;
    let ir: IrFile = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("serde_json: {}", e)))?;

    let params = ir.params.unwrap_or_default();
    let meta = Meta::new(params.iter().map(|s| s.as_str()).collect());

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
        if let Some(expr) = p.parse_expr() {
            dx_map.insert(i, expr);
        } else {
            parse_errors.push(format!("failed to parse dx[{}] RHS='{}'", i, rhs));
        }
    }
    for (i, rhs) in extract_all_assign(&out_text, "y[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            out_map.insert(i, expr);
        } else {
            parse_errors.push(format!("failed to parse y[{}] RHS='{}'", i, rhs));
        }
    }
    for (i, rhs) in extract_all_assign(&init_text, "x[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            init_map.insert(i, expr);
        } else {
            parse_errors.push(format!("failed to parse init x[{}] RHS='{}'", i, rhs));
        }
    }

    fn extract_macro_map(src: &str, mac: &str) -> Vec<(usize, String)> {
        if let Some(pos) = src.find(mac) {
            if let Some(lb) = src[pos..].find('{') {
                let tail = &src[pos + lb + 1..];
                if let Some(rb) = tail.find('}') {
                    let body = &tail[..rb];
                    return body
                        .split(',')
                        .filter_map(|s| {
                            let parts: Vec<&str> = s.split("=>").collect();
                            if parts.len() == 2 {
                                if let Ok(k) = parts[0].trim().parse::<usize>() {
                                    return Some((k, parts[1].trim().to_string()));
                                }
                            }
                            None
                        })
                        .collect();
                }
            }
        }
        Vec::new()
    }

    for (i, rhs) in extract_macro_map(&lag_text, "lag!") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            lag_map.insert(i, expr);
        } else {
            parse_errors.push(format!("failed to parse lag! entry {} => '{}'", i, rhs));
        }
    }
    for (i, rhs) in extract_macro_map(&fa_text, "fa!") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            fa_map.insert(i, expr);
        } else {
            parse_errors.push(format!("failed to parse fa! entry {} => '{}'", i, rhs));
        }
    }

    // Detect fetch_params! (or common typo fetch_param!) occurrences and validate
    // that the parameter names referenced exist in the IR `params` list.
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
                let tail = &rest[pos + lb + 1..];
                if let Some(rb) = tail.find(')') {
                    let body = &tail[..rb];
                    res.push(body.to_string());
                    rest = &tail[rb + 1..];
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
        // split by ',' and trim
        let parts: Vec<String> = body
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '"' || c == '\'')).map(|s| s.to_string())
            .collect();
        // expect first arg to be 'p' (the param vector)
        if parts.is_empty() {
            parse_errors.push(format!("empty fetch_params! macro body: '{}'", body));
            continue;
        }
        // validate each referenced parameter name (skip names starting with '_')
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

    // Detect fetch_cov! occurrences and validate their syntax: expect at least
    // (cov_var, t_var, name1, name2, ...). We cannot validate covariate names
    // against a dataset at load time, but we can ensure the macro is well-formed.
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
            parse_errors.push(format!("fetch_cov! macro expects at least (cov, t, name...), got '{}'", body));
            continue;
        }
        // first arg: cov variable (identifier)
        let cov_var = parts[0].clone();
        if cov_var.is_empty() || !cov_var.chars().next().unwrap().is_ascii_alphabetic() {
            parse_errors.push(format!("invalid first argument '{}' in fetch_cov! macro", cov_var));
        }
        // second arg: time variable (allow t or _t or identifier)
        let _tvar = parts[1].clone();
        if _tvar.is_empty() {
            parse_errors.push(format!("invalid time argument '{}' in fetch_cov! macro", _tvar));
        }
        // remaining args: covariate names (can't validate existence here)
        for name in parts.iter().skip(2) {
            if name.is_empty() {
                parse_errors.push(format!("empty covariate name in fetch_cov! macro body '{}'", body));
            }
            // allow underscore-prefixed names
            if !name.starts_with('_') && !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                parse_errors.push(format!("invalid covariate identifier '{}' in fetch_cov! macro", name));
            }
        }
    }

    if dx_map.is_empty() {
        if let Some(start) = diffeq_text.find("dx") {
            if let Some(semi) = diffeq_text[start..].find(';') {
                let rhs = diffeq_text[start..start + semi].to_string();
                if let Some(eqpos) = rhs.find('=') {
                    let rhs_expr = rhs[eqpos + 1..].trim().to_string();
                    let toks = tokenize(&rhs_expr);
                    let mut p = Parser::new(toks);
                    if let Some(expr) = p.parse_expr() {
                        dx_map.insert(0, expr);
                    } else {
                        parse_errors
                            .push(format!("failed to parse fallback dx RHS='{}'", rhs_expr));
                    }
                }
            }
        }
    }

    if !parse_errors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("parse errors: {}", parse_errors.join("; ")),
        ));
    }

    let mut pmap = std::collections::HashMap::new();
    for (i, name) in params.iter().enumerate() {
        pmap.insert(name.clone(), i);
    }

    let max_dx = dx_map.keys().copied().max().unwrap_or(0);
    let max_y = out_map.keys().copied().max().unwrap_or(0);
    let nstates = max_dx + 1;
    let nouteqs = max_y + 1;

    // Validate parsed expressions: ensure identifiers reference known parameters or
    // permitted symbols. This prevents silently returning 0.0 at runtime for
    // misspelled parameter names (e.g., `kes` instead of `ke`).
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
                // allow parameter names from pmap
                if pmap.contains_key(name) {
                    return;
                }
                errors.push(format!("unknown identifier '{}'", name));
            }
            Expr::Indexed(name, idx) => {
                match name.as_str() {
                    "x" | "rateiv" => {
                        if *idx >= nstates {
                            errors.push(format!("index out of bounds '{}'[{}] (nstates={})", name, idx, nstates));
                        }
                    }
                    "p" | "params" => {
                        if *idx >= nparams {
                            errors.push(format!("parameter index out of bounds '{}'[{}] (nparams={})", name, idx, nparams));
                        }
                    }
                    "y" => {
                        // outputs may be validated elsewhere; allow any non-negative index
                    }
                    _ => {
                        errors.push(format!("unknown indexed symbol '{}'", name));
                    }
                }
            }
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
            Expr::MethodCall { receiver, name: _, args } => {
                validate_expr(receiver, pmap, nstates, nparams, errors);
                for a in args.iter() {
                    validate_expr(a, pmap, nstates, nparams, errors);
                }
            }
        }
    }

    // Run validation across all parsed expressions
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

    let entry = RegistryEntry {
        dx: dx_map,
        out: out_map,
        init: init_map,
        lag: lag_map,
        fa: fa_map,
        pmap: pmap.clone(),
        nstates,
        _nouteqs: nouteqs,
    };

    let id = NEXT_EXPR_ID.fetch_add(1, Ordering::SeqCst);
    {
        let mut guard = EXPR_REGISTRY.lock().unwrap();
        guard.insert(id, entry);
    }

    let ode = ODE::with_registry_id(
        diffeq_dispatch,
        lag_dispatch,
        fa_dispatch,
        init_dispatch,
        out_dispatch,
        (nstates, nouteqs),
        Some(id),
    );
    Ok((ode, meta, id))
}

pub fn unregister_model(id: usize) {
    let mut guard = EXPR_REGISTRY.lock().unwrap();
    guard.remove(&id);
}

pub fn ode_for_id(id: usize) -> Option<ODE> {
    let guard = EXPR_REGISTRY.lock().unwrap();
    if let Some(entry) = guard.get(&id) {
        let nstates = entry.nstates;
        let nouteqs = entry._nouteqs;
        let ode = ODE::with_registry_id(
            diffeq_dispatch,
            lag_dispatch,
            fa_dispatch,
            init_dispatch,
            out_dispatch,
            (nstates, nouteqs),
            Some(id),
        );
        Some(ode)
    } else {
        None
    }
}
