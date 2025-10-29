use diffsol::Vector;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Mutex; // bring zeros/len helpers into scope

use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::simulator::equation::{Meta, ODE};

#[derive(Deserialize, Debug)]
struct IrFile {
    ir_version: Option<String>,
    kind: Option<String>,
    params: Option<Vec<String>>,
    model_text: Option<String>,
}

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
}

// A tiny global registry to hold the parsed expressions for the current
// interpreter-backed ODE. We use a Mutex<Option<...>> and non-capturing
// dispatcher functions (below) so we can pass plain fn pointers to
// ODE::new (which expects function pointer types, not closures).
use std::sync::atomic::{AtomicUsize, Ordering};

// Registry mapping id -> (dx_expr, y_expr, param_name->index)
// Registry entry holds parsed expressions for all supported pieces of a model.
#[derive(Clone, Debug)]
struct RegistryEntry {
    // dx expressions keyed by state index
    dx: HashMap<usize, Expr>,
    // output expressions keyed by output index
    out: HashMap<usize, Expr>,
    // init expressions keyed by state index
    init: HashMap<usize, Expr>,
    // lag/fa maps keyed by index
    lag: HashMap<usize, Expr>,
    fa: HashMap<usize, Expr>,
    // parameter name -> index
    pmap: HashMap<String, usize>,
    // sizes
    nstates: usize,
    _nouteqs: usize,
}

static EXPR_REGISTRY: Lazy<Mutex<HashMap<usize, RegistryEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Global id source for entries in EXPR_REGISTRY
static NEXT_EXPR_ID: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(1));

// Thread-local current registry id used by dispatchers to pick the right entry.
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

// Simple tokenizer for expressions
#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Ident(String),
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
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
            _ => {
                chars.next();
            }
        }
    }
    toks
}

// Recursive descent parser for expressions with operator precedence
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
        let mut node = self.parse_unary()?;
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
        let tok = self.next().cloned()?;
        match tok {
            Token::Num(v) => Some(Expr::Number(v)),
            Token::Ident(id) => {
                // if next is [ then parse index
                if let Some(Token::LBracket) = self.peek() {
                    self.next(); // consume [
                    if let Some(Token::Num(n)) = self.next().cloned() {
                        let idx = n as usize;
                        if let Some(Token::RBracket) = self.next().cloned() {
                            return Some(Expr::Indexed(id.clone(), idx));
                        }
                    }
                    return None;
                }
                Some(Expr::Ident(id.clone()))
            }
            Token::LParen => {
                let expr = self.parse_expr();
                if let Some(Token::RParen) = self.next().cloned() {
                    expr
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// Evaluate expression given runtime variables
fn eval_expr(
    expr: &Expr,
    x: &crate::simulator::V,
    p: &crate::simulator::V,
    rateiv: &crate::simulator::V,
    pmap: Option<&HashMap<String, usize>>,
) -> f64 {
    match expr {
        Expr::Number(v) => *v,
        Expr::Ident(name) => {
            // Try resolve identifier to a parameter index via pmap, if present.
            if let Some(map) = pmap {
                if let Some(idx) = map.get(name) {
                    return p[*idx];
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
            let v = eval_expr(rhs, x, p, rateiv, pmap);
            match op {
                '-' => -v,
                _ => v,
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = eval_expr(lhs, x, p, rateiv, pmap);
            let b = eval_expr(rhs, x, p, rateiv, pmap);
            match op {
                '+' => a + b,
                '-' => a - b,
                '*' => a * b,
                '/' => a / b,
                _ => a,
            }
        }
    }
}

// Non-capturing dispatcher functions that read the global registry and
// evaluate the stored ASTs. These are plain `fn` items so they can be
// passed to `ODE::new` (which expects function pointer types).
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
    // pick registry entry based on current thread-local id
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            // evaluate each dx expression present in the entry
            for (i, expr) in entry.dx.iter() {
                let val = eval_expr(expr, x, p, &rateiv, Some(&entry.pmap));
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
    // create a temporary zero-rate vector for expressions that reference rateiv
    let tmp = crate::simulator::V::zeros(1, diffsol::NalgebraContext);
    let guard = EXPR_REGISTRY.lock().unwrap();
    let cur = CURRENT_EXPR_ID.with(|c| c.get());
    if let Some(id) = cur {
        if let Some(entry) = guard.get(&id) {
            for (i, expr) in entry.out.iter() {
                let val = eval_expr(expr, x, p, &tmp, Some(&entry.pmap));
                y[*i] = val;
            }
        }
    }
}

// Lag dispatcher: returns a HashMap of lag times for compartments
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
                let v = eval_expr(expr, &zero_x, p, &zero_rate, Some(&entry.pmap));
                out.insert(*i, v);
            }
        }
    }
    out
}

// Fa dispatcher: returns a HashMap of fraction absorbed
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
                let v = eval_expr(expr, &zero_x, p, &zero_rate, Some(&entry.pmap));
                out.insert(*i, v);
            }
        }
    }
    out
}

// Init dispatcher: sets initial state values
fn init_dispatch(
    p: &crate::simulator::V,
    _t: crate::simulator::T,
    _cov: &crate::data::Covariates,
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
                );
                x[*i] = v;
            }
        }
    }
}

/// Loads a prototype IR-based ODE and returns an `ODE` and `Meta`.
///
/// This interpreter will attempt to extract a single `dx[0] = <expr>;` assignment
/// and a single `y[0] = <expr>;` assignment from the `model_text` field and
/// compile them into small expression ASTs. It uses parameter ordering from
/// the IR `params` array: callers must ensure `emit_ir` provided the correct
/// parameter ordering.
pub fn load_ir_ode(ir_path: PathBuf) -> Result<(ODE, Meta), io::Error> {
    let contents = fs::read_to_string(&ir_path)?;
    let ir: IrFile = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("serde_json: {}", e)))?;

    let params = ir.params.unwrap_or_default();
    let meta = Meta::new(params.iter().map(|s| s.as_str()).collect());

    // Prepare parameter name -> index map
    let mut pmap = std::collections::HashMap::new();
    for (i, name) in params.iter().enumerate() {
        pmap.insert(name.clone(), i);
    }

    // Extract expressions from model_text
    let model_text = ir.model_text.unwrap_or_default();

    // (removed: unused single-assignment helper)

    // Parse all dx[i] and y[i] assignments, init x[i] assignments, and lag/fa macros.
    let mut dx_map: HashMap<usize, Expr> = HashMap::new();
    let mut out_map: HashMap<usize, Expr> = HashMap::new();
    let mut init_map: HashMap<usize, Expr> = HashMap::new();
    let mut lag_map: HashMap<usize, Expr> = HashMap::new();
    let mut fa_map: HashMap<usize, Expr> = HashMap::new();

    // helper: find all occurrences of a pattern like "dx[<n>]" and capture the RHS until ';'
    fn extract_all_assign(src: &str, lhs_prefix: &str) -> Vec<(usize, String)> {
        let mut res = Vec::new();
        let mut rest = src;
        while let Some(pos) = rest.find(lhs_prefix) {
            let after = &rest[pos + lhs_prefix.len()..];
            // read digits until ']'
            if let Some(rb) = after.find(']') {
                let idx_str = &after[..rb];
                if let Ok(idx) = idx_str.trim().parse::<usize>() {
                    // find '=' somewhere after the bracket
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
            // if we didn't parse, advance to avoid infinite loop
            rest = &rest[pos + lhs_prefix.len()..];
        }
        res
    }

    for (i, rhs) in extract_all_assign(&model_text, "dx[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            dx_map.insert(i, expr);
        }
    }
    for (i, rhs) in extract_all_assign(&model_text, "y[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            out_map.insert(i, expr);
        }
    }
    for (i, rhs) in extract_all_assign(&model_text, "x[") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            init_map.insert(i, expr);
        }
    }

    // Parse lag!{...} and fa!{...} simple maps like 0=>tlag,1=>0.3
    fn extract_macro_map(src: &str, mac: &str) -> Vec<(usize, String)> {
        if let Some(pos) = src.find(mac) {
            if let Some(lb) = src[pos..].find('{') {
                let tail = &src[pos + lb + 1..];
                if let Some(rb) = tail.find('}') {
                    let body = &tail[..rb];
                    // split by ',' and parse 'k => expr'
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

    for (i, rhs) in extract_macro_map(&model_text, "lag!") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            lag_map.insert(i, expr);
        }
    }
    for (i, rhs) in extract_macro_map(&model_text, "fa!") {
        let toks = tokenize(&rhs);
        let mut p = Parser::new(toks);
        if let Some(expr) = p.parse_expr() {
            fa_map.insert(i, expr);
        }
    }

    // Heuristics: if no dx statements found, try to extract single expression inside closure-like text
    if dx_map.is_empty() {
        if let Some(start) = model_text.find("dx") {
            if let Some(semi) = model_text[start..].find(';') {
                let rhs = model_text[start..start + semi].to_string();
                if let Some(eqpos) = rhs.find('=') {
                    let rhs_expr = rhs[eqpos + 1..].trim().to_string();
                    let toks = tokenize(&rhs_expr);
                    let mut p = Parser::new(toks);
                    if let Some(expr) = p.parse_expr() {
                        dx_map.insert(0, expr);
                    }
                }
            }
        }
    }

    // Now build closures. We'll create closures that map parameter names to indices by
    // creating a parameter vector `pvec` where param names are placed at their index.
    use crate::data::Covariates;
    use crate::simulator::{T, V};

    // Build parameter name -> index map
    let mut pmap = std::collections::HashMap::new();
    for (i, name) in params.iter().enumerate() {
        pmap.insert(name.clone(), i);
    }

    // determine sizes from parsed maps
    let max_dx = dx_map.keys().copied().max().unwrap_or(0);
    let max_y = out_map.keys().copied().max().unwrap_or(0);
    let nstates = max_dx + 1;
    let nouteqs = max_y + 1;

    // Construct registry entry and insert
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

    // allocate id and insert into the registry
    let id = NEXT_EXPR_ID.fetch_add(1, Ordering::SeqCst);
    {
        let mut guard = EXPR_REGISTRY.lock().unwrap();
        guard.insert(id, entry);
    }

    // local placeholder closures removed; we use the dispatcher functions

    // Use the dispatcher functions (plain fn pointers) so they can be used
    // with the existing ODE::new signature that expects fn types.
    // Build ODE with proper sizes and dispatchers
    let ode = ODE::with_registry_id(
        diffeq_dispatch,
        lag_dispatch,
        fa_dispatch,
        init_dispatch,
        out_dispatch,
        (nstates, nouteqs),
        Some(id),
    );
    Ok((ode, meta))
}
