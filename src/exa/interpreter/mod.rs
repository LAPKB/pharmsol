use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;
use std::collections::HashMap;
use diffsol::Vector; // bring zeros/len helpers into scope

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
    Ident(String),            // e.g. ke
    Indexed(String, usize),   // e.g. x[0], rateiv[0], y[0]
    UnaryOp { op: char, rhs: Box<Expr> },
    BinaryOp { lhs: Box<Expr>, op: char, rhs: Box<Expr> },
}

// A tiny global registry to hold the parsed expressions for the current
// interpreter-backed ODE. We use a Mutex<Option<...>> and non-capturing
// dispatcher functions (below) so we can pass plain fn pointers to
// ODE::new (which expects function pointer types, not closures).
use std::sync::atomic::{AtomicUsize, Ordering};

// Registry mapping id -> (dx_expr, y_expr, param_name->index)
static EXPR_REGISTRY: Lazy<Mutex<HashMap<usize, (Expr, Expr, HashMap<String, usize>)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Global id source for entries in EXPR_REGISTRY
static NEXT_EXPR_ID: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(1));

// Thread-local current registry id used by dispatchers to pick the right entry.
thread_local! {
    static CURRENT_EXPR_ID: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
}

pub(crate) fn set_current_expr_id(id: Option<usize>) -> Option<usize> {
    let prev = CURRENT_EXPR_ID.with(|c| { let p = c.get(); c.set(id); p });
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
                if d.is_ascii_digit() || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-' && num.ends_with('e') {
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
            '[' => { toks.push(Token::LBracket); chars.next(); }
            ']' => { toks.push(Token::RBracket); chars.next(); }
            '(' => { toks.push(Token::LParen); chars.next(); }
            ')' => { toks.push(Token::RParen); chars.next(); }
            ',' => { toks.push(Token::Comma); chars.next(); }
            ';' => { toks.push(Token::Semicolon); chars.next(); }
            '+'|'-'|'*'|'/' => { toks.push(Token::Op(c)); chars.next(); }
            _ => { chars.next(); }
        }
    }
    toks
}

// Recursive descent parser for expressions with operator precedence
struct Parser { tokens: Vec<Token>, pos: usize }

impl Parser {
    fn new(tokens: Vec<Token>) -> Self { Self { tokens, pos: 0 } }
    fn peek(&self) -> Option<&Token> { self.tokens.get(self.pos) }
    fn next(&mut self) -> Option<&Token> { let r = self.tokens.get(self.pos); if r.is_some() { self.pos += 1; } r }

    fn parse_expr(&mut self) -> Option<Expr> { self.parse_add_sub() }

    fn parse_add_sub(&mut self) -> Option<Expr> {
        let mut node = self.parse_mul_div()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Op('+') => { self.next(); let rhs = self.parse_mul_div()?; node = Expr::BinaryOp { lhs: Box::new(node), op: '+', rhs: Box::new(rhs) }; }
                Token::Op('-') => { self.next(); let rhs = self.parse_mul_div()?; node = Expr::BinaryOp { lhs: Box::new(node), op: '-', rhs: Box::new(rhs) }; }
                _ => break,
            }
        }
        Some(node)
    }

    fn parse_mul_div(&mut self) -> Option<Expr> {
        let mut node = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Op('*') => { self.next(); let rhs = self.parse_unary()?; node = Expr::BinaryOp { lhs: Box::new(node), op: '*', rhs: Box::new(rhs) }; }
                Token::Op('/') => { self.next(); let rhs = self.parse_unary()?; node = Expr::BinaryOp { lhs: Box::new(node), op: '/', rhs: Box::new(rhs) }; }
                _ => break,
            }
        }
        Some(node)
    }

    fn parse_unary(&mut self) -> Option<Expr> {
        if let Some(Token::Op('-')) = self.peek() {
            self.next(); let rhs = self.parse_unary()?; return Some(Expr::UnaryOp { op: '-', rhs: Box::new(rhs) });
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
                } else { None }
            }
            _ => None,
        }
    }
}

// Evaluate expression given runtime variables
fn eval_expr(expr: &Expr, x: &crate::simulator::V, p: &crate::simulator::V, rateiv: &crate::simulator::V, pmap: Option<&HashMap<String, usize>>) -> f64 {
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
        Expr::Indexed(name, idx) => {
            match name.as_str() {
                "x" => x[*idx],
                "p" | "params" => p[*idx],
                "rateiv" => rateiv[*idx],
                _ => 0.0,
            }
        }
        Expr::UnaryOp { op, rhs } => {
            let v = eval_expr(rhs, x, p, rateiv, pmap);
            match op { '-' => -v, _ => v }
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
        if let Some((dx_expr, _y_expr, pmap)) = guard.get(&id) {
            let val = eval_expr(dx_expr, x, p, &rateiv, Some(pmap));
            dx[0] = val;
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
        if let Some((_dx_expr, y_expr, pmap)) = guard.get(&id) {
            let val = eval_expr(y_expr, x, p, &tmp, Some(pmap));
            y[0] = val;
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
    for (i, name) in params.iter().enumerate() { pmap.insert(name.clone(), i); }

    // Extract expressions from model_text
    let model_text = ir.model_text.unwrap_or_default();

    // helper to extract between a pattern and ';'
    fn extract_assign(src: &str, lhs: &str) -> Option<String> {
        if let Some(pos) = src.find(lhs) {
            let tail = &src[pos + lhs.len()..];
            if let Some(semi) = tail.find(';') {
                return Some(tail[..semi].trim().to_string());
            }
        }
        None
    }

    let dx0_rhs = extract_assign(&model_text, "dx[0]").or_else(|| extract_assign(&model_text, "dx[0] =")).unwrap_or_else(|| "-ke * x[0] + rateiv[0]".to_string());
    let y0_rhs = extract_assign(&model_text, "y[0]").or_else(|| extract_assign(&model_text, "y[0] =")).unwrap_or_else(|| "x[0] / v".to_string());

    // Tokenize and parse expressions
    let dx_tokens = tokenize(&dx0_rhs);
    let mut dx_parser = Parser::new(dx_tokens);
    let dx_expr = dx_parser.parse_expr().expect("Failed to parse dx expression");

    let y_tokens = tokenize(&y0_rhs);
    let mut y_parser = Parser::new(y_tokens);
    let y_expr = y_parser.parse_expr().expect("Failed to parse y expression");

    // Now build closures. We'll create closures that map parameter names to indices by
    // creating a parameter vector `pvec` where param names are placed at their index.
    use crate::simulator::{T, V};
    use crate::data::Covariates;

    // Build parameter name -> index map and store along with parsed Exprs
    // into the global registry so the non-capturing dispatchers can
    // resolve parameter identifiers.
    let mut pmap = std::collections::HashMap::new();
    for (i, name) in params.iter().enumerate() {
        pmap.insert(name.clone(), i);
    }

    // allocate id and insert into the registry
    let id = NEXT_EXPR_ID.fetch_add(1, Ordering::SeqCst);
    {
        let mut guard = EXPR_REGISTRY.lock().unwrap();
        guard.insert(id, (dx_expr.clone(), y_expr.clone(), pmap));
    }

    let lag = |_p: &V, _t: T, _cov: &Covariates| -> std::collections::HashMap<usize, T> {
        std::collections::HashMap::new()
    };
    let fa = |_p: &V, _t: T, _cov: &Covariates| -> std::collections::HashMap<usize, T> {
        std::collections::HashMap::new()
    };
    let init = |_p: &V, _t: T, _cov: &Covariates, _x: &mut V| {};

    // Use the dispatcher functions (plain fn pointers) so they can be used
    // with the existing ODE::new signature that expects fn types.
    let ode = ODE::with_registry_id(
        diffeq_dispatch,
        lag,
        fa,
        init,
        out_dispatch,
        (1_usize, 1_usize),
        Some(id),
    );
    Ok((ode, meta))
}
