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
    lag_map: Option<std::collections::HashMap<usize, String>>,
    fa_map: Option<std::collections::HashMap<usize, String>>,
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

    #[test]
    fn test_arithmetic_and_power() {
        let s = "-1 + 2*3 - 4/2 + 2^3"; // -1 + 6 -2 + 8 = 11
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let pvec = V::zeros(1, diffsol::NalgebraContext);
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        assert!((val - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_comparisons_and_logical() {
        let s = "(1 < 2) && (3 >= 2) || (0 == 1)"; // true && true || false => true
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let pvec = V::zeros(1, diffsol::NalgebraContext);
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_if_builtin() {
        let s = "if(1, 2.5, 7.5)"; // should return 2.5
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let pvec = V::zeros(1, diffsol::NalgebraContext);
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        assert!((val - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_dynamic_indexing() {
        let s = "x[(1+1)] * p[0]"; // x[2]*p[0]
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let mut x = V::zeros(4, diffsol::NalgebraContext);
        x[2] = 3.0;
        let mut pvec = V::zeros(1, diffsol::NalgebraContext);
        pvec[0] = 2.0;
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        assert!((val - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_function_whitelist_and_methods() {
        let s = "max(2.0, 3.0) + pow(2.0, 3.0)"; // 3 + 8 = 11
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let pvec = V::zeros(1, diffsol::NalgebraContext);
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        assert!((val - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_macro_parsing_load_ir() {
        use std::env;
        use std::fs;
        let tmp = env::temp_dir().join("exa_test_ir_lag.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = 0.0; }".to_string();
        // lag text contains function calls and commas inside calls
        let lag = Some(
            "|p, t, _cov| { lag!{0 => max(1.0, t * 2.0), 1 => if(t>0, 2.0, 3.0)} }".to_string(),
        );
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            lag,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let res = load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(res.is_ok());
    }
}

#[cfg(test)]
mod load_negative_tests {
    use super::*;
    use std::env;
    use std::fs;

    // Ensure loader returns an error when textual lag/fa are present but
    // structured lag_map/fa_map fields are missing. This verifies we no
    // longer accept fragile runtime macro parsing.
    #[test]
    fn test_loader_errors_when_missing_structured_maps() {
        let tmp = env::temp_dir().join("exa_test_ir_negative.json");
        // Build a minimal IR JSON where lag/fa textual fields are present
        // but lag_map/fa_map are omitted.
        let ir_json = serde_json::json!({
            "ir_version": "1.0",
            "kind": "EqnKind::ODE",
            "params": ["ke", "v"],
            "diffeq": "|x, p, _t, dx, rateiv, _cov| { dx[0] = -ke * x[0] + rateiv[0]; }",
            "lag": "|p, t, _cov| { lag!{0 => t} }",
            "fa": "|p, t, _cov| { fa!{0 => 0.1} }",
            "init": "|p, _t, _cov, x| { }",
            "out": "|x, p, _t, _cov, y| { y[0] = x[0]; }"
        });
        let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
        fs::write(&tmp, s.as_bytes()).expect("write tmp");

        let res = load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(res.is_err(), "loader should reject IR missing structured maps");
    }
}

// --- rest of interpreter implementation follows (copy of original) ---

// Small expression AST for arithmetic used in model RHS and outputs.
#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    Ident(String),              // e.g. ke
    Indexed(String, Box<Expr>), // e.g. x[0], rateiv[0], y[0] where index can be expr
    UnaryOp {
        op: String,
        rhs: Box<Expr>,
    },
    BinaryOp {
        lhs: Box<Expr>,
        op: String,
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
    Ternary {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
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
    static LAST_RUNTIME_ERROR: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

pub(crate) fn set_current_expr_id(id: Option<usize>) -> Option<usize> {
    let prev = CURRENT_EXPR_ID.with(|c| {
        let p = c.get();
        c.set(id);
        p
    });
    prev
}

// Runtime error helpers: interpreter code can set an error message when a
// runtime problem (invalid index, unknown function, etc.) occurs. The
// simulator will poll for this error and convert it into a `PharmsolError`.
pub fn set_runtime_error(msg: String) {
    LAST_RUNTIME_ERROR.with(|c| {
        *c.borrow_mut() = Some(msg);
    });
}

pub fn take_runtime_error() -> Option<String> {
    LAST_RUNTIME_ERROR.with(|c| c.borrow_mut().take())
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
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    Ne,
    And,
    Or,
    Bang,
    Question,
    Colon,
    Semicolon,
}

#[derive(Debug, Clone)]
struct ParseError {
    pos: usize,
    found: Option<Token>,
    expected: Vec<String>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.expected.is_empty() {
            write!(f, "parse error at pos {} found={:?} expected={:?}", self.pos, self.found, self.expected)
        } else if let Some(tok) = &self.found {
            write!(f, "parse error at pos {} found={:?}", self.pos, tok)
        } else {
            write!(f, "parse error at pos {} found=<end>", self.pos)
        }
    }
}

impl std::error::Error for ParseError {}

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
                // allow digits, dot, exponent markers, and a sign only when
                // it follows an exponent marker (e or E)
                if d.is_ascii_digit()
                    || d == '.'
                    || d == 'e'
                    || d == 'E'
                    || ((d == '+' || d == '-') && (num.ends_with('e') || num.ends_with('E')))
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
            '?' => {
                toks.push(Token::Question);
                chars.next();
            }
            ':' => {
                toks.push(Token::Colon);
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
            '<' => {
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    toks.push(Token::Le);
                } else {
                    toks.push(Token::Lt);
                }
            }
            '>' => {
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    toks.push(Token::Ge);
                } else {
                    toks.push(Token::Gt);
                }
            }
            '=' => {
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    toks.push(Token::EqEq);
                } else {
                    // single '=' not used, skip
                }
            }
            '!' => {
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    toks.push(Token::Ne);
                } else {
                    toks.push(Token::Bang);
                }
            }
            '&' => {
                chars.next();
                if let Some(&'&') = chars.peek() {
                    chars.next();
                    toks.push(Token::And);
                }
            }
            '|' => {
                chars.next();
                if let Some(&'|') = chars.peek() {
                    chars.next();
                    toks.push(Token::Or);
                }
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
    expected: Vec<String>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            expected: Vec::new(),
        }
    }
    fn expected_push(&mut self, s: &str) {
        if !self.expected.contains(&s.to_string()) {
            self.expected.push(s.to_string());
        }
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
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Option<Expr> {
        // parse conditional ternary: cond ? then : else
        let cond = self.parse_or()?;
        if let Some(Token::Question) = self.peek().cloned() {
            self.next();
            let then_branch = self.parse_expr()?;
            if let Some(Token::Colon) = self.peek().cloned() {
                self.next();
                let else_branch = self.parse_expr()?;
                return Some(Expr::Ternary {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                });
            } else {
                self.expected_push(":");
                return None;
            }
        }
        Some(cond)
    }

    /// Parse and return a Result with an informative error message on failure.
    fn parse_expr_result(&mut self) -> Result<Expr, ParseError> {
        if let Some(expr) = self.parse_expr() {
            Ok(expr)
        } else {
            Err(ParseError {
                pos: self.pos,
                found: self.peek().cloned(),
                expected: self.expected.clone(),
            })
        }
    }

    fn parse_or(&mut self) -> Option<Expr> {
        let mut node = self.parse_and()?;
        while let Some(Token::Or) = self.peek().cloned() {
            self.next();
            let rhs = self.parse_and()?;
            node = Expr::BinaryOp {
                lhs: Box::new(node),
                op: "||".to_string(),
                rhs: Box::new(rhs),
            };
        }
        Some(node)
    }

    fn parse_and(&mut self) -> Option<Expr> {
        let mut node = self.parse_eq()?;
        while let Some(Token::And) = self.peek().cloned() {
            self.next();
            let rhs = self.parse_eq()?;
            node = Expr::BinaryOp {
                lhs: Box::new(node),
                op: "&&".to_string(),
                rhs: Box::new(rhs),
            };
        }
        Some(node)
    }

    fn parse_eq(&mut self) -> Option<Expr> {
        let mut node = self.parse_cmp()?;
        loop {
            match self.peek() {
                Some(Token::EqEq) => {
                    self.next();
                    let rhs = self.parse_cmp()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "==".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Some(Token::Ne) => {
                    self.next();
                    let rhs = self.parse_cmp()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "!=".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Some(node)
    }

    fn parse_cmp(&mut self) -> Option<Expr> {
        let mut node = self.parse_add_sub()?;
        loop {
            match self.peek() {
                Some(Token::Lt) => {
                    self.next();
                    let rhs = self.parse_add_sub()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "<".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Some(Token::Gt) => {
                    self.next();
                    let rhs = self.parse_add_sub()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: ">".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Some(Token::Le) => {
                    self.next();
                    let rhs = self.parse_add_sub()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "<=".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Some(Token::Ge) => {
                    self.next();
                    let rhs = self.parse_add_sub()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: ">=".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Some(node)
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
                        op: "+".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Token::Op('-') => {
                    self.next();
                    let rhs = self.parse_mul_div()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "-".to_string(),
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
                        op: "*".to_string(),
                        rhs: Box::new(rhs),
                    };
                }
                Token::Op('/') => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    node = Expr::BinaryOp {
                        lhs: Box::new(node),
                        op: "/".to_string(),
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
                op: "^".to_string(),
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
                op: '-'.to_string(),
                rhs: Box::new(rhs),
            });
        }
        if let Some(Token::Bang) = self.peek() {
            self.next();
            let rhs = self.parse_unary()?;
            // represent logical not as Call if needed, but use unary op '!'
            return Some(Expr::UnaryOp {
                op: '!'.to_string(),
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
                        Expr::Call {
                            name: id.clone(),
                            args,
                        }
                    } else {
                        loop {
                            if let Some(expr) = self.parse_expr() {
                                args.push(expr);
                            } else {
                                        self.expected_push("expression");
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
                                        _ => {
                                            self.expected_push(",|)");
                                            return None;
                                        }
                            }
                        }
                        Expr::Call {
                            name: id.clone(),
                            args,
                        }
                    }
                } else if let Some(Token::LBracket) = self.peek() {
                    // Indexing: Ident[expr]
                    // To avoid the inner parse consuming the closing ']' we locate
                    // the matching RBracket in the token stream, parse only the
                    // tokens inside with a fresh Parser, and advance the main
                    // parser past the closing bracket. This supports nested
                    // parentheses and nested brackets inside the index.
                    self.next(); // consume '['
                    #[cfg(test)]
                    {
                        eprintln!(
                            "parsing index: pos={} remaining={:?}",
                            self.pos,
                            &self.tokens[self.pos..]
                        );
                    }
                    let mut depth = 1isize;
                    let mut i = self.pos;
                    while i < self.tokens.len() {
                        match &self.tokens[i] {
                            Token::LBracket => depth += 1,
                            Token::RBracket => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    if i >= self.tokens.len() {
                        self.expected_push("]");
                        return None; // no matching ']'
                    }
                    // parse tokens in range [self.pos, i) as a sub-expression
                    let slice = self.tokens[self.pos..i].to_vec();
                    let mut sub = Parser::new(slice);
                    let idx_expr = sub.parse_expr()?;
                    // advance main parser past the matched RBracket
                    self.pos = i + 1;
                    Expr::Indexed(id.clone(), Box::new(idx_expr))
                } else {
                    Expr::Ident(id.clone())
                }
            }
            Token::LParen => {
                let expr = self.parse_expr();
                if let Some(Token::RParen) = self.next().cloned() {
                    if let Some(e) = expr {
                        e
                    } else {
                        self.expected_push("expression");
                        return None;
                    }
                } else {
                    self.expected_push(")");
                    return None;
                }
            }
            _ => {
                self.expected_push("number|identifier|'('");
                return None;
            }
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
                    self.expected_push("identifier");
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
                                self.expected_push("expression");
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
                                _ => {
                                    self.expected_push(",|)");
                                    return None;
                                }
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
            // allow underscore-prefixed idents as intentional ignored placeholders
            if name.starts_with('_') {
                return 0.0;
            }
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
            // Unknown identifier: set a runtime error so the simulator can fail fast
            set_runtime_error(format!("unknown identifier '{}'", name));
            0.0
        }
        Expr::Indexed(name, idx_expr) => {
            let idxf = eval_expr(idx_expr, x, p, rateiv, pmap, t, cov);
            if !idxf.is_finite() || idxf.is_sign_negative() {
                set_runtime_error(format!(
                    "invalid index expression for '{}' -> {}",
                    name, idxf
                ));
                return 0.0;
            }
            let idx = idxf as usize;
            match name.as_str() {
                "x" => {
                    if idx < x.len() {
                        x[idx]
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'x'[{}] (nstates={})",
                            idx,
                            x.len()
                        ));
                        0.0
                    }
                }
                "p" | "params" => {
                    if idx < p.len() {
                        p[idx]
                    } else {
                        set_runtime_error(format!(
                            "parameter index out of bounds '{}'[{}] (nparams={})",
                            name,
                            idx,
                            p.len()
                        ));
                        0.0
                    }
                }
                "rateiv" => {
                    if idx < rateiv.len() {
                        rateiv[idx]
                    } else {
                        set_runtime_error(format!(
                            "index out of bounds 'rateiv'[{}] (len={})",
                            idx,
                            rateiv.len()
                        ));
                        0.0
                    }
                }
                _ => {
                    set_runtime_error(format!("unknown indexed symbol '{}'", name));
                    0.0
                }
            }
        }
        Expr::UnaryOp { op, rhs } => {
            let v = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
            match op.as_str() {
                "-" => -v,
                "!" => {
                    if v == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => v,
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = eval_expr(lhs, x, p, rateiv, pmap, t, cov);
            // short-circuit for logical && and ||
            match op.as_str() {
                "&&" => {
                    if a == 0.0 {
                        return 0.0;
                    }
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    if b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "||" => {
                    if a != 0.0 {
                        return 1.0;
                    }
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    if b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => {
                    let b = eval_expr(rhs, x, p, rateiv, pmap, t, cov);
                    match op.as_str() {
                        "+" => a + b,
                        "-" => a - b,
                        "*" => a * b,
                        "/" => a / b,
                        "^" => a.powf(b),
                        "<" => {
                            if a < b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ">" => {
                            if a > b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "<=" => {
                            if a <= b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ">=" => {
                            if a >= b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "==" => {
                            if a == b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        "!=" => {
                            if a != b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => a,
                    }
                }
            }
        }
        Expr::Call { name, args } => {
            let mut avals: Vec<f64> = Vec::new();
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            if res == 0.0 {
                // eval_call returns 0.0 for unknown functions — set runtime error
                // so the simulator can pick it up and convert to Err.
                if !matches!(
                    name.as_str(),
                    "min"
                        | "max"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "ln"
                        | "log"
                        | "log10"
                        | "log2"
                        | "pow"
                        | "powf"
                ) {
                    set_runtime_error(format!("unknown function '{}()', returned 0.0", name));
                }
            }
            res
        }
        Expr::Ternary {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = eval_expr(cond, x, p, rateiv, pmap, t, cov);
            if c != 0.0 {
                eval_expr(then_branch, x, p, rateiv, pmap, t, cov)
            } else {
                eval_expr(else_branch, x, p, rateiv, pmap, t, cov)
            }
        }
        Expr::MethodCall {
            receiver,
            name,
            args,
        } => {
            let recv = eval_expr(receiver, x, p, rateiv, pmap, t, cov);
            let mut avals: Vec<f64> = Vec::new();
            avals.push(recv);
            for aexpr in args.iter() {
                avals.push(eval_expr(aexpr, x, p, rateiv, pmap, t, cov));
            }
            let res = eval_call(name.as_str(), &avals);
            if res == 0.0 {
                if !matches!(
                    name.as_str(),
                    "min"
                        | "max"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "ln"
                        | "log"
                        | "log10"
                        | "log2"
                        | "pow"
                        | "powf"
                ) {
                    set_runtime_error(format!("unknown method '{}', returned 0.0", name));
                }
            }
            res
        }
    }
}

fn eval_call(name: &str, args: &[f64]) -> f64 {
    match name {
        "exp" => args.get(0).cloned().unwrap_or(0.0).exp(),
        "if" => {
            let cond = args.get(0).cloned().unwrap_or(0.0);
            if cond != 0.0 {
                args.get(1).cloned().unwrap_or(0.0)
            } else {
                args.get(2).cloned().unwrap_or(0.0)
            }
        }
        "ln" | "log" => args.get(0).cloned().unwrap_or(0.0).ln(),
        "log10" => args.get(0).cloned().unwrap_or(0.0).log10(),
        "log2" => args.get(0).cloned().unwrap_or(0.0).log2(),
        "sqrt" => args.get(0).cloned().unwrap_or(0.0).sqrt(),
        "pow" => {
            let a = args.get(0).cloned().unwrap_or(0.0);
            let b = args.get(1).cloned().unwrap_or(0.0);
            a.powf(b)
        }
        "powf" => {
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

    // Note: textual macro extraction (parsing `lag!{...}` or `fa!{...}` from the
    // raw model text) was removed. Build-time emit_ir should populate
    // `lag_map` and `fa_map` in the IR. If those maps are missing but the
    // textual fields are present the loader will now produce a parse error.

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
        // split by ',' and trim
        let parts: Vec<String> = body
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '"' || c == '\''))
            .map(|s| s.to_string())
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
            parse_errors.push(format!(
                "fetch_cov! macro expects at least (cov, t, name...), got '{}'",
                body
            ));
            continue;
        }
        // first arg: cov variable (identifier)
        let cov_var = parts[0].clone();
        if cov_var.is_empty() || !cov_var.chars().next().unwrap().is_ascii_alphabetic() {
            parse_errors.push(format!(
                "invalid first argument '{}' in fetch_cov! macro",
                cov_var
            ));
        }
        // second arg: time variable (allow t or _t or identifier)
        let _tvar = parts[1].clone();
        if _tvar.is_empty() {
            parse_errors.push(format!(
                "invalid time argument '{}' in fetch_cov! macro",
                _tvar
            ));
        }
        // remaining args: covariate names (can't validate existence here)
        for name in parts.iter().skip(2) {
            if name.is_empty() {
                parse_errors.push(format!(
                    "empty covariate name in fetch_cov! macro body '{}'",
                    body
                ));
            }
            // allow underscore-prefixed names
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
        parse_errors.push("no dx[...] assignments found in diffeq; emit_ir must populate dx entries in the IR".to_string());
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
            Expr::Indexed(name, idx_expr) => {
                // If index is a literal number we can statically validate bounds, otherwise validate the index expression only
                match &**idx_expr {
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
                        // validate nested expressions inside the index
                        validate_expr(other, pmap, nstates, nparams, errors);
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
