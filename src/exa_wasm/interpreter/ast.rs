// AST types for the exa_wasm interpreter
use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    Ident(String),              // e.g. ke
    Param(usize),               // parameter by index (p[0] rewritten to Param(0))
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

#[derive(Debug, Clone)]
pub enum Token {
    Num(f64),
    Bool(bool),
    Ident(String),
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Assign,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Lhs {
    Ident(String),
    Indexed(String, Box<Expr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Expr(Expr),
    Assign(Lhs, Expr),
    Block(Vec<Stmt>),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub pos: usize,
    pub found: Option<Token>,
    pub expected: Vec<String>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.expected.is_empty() {
            write!(
                f,
                "parse error at pos {} found={:?} expected={:?}",
                self.pos, self.found, self.expected
            )
        } else if let Some(tok) = &self.found {
            write!(f, "parse error at pos {} found={:?}", self.pos, tok)
        } else {
            write!(f, "parse error at pos {} found=<end>", self.pos)
        }
    }
}

impl std::error::Error for ParseError {}
