//! Expression AST for the JIT-compiled model DSL.
//!
//! Expressions reference symbolic names (parameters, compartments, covariates,
//! `rateiv[i]`, `bolus[i]`, and the special `t`). The model compiler resolves
//! these names to concrete indices before lowering to Cranelift IR.

use std::fmt;

/// A node in the expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal floating point constant.
    Const(f64),
    /// Identifier — could be a parameter, compartment, covariate, or `t`.
    Ident(String),
    /// `name[index]` indexing. Used for `rateiv[i]` and `bolus[i]`.
    Index(String, usize),
    /// Unary minus.
    Neg(Box<Expr>),
    /// Binary arithmetic.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// Function call: `name(arg1, arg2, ...)`.
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
        };
        f.write_str(s)
    }
}
