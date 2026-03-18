//! Language-agnostic math expression parser and Rust code emitter.
//!
//! This module parses a simple math expression language used in JSON model
//! definitions and transpiles it to Rust code. The expression language is
//! intentionally simple and language-agnostic — it uses standard mathematical
//! notation (e.g. `^` for power, `exp()`, `ln()`) rather than Rust-specific
//! syntax (e.g. `.powf()`, `.ln()`).
//!
//! # Expression Grammar
//!
//! ```text
//! expr     := ternary
//! ternary  := logic ("?" logic ":" logic)?
//! logic    := compare (("and" | "or") compare)*
//! compare  := addition (("==" | "!=" | "<=" | ">=" | "<" | ">") addition)?
//! addition := multiply (("+" | "-") multiply)*
//! multiply := unary (("*" | "/") unary)*
//! unary    := ("-" | "not") unary | power
//! power    := call ("^" unary)?
//! call     := IDENT "(" args ")" | indexing
//! indexing := atom ("[" expr "]")?
//! atom     := NUMBER | IDENT | "(" expr ")"
//! args     := expr ("," expr)*
//! ```
//!
//! # Supported Functions
//!
//! `exp`, `ln`, `log`, `log2`, `log10`, `sqrt`, `abs`, `min`, `max`,
//! `floor`, `ceil`, `sin`, `cos`, `if`
//!
//! # Examples
//!
//! | Expression | Generated Rust |
//! |---|---|
//! | `CL * (wt / 70)^0.75` | `CL * (wt / 70.0).powf(0.75)` |
//! | `exp(-ke * t)` | `(-ke * t).exp()` |
//! | `if(sex == 1, V * 0.8, V)` | `if sex == 1.0 { V * 0.8 } else { V }` |
//! | `ln(x)` | `(x).ln()` |

use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════════
// AST
// ═══════════════════════════════════════════════════════════════════════════════

/// AST node for parsed expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric literal
    Number(f64),
    /// Variable or parameter reference
    Ident(String),
    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary negation
    Neg(Box<Expr>),
    /// Logical not
    Not(Box<Expr>),
    /// Function call (e.g., exp(x), ln(x), sqrt(x))
    FuncCall { name: String, args: Vec<Expr> },
    /// Conditional: if(cond, then, else)
    If {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Array/vector indexing: name[index]
    Index { name: String, index: Box<Expr> },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Pow => write!(f, "^"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gte => write!(f, ">="),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
        }
    }
}

/// Return a numeric precedence level for a binary operator (higher = binds tighter)
fn precedence(op: BinOp) -> u8 {
    match op {
        BinOp::Or => 1,
        BinOp::And => 2,
        BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => 3,
        BinOp::Add | BinOp::Sub => 4,
        BinOp::Mul | BinOp::Div => 5,
        BinOp::Pow => 6,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Question,
    Colon,
    EqEq,
    Neq,
    Lte,
    Gte,
    Lt,
    Gt,
    And,
    Or,
    Not,
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{}", n),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Question => write!(f, "?"),
            Token::Colon => write!(f, ":"),
            Token::EqEq => write!(f, "=="),
            Token::Neq => write!(f, "!="),
            Token::Lte => write!(f, "<="),
            Token::Gte => write!(f, ">="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::And => write!(f, "and"),
            Token::Or => write!(f, "or"),
            Token::Not => write!(f, "not"),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

/// Parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // Numbers (including decimals and scientific notation)
        if ch.is_ascii_digit()
            || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Scientific notation: e.g. 1e-5, 2.5E+3
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            let num: f64 = num_str.parse().map_err(|_| ParseError {
                message: format!("invalid number: '{}'", num_str),
                position: start,
            })?;
            tokens.push(Token::Number(num));
            continue;
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            match word.as_str() {
                "and" => tokens.push(Token::And),
                "or" => tokens.push(Token::Or),
                "not" => tokens.push(Token::Not),
                _ => tokens.push(Token::Ident(word)),
            }
            continue;
        }

        // Two-character operators
        if i + 1 < chars.len() {
            let two: String = chars[i..i + 2].iter().collect();
            match two.as_str() {
                "==" => {
                    tokens.push(Token::EqEq);
                    i += 2;
                    continue;
                }
                "!=" => {
                    tokens.push(Token::Neq);
                    i += 2;
                    continue;
                }
                "<=" => {
                    tokens.push(Token::Lte);
                    i += 2;
                    continue;
                }
                ">=" => {
                    tokens.push(Token::Gte);
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }

        // Single-character operators
        match ch {
            '+' => tokens.push(Token::Plus),
            '-' => tokens.push(Token::Minus),
            '*' => tokens.push(Token::Star),
            '/' => tokens.push(Token::Slash),
            '^' => tokens.push(Token::Caret),
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            '[' => tokens.push(Token::LBracket),
            ']' => tokens.push(Token::RBracket),
            ',' => tokens.push(Token::Comma),
            '?' => tokens.push(Token::Question),
            ':' => tokens.push(Token::Colon),
            '<' => tokens.push(Token::Lt),
            '>' => tokens.push(Token::Gt),
            _ => {
                return Err(ParseError {
                    message: format!("unexpected character: '{}'", ch),
                    position: i,
                });
            }
        }
        i += 1;
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════════════

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        let tok = self.advance();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            Ok(())
        } else {
            Err(ParseError {
                message: format!("expected '{}', got '{}'", expected, tok),
                position: self.pos.saturating_sub(1),
            })
        }
    }

    /// Top-level: parse a full expression
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_ternary()
    }

    /// Ternary: logic ? logic : logic
    fn parse_ternary(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_logic()?;
        if matches!(self.peek(), Token::Question) {
            self.advance(); // consume ?
            let then_expr = self.parse_logic()?;
            self.expect(&Token::Colon)?;
            let else_expr = self.parse_logic()?;
            Ok(Expr::If {
                condition: Box::new(expr),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            })
        } else {
            Ok(expr)
        }
    }

    /// Logic: ("not")? compare (("and" | "or") ("not")? compare)*
    fn parse_logic(&mut self) -> Result<Expr, ParseError> {
        let mut left = if matches!(self.peek(), Token::Not) {
            self.advance();
            Expr::Not(Box::new(self.parse_compare()?))
        } else {
            self.parse_compare()?
        };
        loop {
            let op = match self.peek() {
                Token::And => BinOp::And,
                Token::Or => BinOp::Or,
                _ => break,
            };
            self.advance();
            let right = if matches!(self.peek(), Token::Not) {
                self.advance();
                Expr::Not(Box::new(self.parse_compare()?))
            } else {
                self.parse_compare()?
            };
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// Compare: addition (comp_op addition)?
    fn parse_compare(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_addition()?;
        let op = match self.peek() {
            Token::EqEq => BinOp::Eq,
            Token::Neq => BinOp::Neq,
            Token::Lt => BinOp::Lt,
            Token::Gt => BinOp::Gt,
            Token::Lte => BinOp::Lte,
            Token::Gte => BinOp::Gte,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_addition()?;
        Ok(Expr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Addition: multiply (("+" | "-") multiply)*
    fn parse_addition(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiply()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiply()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// Multiply: unary (("*" | "/") unary)*
    fn parse_multiply(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// Unary: "-" unary | power
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Neg(Box::new(expr)))
            }
            _ => self.parse_power(),
        }
    }

    /// Power: call ("^" unary)?  (right-associative)
    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let base = self.parse_call()?;
        if matches!(self.peek(), Token::Caret) {
            self.advance();
            let exp = self.parse_unary()?; // right-associative
            Ok(Expr::BinOp {
                op: BinOp::Pow,
                left: Box::new(base),
                right: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    /// Call: IDENT "(" args ")" | indexing
    fn parse_call(&mut self) -> Result<Expr, ParseError> {
        if let Token::Ident(name) = self.peek().clone() {
            // Check if next-next is '(' — it's a function call
            if self.tokens.get(self.pos + 1) == Some(&Token::LParen) {
                self.advance(); // consume ident
                self.advance(); // consume '('

                // Special case: `if(cond, then, else)`
                if name == "if" {
                    let condition = self.parse_expr()?;
                    self.expect(&Token::Comma)?;
                    let then_expr = self.parse_expr()?;
                    self.expect(&Token::Comma)?;
                    let else_expr = self.parse_expr()?;
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::If {
                        condition: Box::new(condition),
                        then_expr: Box::new(then_expr),
                        else_expr: Box::new(else_expr),
                    });
                }

                // Regular function call
                let mut args = Vec::new();
                if !matches!(self.peek(), Token::RParen) {
                    args.push(self.parse_expr()?);
                    while matches!(self.peek(), Token::Comma) {
                        self.advance();
                        args.push(self.parse_expr()?);
                    }
                }
                self.expect(&Token::RParen)?;
                return Ok(Expr::FuncCall { name, args });
            }
        }
        self.parse_indexing()
    }

    /// Indexing: atom ("[" expr "]")?
    fn parse_indexing(&mut self) -> Result<Expr, ParseError> {
        // Check for ident[expr] pattern
        if let Token::Ident(name) = self.peek().clone() {
            if self.tokens.get(self.pos + 1) == Some(&Token::LBracket) {
                self.advance(); // consume ident
                self.advance(); // consume '['
                let index = self.parse_expr()?;
                self.expect(&Token::RBracket)?;
                return Ok(Expr::Index {
                    name,
                    index: Box::new(index),
                });
            }
        }
        self.parse_atom()
    }

    /// Atom: NUMBER | IDENT | "(" expr ")"
    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Expr::Number(n))
            }
            Token::Ident(s) => {
                self.advance();
                Ok(Expr::Ident(s))
            }
            Token::LParen => {
                self.advance(); // consume '('
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            tok => Err(ParseError {
                message: format!("unexpected token: '{}'", tok),
                position: self.pos,
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse a math expression string into an AST
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    // Ensure we consumed all tokens
    if !matches!(parser.peek(), Token::Eof) {
        return Err(ParseError {
            message: format!("unexpected token after expression: '{}'", parser.peek()),
            position: parser.pos,
        });
    }
    Ok(expr)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rust Code Emitter
// ═══════════════════════════════════════════════════════════════════════════════

/// Known math functions and their Rust method equivalents
const KNOWN_FUNCTIONS: &[&str] = &[
    "exp", "ln", "log", "log2", "log10", "sqrt", "abs", "min", "max", "floor", "ceil", "sin",
    "cos", "tan", "asin", "acos", "atan",
];

/// Context for Rust code emission
pub struct RustEmitter;

impl RustEmitter {
    /// Create a new emitter
    pub fn new() -> Self {
        Self
    }

    /// Emit Rust code string from an AST
    pub fn emit(&self, expr: &Expr) -> String {
        match expr {
            Expr::Number(n) => format_number(*n),
            Expr::Ident(name) => name.clone(),
            Expr::Neg(inner) => match inner.as_ref() {
                Expr::Ident(_) | Expr::Number(_) | Expr::Index { .. } => {
                    format!("-{}", self.emit(inner))
                }
                _ => format!("-({})", self.emit(inner)),
            },
            Expr::Not(inner) => {
                let s = self.emit(inner);
                format!("!({})", s)
            }
            Expr::BinOp { op, left, right } => {
                if *op == BinOp::Pow {
                    let base = self.emit_wrapped(left);
                    let exp = self.emit(right);
                    format!("{}.powf({})", base, exp)
                } else {
                    let l = self.emit_with_parens(left, *op, true);
                    let r = self.emit_with_parens(right, *op, false);
                    let op_str = match op {
                        BinOp::Add => "+",
                        BinOp::Sub => "-",
                        BinOp::Mul => "*",
                        BinOp::Div => "/",
                        BinOp::Eq => "==",
                        BinOp::Neq => "!=",
                        BinOp::Lt => "<",
                        BinOp::Gt => ">",
                        BinOp::Lte => "<=",
                        BinOp::Gte => ">=",
                        BinOp::And => "&&",
                        BinOp::Or => "||",
                        BinOp::Pow => unreachable!(),
                    };
                    format!("{} {} {}", l, op_str, r)
                }
            }
            Expr::FuncCall { name, args } => self.emit_func_call(name, args),
            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond = self.emit(condition);
                let then_s = self.emit(then_expr);
                let else_s = self.emit(else_expr);
                format!("if {} {{ {} }} else {{ {} }}", cond, then_s, else_s)
            }
            Expr::Index { name, index } => {
                // For array indices, emit integer-valued numbers as integers (x[1] not x[1.0])
                let idx = match index.as_ref() {
                    Expr::Number(n) if *n == n.floor() && *n >= 0.0 => {
                        format!("{}", *n as usize)
                    }
                    _ => self.emit(index),
                };
                format!("{}[{}]", name, idx)
            }
        }
    }

    /// Emit for use as a method receiver (e.g., `x.exp()`).
    /// Only wraps in parens if the expression is complex (not a simple ident/number/index).
    fn emit_wrapped(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(_) | Expr::Number(_) | Expr::Index { .. } => self.emit(expr),
            _ => format!("({})", self.emit(expr)),
        }
    }

    /// Emit a child expression, adding parentheses if the child has lower precedence
    fn emit_with_parens(&self, child: &Expr, parent_op: BinOp, is_left: bool) -> String {
        if let Expr::BinOp { op: child_op, .. } = child {
            let needs_parens = if precedence(*child_op) < precedence(parent_op) {
                true
            } else if precedence(*child_op) == precedence(parent_op) && !is_left {
                // Right-associative grouping needed for same-precedence on the right
                // e.g. a / (b + c) — add parens to avoid a / b + c
                // But a - b - c is OK (left-associative)
                matches!(parent_op, BinOp::Sub | BinOp::Div)
            } else {
                false
            };
            if needs_parens {
                return format!("({})", self.emit(child));
            }
        }
        self.emit(child)
    }

    /// Emit a function call
    fn emit_func_call(&self, name: &str, args: &[Expr]) -> String {
        match name {
            // Single-arg functions that become Rust method calls
            "exp" | "ln" | "sqrt" | "abs" | "floor" | "ceil" | "sin" | "cos" | "tan" | "asin"
            | "acos" | "atan" => {
                if args.len() != 1 {
                    // Emit as-is, validation will catch it
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    return format!("{}({})", name, arg_strs.join(", "));
                }
                let arg = self.emit_wrapped(&args[0]);
                let method = match name {
                    "ln" => "ln",
                    "exp" => "exp",
                    "sqrt" => "sqrt",
                    "abs" => "abs",
                    "floor" => "floor",
                    "ceil" => "ceil",
                    "sin" => "sin",
                    "cos" => "cos",
                    "tan" => "tan",
                    "asin" => "asin",
                    "acos" => "acos",
                    "atan" => "atan",
                    _ => unreachable!(),
                };
                format!("{}.{}()", arg, method)
            }
            // log(x) = ln(x), log(base, x) = log_base(x)
            "log" => {
                if args.len() == 1 {
                    let arg = self.emit_wrapped(&args[0]);
                    format!("{}.ln()", arg)
                } else if args.len() == 2 {
                    let base = self.emit(&args[0]);
                    let x = self.emit_wrapped(&args[1]);
                    format!("{}.log({})", x, base)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    format!("log({})", arg_strs.join(", "))
                }
            }
            "log2" => {
                if args.len() == 1 {
                    let arg = self.emit_wrapped(&args[0]);
                    format!("{}.log2()", arg)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    format!("log2({})", arg_strs.join(", "))
                }
            }
            "log10" => {
                if args.len() == 1 {
                    let arg = self.emit_wrapped(&args[0]);
                    format!("{}.log10()", arg)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    format!("log10({})", arg_strs.join(", "))
                }
            }
            // Two-arg functions
            "min" => {
                if args.len() == 2 {
                    let a = self.emit(&args[0]);
                    let b = self.emit(&args[1]);
                    format!("{}_f64.min({})", a, b)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    format!("min({})", arg_strs.join(", "))
                }
            }
            "max" => {
                if args.len() == 2 {
                    let a = self.emit(&args[0]);
                    let b = self.emit(&args[1]);
                    format!("{}_f64.max({})", a, b)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                    format!("max({})", arg_strs.join(", "))
                }
            }
            // Unknown function — emit as-is
            _ => {
                let arg_strs: Vec<_> = args.iter().map(|a| self.emit(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
        }
    }
}

impl Default for RustEmitter {
    fn default() -> Self {
        Self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse and emit Rust code from a math expression string
pub fn to_rust(input: &str) -> Result<String, ParseError> {
    let expr = parse(input)?;
    let emitter = RustEmitter::new();
    Ok(emitter.emit(&expr))
}

/// Parse and emit Rust code with compartment/state name mappings.
///
/// This is used by the codegen to resolve named compartment references
/// in array indexing, e.g. `rateiv[central]` → `rateiv[1]`.
pub fn to_rust_with_names(
    input: &str,
    name_to_index: &HashMap<String, usize>,
) -> Result<String, ParseError> {
    let mut expr = parse(input)?;
    resolve_named_indices(&mut expr, name_to_index);
    let emitter = RustEmitter::new();
    Ok(emitter.emit(&expr))
}

/// Resolve named array indices to numeric indices using a name map.
/// e.g. `rateiv[central]` with map {central: 1} → `rateiv[1]`
fn resolve_named_indices(expr: &mut Expr, map: &HashMap<String, usize>) {
    match expr {
        Expr::Index { name: _, index } => {
            // If the index is a bare identifier that matches a compartment name, replace with its index
            if let Expr::Ident(ref idx_name) = **index {
                if let Some(&idx) = map.get(idx_name) {
                    *index = Box::new(Expr::Number(idx as f64));
                }
            }
            resolve_named_indices(index, map);
        }
        Expr::BinOp { left, right, .. } => {
            resolve_named_indices(left, map);
            resolve_named_indices(right, map);
        }
        Expr::Neg(inner) | Expr::Not(inner) => resolve_named_indices(inner, map),
        Expr::FuncCall { args, .. } => {
            for arg in args {
                resolve_named_indices(arg, map);
            }
        }
        Expr::If {
            condition,
            then_expr,
            else_expr,
        } => {
            resolve_named_indices(condition, map);
            resolve_named_indices(then_expr, map);
            resolve_named_indices(else_expr, map);
        }
        Expr::Number(_) | Expr::Ident(_) => {}
    }
}

/// Collect all identifiers referenced in an expression
pub fn collect_identifiers(expr: &Expr) -> Vec<String> {
    let mut ids = Vec::new();
    collect_ids_recursive(expr, &mut ids);
    ids.sort();
    ids.dedup();
    ids
}

fn collect_ids_recursive(expr: &Expr, ids: &mut Vec<String>) {
    match expr {
        Expr::Ident(name) => ids.push(name.clone()),
        Expr::Number(_) => {}
        Expr::Neg(inner) | Expr::Not(inner) => collect_ids_recursive(inner, ids),
        Expr::BinOp { left, right, .. } => {
            collect_ids_recursive(left, ids);
            collect_ids_recursive(right, ids);
        }
        Expr::FuncCall { args, .. } => {
            for arg in args {
                collect_ids_recursive(arg, ids);
            }
        }
        Expr::If {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_ids_recursive(condition, ids);
            collect_ids_recursive(then_expr, ids);
            collect_ids_recursive(else_expr, ids);
        }
        Expr::Index { name, index } => {
            ids.push(name.clone());
            collect_ids_recursive(index, ids);
        }
    }
}

/// Check if a function name is a known built-in
pub fn is_known_function(name: &str) -> bool {
    KNOWN_FUNCTIONS.contains(&name)
}

/// Format a number for Rust output — use integer format when exact, otherwise float
fn format_number(n: f64) -> String {
    if n == n.floor() && n.abs() < 1e15 {
        // Integer-valued: emit as float literal for type consistency
        if n == 0.0 {
            "0.0".to_string()
        } else {
            format!("{:.1}", n)
        }
    } else {
        // Full precision
        let s = format!("{}", n);
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{}.0", s)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_arithmetic() {
        assert_eq!(to_rust("a + b").unwrap(), "a + b");
        assert_eq!(to_rust("a - b").unwrap(), "a - b");
        assert_eq!(to_rust("a * b").unwrap(), "a * b");
        assert_eq!(to_rust("a / b").unwrap(), "a / b");
    }

    #[test]
    fn test_power() {
        assert_eq!(to_rust("x^2").unwrap(), "x.powf(2.0)");
        assert_eq!(to_rust("(wt / 70)^0.75").unwrap(), "(wt / 70.0).powf(0.75)");
    }

    #[test]
    fn test_negation() {
        assert_eq!(to_rust("-ka * depot").unwrap(), "-ka * depot");
        assert_eq!(to_rust("-x").unwrap(), "-x");
    }

    #[test]
    fn test_functions() {
        assert_eq!(to_rust("exp(x)").unwrap(), "x.exp()");
        assert_eq!(to_rust("ln(x)").unwrap(), "x.ln()");
        assert_eq!(to_rust("log(x)").unwrap(), "x.ln()");
        assert_eq!(to_rust("sqrt(x)").unwrap(), "x.sqrt()");
        assert_eq!(to_rust("abs(x)").unwrap(), "x.abs()");
        assert_eq!(to_rust("log10(x)").unwrap(), "x.log10()");
        assert_eq!(to_rust("log2(x)").unwrap(), "x.log2()");
        // Complex expressions get wrapped
        assert_eq!(to_rust("exp(a + b)").unwrap(), "(a + b).exp()");
    }

    #[test]
    fn test_min_max() {
        assert_eq!(to_rust("min(a, b)").unwrap(), "a_f64.min(b)");
        assert_eq!(to_rust("max(a, b)").unwrap(), "a_f64.max(b)");
    }

    #[test]
    fn test_if_function() {
        assert_eq!(
            to_rust("if(sex == 1, V * 0.8, V)").unwrap(),
            "if sex == 1.0 { V * 0.8 } else { V }"
        );
    }

    #[test]
    fn test_ternary() {
        assert_eq!(
            to_rust("sex == 1 ? V * 0.8 : V").unwrap(),
            "if sex == 1.0 { V * 0.8 } else { V }"
        );
    }

    #[test]
    fn test_nested_if() {
        assert_eq!(
            to_rust("if(sex == 1, V * 0.8, if(age < 18, V * 1.2, V))").unwrap(),
            "if sex == 1.0 { V * 0.8 } else { if age < 18.0 { V * 1.2 } else { V } }"
        );
    }

    #[test]
    fn test_comparison_operators() {
        assert_eq!(to_rust("a == b").unwrap(), "a == b");
        assert_eq!(to_rust("a != b").unwrap(), "a != b");
        assert_eq!(to_rust("a < b").unwrap(), "a < b");
        assert_eq!(to_rust("a > b").unwrap(), "a > b");
        assert_eq!(to_rust("a <= b").unwrap(), "a <= b");
        assert_eq!(to_rust("a >= b").unwrap(), "a >= b");
    }

    #[test]
    fn test_logical_operators() {
        assert_eq!(to_rust("a > 0 and b > 0").unwrap(), "a > 0.0 && b > 0.0");
        assert_eq!(to_rust("a > 0 or b > 0").unwrap(), "a > 0.0 || b > 0.0");
        assert_eq!(to_rust("not a > 0").unwrap(), "!(a > 0.0)");
    }

    #[test]
    fn test_array_indexing() {
        assert_eq!(to_rust("x[0]").unwrap(), "x[0]");
        assert_eq!(to_rust("x[1] / V").unwrap(), "x[1] / V");
        assert_eq!(to_rust("rateiv[0]").unwrap(), "rateiv[0]");
    }

    #[test]
    fn test_named_indexing() {
        let mut map = HashMap::new();
        map.insert("central".to_string(), 1);
        map.insert("depot".to_string(), 0);
        assert_eq!(
            to_rust_with_names("rateiv[central]", &map).unwrap(),
            "rateiv[1]"
        );
        assert_eq!(to_rust_with_names("x[depot]", &map).unwrap(), "x[0]");
    }

    #[test]
    fn test_complex_pharmacometric_expression() {
        // Allometric scaling
        assert_eq!(
            to_rust("CL * (wt / 70)^0.75").unwrap(),
            "CL * (wt / 70.0).powf(0.75)"
        );

        // Exponential visit effect
        assert_eq!(
            to_rust("CL * exp(theta1 * (pkvisit - 1))").unwrap(),
            "CL * (theta1 * (pkvisit - 1.0)).exp()"
        );

        // Combined allometric + exponential (Neely-style)
        let result = to_rust("cls * exp(theta1 * (pkvisit - 1)) * (wt / 70)^0.75").unwrap();
        assert!(result.contains(".exp()"));
        assert!(result.contains(".powf(0.75)"));
    }

    #[test]
    fn test_scientific_notation() {
        assert_eq!(to_rust("1e-5").unwrap(), "0.00001");
        assert_eq!(to_rust("1.5e3").unwrap(), "1500.0");
    }

    #[test]
    fn test_michaelis_menten() {
        let result = to_rust("-Vmax * (central / V) / (Km + central / V)").unwrap();
        assert_eq!(result, "-Vmax * central / V / (Km + central / V)");
    }

    #[test]
    fn test_ode_expressions() {
        assert_eq!(to_rust("-ka * depot").unwrap(), "-ka * depot");
        let result =
            to_rust("ka * depot - ke * central - k12 * central + k21 * peripheral").unwrap();
        assert!(result.contains("ka * depot"));
        assert!(result.contains("ke * central"));
    }

    #[test]
    fn test_collect_identifiers() {
        let expr = parse("CL * (wt / 70)^0.75 + V").unwrap();
        let ids = collect_identifiers(&expr);
        assert!(ids.contains(&"CL".to_string()));
        assert!(ids.contains(&"wt".to_string()));
        assert!(ids.contains(&"V".to_string()));
        assert!(!ids.contains(&"70".to_string())); // numbers excluded
    }

    #[test]
    fn test_empty_input_error() {
        assert!(parse("").is_err());
    }

    #[test]
    fn test_invalid_syntax_error() {
        assert!(parse("+ +").is_err());
        assert!(parse("a +").is_err());
    }

    #[test]
    fn test_log_with_base() {
        assert_eq!(to_rust("log(10, x)").unwrap(), "x.log(10.0)");
    }

    #[test]
    fn test_nested_if_function() {
        let result = to_rust("if(sex == 1, if(age > 65, V * 0.5, V * 0.8), V)").unwrap();
        assert!(result.contains("if sex == 1.0"));
        assert!(result.contains("if age > 65.0"));
    }

    #[test]
    fn test_chained_comparisons_with_logic() {
        assert_eq!(
            to_rust("a > 0 and b < 10 or c == 5").unwrap(),
            "a > 0.0 && b < 10.0 || c == 5.0"
        );
    }

    #[test]
    fn test_power_right_associativity() {
        // x^2^3 should parse as x^(2^3)
        assert_eq!(to_rust("x^2^3").unwrap(), "x.powf(2.0.powf(3.0))");
    }

    #[test]
    fn test_negative_in_expression() {
        assert_eq!(to_rust("-ka * depot").unwrap(), "-ka * depot");
        assert_eq!(to_rust("-(a + b)").unwrap(), "-(a + b)");
    }

    #[test]
    fn test_division_precedence() {
        // a / (b + c) must preserve parens
        assert_eq!(to_rust("a / (b + c)").unwrap(), "a / (b + c)");
        // a + b / c should not add parens
        assert_eq!(to_rust("a + b / c").unwrap(), "a + b / c");
    }

    #[test]
    fn test_allometric_expression() {
        // Typical allometric scaling: CL * (WT / 70)^0.75
        let result = to_rust("CL * (WT / 70)^0.75").unwrap();
        assert!(result.contains("powf(0.75)"));
        assert!(result.contains("WT / 70.0"));
    }

    #[test]
    fn test_collect_identifiers_complex() {
        let expr = parse("CL * (WT / 70)^0.75 + V").unwrap();
        let ids = collect_identifiers(&expr);
        assert!(ids.contains(&"CL".to_string()));
        assert!(ids.contains(&"WT".to_string()));
        assert!(ids.contains(&"V".to_string()));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_pharmacometric_secondary_equation() {
        // ke = CL / V
        assert_eq!(to_rust("CL / V").unwrap(), "CL / V");
        // Complex clearance: CLs * (wt / 70)^0.75 * if(sex == 1, 0.8, 1.0)
        let result = to_rust("CLs * (wt / 70)^0.75 * if(sex == 1, 0.8, 1.0)").unwrap();
        assert!(result.contains("CLs"));
        assert!(result.contains("powf(0.75)"));
        assert!(result.contains("if sex == 1.0"));
    }

    #[test]
    fn test_michaelis_menten_expression() {
        let result = to_rust("-Vmax * (central / V) / (Km + central / V)").unwrap();
        assert!(result.contains("Vmax"));
        assert!(result.contains("central / V"));
        assert!(result.contains("Km"));
    }

    #[test]
    fn test_ternary_expression() {
        assert_eq!(
            to_rust("sex == 1 ? V * 0.8 : V").unwrap(),
            "if sex == 1.0 { V * 0.8 } else { V }"
        );
    }

    #[test]
    fn test_floor_ceil_functions() {
        assert_eq!(to_rust("floor(x)").unwrap(), "x.floor()");
        assert_eq!(to_rust("ceil(x + 1)").unwrap(), "(x + 1.0).ceil()");
    }

    #[test]
    fn test_trig_functions() {
        assert_eq!(to_rust("sin(x)").unwrap(), "x.sin()");
        assert_eq!(to_rust("cos(x * 2)").unwrap(), "(x * 2.0).cos()");
    }
}
