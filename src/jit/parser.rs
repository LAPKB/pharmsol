//! Pratt parser for the JIT model expression language.
//!
//! Grammar (informal):
//! ```text
//! expr   := term ((`+`|`-`) term)*
//! term   := factor ((`*`|`/`) factor)*
//! factor := unary (`^` factor)?
//! unary  := `-` unary | atom
//! atom   := number | ident | ident `[` int `]` | ident `(` args `)` | `(` expr `)`
//! args   := expr (`,` expr)*
//! ```

use super::ast::{BinOp, Expr};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParseError {
    #[error("unexpected character {0:?} at byte offset {1}")]
    UnexpectedChar(char, usize),
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("expected {expected}, found {found:?}")]
    Expected {
        expected: &'static str,
        found: String,
    },
    #[error("invalid number literal {0:?}")]
    InvalidNumber(String),
    #[error("invalid index literal {0:?}")]
    InvalidIndex(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.bump_char();
            } else {
                break;
            }
        }
    }

    fn next(&mut self) -> Result<Option<Tok>, ParseError> {
        self.skip_ws();
        let Some(c) = self.peek_char() else {
            return Ok(None);
        };
        let start = self.pos;
        let tok = match c {
            '(' => {
                self.bump_char();
                Tok::LParen
            }
            ')' => {
                self.bump_char();
                Tok::RParen
            }
            '[' => {
                self.bump_char();
                Tok::LBracket
            }
            ']' => {
                self.bump_char();
                Tok::RBracket
            }
            ',' => {
                self.bump_char();
                Tok::Comma
            }
            '+' => {
                self.bump_char();
                Tok::Plus
            }
            '-' => {
                self.bump_char();
                Tok::Minus
            }
            '*' => {
                self.bump_char();
                Tok::Star
            }
            '/' => {
                self.bump_char();
                Tok::Slash
            }
            '^' => {
                self.bump_char();
                Tok::Caret
            }
            c if c.is_ascii_digit() || c == '.' => {
                while let Some(c) = self.peek_char() {
                    if c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || c == '+'
                        || c == '-'
                    {
                        // allow exponent sign only after e/E
                        if (c == '+' || c == '-')
                            && !matches!(self.src[..self.pos].chars().last(), Some('e') | Some('E'))
                        {
                            break;
                        }
                        self.bump_char();
                    } else {
                        break;
                    }
                }
                let s = &self.src[start..self.pos];
                let n: f64 = s
                    .parse()
                    .map_err(|_| ParseError::InvalidNumber(s.to_string()))?;
                Tok::Num(n)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                while let Some(c) = self.peek_char() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        self.bump_char();
                    } else {
                        break;
                    }
                }
                Tok::Ident(self.src[start..self.pos].to_string())
            }
            c => return Err(ParseError::UnexpectedChar(c, start)),
        };
        Ok(Some(tok))
    }
}

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn new(src: &str) -> Result<Self, ParseError> {
        let mut lx = Lexer::new(src);
        let mut toks = Vec::new();
        while let Some(t) = lx.next()? {
            toks.push(t);
        }
        Ok(Self { toks, i: 0 })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.i).cloned();
        if t.is_some() {
            self.i += 1;
        }
        t
    }

    fn expect(&mut self, want: &Tok, label: &'static str) -> Result<(), ParseError> {
        match self.bump() {
            Some(ref got) if got == want => Ok(()),
            Some(other) => Err(ParseError::Expected {
                expected: label,
                found: format!("{:?}", other),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_term()?;
        while let Some(t) = self.peek() {
            let op = match t {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_term()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_factor()?;
        while let Some(t) = self.peek() {
            let op = match t {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_factor()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_unary()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.bump();
            // right-associative
            let rhs = self.parse_factor()?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(lhs), Box::new(rhs)))
        } else {
            Ok(lhs)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.bump();
            let inner = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        if matches!(self.peek(), Some(Tok::Plus)) {
            self.bump();
            return self.parse_unary();
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let Some(tok) = self.bump() else {
            return Err(ParseError::UnexpectedEof);
        };
        match tok {
            Tok::Num(n) => Ok(Expr::Const(n)),
            Tok::LParen => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::Ident(name) => match self.peek() {
                Some(Tok::LParen) => {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                    Ok(Expr::Call(name, args))
                }
                Some(Tok::LBracket) => {
                    self.bump();
                    let idx_tok = self.bump().ok_or(ParseError::UnexpectedEof)?;
                    let idx = match idx_tok {
                        Tok::Num(n) if n.fract() == 0.0 && n >= 0.0 => n as usize,
                        other => {
                            return Err(ParseError::InvalidIndex(format!("{:?}", other)));
                        }
                    };
                    self.expect(&Tok::RBracket, "`]`")?;
                    Ok(Expr::Index(name, idx))
                }
                _ => Ok(Expr::Ident(name)),
            },
            other => Err(ParseError::Expected {
                expected: "expression",
                found: format!("{:?}", other),
            }),
        }
    }
}

/// Parse an expression from source text.
pub fn parse(src: &str) -> Result<Expr, ParseError> {
    let mut p = Parser::new(src)?;
    let e = p.parse_expr()?;
    if p.peek().is_some() {
        return Err(ParseError::Expected {
            expected: "end of expression",
            found: format!("{:?}", p.peek().unwrap()),
        });
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let e = parse("ka * depot + 1.5e-2").unwrap();
        assert_eq!(
            e,
            Expr::Bin(
                BinOp::Add,
                Box::new(Expr::Bin(
                    BinOp::Mul,
                    Box::new(Expr::Ident("ka".into())),
                    Box::new(Expr::Ident("depot".into())),
                )),
                Box::new(Expr::Const(0.015)),
            )
        );
    }

    #[test]
    fn rateiv_index() {
        let e = parse("rateiv[0] - (CL/V) * central").unwrap();
        match e {
            Expr::Bin(BinOp::Sub, lhs, _) => {
                assert_eq!(*lhs, Expr::Index("rateiv".into(), 0));
            }
            _ => panic!("expected Sub, got {:?}", e),
        }
    }

    #[test]
    fn calls_and_pow() {
        let e = parse("exp(-ke * t) * 2 ^ 3").unwrap();
        match e {
            Expr::Bin(BinOp::Mul, _, rhs) => assert!(matches!(*rhs, Expr::Bin(BinOp::Pow, _, _))),
            _ => panic!(),
        }
    }
}
