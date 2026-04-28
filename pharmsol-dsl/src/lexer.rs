use crate::diagnostic::{ParseError, Span};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub starts_line: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Model,
    Kind,
    Ode,
    Analytical,
    Sde,
    Parameters,
    Constants,
    Covariates,
    States,
    Routes,
    Derive,
    Dynamics,
    Outputs,
    Init,
    Drift,
    Diffusion,
    Particles,
    If,
    Else,
    For,
    In,
    Let,
    True,
    False,
    Ident(String),
    Number(f64),
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    At,
    Arrow,
    DotDot,
    Eq,
    EqEq,
    Bang,
    BangEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
}

impl TokenKind {
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Model => "`model`".into(),
            TokenKind::Kind => "`kind`".into(),
            TokenKind::Ode => "`ode`".into(),
            TokenKind::Analytical => "`analytical`".into(),
            TokenKind::Sde => "`sde`".into(),
            TokenKind::Parameters => "`parameters`".into(),
            TokenKind::Constants => "`constants`".into(),
            TokenKind::Covariates => "`covariates`".into(),
            TokenKind::States => "`states`".into(),
            TokenKind::Routes => "`routes`".into(),
            TokenKind::Derive => "`derive`".into(),
            TokenKind::Dynamics => "`dynamics`".into(),
            TokenKind::Outputs => "`outputs`".into(),
            TokenKind::Init => "`init`".into(),
            TokenKind::Drift => "`drift`".into(),
            TokenKind::Diffusion => "`diffusion`".into(),
            TokenKind::Particles => "`particles`".into(),
            TokenKind::If => "`if`".into(),
            TokenKind::Else => "`else`".into(),
            TokenKind::For => "`for`".into(),
            TokenKind::In => "`in`".into(),
            TokenKind::Let => "`let`".into(),
            TokenKind::True => "`true`".into(),
            TokenKind::False => "`false`".into(),
            TokenKind::Ident(name) => format!("identifier `{name}`"),
            TokenKind::Number(value) => format!("number `{value}`"),
            TokenKind::LBrace => "`{`".into(),
            TokenKind::RBrace => "`}`".into(),
            TokenKind::LParen => "`(`".into(),
            TokenKind::RParen => "`)`".into(),
            TokenKind::LBracket => "`[`".into(),
            TokenKind::RBracket => "`]`".into(),
            TokenKind::Comma => "`,`".into(),
            TokenKind::Semi => "`;`".into(),
            TokenKind::At => "`@`".into(),
            TokenKind::Arrow => "`->`".into(),
            TokenKind::DotDot => "`..`".into(),
            TokenKind::Eq => "`=`".into(),
            TokenKind::EqEq => "`==`".into(),
            TokenKind::Bang => "`!`".into(),
            TokenKind::BangEq => "`!=`".into(),
            TokenKind::Lt => "`<`".into(),
            TokenKind::LtEq => "`<=`".into(),
            TokenKind::Gt => "`>`".into(),
            TokenKind::GtEq => "`>=`".into(),
            TokenKind::AndAnd => "`&&`".into(),
            TokenKind::OrOr => "`||`".into(),
            TokenKind::Plus => "`+`".into(),
            TokenKind::Minus => "`-`".into(),
            TokenKind::Star => "`*`".into(),
            TokenKind::Slash => "`/`".into(),
            TokenKind::Caret => "`^`".into(),
        }
    }
}

pub fn lex(src: &str) -> Result<Vec<Token>, ParseError> {
    Lexer::new(src).lex()
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn lex(mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        while let Some(saw_newline) = self.skip_ws_and_comments() {
            let start = self.pos;
            let Some(ch) = self.peek_char() else {
                break;
            };
            let kind = match ch {
                '{' => {
                    self.bump_char();
                    TokenKind::LBrace
                }
                '}' => {
                    self.bump_char();
                    TokenKind::RBrace
                }
                '(' => {
                    self.bump_char();
                    TokenKind::LParen
                }
                ')' => {
                    self.bump_char();
                    TokenKind::RParen
                }
                '[' => {
                    self.bump_char();
                    TokenKind::LBracket
                }
                ']' => {
                    self.bump_char();
                    TokenKind::RBracket
                }
                ',' => {
                    self.bump_char();
                    TokenKind::Comma
                }
                ';' => {
                    self.bump_char();
                    TokenKind::Semi
                }
                '@' => {
                    self.bump_char();
                    TokenKind::At
                }
                '+' => {
                    self.bump_char();
                    TokenKind::Plus
                }
                '-' => {
                    self.bump_char();
                    if self.peek_char() == Some('>') {
                        self.bump_char();
                        TokenKind::Arrow
                    } else {
                        TokenKind::Minus
                    }
                }
                '*' => {
                    self.bump_char();
                    TokenKind::Star
                }
                '/' => {
                    self.bump_char();
                    TokenKind::Slash
                }
                '^' => {
                    self.bump_char();
                    TokenKind::Caret
                }
                '=' => {
                    self.bump_char();
                    if self.peek_char() == Some('=') {
                        self.bump_char();
                        TokenKind::EqEq
                    } else {
                        TokenKind::Eq
                    }
                }
                '!' => {
                    self.bump_char();
                    if self.peek_char() == Some('=') {
                        self.bump_char();
                        TokenKind::BangEq
                    } else {
                        TokenKind::Bang
                    }
                }
                '<' => {
                    self.bump_char();
                    if self.peek_char() == Some('=') {
                        self.bump_char();
                        TokenKind::LtEq
                    } else {
                        TokenKind::Lt
                    }
                }
                '>' => {
                    self.bump_char();
                    if self.peek_char() == Some('=') {
                        self.bump_char();
                        TokenKind::GtEq
                    } else {
                        TokenKind::Gt
                    }
                }
                '&' => {
                    self.bump_char();
                    if self.peek_char() == Some('&') {
                        self.bump_char();
                        TokenKind::AndAnd
                    } else {
                        return Err(ParseError::new("expected `&&`", Span::new(start, self.pos)));
                    }
                }
                '|' => {
                    self.bump_char();
                    if self.peek_char() == Some('|') {
                        self.bump_char();
                        TokenKind::OrOr
                    } else {
                        return Err(ParseError::new("expected `||`", Span::new(start, self.pos)));
                    }
                }
                '.' => {
                    if self
                        .peek_nth_char(1)
                        .is_some_and(|next| next.is_ascii_digit())
                    {
                        self.scan_number(start)?
                    } else if self.peek_nth_char(1) == Some('.') {
                        self.bump_char();
                        self.bump_char();
                        TokenKind::DotDot
                    } else {
                        return Err(ParseError::new(
                            "unexpected `.`",
                            Span::new(start, start + 1),
                        ));
                    }
                }
                ch if ch.is_ascii_digit() => self.scan_number(start)?,
                ch if ch.is_ascii_alphabetic() || ch == '_' => self.scan_ident_or_keyword(start),
                other => {
                    return Err(ParseError::new(
                        format!("unexpected character `{other}`"),
                        Span::new(start, start + other.len_utf8()),
                    ));
                }
            };
            tokens.push(Token {
                kind,
                span: Span::new(start, self.pos),
                starts_line: saw_newline || tokens.is_empty(),
            });
        }
        Ok(tokens)
    }

    fn skip_ws_and_comments(&mut self) -> Option<bool> {
        let mut saw_newline = false;
        loop {
            while let Some(ch) = self.peek_char() {
                if !ch.is_whitespace() {
                    break;
                }
                saw_newline |= ch == '\n';
                self.bump_char();
            }

            match (self.peek_char(), self.peek_nth_char(1)) {
                (Some('#'), _) | (Some('/'), Some('/')) => {
                    while let Some(ch) = self.peek_char() {
                        self.bump_char();
                        if ch == '\n' {
                            saw_newline = true;
                            break;
                        }
                    }
                }
                _ => break,
            }
        }

        self.peek_char().map(|_| saw_newline)
    }

    fn scan_ident_or_keyword(&mut self, start: usize) -> TokenKind {
        while self
            .peek_char()
            .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            self.bump_char();
        }

        match &self.src[start..self.pos] {
            "model" => TokenKind::Model,
            "kind" => TokenKind::Kind,
            "ode" => TokenKind::Ode,
            "analytical" => TokenKind::Analytical,
            "sde" => TokenKind::Sde,
            "parameters" => TokenKind::Parameters,
            "constants" => TokenKind::Constants,
            "covariates" => TokenKind::Covariates,
            "states" => TokenKind::States,
            "routes" => TokenKind::Routes,
            "derive" => TokenKind::Derive,
            "dynamics" => TokenKind::Dynamics,
            "outputs" => TokenKind::Outputs,
            "init" => TokenKind::Init,
            "drift" => TokenKind::Drift,
            "diffusion" => TokenKind::Diffusion,
            "particles" => TokenKind::Particles,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "let" => TokenKind::Let,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            ident => TokenKind::Ident(ident.to_string()),
        }
    }

    fn scan_number(&mut self, start: usize) -> Result<TokenKind, ParseError> {
        if self.peek_char() == Some('.') {
            self.bump_char();
        }

        while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
            self.bump_char();
        }

        if self.peek_char() == Some('.') && self.peek_nth_char(1) != Some('.') {
            self.bump_char();
            while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump_char();
            }
        }

        if matches!(self.peek_char(), Some('e') | Some('E')) {
            let checkpoint = self.pos;
            self.bump_char();
            if matches!(self.peek_char(), Some('+') | Some('-')) {
                self.bump_char();
            }

            let exp_start = self.pos;
            while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump_char();
            }

            if self.pos == exp_start {
                return Err(ParseError::new(
                    "expected exponent digits",
                    Span::new(checkpoint, self.pos),
                ));
            }
        }

        let raw = &self.src[start..self.pos];
        let value = raw.parse::<f64>().map_err(|_| {
            ParseError::new(
                format!("invalid number literal `{raw}`"),
                Span::new(start, self.pos),
            )
        })?;
        Ok(TokenKind::Number(value))
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_nth_char(&self, n: usize) -> Option<char> {
        self.src[self.pos..].chars().nth(n)
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }
}
