use crate::exa_wasm::interpreter::ast::{Expr, ParseError, Token};

// Tokenizer + recursive-descent parser
pub fn tokenize(s: &str) -> Vec<Token> {
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
            // treat true/false as boolean tokens
            if id.eq_ignore_ascii_case("true") {
                toks.push(Token::Bool(true));
            } else if id.eq_ignore_ascii_case("false") {
                toks.push(Token::Bool(false));
            } else {
                toks.push(Token::Ident(id));
            }
            continue;
        }
        match c {
            '[' => {
                toks.push(Token::LBracket);
                chars.next();
            }
            '{' => {
                toks.push(Token::LBrace);
                chars.next();
            }
            '}' => {
                toks.push(Token::RBrace);
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
                    toks.push(Token::Assign);
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

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    expected: Vec<String>,
}
impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
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
    pub fn parse_expr(&mut self) -> Option<Expr> {
        self.parse_ternary()
    }
    fn parse_ternary(&mut self) -> Option<Expr> {
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
    pub fn parse_expr_result(&mut self) -> Result<Expr, ParseError> {
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
    fn parse_power(&mut self) -> Option<Expr> {
        let node = self.parse_unary()?;
        if let Some(Token::Op('^')) = self.peek() {
            self.next();
            let rhs = self.parse_power()?;
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
            return Some(Expr::UnaryOp {
                op: '!'.to_string(),
                rhs: Box::new(rhs),
            });
        }
        self.parse_primary()
    }
    fn parse_primary(&mut self) -> Option<Expr> {
        let tok = self.next().cloned()?;
        let mut node = match tok {
            Token::Num(v) => Expr::Number(v),
            Token::Bool(b) => Expr::Bool(b),
            Token::Ident(id) => {
                // function call?
                if let Some(Token::LParen) = self.peek().cloned() {
                    self.next();
                    let mut args: Vec<Expr> = Vec::new();
                    if let Some(Token::RParen) = self.peek().cloned() {
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
                            match self.peek().cloned() {
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
                        // after parsing args, produce the Call node
                        Expr::Call {
                            name: id.clone(),
                            args,
                        }
                    }
                // indexed access?
                } else if let Some(Token::LBracket) = self.peek().cloned() {
                    self.next();
                    // parse index expression
                    let idx = self.parse_expr()?;
                    if let Some(Token::RBracket) = self.peek().cloned() {
                        self.next();
                        Expr::Indexed(id.clone(), Box::new(idx))
                    } else {
                        self.expected_push("]");
                        return None;
                    }
                } else {
                    Expr::Ident(id.clone())
                }
            }
            Token::LParen => {
                let expr = self.parse_expr();
                if let Some(Token::RParen) = self.peek().cloned() {
                    self.next();
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

        // method call chaining: .name(args?)
        loop {
            if let Some(Token::Dot) = self.peek().cloned() {
                self.next();
                let name = if let Some(Token::Ident(n)) = self.next().cloned() {
                    n
                } else {
                    self.expected_push("identifier");
                    return None;
                };
                let mut args: Vec<Expr> = Vec::new();
                if let Some(Token::LParen) = self.peek().cloned() {
                    self.next();
                    if let Some(Token::RParen) = self.peek().cloned() {
                        self.next();
                    } else {
                        loop {
                            if let Some(expr) = self.parse_expr() {
                                args.push(expr);
                            } else {
                                self.expected_push("expression");
                                return None;
                            }
                            match self.peek().cloned() {
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

// Statement parsing (small recursive-descent on top of the expression parser)
impl Parser {
    pub fn parse_statements(&mut self) -> Option<Vec<crate::exa_wasm::interpreter::ast::Stmt>> {
        let mut stmts = Vec::new();
        while let Some(tok) = self.peek() {
            match tok {
                Token::RBrace => break,
                _ => {
                    if let Some(s) = self.parse_statement() {
                        stmts.push(s);
                        continue;
                    } else {
                        return None;
                    }
                }
            }
        }
        Some(stmts)
    }

    fn parse_statement(&mut self) -> Option<crate::exa_wasm::interpreter::ast::Stmt> {
        use crate::exa_wasm::interpreter::ast::{Lhs, Stmt};
        // handle `if` as identifier token
        if let Some(Token::Ident(id)) = self.peek().cloned() {
            if id == "if" {
                // consume 'if'
                self.next();
                // allow optional parens around condition
                let cond = if let Some(Token::LParen) = self.peek().cloned() {
                    self.next();
                    let e = self.parse_expr()?;
                    if let Some(Token::RParen) = self.peek().cloned() {
                        self.next();
                    } else {
                        self.expected_push(")");
                        return None;
                    }
                    e
                } else {
                    self.parse_expr()?
                };
                // then branch must be a block
                let then_branch = if let Some(Token::LBrace) = self.peek().cloned() {
                    self.next();
                    let mut pstmts = Vec::new();
                    while let Some(tok) = self.peek().cloned() {
                        if let Token::RBrace = tok {
                            self.next();
                            break;
                        }
                        pstmts.push(self.parse_statement()?);
                    }
                    Stmt::Block(pstmts)
                } else {
                    // single statement as then branch
                    self.parse_statement()
                        .map(Box::new)
                        .map(|b| *b)
                        .unwrap_or(Stmt::Block(vec![]))
                };
                // optional else
                let else_branch = if let Some(Token::Ident(eid)) = self.peek().cloned() {
                    if eid == "else" {
                        self.next();
                        if let Some(Token::LBrace) = self.peek().cloned() {
                            self.next();
                            let mut estmts = Vec::new();
                            while let Some(tok) = self.peek().cloned() {
                                if let Token::RBrace = tok {
                                    self.next();
                                    break;
                                }
                                estmts.push(self.parse_statement()?);
                            }
                            Some(Box::new(Stmt::Block(estmts)))
                        } else if let Some(Token::Ident(_)) = self.peek().cloned() {
                            Some(Box::new(self.parse_statement()?))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                return Some(Stmt::If {
                    cond,
                    then_branch: Box::new(then_branch),
                    else_branch,
                });
            }
        }

        // Attempt assignment: lookahead without consuming
        if let Some(Token::Ident(_)) = self.peek() {
            // lookahead for simple `Ident =` or `Ident [ ... ] =`
            let mut is_assign = false;
            // check immediate next token
            if let Some(next_tok) = self.tokens.get(self.pos + 1) {
                match next_tok {
                    Token::Assign => is_assign = true,
                    Token::LBracket => {
                        // find matching RBracket
                        let mut depth = 0isize;
                        let mut j = self.pos + 1;
                        while j < self.tokens.len() {
                            match self.tokens[j] {
                                Token::LBracket => depth += 1,
                                Token::RBracket => {
                                    depth -= 1;
                                    if depth == 0 {
                                        // check token after RBracket
                                        if let Some(tok_after) = self.tokens.get(j + 1) {
                                            if let Token::Assign = tok_after {
                                                is_assign = true;
                                            }
                                        }
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                    }
                    _ => {}
                }
            }

            if is_assign {
                // parse lhs
                let lhs = if let Some(Token::Ident(name)) = self.next().cloned() {
                    if let Some(Token::LBracket) = self.peek().cloned() {
                        self.next();
                        let idx = self.parse_expr()?;
                        if let Some(Token::RBracket) = self.peek().cloned() {
                            self.next();
                            Lhs::Indexed(name, Box::new(idx))
                        } else {
                            self.expected_push("]");
                            return None;
                        }
                    } else {
                        Lhs::Ident(name)
                    }
                } else {
                    return None;
                };
                // expect assign
                if let Some(Token::Assign) = self.peek().cloned() {
                    self.next();
                    let rhs = self.parse_expr()?;
                    // expect semicolon
                    if let Some(Token::Semicolon) = self.peek().cloned() {
                        self.next();
                    } else {
                        self.expected_push(";");
                        return None;
                    }
                    return Some(Stmt::Assign(lhs, rhs));
                }
            }
        }

        // Expression statement: expr ;
        let expr = self.parse_expr()?;
        if let Some(Token::Semicolon) = self.peek().cloned() {
            self.next();
        } else {
            self.expected_push(";");
            return None;
        }
        Some(Stmt::Expr(expr))
    }
}
