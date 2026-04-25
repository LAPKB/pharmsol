use super::ast::*;
use super::authoring;
use super::diagnostic::{ParseError, Span};
use super::lexer::{lex, Token, TokenKind};

pub fn parse_module(src: &str) -> Result<Module, ParseError> {
    let leading = strip_leading_layout(src);
    if let Some(rest) = leading.strip_prefix("model") {
        let trimmed = rest.trim_start();
        if trimmed.starts_with('=') {
            return authoring::parse_module(src);
        }
        return Parser::new(src)?.parse_module();
    }
    if leading.is_empty() {
        return Parser::new(src)?.parse_module();
    }
    authoring::parse_module(src)
}

pub fn parse_model(src: &str) -> Result<Model, ParseError> {
    let module = parse_module(src)?;
    match module.models.len() {
        1 => Ok(module.models.into_iter().next().unwrap()),
        0 => Err(ParseError::new(
            "expected a `model` declaration",
            Span::empty(src.len()),
        )),
        _ => Err(ParseError::new(
            "expected exactly one `model` declaration",
            module.models[1].span,
        )),
    }
}

fn strip_leading_layout(src: &str) -> &str {
    let mut rest = src;
    loop {
        rest = rest.trim_start_matches(char::is_whitespace);
        if let Some(stripped) = rest.strip_prefix('#') {
            rest = stripped
                .find('\n')
                .map_or("", |index| &stripped[index + 1..]);
            continue;
        }
        if let Some(stripped) = rest.strip_prefix("//") {
            rest = stripped
                .find('\n')
                .map_or("", |index| &stripped[index + 1..]);
            continue;
        }
        return rest;
    }
}

pub(crate) fn parse_expr_fragment(src: &str) -> Result<Expr, ParseError> {
    let mut parser = Parser::new(src)?;
    let expr = parser.parse_expr(0)?;
    if let Some(token) = parser.peek() {
        return Err(ParseError::new(
            format!(
                "expected end of expression, found {}",
                token.kind.describe()
            ),
            token.span,
        ));
    }
    Ok(expr)
}

pub(crate) fn parse_place_fragment(src: &str) -> Result<Place, ParseError> {
    let mut parser = Parser::new(src)?;
    let place = parser.parse_place()?;
    if let Some(token) = parser.peek() {
        return Err(ParseError::new(
            format!("expected end of place, found {}", token.kind.describe()),
            token.span,
        ));
    }
    Ok(place)
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    src_len: usize,
}

impl Parser {
    fn new(src: &str) -> Result<Self, ParseError> {
        Ok(Self::from_tokens(lex(src)?, src.len()))
    }

    fn from_tokens(tokens: Vec<Token>, src_len: usize) -> Self {
        Self {
            tokens,
            index: 0,
            src_len,
        }
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let start = self.peek_span().unwrap_or_else(|| Span::empty(0));
        let mut models = Vec::new();
        while !self.is_eof() {
            models.push(self.parse_model_decl()?);
        }
        let span = if models.is_empty() {
            Span::empty(0)
        } else {
            start.join(models.last().unwrap().span)
        };
        Ok(Module { models, span })
    }

    fn parse_model_decl(&mut self) -> Result<Model, ParseError> {
        let model_kw = self.expect_simple(|kind| matches!(kind, TokenKind::Model), "`model`")?;
        let name = self.parse_ident()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;

        let mut kind = None;
        let mut items = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            if self.at(|kind| matches!(kind, TokenKind::Kind)) {
                let kind_span = self.bump().unwrap().span;
                if kind.is_some() {
                    return Err(ParseError::new(
                        "duplicate `kind` declaration in model body",
                        kind_span,
                    ));
                }
                kind = Some(self.parse_model_kind()?);
            } else {
                items.push(self.parse_model_item()?);
            }
        }

        let close = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        let kind = kind.ok_or_else(|| {
            ParseError::new(
                format!("model `{}` is missing a `kind` declaration", name.text),
                name.span,
            )
        })?;

        Ok(Model {
            name,
            kind,
            items,
            span: model_kw.span.join(close.span),
        })
    }

    fn parse_model_kind(&mut self) -> Result<ModelKind, ParseError> {
        let token = self.bump().ok_or_else(|| {
            ParseError::new(
                "expected a model kind after `kind`",
                Span::empty(self.src_len),
            )
        })?;
        match token.kind {
            TokenKind::Ode => Ok(ModelKind::Ode),
            TokenKind::Analytical => Ok(ModelKind::Analytical),
            TokenKind::Sde => Ok(ModelKind::Sde),
            other => Err(ParseError::new(
                format!(
                    "expected `ode`, `analytical`, or `sde`, found {}",
                    other.describe()
                ),
                token.span,
            )),
        }
    }

    fn parse_model_item(&mut self) -> Result<ModelItem, ParseError> {
        match self.peek_kind() {
            Some(TokenKind::Parameters) => {
                Ok(ModelItem::Parameters(self.parse_parameters_block()?))
            }
            Some(TokenKind::Constants) => Ok(ModelItem::Constants(self.parse_constants_block()?)),
            Some(TokenKind::Covariates) => {
                Ok(ModelItem::Covariates(self.parse_covariates_block()?))
            }
            Some(TokenKind::States) => Ok(ModelItem::States(self.parse_states_block()?)),
            Some(TokenKind::Routes) => Ok(ModelItem::Routes(self.parse_routes_block()?)),
            Some(TokenKind::Derive) => Ok(ModelItem::Derive(self.parse_statement_block("derive")?)),
            Some(TokenKind::Dynamics) => {
                Ok(ModelItem::Dynamics(self.parse_statement_block("dynamics")?))
            }
            Some(TokenKind::Outputs) => {
                Ok(ModelItem::Outputs(self.parse_statement_block("outputs")?))
            }
            Some(TokenKind::Analytical) => {
                Ok(ModelItem::Analytical(self.parse_analytical_block()?))
            }
            Some(TokenKind::Init) => Ok(ModelItem::Init(self.parse_statement_block("init")?)),
            Some(TokenKind::Drift) => Ok(ModelItem::Drift(self.parse_statement_block("drift")?)),
            Some(TokenKind::Diffusion) => Ok(ModelItem::Diffusion(
                self.parse_statement_block("diffusion")?,
            )),
            Some(TokenKind::Particles) => Ok(ModelItem::Particles(self.parse_particles_decl()?)),
            Some(other) => Err(ParseError::new(
                format!("unexpected token {} in model body", other.describe()),
                self.peek().unwrap().span,
            )),
            None => Err(ParseError::new(
                "unexpected end of input in model body",
                Span::empty(self.src_len),
            )),
        }
    }

    fn parse_parameters_block(&mut self) -> Result<ParametersBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut items = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            items.push(self.parse_ident()?);
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(ParametersBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_constants_block(&mut self) -> Result<ConstantsBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut items = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            items.push(self.parse_binding()?);
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(ConstantsBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_covariates_block(&mut self) -> Result<CovariatesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut items = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            let name = self.parse_ident()?;
            let interpolation = if self.take_if(|kind| matches!(kind, TokenKind::At)).is_some() {
                Some(self.parse_ident()?)
            } else {
                None
            };
            let span = if let Some(annotation) = &interpolation {
                name.span.join(annotation.span)
            } else {
                name.span
            };
            items.push(CovariateDecl {
                name,
                interpolation,
                span,
            });
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(CovariatesBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_states_block(&mut self) -> Result<StatesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut items = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            let name = self.parse_ident()?;
            let mut span = name.span;
            let size = if self
                .take_if(|kind| matches!(kind, TokenKind::LBracket))
                .is_some()
            {
                let expr = self.parse_expr(0)?;
                let close =
                    self.expect_simple(|kind| matches!(kind, TokenKind::RBracket), "`]`")?;
                span = span.join(close.span);
                Some(expr)
            } else {
                None
            };
            items.push(StateDecl { name, size, span });
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(StatesBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_routes_block(&mut self) -> Result<RoutesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut routes = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            routes.push(self.parse_route_decl()?);
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(RoutesBlock {
            routes,
            span: start.join(end.span),
        })
    }

    fn parse_route_decl(&mut self) -> Result<RouteDecl, ParseError> {
        let input = self.parse_ident()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::Arrow), "`->`")?;
        let destination = self.parse_place()?;
        let mut end_span = destination.span;
        let mut properties = Vec::new();
        if self
            .take_if(|kind| matches!(kind, TokenKind::LBrace))
            .is_some()
        {
            while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
                properties.push(self.parse_binding()?);
                self.consume_separators();
            }
            end_span = self
                .expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?
                .span;
        }
        Ok(RouteDecl {
            input: input.clone(),
            destination,
            properties,
            span: input.span.join(end_span),
        })
    }

    fn parse_analytical_block(&mut self) -> Result<AnalyticalBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;

        let kernel_name = self.parse_ident()?;
        if kernel_name.text != "kernel" {
            return Err(ParseError::new(
                format!(
                    "expected `kernel = <identifier>` inside analytical block, found `{}`",
                    kernel_name.text
                ),
                kernel_name.span,
            ));
        }

        self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let kernel = self.parse_ident()?;
        self.consume_separators();
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(AnalyticalBlock {
            kernel,
            span: start.join(end.span),
        })
    }

    fn parse_particles_decl(&mut self) -> Result<ParticlesDecl, ParseError> {
        let start = self.bump().unwrap().span;
        let value = self.parse_expr(0)?;
        Ok(ParticlesDecl {
            span: start.join(value.span),
            value,
        })
    }

    fn parse_statement_block(&mut self, name: &str) -> Result<StatementBlock, ParseError> {
        let start = self.bump().unwrap().span;
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut statements = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            statements.push(self.parse_stmt()?);
            self.consume_separators();
        }
        let end = self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        if statements.is_empty() {
            return Err(ParseError::new(
                format!("`{name}` block must contain at least one statement"),
                start.join(end.span),
            ));
        }
        Ok(StatementBlock {
            statements,
            span: start.join(end.span),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_kind() {
            Some(TokenKind::If) => self.parse_if_stmt(),
            Some(TokenKind::For) => self.parse_for_stmt(),
            Some(TokenKind::Let) => self.parse_let_stmt(),
            _ => self.parse_assign_stmt(),
        }
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.bump().unwrap().span;
        let condition = self.parse_expr(0)?;
        let then_branch = self.parse_stmt_body()?;
        let mut end_span = then_branch.last().map_or(condition.span, |stmt| stmt.span);
        let else_branch = if self
            .take_if(|kind| matches!(kind, TokenKind::Else))
            .is_some()
        {
            if self.at(|kind| matches!(kind, TokenKind::If)) {
                let nested = self.parse_if_stmt()?;
                end_span = nested.span;
                Some(vec![nested])
            } else {
                let branch = self.parse_stmt_body()?;
                end_span = branch.last().map_or(end_span, |stmt| stmt.span);
                Some(branch)
            }
        } else {
            None
        };
        Ok(Stmt {
            kind: StmtKind::If(IfStmt {
                condition,
                then_branch,
                else_branch,
            }),
            span: start.join(end_span),
        })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.bump().unwrap().span;
        let binding = self.parse_ident()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::In), "`in`")?;
        let range_start = self.parse_expr(0)?;
        self.expect_simple(|kind| matches!(kind, TokenKind::DotDot), "`..`")?;
        let range_end = self.parse_expr(0)?;
        let body = self.parse_stmt_body()?;
        let body_end = body.last().map_or(range_end.span, |stmt| stmt.span);
        Ok(Stmt {
            span: start.join(body_end),
            kind: StmtKind::For(ForStmt {
                binding,
                range: RangeExpr {
                    span: range_start.span.join(range_end.span),
                    start: range_start,
                    end: range_end,
                },
                body,
            }),
        })
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.bump().unwrap().span;
        let name = self.parse_ident()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_expr(0)?;
        Ok(Stmt {
            span: start.join(value.span),
            kind: StmtKind::Let(LetStmt { name, value }),
        })
    }

    fn parse_assign_stmt(&mut self) -> Result<Stmt, ParseError> {
        let target = self.parse_assign_target()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_expr(0)?;
        Ok(Stmt {
            span: target.span.join(value.span),
            kind: StmtKind::Assign(AssignStmt { target, value }),
        })
    }

    fn parse_stmt_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let mut statements = Vec::new();
        while !self.at(|kind| matches!(kind, TokenKind::RBrace)) {
            statements.push(self.parse_stmt()?);
            self.consume_separators();
        }
        self.expect_simple(|kind| matches!(kind, TokenKind::RBrace), "`}`")?;
        Ok(statements)
    }

    fn parse_binding(&mut self) -> Result<Binding, ParseError> {
        let name = self.parse_ident()?;
        self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_expr(0)?;
        Ok(Binding {
            span: name.span.join(value.span),
            name,
            value,
        })
    }

    fn parse_place(&mut self) -> Result<Place, ParseError> {
        let name = self.parse_ident()?;
        let mut span = name.span;
        let index = if self
            .take_if(|kind| matches!(kind, TokenKind::LBracket))
            .is_some()
        {
            let expr = self.parse_expr(0)?;
            let close = self.expect_simple(|kind| matches!(kind, TokenKind::RBracket), "`]`")?;
            span = span.join(close.span);
            Some(expr)
        } else {
            None
        };
        Ok(Place { name, index, span })
    }

    fn parse_assign_target(&mut self) -> Result<AssignTarget, ParseError> {
        let name = self.parse_ident()?;
        let mut span = name.span;
        let kind = if self
            .take_if(|kind| matches!(kind, TokenKind::LParen))
            .is_some()
        {
            let args = self.parse_expr_list(TokenKindMatcher::RPAREN)?;
            let close = self.expect_simple(|kind| matches!(kind, TokenKind::RParen), "`)`")?;
            span = span.join(close.span);
            AssignTargetKind::Call { callee: name, args }
        } else if self
            .take_if(|kind| matches!(kind, TokenKind::LBracket))
            .is_some()
        {
            let index = self.parse_expr(0)?;
            let close = self.expect_simple(|kind| matches!(kind, TokenKind::RBracket), "`]`")?;
            span = span.join(close.span);
            AssignTargetKind::Index {
                target: name,
                index,
            }
        } else {
            AssignTargetKind::Name(name)
        };
        Ok(AssignTarget { kind, span })
    }

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        let token = self
            .bump()
            .ok_or_else(|| ParseError::new("expected identifier", Span::empty(self.src_len)))?;
        match token.kind {
            TokenKind::Ident(name) => Ok(Ident::new(name, token.span)),
            other => Err(ParseError::new(
                format!("expected identifier, found {}", other.describe()),
                token.span,
            )),
        }
    }

    fn parse_expr(&mut self, min_precedence: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_prefix_expr()?;
        while let Some((op, precedence, right_assoc)) = self.peek_binary_op() {
            if precedence < min_precedence {
                break;
            }

            self.bump();
            let next_min = if right_assoc {
                precedence
            } else {
                precedence + 1
            };
            let rhs = self.parse_expr(next_min)?;
            let span = lhs.span.join(rhs.span);
            lhs = Expr {
                span,
                kind: ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_prefix_expr(&mut self) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            Some(TokenKind::Plus) => {
                let start = self.bump().unwrap().span;
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: start.join(expr.span),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Plus,
                        expr: Box::new(expr),
                    },
                })
            }
            Some(TokenKind::Minus) => {
                let start = self.bump().unwrap().span;
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: start.join(expr.span),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Minus,
                        expr: Box::new(expr),
                    },
                })
            }
            Some(TokenKind::Bang) => {
                let start = self.bump().unwrap().span;
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: start.join(expr.span),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(expr),
                    },
                })
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary_expr()?;
        loop {
            if self
                .take_if(|kind| matches!(kind, TokenKind::LBracket))
                .is_some()
            {
                let index = self.parse_expr(0)?;
                let close =
                    self.expect_simple(|kind| matches!(kind, TokenKind::RBracket), "`]`")?;
                let span = expr.span.join(close.span);
                expr = Expr {
                    span,
                    kind: ExprKind::Index {
                        target: Box::new(expr),
                        index: Box::new(index),
                    },
                };
                continue;
            }

            if self
                .take_if(|kind| matches!(kind, TokenKind::LParen))
                .is_some()
            {
                let callee = match expr.kind {
                    ExprKind::Name(name) => name,
                    _ => {
                        return Err(ParseError::new("only identifiers can be called", expr.span));
                    }
                };
                let args = self.parse_expr_list(TokenKindMatcher::RPAREN)?;
                let close = self.expect_simple(|kind| matches!(kind, TokenKind::RParen), "`)`")?;
                let span = callee.span.join(close.span);
                expr = Expr {
                    span,
                    kind: ExprKind::Call { callee, args },
                };
                continue;
            }

            break;
        }
        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        let token = self
            .bump()
            .ok_or_else(|| ParseError::new("expected expression", Span::empty(self.src_len)))?;
        match token.kind {
            TokenKind::Number(value) => Ok(Expr {
                kind: ExprKind::Number(value),
                span: token.span,
            }),
            TokenKind::True => Ok(Expr {
                kind: ExprKind::Bool(true),
                span: token.span,
            }),
            TokenKind::False => Ok(Expr {
                kind: ExprKind::Bool(false),
                span: token.span,
            }),
            TokenKind::Ident(name) => Ok(Expr {
                kind: ExprKind::Name(Ident::new(name, token.span)),
                span: token.span,
            }),
            TokenKind::LParen => {
                let expr = self.parse_expr(0)?;
                let close = self.expect_simple(|kind| matches!(kind, TokenKind::RParen), "`)`")?;
                Ok(Expr {
                    span: token.span.join(close.span),
                    kind: expr.kind,
                })
            }
            other => Err(ParseError::new(
                format!("expected expression, found {}", other.describe()),
                token.span,
            )),
        }
    }

    fn parse_expr_list(&mut self, terminator: TokenKindMatcher) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if self.at(terminator.matches) {
            return Ok(args);
        }

        loop {
            args.push(self.parse_expr(0)?);
            if self
                .take_if(|kind| matches!(kind, TokenKind::Comma))
                .is_some()
            {
                continue;
            }
            break;
        }
        Ok(args)
    }

    fn peek_binary_op(&self) -> Option<(BinaryOp, u8, bool)> {
        let kind = self.peek_kind()?;
        let op = match kind {
            TokenKind::OrOr => (BinaryOp::Or, 1, false),
            TokenKind::AndAnd => (BinaryOp::And, 2, false),
            TokenKind::EqEq => (BinaryOp::Eq, 3, false),
            TokenKind::BangEq => (BinaryOp::NotEq, 3, false),
            TokenKind::Lt => (BinaryOp::Lt, 4, false),
            TokenKind::LtEq => (BinaryOp::LtEq, 4, false),
            TokenKind::Gt => (BinaryOp::Gt, 4, false),
            TokenKind::GtEq => (BinaryOp::GtEq, 4, false),
            TokenKind::Plus => (BinaryOp::Add, 5, false),
            TokenKind::Minus => (BinaryOp::Sub, 5, false),
            TokenKind::Star => (BinaryOp::Mul, 6, false),
            TokenKind::Slash => (BinaryOp::Div, 6, false),
            TokenKind::Caret => (BinaryOp::Pow, 7, true),
            _ => return None,
        };
        Some(op)
    }

    fn at<F>(&self, predicate: F) -> bool
    where
        F: FnOnce(&TokenKind) -> bool,
    {
        self.peek_kind().is_some_and(predicate)
    }

    fn take_if<F>(&mut self, predicate: F) -> Option<Token>
    where
        F: FnOnce(&TokenKind) -> bool,
    {
        if self.at(predicate) {
            self.bump()
        } else {
            None
        }
    }

    fn expect_simple<F>(&mut self, predicate: F, expected: &str) -> Result<Token, ParseError>
    where
        F: FnOnce(&TokenKind) -> bool,
    {
        let token = self.bump().ok_or_else(|| {
            ParseError::new(
                format!("expected {expected}, found end of input"),
                Span::empty(self.src_len),
            )
        })?;
        if predicate(&token.kind) {
            Ok(token)
        } else {
            Err(ParseError::new(
                format!("expected {expected}, found {}", token.kind.describe()),
                token.span,
            ))
        }
    }

    fn consume_separators(&mut self) {
        while self
            .take_if(|kind| matches!(kind, TokenKind::Comma | TokenKind::Semi))
            .is_some()
        {}
    }

    fn is_eof(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn peek_kind(&self) -> Option<&TokenKind> {
        self.peek().map(|token| &token.kind)
    }

    fn peek_span(&self) -> Option<Span> {
        self.peek().map(|token| token.span)
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.index).cloned();
        if token.is_some() {
            self.index += 1;
        }
        token
    }
}

#[derive(Clone, Copy)]
struct TokenKindMatcher {
    matches: fn(&TokenKind) -> bool,
}

impl TokenKindMatcher {
    const RPAREN: Self = Self {
        matches: |kind| matches!(kind, TokenKind::RParen),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proposal_two_corpus() {
        let src = include_str!("../../dsl-proposals/02-structured-block-imperative.dsl");
        let module = parse_module(src).expect("proposal 2 fixture parses");
        assert_eq!(module.models.len(), 4);
        assert_eq!(module.models[0].name.text, "one_cmt_oral_iv");
        assert_eq!(module.models[0].kind, ModelKind::Ode);
        assert!(module.models[0]
            .items
            .iter()
            .any(|item| matches!(item, ModelItem::Derive(block) if block.statements.iter().any(|stmt| matches!(stmt.kind, StmtKind::If(_))))));
        assert!(module.models[1]
            .items
            .iter()
            .any(|item| matches!(item, ModelItem::Dynamics(block) if block.statements.iter().any(|stmt| matches!(stmt.kind, StmtKind::For(_))))));
        assert!(module.models[3]
            .items
            .iter()
            .any(|item| matches!(item, ModelItem::Particles(_))));
        assert!(module.models[3]
            .items
            .iter()
            .any(|item| matches!(item, ModelItem::Diffusion(_))));
    }

    #[test]
    fn round_trips_proposal_two_corpus() {
        let src = include_str!("../../dsl-proposals/02-structured-block-imperative.dsl");
        let parsed = parse_module(src).expect("proposal 2 fixture parses");
        let formatted = parsed.to_string();
        let reparsed = parse_module(&formatted).expect("formatted DSL reparses");
        assert_eq!(formatted, reparsed.to_string());
    }

    #[test]
    fn desugars_authoring_fixture_into_canonical_ast() {
        let src = include_str!("../../dsl-proposals/04-user-recommended_style.dsi");
        let expected = include_str!("../../dsl-proposals/04-user-recommended_style.desugared.dsl");

        let parsed = parse_module(src).expect("proposal 4 fixture parses");
        let expected = parse_module(expected).expect("canonical fixture parses");

        assert_eq!(parsed.to_string(), expected.to_string());
    }

    #[test]
    fn reports_authoring_expression_error_at_original_line() {
        let src = r#"states = gut
dx(gut) = 1 +
out(cp) = gut ~ continuous()
"#;

        let err = parse_module(src).expect_err("invalid proposal 4 expression should fail");
        let rendered = err.render(src);
        assert!(rendered.contains("line 2"), "{}", rendered);
    }

    #[test]
    fn reports_route_syntax_error() {
        let src = "model broken { kind ode routes { oral depot } }";
        let err = parse_module(src).expect_err("missing route arrow should fail");
        let rendered = err.render(src);
        assert!(rendered.contains("expected `->`"), "{}", rendered);
    }

    #[test]
    fn reports_missing_kind() {
        let src = "model missing { parameters { ka } }";
        let err = parse_model(src).expect_err("missing kind should fail");
        let rendered = err.render(src);
        assert!(
            rendered.contains("missing a `kind` declaration"),
            "{}",
            rendered
        );
    }
}
