use super::ast::*;
use super::authoring;
use super::diagnostic::{ParseError, Span};
use super::lexer::{lex, Token, TokenKind};

pub fn parse_module(src: &str) -> Result<Module, ParseError> {
    let parsed = (|| {
        let leading = strip_leading_layout(src);
        if let Some(rest) = leading.strip_prefix("model") {
            let trimmed = rest.trim_start();
            if trimmed.starts_with('=') {
                return authoring::parse_module(src);
            }
            return Parser::new(src)?.parse_module();
        }
        if leading.is_empty() {
            Parser::new(src)?.parse_module()
        } else {
            authoring::parse_module(src)
        }
    })();

    parsed.map_err(|error| error.with_source(src))
}

pub fn parse_model(src: &str) -> Result<Model, ParseError> {
    let module = parse_module(src)?;
    match module.models.len() {
        1 => Ok(module.models.into_iter().next().unwrap()),
        0 => Err(
            ParseError::new("expected a `model` declaration", Span::empty(src.len()))
                .with_source(src),
        ),
        _ => Err(ParseError::new(
            "expected exactly one `model` declaration",
            module.models[1].span,
        )
        .with_source(src)),
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
    let parsed = (|| {
        let mut parser = Parser::new(src)?;
        let expr = parser.parse_expr(0)?;
        if let Some(token) = parser.peek() {
            Err(ParseError::new(
                format!(
                    "expected end of expression, found {}",
                    token.kind.describe()
                ),
                token.span,
            ))
        } else {
            Ok(expr)
        }
    })();

    parsed.map_err(|error| error.with_source(src))
}

pub(crate) fn parse_place_fragment(src: &str) -> Result<Place, ParseError> {
    let parsed = (|| {
        let mut parser = Parser::new(src)?;
        let place = parser.parse_place()?;
        if let Some(token) = parser.peek() {
            Err(ParseError::new(
                format!("expected end of place, found {}", token.kind.describe()),
                token.span,
            ))
        } else {
            Ok(place)
        }
    })();

    parsed.map_err(|error| error.with_source(src))
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    src_len: usize,
    layout_boundaries: Vec<LayoutBoundary>,
}

#[derive(Clone, Copy)]
enum LayoutBoundary {
    ModelItem,
    Statement,
    Binding,
    IdentItem,
    RouteDecl,
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
            layout_boundaries: Vec::new(),
        }
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let start = self.peek_span().unwrap_or_else(|| Span::empty(0));
        let mut models = Vec::new();
        let mut errors = Vec::new();
        while !self.is_eof() {
            match self.parse_model_decl() {
                Ok(model) => models.push(model),
                Err(error) => {
                    errors.extend(error.diagnostics().iter().cloned());
                    self.sync_to_top_level_boundary();
                    if self.is_eof() {
                        break;
                    }
                }
            }
        }
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
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
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;

        let (kind, items, mut errors) =
            self.with_layout_boundary(LayoutBoundary::ModelItem, |parser| {
                let mut kind = None;
                let mut items = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    if parser.at(|kind| matches!(kind, TokenKind::Kind)) {
                        let kind_kw = parser.bump().unwrap();
                        if kind.is_some() {
                            errors.push(
                                ParseError::new(
                                    "duplicate `kind` declaration in model body",
                                    kind_kw.span,
                                )
                                .into_diagnostic(),
                            );
                            parser.sync_to_model_body_boundary();
                            continue;
                        }
                        match parser.parse_model_kind(kind_kw) {
                            Ok(model_kind) => kind = Some(model_kind),
                            Err(error) => {
                                errors.extend(error.diagnostics().iter().cloned());
                                parser.sync_to_model_body_boundary();
                            }
                        }
                    } else {
                        match parser.parse_model_item() {
                            Ok(item) => items.push(item),
                            Err(error) => {
                                errors.extend(error.diagnostics().iter().cloned());
                                parser.sync_to_model_body_boundary();
                            }
                        }
                    }
                }
                Ok((kind, items, errors))
            })?;

        let close = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            format!("model `{}` body", name.text),
        ) {
            Ok(close) => close,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        let kind = match kind {
            Some(kind) => kind,
            None => {
                errors.push(
                    ParseError::new(
                        format!("model `{}` is missing a `kind` declaration", name.text),
                        name.span,
                    )
                    .into_diagnostic(),
                );
                return Err(ParseError::from_diagnostics(errors));
            }
        };

        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }

        Ok(Model {
            name,
            kind,
            items,
            span: model_kw.span.join(close.span),
        })
    }

    fn parse_model_kind(&mut self, kind_kw: Token) -> Result<ModelKind, ParseError> {
        self.ensure_not_layout_boundary(
            kind_kw.span,
            "expected a model kind after `kind`",
            "place the model kind on the same line as `kind` or start the next model item after the kind declaration",
        )?;
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
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (items, mut errors) =
            self.with_layout_boundary(LayoutBoundary::IdentItem, |parser| {
                let mut items = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match parser.parse_ident() {
                        Ok(item) => items.push(item),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_layout_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((items, errors))
            })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`parameters` block",
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(ParametersBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_constants_block(&mut self) -> Result<ConstantsBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (items, mut errors) = self.with_layout_boundary(LayoutBoundary::Binding, |parser| {
            let mut items = Vec::new();
            let mut errors = Vec::new();
            while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                match parser.parse_binding() {
                    Ok(item) => items.push(item),
                    Err(error) => {
                        errors.extend(error.diagnostics().iter().cloned());
                        parser.sync_to_layout_boundary();
                        if parser.is_eof() {
                            break;
                        }
                    }
                }
                parser.consume_separators();
            }
            Ok((items, errors))
        })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`constants` block",
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(ConstantsBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_covariates_block(&mut self) -> Result<CovariatesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (items, mut errors) =
            self.with_layout_boundary(LayoutBoundary::IdentItem, |parser| {
                let mut items = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match (|| -> Result<CovariateDecl, ParseError> {
                        let name = parser.parse_ident()?;
                        let interpolation = if let Some(at) =
                            parser.take_if(|kind| matches!(kind, TokenKind::At))
                        {
                            Some(
                                parser.parse_continuation_ident_after(
                                    &at,
                                    "interpolation identifier",
                                )?,
                            )
                        } else {
                            None
                        };
                        let span = if let Some(annotation) = &interpolation {
                            name.span.join(annotation.span)
                        } else {
                            name.span
                        };
                        Ok(CovariateDecl {
                            name,
                            interpolation,
                            span,
                        })
                    })() {
                        Ok(item) => items.push(item),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_layout_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((items, errors))
            })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`covariates` block",
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(CovariatesBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_states_block(&mut self) -> Result<StatesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (items, mut errors) =
            self.with_layout_boundary(LayoutBoundary::IdentItem, |parser| {
                let mut items = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match (|| -> Result<StateDecl, ParseError> {
                        let name = parser.parse_ident()?;
                        let mut span = name.span;
                        let size = if let Some(open_bracket) =
                            parser.take_if(|kind| matches!(kind, TokenKind::LBracket))
                        {
                            let expr = parser.parse_continuation_expr_after(&open_bracket)?;
                            let close = parser.expect_closing(
                                |kind| matches!(kind, TokenKind::RBracket),
                                "`]`",
                                open_bracket.span,
                                format!("state size for `{}`", name.text),
                            )?;
                            span = span.join(close.span);
                            Some(expr)
                        } else {
                            None
                        };
                        Ok(StateDecl { name, size, span })
                    })() {
                        Ok(item) => items.push(item),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_layout_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((items, errors))
            })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`states` block",
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(StatesBlock {
            items,
            span: start.join(end.span),
        })
    }

    fn parse_routes_block(&mut self) -> Result<RoutesBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (routes, mut errors) =
            self.with_layout_boundary(LayoutBoundary::RouteDecl, |parser| {
                let mut routes = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match parser.parse_route_decl() {
                        Ok(route) => routes.push(route),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_layout_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((routes, errors))
            })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`routes` block",
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(RoutesBlock {
            routes,
            span: start.join(end.span),
        })
    }

    fn parse_route_decl(&mut self) -> Result<RouteDecl, ParseError> {
        let input = self.parse_ident()?;
        let arrow = self.expect_simple(|kind| matches!(kind, TokenKind::Arrow), "`->`")?;
        self.ensure_not_layout_boundary(
            arrow.span,
            "expected route destination after `->`",
            "place the destination on the same line as the route arrow or start the next route after this declaration",
        )?;
        let destination = self.parse_place()?;
        let mut end_span = destination.span;
        let mut properties = Vec::new();
        if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LBrace)) {
            let (parsed_properties, mut errors) =
                self.with_layout_boundary(LayoutBoundary::Binding, |parser| {
                    let mut properties = Vec::new();
                    let mut errors = Vec::new();
                    while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                        match parser.parse_binding() {
                            Ok(property) => properties.push(property),
                            Err(error) => {
                                errors.extend(error.diagnostics().iter().cloned());
                                parser.sync_to_layout_boundary();
                                if parser.is_eof() {
                                    break;
                                }
                            }
                        }
                        parser.consume_separators();
                    }
                    Ok((properties, errors))
                })?;
            properties = parsed_properties;
            end_span = match self.expect_closing(
                |kind| matches!(kind, TokenKind::RBrace),
                "`}`",
                open.span,
                format!("property block for route `{}`", input.text),
            ) {
                Ok(end) => end.span,
                Err(error) => {
                    errors.extend(error.diagnostics().iter().cloned());
                    return Err(ParseError::from_diagnostics(errors));
                }
            };
            if !errors.is_empty() {
                return Err(ParseError::from_diagnostics(errors));
            }
        }
        Ok(RouteDecl {
            input: input.clone(),
            destination,
            kind: None,
            properties,
            span: input.span.join(end_span),
        })
    }

    fn parse_analytical_block(&mut self) -> Result<AnalyticalBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;

        let structure_name = self.parse_ident()?;
        if structure_name.text != "structure" {
            return Err(ParseError::new(
                format!(
                    "expected `structure = <identifier>` inside analytical block, found `{}`",
                    structure_name.text
                ),
                structure_name.span,
            ));
        }

        let eq = self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let structure = self.parse_continuation_ident_after(&eq, "structure identifier")?;
        self.consume_separators();
        let end = self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "`analytical` block",
        )?;
        Ok(AnalyticalBlock {
            structure,
            span: start.join(end.span),
        })
    }

    fn parse_particles_decl(&mut self) -> Result<ParticlesDecl, ParseError> {
        let start = self.bump().unwrap();
        let value = self.parse_continuation_expr_after(&start)?;
        Ok(ParticlesDecl {
            span: start.span.join(value.span),
            value,
        })
    }

    fn parse_statement_block(&mut self, name: &str) -> Result<StatementBlock, ParseError> {
        let start = self.bump().unwrap().span;
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (statements, mut errors) =
            self.with_layout_boundary(LayoutBoundary::Statement, |parser| {
                let mut statements = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match parser.parse_stmt() {
                        Ok(statement) => statements.push(statement),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_statement_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((statements, errors))
            })?;
        let end = match self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            format!("`{name}` block"),
        ) {
            Ok(end) => end,
            Err(error) => {
                errors.extend(error.diagnostics().iter().cloned());
                return Err(ParseError::from_diagnostics(errors));
            }
        };
        if statements.is_empty() && errors.is_empty() {
            errors.push(
                ParseError::new(
                    format!("`{name}` block must contain at least one statement"),
                    start.join(end.span),
                )
                .into_diagnostic(),
            );
        }
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
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
        let in_token = self.expect_simple(|kind| matches!(kind, TokenKind::In), "`in`")?;
        let range_start = self.parse_continuation_expr_after(&in_token)?;
        let dotdot = self.expect_simple(|kind| matches!(kind, TokenKind::DotDot), "`..`")?;
        let range_end = self.parse_continuation_expr_after(&dotdot)?;
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
        let eq = self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_continuation_expr_after(&eq)?;
        Ok(Stmt {
            span: start.join(value.span),
            kind: StmtKind::Let(LetStmt { name, value }),
        })
    }

    fn parse_assign_stmt(&mut self) -> Result<Stmt, ParseError> {
        let target = self.parse_assign_target()?;
        let eq = self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_continuation_expr_after(&eq)?;
        Ok(Stmt {
            span: target.span.join(value.span),
            kind: StmtKind::Assign(AssignStmt { target, value }),
        })
    }

    fn parse_stmt_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let open = self.expect_simple(|kind| matches!(kind, TokenKind::LBrace), "`{`")?;
        let (statements, mut errors) =
            self.with_layout_boundary(LayoutBoundary::Statement, |parser| {
                let mut statements = Vec::new();
                let mut errors = Vec::new();
                while !parser.is_eof() && !parser.at(|kind| matches!(kind, TokenKind::RBrace)) {
                    match parser.parse_stmt() {
                        Ok(statement) => statements.push(statement),
                        Err(error) => {
                            errors.extend(error.diagnostics().iter().cloned());
                            parser.sync_to_statement_boundary();
                            if parser.is_eof() {
                                break;
                            }
                        }
                    }
                    parser.consume_separators();
                }
                Ok((statements, errors))
            })?;
        if let Err(error) = self.expect_closing(
            |kind| matches!(kind, TokenKind::RBrace),
            "`}`",
            open.span,
            "statement body",
        ) {
            errors.extend(error.diagnostics().iter().cloned());
        }
        if !errors.is_empty() {
            return Err(ParseError::from_diagnostics(errors));
        }
        Ok(statements)
    }

    fn parse_binding(&mut self) -> Result<Binding, ParseError> {
        let name = self.parse_ident()?;
        let eq = self.expect_simple(|kind| matches!(kind, TokenKind::Eq), "`=`")?;
        let value = self.parse_continuation_expr_after(&eq)?;
        Ok(Binding {
            span: name.span.join(value.span),
            name,
            value,
        })
    }

    fn parse_place(&mut self) -> Result<Place, ParseError> {
        let name = self.parse_ident()?;
        let mut span = name.span;
        let index = if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LBracket)) {
            let expr = self.parse_continuation_expr_after(&open)?;
            let close = self.expect_closing(
                |kind| matches!(kind, TokenKind::RBracket),
                "`]`",
                open.span,
                format!("index for `{}`", name.text),
            )?;
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
        let kind = if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LParen)) {
            let args = self.parse_expr_list(&open, TokenKindMatcher::RPAREN)?;
            let close = self.expect_closing(
                |kind| matches!(kind, TokenKind::RParen),
                "`)`",
                open.span,
                format!("argument list for `{}`", name.text),
            )?;
            span = span.join(close.span);
            AssignTargetKind::Call { callee: name, args }
        } else if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LBracket)) {
            let index = self.parse_continuation_expr_after(&open)?;
            let close = self.expect_closing(
                |kind| matches!(kind, TokenKind::RBracket),
                "`]`",
                open.span,
                format!("index assignment for `{}`", name.text),
            )?;
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

            let operator = self.bump().expect("peeked binary operator");
            let next_min = if right_assoc {
                precedence
            } else {
                precedence + 1
            };
            if self.current_layout_boundary_starts_here() {
                return Err(self.missing_expression_after_token(&operator));
            }
            if !self.peek_kind().is_some_and(Self::starts_expr) {
                return Err(self.missing_expression_after_token(&operator));
            }
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
                let operator = self.bump().unwrap();
                if self.current_layout_boundary_starts_here() {
                    return Err(self.missing_expression_after_token(&operator));
                }
                if !self.peek_kind().is_some_and(Self::starts_expr) {
                    return Err(self.missing_expression_after_token(&operator));
                }
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: operator.span.join(expr.span),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Plus,
                        expr: Box::new(expr),
                    },
                })
            }
            Some(TokenKind::Minus) => {
                let operator = self.bump().unwrap();
                if self.current_layout_boundary_starts_here() {
                    return Err(self.missing_expression_after_token(&operator));
                }
                if !self.peek_kind().is_some_and(Self::starts_expr) {
                    return Err(self.missing_expression_after_token(&operator));
                }
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: operator.span.join(expr.span),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Minus,
                        expr: Box::new(expr),
                    },
                })
            }
            Some(TokenKind::Bang) => {
                let operator = self.bump().unwrap();
                if self.current_layout_boundary_starts_here() {
                    return Err(self.missing_expression_after_token(&operator));
                }
                if !self.peek_kind().is_some_and(Self::starts_expr) {
                    return Err(self.missing_expression_after_token(&operator));
                }
                let expr = self.parse_prefix_expr()?;
                Ok(Expr {
                    span: operator.span.join(expr.span),
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
            if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LBracket)) {
                let index = self.parse_continuation_expr_after(&open)?;
                let close = self.expect_closing(
                    |kind| matches!(kind, TokenKind::RBracket),
                    "`]`",
                    open.span,
                    "index expression",
                )?;
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

            if let Some(open) = self.take_if(|kind| matches!(kind, TokenKind::LParen)) {
                let callee = match expr.kind {
                    ExprKind::Name(name) => name,
                    _ => {
                        return Err(ParseError::new("only identifiers can be called", expr.span));
                    }
                };
                let args = self.parse_expr_list(&open, TokenKindMatcher::RPAREN)?;
                let close = self.expect_closing(
                    |kind| matches!(kind, TokenKind::RParen),
                    "`)`",
                    open.span,
                    format!("argument list for `{}`", callee.text),
                )?;
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
                if self.at(|kind| matches!(kind, TokenKind::RParen)) {
                    return Err(ParseError::new(
                        "expected expression inside `(` ... `)`",
                        token.span,
                    )
                    .with_context_label(token.span, "`(` group starts here")
                    .with_help("add an expression between `(` and `)`"));
                }
                let expr = self.parse_continuation_expr_after(&token)?;
                let close = self.expect_closing(
                    |kind| matches!(kind, TokenKind::RParen),
                    "`)`",
                    token.span,
                    "`(` group",
                )?;
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

    fn parse_expr_list(
        &mut self,
        open: &Token,
        terminator: TokenKindMatcher,
    ) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if self.at(terminator.matches) {
            return Ok(args);
        }

        let mut leader = open.clone();
        loop {
            args.push(self.parse_continuation_expr_after(&leader)?);
            if let Some(comma) = self.take_if(|kind| matches!(kind, TokenKind::Comma)) {
                leader = comma;
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

    fn expect_closing<F>(
        &mut self,
        predicate: F,
        expected: &str,
        open_span: Span,
        context: impl Into<String>,
    ) -> Result<Token, ParseError>
    where
        F: FnOnce(&TokenKind) -> bool,
    {
        let context = context.into();
        let token = self.bump().ok_or_else(|| {
            ParseError::new(
                format!("expected {expected}, found end of input"),
                Span::empty(self.src_len),
            )
            .with_context_label(open_span, format!("{context} opened here"))
            .with_help(format!("insert {expected} to close {context}"))
        })?;
        if predicate(&token.kind) {
            Ok(token)
        } else {
            Err(ParseError::new(
                format!("expected {expected}, found {}", token.kind.describe()),
                token.span,
            )
            .with_context_label(open_span, format!("{context} opened here"))
            .with_help(format!("close {context} with {expected}")))
        }
    }

    fn parse_continuation_expr_after(&mut self, anchor: &Token) -> Result<Expr, ParseError> {
        self.ensure_not_layout_boundary(
            anchor.span,
            format!("expected expression after {}", anchor.kind.describe()),
            format!(
                "continue the expression after {} on the same line or add a separator before the next {}",
                anchor.kind.describe(),
                self.current_boundary_subject()
            ),
        )?;
        self.parse_expr(0)
    }

    fn parse_continuation_ident_after(
        &mut self,
        anchor: &Token,
        expected: &str,
    ) -> Result<Ident, ParseError> {
        self.ensure_not_layout_boundary(
            anchor.span,
            format!("expected {expected} after {}", anchor.kind.describe()),
            format!(
                "place the {expected} on the same line as {} or separate the next {} first",
                anchor.kind.describe(),
                self.current_boundary_subject()
            ),
        )?;
        self.parse_ident()
    }

    fn ensure_not_layout_boundary(
        &self,
        anchor: Span,
        message: impl Into<String>,
        help: impl Into<String>,
    ) -> Result<(), ParseError> {
        if !self.current_layout_boundary_starts_here() {
            return Ok(());
        }
        let mut error = ParseError::new(message, anchor);
        if let Some(token) = self.peek() {
            error = error.with_context_label(token.span, self.current_boundary_label());
        }
        Err(error.with_help(help))
    }

    fn missing_expression_after_token(&self, operator: &Token) -> ParseError {
        let mut error = ParseError::new(
            format!("expected expression after {}", operator.kind.describe()),
            operator.span,
        );
        if self.current_layout_boundary_starts_here() {
            if let Some(token) = self.peek() {
                error = error.with_context_label(token.span, self.current_boundary_label());
            }
            error.with_help(format!(
                "continue the expression after {} on the same line or add a separator before the next {}",
                operator.kind.describe(),
                self.current_boundary_subject()
            ))
        } else {
            error.with_help(format!(
                "add an expression after {}",
                operator.kind.describe()
            ))
        }
    }

    fn consume_separators(&mut self) {
        while self
            .take_if(|kind| matches!(kind, TokenKind::Comma | TokenKind::Semi))
            .is_some()
        {}
    }

    fn with_layout_boundary<T>(
        &mut self,
        boundary: LayoutBoundary,
        parse: impl FnOnce(&mut Self) -> Result<T, ParseError>,
    ) -> Result<T, ParseError> {
        self.layout_boundaries.push(boundary);
        let result = parse(self);
        self.layout_boundaries.pop();
        result
    }

    fn current_layout_boundary(&self) -> Option<LayoutBoundary> {
        self.layout_boundaries.last().copied()
    }

    fn current_layout_boundary_starts_here(&self) -> bool {
        let Some(boundary) = self.current_layout_boundary() else {
            return false;
        };
        let Some(token) = self.peek() else {
            return false;
        };
        token.starts_line && self.matches_layout_boundary(boundary, self.index)
    }

    fn matches_layout_boundary(&self, boundary: LayoutBoundary, index: usize) -> bool {
        let Some(token) = self.tokens.get(index) else {
            return false;
        };
        match boundary {
            LayoutBoundary::ModelItem => matches!(
                token.kind,
                TokenKind::Kind
                    | TokenKind::Parameters
                    | TokenKind::Constants
                    | TokenKind::Covariates
                    | TokenKind::States
                    | TokenKind::Routes
                    | TokenKind::Derive
                    | TokenKind::Dynamics
                    | TokenKind::Outputs
                    | TokenKind::Analytical
                    | TokenKind::Init
                    | TokenKind::Drift
                    | TokenKind::Diffusion
                    | TokenKind::Particles
            ),
            LayoutBoundary::Statement => match &token.kind {
                TokenKind::If | TokenKind::For | TokenKind::Let => true,
                TokenKind::Ident(_) => self.line_starts_assignment_target(index),
                _ => false,
            },
            LayoutBoundary::Binding => self.line_starts_named_assignment(index),
            LayoutBoundary::IdentItem => matches!(token.kind, TokenKind::Ident(_)),
            LayoutBoundary::RouteDecl => self.line_starts_route_decl(index),
        }
    }

    fn line_starts_named_assignment(&self, index: usize) -> bool {
        matches!(
            self.tokens.get(index).map(|token| &token.kind),
            Some(TokenKind::Ident(_))
        ) && self
            .next_same_line_index(index)
            .is_some_and(|next| matches!(self.tokens[next].kind, TokenKind::Eq))
    }

    fn line_starts_route_decl(&self, index: usize) -> bool {
        matches!(
            self.tokens.get(index).map(|token| &token.kind),
            Some(TokenKind::Ident(_))
        ) && self
            .next_same_line_index(index)
            .is_some_and(|next| matches!(self.tokens[next].kind, TokenKind::Arrow))
    }

    fn line_starts_assignment_target(&self, index: usize) -> bool {
        if !matches!(
            self.tokens.get(index).map(|token| &token.kind),
            Some(TokenKind::Ident(_))
        ) {
            return false;
        }

        let Some(next) = self.next_same_line_index(index) else {
            return false;
        };

        match self.tokens[next].kind {
            TokenKind::Eq => true,
            TokenKind::LParen => self
                .find_matching_same_line(next, TokenKindMatcher::LPAREN, TokenKindMatcher::RPAREN)
                .and_then(|close| self.next_same_line_index(close))
                .is_some_and(|after| matches!(self.tokens[after].kind, TokenKind::Eq)),
            TokenKind::LBracket => self
                .find_matching_same_line(
                    next,
                    TokenKindMatcher::LBRACKET,
                    TokenKindMatcher::RBRACKET,
                )
                .and_then(|close| self.next_same_line_index(close))
                .is_some_and(|after| matches!(self.tokens[after].kind, TokenKind::Eq)),
            _ => false,
        }
    }

    fn next_same_line_index(&self, index: usize) -> Option<usize> {
        let next = index + 1;
        let token = self.tokens.get(next)?;
        (!token.starts_line).then_some(next)
    }

    fn find_matching_same_line(
        &self,
        start: usize,
        open: TokenKindMatcher,
        close: TokenKindMatcher,
    ) -> Option<usize> {
        let mut depth = 0usize;
        let mut index = start;
        while let Some(token) = self.tokens.get(index) {
            if index > start && token.starts_line {
                return None;
            }
            if (open.matches)(&token.kind) {
                depth += 1;
            } else if (close.matches)(&token.kind) {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            index += 1;
        }
        None
    }

    fn current_boundary_label(&self) -> &'static str {
        match self.current_layout_boundary() {
            Some(LayoutBoundary::ModelItem) => "next model item starts here",
            Some(LayoutBoundary::Statement) => "next statement starts here",
            Some(LayoutBoundary::Binding) => "next binding starts here",
            Some(LayoutBoundary::IdentItem) => "next declaration starts here",
            Some(LayoutBoundary::RouteDecl) => "next route starts here",
            None => "next item starts here",
        }
    }

    fn current_boundary_subject(&self) -> &'static str {
        match self.current_layout_boundary() {
            Some(LayoutBoundary::ModelItem) => "model item",
            Some(LayoutBoundary::Statement) => "statement",
            Some(LayoutBoundary::Binding) => "binding",
            Some(LayoutBoundary::IdentItem) => "declaration",
            Some(LayoutBoundary::RouteDecl) => "route",
            None => "item",
        }
    }

    fn sync_to_top_level_boundary(&mut self) {
        while let Some(kind) = self.peek_kind() {
            if matches!(kind, TokenKind::Model) {
                break;
            }
            self.bump();
        }
    }

    fn sync_to_model_body_boundary(&mut self) {
        while let Some(kind) = self.peek_kind() {
            if matches!(kind, TokenKind::RBrace)
                || matches!(
                    kind,
                    TokenKind::Kind
                        | TokenKind::Parameters
                        | TokenKind::Constants
                        | TokenKind::Covariates
                        | TokenKind::States
                        | TokenKind::Routes
                        | TokenKind::Derive
                        | TokenKind::Dynamics
                        | TokenKind::Outputs
                        | TokenKind::Analytical
                        | TokenKind::Init
                        | TokenKind::Drift
                        | TokenKind::Diffusion
                        | TokenKind::Particles
                )
            {
                break;
            }
            if matches!(kind, TokenKind::Comma | TokenKind::Semi) {
                self.consume_separators();
                break;
            }
            self.bump();
        }
        self.consume_separators();
    }

    fn sync_to_statement_boundary(&mut self) {
        while let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::RBrace) {
                break;
            }
            if matches!(token.kind, TokenKind::Comma | TokenKind::Semi) {
                self.consume_separators();
                break;
            }
            if self.current_layout_boundary_starts_here() {
                break;
            }
            self.bump();
        }
        self.consume_separators();
    }

    fn sync_to_layout_boundary(&mut self) {
        while let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::RBrace) {
                break;
            }
            if matches!(token.kind, TokenKind::Comma | TokenKind::Semi) {
                self.consume_separators();
                break;
            }
            if self.current_layout_boundary_starts_here() {
                break;
            }
            self.bump();
        }
        self.consume_separators();
    }

    fn starts_expr(kind: &TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Number(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Ident(_)
                | TokenKind::LParen
                | TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Bang
        )
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
    const LPAREN: Self = Self {
        matches: |kind| matches!(kind, TokenKind::LParen),
    };

    const RPAREN: Self = Self {
        matches: |kind| matches!(kind, TokenKind::RParen),
    };

    const LBRACKET: Self = Self {
        matches: |kind| matches!(kind, TokenKind::LBracket),
    };

    const RBRACKET: Self = Self {
        matches: |kind| matches!(kind, TokenKind::RBracket),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        RECOMMENDED_STYLE_AUTHORING, RECOMMENDED_STYLE_CANONICAL, STRUCTURED_BLOCK_CORPUS,
    };

    #[test]
    fn parses_structured_block_corpus() {
        let src = STRUCTURED_BLOCK_CORPUS;
        let module = parse_module(src).expect("structured-block fixture parses");
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
    fn round_trips_structured_block_corpus() {
        let src = STRUCTURED_BLOCK_CORPUS;
        let parsed = parse_module(src).expect("structured-block fixture parses");
        let formatted = parsed.to_string();
        let reparsed = parse_module(&formatted).expect("formatted DSL reparses");
        assert_eq!(formatted, reparsed.to_string());
    }

    #[test]
    fn desugars_authoring_fixture_into_canonical_ast() {
        let src = RECOMMENDED_STYLE_AUTHORING;
        let expected = RECOMMENDED_STYLE_CANONICAL;

        let parsed = parse_module(src).expect("recommended-style fixture parses");
        let expected = parse_module(expected).expect("canonical fixture parses");

        assert_eq!(parsed.to_string(), expected.to_string());
    }

    #[test]
    fn reports_authoring_expression_error_at_original_line() {
        let src = r#"states = gut
dx(gut) = 1 +
out(cp) = gut ~ continuous()
"#;

        let err = parse_module(src).expect_err("invalid recommended-style expression should fail");
        let rendered = err.render(src);
        assert!(rendered.contains("line 2"), "{}", rendered);
    }

    #[test]
    fn authoring_output_annotation_is_optional() {
        let annotated = r#"
name = optional_output_annotation
kind = ode
states = central
ddt(central) = 0
out(cp) = central ~ continuous()
"#;
        let plain = r#"
name = optional_output_annotation
kind = ode
states = central
ddt(central) = 0
out(cp) = central
"#;

        let annotated = parse_module(annotated).expect("annotated authoring model parses");
        let plain = parse_module(plain).expect("plain authoring model parses");

        assert_eq!(annotated.to_string(), plain.to_string());
    }

    #[test]
    fn authoring_dx_and_ddt_lower_equivalently() {
        let dx_src = r#"
name = derivative_alias
kind = ode
states = central
dx(central) = -ke * central
out(cp) = central
"#;
        let ddt_src = r#"
name = derivative_alias
kind = ode
states = central
ddt(central) = -ke * central
out(cp) = central
"#;

        let dx_model = parse_module(dx_src).expect("dx authoring model parses");
        let ddt_model = parse_module(ddt_src).expect("ddt authoring model parses");

        assert_eq!(dx_model.to_string(), ddt_model.to_string());
    }

    #[test]
    fn authoring_rejects_out_target_not_in_declared_outputs() {
        let src = r#"
name = bimodal_ke
kind = ode
params = ke, v
states = central
outputs = cpa
infusion(iv) -> central
ddt(central) = -ke * central
out(cp) = central / v ~ continuous()
"#;

        let err = parse_model(src).expect_err("undeclared output target must fail");
        let rendered = err.render(src);

        assert!(
            rendered.contains("output `cp` is not declared in `outputs = ...`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn reports_route_syntax_error() {
        let src = "model broken { kind ode routes { oral depot } }";
        let err = parse_module(src).expect_err("missing route arrow should fail");
        let rendered = err.render(src);
        assert!(rendered.contains("expected `->`"), "{}", rendered);
        let debugged = format!("{err:?}");
        assert!(debugged.contains("error[DSL1000]"), "{}", debugged);
        assert!(debugged.contains("expected `->`"), "{}", debugged);
    }

    #[test]
    fn reports_delimiter_context_for_unclosed_state_size() {
        let src = "model broken { kind ode states { gut[2 } }";
        let err = parse_module(src).expect_err("missing closing bracket should fail");
        let rendered = err.render(src);
        assert!(rendered.contains("expected `]`"), "{}", rendered);
        assert!(
            rendered.contains("state size for `gut` opened here"),
            "{}",
            rendered
        );
        assert!(
            rendered.contains("close state size for `gut` with `]`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn recovers_multiple_statement_errors_in_one_block() {
        let src = "model broken { kind ode outputs { cp = 1 + ; dv = 2 + ; } }";
        let err = parse_module(src).expect_err("multiple statement errors should be reported");
        let rendered = err.render(src);
        assert_eq!(err.diagnostics().len(), 2, "{}", rendered);
        assert_eq!(
            rendered
                .matches("error[DSL1000]: expected expression after `+`")
                .count(),
            2,
            "{}",
            rendered
        );
        assert!(
            !rendered.contains("must contain at least one statement"),
            "{}",
            rendered
        );
    }

    #[test]
    fn recovers_newline_delimited_statement_errors() {
        let src = r#"
model broken {
    kind ode
    outputs {
        cp = 1 +
        dv = 2
        auc = 3 +
        cmax = 4
    }
}
"#;

        let err =
            parse_module(src).expect_err("newline-delimited statement errors should be reported");
        let rendered = err.render(src);
        assert_eq!(err.diagnostics().len(), 2, "{}", rendered);
        assert_eq!(
            rendered
                .matches("error[DSL1000]: expected expression after `+`")
                .count(),
            2,
            "{}",
            rendered
        );
        assert!(
            !rendered.contains("expected identifier, found `=`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn recovers_newline_delimited_binding_errors() {
        let src = r#"
model broken {
    kind ode
    constants {
        ka = 1 +
        ke = 2
        cl = 3 +
        v = 4
    }
    outputs {
        cp = 1
    }
}
"#;

        let err =
            parse_module(src).expect_err("newline-delimited binding errors should be reported");
        let rendered = err.render(src);
        assert_eq!(err.diagnostics().len(), 2, "{}", rendered);
        assert_eq!(
            rendered
                .matches("error[DSL1000]: expected expression after `+`")
                .count(),
            2,
            "{}",
            rendered
        );
        assert!(
            !rendered.contains("expected identifier, found `=`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn recovers_route_destination_before_next_line_route() {
        let src = r#"
model broken {
    kind ode
    routes {
        oral ->
        iv -> central
    }
    outputs {
        cp = 1
    }
}
"#;

        let err = parse_module(src).expect_err("route destination recovery should fail cleanly");
        let rendered = err.render(src);
        assert_eq!(err.diagnostics().len(), 1, "{}", rendered);
        assert!(
            rendered.contains("expected route destination after `->`"),
            "{}",
            rendered
        );
        assert!(rendered.contains("next route starts here"), "{}", rendered);
    }

    #[test]
    fn recovers_multiple_model_body_errors() {
        let src = "model broken { kind ode nonsense parameters { ka } what }";
        let err = parse_module(src).expect_err("multiple model-body errors should be reported");
        let rendered = err.render(src);
        assert_eq!(err.diagnostics().len(), 2, "{}", rendered);
        assert!(
            rendered.contains("unexpected token identifier `nonsense` in model body"),
            "{}",
            rendered
        );
        assert!(
            rendered.contains("unexpected token identifier `what` in model body"),
            "{}",
            rendered
        );
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
