use std::collections::{BTreeMap, BTreeSet};

use super::diagnostic::{Applicability, DiagnosticSuggestion, ParseError, Span, TextEdit};
use super::parser::{parse_expr_fragment, parse_place_fragment};
use super::syntax::*;
use crate::name_match::{
    common_prefix_len, edit_distance, is_high_confidence_match, is_single_adjacent_transposition,
};
use crate::RATE_FUNCTION_NAME;

const DEFAULT_MODEL_NAME: &str = "main";

pub(super) fn parse_module(src: &str) -> Result<Module, ParseError> {
    AuthoringParser::new(src).parse_module()
}

struct AuthoringParser<'a> {
    src: &'a str,
    name: Option<Ident>,
    explicit_kind: Option<(ModelKind, Span)>,
    parameters: Vec<Ident>,
    constants: Vec<Binding>,
    covariates: Vec<CovariateDecl>,
    states: Vec<StateDecl>,
    declared_derived: BTreeMap<String, Span>,
    declared_outputs: BTreeSet<String>,
    explicit_output_order: Vec<String>,
    explicit_outputs: BTreeMap<String, Span>,
    assigned_outputs: BTreeMap<String, Span>,
    assigned_equations: BTreeMap<String, Span>,
    declared_outputs_span: Option<Span>,
    routes: BTreeMap<(String, SurfaceRouteKind), SurfaceRoute>,
    route_order: Vec<(String, SurfaceRouteKind)>,
    route_modifiers: BTreeMap<String, Vec<Binding>>,
    derive_statements: Vec<Stmt>,
    derivative_statements: Vec<Stmt>,
    output_statements: Vec<Stmt>,
    init_statements: Vec<Stmt>,
    diffusion_statements: Vec<Stmt>,
    particles: Option<ParticlesDecl>,
    analytical: Option<AnalyticalBlock>,
    first_span: Option<Span>,
    last_span: Option<Span>,
}

#[derive(Clone)]
struct SurfaceRoute {
    input: Ident,
    destination: Place,
    kind: SurfaceRouteKind,
    span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SurfaceRouteKind {
    Bolus,
    Infusion,
}

#[derive(Clone)]
enum SurfaceRhs {
    Expr(Expr),
    If {
        condition: Expr,
        then_branch: Box<SurfaceRhs>,
        else_branch: Box<SurfaceRhs>,
        span: Span,
    },
}

type SimilarOutputScore = (usize, usize, usize);
type SimilarOutputMatch = (SimilarOutputScore, (String, Span));

impl<'a> AuthoringParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            name: None,
            explicit_kind: None,
            parameters: Vec::new(),
            constants: Vec::new(),
            covariates: Vec::new(),
            states: Vec::new(),
            declared_derived: BTreeMap::new(),
            declared_outputs: BTreeSet::new(),
            explicit_output_order: Vec::new(),
            explicit_outputs: BTreeMap::new(),
            assigned_outputs: BTreeMap::new(),
            assigned_equations: BTreeMap::new(),
            declared_outputs_span: None,
            routes: BTreeMap::new(),
            route_order: Vec::new(),
            route_modifiers: BTreeMap::new(),
            derive_statements: Vec::new(),
            derivative_statements: Vec::new(),
            output_statements: Vec::new(),
            init_statements: Vec::new(),
            diffusion_statements: Vec::new(),
            particles: None,
            analytical: None,
            first_span: None,
            last_span: None,
        }
    }

    fn parse_module(mut self) -> Result<Module, ParseError> {
        let masked = mask_source_comments(self.src);
        for (start, end) in split_logical_statements(&masked) {
            self.parse_line(&masked[start..end], start)?;
        }

        self.validate_declared_outputs_assigned()?;
        self.validate_declared_derived()?;

        let module_span = match (self.first_span, self.last_span) {
            (Some(first), Some(last)) => first.join(last),
            _ => Span::empty(0),
        };

        if self.routes.is_empty()
            && self.parameters.is_empty()
            && self.constants.is_empty()
            && self.covariates.is_empty()
            && self.states.is_empty()
            && self.derive_statements.is_empty()
            && self.derivative_statements.is_empty()
            && self.output_statements.is_empty()
            && self.init_statements.is_empty()
            && self.diffusion_statements.is_empty()
            && self.analytical.is_none()
            && self.particles.is_none()
        {
            return Ok(Module {
                models: Vec::new(),
                span: module_span,
            });
        }

        let kind = self.determine_kind(module_span)?;
        self.resolve_input_aliases();
        // ODE derivatives and SDE drift take their inputs inline; only
        // analytical models, which have no `dx(...)`, declare routes.
        if matches!(kind, ModelKind::Ode | ModelKind::Sde) {
            self.lower_ode_inputs(kind)?;
        }

        let surface_routes = std::mem::take(&mut self.routes);
        let route_order = std::mem::take(&mut self.route_order);
        let mut route_modifiers = std::mem::take(&mut self.route_modifiers);
        let mut routes = Vec::with_capacity(surface_routes.len());
        for (route_name, kind) in route_order {
            let Some(route) = surface_routes.get(&(route_name.clone(), kind)) else {
                continue;
            };
            let mut span = route.span;
            // Lag and bioavailability are bolus-only properties. When a bolus
            // and an infusion share a label, the modifiers bind to the bolus
            // route; the infusion route declares none.
            let shared_label = surface_routes
                .contains_key(&(route_name.clone(), SurfaceRouteKind::Bolus))
                && surface_routes.contains_key(&(route_name.clone(), SurfaceRouteKind::Infusion));
            let properties = if shared_label && kind == SurfaceRouteKind::Infusion {
                Vec::new()
            } else {
                route_modifiers.remove(&route_name).unwrap_or_default()
            };
            if !properties.is_empty() {
                span = properties
                    .iter()
                    .fold(span, |current, property| current.join(property.span));
            }
            routes.push(RouteDecl {
                input: route.input.clone(),
                destination: route.destination.clone(),
                kind: Some(match route.kind {
                    SurfaceRouteKind::Bolus => RouteKind::Bolus,
                    SurfaceRouteKind::Infusion => RouteKind::Infusion,
                }),
                properties,
                span,
            });
        }

        if let Some((unknown_route, properties)) = route_modifiers.iter().next() {
            return Err(ParseError::new(
                format!("unknown route `{unknown_route}` in dose modifier"),
                properties[0].span,
            ));
        }

        if matches!(kind, ModelKind::Analytical) && !self.derivative_statements.is_empty() {
            return Err(ParseError::new(
                "analytical models cannot declare `dx(...)` equations",
                self.derivative_statements[0].span,
            ));
        }

        if !self.explicit_output_order.is_empty() {
            let output_order = self
                .explicit_output_order
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), index))
                .collect::<BTreeMap<_, _>>();
            self.output_statements.sort_by_key(|statement| {
                output_statement_name(statement)
                    .and_then(|name| output_order.get(name).copied())
                    .unwrap_or(usize::MAX)
            });
        }

        let derivative_statements = std::mem::take(&mut self.derivative_statements);

        let name = self
            .name
            .unwrap_or_else(|| Ident::new(DEFAULT_MODEL_NAME, module_span));
        let mut items = Vec::new();

        if !self.parameters.is_empty() {
            items.push(ModelItem::Parameters(ParametersBlock {
                span: join_ident_spans(&self.parameters),
                items: self.parameters,
            }));
        }
        if !self.constants.is_empty() {
            items.push(ModelItem::Constants(ConstantsBlock {
                span: join_binding_spans(&self.constants),
                items: self.constants,
            }));
        }
        if !self.covariates.is_empty() {
            items.push(ModelItem::Covariates(CovariatesBlock {
                span: join_covariate_spans(&self.covariates),
                items: self.covariates,
            }));
        }
        if !self.states.is_empty() {
            items.push(ModelItem::States(StatesBlock {
                span: join_state_spans(&self.states),
                items: self.states,
            }));
        }
        if !routes.is_empty() {
            items.push(ModelItem::Routes(RoutesBlock {
                span: join_route_spans(&routes),
                routes,
            }));
        }
        if let Some(analytical) = self.analytical {
            items.push(ModelItem::Analytical(analytical));
        }
        if let Some(particles) = self.particles {
            items.push(ModelItem::Particles(particles));
        }
        if !self.derive_statements.is_empty() {
            items.push(ModelItem::Derive(StatementBlock {
                span: join_stmt_spans(&self.derive_statements),
                statements: self.derive_statements,
            }));
        }

        match kind {
            ModelKind::Ode if !derivative_statements.is_empty() => {
                items.push(ModelItem::Dynamics(StatementBlock {
                    span: join_stmt_spans(&derivative_statements),
                    statements: derivative_statements,
                }))
            }
            ModelKind::Sde if !derivative_statements.is_empty() => {
                items.push(ModelItem::Drift(StatementBlock {
                    span: join_stmt_spans(&derivative_statements),
                    statements: derivative_statements,
                }))
            }
            _ => {}
        }

        if !self.init_statements.is_empty() {
            items.push(ModelItem::Init(StatementBlock {
                span: join_stmt_spans(&self.init_statements),
                statements: self.init_statements,
            }));
        }
        if !self.diffusion_statements.is_empty() {
            items.push(ModelItem::Diffusion(StatementBlock {
                span: join_stmt_spans(&self.diffusion_statements),
                statements: self.diffusion_statements,
            }));
        }

        let outputs_span = if !self.output_statements.is_empty() {
            join_stmt_spans(&self.output_statements)
        } else {
            self.declared_outputs_span.unwrap_or(module_span)
        };
        items.push(ModelItem::Outputs(StatementBlock {
            span: outputs_span,
            statements: self.output_statements,
        }));

        Ok(Module {
            span: module_span,
            models: vec![Model {
                name,
                kind,
                span: module_span,
                items,
            }],
        })
    }

    fn parse_line(&mut self, line: &str, line_offset: usize) -> Result<(), ParseError> {
        let comment_cutoff = line.find('#').unwrap_or(line.len());
        let code = &line[..comment_cutoff];
        let leading = code.len() - code.trim_start().len();
        let trailing = code.trim_end().len();
        if leading >= trailing {
            return Ok(());
        }

        let trimmed = &code[leading..trailing];
        let span = Span::new(line_offset + leading, line_offset + trailing);
        self.note_span(span);

        if starts_with_keyword(trimmed, "if") {
            let stmt = self.parse_if_statement(trimmed, span.start, span, 0)?;
            self.derive_statements.push(stmt);
            return Ok(());
        }

        if let Some(arrow) = find_top_level_arrow(trimmed) {
            return self.parse_route_line(trimmed, arrow, span.start, span);
        }

        let eq_index = find_top_level_assignment(trimmed)
            .ok_or_else(|| missing_assignment_error(trimmed, span))?;

        let lhs = &trimmed[..eq_index];
        let rhs = &trimmed[eq_index + 1..];
        let lhs_abs = span.start;
        let rhs_abs = span.start + eq_index + 1;
        let lhs_trimmed = lhs.trim();

        if let Some(rest) = lhs_trimmed.strip_prefix("model") {
            if !rest.trim().is_empty() {
                return Err(ParseError::new("expected `name = <identifier>`", span));
            }
            return Err(ParseError::new(
                "`model = ...` has been renamed to `name = ...`",
                span,
            ));
        }

        if let Some(rest) = lhs_trimmed.strip_prefix("name") {
            if !rest.trim().is_empty() {
                return Err(ParseError::new("expected `name = <identifier>`", span));
            }
            self.name = Some(parse_ident_segment(rhs, rhs_abs)?);
            return Ok(());
        }

        if let Some(rest) = lhs_trimmed.strip_prefix("kind") {
            if !rest.trim().is_empty() {
                return Err(ParseError::new(
                    "expected `kind = <ode|analytical|sde>`",
                    span,
                ));
            }
            let kind_ident = parse_ident_segment(rhs, rhs_abs)?;
            let kind = match kind_ident.text.as_str() {
                "ode" => ModelKind::Ode,
                "analytical" => ModelKind::Analytical,
                "sde" => ModelKind::Sde,
                other => {
                    let mut error =
                        ParseError::new(format!("unknown model kind `{other}`"), kind_ident.span)
                            .with_help("the model kind is one of `ode`, `analytical`, or `sde`");
                    if let Some(candidate) = closest_keyword(other, MODEL_KINDS) {
                        error = error.with_suggestion(did_you_mean(kind_ident.span, candidate));
                    }
                    return Err(error);
                }
            };
            self.explicit_kind = Some((kind, span));
            return Ok(());
        }

        if lhs_trimmed == "params" || lhs_trimmed == "parameters" {
            self.parameters.extend(parse_ident_list(rhs, rhs_abs)?);
            return Ok(());
        }

        if lhs_trimmed == "covariates" {
            self.covariates.extend(parse_covariates_list(rhs, rhs_abs)?);
            return Ok(());
        }

        if lhs_trimmed == "states" {
            self.states.extend(parse_states_list(rhs, rhs_abs)?);
            return Ok(());
        }

        if lhs_trimmed == "derived" {
            for ident in parse_ident_list(rhs, rhs_abs)? {
                if let Some(existing_span) =
                    self.declared_derived.insert(ident.text.clone(), ident.span)
                {
                    return Err(ParseError::new(
                        format!("duplicate derived declaration `{}`", ident.text),
                        ident.span,
                    )
                    .with_context_label(
                        existing_span,
                        format!("derived `{}` first declared here", ident.text),
                    )
                    .with_help(format!(
                        "remove the duplicate `{}` from `derived = ...`",
                        ident.text
                    )));
                }
            }
            return Ok(());
        }

        if lhs_trimmed == "outputs" {
            self.declared_outputs_span = Some(span);
            for ident in parse_output_label_list(rhs, rhs_abs)? {
                self.explicit_output_order.push(ident.text.clone());
                self.declared_outputs.insert(ident.text.clone());
                self.explicit_outputs.insert(ident.text, ident.span);
            }
            return Ok(());
        }

        if lhs_trimmed == "particles" {
            let value = parse_expr_at(rhs, rhs_abs)?;
            self.particles = Some(ParticlesDecl { value, span });
            return Ok(());
        }

        if lhs_trimmed == "function" {
            return Err(ParseError::new(
                "`function = ...` has been renamed to `structure = ...`",
                span,
            ));
        }

        if lhs_trimmed == "structure" {
            let structure = parse_ident_segment(rhs, rhs_abs)?;
            self.analytical = Some(AnalyticalBlock { span, structure });
            return Ok(());
        }

        if let Some(name_segment) = lhs_trimmed.strip_prefix("const ") {
            let name_abs = span.start + (lhs.len() - lhs_trimmed.len()) + "const ".len();
            let name = parse_ident_segment(name_segment, name_abs)?;
            let value = parse_expr_at(rhs, rhs_abs)?;
            self.constants.push(Binding {
                span: name.span.join(value.span),
                name,
                value,
            });
            return Ok(());
        }

        if let Some(call) = parse_call_head(lhs, lhs_abs)? {
            return self.parse_call_assignment(call, rhs, rhs_abs, span);
        }

        // A comma-separated right-hand side is only valid after a declaration
        // header, so a plain name here is most likely a misspelled header.
        if is_valid_ident(lhs_trimmed) && split_top_level(rhs, ',').len() > 1 {
            let lhs_start = lhs_abs + (lhs.len() - lhs.trim_start().len());
            let lhs_span = Span::new(lhs_start, lhs_start + lhs_trimmed.len());
            let mut error = ParseError::new(
                format!("`{lhs_trimmed}` is not a declaration header"),
                lhs_span,
            )
            .with_help(
                "comma-separated lists follow a header: `params`, `covariates`, \
                 `states`, `derived`, or `outputs`",
            );
            if let Some(candidate) = closest_keyword(lhs_trimmed, DECLARATION_HEADERS) {
                error = error.with_suggestion(did_you_mean(lhs_span, candidate));
            }
            return Err(error);
        }

        let target = match parse_ident_segment(lhs, lhs_abs) {
            Ok(target) => target,
            Err(error) => {
                if self.declared_outputs_span.is_none() {
                    return Err(error);
                }

                let target = parse_output_label_segment(lhs, lhs_abs)?;
                if !self.declared_outputs.contains(&target.text) {
                    return Err(self.undeclared_output_error(&target.text, target.span));
                }
                target
            }
        };
        let rhs = parse_surface_rhs(rhs, rhs_abs)?;
        let stmt = build_assignment_statement(
            AssignTarget {
                span: target.span,
                kind: AssignTargetKind::Name(target.clone()),
            },
            rhs,
        );
        if self.declared_outputs.contains(&target.text) {
            self.note_output_assignment(&target, span)?;
            self.output_statements.push(stmt);
        } else {
            self.derive_statements.push(stmt);
        }
        Ok(())
    }

    fn parse_if_statement(
        &mut self,
        text: &str,
        abs_start: usize,
        span: Span,
        depth: usize,
    ) -> Result<Stmt, ParseError> {
        // Statement-level `if` nesting is capped at half of [`MAX_NESTING_DEPTH`]
        // (which remains the limit for expressions): each statement level spans
        // three parser frames (`parse_if_statement` -> `parse_if_body` ->
        // `parse_authoring_statement`), and at the full limit the unoptimized
        // stack usage (~9.5 KB/level) would exhaust a default 2 MiB thread
        // stack before the guard could fire.
        if depth >= crate::parser::MAX_NESTING_DEPTH / 2 {
            return Err(ParseError::new(
                format!(
                    "statement `if` is nested too deeply (maximum nesting depth is {})",
                    crate::parser::MAX_NESTING_DEPTH / 2
                ),
                Span::new(abs_start, abs_start + text.len().min(2)),
            ));
        }
        let rest = &text[2..];
        let rest_leading = rest.len() - rest.trim_start().len();
        let rest = &rest[rest_leading..];
        let rest_abs = abs_start + 2 + rest_leading;
        if !rest.starts_with('(') {
            return Err(ParseError::new(
                "expected `(` after `if`",
                Span::new(rest_abs, rest_abs + rest.len().min(1)),
            ));
        }
        let close = find_matching_delimiter(rest, '(', ')').ok_or_else(|| {
            ParseError::new(
                "unclosed `(` in `if` condition",
                Span::new(rest_abs, rest_abs + rest.len()),
            )
        })?;
        let condition = parse_expr_at(&rest[1..close], rest_abs + 1)?;
        let remaining = &rest[close + 1..];
        let remaining_abs = rest_abs + close + 1;

        let (then_branch, then_consumed) = self.parse_if_body(remaining, remaining_abs, depth)?;

        let after_then = &remaining[then_consumed..];
        let after_then_leading = after_then.len() - after_then.trim_start().len();
        let after_then_trimmed = &after_then[after_then_leading..];
        let after_then_abs = remaining_abs + then_consumed + after_then_leading;

        let else_branch = if after_then_trimmed.is_empty() {
            None
        } else {
            if !starts_with_keyword(after_then_trimmed, "else") {
                return Err(ParseError::new(
                    "expected `else` or end of `if`",
                    Span::new(
                        after_then_abs,
                        after_then_abs + after_then_trimmed.len().min(1),
                    ),
                ));
            }
            let else_rest = &after_then_trimmed[4..];
            let else_rest_leading = else_rest.len() - else_rest.trim_start().len();
            let else_rest_trimmed = &else_rest[else_rest_leading..];
            let else_rest_abs = after_then_abs + 4 + else_rest_leading;
            let (stmts, else_consumed) = if starts_with_keyword(else_rest_trimmed, "if") {
                let nested_span = Span::new(else_rest_abs, else_rest_abs + else_rest_trimmed.len());
                let nested = self.parse_if_statement(
                    else_rest_trimmed,
                    else_rest_abs,
                    nested_span,
                    depth + 1,
                )?;
                (vec![nested], else_rest_trimmed.len())
            } else {
                let (stmts, consumed) =
                    self.parse_if_body(else_rest_trimmed, else_rest_abs, depth)?;
                (stmts, consumed)
            };
            let trailing_src = &else_rest_trimmed[else_consumed..];
            let trailing = trailing_src.trim_start();
            let trailing_leading = trailing_src.len() - trailing.len();
            let trailing_abs = else_rest_abs + else_consumed + trailing_leading;
            if !trailing.is_empty() {
                return Err(ParseError::new(
                    "unexpected tokens after `if`/`else` statement",
                    Span::new(trailing_abs, trailing_abs + trailing.len()),
                ));
            }
            Some(stmts)
        };

        Ok(Stmt {
            span,
            kind: StmtKind::If(IfStmt {
                condition,
                then_branch,
                else_branch,
            }),
        })
    }

    fn parse_if_body(
        &mut self,
        text: &str,
        abs_start: usize,
        depth: usize,
    ) -> Result<(Vec<Stmt>, usize), ParseError> {
        let trimmed = text.trim_start();
        let leading = text.len() - trimmed.len();
        let abs = abs_start + leading;
        if !trimmed.starts_with('{') {
            return Err(ParseError::new(
                "expected `{` to open `if`/`else` body",
                Span::new(abs, abs + trimmed.len().min(1)),
            ));
        }
        let close = find_matching_delimiter(trimmed, '{', '}').ok_or_else(|| {
            ParseError::new(
                "unclosed `{` in `if`/`else` body",
                Span::new(abs, abs + trimmed.len()),
            )
        })?;
        let inner = &trimmed[1..close];
        let inner_abs = abs + 1;
        let mut statements = Vec::new();
        for (start, end) in split_logical_statements(inner) {
            let chunk = &inner[start..end];
            let stmt = self.parse_authoring_statement(chunk, inner_abs + start, depth)?;
            statements.push(stmt);
        }
        Ok((statements, leading + close + 1))
    }

    fn parse_authoring_statement(
        &mut self,
        text: &str,
        abs_start: usize,
        depth: usize,
    ) -> Result<Stmt, ParseError> {
        let trimmed = text.trim();
        let leading = text.len() - text.trim_start().len();
        let abs = abs_start + leading;
        if trimmed.is_empty() {
            return Err(ParseError::new(
                "empty statement in `if` body",
                Span::new(abs, abs),
            ));
        }
        if starts_with_keyword(trimmed, "if") {
            let span = Span::new(abs, abs + trimmed.len());
            return self.parse_if_statement(trimmed, abs, span, depth + 1);
        }
        let eq = find_top_level_assignment(trimmed).ok_or_else(|| {
            ParseError::new(
                "expected an assignment (`name = <expression>`) or `if` in `if` body",
                Span::new(abs, abs + trimmed.len().min(1)),
            )
        })?;
        let lhs = &trimmed[..eq];
        let rhs = &trimmed[eq + 1..];
        let lhs_trimmed = lhs.trim();
        let lhs_leading = lhs.len() - lhs.trim_start().len();
        let lhs_abs = abs + lhs_leading;
        let rhs_trimmed = rhs.trim();
        let rhs_leading = rhs.len() - rhs.trim_start().len();
        let rhs_abs = abs + eq + 1 + rhs_leading;
        if lhs_trimmed.is_empty() || rhs_trimmed.is_empty() {
            return Err(ParseError::new(
                "expected `name = <expression>`",
                Span::new(abs, abs + trimmed.len().min(1)),
            ));
        }
        if lhs_trimmed.contains('(') {
            return Err(ParseError::new(
                "an `if` statement body supports only plain-variable assignments \
                 (`x = <expression>`); to make `ddt(...)`, `out(...)`, or `init(...)` \
                 conditional, use a conditional equation instead, e.g. \
                 `ddt(central) = if (cond) <a> else <b>`",
                Span::new(lhs_abs, lhs_abs + lhs_trimmed.len()),
            ));
        }
        let target = parse_ident_segment(lhs_trimmed, lhs_abs)?;
        let rhs = parse_surface_rhs(rhs_trimmed, rhs_abs)?;
        Ok(build_assignment_statement(
            AssignTarget {
                span: Span::new(lhs_abs, lhs_abs + lhs_trimmed.len()),
                kind: AssignTargetKind::Name(target),
            },
            rhs,
        ))
    }

    // ODE input syntax lowers to the existing routes, event scales, and rates.
    // The execution model and simulator do not need a second input mechanism.
    fn lower_ode_inputs(&mut self, kind: ModelKind) -> Result<(), ParseError> {
        let model = if kind == ModelKind::Sde { "SDE" } else { "ODE" };
        if let Some(route) = self.routes.values().next() {
            return Err(ParseError::new(
                format!("{model} inputs belong in the derivative RHS: use `+ bolus(input)` or `+ infusion(input)`, not a route declaration"),
                route.span,
            ));
        }
        for property in self.route_modifiers.values().flatten() {
            if property.name.text == "bioavailability" {
                return Err(ParseError::new(
                    format!("{model} models no longer accept `fa(input)`; multiply `bolus(input)` by the scale in the derivative RHS"),
                    property.span,
                ));
            }
        }
        let mut statements = std::mem::take(&mut self.derivative_statements);
        self.lower_ode_input_statements(&mut statements, false)?;
        self.derivative_statements = statements;
        Ok(())
    }

    fn lower_ode_input_statements(
        &mut self,
        statements: &mut [Stmt],
        conditional: bool,
    ) -> Result<(), ParseError> {
        for statement in statements {
            match &mut statement.kind {
                StmtKind::Assign(assign) => {
                    let AssignTargetKind::Call { callee, args } = &assign.target.kind else {
                        continue;
                    };
                    if callee.text != "ddt" || args.len() != 1 {
                        continue;
                    }
                    let target = &args[0];
                    let destination = parse_place_at(
                        &self.src[target.span.start..target.span.end],
                        target.span.start,
                    )?;
                    self.lower_ode_input_expr(&mut assign.value, &destination, conditional)?;
                }
                StmtKind::If(branch) => {
                    self.lower_ode_input_statements(&mut branch.then_branch, true)?;
                    if let Some(otherwise) = &mut branch.else_branch {
                        self.lower_ode_input_statements(otherwise, true)?;
                    }
                }
                StmtKind::For(body) => {
                    self.lower_ode_input_statements(&mut body.body, true)?;
                }
                StmtKind::Let(_) => {}
            }
        }
        Ok(())
    }

    fn lower_ode_input_expr(
        &mut self,
        expr: &mut Expr,
        destination: &Place,
        conditional: bool,
    ) -> Result<(), ParseError> {
        let input_argument = match &expr.kind {
            ExprKind::Call { callee, args }
                if matches!(callee.text.as_str(), "bolus" | "infusion") && args.len() == 1 =>
            {
                Some(&args[0])
            }
            ExprKind::Index { target, index } if matches!(&target.kind, ExprKind::Name(name) if matches!(name.text.as_str(), "bolus" | "infusion")) => {
                Some(index.as_ref())
            }
            _ => None,
        };
        if let Some(argument) = input_argument {
            parse_route_label_segment(
                &self.src[argument.span.start..argument.span.end],
                argument.span.start,
            )?;
        }
        if let Some((input, scale)) = scaled_bolus_term(expr) {
            if let Some(state) = first_state_reference(&scale, &self.states) {
                return Err(ParseError::new(
                    format!(
                        "a bolus can only be scaled by values that do not depend on states; `{}` is a state",
                        state.text
                    ),
                    state.span,
                )
                .with_note(
                    "the scale becomes the dose's bioavailability, which is fixed when the dose is given",
                )
                .with_help(
                    "scale the bolus by parameters, covariates, or constants, and move \
                     state-dependent terms out of the bolus term",
                ));
            }
            if conditional {
                return Err(ParseError::new(
                    "put conditional bolus scaling in an event-safe derived value, then use one `bolus(input) * scale` term outside the conditional",
                    expr.span,
                ));
            }
            self.infer_ode_route(
                input.clone(),
                SurfaceRouteKind::Bolus,
                destination,
                expr.span,
            )?;
            if !matches!(scale.kind, ExprKind::Number(1.0)) {
                self.route_modifiers
                    .entry(input.text)
                    .or_default()
                    .push(Binding {
                        name: Ident::new("bioavailability", expr.span),
                        value: scale,
                        span: expr.span,
                    });
            }
            expr.kind = ExprKind::Number(0.0);
            return Ok(());
        }
        if let Some((kind, input)) = ode_input(expr) {
            if kind == SurfaceRouteKind::Infusion {
                self.infer_ode_route(input.clone(), kind, destination, expr.span)?;
                expr.kind = ExprKind::Call {
                    callee: Ident::new(RATE_FUNCTION_NAME, expr.span),
                    args: vec![Expr {
                        span: input.span,
                        kind: ExprKind::Name(input),
                    }],
                };
                return Ok(());
            }
        }
        match &mut expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => {
                let had_bolus = contains_bolus(lhs) || contains_bolus(rhs);
                self.lower_ode_input_expr(lhs, destination, conditional)?;
                self.lower_ode_input_expr(rhs, destination, conditional)?;
                if had_bolus && matches!(lhs.kind, ExprKind::Number(0.0)) {
                    expr.kind = rhs.kind.clone();
                } else if had_bolus && matches!(rhs.kind, ExprKind::Number(0.0)) {
                    expr.kind = lhs.kind.clone();
                }
            }
            ExprKind::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } => {
                let had_bolus = contains_bolus(lhs);
                self.lower_ode_input_expr(lhs, destination, conditional)?;
                reject_bolus_term(rhs)?;
                self.lower_ode_input_expr(rhs, destination, conditional)?;
                if had_bolus && matches!(lhs.kind, ExprKind::Number(0.0)) {
                    expr.kind = ExprKind::Unary {
                        op: UnaryOp::Minus,
                        expr: rhs.clone(),
                    };
                }
            }
            _ => {
                reject_bolus_term(expr)?;
                match &mut expr.kind {
                    ExprKind::Unary { expr, .. } => {
                        self.lower_ode_input_expr(expr, destination, conditional)?
                    }
                    ExprKind::Binary { lhs, rhs, .. } => {
                        self.lower_ode_input_expr(lhs, destination, conditional)?;
                        self.lower_ode_input_expr(rhs, destination, conditional)?;
                    }
                    ExprKind::Call { callee, args } => {
                        if matches!(
                            callee.text.as_str(),
                            "bolus" | "infusion" | RATE_FUNCTION_NAME
                        ) {
                            return Err(ParseError::new(
                                "use `bolus(input_name)` or `infusion(input_name)` in the ODE derivative RHS",
                                expr.span,
                            ));
                        }
                        for arg in args {
                            self.lower_ode_input_expr(arg, destination, conditional)?;
                        }
                    }
                    ExprKind::Index { target, index } => {
                        self.lower_ode_input_expr(target, destination, conditional)?;
                        self.lower_ode_input_expr(index, destination, conditional)?;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Rewrites the Pmetrics spellings `b[n]` and `rateiv[n]` to `bolus[n]` and
    /// `infusion[n]` in derivative equations. An alias is left alone when a
    /// state of the same name exists, since `b[n]` then indexes that state.
    fn resolve_input_aliases(&mut self) {
        let aliases: Vec<(&str, &str)> = INPUT_ALIASES
            .iter()
            .copied()
            .filter(|(alias, _)| !self.states.iter().any(|state| state.name.text == *alias))
            .collect();
        if !aliases.is_empty() {
            rename_input_aliases(&mut self.derivative_statements, &aliases, self.src);
        }
    }

    fn infer_ode_route(
        &mut self,
        input: Ident,
        kind: SurfaceRouteKind,
        destination: &Place,
        span: Span,
    ) -> Result<(), ParseError> {
        parse_route_label_segment(&input.text, input.span.start)?;
        let key = (input.text.clone(), kind);
        if let Some(existing) = self.routes.get(&key) {
            if kind == SurfaceRouteKind::Bolus
                && !expr_matches_place(&place_to_expr(&existing.destination), destination)
            {
                return Err(ParseError::new(
                    format!("input `{}` must have one destination state", input.text),
                    span,
                ));
            }
            if kind == SurfaceRouteKind::Bolus {
                return Err(ParseError::new(
                    format!(
                        "use one `bolus({}) * scale` term for this input",
                        input.text
                    ),
                    span,
                ));
            }
            // Infusions reuse their rate slot wherever referenced. The first
            // destination supplies metadata only; it does not inject a rate.
            return Ok(());
        }
        self.route_order.push(key.clone());
        self.routes.insert(
            key,
            SurfaceRoute {
                input,
                destination: destination.clone(),
                kind,
                span,
            },
        );
        Ok(())
    }

    fn parse_route_line(
        &mut self,
        trimmed: &str,
        arrow: usize,
        line_start: usize,
        span: Span,
    ) -> Result<(), ParseError> {
        let lhs = &trimmed[..arrow];
        let rhs = &trimmed[arrow + 2..];
        let call = parse_call_head(lhs, line_start)?
            .ok_or_else(|| ParseError::new("expected `bolus(route)` or `infusion(route)`", span))?;

        let kind = match call.callee.text.as_str() {
            "bolus" => SurfaceRouteKind::Bolus,
            "infusion" => SurfaceRouteKind::Infusion,
            other => {
                return Err(ParseError::new(
                    format!("unknown route shorthand `{other}`"),
                    call.callee.span,
                ))
            }
        };

        let input = parse_route_label_segment(call.argument, call.argument_start)?;
        let route_name = input.text.clone();
        let destination = parse_place_at(rhs, line_start + arrow + 2)?;
        // Route labels are unique per kind: a bolus and an infusion may share
        // a label (one drug given by either route), but two routes of the
        // same kind may not.
        if self.routes.contains_key(&(route_name.clone(), kind)) {
            return Err(ParseError::new(
                format!("duplicate route `{}`", input.text),
                input.span,
            ));
        }
        self.routes.insert(
            (route_name.clone(), kind),
            SurfaceRoute {
                input,
                destination,
                kind,
                span,
            },
        );
        self.route_order.push((route_name, kind));
        Ok(())
    }

    fn parse_call_assignment(
        &mut self,
        call: CallHead<'_>,
        rhs: &str,
        rhs_abs: usize,
        span: Span,
    ) -> Result<(), ParseError> {
        match call.callee.text.as_str() {
            "lag" | "fa" => {
                let route_name = parse_route_label_segment(call.argument, call.argument_start)?;
                let value = parse_expr_at(rhs, rhs_abs)?;
                let property_name = match call.callee.text.as_str() {
                    "lag" => "lag",
                    "fa" => "bioavailability",
                    _ => unreachable!(),
                };
                let binding = Binding {
                    span,
                    name: Ident::new(property_name, call.callee.span),
                    value,
                };
                let properties = self.route_modifiers.entry(route_name.text).or_default();
                if properties
                    .iter()
                    .any(|property| property.name.text == property_name)
                {
                    return Err(ParseError::new(
                        format!("duplicate route property `{property_name}`"),
                        call.callee.span,
                    ));
                }
                properties.push(binding);
            }
            "dx" | "ddt" => {
                let place = parse_place_at(call.argument, call.argument_start)?;
                self.note_equation_target(format!("dx({})", self.place_key(&place)), span)?;
                let rhs = parse_surface_rhs(rhs, rhs_abs)?;
                let stmt = build_assignment_statement(
                    AssignTarget {
                        span,
                        kind: AssignTargetKind::Call {
                            callee: Ident::new("ddt", call.callee.span),
                            args: vec![place_to_expr(&place)],
                        },
                    },
                    rhs,
                );
                self.derivative_statements.push(stmt);
            }
            "noise" => {
                let place = parse_place_at(call.argument, call.argument_start)?;
                self.note_equation_target(format!("noise({})", self.place_key(&place)), span)?;
                let rhs = parse_surface_rhs(rhs, rhs_abs)?;
                let stmt = build_assignment_statement(
                    AssignTarget {
                        span,
                        kind: AssignTargetKind::Call {
                            callee: Ident::new("noise", call.callee.span),
                            args: vec![place_to_expr(&place)],
                        },
                    },
                    rhs,
                );
                self.diffusion_statements.push(stmt);
            }
            "init" => {
                let place = parse_place_at(call.argument, call.argument_start)?;
                self.note_equation_target(format!("init({})", self.place_key(&place)), span)?;
                let rhs = parse_surface_rhs(rhs, rhs_abs)?;
                let stmt = build_assignment_statement(
                    AssignTarget {
                        span,
                        kind: place_to_assign_target(place),
                    },
                    rhs,
                );
                self.init_statements.push(stmt);
            }
            "out" => {
                let output = parse_output_label_segment(call.argument, call.argument_start)?;
                self.validate_output_target(&output)?;
                self.declared_outputs.insert(output.text.clone());
                self.note_output_assignment(&output, span)?;

                let (expr_rhs, annotation) = split_output_annotation(rhs);
                if let Some((annotation_src, annotation_start)) = annotation {
                    validate_output_annotation(annotation_src, rhs_abs + annotation_start)?;
                }

                let rhs = parse_surface_rhs(expr_rhs, rhs_abs)?;
                let stmt = build_assignment_statement(
                    AssignTarget {
                        span: output.span,
                        kind: AssignTargetKind::Name(output),
                    },
                    rhs,
                );
                self.output_statements.push(stmt);
            }
            other => {
                return Err(ParseError::new(
                    format!("unsupported equation target `{other}`"),
                    call.callee.span,
                ))
            }
        }
        Ok(())
    }

    fn determine_kind(&self, module_span: Span) -> Result<ModelKind, ParseError> {
        let inferred = if !self.diffusion_statements.is_empty() || self.particles.is_some() {
            ModelKind::Sde
        } else if self.analytical.is_some() {
            ModelKind::Analytical
        } else {
            ModelKind::Ode
        };

        let kind = self.explicit_kind.map(|(kind, _)| kind).unwrap_or(inferred);
        let kind_span = self
            .explicit_kind
            .map(|(_, span)| span)
            .unwrap_or(module_span);

        if matches!(kind, ModelKind::Analytical)
            && (!self.diffusion_statements.is_empty() || self.particles.is_some())
        {
            return Err(ParseError::new(
                "analytical models cannot declare particles or noise equations",
                kind_span,
            ));
        }

        if matches!(kind, ModelKind::Ode) && !self.diffusion_statements.is_empty() {
            return Err(ParseError::new(
                "ODE models cannot declare `noise(...)` equations",
                self.diffusion_statements[0].span,
            ));
        }

        if matches!(kind, ModelKind::Sde) {
            if let Some(analytical) = &self.analytical {
                return Err(ParseError::new(
                    "SDE models cannot declare an analytical structure",
                    analytical.span,
                ));
            }
        }

        Ok(kind)
    }

    fn validate_output_target(&self, output: &Ident) -> Result<(), ParseError> {
        if self.declared_outputs_span.is_none() || self.explicit_outputs.contains_key(&output.text)
        {
            return Ok(());
        }

        Err(self.undeclared_output_error(&output.text, output.span))
    }

    fn note_output_assignment(&mut self, output: &Ident, span: Span) -> Result<(), ParseError> {
        self.note_equation_target(format!("out({})", output.text), span)?;
        self.assigned_outputs
            .entry(output.text.clone())
            .or_insert(output.span);
        Ok(())
    }

    /// Identifies the state element a place writes: constant indices are
    /// compared by value, so `a[0]`, `a[00]`, and `a[(0)]` are the same target.
    fn place_key(&self, place: &Place) -> String {
        let Some(index) = &place.index else {
            return place.name.text.clone();
        };
        let index = match index.kind {
            ExprKind::Number(value) if value.fract() == 0.0 => format!("{value}"),
            _ => self.src[index.span.start..index.span.end]
                .split_whitespace()
                .collect(),
        };
        format!("{}[{index}]", place.name.text)
    }

    /// Records a top-level equation target such as `dx(central)`. Each one is
    /// an unconditional assignment, so a second equation for the same target
    /// would silently replace the first.
    fn note_equation_target(&mut self, target: String, span: Span) -> Result<(), ParseError> {
        if let Some(first) = self.assigned_equations.insert(target.clone(), span) {
            return Err(
                ParseError::new(format!("`{target}` is assigned more than once"), span)
                    .with_context_label(first, format!("`{target}` first assigned here"))
                    .with_help(format!(
                        "combine both right-hand sides into a single `{target} = ...` equation; \
                 a second equation would replace the first"
                    )),
            );
        }
        Ok(())
    }

    fn validate_declared_outputs_assigned(&self) -> Result<(), ParseError> {
        let mut diagnostics = Vec::new();
        if !self.explicit_outputs.is_empty() {
            for (name, span) in &self.assigned_outputs {
                if self.explicit_outputs.contains_key(name) {
                    continue;
                }

                diagnostics.push(self.undeclared_output_error(name, *span).into_diagnostic());
            }
        }

        for (name, span) in &self.explicit_outputs {
            if self.assigned_outputs.contains_key(name) {
                continue;
            }

            let mut error = ParseError::new(
                format!("output `{name}` is declared in `outputs = ...` but never assigned"),
                *span,
            )
            .with_help(format!("add `out({name}) = ...` or `{name} = ...`"));
            if let Some(outputs_span) = self.declared_outputs_span {
                error = error.with_context_label(outputs_span, "`outputs = ...` declared here");
            }
            diagnostics.push(error.into_diagnostic());
        }

        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(ParseError::from_diagnostics(diagnostics))
        }
    }

    fn validate_declared_derived(&self) -> Result<(), ParseError> {
        for parameter in &self.parameters {
            if let Some(derived_span) = self.declared_derived.get(&parameter.text).copied() {
                return Err(ParseError::new(
                    format!(
                        "derived name `{}` collides with primary parameter `{}`",
                        parameter.text, parameter.text
                    ),
                    derived_span,
                )
                .with_context_label(
                    parameter.span,
                    format!("primary parameter `{}` declared here", parameter.text),
                )
                .with_help("names declared in `params` and `derived` must be distinct"));
            }
        }

        if self.declared_derived.is_empty() {
            return Ok(());
        }

        let mut assigned = BTreeMap::new();
        collect_named_assign_targets(&self.derive_statements, &mut assigned);

        for (name, span) in &assigned {
            if self.declared_derived.contains_key(name) {
                continue;
            }
            return Err(ParseError::new(
                format!("derived value `{name}` is not declared in `derived = ...`"),
                *span,
            )
            .with_help(format!(
                "add `{name}` to `derived = ...` or rename the assignment target"
            )));
        }

        for (name, span) in &self.declared_derived {
            if assigned.contains_key(name) {
                continue;
            }
            return Err(ParseError::new(
                format!("derived value `{name}` is declared in `derived = ...` but never assigned"),
                *span,
            )
            .with_help(format!(
                "add `{name} = ...` or remove `{name}` from `derived = ...`"
            )));
        }

        Ok(())
    }

    fn undeclared_output_error(&self, name: &str, span: Span) -> ParseError {
        let mut error = ParseError::new(
            format!("output `{name}` is not declared in `outputs = ...`"),
            span,
        )
        .with_help("add the output name to `outputs = ...` or rename the output assignment to match a declared output");

        if let Some((candidate, candidate_span)) = self.best_similar_output_name(name) {
            error = error
                .with_secondary_label(
                    candidate_span,
                    format!("declared output `{candidate}` is here"),
                )
                .with_suggestion(DiagnosticSuggestion {
                    message: format!("did you mean `{candidate}`?"),
                    edits: vec![TextEdit {
                        span,
                        replacement: candidate,
                    }],
                    applicability: Applicability::MaybeIncorrect,
                });
        } else if let Some(outputs_span) = self.declared_outputs_span {
            error = error.with_context_label(outputs_span, "`outputs = ...` declared here");
        }

        error
    }
    fn best_similar_output_name(&self, needle: &str) -> Option<(String, Span)> {
        let original_needle = needle;
        let needle = needle.to_ascii_lowercase();
        let mut best: Option<SimilarOutputMatch> = None;
        let mut tied = false;

        for (candidate, span) in &self.explicit_outputs {
            if candidate == original_needle {
                continue;
            }
            let lookup = candidate.to_ascii_lowercase();
            let distance = if is_single_adjacent_transposition(&needle, &lookup) {
                1
            } else {
                edit_distance(&needle, &lookup)
            };
            let prefix = common_prefix_len(&needle, &lookup);
            if !is_high_confidence_match(&needle, &lookup, distance, prefix) {
                continue;
            }
            let score = (
                distance,
                usize::MAX - prefix,
                needle.len().abs_diff(lookup.len()),
            );
            match &best {
                None => {
                    best = Some((score, (candidate.clone(), *span)));
                    tied = false;
                }
                Some((best_score, _)) if score < *best_score => {
                    best = Some((score, (candidate.clone(), *span)));
                    tied = false;
                }
                Some((best_score, _)) if score == *best_score => tied = true,
                _ => {}
            }
        }

        if tied {
            None
        } else {
            best.map(|(_, candidate)| candidate)
        }
    }

    fn note_span(&mut self, span: Span) {
        self.first_span.get_or_insert(span);
        self.last_span = Some(span);
    }
}

const INPUT_ALIASES: &[(&str, &str)] = &[("b", "bolus"), ("rateiv", "infusion")];

fn rename_input_aliases(statements: &mut [Stmt], aliases: &[(&str, &str)], src: &str) {
    for statement in statements {
        match &mut statement.kind {
            StmtKind::Let(binding) => {
                rename_input_aliases_in_expr(&mut binding.value, aliases, src)
            }
            StmtKind::Assign(assign) => {
                rename_input_aliases_in_expr(&mut assign.value, aliases, src)
            }
            StmtKind::If(branch) => {
                rename_input_aliases_in_expr(&mut branch.condition, aliases, src);
                rename_input_aliases(&mut branch.then_branch, aliases, src);
                if let Some(otherwise) = &mut branch.else_branch {
                    rename_input_aliases(otherwise, aliases, src);
                }
            }
            StmtKind::For(body) => rename_input_aliases(&mut body.body, aliases, src),
        }
    }
}

fn rename_input_aliases_in_expr(expr: &mut Expr, aliases: &[(&str, &str)], src: &str) {
    // Returns true when `ident` was an alias and now names an input.
    let rename = |ident: &mut Ident| {
        let Some((_, canonical)) = aliases.iter().find(|(alias, _)| ident.text == *alias) else {
            return false;
        };
        ident.text = (*canonical).to_string();
        true
    };
    // The parser turns integer arguments of `bolus`/`infusion` into route
    // labels; do the same for the aliases it parsed as plain numbers.
    let label_from_number = |argument: &mut Expr| {
        if let ExprKind::Number(_) = argument.kind {
            let text = &src[argument.span.start..argument.span.end];
            if !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()) {
                argument.kind = ExprKind::Name(Ident::new(text, argument.span));
            }
        }
    };
    match &mut expr.kind {
        ExprKind::Index { target, index } => {
            if let ExprKind::Name(name) = &mut target.kind {
                if rename(name) {
                    label_from_number(index);
                }
            }
            rename_input_aliases_in_expr(index, aliases, src);
        }
        ExprKind::Call { callee, args } => {
            if rename(callee) && args.len() == 1 {
                label_from_number(&mut args[0]);
            }
            for arg in args {
                rename_input_aliases_in_expr(arg, aliases, src);
            }
        }
        ExprKind::Unary { expr, .. } => rename_input_aliases_in_expr(expr, aliases, src),
        ExprKind::Binary { lhs, rhs, .. } => {
            rename_input_aliases_in_expr(lhs, aliases, src);
            rename_input_aliases_in_expr(rhs, aliases, src);
        }
        ExprKind::Number(_) | ExprKind::Bool(_) | ExprKind::Name(_) => {}
    }
}

fn first_state_reference<'e>(expr: &'e Expr, states: &[StateDecl]) -> Option<&'e Ident> {
    match &expr.kind {
        ExprKind::Name(name) if states.iter().any(|state| state.name.text == name.text) => {
            Some(name)
        }
        ExprKind::Index { target, index } => {
            first_state_reference(target, states).or_else(|| first_state_reference(index, states))
        }
        ExprKind::Unary { expr, .. } => first_state_reference(expr, states),
        ExprKind::Binary { lhs, rhs, .. } => {
            first_state_reference(lhs, states).or_else(|| first_state_reference(rhs, states))
        }
        ExprKind::Call { args, .. } => args
            .iter()
            .find_map(|arg| first_state_reference(arg, states)),
        ExprKind::Number(_) | ExprKind::Bool(_) | ExprKind::Name(_) => None,
    }
}

fn ode_input(expr: &Expr) -> Option<(SurfaceRouteKind, Ident)> {
    let (name, argument) = match &expr.kind {
        ExprKind::Call { callee, args } if args.len() == 1 => (&callee.text, &args[0]),
        ExprKind::Index { target, index } => {
            let ExprKind::Name(name) = &target.kind else {
                return None;
            };
            (&name.text, index.as_ref())
        }
        _ => return None,
    };
    let kind = match name.as_str() {
        "bolus" => SurfaceRouteKind::Bolus,
        "infusion" => SurfaceRouteKind::Infusion,
        _ => return None,
    };
    let ExprKind::Name(input) = &argument.kind else {
        return None;
    };
    Some((kind, input.clone()))
}

fn contains_bolus(expr: &Expr) -> bool {
    if matches!(ode_input(expr), Some((SurfaceRouteKind::Bolus, _))) {
        return true;
    }
    match &expr.kind {
        ExprKind::Unary { expr, .. } => contains_bolus(expr),
        ExprKind::Binary { lhs, rhs, .. } => contains_bolus(lhs) || contains_bolus(rhs),
        ExprKind::Call { args, .. } => args.iter().any(contains_bolus),
        ExprKind::Index { target, index } => contains_bolus(target) || contains_bolus(index),
        _ => false,
    }
}

fn reject_bolus_term(expr: &Expr) -> Result<(), ParseError> {
    if contains_bolus(expr) {
        return Err(ParseError::new(
            "bolus inputs must be additive linear terms such as `bolus(input) * scale`; put the scale in an event-safe derived value",
            expr.span,
        ));
    }
    Ok(())
}

fn scaled_bolus_term(expr: &Expr) -> Option<(Ident, Expr)> {
    if let Some((SurfaceRouteKind::Bolus, input)) = ode_input(expr) {
        return Some((
            input,
            Expr {
                span: expr.span,
                kind: ExprKind::Number(1.0),
            },
        ));
    }
    if let ExprKind::Binary {
        op: BinaryOp::Mul,
        lhs,
        rhs,
    } = &expr.kind
    {
        let (input, scale, other) = if !contains_bolus(rhs) {
            let (input, scale) = scaled_bolus_term(lhs)?;
            (input, scale, rhs)
        } else if !contains_bolus(lhs) {
            let (input, scale) = scaled_bolus_term(rhs)?;
            (input, scale, lhs)
        } else {
            return None;
        };
        if matches!(scale.kind, ExprKind::Number(1.0)) {
            return Some((input, other.as_ref().clone()));
        }
        return Some((
            input,
            Expr {
                span: expr.span,
                kind: ExprKind::Binary {
                    op: BinaryOp::Mul,
                    lhs: Box::new(scale),
                    rhs: other.clone(),
                },
            },
        ));
    }
    None
}

struct CallHead<'a> {
    callee: Ident,
    argument: &'a str,
    argument_start: usize,
}

fn parse_call_head<'a>(src: &'a str, abs_start: usize) -> Result<Option<CallHead<'a>>, ParseError> {
    let trimmed = src.trim();
    let leading = src.len() - src.trim_start().len();
    let Some(open) = trimmed.find('(') else {
        return Ok(None);
    };
    let Some(close) = trimmed.rfind(')') else {
        return Err(ParseError::new(
            "expected `)` to close call-style target",
            Span::new(abs_start + leading + open, abs_start + src.len()),
        ));
    };
    if !trimmed[close + 1..].trim().is_empty() {
        return Err(ParseError::new(
            "unexpected trailing tokens after target",
            Span::new(abs_start + leading + close + 1, abs_start + src.len()),
        ));
    }

    let callee_src = &trimmed[..open];
    let argument_src = &trimmed[open + 1..close];
    let callee = parse_ident_segment(callee_src, abs_start + leading)?;
    let argument_start = abs_start + leading + open + 1;
    Ok(Some(CallHead {
        callee,
        argument: argument_src,
        argument_start,
    }))
}

fn parse_ident_list(src: &str, abs_start: usize) -> Result<Vec<Ident>, ParseError> {
    split_top_level(src, ',')
        .into_iter()
        .map(|(segment, start)| parse_ident_segment(segment, abs_start + start))
        .collect()
}

fn parse_output_label_list(src: &str, abs_start: usize) -> Result<Vec<Ident>, ParseError> {
    split_top_level(src, ',')
        .into_iter()
        .map(|(segment, start)| parse_output_label_segment(segment, abs_start + start))
        .collect()
}

fn parse_covariates_list(src: &str, abs_start: usize) -> Result<Vec<CovariateDecl>, ParseError> {
    let mut covariates = Vec::new();
    for (segment, start) in split_top_level(src, ',') {
        let item_abs = abs_start + start;
        let trimmed = segment.trim();
        let leading = segment.len() - segment.trim_start().len();
        let at_index = trimmed.find('@');
        let (name_src, interpolation) = match at_index {
            Some(index) => (&trimmed[..index], Some((&trimmed[index + 1..], index + 1))),
            None => (trimmed, None),
        };
        let name = parse_ident_segment(name_src, item_abs + leading)?;
        let interpolation = interpolation
            .map(|(annotation_src, annotation_offset)| {
                let interpolation =
                    parse_ident_segment(annotation_src, item_abs + leading + annotation_offset)?;
                let normalized = normalize_interpolation_name(&interpolation.text);
                Ok(Ident::new(normalized, interpolation.span))
            })
            .transpose()?;
        covariates.push(CovariateDecl {
            span: name.span.join(
                interpolation
                    .as_ref()
                    .map_or(name.span, |annotation| annotation.span),
            ),
            name,
            interpolation,
        });
    }
    Ok(covariates)
}

fn parse_states_list(src: &str, abs_start: usize) -> Result<Vec<StateDecl>, ParseError> {
    let mut states = Vec::new();
    for (segment, start) in split_top_level(src, ',') {
        let item_abs = abs_start + start;
        let place = parse_place_at(segment, item_abs)?;
        states.push(StateDecl {
            span: place.span,
            name: place.name,
            size: place.index,
        });
    }
    Ok(states)
}

fn parse_ident_segment(src: &str, abs_start: usize) -> Result<Ident, ParseError> {
    let trimmed = src.trim();
    let leading = src.len() - src.trim_start().len();
    if trimmed.is_empty() {
        return Err(ParseError::new(
            "expected identifier",
            Span::new(abs_start, abs_start + src.len()),
        ));
    }
    if !is_valid_ident(trimmed) {
        return Err(ParseError::new(
            format!("expected identifier, found `{trimmed}`"),
            Span::new(abs_start + leading, abs_start + leading + trimmed.len()),
        ));
    }
    Ok(Ident::new(
        trimmed,
        Span::new(abs_start + leading, abs_start + leading + trimmed.len()),
    ))
}

fn parse_output_label_segment(src: &str, abs_start: usize) -> Result<Ident, ParseError> {
    parse_label_segment(src, abs_start, LabelKind::Output)
}

fn parse_route_label_segment(src: &str, abs_start: usize) -> Result<Ident, ParseError> {
    parse_label_segment(src, abs_start, LabelKind::Route)
}

fn parse_label_segment(src: &str, abs_start: usize, kind: LabelKind) -> Result<Ident, ParseError> {
    let trimmed = src.trim();
    let leading = src.len() - src.trim_start().len();
    let span = Span::new(abs_start + leading, abs_start + leading + trimmed.len());
    if trimmed.is_empty() {
        return Err(ParseError::new(
            format!("expected {}", kind.expected()),
            Span::new(abs_start, abs_start + src.len()),
        ));
    }
    if !is_valid_output_label(trimmed) {
        return Err(ParseError::new(
            format!("expected {}, found `{trimmed}`", kind.expected()),
            span,
        ));
    }

    Ok(Ident::new(trimmed, span))
}

#[derive(Clone, Copy)]
enum LabelKind {
    Route,
    Output,
}

impl LabelKind {
    fn expected(self) -> &'static str {
        match self {
            Self::Route => "route label",
            Self::Output => "output label",
        }
    }
}

fn parse_place_at(src: &str, abs_start: usize) -> Result<Place, ParseError> {
    let mut place = parse_place_fragment(src).map_err(|error| error.shifted(abs_start))?;
    shift_place(&mut place, abs_start);
    Ok(place)
}

fn parse_expr_at(src: &str, abs_start: usize) -> Result<Expr, ParseError> {
    let mut expr = parse_expr_fragment(src).map_err(|error| error.shifted(abs_start))?;
    shift_expr(&mut expr, abs_start);
    Ok(expr)
}

fn parse_surface_rhs(src: &str, abs_start: usize) -> Result<SurfaceRhs, ParseError> {
    parse_surface_rhs_at(src, abs_start, 0)
}

fn parse_surface_rhs_at(
    src: &str,
    abs_start: usize,
    depth: usize,
) -> Result<SurfaceRhs, ParseError> {
    let trimmed = src.trim_start();
    let leading = src.len() - trimmed.len();
    if starts_with_keyword(trimmed, "if") {
        return parse_if_rhs(trimmed, abs_start + leading, depth);
    }
    Ok(SurfaceRhs::Expr(parse_expr_at(src, abs_start)?))
}

fn parse_surface_rhs_body(
    src: &str,
    abs_start: usize,
    depth: usize,
) -> Result<(SurfaceRhs, usize), ParseError> {
    let trimmed = src.trim_start();
    let leading = src.len() - trimmed.len();
    let abs = abs_start + leading;
    if trimmed.starts_with('{') {
        let close = find_matching_delimiter(trimmed, '{', '}').ok_or_else(|| {
            ParseError::new(
                "unclosed `{` in conditional expression",
                Span::new(abs, abs + trimmed.len()),
            )
        })?;
        let inner = &trimmed[1..close];
        let inner_leading = inner.len() - inner.trim_start().len();
        let inner_trimmed = &inner[inner_leading..];
        let inner_trailing = inner_trimmed.trim_end().len();
        let inner_trimmed = &inner_trimmed[..inner_trailing];
        let inner_abs = abs + 1 + inner_leading;
        let rhs = parse_surface_rhs_at(inner_trimmed, inner_abs, depth + 1)?;
        return Ok((rhs, leading + close + 1));
    }
    let rhs = parse_surface_rhs_at(trimmed, abs, depth + 1)?;
    Ok((rhs, leading + trimmed.len()))
}

fn parse_if_rhs(src: &str, abs_start: usize, depth: usize) -> Result<SurfaceRhs, ParseError> {
    if depth >= crate::parser::MAX_NESTING_DEPTH {
        return Err(ParseError::new(
            format!(
                "conditional expression is nested too deeply (maximum nesting depth is {})",
                crate::parser::MAX_NESTING_DEPTH
            ),
            Span::new(abs_start, abs_start + src.len().min(2)),
        ));
    }
    let rest = &src[2..];
    let rest_leading = rest.len() - rest.trim_start().len();
    let rest = &rest[rest_leading..];
    let rest_abs = abs_start + 2 + rest_leading;
    if !rest.starts_with('(') {
        return Err(ParseError::new(
            "expected `(` after `if` in conditional expression",
            Span::new(rest_abs, rest_abs + rest.len().min(1)),
        ));
    }

    let close = find_matching_delimiter(rest, '(', ')').ok_or_else(|| {
        ParseError::new(
            "unclosed `(` in conditional expression",
            Span::new(rest_abs, rest_abs + rest.len()),
        )
    })?;

    let condition_src = &rest[1..close];
    let remaining = &rest[close + 1..];
    let remaining_abs = rest_abs + close + 1;
    let else_index = find_top_level_keyword(remaining, "else").ok_or_else(|| {
        ParseError::new(
            "expected `else` in conditional expression",
            Span::new(remaining_abs, remaining_abs + remaining.len()),
        )
    })?;

    let condition = parse_expr_at(condition_src, rest_abs + 1)?;
    let then_src = &remaining[..else_index];
    let (then_branch, then_consumed) = parse_surface_rhs_body(then_src, remaining_abs, depth)?;
    let then_trailing_src = &then_src[then_consumed..];
    let then_trailing = then_trailing_src.trim_start();
    let then_trailing_leading = then_trailing_src.len() - then_trailing.len();
    let then_trailing_pos = remaining_abs + then_consumed + then_trailing_leading;
    if !then_trailing.is_empty() {
        return Err(ParseError::new(
            "unexpected tokens after `if`/`else` expression",
            Span::new(then_trailing_pos, then_trailing_pos + then_trailing.len()),
        ));
    }
    let else_src = &remaining[else_index + 4..];
    let else_abs = remaining_abs + else_index + 4;
    let (else_branch, else_consumed) = parse_surface_rhs_body(else_src, else_abs, depth)?;
    let trailing_src = &else_src[else_consumed..];
    let trailing = trailing_src.trim_start();
    let trailing_leading = trailing_src.len() - trailing.len();
    let trailing_pos = else_abs + else_consumed + trailing_leading;
    if !trailing.is_empty() {
        return Err(ParseError::new(
            "unexpected tokens after `if`/`else` expression",
            Span::new(trailing_pos, trailing_pos + trailing.len()),
        ));
    }
    let span = Span::new(abs_start, remaining_abs + remaining.len());

    Ok(SurfaceRhs::If {
        condition,
        then_branch: Box::new(then_branch),
        else_branch: Box::new(else_branch),
        span,
    })
}

fn build_assignment_statement(target: AssignTarget, rhs: SurfaceRhs) -> Stmt {
    match rhs {
        SurfaceRhs::Expr(value) => Stmt {
            span: target.span.join(value.span),
            kind: StmtKind::Assign(AssignStmt { target, value }),
        },
        SurfaceRhs::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => Stmt {
            span: target.span.join(span),
            kind: StmtKind::If(IfStmt {
                condition,
                then_branch: vec![build_assignment_statement(target.clone(), *then_branch)],
                else_branch: Some(vec![build_assignment_statement(target, *else_branch)]),
            }),
        },
    }
}

fn place_to_expr(place: &Place) -> Expr {
    let mut expr = Expr {
        span: place.name.span,
        kind: ExprKind::Name(place.name.clone()),
    };
    if let Some(index) = &place.index {
        expr = Expr {
            span: place.span,
            kind: ExprKind::Index {
                target: Box::new(expr),
                index: Box::new(index.clone()),
            },
        };
    }
    expr
}

fn place_to_assign_target(place: Place) -> AssignTargetKind {
    match place.index {
        Some(index) => AssignTargetKind::Index {
            target: place.name,
            index,
        },
        None => AssignTargetKind::Name(place.name),
    }
}

fn split_output_annotation(src: &str) -> (&str, Option<(&str, usize)>) {
    match src.find('~') {
        Some(index) => (&src[..index], Some((&src[index + 1..], index + 1))),
        None => (src, None),
    }
}

fn validate_output_annotation(src: &str, abs_start: usize) -> Result<(), ParseError> {
    let annotation = parse_expr_at(src, abs_start)?;
    match annotation.kind {
        ExprKind::Call { callee, args } if callee.text == "continuous" && args.is_empty() => Ok(()),
        _ => Err(ParseError::new(
            "expected the output annotation `continuous()`",
            annotation.span,
        )),
    }
}

fn expr_matches_place(expr: &Expr, destination: &Place) -> bool {
    match (&expr.kind, &destination.index) {
        (ExprKind::Name(name), None) => name.text == destination.name.text,
        (ExprKind::Index { target, index }, Some(destination_index)) => {
            matches!(&target.kind, ExprKind::Name(name) if name.text == destination.name.text)
                && **index == *destination_index
        }
        _ => false,
    }
}

fn shift_place(place: &mut Place, delta: usize) {
    place.span = place.span.shifted(delta);
    place.name.span = place.name.span.shifted(delta);
    if let Some(index) = &mut place.index {
        shift_expr(index, delta);
    }
}

fn shift_expr(expr: &mut Expr, delta: usize) {
    expr.span = expr.span.shifted(delta);
    match &mut expr.kind {
        ExprKind::Name(name) => {
            name.span = name.span.shifted(delta);
        }
        ExprKind::Unary { expr, .. } => shift_expr(expr, delta),
        ExprKind::Binary { lhs, rhs, .. } => {
            shift_expr(lhs, delta);
            shift_expr(rhs, delta);
        }
        ExprKind::Call { callee, args } => {
            callee.span = callee.span.shifted(delta);
            for arg in args {
                shift_expr(arg, delta);
            }
        }
        ExprKind::Index { target, index } => {
            shift_expr(target, delta);
            shift_expr(index, delta);
        }
        ExprKind::Number(_) | ExprKind::Bool(_) => {}
    }
}

fn normalize_interpolation_name(name: &str) -> String {
    match name {
        "cf" | "carryforward" => "carry_forward".to_string(),
        other => other.to_string(),
    }
}

fn starts_with_keyword(src: &str, keyword: &str) -> bool {
    src.strip_prefix(keyword).is_some_and(|rest| {
        rest.is_empty() || {
            let next = rest.chars().next().unwrap();
            !next.is_ascii_alphanumeric() && !rest.starts_with('_')
        }
    })
}

fn split_top_level(src: &str, delimiter: char) -> Vec<(&str, usize)> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    for (index, ch) in src.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            _ => {}
        }
        if ch == delimiter && paren_depth == 0 && bracket_depth == 0 {
            items.push((&src[start..index], start));
            start = index + ch.len_utf8();
        }
    }
    items.push((&src[start..], start));
    items
}

fn find_top_level_arrow(src: &str) -> Option<usize> {
    find_top_level_operator(src, "->")
}

fn find_top_level_assignment(src: &str) -> Option<usize> {
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let bytes = src.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] as char {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '=' if paren_depth == 0 && bracket_depth == 0 => {
                let prev = index.checked_sub(1).and_then(|idx| bytes.get(idx)).copied();
                let next = bytes.get(index + 1).copied();
                if matches!(prev, Some(b'!') | Some(b'=') | Some(b'<') | Some(b'>'))
                    || matches!(next, Some(b'='))
                {
                    index += 1;
                    continue;
                }
                return Some(index);
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn find_top_level_operator(src: &str, operator: &str) -> Option<usize> {
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let bytes = src.as_bytes();
    let op = operator.as_bytes();
    let mut index = 0;
    while index + op.len() <= bytes.len() {
        match bytes[index] as char {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            _ => {}
        }
        if paren_depth == 0 && bracket_depth == 0 && &bytes[index..index + op.len()] == op {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn find_top_level_keyword(src: &str, keyword: &str) -> Option<usize> {
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let bytes = src.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let mut index = 0;
    while index + keyword_bytes.len() <= bytes.len() {
        let byte = bytes[index];
        if in_string {
            if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            b'{' => brace_depth += 1,
            b'}' => brace_depth -= 1,
            _ => {}
        }
        if paren_depth == 0
            && bracket_depth == 0
            && brace_depth == 0
            && &bytes[index..index + keyword_bytes.len()] == keyword_bytes
        {
            let prev = index.checked_sub(1).and_then(|idx| bytes.get(idx)).copied();
            let next = bytes.get(index + keyword_bytes.len()).copied();
            let prev_ok = prev.is_none_or(|byte| !is_ident_byte(byte));
            let next_ok = next.is_none_or(|byte| !is_ident_byte(byte));
            if prev_ok && next_ok {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

const MODEL_KINDS: &[&str] = &["ode", "analytical", "sde"];

const DECLARATION_HEADERS: &[&str] = &[
    "name",
    "kind",
    "params",
    "parameters",
    "covariates",
    "states",
    "derived",
    "outputs",
    "particles",
    "structure",
];

/// Returns the candidate closest to `needle`, if it is a confident match.
fn closest_keyword(needle: &str, candidates: &[&'static str]) -> Option<&'static str> {
    let needle = needle.to_ascii_lowercase();
    candidates
        .iter()
        .filter(|candidate| **candidate != needle)
        .map(|candidate| {
            let distance = if is_single_adjacent_transposition(&needle, candidate) {
                1
            } else {
                edit_distance(&needle, candidate)
            };
            (distance, common_prefix_len(&needle, candidate), *candidate)
        })
        .filter(|(distance, prefix, candidate)| {
            is_high_confidence_match(&needle, candidate, *distance, *prefix)
        })
        .min_by_key(|(distance, prefix, _)| (*distance, std::cmp::Reverse(*prefix)))
        .map(|(_, _, candidate)| candidate)
}

fn did_you_mean(span: Span, candidate: &str) -> DiagnosticSuggestion {
    DiagnosticSuggestion {
        message: format!("did you mean `{candidate}`?"),
        edits: vec![TextEdit {
            span,
            replacement: candidate.to_string(),
        }],
        applicability: Applicability::MaybeIncorrect,
    }
}

/// Explains a top-level statement that has no `=`, `->`, or leading `if`.
fn missing_assignment_error(trimmed: &str, span: Span) -> ParseError {
    if let Some(index) = find_top_level_operator(trimmed, "==") {
        let operator = Span::new(span.start + index, span.start + index + 2);
        return ParseError::new("expected `=` in equation, found `==`", operator)
            .with_help("`==` compares two values; a single `=` assigns the right-hand side")
            .with_suggestion(DiagnosticSuggestion {
                message: "use `=` to assign".to_string(),
                edits: vec![TextEdit {
                    span: operator,
                    replacement: "=".to_string(),
                }],
                applicability: Applicability::MaybeIncorrect,
            });
    }
    ParseError::new("expected a declaration, equation, or route shorthand", span).with_help(
        "statements have the form `params = a, b`, `name = <expression>`, \
         `dx(state) = <expression>`, or `input -> state`",
    )
}

/// Replaces every `#` comment with spaces. Byte offsets are preserved, so spans
/// computed on the masked text point into the original source.
fn mask_source_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for segment in src.split_inclusive('\n') {
        let (line, newline) = match segment.strip_suffix('\n') {
            Some(line) => (line, "\n"),
            None => (segment, ""),
        };
        let cutoff = line.find('#').unwrap_or(line.len());
        out.push_str(&line[..cutoff]);
        out.push_str(&" ".repeat(line.len() - cutoff));
        out.push_str(newline);
    }
    out
}

/// Operators that, at the end of a line, continue the statement on the next
/// line. `=` is deliberately absent so a missing right-hand side is reported at
/// its own line instead of swallowing the next statement.
const TRAILING_CONTINUATIONS: &[&str] = &[
    "+", "-", "*", "/", "^", ",", "&&", "||", "<", ">", "<=", ">=", "==", "!=",
];

/// Operators that, at the start of a line, continue the previous statement.
/// No statement can begin with one of these.
const LEADING_CONTINUATIONS: &[&str] = &[
    "+", "-", "*", "/", "^", ",", "&&", "||", "<", ">", "==", "!=",
];

/// Splits comment-masked authoring source into the trimmed byte ranges of its
/// logical statements.
///
/// A statement ends at a top-level `;` or newline. It continues past a newline
/// while a `{` block is open, while a `(` or `[` group is open and the next line
/// is indented further than the statement's first line or closes the group,
/// when the line ends with a binary operator or comma and the next line is
/// indented further, and when the next line starts with a binary operator or
/// `else`. Otherwise the next line starts a new statement, so an unclosed group
/// or dangling operator is reported on its own line instead of absorbing the
/// statement after it. The same rules apply inside `if` bodies.
fn split_logical_statements(src: &str) -> Vec<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut group_depth = 0i32;
    let mut brace_depth = 0i32;
    for (index, byte) in bytes.iter().enumerate() {
        let boundary = match byte {
            b'(' | b'[' => {
                group_depth += 1;
                false
            }
            b')' | b']' => {
                group_depth -= 1;
                false
            }
            b'{' => {
                brace_depth += 1;
                false
            }
            b'}' => {
                brace_depth -= 1;
                false
            }
            b';' => group_depth <= 0 && brace_depth <= 0,
            b'\n' => !continues_after_newline(src, start, index, group_depth, brace_depth),
            _ => false,
        };
        if boundary {
            push_trimmed_range(src, start, index, &mut ranges);
            start = index + 1;
            group_depth = 0;
            brace_depth = 0;
        }
    }
    push_trimmed_range(src, start, src.len(), &mut ranges);
    ranges
}

fn continues_after_newline(
    src: &str,
    start: usize,
    newline: usize,
    group_depth: i32,
    brace_depth: i32,
) -> bool {
    if brace_depth > 0 {
        return true;
    }
    let Some(next_line) = next_nonblank_line(src, newline + 1) else {
        return false;
    };
    let next = next_line.trim_start();
    // Continuation lines are indented past the line the statement starts on.
    let statement_line = &src[start..newline];
    let first = start + statement_line.len() - statement_line.trim_start().len();
    let indented = indentation(next_line) > indentation(line_at(src, first));
    if group_depth > 0 {
        return indented || next.starts_with([')', ']']);
    }
    let current = src[start..newline].trim_end();
    (indented
        && !current.trim_start().is_empty()
        && TRAILING_CONTINUATIONS
            .iter()
            .any(|operator| current.ends_with(operator)))
        || starts_with_keyword(next, "else")
        || LEADING_CONTINUATIONS
            .iter()
            .any(|operator| next.starts_with(operator))
}

fn line_at(src: &str, offset: usize) -> &str {
    let start = src[..offset].rfind('\n').map_or(0, |index| index + 1);
    let end = src[offset..]
        .find('\n')
        .map_or(src.len(), |index| offset + index);
    &src[start..end]
}

fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn next_nonblank_line(src: &str, from: usize) -> Option<&str> {
    src[from..].split('\n').find(|line| !line.trim().is_empty())
}

fn push_trimmed_range(src: &str, start: usize, end: usize, ranges: &mut Vec<(usize, usize)>) {
    let text = &src[start..end];
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    let leading = text.len() - text.trim_start().len();
    ranges.push((start + leading, start + leading + trimmed.len()));
}

fn find_matching_delimiter(src: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    for (index, ch) in src.char_indices() {
        if in_string {
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn is_valid_ident(src: &str) -> bool {
    let mut chars = src.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn is_valid_output_label(src: &str) -> bool {
    is_valid_ident(src) || src.chars().all(|ch| ch.is_ascii_digit())
}

fn is_ident_byte(byte: u8) -> bool {
    (byte as char).is_ascii_alphanumeric() || byte == b'_'
}

fn join_ident_spans(items: &[Ident]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}

fn join_binding_spans(items: &[Binding]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}

fn join_covariate_spans(items: &[CovariateDecl]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}

fn output_statement_name(statement: &Stmt) -> Option<&str> {
    match &statement.kind {
        StmtKind::Assign(assign) => match &assign.target.kind {
            AssignTargetKind::Name(name) => Some(name.text.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn collect_named_assign_targets(statements: &[Stmt], names: &mut BTreeMap<String, Span>) {
    for statement in statements {
        match &statement.kind {
            StmtKind::Assign(assign) => {
                if let AssignTargetKind::Name(name) = &assign.target.kind {
                    names.entry(name.text.clone()).or_insert(name.span);
                }
            }
            StmtKind::If(if_stmt) => {
                collect_named_assign_targets(&if_stmt.then_branch, names);
                if let Some(branch) = &if_stmt.else_branch {
                    collect_named_assign_targets(branch, names);
                }
            }
            StmtKind::For(for_stmt) => collect_named_assign_targets(&for_stmt.body, names),
            _ => {}
        }
    }
}

fn join_state_spans(items: &[StateDecl]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}

fn join_route_spans(items: &[RouteDecl]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}

fn join_stmt_spans(items: &[Stmt]) -> Span {
    items
        .iter()
        .map(|item| item.span)
        .reduce(Span::join)
        .unwrap_or_else(|| Span::empty(0))
}
