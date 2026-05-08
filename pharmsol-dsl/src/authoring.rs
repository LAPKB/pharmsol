use std::collections::{BTreeMap, BTreeSet};

use super::ast::*;
use super::diagnostic::{Applicability, DiagnosticSuggestion, ParseError, Span, TextEdit};
use super::parser::{parse_expr_fragment, parse_place_fragment};

const DEFAULT_MODEL_NAME: &str = "main";
const NUMERIC_ROUTE_PREFIX: &str = "input_";
const NUMERIC_OUTPUT_PREFIX: &str = "outeq_";

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
    declared_derived: BTreeSet<String>,
    declared_outputs: BTreeSet<String>,
    explicit_output_order: Vec<String>,
    explicit_outputs: BTreeMap<String, Span>,
    assigned_outputs: BTreeMap<String, Span>,
    declared_outputs_span: Option<Span>,
    routes: BTreeMap<String, SurfaceRoute>,
    route_order: Vec<String>,
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

#[derive(Clone, Copy, PartialEq, Eq)]
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
            declared_derived: BTreeSet::new(),
            declared_outputs: BTreeSet::new(),
            explicit_output_order: Vec::new(),
            explicit_outputs: BTreeMap::new(),
            assigned_outputs: BTreeMap::new(),
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
        let mut offset = 0;
        for segment in self.src.split_inclusive('\n') {
            let line = segment.strip_suffix('\n').unwrap_or(segment);
            self.parse_line(line, offset)?;
            offset += segment.len();
        }

        if !self.src.is_empty() && !self.src.ends_with('\n') && offset < self.src.len() {
            self.parse_line(&self.src[offset..], offset)?;
        }

        self.validate_declared_outputs_assigned()?;

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

        let surface_routes = std::mem::take(&mut self.routes);
        let route_order = std::mem::take(&mut self.route_order);
        let mut route_modifiers = std::mem::take(&mut self.route_modifiers);
        let mut routes = Vec::with_capacity(surface_routes.len());
        for route_name in route_order {
            let Some(route) = surface_routes.get(&route_name) else {
                continue;
            };
            let mut span = route.span;
            let properties = route_modifiers.remove(&route_name).unwrap_or_default();
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

        let kind = self.determine_kind(module_span)?;
        if matches!(kind, ModelKind::Analytical) && !self.derivative_statements.is_empty() {
            return Err(ParseError::new(
                "analytical authoring models cannot declare `dx(...)` equations",
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

        let mut derivative_statements = std::mem::take(&mut self.derivative_statements);
        inject_infusion_rates(&surface_routes, &routes, &mut derivative_statements);

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
        if leading == trailing {
            return Ok(());
        }

        let trimmed = &code[leading..trailing];
        let span = Span::new(line_offset + leading, line_offset + trailing);
        self.note_span(span);

        if find_top_level_arrow(trimmed).is_some() {
            return self.parse_route_line(trimmed, span.start, span);
        }

        let eq_index = find_top_level_assignment(trimmed).ok_or_else(|| {
            ParseError::new(
                "expected an authoring declaration, equation, or route shorthand",
                span,
            )
        })?;

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
                    return Err(ParseError::new(
                        format!("unknown model kind `{other}`"),
                        kind_ident.span,
                    ))
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
                self.declared_derived.insert(ident.text);
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

        if lhs_trimmed == "kernel" {
            return Err(ParseError::new(
                "`kernel = ...` has been renamed to `structure = ...`",
                span,
            ));
        }

        if lhs_trimmed == "structure" {
            let structure = parse_ident_segment(rhs, rhs_abs)?;
            self.analytical = Some(AnalyticalBlock { span, structure });
            return Ok(());
        }

        if let Some(name_segment) = lhs_trimmed.strip_prefix("const ") {
            let name_abs = span.start + (lhs.find("const").unwrap() + "const ".len());
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
            self.note_output_assignment(&target);
            self.output_statements.push(stmt);
        } else {
            self.derive_statements.push(stmt);
        }
        Ok(())
    }

    fn parse_route_line(
        &mut self,
        trimmed: &str,
        line_start: usize,
        span: Span,
    ) -> Result<(), ParseError> {
        let arrow = find_top_level_arrow(trimmed).unwrap();
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
        if self.routes.contains_key(&route_name) {
            return Err(ParseError::new(
                format!("duplicate route `{}`", input.text),
                input.span,
            ));
        }
        self.routes.insert(
            route_name.clone(),
            SurfaceRoute {
                input,
                destination,
                kind,
                span,
            },
        );
        self.route_order.push(route_name);
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
                self.note_output_assignment(&output);

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
                    format!("unsupported authoring equation target `{other}`"),
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
                "analytical authoring models cannot declare particles or noise equations",
                kind_span,
            ));
        }

        if matches!(kind, ModelKind::Ode) && !self.diffusion_statements.is_empty() {
            return Err(ParseError::new(
                "ODE authoring models cannot declare `noise(...)` equations",
                self.diffusion_statements[0].span,
            ));
        }

        if matches!(kind, ModelKind::Sde) {
            if let Some(analytical) = &self.analytical {
                return Err(ParseError::new(
                    "SDE authoring models cannot declare an analytical structure",
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

    fn note_output_assignment(&mut self, output: &Ident) {
        self.assigned_outputs
            .entry(output.text.clone())
            .or_insert(output.span);
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

fn inject_infusion_rates(
    surface_routes: &BTreeMap<String, SurfaceRoute>,
    routes: &[RouteDecl],
    derivative_statements: &mut Vec<Stmt>,
) {
    for route in routes {
        let Some(surface_route) = surface_routes.get(&route.input.text) else {
            continue;
        };
        if surface_route.kind != SurfaceRouteKind::Infusion {
            continue;
        }

        let rate_expr = Expr {
            span: surface_route.span,
            kind: ExprKind::Call {
                callee: Ident::new("rate", surface_route.input.span),
                args: vec![Expr {
                    span: surface_route.input.span,
                    kind: ExprKind::Name(surface_route.input.clone()),
                }],
            },
        };

        if !augment_derivative_statements(derivative_statements, &route.destination, &rate_expr) {
            derivative_statements.push(Stmt {
                span: surface_route.span,
                kind: StmtKind::Assign(AssignStmt {
                    target: AssignTarget {
                        span: surface_route.span,
                        kind: AssignTargetKind::Call {
                            callee: Ident::new("ddt", surface_route.span),
                            args: vec![place_to_expr(&route.destination)],
                        },
                    },
                    value: rate_expr,
                }),
            });
        }
    }
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
            "expected `)` to close call-style authoring target",
            Span::new(abs_start + leading + open, abs_start + src.len()),
        ));
    };
    if !trimmed[close + 1..].trim().is_empty() {
        return Err(ParseError::new(
            "unexpected trailing tokens after authoring target",
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

    if let Some(suffix) = bare_numeric_label(trimmed) {
        let replacement = kind.canonical_label(suffix);
        return Err(ParseError::new(
            format!(
                "bare numeric {} labels are not allowed in the DSL; use `{replacement}` instead",
                kind.noun()
            ),
            span,
        )
        .with_help(format!(
            "numeric {} labels must use the `{}` prefix in authored DSL",
            kind.noun(),
            kind.prefix_pattern()
        ))
        .with_suggestion(DiagnosticSuggestion {
            message: format!("use `{replacement}`"),
            edits: vec![TextEdit { span, replacement }],
            applicability: Applicability::Always,
        }));
    }

    if let Some(suffix) = canonical_numeric_suffix(trimmed, kind.wrong_prefix()) {
        let replacement = kind.canonical_label(suffix);
        return Err(ParseError::new(
            format!(
                "`{trimmed}` is {} label and cannot be used as {}; use `{replacement}` here",
                kind.wrong_kind_phrase(),
                kind.noun_phrase()
            ),
            span,
        )
        .with_help(format!(
            "numeric {} labels use the `{}` prefix",
            kind.noun(),
            kind.prefix_pattern()
        ))
        .with_suggestion(DiagnosticSuggestion {
            message: format!("use `{replacement}`"),
            edits: vec![TextEdit { span, replacement }],
            applicability: Applicability::Always,
        }));
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

    fn noun(self) -> &'static str {
        match self {
            Self::Route => "route",
            Self::Output => "output",
        }
    }

    fn noun_phrase(self) -> &'static str {
        match self {
            Self::Route => "a route",
            Self::Output => "an output",
        }
    }

    fn wrong_kind_phrase(self) -> &'static str {
        match self {
            Self::Route => "an output",
            Self::Output => "a route",
        }
    }

    fn canonical_label(self, suffix: &str) -> String {
        match self {
            Self::Route => format!("{NUMERIC_ROUTE_PREFIX}{suffix}"),
            Self::Output => format!("{NUMERIC_OUTPUT_PREFIX}{suffix}"),
        }
    }

    fn wrong_prefix(self) -> &'static str {
        match self {
            Self::Route => NUMERIC_OUTPUT_PREFIX,
            Self::Output => NUMERIC_ROUTE_PREFIX,
        }
    }

    fn prefix_pattern(self) -> &'static str {
        match self {
            Self::Route => "input_<n>",
            Self::Output => "outeq_<n>",
        }
    }
}

fn bare_numeric_label(src: &str) -> Option<&str> {
    (!src.is_empty() && src.chars().all(|ch| ch.is_ascii_digit())).then_some(src)
}

fn canonical_numeric_suffix<'a>(src: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = src.strip_prefix(prefix)?;
    (!suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit())).then_some(suffix)
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
    let trimmed = src.trim_start();
    let leading = src.len() - trimmed.len();
    if starts_with_keyword(trimmed, "if") {
        return parse_if_rhs(trimmed, abs_start + leading);
    }
    Ok(SurfaceRhs::Expr(parse_expr_at(src, abs_start)?))
}

fn parse_if_rhs(src: &str, abs_start: usize) -> Result<SurfaceRhs, ParseError> {
    let rest = &src[2..];
    let rest_leading = rest.len() - rest.trim_start().len();
    let rest = &rest[rest_leading..];
    let rest_abs = abs_start + 2 + rest_leading;
    if !rest.starts_with('(') {
        return Err(ParseError::new(
            "expected `(` after `if` in authoring conditional expression",
            Span::new(rest_abs, rest_abs + rest.len().min(1)),
        ));
    }

    let close = find_matching_delimiter(rest, '(', ')').ok_or_else(|| {
        ParseError::new(
            "unclosed `(` in authoring conditional expression",
            Span::new(rest_abs, rest_abs + rest.len()),
        )
    })?;

    let condition_src = &rest[1..close];
    let remaining = &rest[close + 1..];
    let remaining_abs = rest_abs + close + 1;
    let else_index = find_top_level_keyword(remaining, "else").ok_or_else(|| {
        ParseError::new(
            "expected `else` in authoring conditional expression",
            Span::new(remaining_abs, remaining_abs + remaining.len()),
        )
    })?;

    let condition = parse_expr_at(condition_src, rest_abs + 1)?;
    let then_branch = parse_surface_rhs(&remaining[..else_index], remaining_abs)?;
    let else_branch =
        parse_surface_rhs(&remaining[else_index + 4..], remaining_abs + else_index + 4)?;
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

fn is_high_confidence_match(needle: &str, candidate: &str, distance: usize, prefix: usize) -> bool {
    let max_len = needle.len().max(candidate.len());
    let max_distance = match max_len {
        0..=4 => 1,
        5..=8 => 2,
        _ => 3,
    };
    distance <= max_distance && (prefix > 0 || distance <= 1)
}

fn common_prefix_len(lhs: &str, rhs: &str) -> usize {
    lhs.chars()
        .zip(rhs.chars())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count()
}

fn is_single_adjacent_transposition(lhs: &str, rhs: &str) -> bool {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.len() != rhs.len() {
        return false;
    }

    let differing = lhs
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter_map(|(index, (lhs, rhs))| (lhs != rhs).then_some(index))
        .collect::<Vec<_>>();

    if differing.len() != 2 || differing[1] != differing[0] + 1 {
        return false;
    }

    let first = differing[0];
    lhs[first] == rhs[first + 1] && lhs[first + 1] == rhs[first]
}

fn edit_distance(lhs: &str, rhs: &str) -> usize {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.is_empty() {
        return rhs.len();
    }
    if rhs.is_empty() {
        return lhs.len();
    }

    let mut previous: Vec<usize> = (0..=rhs.len()).collect();
    let mut current = vec![0; rhs.len() + 1];

    for (lhs_index, lhs_char) in lhs.iter().enumerate() {
        current[0] = lhs_index + 1;
        for (rhs_index, rhs_char) in rhs.iter().enumerate() {
            let substitution_cost = usize::from(lhs_char != rhs_char);
            current[rhs_index + 1] = (current[rhs_index] + 1)
                .min(previous[rhs_index + 1] + 1)
                .min(previous[rhs_index] + substitution_cost);
        }
        previous.clone_from_slice(&current);
    }

    previous[rhs.len()]
}

fn augment_derivative_statements(
    statements: &mut [Stmt],
    destination: &Place,
    rate_expr: &Expr,
) -> bool {
    let mut matched = false;
    for stmt in statements {
        match &mut stmt.kind {
            StmtKind::Assign(assign) if derivative_target_matches(&assign.target, destination) => {
                let value = Expr {
                    span: assign.value.span.join(rate_expr.span),
                    kind: ExprKind::Binary {
                        op: BinaryOp::Add,
                        lhs: Box::new(assign.value.clone()),
                        rhs: Box::new(rate_expr.clone()),
                    },
                };
                assign.value = value;
                stmt.span = stmt.span.join(rate_expr.span);
                matched = true;
            }
            StmtKind::If(if_stmt) => {
                let then_match =
                    augment_derivative_statements(&mut if_stmt.then_branch, destination, rate_expr);
                let else_match = if_stmt.else_branch.as_mut().is_some_and(|branch| {
                    augment_derivative_statements(branch, destination, rate_expr)
                });
                matched |= then_match || else_match;
            }
            StmtKind::For(for_stmt) => {
                matched |=
                    augment_derivative_statements(&mut for_stmt.body, destination, rate_expr);
            }
            _ => {}
        }
    }
    matched
}

fn derivative_target_matches(target: &AssignTarget, destination: &Place) -> bool {
    match &target.kind {
        AssignTargetKind::Call { callee, args } if callee.text == "ddt" && args.len() == 1 => {
            expr_matches_place(&args[0], destination)
        }
        _ => false,
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
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let bytes = src.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let mut index = 0;
    while index + keyword_bytes.len() <= bytes.len() {
        match bytes[index] as char {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            _ => {}
        }
        if paren_depth == 0
            && bracket_depth == 0
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

fn find_matching_delimiter(src: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0;
    for (index, ch) in src.char_indices() {
        if ch == open {
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
