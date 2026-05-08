use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::ast as syntax;
use crate::diagnostic::{
    Applicability, Diagnostic, DiagnosticPhase, DiagnosticReport, DiagnosticSuggestion, Span,
    TextEdit, DSL_SEMANTIC_GENERIC,
};
use crate::ir::*;
use crate::ModelKind;

const RESERVED_NAMES: &[&str] = &[
    "abs",
    "bioavailability",
    "carry_forward",
    "ceil",
    "ddt",
    "exp",
    "floor",
    "lag",
    "linear",
    "ln",
    "locf",
    "log",
    "log10",
    "log2",
    "max",
    "min",
    "noise",
    "pow",
    "rate",
    "round",
    "sin",
    "cos",
    "tan",
    "sqrt",
];

const RATE_FUNCTION_NAME: &str = "rate";
const NUMERIC_ROUTE_PREFIX: &str = "input_";
const NUMERIC_OUTPUT_PREFIX: &str = "outeq_";

#[derive(Default)]
struct SemanticAssist {
    context_labels: Vec<(Span, String)>,
    secondary_labels: Vec<(Span, String)>,
    helps: Vec<String>,
    suggestions: Vec<DiagnosticSuggestion>,
}

impl SemanticAssist {
    fn context_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.context_labels.push((span, message.into()));
        self
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.helps.push(help.into());
        self
    }

    fn replacement_suggestion(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
        applicability: Applicability,
    ) -> Self {
        self.suggestions.push(DiagnosticSuggestion {
            message: message.into(),
            edits: vec![TextEdit {
                span,
                replacement: replacement.into(),
            }],
            applicability,
        });
        self
    }

    fn apply(self, mut error: SemanticError) -> SemanticError {
        for (span, message) in self.context_labels {
            error = error.with_context_label(span, message);
        }
        for (span, message) in self.secondary_labels {
            error = error.with_secondary_label(span, message);
        }
        for help in self.helps {
            error = error.with_help(help);
        }
        for suggestion in self.suggestions {
            error = error.with_suggestion(suggestion);
        }
        error
    }
}

struct SimilarNameCandidate {
    lookup_name: String,
    assist: SemanticAssist,
}

impl SimilarNameCandidate {
    fn new(lookup_name: impl Into<String>, assist: SemanticAssist) -> Self {
        Self {
            lookup_name: lookup_name.into(),
            assist,
        }
    }
}

pub fn analyze_module(module: &syntax::Module) -> Result<TypedModule, SemanticError> {
    let mut models = Vec::with_capacity(module.models.len());
    for model in &module.models {
        models.push(analyze_model(model)?);
    }
    Ok(TypedModule {
        models,
        span: module.span,
    })
}

pub fn analyze_model(model: &syntax::Model) -> Result<TypedModel, SemanticError> {
    Analyzer::new(model).analyze()
}

#[derive(Clone, PartialEq, Eq)]
pub struct SemanticError {
    diagnostic: Box<Diagnostic>,
    source: Option<Arc<str>>,
}

impl SemanticError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            diagnostic: Box::new(Diagnostic::error(
                DSL_SEMANTIC_GENERIC,
                DiagnosticPhase::Semantic,
                message,
                span,
            )),
            source: None,
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.diagnostic.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.diagnostic.helps.push(help.into());
        self
    }

    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic = Box::new(self.diagnostic.with_secondary_label(span, message));
        self
    }

    pub fn with_context_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic = Box::new(self.diagnostic.with_context_label(span, message));
        self
    }

    pub fn with_suggestion(mut self, suggestion: DiagnosticSuggestion) -> Self {
        self.diagnostic = Box::new(self.diagnostic.with_suggestion(suggestion));
        self
    }

    pub fn diagnostic(&self) -> &Diagnostic {
        self.diagnostic.as_ref()
    }

    pub fn into_diagnostic(self) -> Diagnostic {
        *self.diagnostic
    }

    pub fn render(&self, src: &str) -> String {
        self.diagnostic.render(src)
    }

    pub fn diagnostic_report(&self, source_name: impl Into<String>) -> DiagnosticReport {
        DiagnosticReport::from_diagnostics(
            source_name,
            self.source(),
            std::slice::from_ref(self.diagnostic.as_ref()),
        )
    }

    pub fn with_source(mut self, source: impl Into<Arc<str>>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
}

impl fmt::Debug for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = self.source() {
            return f.write_str(&self.render(source));
        }
        let span = self.diagnostic.primary_span();
        write!(
            f,
            "{} at bytes {}..{}",
            self.diagnostic.message, span.start, span.end
        )
    }
}

impl std::error::Error for SemanticError {}

struct Analyzer<'a> {
    model: &'a syntax::Model,
    symbols: Vec<PendingSymbol>,
    globals: Globals,
}

impl<'a> Analyzer<'a> {
    fn new(model: &'a syntax::Model) -> Self {
        Self {
            model,
            symbols: Vec::new(),
            globals: Globals::default(),
        }
    }

    fn analyze(mut self) -> Result<TypedModel, SemanticError> {
        let sections = ModelSections::from_model(self.model)?;

        let parameters = self.register_parameters(sections.parameters)?;
        let constants = self.resolve_and_register_constants(sections.constants)?;
        let covariates = self.register_covariates(sections.covariates)?;
        let states = self.register_states(sections.states)?;
        let routes = self.register_routes(sections.routes)?;

        let derived = self.register_implicit_symbols(
            sections.derive.map(|block| block.statements.as_slice()),
            SymbolKind::Derived,
        )?;
        let outputs = self.register_implicit_symbols(
            Some(
                &sections
                    .outputs
                    .ok_or_else(|| {
                        SemanticError::new(
                            format!(
                                "model `{}` is missing an `outputs` block",
                                self.model.name.text
                            ),
                            self.model.span,
                        )
                    })?
                    .statements,
            ),
            SymbolKind::Output,
        )?;

        self.validate_kind_requirements(&sections, &states)?;

        let derive_result = if let Some(block) = sections.derive {
            Some(self.analyze_statement_block(block, BlockContext::Derive, BTreeSet::new())?)
        } else {
            None
        };
        let available_derived = derive_result
            .as_ref()
            .map(|result| result.available_derived.clone())
            .unwrap_or_default();

        let dynamics = if let Some(block) = sections.dynamics {
            Some(self.analyze_statement_block(
                block,
                BlockContext::Dynamics,
                available_derived.clone(),
            )?)
        } else {
            None
        };
        let init = if let Some(block) = sections.init {
            Some(self.analyze_statement_block(
                block,
                BlockContext::Init,
                available_derived.clone(),
            )?)
        } else {
            None
        };
        let drift = if let Some(block) = sections.drift {
            Some(self.analyze_statement_block(
                block,
                BlockContext::Drift,
                available_derived.clone(),
            )?)
        } else {
            None
        };
        let diffusion = if let Some(block) = sections.diffusion {
            Some(self.analyze_statement_block(
                block,
                BlockContext::Diffusion,
                available_derived.clone(),
            )?)
        } else {
            None
        };
        let outputs_block = self.analyze_statement_block(
            sections.outputs.expect("outputs checked above"),
            BlockContext::Outputs,
            available_derived,
        )?;

        self.validate_kind_blocks(
            self.model.kind,
            ModelKindBlocks {
                dynamics: dynamics.as_ref(),
                drift: drift.as_ref(),
                diffusion: diffusion.as_ref(),
                analytical: sections.analytical,
                particles: sections.particles,
            },
            &states,
        )?;

        self.validate_output_assignments(&outputs, &outputs_block)?;
        if let Some(result) = &dynamics {
            self.validate_state_coverage(result, &states, "dynamics")?;
        }
        if let Some(result) = &drift {
            self.validate_state_coverage(result, &states, "drift")?;
        }

        let particles = if let Some(decl) = sections.particles {
            Some(self.expect_const_usize(&decl.value, "particles", true)?)
        } else {
            None
        };

        let analytical = if let Some(block) = sections.analytical {
            let structure =
                AnalyticalKernel::from_name(&block.structure.text).ok_or_else(|| {
                    SemanticError::new(
                        format!("unknown analytical structure `{}`", block.structure.text),
                        block.structure.span,
                    )
                })?;
            let state_components = states
                .iter()
                .map(|state| state.size.unwrap_or(1))
                .sum::<usize>();
            if state_components != structure.state_count() {
                return Err(SemanticError::new(
                    format!(
                        "analytical structure `{}` expects {} state value(s), but model declares {}",
                        block.structure.text,
                        structure.state_count(),
                        state_components
                    ),
                    block.structure.span,
                ));
            }
            Some(TypedAnalytical {
                structure,
                span: block.span,
            })
        } else {
            None
        };

        let model_name = self.model.name.text.clone();
        let model_kind = self.model.kind;
        let model_span = self.model.span;
        let symbols = self.finalize_symbols()?;
        Ok(TypedModel {
            name: model_name,
            kind: model_kind,
            symbols,
            parameters,
            constants,
            covariates,
            states,
            routes,
            derived,
            outputs,
            particles,
            analytical,
            derive: derive_result.map(|result| result.block),
            dynamics: dynamics.map(|result| result.block),
            outputs_block: outputs_block.block,
            init: init.map(|result| result.block),
            drift: drift.map(|result| result.block),
            diffusion: diffusion.map(|result| result.block),
            span: model_span,
        })
    }

    fn register_parameters(
        &mut self,
        block: Option<&syntax::ParametersBlock>,
    ) -> Result<Vec<SymbolId>, SemanticError> {
        let mut parameters = Vec::new();
        if let Some(block) = block {
            for ident in &block.items {
                let id = self.insert_global_symbol(
                    &ident.text,
                    SymbolKind::Parameter,
                    PendingSymbolType::Scalar(Some(ValueType::Real)),
                    ident.span,
                )?;
                self.globals.parameters.insert(ident.text.clone(), id);
                parameters.push(id);
            }
        }
        Ok(parameters)
    }

    fn resolve_and_register_constants(
        &mut self,
        block: Option<&syntax::ConstantsBlock>,
    ) -> Result<Vec<TypedConstant>, SemanticError> {
        let Some(block) = block else {
            return Ok(Vec::new());
        };

        let mut bindings = BTreeMap::new();
        for binding in &block.items {
            if let Some(existing) = bindings.insert(binding.name.text.clone(), binding) {
                return Err(SemanticAssist::default()
                    .context_label(
                        existing.name.span,
                        format!("constant `{}` first declared here", binding.name.text),
                    )
                    .help(format!(
                        "rename this constant to a unique name such as `{}_2`",
                        binding.name.text
                    ))
                    .replacement_suggestion(
                        binding.name.span,
                        format!("{}_2", binding.name.text),
                        format!("rename this constant to `{}_2`", binding.name.text),
                        Applicability::MaybeIncorrect,
                    )
                    .apply(SemanticError::new(
                        format!("duplicate constant `{}`", binding.name.text),
                        binding.name.span,
                    )));
            }
        }

        let mut visiting = BTreeSet::new();
        let mut typed = Vec::new();
        for binding in &block.items {
            let value = self.evaluate_const_expr(&binding.value, &bindings, &mut visiting)?;
            let id = self.insert_global_symbol(
                &binding.name.text,
                SymbolKind::Constant,
                PendingSymbolType::Scalar(Some(value.value_type())),
                binding.name.span,
            )?;
            self.globals.constants.insert(binding.name.text.clone(), id);
            self.globals
                .constant_values
                .insert(binding.name.text.clone(), value.clone());
            typed.push(TypedConstant {
                symbol: id,
                value,
                span: binding.span,
            });
        }
        Ok(typed)
    }

    fn register_covariates(
        &mut self,
        block: Option<&syntax::CovariatesBlock>,
    ) -> Result<Vec<TypedCovariate>, SemanticError> {
        let mut covariates = Vec::new();
        if let Some(block) = block {
            for covariate in &block.items {
                let interpolation = match covariate
                    .interpolation
                    .as_ref()
                    .map(|value| value.text.as_str())
                {
                    None => None,
                    Some("linear") => Some(CovariateInterpolation::Linear),
                    Some("locf") | Some("carry_forward") => Some(CovariateInterpolation::Locf),
                    Some(other) => {
                        return Err(SemanticError::new(
                            format!("unknown covariate interpolation `{other}`"),
                            covariate.interpolation.as_ref().unwrap().span,
                        )
                        .with_note("supported interpolation names are `linear`, `locf`, and `carry_forward`"));
                    }
                };
                let id = self.insert_global_symbol(
                    &covariate.name.text,
                    SymbolKind::Covariate,
                    PendingSymbolType::Scalar(Some(ValueType::Real)),
                    covariate.name.span,
                )?;
                self.globals
                    .covariates
                    .insert(covariate.name.text.clone(), id);
                covariates.push(TypedCovariate {
                    symbol: id,
                    interpolation,
                    span: covariate.span,
                });
            }
        }
        Ok(covariates)
    }

    fn register_states(
        &mut self,
        block: Option<&syntax::StatesBlock>,
    ) -> Result<Vec<TypedState>, SemanticError> {
        let Some(block) = block else {
            return Err(SemanticError::new(
                format!(
                    "model `{}` is missing a `states` block",
                    self.model.name.text
                ),
                self.model.span,
            ));
        };

        let mut states = Vec::new();
        for state in &block.items {
            let size = match &state.size {
                Some(expr) => Some(self.expect_const_usize(expr, "state array size", true)?),
                None => None,
            };
            let pending_type = match size {
                Some(size) => PendingSymbolType::Array {
                    element: ValueType::Real,
                    size,
                },
                None => PendingSymbolType::Scalar(Some(ValueType::Real)),
            };
            let id = self.insert_global_symbol(
                &state.name.text,
                SymbolKind::State,
                pending_type,
                state.name.span,
            )?;
            self.globals
                .states
                .insert(state.name.text.clone(), StateEntry { symbol: id, size });
            states.push(TypedState {
                symbol: id,
                size,
                span: state.span,
            });
        }
        Ok(states)
    }

    fn register_routes(
        &mut self,
        block: Option<&syntax::RoutesBlock>,
    ) -> Result<Vec<TypedRoute>, SemanticError> {
        let mut routes = Vec::new();
        if let Some(block) = block {
            for route in &block.routes {
                self.validate_route_label_name(&route.input)?;
                let id = self.insert_global_symbol(
                    &route.input.text,
                    SymbolKind::Route,
                    PendingSymbolType::Route,
                    route.input.span,
                )?;
                self.globals.routes.insert(route.input.text.clone(), id);
                let destination = self.analyze_state_place_const(&route.destination)?;
                let mut seen_props = BTreeMap::new();
                let mut properties = Vec::new();
                for property in &route.properties {
                    let kind = match property.name.text.as_str() {
                        "lag" => RoutePropertyKind::Lag,
                        "bioavailability" => RoutePropertyKind::Bioavailability,
                        other => {
                            return Err(SemanticError::new(
                                format!("unknown route property `{other}`"),
                                property.name.span,
                            )
                            .with_note(
                                "supported route properties are `lag` and `bioavailability`",
                            ));
                        }
                    };
                    if let Some(existing_span) = seen_props.insert(kind, property.name.span) {
                        return Err(SemanticAssist::default()
                            .context_label(
                                existing_span,
                                format!(
                                    "route property `{}` first declared here",
                                    property.name.text
                                ),
                            )
                            .help(format!(
                                "each route can declare `{}` at most once",
                                property.name.text
                            ))
                            .apply(SemanticError::new(
                                format!("duplicate route property `{}`", property.name.text),
                                property.name.span,
                            )));
                    }
                    let env = BlockEnv::new(BTreeSet::new());
                    let value = self.analyze_expr(&property.value, &env)?;
                    self.expect_numeric(&value, "route property", property.value.span)?;
                    properties.push(TypedRouteProperty {
                        kind,
                        value,
                        span: property.span,
                    });
                }
                routes.push(TypedRoute {
                    symbol: id,
                    kind: route.kind,
                    destination,
                    properties,
                    span: route.span,
                });
            }
        }
        Ok(routes)
    }

    fn register_implicit_symbols(
        &mut self,
        statements: Option<&[syntax::Stmt]>,
        kind: SymbolKind,
    ) -> Result<Vec<SymbolId>, SemanticError> {
        let mut collected_idents = Vec::new();
        let Some(statements) = statements else {
            return Ok(Vec::new());
        };

        let mut seen = BTreeSet::new();
        collect_bare_assignment_names(statements, &mut seen, &mut collected_idents);
        let mut symbols = Vec::new();
        for ident in collected_idents {
            if matches!(kind, SymbolKind::Output) {
                self.validate_output_label_name(&ident)?;
            }
            let id = self.insert_global_symbol(
                &ident.text,
                kind,
                PendingSymbolType::Scalar(None),
                ident.span,
            )?;
            match kind {
                SymbolKind::Derived => {
                    self.globals.derived.insert(ident.text.clone(), id);
                }
                SymbolKind::Output => {
                    self.globals.outputs.insert(ident.text.clone(), id);
                }
                _ => unreachable!(),
            }
            symbols.push(id);
        }
        Ok(symbols)
    }

    fn analyze_statement_block(
        &mut self,
        block: &syntax::StatementBlock,
        context: BlockContext,
        available_derived: BTreeSet<SymbolId>,
    ) -> Result<BlockAnalysis, SemanticError> {
        let env = BlockEnv::new(available_derived);
        let (statements, env, touched_states) =
            self.analyze_stmt_list(&block.statements, context, env)?;
        Ok(BlockAnalysis {
            block: TypedStatementBlock {
                context,
                statements,
                span: block.span,
            },
            available_derived: env.available_derived,
            definite_targets: env.definite_targets,
            touched_states,
        })
    }

    fn analyze_stmt_list(
        &mut self,
        statements: &[syntax::Stmt],
        context: BlockContext,
        mut env: BlockEnv,
    ) -> Result<(Vec<TypedStmt>, BlockEnv, BTreeSet<SymbolId>), SemanticError> {
        let mut typed = Vec::with_capacity(statements.len());
        let mut touched_states = BTreeSet::new();

        for stmt in statements {
            match &stmt.kind {
                syntax::StmtKind::Let(let_stmt) => {
                    let value = self.analyze_expr(&let_stmt.value, &env)?;
                    let symbol = self.insert_local_symbol(
                        &mut env,
                        &let_stmt.name,
                        value.ty,
                        SymbolKind::Local,
                    )?;
                    typed.push(TypedStmt {
                        kind: TypedStmtKind::Let(TypedLetStmt { symbol, value }),
                        span: stmt.span,
                    });
                }
                syntax::StmtKind::Assign(assign) => {
                    let target = self.analyze_assign_target(&assign.target, context, &env)?;
                    let value = self.analyze_expr(&assign.value, &env)?;
                    self.expect_numeric(&value, "assignment value", assign.value.span)?;
                    match &target.kind {
                        TypedAssignTargetKind::Derived(symbol) => {
                            self.merge_symbol_type(*symbol, value.ty, assign.value.span)?;
                            env.available_derived.insert(*symbol);
                            env.definite_targets.insert(*symbol);
                        }
                        TypedAssignTargetKind::Output(symbol) => {
                            self.merge_symbol_type(*symbol, value.ty, assign.value.span)?;
                            env.definite_targets.insert(*symbol);
                        }
                        TypedAssignTargetKind::StateInit(place)
                        | TypedAssignTargetKind::Derivative(place)
                        | TypedAssignTargetKind::Noise(place) => {
                            touched_states.insert(place.state);
                        }
                    }
                    typed.push(TypedStmt {
                        kind: TypedStmtKind::Assign(TypedAssignStmt { target, value }),
                        span: stmt.span,
                    });
                }
                syntax::StmtKind::If(if_stmt) => {
                    let condition = self.analyze_expr(&if_stmt.condition, &env)?;
                    self.expect_bool(&condition, "if condition", if_stmt.condition.span)?;

                    let then_env = env.child_scope();
                    let (then_branch, then_env, then_states) =
                        self.analyze_stmt_list(&if_stmt.then_branch, context, then_env)?;
                    let mut branch_states = then_states;

                    let (else_branch, next_available, next_targets) = if let Some(else_branch) =
                        &if_stmt.else_branch
                    {
                        let else_env = env.child_scope();
                        let (else_typed, else_env, else_states) =
                            self.analyze_stmt_list(else_branch, context, else_env)?;
                        branch_states.extend(else_states);
                        let available = if context == BlockContext::Derive {
                            intersect_sets(&then_env.available_derived, &else_env.available_derived)
                        } else {
                            env.available_derived.clone()
                        };
                        let targets =
                            if matches!(context, BlockContext::Derive | BlockContext::Outputs) {
                                intersect_sets(
                                    &then_env.definite_targets,
                                    &else_env.definite_targets,
                                )
                            } else {
                                env.definite_targets.clone()
                            };
                        (Some(else_typed), available, targets)
                    } else {
                        let available = env.available_derived.clone();
                        let targets = env.definite_targets.clone();
                        (None, available, targets)
                    };

                    env.available_derived = next_available;
                    env.definite_targets = next_targets;
                    touched_states.extend(branch_states);
                    typed.push(TypedStmt {
                        kind: TypedStmtKind::If(TypedIfStmt {
                            condition,
                            then_branch,
                            else_branch,
                        }),
                        span: stmt.span,
                    });
                }
                syntax::StmtKind::For(for_stmt) => {
                    let start = self.analyze_expr(&for_stmt.range.start, &env)?;
                    let end = self.analyze_expr(&for_stmt.range.end, &env)?;
                    self.expect_int(&start, "for-loop range start", for_stmt.range.start.span)?;
                    self.expect_int(&end, "for-loop range end", for_stmt.range.end.span)?;

                    let mut loop_env = env.child_scope();
                    let binding = self.insert_local_symbol(
                        &mut loop_env,
                        &for_stmt.binding,
                        ValueType::Int,
                        SymbolKind::LoopBinding,
                    )?;
                    let (body, _loop_env, body_states) =
                        self.analyze_stmt_list(&for_stmt.body, context, loop_env)?;
                    touched_states.extend(body_states);
                    typed.push(TypedStmt {
                        kind: TypedStmtKind::For(TypedForStmt {
                            binding,
                            range: TypedRangeExpr {
                                start,
                                end,
                                span: for_stmt.range.span,
                            },
                            body,
                        }),
                        span: stmt.span,
                    });
                }
            }
        }

        Ok((typed, env, touched_states))
    }

    fn analyze_assign_target(
        &mut self,
        target: &syntax::AssignTarget,
        context: BlockContext,
        env: &BlockEnv,
    ) -> Result<TypedAssignTarget, SemanticError> {
        let kind = match context {
            BlockContext::Derive => match &target.kind {
                syntax::AssignTargetKind::Name(name) => {
                    let Some(symbol) = self.globals.derived.get(&name.text).copied() else {
                        return Err(SemanticError::new(
                            format!("`{}` is not a valid derive target", name.text),
                            name.span,
                        ));
                    };
                    TypedAssignTargetKind::Derived(symbol)
                }
                _ => {
                    return Err(SemanticError::new(
                        "derive assignments must target a bare identifier",
                        target.span,
                    ))
                }
            },
            BlockContext::Outputs => match &target.kind {
                syntax::AssignTargetKind::Name(name) => {
                    let Some(symbol) = self.globals.outputs.get(&name.text).copied() else {
                        return Err(SemanticError::new(
                            format!("`{}` is not a valid output target", name.text),
                            name.span,
                        ));
                    };
                    TypedAssignTargetKind::Output(symbol)
                }
                _ => {
                    return Err(SemanticError::new(
                        "outputs assignments must target a bare identifier",
                        target.span,
                    ))
                }
            },
            BlockContext::Init => {
                TypedAssignTargetKind::StateInit(self.analyze_runtime_state_place(target, env)?)
            }
            BlockContext::Dynamics | BlockContext::Drift => {
                let place = self.expect_call_state_target(target, "ddt")?;
                TypedAssignTargetKind::Derivative(
                    self.analyze_runtime_state_place_expr(&place, env)?,
                )
            }
            BlockContext::Diffusion => {
                let place = self.expect_call_state_target(target, "noise")?;
                TypedAssignTargetKind::Noise(self.analyze_runtime_state_place_expr(&place, env)?)
            }
        };
        Ok(TypedAssignTarget {
            kind,
            span: target.span,
        })
    }

    fn expect_call_state_target(
        &self,
        target: &syntax::AssignTarget,
        expected: &str,
    ) -> Result<syntax::Place, SemanticError> {
        match &target.kind {
            syntax::AssignTargetKind::Call { callee, args }
                if callee.text == expected && args.len() == 1 =>
            {
                self.place_from_expr(&args[0])
            }
            syntax::AssignTargetKind::Call { callee, .. } => Err(SemanticError::new(
                format!(
                    "expected `{expected}(...)` assignment target, found `{}`",
                    callee.text
                ),
                target.span,
            )),
            _ => Err(SemanticError::new(
                format!("expected `{expected}(...)` assignment target"),
                target.span,
            )),
        }
    }

    fn place_from_expr(&self, expr: &syntax::Expr) -> Result<syntax::Place, SemanticError> {
        match &expr.kind {
            syntax::ExprKind::Name(name) => Ok(syntax::Place {
                name: name.clone(),
                index: None,
                span: expr.span,
            }),
            syntax::ExprKind::Index { target, index } => match &target.kind {
                syntax::ExprKind::Name(name) => Ok(syntax::Place {
                    name: name.clone(),
                    index: Some((**index).clone()),
                    span: expr.span,
                }),
                _ => Err(SemanticError::new(
                    "indexed assignment targets must index a state identifier",
                    expr.span,
                )),
            },
            _ => Err(SemanticError::new(
                "expected a state reference in assignment target",
                expr.span,
            )),
        }
    }

    fn analyze_runtime_state_place(
        &self,
        target: &syntax::AssignTarget,
        env: &BlockEnv,
    ) -> Result<TypedStatePlace, SemanticError> {
        let place = match &target.kind {
            syntax::AssignTargetKind::Name(name) => syntax::Place {
                name: name.clone(),
                index: None,
                span: target.span,
            },
            syntax::AssignTargetKind::Index { target, index } => syntax::Place {
                name: target.clone(),
                index: Some(index.clone()),
                span: target.span,
            },
            syntax::AssignTargetKind::Call { .. } => {
                return Err(SemanticError::new(
                    "unexpected call target in runtime state assignment",
                    target.span,
                ))
            }
        };
        self.analyze_runtime_state_place_expr(&place, env)
    }

    fn analyze_runtime_state_place_expr(
        &self,
        place: &syntax::Place,
        env: &BlockEnv,
    ) -> Result<TypedStatePlace, SemanticError> {
        let state = self.globals.states.get(&place.name.text).ok_or_else(|| {
            let error = SemanticError::new(
                format!("unknown state `{}`", place.name.text),
                place.name.span,
            );
            match self.assist_for_unknown_state(&place.name) {
                Some(assist) => assist.apply(error),
                None => error,
            }
        })?;
        let index = match (&state.size, &place.index) {
            (Some(_), Some(index)) => {
                let index = self.analyze_expr(index, env)?;
                self.expect_int(&index, "state index", index.span)?;
                Some(Box::new(index))
            }
            (Some(_), None) => {
                return Err(SemanticError::new(
                    format!("state array `{}` requires an index", place.name.text),
                    place.span,
                ))
            }
            (None, Some(_)) => {
                return Err(SemanticError::new(
                    format!(
                        "state `{}` is scalar and cannot be indexed",
                        place.name.text
                    ),
                    place.span,
                ))
            }
            (None, None) => None,
        };
        Ok(TypedStatePlace {
            state: state.symbol,
            index,
            span: place.span,
        })
    }

    fn analyze_state_place_const(
        &self,
        place: &syntax::Place,
    ) -> Result<TypedStatePlace, SemanticError> {
        let state = self.globals.states.get(&place.name.text).ok_or_else(|| {
            let error = SemanticError::new(
                format!("unknown state `{}`", place.name.text),
                place.name.span,
            );
            match self.assist_for_unknown_state(&place.name) {
                Some(assist) => assist.apply(error),
                None => error,
            }
        })?;
        let index = match (&state.size, &place.index) {
            (Some(_), Some(index)) => {
                let value = self.expect_const_usize(index, "route destination index", false)?;
                Some(Box::new(TypedExpr {
                    kind: TypedExprKind::Literal(ConstValue::Int(value as i64)),
                    ty: ValueType::Int,
                    constant: Some(ConstValue::Int(value as i64)),
                    span: index.span,
                }))
            }
            (Some(_), None) => {
                return Err(SemanticError::new(
                    format!("state array `{}` requires an index", place.name.text),
                    place.span,
                ))
            }
            (None, Some(_)) => {
                return Err(SemanticError::new(
                    format!(
                        "state `{}` is scalar and cannot be indexed",
                        place.name.text
                    ),
                    place.span,
                ))
            }
            (None, None) => None,
        };
        Ok(TypedStatePlace {
            state: state.symbol,
            index,
            span: place.span,
        })
    }

    fn analyze_expr(
        &self,
        expr: &syntax::Expr,
        env: &BlockEnv,
    ) -> Result<TypedExpr, SemanticError> {
        match &expr.kind {
            syntax::ExprKind::Number(value) => {
                let constant = number_to_const(*value);
                Ok(TypedExpr {
                    kind: TypedExprKind::Literal(constant.clone()),
                    ty: constant.value_type(),
                    constant: Some(constant),
                    span: expr.span,
                })
            }
            syntax::ExprKind::Bool(value) => {
                let constant = ConstValue::Bool(*value);
                Ok(TypedExpr {
                    kind: TypedExprKind::Literal(constant.clone()),
                    ty: ValueType::Bool,
                    constant: Some(constant),
                    span: expr.span,
                })
            }
            syntax::ExprKind::Name(name) => self.analyze_name_expr(name, expr.span, env),
            syntax::ExprKind::Unary { op, expr: inner } => {
                let inner = self.analyze_expr(inner, env)?;
                let ty = match op {
                    syntax::UnaryOp::Not => {
                        self.expect_bool(&inner, "unary `!` operand", inner.span)?;
                        ValueType::Bool
                    }
                    syntax::UnaryOp::Plus | syntax::UnaryOp::Minus => {
                        self.expect_numeric(&inner, "unary numeric operand", inner.span)?;
                        inner.ty
                    }
                };
                let op = match op {
                    syntax::UnaryOp::Plus => TypedUnaryOp::Plus,
                    syntax::UnaryOp::Minus => TypedUnaryOp::Minus,
                    syntax::UnaryOp::Not => TypedUnaryOp::Not,
                };
                let constant = inner
                    .constant
                    .as_ref()
                    .and_then(|value| fold_unary(op, value));
                Ok(TypedExpr {
                    kind: TypedExprKind::Unary {
                        op,
                        expr: Box::new(inner),
                    },
                    ty,
                    constant,
                    span: expr.span,
                })
            }
            syntax::ExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.analyze_expr(lhs, env)?;
                let rhs = self.analyze_expr(rhs, env)?;
                let op = map_binary_op(*op);
                let ty = self.binary_result_type(op, &lhs, &rhs, expr.span)?;
                let constant = match (&lhs.constant, &rhs.constant) {
                    (Some(lhs), Some(rhs)) => fold_binary(op, lhs, rhs),
                    _ => None,
                };
                Ok(TypedExpr {
                    kind: TypedExprKind::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    ty,
                    constant,
                    span: expr.span,
                })
            }
            syntax::ExprKind::Call { callee, args } => {
                self.analyze_call(callee, args, expr.span, env)
            }
            syntax::ExprKind::Index { target, index } => {
                let index = self.analyze_expr(index, env)?;
                self.expect_int(&index, "index expression", index.span)?;
                match &target.kind {
                    syntax::ExprKind::Name(name) => {
                        let state = self.globals.states.get(&name.text).ok_or_else(|| {
                            SemanticError::new(
                                format!(
                                    "only state arrays can be indexed; `{}` is not a state",
                                    name.text
                                ),
                                name.span,
                            )
                        })?;
                        if state.size.is_none() {
                            return Err(SemanticError::new(
                                format!("state `{}` is scalar and cannot be indexed", name.text),
                                expr.span,
                            ));
                        }
                        Ok(TypedExpr {
                            kind: TypedExprKind::StateValue(TypedStatePlace {
                                state: state.symbol,
                                index: Some(Box::new(index)),
                                span: expr.span,
                            }),
                            ty: ValueType::Real,
                            constant: None,
                            span: expr.span,
                        })
                    }
                    _ => Err(SemanticError::new(
                        "only state arrays can be indexed",
                        expr.span,
                    )),
                }
            }
        }
    }

    fn analyze_name_expr(
        &self,
        name: &syntax::Ident,
        span: Span,
        env: &BlockEnv,
    ) -> Result<TypedExpr, SemanticError> {
        if let Some(symbol) = env.lookup_local(&name.text) {
            let ty = self.scalar_symbol_type(symbol).ok_or_else(|| {
                SemanticError::new(
                    format!("local `{}` does not resolve to a scalar value", name.text),
                    span,
                )
            })?;
            return Ok(TypedExpr {
                kind: TypedExprKind::Symbol(symbol),
                ty,
                constant: None,
                span,
            });
        }

        if let Some(symbol) = self.globals.parameters.get(&name.text).copied() {
            return Ok(TypedExpr {
                kind: TypedExprKind::Symbol(symbol),
                ty: ValueType::Real,
                constant: None,
                span,
            });
        }

        if let Some(symbol) = self.globals.constants.get(&name.text).copied() {
            let constant = self.globals.constant_values.get(&name.text).cloned();
            let ty = self
                .scalar_symbol_type(symbol)
                .expect("constant type must be known");
            return Ok(TypedExpr {
                kind: TypedExprKind::Symbol(symbol),
                ty,
                constant,
                span,
            });
        }

        if let Some(symbol) = self.globals.covariates.get(&name.text).copied() {
            return Ok(TypedExpr {
                kind: TypedExprKind::Symbol(symbol),
                ty: ValueType::Real,
                constant: None,
                span,
            });
        }

        if let Some(state) = self.globals.states.get(&name.text) {
            if state.size.is_some() {
                return Err(SemanticError::new(
                    format!("state array `{}` requires an index", name.text),
                    span,
                ));
            }
            let place = TypedStatePlace {
                state: state.symbol,
                index: None,
                span,
            };
            return Ok(TypedExpr {
                kind: TypedExprKind::StateValue(place),
                ty: ValueType::Real,
                constant: None,
                span,
            });
        }

        if let Some(symbol) = self.globals.derived.get(&name.text).copied() {
            if !env.available_derived.contains(&symbol) {
                return Err(SemanticError::new(
                    format!(
                        "derived value `{}` is not definitely assigned at this point",
                        name.text
                    ),
                    span,
                ));
            }
            let ty = self.scalar_symbol_type(symbol).ok_or_else(|| {
                SemanticError::new(
                    format!(
                        "derived value `{}` does not have a resolved type yet",
                        name.text
                    ),
                    span,
                )
            })?;
            return Ok(TypedExpr {
                kind: TypedExprKind::Symbol(symbol),
                ty,
                constant: None,
                span,
            });
        }

        if self.globals.routes.contains_key(&name.text) {
            let route = self.globals.routes[&name.text];
            return Err(self
                .assist_for_route_scalar(route, span)
                .apply(SemanticError::new(
                    format!(
                        "route `{}` cannot be used as a scalar value; use `rate({})`",
                        name.text, name.text
                    ),
                    span,
                )));
        }

        if self.globals.outputs.contains_key(&name.text) {
            let output = self.globals.outputs[&name.text];
            return Err(self
                .assist_for_output_scope(output)
                .apply(SemanticError::new(
                    format!("output `{}` is not in expression scope", name.text),
                    span,
                )));
        }

        let error = SemanticError::new(format!("unknown identifier `{}`", name.text), span);
        Err(match self.assist_for_unknown_identifier(name, span, env) {
            Some(assist) => assist.apply(error),
            None => error,
        })
    }

    fn analyze_call(
        &self,
        callee: &syntax::Ident,
        args: &[syntax::Expr],
        span: Span,
        env: &BlockEnv,
    ) -> Result<TypedExpr, SemanticError> {
        if callee.text == "rate" {
            if args.len() != 1 {
                return Err(SemanticError::new(
                    format!(
                        "`rate` expects exactly one route argument, got {}",
                        args.len()
                    ),
                    callee.span,
                ));
            }
            if let syntax::ExprKind::Number(value) = &args[0].kind {
                if let Some(suffix) = numeric_label_literal_suffix(*value) {
                    return Err(self.bare_numeric_route_error(args[0].span, &suffix));
                }
            }
            let syntax::ExprKind::Name(route_name) = &args[0].kind else {
                return Err(SemanticError::new(
                    "`rate` expects a route identifier argument",
                    args[0].span,
                ));
            };
            self.validate_route_label_name(route_name)?;
            let route = self
                .globals
                .routes
                .get(&route_name.text)
                .copied()
                .ok_or_else(|| {
                    let error = SemanticError::new(
                        format!("unknown route `{}` in `rate(...)`", route_name.text),
                        route_name.span,
                    );
                    match self.assist_for_unknown_route(route_name) {
                        Some(assist) => assist.apply(error),
                        None => error,
                    }
                })?;
            return Ok(TypedExpr {
                kind: TypedExprKind::Call {
                    callee: TypedCall::Rate(route),
                    args: Vec::new(),
                },
                ty: ValueType::Real,
                constant: None,
                span,
            });
        }

        let intrinsic = MathIntrinsic::from_name(&callee.text).ok_or_else(|| {
            let error =
                SemanticError::new(format!("unknown function `{}`", callee.text), callee.span);
            match self.assist_for_unknown_function(callee) {
                Some(assist) => assist.apply(error),
                None => error,
            }
        })?;
        let expected_arity = intrinsic.arity();
        match expected_arity {
            IntrinsicArity::Exact(expected) if expected != args.len() => {
                return Err(SemanticError::new(
                    format!(
                        "function `{}` expects {} argument(s), got {}",
                        callee.text,
                        expected,
                        args.len()
                    ),
                    callee.span,
                ))
            }
            _ => {}
        }

        let mut typed_args = Vec::with_capacity(args.len());
        for arg in args {
            let typed = self.analyze_expr(arg, env)?;
            self.expect_numeric(&typed, &format!("`{}` argument", callee.text), arg.span)?;
            typed_args.push(typed);
        }
        let ty = call_result_type(intrinsic, &typed_args);
        let constant = typed_args
            .iter()
            .map(|arg| arg.constant.clone())
            .collect::<Option<Vec<_>>>()
            .and_then(|values| fold_call(intrinsic, &values));
        Ok(TypedExpr {
            kind: TypedExprKind::Call {
                callee: TypedCall::Math(intrinsic),
                args: typed_args,
            },
            ty,
            constant,
            span,
        })
    }

    fn binary_result_type(
        &self,
        op: TypedBinaryOp,
        lhs: &TypedExpr,
        rhs: &TypedExpr,
        span: Span,
    ) -> Result<ValueType, SemanticError> {
        match op {
            TypedBinaryOp::Or | TypedBinaryOp::And => {
                self.expect_bool(lhs, "logical operand", lhs.span)?;
                self.expect_bool(rhs, "logical operand", rhs.span)?;
                Ok(ValueType::Bool)
            }
            TypedBinaryOp::Eq | TypedBinaryOp::NotEq => {
                if lhs.ty != rhs.ty {
                    return Err(SemanticError::new(
                        format!(
                            "equality comparison requires matching operand types, found {:?} and {:?}",
                            lhs.ty, rhs.ty
                        ),
                        span,
                    ));
                }
                Ok(ValueType::Bool)
            }
            TypedBinaryOp::Lt | TypedBinaryOp::LtEq | TypedBinaryOp::Gt | TypedBinaryOp::GtEq => {
                self.expect_numeric(lhs, "comparison operand", lhs.span)?;
                self.expect_numeric(rhs, "comparison operand", rhs.span)?;
                Ok(ValueType::Bool)
            }
            TypedBinaryOp::Add | TypedBinaryOp::Sub | TypedBinaryOp::Mul => {
                self.expect_numeric(lhs, "arithmetic operand", lhs.span)?;
                self.expect_numeric(rhs, "arithmetic operand", rhs.span)?;
                Ok(promote_numeric(lhs.ty, rhs.ty))
            }
            TypedBinaryOp::Div | TypedBinaryOp::Pow => {
                self.expect_numeric(lhs, "arithmetic operand", lhs.span)?;
                self.expect_numeric(rhs, "arithmetic operand", rhs.span)?;
                Ok(ValueType::Real)
            }
        }
    }

    fn expect_numeric(
        &self,
        expr: &TypedExpr,
        context: &str,
        span: Span,
    ) -> Result<(), SemanticError> {
        if expr.ty.is_numeric() {
            Ok(())
        } else {
            Err(SemanticError::new(
                format!("{context} must be numeric, found {:?}", expr.ty),
                span,
            ))
        }
    }

    fn expect_bool(
        &self,
        expr: &TypedExpr,
        context: &str,
        span: Span,
    ) -> Result<(), SemanticError> {
        if expr.ty == ValueType::Bool {
            Ok(())
        } else {
            Err(SemanticError::new(
                format!("{context} must be boolean, found {:?}", expr.ty),
                span,
            ))
        }
    }

    fn expect_int(&self, expr: &TypedExpr, context: &str, span: Span) -> Result<(), SemanticError> {
        if expr.ty == ValueType::Int {
            Ok(())
        } else {
            Err(SemanticError::new(
                format!("{context} must be integer-valued, found {:?}", expr.ty),
                span,
            ))
        }
    }

    fn expect_const_usize(
        &self,
        expr: &syntax::Expr,
        context: &str,
        strictly_positive: bool,
    ) -> Result<usize, SemanticError> {
        let value = self.evaluate_const_expr(expr, &BTreeMap::new(), &mut BTreeSet::new())?;
        let Some(value) = value.as_i64() else {
            return Err(SemanticError::new(
                format!("{context} must be an integer constant"),
                expr.span,
            ));
        };
        if value < 0 || (strictly_positive && value == 0) {
            return Err(SemanticError::new(
                format!(
                    "{context} must be {}",
                    if strictly_positive {
                        "positive"
                    } else {
                        "non-negative"
                    }
                ),
                expr.span,
            ));
        }
        Ok(value as usize)
    }

    fn evaluate_const_expr(
        &self,
        expr: &syntax::Expr,
        bindings: &BTreeMap<String, &syntax::Binding>,
        visiting: &mut BTreeSet<String>,
    ) -> Result<ConstValue, SemanticError> {
        match &expr.kind {
            syntax::ExprKind::Number(value) => Ok(number_to_const(*value)),
            syntax::ExprKind::Bool(value) => Ok(ConstValue::Bool(*value)),
            syntax::ExprKind::Name(name) => {
                if let Some(value) = self.globals.constant_values.get(&name.text) {
                    return Ok(value.clone());
                }
                let binding = bindings.get(&name.text).ok_or_else(|| {
                    SemanticError::new(
                        format!(
                            "unknown constant `{}` in compile-time expression",
                            name.text
                        ),
                        name.span,
                    )
                })?;
                if !visiting.insert(name.text.clone()) {
                    return Err(SemanticError::new(
                        format!("constant `{}` forms a dependency cycle", name.text),
                        name.span,
                    ));
                }
                let value = self.evaluate_const_expr(&binding.value, bindings, visiting)?;
                visiting.remove(&name.text);
                Ok(value)
            }
            syntax::ExprKind::Unary { op, expr } => {
                let value = self.evaluate_const_expr(expr, bindings, visiting)?;
                let op = match op {
                    syntax::UnaryOp::Plus => TypedUnaryOp::Plus,
                    syntax::UnaryOp::Minus => TypedUnaryOp::Minus,
                    syntax::UnaryOp::Not => TypedUnaryOp::Not,
                };
                fold_unary(op, &value).ok_or_else(|| {
                    SemanticError::new("invalid constant unary operation", expr.span)
                })
            }
            syntax::ExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.evaluate_const_expr(lhs, bindings, visiting)?;
                let rhs = self.evaluate_const_expr(rhs, bindings, visiting)?;
                fold_binary(map_binary_op(*op), &lhs, &rhs).ok_or_else(|| {
                    SemanticError::new("invalid constant binary operation", expr.span)
                })
            }
            syntax::ExprKind::Call { callee, args } => {
                if callee.text == "rate" {
                    return Err(SemanticError::new(
                        "`rate(...)` cannot appear in a compile-time expression",
                        callee.span,
                    ));
                }
                let intrinsic = MathIntrinsic::from_name(&callee.text).ok_or_else(|| {
                    SemanticError::new(
                        format!("unknown compile-time function `{}`", callee.text),
                        callee.span,
                    )
                })?;
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.evaluate_const_expr(arg, bindings, visiting)?);
                }
                fold_call(intrinsic, &values).ok_or_else(|| {
                    SemanticError::new(
                        format!("invalid compile-time call to `{}`", callee.text),
                        expr.span,
                    )
                })
            }
            syntax::ExprKind::Index { .. } => Err(SemanticError::new(
                "indexing is not allowed in compile-time expressions",
                expr.span,
            )),
        }
    }

    fn insert_global_symbol(
        &mut self,
        name: &str,
        kind: SymbolKind,
        ty: PendingSymbolType,
        span: Span,
    ) -> Result<SymbolId, SemanticError> {
        if RESERVED_NAMES.contains(&name) {
            return Err(SemanticAssist::default()
                .help(format!(
                    "rename `{name}` to a non-reserved identifier such as `{}_value`",
                    name
                ))
                .replacement_suggestion(
                    span,
                    format!("{}_value", name),
                    format!("rename `{name}` to `{}_value`", name),
                    Applicability::MaybeIncorrect,
                )
                .apply(SemanticError::new(
                    format!("`{name}` is reserved by the DSL and cannot be used as a symbol name"),
                    span,
                )));
        }
        if let Some(existing) = self.globals.all_names.get(name).copied() {
            let existing_kind = self.symbols.get(existing).expect("valid symbol id").kind;
            if !allows_route_output_name_overlap(existing_kind, kind) {
                return Err(SemanticAssist::default()
                    .context_label(
                        self.symbol_span(existing),
                        self.symbol_declared_here(existing),
                    )
                    .help(format!(
                        "rename this declaration to a unique name such as `{}_2`",
                        name
                    ))
                    .replacement_suggestion(
                        span,
                        format!("{}_2", name),
                        format!("rename this declaration to `{}_2`", name),
                        Applicability::MaybeIncorrect,
                    )
                    .apply(SemanticError::new(
                        format!(
                            "symbol name `{name}` collides with existing `{}`",
                            self.symbol_name(existing)
                        ),
                        span,
                    )));
            }
        }
        let id = self.symbols.len();
        self.symbols.push(PendingSymbol {
            id,
            name: name.to_string(),
            kind,
            ty,
            span,
        });
        self.globals.all_names.entry(name.to_string()).or_insert(id);
        Ok(id)
    }

    fn validate_route_label_name(&self, label: &syntax::Ident) -> Result<(), SemanticError> {
        if let Some(suffix) = bare_numeric_label(&label.text) {
            return Err(self.bare_numeric_route_error(label.span, suffix));
        }
        if let Some(suffix) = canonical_numeric_suffix(&label.text, NUMERIC_OUTPUT_PREFIX) {
            return Err(self.wrong_prefix_route_error(label, suffix));
        }
        Ok(())
    }

    fn validate_output_label_name(&self, label: &syntax::Ident) -> Result<(), SemanticError> {
        if let Some(suffix) = bare_numeric_label(&label.text) {
            return Err(self.bare_numeric_output_error(label.span, suffix));
        }
        if let Some(suffix) = canonical_numeric_suffix(&label.text, NUMERIC_ROUTE_PREFIX) {
            return Err(self.wrong_prefix_output_error(label, suffix));
        }
        Ok(())
    }

    fn bare_numeric_route_error(&self, span: Span, suffix: &str) -> SemanticError {
        let replacement = format!("{NUMERIC_ROUTE_PREFIX}{suffix}");
        SemanticAssist::default()
            .help("numeric route labels must use the `input_<n>` form in authored DSL")
            .replacement_suggestion(
                span,
                replacement.clone(),
                format!("use `{replacement}`"),
                Applicability::Always,
            )
            .apply(SemanticError::new(
                format!(
                    "bare numeric route labels are not allowed in the DSL; use `{replacement}` instead"
                ),
                span,
            ))
    }

    fn bare_numeric_output_error(&self, span: Span, suffix: &str) -> SemanticError {
        let replacement = format!("{NUMERIC_OUTPUT_PREFIX}{suffix}");
        SemanticAssist::default()
            .help("numeric output labels must use the `outeq_<n>` form in authored DSL")
            .replacement_suggestion(
                span,
                replacement.clone(),
                format!("use `{replacement}`"),
                Applicability::Always,
            )
            .apply(SemanticError::new(
                format!(
                    "bare numeric output labels are not allowed in the DSL; use `{replacement}` instead"
                ),
                span,
            ))
    }

    fn wrong_prefix_route_error(&self, label: &syntax::Ident, suffix: &str) -> SemanticError {
        let replacement = format!("{NUMERIC_ROUTE_PREFIX}{suffix}");
        SemanticAssist::default()
            .help("numeric route labels use the `input_<n>` prefix")
            .replacement_suggestion(
                label.span,
                replacement.clone(),
                format!("use `{replacement}`"),
                Applicability::Always,
            )
            .apply(SemanticError::new(
                format!(
                    "`{}` is an output label and cannot be used as a route; use `{replacement}` here",
                    label.text
                ),
                label.span,
            ))
    }

    fn wrong_prefix_output_error(&self, label: &syntax::Ident, suffix: &str) -> SemanticError {
        let replacement = format!("{NUMERIC_OUTPUT_PREFIX}{suffix}");
        SemanticAssist::default()
            .help("numeric output labels use the `outeq_<n>` prefix")
            .replacement_suggestion(
                label.span,
                replacement.clone(),
                format!("use `{replacement}`"),
                Applicability::Always,
            )
            .apply(SemanticError::new(
                format!(
                    "`{}` is a route label and cannot be used as an output target; use `{replacement}` here",
                    label.text
                ),
                label.span,
            ))
    }

    fn insert_local_symbol(
        &mut self,
        env: &mut BlockEnv,
        ident: &syntax::Ident,
        ty: ValueType,
        kind: SymbolKind,
    ) -> Result<SymbolId, SemanticError> {
        if let Some(existing) = env
            .lookup_local(&ident.text)
            .or_else(|| self.globals.all_names.get(&ident.text).copied())
        {
            return Err(SemanticAssist::default()
                .context_label(
                    self.symbol_span(existing),
                    self.symbol_declared_here(existing),
                )
                .help(format!(
                    "rename this local binding to a unique name such as `{}_local`",
                    ident.text
                ))
                .replacement_suggestion(
                    ident.span,
                    format!("{}_local", ident.text),
                    format!("rename this local binding to `{}_local`", ident.text),
                    Applicability::MaybeIncorrect,
                )
                .apply(SemanticError::new(
                    format!(
                        "local symbol `{}` would shadow an existing symbol",
                        ident.text
                    ),
                    ident.span,
                )));
        }
        let id = self.symbols.len();
        self.symbols.push(PendingSymbol {
            id,
            name: ident.text.clone(),
            kind,
            ty: PendingSymbolType::Scalar(Some(ty)),
            span: ident.span,
        });
        env.insert_local(ident.text.clone(), id);
        Ok(id)
    }

    fn merge_symbol_type(
        &mut self,
        symbol: SymbolId,
        ty: ValueType,
        span: Span,
    ) -> Result<(), SemanticError> {
        let entry = self.symbols.get_mut(symbol).expect("valid symbol id");
        match &mut entry.ty {
            PendingSymbolType::Scalar(slot) => match slot {
                None => *slot = Some(ty),
                Some(existing) if *existing == ty => {}
                Some(existing) if existing.is_numeric() && ty.is_numeric() => {
                    *slot = Some(promote_numeric(*existing, ty));
                }
                Some(existing) => {
                    return Err(SemanticError::new(
                        format!(
                            "symbol `{}` is assigned incompatible types {:?} and {:?}",
                            entry.name, existing, ty
                        ),
                        span,
                    ));
                }
            },
            PendingSymbolType::Array { .. } | PendingSymbolType::Route => {
                return Err(SemanticError::new(
                    format!(
                        "symbol `{}` is not assignable as a scalar target",
                        entry.name
                    ),
                    span,
                ));
            }
        }
        Ok(())
    }

    fn scalar_symbol_type(&self, symbol: SymbolId) -> Option<ValueType> {
        match &self.symbols.get(symbol)?.ty {
            PendingSymbolType::Scalar(Some(ty)) => Some(*ty),
            PendingSymbolType::Scalar(None) => None,
            PendingSymbolType::Array { .. } | PendingSymbolType::Route => None,
        }
    }

    fn symbol_name(&self, symbol: SymbolId) -> &str {
        &self.symbols[symbol].name
    }

    fn symbol_span(&self, symbol: SymbolId) -> Span {
        self.symbols[symbol].span
    }

    fn symbol_kind_label(&self, symbol: SymbolId) -> &'static str {
        match self.symbols[symbol].kind {
            SymbolKind::Parameter => "parameter",
            SymbolKind::Constant => "constant",
            SymbolKind::Covariate => "covariate",
            SymbolKind::State => "state",
            SymbolKind::Route => "route",
            SymbolKind::Derived => "derived value",
            SymbolKind::Output => "output",
            SymbolKind::Local => "local",
            SymbolKind::LoopBinding => "loop binding",
        }
    }

    fn symbol_declared_here(&self, symbol: SymbolId) -> String {
        format!(
            "{} `{}` declared here",
            self.symbol_kind_label(symbol),
            self.symbol_name(symbol)
        )
    }

    fn assist_for_symbol_replacement(&self, symbol: SymbolId, span: Span) -> SemanticAssist {
        let name = self.symbol_name(symbol).to_string();
        SemanticAssist::default()
            .context_label(self.symbol_span(symbol), self.symbol_declared_here(symbol))
            .replacement_suggestion(
                span,
                name.clone(),
                format!("did you mean `{name}`?"),
                Applicability::MaybeIncorrect,
            )
    }

    fn assist_for_route_scalar(&self, route: SymbolId, span: Span) -> SemanticAssist {
        let name = self.symbol_name(route).to_string();
        SemanticAssist::default()
            .context_label(self.symbol_span(route), self.symbol_declared_here(route))
            .help(format!("route inputs are read through `rate({name})`"))
            .replacement_suggestion(
                span,
                format!("rate({name})"),
                format!("did you mean `rate({name})`?"),
                Applicability::MaybeIncorrect,
            )
    }

    fn assist_for_output_scope(&self, output: SymbolId) -> SemanticAssist {
        SemanticAssist::default()
            .context_label(self.symbol_span(output), self.symbol_declared_here(output))
            .help(
                "outputs are assignment targets inside the `outputs` block and are not available as expression values",
            )
    }

    fn assist_for_unknown_identifier(
        &self,
        name: &syntax::Ident,
        span: Span,
        env: &BlockEnv,
    ) -> Option<SemanticAssist> {
        let mut seen = BTreeSet::new();
        let mut candidates = Vec::new();

        for scope in env.locals.iter().rev() {
            for (candidate_name, symbol) in scope {
                if seen.insert(candidate_name.clone()) {
                    candidates.push(SimilarNameCandidate::new(
                        candidate_name.clone(),
                        self.assist_for_symbol_replacement(*symbol, span),
                    ));
                }
            }
        }

        for symbol in self
            .globals
            .parameters
            .values()
            .chain(self.globals.constants.values())
            .chain(self.globals.covariates.values())
            .chain(
                self.globals
                    .states
                    .values()
                    .filter(|entry| entry.size.is_none())
                    .map(|entry| &entry.symbol),
            )
        {
            let candidate_name = self.symbol_name(*symbol).to_string();
            if seen.insert(candidate_name.clone()) {
                candidates.push(SimilarNameCandidate::new(
                    candidate_name,
                    self.assist_for_symbol_replacement(*symbol, span),
                ));
            }
        }

        for symbol in &env.available_derived {
            let candidate_name = self.symbol_name(*symbol).to_string();
            if seen.insert(candidate_name.clone()) {
                candidates.push(SimilarNameCandidate::new(
                    candidate_name,
                    self.assist_for_symbol_replacement(*symbol, span),
                ));
            }
        }

        for symbol in self.globals.routes.values() {
            let candidate_name = self.symbol_name(*symbol).to_string();
            if seen.insert(candidate_name.clone()) {
                candidates.push(SimilarNameCandidate::new(
                    candidate_name,
                    self.assist_for_route_scalar(*symbol, span),
                ));
            }
        }

        best_similar_name_assist(&name.text, candidates)
    }

    fn assist_for_unknown_state(&self, state_name: &syntax::Ident) -> Option<SemanticAssist> {
        let candidates = self
            .globals
            .states
            .values()
            .map(|entry| {
                SimilarNameCandidate::new(
                    self.symbol_name(entry.symbol).to_string(),
                    self.assist_for_symbol_replacement(entry.symbol, state_name.span),
                )
            })
            .collect::<Vec<_>>();
        best_similar_name_assist(&state_name.text, candidates)
    }

    fn assist_for_unknown_route(&self, route_name: &syntax::Ident) -> Option<SemanticAssist> {
        let candidates = self
            .globals
            .routes
            .values()
            .map(|symbol| {
                let name = self.symbol_name(*symbol).to_string();
                SimilarNameCandidate::new(
                    name.clone(),
                    SemanticAssist::default()
                        .context_label(
                            self.symbol_span(*symbol),
                            self.symbol_declared_here(*symbol),
                        )
                        .replacement_suggestion(
                            route_name.span,
                            name.clone(),
                            format!("did you mean `{name}`?"),
                            Applicability::MaybeIncorrect,
                        ),
                )
            })
            .collect::<Vec<_>>();
        best_similar_name_assist(&route_name.text, candidates)
    }

    fn assist_for_unknown_function(&self, callee: &syntax::Ident) -> Option<SemanticAssist> {
        let mut candidates = MathIntrinsic::ALL
            .iter()
            .map(|intrinsic| {
                let name = intrinsic.name().to_string();
                SimilarNameCandidate::new(
                    name.clone(),
                    SemanticAssist::default().replacement_suggestion(
                        callee.span,
                        name.clone(),
                        format!("did you mean `{name}`?"),
                        Applicability::MaybeIncorrect,
                    ),
                )
            })
            .collect::<Vec<_>>();
        candidates.push(SimilarNameCandidate::new(
            RATE_FUNCTION_NAME,
            SemanticAssist::default()
                .help("`rate` reads route inputs as `rate(route)`")
                .replacement_suggestion(
                    callee.span,
                    RATE_FUNCTION_NAME,
                    "did you mean `rate`?",
                    Applicability::MaybeIncorrect,
                ),
        ));
        best_similar_name_assist(&callee.text, candidates)
    }

    fn finalize_symbols(self) -> Result<Vec<Symbol>, SemanticError> {
        self.symbols
            .into_iter()
            .map(|symbol| {
                let ty = match symbol.ty {
                    PendingSymbolType::Scalar(Some(ty)) => SymbolType::Scalar(ty),
                    PendingSymbolType::Scalar(None) => {
                        return Err(SemanticError::new(
                            format!(
                                "symbol `{}` does not have a resolved scalar type",
                                symbol.name
                            ),
                            symbol.span,
                        ))
                    }
                    PendingSymbolType::Array { element, size } => {
                        SymbolType::Array { element, size }
                    }
                    PendingSymbolType::Route => SymbolType::Route,
                };
                Ok(Symbol {
                    id: symbol.id,
                    name: symbol.name,
                    kind: symbol.kind,
                    ty,
                    span: symbol.span,
                })
            })
            .collect()
    }

    fn validate_kind_requirements(
        &self,
        sections: &ModelSections<'_>,
        states: &[TypedState],
    ) -> Result<(), SemanticError> {
        if states.is_empty() {
            return Err(SemanticError::new(
                format!(
                    "model `{}` must declare at least one state",
                    self.model.name.text
                ),
                self.model.span,
            ));
        }
        if sections.outputs.is_none() {
            return Err(SemanticError::new(
                format!(
                    "model `{}` is missing an `outputs` block",
                    self.model.name.text
                ),
                self.model.span,
            ));
        }
        Ok(())
    }

    fn validate_kind_blocks(
        &self,
        kind: ModelKind,
        blocks: ModelKindBlocks<'_>,
        states: &[TypedState],
    ) -> Result<(), SemanticError> {
        match kind {
            ModelKind::Ode => {
                if blocks.dynamics.is_none() {
                    return Err(SemanticError::new(
                        "ODE models require a `dynamics` block",
                        self.model.span,
                    ));
                }
                if blocks.drift.is_some() || blocks.diffusion.is_some() {
                    return Err(SemanticError::new(
                        "ODE models cannot declare `drift` or `diffusion` blocks",
                        self.model.span,
                    ));
                }
                if blocks.analytical.is_some() {
                    return Err(SemanticError::new(
                        "ODE models cannot declare an `analytical` block",
                        self.model.span,
                    ));
                }
                if let Some(particles_decl) = blocks.particles {
                    return Err(SemanticError::new(
                        "ODE models cannot declare `particles`",
                        particles_decl.span,
                    ));
                }
            }
            ModelKind::Analytical => {
                if blocks.analytical.is_none() {
                    return Err(SemanticError::new(
                        "analytical models require an `analytical` block",
                        self.model.span,
                    ));
                }
                if blocks.dynamics.is_some() || blocks.drift.is_some() || blocks.diffusion.is_some()
                {
                    return Err(SemanticError::new(
                        "analytical models cannot declare `dynamics`, `drift`, or `diffusion` blocks",
                        self.model.span,
                    ));
                }
                if let Some(particles_decl) = blocks.particles {
                    return Err(SemanticError::new(
                        "analytical models cannot declare `particles`",
                        particles_decl.span,
                    ));
                }
            }
            ModelKind::Sde => {
                if blocks.drift.is_none() || blocks.diffusion.is_none() {
                    return Err(SemanticError::new(
                        "SDE models require both `drift` and `diffusion` blocks",
                        self.model.span,
                    ));
                }
                if blocks.dynamics.is_some() {
                    return Err(SemanticError::new(
                        "SDE models cannot declare a `dynamics` block",
                        self.model.span,
                    ));
                }
                if blocks.analytical.is_some() {
                    return Err(SemanticError::new(
                        "SDE models cannot declare an `analytical` block",
                        self.model.span,
                    ));
                }
                if blocks.particles.is_none() {
                    return Err(SemanticError::new(
                        "SDE models require `particles`",
                        self.model.span,
                    ));
                }
            }
        }

        if states.is_empty() {
            return Err(SemanticError::new(
                "typed model validation requires at least one state",
                self.model.span,
            ));
        }
        Ok(())
    }

    fn validate_output_assignments(
        &self,
        outputs: &[SymbolId],
        block: &BlockAnalysis,
    ) -> Result<(), SemanticError> {
        for output in outputs {
            if !block.definite_targets.contains(output) {
                return Err(SemanticError::new(
                    format!(
                        "output `{}` is not definitely assigned on all control-flow paths",
                        self.symbol_name(*output)
                    ),
                    block.block.span,
                ));
            }
        }
        Ok(())
    }

    fn validate_state_coverage(
        &self,
        block: &BlockAnalysis,
        states: &[TypedState],
        block_name: &str,
    ) -> Result<(), SemanticError> {
        for state in states {
            if !block.touched_states.contains(&state.symbol) {
                return Err(SemanticError::new(
                    format!(
                        "{block_name} block does not assign `{}`",
                        self.symbol_name(state.symbol)
                    ),
                    block.block.span,
                ));
            }
        }
        Ok(())
    }
}

fn allows_route_output_name_overlap(existing: SymbolKind, new: SymbolKind) -> bool {
    matches!(
        (existing, new),
        (SymbolKind::Route, SymbolKind::Output) | (SymbolKind::Output, SymbolKind::Route)
    )
}

fn bare_numeric_label(src: &str) -> Option<&str> {
    (!src.is_empty() && src.chars().all(|ch| ch.is_ascii_digit())).then_some(src)
}

fn canonical_numeric_suffix<'a>(src: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = src.strip_prefix(prefix)?;
    (!suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit())).then_some(suffix)
}

fn numeric_label_literal_suffix(value: f64) -> Option<String> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64)
        .then(|| (value as usize).to_string())
}

#[derive(Default)]
struct Globals {
    all_names: BTreeMap<String, SymbolId>,
    parameters: BTreeMap<String, SymbolId>,
    constants: BTreeMap<String, SymbolId>,
    constant_values: BTreeMap<String, ConstValue>,
    covariates: BTreeMap<String, SymbolId>,
    states: BTreeMap<String, StateEntry>,
    routes: BTreeMap<String, SymbolId>,
    derived: BTreeMap<String, SymbolId>,
    outputs: BTreeMap<String, SymbolId>,
}

#[derive(Debug, Clone, Copy)]
struct StateEntry {
    symbol: SymbolId,
    size: Option<usize>,
}

#[derive(Clone)]
struct BlockEnv {
    locals: Vec<BTreeMap<String, SymbolId>>,
    available_derived: BTreeSet<SymbolId>,
    definite_targets: BTreeSet<SymbolId>,
}

impl BlockEnv {
    fn new(available_derived: BTreeSet<SymbolId>) -> Self {
        Self {
            locals: vec![BTreeMap::new()],
            available_derived,
            definite_targets: BTreeSet::new(),
        }
    }

    fn child_scope(&self) -> Self {
        let mut next = self.clone();
        next.locals.push(BTreeMap::new());
        next
    }

    fn insert_local(&mut self, name: String, symbol: SymbolId) {
        self.locals
            .last_mut()
            .expect("local scope")
            .insert(name, symbol);
    }

    fn lookup_local(&self, name: &str) -> Option<SymbolId> {
        self.locals
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }
}

struct BlockAnalysis {
    block: TypedStatementBlock,
    available_derived: BTreeSet<SymbolId>,
    definite_targets: BTreeSet<SymbolId>,
    touched_states: BTreeSet<SymbolId>,
}

#[derive(Clone)]
enum PendingSymbolType {
    Scalar(Option<ValueType>),
    Array { element: ValueType, size: usize },
    Route,
}

struct PendingSymbol {
    id: SymbolId,
    name: String,
    kind: SymbolKind,
    ty: PendingSymbolType,
    span: Span,
}

struct ModelKindBlocks<'a> {
    dynamics: Option<&'a BlockAnalysis>,
    drift: Option<&'a BlockAnalysis>,
    diffusion: Option<&'a BlockAnalysis>,
    analytical: Option<&'a syntax::AnalyticalBlock>,
    particles: Option<&'a syntax::ParticlesDecl>,
}

#[derive(Default)]
struct ModelSections<'a> {
    parameters: Option<&'a syntax::ParametersBlock>,
    constants: Option<&'a syntax::ConstantsBlock>,
    covariates: Option<&'a syntax::CovariatesBlock>,
    states: Option<&'a syntax::StatesBlock>,
    routes: Option<&'a syntax::RoutesBlock>,
    derive: Option<&'a syntax::StatementBlock>,
    dynamics: Option<&'a syntax::StatementBlock>,
    outputs: Option<&'a syntax::StatementBlock>,
    analytical: Option<&'a syntax::AnalyticalBlock>,
    init: Option<&'a syntax::StatementBlock>,
    drift: Option<&'a syntax::StatementBlock>,
    diffusion: Option<&'a syntax::StatementBlock>,
    particles: Option<&'a syntax::ParticlesDecl>,
}

impl<'a> ModelSections<'a> {
    fn from_model(model: &'a syntax::Model) -> Result<Self, SemanticError> {
        let mut sections = Self::default();
        for item in &model.items {
            match item {
                syntax::ModelItem::Parameters(block) => {
                    set_once(&mut sections.parameters, block, "parameters")?
                }
                syntax::ModelItem::Constants(block) => {
                    set_once(&mut sections.constants, block, "constants")?
                }
                syntax::ModelItem::Covariates(block) => {
                    set_once(&mut sections.covariates, block, "covariates")?
                }
                syntax::ModelItem::States(block) => {
                    set_once(&mut sections.states, block, "states")?
                }
                syntax::ModelItem::Routes(block) => {
                    set_once(&mut sections.routes, block, "routes")?
                }
                syntax::ModelItem::Derive(block) => {
                    set_once(&mut sections.derive, block, "derive")?
                }
                syntax::ModelItem::Dynamics(block) => {
                    set_once(&mut sections.dynamics, block, "dynamics")?
                }
                syntax::ModelItem::Outputs(block) => {
                    set_once(&mut sections.outputs, block, "outputs")?
                }
                syntax::ModelItem::Analytical(block) => {
                    set_once(&mut sections.analytical, block, "analytical")?
                }
                syntax::ModelItem::Init(block) => set_once(&mut sections.init, block, "init")?,
                syntax::ModelItem::Drift(block) => set_once(&mut sections.drift, block, "drift")?,
                syntax::ModelItem::Diffusion(block) => {
                    set_once(&mut sections.diffusion, block, "diffusion")?
                }
                syntax::ModelItem::Particles(block) => {
                    set_once(&mut sections.particles, block, "particles")?
                }
            }
        }
        Ok(sections)
    }
}

fn set_once<'a, T>(slot: &mut Option<&'a T>, value: &'a T, name: &str) -> Result<(), SemanticError>
where
    T: HasSpan,
{
    if let Some(existing) = *slot {
        return Err(SemanticAssist::default()
            .context_label(
                existing.span(),
                format!("`{name}` section first declared here"),
            )
            .help(format!("each model can declare `{name}` at most once"))
            .apply(SemanticError::new(
                format!("duplicate `{name}` section in model body"),
                value.span(),
            )));
    }
    *slot = Some(value);
    Ok(())
}

fn best_similar_name_assist(
    needle: &str,
    candidates: Vec<SimilarNameCandidate>,
) -> Option<SemanticAssist> {
    let original_needle = needle;
    let needle = needle.to_ascii_lowercase();
    let mut best: Option<((usize, usize, usize), SemanticAssist)> = None;
    let mut tied = false;

    for candidate in candidates {
        if candidate.lookup_name == original_needle {
            continue;
        }
        let lookup = candidate.lookup_name.to_ascii_lowercase();
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
                best = Some((score, candidate.assist));
                tied = false;
            }
            Some((best_score, _)) if score < *best_score => {
                best = Some((score, candidate.assist));
                tied = false;
            }
            Some((best_score, _)) if score == *best_score => tied = true,
            _ => {}
        }
    }

    if tied {
        None
    } else {
        best.map(|(_, assist)| assist)
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

trait HasSpan {
    fn span(&self) -> Span;
}

impl HasSpan for syntax::ParametersBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::ConstantsBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::CovariatesBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::StatesBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::RoutesBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::StatementBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::AnalyticalBlock {
    fn span(&self) -> Span {
        self.span
    }
}
impl HasSpan for syntax::ParticlesDecl {
    fn span(&self) -> Span {
        self.span
    }
}

fn collect_bare_assignment_names(
    statements: &[syntax::Stmt],
    seen: &mut BTreeSet<String>,
    output: &mut Vec<syntax::Ident>,
) {
    for statement in statements {
        match &statement.kind {
            syntax::StmtKind::Assign(assign) => {
                if let syntax::AssignTargetKind::Name(name) = &assign.target.kind {
                    if seen.insert(name.text.clone()) {
                        output.push(name.clone());
                    }
                }
            }
            syntax::StmtKind::If(if_stmt) => {
                collect_bare_assignment_names(&if_stmt.then_branch, seen, output);
                if let Some(else_branch) = &if_stmt.else_branch {
                    collect_bare_assignment_names(else_branch, seen, output);
                }
            }
            syntax::StmtKind::For(for_stmt) => {
                collect_bare_assignment_names(&for_stmt.body, seen, output);
            }
            syntax::StmtKind::Let(_) => {}
        }
    }
}

fn number_to_const(value: f64) -> ConstValue {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value <= i64::MAX as f64
    {
        ConstValue::Int(value as i64)
    } else {
        ConstValue::Real(value)
    }
}

fn promote_numeric(lhs: ValueType, rhs: ValueType) -> ValueType {
    if lhs == ValueType::Real || rhs == ValueType::Real {
        ValueType::Real
    } else {
        ValueType::Int
    }
}

fn intersect_sets(set_a: &BTreeSet<SymbolId>, set_b: &BTreeSet<SymbolId>) -> BTreeSet<SymbolId> {
    set_a.intersection(set_b).copied().collect()
}

fn map_binary_op(op: syntax::BinaryOp) -> TypedBinaryOp {
    match op {
        syntax::BinaryOp::Or => TypedBinaryOp::Or,
        syntax::BinaryOp::And => TypedBinaryOp::And,
        syntax::BinaryOp::Eq => TypedBinaryOp::Eq,
        syntax::BinaryOp::NotEq => TypedBinaryOp::NotEq,
        syntax::BinaryOp::Lt => TypedBinaryOp::Lt,
        syntax::BinaryOp::LtEq => TypedBinaryOp::LtEq,
        syntax::BinaryOp::Gt => TypedBinaryOp::Gt,
        syntax::BinaryOp::GtEq => TypedBinaryOp::GtEq,
        syntax::BinaryOp::Add => TypedBinaryOp::Add,
        syntax::BinaryOp::Sub => TypedBinaryOp::Sub,
        syntax::BinaryOp::Mul => TypedBinaryOp::Mul,
        syntax::BinaryOp::Div => TypedBinaryOp::Div,
        syntax::BinaryOp::Pow => TypedBinaryOp::Pow,
    }
}

fn call_result_type(intrinsic: MathIntrinsic, args: &[TypedExpr]) -> ValueType {
    match intrinsic {
        MathIntrinsic::Abs => args.first().map_or(ValueType::Real, |arg| arg.ty),
        MathIntrinsic::Min | MathIntrinsic::Max => args
            .iter()
            .map(|arg| arg.ty)
            .reduce(promote_numeric)
            .unwrap_or(ValueType::Real),
        MathIntrinsic::Floor
        | MathIntrinsic::Ceil
        | MathIntrinsic::Exp
        | MathIntrinsic::Ln
        | MathIntrinsic::Log
        | MathIntrinsic::Log10
        | MathIntrinsic::Log2
        | MathIntrinsic::Pow
        | MathIntrinsic::Round
        | MathIntrinsic::Sin
        | MathIntrinsic::Cos
        | MathIntrinsic::Tan
        | MathIntrinsic::Sqrt => ValueType::Real,
    }
}

fn fold_unary(op: TypedUnaryOp, value: &ConstValue) -> Option<ConstValue> {
    match (op, value) {
        (TypedUnaryOp::Plus, ConstValue::Int(value)) => Some(ConstValue::Int(*value)),
        (TypedUnaryOp::Plus, ConstValue::Real(value)) => Some(ConstValue::Real(*value)),
        (TypedUnaryOp::Minus, ConstValue::Int(value)) => Some(ConstValue::Int(-value)),
        (TypedUnaryOp::Minus, ConstValue::Real(value)) => Some(ConstValue::Real(-value)),
        (TypedUnaryOp::Not, ConstValue::Bool(value)) => Some(ConstValue::Bool(!value)),
        _ => None,
    }
}

fn fold_binary(op: TypedBinaryOp, lhs: &ConstValue, rhs: &ConstValue) -> Option<ConstValue> {
    match op {
        TypedBinaryOp::Or => Some(ConstValue::Bool(
            matches!(lhs, ConstValue::Bool(true)) || matches!(rhs, ConstValue::Bool(true)),
        )),
        TypedBinaryOp::And => Some(ConstValue::Bool(
            matches!(lhs, ConstValue::Bool(true)) && matches!(rhs, ConstValue::Bool(true)),
        )),
        TypedBinaryOp::Eq => Some(ConstValue::Bool(lhs == rhs)),
        TypedBinaryOp::NotEq => Some(ConstValue::Bool(lhs != rhs)),
        TypedBinaryOp::Lt => Some(ConstValue::Bool(lhs.as_f64()? < rhs.as_f64()?)),
        TypedBinaryOp::LtEq => Some(ConstValue::Bool(lhs.as_f64()? <= rhs.as_f64()?)),
        TypedBinaryOp::Gt => Some(ConstValue::Bool(lhs.as_f64()? > rhs.as_f64()?)),
        TypedBinaryOp::GtEq => Some(ConstValue::Bool(lhs.as_f64()? >= rhs.as_f64()?)),
        TypedBinaryOp::Add => fold_numeric(
            lhs,
            rhs,
            |left, right| left + right,
            |left, right| left + right,
        ),
        TypedBinaryOp::Sub => fold_numeric(
            lhs,
            rhs,
            |left, right| left - right,
            |left, right| left - right,
        ),
        TypedBinaryOp::Mul => fold_numeric(
            lhs,
            rhs,
            |left, right| left * right,
            |left, right| left * right,
        ),
        TypedBinaryOp::Div => Some(ConstValue::Real(lhs.as_f64()? / rhs.as_f64()?)),
        TypedBinaryOp::Pow => Some(ConstValue::Real(lhs.as_f64()?.powf(rhs.as_f64()?))),
    }
}

fn fold_numeric(
    lhs: &ConstValue,
    rhs: &ConstValue,
    int_op: impl FnOnce(i64, i64) -> i64,
    real_op: impl FnOnce(f64, f64) -> f64,
) -> Option<ConstValue> {
    match (lhs, rhs) {
        (ConstValue::Int(lhs), ConstValue::Int(rhs)) => Some(ConstValue::Int(int_op(*lhs, *rhs))),
        _ => Some(ConstValue::Real(real_op(lhs.as_f64()?, rhs.as_f64()?))),
    }
}

fn fold_call(intrinsic: MathIntrinsic, values: &[ConstValue]) -> Option<ConstValue> {
    match intrinsic {
        MathIntrinsic::Abs => match values.first()? {
            ConstValue::Int(value) => Some(ConstValue::Int(value.abs())),
            ConstValue::Real(value) => Some(ConstValue::Real(value.abs())),
            ConstValue::Bool(_) => None,
        },
        MathIntrinsic::Ceil => Some(ConstValue::Real(values.first()?.as_f64()?.ceil())),
        MathIntrinsic::Exp => Some(ConstValue::Real(values.first()?.as_f64()?.exp())),
        MathIntrinsic::Floor => Some(ConstValue::Real(values.first()?.as_f64()?.floor())),
        MathIntrinsic::Ln | MathIntrinsic::Log => {
            Some(ConstValue::Real(values.first()?.as_f64()?.ln()))
        }
        MathIntrinsic::Log10 => Some(ConstValue::Real(values.first()?.as_f64()?.log10())),
        MathIntrinsic::Log2 => Some(ConstValue::Real(values.first()?.as_f64()?.log2())),
        MathIntrinsic::Max => Some(ConstValue::Real(
            values.first()?.as_f64()?.max(values.get(1)?.as_f64()?),
        )),
        MathIntrinsic::Min => Some(ConstValue::Real(
            values.first()?.as_f64()?.min(values.get(1)?.as_f64()?),
        )),
        MathIntrinsic::Pow => Some(ConstValue::Real(
            values.first()?.as_f64()?.powf(values.get(1)?.as_f64()?),
        )),
        MathIntrinsic::Round => Some(ConstValue::Real(values.first()?.as_f64()?.round())),
        MathIntrinsic::Sin => Some(ConstValue::Real(values.first()?.as_f64()?.sin())),
        MathIntrinsic::Cos => Some(ConstValue::Real(values.first()?.as_f64()?.cos())),
        MathIntrinsic::Tan => Some(ConstValue::Real(values.first()?.as_f64()?.tan())),
        MathIntrinsic::Sqrt => Some(ConstValue::Real(values.first()?.as_f64()?.sqrt())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        RECOMMENDED_STYLE_AUTHORING, RECOMMENDED_STYLE_CANONICAL, STRUCTURED_BLOCK_CORPUS,
    };
    use crate::RouteKind;
    use crate::{parse_model, parse_module};

    #[test]
    fn analyzes_structured_block_corpus() {
        let src = STRUCTURED_BLOCK_CORPUS;
        let module = parse_module(src).expect("structured-block fixture parses");
        let typed = analyze_module(&module).expect("structured-block fixture analyzes");

        assert_eq!(typed.models.len(), 4);
        let transit = &typed.models[1];
        assert_eq!(transit.kind, ModelKind::Ode);
        assert_eq!(transit.states[0].size, Some(4));
        assert!(transit.dynamics.is_some());

        let analytical = &typed.models[2];
        assert!(matches!(
            analytical.analytical.as_ref().map(|value| value.structure),
            Some(AnalyticalKernel::OneCompartmentWithAbsorption)
        ));

        let sde = &typed.models[3];
        assert_eq!(sde.particles, Some(1000));
        assert!(sde.drift.is_some());
        assert!(sde.diffusion.is_some());
    }

    #[test]
    fn derives_values_across_if_branches() {
        let src = STRUCTURED_BLOCK_CORPUS;
        let model = parse_model(src.split("\n\n\n").next().unwrap()).expect("single model parses");
        let typed = analyze_model(&model).expect("single model analyzes");
        let ke_symbol = typed
            .symbols
            .iter()
            .find(|symbol| symbol.name == "ke")
            .expect("derived symbol exists");
        assert!(matches!(ke_symbol.ty, SymbolType::Scalar(ValueType::Real)));
    }

    #[test]
    fn authoring_fixture_preserves_route_kind_while_remaining_equivalent() {
        let authoring_surface = RECOMMENDED_STYLE_AUTHORING;
        let canonical = RECOMMENDED_STYLE_CANONICAL;

        let authoring_model = parse_model(authoring_surface).expect("authoring model parses");
        let canonical_model = parse_model(canonical).expect("canonical model parses");

        let authoring_typed = analyze_model(&authoring_model).expect("authoring model analyzes");
        let canonical_typed = analyze_model(&canonical_model).expect("canonical model analyzes");

        assert_eq!(
            typed_model_signature(&authoring_typed),
            typed_model_signature(&canonical_typed)
        );
        assert_eq!(authoring_typed.routes[0].kind, Some(RouteKind::Bolus));
        assert_eq!(canonical_typed.routes[0].kind, None);
    }

    #[test]
    fn rejects_unknown_route_in_rate_call() {
        let src = r#"
model broken {
  kind ode
  states { central }
  dynamics {
    ddt(central) = rate(oral)
  }
  outputs {
    cp = central
  }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("unknown route must fail");
        assert!(err.render(src).contains("unknown route `oral`"));
    }

    #[test]
    fn suggests_similar_state_name_for_unknown_identifier() {
        let src = r#"
model broken {
    kind ode
    states { central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = cental
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("unknown identifier must fail");

        assert!(err
            .diagnostic()
            .suggestions
            .iter()
            .any(|suggestion| suggestion.message.contains("did you mean `central`?")));
        assert!(err
            .render(src)
            .contains("suggestion: did you mean `central`?"));
    }

    #[test]
    fn suggests_case_variant_for_unknown_identifier() {
        let src = r#"
model broken {
    kind ode
    parameters { Ke }
    states { central }
    dynamics {
        ddt(central) = -ke * central
    }
    outputs {
        cp = central
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("case-mismatched identifier must fail");

        assert!(err
            .diagnostic()
            .suggestions
            .iter()
            .any(|suggestion| suggestion.message.contains("did you mean `Ke`?")));
        assert!(err.render(src).contains("suggestion: did you mean `Ke`?"));
    }

    #[test]
    fn suggests_similar_intrinsic_for_unknown_function() {
        let src = r#"
model broken {
    kind ode
    states { central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = sqt(central)
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("unknown function must fail");

        assert!(err
            .diagnostic()
            .suggestions
            .iter()
            .any(|suggestion| suggestion.message.contains("did you mean `sqrt`?")));
        assert!(err.render(src).contains("suggestion: did you mean `sqrt`?"));
    }

    #[test]
    fn route_scalar_usage_reports_help_and_context() {
        let src = r#"
model broken {
    kind ode
    states { central }
    routes { oral -> central }
    dynamics {
        ddt(central) = oral
    }
    outputs {
        cp = central
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("route scalar usage must fail");

        assert!(err
            .diagnostic()
            .helps
            .iter()
            .any(|help| help.contains("route inputs are read through `rate(oral)`")));
        assert!(err.render(src).contains("route `oral` declared here"));
        assert!(err
            .render(src)
            .contains("suggestion: did you mean `rate(oral)`?"));
    }

    #[test]
    fn output_scope_violation_reports_help_and_context() {
        let src = r#"
model broken {
    kind ode
    states { central }
    dynamics {
        ddt(central) = cp
    }
    outputs {
        cp = central
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("output scope violation must fail");

        assert!(
            err.diagnostic()
                .helps
                .iter()
                .any(|help| help
                    .contains("outputs are assignment targets inside the `outputs` block"))
        );
        assert!(err.render(src).contains("output `cp` declared here"));
    }

    #[test]
    fn reserved_name_reports_rename_suggestion() {
        let src = r#"
model broken {
    kind ode
    parameters { log }
    states { central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = central
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("reserved name must fail");

        assert!(err.render(src).contains("rename `log` to `log_value`"));
        assert!(err
            .diagnostic()
            .suggestions
            .iter()
            .any(|suggestion| suggestion
                .edits
                .iter()
                .any(|edit| edit.replacement == "log_value")));
    }

    #[test]
    fn duplicate_constant_points_to_first_declaration() {
        let src = r#"
model broken {
    kind ode
    constants {
        ka = 1
        ka = 2
    }
    states { central }
    dynamics {
        ddt(central) = 0
    }
    outputs {
        cp = central
    }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("duplicate constant must fail");

        assert!(err
            .render(src)
            .contains("constant `ka` first declared here"));
        assert!(err.render(src).contains("rename this constant to `ka_2`"));
    }

    #[test]
    fn rejects_missing_output_assignment_on_all_paths() {
        let src = r#"
model broken {
  kind ode
  states { central }
  dynamics {
    ddt(central) = 0
  }
  outputs {
    if true {
      cp = central
    }
  }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("partial output assignment must fail");
        assert!(err
            .render(src)
            .contains("output `cp` is not definitely assigned on all control-flow paths"));
    }

    #[test]
    fn rejects_non_integer_array_size() {
        let src = r#"
model broken {
  kind ode
  constants { n = 1.5 }
  states { transit[n] }
  dynamics {
    ddt(transit[0]) = 0
  }
  outputs {
    cp = 0
  }
}
"#;
        let model = parse_model(src).expect("model parses");
        let err = analyze_model(&model).expect_err("non-integer array size must fail");
        assert!(err
            .render(src)
            .contains("state array size must be an integer constant"));
    }

    fn typed_model_signature(model: &TypedModel) -> String {
        let mut lines = Vec::new();
        lines.push(format!("kind:{:?}", model.kind));
        lines.push(format!(
            "parameters:{}",
            join_names(model, &model.parameters)
        ));
        lines.push(format!("constants:{}", join_constants(model)));
        lines.push(format!("covariates:{}", join_covariates(model)));
        lines.push(format!("states:{}", join_states(model)));
        lines.push(format!("routes:{}", join_routes(model)));
        lines.push(format!("derived:{}", join_names(model, &model.derived)));
        lines.push(format!("outputs:{}", join_names(model, &model.outputs)));
        lines.push(format!("particles:{:?}", model.particles));
        lines.push(format!(
            "analytical:{:?}",
            model.analytical.as_ref().map(|value| value.structure)
        ));
        lines.push(format!(
            "derive:{}",
            model
                .derive
                .as_ref()
                .map(|block| block_signature(model, block))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "dynamics:{}",
            model
                .dynamics
                .as_ref()
                .map(|block| block_signature(model, block))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "init:{}",
            model
                .init
                .as_ref()
                .map(|block| block_signature(model, block))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "drift:{}",
            model
                .drift
                .as_ref()
                .map(|block| block_signature(model, block))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "diffusion:{}",
            model
                .diffusion
                .as_ref()
                .map(|block| block_signature(model, block))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "outputs_block:{}",
            block_signature(model, &model.outputs_block)
        ));
        lines.join("\n")
    }

    fn join_names(model: &TypedModel, ids: &[SymbolId]) -> String {
        ids.iter()
            .map(|id| symbol_name(model, *id))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn join_constants(model: &TypedModel) -> String {
        model
            .constants
            .iter()
            .map(|constant| {
                format!(
                    "{}={:?}",
                    symbol_name(model, constant.symbol),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn join_covariates(model: &TypedModel) -> String {
        model
            .covariates
            .iter()
            .map(|covariate| {
                format!(
                    "{}@{:?}",
                    symbol_name(model, covariate.symbol),
                    covariate.interpolation
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn join_states(model: &TypedModel) -> String {
        model
            .states
            .iter()
            .map(|state| format!("{}[{:#?}]", symbol_name(model, state.symbol), state.size))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn join_routes(model: &TypedModel) -> String {
        model
            .routes
            .iter()
            .map(|route| {
                let destination = state_place_signature(model, &route.destination);
                let properties = route
                    .properties
                    .iter()
                    .map(|property| {
                        format!(
                            "{:?}={}",
                            property.kind,
                            expr_signature(model, &property.value)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                format!(
                    "{}->{}{{{}}}",
                    symbol_name(model, route.symbol),
                    destination,
                    properties
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn block_signature(model: &TypedModel, block: &TypedStatementBlock) -> String {
        block
            .statements
            .iter()
            .map(|stmt| stmt_signature(model, stmt))
            .collect::<Vec<_>>()
            .join(";")
    }

    fn stmt_signature(model: &TypedModel, stmt: &TypedStmt) -> String {
        match &stmt.kind {
            TypedStmtKind::Let(value) => format!(
                "let({}:{})",
                symbol_name(model, value.symbol),
                expr_signature(model, &value.value)
            ),
            TypedStmtKind::Assign(value) => format!(
                "assign({}={})",
                assign_target_signature(model, &value.target),
                expr_signature(model, &value.value)
            ),
            TypedStmtKind::If(value) => format!(
                "if({}){{{}}}else{{{}}}",
                expr_signature(model, &value.condition),
                value
                    .then_branch
                    .iter()
                    .map(|stmt| stmt_signature(model, stmt))
                    .collect::<Vec<_>>()
                    .join(";"),
                value
                    .else_branch
                    .as_ref()
                    .map(|branch| branch
                        .iter()
                        .map(|stmt| stmt_signature(model, stmt))
                        .collect::<Vec<_>>()
                        .join(";"))
                    .unwrap_or_default()
            ),
            TypedStmtKind::For(value) => format!(
                "for({}:{}..{}){{{}}}",
                symbol_name(model, value.binding),
                expr_signature(model, &value.range.start),
                expr_signature(model, &value.range.end),
                value
                    .body
                    .iter()
                    .map(|stmt| stmt_signature(model, stmt))
                    .collect::<Vec<_>>()
                    .join(";")
            ),
        }
    }

    fn assign_target_signature(model: &TypedModel, target: &TypedAssignTarget) -> String {
        match &target.kind {
            TypedAssignTargetKind::Derived(symbol) => {
                format!("derived:{}", symbol_name(model, *symbol))
            }
            TypedAssignTargetKind::Output(symbol) => {
                format!("output:{}", symbol_name(model, *symbol))
            }
            TypedAssignTargetKind::StateInit(place) => {
                format!("init:{}", state_place_signature(model, place))
            }
            TypedAssignTargetKind::Derivative(place) => {
                format!("ddt:{}", state_place_signature(model, place))
            }
            TypedAssignTargetKind::Noise(place) => {
                format!("noise:{}", state_place_signature(model, place))
            }
        }
    }

    fn state_place_signature(model: &TypedModel, place: &TypedStatePlace) -> String {
        let name = symbol_name(model, place.state);
        match &place.index {
            Some(index) => format!("{}[{}]", name, expr_signature(model, index)),
            None => name,
        }
    }

    fn expr_signature(model: &TypedModel, expr: &TypedExpr) -> String {
        match &expr.kind {
            TypedExprKind::Literal(value) => format!("lit:{value:?}:{:?}", expr.ty),
            TypedExprKind::Symbol(symbol) => {
                format!("sym:{}:{:?}", symbol_name(model, *symbol), expr.ty)
            }
            TypedExprKind::StateValue(place) => format!(
                "state:{}:{:?}",
                state_place_signature(model, place),
                expr.ty
            ),
            TypedExprKind::Unary { op, expr: inner } => {
                format!("un:{op:?}:{}", expr_signature(model, inner))
            }
            TypedExprKind::Binary { op, lhs, rhs } => format!(
                "bin:{op:?}:{}:{}:{:?}",
                expr_signature(model, lhs),
                expr_signature(model, rhs),
                expr.ty
            ),
            TypedExprKind::Call { callee, args } => format!(
                "call:{}({})",
                match callee {
                    TypedCall::Math(intrinsic) => format!("math:{intrinsic:?}"),
                    TypedCall::Rate(symbol) => format!("rate:{}", symbol_name(model, *symbol)),
                },
                args.iter()
                    .map(|arg| expr_signature(model, arg))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }

    fn symbol_name(model: &TypedModel, symbol: SymbolId) -> String {
        model
            .symbols
            .iter()
            .find(|entry| entry.id == symbol)
            .map(|entry| entry.name.clone())
            .unwrap_or_else(|| format!("#{symbol}"))
    }
}
