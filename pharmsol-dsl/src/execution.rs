//! Ready-to-run model compiled from an analyzed model.
//!
//! An [`ExecutionModel`] carries everything a simulation backend needs:
//! resolved symbols, dense buffer layouts, and the model functions
//! (dynamics, outputs, init, and friends) as straight-line programs.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::{
    AnalyticalKernel, AnalyzedAssignTargetKind, AnalyzedBinaryOp, AnalyzedCall, AnalyzedExpr,
    AnalyzedExprKind, AnalyzedModel, AnalyzedModule, AnalyzedRangeExpr, AnalyzedStatePlace,
    AnalyzedStatementBlock, AnalyzedStmt, AnalyzedStmtKind, AnalyzedUnaryOp, ConstValue,
    CovariateInterpolation, Diagnostic, DiagnosticPhase, DiagnosticReport, MathFunction, ModelKind,
    RouteKind, RoutePropertyKind, Span, Symbol, SymbolId, SymbolKind, SymbolType, ValueType,
    DSL_COMPILE_GENERIC,
};

/// Compiles every model in an analyzed module into its ready-to-run form.
///
/// This is the final pipeline stage after [`parse_module`](crate::parse_module)
/// and [`analyze_module`](crate::analyze_module).
pub fn compile_analyzed_module(module: &AnalyzedModule) -> Result<ExecutionModule, CompileError> {
    let mut models = Vec::with_capacity(module.models.len());
    for model in &module.models {
        models.push(compile_analyzed_model(model)?);
    }
    Ok(ExecutionModule {
        models,
        span: module.span,
    })
}

/// Compiles an analyzed model into its ready-to-run [`ExecutionModel`].
///
/// This is the final pipeline stage after [`parse_model`](crate::parse_model)
/// and [`analyze_model`](crate::analyze_model).
pub fn compile_analyzed_model(model: &AnalyzedModel) -> Result<ExecutionModel, CompileError> {
    ModelCompiler::new(model)?.compile()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionModule {
    pub models: Vec<ExecutionModel>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionModel {
    pub name: String,
    pub kind: ModelKind,
    pub metadata: ExecutionMetadata,
    pub layout: ExecutionLayout,
    pub functions: Vec<ModelFunction>,
    pub span: Span,
}

impl ExecutionModel {
    pub fn function(&self, kind: ModelFunctionKind) -> Option<&ModelFunction> {
        self.functions.iter().find(|function| function.kind == kind)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionMetadata {
    pub constants: Vec<ExecutionConstant>,
    pub parameters: Vec<ExecutionSlot>,
    pub covariates: Vec<ExecutionCovariate>,
    pub states: Vec<ExecutionState>,
    pub routes: Vec<ExecutionRoute>,
    pub derived: Vec<ExecutionSlot>,
    pub outputs: Vec<ExecutionSlot>,
    pub particles: Option<usize>,
    pub analytical: Option<AnalyticalKernel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionConstant {
    pub symbol: SymbolId,
    pub name: String,
    pub value: ConstValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionSlot {
    pub symbol: SymbolId,
    pub name: String,
    pub index: usize,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionCovariate {
    pub symbol: SymbolId,
    pub name: String,
    pub index: usize,
    pub interpolation: Option<CovariateInterpolation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionState {
    pub symbol: SymbolId,
    pub name: String,
    pub offset: usize,
    pub len: usize,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionRoute {
    pub symbol: SymbolId,
    pub name: String,
    pub declaration_index: usize,
    pub index: usize,
    pub kind: Option<RouteKind>,
    pub destination: RouteDestination,
    pub has_lag: bool,
    pub has_bioavailability: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteDestination {
    pub state: SymbolId,
    pub state_name: String,
    pub state_offset: usize,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionLayout {
    pub scalar: ScalarType,
    pub parameter_buffer: BufferLayout,
    pub covariate_buffer: BufferLayout,
    pub state_buffer: BufferLayout,
    pub derived_buffer: BufferLayout,
    pub output_buffer: BufferLayout,
    pub route_buffer: BufferLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferLayout {
    pub kind: BufferKind,
    pub len: usize,
    pub slots: Vec<BufferSlot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    Parameters,
    Covariates,
    States,
    Derived,
    Outputs,
    Routes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlot {
    pub name: String,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelFunction {
    pub kind: ModelFunctionKind,
    pub signature: FunctionSignature,
    pub body: FunctionBody,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelFunctionKind {
    Derive,
    Dynamics,
    Outputs,
    Init,
    Drift,
    Diffusion,
    RouteLag,
    RouteBioavailability,
    Analytical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    pub args: Vec<FunctionArgument>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionArgument {
    pub kind: FunctionArgumentKind,
    pub access: Access,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionArgumentKind {
    Time,
    Parameters,
    Covariates,
    States,
    RouteInputs,
    Derived,
    Outputs,
    StateDerivatives,
    InitialState,
    StateNoise,
    RouteLag,
    RouteBioavailability,
    AnalyticalState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionBody {
    Statements(ExecutionProgram),
    AnalyticalBuiltin(AnalyticalKernel),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionProgram {
    pub locals: Vec<ExecutionLocal>,
    pub body: ExecutionBlock,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionLocal {
    pub symbol: SymbolId,
    pub name: String,
    pub index: usize,
    pub ty: ValueType,
    pub kind: SymbolKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBlock {
    pub statements: Vec<ExecutionStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionStmt {
    pub kind: ExecutionStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStmtKind {
    Let(ExecutionLetStmt),
    Assign(ExecutionAssignStmt),
    If(ExecutionIfStmt),
    For(ExecutionForStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionLetStmt {
    pub local: usize,
    pub value: ExecutionExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionAssignStmt {
    pub target: ExecutionTarget,
    pub value: ExecutionExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionIfStmt {
    pub condition: ExecutionExpr,
    pub then_branch: Vec<ExecutionStmt>,
    pub else_branch: Option<Vec<ExecutionStmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionForStmt {
    pub local: usize,
    pub range: ExecutionRange,
    pub body: Vec<ExecutionStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionRange {
    pub start: ExecutionExpr,
    pub end: ExecutionExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionTarget {
    pub kind: ExecutionTargetKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionTargetKind {
    Derived(usize),
    Output(usize),
    StateInit(ExecutionStateRef),
    StateDerivative(ExecutionStateRef),
    StateNoise(ExecutionStateRef),
    RouteLag(usize),
    RouteBioavailability(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionStateRef {
    pub symbol: SymbolId,
    pub base_offset: usize,
    pub len: usize,
    pub index: Option<Box<ExecutionExpr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionExpr {
    pub kind: ExecutionExprKind,
    pub ty: ValueType,
    pub constant: Option<ConstValue>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionExprKind {
    Literal(ConstValue),
    Load(ExecutionLoad),
    Unary {
        op: AnalyzedUnaryOp,
        expr: Box<ExecutionExpr>,
    },
    Binary {
        op: AnalyzedBinaryOp,
        lhs: Box<ExecutionExpr>,
        rhs: Box<ExecutionExpr>,
    },
    Call {
        callee: ExecutionCall,
        args: Vec<ExecutionExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionLoad {
    Parameter(usize),
    Covariate(usize),
    State(ExecutionStateRef),
    Derived(usize),
    Local(usize),
    RouteInput { route: SymbolId, index: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionCall {
    Math(MathFunction),
}

#[derive(Clone, PartialEq, Eq)]
pub struct CompileError {
    diagnostic: Box<Diagnostic>,
    source: Option<Arc<str>>,
}

impl CompileError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            diagnostic: Box::new(Diagnostic::error(
                DSL_COMPILE_GENERIC,
                DiagnosticPhase::Compile,
                message,
                span,
            )),
            source: None,
        }
    }

    fn with_note(mut self, note: impl Into<String>) -> Self {
        self.diagnostic.notes.push(note.into());
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

impl fmt::Debug for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = self.source() {
            return f.write_str(&self.render(source));
        }
        let span = self.diagnostic.primary_span();
        write!(
            f,
            "error[{}]: {} (at bytes {}..{})",
            self.diagnostic.code, self.diagnostic.message, span.start, span.end
        )
    }
}

impl std::error::Error for CompileError {}

struct ModelCompiler<'a> {
    model: &'a AnalyzedModel,
    metadata: ExecutionMetadata,
    symbol_map: BTreeMap<SymbolId, &'a Symbol>,
    parameter_slots: BTreeMap<SymbolId, usize>,
    covariate_slots: BTreeMap<SymbolId, usize>,
    state_slots: BTreeMap<SymbolId, StateLayout>,
    route_slots: BTreeMap<SymbolId, usize>,
    derived_slots: BTreeMap<SymbolId, usize>,
    output_slots: BTreeMap<SymbolId, usize>,
}

#[derive(Debug, Clone, Copy)]
struct StateLayout {
    offset: usize,
    len: usize,
}

impl<'a> ModelCompiler<'a> {
    fn new(model: &'a AnalyzedModel) -> Result<Self, CompileError> {
        let symbol_map: BTreeMap<SymbolId, &Symbol> = model
            .symbols
            .iter()
            .map(|symbol| (symbol.id, symbol))
            .collect();

        let constants = model
            .constants
            .iter()
            .map(|constant| {
                let symbol = lookup_symbol(&symbol_map, constant.symbol, constant.span)?;
                Ok(ExecutionConstant {
                    symbol: constant.symbol,
                    name: symbol.name.clone(),
                    value: constant.value.clone(),
                    span: constant.span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let mut parameter_slots = BTreeMap::new();
        let parameters = model
            .parameters
            .iter()
            .enumerate()
            .map(|(index, symbol_id)| {
                let symbol = lookup_symbol(&symbol_map, *symbol_id, model.span)?;
                parameter_slots.insert(*symbol_id, index);
                Ok(ExecutionSlot {
                    symbol: *symbol_id,
                    name: symbol.name.clone(),
                    index,
                    span: symbol.span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let mut covariate_slots = BTreeMap::new();
        let covariates = model
            .covariates
            .iter()
            .enumerate()
            .map(|(index, covariate)| {
                let symbol = lookup_symbol(&symbol_map, covariate.symbol, covariate.span)?;
                covariate_slots.insert(covariate.symbol, index);
                Ok(ExecutionCovariate {
                    symbol: covariate.symbol,
                    name: symbol.name.clone(),
                    index,
                    interpolation: covariate.interpolation,
                    span: covariate.span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let mut state_slots = BTreeMap::new();
        let mut states = Vec::with_capacity(model.states.len());
        let mut next_state_offset = 0usize;
        for state in &model.states {
            let symbol = lookup_symbol(&symbol_map, state.symbol, state.span)?;
            let len = state.size.unwrap_or(1);
            state_slots.insert(
                state.symbol,
                StateLayout {
                    offset: next_state_offset,
                    len,
                },
            );
            states.push(ExecutionState {
                symbol: state.symbol,
                name: symbol.name.clone(),
                offset: next_state_offset,
                len,
                span: state.span,
            });
            next_state_offset = next_state_offset.checked_add(len).ok_or_else(|| {
                CompileError::new(
                    "combined state sizes exceed the supported state space",
                    state.span,
                )
            })?;
        }

        let uses_authoring_route_kinds =
            !model.routes.is_empty() && model.routes.iter().all(|route| route.kind.is_some());
        let mut route_slots = BTreeMap::new();
        let mut routes = Vec::with_capacity(model.routes.len());
        let mut next_bolus_index = 0usize;
        let mut next_infusion_index = 0usize;
        for (declaration_index, route) in model.routes.iter().enumerate() {
            let symbol = lookup_symbol(&symbol_map, route.symbol, route.span)?;
            if route.kind == Some(RouteKind::Infusion) {
                if let Some(property) = route.properties.first() {
                    let label = match property.kind {
                        RoutePropertyKind::Lag => "lag",
                        RoutePropertyKind::Bioavailability => "bioavailability",
                    };
                    return Err(CompileError::new(
                        format!(
                            "DSL authoring does not allow `{label}` on infusion route `{}`",
                            symbol.name
                        ),
                        property.span,
                    )
                    .with_note("lag and bioavailability are bolus-only route properties"));
                }
            }
            // Authoring source always records a kind per route; canonical
            // `model {}` source never does. Mixed models fall back to
            // declaration order.
            let index = match (uses_authoring_route_kinds, route.kind) {
                (true, Some(RouteKind::Bolus)) => {
                    let index = next_bolus_index;
                    next_bolus_index += 1;
                    index
                }
                (true, Some(RouteKind::Infusion)) => {
                    let index = next_infusion_index;
                    next_infusion_index += 1;
                    index
                }
                _ => declaration_index,
            };
            route_slots.insert(route.symbol, index);
            let destination =
                compile_route_destination(&symbol_map, &state_slots, &route.destination)?;
            routes.push(ExecutionRoute {
                symbol: route.symbol,
                name: symbol.name.clone(),
                declaration_index,
                index,
                kind: route.kind,
                destination,
                has_lag: route
                    .properties
                    .iter()
                    .any(|property| property.kind == RoutePropertyKind::Lag),
                has_bioavailability: route
                    .properties
                    .iter()
                    .any(|property| property.kind == RoutePropertyKind::Bioavailability),
                span: route.span,
            });
        }

        let mut derived_slots = BTreeMap::new();
        let derived = model
            .derived
            .iter()
            .enumerate()
            .map(|(index, symbol_id)| {
                let symbol = lookup_symbol(&symbol_map, *symbol_id, model.span)?;
                derived_slots.insert(*symbol_id, index);
                Ok(ExecutionSlot {
                    symbol: *symbol_id,
                    name: symbol.name.clone(),
                    index,
                    span: symbol.span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let mut output_slots = BTreeMap::new();
        let outputs = model
            .outputs
            .iter()
            .enumerate()
            .map(|(index, symbol_id)| {
                let symbol = lookup_symbol(&symbol_map, *symbol_id, model.span)?;
                output_slots.insert(*symbol_id, index);
                Ok(ExecutionSlot {
                    symbol: *symbol_id,
                    name: symbol.name.clone(),
                    index,
                    span: symbol.span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        Ok(Self {
            model,
            metadata: ExecutionMetadata {
                constants,
                parameters,
                covariates,
                states,
                routes,
                derived,
                outputs,
                particles: model.particles,
                analytical: model
                    .analytical
                    .as_ref()
                    .map(|analytical| analytical.structure),
            },
            symbol_map,
            parameter_slots,
            covariate_slots,
            state_slots,
            route_slots,
            derived_slots,
            output_slots,
        })
    }

    fn compile(self) -> Result<ExecutionModel, CompileError> {
        let layout = self.build_layout();
        let mut functions = Vec::new();

        if let Some(block) = &self.model.derive {
            functions.push(self.compile_statement_function(ModelFunctionKind::Derive, block)?);
        }
        if let Some(block) = &self.model.init {
            functions.push(self.compile_init_function(block)?);
        }
        if let Some(block) = &self.model.dynamics {
            functions.push(self.compile_statement_function(ModelFunctionKind::Dynamics, block)?);
        }
        if let Some(block) = &self.model.drift {
            functions.push(self.compile_statement_function(ModelFunctionKind::Drift, block)?);
        }
        if let Some(block) = &self.model.diffusion {
            functions.push(self.compile_statement_function(ModelFunctionKind::Diffusion, block)?);
        }
        if let Some(function) = self
            .compile_route_property_function(RoutePropertyKind::Lag, ModelFunctionKind::RouteLag)?
        {
            functions.push(function);
        }
        if let Some(function) = self.compile_route_property_function(
            RoutePropertyKind::Bioavailability,
            ModelFunctionKind::RouteBioavailability,
        )? {
            functions.push(function);
        }
        if let Some(analytical) = &self.model.analytical {
            functions.push(ModelFunction {
                kind: ModelFunctionKind::Analytical,
                signature: signature_for(ModelFunctionKind::Analytical),
                body: FunctionBody::AnalyticalBuiltin(analytical.structure),
                span: analytical.span,
            });
        }
        functions.push(
            self.compile_statement_function(ModelFunctionKind::Outputs, &self.model.outputs_block)?,
        );

        Ok(ExecutionModel {
            name: self.model.name.clone(),
            kind: self.model.kind,
            metadata: self.metadata,
            layout,
            functions,
            span: self.model.span,
        })
    }

    fn build_layout(&self) -> ExecutionLayout {
        ExecutionLayout {
            scalar: ScalarType::F64,
            parameter_buffer: BufferLayout {
                kind: BufferKind::Parameters,
                len: self.metadata.parameters.len(),
                slots: self
                    .metadata
                    .parameters
                    .iter()
                    .map(|slot| BufferSlot {
                        name: slot.name.clone(),
                        offset: slot.index,
                        len: 1,
                    })
                    .collect(),
            },
            covariate_buffer: BufferLayout {
                kind: BufferKind::Covariates,
                len: self.metadata.covariates.len(),
                slots: self
                    .metadata
                    .covariates
                    .iter()
                    .map(|slot| BufferSlot {
                        name: slot.name.clone(),
                        offset: slot.index,
                        len: 1,
                    })
                    .collect(),
            },
            state_buffer: BufferLayout {
                kind: BufferKind::States,
                len: self.metadata.states.iter().map(|state| state.len).sum(),
                slots: self
                    .metadata
                    .states
                    .iter()
                    .map(|state| BufferSlot {
                        name: state.name.clone(),
                        offset: state.offset,
                        len: state.len,
                    })
                    .collect(),
            },
            derived_buffer: BufferLayout {
                kind: BufferKind::Derived,
                len: self.metadata.derived.len(),
                slots: self
                    .metadata
                    .derived
                    .iter()
                    .map(|slot| BufferSlot {
                        name: slot.name.clone(),
                        offset: slot.index,
                        len: 1,
                    })
                    .collect(),
            },
            output_buffer: BufferLayout {
                kind: BufferKind::Outputs,
                len: self.metadata.outputs.len(),
                slots: self
                    .metadata
                    .outputs
                    .iter()
                    .map(|slot| BufferSlot {
                        name: slot.name.clone(),
                        offset: slot.index,
                        len: 1,
                    })
                    .collect(),
            },
            route_buffer: BufferLayout {
                kind: BufferKind::Routes,
                len: self
                    .metadata
                    .routes
                    .iter()
                    .map(|route| route.index + 1)
                    .max()
                    .unwrap_or(0),
                slots: self
                    .metadata
                    .routes
                    .iter()
                    .map(|route| BufferSlot {
                        name: route.name.clone(),
                        offset: route.index,
                        len: 1,
                    })
                    .collect(),
            },
        }
    }

    fn compile_statement_function(
        &self,
        kind: ModelFunctionKind,
        block: &AnalyzedStatementBlock,
    ) -> Result<ModelFunction, CompileError> {
        let mut locals = CompileLocals::default();
        let statements = block
            .statements
            .iter()
            .map(|stmt| self.compile_stmt(stmt, &mut locals))
            .collect::<Result<Vec<_>, CompileError>>()?;

        Ok(ModelFunction {
            kind,
            signature: signature_for(kind),
            body: FunctionBody::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: block.span,
                },
            }),
            span: block.span,
        })
    }

    fn compile_init_function(
        &self,
        block: &AnalyzedStatementBlock,
    ) -> Result<ModelFunction, CompileError> {
        let mut locals = CompileLocals::default();
        let mut statements = self
            .metadata
            .states
            .iter()
            .flat_map(|state| {
                let base = (0..state.len).map(|component| ExecutionStmt {
                    kind: ExecutionStmtKind::Assign(ExecutionAssignStmt {
                        target: ExecutionTarget {
                            kind: ExecutionTargetKind::StateInit(ExecutionStateRef {
                                symbol: state.symbol,
                                base_offset: state.offset,
                                len: state.len,
                                index: if state.len == 1 {
                                    None
                                } else {
                                    Some(Box::new(literal_int(component as i64, state.span)))
                                },
                                span: state.span,
                            }),
                            span: state.span,
                        },
                        value: literal_real(0.0, state.span),
                    }),
                    span: state.span,
                });
                base.collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        statements.extend(
            block
                .statements
                .iter()
                .map(|stmt| self.compile_stmt(stmt, &mut locals))
                .collect::<Result<Vec<_>, CompileError>>()?,
        );

        Ok(ModelFunction {
            kind: ModelFunctionKind::Init,
            signature: signature_for(ModelFunctionKind::Init),
            body: FunctionBody::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: block.span,
                },
            }),
            span: block.span,
        })
    }

    fn compile_route_property_function(
        &self,
        property_kind: RoutePropertyKind,
        kind: ModelFunctionKind,
    ) -> Result<Option<ModelFunction>, CompileError> {
        if !self.model.routes.iter().any(|route| {
            route
                .properties
                .iter()
                .any(|property| property.kind == property_kind)
        }) {
            return Ok(None);
        }

        let mut statements = Vec::with_capacity(self.model.routes.len());
        let mut locals = CompileLocals::default();
        let default_value = match property_kind {
            RoutePropertyKind::Lag => literal_real(0.0, self.model.span),
            RoutePropertyKind::Bioavailability => literal_real(1.0, self.model.span),
        };
        let route_len = self
            .metadata
            .routes
            .iter()
            .map(|route| route.index + 1)
            .max()
            .unwrap_or(0);
        for route_index in 0..route_len {
            let target_kind = match property_kind {
                RoutePropertyKind::Lag => ExecutionTargetKind::RouteLag(route_index),
                RoutePropertyKind::Bioavailability => {
                    ExecutionTargetKind::RouteBioavailability(route_index)
                }
            };
            statements.push(ExecutionStmt {
                kind: ExecutionStmtKind::Assign(ExecutionAssignStmt {
                    target: ExecutionTarget {
                        kind: target_kind,
                        span: self.model.span,
                    },
                    value: default_value.clone(),
                }),
                span: self.model.span,
            });
        }
        for route in &self.model.routes {
            if route.kind == Some(RouteKind::Infusion) {
                continue;
            }
            let route_name = self.symbol_name(route.symbol)?.to_string();
            let route_index = *self.route_slots.get(&route.symbol).ok_or_else(|| {
                CompileError::new(
                    format!("route `{}` has no execution slot", route_name),
                    route.span,
                )
            })?;
            let expression = match route
                .properties
                .iter()
                .find(|property| property.kind == property_kind)
            {
                Some(property) => self.compile_expr(&property.value, &mut locals)?,
                None => continue,
            };
            let target_kind = match property_kind {
                RoutePropertyKind::Lag => ExecutionTargetKind::RouteLag(route_index),
                RoutePropertyKind::Bioavailability => {
                    ExecutionTargetKind::RouteBioavailability(route_index)
                }
            };
            statements.push(ExecutionStmt {
                kind: ExecutionStmtKind::Assign(ExecutionAssignStmt {
                    target: ExecutionTarget {
                        kind: target_kind,
                        span: route.span,
                    },
                    value: expression,
                }),
                span: route.span,
            });
        }

        Ok(Some(ModelFunction {
            kind,
            signature: signature_for(kind),
            body: FunctionBody::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: self.model.span,
                },
            }),
            span: self.model.span,
        }))
    }

    fn compile_stmt(
        &self,
        stmt: &AnalyzedStmt,
        locals: &mut CompileLocals,
    ) -> Result<ExecutionStmt, CompileError> {
        let kind = match &stmt.kind {
            AnalyzedStmtKind::Let(let_stmt) => {
                let local = locals.local_slot(let_stmt.symbol, self)?;
                ExecutionStmtKind::Let(ExecutionLetStmt {
                    local,
                    value: self.compile_expr(&let_stmt.value, locals)?,
                })
            }
            AnalyzedStmtKind::Assign(assign) => ExecutionStmtKind::Assign(ExecutionAssignStmt {
                target: self.compile_target(&assign.target.kind, assign.target.span, locals)?,
                value: self.compile_expr(&assign.value, locals)?,
            }),
            AnalyzedStmtKind::If(if_stmt) => ExecutionStmtKind::If(ExecutionIfStmt {
                condition: self.compile_expr(&if_stmt.condition, locals)?,
                then_branch: if_stmt
                    .then_branch
                    .iter()
                    .map(|stmt| self.compile_stmt(stmt, locals))
                    .collect::<Result<Vec<_>, _>>()?,
                else_branch: if_stmt
                    .else_branch
                    .as_ref()
                    .map(|branch| {
                        branch
                            .iter()
                            .map(|stmt| self.compile_stmt(stmt, locals))
                            .collect::<Result<Vec<_>, CompileError>>()
                    })
                    .transpose()?,
            }),
            AnalyzedStmtKind::For(for_stmt) => {
                let local = locals.local_slot(for_stmt.binding, self)?;
                ExecutionStmtKind::For(ExecutionForStmt {
                    local,
                    range: self.compile_range(&for_stmt.range, locals)?,
                    body: for_stmt
                        .body
                        .iter()
                        .map(|stmt| self.compile_stmt(stmt, locals))
                        .collect::<Result<Vec<_>, _>>()?,
                })
            }
        };

        Ok(ExecutionStmt {
            kind,
            span: stmt.span,
        })
    }

    fn compile_range(
        &self,
        range: &AnalyzedRangeExpr,
        locals: &mut CompileLocals,
    ) -> Result<ExecutionRange, CompileError> {
        Ok(ExecutionRange {
            start: self.compile_expr(&range.start, locals)?,
            end: self.compile_expr(&range.end, locals)?,
            span: range.span,
        })
    }

    fn compile_target(
        &self,
        target: &AnalyzedAssignTargetKind,
        span: Span,
        locals: &mut CompileLocals,
    ) -> Result<ExecutionTarget, CompileError> {
        let kind = match target {
            AnalyzedAssignTargetKind::Derived(symbol) => {
                ExecutionTargetKind::Derived(self.slot_for_derived(*symbol, span)?)
            }
            AnalyzedAssignTargetKind::Output(symbol) => {
                ExecutionTargetKind::Output(self.slot_for_output(*symbol, span)?)
            }
            AnalyzedAssignTargetKind::StateInit(place) => {
                ExecutionTargetKind::StateInit(self.compile_state_ref(place, locals)?)
            }
            AnalyzedAssignTargetKind::Derivative(place) => {
                ExecutionTargetKind::StateDerivative(self.compile_state_ref(place, locals)?)
            }
            AnalyzedAssignTargetKind::Noise(place) => {
                ExecutionTargetKind::StateNoise(self.compile_state_ref(place, locals)?)
            }
        };
        Ok(ExecutionTarget { kind, span })
    }

    fn compile_expr(
        &self,
        expr: &AnalyzedExpr,
        locals: &mut CompileLocals,
    ) -> Result<ExecutionExpr, CompileError> {
        if let Some(constant) = &expr.constant {
            return Ok(ExecutionExpr {
                kind: ExecutionExprKind::Literal(constant.clone()),
                ty: expr.ty,
                constant: Some(constant.clone()),
                span: expr.span,
            });
        }

        let kind = match &expr.kind {
            AnalyzedExprKind::Literal(constant) => ExecutionExprKind::Literal(constant.clone()),
            AnalyzedExprKind::Symbol(symbol) => {
                let symbol_info = lookup_symbol(&self.symbol_map, *symbol, expr.span)?;
                match symbol_info.kind {
                    SymbolKind::Parameter => ExecutionExprKind::Load(ExecutionLoad::Parameter(
                        self.slot_for_parameter(*symbol, expr.span)?,
                    )),
                    SymbolKind::Covariate => ExecutionExprKind::Load(ExecutionLoad::Covariate(
                        self.slot_for_covariate(*symbol, expr.span)?,
                    )),
                    SymbolKind::Derived => ExecutionExprKind::Load(ExecutionLoad::Derived(
                        self.slot_for_derived(*symbol, expr.span)?,
                    )),
                    SymbolKind::Local | SymbolKind::LoopBinding => ExecutionExprKind::Load(
                        ExecutionLoad::Local(locals.local_slot(*symbol, self)?),
                    ),
                    SymbolKind::Constant => {
                        return Err(CompileError::new(
                            format!(
                                "constant `{}` should have been folded before execution compilation",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                    SymbolKind::State => {
                        return Err(CompileError::new(
                            format!(
                                "state `{}` should compile through a state reference",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                    SymbolKind::Route => {
                        return Err(CompileError::new(
                            format!(
                                "route `{}` is not a scalar execution input",
                                symbol_info.name
                            ),
                            expr.span,
                        )
                        .with_note("routes must compile through `rate(route)` or route metadata"));
                    }
                    SymbolKind::Output => {
                        return Err(CompileError::new(
                            format!(
                                "output `{}` cannot be read inside execution functions",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                }
            }
            AnalyzedExprKind::StateValue(place) => ExecutionExprKind::Load(ExecutionLoad::State(
                self.compile_state_ref(place, locals)?,
            )),
            AnalyzedExprKind::Unary { op, expr } => ExecutionExprKind::Unary {
                op: *op,
                expr: Box::new(self.compile_expr(expr, locals)?),
            },
            AnalyzedExprKind::Binary { op, lhs, rhs } => ExecutionExprKind::Binary {
                op: *op,
                lhs: Box::new(self.compile_expr(lhs, locals)?),
                rhs: Box::new(self.compile_expr(rhs, locals)?),
            },
            AnalyzedExprKind::Call { callee, args } => match callee {
                AnalyzedCall::Math(intrinsic) => ExecutionExprKind::Call {
                    callee: ExecutionCall::Math(*intrinsic),
                    args: args
                        .iter()
                        .map(|arg| self.compile_expr(arg, locals))
                        .collect::<Result<Vec<_>, _>>()?,
                },
                AnalyzedCall::Rate(route) => {
                    let route_name = self.symbol_name(*route)?.to_string();
                    let route_index = *self.route_slots.get(route).ok_or_else(|| {
                        CompileError::new(
                            format!("route `{}` has no execution slot", route_name),
                            expr.span,
                        )
                    })?;
                    ExecutionExprKind::Load(ExecutionLoad::RouteInput {
                        route: *route,
                        index: route_index,
                    })
                }
            },
        };

        Ok(ExecutionExpr {
            kind,
            ty: expr.ty,
            constant: None,
            span: expr.span,
        })
    }

    fn compile_state_ref(
        &self,
        place: &AnalyzedStatePlace,
        locals: &mut CompileLocals,
    ) -> Result<ExecutionStateRef, CompileError> {
        let state_name = self.symbol_name(place.state)?.to_string();
        let layout = self.state_slots.get(&place.state).copied().ok_or_else(|| {
            CompileError::new(
                format!("state `{}` has no execution layout", state_name),
                place.span,
            )
        })?;
        let index = place
            .index
            .as_ref()
            .map(|index| self.compile_expr(index, locals))
            .transpose()?
            .map(Box::new);
        Ok(ExecutionStateRef {
            symbol: place.state,
            base_offset: layout.offset,
            len: layout.len,
            index,
            span: place.span,
        })
    }

    fn slot_for_parameter(&self, symbol: SymbolId, span: Span) -> Result<usize, CompileError> {
        self.parameter_slots.get(&symbol).copied().ok_or_else(|| {
            CompileError::new(
                format!(
                    "parameter `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_covariate(&self, symbol: SymbolId, span: Span) -> Result<usize, CompileError> {
        self.covariate_slots.get(&symbol).copied().ok_or_else(|| {
            CompileError::new(
                format!(
                    "covariate `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_derived(&self, symbol: SymbolId, span: Span) -> Result<usize, CompileError> {
        self.derived_slots.get(&symbol).copied().ok_or_else(|| {
            CompileError::new(
                format!(
                    "derived value `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_output(&self, symbol: SymbolId, span: Span) -> Result<usize, CompileError> {
        self.output_slots.get(&symbol).copied().ok_or_else(|| {
            CompileError::new(
                format!(
                    "output `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn symbol_name(&self, symbol: SymbolId) -> Result<&str, CompileError> {
        Ok(&lookup_symbol(&self.symbol_map, symbol, self.model.span)?.name)
    }
}

#[derive(Default)]
struct CompileLocals {
    locals: Vec<ExecutionLocal>,
    slots: BTreeMap<SymbolId, usize>,
}

impl CompileLocals {
    fn local_slot(
        &mut self,
        symbol: SymbolId,
        compiler: &ModelCompiler<'_>,
    ) -> Result<usize, CompileError> {
        if let Some(slot) = self.slots.get(&symbol).copied() {
            return Ok(slot);
        }
        let symbol_info = lookup_symbol(&compiler.symbol_map, symbol, compiler.model.span)?;
        let ty = match symbol_info.ty {
            SymbolType::Scalar(ty) => ty,
            SymbolType::Array { .. } => {
                return Err(CompileError::new(
                    format!("local `{}` must be scalar", symbol_info.name),
                    symbol_info.span,
                ));
            }
            SymbolType::Route => {
                return Err(CompileError::new(
                    format!("local `{}` cannot be a route handle", symbol_info.name),
                    symbol_info.span,
                ));
            }
        };
        let slot = self.locals.len();
        self.locals.push(ExecutionLocal {
            symbol,
            name: symbol_info.name.clone(),
            index: slot,
            ty,
            kind: symbol_info.kind,
            span: symbol_info.span,
        });
        self.slots.insert(symbol, slot);
        Ok(slot)
    }
}

fn signature_for(kind: ModelFunctionKind) -> FunctionSignature {
    let args = match kind {
        ModelFunctionKind::Derive => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Output),
        ],
        ModelFunctionKind::Dynamics => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::StateDerivatives, Access::Output),
        ],
        ModelFunctionKind::Outputs => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::Outputs, Access::Output),
        ],
        ModelFunctionKind::Init => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::InitialState, Access::Output),
        ],
        ModelFunctionKind::Drift => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::StateDerivatives, Access::Output),
        ],
        ModelFunctionKind::Diffusion => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::StateNoise, Access::Output),
        ],
        ModelFunctionKind::RouteLag => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::RouteLag, Access::Output),
        ],
        ModelFunctionKind::RouteBioavailability => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::RouteBioavailability, Access::Output),
        ],
        ModelFunctionKind::Analytical => vec![
            arg(FunctionArgumentKind::Time, Access::Input),
            arg(FunctionArgumentKind::States, Access::Input),
            arg(FunctionArgumentKind::Parameters, Access::Input),
            arg(FunctionArgumentKind::Covariates, Access::Input),
            arg(FunctionArgumentKind::RouteInputs, Access::Input),
            arg(FunctionArgumentKind::Derived, Access::Input),
            arg(FunctionArgumentKind::AnalyticalState, Access::Output),
        ],
    };
    FunctionSignature { args }
}

fn arg(kind: FunctionArgumentKind, access: Access) -> FunctionArgument {
    FunctionArgument { kind, access }
}

fn lookup_symbol<'a>(
    symbols: &'a BTreeMap<SymbolId, &'a Symbol>,
    symbol: SymbolId,
    span: Span,
) -> Result<&'a Symbol, CompileError> {
    symbols.get(&symbol).copied().ok_or_else(|| {
        CompileError::new(
            format!("symbol id {symbol} is missing from the analyzed model symbol table"),
            span,
        )
    })
}

fn compile_route_destination(
    symbols: &BTreeMap<SymbolId, &Symbol>,
    state_slots: &BTreeMap<SymbolId, StateLayout>,
    destination: &AnalyzedStatePlace,
) -> Result<RouteDestination, CompileError> {
    let symbol = lookup_symbol(symbols, destination.state, destination.span)?;
    let layout = state_slots
        .get(&destination.state)
        .copied()
        .ok_or_else(|| {
            CompileError::new(
                format!("state `{}` has no execution layout", symbol.name),
                destination.span,
            )
        })?;
    let element = match &destination.index {
        None => 0,
        Some(index) => constant_index(index, destination.span)?,
    };
    if element >= layout.len {
        return Err(CompileError::new(
            format!(
                "route destination for `{}` indexes element {}, but state length is {}",
                symbol.name, element, layout.len
            ),
            destination.span,
        ));
    }
    Ok(RouteDestination {
        state: destination.state,
        state_name: symbol.name.clone(),
        state_offset: layout.offset + element,
        span: destination.span,
    })
}

fn constant_index(expr: &AnalyzedExpr, span: Span) -> Result<usize, CompileError> {
    let value = expr
        .constant
        .as_ref()
        .and_then(ConstValue::as_i64)
        .ok_or_else(|| CompileError::new("expected a compile-time integer index", span))?;
    if value < 0 {
        return Err(CompileError::new(
            "expected a non-negative compile-time index",
            span,
        ));
    }
    Ok(value as usize)
}

fn literal_real(value: f64, span: Span) -> ExecutionExpr {
    ExecutionExpr {
        kind: ExecutionExprKind::Literal(ConstValue::Real(value)),
        ty: ValueType::Real,
        constant: Some(ConstValue::Real(value)),
        span,
    }
}

fn literal_int(value: i64, span: Span) -> ExecutionExpr {
    ExecutionExpr {
        kind: ExecutionExprKind::Literal(ConstValue::Int(value)),
        ty: ValueType::Int,
        constant: Some(ConstValue::Int(value)),
        span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::STRUCTURED_BLOCK_CORPUS;
    use crate::{analyze_module, parse_module};

    #[test]
    fn compiles_structured_block_corpus_into_execution_models() {
        let execution = structured_block_execution();
        assert_eq!(execution.models.len(), 4);

        let ode = find_model(&execution, "one_cmt_oral_iv");
        assert_eq!(ode.layout.parameter_buffer.len, 5);
        assert_eq!(ode.layout.covariate_buffer.len, 1);
        assert_eq!(ode.layout.state_buffer.len, 2);
        assert_eq!(ode.layout.derived_buffer.len, 3);
        assert_eq!(ode.layout.output_buffer.len, 1);
        assert_eq!(ode.layout.route_buffer.len, 2);
        assert_eq!(ode.metadata.routes[0].destination.state_offset, 0);
        assert_eq!(
            function_kinds(ode),
            vec![
                ModelFunctionKind::Derive,
                ModelFunctionKind::Dynamics,
                ModelFunctionKind::RouteLag,
                ModelFunctionKind::RouteBioavailability,
                ModelFunctionKind::Outputs,
            ]
        );
    }

    #[test]
    fn authoring_routes_share_input_indices_by_kind_local_ordinal() {
        let src = r#"name = shared_authoring
kind = ode

params = ka, ke, v, tlag, f_oral
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(cp) = central / v ~ continuous()
"#;

        let model = crate::parse_model(src).expect("authoring model parses");
        let analyzed = crate::analyze_model(&model).expect("authoring model analyzes");
        let compiled = crate::compile_analyzed_model(&analyzed).expect("authoring model compiles");

        assert_eq!(compiled.layout.route_buffer.len, 1);
        assert_eq!(compiled.metadata.routes.len(), 2);
        assert_eq!(compiled.metadata.routes[0].kind, Some(RouteKind::Bolus));
        assert_eq!(compiled.metadata.routes[1].kind, Some(RouteKind::Infusion));
        assert_eq!(compiled.metadata.routes[0].declaration_index, 0);
        assert_eq!(compiled.metadata.routes[1].declaration_index, 1);
        assert_eq!(compiled.metadata.routes[0].index, 0);
        assert_eq!(compiled.metadata.routes[1].index, 0);
        assert!(compiled.metadata.routes[0].has_lag);
        assert!(compiled.metadata.routes[0].has_bioavailability);
        assert!(!compiled.metadata.routes[1].has_lag);
        assert!(!compiled.metadata.routes[1].has_bioavailability);
    }

    #[test]
    fn canonical_numeric_channel_names_flow_into_execution_metadata_and_abi() {
        let src = r#"name = canonical_numeric_channels
kind = ode

params = ke, v
states = depot, central
outputs = cp, outeq_2

bolus(input_10) -> depot
infusion(iv) -> central

dx(depot) = -ke * depot
dx(central) = rate(input_10) - ke * central

out(cp) = central / v
out(outeq_2) = depot / v
"#;

        let model = crate::parse_model(src).expect("authoring model parses");
        let analyzed = crate::analyze_model(&model).expect("authoring model analyzes");
        let compiled = crate::compile_analyzed_model(&analyzed).expect("authoring model compiles");

        assert_eq!(
            compiled
                .metadata
                .routes
                .iter()
                .map(|route| route.name.as_str())
                .collect::<Vec<_>>(),
            vec!["input_10", "iv"]
        );
        assert_eq!(
            compiled
                .metadata
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cp", "outeq_2"]
        );
        assert_eq!(
            compiled
                .layout
                .route_buffer
                .slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            vec!["input_10", "iv"]
        );
        assert_eq!(
            compiled
                .layout
                .output_buffer
                .slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cp", "outeq_2"]
        );
    }

    #[test]
    fn authoring_routes_reject_infusion_lag_properties() {
        let src = r#"name = invalid_infusion_lag
kind = ode

params = ke, v, tlag
states = central
outputs = cp

infusion(iv) -> central
lag(iv) = tlag

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

        let model = crate::parse_model(src).expect("authoring model parses");
        let analyzed = crate::analyze_model(&model).expect("authoring model analyzes");
        let error = crate::compile_analyzed_model(&analyzed)
            .expect_err("infusion lag should fail during compilation");

        assert!(error
            .to_string()
            .contains("DSL authoring does not allow `lag` on infusion route `iv`"));
    }

    #[test]
    fn authoring_routes_reject_infusion_bioavailability_properties() {
        let src = r#"name = invalid_infusion_fa
kind = ode

params = ke, v, f_iv
states = central
outputs = cp

infusion(iv) -> central
fa(iv) = f_iv

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

        let model = crate::parse_model(src).expect("authoring model parses");
        let analyzed = crate::analyze_model(&model).expect("authoring model analyzes");
        let error = crate::compile_analyzed_model(&analyzed)
            .expect_err("infusion bioavailability should fail during compilation");

        assert!(error
            .to_string()
            .contains("DSL authoring does not allow `bioavailability` on infusion route `iv`"));
    }

    #[test]
    fn flattens_array_states_and_preserves_loop_structure() {
        let execution = structured_block_execution();
        let transit = find_model(&execution, "transit_absorption");
        assert_eq!(transit.layout.state_buffer.len, 5);
        assert_eq!(transit.metadata.states[0].name, "transit");
        assert_eq!(transit.metadata.states[0].offset, 0);
        assert_eq!(transit.metadata.states[0].len, 4);
        assert_eq!(transit.metadata.states[1].name, "central");
        assert_eq!(transit.metadata.states[1].offset, 4);
        assert!(transit.function(ModelFunctionKind::RouteLag).is_none());
        assert!(transit
            .function(ModelFunctionKind::RouteBioavailability)
            .is_none());

        let dynamics = transit
            .function(ModelFunctionKind::Dynamics)
            .expect("dynamics function");
        let FunctionBody::Statements(program) = &dynamics.body else {
            panic!("expected statement-based dynamics function");
        };
        assert!(program
            .body
            .statements
            .iter()
            .any(|stmt| matches!(stmt.kind, ExecutionStmtKind::For(_))));
    }

    #[test]
    fn analytical_models_compile_to_builtin_execution_functions() {
        let execution = structured_block_execution();
        let analytical = find_model(&execution, "one_cmt_abs");
        let function = analytical
            .function(ModelFunctionKind::Analytical)
            .expect("analytical function");
        assert_eq!(
            function.signature.args,
            vec![
                arg(FunctionArgumentKind::Time, Access::Input),
                arg(FunctionArgumentKind::States, Access::Input),
                arg(FunctionArgumentKind::Parameters, Access::Input),
                arg(FunctionArgumentKind::Covariates, Access::Input),
                arg(FunctionArgumentKind::RouteInputs, Access::Input),
                arg(FunctionArgumentKind::Derived, Access::Input),
                arg(FunctionArgumentKind::AnalyticalState, Access::Output),
            ]
        );
        assert!(matches!(
            function.body,
            FunctionBody::AnalyticalBuiltin(AnalyticalKernel::OneCompartmentWithAbsorption)
        ));
    }

    #[test]
    fn sde_models_emit_runtime_functions_and_zero_filled_init() {
        let execution = structured_block_execution();
        let sde = find_model(&execution, "vanco_sde");
        assert_eq!(sde.metadata.particles, Some(1000));
        assert_eq!(
            function_kinds(sde),
            vec![
                ModelFunctionKind::Init,
                ModelFunctionKind::Drift,
                ModelFunctionKind::Diffusion,
                ModelFunctionKind::Outputs,
            ]
        );

        let init = sde
            .function(ModelFunctionKind::Init)
            .expect("init function");
        let FunctionBody::Statements(program) = &init.body else {
            panic!("expected statement init function");
        };
        assert!(program.body.statements.len() > sde.metadata.states.len());
        assert!(matches!(
            program.body.statements[0].kind,
            ExecutionStmtKind::Assign(ExecutionAssignStmt {
                target: ExecutionTarget {
                    kind: ExecutionTargetKind::StateInit(_),
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn route_property_functions_fill_defaults_for_unconfigured_routes() {
        let execution = structured_block_execution();
        let ode = find_model(&execution, "one_cmt_oral_iv");
        let lag = ode
            .function(ModelFunctionKind::RouteLag)
            .expect("lag function");
        let bio = ode
            .function(ModelFunctionKind::RouteBioavailability)
            .expect("bioavailability function");

        let FunctionBody::Statements(lag_program) = &lag.body else {
            panic!("expected statement lag function");
        };
        let FunctionBody::Statements(bio_program) = &bio.body else {
            panic!("expected statement bioavailability function");
        };

        assert_eq!(lag_program.body.statements.len(), 3);
        assert_eq!(bio_program.body.statements.len(), 3);
        assert!(matches!(
            lag_program.body.statements[1].kind,
            ExecutionStmtKind::Assign(ExecutionAssignStmt {
                value: ExecutionExpr {
                    kind: ExecutionExprKind::Literal(ConstValue::Real(value)),
                    ..
                },
                ..
            }) if value == 0.0
        ));
        assert!(matches!(
            bio_program.body.statements[1].kind,
            ExecutionStmtKind::Assign(ExecutionAssignStmt {
                value: ExecutionExpr {
                    kind: ExecutionExprKind::Literal(ConstValue::Real(value)),
                    ..
                },
                ..
            }) if value == 1.0
        ));
    }

    fn structured_block_execution() -> ExecutionModule {
        let src = STRUCTURED_BLOCK_CORPUS;
        let module = parse_module(src).expect("structured-block fixture parses");
        let analyzed = analyze_module(&module).expect("structured-block fixture analyzes");
        compile_analyzed_module(&analyzed).expect("execution compilation succeeds")
    }

    fn find_model<'a>(module: &'a ExecutionModule, name: &str) -> &'a ExecutionModel {
        module
            .models
            .iter()
            .find(|model| model.name == name)
            .unwrap_or_else(|| panic!("missing model {name}"))
    }

    fn function_kinds(model: &ExecutionModel) -> Vec<ModelFunctionKind> {
        model
            .functions
            .iter()
            .map(|function| function.kind)
            .collect()
    }
}
