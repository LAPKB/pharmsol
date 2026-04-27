use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::TypedAssignTargetKind;
use super::{
    AnalyticalKernel, ConstValue, CovariateInterpolation, Diagnostic, DiagnosticPhase,
    DiagnosticReport, ModelKind, RoutePropertyKind, Span, Symbol, SymbolId, SymbolKind,
    TypedBinaryOp, TypedCall, TypedExpr, TypedExprKind, TypedModel, TypedModule, TypedRangeExpr,
    TypedStatePlace, TypedStatementBlock, TypedStmt, TypedStmtKind, TypedUnaryOp, ValueType,
    DSL_LOWERING_GENERIC,
};

pub fn lower_typed_module(module: &TypedModule) -> Result<ExecutionModule, LoweringError> {
    let mut models = Vec::with_capacity(module.models.len());
    for model in &module.models {
        models.push(lower_typed_model(model)?);
    }
    Ok(ExecutionModule {
        models,
        span: module.span,
    })
}

pub fn lower_typed_model(model: &TypedModel) -> Result<ExecutionModel, LoweringError> {
    ExecutionLowerer::new(model)?.lower()
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
    pub abi: ExecutionAbi,
    pub kernels: Vec<ExecutionKernel>,
    pub span: Span,
}

impl ExecutionModel {
    pub fn kernel(&self, role: KernelRole) -> Option<&ExecutionKernel> {
        self.kernels.iter().find(|kernel| kernel.role == role)
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
    pub index: usize,
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
pub struct ExecutionAbi {
    pub scalar: ScalarAbi,
    pub calling_convention: CallingConvention,
    pub parameter_buffer: DenseBufferLayout,
    pub covariate_buffer: DenseBufferLayout,
    pub state_buffer: DenseBufferLayout,
    pub derived_buffer: DenseBufferLayout,
    pub output_buffer: DenseBufferLayout,
    pub route_buffer: DenseBufferLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarAbi {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallingConvention {
    DenseF64Buffers,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBufferLayout {
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
pub struct ExecutionKernel {
    pub role: KernelRole,
    pub signature: KernelSignature,
    pub implementation: KernelImplementation,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelRole {
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
pub struct KernelSignature {
    pub args: Vec<KernelArgument>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelArgument {
    pub kind: KernelArgumentKind,
    pub access: KernelAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelArgumentKind {
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
pub enum KernelAccess {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KernelImplementation {
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
        op: TypedUnaryOp,
        expr: Box<ExecutionExpr>,
    },
    Binary {
        op: TypedBinaryOp,
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
    RouteInput(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionCall {
    Math(super::MathIntrinsic),
}

#[derive(Clone, PartialEq, Eq)]
pub struct LoweringError {
    diagnostic: Box<Diagnostic>,
    source: Option<Arc<str>>,
}

impl LoweringError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            diagnostic: Box::new(Diagnostic::error(
                DSL_LOWERING_GENERIC,
                DiagnosticPhase::Lowering,
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

impl fmt::Debug for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for LoweringError {
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

impl std::error::Error for LoweringError {}

struct ExecutionLowerer<'a> {
    model: &'a TypedModel,
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

impl<'a> ExecutionLowerer<'a> {
    fn new(model: &'a TypedModel) -> Result<Self, LoweringError> {
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
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
            next_state_offset += len;
        }

        let mut route_slots = BTreeMap::new();
        let routes = model
            .routes
            .iter()
            .enumerate()
            .map(|(index, route)| {
                let symbol = lookup_symbol(&symbol_map, route.symbol, route.span)?;
                route_slots.insert(route.symbol, index);
                let destination =
                    lower_route_destination(&symbol_map, &state_slots, &route.destination)?;
                Ok(ExecutionRoute {
                    symbol: route.symbol,
                    name: symbol.name.clone(),
                    index,
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
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
            .collect::<Result<Vec<_>, LoweringError>>()?;

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
                    .map(|analytical| analytical.kernel),
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

    fn lower(self) -> Result<ExecutionModel, LoweringError> {
        let abi = self.build_abi();
        let mut kernels = Vec::new();

        if let Some(block) = &self.model.derive {
            kernels.push(self.lower_statement_kernel(KernelRole::Derive, block)?);
        }
        if let Some(block) = &self.model.init {
            kernels.push(self.lower_init_kernel(block)?);
        }
        if let Some(block) = &self.model.dynamics {
            kernels.push(self.lower_statement_kernel(KernelRole::Dynamics, block)?);
        }
        if let Some(block) = &self.model.drift {
            kernels.push(self.lower_statement_kernel(KernelRole::Drift, block)?);
        }
        if let Some(block) = &self.model.diffusion {
            kernels.push(self.lower_statement_kernel(KernelRole::Diffusion, block)?);
        }
        if let Some(kernel) =
            self.lower_route_property_kernel(RoutePropertyKind::Lag, KernelRole::RouteLag)?
        {
            kernels.push(kernel);
        }
        if let Some(kernel) = self.lower_route_property_kernel(
            RoutePropertyKind::Bioavailability,
            KernelRole::RouteBioavailability,
        )? {
            kernels.push(kernel);
        }
        if let Some(analytical) = &self.model.analytical {
            kernels.push(ExecutionKernel {
                role: KernelRole::Analytical,
                signature: signature_for(KernelRole::Analytical),
                implementation: KernelImplementation::AnalyticalBuiltin(analytical.kernel),
                span: analytical.span,
            });
        }
        kernels.push(self.lower_statement_kernel(KernelRole::Outputs, &self.model.outputs_block)?);

        Ok(ExecutionModel {
            name: self.model.name.clone(),
            kind: self.model.kind,
            metadata: self.metadata,
            abi,
            kernels,
            span: self.model.span,
        })
    }

    fn build_abi(&self) -> ExecutionAbi {
        ExecutionAbi {
            scalar: ScalarAbi::F64,
            calling_convention: CallingConvention::DenseF64Buffers,
            parameter_buffer: DenseBufferLayout {
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
            covariate_buffer: DenseBufferLayout {
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
            state_buffer: DenseBufferLayout {
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
            derived_buffer: DenseBufferLayout {
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
            output_buffer: DenseBufferLayout {
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
            route_buffer: DenseBufferLayout {
                kind: BufferKind::Routes,
                len: self.metadata.routes.len(),
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

    fn lower_statement_kernel(
        &self,
        role: KernelRole,
        block: &TypedStatementBlock,
    ) -> Result<ExecutionKernel, LoweringError> {
        let mut locals = LocalLowering::default();
        let statements = block
            .statements
            .iter()
            .map(|stmt| self.lower_stmt(stmt, &mut locals))
            .collect::<Result<Vec<_>, LoweringError>>()?;

        Ok(ExecutionKernel {
            role,
            signature: signature_for(role),
            implementation: KernelImplementation::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: block.span,
                },
            }),
            span: block.span,
        })
    }

    fn lower_init_kernel(
        &self,
        block: &TypedStatementBlock,
    ) -> Result<ExecutionKernel, LoweringError> {
        let mut locals = LocalLowering::default();
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
                .map(|stmt| self.lower_stmt(stmt, &mut locals))
                .collect::<Result<Vec<_>, LoweringError>>()?,
        );

        Ok(ExecutionKernel {
            role: KernelRole::Init,
            signature: signature_for(KernelRole::Init),
            implementation: KernelImplementation::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: block.span,
                },
            }),
            span: block.span,
        })
    }

    fn lower_route_property_kernel(
        &self,
        property_kind: RoutePropertyKind,
        role: KernelRole,
    ) -> Result<Option<ExecutionKernel>, LoweringError> {
        if !self.model.routes.iter().any(|route| {
            route
                .properties
                .iter()
                .any(|property| property.kind == property_kind)
        }) {
            return Ok(None);
        }

        let mut statements = Vec::with_capacity(self.model.routes.len());
        let mut locals = LocalLowering::default();
        for route in &self.model.routes {
            let route_name = self.symbol_name(route.symbol)?.to_string();
            let route_index = *self.route_slots.get(&route.symbol).ok_or_else(|| {
                LoweringError::new(
                    format!("route `{}` has no execution slot", route_name),
                    route.span,
                )
            })?;
            let expression = match route
                .properties
                .iter()
                .find(|property| property.kind == property_kind)
            {
                Some(property) => self.lower_expr(&property.value, &mut locals)?,
                None if property_kind == RoutePropertyKind::Lag => literal_real(0.0, route.span),
                None => literal_real(1.0, route.span),
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

        Ok(Some(ExecutionKernel {
            role,
            signature: signature_for(role),
            implementation: KernelImplementation::Statements(ExecutionProgram {
                locals: locals.locals,
                body: ExecutionBlock {
                    statements,
                    span: self.model.span,
                },
            }),
            span: self.model.span,
        }))
    }

    fn lower_stmt(
        &self,
        stmt: &TypedStmt,
        locals: &mut LocalLowering,
    ) -> Result<ExecutionStmt, LoweringError> {
        let kind = match &stmt.kind {
            TypedStmtKind::Let(let_stmt) => {
                let local = locals.local_slot(let_stmt.symbol, self)?;
                ExecutionStmtKind::Let(ExecutionLetStmt {
                    local,
                    value: self.lower_expr(&let_stmt.value, locals)?,
                })
            }
            TypedStmtKind::Assign(assign) => ExecutionStmtKind::Assign(ExecutionAssignStmt {
                target: self.lower_target(&assign.target.kind, assign.target.span, locals)?,
                value: self.lower_expr(&assign.value, locals)?,
            }),
            TypedStmtKind::If(if_stmt) => ExecutionStmtKind::If(ExecutionIfStmt {
                condition: self.lower_expr(&if_stmt.condition, locals)?,
                then_branch: if_stmt
                    .then_branch
                    .iter()
                    .map(|stmt| self.lower_stmt(stmt, locals))
                    .collect::<Result<Vec<_>, _>>()?,
                else_branch: if_stmt
                    .else_branch
                    .as_ref()
                    .map(|branch| {
                        branch
                            .iter()
                            .map(|stmt| self.lower_stmt(stmt, locals))
                            .collect::<Result<Vec<_>, LoweringError>>()
                    })
                    .transpose()?,
            }),
            TypedStmtKind::For(for_stmt) => {
                let local = locals.local_slot(for_stmt.binding, self)?;
                ExecutionStmtKind::For(ExecutionForStmt {
                    local,
                    range: self.lower_range(&for_stmt.range, locals)?,
                    body: for_stmt
                        .body
                        .iter()
                        .map(|stmt| self.lower_stmt(stmt, locals))
                        .collect::<Result<Vec<_>, _>>()?,
                })
            }
        };

        Ok(ExecutionStmt {
            kind,
            span: stmt.span,
        })
    }

    fn lower_range(
        &self,
        range: &TypedRangeExpr,
        locals: &mut LocalLowering,
    ) -> Result<ExecutionRange, LoweringError> {
        Ok(ExecutionRange {
            start: self.lower_expr(&range.start, locals)?,
            end: self.lower_expr(&range.end, locals)?,
            span: range.span,
        })
    }

    fn lower_target(
        &self,
        target: &TypedAssignTargetKind,
        span: Span,
        locals: &mut LocalLowering,
    ) -> Result<ExecutionTarget, LoweringError> {
        let kind = match target {
            TypedAssignTargetKind::Derived(symbol) => {
                ExecutionTargetKind::Derived(self.slot_for_derived(*symbol, span)?)
            }
            TypedAssignTargetKind::Output(symbol) => {
                ExecutionTargetKind::Output(self.slot_for_output(*symbol, span)?)
            }
            TypedAssignTargetKind::StateInit(place) => {
                ExecutionTargetKind::StateInit(self.lower_state_ref(place, locals)?)
            }
            TypedAssignTargetKind::Derivative(place) => {
                ExecutionTargetKind::StateDerivative(self.lower_state_ref(place, locals)?)
            }
            TypedAssignTargetKind::Noise(place) => {
                ExecutionTargetKind::StateNoise(self.lower_state_ref(place, locals)?)
            }
        };
        Ok(ExecutionTarget { kind, span })
    }

    fn lower_expr(
        &self,
        expr: &TypedExpr,
        locals: &mut LocalLowering,
    ) -> Result<ExecutionExpr, LoweringError> {
        if let Some(constant) = &expr.constant {
            return Ok(ExecutionExpr {
                kind: ExecutionExprKind::Literal(constant.clone()),
                ty: expr.ty,
                constant: Some(constant.clone()),
                span: expr.span,
            });
        }

        let kind = match &expr.kind {
            TypedExprKind::Literal(constant) => ExecutionExprKind::Literal(constant.clone()),
            TypedExprKind::Symbol(symbol) => {
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
                        return Err(LoweringError::new(
                            format!(
                                "constant `{}` should have been folded before execution lowering",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                    SymbolKind::State => {
                        return Err(LoweringError::new(
                            format!(
                                "state `{}` should lower through a state reference",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                    SymbolKind::Route => {
                        return Err(LoweringError::new(
                            format!(
                                "route `{}` is not a scalar execution input",
                                symbol_info.name
                            ),
                            expr.span,
                        )
                        .with_note("routes must lower through `rate(route)` or route metadata"));
                    }
                    SymbolKind::Output => {
                        return Err(LoweringError::new(
                            format!(
                                "output `{}` cannot be read inside execution kernels",
                                symbol_info.name
                            ),
                            expr.span,
                        ));
                    }
                }
            }
            TypedExprKind::StateValue(place) => {
                ExecutionExprKind::Load(ExecutionLoad::State(self.lower_state_ref(place, locals)?))
            }
            TypedExprKind::Unary { op, expr } => ExecutionExprKind::Unary {
                op: *op,
                expr: Box::new(self.lower_expr(expr, locals)?),
            },
            TypedExprKind::Binary { op, lhs, rhs } => ExecutionExprKind::Binary {
                op: *op,
                lhs: Box::new(self.lower_expr(lhs, locals)?),
                rhs: Box::new(self.lower_expr(rhs, locals)?),
            },
            TypedExprKind::Call { callee, args } => match callee {
                TypedCall::Math(intrinsic) => ExecutionExprKind::Call {
                    callee: ExecutionCall::Math(*intrinsic),
                    args: args
                        .iter()
                        .map(|arg| self.lower_expr(arg, locals))
                        .collect::<Result<Vec<_>, _>>()?,
                },
                TypedCall::Rate(route) => {
                    let route_name = self.symbol_name(*route)?.to_string();
                    let route_index = *self.route_slots.get(route).ok_or_else(|| {
                        LoweringError::new(
                            format!("route `{}` has no execution slot", route_name),
                            expr.span,
                        )
                    })?;
                    ExecutionExprKind::Load(ExecutionLoad::RouteInput(route_index))
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

    fn lower_state_ref(
        &self,
        place: &TypedStatePlace,
        locals: &mut LocalLowering,
    ) -> Result<ExecutionStateRef, LoweringError> {
        let state_name = self.symbol_name(place.state)?.to_string();
        let layout = self.state_slots.get(&place.state).copied().ok_or_else(|| {
            LoweringError::new(
                format!("state `{}` has no execution layout", state_name),
                place.span,
            )
        })?;
        let index = place
            .index
            .as_ref()
            .map(|index| self.lower_expr(index, locals))
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

    fn slot_for_parameter(&self, symbol: SymbolId, span: Span) -> Result<usize, LoweringError> {
        self.parameter_slots.get(&symbol).copied().ok_or_else(|| {
            LoweringError::new(
                format!(
                    "parameter `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_covariate(&self, symbol: SymbolId, span: Span) -> Result<usize, LoweringError> {
        self.covariate_slots.get(&symbol).copied().ok_or_else(|| {
            LoweringError::new(
                format!(
                    "covariate `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_derived(&self, symbol: SymbolId, span: Span) -> Result<usize, LoweringError> {
        self.derived_slots.get(&symbol).copied().ok_or_else(|| {
            LoweringError::new(
                format!(
                    "derived value `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn slot_for_output(&self, symbol: SymbolId, span: Span) -> Result<usize, LoweringError> {
        self.output_slots.get(&symbol).copied().ok_or_else(|| {
            LoweringError::new(
                format!(
                    "output `{}` has no ABI slot",
                    self.symbol_name(symbol).unwrap_or("<unknown>")
                ),
                span,
            )
        })
    }

    fn symbol_name(&self, symbol: SymbolId) -> Result<&str, LoweringError> {
        Ok(&lookup_symbol(&self.symbol_map, symbol, self.model.span)?.name)
    }
}

#[derive(Default)]
struct LocalLowering {
    locals: Vec<ExecutionLocal>,
    slots: BTreeMap<SymbolId, usize>,
}

impl LocalLowering {
    fn local_slot(
        &mut self,
        symbol: SymbolId,
        lowerer: &ExecutionLowerer<'_>,
    ) -> Result<usize, LoweringError> {
        if let Some(slot) = self.slots.get(&symbol).copied() {
            return Ok(slot);
        }
        let symbol_info = lookup_symbol(&lowerer.symbol_map, symbol, lowerer.model.span)?;
        let ty = match symbol_info.ty {
            super::SymbolType::Scalar(ty) => ty,
            super::SymbolType::Array { .. } => {
                return Err(LoweringError::new(
                    format!("local `{}` must be scalar", symbol_info.name),
                    symbol_info.span,
                ));
            }
            super::SymbolType::Route => {
                return Err(LoweringError::new(
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

fn signature_for(role: KernelRole) -> KernelSignature {
    let args = match role {
        KernelRole::Derive => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Output),
        ],
        KernelRole::Dynamics => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::StateDerivatives, KernelAccess::Output),
        ],
        KernelRole::Outputs => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::Outputs, KernelAccess::Output),
        ],
        KernelRole::Init => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::InitialState, KernelAccess::Output),
        ],
        KernelRole::Drift => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::StateDerivatives, KernelAccess::Output),
        ],
        KernelRole::Diffusion => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::StateNoise, KernelAccess::Output),
        ],
        KernelRole::RouteLag => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::RouteLag, KernelAccess::Output),
        ],
        KernelRole::RouteBioavailability => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(
                KernelArgumentKind::RouteBioavailability,
                KernelAccess::Output,
            ),
        ],
        KernelRole::Analytical => vec![
            arg(KernelArgumentKind::Time, KernelAccess::Input),
            arg(KernelArgumentKind::States, KernelAccess::Input),
            arg(KernelArgumentKind::Parameters, KernelAccess::Input),
            arg(KernelArgumentKind::Covariates, KernelAccess::Input),
            arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
            arg(KernelArgumentKind::Derived, KernelAccess::Input),
            arg(KernelArgumentKind::AnalyticalState, KernelAccess::Output),
        ],
    };
    KernelSignature { args }
}

fn arg(kind: KernelArgumentKind, access: KernelAccess) -> KernelArgument {
    KernelArgument { kind, access }
}

fn lookup_symbol<'a>(
    symbols: &'a BTreeMap<SymbolId, &'a Symbol>,
    symbol: SymbolId,
    span: Span,
) -> Result<&'a Symbol, LoweringError> {
    symbols.get(&symbol).copied().ok_or_else(|| {
        LoweringError::new(
            format!("symbol id {symbol} is missing from the typed model symbol table"),
            span,
        )
    })
}

fn lower_route_destination(
    symbols: &BTreeMap<SymbolId, &Symbol>,
    state_slots: &BTreeMap<SymbolId, StateLayout>,
    destination: &TypedStatePlace,
) -> Result<RouteDestination, LoweringError> {
    let symbol = lookup_symbol(symbols, destination.state, destination.span)?;
    let layout = state_slots
        .get(&destination.state)
        .copied()
        .ok_or_else(|| {
            LoweringError::new(
                format!("state `{}` has no execution layout", symbol.name),
                destination.span,
            )
        })?;
    let element = match &destination.index {
        None => 0,
        Some(index) => constant_index(index, destination.span)?,
    };
    if element >= layout.len {
        return Err(LoweringError::new(
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

fn constant_index(expr: &TypedExpr, span: Span) -> Result<usize, LoweringError> {
    let value = expr
        .constant
        .as_ref()
        .and_then(ConstValue::as_i64)
        .ok_or_else(|| LoweringError::new("expected a compile-time integer index", span))?;
    if value < 0 {
        return Err(LoweringError::new(
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
    use crate::dsl::{analyze_module, parse_module};

    #[test]
    fn lowers_proposal_two_corpus_into_execution_models() {
        let execution = proposal_execution();
        assert_eq!(execution.models.len(), 4);

        let ode = find_model(&execution, "one_cmt_oral_iv");
        assert_eq!(ode.abi.parameter_buffer.len, 5);
        assert_eq!(ode.abi.covariate_buffer.len, 1);
        assert_eq!(ode.abi.state_buffer.len, 2);
        assert_eq!(ode.abi.derived_buffer.len, 3);
        assert_eq!(ode.abi.output_buffer.len, 1);
        assert_eq!(ode.abi.route_buffer.len, 2);
        assert_eq!(ode.metadata.routes[0].destination.state_offset, 0);
        assert_eq!(
            kernel_roles(ode),
            vec![
                KernelRole::Derive,
                KernelRole::Dynamics,
                KernelRole::RouteLag,
                KernelRole::RouteBioavailability,
                KernelRole::Outputs,
            ]
        );
    }

    #[test]
    fn flattens_array_states_and_preserves_loop_structure() {
        let execution = proposal_execution();
        let transit = find_model(&execution, "transit_absorption");
        assert_eq!(transit.abi.state_buffer.len, 5);
        assert_eq!(transit.metadata.states[0].name, "transit");
        assert_eq!(transit.metadata.states[0].offset, 0);
        assert_eq!(transit.metadata.states[0].len, 4);
        assert_eq!(transit.metadata.states[1].name, "central");
        assert_eq!(transit.metadata.states[1].offset, 4);
        assert!(transit.kernel(KernelRole::RouteLag).is_none());
        assert!(transit.kernel(KernelRole::RouteBioavailability).is_none());

        let dynamics = transit
            .kernel(KernelRole::Dynamics)
            .expect("dynamics kernel");
        let KernelImplementation::Statements(program) = &dynamics.implementation else {
            panic!("expected statement-based dynamics kernel");
        };
        assert!(program
            .body
            .statements
            .iter()
            .any(|stmt| matches!(stmt.kind, ExecutionStmtKind::For(_))));
    }

    #[test]
    fn analytical_models_lower_to_builtin_execution_kernels() {
        let execution = proposal_execution();
        let analytical = find_model(&execution, "one_cmt_abs");
        let kernel = analytical
            .kernel(KernelRole::Analytical)
            .expect("analytical kernel");
        assert_eq!(
            kernel.signature.args,
            vec![
                arg(KernelArgumentKind::Time, KernelAccess::Input),
                arg(KernelArgumentKind::States, KernelAccess::Input),
                arg(KernelArgumentKind::Parameters, KernelAccess::Input),
                arg(KernelArgumentKind::Covariates, KernelAccess::Input),
                arg(KernelArgumentKind::RouteInputs, KernelAccess::Input),
                arg(KernelArgumentKind::Derived, KernelAccess::Input),
                arg(KernelArgumentKind::AnalyticalState, KernelAccess::Output),
            ]
        );
        assert!(matches!(
            kernel.implementation,
            KernelImplementation::AnalyticalBuiltin(AnalyticalKernel::OneCompartmentWithAbsorption)
        ));
    }

    #[test]
    fn sde_models_emit_runtime_kernels_and_zero_filled_init() {
        let execution = proposal_execution();
        let sde = find_model(&execution, "vanco_sde");
        assert_eq!(sde.metadata.particles, Some(1000));
        assert_eq!(
            kernel_roles(sde),
            vec![
                KernelRole::Init,
                KernelRole::Drift,
                KernelRole::Diffusion,
                KernelRole::Outputs,
            ]
        );

        let init = sde.kernel(KernelRole::Init).expect("init kernel");
        let KernelImplementation::Statements(program) = &init.implementation else {
            panic!("expected statement init kernel");
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
    fn route_property_kernels_fill_defaults_for_unconfigured_routes() {
        let execution = proposal_execution();
        let ode = find_model(&execution, "one_cmt_oral_iv");
        let lag = ode.kernel(KernelRole::RouteLag).expect("lag kernel");
        let bio = ode
            .kernel(KernelRole::RouteBioavailability)
            .expect("bioavailability kernel");

        let KernelImplementation::Statements(lag_program) = &lag.implementation else {
            panic!("expected statement lag kernel");
        };
        let KernelImplementation::Statements(bio_program) = &bio.implementation else {
            panic!("expected statement bioavailability kernel");
        };

        assert_eq!(lag_program.body.statements.len(), 2);
        assert_eq!(bio_program.body.statements.len(), 2);
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

    fn proposal_execution() -> ExecutionModule {
        let src = include_str!("../../tests/fixtures/dsl/02-structured-block-imperative.dsl");
        let module = parse_module(src).expect("proposal parses");
        let typed = analyze_module(&module).expect("proposal analyzes");
        lower_typed_module(&typed).expect("execution lowering succeeds")
    }

    fn find_model<'a>(module: &'a ExecutionModule, name: &str) -> &'a ExecutionModel {
        module
            .models
            .iter()
            .find(|model| model.name == name)
            .unwrap_or_else(|| panic!("missing model {name}"))
    }

    fn kernel_roles(model: &ExecutionModel) -> Vec<KernelRole> {
        model.kernels.iter().map(|kernel| kernel.role).collect()
    }
}
