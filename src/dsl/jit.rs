use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;

use cranelift::codegen::settings;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

pub use super::native::CompiledModelFunction;
use super::native::{
    CompiledNativeModel, NativeAnalyticalModel, NativeExecutionArtifact, NativeModelInfo,
    NativeOdeModel, NativeSdeModel,
};
use pharmsol_dsl::execution::{
    ExecutionBlock, ExecutionCall, ExecutionExpr, ExecutionExprKind, ExecutionForStmt,
    ExecutionIfStmt, ExecutionLoad, ExecutionModel, ExecutionProgram, ExecutionStateRef,
    ExecutionStmt, ExecutionStmtKind, ExecutionTarget, ExecutionTargetKind, FunctionBody,
    ModelFunction, ModelFunctionKind,
};
use pharmsol_dsl::{
    AnalyzedBinaryOp, AnalyzedUnaryOp, ConstValue, Diagnostic, DiagnosticPhase, DiagnosticReport,
    MathFunction, ModelKind, Span, ValueType, DSL_BACKEND_GENERIC,
};

mod externs {
    pub extern "C" fn exp_(x: f64) -> f64 {
        x.exp()
    }

    pub extern "C" fn ln_(x: f64) -> f64 {
        x.ln()
    }

    pub extern "C" fn log10_(x: f64) -> f64 {
        x.log10()
    }

    pub extern "C" fn log2_(x: f64) -> f64 {
        x.log2()
    }

    pub extern "C" fn sqrt_(x: f64) -> f64 {
        x.sqrt()
    }

    pub extern "C" fn abs_(x: f64) -> f64 {
        x.abs()
    }

    pub extern "C" fn ceil_(x: f64) -> f64 {
        x.ceil()
    }

    pub extern "C" fn floor_(x: f64) -> f64 {
        x.floor()
    }

    pub extern "C" fn round_(x: f64) -> f64 {
        x.round()
    }

    pub extern "C" fn sin_(x: f64) -> f64 {
        x.sin()
    }

    pub extern "C" fn cos_(x: f64) -> f64 {
        x.cos()
    }

    pub extern "C" fn tan_(x: f64) -> f64 {
        x.tan()
    }

    pub extern "C" fn pow_(a: f64, b: f64) -> f64 {
        a.powf(b)
    }
}

pub type JitExecutionArtifact = NativeExecutionArtifact;
pub type JitOdeModel = NativeOdeModel;
pub type JitAnalyticalModel = NativeAnalyticalModel;
pub type JitSdeModel = NativeSdeModel;
pub type CompiledJitModel = CompiledNativeModel;

/// Error reported while lowering an execution model into native in-process JIT
/// code.
///
/// The error retains the backend diagnostic so callers can render the message
/// against the original DSL source when available.
#[derive(Clone, PartialEq, Eq)]
pub struct JitCompileError {
    diagnostic: Box<Diagnostic>,
    source: Option<Arc<str>>,
}

impl JitCompileError {
    fn new(message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            diagnostic: Box::new(Diagnostic::error(
                DSL_BACKEND_GENERIC,
                DiagnosticPhase::Backend,
                message,
                span.unwrap_or_default(),
            )),
            source: None,
        }
    }

    pub fn diagnostic(&self) -> &Diagnostic {
        self.diagnostic.as_ref()
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

impl std::fmt::Display for JitCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Debug for JitCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for JitCompileError {}

#[derive(Clone, Copy)]
struct ExternIds {
    exp: cranelift_module::FuncId,
    ln: cranelift_module::FuncId,
    log10: cranelift_module::FuncId,
    log2: cranelift_module::FuncId,
    sqrt: cranelift_module::FuncId,
    abs: cranelift_module::FuncId,
    ceil: cranelift_module::FuncId,
    floor: cranelift_module::FuncId,
    round: cranelift_module::FuncId,
    sin: cranelift_module::FuncId,
    cos: cranelift_module::FuncId,
    tan: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
}

#[derive(Clone, Copy)]
struct ExternRefs {
    exp: codegen::ir::FuncRef,
    ln: codegen::ir::FuncRef,
    log10: codegen::ir::FuncRef,
    log2: codegen::ir::FuncRef,
    sqrt: codegen::ir::FuncRef,
    abs: codegen::ir::FuncRef,
    ceil: codegen::ir::FuncRef,
    floor: codegen::ir::FuncRef,
    round: codegen::ir::FuncRef,
    sin: codegen::ir::FuncRef,
    cos: codegen::ir::FuncRef,
    tan: codegen::ir::FuncRef,
    pow: codegen::ir::FuncRef,
}

#[derive(Clone, Copy)]
struct FunctionArgs {
    _time: Value,
    states: Value,
    params: Value,
    covariates: Value,
    routes: Value,
    derived: Value,
    out: Value,
}

#[derive(Clone, Copy)]
struct LocalBinding {
    variable: Variable,
    ty: ValueType,
}

struct EmitEnv<'a> {
    _ptr_ty: Type,
    args: FunctionArgs,
    externs: ExternRefs,
    locals: &'a BTreeMap<usize, LocalBinding>,
}

#[derive(Clone, Copy)]
struct LoweredValue {
    value: Value,
    ty: ValueType,
}

/// Compile one compiled execution model into a reusable JIT function artifact.
///
/// This builds the raw Cranelift-compiled function bundle for all roles present in
/// the model. Most callers should use [`compile_execution_model_to_jit`] instead.
pub fn compile_execution_artifact(
    model: &ExecutionModel,
) -> Result<NativeExecutionArtifact, JitCompileError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "false")
        .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))?;

    let isa_builder = cranelift_native::builder()
        .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol("pm_exp", externs::exp_ as *const u8);
    builder.symbol("pm_ln", externs::ln_ as *const u8);
    builder.symbol("pm_log10", externs::log10_ as *const u8);
    builder.symbol("pm_log2", externs::log2_ as *const u8);
    builder.symbol("pm_sqrt", externs::sqrt_ as *const u8);
    builder.symbol("pm_abs", externs::abs_ as *const u8);
    builder.symbol("pm_ceil", externs::ceil_ as *const u8);
    builder.symbol("pm_floor", externs::floor_ as *const u8);
    builder.symbol("pm_round", externs::round_ as *const u8);
    builder.symbol("pm_sin", externs::sin_ as *const u8);
    builder.symbol("pm_cos", externs::cos_ as *const u8);
    builder.symbol("pm_tan", externs::tan_ as *const u8);
    builder.symbol("pm_pow", externs::pow_ as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_ty = module.target_config().pointer_type();
    let externs = declare_externs(&mut module, model.span)?;
    let mut ctx = module.make_context();
    let mut builder_context = FunctionBuilderContext::new();

    let derive = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Derive,
    )?;
    let dynamics = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Dynamics,
    )?;
    let outputs = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Outputs,
    )?
    .ok_or_else(|| JitCompileError::new("missing outputs function", Some(model.span)))?;
    let init = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Init,
    )?;
    let drift = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Drift,
    )?;
    let diffusion = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::Diffusion,
    )?;
    let route_lag = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::RouteLag,
    )?;
    let route_bioavailability = compile_role_function(
        &mut module,
        &mut ctx,
        &mut builder_context,
        ptr_ty,
        externs,
        model,
        ModelFunctionKind::RouteBioavailability,
    )?;

    module
        .finalize_definitions()
        .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))?;

    Ok(NativeExecutionArtifact::from_jit_module(
        model.name.clone(),
        derive.map(|id| function_pointer(&mut module, id)),
        dynamics.map(|id| function_pointer(&mut module, id)),
        function_pointer(&mut module, outputs),
        init.map(|id| function_pointer(&mut module, id)),
        drift.map(|id| function_pointer(&mut module, id)),
        diffusion.map(|id| function_pointer(&mut module, id)),
        route_lag.map(|id| function_pointer(&mut module, id)),
        route_bioavailability.map(|id| function_pointer(&mut module, id)),
        module,
    ))
}

fn declare_externs(module: &mut JITModule, span: Span) -> Result<ExternIds, JitCompileError> {
    let declare_unary = |module: &mut JITModule, symbol: &str| -> Result<_, JitCompileError> {
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(types::F64));
        signature.returns.push(AbiParam::new(types::F64));
        module
            .declare_function(symbol, Linkage::Import, &signature)
            .map_err(|error| JitCompileError::new(error.to_string(), Some(span)))
    };
    let declare_binary = |module: &mut JITModule, symbol: &str| -> Result<_, JitCompileError> {
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(types::F64));
        signature.params.push(AbiParam::new(types::F64));
        signature.returns.push(AbiParam::new(types::F64));
        module
            .declare_function(symbol, Linkage::Import, &signature)
            .map_err(|error| JitCompileError::new(error.to_string(), Some(span)))
    };

    Ok(ExternIds {
        exp: declare_unary(module, "pm_exp")?,
        ln: declare_unary(module, "pm_ln")?,
        log10: declare_unary(module, "pm_log10")?,
        log2: declare_unary(module, "pm_log2")?,
        sqrt: declare_unary(module, "pm_sqrt")?,
        abs: declare_unary(module, "pm_abs")?,
        ceil: declare_unary(module, "pm_ceil")?,
        floor: declare_unary(module, "pm_floor")?,
        round: declare_unary(module, "pm_round")?,
        sin: declare_unary(module, "pm_sin")?,
        cos: declare_unary(module, "pm_cos")?,
        tan: declare_unary(module, "pm_tan")?,
        pow: declare_binary(module, "pm_pow")?,
    })
}

fn compile_role_function(
    module: &mut JITModule,
    ctx: &mut cranelift::codegen::Context,
    builder_context: &mut FunctionBuilderContext,
    ptr_ty: Type,
    externs: ExternIds,
    model: &ExecutionModel,
    role: ModelFunctionKind,
) -> Result<Option<cranelift_module::FuncId>, JitCompileError> {
    let Some(function) = model.function(role) else {
        return Ok(None);
    };
    let FunctionBody::Statements(program) = &function.body else {
        return Ok(None);
    };

    let function_name = format!("{}_{}", model.name, function_kind_name(role));
    let function_id = emit_statement_function(
        module,
        ctx,
        builder_context,
        ptr_ty,
        externs,
        &function_name,
        function,
        program,
    )?;
    Ok(Some(function_id))
}

#[allow(clippy::too_many_arguments)]
fn emit_statement_function(
    module: &mut JITModule,
    ctx: &mut cranelift::codegen::Context,
    builder_context: &mut FunctionBuilderContext,
    ptr_ty: Type,
    externs: ExternIds,
    function_name: &str,
    function: &ModelFunction,
    program: &ExecutionProgram,
) -> Result<cranelift_module::FuncId, JitCompileError> {
    ctx.func.signature = dense_function_signature(module);

    let function_id = module
        .declare_function(function_name, Linkage::Local, &ctx.func.signature)
        .map_err(|error| JitCompileError::new(error.to_string(), Some(function.span)))?;

    let mut builder = FunctionBuilder::new(&mut ctx.func, builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let params = builder.block_params(entry);
    let args = FunctionArgs {
        _time: params[0],
        states: params[1],
        params: params[2],
        covariates: params[3],
        routes: params[4],
        derived: params[5],
        out: params[6],
    };

    let externs = ExternRefs {
        exp: module.declare_func_in_func(externs.exp, builder.func),
        ln: module.declare_func_in_func(externs.ln, builder.func),
        log10: module.declare_func_in_func(externs.log10, builder.func),
        log2: module.declare_func_in_func(externs.log2, builder.func),
        sqrt: module.declare_func_in_func(externs.sqrt, builder.func),
        abs: module.declare_func_in_func(externs.abs, builder.func),
        ceil: module.declare_func_in_func(externs.ceil, builder.func),
        floor: module.declare_func_in_func(externs.floor, builder.func),
        round: module.declare_func_in_func(externs.round, builder.func),
        sin: module.declare_func_in_func(externs.sin, builder.func),
        cos: module.declare_func_in_func(externs.cos, builder.func),
        tan: module.declare_func_in_func(externs.tan, builder.func),
        pow: module.declare_func_in_func(externs.pow, builder.func),
    };

    let mut locals = BTreeMap::new();
    for local in &program.locals {
        let variable = builder.declare_var(clif_type(local.ty));
        let initial = zero_value(&mut builder, local.ty);
        builder.def_var(variable, initial);
        locals.insert(
            local.index,
            LocalBinding {
                variable,
                ty: local.ty,
            },
        );
    }

    let env = EmitEnv {
        _ptr_ty: ptr_ty,
        args,
        externs,
        locals: &locals,
    };
    emit_block(&mut builder, &env, &program.body)?;
    builder.ins().return_(&[]);
    builder.finalize();

    module
        .define_function(function_id, ctx)
        .map_err(|error| JitCompileError::new(error.to_string(), Some(function.span)))?;
    module.clear_context(ctx);
    Ok(function_id)
}

fn dense_function_signature(module: &mut JITModule) -> cranelift::codegen::ir::Signature {
    let mut signature = module.make_signature();
    let ptr_ty = module.target_config().pointer_type();
    signature.params.push(AbiParam::new(types::F64));
    for _ in 0..6 {
        signature.params.push(AbiParam::new(ptr_ty));
    }
    signature
}

fn function_pointer(
    module: &mut JITModule,
    function_id: cranelift_module::FuncId,
) -> CompiledModelFunction {
    unsafe { mem::transmute(module.get_finalized_function(function_id)) }
}

fn function_kind_name(role: ModelFunctionKind) -> &'static str {
    match role {
        ModelFunctionKind::Derive => "derive",
        ModelFunctionKind::Dynamics => "dynamics",
        ModelFunctionKind::Outputs => "outputs",
        ModelFunctionKind::Init => "init",
        ModelFunctionKind::Drift => "drift",
        ModelFunctionKind::Diffusion => "diffusion",
        ModelFunctionKind::RouteLag => "route_lag",
        ModelFunctionKind::RouteBioavailability => "route_bioavailability",
        ModelFunctionKind::Analytical => "analytical",
    }
}

fn emit_block(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    block: &ExecutionBlock,
) -> Result<(), JitCompileError> {
    for statement in &block.statements {
        emit_stmt(builder, env, statement)?;
    }
    Ok(())
}

fn emit_stmt(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    statement: &ExecutionStmt,
) -> Result<(), JitCompileError> {
    match &statement.kind {
        ExecutionStmtKind::Let(let_stmt) => {
            let value = lower_expr(builder, env, &let_stmt.value)?;
            let binding = env.locals.get(&let_stmt.local).ok_or_else(|| {
                JitCompileError::new(
                    format!("unknown local slot {}", let_stmt.local),
                    Some(statement.span),
                )
            })?;
            let coerced = cast_value(builder, value, binding.ty, statement.span)?;
            builder.def_var(binding.variable, coerced.value);
            Ok(())
        }
        ExecutionStmtKind::Assign(assign_stmt) => {
            let value = lower_expr(builder, env, &assign_stmt.value)?;
            store_target(builder, env, &assign_stmt.target, value, statement.span)
        }
        ExecutionStmtKind::If(if_stmt) => emit_if(builder, env, if_stmt, statement.span),
        ExecutionStmtKind::For(for_stmt) => emit_for(builder, env, for_stmt, statement.span),
    }
}

fn emit_if(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    if_stmt: &ExecutionIfStmt,
    span: Span,
) -> Result<(), JitCompileError> {
    let condition = lower_expr(builder, env, &if_stmt.condition)?;
    let cond_value = as_bool(builder, condition, span)?;
    let then_block = builder.create_block();
    let else_block = builder.create_block();
    let merge_block = builder.create_block();

    builder
        .ins()
        .brif(cond_value, then_block, &[], else_block, &[]);

    builder.switch_to_block(then_block);
    emit_stmt_list(builder, env, &if_stmt.then_branch)?;
    builder.ins().jump(merge_block, &[]);
    builder.seal_block(then_block);

    builder.switch_to_block(else_block);
    if let Some(else_branch) = &if_stmt.else_branch {
        emit_stmt_list(builder, env, else_branch)?;
    }
    builder.ins().jump(merge_block, &[]);
    builder.seal_block(else_block);

    builder.switch_to_block(merge_block);
    builder.seal_block(merge_block);
    Ok(())
}

fn emit_for(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    for_stmt: &ExecutionForStmt,
    span: Span,
) -> Result<(), JitCompileError> {
    let binding = env.locals.get(&for_stmt.local).ok_or_else(|| {
        JitCompileError::new(
            format!("unknown loop local slot {}", for_stmt.local),
            Some(span),
        )
    })?;
    let start_expr = lower_expr(builder, env, &for_stmt.range.start)?;
    let start = cast_value(builder, start_expr, ValueType::Int, span)?;
    let end_expr = lower_expr(builder, env, &for_stmt.range.end)?;
    let end = cast_value(builder, end_expr, ValueType::Int, span)?;

    builder.def_var(binding.variable, start.value);

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    builder.ins().jump(loop_header, &[]);
    builder.switch_to_block(loop_header);

    let current = builder.use_var(binding.variable);
    let keep_going = builder
        .ins()
        .icmp(IntCC::SignedLessThan, current, end.value);
    builder
        .ins()
        .brif(keep_going, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);
    emit_stmt_list(builder, env, &for_stmt.body)?;
    let current = builder.use_var(binding.variable);
    let next = builder.ins().iadd_imm(current, 1);
    builder.def_var(binding.variable, next);
    builder.ins().jump(loop_header, &[]);
    builder.seal_block(loop_body);
    builder.seal_block(loop_header);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);
    Ok(())
}

fn emit_stmt_list(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    statements: &[ExecutionStmt],
) -> Result<(), JitCompileError> {
    for statement in statements {
        emit_stmt(builder, env, statement)?;
    }
    Ok(())
}

fn store_target(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    target: &ExecutionTarget,
    value: LoweredValue,
    span: Span,
) -> Result<(), JitCompileError> {
    let value = cast_value(builder, value, ValueType::Real, span)?;
    match &target.kind {
        ExecutionTargetKind::Derived(index)
        | ExecutionTargetKind::Output(index)
        | ExecutionTargetKind::RouteLag(index)
        | ExecutionTargetKind::RouteBioavailability(index) => {
            store_fixed(builder, env.args.out, *index, value.value);
            Ok(())
        }
        ExecutionTargetKind::StateInit(state_ref)
        | ExecutionTargetKind::StateDerivative(state_ref)
        | ExecutionTargetKind::StateNoise(state_ref) => {
            store_state_ref(builder, env, env.args.out, state_ref, value.value)
        }
    }
}

fn lower_expr(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    expr: &ExecutionExpr,
) -> Result<LoweredValue, JitCompileError> {
    match &expr.kind {
        ExecutionExprKind::Literal(value) => Ok(lower_literal(builder, value)),
        ExecutionExprKind::Load(load) => lower_load(builder, env, load, expr.ty, expr.span),
        ExecutionExprKind::Unary { op, expr: inner } => {
            let value = lower_expr(builder, env, inner)?;
            lower_unary(builder, env, *op, value, expr.ty, expr.span)
        }
        ExecutionExprKind::Binary { op, lhs, rhs } => {
            let lhs = lower_expr(builder, env, lhs)?;
            let rhs = lower_expr(builder, env, rhs)?;
            lower_binary(builder, env, *op, lhs, rhs, expr.ty, expr.span)
        }
        ExecutionExprKind::Call { callee, args } => {
            let compiled = args
                .iter()
                .map(|arg| lower_expr(builder, env, arg))
                .collect::<Result<Vec<_>, _>>()?;
            lower_call(builder, env, callee, &compiled, expr.ty, expr.span)
        }
    }
}

fn lower_literal(builder: &mut FunctionBuilder<'_>, value: &ConstValue) -> LoweredValue {
    match value {
        ConstValue::Int(number) => LoweredValue {
            value: builder.ins().iconst(types::I64, *number),
            ty: ValueType::Int,
        },
        ConstValue::Real(number) => LoweredValue {
            value: builder.ins().f64const(*number),
            ty: ValueType::Real,
        },
        ConstValue::Bool(value) => LoweredValue {
            value: builder.ins().iconst(types::I64, if *value { 1 } else { 0 }),
            ty: ValueType::Bool,
        },
    }
}

fn lower_load(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    load: &ExecutionLoad,
    ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    let value = match load {
        ExecutionLoad::Parameter(index) => load_fixed(builder, env.args.params, *index, ty),
        ExecutionLoad::Covariate(index) => load_fixed(builder, env.args.covariates, *index, ty),
        ExecutionLoad::Derived(index) => load_fixed(builder, env.args.derived, *index, ty),
        ExecutionLoad::RouteInput { index, .. } => load_fixed(builder, env.args.routes, *index, ty),
        ExecutionLoad::Local(index) => {
            let binding = env.locals.get(index).ok_or_else(|| {
                JitCompileError::new(format!("unknown local slot {index}"), Some(span))
            })?;
            builder.use_var(binding.variable)
        }
        ExecutionLoad::State(state_ref) => {
            load_state_ref(builder, env, env.args.states, state_ref, ty)?
        }
    };

    Ok(LoweredValue { value, ty })
}

fn lower_unary(
    builder: &mut FunctionBuilder<'_>,
    _env: &EmitEnv<'_>,
    op: AnalyzedUnaryOp,
    value: LoweredValue,
    target_ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    match op {
        AnalyzedUnaryOp::Plus => cast_value(builder, value, target_ty, span),
        AnalyzedUnaryOp::Minus => {
            let value = cast_value(builder, value, target_ty, span)?;
            let compiled = match target_ty {
                ValueType::Real => builder.ins().fneg(value.value),
                ValueType::Int => builder.ins().ineg(value.value),
                ValueType::Bool => {
                    return Err(JitCompileError::new(
                        "cannot negate a boolean expression",
                        Some(span),
                    ))
                }
            };
            Ok(LoweredValue {
                value: compiled,
                ty: target_ty,
            })
        }
        AnalyzedUnaryOp::Not => {
            let condition = as_bool(builder, value, span)?;
            let condition_i64 = bool_to_i64(builder, condition);
            let is_zero = builder.ins().icmp_imm(IntCC::Equal, condition_i64, 0);
            let compiled = bool_to_i64(builder, is_zero);
            Ok(LoweredValue {
                value: compiled,
                ty: ValueType::Bool,
            })
        }
    }
}

fn lower_binary(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    op: AnalyzedBinaryOp,
    lhs: LoweredValue,
    rhs: LoweredValue,
    target_ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    match op {
        AnalyzedBinaryOp::Or => {
            let lhs = as_i64_bool(builder, lhs, span)?;
            let rhs = as_i64_bool(builder, rhs, span)?;
            Ok(LoweredValue {
                value: builder.ins().bor(lhs, rhs),
                ty: ValueType::Bool,
            })
        }
        AnalyzedBinaryOp::And => {
            let lhs = as_i64_bool(builder, lhs, span)?;
            let rhs = as_i64_bool(builder, rhs, span)?;
            Ok(LoweredValue {
                value: builder.ins().band(lhs, rhs),
                ty: ValueType::Bool,
            })
        }
        AnalyzedBinaryOp::Eq | AnalyzedBinaryOp::NotEq => {
            let value = lower_equality(builder, lhs, rhs, target_ty, op, span)?;
            Ok(LoweredValue {
                value,
                ty: ValueType::Bool,
            })
        }
        AnalyzedBinaryOp::Lt
        | AnalyzedBinaryOp::LtEq
        | AnalyzedBinaryOp::Gt
        | AnalyzedBinaryOp::GtEq => {
            let value = lower_comparison(builder, lhs, rhs, span, op)?;
            Ok(LoweredValue {
                value,
                ty: ValueType::Bool,
            })
        }
        AnalyzedBinaryOp::Add | AnalyzedBinaryOp::Sub | AnalyzedBinaryOp::Mul => {
            let lhs = cast_value(builder, lhs, target_ty, span)?;
            let rhs = cast_value(builder, rhs, target_ty, span)?;
            let value = match (op, target_ty) {
                (AnalyzedBinaryOp::Add, ValueType::Real) => {
                    builder.ins().fadd(lhs.value, rhs.value)
                }
                (AnalyzedBinaryOp::Sub, ValueType::Real) => {
                    builder.ins().fsub(lhs.value, rhs.value)
                }
                (AnalyzedBinaryOp::Mul, ValueType::Real) => {
                    builder.ins().fmul(lhs.value, rhs.value)
                }
                (AnalyzedBinaryOp::Add, ValueType::Int) => builder.ins().iadd(lhs.value, rhs.value),
                (AnalyzedBinaryOp::Sub, ValueType::Int) => builder.ins().isub(lhs.value, rhs.value),
                (AnalyzedBinaryOp::Mul, ValueType::Int) => builder.ins().imul(lhs.value, rhs.value),
                _ => {
                    return Err(JitCompileError::new(
                        "invalid arithmetic operand types",
                        Some(span),
                    ))
                }
            };
            Ok(LoweredValue {
                value,
                ty: target_ty,
            })
        }
        AnalyzedBinaryOp::Div => {
            let lhs = cast_value(builder, lhs, ValueType::Real, span)?;
            let rhs = cast_value(builder, rhs, ValueType::Real, span)?;
            Ok(LoweredValue {
                value: builder.ins().fdiv(lhs.value, rhs.value),
                ty: ValueType::Real,
            })
        }
        AnalyzedBinaryOp::Pow => lower_call(
            builder,
            env,
            &ExecutionCall::Math(MathFunction::Pow),
            &[lhs, rhs],
            target_ty,
            span,
        ),
    }
}

fn lower_call(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    callee: &ExecutionCall,
    args: &[LoweredValue],
    target_ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    match callee {
        ExecutionCall::Math(intrinsic) => {
            lower_math_call(builder, env, *intrinsic, args, target_ty, span)
        }
    }
}

fn lower_math_call(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    intrinsic: MathFunction,
    args: &[LoweredValue],
    target_ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    match intrinsic {
        MathFunction::Max | MathFunction::Min => {
            if args.len() != 2 {
                return Err(JitCompileError::new(
                    format!("{:?} expects 2 arguments", intrinsic),
                    Some(span),
                ));
            }
            let lhs = cast_value(builder, args[0], target_ty, span)?;
            let rhs = cast_value(builder, args[1], target_ty, span)?;
            let value = match target_ty {
                ValueType::Real => {
                    let compare = builder.ins().fcmp(
                        if intrinsic == MathFunction::Max {
                            FloatCC::GreaterThan
                        } else {
                            FloatCC::LessThan
                        },
                        lhs.value,
                        rhs.value,
                    );
                    builder.ins().select(compare, lhs.value, rhs.value)
                }
                ValueType::Int => {
                    let compare = builder.ins().icmp(
                        if intrinsic == MathFunction::Max {
                            IntCC::SignedGreaterThan
                        } else {
                            IntCC::SignedLessThan
                        },
                        lhs.value,
                        rhs.value,
                    );
                    builder.ins().select(compare, lhs.value, rhs.value)
                }
                ValueType::Bool => {
                    return Err(JitCompileError::new(
                        "min/max do not accept boolean arguments",
                        Some(span),
                    ))
                }
            };
            Ok(LoweredValue {
                value,
                ty: target_ty,
            })
        }
        MathFunction::Abs if target_ty == ValueType::Int => {
            let value = cast_value(builder, args[0], ValueType::Int, span)?;
            let is_negative = builder
                .ins()
                .icmp_imm(IntCC::SignedLessThan, value.value, 0);
            let negated = builder.ins().ineg(value.value);
            Ok(LoweredValue {
                value: builder.ins().select(is_negative, negated, value.value),
                ty: ValueType::Int,
            })
        }
        _ => {
            let function = match intrinsic {
                MathFunction::Abs => env.externs.abs,
                MathFunction::Ceil => env.externs.ceil,
                MathFunction::Exp => env.externs.exp,
                MathFunction::Floor => env.externs.floor,
                MathFunction::Ln | MathFunction::Log => env.externs.ln,
                MathFunction::Log10 => env.externs.log10,
                MathFunction::Log2 => env.externs.log2,
                MathFunction::Pow => env.externs.pow,
                MathFunction::Round => env.externs.round,
                MathFunction::Sin => env.externs.sin,
                MathFunction::Cos => env.externs.cos,
                MathFunction::Tan => env.externs.tan,
                MathFunction::Sqrt => env.externs.sqrt,
                MathFunction::Max | MathFunction::Min => unreachable!(),
            };

            let mut call_args = Vec::with_capacity(args.len());
            for arg in args {
                let arg = cast_value(builder, *arg, ValueType::Real, span)?;
                call_args.push(arg.value);
            }
            let call = builder.ins().call(function, &call_args);
            let result = builder.inst_results(call)[0];
            cast_value(
                builder,
                LoweredValue {
                    value: result,
                    ty: ValueType::Real,
                },
                target_ty,
                span,
            )
        }
    }
}

fn lower_equality(
    builder: &mut FunctionBuilder<'_>,
    lhs: LoweredValue,
    rhs: LoweredValue,
    target_ty: ValueType,
    op: AnalyzedBinaryOp,
    span: Span,
) -> Result<Value, JitCompileError> {
    let predicate = match op {
        AnalyzedBinaryOp::Eq => true,
        AnalyzedBinaryOp::NotEq => false,
        _ => unreachable!(),
    };

    let comparison = match target_ty {
        ValueType::Real => {
            let lhs = cast_value(builder, lhs, ValueType::Real, span)?;
            let rhs = cast_value(builder, rhs, ValueType::Real, span)?;
            builder.ins().fcmp(
                if predicate {
                    FloatCC::Equal
                } else {
                    FloatCC::NotEqual
                },
                lhs.value,
                rhs.value,
            )
        }
        ValueType::Int | ValueType::Bool => {
            let lhs = cast_value(builder, lhs, target_ty, span)?;
            let rhs = cast_value(builder, rhs, target_ty, span)?;
            builder.ins().icmp(
                if predicate {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                },
                lhs.value,
                rhs.value,
            )
        }
    };

    Ok(bool_to_i64(builder, comparison))
}

fn lower_comparison(
    builder: &mut FunctionBuilder<'_>,
    lhs: LoweredValue,
    rhs: LoweredValue,
    span: Span,
    op: AnalyzedBinaryOp,
) -> Result<Value, JitCompileError> {
    let comparison = if lhs.ty == ValueType::Real || rhs.ty == ValueType::Real {
        let lhs = cast_value(builder, lhs, ValueType::Real, span)?;
        let rhs = cast_value(builder, rhs, ValueType::Real, span)?;
        builder.ins().fcmp(
            match op {
                AnalyzedBinaryOp::Lt => FloatCC::LessThan,
                AnalyzedBinaryOp::LtEq => FloatCC::LessThanOrEqual,
                AnalyzedBinaryOp::Gt => FloatCC::GreaterThan,
                AnalyzedBinaryOp::GtEq => FloatCC::GreaterThanOrEqual,
                _ => unreachable!(),
            },
            lhs.value,
            rhs.value,
        )
    } else {
        let lhs = cast_value(builder, lhs, ValueType::Int, span)?;
        let rhs = cast_value(builder, rhs, ValueType::Int, span)?;
        builder.ins().icmp(
            match op {
                AnalyzedBinaryOp::Lt => IntCC::SignedLessThan,
                AnalyzedBinaryOp::LtEq => IntCC::SignedLessThanOrEqual,
                AnalyzedBinaryOp::Gt => IntCC::SignedGreaterThan,
                AnalyzedBinaryOp::GtEq => IntCC::SignedGreaterThanOrEqual,
                _ => unreachable!(),
            },
            lhs.value,
            rhs.value,
        )
    };
    Ok(bool_to_i64(builder, comparison))
}

fn cast_value(
    builder: &mut FunctionBuilder<'_>,
    value: LoweredValue,
    target_ty: ValueType,
    span: Span,
) -> Result<LoweredValue, JitCompileError> {
    if value.ty == target_ty {
        return Ok(value);
    }

    let compiled = match (value.ty, target_ty) {
        (ValueType::Int, ValueType::Real) => builder.ins().fcvt_from_sint(types::F64, value.value),
        (ValueType::Bool, ValueType::Real) => {
            let condition = as_bool(builder, value, span)?;
            let as_int = bool_to_i64(builder, condition);
            builder.ins().fcvt_from_sint(types::F64, as_int)
        }
        (ValueType::Real, ValueType::Int) => builder.ins().fcvt_to_sint(types::I64, value.value),
        (ValueType::Bool, ValueType::Int) => as_i64_bool(builder, value, span)?,
        (ValueType::Int, ValueType::Bool) | (ValueType::Real, ValueType::Bool) => {
            let condition = as_bool(builder, value, span)?;
            bool_to_i64(builder, condition)
        }
        (ValueType::Bool, ValueType::Bool) => value.value,
        _ => {
            return Err(JitCompileError::new(
                format!("unsupported cast from {:?} to {:?}", value.ty, target_ty),
                Some(span),
            ))
        }
    };

    Ok(LoweredValue {
        value: compiled,
        ty: target_ty,
    })
}

fn as_bool(
    builder: &mut FunctionBuilder<'_>,
    value: LoweredValue,
    _span: Span,
) -> Result<Value, JitCompileError> {
    match value.ty {
        ValueType::Bool | ValueType::Int => {
            Ok(builder.ins().icmp_imm(IntCC::NotEqual, value.value, 0))
        }
        ValueType::Real => {
            let zero = builder.ins().f64const(0.0);
            Ok(builder.ins().fcmp(FloatCC::NotEqual, value.value, zero))
        }
    }
}

fn as_i64_bool(
    builder: &mut FunctionBuilder<'_>,
    value: LoweredValue,
    span: Span,
) -> Result<Value, JitCompileError> {
    let condition = as_bool(builder, value, span)?;
    Ok(bool_to_i64(builder, condition))
}

fn bool_to_i64(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let one = builder.ins().iconst(types::I64, 1);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().select(value, one, zero)
}

fn clif_type(ty: ValueType) -> Type {
    match ty {
        ValueType::Real => types::F64,
        ValueType::Int | ValueType::Bool => types::I64,
    }
}

fn zero_value(builder: &mut FunctionBuilder<'_>, ty: ValueType) -> Value {
    match ty {
        ValueType::Real => builder.ins().f64const(0.0),
        ValueType::Int | ValueType::Bool => builder.ins().iconst(types::I64, 0),
    }
}

fn load_fixed(
    builder: &mut FunctionBuilder<'_>,
    base: Value,
    index: usize,
    ty: ValueType,
) -> Value {
    builder
        .ins()
        .load(clif_type(ty), MemFlags::new(), base, (index * 8) as i32)
}

fn store_fixed(builder: &mut FunctionBuilder<'_>, base: Value, index: usize, value: Value) {
    builder
        .ins()
        .store(MemFlags::new(), value, base, (index * 8) as i32);
}

fn load_state_ref(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    base: Value,
    state_ref: &ExecutionStateRef,
    ty: ValueType,
) -> Result<Value, JitCompileError> {
    let address = state_address(builder, env, base, state_ref)?;
    Ok(builder
        .ins()
        .load(clif_type(ty), MemFlags::new(), address, 0))
}

fn store_state_ref(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    base: Value,
    state_ref: &ExecutionStateRef,
    value: Value,
) -> Result<(), JitCompileError> {
    let address = state_address(builder, env, base, state_ref)?;
    builder.ins().store(MemFlags::new(), value, address, 0);
    Ok(())
}

fn state_address(
    builder: &mut FunctionBuilder<'_>,
    env: &EmitEnv<'_>,
    base: Value,
    state_ref: &ExecutionStateRef,
) -> Result<Value, JitCompileError> {
    let element_index = if let Some(index_expr) = &state_ref.index {
        let index_expr = lower_expr(builder, env, index_expr)?;
        let index = cast_value(builder, index_expr, ValueType::Int, state_ref.span)?;
        builder
            .ins()
            .iadd_imm(index.value, state_ref.base_offset as i64)
    } else {
        builder
            .ins()
            .iconst(types::I64, state_ref.base_offset as i64)
    };
    let byte_offset = builder.ins().imul_imm(element_index, 8);
    Ok(builder.ins().iadd(base, byte_offset))
}

/// Compile an [`ExecutionModel`](pharmsol_dsl::ExecutionModel) to the native
/// in-process JIT backend.
///
/// Use this low-level entrypoint when you already own the parse, analyze, and
/// lower steps and want the JIT backend directly instead of the higher-level
/// runtime facade.
///
/// This function requires the `dsl-jit` feature.
///
/// ```rust,no_run
/// use pharmsol::dsl::{
///     analyze_model, compile_execution_model_to_jit, compile_analyzed_model, parse_model,
/// };
///
/// let parsed = parse_model(
///     r#"
/// model implicit_route_injection {
///     kind ode
///     states { central }
///     routes { iv -> central }
///     dynamics {
///         ddt(central) = 0
///     }
///     outputs {
///         cp = central
///     }
/// }
/// "#,
/// )?;
/// let analyzed = analyze_model(&parsed)?;
/// let execution = compile_analyzed_model(&analyzed)?;
/// let compiled = compile_execution_model_to_jit(&execution)?;
/// # let _ = compiled;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_execution_model_to_jit(
    model: &ExecutionModel,
) -> Result<CompiledJitModel, JitCompileError> {
    match model.kind {
        ModelKind::Ode => Ok(CompiledJitModel::Ode(compile_ode_model_to_jit(model)?)),
        ModelKind::Analytical => Ok(CompiledJitModel::Analytical(
            compile_analytical_model_to_jit(model)?,
        )),
        ModelKind::Sde => Ok(CompiledJitModel::Sde(compile_sde_model_to_jit(model)?)),
    }
}

/// Compile an ODE execution model to the native in-process JIT backend.
pub fn compile_ode_model_to_jit(model: &ExecutionModel) -> Result<JitOdeModel, JitCompileError> {
    if model.kind != ModelKind::Ode {
        return Err(JitCompileError::new(
            format!(
                "model `{}` is {:?}, not an ODE model",
                model.name, model.kind
            ),
            Some(model.span),
        ));
    }
    JitOdeModel::new(
        NativeModelInfo::from_execution_model(model),
        compile_execution_artifact(model)?,
    )
    .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))
}

/// Compile an analytical execution model to the native in-process JIT backend.
pub fn compile_analytical_model_to_jit(
    model: &ExecutionModel,
) -> Result<JitAnalyticalModel, JitCompileError> {
    if model.kind != ModelKind::Analytical {
        return Err(JitCompileError::new(
            format!(
                "model `{}` is {:?}, not an analytical model",
                model.name, model.kind
            ),
            Some(model.span),
        ));
    }
    JitAnalyticalModel::new(
        NativeModelInfo::from_execution_model(model),
        compile_execution_artifact(model)?,
    )
    .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))
}

/// Compile an SDE execution model to the native in-process JIT backend.
pub fn compile_sde_model_to_jit(model: &ExecutionModel) -> Result<JitSdeModel, JitCompileError> {
    if model.kind != ModelKind::Sde {
        return Err(JitCompileError::new(
            format!(
                "model `{}` is {:?}, not an SDE model",
                model.name, model.kind
            ),
            Some(model.span),
        ));
    }
    JitSdeModel::new(
        NativeModelInfo::from_execution_model(model),
        compile_execution_artifact(model)?,
    )
    .map_err(|error| JitCompileError::new(error.to_string(), Some(model.span)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equation::ode::{ExplicitRkTableau, OdeSolver};
    use crate::equation::{ODE, SDE};
    use crate::simulator::equation::analytical::one_compartment_with_absorption;
    use crate::simulator::equation::{Equation, Predictions as PredictionTrait};
    use crate::test_fixtures::STRUCTURED_BLOCK_CORPUS;
    use crate::{equation, Parameters, Subject, SubjectBuilderExt};
    use approx::assert_relative_eq;
    use diffsol::Vector;
    use pharmsol_dsl::execution::BufferLayout;

    fn load_corpus_model(name: &str) -> ExecutionModel {
        let source = STRUCTURED_BLOCK_CORPUS;
        let parsed = pharmsol_dsl::parse_module(source).expect("parse corpus module");
        let analyzed = pharmsol_dsl::analyze_module(&parsed).expect("analyze corpus module");
        let model = analyzed
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model present in corpus module");
        pharmsol_dsl::compile_analyzed_model(model).expect("lower corpus model")
    }

    #[test]
    fn jit_compile_error_exposes_backend_diagnostic_report() {
        let source = STRUCTURED_BLOCK_CORPUS;
        let model = load_corpus_model("one_cmt_oral_iv");
        let error = compile_sde_model_to_jit(&model)
            .expect_err("ODE model should not compile through the SDE JIT entrypoint")
            .with_source(source);

        let diagnostic = error.diagnostic();
        assert_eq!(diagnostic.phase, pharmsol_dsl::DiagnosticPhase::Backend);
        assert_eq!(diagnostic.code, pharmsol_dsl::DSL_BACKEND_GENERIC);
        assert!(diagnostic.message.contains("not an SDE model"));

        let rendered = error.render(source);
        assert!(rendered.contains("error[DSL4000]"), "{}", rendered);
        assert!(rendered.contains("not an SDE model"), "{}", rendered);

        let report = error.diagnostic_report("model.dsl");
        assert_eq!(report.source.name, "model.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL4000");
        assert_eq!(report.diagnostics[0].phase, "backend");
        assert!(report.diagnostics[0].labels[0].span.start_line.is_some());

        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL4000]"), "{}", debugged);
    }

    #[test]
    fn authoring_runtime_shares_input_between_bolus_and_infusion_routes() {
        let source = r#"
name = shared_authoring
kind = ode

params = ka, ke, v
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(cp) = central / v ~ continuous()
"#;
        let parsed = pharmsol_dsl::parse_model(source).expect("authoring model parses");
        let analyzed = pharmsol_dsl::analyze_model(&parsed).expect("authoring model analyzes");
        let model =
            pharmsol_dsl::compile_analyzed_model(&analyzed).expect("authoring model lowers");
        let jit = compile_ode_model_to_jit(&model)
            .expect("compile jit ode model")
            .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

        let oral = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "oral")
            .map(|route| route.index)
            .expect("oral route");
        let iv = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "iv")
            .map(|route| route.index)
            .expect("iv route");
        let cp = jit
            .info()
            .outputs
            .iter()
            .find(|output| output.name == "cp")
            .map(|output| output.index)
            .expect("cp output");
        assert_eq!(oral, 0);
        assert_eq!(iv, 0);
        assert_eq!(cp, 0);

        let jit_subject = Subject::builder("ode")
            .bolus(0.0, 120.0, "oral")
            .infusion(6.0, 60.0, "iv", 2.0)
            .observation(0.5, 0.0, "cp")
            .observation(1.0, 0.0, "cp")
            .observation(2.0, 0.0, "cp")
            .observation(6.0, 0.0, "cp")
            .observation(7.0, 0.0, "cp")
            .observation(9.0, 0.0, "cp")
            .build();

        let reference_subject = Subject::builder("ode")
            .bolus(0.0, 120.0, "oral")
            .infusion(6.0, 60.0, "iv", 2.0)
            .observation(0.5, 0.0, "cp")
            .observation(1.0, 0.0, "cp")
            .observation(2.0, 0.0, "cp")
            .observation(6.0, 0.0, "cp")
            .observation(7.0, 0.0, "cp")
            .observation(9.0, 0.0, "cp")
            .build();

        let support = Parameters::with_model(
            &crate::dsl::CompiledRuntimeModel::Ode(jit.clone()),
            [("ka", 1.2), ("ke", 0.15), ("v", 40.0)],
        )
        .expect("valid named parameters");
        let jit_predictions = jit
            .estimate_predictions(&jit_subject, &support)
            .expect("jit predictions");

        let reference = ODE::new(
            |x, p, _t, dx, bolus, rateiv, _cov| {
                let ka = p[0];
                let ke = p[1];
                dx[0] = -ka * x[0] + bolus[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
            },
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                y[0] = x[1] / p[2];
            },
        )
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
        .with_metadata(
            equation::metadata::new("shared_authoring")
                .parameters(["ka", "ke", "v"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .routes([
                    equation::Route::bolus("oral")
                        .to_state("depot")
                        .expect_explicit_input(),
                    equation::Route::infusion("iv")
                        .to_state("central")
                        .expect_explicit_input(),
                ]),
        )
        .expect("reference ode metadata should validate");

        let reference_predictions = reference
            .estimate_predictions(&reference_subject, &support)
            .expect("reference ode predictions");

        for (jit_pred, reference_pred) in jit_predictions
            .predictions()
            .iter()
            .zip(reference_predictions.predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                reference_pred.prediction(),
                max_relative = 1e-4
            );
        }
    }

    fn slot_index(layout: &BufferLayout, name: &str) -> usize {
        layout
            .slots
            .iter()
            .find(|slot| slot.name == name)
            .expect("slot present")
            .offset
    }

    #[test]
    fn compiles_dense_execution_functions_for_ode_models() {
        let model = load_corpus_model("one_cmt_oral_iv");
        let artifact = compile_execution_artifact(&model).expect("compile execution artifact");

        let mut derived = vec![0.0; model.layout.derived_buffer.len];
        let mut dx = vec![0.0; model.layout.state_buffer.len];
        let mut out = vec![0.0; model.layout.output_buffer.len];
        let states = [100.0, 0.0];
        let params = [1.0, 5.0, 50.0, 1.5, 0.8];
        let covariates = [70.0];
        let routes = [0.0, 0.0];

        let derive = artifact.derive.expect("derive function present");
        unsafe {
            derive(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                derived.as_mut_ptr(),
            );
            artifact.dynamics.expect("dynamics function present")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                dx.as_mut_ptr(),
            );
            (artifact.outputs)(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                out.as_mut_ptr(),
            );
        }

        let derived_layout = &model.layout.derived_buffer;
        assert!((derived[slot_index(derived_layout, "cl_i")] - 5.0).abs() < 1e-12);
        assert!((derived[slot_index(derived_layout, "v_i")] - 50.0).abs() < 1e-12);
        assert!((derived[slot_index(derived_layout, "ke")] - 0.1).abs() < 1e-12);

        let state_layout = &model.layout.state_buffer;
        assert!((dx[slot_index(state_layout, "depot")] + 100.0).abs() < 1e-12);
        assert!((dx[slot_index(state_layout, "central")] - 100.0).abs() < 1e-12);

        let output_layout = &model.layout.output_buffer;
        assert_eq!(out[slot_index(output_layout, "cp")], 0.0);
    }

    #[test]
    fn jit_ode_wrapper_matches_existing_ode_predictions() {
        let model = load_corpus_model("one_cmt_oral_iv");
        let jit = compile_ode_model_to_jit(&model)
            .expect("compile jit ode model")
            .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

        let oral = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "oral")
            .map(|route| route.index)
            .expect("oral route");
        let iv = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "iv")
            .map(|route| route.index)
            .expect("iv route");
        let cp = jit
            .info()
            .outputs
            .iter()
            .find(|output| output.name == "cp")
            .map(|output| output.index)
            .expect("cp output");
        assert_eq!(oral, 0);
        assert_eq!(iv, 1);
        assert_eq!(cp, 0);

        let jit_subject = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, "oral")
            .infusion(6.0, 60.0, "iv", 2.0)
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(6.0, "cp")
            .missing_observation(7.0, "cp")
            .missing_observation(9.0, "cp")
            .build();

        let reference_subject = Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, "oral")
            .infusion(6.0, 60.0, "iv", 2.0)
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(6.0, "cp")
            .missing_observation(7.0, "cp")
            .missing_observation(9.0, "cp")
            .build();

        let support = Parameters::with_model(
            &crate::dsl::CompiledRuntimeModel::Ode(jit.clone()),
            [
                ("ka", 1.2),
                ("cl", 5.0),
                ("v", 40.0),
                ("tlag", 0.5),
                ("f_oral", 0.8),
            ],
        )
        .expect("valid named parameters");
        let jit_predictions = jit
            .estimate_predictions(&jit_subject, &support)
            .expect("jit predictions");

        let reference = ODE::new(
            |x, p, t, dx, bolus, rateiv, cov| {
                let wt: f64 = cov
                    .get_covariate("wt")
                    .map(|values| values.interpolate(t).unwrap())
                    .unwrap();
                let ka = p[0];
                let cl = p[1];
                let v = p[2];
                let cl_i = cl * (wt / 70.0).powf(0.75);
                let v_i = if wt > 120.0 { v * 1.15 } else { v };
                let ke = cl_i / v_i;
                dx[0] = -ka * x[0] + bolus[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[1] + bolus[1];
            },
            |p, _t, _cov| {
                let mut lag = std::collections::HashMap::new();
                lag.insert(0, p[3]);
                lag
            },
            |p, _t, _cov| {
                let mut fa = std::collections::HashMap::new();
                fa.insert(0, p[4]);
                fa
            },
            |_p, _t, _cov, _x| {},
            |x, p, t, cov, y| {
                let wt: f64 = cov
                    .get_covariate("wt")
                    .map(|values| values.interpolate(t).unwrap())
                    .unwrap();
                let v = p[2];
                let v_i = if wt > 120.0 { v * 1.15 } else { v };
                y[0] = x[1] / v_i;
            },
        )
        .with_nstates(2)
        .with_ndrugs(2)
        .with_nout(1)
        .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
        .with_metadata(
            equation::metadata::new("one_cmt_oral_iv")
                .parameters(["ka", "cl", "v", "tlag", "f_oral"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .routes([
                    equation::Route::bolus("oral")
                        .to_state("depot")
                        .expect_explicit_input(),
                    equation::Route::infusion("iv_pad")
                        .to_state("central")
                        .expect_explicit_input(),
                    equation::Route::infusion("iv")
                        .to_state("central")
                        .expect_explicit_input(),
                ]),
        )
        .expect("reference ode metadata should validate");

        let reference_predictions = reference
            .estimate_predictions(&reference_subject, &support)
            .expect("reference ode predictions");

        for (jit_pred, reference_pred) in jit_predictions
            .predictions()
            .iter()
            .zip(reference_predictions.predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                reference_pred.prediction(),
                max_relative = 1e-4
            );
        }
    }

    #[test]
    fn jit_analytical_wrapper_matches_existing_analytical_predictions() {
        let model = load_corpus_model("one_cmt_abs");
        let jit = compile_analytical_model_to_jit(&model).expect("compile jit analytical model");

        let oral = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "oral")
            .map(|route| route.index)
            .expect("oral route");
        let cp = jit
            .info()
            .outputs
            .iter()
            .find(|output| output.name == "cp")
            .map(|output| output.index)
            .expect("cp output");
        assert_eq!(oral, 0);
        assert_eq!(cp, 0);

        let jit_subject = Subject::builder("analytical")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let reference_subject = Subject::builder("analytical")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let support = Parameters::with_model(&jit, [("ka", 1.0), ("ke", 0.15), ("v", 25.0)])
            .expect("valid named parameters");
        let jit_predictions = jit
            .estimate_predictions(&jit_subject, &support)
            .expect("jit analytical predictions");

        let reference = equation::Analytical::new(
            one_compartment_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                y[0] = x[1] / p[2];
            },
        )
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            equation::metadata::new("one_cmt_abs")
                .kind(equation::ModelKind::Analytical)
                .parameters(["ka", "ke", "v"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(equation::Route::bolus("oral").to_state("depot"))
                .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
        )
        .expect("reference analytical metadata should validate");

        let reference_predictions = reference
            .estimate_predictions(&reference_subject, &support)
            .expect("reference analytical predictions");

        for (jit_pred, reference_pred) in jit_predictions
            .predictions()
            .iter()
            .zip(reference_predictions.predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                reference_pred.prediction(),
                max_relative = 1e-4
            );
        }
    }

    #[test]
    fn jit_analytical_wrapper_supports_mixed_primary_and_derived_structure_inputs() {
        let source = r#"
model analytical_mixed {
    kind analytical
    parameters { ka, v, ke0 }
    states { depot, central }
    routes { oral -> depot }
    derive {
        ke = ke0
    }
    analytical {
        structure = one_compartment_with_absorption
    }
    outputs {
        cp = central / v
    }
}
"#;
        let parsed = pharmsol_dsl::parse_model(source).expect("analytical model parses");
        let analyzed = pharmsol_dsl::analyze_model(&parsed).expect("analytical model analyzes");
        let model =
            pharmsol_dsl::compile_analyzed_model(&analyzed).expect("analytical model lowers");
        let jit = compile_analytical_model_to_jit(&model).expect("compile jit analytical model");

        assert_eq!(jit.info().derived, vec!["ke".to_string()]);

        let jit_subject = Subject::builder("analytical")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let reference_subject = Subject::builder("analytical")
            .bolus(0.0, 100.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let jit_support = Parameters::with_model(&jit, [("ka", 1.0), ("v", 25.0), ("ke0", 0.15)])
            .expect("valid named parameters");
        let jit_predictions = jit
            .estimate_predictions(&jit_subject, &jit_support)
            .expect("jit analytical predictions");

        let reference = equation::Analytical::new(
            one_compartment_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                y[0] = x[1] / p[2];
            },
        )
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            equation::metadata::new("analytical_mixed")
                .kind(equation::ModelKind::Analytical)
                .parameters(["ka", "v", "ke0"])
                .states(["depot", "central"])
                .outputs(["cp"])
                .route(equation::Route::bolus("oral").to_state("depot"))
                .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
        )
        .expect("reference analytical metadata should validate");

        let reference_support = crate::parameters::dense(vec![1.0, 0.15, 25.0]);
        let reference_predictions = reference
            .estimate_predictions(&reference_subject, &reference_support)
            .expect("reference analytical predictions");

        for (jit_pred, reference_pred) in jit_predictions
            .predictions()
            .iter()
            .zip(reference_predictions.predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                reference_pred.prediction(),
                max_relative = 1e-4
            );
        }
    }

    #[test]
    fn jit_sde_wrapper_matches_zero_diffusion_reference() {
        let model = load_corpus_model("vanco_sde");
        let jit = compile_sde_model_to_jit(&model)
            .expect("compile jit sde model")
            .with_particles(64);

        let oral = jit
            .info()
            .routes
            .iter()
            .find(|route| route.name == "oral")
            .map(|route| route.index)
            .expect("oral route");
        let cp = jit
            .info()
            .outputs
            .iter()
            .find(|output| output.name == "cp")
            .map(|output| output.index)
            .expect("cp output");
        assert_eq!(oral, 0);
        assert_eq!(cp, 0);

        let jit_subject = Subject::builder("sde")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 80.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let reference_subject = Subject::builder("sde")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 80.0, "oral")
            .missing_observation(0.5, "cp")
            .missing_observation(1.0, "cp")
            .missing_observation(2.0, "cp")
            .missing_observation(4.0, "cp")
            .build();

        let support = Parameters::with_model(
            &jit,
            [
                ("ka", 1.1),
                ("ke0", 0.2),
                ("kcp", 0.12),
                ("kpc", 0.08),
                ("vol", 15.0),
                ("ske", 0.0),
            ],
        )
        .expect("valid named parameters");
        let jit_predictions = jit
            .estimate_predictions(&jit_subject, &support)
            .expect("jit sde predictions");

        let reference = SDE::new(
            |x, p, _t, dx, _rateiv, _cov| {
                let ka = p[0];
                let ke0 = p[1];
                let kcp = p[2];
                let kpc = p[3];
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - (x[3] + kcp) * x[1] + kpc * x[2];
                dx[2] = kcp * x[1] - kpc * x[2];
                dx[3] = -x[3] + ke0;
            },
            |p, sigma| {
                sigma.fill(0.0);
                sigma[3] = p[5];
            },
            |_p, _t, _cov| std::collections::HashMap::new(),
            |_p, _t, _cov| std::collections::HashMap::new(),
            |p, _t, _cov, x| {
                x[3] = p[1];
            },
            |x, p, t, cov, y| {
                let wt: f64 = cov
                    .get_covariate("wt")
                    .map(|values| values.interpolate(t).unwrap())
                    .unwrap();
                y[0] = x[1] / (p[4] * wt);
            },
            64,
        )
        .with_nstates(4)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            equation::metadata::new("vanco_sde")
                .parameters(["ka", "ke0", "kcp", "kpc", "vol", "ske"])
                .states(["depot", "central", "peripheral", "ke"])
                .outputs(["cp"])
                .route(
                    equation::Route::bolus("oral")
                        .to_state("depot")
                        .expect_explicit_input(),
                )
                .particles(64),
        )
        .expect("reference sde metadata should validate");

        let reference_predictions = reference
            .estimate_predictions(&reference_subject, &support)
            .expect("reference sde predictions");

        for (jit_pred, reference_pred) in jit_predictions
            .get_predictions()
            .iter()
            .zip(reference_predictions.get_predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                reference_pred.prediction(),
                max_relative = 1e-4
            );
        }
    }
}
