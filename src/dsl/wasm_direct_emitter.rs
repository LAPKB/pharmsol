use std::collections::BTreeMap;

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, GlobalSection, GlobalType, ImportSection, Instruction, MemArg,
    MemorySection, MemoryType, Module, TypeSection, ValType,
};

use super::compiled_backend_abi::{
    compiled_kernel_symbol, encode_compiled_model_info, ALLOC_F64_BUFFER_SYMBOL,
    API_VERSION_SYMBOL, FREE_F64_BUFFER_SYMBOL, MODEL_INFO_JSON_LEN_SYMBOL,
    MODEL_INFO_JSON_PTR_SYMBOL,
};
use super::execution::{
    ExecutionExpr, ExecutionExprKind, ExecutionKernel, ExecutionLoad, ExecutionModel,
    ExecutionProgram, ExecutionStateRef, ExecutionStmt, ExecutionStmtKind, ExecutionTargetKind,
    KernelImplementation, KernelRole,
};
use super::{ConstValue, MathIntrinsic, ModelKind, TypedBinaryOp, TypedUnaryOp, ValueType};
use crate::dsl::WasmError;

const PAGE_SIZE: usize = 65_536;
const ABI_PTR_ALIGNMENT: usize = std::mem::size_of::<f64>();
const MODEL_INFO_PTR: i32 = 0;

const API_VERSION_TYPE: u32 = 0;
const ALLOC_TYPE: u32 = 1;
const FREE_TYPE: u32 = 2;
const KERNEL_TYPE: u32 = 3;
const UNARY_REAL_IMPORT_TYPE: u32 = 4;
const BINARY_REAL_IMPORT_TYPE: u32 = 5;

const HEAP_PTR_GLOBAL: u32 = 0;

const KERNEL_PARAM_STATES: u32 = 1;
const KERNEL_PARAM_PARAMS: u32 = 2;
const KERNEL_PARAM_COVARIATES: u32 = 3;
const KERNEL_PARAM_ROUTES: u32 = 4;
const KERNEL_PARAM_DERIVED: u32 = 5;
const KERNEL_PARAM_OUT: u32 = 6;
const FIRST_WASM_LOCAL_INDEX: u32 = KERNEL_PARAM_OUT + 1;

pub(crate) const DIRECT_WASM_IMPORT_MODULE: &str = "pharmsol_dsl_host_math";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectUnaryMathImport {
    pub name: &'static str,
    pub intrinsic: MathIntrinsic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectBinaryMathImport {
    pub name: &'static str,
    pub intrinsic: MathIntrinsic,
}

pub(crate) const DIRECT_WASM_UNARY_MATH_IMPORTS: [DirectUnaryMathImport; 8] = [
    DirectUnaryMathImport {
        name: "exp",
        intrinsic: MathIntrinsic::Exp,
    },
    DirectUnaryMathImport {
        name: "ln",
        intrinsic: MathIntrinsic::Ln,
    },
    DirectUnaryMathImport {
        name: "log10",
        intrinsic: MathIntrinsic::Log10,
    },
    DirectUnaryMathImport {
        name: "log2",
        intrinsic: MathIntrinsic::Log2,
    },
    DirectUnaryMathImport {
        name: "round",
        intrinsic: MathIntrinsic::Round,
    },
    DirectUnaryMathImport {
        name: "sin",
        intrinsic: MathIntrinsic::Sin,
    },
    DirectUnaryMathImport {
        name: "cos",
        intrinsic: MathIntrinsic::Cos,
    },
    DirectUnaryMathImport {
        name: "tan",
        intrinsic: MathIntrinsic::Tan,
    },
];

pub(crate) const DIRECT_WASM_BINARY_MATH_IMPORTS: [DirectBinaryMathImport; 1] =
    [DirectBinaryMathImport {
        name: "pow",
        intrinsic: MathIntrinsic::Pow,
    }];

const DIRECT_SUPPORTED_STATEMENT_KERNEL_ROLES: [KernelRole; 8] = [
    KernelRole::Derive,
    KernelRole::Dynamics,
    KernelRole::Outputs,
    KernelRole::Init,
    KernelRole::Drift,
    KernelRole::Diffusion,
    KernelRole::RouteLag,
    KernelRole::RouteBioavailability,
];

#[derive(Debug, Clone, Copy)]
struct WasmLocalBinding {
    wasm_local: u32,
    ty: ValueType,
}

struct KernelEmitState<'a> {
    model_name: &'a str,
    locals: BTreeMap<usize, WasmLocalBinding>,
    next_hidden_i64_local: u32,
}

pub(crate) fn compile_execution_model_to_wasm_bytes(
    model: &ExecutionModel,
    api_version: u32,
) -> Result<Vec<u8>, WasmError> {
    let kernels = collect_direct_statement_kernels(model)?;
    let metadata = encode_compiled_model_info(model, api_version)?;
    let metadata_bytes = metadata.into_bytes();
    let aligned_heap_start = align_to_f64_boundary(metadata_bytes.len())?;

    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I32]);
    types.ty().function([ValType::I32], [ValType::I32]);
    types.ty().function([ValType::I32, ValType::I32], []);
    types.ty().function(
        [
            ValType::F64,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        [],
    );
    types.ty().function([ValType::F64], [ValType::F64]);
    types.ty().function([ValType::F64, ValType::F64], [ValType::F64]);
    module.section(&types);

    let mut imports = ImportSection::new();
    for import in DIRECT_WASM_UNARY_MATH_IMPORTS {
        imports.import(
            DIRECT_WASM_IMPORT_MODULE,
            import.name,
            EntityType::Function(UNARY_REAL_IMPORT_TYPE),
        );
    }
    for import in DIRECT_WASM_BINARY_MATH_IMPORTS {
        imports.import(
            DIRECT_WASM_IMPORT_MODULE,
            import.name,
            EntityType::Function(BINARY_REAL_IMPORT_TYPE),
        );
    }
    module.section(&imports);

    let mut functions = FunctionSection::new();
    functions.function(API_VERSION_TYPE);
    functions.function(API_VERSION_TYPE);
    functions.function(API_VERSION_TYPE);
    functions.function(ALLOC_TYPE);
    functions.function(FREE_TYPE);
    for _ in &kernels {
        functions.function(KERNEL_TYPE);
    }
    module.section(&functions);

    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: pages_for_bytes(usize::try_from(aligned_heap_start.max(1)).map_err(|_| {
            WasmError::Emit(format!(
                "direct backend memory size overflow for model `{}`",
                model.name
            ))
        })?) as u64,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&memories);

    let mut globals = GlobalSection::new();
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &ConstExpr::i32_const(aligned_heap_start),
    );
    module.section(&globals);

    let first_defined_function_index = direct_wasm_import_count() as u32;
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export(API_VERSION_SYMBOL, ExportKind::Func, first_defined_function_index);
    exports.export(
        MODEL_INFO_JSON_PTR_SYMBOL,
        ExportKind::Func,
        first_defined_function_index + 1,
    );
    exports.export(
        MODEL_INFO_JSON_LEN_SYMBOL,
        ExportKind::Func,
        first_defined_function_index + 2,
    );
    exports.export(
        ALLOC_F64_BUFFER_SYMBOL,
        ExportKind::Func,
        first_defined_function_index + 3,
    );
    exports.export(
        FREE_F64_BUFFER_SYMBOL,
        ExportKind::Func,
        first_defined_function_index + 4,
    );
    for (kernel_index, kernel) in kernels.iter().enumerate() {
        let symbol = compiled_kernel_symbol(kernel.role).ok_or_else(|| {
            WasmError::DirectBackendUnsupported {
                model: model.name.clone(),
                reason: format!("missing compiled kernel symbol for role {:?}", kernel.role),
            }
        })?;
        exports.export(
            symbol,
            ExportKind::Func,
            first_defined_function_index + 5 + kernel_index as u32,
        );
    }
    module.section(&exports);

    let mut code = CodeSection::new();
    code.function(&const_i32_function(api_version as i32));
    code.function(&const_i32_function(MODEL_INFO_PTR));
    code.function(&const_i32_function(i32::try_from(metadata_bytes.len()).map_err(|_| {
        WasmError::Emit(format!(
            "direct backend metadata for model `{}` is too large to address with wasm32",
            model.name
        ))
    })?));
    code.function(&alloc_function());
    code.function(&free_function());
    for kernel in &kernels {
        code.function(&emit_statement_kernel(model, kernel)?);
    }
    module.section(&code);

    let mut data = DataSection::new();
    data.active(0, &ConstExpr::i32_const(MODEL_INFO_PTR), metadata_bytes);
    module.section(&data);

    Ok(module.finish())
}

fn collect_direct_statement_kernels(model: &ExecutionModel) -> Result<Vec<&ExecutionKernel>, WasmError> {
    if !matches!(model.kind, ModelKind::Ode | ModelKind::Analytical | ModelKind::Sde) {
        return Err(WasmError::DirectBackendUnsupported {
            model: model.name.clone(),
            reason: format!(
                "direct emission currently supports ODE, analytical, and SDE models only; {:?} remains in later slices",
                model.kind
            ),
        });
    }

    let unsupported_roles = model
        .kernels
        .iter()
        .filter(|kernel| {
            kernel.role != KernelRole::Analytical
                && !DIRECT_SUPPORTED_STATEMENT_KERNEL_ROLES.contains(&kernel.role)
        })
        .map(|kernel| format!("{:?}", kernel.role))
        .collect::<Vec<_>>();
    if !unsupported_roles.is_empty() {
        return Err(WasmError::DirectBackendUnsupported {
            model: model.name.clone(),
            reason: format!(
                "direct emission supports only metadata-only analytical kernels plus statement kernels {:?}; found {}",
                DIRECT_SUPPORTED_STATEMENT_KERNEL_ROLES,
                unsupported_roles.join(", ")
            ),
        });
    }

    if model.kernel(KernelRole::Outputs).is_none() {
        return Err(WasmError::DirectBackendUnsupported {
            model: model.name.clone(),
            reason: "direct emitter requires an outputs kernel".to_string(),
        });
    }

    let mut kernels = Vec::new();
    for role in DIRECT_SUPPORTED_STATEMENT_KERNEL_ROLES {
        if let Some(kernel) = model.kernel(role) {
            match kernel.implementation {
                KernelImplementation::Statements(_) => kernels.push(kernel),
                KernelImplementation::AnalyticalBuiltin(_) => {
                    return Err(WasmError::DirectBackendUnsupported {
                        model: model.name.clone(),
                        reason: format!(
                            "direct emitter does not support non-metadata analytical builtins for {:?}",
                            role
                        ),
                    })
                }
            }
        }
    }

    if let Some(kernel) = model.kernel(KernelRole::Analytical) {
        if !matches!(kernel.implementation, KernelImplementation::AnalyticalBuiltin(_)) {
            return Err(WasmError::DirectBackendUnsupported {
                model: model.name.clone(),
                reason: "direct emitter expects analytical execution to remain metadata-driven".to_string(),
            });
        }
        if model.metadata.analytical.is_none() {
            return Err(WasmError::Emit(format!(
                "analytical model `{}` lowered an analytical kernel without analytical metadata",
                model.name
            )));
        }
    }

    Ok(kernels)
}

fn const_i32_function(value: i32) -> Function {
    let mut function = Function::new([]);
    function.instruction(&Instruction::I32Const(value));
    function.instruction(&Instruction::End);
    function
}

fn alloc_function() -> Function {
    const PARAM_LEN: u32 = 0;
    const LOCAL_OLD_PTR: u32 = 1;
    const LOCAL_BYTES: u32 = 2;
    const LOCAL_NEW_END: u32 = 3;

    let mut function = Function::new([(3, ValType::I32)]);
    function.instruction(&Instruction::LocalGet(PARAM_LEN));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Result(ValType::I32)));
    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::Else);
    function.instruction(&Instruction::GlobalGet(HEAP_PTR_GLOBAL));
    function.instruction(&Instruction::LocalSet(LOCAL_OLD_PTR));
    function.instruction(&Instruction::LocalGet(PARAM_LEN));
    function.instruction(&Instruction::I32Const(std::mem::size_of::<f64>() as i32));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::LocalSet(LOCAL_BYTES));
    function.instruction(&Instruction::LocalGet(LOCAL_OLD_PTR));
    function.instruction(&Instruction::LocalGet(LOCAL_BYTES));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(LOCAL_NEW_END));
    function.instruction(&Instruction::LocalGet(LOCAL_NEW_END));
    function.instruction(&Instruction::MemorySize(0));
    function.instruction(&Instruction::I32Const(PAGE_SIZE as i32));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32GtU);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(LOCAL_NEW_END));
    function.instruction(&Instruction::MemorySize(0));
    function.instruction(&Instruction::I32Const(PAGE_SIZE as i32));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::I32Const((PAGE_SIZE - 1) as i32));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Const(PAGE_SIZE as i32));
    function.instruction(&Instruction::I32DivU);
    function.instruction(&Instruction::MemoryGrow(0));
    function.instruction(&Instruction::I32Const(-1));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(LOCAL_NEW_END));
    function.instruction(&Instruction::GlobalSet(HEAP_PTR_GLOBAL));
    function.instruction(&Instruction::LocalGet(LOCAL_OLD_PTR));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

fn free_function() -> Function {
    let mut function = Function::new([]);
    function.instruction(&Instruction::End);
    function
}

fn emit_statement_kernel(model: &ExecutionModel, kernel: &ExecutionKernel) -> Result<Function, WasmError> {
    let program = match &kernel.implementation {
        KernelImplementation::Statements(program) => program,
        KernelImplementation::AnalyticalBuiltin(_) => {
            return Err(WasmError::DirectBackendUnsupported {
                model: model.name.clone(),
                reason: format!(
                    "direct W04 emitter does not support analytical builtins for {:?}",
                    kernel.role
                ),
            })
        }
    };

    let hidden_i64_locals = count_hidden_i64_locals_in_statements(&program.body.statements);

    let mut locals = Vec::new();
    let mut local_bindings = BTreeMap::new();
    let mut next_local_index = FIRST_WASM_LOCAL_INDEX;
    for local in &program.locals {
        locals.push((1, wasm_val_type(local.ty)));
        local_bindings.insert(
            local.index,
            WasmLocalBinding {
                wasm_local: next_local_index,
                ty: local.ty,
            },
        );
        next_local_index += 1;
    }
    for _ in 0..hidden_i64_locals {
        locals.push((1, ValType::I64));
    }

    let mut function = Function::new(locals);
    let mut state = KernelEmitState {
        model_name: &model.name,
        locals: local_bindings,
        next_hidden_i64_local: next_local_index,
    };
    emit_program(&mut state, program, &mut function)?;
    function.instruction(&Instruction::End);
    Ok(function)
}

fn emit_program(
    state: &mut KernelEmitState<'_>,
    program: &ExecutionProgram,
    function: &mut Function,
) -> Result<(), WasmError> {
    for statement in &program.body.statements {
        emit_statement(state, statement, function)?;
    }
    Ok(())
}

fn emit_statement(
    state: &mut KernelEmitState<'_>,
    statement: &ExecutionStmt,
    function: &mut Function,
) -> Result<(), WasmError> {
    match &statement.kind {
        ExecutionStmtKind::Let(binding) => {
            let local = state.local(binding.local)?;
            emit_expr(state, &binding.value, function)?;
            emit_cast_stack(binding.value.ty, local.ty, function, state.model_name)?;
            function.instruction(&Instruction::LocalSet(local.wasm_local));
            Ok(())
        }
        ExecutionStmtKind::Assign(assign) => emit_assignment(state, assign, function),
        ExecutionStmtKind::If(if_stmt) => emit_if(state, if_stmt, function),
        ExecutionStmtKind::For(for_stmt) => emit_for(state, for_stmt, function),
    }
}

fn emit_assignment(
    state: &mut KernelEmitState<'_>,
    assign: &super::execution::ExecutionAssignStmt,
    function: &mut Function,
) -> Result<(), WasmError> {
    function.instruction(&Instruction::LocalGet(KERNEL_PARAM_OUT));
    emit_target_byte_offset(state, &assign.target.kind, function)?;
    function.instruction(&Instruction::I32Add);
    emit_expr(state, &assign.value, function)?;
    emit_cast_stack(assign.value.ty, ValueType::Real, function, state.model_name)?;
    function.instruction(&Instruction::F64Store(f64_memarg()));
    Ok(())
}

fn emit_if(
    state: &mut KernelEmitState<'_>,
    if_stmt: &super::execution::ExecutionIfStmt,
    function: &mut Function,
) -> Result<(), WasmError> {
    emit_expr(state, &if_stmt.condition, function)?;
    emit_cast_stack(if_stmt.condition.ty, ValueType::Bool, function, state.model_name)?;
    function.instruction(&Instruction::If(BlockType::Empty));
    for statement in &if_stmt.then_branch {
        emit_statement(state, statement, function)?;
    }
    if let Some(else_branch) = &if_stmt.else_branch {
        function.instruction(&Instruction::Else);
        for statement in else_branch {
            emit_statement(state, statement, function)?;
        }
    }
    function.instruction(&Instruction::End);
    Ok(())
}

fn emit_for(
    state: &mut KernelEmitState<'_>,
    for_stmt: &super::execution::ExecutionForStmt,
    function: &mut Function,
) -> Result<(), WasmError> {
    let loop_local = state.local(for_stmt.local)?;
    if loop_local.ty != ValueType::Int {
        return Err(WasmError::Emit(format!(
            "direct wasm loop local {} must lower to int, found {:?}",
            for_stmt.local, loop_local.ty
        )));
    }
    let loop_end_local = state.take_hidden_i64_local();

    emit_expr(state, &for_stmt.range.start, function)?;
    emit_cast_stack(
        for_stmt.range.start.ty,
        ValueType::Int,
        function,
        state.model_name,
    )?;
    function.instruction(&Instruction::LocalSet(loop_local.wasm_local));

    emit_expr(state, &for_stmt.range.end, function)?;
    emit_cast_stack(
        for_stmt.range.end.ty,
        ValueType::Int,
        function,
        state.model_name,
    )?;
    function.instruction(&Instruction::LocalSet(loop_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(loop_local.wasm_local));
    function.instruction(&Instruction::LocalGet(loop_end_local));
    function.instruction(&Instruction::I64GeS);
    function.instruction(&Instruction::BrIf(1));

    for statement in &for_stmt.body {
        emit_statement(state, statement, function)?;
    }

    function.instruction(&Instruction::LocalGet(loop_local.wasm_local));
    function.instruction(&Instruction::I64Const(1));
    function.instruction(&Instruction::I64Add);
    function.instruction(&Instruction::LocalSet(loop_local.wasm_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    Ok(())
}

fn emit_expr(
    state: &mut KernelEmitState<'_>,
    expr: &ExecutionExpr,
    function: &mut Function,
) -> Result<(), WasmError> {
    match &expr.kind {
        ExecutionExprKind::Literal(value) => emit_literal(value, function),
        ExecutionExprKind::Load(load) => emit_load(state, load, expr.ty, function),
        ExecutionExprKind::Unary { op, expr: inner } => {
            emit_expr(state, inner, function)?;
            emit_cast_stack(inner.ty, expr.ty, function, state.model_name)?;
            match op {
                TypedUnaryOp::Plus => Ok(()),
                TypedUnaryOp::Minus => match expr.ty {
                    ValueType::Real => {
                        function.instruction(&Instruction::F64Neg);
                        Ok(())
                    }
                    ValueType::Int => {
                        function.instruction(&Instruction::I64Const(-1));
                        function.instruction(&Instruction::I64Mul);
                        Ok(())
                    }
                    ValueType::Bool => Err(WasmError::DirectBackendUnsupported {
                        model: state.model_name.to_string(),
                        reason: "cannot apply unary minus to a boolean expression".to_string(),
                    }),
                },
                TypedUnaryOp::Not => {
                    emit_cast_stack(expr.ty, ValueType::Bool, function, state.model_name)?;
                    function.instruction(&Instruction::I32Eqz);
                    Ok(())
                }
            }
        }
        ExecutionExprKind::Binary { op, lhs, rhs } => {
            emit_binary(state, *op, lhs, rhs, expr.ty, function)
        }
        ExecutionExprKind::Call { callee, args } => {
            let intrinsic = match callee {
                super::execution::ExecutionCall::Math(intrinsic) => *intrinsic,
            };
            let arg_refs = args.iter().collect::<Vec<_>>();
            emit_math_call(state, intrinsic, &arg_refs, expr.ty, function)
        }
    }
}

fn emit_binary(
    state: &mut KernelEmitState<'_>,
    op: TypedBinaryOp,
    lhs: &ExecutionExpr,
    rhs: &ExecutionExpr,
    result_ty: ValueType,
    function: &mut Function,
) -> Result<(), WasmError> {
    match op {
        TypedBinaryOp::Or => {
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, ValueType::Bool, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, ValueType::Bool, function, state.model_name)?;
            function.instruction(&Instruction::I32Or);
        }
        TypedBinaryOp::And => {
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, ValueType::Bool, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, ValueType::Bool, function, state.model_name)?;
            function.instruction(&Instruction::I32And);
        }
        TypedBinaryOp::Eq | TypedBinaryOp::NotEq => {
            let operand_ty = if lhs.ty == ValueType::Real || rhs.ty == ValueType::Real {
                ValueType::Real
            } else if lhs.ty == ValueType::Bool && rhs.ty == ValueType::Bool {
                ValueType::Bool
            } else {
                ValueType::Int
            };
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, operand_ty, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, operand_ty, function, state.model_name)?;
            match (operand_ty, op) {
                (ValueType::Real, TypedBinaryOp::Eq) => function.instruction(&Instruction::F64Eq),
                (ValueType::Real, TypedBinaryOp::NotEq) => function.instruction(&Instruction::F64Ne),
                (ValueType::Int, TypedBinaryOp::Eq) => function.instruction(&Instruction::I64Eq),
                (ValueType::Int, TypedBinaryOp::NotEq) => function.instruction(&Instruction::I64Ne),
                (ValueType::Bool, TypedBinaryOp::Eq) => function.instruction(&Instruction::I32Eq),
                (ValueType::Bool, TypedBinaryOp::NotEq) => function.instruction(&Instruction::I32Ne),
                _ => unreachable!(),
            };
        }
        TypedBinaryOp::Lt | TypedBinaryOp::LtEq | TypedBinaryOp::Gt | TypedBinaryOp::GtEq => {
            let operand_ty = if lhs.ty == ValueType::Real || rhs.ty == ValueType::Real {
                ValueType::Real
            } else {
                ValueType::Int
            };
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, operand_ty, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, operand_ty, function, state.model_name)?;
            match (operand_ty, op) {
                (ValueType::Real, TypedBinaryOp::Lt) => function.instruction(&Instruction::F64Lt),
                (ValueType::Real, TypedBinaryOp::LtEq) => function.instruction(&Instruction::F64Le),
                (ValueType::Real, TypedBinaryOp::Gt) => function.instruction(&Instruction::F64Gt),
                (ValueType::Real, TypedBinaryOp::GtEq) => function.instruction(&Instruction::F64Ge),
                (ValueType::Int, TypedBinaryOp::Lt) => function.instruction(&Instruction::I64LtS),
                (ValueType::Int, TypedBinaryOp::LtEq) => function.instruction(&Instruction::I64LeS),
                (ValueType::Int, TypedBinaryOp::Gt) => function.instruction(&Instruction::I64GtS),
                (ValueType::Int, TypedBinaryOp::GtEq) => function.instruction(&Instruction::I64GeS),
                _ => unreachable!(),
            };
        }
        TypedBinaryOp::Add | TypedBinaryOp::Sub | TypedBinaryOp::Mul => {
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, result_ty, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, result_ty, function, state.model_name)?;
            match (result_ty, op) {
                (ValueType::Real, TypedBinaryOp::Add) => function.instruction(&Instruction::F64Add),
                (ValueType::Real, TypedBinaryOp::Sub) => function.instruction(&Instruction::F64Sub),
                (ValueType::Real, TypedBinaryOp::Mul) => function.instruction(&Instruction::F64Mul),
                (ValueType::Int, TypedBinaryOp::Add) => function.instruction(&Instruction::I64Add),
                (ValueType::Int, TypedBinaryOp::Sub) => function.instruction(&Instruction::I64Sub),
                (ValueType::Int, TypedBinaryOp::Mul) => function.instruction(&Instruction::I64Mul),
                _ => {
                    return Err(WasmError::DirectBackendUnsupported {
                        model: state.model_name.to_string(),
                        reason: format!(
                            "direct W04 emitter cannot apply {:?} to {:?}",
                            op, result_ty
                        ),
                    })
                }
            };
        }
        TypedBinaryOp::Div => {
            emit_expr(state, lhs, function)?;
            emit_cast_stack(lhs.ty, ValueType::Real, function, state.model_name)?;
            emit_expr(state, rhs, function)?;
            emit_cast_stack(rhs.ty, ValueType::Real, function, state.model_name)?;
            function.instruction(&Instruction::F64Div);
        }
        TypedBinaryOp::Pow => emit_math_call(state, MathIntrinsic::Pow, &[lhs, rhs], result_ty, function)?,
    }
    Ok(())
}

fn emit_math_call(
    state: &mut KernelEmitState<'_>,
    intrinsic: MathIntrinsic,
    args: &[&ExecutionExpr],
    result_ty: ValueType,
    function: &mut Function,
) -> Result<(), WasmError> {
    match intrinsic {
        MathIntrinsic::Max | MathIntrinsic::Min => {
            if args.len() != 2 {
                return Err(WasmError::DirectBackendUnsupported {
                    model: state.model_name.to_string(),
                    reason: format!("{} expects 2 arguments", intrinsic.name()),
                });
            }
            match result_ty {
                ValueType::Real => {
                    emit_expr(state, args[0], function)?;
                    emit_cast_stack(args[0].ty, ValueType::Real, function, state.model_name)?;
                    emit_expr(state, args[1], function)?;
                    emit_cast_stack(args[1].ty, ValueType::Real, function, state.model_name)?;
                    if intrinsic == MathIntrinsic::Max {
                        function.instruction(&Instruction::F64Max);
                    } else {
                        function.instruction(&Instruction::F64Min);
                    }
                }
                ValueType::Int => {
                    let lhs_local = state.take_hidden_i64_local();
                    let rhs_local = state.take_hidden_i64_local();
                    emit_expr(state, args[0], function)?;
                    emit_cast_stack(args[0].ty, ValueType::Int, function, state.model_name)?;
                    function.instruction(&Instruction::LocalSet(lhs_local));
                    emit_expr(state, args[1], function)?;
                    emit_cast_stack(args[1].ty, ValueType::Int, function, state.model_name)?;
                    function.instruction(&Instruction::LocalSet(rhs_local));
                    function.instruction(&Instruction::LocalGet(lhs_local));
                    function.instruction(&Instruction::LocalGet(rhs_local));
                    function.instruction(&Instruction::LocalGet(lhs_local));
                    function.instruction(&Instruction::LocalGet(rhs_local));
                    if intrinsic == MathIntrinsic::Max {
                        function.instruction(&Instruction::I64GtS);
                    } else {
                        function.instruction(&Instruction::I64LtS);
                    }
                    function.instruction(&Instruction::Select);
                }
                ValueType::Bool => {
                    return Err(WasmError::DirectBackendUnsupported {
                        model: state.model_name.to_string(),
                        reason: format!("{} does not accept boolean arguments", intrinsic.name()),
                    })
                }
            }
        }
        MathIntrinsic::Abs if result_ty == ValueType::Int => {
            if args.len() != 1 {
                return Err(WasmError::DirectBackendUnsupported {
                    model: state.model_name.to_string(),
                    reason: "abs expects 1 argument".to_string(),
                });
            }
            let value_local = state.take_hidden_i64_local();
            emit_expr(state, args[0], function)?;
            emit_cast_stack(args[0].ty, ValueType::Int, function, state.model_name)?;
            function.instruction(&Instruction::LocalSet(value_local));
            function.instruction(&Instruction::LocalGet(value_local));
            function.instruction(&Instruction::I64Const(-1));
            function.instruction(&Instruction::I64Mul);
            function.instruction(&Instruction::LocalGet(value_local));
            function.instruction(&Instruction::LocalGet(value_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::Select);
        }
        _ => {
            let expected_arity = if intrinsic == MathIntrinsic::Pow { 2 } else { 1 };
            if args.len() != expected_arity {
                return Err(WasmError::DirectBackendUnsupported {
                    model: state.model_name.to_string(),
                    reason: format!("{} expects {} arguments", intrinsic.name(), expected_arity),
                });
            }
            for arg in args {
                emit_expr(state, arg, function)?;
                emit_cast_stack(arg.ty, ValueType::Real, function, state.model_name)?;
            }
            match intrinsic {
                MathIntrinsic::Abs => function.instruction(&Instruction::F64Abs),
                MathIntrinsic::Ceil => function.instruction(&Instruction::F64Ceil),
                MathIntrinsic::Exp => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Exp)?)),
                MathIntrinsic::Floor => function.instruction(&Instruction::F64Floor),
                MathIntrinsic::Ln | MathIntrinsic::Log => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Ln)?)),
                MathIntrinsic::Log10 => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Log10)?)),
                MathIntrinsic::Log2 => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Log2)?)),
                MathIntrinsic::Pow => function.instruction(&Instruction::Call(binary_math_import_index(MathIntrinsic::Pow)?)),
                MathIntrinsic::Round => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Round)?)),
                MathIntrinsic::Sin => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Sin)?)),
                MathIntrinsic::Cos => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Cos)?)),
                MathIntrinsic::Tan => function.instruction(&Instruction::Call(unary_math_import_index(MathIntrinsic::Tan)?)),
                MathIntrinsic::Sqrt => function.instruction(&Instruction::F64Sqrt),
                MathIntrinsic::Max | MathIntrinsic::Min => unreachable!(),
            };
            emit_cast_stack(ValueType::Real, result_ty, function, state.model_name)?;
        }
    }
    Ok(())
}

fn emit_literal(value: &ConstValue, function: &mut Function) -> Result<(), WasmError> {
    match value {
        ConstValue::Int(value) => function.instruction(&Instruction::I64Const(*value)),
        ConstValue::Real(value) => function.instruction(&Instruction::F64Const((*value).into())),
        ConstValue::Bool(value) => function.instruction(&Instruction::I32Const(i32::from(*value))),
    };
    Ok(())
}

fn emit_load(
    state: &mut KernelEmitState<'_>,
    load: &ExecutionLoad,
    target_ty: ValueType,
    function: &mut Function,
) -> Result<(), WasmError> {
    match load {
        ExecutionLoad::Parameter(index) => {
            emit_dense_load(function, KERNEL_PARAM_PARAMS, *index, target_ty, state.model_name)
        }
        ExecutionLoad::Covariate(index) => emit_dense_load(
            function,
            KERNEL_PARAM_COVARIATES,
            *index,
            target_ty,
            state.model_name,
        ),
        ExecutionLoad::Derived(index) => {
            emit_dense_load(function, KERNEL_PARAM_DERIVED, *index, target_ty, state.model_name)
        }
        ExecutionLoad::Local(index) => {
            let local = state.local(*index)?;
            function.instruction(&Instruction::LocalGet(local.wasm_local));
            emit_cast_stack(local.ty, target_ty, function, state.model_name)
        }
        ExecutionLoad::RouteInput(index) => {
            emit_dense_load(function, KERNEL_PARAM_ROUTES, *index, target_ty, state.model_name)
        }
        ExecutionLoad::State(state_ref) => {
            emit_state_load(state, KERNEL_PARAM_STATES, state_ref, target_ty, function)
        }
    }
}

fn emit_dense_load(
    function: &mut Function,
    base_ptr_local: u32,
    slot: usize,
    target_ty: ValueType,
    model_name: &str,
) -> Result<(), WasmError> {
    function.instruction(&Instruction::LocalGet(base_ptr_local));
    function.instruction(&Instruction::I32Const(byte_offset(slot)?));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::F64Load(f64_memarg()));
    emit_cast_stack(ValueType::Real, target_ty, function, model_name)
}

fn emit_state_load(
    state: &mut KernelEmitState<'_>,
    base_ptr_local: u32,
    state_ref: &ExecutionStateRef,
    target_ty: ValueType,
    function: &mut Function,
) -> Result<(), WasmError> {
    function.instruction(&Instruction::LocalGet(base_ptr_local));
    emit_state_ref_byte_offset(state, state_ref, function)?;
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::F64Load(f64_memarg()));
    emit_cast_stack(ValueType::Real, target_ty, function, state.model_name)
}

fn emit_target_byte_offset(
    state: &mut KernelEmitState<'_>,
    target: &ExecutionTargetKind,
    function: &mut Function,
) -> Result<(), WasmError> {
    match target {
        ExecutionTargetKind::Derived(index)
        | ExecutionTargetKind::Output(index)
        | ExecutionTargetKind::RouteLag(index)
        | ExecutionTargetKind::RouteBioavailability(index) => {
            function.instruction(&Instruction::I32Const(byte_offset(*index)?));
            Ok(())
        }
        ExecutionTargetKind::StateInit(state_ref)
        | ExecutionTargetKind::StateDerivative(state_ref)
        | ExecutionTargetKind::StateNoise(state_ref) => {
            emit_state_ref_byte_offset(state, state_ref, function)
        }
    }
}

fn emit_state_ref_byte_offset(
    state: &mut KernelEmitState<'_>,
    state_ref: &ExecutionStateRef,
    function: &mut Function,
) -> Result<(), WasmError> {
    if let Some(index_expr) = &state_ref.index {
        emit_expr(state, index_expr, function)?;
        emit_cast_stack(index_expr.ty, ValueType::Int, function, state.model_name)?;
        function.instruction(&Instruction::I32WrapI64);
        function.instruction(&Instruction::I32Const(std::mem::size_of::<f64>() as i32));
        function.instruction(&Instruction::I32Mul);
        if state_ref.base_offset != 0 {
            function.instruction(&Instruction::I32Const(byte_offset(state_ref.base_offset)?));
            function.instruction(&Instruction::I32Add);
        }
    } else {
        function.instruction(&Instruction::I32Const(byte_offset(state_ref.base_offset)?));
    }
    Ok(())
}

fn emit_cast_stack(
    from: ValueType,
    to: ValueType,
    function: &mut Function,
    model_name: &str,
) -> Result<(), WasmError> {
    if from == to {
        return Ok(());
    }

    match (from, to) {
        (ValueType::Int, ValueType::Real) => {
            function.instruction(&Instruction::F64ConvertI64S);
        }
        (ValueType::Bool, ValueType::Real) => {
            function.instruction(&Instruction::I64ExtendI32U);
            function.instruction(&Instruction::F64ConvertI64S);
        }
        (ValueType::Real, ValueType::Int) => {
            function.instruction(&Instruction::I64TruncSatF64S);
        }
        (ValueType::Bool, ValueType::Int) => {
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (ValueType::Int, ValueType::Bool) => {
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64Ne);
        }
        (ValueType::Real, ValueType::Bool) => {
            function.instruction(&Instruction::F64Const(0.0.into()));
            function.instruction(&Instruction::F64Ne);
        }
        _ => {
            return Err(WasmError::DirectBackendUnsupported {
                model: model_name.to_string(),
                reason: format!("unsupported direct wasm cast from {:?} to {:?}", from, to),
            })
        }
    };
    Ok(())
}

fn byte_offset(index: usize) -> Result<i32, WasmError> {
    let offset = index
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| WasmError::Emit("direct backend byte offset overflow".to_string()))?;
    i32::try_from(offset).map_err(|_| WasmError::Emit("direct backend byte offset overflow".to_string()))
}

fn wasm_val_type(ty: ValueType) -> ValType {
    match ty {
        ValueType::Real => ValType::F64,
        ValueType::Int => ValType::I64,
        ValueType::Bool => ValType::I32,
    }
}

fn count_hidden_i64_locals_in_statements(statements: &[ExecutionStmt]) -> usize {
    statements.iter().map(count_hidden_i64_locals_in_statement).sum()
}

fn count_hidden_i64_locals_in_statement(statement: &ExecutionStmt) -> usize {
    match &statement.kind {
        ExecutionStmtKind::Let(binding) => count_hidden_i64_locals_in_expr(&binding.value),
        ExecutionStmtKind::Assign(assign) => count_hidden_i64_locals_in_expr(&assign.value),
        ExecutionStmtKind::If(if_stmt) => {
            count_hidden_i64_locals_in_expr(&if_stmt.condition)
                + count_hidden_i64_locals_in_statements(&if_stmt.then_branch)
                + if_stmt
                    .else_branch
                    .as_ref()
                    .map_or(0, |branch| count_hidden_i64_locals_in_statements(branch))
        }
        ExecutionStmtKind::For(for_stmt) => {
            1 + count_hidden_i64_locals_in_expr(&for_stmt.range.start)
                + count_hidden_i64_locals_in_expr(&for_stmt.range.end)
                + count_hidden_i64_locals_in_statements(&for_stmt.body)
        }
    }
}

fn count_hidden_i64_locals_in_expr(expr: &ExecutionExpr) -> usize {
    match &expr.kind {
        ExecutionExprKind::Literal(_) | ExecutionExprKind::Load(_) => 0,
        ExecutionExprKind::Unary { expr: inner, .. } => count_hidden_i64_locals_in_expr(inner),
        ExecutionExprKind::Binary { lhs, rhs, .. } => {
            count_hidden_i64_locals_in_expr(lhs) + count_hidden_i64_locals_in_expr(rhs)
        }
        ExecutionExprKind::Call { callee, args } => {
            let arg_cost = args.iter().map(count_hidden_i64_locals_in_expr).sum::<usize>();
            let local_cost = match callee {
                super::execution::ExecutionCall::Math(MathIntrinsic::Max | MathIntrinsic::Min)
                    if expr.ty == ValueType::Int =>
                {
                    2
                }
                super::execution::ExecutionCall::Math(MathIntrinsic::Abs) if expr.ty == ValueType::Int => 1,
                _ => 0,
            };
            arg_cost + local_cost
        }
    }
}

fn direct_wasm_import_count() -> usize {
    DIRECT_WASM_UNARY_MATH_IMPORTS.len() + DIRECT_WASM_BINARY_MATH_IMPORTS.len()
}

fn unary_math_import_index(intrinsic: MathIntrinsic) -> Result<u32, WasmError> {
    DIRECT_WASM_UNARY_MATH_IMPORTS
        .iter()
        .position(|import| import.intrinsic == intrinsic)
        .map(|index| index as u32)
        .ok_or_else(|| WasmError::Emit(format!("missing unary direct wasm import for {intrinsic:?}")))
}

fn binary_math_import_index(intrinsic: MathIntrinsic) -> Result<u32, WasmError> {
    DIRECT_WASM_BINARY_MATH_IMPORTS
        .iter()
        .position(|import| import.intrinsic == intrinsic)
        .map(|index| DIRECT_WASM_UNARY_MATH_IMPORTS.len() as u32 + index as u32)
        .ok_or_else(|| WasmError::Emit(format!("missing binary direct wasm import for {intrinsic:?}")))
}

fn align_to_f64_boundary(len: usize) -> Result<i32, WasmError> {
    let aligned = len
        .checked_add(ABI_PTR_ALIGNMENT - 1)
        .and_then(|value| value.checked_div(ABI_PTR_ALIGNMENT))
        .and_then(|value| value.checked_mul(ABI_PTR_ALIGNMENT))
        .ok_or_else(|| WasmError::Emit("direct backend heap alignment overflow".to_string()))?;
    i32::try_from(aligned).map_err(|_| WasmError::Emit("direct backend heap alignment overflow".to_string()))
}

fn pages_for_bytes(bytes: usize) -> usize {
    bytes.div_ceil(PAGE_SIZE)
}

fn f64_memarg() -> MemArg {
    MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }
}

impl KernelEmitState<'_> {
    fn local(&self, index: usize) -> Result<WasmLocalBinding, WasmError> {
        self.locals.get(&index).copied().ok_or_else(|| {
            WasmError::Emit(format!("unknown direct wasm local slot {index}"))
        })
    }

    fn take_hidden_i64_local(&mut self) -> u32 {
        let local = self.next_hidden_i64_local;
        self.next_hidden_i64_local += 1;
        local
    }
}

#[cfg(test)]
pub(crate) fn w03_minimal_outputs_execution_model() -> ExecutionModel {
    use super::execution::{
        BufferKind, BufferSlot, CallingConvention, DenseBufferLayout, ExecutionAbi, ExecutionBlock,
        ExecutionMetadata, ExecutionProgram, ExecutionSlot, ExecutionState, ExecutionStateRef,
        ExecutionTarget, KernelAccess, KernelArgument, KernelArgumentKind, KernelSignature,
        ScalarAbi,
    };
    use super::Span;

    let span = Span::empty(0);
    let parameter_slots = vec![
        ExecutionSlot {
            symbol: 0,
            name: "cl".to_string(),
            index: 0,
            span,
        },
        ExecutionSlot {
            symbol: 1,
            name: "v".to_string(),
            index: 1,
            span,
        },
    ];
    let state_slots = vec![ExecutionState {
        symbol: 2,
        name: "central".to_string(),
        offset: 0,
        len: 1,
        span,
    }];
    let output_slots = vec![ExecutionSlot {
        symbol: 3,
        name: "cp".to_string(),
        index: 0,
        span,
    }];

    ExecutionModel {
        name: "direct_w03_minimal".to_string(),
        kind: ModelKind::Ode,
        metadata: ExecutionMetadata {
            constants: vec![],
            parameters: parameter_slots.clone(),
            covariates: vec![],
            states: state_slots.clone(),
            routes: vec![],
            derived: vec![],
            outputs: output_slots.clone(),
            particles: None,
            analytical: None,
        },
        abi: ExecutionAbi {
            scalar: ScalarAbi::F64,
            calling_convention: CallingConvention::DenseF64Buffers,
            parameter_buffer: DenseBufferLayout {
                kind: BufferKind::Parameters,
                len: 2,
                slots: vec![
                    BufferSlot {
                        name: "cl".to_string(),
                        offset: 0,
                        len: 1,
                    },
                    BufferSlot {
                        name: "v".to_string(),
                        offset: 1,
                        len: 1,
                    },
                ],
            },
            covariate_buffer: DenseBufferLayout {
                kind: BufferKind::Covariates,
                len: 0,
                slots: vec![],
            },
            state_buffer: DenseBufferLayout {
                kind: BufferKind::States,
                len: 1,
                slots: vec![BufferSlot {
                    name: "central".to_string(),
                    offset: 0,
                    len: 1,
                }],
            },
            derived_buffer: DenseBufferLayout {
                kind: BufferKind::Derived,
                len: 0,
                slots: vec![],
            },
            output_buffer: DenseBufferLayout {
                kind: BufferKind::Outputs,
                len: 1,
                slots: vec![BufferSlot {
                    name: "cp".to_string(),
                    offset: 0,
                    len: 1,
                }],
            },
            route_buffer: DenseBufferLayout {
                kind: BufferKind::Routes,
                len: 0,
                slots: vec![],
            },
        },
        kernels: vec![ExecutionKernel {
            role: KernelRole::Outputs,
            signature: KernelSignature {
                args: vec![
                    KernelArgument {
                        kind: KernelArgumentKind::Time,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::States,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::Parameters,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::Covariates,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::RouteInputs,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::Derived,
                        access: KernelAccess::Input,
                    },
                    KernelArgument {
                        kind: KernelArgumentKind::Outputs,
                        access: KernelAccess::Output,
                    },
                ],
            },
            implementation: KernelImplementation::Statements(ExecutionProgram {
                locals: vec![],
                body: ExecutionBlock {
                    statements: vec![ExecutionStmt {
                        kind: ExecutionStmtKind::Assign(super::execution::ExecutionAssignStmt {
                            target: ExecutionTarget {
                                kind: ExecutionTargetKind::Output(0),
                                span,
                            },
                            value: ExecutionExpr {
                                kind: ExecutionExprKind::Binary {
                                    op: TypedBinaryOp::Add,
                                    lhs: Box::new(ExecutionExpr {
                                        kind: ExecutionExprKind::Load(ExecutionLoad::State(
                                            ExecutionStateRef {
                                                symbol: 2,
                                                base_offset: 0,
                                                len: 1,
                                                index: None,
                                                span,
                                            },
                                        )),
                                        ty: ValueType::Real,
                                        constant: None,
                                        span,
                                    }),
                                    rhs: Box::new(ExecutionExpr {
                                        kind: ExecutionExprKind::Binary {
                                            op: TypedBinaryOp::Div,
                                            lhs: Box::new(ExecutionExpr {
                                                kind: ExecutionExprKind::Binary {
                                                    op: TypedBinaryOp::Add,
                                                    lhs: Box::new(ExecutionExpr {
                                                        kind: ExecutionExprKind::Load(
                                                            ExecutionLoad::Parameter(0),
                                                        ),
                                                        ty: ValueType::Real,
                                                        constant: None,
                                                        span,
                                                    }),
                                                    rhs: Box::new(ExecutionExpr {
                                                        kind: ExecutionExprKind::Load(
                                                            ExecutionLoad::Parameter(1),
                                                        ),
                                                        ty: ValueType::Real,
                                                        constant: None,
                                                        span,
                                                    }),
                                                },
                                                ty: ValueType::Real,
                                                constant: None,
                                                span,
                                            }),
                                            rhs: Box::new(ExecutionExpr {
                                                kind: ExecutionExprKind::Literal(ConstValue::Real(2.0)),
                                                ty: ValueType::Real,
                                                constant: Some(ConstValue::Real(2.0)),
                                                span,
                                            }),
                                        },
                                        ty: ValueType::Real,
                                        constant: None,
                                        span,
                                    }),
                                },
                                ty: ValueType::Real,
                                constant: None,
                                span,
                            },
                        }),
                        span,
                    }],
                    span,
                },
            }),
            span,
        }],
        span,
    }
}

#[cfg(all(test, feature = "dsl-wasm"))]
mod tests {
    use super::*;
    use wasmtime::{Engine, Module as WasmtimeModule, Store};

    #[test]
    fn direct_emitter_builds_valid_outputs_only_wasm_module() {
        let model = w03_minimal_outputs_execution_model();
        let bytes = compile_execution_model_to_wasm_bytes(&model, 1).expect("emit direct wasm bytes");

        let engine = Engine::default();
        let module = WasmtimeModule::new(&engine, &bytes).expect("compile wasm bytes");
        let mut store = Store::new(&engine, ());
        let linker = super::super::wasm::configured_wasm_linker(&engine)
            .expect("configured wasm linker");
        linker
            .instantiate(&mut store, &module)
            .expect("instantiate direct wasm module");
    }

    #[test]
    fn direct_emitter_compiles_real_ode_corpus_model() {
        let source = include_str!("../../dsl-proposals/02-structured-block-imperative.dsl");
        let parsed = crate::dsl::parse_module(source).expect("parse proposal source");
        let typed = crate::dsl::analyze_module(&parsed).expect("analyze proposal source");
        let model = typed
            .models
            .iter()
            .find(|model| model.name == "one_cmt_oral_iv")
            .expect("ode corpus model");
        let execution = crate::dsl::lower_typed_model(model).expect("lower proposal model");

        let bytes = compile_execution_model_to_wasm_bytes(&execution, 1)
            .expect("emit direct ode wasm bytes");
        let engine = Engine::default();
        let module = WasmtimeModule::new(&engine, &bytes).expect("compile direct ode wasm bytes");
        let mut store = Store::new(&engine, ());
        let linker = super::super::wasm::configured_wasm_linker(&engine)
            .expect("configured wasm linker");
        linker
            .instantiate(&mut store, &module)
            .expect("instantiate direct ode wasm module");
    }
}