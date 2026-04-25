use std::fmt::Write;

use super::execution::{
    ExecutionBlock, ExecutionCall, ExecutionExpr, ExecutionExprKind, ExecutionLoad, ExecutionModel,
    ExecutionProgram, ExecutionStateRef, ExecutionStmt, ExecutionStmtKind, ExecutionTargetKind,
    KernelImplementation, KernelRole,
};
use super::native::NativeModelInfo;
use super::{MathIntrinsic, TypedBinaryOp, TypedUnaryOp, ValueType};

pub const API_VERSION_SYMBOL: &str = "pharmsol_dsl_api_version";
pub const MODEL_INFO_JSON_PTR_SYMBOL: &str = "pharmsol_dsl_model_info_json_ptr";
pub const MODEL_INFO_JSON_LEN_SYMBOL: &str = "pharmsol_dsl_model_info_json_len";
pub const DERIVE_SYMBOL: &str = "pharmsol_dsl_kernel_derive";
pub const DYNAMICS_SYMBOL: &str = "pharmsol_dsl_kernel_dynamics";
pub const OUTPUTS_SYMBOL: &str = "pharmsol_dsl_kernel_outputs";
pub const INIT_SYMBOL: &str = "pharmsol_dsl_kernel_init";
pub const DRIFT_SYMBOL: &str = "pharmsol_dsl_kernel_drift";
pub const DIFFUSION_SYMBOL: &str = "pharmsol_dsl_kernel_diffusion";
pub const ROUTE_LAG_SYMBOL: &str = "pharmsol_dsl_kernel_route_lag";
pub const ROUTE_BIOAVAILABILITY_SYMBOL: &str = "pharmsol_dsl_kernel_route_bioavailability";
pub const ALLOC_F64_BUFFER_SYMBOL: &str = "pharmsol_dsl_alloc_f64_buffer";
pub const FREE_F64_BUFFER_SYMBOL: &str = "pharmsol_dsl_free_f64_buffer";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustBackendFlavor {
    #[cfg(feature = "dsl-aot")]
    NativeAot { api_version: u32 },
    #[cfg(feature = "dsl-wasm")]
    Wasm { api_version: u32 },
}

impl RustBackendFlavor {
    fn api_version(self) -> u32 {
        match self {
            #[cfg(all(feature = "dsl-aot", feature = "dsl-wasm"))]
            Self::NativeAot { api_version } | Self::Wasm { api_version } => api_version,
            #[cfg(all(feature = "dsl-aot", not(feature = "dsl-wasm")))]
            Self::NativeAot { api_version } => api_version,
            #[cfg(all(not(feature = "dsl-aot"), feature = "dsl-wasm"))]
            Self::Wasm { api_version } => api_version,
        }
    }

    fn emits_wasm_allocators(self) -> bool {
        #[cfg(feature = "dsl-wasm")]
        {
            matches!(self, Self::Wasm { .. })
        }
        #[cfg(not(feature = "dsl-wasm"))]
        {
            let _ = self;
            false
        }
    }
}

pub fn emit_rust_backend_source(
    model: &ExecutionModel,
    flavor: RustBackendFlavor,
) -> Result<String, String> {
    let model_info = NativeModelInfo::from_execution_model(model);
    let model_info_json = serde_json::to_string(&model_info).map_err(|error| error.to_string())?;

    let mut source = String::new();
    writeln!(source, "#![allow(dead_code)]").unwrap();
    writeln!(source, "#![allow(unused_mut)]").unwrap();
    writeln!(source, "#![allow(unused_variables)]").unwrap();
    writeln!(source).unwrap();
    writeln!(
        source,
        "const MODEL_INFO_JSON: &str = {:?};",
        model_info_json
    )
    .unwrap();
    writeln!(
        source,
        "const PHARMSOL_DSL_API_VERSION: u32 = {};",
        flavor.api_version()
    )
    .unwrap();
    writeln!(source).unwrap();
    writeln!(source, "#[inline]").unwrap();
    writeln!(
        source,
        "unsafe fn load_f64(ptr: *const f64, index: usize) -> f64 {{ *ptr.add(index) }}"
    )
    .unwrap();
    writeln!(source, "#[inline]").unwrap();
    writeln!(source, "unsafe fn store_f64(ptr: *mut f64, index: usize, value: f64) {{ *ptr.add(index) = value; }}").unwrap();
    writeln!(source).unwrap();
    writeln!(source, "#[no_mangle]").unwrap();
    writeln!(
        source,
        "pub extern \"C\" fn {API_VERSION_SYMBOL}() -> u32 {{ PHARMSOL_DSL_API_VERSION }}"
    )
    .unwrap();
    writeln!(source, "#[no_mangle]").unwrap();
    writeln!(source, "pub extern \"C\" fn {MODEL_INFO_JSON_PTR_SYMBOL}() -> *const u8 {{ MODEL_INFO_JSON.as_ptr() }}").unwrap();
    writeln!(source, "#[no_mangle]").unwrap();
    writeln!(
        source,
        "pub extern \"C\" fn {MODEL_INFO_JSON_LEN_SYMBOL}() -> usize {{ MODEL_INFO_JSON.len() }}"
    )
    .unwrap();
    writeln!(source).unwrap();

    if flavor.emits_wasm_allocators() {
        writeln!(source, "#[no_mangle]").unwrap();
        writeln!(
            source,
            "pub extern \"C\" fn {ALLOC_F64_BUFFER_SYMBOL}(len: usize) -> *mut f64 {{"
        )
        .unwrap();
        writeln!(source, "    if len == 0 {{").unwrap();
        writeln!(source, "        return core::ptr::null_mut();").unwrap();
        writeln!(source, "    }}").unwrap();
        writeln!(
            source,
            "    let mut buffer = Vec::<f64>::with_capacity(len);"
        )
        .unwrap();
        writeln!(source, "    let ptr = buffer.as_mut_ptr();").unwrap();
        writeln!(source, "    core::mem::forget(buffer);").unwrap();
        writeln!(source, "    ptr").unwrap();
        writeln!(source, "}}").unwrap();
        writeln!(source).unwrap();
        writeln!(source, "#[no_mangle]").unwrap();
        writeln!(
            source,
            "pub unsafe extern \"C\" fn {FREE_F64_BUFFER_SYMBOL}(ptr: *mut f64, len: usize) {{"
        )
        .unwrap();
        writeln!(source, "    if ptr.is_null() || len == 0 {{").unwrap();
        writeln!(source, "        return;").unwrap();
        writeln!(source, "    }}").unwrap();
        writeln!(
            source,
            "    drop(Vec::<f64>::from_raw_parts(ptr, len, len));"
        )
        .unwrap();
        writeln!(source, "}}").unwrap();
        writeln!(source).unwrap();
    }

    for kernel in &model.kernels {
        if let Some(symbol) = kernel_symbol_name(kernel.role) {
            if let KernelImplementation::Statements(program) = &kernel.implementation {
                emit_statement_kernel(&mut source, program, symbol)?;
                writeln!(source).unwrap();
            }
        }
    }

    Ok(source)
}

fn kernel_symbol_name(role: KernelRole) -> Option<&'static str> {
    match role {
        KernelRole::Derive => Some(DERIVE_SYMBOL),
        KernelRole::Dynamics => Some(DYNAMICS_SYMBOL),
        KernelRole::Outputs => Some(OUTPUTS_SYMBOL),
        KernelRole::Init => Some(INIT_SYMBOL),
        KernelRole::Drift => Some(DRIFT_SYMBOL),
        KernelRole::Diffusion => Some(DIFFUSION_SYMBOL),
        KernelRole::RouteLag => Some(ROUTE_LAG_SYMBOL),
        KernelRole::RouteBioavailability => Some(ROUTE_BIOAVAILABILITY_SYMBOL),
        KernelRole::Analytical => None,
    }
}

fn emit_statement_kernel(
    source: &mut String,
    program: &ExecutionProgram,
    symbol: &'static str,
) -> Result<(), String> {
    writeln!(source, "#[no_mangle]").unwrap();
    writeln!(source, "pub unsafe extern \"C\" fn {symbol}(").unwrap();
    writeln!(source, "    t: f64,").unwrap();
    writeln!(source, "    states: *const f64,").unwrap();
    writeln!(source, "    params: *const f64,").unwrap();
    writeln!(source, "    covariates: *const f64,").unwrap();
    writeln!(source, "    routes: *const f64,").unwrap();
    writeln!(source, "    derived: *const f64,").unwrap();
    writeln!(source, "    out: *mut f64,").unwrap();
    writeln!(source, ") {{").unwrap();

    for local in &program.locals {
        writeln!(
            source,
            "    let mut local_{}: {} = {};",
            local.index,
            rust_type(local.ty),
            rust_zero(local.ty)
        )
        .unwrap();
    }

    emit_block(source, &program.body, 1)?;
    writeln!(source, "}}").unwrap();
    Ok(())
}

fn emit_block(source: &mut String, block: &ExecutionBlock, indent: usize) -> Result<(), String> {
    for statement in &block.statements {
        emit_stmt(source, statement, indent)?;
    }
    Ok(())
}

fn emit_stmt(source: &mut String, statement: &ExecutionStmt, indent: usize) -> Result<(), String> {
    match &statement.kind {
        ExecutionStmtKind::Let(let_stmt) => {
            let value = emit_expr(&let_stmt.value)?;
            push_line(
                source,
                indent,
                &format!("local_{} = {};", let_stmt.local, value.rendered),
            );
        }
        ExecutionStmtKind::Assign(assign) => {
            let value = emit_expr(&assign.value)?;
            let value = cast_expr(value.rendered, value.ty, ValueType::Real);
            let target = emit_target(&assign.target.kind)?;
            push_line(
                source,
                indent,
                &format!("unsafe {{ store_f64(out, {target}, {value}); }}"),
            );
        }
        ExecutionStmtKind::If(if_stmt) => {
            let condition = emit_expr(&if_stmt.condition)?;
            let condition = cast_expr(condition.rendered, condition.ty, ValueType::Bool);
            push_line(source, indent, &format!("if {condition} {{"));
            for nested in &if_stmt.then_branch {
                emit_stmt(source, nested, indent + 1)?;
            }
            if let Some(else_branch) = &if_stmt.else_branch {
                push_line(source, indent, "} else {");
                for nested in else_branch {
                    emit_stmt(source, nested, indent + 1)?;
                }
            }
            push_line(source, indent, "}");
        }
        ExecutionStmtKind::For(for_stmt) => {
            let start = emit_expr(&for_stmt.range.start)?;
            let end = emit_expr(&for_stmt.range.end)?;
            let start = cast_expr(start.rendered, start.ty, ValueType::Int);
            let end = cast_expr(end.rendered, end.ty, ValueType::Int);
            push_line(
                source,
                indent,
                &format!(
                    "for __loop_local_{} in ({start})..({end}) {{",
                    for_stmt.local
                ),
            );
            push_line(
                source,
                indent + 1,
                &format!(
                    "local_{} = __loop_local_{};",
                    for_stmt.local, for_stmt.local
                ),
            );
            for nested in &for_stmt.body {
                emit_stmt(source, nested, indent + 1)?;
            }
            push_line(source, indent, "}");
        }
    }
    Ok(())
}

fn emit_target(target: &ExecutionTargetKind) -> Result<String, String> {
    Ok(match target {
        ExecutionTargetKind::Derived(index)
        | ExecutionTargetKind::Output(index)
        | ExecutionTargetKind::RouteLag(index)
        | ExecutionTargetKind::RouteBioavailability(index) => index.to_string(),
        ExecutionTargetKind::StateInit(state)
        | ExecutionTargetKind::StateDerivative(state)
        | ExecutionTargetKind::StateNoise(state) => emit_state_ref_index(state)?,
    })
}

fn emit_state_ref_index(state: &ExecutionStateRef) -> Result<String, String> {
    Ok(match &state.index {
        Some(index) => {
            let index = emit_expr(index)?;
            let index = cast_expr(index.rendered, index.ty, ValueType::Int);
            format!("{} + ({index} as usize)", state.base_offset)
        }
        None => state.base_offset.to_string(),
    })
}

#[derive(Debug, Clone)]
struct RenderedExpr {
    rendered: String,
    ty: ValueType,
}

fn emit_expr(expr: &ExecutionExpr) -> Result<RenderedExpr, String> {
    let rendered = match &expr.kind {
        ExecutionExprKind::Literal(value) => match value {
            super::ConstValue::Int(value) => format!("{value}i64"),
            super::ConstValue::Real(value) => format!("{value:?}"),
            super::ConstValue::Bool(value) => value.to_string(),
        },
        ExecutionExprKind::Load(load) => emit_load(load, expr.ty)?,
        ExecutionExprKind::Unary { op, expr: inner } => {
            let inner = emit_expr(inner)?;
            match op {
                TypedUnaryOp::Plus => cast_expr(inner.rendered, inner.ty, expr.ty),
                TypedUnaryOp::Minus => match expr.ty {
                    ValueType::Real | ValueType::Int => {
                        format!("-({})", cast_expr(inner.rendered, inner.ty, expr.ty))
                    }
                    ValueType::Bool => {
                        return Err("cannot emit unary minus for boolean expressions".to_string())
                    }
                },
                TypedUnaryOp::Not => {
                    format!(
                        "!({})",
                        cast_expr(inner.rendered, inner.ty, ValueType::Bool)
                    )
                }
            }
        }
        ExecutionExprKind::Binary { op, lhs, rhs } => emit_binary_expr(*op, lhs, rhs, expr.ty)?,
        ExecutionExprKind::Call { callee, args } => emit_call_expr(callee, args, expr.ty)?,
    };

    Ok(RenderedExpr {
        rendered,
        ty: expr.ty,
    })
}

fn emit_load(load: &ExecutionLoad, ty: ValueType) -> Result<String, String> {
    let raw = match load {
        ExecutionLoad::Parameter(index) => format!("load_f64(params, {index})"),
        ExecutionLoad::Covariate(index) => format!("load_f64(covariates, {index})"),
        ExecutionLoad::Derived(index) => format!("load_f64(derived, {index})"),
        ExecutionLoad::Local(index) => return Ok(format!("local_{index}")),
        ExecutionLoad::RouteInput(index) => format!("load_f64(routes, {index})"),
        ExecutionLoad::State(state) => {
            let index = emit_state_ref_index(state)?;
            format!("load_f64(states, {index})")
        }
    };
    Ok(cast_expr(raw, ValueType::Real, ty))
}

fn emit_binary_expr(
    op: TypedBinaryOp,
    lhs: &ExecutionExpr,
    rhs: &ExecutionExpr,
    result_ty: ValueType,
) -> Result<String, String> {
    let lhs = emit_expr(lhs)?;
    let rhs = emit_expr(rhs)?;
    Ok(match op {
        TypedBinaryOp::Or => format!(
            "({}) || ({})",
            cast_expr(lhs.rendered, lhs.ty, ValueType::Bool),
            cast_expr(rhs.rendered, rhs.ty, ValueType::Bool)
        ),
        TypedBinaryOp::And => format!(
            "({}) && ({})",
            cast_expr(lhs.rendered, lhs.ty, ValueType::Bool),
            cast_expr(rhs.rendered, rhs.ty, ValueType::Bool)
        ),
        TypedBinaryOp::Eq | TypedBinaryOp::NotEq => {
            let operand_ty = if lhs.ty == ValueType::Real || rhs.ty == ValueType::Real {
                ValueType::Real
            } else if lhs.ty == ValueType::Bool && rhs.ty == ValueType::Bool {
                ValueType::Bool
            } else {
                ValueType::Int
            };
            let operator = if op == TypedBinaryOp::Eq { "==" } else { "!=" };
            format!(
                "({}) {operator} ({})",
                cast_expr(lhs.rendered, lhs.ty, operand_ty),
                cast_expr(rhs.rendered, rhs.ty, operand_ty)
            )
        }
        TypedBinaryOp::Lt | TypedBinaryOp::LtEq | TypedBinaryOp::Gt | TypedBinaryOp::GtEq => {
            let operand_ty = if lhs.ty == ValueType::Real || rhs.ty == ValueType::Real {
                ValueType::Real
            } else {
                ValueType::Int
            };
            let operator = match op {
                TypedBinaryOp::Lt => "<",
                TypedBinaryOp::LtEq => "<=",
                TypedBinaryOp::Gt => ">",
                TypedBinaryOp::GtEq => ">=",
                _ => unreachable!(),
            };
            format!(
                "({}) {operator} ({})",
                cast_expr(lhs.rendered, lhs.ty, operand_ty),
                cast_expr(rhs.rendered, rhs.ty, operand_ty)
            )
        }
        TypedBinaryOp::Add | TypedBinaryOp::Sub | TypedBinaryOp::Mul => {
            let operator = match op {
                TypedBinaryOp::Add => "+",
                TypedBinaryOp::Sub => "-",
                TypedBinaryOp::Mul => "*",
                _ => unreachable!(),
            };
            format!(
                "({}) {operator} ({})",
                cast_expr(lhs.rendered, lhs.ty, result_ty),
                cast_expr(rhs.rendered, rhs.ty, result_ty)
            )
        }
        TypedBinaryOp::Div => format!(
            "({}) / ({})",
            cast_expr(lhs.rendered, lhs.ty, ValueType::Real),
            cast_expr(rhs.rendered, rhs.ty, ValueType::Real)
        ),
        TypedBinaryOp::Pow => {
            let lhs = cast_expr(lhs.rendered, lhs.ty, ValueType::Real);
            let rhs = cast_expr(rhs.rendered, rhs.ty, ValueType::Real);
            cast_expr(format!("({lhs}).powf({rhs})"), ValueType::Real, result_ty)
        }
    })
}

fn emit_call_expr(
    callee: &ExecutionCall,
    args: &[ExecutionExpr],
    result_ty: ValueType,
) -> Result<String, String> {
    match callee {
        ExecutionCall::Math(intrinsic) => emit_math_call(*intrinsic, args, result_ty),
    }
}

fn emit_math_call(
    intrinsic: MathIntrinsic,
    args: &[ExecutionExpr],
    result_ty: ValueType,
) -> Result<String, String> {
    let args = args.iter().map(emit_expr).collect::<Result<Vec<_>, _>>()?;
    Ok(match intrinsic {
        MathIntrinsic::Max | MathIntrinsic::Min => {
            if args.len() != 2 {
                return Err(format!("{intrinsic:?} expects 2 arguments"));
            }
            match result_ty {
                ValueType::Real => {
                    let lhs = cast_expr(args[0].rendered.clone(), args[0].ty, ValueType::Real);
                    let rhs = cast_expr(args[1].rendered.clone(), args[1].ty, ValueType::Real);
                    let method = if intrinsic == MathIntrinsic::Max {
                        "max"
                    } else {
                        "min"
                    };
                    format!("({lhs}).{method}({rhs})")
                }
                ValueType::Int => {
                    let lhs = cast_expr(args[0].rendered.clone(), args[0].ty, ValueType::Int);
                    let rhs = cast_expr(args[1].rendered.clone(), args[1].ty, ValueType::Int);
                    let function = if intrinsic == MathIntrinsic::Max {
                        "std::cmp::max"
                    } else {
                        "std::cmp::min"
                    };
                    format!("{function}({lhs}, {rhs})")
                }
                ValueType::Bool => {
                    return Err("min/max do not accept boolean arguments".to_string())
                }
            }
        }
        MathIntrinsic::Abs if result_ty == ValueType::Int => {
            let value = cast_expr(args[0].rendered.clone(), args[0].ty, ValueType::Int);
            format!("({value}).abs()")
        }
        _ => {
            let function = match intrinsic {
                MathIntrinsic::Abs => "abs",
                MathIntrinsic::Ceil => "ceil",
                MathIntrinsic::Exp => "exp",
                MathIntrinsic::Floor => "floor",
                MathIntrinsic::Ln | MathIntrinsic::Log => "ln",
                MathIntrinsic::Log10 => "log10",
                MathIntrinsic::Log2 => "log2",
                MathIntrinsic::Pow => {
                    if args.len() != 2 {
                        return Err("pow expects 2 arguments".to_string());
                    }
                    let lhs = cast_expr(args[0].rendered.clone(), args[0].ty, ValueType::Real);
                    let rhs = cast_expr(args[1].rendered.clone(), args[1].ty, ValueType::Real);
                    return Ok(cast_expr(
                        format!("({lhs}).powf({rhs})"),
                        ValueType::Real,
                        result_ty,
                    ));
                }
                MathIntrinsic::Round => "round",
                MathIntrinsic::Sin => "sin",
                MathIntrinsic::Cos => "cos",
                MathIntrinsic::Tan => "tan",
                MathIntrinsic::Sqrt => "sqrt",
                MathIntrinsic::Max | MathIntrinsic::Min => unreachable!(),
            };
            let value = cast_expr(args[0].rendered.clone(), args[0].ty, ValueType::Real);
            cast_expr(
                format!("({value}).{function}()"),
                ValueType::Real,
                result_ty,
            )
        }
    })
}

fn cast_expr(expr: String, from: ValueType, to: ValueType) -> String {
    if from == to {
        return expr;
    }

    match (from, to) {
        (ValueType::Int, ValueType::Real) => format!("({expr}) as f64"),
        (ValueType::Bool, ValueType::Real) => format!("if {expr} {{ 1.0 }} else {{ 0.0 }}"),
        (ValueType::Real, ValueType::Int) => format!("({expr}) as i64"),
        (ValueType::Bool, ValueType::Int) => format!("if {expr} {{ 1i64 }} else {{ 0i64 }}"),
        (ValueType::Real, ValueType::Bool) => format!("({expr}) != 0.0"),
        (ValueType::Int, ValueType::Bool) => format!("({expr}) != 0"),
        _ => expr,
    }
}

fn rust_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Real => "f64",
        ValueType::Bool => "bool",
    }
}

fn rust_zero(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "0i64",
        ValueType::Real => "0.0",
        ValueType::Bool => "false",
    }
}

fn push_line(source: &mut String, indent: usize, line: &str) {
    for _ in 0..indent {
        source.push_str("    ");
    }
    source.push_str(line);
    source.push('\n');
}
