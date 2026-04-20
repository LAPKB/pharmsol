//! Cranelift JIT lowering for JIT model expressions.
//!
//! Lowers each parsed [`Expr`] into Cranelift IR and produces up to five
//! `extern "C"` function pointers per model:
//!
//! ```text
//! extern "C" fn rhs(t, x, dx, p, rateiv, cov);     // always
//! extern "C" fn out(t, x, p, cov, y);              // always
//! extern "C" fn init(t, p, cov, x_out);            // if any init() declared
//! extern "C" fn lag (t, p, cov, lag_out[ndrugs]);  // if any lag(N) declared
//! extern "C" fn fa  (t, p, cov, fa_out[ndrugs]);   // if any fa(N)  declared
//! ```
//!
//! `lag`/`fa` writers fully overwrite their output buffers, using `0.0` /
//! `1.0` defaults for any channel without an explicit declaration. `init`
//! does the same with default `0.0` for compartments.
//!
//! `let` bindings are model-wide named scalar expressions. For each emitted
//! function, the lets transitively reachable from that function's expressions
//! are evaluated once at function entry and reused via SSA Values.

use std::collections::{HashMap, HashSet};
use std::mem;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::ast::{BinOp, Expr};
use super::model::{Equation, LetBinding, ModelError};

/// Native libm-backed extern functions used by JIT'd code.
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
    pub extern "C" fn sqrt_(x: f64) -> f64 {
        x.sqrt()
    }
    pub extern "C" fn abs_(x: f64) -> f64 {
        x.abs()
    }
    pub extern "C" fn pow_(a: f64, b: f64) -> f64 {
        a.powf(b)
    }
}

pub type RhsFn = unsafe extern "C" fn(
    t: f64,
    x: *const f64,
    dx: *mut f64,
    p: *const f64,
    rateiv: *const f64,
    cov: *const f64,
);

pub type OutFn =
    unsafe extern "C" fn(t: f64, x: *const f64, p: *const f64, cov: *const f64, y: *mut f64);

pub type InitFn = unsafe extern "C" fn(t: f64, p: *const f64, cov: *const f64, x_out: *mut f64);
pub type LagFn = unsafe extern "C" fn(t: f64, p: *const f64, cov: *const f64, out: *mut f64);
pub type FaFn = unsafe extern "C" fn(t: f64, p: *const f64, cov: *const f64, out: *mut f64);

/// Owned JIT artifact. Holds the JITModule alive so the function pointers
/// remain valid for as long as this struct exists.
pub struct JitArtifact {
    pub rhs: RhsFn,
    pub out: OutFn,
    pub init: Option<InitFn>,
    pub lag: Option<LagFn>,
    pub fa: Option<FaFn>,
    _module: JITModule,
}

unsafe impl Send for JitArtifact {}
unsafe impl Sync for JitArtifact {}

impl std::fmt::Debug for JitArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitArtifact")
            .field("rhs", &(self.rhs as *const ()))
            .field("out", &(self.out as *const ()))
            .field("init", &self.init.map(|p| p as *const ()))
            .field("lag", &self.lag.map(|p| p as *const ()))
            .field("fa", &self.fa.map(|p| p as *const ()))
            .finish()
    }
}

/// Symbols available in a given function's expressions. Lets are resolved
/// separately via `Env::let_idx`.
#[derive(Clone, Copy)]
struct Ctx {
    has_state: bool,
    has_rateiv: bool,
}

const CTX_RHS: Ctx = Ctx {
    has_state: true,
    has_rateiv: true,
};
const CTX_OUT: Ctx = Ctx {
    has_state: true,
    has_rateiv: false,
};
const CTX_AUX: Ctx = Ctx {
    has_state: false,
    has_rateiv: false,
};

struct Env<'a> {
    compartments: &'a [String],
    params: &'a [String],
    covariates: &'a [String],
    ndrugs: usize,
    let_idx: HashMap<String, usize>,
    lets: &'a [LetBinding],
}

#[derive(Clone, Copy)]
struct PtrArgs {
    t: Value,
    x: Option<Value>,
    p: Value,
    rateiv: Option<Value>,
    cov: Value,
}

#[derive(Clone, Copy)]
struct ExternIds {
    exp: cranelift_module::FuncId,
    ln: cranelift_module::FuncId,
    log10: cranelift_module::FuncId,
    sqrt: cranelift_module::FuncId,
    abs: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
}

#[derive(Clone, Copy)]
struct ExternRefs {
    exp: codegen::ir::FuncRef,
    ln: codegen::ir::FuncRef,
    log10: codegen::ir::FuncRef,
    sqrt: codegen::ir::FuncRef,
    abs: codegen::ir::FuncRef,
    pow: codegen::ir::FuncRef,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_artifact(
    name: &str,
    compartments: &[String],
    params: &[String],
    covariates: &[String],
    ndrugs: usize,
    dxdt: &[Equation],
    outputs: &[Equation],
    lets: &[LetBinding],
    inits: &[Option<Expr>], // length = nstates
    lags: &[Option<Expr>],  // length = ndrugs
    fas: &[Option<Expr>],   // length = ndrugs
) -> Result<JitArtifact, ModelError> {
    let let_idx: HashMap<String, usize> = lets
        .iter()
        .enumerate()
        .map(|(i, lb)| (lb.name.clone(), i))
        .collect();
    let env = Env {
        compartments,
        params,
        covariates,
        ndrugs,
        let_idx,
        lets,
    };

    // Validate every expression up front.
    for eq in dxdt {
        validate(&eq.expr, &env, CTX_RHS, &format!("dxdt({})", eq.target))?;
    }
    for eq in outputs {
        validate(&eq.expr, &env, CTX_OUT, &format!("output({})", eq.target))?;
    }
    for (i, init) in inits.iter().enumerate() {
        if let Some(e) = init {
            validate(e, &env, CTX_AUX, &format!("init({})", compartments[i]))?;
        }
    }
    for (i, lag) in lags.iter().enumerate() {
        if let Some(e) = lag {
            validate(e, &env, CTX_AUX, &format!("lag({i})"))?;
        }
    }
    for (i, fa) in fas.iter().enumerate() {
        if let Some(e) = fa {
            validate(e, &env, CTX_AUX, &format!("fa({i})"))?;
        }
    }
    // Validate let bodies in the most-permissive context (rhs). Per-function
    // emission re-validates them in their actual usage context.
    for lb in lets {
        validate(&lb.expr, &env, CTX_RHS, &format!("let {}", lb.name))?;
    }
    // Cycle check on let bindings.
    for i in 0..lets.len() {
        let mut visiting = HashSet::new();
        check_let_no_cycle(i, &env, &mut visiting)?;
    }

    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    let isa_builder =
        cranelift_native::builder().map_err(|e| ModelError::Codegen(e.to_string()))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol("pm_exp", externs::exp_ as *const u8);
    builder.symbol("pm_ln", externs::ln_ as *const u8);
    builder.symbol("pm_log10", externs::log10_ as *const u8);
    builder.symbol("pm_sqrt", externs::sqrt_ as *const u8);
    builder.symbol("pm_abs", externs::abs_ as *const u8);
    builder.symbol("pm_pow", externs::pow_ as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_ty = module.target_config().pointer_type();

    let extern_unary = |module: &mut JITModule, sym: &str| -> Result<_, ModelError> {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));
        module
            .declare_function(sym, Linkage::Import, &sig)
            .map_err(|e| ModelError::Codegen(e.to_string()))
    };
    let extern_binary = |module: &mut JITModule, sym: &str| -> Result<_, ModelError> {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));
        module
            .declare_function(sym, Linkage::Import, &sig)
            .map_err(|e| ModelError::Codegen(e.to_string()))
    };
    let externs = ExternIds {
        exp: extern_unary(&mut module, "pm_exp")?,
        ln: extern_unary(&mut module, "pm_ln")?,
        log10: extern_unary(&mut module, "pm_log10")?,
        sqrt: extern_unary(&mut module, "pm_sqrt")?,
        abs: extern_unary(&mut module, "pm_abs")?,
        pow: extern_binary(&mut module, "pm_pow")?,
    };

    let mut ctx = module.make_context();
    let mut fbc = FunctionBuilderContext::new();

    let rhs_id = emit_rhs(
        &mut module,
        &mut ctx,
        &mut fbc,
        ptr_ty,
        &externs,
        &env,
        name,
        dxdt,
    )?;
    let out_id = emit_out(
        &mut module,
        &mut ctx,
        &mut fbc,
        ptr_ty,
        &externs,
        &env,
        name,
        outputs,
    )?;

    let init_id = if inits.iter().any(|x| x.is_some()) {
        Some(emit_aux(
            &mut module,
            &mut ctx,
            &mut fbc,
            ptr_ty,
            &externs,
            &env,
            &format!("{name}_init"),
            inits,
            0.0,
        )?)
    } else {
        None
    };
    let lag_id = if lags.iter().any(|x| x.is_some()) {
        Some(emit_aux(
            &mut module,
            &mut ctx,
            &mut fbc,
            ptr_ty,
            &externs,
            &env,
            &format!("{name}_lag"),
            lags,
            0.0,
        )?)
    } else {
        None
    };
    let fa_id = if fas.iter().any(|x| x.is_some()) {
        Some(emit_aux(
            &mut module,
            &mut ctx,
            &mut fbc,
            ptr_ty,
            &externs,
            &env,
            &format!("{name}_fa"),
            fas,
            1.0,
        )?)
    } else {
        None
    };

    module
        .finalize_definitions()
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    let rhs: RhsFn = unsafe { mem::transmute(module.get_finalized_function(rhs_id)) };
    let out: OutFn = unsafe { mem::transmute(module.get_finalized_function(out_id)) };
    let init =
        init_id.map(|id| unsafe { mem::transmute::<_, InitFn>(module.get_finalized_function(id)) });
    let lag =
        lag_id.map(|id| unsafe { mem::transmute::<_, LagFn>(module.get_finalized_function(id)) });
    let fa =
        fa_id.map(|id| unsafe { mem::transmute::<_, FaFn>(module.get_finalized_function(id)) });

    Ok(JitArtifact {
        rhs,
        out,
        init,
        lag,
        fa,
        _module: module,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_rhs(
    module: &mut JITModule,
    ctx: &mut codegen::Context,
    fbc: &mut FunctionBuilderContext,
    ptr_ty: codegen::ir::Type,
    externs: &ExternIds,
    env: &Env<'_>,
    name: &str,
    dxdt: &[Equation],
) -> Result<cranelift_module::FuncId, ModelError> {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::F64));
    for _ in 0..5 {
        sig.params.push(AbiParam::new(ptr_ty));
    }
    let id = module
        .declare_function(&format!("{name}_rhs"), Linkage::Export, &sig)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    ctx.func.signature = sig;

    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);
        let p = fb.block_params(block).to_vec();
        let pa = PtrArgs {
            t: p[0],
            x: Some(p[1]),
            p: p[3],
            rateiv: Some(p[4]),
            cov: p[5],
        };
        let dx_ptr = p[2];
        let frefs = declare_refs(module, &mut fb, externs);
        let lvals = emit_lets(
            &mut fb,
            env,
            &pa,
            &frefs,
            CTX_RHS,
            dxdt.iter().map(|e| &e.expr),
        )?;
        for (i, eq) in dxdt.iter().enumerate() {
            let v = lower(&eq.expr, &mut fb, env, &pa, &frefs, CTX_RHS, &lvals)
                .map_err(|e| ModelError::Codegen(format!("dxdt[{i}] {}: {e}", eq.target)))?;
            let offset = (i * mem::size_of::<f64>()) as i32;
            fb.ins().store(MemFlags::trusted(), v, dx_ptr, offset);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }
    module
        .define_function(id, ctx)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    module.clear_context(ctx);
    Ok(id)
}

#[allow(clippy::too_many_arguments)]
fn emit_out(
    module: &mut JITModule,
    ctx: &mut codegen::Context,
    fbc: &mut FunctionBuilderContext,
    ptr_ty: codegen::ir::Type,
    externs: &ExternIds,
    env: &Env<'_>,
    name: &str,
    outputs: &[Equation],
) -> Result<cranelift_module::FuncId, ModelError> {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::F64));
    for _ in 0..4 {
        sig.params.push(AbiParam::new(ptr_ty));
    }
    let id = module
        .declare_function(&format!("{name}_out"), Linkage::Export, &sig)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    ctx.func.signature = sig;

    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);
        let p = fb.block_params(block).to_vec();
        let pa = PtrArgs {
            t: p[0],
            x: Some(p[1]),
            p: p[2],
            rateiv: None,
            cov: p[3],
        };
        let y_ptr = p[4];
        let frefs = declare_refs(module, &mut fb, externs);
        let lvals = emit_lets(
            &mut fb,
            env,
            &pa,
            &frefs,
            CTX_OUT,
            outputs.iter().map(|e| &e.expr),
        )?;
        for (i, eq) in outputs.iter().enumerate() {
            let v = lower(&eq.expr, &mut fb, env, &pa, &frefs, CTX_OUT, &lvals)
                .map_err(|e| ModelError::Codegen(format!("output[{i}] {}: {e}", eq.target)))?;
            let offset = (i * mem::size_of::<f64>()) as i32;
            fb.ins().store(MemFlags::trusted(), v, y_ptr, offset);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }
    module
        .define_function(id, ctx)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    module.clear_context(ctx);
    Ok(id)
}

#[allow(clippy::too_many_arguments)]
fn emit_aux(
    module: &mut JITModule,
    ctx: &mut codegen::Context,
    fbc: &mut FunctionBuilderContext,
    ptr_ty: codegen::ir::Type,
    externs: &ExternIds,
    env: &Env<'_>,
    name: &str,
    slots: &[Option<Expr>],
    default: f64,
) -> Result<cranelift_module::FuncId, ModelError> {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::F64));
    for _ in 0..3 {
        sig.params.push(AbiParam::new(ptr_ty));
    }
    let id = module
        .declare_function(name, Linkage::Export, &sig)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    ctx.func.signature = sig;

    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);
        let pp = fb.block_params(block).to_vec();
        let pa = PtrArgs {
            t: pp[0],
            x: None,
            p: pp[1],
            rateiv: None,
            cov: pp[2],
        };
        let out_ptr = pp[3];
        let frefs = declare_refs(module, &mut fb, externs);
        let lvals = emit_lets(
            &mut fb,
            env,
            &pa,
            &frefs,
            CTX_AUX,
            slots.iter().filter_map(|s| s.as_ref()),
        )?;
        for (i, slot) in slots.iter().enumerate() {
            let v = match slot {
                Some(e) => lower(e, &mut fb, env, &pa, &frefs, CTX_AUX, &lvals)
                    .map_err(|err| ModelError::Codegen(format!("{name}[{i}]: {err}")))?,
                None => fb.ins().f64const(default),
            };
            let offset = (i * mem::size_of::<f64>()) as i32;
            fb.ins().store(MemFlags::trusted(), v, out_ptr, offset);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }
    module
        .define_function(id, ctx)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    module.clear_context(ctx);
    Ok(id)
}

fn declare_refs(module: &mut JITModule, fb: &mut FunctionBuilder<'_>, e: &ExternIds) -> ExternRefs {
    ExternRefs {
        exp: module.declare_func_in_func(e.exp, fb.func),
        ln: module.declare_func_in_func(e.ln, fb.func),
        log10: module.declare_func_in_func(e.log10, fb.func),
        sqrt: module.declare_func_in_func(e.sqrt, fb.func),
        abs: module.declare_func_in_func(e.abs, fb.func),
        pow: module.declare_func_in_func(e.pow, fb.func),
    }
}

/// Compute and emit the let bindings transitively reachable from
/// `target_exprs` in the given context, in source order. Returns a map from
/// let-name to the resulting Cranelift Value (a single SSA value reused by
/// all consumers).
fn emit_lets<'a, I>(
    fb: &mut FunctionBuilder<'_>,
    env: &Env<'_>,
    pa: &PtrArgs,
    fr: &ExternRefs,
    ctx: Ctx,
    target_exprs: I,
) -> Result<HashMap<String, Value>, ModelError>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let mut needed = vec![false; env.lets.len()];
    let mut frontier: Vec<&Expr> = target_exprs.into_iter().collect();
    while let Some(e) = frontier.pop() {
        collect_let_uses(e, env, &mut needed, &mut frontier);
    }

    let mut vals: HashMap<String, Value> = HashMap::new();
    for (i, lb) in env.lets.iter().enumerate() {
        if !needed[i] {
            continue;
        }
        validate(
            &lb.expr,
            env,
            ctx,
            &format!("let {} (in this scope)", lb.name),
        )?;
        let v = lower(&lb.expr, fb, env, pa, fr, ctx, &vals)
            .map_err(|e| ModelError::Codegen(format!("let {}: {e}", lb.name)))?;
        vals.insert(lb.name.clone(), v);
    }
    Ok(vals)
}

fn collect_let_uses<'a>(
    e: &'a Expr,
    env: &'a Env<'_>,
    needed: &mut [bool],
    frontier: &mut Vec<&'a Expr>,
) {
    match e {
        Expr::Const(_) | Expr::Index(_, _) => {}
        Expr::Ident(name) => {
            if let Some(&i) = env.let_idx.get(name) {
                if !needed[i] {
                    needed[i] = true;
                    frontier.push(&env.lets[i].expr);
                }
            }
        }
        Expr::Neg(inner) => collect_let_uses(inner, env, needed, frontier),
        Expr::Bin(_, a, b) => {
            collect_let_uses(a, env, needed, frontier);
            collect_let_uses(b, env, needed, frontier);
        }
        Expr::Call(_, args) => {
            for a in args {
                collect_let_uses(a, env, needed, frontier);
            }
        }
    }
}

fn check_let_no_cycle(
    start: usize,
    env: &Env<'_>,
    visiting: &mut HashSet<usize>,
) -> Result<(), ModelError> {
    if !visiting.insert(start) {
        return Err(ModelError::LetCycle(env.lets[start].name.clone()));
    }
    let mut deps: Vec<usize> = Vec::new();
    collect_let_dep_indices(&env.lets[start].expr, env, &mut deps);
    for d in deps {
        check_let_no_cycle(d, env, visiting)?;
    }
    visiting.remove(&start);
    Ok(())
}

fn collect_let_dep_indices(e: &Expr, env: &Env<'_>, out: &mut Vec<usize>) {
    match e {
        Expr::Const(_) | Expr::Index(_, _) => {}
        Expr::Ident(name) => {
            if let Some(&i) = env.let_idx.get(name) {
                out.push(i);
            }
        }
        Expr::Neg(inner) => collect_let_dep_indices(inner, env, out),
        Expr::Bin(_, a, b) => {
            collect_let_dep_indices(a, env, out);
            collect_let_dep_indices(b, env, out);
        }
        Expr::Call(_, args) => {
            for a in args {
                collect_let_dep_indices(a, env, out);
            }
        }
    }
}

fn validate(expr: &Expr, env: &Env<'_>, ctx: Ctx, where_: &str) -> Result<(), ModelError> {
    match expr {
        Expr::Const(_) => Ok(()),
        Expr::Ident(name) => {
            if name == "t"
                || env.let_idx.contains_key(name)
                || env.params.iter().any(|p| p == name)
                || env.covariates.iter().any(|c| c == name)
                || (ctx.has_state && env.compartments.iter().any(|c| c == name))
            {
                Ok(())
            } else {
                Err(ModelError::UnresolvedIdent {
                    ident: name.clone(),
                    context: where_.to_string(),
                })
            }
        }
        Expr::Index(name, idx) => match name.as_str() {
            "rateiv" => {
                if !ctx.has_rateiv {
                    return Err(ModelError::UnresolvedIdent {
                        ident: format!("rateiv[{idx}]"),
                        context: where_.to_string(),
                    });
                }
                if *idx >= env.ndrugs {
                    Err(ModelError::RateIvOutOfRange(*idx, env.ndrugs))
                } else {
                    Ok(())
                }
            }
            "bolus" => {
                if *idx >= env.ndrugs {
                    Err(ModelError::BolusOutOfRange(*idx, env.ndrugs))
                } else {
                    Ok(())
                }
            }
            other => Err(ModelError::InvalidIndexTarget(other.to_string())),
        },
        Expr::Neg(e) => validate(e, env, ctx, where_),
        Expr::Bin(_, l, r) => {
            validate(l, env, ctx, where_)?;
            validate(r, env, ctx, where_)
        }
        Expr::Call(name, args) => {
            let expected = match name.as_str() {
                "exp" | "ln" | "log" | "log10" | "sqrt" | "abs" => 1,
                "pow" => 2,
                _ => return Err(ModelError::UnknownFunction { name: name.clone() }),
            };
            if args.len() != expected {
                return Err(ModelError::BadArity {
                    name: name.clone(),
                    expected,
                    got: args.len(),
                });
            }
            for a in args {
                validate(a, env, ctx, where_)?;
            }
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn lower(
    expr: &Expr,
    fb: &mut FunctionBuilder<'_>,
    env: &Env<'_>,
    pa: &PtrArgs,
    fr: &ExternRefs,
    ctx: Ctx,
    lvals: &HashMap<String, Value>,
) -> Result<Value, String> {
    Ok(match expr {
        Expr::Const(c) => fb.ins().f64const(*c),
        Expr::Ident(name) => {
            if name == "t" {
                pa.t
            } else if let Some(v) = lvals.get(name) {
                *v
            } else if let Some(i) = ctx
                .has_state
                .then(|| env.compartments.iter().position(|c| c == name))
                .flatten()
            {
                let x = pa.x.expect("ctx.has_state implies x pointer is present");
                load_f64(fb, x, i)
            } else if let Some(i) = env.params.iter().position(|p| p == name) {
                load_f64(fb, pa.p, i)
            } else if let Some(i) = env.covariates.iter().position(|c| c == name) {
                load_f64(fb, pa.cov, i)
            } else {
                return Err(format!("unresolved identifier {name:?}"));
            }
        }
        Expr::Index(name, idx) => match name.as_str() {
            "rateiv" => {
                if !ctx.has_rateiv {
                    return Err("rateiv[] is only valid in dxdt expressions".into());
                }
                let ptr = pa.rateiv.expect("rhs has rateiv");
                load_f64(fb, ptr, *idx)
            }
            "bolus" => fb.ins().f64const(0.0),
            other => return Err(format!("invalid index target {other:?}")),
        },
        Expr::Neg(e) => {
            let v = lower(e, fb, env, pa, fr, ctx, lvals)?;
            fb.ins().fneg(v)
        }
        Expr::Bin(op, l, r) => {
            let lv = lower(l, fb, env, pa, fr, ctx, lvals)?;
            let rv = lower(r, fb, env, pa, fr, ctx, lvals)?;
            match op {
                BinOp::Add => fb.ins().fadd(lv, rv),
                BinOp::Sub => fb.ins().fsub(lv, rv),
                BinOp::Mul => fb.ins().fmul(lv, rv),
                BinOp::Div => fb.ins().fdiv(lv, rv),
                BinOp::Pow => {
                    let call = fb.ins().call(fr.pow, &[lv, rv]);
                    fb.inst_results(call)[0]
                }
            }
        }
        Expr::Call(name, args) => {
            let lowered: Vec<Value> = args
                .iter()
                .map(|a| lower(a, fb, env, pa, fr, ctx, lvals))
                .collect::<Result<_, _>>()?;
            let func_ref = match name.as_str() {
                "exp" => fr.exp,
                "ln" | "log" => fr.ln,
                "log10" => fr.log10,
                "sqrt" => fr.sqrt,
                "abs" => fr.abs,
                "pow" => fr.pow,
                other => return Err(format!("unknown function {other:?}")),
            };
            let call = fb.ins().call(func_ref, &lowered);
            fb.inst_results(call)[0]
        }
    })
}

#[inline]
fn load_f64(fb: &mut FunctionBuilder<'_>, base: Value, idx: usize) -> Value {
    let offset = (idx * mem::size_of::<f64>()) as i32;
    fb.ins().load(types::F64, MemFlags::trusted(), base, offset)
}
