//! Cranelift JIT lowering for JIT model expressions.
//!
//! Lowers each parsed [`Expr`] into Cranelift IR and produces two `extern "C"`
//! function pointers per model:
//!
//! ```text
//! extern "C" fn rhs(
//!     t:      f64,
//!     x:      *const f64,   // length = nstates
//!     dx:     *mut f64,     // length = nstates
//!     p:      *const f64,   // length = nparams
//!     rateiv: *const f64,   // length = ndrugs
//!     cov:    *const f64,   // length = ncov (pre-resolved at the call time)
//! );
//!
//! extern "C" fn out(
//!     t: f64,
//!     x: *const f64,
//!     p: *const f64,
//!     cov: *const f64,
//!     y: *mut f64,          // length = nout
//! );
//! ```
//!
//! The bolus contribution is handled in pharmsol's event loop and is therefore
//! always passed as a zero buffer here, matching pharmsol's `DiffEq` semantics
//! when using the diffsol RHS.
//!
//! Identifiers are resolved against the model's compartment, parameter, and
//! covariate name lists. `t` is the special time variable. `rateiv[i]` and
//! `bolus[i]` are explicit indexed accesses.

use std::collections::HashMap;
use std::mem;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::ast::{BinOp, Expr};
use super::model::{Equation, ModelError};

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

/// Signature of a JIT-compiled RHS function.
pub type RhsFn = unsafe extern "C" fn(
    t: f64,
    x: *const f64,
    dx: *mut f64,
    p: *const f64,
    rateiv: *const f64,
    cov: *const f64,
);

/// Signature of a JIT-compiled output function.
pub type OutFn = unsafe extern "C" fn(
    t: f64,
    x: *const f64,
    p: *const f64,
    cov: *const f64,
    y: *mut f64,
);

/// Owned JIT artifact. Holds the JITModule alive so the function pointers
/// remain valid for as long as this struct exists.
pub struct JitArtifact {
    // Order matters: `_module` must outlive the function pointers.
    pub rhs: RhsFn,
    pub out: OutFn,
    _module: JITModule,
}

// Function pointers into JIT-allocated executable memory are safe to share
// across threads — the underlying memory is read-only after finalization.
unsafe impl Send for JitArtifact {}
unsafe impl Sync for JitArtifact {}

impl std::fmt::Debug for JitArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitArtifact")
            .field("rhs", &(self.rhs as *const ()))
            .field("out", &(self.out as *const ()))
            .finish()
    }
}

/// Symbol-resolution table for lowering an expression.
struct Env<'a> {
    compartments: &'a [String],
    params: &'a [String],
    covariates: &'a [String],
    ndrugs: usize,
}

/// Pointer-parameter slots in the lowered functions.
#[derive(Clone, Copy)]
struct PtrArgs {
    t: Value,
    x: Value,
    p: Value,
    rateiv: Option<Value>, // present only in rhs
    cov: Value,
}

pub(crate) fn compile_artifact(
    name: &str,
    compartments: &[String],
    params: &[String],
    covariates: &[String],
    ndrugs: usize,
    dxdt: &[Equation],
    outputs: &[Equation],
) -> Result<JitArtifact, ModelError> {
    let env = Env {
        compartments,
        params,
        covariates,
        ndrugs,
    };

    // Pre-validate identifiers in every expression so we surface clean errors
    // before we touch Cranelift.
    for eq in dxdt {
        validate(&eq.expr, &env, &format!("dxdt({})", eq.target))?;
    }
    for eq in outputs {
        validate(&eq.expr, &env, &format!("output({})", eq.target))?;
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

    let mut builder =
        JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    // Register libm-style externs.
    builder.symbol("pm_exp", externs::exp_ as *const u8);
    builder.symbol("pm_ln", externs::ln_ as *const u8);
    builder.symbol("pm_log10", externs::log10_ as *const u8);
    builder.symbol("pm_sqrt", externs::sqrt_ as *const u8);
    builder.symbol("pm_abs", externs::abs_ as *const u8);
    builder.symbol("pm_pow", externs::pow_ as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_ty = module.target_config().pointer_type();

    // Declare extern functions.
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
    let exp_id = extern_unary(&mut module, "pm_exp")?;
    let ln_id = extern_unary(&mut module, "pm_ln")?;
    let log10_id = extern_unary(&mut module, "pm_log10")?;
    let sqrt_id = extern_unary(&mut module, "pm_sqrt")?;
    let abs_id = extern_unary(&mut module, "pm_abs")?;
    let pow_id = extern_binary(&mut module, "pm_pow")?;

    let externs = ExternIds {
        exp: exp_id,
        ln: ln_id,
        log10: log10_id,
        sqrt: sqrt_id,
        abs: abs_id,
        pow: pow_id,
    };

    // ---- rhs ----
    let mut rhs_sig = module.make_signature();
    rhs_sig.params.push(AbiParam::new(types::F64)); // t
    for _ in 0..5 {
        rhs_sig.params.push(AbiParam::new(ptr_ty));
    }
    let rhs_id = module
        .declare_function(
            &format!("{name}_rhs"),
            Linkage::Export,
            &rhs_sig,
        )
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    let mut ctx = module.make_context();
    ctx.func.signature = rhs_sig;
    let mut fbc = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);
        let params = fb.block_params(block).to_vec();
        let pa = PtrArgs {
            t: params[0],
            x: params[1],
            // params[2] = dx (write target)
            p: params[3],
            rateiv: Some(params[4]),
            cov: params[5],
        };
        let dx_ptr = params[2];

        // Declare extern fn refs in the function so we can call them.
        let exp_ref = module.declare_func_in_func(externs.exp, fb.func);
        let ln_ref = module.declare_func_in_func(externs.ln, fb.func);
        let log10_ref = module.declare_func_in_func(externs.log10, fb.func);
        let sqrt_ref = module.declare_func_in_func(externs.sqrt, fb.func);
        let abs_ref = module.declare_func_in_func(externs.abs, fb.func);
        let pow_ref = module.declare_func_in_func(externs.pow, fb.func);
        let frefs = ExternRefs {
            exp: exp_ref,
            ln: ln_ref,
            log10: log10_ref,
            sqrt: sqrt_ref,
            abs: abs_ref,
            pow: pow_ref,
        };

        for (i, eq) in dxdt.iter().enumerate() {
            let v = lower(&eq.expr, &mut fb, &env, &pa, &frefs, true)
                .map_err(|e| ModelError::Codegen(format!("dxdt[{i}] {}: {e}", eq.target)))?;
            let offset = (i * mem::size_of::<f64>()) as i32;
            fb.ins().store(MemFlags::trusted(), v, dx_ptr, offset);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }
    module
        .define_function(rhs_id, &mut ctx)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    module.clear_context(&mut ctx);

    // ---- out ----
    let mut out_sig = module.make_signature();
    out_sig.params.push(AbiParam::new(types::F64)); // t
    for _ in 0..4 {
        out_sig.params.push(AbiParam::new(ptr_ty));
    }
    let out_id = module
        .declare_function(
            &format!("{name}_out"),
            Linkage::Export,
            &out_sig,
        )
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    ctx.func.signature = out_sig;
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);
        let params = fb.block_params(block).to_vec();
        let pa = PtrArgs {
            t: params[0],
            x: params[1],
            p: params[2],
            rateiv: None,
            cov: params[3],
        };
        let y_ptr = params[4];

        let exp_ref = module.declare_func_in_func(externs.exp, fb.func);
        let ln_ref = module.declare_func_in_func(externs.ln, fb.func);
        let log10_ref = module.declare_func_in_func(externs.log10, fb.func);
        let sqrt_ref = module.declare_func_in_func(externs.sqrt, fb.func);
        let abs_ref = module.declare_func_in_func(externs.abs, fb.func);
        let pow_ref = module.declare_func_in_func(externs.pow, fb.func);
        let frefs = ExternRefs {
            exp: exp_ref,
            ln: ln_ref,
            log10: log10_ref,
            sqrt: sqrt_ref,
            abs: abs_ref,
            pow: pow_ref,
        };

        for (i, eq) in outputs.iter().enumerate() {
            let v = lower(&eq.expr, &mut fb, &env, &pa, &frefs, false)
                .map_err(|e| ModelError::Codegen(format!("output[{i}] {}: {e}", eq.target)))?;
            let offset = (i * mem::size_of::<f64>()) as i32;
            fb.ins().store(MemFlags::trusted(), v, y_ptr, offset);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }
    module
        .define_function(out_id, &mut ctx)
        .map_err(|e| ModelError::Codegen(e.to_string()))?;
    module.clear_context(&mut ctx);

    module
        .finalize_definitions()
        .map_err(|e| ModelError::Codegen(e.to_string()))?;

    let rhs_ptr = module.get_finalized_function(rhs_id);
    let out_ptr = module.get_finalized_function(out_id);

    let rhs: RhsFn = unsafe { mem::transmute(rhs_ptr) };
    let out: OutFn = unsafe { mem::transmute(out_ptr) };

    Ok(JitArtifact {
        rhs,
        out,
        _module: module,
    })
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

fn validate(expr: &Expr, env: &Env<'_>, ctx: &str) -> Result<(), ModelError> {
    match expr {
        Expr::Const(_) => Ok(()),
        Expr::Ident(name) => {
            if name == "t"
                || env.compartments.iter().any(|c| c == name)
                || env.params.iter().any(|p| p == name)
                || env.covariates.iter().any(|c| c == name)
            {
                Ok(())
            } else {
                Err(ModelError::UnresolvedIdent {
                    ident: name.clone(),
                    context: ctx.to_string(),
                })
            }
        }
        Expr::Index(name, idx) => match name.as_str() {
            "rateiv" => {
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
        Expr::Neg(e) => validate(e, env, ctx),
        Expr::Bin(_, l, r) => {
            validate(l, env, ctx)?;
            validate(r, env, ctx)
        }
        Expr::Call(name, args) => {
            let expected = match name.as_str() {
                "exp" | "ln" | "log" | "log10" | "sqrt" | "abs" => 1,
                "pow" => 2,
                _ => {
                    return Err(ModelError::UnknownFunction { name: name.clone() });
                }
            };
            if args.len() != expected {
                return Err(ModelError::BadArity {
                    name: name.clone(),
                    expected,
                    got: args.len(),
                });
            }
            for a in args {
                validate(a, env, ctx)?;
            }
            Ok(())
        }
    }
}

/// Lower an expression to a single `f64` Cranelift `Value`.
fn lower(
    expr: &Expr,
    fb: &mut FunctionBuilder<'_>,
    env: &Env<'_>,
    pa: &PtrArgs,
    fr: &ExternRefs,
    in_rhs: bool,
) -> Result<Value, String> {
    Ok(match expr {
        Expr::Const(c) => fb.ins().f64const(*c),
        Expr::Ident(name) => {
            if name == "t" {
                pa.t
            } else if let Some(i) = env.compartments.iter().position(|c| c == name) {
                load_f64(fb, pa.x, i)
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
                if !in_rhs {
                    return Err("rateiv[] is only valid in dxdt expressions".into());
                }
                let ptr = pa.rateiv.expect("rhs has rateiv");
                load_f64(fb, ptr, *idx)
            }
            "bolus" => {
                // Bolus is handled by pharmsol's event loop; expose as 0.0.
                fb.ins().f64const(0.0)
            }
            other => return Err(format!("invalid index target {other:?}")),
        },
        Expr::Neg(e) => {
            let v = lower(e, fb, env, pa, fr, in_rhs)?;
            fb.ins().fneg(v)
        }
        Expr::Bin(op, l, r) => {
            let lv = lower(l, fb, env, pa, fr, in_rhs)?;
            let rv = lower(r, fb, env, pa, fr, in_rhs)?;
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
                .map(|a| lower(a, fb, env, pa, fr, in_rhs))
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

// Suppress unused-import warning when the parser HashMap isn't referenced.
#[allow(dead_code)]
fn _hashmap_keep() -> HashMap<String, usize> {
    HashMap::new()
}
