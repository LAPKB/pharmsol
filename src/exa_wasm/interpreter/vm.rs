use serde::{Deserialize, Serialize};

/// Production-grade opcode set for the exa_wasm VM.
/// Keep names compatible with earlier POC where reasonable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Opcode {
    // stack and constants
    PushConst(f64),    // push constant
    LoadParam(usize),  // push p[idx]
    LoadX(usize),      // push x[idx]
    LoadRateiv(usize), // push rateiv[idx]
    LoadLocal(usize),  // push local slot
    LoadT,             // push t
    // dynamic indexed loads/stores (index evaluated at runtime)
    LoadParamDyn,  // pop index -> push p[idx]
    LoadXDyn,      // pop index -> push x[idx]
    LoadRateivDyn, // pop index -> push rateiv[idx]

    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,

    // comparisons / logical (push 0.0/1.0)
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,

    // control flow
    Jump(usize),        // absolute pc
    JumpIfFalse(usize), // pop cond, if false jump

    // builtin call: index into func table, arg count
    CallBuiltin(usize, usize),

    // stores
    StoreDx(usize),    // pop value and assign to dx[index]
    StoreX(usize),     // pop value into x[index]
    StoreY(usize),     // pop value into y[index]
    StoreLocal(usize), // pop value into local slot
    // dynamic stores: pop value then pop index (index is f64 -> usize)
    StoreDxDyn, // pop value, pop index -> assign to dx[idx]
    StoreXDyn,  // pop value, pop index -> assign to x[idx]
    StoreYDyn,  // pop value, pop index -> assign to y[idx]
}

/// Execute a sequence of opcodes with full VM context.
/// `assign_indexed` is called for dx/x/y assignments (name, idx, val).
pub fn run_bytecode_full<F>(
    code: &[Opcode],
    x: &[f64],
    p: &[f64],
    rateiv: &[f64],
    t: f64,
    locals: &mut [f64],
    funcs: &Vec<String>,
    builtins_dispatch: &dyn Fn(&str, &[f64]) -> f64,
    mut assign_indexed: F,
) where
    F: FnMut(&str, usize, f64),
{
    let mut stack: Vec<f64> = Vec::new();
    let mut pc: usize = 0;
    let code_len = code.len();
    while pc < code_len {
        match &code[pc] {
            Opcode::PushConst(v) => {
                stack.push(*v);
                pc += 1;
            }
            Opcode::LoadParam(i) => {
                let v = if *i < p.len() { p[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadX(i) => {
                let v = if *i < x.len() { x[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadRateiv(i) => {
                let v = if *i < rateiv.len() { rateiv[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadParamDyn => {
                // index is expected on stack as f64
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < p.len() { p[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadXDyn => {
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < x.len() { x[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadRateivDyn => {
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < rateiv.len() { rateiv[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadLocal(i) => {
                let v = if *i < locals.len() { locals[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadT => {
                stack.push(t);
                pc += 1;
            }
            Opcode::Add => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a + b);
                pc += 1;
            }
            Opcode::Sub => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a - b);
                pc += 1;
            }
            Opcode::Mul => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a * b);
                pc += 1;
            }
            Opcode::Div => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a / b);
                pc += 1;
            }
            Opcode::Pow => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.powf(b));
                pc += 1;
            }
            Opcode::Lt => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a < b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Gt => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a > b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Le => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a <= b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Ge => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a >= b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Eq => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a == b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Ne => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a != b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Jump(addr) => {
                pc = *addr;
            }
            Opcode::JumpIfFalse(addr) => {
                let c = stack.pop().unwrap_or(0.0);
                if c == 0.0 {
                    pc = *addr;
                } else {
                    pc += 1;
                }
            }
            Opcode::CallBuiltin(func_idx, argc) => {
                // pop args in reverse order
                let mut args: Vec<f64> = Vec::with_capacity(*argc);
                for _ in 0..*argc {
                    args.push(stack.pop().unwrap_or(0.0));
                }
                args.reverse();
                let func_name = funcs.get(*func_idx).map(|s| s.as_str()).unwrap_or("");
                let res = builtins_dispatch(func_name, &args);
                stack.push(res);
                pc += 1;
            }
            Opcode::StoreDx(i) => {
                let v = stack.pop().unwrap_or(0.0);
                assign_indexed("dx", *i, v);
                pc += 1;
            }
            Opcode::StoreX(i) => {
                let v = stack.pop().unwrap_or(0.0);
                assign_indexed("x", *i, v);
                pc += 1;
            }
            Opcode::StoreY(i) => {
                let v = stack.pop().unwrap_or(0.0);
                assign_indexed("y", *i, v);
                pc += 1;
            }
            Opcode::StoreDxDyn => {
                // pop value then index
                let v = stack.pop().unwrap_or(0.0);
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                assign_indexed("dx", idx, v);
                pc += 1;
            }
            Opcode::StoreXDyn => {
                let v = stack.pop().unwrap_or(0.0);
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                assign_indexed("x", idx, v);
                pc += 1;
            }
            Opcode::StoreYDyn => {
                let v = stack.pop().unwrap_or(0.0);
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                assign_indexed("y", idx, v);
                pc += 1;
            }
            Opcode::StoreLocal(i) => {
                let v = stack.pop().unwrap_or(0.0);
                if *i < locals.len() {
                    locals[*i] = v;
                }
                pc += 1;
            }
        }
    }
}

/// Backwards-compatible lightweight runner used by some unit tests and the
/// legacy emit POC. Runs a minimal subset (params + arithmetic + StoreDx).
pub fn run_bytecode<F>(code: &[Opcode], p: &[f64], mut assign_dx: F)
where
    F: FnMut(usize, f64),
{
    // emulate a minimal environment
    let x: Vec<f64> = Vec::new();
    let rateiv: Vec<f64> = Vec::new();
    let mut locals: Vec<f64> = Vec::new();
    let funcs: Vec<String> = Vec::new();
    let builtins = |_: &str, _: &[f64]| -> f64 { 0.0 };
    run_bytecode_full(
        code,
        &x,
        p,
        &rateiv,
        0.0,
        &mut locals,
        &funcs,
        &builtins,
        |n, i, v| {
            if n == "dx" {
                assign_dx(i, v);
            }
        },
    );
}

/// Run a sequence of opcodes and return the top-of-stack value at the end.
/// This is useful for bytecode fragments that compute an expression value
/// (e.g., lag/fa entries) rather than performing stores.
pub fn run_bytecode_eval(
    code: &[Opcode],
    x: &[f64],
    p: &[f64],
    rateiv: &[f64],
    t: f64,
    locals: &mut [f64],
    funcs: &Vec<String>,
    builtins_dispatch: &dyn Fn(&str, &[f64]) -> f64,
) -> f64 {
    let mut stack: Vec<f64> = Vec::new();
    let mut pc: usize = 0;
    let code_len = code.len();
    while pc < code_len {
        match &code[pc] {
            Opcode::PushConst(v) => {
                stack.push(*v);
                pc += 1;
            }
            Opcode::LoadParam(i) => {
                let v = if *i < p.len() { p[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadX(i) => {
                let v = if *i < x.len() { x[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadRateiv(i) => {
                let v = if *i < rateiv.len() { rateiv[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadParamDyn => {
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < p.len() { p[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadXDyn => {
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < x.len() { x[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadRateivDyn => {
                let idxf = stack.pop().unwrap_or(0.0);
                let idx = idxf as usize;
                let v = if idx < rateiv.len() { rateiv[idx] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadLocal(i) => {
                let v = if *i < locals.len() { locals[*i] } else { 0.0 };
                stack.push(v);
                pc += 1;
            }
            Opcode::LoadT => {
                stack.push(t);
                pc += 1;
            }
            Opcode::Add => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a + b);
                pc += 1;
            }
            Opcode::Sub => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a - b);
                pc += 1;
            }
            Opcode::Mul => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a * b);
                pc += 1;
            }
            Opcode::Div => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a / b);
                pc += 1;
            }
            Opcode::Pow => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.powf(b));
                pc += 1;
            }
            Opcode::Lt => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a < b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Gt => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a > b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Le => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a <= b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Ge => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a >= b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Eq => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a == b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Ne => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a != b { 1.0 } else { 0.0 });
                pc += 1;
            }
            Opcode::Jump(addr) => {
                pc = *addr;
            }
            Opcode::JumpIfFalse(addr) => {
                let c = stack.pop().unwrap_or(0.0);
                if c == 0.0 {
                    pc = *addr;
                } else {
                    pc += 1;
                }
            }
            Opcode::CallBuiltin(func_idx, argc) => {
                let mut args: Vec<f64> = Vec::with_capacity(*argc);
                for _ in 0..*argc {
                    args.push(stack.pop().unwrap_or(0.0));
                }
                args.reverse();
                let func_name = funcs.get(*func_idx).map(|s| s.as_str()).unwrap_or("");
                let res = builtins_dispatch(func_name, &args);
                stack.push(res);
                pc += 1;
            }
            Opcode::StoreDx(i) => {
                // for eval, treat like push value (no-op)
                let _ = stack.pop().unwrap_or(0.0);
                pc += 1;
            }
            Opcode::StoreX(i) => {
                let _ = stack.pop().unwrap_or(0.0);
                pc += 1;
            }
            Opcode::StoreY(i) => {
                let _ = stack.pop().unwrap_or(0.0);
                pc += 1;
            }
            Opcode::StoreLocal(i) => {
                let v = stack.pop().unwrap_or(0.0);
                if *i < locals.len() {
                    locals[*i] = v;
                }
                pc += 1;
            }
            Opcode::StoreDxDyn | Opcode::StoreXDyn | Opcode::StoreYDyn => {
                // pop value then index and ignore for eval
                let _v = stack.pop().unwrap_or(0.0);
                let _idxf = stack.pop().unwrap_or(0.0);
                pc += 1;
            }
        }
    }

    stack.pop().unwrap_or(0.0)
}
