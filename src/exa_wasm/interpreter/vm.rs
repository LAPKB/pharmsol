use serde::{Deserialize, Serialize};

/// A tiny stack-based bytecode for proof-of-concept evaluation.
/// Opcodes are intentionally minimal for the POC.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Opcode {
    PushConst(f64),   // push constant
    LoadParam(usize), // push p[idx]
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    StoreDx(usize), // pop value and assign to dx[index]
}

/// Execute a sequence of opcodes.
/// - `p` is the parameter vector
/// - `assign_dx` is a closure to receive (idx, value)
pub fn run_bytecode<F>(code: &[Opcode], p: &[f64], mut assign_dx: F)
where
    F: FnMut(usize, f64),
{
    let mut stack: Vec<f64> = Vec::new();
    for op in code.iter() {
        match op {
            Opcode::PushConst(v) => stack.push(*v),
            Opcode::LoadParam(i) => {
                let v = if *i < p.len() { p[*i] } else { 0.0 };
                stack.push(v);
            }
            Opcode::Add => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a + b);
            }
            Opcode::Sub => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a - b);
            }
            Opcode::Mul => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a * b);
            }
            Opcode::Div => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a / b);
            }
            Opcode::Pow => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.powf(b));
            }
            Opcode::StoreDx(i) => {
                let v = stack.pop().unwrap_or(0.0);
                assign_dx(*i, v);
            }
        }
    }
}
