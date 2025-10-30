use crate::exa_wasm::interpreter::ast::{Expr, Stmt, Lhs};

#[derive(Debug, PartialEq)]
pub enum Type {
    Number,
    Bool,
}

#[derive(Debug)]
pub struct TypeError(pub String);

impl From<String> for TypeError {
    fn from(s: String) -> Self {
        TypeError(s)
    }
}

// Very small, conservative type-checker: it walks expressions/statements and
// reports obvious mismatches. It intentionally accepts coercions that the
// evaluator also accepts (number <-> bool coercion), but flags use of boolean
// results where numeric-only result is required (for example, assigning a
// boolean into dx/x/y indexed targets).

fn type_of_binary_op(lhs: &Type, op: &str, rhs: &Type) -> Result<Type, TypeError> {
    use Type::*;
    match op {
        "&&" | "||" => Ok(Bool),
        "<" | ">" | "<=" | ">=" | "==" | "!=" => Ok(Bool),
        "+" | "-" | "*" | "/" | "^" => Ok(Number),
        _ => Ok(Number),
    }
}

pub fn check_expr(expr: &Expr) -> Result<Type, TypeError> {
    use Expr::*;
    match expr {
        Bool(_) => Ok(Type::Bool),
        Number(_) => Ok(Type::Number),
        Ident(_) => Ok(Type::Number), // identifiers resolve to numbers or coercible values
        Indexed(_, idx) => {
            // index expression must be numeric
            match check_expr(idx)? {
                Type::Number => Ok(Type::Number),
                _ => Err(TypeError("index expression must be numeric".to_string())),
            }
        }
        UnaryOp { op, rhs } => {
            let t = check_expr(rhs)?;
            match op.as_str() {
                "!" => Ok(Type::Bool),
                "-" => Ok(Type::Number),
                _ => Ok(t),
            }
        }
        BinaryOp { lhs, op, rhs } => {
            let lt = check_expr(lhs)?;
            let rt = check_expr(rhs)?;
            type_of_binary_op(&lt, op, &rt)
        }
        Call { name: _, args } => {
            // assume numeric-returning functions unless the name is known
            for a in args.iter() {
                let _ = check_expr(a)?; // ensure args type-check
            }
            Ok(Type::Number)
        }
        MethodCall { receiver, name: _, args } => {
            let _ = check_expr(receiver)?;
            for a in args.iter() {
                let _ = check_expr(a)?;
            }
            Ok(Type::Number)
        }
        Ternary { cond, then_branch, else_branch } => {
            match check_expr(cond)? {
                Type::Bool | Type::Number => {
                    let t1 = check_expr(then_branch)?;
                    let t2 = check_expr(else_branch)?;
                    // if branches disagree, prefer Number (coercion)
                    if t1 == t2 { Ok(t1) } else { Ok(Type::Number) }
                }
            }
        }
    }
}

pub fn check_stmt(stmt: &Stmt) -> Result<(), TypeError> {
    use Stmt::*;
    match stmt {
        Expr(e) => {
            let _ = check_expr(e)?;
            Ok(())
        }
        Assign(lhs, rhs) => {
            // lhs type: if assigning into indexed target x/dx/y -> numeric required
            match lhs {
                Lhs::Ident(_) => {
                    let _ = check_expr(rhs)?;
                    Ok(())
                }
                Lhs::Indexed(name, idx_expr) => {
                    // index expression numeric
                    match check_expr(idx_expr)? {
                        Type::Number => {}
                        _ => return Err(TypeError("index expression must be numeric".to_string())),
                    }
                    // rhs must be numeric for indexed assignment
                    match check_expr(rhs)? {
                        Type::Number => Ok(()),
                        Type::Bool => Err(TypeError(format!("cannot assign boolean to indexed target '{}'", name))),
                    }
                }
            }
        }
        Block(v) => {
            for s in v.iter() {
                check_stmt(s)?;
            }
            Ok(())
        }
        If { cond, then_branch, else_branch } => {
            // condition must be boolean or numeric (coercible) — allow both
            match check_expr(cond)? {
                Type::Bool | Type::Number => {}
            }
            check_stmt(then_branch)?;
            if let Some(eb) = else_branch {
                check_stmt(eb)?;
            }
            Ok(())
        }
    }
}

pub fn check_statements(stmts: &[Stmt]) -> Result<(), TypeError> {
    for s in stmts.iter() {
        check_stmt(s)?;
    }
    Ok(())
}
