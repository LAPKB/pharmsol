use crate::exa_wasm::interpreter::ast::{Expr, Lhs, Stmt};

#[derive(Debug, PartialEq)]
pub enum Type {
    Number,
    Bool,
}

pub enum TypeError {
    UnknownFunction(String),
    Arity {
        name: String,
        expected: String,
        got: usize,
    },
    IndexNotNumeric,
    AssignBooleanToIndexed(String),
    Msg(String),
}

impl std::fmt::Debug for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::UnknownFunction(n) => write!(f, "UnknownFunction({})", n),
            TypeError::Arity {
                name,
                expected,
                got,
            } => write!(
                f,
                "Arity {{ name: {}, expected: {}, got: {} }}",
                name, expected, got
            ),
            TypeError::IndexNotNumeric => write!(f, "IndexNotNumeric"),
            TypeError::AssignBooleanToIndexed(n) => write!(f, "AssignBooleanToIndexed({})", n),
            TypeError::Msg(s) => write!(f, "Msg({})", s),
        }
    }
}

impl From<String> for TypeError {
    fn from(s: String) -> Self {
        TypeError::Msg(s)
    }
}

fn type_of_binary_op(_lhs: &Type, op: &str, _rhs: &Type) -> Result<Type, TypeError> {
    use Type::*;
    match op {
        "&&" | "||" => Ok(Bool),
        "<" | ">" | "<=" | ">=" | "==" | "!=" => Ok(Bool),
        "+" | "-" | "*" | "/" | "^" => Ok(Number),
        _ => Ok(Number),
    }
}

// Minimal conservative type checker
pub fn check_expr(expr: &Expr) -> Result<Type, TypeError> {
    use Expr::*;
    match expr {
        Bool(_) => Ok(Type::Bool),
        Number(_) => Ok(Type::Number),
        Ident(_) => Ok(Type::Number),
        Param(_) => Ok(Type::Number),
        Indexed(_, idx) => match check_expr(idx)? {
            Type::Number => Ok(Type::Number),
            _ => Err(TypeError::IndexNotNumeric),
        },
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
        Call { name, args } => {
            // ensure args type-check
            for a in args.iter() {
                let _ = check_expr(a)?;
            }
            // check known builtin and arity via shared builtins module
            if !crate::exa_wasm::interpreter::builtins::is_known_function(name) {
                return Err(TypeError::UnknownFunction(name.clone()));
            }
            if let Some(range) = crate::exa_wasm::interpreter::builtins::arg_count_range(name) {
                if !range.contains(&args.len()) {
                    let lo = *range.start();
                    let hi = *range.end();
                    let expect = if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{}..={}", lo, hi)
                    };
                    return Err(TypeError::Arity {
                        name: name.clone(),
                        expected: expect,
                        got: args.len(),
                    });
                }
            }
            Ok(Type::Number)
        }
        MethodCall {
            receiver,
            name: _,
            args,
        } => {
            let _ = check_expr(receiver)?;
            for a in args.iter() {
                let _ = check_expr(a)?;
            }
            Ok(Type::Number)
        }
        Ternary {
            cond,
            then_branch,
            else_branch,
        } => match check_expr(cond)? {
            Type::Bool | Type::Number => {
                let t1 = check_expr(then_branch)?;
                let t2 = check_expr(else_branch)?;
                if t1 == t2 {
                    Ok(t1)
                } else {
                    Ok(Type::Number)
                }
            }
        },
    }
}

pub fn check_stmt(stmt: &Stmt) -> Result<(), TypeError> {
    use Stmt::*;
    match stmt {
        Expr(e) => {
            let _ = check_expr(e)?;
            Ok(())
        }
        Assign(lhs, rhs) => match lhs {
            Lhs::Ident(_) => {
                let _ = check_expr(rhs)?;
                Ok(())
            }
            Lhs::Indexed(name, idx_expr) => {
                match check_expr(idx_expr)? {
                    Type::Number => {}
                    _ => return Err(TypeError::IndexNotNumeric),
                }
                match check_expr(rhs)? {
                    Type::Number => Ok(()),
                    Type::Bool => Err(TypeError::AssignBooleanToIndexed(name.clone())),
                }
            }
        },
        Block(v) => {
            for s in v.iter() {
                check_stmt(s)?;
            }
            Ok(())
        }
        If {
            cond,
            then_branch,
            else_branch,
        } => {
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
