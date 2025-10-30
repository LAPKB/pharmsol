mod ast;
mod builtins;
mod dispatch;
mod eval;
mod loader;
mod loader_helpers;
mod parser;
mod registry;
mod typecheck;
mod vm;

pub use loader::load_ir_ode;
pub use parser::tokenize;
pub use parser::Parser;
pub use registry::{
    ode_for_id, set_current_expr_id, set_runtime_error, take_runtime_error, unregister_model,
};

pub use vm::{run_bytecode, Opcode};

// Re-export some AST and helper symbols for other sibling modules (e.g. build)
pub use ast::{Expr, Lhs, Stmt};
pub use loader_helpers::{extract_closure_body, strip_macro_calls};

// Keep a small set of unit tests that exercise the parser/eval and loader
// wiring. Runtime dispatch and registry behavior live in the `dispatch`
// and `registry` modules respectively.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::exa_wasm::interpreter::eval::eval_expr;
    use diffsol::Vector;

    #[test]
    fn test_tokenize_and_parse_simple() {
        let s = "-ke * x[0] + rateiv[0] / 2";
        let toks = tokenize(s);
        let mut p = Parser::new(toks);
        let expr = p.parse_expr().expect("parse failed");
        // evaluate with dummy vectors
        use crate::simulator::V;
        let x = V::zeros(1, diffsol::NalgebraContext);
        let mut pvec = V::zeros(1, diffsol::NalgebraContext);
        pvec[0] = 3.0; // ke
        let rateiv = V::zeros(1, diffsol::NalgebraContext);
        // evaluation should succeed (ke resolves via pmap not provided -> 0)
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, None, Some(0.0), None);
        // numeric result must be finite
        assert!(val.as_number().is_finite());
    }

    #[test]
    fn test_unknown_function_sets_runtime_error() {
        use crate::exa_wasm::interpreter::eval::eval_call;
        // clear any prior runtime error
        crate::exa_wasm::interpreter::take_runtime_error();
        // call an unknown function
        let val = eval_call("this_function_does_not_exist", &[]);
        // evaluator returns Number(0.0) for unknowns but should set a runtime error
        use crate::exa_wasm::interpreter::eval::Value;
        assert_eq!(val, Value::Number(0.0));
        let err = crate::exa_wasm::interpreter::take_runtime_error();
        assert!(err.is_some(), "expected runtime error for unknown function");
        let msg = err.unwrap();
        assert!(
            msg.contains("unknown function"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn test_eval_call_rejects_wrong_arity() {
        use crate::exa_wasm::interpreter::eval::eval_call;
        use crate::exa_wasm::interpreter::eval::Value;
        // clear any prior runtime error
        crate::exa_wasm::interpreter::take_runtime_error();
        // call pow with wrong arity (should be 2 args)
        let val = eval_call("pow", &[Value::Number(1.0)]);
        assert_eq!(val, Value::Number(0.0));
        let err = crate::exa_wasm::interpreter::take_runtime_error();
        assert!(err.is_some(), "expected runtime error for wrong arity");
        let msg = err.unwrap();
        assert!(
            msg.contains("wrong arity") || msg.contains("unknown function"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn test_loader_errors_on_unknown_function() {
        use std::env;
        use std::fs;
        let tmp = env::temp_dir().join("exa_test_ir_unknown_fn.json");
        let ir_json = serde_json::json!({
            "ir_version": "1.0",
            "kind": "EqnKind::ODE",
            "params": ["ke","v"],
            "diffeq": "|x, p, _t, dx, rateiv, _cov| { dx[0] = foobar(1.0); }",
            "lag": "",
            "fa": "",
            "init": "",
            "out": ""
        });
        let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
        fs::write(&tmp, s.as_bytes()).expect("write tmp");

        let res = crate::exa_wasm::interpreter::loader::load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(
            res.is_err(),
            "loader should reject IR with unknown function calls"
        );
    }

    #[test]
    fn test_macro_parsing_load_ir() {
        use std::env;
        use std::fs;
        let tmp = env::temp_dir().join("exa_test_ir_lag.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = 0.0; }".to_string();
        // lag text contains function calls and commas inside calls
        let lag = Some(
            "|p, t, _cov| { lag!{0 => max(1.0, t * 2.0), 1 => if(t>0, 2.0, 3.0)} }".to_string(),
        );
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            lag,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let res = load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(res.is_ok());
    }

    #[test]
    fn test_emit_ir_includes_diffeq_ast_and_schema() {
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_emit_ir_diffeq_ast_and_schema.json");
        let diffeq =
            "|x, p, _t, dx, rateiv, _cov| { if (t > 0) { dx[0] = 1.0; } else { dx[0] = 2.0; } }"
                .to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec!["ke".to_string()],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();
        assert!(
            v.get("diffeq_ast").is_some(),
            "emit_ir should include diffeq_ast"
        );
        // schema metadata should be present
        assert!(
            v.get("ir_schema").is_some(),
            "emit_ir should include ir_schema"
        );
    }

    #[test]
    fn test_emit_ir_includes_out_and_init_ast() {
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_emit_ir_out_init_ast.json");
        let out = "|x, p, _t, _cov, y| { y[0] = x[0] + 1.0; }".to_string();
        let init = "|p, _t, _cov, x| { x[0] = 0.0; }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            "".to_string(),
            None,
            None,
            Some(init.clone()),
            Some(out.clone()),
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();
        assert!(v.get("out_ast").is_some(), "emit_ir should include out_ast");
        assert!(
            v.get("init_ast").is_some(),
            "emit_ir should include init_ast"
        );
    }

    #[test]
    fn test_emit_ir_includes_bytecode_map_and_vm_exec() {
        use crate::exa_wasm::interpreter::{run_bytecode, Opcode};
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_emit_ir_bytecode.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = ke * 2.0; }".to_string();
        let params = vec!["ke".to_string()];
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            params.clone(),
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();
        // ensure bytecode_map present
        let bc = v
            .get("bytecode_map")
            .expect("bytecode_map should be present")
            .clone();
        // deserialize into map
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc).expect("deserialize bytecode_map");
        assert!(map.contains_key(&0usize));
        let code = map.get(&0usize).unwrap();
        // execute bytecode with p = [3.0]
        let pvals = vec![3.0f64];
        let mut assigned: Option<(usize, f64)> = None;
        run_bytecode(&code, &pvals, |i, v| {
            assigned = Some((i, v));
        });
        assert!(assigned.is_some());
        let (i, val) = assigned.unwrap();
        assert_eq!(i, 0usize);
        assert_eq!(val, 6.0f64);
    }

    #[test]
    fn test_loader_rewrites_params_to_param_nodes() {
        use crate::exa_wasm::interpreter::ast::{Expr, Stmt};
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_ir_param_rewrite.json");
        let ir_json = serde_json::json!({
            "ir_version": "1.0",
            "kind": "EqnKind::ODE",
            "params": ["ke", "v"],
            "diffeq": "|x, p, _t, dx, rateiv, _cov| { dx[0] = ke * x[0]; }",
            "lag": "",
            "fa": "",
            "init": "",
            "out": ""
        });
        let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
        fs::write(&tmp, s.as_bytes()).expect("write tmp");

        let res = crate::exa_wasm::interpreter::loader::load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(res.is_ok(), "loader should accept valid IR");
        let (_ode, _meta, id) = res.unwrap();
        let entry = crate::exa_wasm::interpreter::registry::get_entry(id).expect("entry");

        fn contains_param_in_expr(e: &Expr, idx: usize) -> bool {
            match e {
                Expr::Param(i) => *i == idx,
                Expr::BinaryOp { lhs, rhs, .. } => {
                    contains_param_in_expr(lhs, idx) || contains_param_in_expr(rhs, idx)
                }
                Expr::UnaryOp { rhs, .. } => contains_param_in_expr(rhs, idx),
                Expr::Call { args, .. } => args.iter().any(|a| contains_param_in_expr(a, idx)),
                Expr::MethodCall { receiver, args, .. } => {
                    contains_param_in_expr(receiver, idx)
                        || args.iter().any(|a| contains_param_in_expr(a, idx))
                }
                Expr::Indexed(_, idx_expr) => contains_param_in_expr(idx_expr, idx),
                Expr::Ternary {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    contains_param_in_expr(cond, idx)
                        || contains_param_in_expr(then_branch, idx)
                        || contains_param_in_expr(else_branch, idx)
                }
                _ => false,
            }
        }

        fn contains_param(stmt: &Stmt, idx: usize) -> bool {
            match stmt {
                Stmt::Assign(_, rhs) => contains_param_in_expr(rhs, idx),
                Stmt::Block(v) => v.iter().any(|s| contains_param(s, idx)),
                Stmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    contains_param(then_branch, idx)
                        || else_branch
                            .as_ref()
                            .map(|b| contains_param(b, idx))
                            .unwrap_or(false)
                }
                Stmt::Expr(e) => contains_param_in_expr(e, idx),
            }
        }

        assert!(
            entry.diffeq_stmts.iter().any(|s| contains_param(s, 0)),
            "expected Param(0) in diffeq stmts"
        );
    }

    #[test]
    fn test_eval_param_expr() {
        use crate::exa_wasm::interpreter::ast::Expr;
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::simulator::V;

        let expr = Expr::Param(0);
        // create simple vectors
        use diffsol::NalgebraContext;
        let x = V::zeros(1, NalgebraContext);
        let mut p = V::zeros(1, NalgebraContext);
        p[0] = 3.1415;
        let rateiv = V::zeros(1, NalgebraContext);

        let val = eval_expr(&expr, &x, &p, &rateiv, None, None, Some(0.0), None);
        assert_eq!(val.as_number(), 3.1415);
    }

    #[test]
    fn test_loader_accepts_preparsed_ast_in_ir() {
        use crate::exa_wasm::interpreter::ast::{Expr, Lhs, Stmt};
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_ir_preparsed_ast.json");
        // build a tiny diffeq AST: dx[0] = 1.0;
        let lhs = Lhs::Indexed("dx".to_string(), Box::new(Expr::Number(0.0)));
        let stmt = Stmt::Assign(lhs, Expr::Number(1.0));
        let diffeq_ast = vec![stmt];

        let ir_json = serde_json::json!({
            "ir_version": "1.0",
            "kind": "EqnKind::ODE",
            "params": [],
            "diffeq": "",
            "diffeq_ast": diffeq_ast,
            "lag": "",
            "fa": "",
            "init": "",
            "out": ""
        });
        let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
        fs::write(&tmp, s.as_bytes()).expect("write tmp");

        let res = crate::exa_wasm::interpreter::loader::load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(
            res.is_ok(),
            "loader should accept IR with pre-parsed diffeq_ast"
        );
    }

    #[test]
    fn test_loader_rejects_builtin_wrong_arity() {
        use std::env;
        use std::fs;
        let tmp = env::temp_dir().join("exa_test_ir_bad_arity.json");
        let ir_json = serde_json::json!({
            "ir_version": "1.0",
            "kind": "EqnKind::ODE",
            "params": ["ke"],
            "diffeq": "|x, p, _t, dx, rateiv, _cov| { dx[0] = pow(1.0); }",
            "lag": "",
            "fa": "",
            "init": "",
            "out": ""
        });
        let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
        fs::write(&tmp, s.as_bytes()).expect("write tmp");

        let res = crate::exa_wasm::interpreter::loader::load_ir_ode(tmp.clone());
        fs::remove_file(tmp).ok();
        assert!(
            res.is_err(),
            "loader should reject builtin calls with wrong arity"
        );
    }

    mod load_negative_tests {
        use super::*;
        use std::env;
        use std::fs;

        #[test]
        fn test_loader_errors_when_missing_structured_maps() {
            let tmp = env::temp_dir().join("exa_test_ir_negative.json");
            let ir_json = serde_json::json!({
                "ir_version": "1.0",
                "kind": "EqnKind::ODE",
                "params": ["ke", "v"],
                "diffeq": "|x, p, _t, dx, rateiv, _cov| { dx[0] = -ke * x[0] + rateiv[0]; }",
                "lag": "|p, t, _cov| { lag!{0 => t} }",
                "fa": "|p, t, _cov| { fa!{0 => 0.1} }",
                "init": "|p, _t, _cov, x| { }",
                "out": "|x, p, _t, _cov, y| { y[0] = x[0]; }"
            });
            let s = serde_json::to_string_pretty(&ir_json).expect("serialize");
            fs::write(&tmp, s.as_bytes()).expect("write tmp");

            let res = load_ir_ode(tmp.clone());
            fs::remove_file(tmp).ok();
            assert!(
                res.is_err(),
                "loader should reject IR missing structured maps"
            );
        }
    }
}
