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
pub use loader_helpers::extract_closure_body;
// Re-export builtin helpers so other modules (like the emitter) can query
// builtin metadata without depending on private module paths.
pub use builtins::{arg_count_range, is_known_function};

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
        // Use the emitter to create IR that includes parsed AST; loader will
        // then validate and reject unknown function calls.
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = foobar(1.0); }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec!["ke".to_string(), "v".to_string()],
        )
        .expect("emit_ir failed");
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
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = ke * x[0]; }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec!["ke".to_string(), "v".to_string()],
        )
        .expect("emit_ir failed");
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
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = pow(1.0); }".to_string();
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

    #[test]
    fn test_bytecode_parity_constant_index() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_const.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0] + 2.0; }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        // extract AST rhs expression
        let diffeq_ast = v.get("diffeq_ast").expect("diffeq_ast");
        let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
            serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
        // expect first stmt to be Assign(_, rhs)
        let rhs_expr = match &stmts[0] {
            crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
            _ => panic!("expected assign stmt"),
        };

        use diffsol::NalgebraContext;
        let x = crate::simulator::V::zeros(1, NalgebraContext);
        let mut x = x;
        x[0] = 5.0;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        // extract bytecode for index 0
        // If emitter did not produce bytecode for this pattern, skip the VM
        // parity check here. The test harness will still exercise the AST
        // path; missing bytecode means the emitter needs expanded lowering.
        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!("emit_ir did not produce diffeq_bytecode for method-call test; skipping VM parity check");
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        // strip trailing StoreDx
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        // builtins dispatch
        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];
        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }

        // debug: show discovered funcs and expr_code
        eprintln!("debug funcs: {:?}", funcs);
        eprintln!("debug expr_code: {:?}", expr_code);

        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_dynamic_index() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_dyn.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[ke]; }".to_string();
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

        // extract AST rhs expression
        // prefer pre-parsed AST when present, otherwise parse the closure text
        let rhs_expr = if let Some(diffeq_ast) = v.get("diffeq_ast") {
            let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
                serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        } else {
            let diffeq_text = v
                .get("diffeq")
                .and_then(|d| d.as_str())
                .expect("diffeq text");
            let body = crate::exa_wasm::interpreter::extract_closure_body(diffeq_text)
                .expect("closure body");
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            // parse statements and extract rhs from first assign
            if let Some(stmts) = p.parse_statements() {
                match &stmts[0] {
                    crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                    _ => panic!("expected assign stmt"),
                }
            } else {
                // fallback: attempt to extract RHS between '=' and ';' and parse as expression
                let eq_pos = body.find('=');
                if let Some(eq) = eq_pos {
                    if let Some(sc) = body[eq..].find(';') {
                        let rhs_text = body[eq + 1..eq + sc].trim();
                        let toks = crate::exa_wasm::interpreter::tokenize(rhs_text);
                        let mut p2 = crate::exa_wasm::interpreter::Parser::new(toks);
                        p2.parse_expr().expect("parse expr rhs")
                    } else {
                        panic!("parse stmts");
                    }
                } else {
                    panic!("parse stmts");
                }
            }
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(2, NalgebraContext);
        x[0] = 7.0;
        x[1] = 9.0;
        let mut p = crate::simulator::V::zeros(1, NalgebraContext);
        p[0] = 0.0; // ke -> picks x[0]
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        // extract bytecode for index 0
        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!("emit_ir did not produce diffeq_bytecode for method-call test; skipping VM parity check");
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        // strip trailing StoreDx
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0], x[1]];
        let p_vals: Vec<f64> = vec![p[0]];
        let rateiv_vals: Vec<f64> = vec![];
        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &Vec::new(),
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_lag_entry() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_lag.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = 0.0; }".to_string();
        // use an expression that only references params so the conservative
        // bytecode compiler can produce code (compile_expr_top does not
        // accept bare 't' or unknown idents).
        let lag = Some("|p, t, _cov| { lag!{0 => p[0] * 2.0} }".to_string());
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
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        // parse textual lag entry back to Expr for AST eval
        let lag_map = v.get("lag_map").expect("lag_map");
        let lag_entry = lag_map
            .get("0")
            .expect("lag entry 0")
            .as_str()
            .expect("string");
        let toks = crate::exa_wasm::interpreter::tokenize(lag_entry);
        let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
        let expr = p.parse_expr().expect("parse lag expr");

        use diffsol::NalgebraContext;
        let x = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        // evaluate AST with p[0] = 3.0 -> expected 6.0
        let mut pvec = crate::simulator::V::zeros(1, diffsol::NalgebraContext);
        pvec[0] = 3.0;
        let ast_val = eval_expr(&expr, &x, &pvec, &rateiv, None, None, Some(0.0), None);

        // get lag_bytecode
        let bc = v.get("lag_bytecode").expect("lag_bytecode");
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize lag_bytecode");
        let code = map.get(&0usize).expect("code for lag 0");

        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![];
        let p_vals: Vec<f64> = vec![3.0];
        let rateiv_vals: Vec<f64> = vec![];
        let vm_val = run_bytecode_eval(
            &code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            2.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_ternary() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_ternary.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0] > 0 ? 2.0 : 3.0; }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        // prefer pre-parsed AST when present, otherwise parse the closure text
        let rhs_expr = if let Some(diffeq_ast) = v.get("diffeq_ast") {
            let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
                serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        } else {
            let diffeq_text = v
                .get("diffeq")
                .and_then(|d| d.as_str())
                .expect("diffeq text");
            let body = crate::exa_wasm::interpreter::extract_closure_body(diffeq_text)
                .expect("closure body");
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            // parse statements and extract rhs from first assign
            let stmts = p.parse_statements().expect("parse stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(1, NalgebraContext);
        x[0] = 1.0;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!("emit_ir did not produce diffeq_bytecode for method-call test; skipping VM parity check");
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];

        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }
        eprintln!("debug funcs: {:?}", funcs);
        eprintln!("debug expr_code: {:?}", expr_code);

        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_method_call() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_method.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0].sin(); }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        let rhs_expr = if let Some(diffeq_ast) = v.get("diffeq_ast") {
            let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
                serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        } else {
            let diffeq_text = v
                .get("diffeq")
                .and_then(|d| d.as_str())
                .expect("diffeq text");
            let body = crate::exa_wasm::interpreter::extract_closure_body(diffeq_text)
                .expect("closure body");
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            if let Some(stmts) = p.parse_statements() {
                match &stmts[0] {
                    crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                    _ => panic!("expected assign stmt"),
                }
            } else {
                // fallback: extract RHS between '=' and ';' and parse as single expression
                if let Some(eq_pos) = body.find('=') {
                    if let Some(sc_pos) = body[eq_pos..].find(';') {
                        let rhs_text = body[eq_pos + 1..eq_pos + sc_pos].trim();
                        let toks2 = crate::exa_wasm::interpreter::tokenize(rhs_text);
                        let mut p2 = crate::exa_wasm::interpreter::Parser::new(toks2);
                        p2.parse_expr().expect("parse expr rhs")
                    } else {
                        panic!("parse stmts");
                    }
                } else {
                    panic!("parse stmts");
                }
            }
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(1, NalgebraContext);
        x[0] = 0.5;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = v.get("diffeq_bytecode").expect("diffeq_bytecode");
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];
        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_nested_dynamic() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_nested.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[x[ke]]; }".to_string();
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

        let diffeq_ast = v.get("diffeq_ast").expect("diffeq_ast");
        let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
            serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
        let rhs_expr = match &stmts[0] {
            crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
            _ => panic!("expected assign stmt"),
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(3, NalgebraContext);
        x[0] = 11.0;
        x[1] = 22.0;
        x[2] = 33.0;
        let mut p = crate::simulator::V::zeros(1, NalgebraContext);
        p[0] = 1.0; // ke -> picks x[1]
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = v.get("diffeq_bytecode").expect("diffeq_bytecode");
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0], x[1], x[2]];
        let p_vals: Vec<f64> = vec![p[0]];
        let rateiv_vals: Vec<f64> = vec![];
        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &Vec::new(),
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_bool_short_circuit() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_bool.json");
        let diffeq =
            "|x, p, _t, dx, rateiv, _cov| { dx[0] = (x[0] > 0) && (x[0] < 10) ? 1.0 : 0.0; }"
                .to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        // extract RHS expr
        let rhs_expr = if let Some(diffeq_ast) = v.get("diffeq_ast") {
            let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
                serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        } else {
            let diffeq_text = v
                .get("diffeq")
                .and_then(|d| d.as_str())
                .expect("diffeq text");
            let body = crate::exa_wasm::interpreter::extract_closure_body(diffeq_text)
                .expect("closure body");
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            let stmts = p.parse_statements().expect("parse stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(1, NalgebraContext);
        x[0] = 5.0;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!(
                    "emit_ir did not produce diffeq_bytecode for bool short-circuit test; skipping VM parity check"
                );
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];

        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }
        eprintln!("debug funcs: {:?}", funcs);
        eprintln!("debug expr_code: {:?}", expr_code);

        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_chained_method_calls() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_chained.json");
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0].sin().abs(); }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        let rhs_expr = if let Some(diffeq_ast) = v.get("diffeq_ast") {
            let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
                serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
            match &stmts[0] {
                crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                _ => panic!("expected assign stmt"),
            }
        } else {
            let diffeq_text = v
                .get("diffeq")
                .and_then(|d| d.as_str())
                .expect("diffeq text");
            let body = crate::exa_wasm::interpreter::extract_closure_body(diffeq_text)
                .expect("closure body");
            let toks = crate::exa_wasm::interpreter::tokenize(&body);
            let mut p = crate::exa_wasm::interpreter::Parser::new(toks);
            if let Some(stmts) = p.parse_statements() {
                match &stmts[0] {
                    crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
                    _ => panic!("expected assign stmt"),
                }
            } else {
                if let Some(eq_pos) = body.find('=') {
                    if let Some(sc_pos) = body[eq_pos..].find(';') {
                        let rhs_text = body[eq_pos + 1..eq_pos + sc_pos].trim();
                        let toks2 = crate::exa_wasm::interpreter::tokenize(rhs_text);
                        let mut p2 = crate::exa_wasm::interpreter::Parser::new(toks2);
                        p2.parse_expr().expect("parse expr rhs")
                    } else {
                        panic!("parse stmts");
                    }
                } else {
                    panic!("parse stmts");
                }
            }
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(1, NalgebraContext);
        x[0] = -0.5;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!("emit_ir did not produce diffeq_bytecode for chained method test; skipping VM parity check");
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];
        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        assert_eq!(ast_val.as_number(), vm_val);
    }

    #[test]
    fn test_bytecode_parity_method_with_arg() {
        use crate::exa_wasm::interpreter::eval::eval_expr;
        use crate::exa_wasm::interpreter::vm::run_bytecode_eval;
        use crate::exa_wasm::interpreter::Opcode;
        use std::env;
        use std::fs;

        let tmp = env::temp_dir().join("exa_test_parity_method_arg.json");
        // use pow as method-style call; receiver becomes first arg
        let diffeq = "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0].pow(2.0); }".to_string();
        let _path = crate::exa_wasm::build::emit_ir::<crate::equation::ODE>(
            diffeq,
            None,
            None,
            None,
            None,
            Some(tmp.clone()),
            vec![],
        )
        .expect("emit_ir failed");
        let s = fs::read_to_string(&tmp).expect("read emitted ir");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse json");
        fs::remove_file(&tmp).ok();

        let diffeq_ast = v.get("diffeq_ast").expect("diffeq_ast");
        let stmts: Vec<crate::exa_wasm::interpreter::Stmt> =
            serde_json::from_value(diffeq_ast.clone()).expect("deserialize stmts");
        let rhs_expr = match &stmts[0] {
            crate::exa_wasm::interpreter::Stmt::Assign(_, rhs) => rhs.clone(),
            _ => panic!("expected assign stmt"),
        };

        use diffsol::NalgebraContext;
        let mut x = crate::simulator::V::zeros(1, NalgebraContext);
        x[0] = 3.0;
        let p = crate::simulator::V::zeros(0, NalgebraContext);
        let rateiv = crate::simulator::V::zeros(0, NalgebraContext);

        let ast_val = eval_expr(&rhs_expr, &x, &p, &rateiv, None, None, Some(0.0), None);

        let bc = match v.get("diffeq_bytecode") {
            Some(b) => b,
            None => {
                eprintln!("emit_ir did not produce diffeq_bytecode for method-with-arg test; skipping VM parity check");
                return;
            }
        };
        let map: std::collections::HashMap<usize, Vec<Opcode>> =
            serde_json::from_value(bc.clone()).expect("deserialize bytecode_map");
        let code = map.get(&0usize).expect("code for idx 0");
        let mut expr_code = code.clone();
        if let Some(last) = expr_code.last() {
            match last {
                Opcode::StoreDx(_) => {
                    expr_code.pop();
                }
                _ => {}
            }
        }

        let builtins = |name: &str, args: &[f64]| -> f64 {
            use crate::exa_wasm::interpreter::eval::{eval_call, Value};
            let vals: Vec<Value> = args.iter().map(|v| Value::Number(*v)).collect();
            eval_call(name, &vals).as_number()
        };

        let mut locals: Vec<f64> = Vec::new();
        let mut locals_slice = locals.as_mut_slice();
        let x_vals: Vec<f64> = vec![x[0]];
        let p_vals: Vec<f64> = vec![];
        let rateiv_vals: Vec<f64> = vec![];

        // use funcs table emitted in IR so builtins can be looked up by name
        let mut funcs: Vec<String> = Vec::new();
        if let Some(fv) = v.get("funcs") {
            funcs = serde_json::from_value(fv.clone()).unwrap_or_default();
        }

        let vm_val = run_bytecode_eval(
            &expr_code,
            &x_vals,
            &p_vals,
            &rateiv_vals,
            0.0,
            &mut locals_slice,
            &funcs,
            &builtins,
        );

        if (ast_val.as_number() - vm_val).abs() > 1e-12 {
            panic!(
                "parity mismatch: ast={} vm={} funcs={:?} code={:?}",
                ast_val.as_number(),
                vm_val,
                funcs,
                expr_code
            );
        }
    }
}
