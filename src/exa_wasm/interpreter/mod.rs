mod ast;
mod dispatch;
mod eval;
mod loader;
mod parser;
mod registry;

pub use loader::load_ir_ode;
pub use parser::tokenize;
pub use parser::Parser;
pub use registry::{
    ode_for_id, set_current_expr_id, set_runtime_error, take_runtime_error, unregister_model,
};

// Keep a small set of unit tests that exercise the parser/eval and loader
// wiring. Runtime dispatch and registry behavior live in the `dispatch`
// and `registry` modules respectively.
#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::Vector;
    use crate::exa_wasm::interpreter::eval::eval_expr;

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
        let val = eval_expr(&expr, &x, &pvec, &rateiv, None, Some(0.0), None);
        // numeric result must be finite
        assert!(val.is_finite());
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
