use std::env;
use std::fs;
// example: emit IR and load via the runtime

fn main() {
    let tmp = env::temp_dir();

    // Model 1: simple dx assignment
    let diffeq1 = "|x, p, _t, dx, rateiv, _cov| { dx[0] = -ke * x[0]; }".to_string();
    let path1 = tmp.join("exa_example_model1.json");
    let _ = pharmsol::exa_wasm::build::emit_ir::<pharmsol::equation::ODE>(
        diffeq1,
        None,
        None,
        None,
        None,
        Some(path1.clone()),
        vec!["ke".to_string()],
    ) .expect("emit_ir model1");

    // Model 2: prelude/local and rate
    let diffeq2 = "|x, p, _t, dx, rateiv, _cov| { ke = 0.5; dx[0] = -ke * x[0] + rateiv[0]; }".to_string();
    let path2 = tmp.join("exa_example_model2.json");
    let _ = pharmsol::exa_wasm::build::emit_ir::<pharmsol::equation::ODE>(
        diffeq2,
        None,
        None,
        None,
        None,
        Some(path2.clone()),
        vec!["ke".to_string()],
    ) .expect("emit_ir model2");

    // Model 3: builtin and ternary
    let diffeq3 = "|x, p, _t, dx, rateiv, _cov| { dx[0] = if(t>0, exp(-ke * t) * x[0], 0.0); }".to_string();
    let path3 = tmp.join("exa_example_model3.json");
    let _ = pharmsol::exa_wasm::build::emit_ir::<pharmsol::equation::ODE>(
        diffeq3,
        None,
        None,
        None,
        None,
        Some(path3.clone()),
        vec!["ke".to_string()],
    ) .expect("emit_ir model3");

    println!("Emitted IR to:\n  {:?}\n  {:?}\n  {:?}", path1, path2, path3);

    // Load them via the public API and print emitted IR metadata from the
    // emitted JSON (avoids accessing private registry internals from an
    // example binary).
    for p in [&path1, &path2, &path3] {
        // try to load via runtime loader (public re-export)
        match pharmsol::exa_wasm::load_ir_ode(p.clone()) {
            Ok((_ode, _meta, id)) => {
                println!("loader accepted model, registry id={}", id);
            }
            Err(e) => {
                eprintln!("loader rejected model {:?}: {}", p, e);
            }
        }

        // read raw IR and display bytecode/funcs/locals metadata
        match fs::read_to_string(p) {
            Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(v) => {
                    let has_bc = v.get("diffeq_bytecode").is_some();
                    let funcs = v
                        .get("funcs")
                        .and_then(|j| j.as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>()))
                        .unwrap_or_default();
                    let locals = v
                        .get("locals")
                        .and_then(|j| j.as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>()))
                        .unwrap_or_default();
                    println!("IR {:?}: diffeq_bytecode={} funcs={:?} locals={:?}", p.file_name().unwrap_or_default(), has_bc, funcs, locals);
                }
                Err(e) => eprintln!("failed to parse emitted IR {:?}: {}", p, e),
            },
            Err(e) => eprintln!("failed to read emitted IR {:?}: {}", p, e),
        }
    }

    // cleanup
    let _ = fs::remove_file(&path1);
    let _ = fs::remove_file(&path2);
    let _ = fs::remove_file(&path3);
}
