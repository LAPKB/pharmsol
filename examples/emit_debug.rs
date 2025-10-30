fn main() {
    use pharmsol::equation;
    use pharmsol::exa_wasm::build::emit_ir;

    // Simple helper example that emits IR for a small model and prints the
    // location of the generated IR file. Keep example minimal and only use
    // public APIs so it doesn't depend on internal interpreter modules.
    let out = emit_ir::<equation::ODE>(
        "|x, p, _t, dx, rateiv, _cov| { dx[0] = x[0].sin(); }".to_string(),
        None,
        None,
        None,
        None,
        None,
        vec![],
    )
    .expect("emit_ir");
    println!("wrote IR to: {}", out);
}
