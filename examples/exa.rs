//cargo run --example exa --features exa

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::*;
    use std::path::PathBuf;
    let subject = Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 1.645776, 0)
        .observation(1.0, 1.216442, 0)
        .observation(2.0, 0.4622729, 0)
        .observation(3.0, 0.1697458, 0)
        .observation(4.0, 0.06382178, 0)
        .observation(6.0, 0.009099384, 0)
        .observation(8.0, 0.001017932, 0)
        .build();

    let ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let model_path = exa::build::compile(
        format!(
            r#"
            equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {{
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        }},
        |_p| lag! {{}},
        |_p| fa! {{}},
        |_p, _t, _cov, _x| {{}},
        |x, p, _t, _cov, y| {{
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        }},
        (1, 1),
    )
    "#
        ),
        Some(PathBuf::from("model.pkm")),
        vec!["ke".to_string(), "v".to_string()],
        |a, b| println!("{}: {}", a, b),
    )
    .unwrap();
    let model_path = PathBuf::from(model_path);
    let (_lib, (dyn_ode, _meta)) = unsafe { exa::load::load_ode(model_path) };
    let dyn_op = dyn_ode.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!("Dyn_model: {:#?}", dyn_op.flat_predictions());

    let op = ode.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!("ODE: {:#?}", op.flat_predictions());
}

#[cfg(not(feature = "exa"))]
fn main() {
    panic!("This example requires the 'exa' feature. Please run with `cargo run --example exa --features exa`");
}
