use pharmsol::*;

fn example_subject() -> Subject {
    Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(3.0, 0.1, 0)
        .observation(6.0, 0.4, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 1.1, 0)
        .build()
}
fn main() {
    let subject = example_subject();
    let ode = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _tlag, _v);
            dbg!(&dx);
            dbg!(&x);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 2),
    );
    let pred = ode.estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 70.0]);
    println!("{:?}", pred);
}
