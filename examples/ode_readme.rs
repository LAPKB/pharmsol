fn main() {
    use data::Subject;
    use pharmsol::*;

    let subject = Subject::builder("id1")
        .bolus(0.0, 100.0, 0)
        .repeat(2, 0.5)
        .observation(0.5, 0.1, 0)
        .observation(1.0, 0.4, 0)
        .observation(2.0, 1.0, 0)
        .observation(2.5, 1.1, 0)
        .covariate("wt", 0.0, 80.0)
        .covariate("wt", 1.0, 83.0)
        .covariate("age", 0.0, 25.0)
        .build();
    println!("{subject}");
    let ode = simulator::Equation::new_ode(
        |x, p, t, dx, _rateiv, cov| {
            fetch_cov!(cov, t, wt, age);
            fetch_params!(p, ka, ke, _tlag, _v);
            // Secondary Eqs

            let ke = ke * wt.powf(0.75) * (age / 25.0).powf(0.5);

            //Struct
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
        (2, 1),
    );

    let op = ode.simulate_subject(&subject, &vec![0.3, 0.5, 0.1, 70.0], false);
    println!("{op:#?}");
}
