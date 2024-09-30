fn main() {
    use pharmsol::*;

    let subject = Subject::builder("1")
        .infusion(0., 500.0, 0, 0.5)
        .observation(0.5, 1.6457759999999999, 0)
        .observation(1., 1.216442, 0)
        .observation(2., 0.46227289999999999, 0)
        .observation(3., 0.1697458, 0)
        .observation(4., 0.063821779999999995, 0)
        .observation(6., 0.0090993840000000003, 0)
        .observation(8., 0.001017932, 0)
        .build();

    let an = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

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

    let net = equation::ODENet::new(
        vec![dmatrix![-1.0], dmatrix![0.0]],
        vec![],
        vec![],
        vec![OutEq::new(0, Div(X(0), P(1)))],
        (1, 1),
    );
    let em = ErrorModel::new((0.0, 0.05, 0.0, 0.0), 0.0, &ErrorType::Add);
    let ll = an.estimate_likelihood(
        &subject,
        &vec![1.02282724609375, 194.51904296875],
        &em,
        false,
    );
    let op = an.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!(
        "Analytical: \n-2ll:{:#?}\n{:#?}",
        -2.0 * ll,
        op.flat_predictions()
    );

    let ll = ode.estimate_likelihood(
        &subject,
        &vec![1.02282724609375, 194.51904296875],
        &em,
        false,
    );
    let op = ode.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!("ODE: \n-2ll:{:#?}\n{:#?}", -2.0 * ll, op.flat_predictions());

    let ll = net.estimate_likelihood(
        &subject,
        &vec![1.02282724609375, 194.51904296875],
        &em,
        false,
    );
    let op = net.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!(
        "ODENet: \n-2ll:{:#?}\n{:#?}",
        -2.0 * ll,
        op.flat_predictions()
    );
}
