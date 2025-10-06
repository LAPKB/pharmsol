fn main() {
    use pharmsol::*;

    let subject = Subject::builder("1")
        .infusion(0., 500.0, 0, 0.5)
        .observation(0.5, 1.645_776, 0)
        .observation(1., 1.216442, 0)
        .observation(2., 0.462_272_9, 0)
        .observation(3., 0.1697458, 0)
        .observation(4., 0.063_821_78, 0)
        .observation(6., 0.009_099_384, 0)
        .observation(8., 0.001017932, 0)
        .missing_observation(12.0, 0)
        .build();

    let an = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let mut ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();

    ems = ems
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();
    // Compute likelihoods and predictions for both models
    let ll_an = an
        .estimate_likelihood(
            &subject,
            &vec![1.02282724609375, 194.51904296875],
            &ems,
            false,
        )
        .unwrap();
    let op_an = an
        .estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875])
        .unwrap();

    let ll_ode = ode
        .estimate_likelihood(
            &subject,
            &vec![1.02282724609375, 194.51904296875],
            &ems,
            false,
        )
        .unwrap();
    let op_ode = ode
        .estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875])
        .unwrap();

    // Display likelihoods side by side
    println!("Likelihoods:");
    println!("Analytical\tODE");
    println!("{:.6}\t:{:.6}", -2.0 * ll_an, -2.0 * ll_ode);
    println!();

    // Display predictions
    println!("Predictions:");
    println!("Analytical\tODE\tDifference");
    op_an
        .flat_predictions()
        .iter()
        .zip(op_ode.flat_predictions())
        .for_each(|(a, b)| println!("{:.9}\t{:.9}\t{:.9}", a, b, a - b));
}
