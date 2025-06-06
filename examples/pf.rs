use error_model::ErrorModel;
use pharmsol::*;
fn main() {
    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.2, 16.6434, 0)
        .observation(0.4, 14.3233, 0)
        .observation(0.6, 9.8468, 0)
        .observation(0.8, 9.4177, 0)
        .observation(1.0, 7.5170, 0)
        .build();

    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            dx[0] = -x[0] * x[1]; // ke *x[0]
            dx[1] = -x[1] + p[0]; // mean reverting
        },
        |_p, d| {
            d[0] = 1.0;
            d[1] = 0.01;
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| x[1] = 1.0,
        |x, _p, _t, _cov, y| {
            y[0] = x[0];
        },
        (2, 1),
        10000,
    );

    let err0 = ErrorModel::additive(ErrorPoly::new(0.5, 0.00, 0.0, 0.0), 0.0);
    let ems = ErrorModels::new().add(0, err0).unwrap();

    let ll = sde
        .estimate_likelihood(&subject, &vec![1.0], &ems, false)
        .unwrap();

    dbg!(sde
        .estimate_likelihood(&subject, &vec![1.0], &ems, false)
        .unwrap());
    println!("{ll:#?}");
}
