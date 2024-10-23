use em::EM;
use nalgebra::DVector;
use pharmsol::*;
fn main() {
    let mut m = EM::new(
        |x, p, _t, dx, _rateiv, _cov| {
            dx[0] = -x[0] * x[1]; // ke *x[0]
            dx[1] = -x[1] + p[0]; // mean reverting
        },
        |_p, d| {
            d[0] = 1.0;
            d[1] = 0.01;
        },
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![20.0, 1.0]),
        Covariates::new(),
        DVector::from_vec(vec![]),
    );

    let sol = m.solve(0.0, 0.2, 10);
    println!("{:?}", sol);
}
