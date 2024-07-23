use pharmsol::*;
fn main() {
    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.0, -1.0, 0)
        .repeat(12, 0.5)
        .build();

    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ke0, _ske, _v);
            let ke = x[1];
            dx[1] = -ke + ke0;

            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0, ske, _v);
            d[1] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _ske, _v);
            x[1] = ke0
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _ske, v);
            y[0] = x[0] / v;
        },
        (2, 1),
    );

    let trajectories = sde.simulate_trajectories(&subject, &vec![0.3, 0.1, 50.0], 10);
    dbg!(trajectories);
}
