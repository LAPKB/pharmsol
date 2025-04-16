#![allow(non_snake_case)]
#![allow(unused_variables)]

use pharmsol::{simulator::model::Model, *};

fn main() {
    let ode = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_cov!(cov, t, WT);
            fetch_params!(p, CL0, V0, Vp0, Q0);

            let WTc: f64 = WT / 85.0;
            let CL = CL0 * WTc.powf(0.75);
            let V = V0 * WTc;
            let Vp = Vp0 * WTc;
            let Q = Q0 * WTc.powf(0.75);
            let Ke = CL / V;
            let KCP = Q / V;
            let KPC = Q / Vp;

            dx[0] = -Ke * x[0] - KCP * x[0] + KPC * x[1] + rateiv[0];
            dx[1] = KCP * x[0] - KPC * x[1];
        },
        |p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, WT);
            fetch_params!(p, CL0, V0, Vp0, Q0);
            let V = V0 / (WT / 85.0);
            y[0] = x[0] / V;
        },
        (2, 1),
    );

    let subject = data::Subject::builder("id1")
        .infusion(0.0, 3235.0, 0, 0.005)
        .observation(0.5, 0.1, 0)
        .repeat(120, 0.1)
        .build();
    let mut model = ode.initialize_model(&subject, vec![0.1, 0.1, 0.1, 0.1, 70.0]);
    let op = model.estimate_outputs();

    dbg!(op);
}
