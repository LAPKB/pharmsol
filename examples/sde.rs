use pharmsol::{prelude::data::read_pmetrics, *};

fn one_c_ode() -> ODE {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke);
            dx[0] = -ke * x[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| {
            y[0] = x[0] / 50.0;
        },
        (1, 1),
    );
    ode
}

fn one_c_sde() -> SDE {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _ske);
            // let ke0 = 1.2;
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0, ske);
            d[1] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _ske);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _ske);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
        2,
    );
    sde
}

fn three_c_ode() -> ODE {
    let ode = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ka, ke, kcp, kpc, _vol);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (ke + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _kcp, _kpc, vol);
            y[0] = x[1] / vol;
        },
        (3, 3),
    );
    ode
}

fn three_c_sde() -> SDE {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke0, kcp, kpc, _vol, _ske);
            dx[3] = -x[3] + ke0;
            let ke = x[3];
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (ke + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |p, d| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, _vol, ske);
            d[3] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _kcp, _kpc, _vol, _ske);
            x[3] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke0, _kcp0, _kpc0, vol, _ske);
            y[0] = x[1] / vol;
        },
        (4, 1),
        2,
    );
    sde
}

fn main() {
    let ode = one_c_ode();
    let sde = one_c_sde();

    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.0, -1.0, 0)
        .repeat(5, 0.2)
        .build();

    let ode_predictions = ode.estimate_predictions(&subject, &vec![1.0]);
    let sde_predictions = sde.estimate_predictions(&subject, &vec![1.0, 0.0]);

    let mut sde_flat_predictions: Vec<Vec<f64>> = Vec::new();
    for trajectory in sde_predictions.rows() {
        let mut flat_predictions: Vec<f64> = Vec::new();
        for prediction in trajectory.iter() {
            flat_predictions.push(prediction.prediction());
        }
        sde_flat_predictions.push(flat_predictions);
    }
    println!("One compartment model");
    dbg!(ode_predictions.flat_predictions());
    dbg!(sde_flat_predictions);
    println!("---------------------------------");

    let ode = three_c_ode();
    let sde = three_c_sde();

    let spp_ode = vec![
        0.9235121835947036,
        1.9836121537923812,
        0.6279836881637573,
        1.426409281349182,
        11.543784689903259,
    ];
    let spp_sde = vec![
        0.9235121835947036,
        1.9836121537923812,
        0.6279836881637573,
        1.426409281349182,
        11.543784689903259,
        0.0,
    ];

    let ode_predictions = ode.estimate_predictions(&subject, &spp_ode);
    let sde_predictions = sde.estimate_predictions(&subject, &spp_sde);

    let mut sde_flat_predictions: Vec<Vec<f64>> = Vec::new();
    for trajectory in sde_predictions.rows() {
        let mut flat_predictions: Vec<f64> = Vec::new();
        for prediction in trajectory.iter() {
            flat_predictions.push(prediction.prediction());
        }
        sde_flat_predictions.push(flat_predictions);
    }

    println!("Three compartment model");
    dbg!(ode_predictions.flat_predictions());
    dbg!(sde_flat_predictions);
    println!("---------------------------------");

    let ode = three_c_ode();
    let sde = three_c_sde();

    let data = read_pmetrics("../PMcore/examples/w_vanco_sde/test.csv").unwrap();
    let subject = data.get_subject("51").unwrap();

    let ode_predictions = ode.estimate_predictions(&subject, &spp_ode);
    let sde_predictions = sde.estimate_predictions(&subject, &spp_sde);

    let mut sde_flat_predictions: Vec<Vec<f64>> = Vec::new();
    for trajectory in sde_predictions.rows() {
        let mut flat_predictions: Vec<f64> = Vec::new();
        for prediction in trajectory.iter() {
            flat_predictions.push(prediction.prediction());
        }
        sde_flat_predictions.push(flat_predictions);
    }

    println!("Three compartment model - Vanco");
    dbg!(ode_predictions.flat_predictions());
    dbg!(sde_flat_predictions);
    println!("---------------------------------");
}
