use std::{fs::File, path::Path};

use anyhow::Error;
use error_model::{ErrorModel, ErrorType};
use ndarray::{Array1, Array2};
use pharmsol::*;
use prelude::simulator::pf_psi;
use rand_distr::Distribution;
fn main() {
    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.0, -1.0, 0)
        .repeat(12, 0.5)
        .build();

    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _ske, _v);
            let ke = x[1];
            dx[1] = -ke + ke0;
            // user defined
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

    let ke_dist = rand_distr::Normal::new(0.7, 0.15).unwrap();
    let v_dist = rand_distr::Normal::new(50.0, 10.0).unwrap();

    let mut support_points = vec![];
    for _ in 0..10 {
        let ke = ke_dist.sample(&mut rand::thread_rng());
        let v = v_dist.sample(&mut rand::thread_rng());
        support_points.push(vec![ke, 0.1, v]);
    }

    let mut data = vec![];
    for (i, spp) in support_points.iter().enumerate() {
        let trajectory = sde.simulate_trajectories(&subject, &spp, 1);
        let mut sb = data::Subject::builder(format!("id{}", i)).bolus(0.0, 20.0, 0);
        for (t, point) in trajectory.iter().enumerate() {
            sb = sb.observation((t + 1) as f64 * 0.5, *point.first().unwrap(), 0);
        }
        data.push(sb.build());
    }
    let data = data::Data::new(data);
    data.write_pmetrics(&File::create(Path::new("test.csv")).unwrap());
    let mut theta = Array2::zeros((1, 3));
    theta[[0, 0]] = 0.7;
    theta[[0, 1]] = 0.1;
    theta[[0, 2]] = 50.0;

    let ll = sde.particle_filter(
        &data.get_subjects().first().unwrap(),
        &vec![0.7, 0.1, 50.0],
        1000,
        &ErrorModel::new((0.0, 0.0, 0.0, 0.0), 0.5, &ErrorType::Add),
    );

    let ll = pf_psi(
        &sde,
        &data,
        &theta,
        &ErrorModel::new((0.0, 0.0, 0.0, 0.0), 0.5, &ErrorType::Add),
        1000,
    );
    dbg!(ll);
}
