use std::io::Write;
use std::{fs::File, path::Path};

// use error_model::{ErrorModel, ErrorType};
// use ndarray::Array2;
use pharmsol::*;
// use prelude::simulator::pf_psi;
use rand_distr::Distribution;
fn main() {
    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.0, -1.0, 0)
        .repeat(5, 0.2)
        .build();

    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0);
            // let ke0 = 1.2;
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0);
            d[1] = 0.1;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
        1,
    );

    let ke_dist = rand_distr::Normal::new(1.2, 0.12).unwrap();
    // let v_dist = rand_distr::Normal::new(50.0, 10.0).unwrap();
    // let ske_dist = rand_distr::Normal::new(0.1, 0.01).unwrap();

    let mut support_points = vec![];
    let mut file = File::create("spp.csv").unwrap();
    for _ in 0..100 {
        let ke = ke_dist.sample(&mut rand::rng());
        // let ke = 1.2;

        let ske = 0.1;
        // let ske = ske_dist.sample(&mut rand::thread_rng());
        // let v = v_dist.sample(&mut rand::thread_rng());
        let v = 50.0;
        support_points.push(vec![ke]);
        println!("{ke}, {ske}, {v}");
        writeln!(file, "{}, {}, {}", ke, ske, v).unwrap();
    }

    let mut data = vec![];
    for (i, spp) in support_points.iter().enumerate() {
        let trajectories = sde.estimate_predictions(&subject, spp).unwrap();
        let trajectory = trajectories.row(0);
        // dbg!(&trajectory);
        let mut sb = data::Subject::builder(format!("id{}", i)).bolus(0.0, 20.0, 0);
        for (t, point) in trajectory.iter().enumerate() {
            sb = sb.observation((t) as f64 * 0.2, point.prediction(), 0);
        }
        data.push(sb.build());
    }
    let data = data::Data::new(data);
    data.write_pmetrics(&File::create(Path::new("test.csv")).unwrap());
    // let mut theta = Array2::zeros((1, 3));
    // theta[[0, 0]] = 0.7;
    // theta[[0, 1]] = 0.1;
    // theta[[0, 2]] = 50.0;

    // let _ll = sde.particle_filter(
    //     &data.get_subjects().first().unwrap(),
    //     &vec![0.7, 0.1, 50.0],
    //     1000,
    //     &ErrorModel::new((0.0, 0.0, 0.0, 0.0), 0.5, &ErrorType::Add),
    // );

    // let ll = pf_psi(
    //     &sde,
    //     &data,
    //     &theta,
    //     &ErrorModel::new((0.0, 0.0, 0.0, 0.0), 0.5, &ErrorType::Add),
    //     1000,
    // );
    // dbg!(ll);
}
