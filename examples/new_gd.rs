use csv::ReaderBuilder;
use pharmsol::*;
use prelude::{data::read_pmetrics, simulator::Prediction};
 
fn main() {
    // path to theta
    let path = "../PMcore/outputs/theta.csv";
    // read theta into an Array2<f64>
    let mut rdr = ReaderBuilder::new().from_path(path).unwrap();
 
    let mut spps = vec![];
    for result in rdr.records() {
        let record = result.unwrap();
        let mut row = vec![];
        for field in record.iter() {
            row.push(field.parse::<f64>().unwrap());
        }
        spps.push(row);
    }
 
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
            fetch_params!(p, ke0, ske);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _ske);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
        1,
    );

    let data = read_pmetrics("../PMcore/examples/iov/test.csv").unwrap();
 
    for (i, spp) in spps.iter().enumerate() {
        for (j, subject) in data.get_subjects().iter().enumerate() {
            let trajectories: ndarray::Array2<Prediction> =
                sde.estimate_predictions(&subject, &spp);
            let trajectory = trajectories.row(0);
            println!("{}, {}", i, j);
            dbg!(trajectory);
        }
    }
}
