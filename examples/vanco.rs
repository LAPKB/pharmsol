#![allow(dead_code)]
#![allow(unused_variables)]
use pharmsol::{prelude::data::read_pmetrics, *};

fn main() {
    let ode = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_cov!(cov, t, wt, crcl, agecat);
            fetch_params!(p, ke0, kcp, kpc, v0);
            let v = v0 * wt;
            let ke = ke0 * wt.powf(-(0.25)) * crcl;

            dx[0] = (-(ke) * x[0]) - (kcp * x[0]) + kpc * x[1] + rateiv[0];
            dx[1] = (kcp * x[0]) - (kpc * x[1]);
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt, crcl, agecat);
            fetch_params!(p, ke0, kcp, kpc, v0);
            let v = v0 * wt;
            let ke = ke0 * wt.powf(-(0.25)) * crcl;

            y[0] = x[0] / v;
        },
        (2, 1),
    );
    let an = equation::Analytical::new(
        two_compartments,
        |p, t, cov| {
            fetch_cov!(cov, t, wt, crcl, agecat);
            fetch_params!(&p, ke0, kcp, kpc, v0);
            let ke = ke0 * wt.powf(-(0.25)) * crcl;
            let v = v0 * wt;
            p[0] = ke;
            p[1] = kcp;
            p[2] = kpc;
            p[3] = v;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt, crcl, agecat);
            fetch_params!(p, ke0, kcp, kpc, v0);
            let v = v0 * wt;

            y[0] = x[0] / v;
        },
        (2, 1),
    );
    let data = read_pmetrics("examples/vanco.csv").unwrap();
    for subject in data.get_subjects() {
        let op_ode = ode
            .estimate_predictions(&subject, &vec![0.02, 2.0, 2.5, 1.0])
            .unwrap();
        let op_analytical = an
            .estimate_predictions(&subject, &vec![0.02, 2.0, 2.5, 1.0])
            .unwrap();
        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("SUBJECT {}", subject.id());
        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        let mut sum_delta_error = 0.0;
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            sum_delta_error += (od - an).abs();
        }
        println!("sum error: {}", sum_delta_error);
        println!("==========END============");
    }
}
