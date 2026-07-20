//! Reproduction for issue #292 - expansion of ADDL for long repetitive regimens.

use pharmsol::data::parser::DataRow;
use pharmsol::data::row::build_data;
use pharmsol::prelude::*;
use pharmsol::Parameters;

fn main() -> Result<(), pharmsol::PharmsolError> {
    let ode = ode! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let mut rows: Vec<DataRow> = Vec::new();

    rows.push(
        DataRow::builder("51", 0.0)
            .evid(1)
            .dose(500.0)
            .dur(0.5)
            .input("iv")
            .addl(-1)
            .ii(48.0)
            .build(),
    );

    let obs = [
        (0.5, 1.6457759999999999),
        (1.0, 1.216442),
        (2.0, 0.46227289999999999),
        (3.0, 0.1697458),
        (4.0, 0.063821779999999995),
        (6.0, 0.0090993840000000003),
        (8.0, 0.001017932),
        (12.0, 1.90e-05),
        (18.0, 4.60e-08),
        (24.0, 1.00e-08),
    ];
    for (t, val) in obs {
        rows.push(
            DataRow::builder("51", t)
                .evid(0)
                .out(val)
                .outeq("cp")
                .build(),
        );
    }

    rows.push(
        DataRow::builder("51", 25.0)
            .evid(1)
            .dose(20000.0)
            .dur(0.0001)
            .input("iv")
            .addl(24)
            .ii(120.0)
            .build(),
    );

    let data = build_data(rows)?;
    let subject = data.get_subject("51").unwrap();

    let ems = AssayErrorModels::new().add(
        "cp",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
    )?;

    let params =
        Parameters::with_model(&ode, [("ke", 2.0983288402557374), ("v", 118.2103306055069)])
            .expect("valid named parameters");

    for (name, solver) in [
        ("Bdf", OdeSolver::Bdf),
        ("Tsit45", OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
        ("TrBdf2", OdeSolver::Sdirk(SdirkTableau::TrBdf2)),
        ("Esdirk34", OdeSolver::Sdirk(SdirkTableau::Esdirk34)),
    ] {
        let ode = ode.clone().with_solver(solver);
        print!("[{name}] ");
        match ode.estimate_log_likelihood(subject, &params, &ems) {
            Ok(ll) => println!("log-likelihood = {ll}"),
            Err(e) => println!("ERROR: {e}"),
        }
    }

    Ok(())
}
