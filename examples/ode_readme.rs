fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::prelude::*;

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

    let subject = Subject::builder("id1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .build();

    let predictions = ode.estimate_predictions(&subject, &[1.022, 194.0])?;
    println!(
        "state central => {}",
        ode.state_index("central").expect("central state exists")
    );
    println!("prediction times => {:?}", predictions.flat_times());
    println!("predictions => {:?}", predictions.flat_predictions());

    Ok(())
}
