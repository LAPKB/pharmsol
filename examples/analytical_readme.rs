fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::prelude::*;

    let analytical = analytical! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: {
            infusion(iv) -> central,
        },
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let iv = analytical.route_index("iv").expect("iv route exists");
    let cp = analytical.output_index("cp").expect("cp output exists");

    let subject = Subject::builder("analytical_readme")
        .infusion(0.0, 500.0, iv, 0.5)
        .missing_observation(0.5, cp)
        .missing_observation(1.0, cp)
        .missing_observation(2.0, cp)
        .missing_observation(4.0, cp)
        .build();

    let predictions = analytical.estimate_predictions(&subject, &[1.022, 194.0])?;

    println!("times => {:?}", predictions.flat_times());
    println!("predictions => {:?}", predictions.flat_predictions());

    Ok(())
}