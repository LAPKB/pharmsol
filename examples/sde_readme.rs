fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::{Parameters, prelude::*};

    let sde = sde! {
        name: "one_cmt_sde",
        params: [ke, sigma_ke, v],
        states: [central],
        outputs: [cp],
        particles: 16,
        routes: [
            infusion(iv) -> central,
        ],
        drift: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central];
        },
        diffusion: |_p, sigma| {
            sigma[central] = sigma_ke;
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = Subject::builder("sde_readme")
        .infusion(0.0, 500.0, "iv", 0.5)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .build();

    let parameters =
        Parameters::with_model(&sde, [("ke", 1.022), ("sigma_ke", 0.0), ("v", 194.0)])
            .expect("valid named parameters");
    let predictions = sde.estimate_predictions(&subject, &parameters)?;

    println!("first prediction => {}", predictions[[0, 0]].prediction());
    println!("prediction grid shape => {:?}", predictions.dim());

    Ok(())
}
