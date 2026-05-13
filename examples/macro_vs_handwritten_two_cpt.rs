//! Compares a declaration-first macro ODE with the equivalent handwritten ODE
//! on a two-compartment IV problem that shares one numeric input across
//! a loading bolus and a maintenance infusion.
//!
//! This keeps the macro story as the default surface while showing the
//! low-level API as an explicit advanced comparison path.

use pharmsol::prelude::*;

fn macro_model() -> equation::ODE {
    ode! {
        name: "two_cpt_shared_input_parity",
        params: [ke, kcp, kpc, v],
        states: [central, peripheral],
        outputs: [cp],
        routes: [
            bolus(load) -> central,
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_model() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, bolus, rateiv, _cov| {
            fetch_params!(p, ke, kcp, kpc, _v);
            dx[0] = -ke * x[0] - kcp * x[0] + kpc * x[1] + rateiv[0] + bolus[0];
            dx[1] = kcp * x[0] - kpc * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _kcp, _kpc, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("two_cpt_shared_input_parity")
            .parameters(["ke", "kcp", "kpc", "v"])
            .states(["central", "peripheral"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("load")
                    .to_state("central")
                    .inject_input_to_destination(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ]),
    )
    .expect("handwritten two-compartment metadata should validate")
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn main() -> Result<(), pharmsol::PharmsolError> {
    let macro_ode = macro_model();
    let handwritten_ode = handwritten_model();
    let macro_metadata = macro_ode.metadata().expect("macro metadata exists");

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());
    assert_eq!(
        macro_metadata
            .route("load")
            .map(|route| route.input_index()),
        macro_metadata.route("iv").map(|route| route.input_index()),
        "load and iv should share one numeric input"
    );
    assert!(macro_metadata.output("cp").is_some());

    let subject = Subject::builder("macro-vs-handwritten-two-cpt")
        .bolus(0.0, 100.0, "load")
        .infusion(12.0, 200.0, "iv", 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .missing_observation(8.0, "cp")
        .missing_observation(12.0, "cp")
        .missing_observation(12.5, "cp")
        .missing_observation(13.0, "cp")
        .missing_observation(14.0, "cp")
        .missing_observation(16.0, "cp")
        .missing_observation(24.0, "cp")
        .build();

    let params = [0.1, 0.05, 0.03, 50.0];
    let macro_predictions = macro_ode.estimate_predictions(&subject, &params)?;
    let handwritten_predictions = handwritten_ode.estimate_predictions(&subject, &params)?;

    let macro_flat = macro_predictions.flat_predictions();
    let handwritten_flat = handwritten_predictions.flat_predictions();
    let diff = max_abs_diff(&macro_flat, &handwritten_flat);

    assert!(
        diff <= 1e-10,
        "macro and handwritten two-compartment predictions diverged: {diff:e}"
    );

    println!("two-compartment parity max abs diff: {diff:e}");
    for ((time, macro_pred), handwritten_pred) in macro_predictions
        .flat_times()
        .iter()
        .zip(macro_flat.iter())
        .zip(handwritten_flat.iter())
    {
        println!("t={time:>5.1}  macro={macro_pred:>12.8}  handwritten={handwritten_pred:>12.8}");
    }

    Ok(())
}
