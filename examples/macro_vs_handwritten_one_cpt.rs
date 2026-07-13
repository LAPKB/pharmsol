//! Compares a declaration-first macro ODE with the equivalent handwritten ODE.
//!
//! This is the advanced comparison path for users who want to confirm that the
//! preferred macro surface and the low-level API produce the same metadata and
//! predictions on the same one-compartment IV problem.

use pharmsol::{prelude::*, Parameters};

fn macro_model() -> equation::ODE {
    ode! {
        name: "one_cpt_macro_parity",
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
    }
}

fn handwritten_model() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = rateiv[0] - ke * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cpt_macro_parity")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("handwritten one-compartment metadata should validate")
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

    assert_eq!(macro_ode.metadata(), handwritten_ode.metadata());

    let subject = Subject::builder("macro-vs-handwritten-one-cpt")
        .infusion(0.0, 500.0, "iv", 0.5)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .missing_observation(8.0, "cp")
        .build();

    let params = Parameters::with_model(&macro_ode, [("ke", 1.022), ("v", 194.0)])
        .expect("valid named parameters");
    let macro_predictions = macro_ode.estimate_predictions(&subject, &params)?;
    let handwritten_predictions = handwritten_ode.estimate_predictions(&subject, &params)?;

    let macro_flat = macro_predictions.flat_predictions();
    let handwritten_flat = handwritten_predictions.flat_predictions();
    let diff = max_abs_diff(&macro_flat, &handwritten_flat);

    assert!(
        diff <= 1e-10,
        "macro and handwritten one-compartment predictions diverged: {diff:e}"
    );

    println!("one-compartment parity max abs diff: {diff:e}");
    for ((time, macro_pred), handwritten_pred) in macro_predictions
        .flat_times()
        .iter()
        .zip(macro_flat.iter())
        .zip(handwritten_flat.iter())
    {
        println!("t={time:>4.1}  macro={macro_pred:>12.8}  handwritten={handwritten_pred:>12.8}");
    }

    Ok(())
}
