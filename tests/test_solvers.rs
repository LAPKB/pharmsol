//! Integration tests for the ODE solver selection API.
//!
//! Runs a one-compartment model under every OdeSolver variant and checks
//! that predictions agree with Bdf (the default) within tolerance.

use pharmsol::prelude::*;

fn subject() -> Subject {
    Subject::builder("id1")
        .bolus(0.0, 100.0, "iv_bolus")
        .infusion(12.0, 200.0, "iv", 2.0)
        .observation(0.5, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .observation(12.5, 0.0, "cp")
        .observation(14.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build()
}

fn one_cpt(solver: OdeSolver) -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0] + b[0];
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
    .with_solver(solver)
    .with_metadata(
        equation::metadata::new("solver_selection_one_cpt")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("iv_bolus")
                    .to_state("central")
                    .expect_explicit_input(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("solver selection metadata should validate")
}

fn preds(solver: OdeSolver) -> Vec<f64> {
    let sub = subject();
    let model = one_cpt(solver);
    let parameters = Parameters::with_model(&model, [("ke", 0.1), ("v", 50.0)])
        .expect("solver selection parameters should validate");
    model
        .estimate_predictions(&sub, &parameters)
        .unwrap()
        .flat_predictions()
        .to_vec()
}

#[test]
fn bdf_produces_predictions() {
    let p = preds(OdeSolver::Bdf);
    assert!(!p.is_empty());
    assert!(p.iter().all(|v| v.is_finite()));
}

#[test]
fn tsit45_matches_bdf() {
    let ref_p = preds(OdeSolver::Bdf);
    let test_p = preds(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    assert_eq!(ref_p.len(), test_p.len());
    for (r, t) in ref_p.iter().zip(&test_p) {
        assert!((r - t).abs() < 0.01, "Tsit45 diverged: {r} vs {t}");
    }
}

#[test]
fn tr_bdf2_matches_bdf() {
    let ref_p = preds(OdeSolver::Bdf);
    let test_p = preds(OdeSolver::Sdirk(SdirkTableau::TrBdf2));
    assert_eq!(ref_p.len(), test_p.len());
    for (r, t) in ref_p.iter().zip(&test_p) {
        assert!((r - t).abs() < 0.01, "TrBdf2 diverged: {r} vs {t}");
    }
}

#[test]
fn esdirk34_matches_bdf() {
    let ref_p = preds(OdeSolver::Bdf);
    let test_p = preds(OdeSolver::Sdirk(SdirkTableau::Esdirk34));
    assert_eq!(ref_p.len(), test_p.len());
    for (r, t) in ref_p.iter().zip(&test_p) {
        assert!((r - t).abs() < 0.01, "Esdirk34 diverged: {r} vs {t}");
    }
}
