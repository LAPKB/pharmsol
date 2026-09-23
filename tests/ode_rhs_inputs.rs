use approx::assert_relative_eq;
use pharmsol::prelude::*;
use pharmsol::{lag, ode, SubjectBuilderExt};

#[test]
fn macro_rhs_inputs_use_existing_simulator_vectors() {
    let model = ode! {
        name: "rhs_inputs",
        params: [f, iv_scale],
        states: [gut, central],
        outputs: [cp],
        diffeq: |_x, _t, dx| {
            let oral_scale = f * 2.0;
            dx[gut] = bolus[oral] * oral_scale;
            dx[central] = infusion[iv] * iv_scale;
        },
        lag: |_t| lag! { oral => 0.5 },
        out: |x, _t, y| { y[cp] = x[gut] + x[central]; },
    };
    let subject = Subject::builder("rhs-inputs")
        .bolus(0.0, 10.0, "oral")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(0.25, "cp")
        .missing_observation(1.5, "cp")
        .build();
    let predictions = model
        .estimate_predictions_dense(&subject, &[1.5, 2.0])
        .unwrap();
    assert_relative_eq!(
        predictions.predictions()[0].prediction(),
        50.0,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        predictions.predictions()[1].prediction(),
        230.0,
        epsilon = 1e-5
    );

    #[cfg(feature = "dsl")]
    {
        use pharmsol::dsl::{compile_module_source_to_runtime, RuntimePredictions};
        let source = "name = rhs_inputs\nkind = ode\nparams = f, iv_scale\nstates = gut, central\noral_scale = f * 2\nlag(oral) = 0.5\ndx(gut) = bolus(oral) * oral_scale\ndx(central) = infusion(iv) * iv_scale\nout(cp) = gut + central\n";
        let compiled =
            compile_module_source_to_runtime(source, Some("rhs_inputs"), |_, _| {}).unwrap();
        let parameters =
            pharmsol::Parameters::with_model(&compiled, [("f", 1.5), ("iv_scale", 2.0)]).unwrap();
        let RuntimePredictions::Subject(actual) = compiled
            .estimate_predictions(&subject, &parameters)
            .unwrap()
        else {
            panic!("expected ODE predictions");
        };
        for (actual, expected) in actual.predictions().iter().zip(predictions.predictions()) {
            assert_relative_eq!(actual.prediction(), expected.prediction(), epsilon = 1e-5);
        }
    }
}
