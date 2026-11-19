//! Predictions are addressed by output label, end to end.
//!
//! Predictions keep the label of the observation they came from, so a caller
//! can pair each one with whatever it keys by output (an error model, a
//! plot series) without relying on declaration or insertion order.

use pharmsol::prelude::*;

fn two_output_ode() -> equation::ODE {
    ode! {
        name: "two_output",
        params: [ke, v],
        states: [central],
        outputs: [cp, effect],
        diffeq: |x, _t, dx| {
            dx[central] = bolus[iv] - ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
            y[effect] = 2.0 * x[central] / v;
        },
    }
}

fn observed_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .observation(1.0, 8.0, "effect")
        .observation(2.0, 7.0, "effect")
        .build();
    Data::new(vec![subject])
}

fn parameters(model: &equation::ODE) -> Parameters {
    Parameters::with_model(model, [("ke", 0.2), ("v", 20.0)]).expect("valid named parameters")
}

#[test]
fn predictions_keep_their_output_label() -> Result<(), PharmsolError> {
    let ode = two_output_ode();
    let data = observed_data();
    let subject = &data.subjects()[0];
    let predictions = ode.predict(subject, &parameters(&ode))?;

    let mut labels: Vec<&str> = predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.outeq().as_str())
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, vec!["cp", "cp", "effect", "effect"]);

    // Each label gets its own output: `effect` is twice `cp` at every time.
    for prediction in predictions.predictions() {
        if prediction.outeq().as_str() == "effect" {
            let cp = predictions
                .predictions()
                .iter()
                .find(|other| other.outeq().as_str() == "cp" && other.time() == prediction.time())
                .expect("a cp prediction at the same time");
            assert!((prediction.prediction() - 2.0 * cp.prediction()).abs() < 1e-12);
        }
    }

    // Turning a prediction back into an observation (e.g. to build simulated
    // data) keeps the label, so the observation resolves to the same output.
    for prediction in predictions.predictions() {
        assert_eq!(prediction.to_observation().outeq(), prediction.outeq());
    }
    Ok(())
}

#[test]
fn output_labels_serialize_as_plain_strings() {
    let label: OutputLabel = serde_json::from_str("\"cp\"").expect("label deserializes");
    assert_eq!(label, OutputLabel::new("cp"));
    assert_eq!(serde_json::to_string(&label).unwrap(), "\"cp\"");
}
