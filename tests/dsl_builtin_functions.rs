#![cfg(feature = "dsl")]

use approx::assert_relative_eq;
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};
use pharmsol::{effect50, Parameters, Subject, SubjectBuilderExt};
use pharmsol_dsl::MathFunction;

const PARAMS: [(&str, f64); 5] = [
    ("p1", 0.7),
    ("p2", 1.3),
    ("p3", 0.4),
    ("p4", 2.2),
    ("p5", 0.9),
];

fn compile(output: &str) -> CompiledRuntimeModel {
    let source = format!(
        "name = builtin\nkind = ode\nparams = p1, p2, p3, p4, p5\nstates = central\noutputs = y\ndx(central) = 0\nout(y) = {output}\n"
    );
    compile_module_source_to_runtime(&source, Some("builtin"), |_, _| {})
        .unwrap_or_else(|error| panic!("`{output}` compiles: {error}"))
}

fn predict(model: &CompiledRuntimeModel) -> f64 {
    let subject = Subject::builder("builtin")
        .missing_observation(0.0, "y")
        .build();
    let parameters = Parameters::with_model(model, PARAMS).expect("valid parameters");
    model
        .predict(&subject, &parameters)
        .expect("runtime prediction")
        .into_subject()
        .expect("ODE runtime returns subject predictions")
        .predictions()[0]
        .prediction()
}

#[test]
fn jit_matches_the_reference_evaluator_for_every_math_function() {
    let values = PARAMS.map(|(_, value)| value);
    for function in MathFunction::ALL {
        let arity = function.signature().params.len();
        let args = PARAMS[..arity]
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ");
        let expected = function
            .evaluate(&values[..arity])
            .expect("one value per parameter");
        let actual = predict(&compile(&format!("{}({args})", function.name())));
        assert_relative_eq!(actual, expected, max_relative = 1e-12);
    }
}

#[test]
fn jit_effect50_binds_named_arguments_and_defaults() {
    assert_relative_eq!(predict(&compile("effect50()")), 0.5, max_relative = 1e-12);
    assert_relative_eq!(
        predict(&compile("effect50(p1, p2, p3, p4, p5)")),
        effect50(0.7, 1.3, 0.4, 2.2, 0.9),
        max_relative = 1e-12
    );
    assert_relative_eq!(
        predict(&compile("effect50(x = p5, e50 = p3, max = 10)")),
        effect50(0.0, 10.0, 0.4, 1.0, 0.9),
        max_relative = 1e-12
    );
}
