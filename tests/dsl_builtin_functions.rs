#![cfg(feature = "dsl")]

use approx::assert_relative_eq;
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel};
use pharmsol::{effect50, Parameters, Subject, SubjectBuilderExt};
use pharmsol_dsl::{MathFunction, ParamKind};

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
    for function in MathFunction::ALL {
        for condition in [true, false] {
            // Numeric parameters take p1, p2, ... in turn; a boolean one takes
            // a comparison so both branches of `ifelse` are exercised.
            let mut numeric = PARAMS.iter();
            let (args, values): (Vec<_>, Vec<_>) = function
                .signature()
                .params
                .iter()
                .map(|param| match param.kind {
                    ParamKind::Numeric => {
                        let (name, value) = numeric.next().expect("enough parameters");
                        (name.to_string(), *value)
                    }
                    ParamKind::Bool if condition => ("p2 > p1".to_string(), 1.0),
                    ParamKind::Bool => ("p1 > p2".to_string(), 0.0),
                })
                .unzip();
            let expected = function.evaluate(&values).expect("one value per parameter");
            let call = format!("{}({})", function.name(), args.join(", "));
            assert_relative_eq!(predict(&compile(&call)), expected, max_relative = 1e-12);
        }
    }
}

#[test]
fn jit_ifelse_selects_a_branch_from_runtime_conditions() {
    assert_relative_eq!(predict(&compile("ifelse(central >= 0, 2, 3)")), 2.0);
    assert_relative_eq!(predict(&compile("ifelse(t > 1, 2, p1)")), 0.7);
    assert_relative_eq!(
        predict(&compile(
            "ifelse(no = 0, yes = sigmoid(p1, k = p4), test = p5 < 1)"
        )),
        pharmsol_dsl::sigmoid(0.7, 2.2),
        max_relative = 1e-12
    );
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
