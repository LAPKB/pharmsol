use approx::assert_relative_eq;
#[cfg(feature = "dsl-jit")]
use pharmsol::dsl::{self, RuntimeCompilationTarget, RuntimePredictions};
#[cfg(feature = "dsl-jit")]
use pharmsol::equation::RouteInputPolicy;
use pharmsol::equation::{
    self, AnalyticalKernel, RouteKind as HandwrittenRouteKind, ValidatedModelMetadata,
};
use pharmsol::prelude::*;
#[cfg(feature = "dsl-jit")]
use pharmsol::Predictions;
use pharmsol_dsl::{
    analyze_model, lower_typed_model, parse_model, CovariateInterpolation, ExecutionModel,
    ModelKind, RouteKind as DslRouteKind,
};

const ODE_DSL: &str = r#"
name = one_cmt_oral_iv
kind = ode

params = ka, cl, v, tlag, f_oral
covariates = wt @linear
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral

dx(depot) = -ka * depot
dx(central) = ka * depot - (cl / v) * central

out(cp) = central / (v * (wt / 70.0)) ~ continuous()
"#;

const ODE_MACRO_DSL: &str = r#"
name = one_cmt_oral_covariate_parity
kind = ode

params = ka, cl, v, tlag, f_oral
covariates = wt @linear
states = depot, central
outputs = cp

bolus(oral) -> depot
lag(oral) = tlag
fa(oral) = f_oral

dx(depot) = -ka * depot
dx(central) = ka * depot - (cl / v) * central

out(cp) = central / (v * (wt / 70.0)) ~ continuous()
"#;

const ODE_MULTI_DIGIT_OUTPUT_ORDER_DSL: &str = r#"
name = multi_digit_output_order
kind = ode

params = ke, v
states = central
outputs = 2, 10, 11

infusion(iv) -> central

dx(central) = -ke * central

out(10) = central / v ~ continuous()
out(2) = central / v ~ continuous()
out(11) = central / v ~ continuous()
"#;

const ODE_NUMERIC_ROUTE_LABELS_AUTHORING_DSL: &str = r#"
name = authoring_numeric_routes
kind = ode

states = first, second
outputs = cp

bolus(10) -> first
bolus(11) -> second

dx(first) = 0
dx(second) = 0

out(cp) = first + second ~ continuous()
"#;

const ODE_NUMERIC_ROUTE_LABELS_STRUCTURED_DSL: &str = r#"model structured_numeric_routes {
    kind ode
    states {
        first,
        second,
    }
    routes {
        10 -> first
        11 -> second
    }
    dynamics {
        ddt(first) = 0
        ddt(second) = 0
    }
    outputs {
        cp = first + second
    }
}
"#;

const ODE_INVALID_INFUSION_LAG_DSL: &str = r#"
name = invalid_infusion_lag_parity
kind = ode

params = ke, v, tlag
states = central
outputs = cp

infusion(iv) -> central
lag(iv) = tlag

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const ODE_RUNTIME_SHARED_INPUT_DSL: &str = r#"
name = shared_input_one_cpt
kind = ode

params = ka, ke, v, tlag, f_oral
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(cp) = central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const ODE_RUNTIME_MIXED_OUTPUT_LABELS_DSL: &str = r#"
name = mixed_output_labels_runtime
kind = ode

params = ke, v
states = central
outputs = cp, 0, 1

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
out(0) = 2 * central / v ~ continuous()
out(1) = 3 * central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const ODE_RUNTIME_UNDECLARED_NUMERIC_OUTPUT_LABEL_DSL: &str = r#"
name = undeclared_numeric_output_runtime
kind = ode

params = ke, v
states = central
outputs = a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10

infusion(iv) -> central

dx(central) = -ke * central

out(a0) = central / v ~ continuous()
out(a1) = central / v ~ continuous()
out(a2) = central / v ~ continuous()
out(a3) = central / v ~ continuous()
out(a4) = central / v ~ continuous()
out(a5) = central / v ~ continuous()
out(a6) = central / v ~ continuous()
out(a7) = central / v ~ continuous()
out(a8) = central / v ~ continuous()
out(a9) = central / v ~ continuous()
out(a10) = central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const ODE_RUNTIME_UNDECLARED_NUMERIC_INPUT_LABEL_DSL: &str = r#"
name = undeclared_numeric_input_runtime
kind = ode

params = ke, v
states = central
outputs = cp

bolus(r0) -> central
bolus(r1) -> central
bolus(r2) -> central
bolus(r3) -> central
bolus(r4) -> central
bolus(r5) -> central
bolus(r6) -> central
bolus(r7) -> central
bolus(r8) -> central
bolus(r9) -> central
bolus(r10) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

const ANALYTICAL_DSL: &str = r#"
name = one_cmt_abs_parity
kind = analytical

params = ka, ke, v
states = depot, central
outputs = cp

bolus(oral) -> depot
structure = one_compartment_with_absorption

out(cp) = central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const ANALYTICAL_RUNTIME_SHARED_INPUT_DSL: &str = r#"
name = one_cmt_abs_shared
kind = analytical

params = ka, ke, v, tlag, f_oral
states = gut, central
outputs = cp

bolus(oral) -> gut
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral
structure = one_compartment_with_absorption

out(cp) = central / v ~ continuous()
"#;

const SDE_DSL: &str = r#"
name = one_cmt_sde_parity
kind = sde

params = ka, ke, v, sigma
covariates = wt @locf
states = depot, central
outputs = cp

bolus(oral) -> depot
particles = 256

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central
noise(central) = sigma

out(cp) = central / (v * wt) ~ continuous()
"#;

const SDE_MACRO_DSL: &str = r#"
name = one_cmt_sde_macro_parity
kind = sde

params = ka, ke, v, sigma
states = depot, central
outputs = cp

bolus(oral) -> depot
particles = 256

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central
noise(central) = sigma

out(cp) = central / v ~ continuous()
"#;

#[cfg(feature = "dsl-jit")]
const SDE_RUNTIME_SHARED_INPUT_DSL: &str = r#"
name = one_cmt_shared_sde
kind = sde

params = ka, ke, sigma_ke, v, tlag, f_oral
states = gut, central
outputs = cp
particles = 8

bolus(oral) -> gut
infusion(iv) -> central
lag(oral) = tlag
fa(oral) = f_oral

dx(gut) = -ka * gut
dx(central) = ka * gut - ke * central
noise(central) = sigma_ke

out(cp) = central / v ~ continuous()
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetadataParityView {
    name: String,
    kind: ModelKind,
    parameters: Vec<NamedIndex>,
    covariates: Vec<CovariateParity>,
    states: Vec<NamedIndex>,
    route_input_count: usize,
    routes: Vec<RouteParity>,
    outputs: Vec<NamedIndex>,
    analytical_kernel: Option<AnalyticalKernel>,
    particles: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NamedIndex {
    name: String,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CovariateParity {
    name: String,
    index: usize,
    interpolation: Option<CovariateInterpolation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RouteParity {
    name: String,
    kind: Option<RouteKindParity>,
    declaration_index: usize,
    input_index: usize,
    destination_name: String,
    destination_index: usize,
    has_lag: bool,
    has_bioavailability: bool,
}

#[cfg(feature = "dsl-jit")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct RouteInputPolicyParity {
    name: String,
    declaration_index: usize,
    input_index: usize,
    input_policy: RouteInputPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteKindParity {
    Bolus,
    Infusion,
}

impl RouteKindParity {
    fn from_dsl(kind: DslRouteKind) -> Self {
        match kind {
            DslRouteKind::Bolus => Self::Bolus,
            DslRouteKind::Infusion => Self::Infusion,
        }
    }

    fn from_handwritten(kind: HandwrittenRouteKind) -> Self {
        match kind {
            HandwrittenRouteKind::Bolus => Self::Bolus,
            HandwrittenRouteKind::Infusion => Self::Infusion,
        }
    }
}

fn load_execution_model(src: &str) -> ExecutionModel {
    let parsed = parse_model(src).expect("DSL model should parse");
    let typed = analyze_model(&parsed).expect("DSL model should analyze");
    lower_typed_model(&typed).expect("DSL model should lower")
}

#[cfg(feature = "dsl-jit")]
fn compile_runtime_jit_model(src: &str, model_name: &str) -> dsl::CompiledRuntimeModel {
    dsl::compile_module_source_to_runtime(
        src,
        Some(model_name),
        RuntimeCompilationTarget::Jit,
        |_, _| {},
    )
    .expect("DSL runtime model should compile")
}

#[cfg(feature = "dsl-jit")]
fn shared_input_prediction_subject() -> Subject {
    Subject::builder("authoring-parity-shared-input")
        .bolus(0.0, 100.0, "oral")
        .infusion(6.0, 60.0, "iv", 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(6.5, "cp")
        .missing_observation(7.0, "cp")
        .missing_observation(8.0, "cp")
        .build()
}

fn dsl_metadata_view(src: &str) -> MetadataParityView {
    let model = load_execution_model(src);

    let parameters = model
        .metadata
        .parameters
        .iter()
        .map(|parameter| NamedIndex {
            name: parameter.name.clone(),
            index: parameter.index,
        })
        .collect();
    let covariates = model
        .metadata
        .covariates
        .iter()
        .map(|covariate| CovariateParity {
            name: covariate.name.clone(),
            index: covariate.index,
            interpolation: covariate.interpolation,
        })
        .collect();
    let states = model
        .metadata
        .states
        .iter()
        .map(|state| NamedIndex {
            name: state.name.clone(),
            index: state.offset,
        })
        .collect();
    let outputs = model
        .metadata
        .outputs
        .iter()
        .map(|output| NamedIndex {
            name: output.name.clone(),
            index: output.index,
        })
        .collect();
    let routes = model
        .metadata
        .routes
        .iter()
        .map(|route| RouteParity {
            name: route.name.clone(),
            kind: route.kind.map(RouteKindParity::from_dsl),
            declaration_index: route.declaration_index,
            input_index: route.index,
            destination_name: route.destination.state_name.clone(),
            destination_index: route.destination.state_offset,
            has_lag: route.has_lag,
            has_bioavailability: route.has_bioavailability,
        })
        .collect();

    MetadataParityView {
        name: model.name,
        kind: model.kind,
        parameters,
        covariates,
        states,
        route_input_count: model.abi.route_buffer.len,
        routes,
        outputs,
        analytical_kernel: model.metadata.analytical,
        particles: model.metadata.particles,
    }
}

#[cfg(feature = "dsl-jit")]
fn dsl_route_input_policy_view(src: &str) -> Vec<RouteInputPolicyParity> {
    let model = load_execution_model(src);
    let info = dsl::NativeModelInfo::from_execution_model(&model);

    info.routes
        .into_iter()
        .map(|route| RouteInputPolicyParity {
            name: route.name,
            declaration_index: route.declaration_index,
            input_index: route.index,
            input_policy: if route.inject_input_to_destination {
                RouteInputPolicy::InjectToDestination
            } else {
                RouteInputPolicy::ExplicitInputVector
            },
        })
        .collect()
}

fn validated_metadata_view(metadata: &ValidatedModelMetadata) -> MetadataParityView {
    MetadataParityView {
        name: metadata.name().to_string(),
        kind: metadata.kind(),
        parameters: metadata
            .parameters()
            .iter()
            .enumerate()
            .map(|(index, parameter)| NamedIndex {
                name: parameter.name().to_string(),
                index,
            })
            .collect(),
        covariates: metadata
            .covariates()
            .iter()
            .enumerate()
            .map(|(index, covariate)| CovariateParity {
                name: covariate.name().to_string(),
                index,
                interpolation: covariate.interpolation(),
            })
            .collect(),
        states: metadata
            .states()
            .iter()
            .enumerate()
            .map(|(index, state)| NamedIndex {
                name: state.name().to_string(),
                index,
            })
            .collect(),
        route_input_count: metadata.route_input_count(),
        routes: metadata
            .routes()
            .iter()
            .map(|route| RouteParity {
                name: route.name().to_string(),
                kind: Some(RouteKindParity::from_handwritten(route.kind())),
                declaration_index: route.declaration_index(),
                input_index: route.input_index(),
                destination_name: route.destination().to_string(),
                destination_index: route.destination_index(),
                has_lag: route.has_lag(),
                has_bioavailability: route.has_bioavailability(),
            })
            .collect(),
        outputs: metadata
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, output)| NamedIndex {
                name: output.name().to_string(),
                index,
            })
            .collect(),
        analytical_kernel: metadata.analytical_kernel(),
        particles: metadata.particles(),
    }
}

#[cfg(feature = "dsl-jit")]
fn handwritten_route_input_policy_view(
    metadata: &ValidatedModelMetadata,
) -> Vec<RouteInputPolicyParity> {
    metadata
        .routes()
        .iter()
        .map(|route| RouteInputPolicyParity {
            name: route.name().to_string(),
            declaration_index: route.declaration_index(),
            input_index: route.input_index(),
            input_policy: route
                .input_policy()
                .expect("route input policy should be explicit in this handwritten fixture"),
        })
        .collect()
}

fn macro_ode_model() -> equation::ODE {
    ode! {
        name: "one_cmt_oral_covariate_parity",
        params: [ka, cl, v, tlag, f_oral],
        covariates: [wt],
        states: [depot, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> depot,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - (cl / v) * x[central];
        },
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        out: |x, _p, t, cov, y| {
            fetch_cov!(cov, t, wt);
            y[cp] = x[central] / (v * (wt / 70.0));
        },
    }
}

fn handwritten_ode_macro_model() -> equation::ODE {
    equation::ODE::new(
        |_x, _p, _t, dx, _bolus, _rateiv, _cov| {
            dx[0] = 0.0;
            dx[1] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_oral_covariate_parity")
            .parameters(["ka", "cl", "v", "tlag", "f_oral"])
            .covariates([equation::Covariate::continuous("wt")])
            .states(["depot", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
            ),
    )
    .expect("handwritten macro-shape ODE metadata should validate")
}

fn handwritten_ode_model() -> equation::ODE {
    equation::ODE::new(
        |_x, _p, _t, dx, _bolus, _rateiv, _cov| {
            dx[0] = 0.0;
            dx[1] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_oral_iv")
            .parameters(["ka", "cl", "v", "tlag", "f_oral"])
            .covariates([equation::Covariate::continuous("wt")])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("handwritten ODE metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_macro_ode() -> equation::ODE {
    ode! {
        name: "shared_input_one_cpt",
        params: [ka, ke, v, tlag, f_oral],
        states: [depot, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> depot,
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - ke * x[central];
        },
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_handwritten_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, bolus, rateiv, _cov| {
            fetch_params!(p, ka, ke, _v, _tlag, _f_oral);
            dx[0] = bolus[0] - ka * x[0];
            dx[1] = ka * x[0] + rateiv[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("shared_input_one_cpt")
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .inject_input_to_destination(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ]),
    )
    .expect("handwritten shared-input ODE metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn runtime_mismatched_shared_input_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _bolus, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v, _tlag, _f_oral);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("shared_input_one_cpt_mismatched")
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .with_lag()
                    .with_bioavailability()
                    .expect_explicit_input(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("mismatched shared-input ODE metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_macro_analytical() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs_shared",
        params: [ka, ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        structure: one_compartment_with_absorption,
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_handwritten_analytical() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_abs_shared")
            .kind(equation::ModelKind::Analytical)
            .parameters(["ka", "ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv").to_state("central"),
            ])
            .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .expect("handwritten shared-input analytical metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_macro_sde() -> equation::SDE {
    sde! {
        name: "one_cmt_shared_sde",
        params: [ka, ke, sigma_ke, v, tlag, f_oral],
        states: [gut, central],
        outputs: [cp],
        particles: 8,
        routes: [
            bolus(oral) -> gut,
            infusion(iv) -> central,
        ],
        drift: |x, _p, _t, dx, _cov| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        diffusion: |_p, sigma| {
            sigma[gut] = 0.0;
            sigma[central] = 0.0 * sigma_ke;
        },
        lag: |_p, _t, _cov| {
            lag! { oral => tlag }
        },
        fa: |_p, _t, _cov| {
            fa! { oral => f_oral }
        },
        init: |_p, _t, _cov, x| {
            x[gut] = 0.0;
            x[central] = 0.0;
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

#[cfg(feature = "dsl-jit")]
fn runtime_shared_input_handwritten_sde() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ka, ke, _sigma_ke, _v, _tlag, _f_oral);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] + rateiv[0] - ke * x[1];
        },
        |_p, sigma| {
            sigma[0] = 0.0;
            sigma[1] = 0.0;
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, tlag, _f_oral);
            lag! { 0 => tlag }
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, _sigma_ke, _v, _tlag, f_oral);
            fa! { 0 => f_oral }
        },
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _sigma_ke, v, _tlag, _f_oral);
            y[0] = x[1] / v;
        },
        8,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_shared_sde")
            .kind(equation::ModelKind::Sde)
            .parameters(["ka", "ke", "sigma_ke", "v", "tlag", "f_oral"])
            .states(["gut", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("gut")
                    .inject_input_to_destination()
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .inject_input_to_destination(),
            ])
            .particles(8),
    )
    .expect("handwritten shared-input SDE metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn assert_prediction_vectors_close(left: &[f64], right: &[f64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let diff = (left_value - right_value).abs();
        assert!(
            diff <= tolerance,
            "prediction mismatch: left={left_value:.12}, right={right_value:.12}, diff={diff:.12}, tolerance={tolerance:.12}"
        );
    }
}

#[cfg(feature = "dsl-jit")]
fn assert_prediction_vectors_diverge(left: &[f64], right: &[f64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter()
            .zip(right.iter())
            .any(|(left_value, right_value)| (left_value - right_value).abs() > tolerance),
        "expected prediction vectors to diverge beyond tolerance {tolerance:.12}"
    );
}

#[cfg(feature = "dsl-jit")]
fn particle_prediction_means(predictions: &ndarray::Array2<Prediction>) -> Vec<f64> {
    predictions
        .get_predictions()
        .into_iter()
        .map(|prediction| prediction.prediction())
        .collect()
}

fn macro_analytical_model() -> equation::Analytical {
    analytical! {
        name: "one_cmt_abs_parity",
        params: [ka, ke, v],
        states: [depot, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> depot,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_analytical_model() -> equation::Analytical {
    equation::Analytical::new(
        equation::one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_abs_parity")
            .kind(ModelKind::Analytical)
            .parameters(["ka", "ke", "v"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .route(equation::Route::bolus("oral").to_state("depot"))
            .analytical_kernel(AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .expect("handwritten analytical metadata should validate")
}

fn macro_sde_model() -> equation::SDE {
    sde! {
        name: "one_cmt_sde_macro_parity",
        params: [ka, ke, v, sigma],
        states: [depot, central],
        outputs: [cp],
        particles: 256,
        routes: [
            bolus(oral) -> depot,
        ],
        drift: |x, _p, _t, dx, _cov| {
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - ke * x[central];
        },
        diffusion: |_p, sigma_values| {
            sigma_values[depot] = 0.0;
            sigma_values[central] = sigma;
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn handwritten_sde_model() -> equation::SDE {
    equation::SDE::new(
        |_x, _p, _t, dx, _rateiv, _cov| {
            dx[0] = 0.0;
            dx[1] = 0.0;
        },
        |_p, sigma| {
            sigma[0] = 0.0;
            sigma[1] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
        256,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_sde_parity")
            .kind(ModelKind::Sde)
            .parameters(["ka", "ke", "v", "sigma"])
            .covariates([equation::Covariate::locf("wt")])
            .states(["depot", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .inject_input_to_destination(),
            )
            .particles(256),
    )
    .expect("handwritten SDE metadata should validate")
}

fn handwritten_sde_macro_model() -> equation::SDE {
    equation::SDE::new(
        |_x, _p, _t, dx, _rateiv, _cov| {
            dx[0] = 0.0;
            dx[1] = 0.0;
        },
        |_p, sigma_values| {
            sigma_values[0] = 0.0;
            sigma_values[1] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
        256,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_sde_macro_parity")
            .kind(ModelKind::Sde)
            .parameters(["ka", "ke", "v", "sigma"])
            .states(["depot", "central"])
            .outputs(["cp"])
            .route(
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .inject_input_to_destination(),
            )
            .particles(256),
    )
    .expect("handwritten macro-shape SDE metadata should validate")
}

#[cfg(feature = "dsl-jit")]
fn mismatched_ode_model() -> equation::ODE {
    equation::ODE::new(
        |_x, _p, _t, dx, _bolus, _rateiv, _cov| {
            dx[0] = 0.0;
            dx[1] = 0.0;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |_x, _p, _t, _cov, y| {
            y[0] = 0.0;
        },
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("one_cmt_oral_iv")
            .parameters(["ka", "cl", "v", "tlag", "f_oral"])
            .covariates([equation::Covariate::continuous("wt")])
            .states(["depot", "central"])
            .outputs(["cp"])
            .routes([
                equation::Route::bolus("oral")
                    .to_state("depot")
                    .expect_explicit_input()
                    .with_lag()
                    .with_bioavailability(),
                equation::Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .expect("mismatched ODE metadata should validate")
}

#[test]
fn ode_dsl_and_handwritten_metadata_agree_on_public_shape() {
    let handwritten = handwritten_ode_model();
    let dsl_view = dsl_metadata_view(ODE_DSL);
    let handwritten_view = validated_metadata_view(
        handwritten
            .metadata()
            .expect("handwritten ODE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
}

#[test]
fn ode_dsl_declared_output_order_controls_dense_indices_for_multi_digit_labels() {
    let dsl_view = dsl_metadata_view(ODE_MULTI_DIGIT_OUTPUT_ORDER_DSL);

    assert_eq!(
        dsl_view.outputs,
        vec![
            NamedIndex {
                name: "2".to_string(),
                index: 0,
            },
            NamedIndex {
                name: "10".to_string(),
                index: 1,
            },
            NamedIndex {
                name: "11".to_string(),
                index: 2,
            },
        ]
    );
}

#[test]
fn ode_authoring_dsl_supports_multi_digit_numeric_route_labels() {
    let dsl_view = dsl_metadata_view(ODE_NUMERIC_ROUTE_LABELS_AUTHORING_DSL);

    assert_eq!(dsl_view.route_input_count, 2);
    assert_eq!(
        dsl_view.routes,
        vec![
            RouteParity {
                name: "10".to_string(),
                kind: Some(RouteKindParity::Bolus),
                declaration_index: 0,
                input_index: 0,
                destination_name: "first".to_string(),
                destination_index: 0,
                has_lag: false,
                has_bioavailability: false,
            },
            RouteParity {
                name: "11".to_string(),
                kind: Some(RouteKindParity::Bolus),
                declaration_index: 1,
                input_index: 1,
                destination_name: "second".to_string(),
                destination_index: 1,
                has_lag: false,
                has_bioavailability: false,
            },
        ]
    );
}

#[test]
fn ode_structured_dsl_supports_multi_digit_numeric_route_labels() {
    let dsl_view = dsl_metadata_view(ODE_NUMERIC_ROUTE_LABELS_STRUCTURED_DSL);

    assert_eq!(dsl_view.route_input_count, 2);
    assert_eq!(
        dsl_view.routes,
        vec![
            RouteParity {
                name: "10".to_string(),
                kind: None,
                declaration_index: 0,
                input_index: 0,
                destination_name: "first".to_string(),
                destination_index: 0,
                has_lag: false,
                has_bioavailability: false,
            },
            RouteParity {
                name: "11".to_string(),
                kind: None,
                declaration_index: 1,
                input_index: 1,
                destination_name: "second".to_string(),
                destination_index: 1,
                has_lag: false,
                has_bioavailability: false,
            },
        ]
    );
}

#[test]
fn ode_macro_dsl_and_handwritten_metadata_agree_on_macro_authorable_shape() {
    let handwritten = handwritten_ode_macro_model();
    let macro_model = macro_ode_model();
    let dsl_view = dsl_metadata_view(ODE_MACRO_DSL);
    let handwritten_view = validated_metadata_view(
        handwritten
            .metadata()
            .expect("handwritten macro-shape ODE metadata should exist"),
    );
    let macro_view = validated_metadata_view(
        macro_model
            .metadata()
            .expect("macro ODE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
    assert_eq!(macro_view, dsl_view);
}

#[test]
fn analytical_dsl_macro_and_handwritten_metadata_agree_on_public_shape() {
    let handwritten = handwritten_analytical_model();
    let macro_model = macro_analytical_model();
    let dsl_view = dsl_metadata_view(ANALYTICAL_DSL);
    let handwritten_view = validated_metadata_view(
        handwritten
            .metadata()
            .expect("handwritten analytical metadata should exist"),
    );
    let macro_view = validated_metadata_view(
        macro_model
            .metadata()
            .expect("macro analytical metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
    assert_eq!(macro_view, dsl_view);
}

#[test]
fn sde_dsl_and_handwritten_metadata_agree_on_public_shape() {
    let handwritten = handwritten_sde_model();
    let dsl_view = dsl_metadata_view(SDE_DSL);
    let handwritten_view = validated_metadata_view(
        handwritten
            .metadata()
            .expect("handwritten SDE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
}

#[test]
fn sde_macro_dsl_and_handwritten_metadata_agree_on_macro_authorable_shape() {
    let handwritten = handwritten_sde_macro_model();
    let macro_model = macro_sde_model();
    let dsl_view = dsl_metadata_view(SDE_MACRO_DSL);
    let handwritten_view = validated_metadata_view(
        handwritten
            .metadata()
            .expect("handwritten macro-shape SDE metadata should exist"),
    );
    let macro_view = validated_metadata_view(
        macro_model
            .metadata()
            .expect("macro SDE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
    assert_eq!(macro_view, dsl_view);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn ode_route_input_policies_agree_with_handwritten_metadata() {
    let dsl_view = dsl_route_input_policy_view(ODE_DSL);
    let handwritten = handwritten_ode_model();
    let handwritten_view = handwritten_route_input_policy_view(
        handwritten
            .metadata()
            .expect("handwritten ODE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn sde_route_input_policies_agree_with_handwritten_metadata() {
    let dsl_view = dsl_route_input_policy_view(SDE_DSL);
    let handwritten = handwritten_sde_model();
    let handwritten_view = handwritten_route_input_policy_view(
        handwritten
            .metadata()
            .expect("handwritten SDE metadata should exist"),
    );

    assert_eq!(handwritten_view, dsl_view);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn route_input_policy_mismatches_are_detected_explicitly() {
    let dsl_view = dsl_route_input_policy_view(ODE_DSL);
    let handwritten = mismatched_ode_model();
    let handwritten_view = handwritten_route_input_policy_view(
        handwritten
            .metadata()
            .expect("mismatched handwritten metadata should exist"),
    );

    assert_ne!(handwritten_view, dsl_view);
    assert_eq!(dsl_view[0].name, "oral");
    assert_eq!(
        dsl_view[0].input_policy,
        RouteInputPolicy::InjectToDestination
    );
    assert_eq!(
        handwritten_view[0].input_policy,
        RouteInputPolicy::ExplicitInputVector
    );
}

#[test]
fn invalid_dsl_infusion_route_properties_fail_explicitly() {
    let model =
        parse_model(ODE_INVALID_INFUSION_LAG_DSL).expect("invalid DSL fixture should parse");
    let typed = analyze_model(&model).expect("invalid DSL fixture should analyze");
    let error = lower_typed_model(&typed)
        .err()
        .expect("infusion lag should fail during lowering");

    assert!(error
        .to_string()
        .contains("DSL authoring does not allow `lag` on infusion route `iv`"));
}

#[cfg(feature = "dsl-jit")]
#[test]
fn ode_runtime_jit_macro_and_handwritten_predictions_agree_on_shared_input_shape() {
    let runtime_model =
        compile_runtime_jit_model(ODE_RUNTIME_SHARED_INPUT_DSL, "shared_input_one_cpt");
    let macro_model = runtime_shared_input_macro_ode();
    let handwritten_model = runtime_shared_input_handwritten_ode();

    let oral = runtime_model
        .route_index("oral")
        .expect("runtime oral route should exist");
    let iv = runtime_model
        .route_index("iv")
        .expect("runtime iv route should exist");
    let cp = runtime_model
        .output_index("cp")
        .expect("runtime cp output should exist");
    let subject = shared_input_prediction_subject();
    let support_point = [1.0, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(oral, 0);
    assert_eq!(iv, oral);
    assert_eq!(cp, 0);
    assert_eq!(macro_model.route_index("oral"), Some(oral));
    assert_eq!(macro_model.route_index("iv"), Some(iv));
    assert_eq!(handwritten_model.route_index("oral"), Some(oral));
    assert_eq!(handwritten_model.route_index("iv"), Some(iv));

    let runtime_predictions = match runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect("runtime ODE model should simulate")
    {
        RuntimePredictions::Subject(predictions) => predictions.flat_predictions().to_vec(),
        RuntimePredictions::Particles(_) => panic!("ODE runtime should return subject predictions"),
    };
    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro ODE model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten ODE model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_vectors_close(&runtime_predictions, &macro_predictions, 1e-4);
    assert_prediction_vectors_close(&runtime_predictions, &handwritten_predictions, 1e-4);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn analytical_runtime_jit_macro_and_handwritten_predictions_agree_on_shared_input_shape() {
    let runtime_model =
        compile_runtime_jit_model(ANALYTICAL_RUNTIME_SHARED_INPUT_DSL, "one_cmt_abs_shared");
    let macro_model = runtime_shared_input_macro_analytical();
    let handwritten_model = runtime_shared_input_handwritten_analytical();

    let oral = runtime_model
        .route_index("oral")
        .expect("runtime oral route should exist");
    let iv = runtime_model
        .route_index("iv")
        .expect("runtime iv route should exist");
    let cp = runtime_model
        .output_index("cp")
        .expect("runtime cp output should exist");
    let subject = shared_input_prediction_subject();
    let support_point = [1.1, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(oral, 0);
    assert_eq!(iv, oral);
    assert_eq!(cp, 0);
    assert_eq!(macro_model.route_index("oral"), Some(oral));
    assert_eq!(macro_model.route_index("iv"), Some(iv));
    assert_eq!(handwritten_model.route_index("oral"), Some(oral));
    assert_eq!(handwritten_model.route_index("iv"), Some(iv));

    let runtime_predictions = match runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect("runtime analytical model should simulate")
    {
        RuntimePredictions::Subject(predictions) => predictions.flat_predictions().to_vec(),
        RuntimePredictions::Particles(_) => {
            panic!("analytical runtime should return subject predictions")
        }
    };
    let macro_predictions = macro_model
        .estimate_predictions(&subject, &support_point)
        .expect("macro analytical model should simulate")
        .flat_predictions()
        .to_vec();
    let handwritten_predictions = handwritten_model
        .estimate_predictions(&subject, &support_point)
        .expect("handwritten analytical model should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_vectors_close(&runtime_predictions, &macro_predictions, 1e-8);
    assert_prediction_vectors_close(&runtime_predictions, &handwritten_predictions, 1e-8);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn sde_runtime_jit_macro_and_handwritten_predictions_agree_on_shared_input_shape() {
    let runtime_model =
        compile_runtime_jit_model(SDE_RUNTIME_SHARED_INPUT_DSL, "one_cmt_shared_sde");
    let macro_model = runtime_shared_input_macro_sde();
    let handwritten_model = runtime_shared_input_handwritten_sde();

    let oral = runtime_model
        .route_index("oral")
        .expect("runtime oral route should exist");
    let iv = runtime_model
        .route_index("iv")
        .expect("runtime iv route should exist");
    let cp = runtime_model
        .output_index("cp")
        .expect("runtime cp output should exist");
    let subject = shared_input_prediction_subject();
    let support_point = [1.1, 0.2, 0.0, 10.0, 0.25, 0.8];

    assert_eq!(oral, 0);
    assert_eq!(iv, oral);
    assert_eq!(cp, 0);
    assert_eq!(macro_model.route_index("oral"), Some(oral));
    assert_eq!(macro_model.route_index("iv"), Some(iv));
    assert_eq!(handwritten_model.route_index("oral"), Some(oral));
    assert_eq!(handwritten_model.route_index("iv"), Some(iv));

    let runtime_predictions = match runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect("runtime SDE model should simulate")
    {
        RuntimePredictions::Particles(predictions) => particle_prediction_means(&predictions),
        RuntimePredictions::Subject(_) => panic!("SDE runtime should return particle predictions"),
    };
    let macro_predictions = particle_prediction_means(
        &macro_model
            .estimate_predictions(&subject, &support_point)
            .expect("macro SDE model should simulate"),
    );
    let handwritten_predictions = particle_prediction_means(
        &handwritten_model
            .estimate_predictions(&subject, &support_point)
            .expect("handwritten SDE model should simulate"),
    );

    assert_prediction_vectors_close(&runtime_predictions, &macro_predictions, 1e-4);
    assert_prediction_vectors_close(&runtime_predictions, &handwritten_predictions, 1e-4);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn route_input_policy_runtime_mismatches_are_detected_explicitly() {
    let runtime_model =
        compile_runtime_jit_model(ODE_RUNTIME_SHARED_INPUT_DSL, "shared_input_one_cpt");
    let mismatched_model = runtime_mismatched_shared_input_ode();

    let oral = runtime_model
        .route_index("oral")
        .expect("runtime oral route should exist");
    let iv = runtime_model
        .route_index("iv")
        .expect("runtime iv route should exist");
    let cp = runtime_model
        .output_index("cp")
        .expect("runtime cp output should exist");
    let subject = shared_input_prediction_subject();
    let support_point = [1.0, 0.2, 10.0, 0.25, 0.8];

    assert_eq!(oral, 0);
    assert_eq!(iv, oral);
    assert_eq!(cp, 0);
    assert_eq!(mismatched_model.route_index("oral"), Some(oral));
    assert_eq!(mismatched_model.route_index("iv"), Some(iv));

    let runtime_predictions = match runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect("runtime ODE model should simulate")
    {
        RuntimePredictions::Subject(predictions) => predictions.flat_predictions().to_vec(),
        RuntimePredictions::Particles(_) => panic!("ODE runtime should return subject predictions"),
    };
    let mismatched_predictions = mismatched_model
        .estimate_predictions(&subject, &support_point)
        .expect("mismatched handwritten ODE should simulate")
        .flat_predictions()
        .to_vec();

    assert_prediction_vectors_diverge(&runtime_predictions, &mismatched_predictions, 1e-4);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn ode_runtime_jit_preserves_mixed_output_labels() {
    let runtime_model = compile_runtime_jit_model(
        ODE_RUNTIME_MIXED_OUTPUT_LABELS_DSL,
        "mixed_output_labels_runtime",
    );
    let subject = Subject::builder("runtime-mixed-output-labels")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(0.5, "cp")
        .missing_observation(0.5, "0")
        .missing_observation(0.5, "1")
        .build();
    let support_point = [0.2, 10.0];

    assert_eq!(runtime_model.output_index("cp"), Some(0));
    assert_eq!(runtime_model.output_index("0"), Some(1));
    assert_eq!(runtime_model.output_index("1"), Some(2));

    let predictions = match runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect("runtime mixed-output model should simulate")
    {
        RuntimePredictions::Subject(predictions) => predictions.flat_predictions().to_vec(),
        RuntimePredictions::Particles(_) => panic!("ODE runtime should return subject predictions"),
    };

    assert_eq!(predictions.len(), 3);
    assert_relative_eq!(predictions[1], 2.0 * predictions[0], epsilon = 1e-6);
    assert_relative_eq!(predictions[2], 3.0 * predictions[0], epsilon = 1e-6);
}

#[cfg(feature = "dsl-jit")]
#[test]
fn ode_runtime_jit_rejects_undeclared_numeric_output_labels_even_when_dense_index_exists() {
    let runtime_model = compile_runtime_jit_model(
        ODE_RUNTIME_UNDECLARED_NUMERIC_OUTPUT_LABEL_DSL,
        "undeclared_numeric_output_runtime",
    );
    let subject = Subject::builder("runtime-undeclared-numeric-output")
        .infusion(0.0, 100.0, "iv", 1.0)
        .missing_observation(0.5, "10")
        .build();
    let support_point = [0.2, 10.0];

    let error = runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect_err("undeclared numeric output label should fail");

    assert!(matches!(
        error,
        dsl::RuntimeError::Runtime(PharmsolError::UnknownOutputLabel { label }) if label == "10"
    ));
}

#[cfg(feature = "dsl-jit")]
#[test]
fn ode_runtime_jit_rejects_undeclared_numeric_input_labels_even_when_dense_index_exists() {
    let runtime_model = compile_runtime_jit_model(
        ODE_RUNTIME_UNDECLARED_NUMERIC_INPUT_LABEL_DSL,
        "undeclared_numeric_input_runtime",
    );
    let subject = Subject::builder("runtime-undeclared-numeric-input")
        .bolus(0.0, 100.0, "10")
        .missing_observation(0.5, "cp")
        .build();
    let support_point = [0.2, 10.0];

    let error = runtime_model
        .estimate_predictions(&subject, &support_point)
        .expect_err("undeclared numeric input label should fail");

    assert!(matches!(
        error,
        dsl::RuntimeError::Runtime(PharmsolError::UnknownInputLabel { label }) if label == "10"
    ));
}
