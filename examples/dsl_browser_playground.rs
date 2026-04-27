//! Run with:
//! cargo run --example dsl_browser_playground --features dsl-wasm

#[cfg(feature = "dsl-wasm")]
mod playground {
    use std::collections::{hash_map::DefaultHasher, HashMap};
    use std::fs;
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;
    use std::process::Command;

    use pharmsol::dsl::{self, CompiledRuntimeModel, RuntimePredictions};
    use pharmsol::{Subject, SubjectBuilderExt};
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

    const INDEX_HTML: &str = include_str!("dsl_browser_playground/index.html");
    const APP_JS: &str = include_str!("dsl_browser_playground/app.js");
    const COMPILE_WORKER_JS: &str = include_str!("dsl_browser_playground/compile-worker.js");
    const STYLES_CSS: &str = include_str!("dsl_browser_playground/styles.css");

    const DEFAULT_MODEL_NAME: &str = "bimodal_ke";
    const DEFAULT_MODEL_SOURCE: &str = r#"
model = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

    const COMPILER_BRIDGE_MANIFEST: &str = "tests/browser-e2e/compiler-bridge/Cargo.toml";
    const COMPILER_BRIDGE_TARGET_DIR: &str = "target/dsl-browser-playground-compile-bridge";
    const COMPILER_BRIDGE_WASM_RELATIVE: &str =
        "wasm32-unknown-unknown/release/browser_compile_bridge.wasm";

    struct AppState {
        compiler_bridge_wasm: Vec<u8>,
        compiled_cache: HashMap<u64, CompiledRuntimeModel>,
    }

    impl AppState {
        fn new(compiler_bridge_wasm: Vec<u8>) -> Self {
            Self {
                compiler_bridge_wasm,
                compiled_cache: HashMap::new(),
            }
        }

        fn compiled_from_bytes(
            &mut self,
            wasm_bytes: &[u8],
        ) -> Result<&CompiledRuntimeModel, String> {
            let cache_key = artifact_hash(wasm_bytes);
            if !self.compiled_cache.contains_key(&cache_key) {
                let model =
                    dsl::load_runtime_wasm_bytes(wasm_bytes).map_err(|error| error.to_string())?;
                self.compiled_cache.clear();
                self.compiled_cache.insert(cache_key, model);
            }

            self.compiled_cache.get(&cache_key).ok_or_else(|| {
                "The compiled artifact could not be cached for simulation.".to_string()
            })
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct DoseDraft {
        time: f64,
        amount: f64,
        route: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct InfusionDraft {
        time: f64,
        amount: f64,
        route: String,
        duration: f64,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ObservationDraft {
        time: f64,
        output: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct SubjectDraft {
        id: String,
        boluses: Vec<DoseDraft>,
        infusions: Vec<InfusionDraft>,
        observations: Vec<ObservationDraft>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct SimulateRequest {
        wasm_bytes: Vec<u8>,
        parameter_values: HashMap<String, f64>,
        covariate_values: HashMap<String, f64>,
        subject: SubjectDraft,
    }

    #[derive(Clone, Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct PredictionPoint {
        time: f64,
        output: String,
        value: f64,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        println!("Preparing browser compile bridge...");
        let compiler_bridge_wasm = prepare_browser_compile_bridge()?;

        let server = Server::http("127.0.0.1:0")
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        let url = format!("http://{}", server.server_addr());
        let mut state = AppState::new(compiler_bridge_wasm);

        println!("Opening DSL browser playground at {url}");
        if should_open_browser() {
            if let Err(error) = webbrowser::open(&url) {
                eprintln!("Could not open the browser automatically: {error}");
                eprintln!("Open this URL manually: {url}");
            }
        } else {
            eprintln!("Automatic browser launch disabled. Open this URL manually: {url}");
        }
        println!("Press Ctrl+C to stop the local server.");

        for request in server.incoming_requests() {
            if let Err(error) = handle_request(request, &mut state) {
                eprintln!("request handling failed: {error}");
            }
        }

        Ok(())
    }

    fn handle_request(
        mut request: Request,
        state: &mut AppState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match (request.method(), request.url()) {
            (&Method::Get, "/") => respond_text(request, INDEX_HTML, "text/html; charset=utf-8"),
            (&Method::Get, "/app.js") => {
                respond_text(request, APP_JS, "text/javascript; charset=utf-8")
            }
            (&Method::Get, "/compile-worker.js") => {
                respond_text(request, COMPILE_WORKER_JS, "text/javascript; charset=utf-8")
            }
            (&Method::Get, "/styles.css") => {
                respond_text(request, STYLES_CSS, "text/css; charset=utf-8")
            }
            (&Method::Get, "/browser_compile_bridge.wasm") => respond_bytes(
                request,
                state.compiler_bridge_wasm.clone(),
                "application/wasm",
            ),
            (&Method::Get, "/api/defaults") => respond_json(
                request,
                &json!({
                    "defaultModelName": DEFAULT_MODEL_NAME,
                    "defaultModelSource": DEFAULT_MODEL_SOURCE.trim(),
                }),
                StatusCode(200),
            ),
            (&Method::Post, "/api/simulate") => {
                let payload: SimulateRequest = read_json(&mut request)?;
                let body = simulate_response(state, payload);
                respond_json(request, &body, StatusCode(200));
            }
            _ => respond_json(
                request,
                &json!({
                    "ok": false,
                    "message": "Route not found"
                }),
                StatusCode(404),
            ),
        }
        Ok(())
    }

    fn simulate_response(state: &mut AppState, payload: SimulateRequest) -> Value {
        if payload.wasm_bytes.is_empty() {
            return json!({
                "ok": false,
                "message": "Compile the model in the browser before running a simulation.",
            });
        }

        let model = match state.compiled_from_bytes(&payload.wasm_bytes) {
            Ok(model) => model,
            Err(message) => {
                return json!({
                    "ok": false,
                    "message": message,
                });
            }
        };

        let support_point = match support_point_values(
            model.info().parameters.as_slice(),
            &payload.parameter_values,
        ) {
            Ok(values) => values,
            Err(message) => {
                return json!({
                    "ok": false,
                    "message": message,
                });
            }
        };

        let subject = match build_subject(model, &payload.subject, &payload.covariate_values) {
            Ok(subject) => subject,
            Err(message) => {
                return json!({
                    "ok": false,
                    "message": message,
                });
            }
        };

        match model.estimate_predictions(&subject, &support_point) {
            Ok(RuntimePredictions::Subject(predictions)) => json!({
                "ok": true,
                "predictionKind": "subject",
                "summary": format!(
                    "{} structural predictions returned from the browser-compiled WASM artifact",
                    predictions.predictions().len()
                ),
                "scientificNotes": scientific_notes(false, None),
                "predictions": predictions.predictions().iter().map(|prediction| {
                    PredictionPoint {
                        time: prediction.time(),
                        output: output_name(model, prediction.outeq()),
                        value: prediction.prediction(),
                    }
                }).collect::<Vec<_>>(),
            }),
            Ok(RuntimePredictions::Particles(predictions)) => {
                let mean_predictions = (0..predictions.ncols())
                    .map(|column| {
                        let template = &predictions[(0, column)];
                        let mean = (0..predictions.nrows())
                            .map(|row| predictions[(row, column)].prediction())
                            .sum::<f64>()
                            / predictions.nrows() as f64;
                        PredictionPoint {
                            time: template.time(),
                            output: output_name(model, template.outeq()),
                            value: mean,
                        }
                    })
                    .collect::<Vec<_>>();
                json!({
                    "ok": true,
                    "predictionKind": "particles",
                    "particleCount": predictions.nrows(),
                    "summary": format!(
                        "{} particle-mean predictions returned from the browser-compiled WASM artifact",
                        mean_predictions.len()
                    ),
                    "scientificNotes": scientific_notes(true, Some(predictions.nrows())),
                    "predictions": mean_predictions,
                })
            }
            Err(error) => json!({
                "ok": false,
                "message": error.to_string(),
            }),
        }
    }

    fn support_point_values(
        parameter_names: &[String],
        parameter_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, String> {
        parameter_names
            .iter()
            .map(|name| {
                let value = parameter_values
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| default_parameter_value(name));
                validate_parameter_value(name, value)?;
                Ok(value)
            })
            .collect()
    }

    fn build_subject(
        model: &CompiledRuntimeModel,
        subject: &SubjectDraft,
        covariate_values: &HashMap<String, f64>,
    ) -> Result<Subject, String> {
        let info = model.info();
        let subject_id = if subject.id.trim().is_empty() {
            format!("{}_subject", info.name)
        } else {
            subject.id.trim().to_string()
        };
        let mut builder = Subject::builder(subject_id);

        for covariate in &info.covariates {
            let value = covariate_values
                .get(&covariate.name)
                .copied()
                .unwrap_or_else(|| default_covariate_value(&covariate.name));
            ensure_finite(&format!("covariate `{}`", covariate.name), value)?;
            builder = builder.covariate(&covariate.name, 0.0, value);
        }

        if subject.observations.is_empty() {
            return Err(
                "Add at least one missing observation time so the simulator knows when to evaluate outputs."
                    .to_string(),
            );
        }

        for bolus in &subject.boluses {
            ensure_nonnegative("bolus time", bolus.time)?;
            ensure_positive("bolus amount", bolus.amount)?;
            let route = model.route_index(&bolus.route).ok_or_else(|| {
                format!(
                    "Unknown bolus route `{}` for model `{}`",
                    bolus.route, info.name
                )
            })?;
            builder = builder.bolus(bolus.time, bolus.amount, route);
        }

        for infusion in &subject.infusions {
            ensure_nonnegative("infusion time", infusion.time)?;
            ensure_positive("infusion amount", infusion.amount)?;
            ensure_positive("infusion duration", infusion.duration)?;
            let route = model.route_index(&infusion.route).ok_or_else(|| {
                format!(
                    "Unknown infusion route `{}` for model `{}`",
                    infusion.route, info.name
                )
            })?;
            builder = builder.infusion(infusion.time, infusion.amount, route, infusion.duration);
        }

        for observation in &subject.observations {
            ensure_nonnegative("observation time", observation.time)?;
            let output = model.output_index(&observation.output).ok_or_else(|| {
                format!(
                    "Unknown observation output `{}` for model `{}`",
                    observation.output, info.name
                )
            })?;
            builder = builder.missing_observation(observation.time, output);
        }

        Ok(builder.build())
    }

    fn scientific_notes(is_particle_model: bool, particle_count: Option<usize>) -> Vec<String> {
        let mut notes = vec![
            "Predictions are structural model outputs evaluated exactly at the missing-observation times in the subject draft.".to_string(),
            "Compilation runs in a browser worker through the direct DSL-to-WASM compiler, and host-side simulation reuses that same compiled WASM artifact.".to_string(),
        ];
        if is_particle_model {
            notes.push(format!(
                "This model is stochastic, so the chart reports particle means across {} particles rather than one deterministic trajectory.",
                particle_count.unwrap_or(0)
            ));
        } else {
            notes.push(
                "No residual error model is added here; the chart shows the model-predicted trajectory only.".to_string(),
            );
        }
        notes
    }

    fn output_name(model: &CompiledRuntimeModel, outeq: usize) -> String {
        model
            .info()
            .outputs
            .get(outeq)
            .map(|output| output.name.clone())
            .unwrap_or_else(|| format!("output_{outeq}"))
    }

    fn default_parameter_value(name: &str) -> f64 {
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "ka" => 1.2,
            "cl" | "cl_i" => 5.0,
            "v" | "vc" | "vol" => 40.0,
            "ke" | "ke0" => 0.25,
            "kcp" | "kpc" | "q" => 0.12,
            "tlag" | "lag" => 0.25,
            "f" | "fa" | "f_oral" => 0.8,
            "ske" | "sigma" | "omega" => 0.05,
            _ if lower.starts_with('f') => 0.8,
            _ => 1.0,
        }
    }

    fn default_covariate_value(name: &str) -> f64 {
        let lower = name.to_ascii_lowercase();
        if lower.contains("wt") || lower.contains("weight") {
            70.0
        } else if lower.contains("age") {
            50.0
        } else if lower.contains("ht") || lower.contains("height") {
            170.0
        } else {
            1.0
        }
    }

    fn validate_parameter_value(name: &str, value: f64) -> Result<(), String> {
        ensure_finite(&format!("parameter `{name}`"), value)?;
        let lower = name.to_ascii_lowercase();

        if matches!(lower.as_str(), "f" | "fa" | "f_oral") || lower.starts_with('f') {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "Parameter `{name}` must stay between 0 and 1 for this playground."
                ));
            }
            return Ok(());
        }

        if matches!(lower.as_str(), "tlag" | "lag") {
            if value < 0.0 {
                return Err(format!("Parameter `{name}` may not be negative."));
            }
            return Ok(());
        }

        if parameter_scale(name) == "log" && value <= 0.0 {
            return Err(format!(
                "Parameter `{name}` must stay strictly positive for stable pharmacometric interpretation."
            ));
        }

        Ok(())
    }

    fn parameter_scale(name: &str) -> &'static str {
        let lower = name.to_ascii_lowercase();
        if matches!(lower.as_str(), "tlag" | "lag" | "f" | "fa" | "f_oral")
            || lower.starts_with('f')
        {
            "linear"
        } else {
            "log"
        }
    }

    fn ensure_finite(label: &str, value: f64) -> Result<(), String> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(format!("{label} must be finite."))
        }
    }

    fn ensure_nonnegative(label: &str, value: f64) -> Result<(), String> {
        ensure_finite(label, value)?;
        if value < 0.0 {
            Err(format!("{label} must be non-negative."))
        } else {
            Ok(())
        }
    }

    fn ensure_positive(label: &str, value: f64) -> Result<(), String> {
        ensure_finite(label, value)?;
        if value <= 0.0 {
            Err(format!("{label} must be greater than zero."))
        } else {
            Ok(())
        }
    }

    fn artifact_hash(bytes: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    fn prepare_browser_compile_bridge() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let manifest_path = manifest_dir.join(COMPILER_BRIDGE_MANIFEST);
        let target_dir = manifest_dir.join(COMPILER_BRIDGE_TARGET_DIR);
        let wasm_path = target_dir.join(COMPILER_BRIDGE_WASM_RELATIVE);

        let status = Command::new("cargo")
            .current_dir(&manifest_dir)
            .args([
                "build",
                "--manifest-path",
                manifest_path
                    .to_str()
                    .expect("manifest path to be valid UTF-8"),
                "--target",
                "wasm32-unknown-unknown",
                "--release",
                "--target-dir",
                target_dir.to_str().expect("target dir to be valid UTF-8"),
            ])
            .status()?;

        if !status.success() {
            return Err(std::io::Error::other(
                "failed to build the browser compile bridge; install wasm32-unknown-unknown with `rustup target add wasm32-unknown-unknown`",
            )
            .into());
        }

        Ok(fs::read(wasm_path)?)
    }

    fn should_open_browser() -> bool {
        std::env::var("PHARMSOL_PLAYGROUND_OPEN_BROWSER")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                normalized != "0" && normalized != "false" && normalized != "no"
            })
            .unwrap_or(true)
    }

    fn read_json<T: DeserializeOwned>(
        request: &mut Request,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let mut body = String::new();
        request.as_reader().read_to_string(&mut body)?;
        Ok(serde_json::from_str(&body)?)
    }

    fn respond_text(request: Request, body: &str, content_type: &str) {
        let response =
            Response::from_string(body).with_header(header("Content-Type", content_type));
        let _ = request.respond(response);
    }

    fn respond_bytes(request: Request, body: Vec<u8>, content_type: &str) {
        let response = Response::from_data(body).with_header(header("Content-Type", content_type));
        let _ = request.respond(response);
    }

    fn respond_json(request: Request, body: &Value, status: StatusCode) {
        let response = Response::from_string(
            serde_json::to_string_pretty(body).unwrap_or_else(|_| "{}".to_string()),
        )
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json; charset=utf-8"));
        let _ = request.respond(response);
    }

    fn header(name: &str, value: &str) -> Header {
        Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid static header")
    }
}

#[cfg(feature = "dsl-wasm")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    playground::run()
}

#[cfg(not(feature = "dsl-wasm"))]
fn main() {
    eprintln!("Run with: cargo run --example dsl_browser_playground --features dsl-wasm");
    std::process::exit(1);
}
