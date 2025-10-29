use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

use rand::Rng;
use rand_distr::Alphanumeric;

/// Emit a minimal JSON IR for a model (WASM-friendly emitter).
pub fn emit_ir<E: crate::Equation>(
    diffeq_txt: String,
    lag_txt: Option<String>,
    fa_txt: Option<String>,
    init_txt: Option<String>,
    out_txt: Option<String>,
    output: Option<PathBuf>,
    params: Vec<String>,
) -> Result<String, io::Error> {
    use serde_json::json;

    let ir_obj = json!({
        "ir_version": "1.0",
        "kind": E::kind().to_str(),
        "params": params,
        "diffeq": diffeq_txt,
        "lag": lag_txt,
        "fa": fa_txt,
        "init": init_txt,
        "out": out_txt,
    });

    let output_path = output.unwrap_or_else(|| {
        let random_suffix: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(5)
            .map(char::from)
            .collect();
        let default_name = format!("model_ir_{}_{}.json", env::consts::OS, random_suffix);
        env::temp_dir().join("exa_tmp").with_file_name(default_name)
    });

    let serialized = serde_json::to_vec_pretty(&ir_obj)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde_json error: {}", e)))?;

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    fs::write(&output_path, serialized)?;
    Ok(output_path.to_string_lossy().to_string())
}
