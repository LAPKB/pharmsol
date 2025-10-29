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
    use std::collections::HashMap;

    // Extract structured lag/fa maps from macro text so the runtime does not
    // need to re-parse macro bodies. These will be empty maps if not present.
    fn extract_macro_map(src: &str, mac: &str) -> HashMap<usize, String> {
        let mut res = HashMap::new();
        let mut search = 0usize;
        while let Some(pos) = src[search..].find(mac) {
            let start = search + pos;
            if let Some(lb_rel) = src[start..].find('{') {
                let lb = start + lb_rel;
                let mut depth: isize = 0;
                let mut i = lb;
                let bytes = src.as_bytes();
                let len = src.len();
                let mut end_opt: Option<usize> = None;
                while i < len {
                    match bytes[i] as char {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end_opt = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if let Some(rb) = end_opt {
                    let body = &src[lb + 1..rb];
                    // split top-level entries by commas not inside parentheses/braces
                    let mut entry = String::new();
                    let mut paren = 0isize;
                    let mut brace = 0isize;
                    for ch in body.chars() {
                        match ch {
                            '(' => {
                                paren += 1;
                                entry.push(ch);
                            }
                            ')' => {
                                paren -= 1;
                                entry.push(ch);
                            }
                            '{' => {
                                brace += 1;
                                entry.push(ch);
                            }
                            '}' => {
                                brace -= 1;
                                entry.push(ch);
                            }
                            ',' if paren == 0 && brace == 0 => {
                                let parts: Vec<&str> = entry.split("=>").collect();
                                if parts.len() == 2 {
                                    if let Ok(k) = parts[0].trim().parse::<usize>() {
                                        res.insert(k, parts[1].trim().to_string());
                                    }
                                }
                                entry.clear();
                            }
                            _ => entry.push(ch),
                        }
                    }
                    if !entry.trim().is_empty() {
                        let parts: Vec<&str> = entry.split("=>").collect();
                        if parts.len() == 2 {
                            if let Ok(k) = parts[0].trim().parse::<usize>() {
                                res.insert(k, parts[1].trim().to_string());
                            }
                        }
                    }
                    search = rb + 1;
                    continue;
                }
            }
            search = start + mac.len();
        }
        res
    }

    let lag_map = extract_macro_map(lag_txt.as_deref().unwrap_or(""), "lag!");
    let fa_map = extract_macro_map(fa_txt.as_deref().unwrap_or(""), "fa!");

    let ir_obj = json!({
        "ir_version": "1.0",
        "kind": E::kind().to_str(),
        "params": params,
        "diffeq": diffeq_txt,
        "lag": lag_txt,
        "fa": fa_txt,
        "lag_map": lag_map,
        "fa_map": fa_map,
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
