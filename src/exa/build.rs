use std::env;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::io::{self};
use std::path::PathBuf;

use rand::Rng;
use rand_distr::Alphanumeric;
use std::process::{Command, Stdio};
use std::thread;

use crate::Equation;

/// Compiles model text into a dynamically loadable library.
///
/// This function creates a Rust project from a template, injects the model text,
/// compiles it, and outputs the resulting library file.
///
/// # Arguments
///
/// * `model_txt` - The model text defining ODE equations.
/// * `output` - Optional path for the output file. If not provided, a random name is generated.
/// * `params` - List of parameter names for the model.
/// * `event_callback` - Callback function for emitting events during compilation.
///
/// # Returns
///
/// The path to the compiled model library file, or an error if compilation failed.
pub fn compile<E: Equation>(
    model_txt: String,
    output: Option<PathBuf>,
    params: Vec<String>,
    event_callback: fn(String, String),
) -> Result<String, io::Error> {
    let _template_path = match create_template() {
        Ok(path) => path,
        Err(e) => {
            event_callback(
                "build-log".into(),
                format!("Failed to create template: {}", e),
            );
            return Err(e);
        }
    };
    let template_path = match inject_model::<E>(model_txt, params) {
        Ok(path) => path,
        Err(e) => {
            event_callback("build-log".into(), format!("Failed to inject model: {}", e));
            return Err(e);
        }
    };

    let dynlib_path = match build_template(template_path.clone(), event_callback) {
        Ok(path) => path,
        Err(e) => {
            event_callback(
                "build-log".into(),
                format!("Failed to build template: {}", e),
            );
            return Err(e);
        }
    };
    let output_path = output.unwrap_or_else(|| {
        let random_suffix: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(5)
            .map(char::from)
            .collect();
        let default_name = format!(
            "model_{}_{}_{}.pkm",
            env::consts::OS,
            env::consts::ARCH,
            random_suffix
        );
        let temp_dir = env::temp_dir().join("exa_tmp");
        temp_dir.with_file_name(default_name)
    });

    fs::copy(&dynlib_path, &output_path).expect("Failed to copy dynamic library to output path");
    Ok(output_path.to_string_lossy().to_string())
}

/// Creates a dummy compilation for testing purposes.
///
/// This function creates a template and builds it without injecting any model text.
///
/// # Arguments
///
/// * `event_callback` - Callback function for emitting events during compilation.
///
/// # Returns
///
/// The path to the template directory, or an error if creation failed.
pub fn dummy_compile(event_callback: fn(String, String)) -> Result<String, io::Error> {
    let template_path = create_template()?;
    build_template(template_path.clone(), event_callback)?;
    Ok(template_path.to_string_lossy().to_string())
}

/// Returns the path to the template directory.
///
/// # Returns
///
/// A string representing the path to the template directory.
pub fn template_path() -> String {
    env::temp_dir()
        .join("exa_tmp")
        .join("template")
        .to_string_lossy()
        .to_string()
}

/// Clears all build artifacts from the temporary directory.
///
/// This function removes the entire temporary directory used for building models.
pub fn clear_build() {
    let temp_dir = env::temp_dir().join("exa_tmp");
    if temp_dir.exists() {
        fs::remove_dir_all(temp_dir).expect("Failed to remove temporary directory");
    }
}

/// Creates a new template project for model compilation.
///
/// This function creates a Rust project structure with the necessary dependencies
/// for compiling ODE models.
///
/// # Returns
///
/// The path to the created template directory, or an error if creation failed.
fn create_template() -> Result<PathBuf, io::Error> {
    let temp_dir = env::temp_dir().join("exa_tmp");
    if !temp_dir.exists() {
        fs::create_dir_all(&temp_dir)?;
    }
    let template_dir = temp_dir.join("template");
    let cargo_toml_path = template_dir.join("Cargo.toml");

    // Get the current package version
    let pkg_version = env!("CARGO_PKG_VERSION");
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let pharmsol_dep = if std::env::var("PHARMSOL_LOCAL_EXA").is_ok() {
        let manifest_path =
            std::fs::canonicalize(manifest_dir).unwrap_or_else(|_| PathBuf::from(manifest_dir));
        let manifest_str = manifest_path.to_string_lossy();

        if manifest_str.contains('\'') {
            let escaped = manifest_str.replace('\\', "\\\\").replace('"', "\\\"");
            format!(r#"pharmsol = {{ path = "{}" }}"#, escaped)
        } else {
            format!(r#"pharmsol = {{ path = '{}' }}"#, manifest_str)
        }
    } else {
        format!(r#"pharmsol = {{ version = "{}" }}"#, pkg_version)
    };

    let cargo_toml_content = format!(
        r#"
        [package]
        name = "model_lib"
        version = "0.1.0"
        edition = "2021"

        [lib]
        crate-type = ["cdylib"]

        [dependencies]
        {pharmsol_dep}
        "#,
    );

    if !template_dir.exists() {
        let output = Command::new("cargo")
            .arg("new")
            .arg("template")
            .arg("--lib")
            .current_dir(&temp_dir)
            .output()
            .expect("Failed to create cargo project");

        io::stderr().write_all(&output.stderr)?;
        io::stdout().write_all(&output.stdout)?;

        fs::write(cargo_toml_path, cargo_toml_content)?;
    } else if !cargo_toml_path.exists() {
        fs::write(cargo_toml_path, cargo_toml_content)?;
    };
    Ok(template_dir)
}

/// Injects model text and parameters into the template project.
///
/// # Arguments
///
/// * `model_txt` - The model text to inject into the template.
/// * `params` - List of parameter names for the model.
///
/// # Returns
///
/// The path to the template directory, or an error if injection failed.
fn inject_model<E: Equation>(model_txt: String, params: Vec<String>) -> Result<PathBuf, io::Error> {
    let template_dir = env::temp_dir().join("exa_tmp").join("template");
    let lib_rs_path = template_dir.join("src").join("lib.rs");
    let lib_rs_content = format!(
        r#"
        #![allow(dead_code)]
        #![allow(unused_variables)]
        use std::ffi::c_void;
        use pharmsol::*;
    
        pub fn eqn() -> impl Equation {{
            {}
        }}
    
        #[no_mangle]
        pub extern "C" fn create_eqn_ptr() -> *mut c_void {{
            let eqn = Box::new(eqn());
            Box::into_raw(eqn) as *mut c_void
        }}

        #[no_mangle]
        pub extern "C" fn equation_kind() -> EqnKind{{
            {}
        }}
    
       #[no_mangle]
        pub extern "C" fn metadata_ptr() -> *mut c_void{{
        let meta = Box::new(equation::Meta::new(vec![{}]));
        Box::into_raw(meta) as *mut c_void
        }}
        "#,
        model_txt,
        E::kind().to_str(),
        params
            .iter()
            .map(|p| format!("\"{}\"", p))
            .collect::<Vec<String>>()
            .join(", ")
    );
    fs::write(lib_rs_path, lib_rs_content)?;
    Command::new("cargo")
        .arg("fmt")
        .current_dir(&template_dir)
        .output()
        .expect("Failed to format cargo project");
    Ok(template_dir)
}

/// Builds the template project and creates a dynamic library.
///
/// # Arguments
///
/// * `template_path` - Path to the template project.
/// * `event_callback` - Callback function for emitting events during compilation.
///
/// # Returns
///
/// The path to the compiled dynamic library, or an error if build failed.
fn build_template(
    template_path: PathBuf,
    event_callback: fn(String, String),
) -> Result<PathBuf, io::Error> {
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--release")
        .arg("--quiet")
        .current_dir(&template_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    let stdout_handle = stream_output(stdout, event_callback);
    let stderr_handle = stream_output(stderr, event_callback);

    let status = child.wait()?;
    stdout_handle
        .join()
        .expect("Failed to join stdout thread")?;
    stderr_handle
        .join()
        .expect("Failed to join stderr thread")?;

    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to build the template",
        ));
    }

    let dynlib_name = if cfg!(target_os = "windows") {
        "model_lib.dll"
    } else if cfg!(target_os = "macos") {
        "libmodel_lib.dylib"
    } else {
        "libmodel_lib.so"
    };

    Ok(template_path
        .join("target")
        .join("release")
        .join(dynlib_name))
}

/// Streams output from a reader to a Tauri event channel.
///
/// # Arguments
///
/// * `reader` - The reader to stream output from.
/// * `event_callback` - Callback function for emitting events during compilation.
///
/// # Returns
///
/// A join handle for the streaming thread.
fn stream_output<R: Read + Send + 'static>(
    reader: R,
    event_callback: fn(String, String),
) -> thread::JoinHandle<Result<(), io::Error>> {
    thread::spawn(move || {
        let mut buffer = [0; 4096];
        let mut reader = io::BufReader::new(reader);

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }

            let output = String::from_utf8_lossy(&buffer[..n]).to_string();
            event_callback("build-log-internal".into(), output);
        }
        Ok(())
    })
}
