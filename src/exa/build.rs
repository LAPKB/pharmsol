use std::env;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::io::{self};
use std::path::PathBuf;
use std::sync::Arc;

use rand::RngExt;
use rand_distr::Alphanumeric;
use std::process::{Command, Stdio};
use std::thread;

use crate::Equation;

/// Finds the cargo executable, checking common installation locations.
///
/// This is necessary because bundled GUI applications (like macOS .app bundles,
/// Windows .exe installers, or Linux AppImages) don't inherit the user's shell
/// PATH, so `cargo` may not be found directly.
///
/// Checks in order:
/// 1. Direct `cargo` command (if PATH is set correctly)
/// 2. `CARGO_HOME/bin/cargo` (custom rustup installation)
/// 3. Standard rustup location: `~/.cargo/bin/cargo`
/// 4. Platform-specific fallback locations
///
/// # Returns
///
/// The path to the cargo executable as a String.
fn find_cargo() -> String {
    // First, check if cargo is directly available in PATH
    if let Ok(output) = Command::new("cargo").arg("--version").output() {
        if output.status.success() {
            return "cargo".to_string();
        }
    }

    // Check CARGO_HOME environment variable (custom rustup installation)
    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        let cargo_path = PathBuf::from(&cargo_home)
            .join("bin")
            .join(cargo_exe_name());
        if cargo_path.exists() {
            return cargo_path.to_string_lossy().to_string();
        }
    }

    // Get home directory (works on all platforms)
    // Unix: $HOME, Windows: %USERPROFILE%
    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_default();

    if !home.is_empty() {
        // Standard rustup installation path: ~/.cargo/bin/cargo
        let standard_path = PathBuf::from(&home)
            .join(".cargo")
            .join("bin")
            .join(cargo_exe_name());
        if standard_path.exists() {
            return standard_path.to_string_lossy().to_string();
        }
    }

    // Platform-specific fallback locations
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            "C:\\Program Files\\Rust stable MSVC\\bin\\cargo.exe",
            "C:\\Program Files\\Rust stable GNU\\bin\\cargo.exe",
        ];
        for candidate in &candidates {
            if PathBuf::from(candidate).exists() {
                return candidate.to_string();
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let candidates = ["/opt/homebrew/bin/cargo", "/usr/local/bin/cargo"];
        for candidate in &candidates {
            if PathBuf::from(candidate).exists() {
                return candidate.to_string();
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let candidates = ["/usr/local/bin/cargo", "/usr/bin/cargo", "/snap/bin/cargo"];
        for candidate in &candidates {
            if PathBuf::from(candidate).exists() {
                return candidate.to_string();
            }
        }
    }

    // Fallback to "cargo" and let it fail with a clear error
    "cargo".to_string()
}

/// Returns the cargo executable name for the current platform.
#[inline]
fn cargo_exe_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "cargo.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "cargo"
    }
}

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
    template_path: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<String, io::Error> {
    let event_callback = Arc::new(event_callback);

    let template_dir = match create_template(template_path.clone()) {
        Ok(path) => path,
        Err(e) => {
            event_callback(
                "build-log".into(),
                format!("Failed to create template: {}", e),
            );
            return Err(e);
        }
    };

    match inject_model::<E>(model_txt, params, template_dir.clone()) {
        Ok(()) => (),
        Err(e) => {
            event_callback("build-log".into(), format!("Failed to inject model: {}", e));
            return Err(e);
        }
    };

    let dynlib_path = match build_template(template_dir.clone(), event_callback.clone()) {
        Ok(path) => path,
        Err(e) => {
            event_callback(
                "build-log".into(),
                format!("Failed to build template: {}", e),
            );
            return Err(e);
        }
    };

    // Emit completion message
    event_callback(
        "build-complete".into(),
        "Compilation finished successfully".into(),
    );

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
        let temp_dir = PathBuf::from(template_path);
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
/// The path to the template directory, or an error if creation failed.
pub fn dummy_compile(
    template_path: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<String, io::Error> {
    let event_callback = Arc::new(event_callback);

    let template_dir = create_template(template_path.clone())?;
    build_template(template_dir.clone(), event_callback.clone())?;

    // Emit completion message
    event_callback(
        "build-complete".into(),
        "Compilation finished successfully".into(),
    );

    Ok(template_dir.to_string_lossy().to_string())
}

/// Creates a new template project for model compilation.
///
/// This function creates a Rust project structure with the necessary dependencies
/// for compiling ODE models.
///
/// # Returns
/// The path to the created template directory, or an error if creation failed.
fn create_template(temp_dir: PathBuf) -> Result<PathBuf, io::Error> {
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
        let output = Command::new(find_cargo())
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

/// Utility function to get the temporary path for building models.
///
/// # Returns
///
/// A PathBuf representing the path to the temporary directory.
pub fn temp_path() -> PathBuf {
    env::temp_dir().join("exa_tmp")
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
/// Returns `()` on success, or an error if injection failed.
fn inject_model<E: Equation>(
    model_txt: String,
    params: Vec<String>,
    template_dir: PathBuf,
) -> Result<(), io::Error> {
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
    // cargo fmt is optional - don't fail if it's not available
    let _ = Command::new(find_cargo())
        .arg("fmt")
        .current_dir(&template_dir)
        .output();
    Ok(())
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
    event_callback: Arc<dyn Fn(String, String) + Send + Sync + 'static>,
) -> Result<PathBuf, io::Error> {
    let cargo_path = find_cargo();
    let mut command = Command::new(&cargo_path);
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

    let stdout_handle = stream_output(stdout, event_callback.clone());
    let stderr_handle = stream_output(stderr, event_callback.clone());

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
    event_callback: Arc<dyn Fn(String, String) + Send + Sync + 'static>,
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
