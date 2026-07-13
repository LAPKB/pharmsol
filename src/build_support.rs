use std::env;
use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[allow(unused_mut)]
fn new_command(program: &str) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

pub(crate) fn find_cargo() -> String {
    if let Ok(output) = Command::new("cargo").arg("--version").output() {
        if output.status.success() {
            return "cargo".to_string();
        }
    }

    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        let cargo_path = PathBuf::from(&cargo_home)
            .join("bin")
            .join(cargo_exe_name());
        if cargo_path.exists() {
            return cargo_path.to_string_lossy().to_string();
        }
    }

    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_default();

    if !home.is_empty() {
        let standard_path = PathBuf::from(&home)
            .join(".cargo")
            .join("bin")
            .join(cargo_exe_name());
        if standard_path.exists() {
            return standard_path.to_string_lossy().to_string();
        }
    }

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

    "cargo".to_string()
}

#[cfg(test)]
pub(crate) fn find_rustup() -> Option<String> {
    if let Ok(output) = Command::new("rustup").arg("--version").output() {
        if output.status.success() {
            return Some("rustup".to_string());
        }
    }

    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        let rustup_path = PathBuf::from(&cargo_home)
            .join("bin")
            .join(rust_tool_exe_name("rustup"));
        if rustup_path.exists() {
            return Some(rustup_path.to_string_lossy().to_string());
        }
    }

    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_default();

    if !home.is_empty() {
        let standard_path = PathBuf::from(&home)
            .join(".cargo")
            .join("bin")
            .join(rust_tool_exe_name("rustup"));
        if standard_path.exists() {
            return Some(standard_path.to_string_lossy().to_string());
        }
    }

    None
}

#[cfg(test)]
pub(crate) fn rustc_host_target() -> Result<String, io::Error> {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = new_command(&rustc).arg("-vV").output()?;
    if !output.status.success() {
        return Err(io::Error::other("failed to run `rustc -vV`"));
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(|line| line.trim().to_string())
        .ok_or_else(|| io::Error::other("`rustc -vV` did not report a host target"))
}

#[cfg(test)]
pub(crate) fn rustup_installed_targets() -> Result<Vec<String>, io::Error> {
    let rustup = find_rustup()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "rustup was not found"))?;
    let output = new_command(&rustup)
        .args(["target", "list", "--installed"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            "failed to run `rustup target list --installed`",
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn cargo_exe_name() -> &'static str {
    rust_tool_exe_name("cargo")
}

fn rust_tool_exe_name(tool: &'static str) -> &'static str {
    #[cfg(target_os = "windows")]
    {
        match tool {
            "cargo" => "cargo.exe",
            "rustup" => "rustup.exe",
            _ => tool,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        tool
    }
}

pub(crate) fn create_cargo_template(
    temp_dir: PathBuf,
    cargo_toml_content: &str,
) -> Result<PathBuf, io::Error> {
    if !temp_dir.exists() {
        fs::create_dir_all(&temp_dir)?;
    }

    let template_dir = temp_dir.join("template");
    let cargo_toml_path = template_dir.join("Cargo.toml");
    let src_dir = template_dir.join("src");
    let needs_scaffold = !template_dir.exists() || !src_dir.exists();

    if needs_scaffold {
        if template_dir.exists() {
            fs::remove_dir_all(&template_dir)?;
        }
        fs::create_dir_all(&src_dir)?;
        fs::write(&cargo_toml_path, cargo_toml_content)?;
    } else if !cargo_toml_path.exists() {
        fs::write(&cargo_toml_path, cargo_toml_content)?;
    } else {
        let existing_content = fs::read_to_string(&cargo_toml_path)?;
        if existing_content.trim() != cargo_toml_content.trim() {
            tracing::info!("template manifest changed, invalidating generated artifact cache");
            fs::write(&cargo_toml_path, cargo_toml_content)?;
            let target_dir = template_dir.join("target");
            if target_dir.exists() {
                fs::remove_dir_all(&target_dir)?;
            }
        }
    }

    Ok(template_dir)
}

pub(crate) fn write_template_source(
    template_dir: impl AsRef<Path>,
    source: &str,
) -> Result<(), io::Error> {
    let template_dir = template_dir.as_ref();
    fs::write(template_dir.join("src").join("lib.rs"), source)?;

    let cargo_path = find_cargo();
    let _ = new_command(&cargo_path)
        .arg("fmt")
        .current_dir(template_dir)
        .output();
    Ok(())
}

pub(crate) fn build_cargo_template(
    template_path: PathBuf,
    event_callback: Arc<dyn Fn(String, String) + Send + Sync + 'static>,
    backend: &'static str,
    model_name: String,
    target: Option<&str>,
    artifact_path: &[&str],
) -> Result<PathBuf, io::Error> {
    let cargo_path = find_cargo();
    let target_dir = template_path.join("target");

    let mut started_message = format!(
        "Compiling {backend} model `{}` in {}",
        model_name,
        template_path.display()
    );
    if let Some(target) = target {
        started_message.push_str(&format!(" for target `{target}`"));
    }
    event_callback("started".into(), started_message);

    let mut command = new_command(&cargo_path);
    command
        .arg("build")
        .arg("--release")
        // .arg("--quiet")
        .arg("--target-dir")
        .arg(&target_dir)
        .current_dir(&template_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(target) = target {
        command.arg("--target").arg(target);
    }

    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    let stdout_handle = stream_output(stdout, event_callback.clone(), model_name.clone());
    let stderr_handle = stream_output(stderr, event_callback.clone(), model_name);

    let status = child.wait()?;
    stdout_handle
        .join()
        .expect("Failed to join stdout thread")?;
    stderr_handle
        .join()
        .expect("Failed to join stderr thread")?;

    if !status.success() {
        return Err(io::Error::other("Failed to build the template"));
    }

    let mut output_path = target_dir;
    for segment in artifact_path {
        output_path = output_path.join(segment);
    }
    Ok(output_path)
}

#[cfg(feature = "dsl-aot")]
pub(crate) fn native_cdylib_filename_for_target(
    crate_name: &str,
    cargo_target: Option<&str>,
) -> String {
    if target_uses_windows_dll(cargo_target) {
        format!("{crate_name}.dll")
    } else if target_uses_apple_dylib(cargo_target) {
        format!("lib{crate_name}.dylib")
    } else {
        format!("lib{crate_name}.so")
    }
}

#[cfg(feature = "dsl-aot")]
fn target_uses_windows_dll(cargo_target: Option<&str>) -> bool {
    cargo_target.map_or(cfg!(target_os = "windows"), |target| {
        target.contains("windows")
    })
}

#[cfg(feature = "dsl-aot")]
fn target_uses_apple_dylib(cargo_target: Option<&str>) -> bool {
    cargo_target.map_or(cfg!(target_os = "macos"), |target| {
        target.contains("apple") || target.contains("darwin") || target.contains("ios")
    })
}

fn stream_output<R: Read + Send + 'static>(
    reader: R,
    event_callback: Arc<dyn Fn(String, String) + Send + Sync + 'static>,
    model_name: String,
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
            let _ = &model_name;
            event_callback("log".into(), output);
        }

        Ok(())
    })
}
