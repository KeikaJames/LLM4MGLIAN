use std::io::Write;
use std::process::{Command, Stdio};

pub fn convert(mode: &str, input: &str) -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dart_dir = std::path::Path::new(manifest_dir).join("../mongol_code-master");
    let script = dart_dir.join("tool/rust_bridge_convert.dart");
    let dart_home = dart_dir.join(".dart-home");

    let mut child = Command::new("dart")
        .arg("run")
        .arg(script)
        .arg(mode)
        .current_dir(dart_dir)
        .env("HOME", dart_home)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start Dart conversion process");

    child
        .stdin
        .as_mut()
        .expect("failed to open Dart stdin")
        .write_all(input.as_bytes())
        .expect("failed to write conversion input");

    let output = child
        .wait_with_output()
        .expect("failed to wait for Dart conversion process");

    if !output.status.success() {
        panic!(
            "Dart conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    String::from_utf8(output.stdout).expect("Dart conversion returned invalid UTF-8")
}
