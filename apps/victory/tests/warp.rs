#![cfg(all(windows, not(feature = "diagnostics")))]

use std::process::Command;

#[test]
fn software_rendering_request_requires_diagnostics() {
    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .args(["run", "missing-warp-test.exe", "--execute"])
        .env("VICTORY_D3D12_WARP", "1")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("VICTORY_D3D12_WARP requires a diagnostics build"), "{stderr}");
}
