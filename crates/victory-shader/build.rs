//! Computes [`TRANSLATOR_FINGERPRINT`](crate::TRANSLATOR_FINGERPRINT) at
//! compile time.
//!
//! `victory-gpu`'s shader cache keys every translation on that fingerprint,
//! so it must cover exactly what can change what
//! [`translate`](crate::emit::translate) produces, and nothing else:
//! hashing the whole executable would discard every cached translation
//! whenever anything else in VIctory is rebuilt.
//!
//! The fingerprint is the BLAKE3 of these length-prefixed parts, in order:
//!
//! * every file under `src/`, each preceded by its path relative to the
//!   manifest (with `/` separators), sorted by that path,
//! * this crate's `Cargo.toml`,
//! * the workspace's `Cargo.lock`,
//! * the target triple and enabled target features,
//! * the enabled feature names (uppercased, as cargo reports them),
//!   sorted,
//! * the first line of `rustc --version`.
//!
//! The build script reruns when any of those change
//! (`cargo:rerun-if-changed`); a run always hashes all of them, so two
//! checkouts of the same sources, built the same way, agree.

use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let manifest_dir = Path::new(&manifest_dir);

    let mut h = blake3::Hasher::new();
    feed(&mut h, b"victory-shader translator fingerprint");

    let mut files = Vec::new();
    let sources = manifest_dir.join("src");
    println!("cargo:rerun-if-changed={}", sources.display());
    collect(&sources, manifest_dir, &mut files);
    files.sort();
    for (rel, path) in &files {
        feed(&mut h, rel.as_bytes());
        feed(&mut h, &read(path));
    }

    let manifest = manifest_dir.join("Cargo.toml");
    feed(&mut h, &read(&manifest));
    println!("cargo:rerun-if-changed={}", manifest.display());

    // The lockfile is at the workspace root: it picks the versions of the
    // dependencies the translator links against.
    let lockfile = manifest_dir.join("../../Cargo.lock");
    feed(&mut h, &read(&lockfile));
    println!("cargo:rerun-if-changed={}", lockfile.display());

    let target = std::env::var("TARGET").expect("cargo sets TARGET");
    feed(&mut h, target.as_bytes());
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    feed(&mut h, target_features.as_bytes());

    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(name, _)| name.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .collect();
    features.sort();
    feed(&mut h, features.join("\0").as_bytes());

    feed(&mut h, rustc_version().as_bytes());

    let fingerprint = h.finalize().to_hex();
    println!("cargo:rustc-env=VICTORY_SHADER_FINGERPRINT={fingerprint}");
}

/// Hashes one part, preceded by its length so no two different part
/// sequences concatenate to the same bytes.
fn feed(h: &mut blake3::Hasher, part: &[u8]) {
    h.update(&(part.len() as u64).to_le_bytes());
    h.update(part);
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn collect(dir: &Path, manifest_dir: &Path, files: &mut Vec<(String, PathBuf)>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap_or_else(|e| panic!("directory entry: {e}")).path();
        if path.is_dir() {
            collect(&path, manifest_dir, files);
        } else {
            let rel = path
                .strip_prefix(manifest_dir)
                .expect("a source file under the manifest")
                .to_string_lossy()
                .replace('\\', "/");
            files.push((rel, path));
        }
    }
}

/// What the compiler calls itself: a different toolchain may codegen the
/// translator differently, so it is part of the fingerprint.
fn rustc_version() -> String {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let ran = std::process::Command::new(rustc).arg("--version").output();
    match ran {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned(),
        // Still deterministic: every build without it agrees.
        _ => "<rustc --version unavailable>".to_owned(),
    }
}
