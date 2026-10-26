//! Compiles the Metal kernels a build embeds, the way upstream's build.zig does.
//!
//! Port of `build.zig`'s pipeline (lines 70-97 and 246-256):
//!   * `mlxMetalStd`: the Metal language version MLX's kernels need on this macOS
//!     (macOS 27: metal4.1, 26: metal4.0, 15: metal3.2, older: metal3.1).
//!   * `metallib`: `xcrun -sdk macosx metal <flags> -c SRC -o X.air`, then
//!     `xcrun -sdk macosx metallib -o X.metallib X.air...`.
//!   * The flags for hand-written kernels: `-std=<std> -fmetal-math-mode=safe
//!     -fmetal-math-fp32-functions=fast`.
//!
//! Phase 0 compiles `kernels/metal/runtime_tests.metal` — upstream's own small kernels
//! for "the Metal runtime's own GPU tests and dispatch benchmarks" — so the whole
//! pipeline is proven end to end before family kernels arrive in Phase 1.
//!
//! When the Metal toolchain is absent the build still succeeds: a placeholder
//! metallib is embedded and `kernels_builtin()` returns false, so the GPU tests skip
//! and the server keeps building (the same shape as upstream's engine-less builds).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Upstream: `build.zig:70  fn mlxMetalStd`.
fn metal_std() -> &'static str {
    if let Ok(out) = Command::new("sw_vers").arg("-productVersion").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        let major: u64 = text
            .trim()
            .split('.')
            .next()
            .and_then(|part| part.parse().ok())
            .unwrap_or(15);
        return match major {
            m if m >= 27 => "metal4.1",
            26 => "metal4.0",
            15..=25 => "metal3.2",
            _ => "metal3.1",
        };
    }
    "metal3.2"
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // The kernels live at the workspace root, one level above the crate.
    let kernel = manifest
        .join("../..")
        .join("kernels/metal/runtime_tests.metal");
    let kernel = kernel
        .canonicalize()
        .unwrap_or_else(|_| Path::new("kernels/metal/runtime_tests.metal").to_path_buf());
    println!("cargo:rerun-if-changed={}", kernel.display());
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let air = out_dir.join("runtime_tests.air");
    let lib = out_dir.join("runtime_tests.metallib");

    let std_flag = format!("-std={}", metal_std());
    let compile = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal"])
        .args([
            std_flag.as_str(),
            "-fmetal-math-mode=safe",
            "-fmetal-math-fp32-functions=fast",
            "-c",
        ])
        .arg(&kernel)
        .arg("-o")
        .arg(&air)
        .output();
    let linked = match compile {
        Ok(run) if run.status.success() => Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", "-o"])
            .arg(&lib)
            .arg(&air)
            .output(),
        Ok(run) => {
            eprintln!("metal compile failed on {}", kernel.display());
            eprintln!("{}", String::from_utf8_lossy(&run.stderr));
            Ok(Command::new("false").output().unwrap())
        }
        Err(err) => {
            eprintln!("could not run xcrun: {}", err);
            Ok(Command::new("false").output().unwrap())
        }
    };

    if let Ok(run) = &linked {
        if run.status.success() && fs::metadata(&lib).is_ok() {
            println!("cargo:rustc-env=TF_METALLIB_PATH={}", lib.display());
            println!("cargo:rustc-env=TF_METAL_KERNELS=1");
            return;
        }
        if !run.status.success() {
            eprintln!("metallib link failed: {}", String::from_utf8_lossy(&run.stderr));
        }
    }
    println!(
        "cargo:warning=no Metal toolchain (xcrun); GPU kernel tests are skipped and engine open will refuse until one is installed"
    );
    fs::write(&lib, []).expect("writing the placeholder metallib");
    println!("cargo:rustc-env=TF_METALLIB_PATH={}", lib.display());
    println!("cargo:rustc-env=TF_METAL_KERNELS=0");
}
