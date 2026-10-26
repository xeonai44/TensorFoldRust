//! The fatbin half of the CUDA build: each .cu in `kernels/cuda`
//! compiled to a fatbin with the flags its Python extension passes, the
//! images embedded in the crate.
//!
//! Port of `zig/build/cuda.zig`'s pipeline (the `kernels` table, the
//! `torch_flags`/`torch_ops`/`glue` flag sets, and `fatbin()`
//! (cuda.zig:198)) onto cargo's build script:
//!
//! * `TENSORFOLD_NVCC` — nvcc (or a wrapper) that builds the CUDA
//!   kernel fatbins (`-Dnvcc`)
//! * `TENSORFOLD_FATBINS` — absolute directory of prebuilt
//!   `<name>.fatbin` files to embed (`-Dfatbins`)
//! * `TENSORFOLD_SM` — SASS targets, comma separated (default 121;
//!   `-Dsm`)
//!
//! Neither `TENSORFOLD_NVCC` nor `TENSORFOLD_FATBINS` builds the crate
//! host-only: `kernels::available` is false and every image is empty —
//! exactly upstream's no-nvcc, no-fatbins build. The nvcc version text
//! is an input of every fatbin (cuda.zig:90: "a new nvcc rebuilds them
//! all"), so the stamp each kernel is skipped by carries it; the
//! per-kernel depfile (`-MD -MF`) feeds the stamp the headers the way
//! upstream's build consumes it. The source tree is watched coarsely
//! (any touch reruns the script); each kernel's stamp decides whether
//! nvcc runs again — Zig's per-step laziness inside cargo's one
//! script.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and
/// which half/bf16 operators the headers define. Upstream:
/// `torch_flags` (cuda.zig:42).
const TORCH_FLAGS: &[&str] = &[
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
];

/// The torch-op replacements' qualification flags (runs/006): no
/// contraction, no flush to zero. Upstream: `torch_ops` (cuda.zig:9).
const TORCH_OPS: &[&str] = &["-O3", "--fmad=false", "--ftz=false"];

/// IEEE division and square root, no contraction or flush: host glue
/// references match bit for bit. Upstream: `glue` (cuda.zig:12).
const GLUE: &[&str] = &[
    "-O3",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// Each .cu in `kernels/cuda` (`src`, else `name`) with the flags its
/// Python extension passes in `extra_cuda_cflags`. Upstream: the
/// `kernels` table (cuda.zig:14) — arch_specific (the `a` gencode
/// suffix) is carried for the kernels that turn it on.
struct Kernel {
    name: &'static str,
    flags: &'static [&'static str],
    src: Option<&'static str>,
    arch_specific: bool,
}

const KERNELS: &[Kernel] = &[
    Kernel { name: "gdn", flags: &["-O3", "--fmad=false"], src: None, arch_specific: false }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    Kernel { name: "probe", flags: &["-O3"], src: None, arch_specific: false },
    Kernel { name: "qmm_group", flags: &["-O3"], src: None, arch_specific: false }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    Kernel { name: "qmm_prefill", flags: &["-O3"], src: None, arch_specific: false },
    Kernel { name: "experts", flags: &["-O3"], src: None, arch_specific: false }, // cuda/experts.py, tensorfold_experts_v7
    Kernel { name: "experts_prefill", flags: &["-O3"], src: None, arch_specific: false },
    Kernel { name: "experts_pack", flags: &["-O3"], src: None, arch_specific: false },
    Kernel { name: "prefill_attention", flags: &["-O3", "--fmad=false"], src: None, arch_specific: false }, // tensorfold_prefill_attention_v1
    Kernel { name: "scan_rows", flags: &["-O3", "--fmad=false"], src: None, arch_specific: false }, // nemotron_h/cuda/mamba.py
    Kernel { name: "nemotron_ops", flags: &["-O3"], src: None, arch_specific: false }, // ours: Nemotron's layouts and the serial feed
    Kernel { name: "lane_gemv", flags: &["-O3"], src: None, arch_specific: false }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    Kernel { name: "sample", flags: &["-O3", "--fmad=false", "--ftz=false"], src: None, arch_specific: false }, // ours: the Metal engine's keyed draws
    Kernel { name: "nemotron_norms", flags: GLUE, src: None, arch_specific: false }, // ours, each with a host reference (glue_ref.zig): Nemotron's glue
    Kernel { name: "nemotron_route", flags: GLUE, src: None, arch_specific: false },
    Kernel { name: "nemotron_mamba", flags: GLUE, src: None, arch_specific: false },
    Kernel { name: "nemotron_attention", flags: GLUE, src: None, arch_specific: false },
    Kernel { name: "nemotron_keyed", flags: GLUE, src: None, arch_specific: false },
    Kernel { name: "fp8_lane", flags: &["-O3"], src: None, arch_specific: false }, // nvfp4/qmmf.cu's FP8G device code, tensorfold_nvfp4_v3
    Kernel { name: "torch_argmax", flags: TORCH_OPS, src: Some("torch_ops/argmax"), arch_specific: false },
    Kernel { name: "torch_topk", flags: TORCH_OPS, src: Some("torch_ops/topk"), arch_specific: false },
    Kernel { name: "torch_pointwise", flags: TORCH_OPS, src: Some("torch_ops/pointwise"), arch_specific: false },
    Kernel { name: "torch_indexing", flags: TORCH_OPS, src: Some("torch_ops/indexing"), arch_specific: false },
    Kernel { name: "torch_movement", flags: TORCH_OPS, src: Some("torch_ops/movement"), arch_specific: false },
    Kernel { name: "torch_nemotron_constants", flags: TORCH_OPS, src: Some("torch_ops/nemotron_constants"), arch_specific: false },
];

fn main() {
    println!("cargo:rustc-check-cfg=cfg(kernels)");
    for var in ["TENSORFOLD_NVCC", "TENSORFOLD_FATBINS", "TENSORFOLD_SM"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sources = fs::canonicalize(manifest.join("../../kernels/cuda"))
        .expect("kernels/cuda stands in the workspace");
    println!("cargo:rerun-if-changed={}", sources.display());
    let nvcc = env::var("TENSORFOLD_NVCC").ok();
    let prebuilt = env::var("TENSORFOLD_FATBINS").ok();
    if nvcc.is_none() && prebuilt.is_none() {
        return; // host-only: available is false and every image is empty
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sms = env::var("TENSORFOLD_SM").unwrap_or_else(|_| "121".into());
    if let Some(dir) = &prebuilt {
        embed_prebuilt(&out_dir, dir);
    } else {
        build_fatbins(&out_dir, &nvcc.unwrap(), &sms);
    }
    println!("cargo:rustc-cfg=kernels");
}

/// Validate the complete named input set and copy it into the build —
/// upstream: `checkDistFatbins`'s "missing or empty CUDA fatbin"
/// refusal, plus `targets`' prebuilt wiring (cuda.zig:99).
fn embed_prebuilt(out_dir: &Path, dir: &str) {
    for kern in KERNELS {
        let from = Path::new(dir).join(format!("{}.fatbin", kern.name));
        let meta = fs::metadata(&from)
            .unwrap_or_else(|_| panic!("missing or empty CUDA fatbin: {}", from.display()));
        if meta.len() == 0 {
            panic!("missing or empty CUDA fatbin: {}", from.display());
        }
        let to = out_dir.join(format!("fatbin_{}", kern.name));
        fs::copy(&from, &to).expect("the fatbin copied into the build");
    }
}

/// `nvcc -fatbin` with torch's flags, the kernel's own and one
/// `-gencode` per SASS target, as the Python build passes them —
/// upstream: `fatbin` (cuda.zig:198). Each kernel skips when its whole
/// input set (the version text, the targets, the flags, the source and
/// every header its depfile listed) hashes to the stamp it left.
fn build_fatbins(out_dir: &Path, nvcc: &str, sms: &str) {
    let version = match Command::new(nvcc).arg("--version").output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        Ok(out) => panic!(
            "{nvcc} --version failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => panic!("cannot run {nvcc}: {e}"),
    };
    // The toolchain itself is a build input: a new nvcc rebuilds. A bare
    // name from PATH resolves through `which` (the --version above
    // proved it runnable, so this cannot fail in practice).
    let real = Path::new(nvcc);
    let watched = if real.exists() {
        real.to_path_buf()
    } else {
        let which = Command::new("which")
            .arg(nvcc)
            .output()
            .unwrap_or_else(|e| panic!("cannot resolve {nvcc} through which: {e}"));
        assert!(
            which.status.success(),
            "{nvcc} ran for --version but which cannot find it"
        );
        PathBuf::from(String::from_utf8_lossy(&which.stdout).trim_end())
    };
    if let Ok(link) = fs::read_link(&watched) {
        println!("cargo:rerun-if-changed={}", link.display());
    }
    println!("cargo:rerun-if-changed={}", watched.display());
    std::thread::scope(|scope| {
        let mut waits = Vec::with_capacity(KERNELS.len());
        for kern in KERNELS {
            let nvcc = nvcc.to_string();
            let sms = sms.to_string();
            let version = version.as_str();
            waits.push(scope.spawn(move || build_one(out_dir, &nvcc, &sms, version, kern)));
        }
        for wait in waits {
            wait.join()
                .expect("the fatbin build thread")
                .expect("the fatbin built");
        }
    });
}

fn build_one(out_dir: &Path, nvcc: &str, sms: &str, version: &str, kern: &Kernel) -> Result<(), String> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = fs::canonicalize(
        manifest
            .join("../../kernels/cuda")
            .join(format!("{}.cu", kern.src.unwrap_or(kern.name))),
    )
    .map_err(|e| format!("the kernel source: {e}"))?;
    let fatbin = out_dir.join(format!("fatbin_{}", kern.name));
    let depfile = out_dir.join(format!("{}.d", kern.name));
    let stamp_file = out_dir.join(format!("{}.stamp", kern.name));
    let mut stamp = fnv(version.as_bytes());
    stamp = fnv_mix(stamp, sms.as_bytes());
    stamp = fnv_mix(stamp, kern.flags.concat().as_bytes());
    stamp = fnv_mix(stamp, kern.src.unwrap_or(kern.name).as_bytes());
    stamp = fnv_mix(stamp, kern.arch_specific.to_string().as_bytes());
    stamp = fnv_mix(stamp, &fs::read(&src).map_err(|e| format!("{}: {e}", src.display()))?);
    if let Ok(old) = fs::read_to_string(&depfile) {
        for header in dep_paths(&old, &fatbin) {
            if let Ok(bytes) = fs::read(&header) {
                stamp = fnv_mix(stamp, &bytes);
            }
        }
    }
    let old = fs::read_to_string(&stamp_file).unwrap_or_default();
    if old == stamp.to_string() && fatbin.exists() {
        return Ok(()); // the same nvcc, targets, flags, source and headers
    }
    let mut cmd = Command::new(nvcc);
    cmd.arg("-fatbin");
    cmd.args(TORCH_FLAGS);
    cmd.args(kern.flags);
    let a = if kern.arch_specific { "a" } else { "" };
    for sm in sms.split(',') {
        cmd.arg(format!("-gencode=arch=compute_{sm}{a},code=sm_{sm}{a}"));
    }
    cmd.args(["-MD", "-MF"]).arg(&depfile);
    cmd.arg("-o").arg(&fatbin);
    cmd.arg(&src);
    let out = cmd.output().map_err(|e| format!("{nvcc}: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvcc failed for {}:\n{}", kern.name, String::from_utf8_lossy(&out.stderr)));
    }
    // The new depfile's headers enter the next stamp.
    if let Ok(fresh) = fs::read_to_string(&depfile) {
        for header in dep_paths(&fresh, &fatbin) {
            if let Ok(bytes) = fs::read(&header) {
                stamp = fnv_mix(stamp, &bytes);
            }
        }
    }
    fs::write(&stamp_file, stamp.to_string()).expect("the stamp written");
    Ok(())
}

/// The depfile's dependency paths: Make syntax (line continuations,
/// escaped spaces), the target before the first colon dropped.
fn dep_paths(dep: &str, target: &Path) -> Vec<PathBuf> {
    let mut text = String::with_capacity(dep.len());
    let mut lines = dep.lines();
    if let Some(first) = lines.next() {
        // The first line holds "<target>: <first deps>"; the target is
        // nvcc's own -o name, the output, not an input.
        let after = first.split_once(':').map(|(_, rest)| rest).unwrap_or("");
        text.push_str(after);
    }
    for line in lines {
        text.push(' ');
        text.push_str(line.trim_end_matches('\\'));
    }
    let mut paths = Vec::new();
    let mut name = String::new();
    let mut bytes = text.bytes().peekable();
    while let Some(b) = bytes.next() {
        match b {
            b'\\' => {
                if let Some(&next) = bytes.peek() {
                    if next == b' ' || next == b'\\' {
                        name.push(next as char);
                        bytes.next();
                        continue;
                    }
                }
                name.push('\\');
            }
            b' ' | b'\n' | b'\t' => {
                if !name.is_empty() {
                    let path = PathBuf::from(std::mem::take(&mut name));
                    if path != target && path.exists() {
                        paths.push(path);
                    }
                }
            }
            _ => name.push(b as char),
        }
    }
    if !name.is_empty() {
        let path = PathBuf::from(name);
        if path != target && path.exists() {
            paths.push(path);
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

/// FNV-1a: change detection for the stamp, nothing more.
fn fnv(bytes: &[u8]) -> u64 {
    fnv_mix(0xcbf29ce484222325, bytes)
}

fn fnv_mix(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
