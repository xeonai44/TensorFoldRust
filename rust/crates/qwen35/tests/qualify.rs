#![cfg(target_os = "macos")]
//! Integration: the pinned checkpoint's real weights on this GPU.
//!
//! Mirrors what upstream does at every Metal backend open (backend.zig:26): load
//! the model, then run the qualify exactness check — wider lanes admit only when
//! their rows, committed caches and keep rolls equal one-row execution on the same
//! weights. For the port this is the Phase 1a exit gate
//! (docs/PORTING-PLAN.md): the Rust engine must make the same admission the Zig
//! engine would on the same machine.
//!
//! The model directory defaults to the pinned checkpoint
//! (`mlx-community/Qwen3.5-2B-MLX-4bit` @ 93760be4, downloaded with hash
//! verification); `QWEN35_MODEL_DIR` points elsewhere. Without a model or a Metal
//! device the test skips with a notice, the way upstream's GPU tests do.

use std::path::PathBuf;
use std::sync::Arc;
use tensorfold_qwen35::model::Model;
use tensorfold_qwen35::qualify;
use tensorfold_qwen35::state::{Scratch, BATCH_ROWS};

fn model_dir() -> PathBuf {
    std::env::var("QWEN35_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit"))
}

#[test]
fn pinned_checkpoint_qualifies() {
    let dir = model_dir();
    if !dir.join("config.json").is_file() {
        eprintln!("skipping: no model at {} (set QWEN35_MODEL_DIR)", dir.display());
        return;
    }
    if !tensorfold_metal::kernels_builtin() {
        eprintln!("skipping: no Metal toolchain in this build");
        return;
    }
    let started = std::time::Instant::now();
    let model = Model::load(&dir).expect("the pinned checkpoint loads");
    eprintln!(
        "loaded {} in {:.2}s, {:.2} GiB resident, {} pipelines compiled",
        dir.display(),
        started.elapsed().as_secs_f64(),
        model.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        tensorfold_qwen35::kernels::TOTAL,
    );
    // Upstream's backend init (backend.zig:24): the wide scratch at
    // max(chunk, batch_rows) rows; capacity is the engine's context bound.
    let mut wide = Scratch::new(Arc::clone(&model.gpu), 128.max(BATCH_ROWS), 256);
    let started = std::time::Instant::now();
    qualify::check(&model, &mut wide).expect("real-weight forwards stay exact");
    eprintln!(
        "qualified: wider lanes' rows, committed caches and keep rolls equal one-row execution ({:.2}s)",
        started.elapsed().as_secs_f64()
    );
}
