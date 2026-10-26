#![cfg(target_os = "macos")]
//! Integration: the lane core driving the qwen35 Metal backend on the pinned
//! checkpoint's real weights.
//!
//! The Phase 1b engine gate (docs/PORTING-PLAN.md): the round loop's shared rounds
//! and one-row paths must produce the same tokens on the real model — the
//! engine_test.zig contract, but with the GPU backend instead of the fake. A
//! serial one-row reference (the lane core with a one-lane config, one stream,
//! `drafts=false` — upstream's plain reference) anchors the loop.
//!
//! The model directory defaults to the pinned checkpoint
//! (`mlx-community/Qwen3.5-2B-MLX-4bit` @ 93760be4); `QWEN35_MODEL_DIR` points
//! elsewhere. Without a model or a Metal device the test skips with a notice, the
//! way upstream's GPU tests do.

use std::path::PathBuf;
use std::sync::Arc;
use tensorfold_engine::Sampling;
use tensorfold_lanes::config::{Config, Model};
use tensorfold_lanes::engine::Engine;
use tensorfold_lanes::stream::{Spec, Stream};
use tensorfold_lanes::WallClock;
use tensorfold_qwen35::backend::{Metal, Options};
use tensorfold_qwen35::model::Model as QwenModel;

fn model_dir() -> PathBuf {
    std::env::var("QWEN35_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit"))
}

/// The lane config from the backend's facts, at upstream's widths: the model's
/// own geometry bounds the rows, and `lanes` bounds the streams a shared round
/// admits (upstream serve uses --parallel's number; the test exercises the
/// shared path with a small bound).
fn lane_config(backend: &Metal, max_rows: u32) -> Config {
    let facts: Model<'_> = backend.facts();
    Config::new(&facts, max_rows, max_rows.saturating_sub(1).max(1)).unwrap()
}

fn generate(
    model: &Arc<QwenModel>,
    options: &Options,
    cfg: &Config,
    prompts: &[Vec<u32>],
    max_new: u32,
    sampling: Option<Sampling>,
) -> Vec<Vec<u32>> {
    let mut backend = Metal::new(Arc::clone(model), *options).unwrap();
    let mut clock = WallClock::default();
    let mut engine = Engine::init(cfg, &mut backend, &mut clock);
    let mut streams: Vec<Box<Stream<'_>>> = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let id: &'static str = Box::leak(format!("s{}", i).into_boxed_str());
        streams.push(Box::new(Stream::init(Spec {
            id,
            prompt,
            max_new,
            eos: &[],
            sampling,
            drafts: false,
            ..Spec::new(id, prompt, max_new)
        })));
    }
    for i in 0..streams.len() {
        unsafe { engine.add_stream(&mut *streams[i]) }.unwrap();
    }
    while engine.active_count() > 0 {
        engine.step().unwrap();
    }
    streams.iter().map(|s| s.emitted().to_vec()).collect()
}

#[test]
fn shared_rounds_equal_solo_on_real_weights() {
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
    let model = Arc::new(QwenModel::load(&dir).unwrap());
    eprintln!(
        "loaded {} in {:.2}s, {:.2} GiB resident",
        dir.display(),
        started.elapsed().as_secs_f64(),
        model.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    // A short prompt of ordinary tokens, enough to warm every layer kind (the
    // 3:1 pattern runs the full attention layer at index 3).
    let prompt: Vec<u32> = [825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373]
        .iter()
        .copied()
        .collect();
    let options = Options::new(2048);
    let max_new = 24;

    // One stream, plain one-row rounds — the serial reference (upstream's
    // `--no-drafts` plain path with family_streams off by construction: one
    // stream never shares).
    let solo_cfg = {
        let backend = Metal::new(Arc::clone(&model), options).unwrap();
        lane_config(&backend, 16)
    };
    let solo = generate(&model, &options, &solo_cfg, &[prompt.clone()], max_new, None);

    // Two streams at once, the same prompt twice: the shared-round path must
    // commit what each commits alone (engine_test.zig:150's contract, on the
    // real model).
    let shared_cfg = {
        let backend = Metal::new(Arc::clone(&model), options).unwrap();
        lane_config(&backend, 32)
    };
    let shared = generate(
        &model,
        &options,
        &shared_cfg,
        &[prompt.clone(), prompt.clone()],
        max_new,
        None,
    );

    assert_eq!(solo[0], shared[0], "the first stream's shared tokens differ from solo");
    assert_eq!(shared[0], shared[1], "the two shared streams differ from each other");
    eprintln!(
        "generated {} tokens; shared rounds equal solo ({:.2}s total)",
        solo[0].len(),
        started.elapsed().as_secs_f64()
    );
    assert!(!solo[0].is_empty());
}
