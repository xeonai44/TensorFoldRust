//! Opening an engine for a checkpoint: what a server asks, and the engine-less build.
//!
//! Port of the `Open`/`Opened` declarations of `zig/src/core/engine_api.zig`
//! (lines 162-192) plus `zig/src/server/engines.zig`'s open shape and the engine-less
//! backend `zig/src/native/none.zig`.

use crate::engine::Engine;
use crate::Family;
use std::path::Path;

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an
/// engine reads.
///
/// Upstream: `engine_api.zig:162  pub const Open = struct { ... }`. Defaults mirror the
/// Zig ones: `lanes = 8`, `drafts = true`, `drafter_bits = 4`, `learn_min_free_gib = 4`,
/// `learn_gib = 32`.
#[derive(Clone, Copy, Debug)]
pub struct Open<'a> {
    pub dir: &'a Path,
    pub model_type: &'a str,
    pub context: Option<i64>,
    pub lanes: u32,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses
    /// instead of serving fewer.
    pub lanes_fixed: bool,
    pub drafts: bool,
    /// --drafter: a draft model's directory for families that load one (Qwen3.8-27B's
    /// DFlash2); None: no drafter.
    pub drafter: Option<&'a str>,
    /// --drafter-bits: 4 prepares the drafter's weights in 4 bits; 0 keeps them bf16.
    pub drafter_bits: u8,
    pub speed_up: Option<&'a str>,
    pub prompt_cache_gib: Option<f64>,
    pub prompt_cache_over_cap: bool,
    /// --learn: where shared prompt states are kept on disk for later sessions and
    /// servers (None: off).
    pub learn: Option<&'a str>,
    /// --learn-min-free-gib: filesystem space kept free before any learned-state write.
    pub learn_min_free_gib: f64,
    /// --learn-gib: what learned states may take on disk, every model and build
    /// together.
    pub learn_gib: f64,
    /// --slide: Sliding Weights learns into the served weights, live (off: learn
    /// requests are refused).
    pub slide: bool,
    /// --device and --segments (CUDA); None: the backend's environment fallback, then
    /// its default.
    pub device: Option<u32>,
    pub segments: Option<u32>,
}

impl<'a> Open<'a> {
    pub fn new(dir: &'a Path, model_type: &'a str) -> Self {
        Open {
            dir,
            model_type,
            context: None,
            lanes: 8,
            lanes_fixed: false,
            drafts: true,
            drafter: None,
            drafter_bits: 4,
            speed_up: None,
            prompt_cache_gib: None,
            prompt_cache_over_cap: false,
            learn: None,
            learn_min_free_gib: 4.0,
            learn_gib: 32.0,
            slide: false,
            device: None,
            segments: None,
        }
    }
}

/// An opened engine. The Zig record carries an explicit `close` function pointer
/// (`engine_api.zig:192`); in Rust the `Box<dyn Engine>` drop runs it, so `close` is the
/// drop.
pub struct Opened {
    pub engine: Box<dyn Engine>,
}

/// The backend a binary was built with: what it serves and how it opens engines.
///
/// Port of the `native.open`/`native.backends`/`native.families`/`native.chip` shape the
/// server's `engines.zig` calls, and of `zig/src/native/none.zig` (the engine-less
/// build) as the first implementation.
pub trait Backend: Send + Sync {
    /// The chip name the backend reports for `capabilities --json`, or None.
    fn chip(&self) -> Option<String>;
    /// The backend names (`native.backends`); empty for the engine-less build.
    fn names(&self) -> &'static [&'static str];
    /// The families this backend reads (`native.families`); empty for the engine-less
    /// build.
    fn families(&self) -> &'static [Family];
    /// The engine for the checkpoint in `request.dir`, or None with `problem` saying why
    /// (`native.open`).
    fn open(&self, request: &Open<'_>, problem: &mut String) -> Option<Opened>;
}

/// No native backend: the engine-less build the server's tests and parity suite run
/// (port of `zig/src/native/none.zig`).
pub struct NoneBackend;

impl Backend for NoneBackend {
    fn chip(&self) -> Option<String> {
        None
    }

    fn names(&self) -> &'static [&'static str] {
        &[]
    }

    fn families(&self) -> &'static [Family] {
        &[]
    }

    fn open(&self, request: &Open<'_>, problem: &mut String) -> Option<Opened> {
        // Upstream: none.zig:13 — the message points 0.6.6's Python engine at the
        // model types the native build does not serve.
        *problem = format!(
            "the native engine has no backend for {} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6",
            request.model_type
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors none.zig: the engine-less backend names every checkpoint a problem.
    #[test]
    fn none_backend_refuses_with_the_upstream_message() {
        let b = NoneBackend;
        assert!(b.names().is_empty());
        assert!(b.families().is_empty());
        let mut problem = String::new();
        let dir = std::path::Path::new("/models/x");
        let out = b.open(&Open::new(dir, "qwen3_5"), &mut problem);
        assert!(out.is_none());
        assert!(problem.contains("no backend for qwen3_5 checkpoints"));
    }
}
