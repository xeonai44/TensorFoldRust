//! Nemotron's CUDA engine, MTP head and lane backend behind the native
//! family interface.
//!
//! Port of `zig/src/families/nemotron/cuda_native.zig` (121 lines) —
//! the Loaded owns the boxed engine, its lanes backend (through the raw
//! bridge), the reuse target and the carveout; the prompt-cache
//! snapshots bridge the target to the lanes crate's store.

use tensorfold_cuda::{Carveout, Context};
use tensorfold_lanes::prompt_cache::{Saved, Snapshots, StoreError};
use tensorfold_lanes::Model;
use tensorfold_lanes::stream::Stream;

use crate::cuda_engine as engine;
use crate::cuda_engine::Engine;
use crate::cuda_lanes::Cuda;
use crate::cuda_reuse;
use crate::cuda_state::{MAX_ROWS, PREFILL_ROWS};

pub const MODEL_TYPE: &str = "nemotron_h";
pub const FORMATS: &[&str] = &["mlx-q4g64"];
pub const DEFAULT_CONTEXT: i64 = engine::DEFAULT_CONTEXT as i64;
pub const MAX_SEGMENTS: u32 = crate::cuda_segments::MAX as u32;
pub const PROMPT_ROWS: u32 = PREFILL_ROWS as u32;

/// Upstream: `Options` (cuda_native.zig:23).
#[derive(Clone, Copy)]
pub struct Options {
    pub context: usize,
    pub drafts: bool,
    pub segments: usize,
}

/// What the native server drives: the lane backend, the facts its
/// round loop reads, and how to free it. Upstream: `Loaded`
/// (cuda_native.zig:29) — the port's Loaded owns everything the
/// backend reaches (the engine boxed, the Cuda holding its raw
/// bridge), so the free is the Drop.
pub struct Loaded<'a, 'd> {
    /// The engine, boxed: the backend's and the target's bridge; the
    /// carveout it may hold KV planes of (dropped after the engine).
    e: Box<Engine<'a, 'd>>,
    /// The lanes backend over the engine (the raw bridge).
    pub cuda: Cuda<'a, 'd>,
    #[allow(dead_code)] // owned for its lifetime, not read
    carve: Option<Box<Carveout<'d>>>,
    target: cuda_reuse::Target<'a, 'd>,
}

impl<'a, 'd> Loaded<'a, 'd> {
    /// Upstream: `open` (cuda_native.zig:46) — the carveout first,
    /// then the engine, the head, both draw modes' graphs (the
    /// sampled sets with a placeholder rule; each stream uploads its
    /// own at prefill), the lanes, and the measurement.
    pub fn open(
        ctx: &'d Context<'d>,
        dir: &std::path::Path,
        kernels: Option<&str>,
        o: Options,
    ) -> Result<Loaded<'a, 'd>, crate::error::Error> {
        let carve = open_carveout(ctx)?;
        let carve_ref: Option<&'a Carveout<'d>> = match &carve {
            Some(c) => Some(unsafe {
                // SAFETY: the carveout outlives the engine (the Loaded
                // drops it after); the reference's lifetime is the
                // Loaded's own scope.
                &*(&**c as *const Carveout<'d>)
            }),
            None => None,
        };
        let mut e = Engine::init(
            ctx,
            dir,
            kernels,
            &engine_options(o, carve_ref),
        )?;
        let has_head = e.head.is_some();
        // Own-sequence graphs support both draw modes; drafting also
        // captures the head (upstream cuda_native.zig:51-59).
        if has_head {
            e.with_head(|h, e| h.capture(e))
                .ok_or(crate::error::Error::NoMtpHead)??;
        } else {
            e.capture_windows()?; // the one-row window: Engine::init captured windows only when drafting
        }
        // the sampled mode's sets: a placeholder rule now, each
        // stream's own uploaded at its prefill
        e.set_sampling(Some(tensorfold_lanes::Sampling {
            seed: 0,
            temperature: 1.0,
            ..Default::default()
        }))?;
        e.capture_windows()?;
        if has_head {
            e.with_head(|h, e| h.capture(e))
                .ok_or(crate::error::Error::NoMtpHead)??;
        }
        e.set_sampling(None)?;
        let mut e: Box<Engine<'a, 'd>> = e;
        let e_ptr: *mut Engine<'a, 'd> = &mut *e;
        let mut cuda = Cuda::init(e_ptr)?;
        let mut target = cuda_reuse::Target { e: e_ptr };
        let _ = &mut target;
        cuda.measure(dir)?;
        Ok(Loaded {
            e,
            carve,
            cuda,
            target,
        })
    }

    /// The backend the lane core drives. Upstream: `backend`
    /// (cuda_lanes.zig:60) — the port's Backend is a trait; this is
    /// the reference to it.
    pub fn backend(&mut self) -> &mut Cuda<'a, 'd> {
        &mut self.cuda
    }

    /// Upstream: `facts` (cuda_lanes.zig:75).
    pub fn facts(&self) -> Model<'_> {
        self.cuda.facts()
    }

    /// Upstream: `rows` (cuda_native.zig:68).
    pub fn rows(&self) -> u32 {
        if self.e.head.is_some() { MAX_ROWS as u32 } else { 1 }
    }

    /// Device bytes each admitted stream allocates for its own sequence
    /// (caches, state, head caches). Upstream: `stream_bytes`
    /// (cuda_native.zig:69).
    pub fn stream_bytes(&mut self) -> usize {
        self.e.seq_bytes()
    }

    /// Streams that allocate nothing: the first takes the engine's own
    /// sequence, whose buffers the graphs hold. Upstream:
    /// `free_streams` (cuda_native.zig:71).
    pub const FREE_STREAMS: u32 = 1;

    /// The prompt cache copies this engine (cuda_reuse.zig). Upstream:
    /// `target` (cuda_native.zig:73).
    pub fn target(&mut self) -> &mut cuda_reuse::Target<'a, 'd> {
        &mut self.target
    }

    /// The lone driver for a drafted stream (null: every stream in the
    /// lane core). Upstream: `lone` (cuda_native.zig:74) — the port's
    /// run over this Loaded.
    pub fn lone_run(
        &mut self,
        s: &mut Stream<'_>,
        hooks: crate::cuda_lone::Hooks<'_>,
    ) -> Result<bool, crate::error::Error> {
        if self.e.head.is_none() {
            return Ok(false); // upstream: lone null without a head
        }
        crate::cuda_lone::run(&mut self.cuda, s, hooks)
    }
}

impl Drop for Loaded<'_, '_> {
    fn drop(&mut self) {
        // upstream release (cuda_native.zig:91): the lanes, the head
        // and engine (the port's drops), then the carveout
        self.cuda.deinit();
        self.e.deinit();
    }
}

/// The engine's options from the family's. Upstream: the `.{ ... }`
/// at cuda_native.zig:49.
fn engine_options<'a, 'd>(
    o: Options,
    carve: Option<&'a Carveout<'d>>,
) -> engine::Options<'a, 'd> {
    let mut opts = engine::Options::new(carve);
    opts.context = Some(o.context);
    opts.mtp = o.drafts;
    opts.graphs = true;
    opts.segments = o.segments;
    opts
}

/// The snapshots bridge: the reuse target as the lanes store's
/// family face (upstream: `nemotron_snaps`, the vtable at
/// native/cuda.zig:15).
pub struct Snaps {
    pub target: *mut cuda_reuse::Target<'static, 'static>,
}

// Safety: the target pointer is the Loaded's (alive as long as the
// store the host builds over it); the store moves to the engine thread
// once, before any use.
unsafe impl Send for Snaps {}

impl Snapshots for Snaps {
    fn bytes(&self, at: u32) -> u64 {
        cuda_reuse::bytes(self.target.cast::<core::ffi::c_void>(), at)
    }

    fn save(
        &mut self,
        _owner: *mut core::ffi::c_void,
        at: u32,
    ) -> Result<Saved, StoreError> {
        // SAFETY: the Loaded's target; the engine stands at `at` with
        // no round in flight (the lane core's keep between chunks).
        let saved = unsafe { cuda_reuse::save(self.target.cast::<core::ffi::c_void>(), std::ptr::null_mut(), at) }
            .map_err(store_error)?;
        Ok(Box::new(RawCopy(saved)))
    }

    fn restore(
        &mut self,
        _owner: *mut core::ffi::c_void,
        saved: &Saved,
    ) -> Result<(), StoreError> {
        // SAFETY: the Loaded's target; the saved pointer the store
        // kept (the family's own bridge).
        let p = Self::saved_ptr(saved);
        // SAFETY: the Loaded's target; the saved pointer the store kept.
        unsafe { cuda_reuse::restore(self.target, p, 0) }.map_err(store_error)
    }

    fn drop_saved(&mut self, saved: Saved) {
        let p = Self::saved_ptr(&saved);
        drop(saved);
        // SAFETY: the family's own copy, freed once.
        unsafe { cuda_reuse::drop_saved(p) }
    }

    fn saved_ptr(saved: &Saved) -> *mut core::ffi::c_void {
        let inner = saved
            .downcast_ref::<RawCopy>()
            .expect("the family's raw copy");
        inner.0
    }
}

/// The family's raw saved copy inside the store's box (upstream's
/// Saved is the thin pointer itself).
struct RawCopy(*mut core::ffi::c_void);

// Safety: the copy lives on the engine thread once saved (the store
// moves there once, before any use) — the same bridge as the target's.
unsafe impl Send for RawCopy {}

fn store_error(e: crate::error::Error) -> StoreError {
    match e {
        crate::error::Error::NotAtMark => StoreError::NotAtMark,
        _ => StoreError::CopyFailed,
    }
}

/// A request this engine refuses, in words; None: none of its own.
/// Upstream: `explain` (cuda_native.zig:83).
pub fn explain(err: &crate::error::Error) -> Option<String> {
    match err {
        crate::error::Error::PromptTooLong => Some(
            "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens".into(),
        ),
        crate::error::Error::Cuda(tensorfold_cuda::Error::OutOfDeviceMemory) => Some(
            "the GPU had no memory left for this request's caches: retry once another request ends".into(),
        ),
        _ => None,
    }
}

/// TF_CUDA_CARVEOUT=1: the display carveout KV planes go to first
/// (cuda/carveout.zig); None when it is off. Upstream:
/// `openCarveout` (cuda_native.zig:101).
fn open_carveout<'d>(
    ctx: &'d Context<'d>,
) -> Result<Option<Box<Carveout<'d>>>, crate::error::Error> {
    let env = |n: &str| std::env::var(n).ok();
    let bytes = match tensorfold_cuda::carveout::requested(false, env("TF_CUDA_CARVEOUT").as_deref(), env("TF_CUDA_CARVEOUT_MIB").as_deref()) {
        Ok(b) => match b {
            Some(b) => b,
            None => return Ok(None),
        },
        Err(_) => return Ok(None),
    };
    let card = env("TF_DRM_CARD")
        .unwrap_or_else(|| tensorfold_cuda::carveout::DEFAULT_CARD.to_string());
    match Carveout::open(ctx.d, &card, bytes) {
        Ok(c) => {
            eprintln!(
                "display carveout: {} MiB from {card}; KV planes go there first",
                bytes >> 20
            );
            Ok(Some(Box::new(c)))
        }
        Err(err) => {
            eprintln!(
                "display carveout of {} MiB from {card}: {err} (needs nvidia_drm modeset=1, the card's device node and no display in use)",
                bytes >> 20
            );
            Err(err.into())
        }
    }
}
