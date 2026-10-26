//! The CUDA serve branch: the memory plan, the family open, the lane
//! host (native/cuda.zig's openWith + Host.serve), the way the Metal
//! branch wires its family on macOS.
//!
//! Port of `zig/src/native/cuda.zig`'s open half (cuda.zig:306-403)
//! onto the port's serve shape. Linux only (the CUDA family's target);
//! every refusal names the variable or flag that set it, upstream's
//! words.

use std::sync::Arc;

use tensorfold_cuda::{abi, usage, Context, Driver};
use tensorfold_lanes::backend::{Backend, DraftRequest, Feed, Verified, Window};
use tensorfold_lanes::stream::Stream;
use tensorfold_lanes::WallClock;

use crate::serve::ServeFlags;
use tensorfold_nemotron::cuda_memory as budget;
use tensorfold_nemotron::native_decisions as decide;
use tensorfold_nemotron::{cuda_engine, cuda_native};

const GIB: f64 = (1u64 << 30) as f64;

/// The engine the server drives plus the lines it prints.
pub struct Opened {
    pub engine: Arc<dyn tensorfold_engine::Engine>,
    pub startup: String,
    pub chunk_step: u32,
    pub context_window: u32,
    pub prefill_step: u32,
    pub prompt_cache: bool,
}

/// native/cuda.zig's openWith + Host.serve: the pool, the weights
/// check, the family open, the admission, the cache, the lane host.
///
/// Every problem is upstream's own words; `problem` carries it back.
#[allow(clippy::too_many_arguments)]
pub fn open(
    flags: &ServeFlags,
    dir: &std::path::Path,
    drafts: bool,
    problem: &mut String,
) -> Option<Opened> {
    // The window (cuda.zig:307-315): the model's, the flag inside it.
    let native = decide::model_context(dir);
    let window = match decide::context_window(flags.context, native, cuda_native::DEFAULT_CONTEXT) {
        Ok(w) => w,
        Err(e) => {
            *problem = match e {
                decide::WindowError::Negative => {
                    format!("--context {}: a token count, or 0 for the model's window", flags.context.unwrap_or(0))
                }
                decide::WindowError::NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count".into(),
                decide::WindowError::PastNative => format!(
                    "--context {} exceeds this model's {}-token window",
                    flags.context.unwrap_or(0),
                    native
                ),
            };
            return None;
        }
    };
    let (lanes, lanes_fixed) = flags.lanes();
    // The device and the segments (cuda.zig:316-322): the flag, else
    // the variable, else the default — upstream's setting rule.
    let device = match decide::setting(flags.device, "TF_CUDA_DEVICE", 0, problem) {
        Some(d) => d,
        None => return None,
    };
    let segments = match decide::setting(flags.segments, "TF_CUDA_SEGMENTS", 1, problem) {
        Some(s) => s,
        None => return None,
    };
    if segments < 1 || segments > cuda_native::MAX_SEGMENTS {
        let named = if flags.segments.is_some() { "--segments " } else { "TF_CUDA_SEGMENTS=" };
        *problem = format!(
            "{named}{segments}: whole prompt chunks a call, 1 to {}",
            cuda_native::MAX_SEGMENTS
        );
        return None;
    }
    // The GPU (cuda.zig:323-350).
    let driver = match Driver::open() {
        Ok(d) => d,
        Err(e) => {
            *problem = format!("no CUDA driver ({e})");
            return None;
        }
    };
    let count = match driver.device_count() {
        Ok(c) => c,
        Err(e) => {
            *problem = format!("no CUDA device count ({e})");
            return None;
        }
    };
    if device as i32 >= count {
        let named = if flags.device.is_some() { "--device " } else { "TF_CUDA_DEVICE=" };
        *problem = format!(
            "{named}{device}: this machine has {count} CUDA device{} (0 to {})",
            if count == 1 { "" } else { "s" },
            count.max(1) - 1
        );
        return None;
    }
    // the leak the lane thread needs (the process holds the GPU)
    let driver: &'static Driver = Box::leak(Box::new(driver));
    let ctx = match Context::init(driver, device as i32) {
        Ok(c) => c,
        Err(e) => {
            *problem = format!("the CUDA device does not open ({e})");
            return None;
        }
    };
    let ctx: &'static Context<'static> = Box::leak(Box::new(ctx));
    if let Err(e) = ctx.make_current() {
        *problem = format!("the CUDA context does not bind ({e})");
        return None;
    }
    let capability = ctx.capability().unwrap_or(0);
    let name = match ctx.name(&mut [0u8; 256]) {
        Ok(n) => n,
        Err(_) => "GPU".into(),
    };
    let env = |n: &str| std::env::var(n).ok();
    let kernels: Option<String> = match tensorfold_cuda::aot::pick(
        env("TENSORFOLD_CUDA_KERNELS").as_deref(),
        capability,
    ) {
        tensorfold_cuda::aot::Pick::Native => None,
        tensorfold_cuda::aot::Pick::Captured(dir) => Some(dir),
        tensorfold_cuda::aot::Pick::Missing(dir) => {
            *problem = format!(
                "TENSORFOLD_CUDA_KERNELS={dir} holds no aot.json: give a captured set's folder, or native for our own kernels"
            );
            return None;
        }
    };
    // The pool (cuda.zig:351 + 110-129): the reserve, the cap, the
    // meminfo for unified.
    let before = match pool(ctx, problem) {
        Some(p) => p,
        None => return None,
    };
    let weights = decide::weight_bytes(dir);
    let held0 = usage(false).device as u64;
    if weights > before.room(held0) {
        *problem = format!(
            "the checkpoint's {:.1} GiB of weights do not fit the {:.1} GiB the CUDA memory budget grants ({:.1} GiB free less a {:.1} GiB reserve{}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
            weights as f64 / GIB,
            before.room(held0) as f64 / GIB,
            before.free as f64 / GIB,
            before.reserve as f64 / GIB,
            if before.limit.is_some() { ", under TENSORFOLD_CUDA_MEMORY_LIMIT_GB" } else { "" },
        );
        return None;
    }
    // The family (cuda.zig:358-361).
    let mut loaded = match cuda_native::Loaded::open(
        ctx,
        dir,
        kernels.as_deref(),
        cuda_native::Options {
            context: window as usize,
            drafts,
            segments: segments as usize,
        },
    ) {
        Ok(l) => l,
        Err(e) => {
            *problem = format!(
                "the native CUDA engine cannot load {} with {} glue kernels ({e})",
                dir.display(),
                kernels.as_deref().unwrap_or("its own"),
            );
            return None;
        }
    };
    let model = (usage(false).device as u64).saturating_sub(held0);
    let after = match pool(ctx, problem) {
        Some(p) => p,
        None => return None,
    };
    let room = after.room(model);
    let free = cuda_native::Loaded::FREE_STREAMS;
    let stream_bytes = loaded.stream_bytes();
    let streams = match budget::admit(room as u64, stream_bytes as u64, free, lanes, lanes_fixed) {
        Ok(s) => s,
        Err(e) => {
            *problem = match e {
                budget::MemoryError::NoStream => format!(
                    "the CUDA memory budget fits no stream: one at a {window}-token window takes {:.2} GiB, and {:.2} GiB is left after the model's {:.2} GiB and the {:.1} GiB reserve; lower --context, or free device memory",
                    stream_bytes as f64 / GIB,
                    room as f64 / GIB,
                    model as f64 / GIB,
                    after.reserve as f64 / GIB,
                ),
                budget::MemoryError::TooMany => format!(
                    "--parallel {lanes} needs {:.2} GiB for its streams at a {window}-token window, and the CUDA memory budget leaves {:.2} GiB: serve --parallel {}, or lower --context",
                    stream_bytes as f64 / GIB * lanes.saturating_sub(free) as f64,
                    room as f64 / GIB,
                    free + (room / stream_bytes.max(1) as u64) as u32,
                ),
                _ => "the CUDA memory budget refuses".into(),
            };
            return None;
        }
    };
    // The cache (cuda.zig:377-390): the flag, else 16 GiB, inside the
    // room; the store over the family's snapshots.
    let cache = budget::cache_bytes(
        flags.prompt_cache_gib,
        room as u64,
        stream_bytes as u64 * (streams.saturating_sub(free)) as u64,
    );
    let segments_note = if segments > 1 {
        format!(", {segments} staggered segments a call")
    } else {
        String::new()
    };
    let cache_note = if cache > 0 {
        format!(", a {:.1} GiB prompt cache", cache as f64 / GIB)
    } else {
        ", no prompt cache".to_string()
    };
    let kernels_note = match &kernels {
        Some(dir) => format!("glue kernels captured at {dir}"),
        None => "own glue kernels".to_string(),
    };
    let startup = format!(
        "CUDA sm_{capability} device {device} ({name}{}): model {:.2} GiB; {streams} stream{} at once, {:.2} GiB each at a {window}-token window, of {:.1} GiB left after a {:.1} GiB reserve; prompts in {}-row chunks{segments_note}{cache_note}; {kernels_note}",
        if after.unified { ", memory shared with the host" } else { "" },
        model as f64 / GIB,
        if streams == 1 { "" } else { "s" },
        stream_bytes as f64 / GIB,
        room as f64 / GIB,
        after.reserve as f64 / GIB,
        cuda_native::PROMPT_ROWS,
    );
    // The lane core (cuda.zig:190-202's serve): the config from the
    // facts, the engine over the bound backend, the host with the
    // memory read, the cache, the explain, the lone driver.
    let facts = loaded.facts();
    let rows = loaded.rows();
    let core_config = match tensorfold_lanes::config::Config::new(&facts, rows, rows - 1) {
        Ok(c) => c,
        Err(e) => {
            *problem = format!("the lane core refuses the family's facts ({e})");
            return None;
        }
    };
    let core_config: &'static tensorfold_lanes::config::Config = Box::leak(Box::new(core_config));
    // the bound backend: the context current on the engine thread
    let bound: &'static mut Bound = Box::leak(Box::new(Bound {
        inner: &mut loaded.cuda,
        ctx,
    }));
    let clock: &'static mut WallClock = Box::leak(Box::new(WallClock::default()));
    let core = tensorfold_lanes::engine::Engine::init(core_config, bound, clock);
    let info = tensorfold_engine::Info {
        lanes: streams,
        context_window: window as u32,
        prefill_step: cuda_native::PROMPT_ROWS,
        prompt_cache: cache > 0,
        ..tensorfold_engine::Info::default()
    };
    let mut host_builder = tensorfold_lanes::LaneHost::new(core, info);
    // the memory read (cuda.zig:131): the device counters as the
    // server's /v1/memory
    host_builder = host_builder.with_memory(read_memory);
    if cache > 0 {
        // the store over the family's target (the Loaded's bridge)
        let target: *mut tensorfold_nemotron::cuda_reuse::Target<'static, 'static> =
            loaded.target() as *mut _;
        let rules = tensorfold_lanes::prompt_cache::Rules {
            lookahead: 1,
            planned: true,
            grid: cuda_native::PROMPT_ROWS,
            ..Default::default()
        };
        let store = tensorfold_lanes::prompt_cache::Store::new(
            cuda_native::Snaps { target },
            rules,
            cache,
        );
        host_builder = host_builder.with_cache(Box::new(store));
    }
    let host = host_builder.start();
    let engine: Arc<dyn tensorfold_engine::Engine> = host.clone();
    Some(Opened {
        engine,
        startup,
        chunk_step: cuda_native::PROMPT_ROWS,
        context_window: window as u32,
        prefill_step: cuda_native::PROMPT_ROWS,
        prompt_cache: cache > 0,
    })
}

/// The pool now (cuda.zig:110-129): the card's counts (meminfo's when
/// unified), the reserve, the limit.
fn pool(ctx: &Context<'_>, problem: &mut String) -> Option<budget::Pool> {
    let unified = ctx.attribute(abi::DEVICE_ATTRIBUTE_INTEGRATED).unwrap_or(0) != 0;
    let card = ctx.mem_info().ok()?;
    let text = if unified {
        std::fs::read_to_string("/proc/meminfo").ok()
    } else {
        None
    };
    let available = match budget::counts(unified, budget::MemInfo { total: card.total as u64, available: card.free as u64 }, text.as_deref()) {
        Ok(a) => a,
        Err(_) => {
            *problem = "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing CUDA unified-memory admission".into();
            return None;
        }
    };
    let reserve = match budget::reserve_bytes(std::env::var("TENSORFOLD_MEMORY_RESERVE_GIB").ok().as_deref(), available.total, unified) {
        Ok(r) => r,
        Err(_) => {
            *problem = format!(
                "TENSORFOLD_MEMORY_RESERVE_GIB={}: a number of GiB from {} to the memory's size",
                std::env::var("TENSORFOLD_MEMORY_RESERVE_GIB").unwrap_or_default(),
                budget::floor(unified).least,
            );
            return None;
        }
    };
    let limit = match budget::limit_bytes(std::env::var("TENSORFOLD_CUDA_MEMORY_LIMIT_GB").ok().as_deref()) {
        Ok(l) => l,
        Err(_) => {
            *problem = format!(
                "TENSORFOLD_CUDA_MEMORY_LIMIT_GB={}: a positive number of GiB whose byte count fits in a 64-bit size",
                std::env::var("TENSORFOLD_CUDA_MEMORY_LIMIT_GB").unwrap_or_default()
            );
            return None;
        }
    };
    Some(budget::Pool {
        free: available.available,
        total: available.total,
        reserve,
        limit,
        unified,
    })
}

/// The /v1/memory read (cuda.zig:131-134): the device counters.
fn read_memory(reset_peak: bool) -> Option<tensorfold_engine::Memory> {
    let u = usage(reset_peak);
    Some(tensorfold_engine::Memory {
        active: u.device,
        cache: 0,
        peak: u.peak,
        pinned: u.host,
    })
}

/// The family's backend with the context bound on the calling thread
/// (cuda.zig:180-294's bind wrapper over the vtable): the port's
/// trait delegation, the raw inner as the Loaded's bridge documents.
struct Bound {
    inner: *mut tensorfold_nemotron::cuda_lanes::Cuda<'static, 'static>,
    ctx: &'static Context<'static>,
}

impl Bound {
    fn bind(&self) {
        if let Err(e) = self.ctx.make_current() {
            eprintln!("cuCtxSetCurrent on the lane thread: {e}");
        }
    }
    fn inner(&mut self) -> &mut tensorfold_nemotron::cuda_lanes::Cuda<'static, 'static> {
        unsafe { &mut *self.inner }
    }
}

impl Backend for Bound {
    fn prefill(&mut self, s: *mut Stream<'_>) -> Result<(), tensorfold_lanes::Error> {
        self.bind();
        self.inner().prefill(s)
    }
    fn first(&mut self, s: *mut Stream<'_>, position: u64) -> Result<u64, tensorfold_lanes::Error> {
        self.bind();
        self.inner().first(s, position)
    }
    fn queue(&mut self, s: *mut Stream<'_>, feed: Feed, position: u64) -> Result<u64, tensorfold_lanes::Error> {
        self.bind();
        self.inner().queue(s, feed, position)
    }
    fn read(&mut self, handle: u64) -> Result<u32, tensorfold_lanes::Error> {
        self.bind();
        self.inner().read(handle)
    }
    fn verify(&mut self, windows: &[Window<'_>], out: &mut [Verified<'_>]) -> Result<(), tensorfold_lanes::Error> {
        self.bind();
        self.inner().verify(windows, out)
    }
    fn keep(&mut self, windows: &[Window<'_>], paths: &[&[u32]]) -> Result<(), tensorfold_lanes::Error> {
        self.bind();
        self.inner().keep(windows, paths)
    }
    fn draft(&mut self, requests: &[DraftRequest<'_, '_>]) -> Result<(), tensorfold_lanes::Error> {
        self.bind();
        self.inner().draft(requests)
    }
    fn first_row(&mut self, s: *mut Stream<'_>) -> Result<tensorfold_lanes::LogprobRow, tensorfold_lanes::Error> {
        self.bind();
        self.inner().first_row(s)
    }
    fn release(&mut self, s: *mut Stream<'_>) {
        self.bind();
        self.inner().release(s)
    }
}
