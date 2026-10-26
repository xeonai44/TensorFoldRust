//! The Nemotron CUDA engine (engine.py): weights, kernels and one
//! sequence's buffers; prefill, windows and commits.
//!
//! Port of `zig/src/families/nemotron/cuda_engine.zig` (475 lines).
//! Upstream's pointer graph — an engine whose Head holds the engine
//! back, a `bound` sequence that may be the engine's own view —
//! becomes two named unsafe shapes, each with its invariant in one
//! place: `with_head` (the head lives in a stable Box, no Head↔Engine
//! field pair is ever written through both references at once, and
//! nothing replaces `head` while its borrow is live) and `bound` (a
//! raw `*mut Seq` to the own view or a caller's boxed sequence, whose
//! address the caller keeps until `free_seq`).

use tensorfold_cuda::{
    abi, Carveout, Context, Event, Exec, HostBuffer, Stream,
    graph, segments::Runner,
};
use tensorfold_lanes::Sampling;

use crate::config::Config;
use crate::cuda_forward::{Forward, Seg, Side};
use crate::cuda_kernels::{Kernels, Ops};
use crate::cuda_mtp::Head;
use crate::cuda_sampler as sampler;
use crate::cuda_state::{Buffers, Seq, MAX_ROWS, PREFILL_ROWS, CHUNK_KEYS};
use crate::cuda_weights as weights;
use crate::error::Error;
use tensorfold_core::heat;

/// nemotron_h.cuda.CONTEXT: prompt plus reply tokens when --context is
/// not given, as `tensorfold serve` sizes it. Upstream:
/// `default_context` (cuda_engine.zig:20).
pub const DEFAULT_CONTEXT: usize = 16384;

/// Upstream: `Options` (cuda_engine.zig:22).
pub struct Options<'a, 'd> {
    pub context: Option<usize>,
    pub mtp: bool,
    pub graphs: bool,
    /// The rule the graphs compile in; null or temperature 0 decodes
    /// greedily.
    pub sampling: Option<Sampling>,
    /// Whole prompt chunks a call runs as staggered segments (1: one
    /// chunk at a time).
    pub segments: usize,
    /// KV planes go to the display carveout first while it holds them
    /// (it outlives the engine).
    pub carveout: Option<&'a Carveout<'d>>,
}

impl<'a, 'd> Options<'a, 'd> {
    pub fn new(carveout: Option<&'a Carveout<'d>>) -> Options<'a, 'd> {
        Options {
            context: None,
            mtp: true,
            graphs: true,
            sampling: None,
            segments: 1,
            carveout,
        }
    }
}

/// Asked before each prompt chunk (or segmented call): true stops the
/// prompt with Cancelled. Upstream: `Cancel` (cuda_engine.zig:32).
pub struct Cancel<'c> {
    pub check: &'c mut dyn FnMut() -> bool,
}

impl Cancel<'_> {
    pub fn now(c: Option<&mut Cancel<'_>>) -> bool {
        match c {
            Some(x) => (x.check)(),
            None => false,
        }
    }
}

/// Grid points where a prompt pass copies its state (ascending),
/// called once each chunk ending there is queued. Upstream: `Keep`
/// (cuda_engine.zig:43).
pub struct Keep<'k> {
    pub at: &'k [u32],
    pub call: &'k mut dyn FnMut(u32),
}

/// Serial rounds a host keeps queued ahead of the one it reads.
/// Upstream: `lookahead` (cuda_engine.zig:50).
pub const LOOKAHEAD: usize = 4;

/// Streams a shared window holds at most (their rows together at most
/// MAX_ROWS). Upstream: `max_streams` (cuda_engine.zig:53).
pub const MAX_STREAMS: usize = 8;

// pinned words: the window's meta row, its ids, its sampled ids, a
// prompt chunk's ids, then 64 a shared window's stream
const PIN_META: usize = 0;
const PIN_IDS: usize = 4;
const PIN_SAMPLED: usize = 32;
const PIN_PROMPT: usize = 64;
const PIN_SHARED: usize = PIN_PROMPT + crate::cuda_segments::MAX * PREFILL_ROWS;

/// One stream's part of a shared window: its sequence, the ids its
/// first rows take from the host, and its rows. Upstream: `Shared`
/// (cuda_engine.zig:63) — the sequence an external caller owns and
/// the engine binds.
pub struct Shared<'a, 'd> {
    pub seq: *mut Seq<'a, 'd>,
    pub ids: &'a [u32],
    pub rows: usize,
}

/// Upstream: `Engine` (cuda_engine.zig:65) — boxed by its owner so the
/// head's raw self-pointer stays valid; the port's Drop runs deinit.
pub struct Engine<'a, 'd> {
    pub ctx: &'d Context<'d>,
    pub stream: Stream<'d>,
    /// The decode MoE's shared expert runs here.
    pub side: Side<'d>,
    pub k: Kernels<'d>,
    pub w: weights::Weights<'d>,
    pub b: Buffers<'a, 'd>,
    pub c: Config,
    pub max_len: usize,
    /// Attention chunk programs a row: a captured _chunk variant's
    /// NCH covering max_len (bits ignore it).
    pub nch: usize,
    pub pinned: HostBuffer<'d>,
    /// Mapped: a serial round's kernel writes its token here, by
    /// position.
    pub history: HostBuffer<'d>,
    pub history_dev: u64,
    pub serial: Option<Exec<'d>>,
    /// [greedy, sampled]: argmax or sample.cu.
    pub windows: [[Option<Exec<'d>>; MAX_ROWS + 1]; 2],
    pub done: [Event<'d>; LOOKAHEAD],
    /// The last window's uploads have read the pinned words.
    pub copied: Event<'d>,
    pub sampled_ready: Event<'d>,
    pub pos: usize,
    pub parity: usize,
    pub prev_keep: usize,
    pub rows: usize,
    pub sampling: Option<Sampling>,
    /// The engine's own buffers, which the graphs were captured on.
    pub own: Seq<'a, 'd>,
    /// The sequence the calls act on: the own view or a caller's boxed
    /// sequence (the raw shape upstream's `bound: *state.Seq` is).
    pub bound: *mut Seq<'a, 'd>,
    pub head: Option<Box<Head<'d>>>,
    pub load_seconds: f64,
    /// Options.segments.
    pub segments: usize,
    /// Their streams and scratch, made at load (or when set_segments
    /// asks for more).
    pub seg: Option<crate::cuda_segments::Segments<'a, 'd>>,
    /// Options.carveout; it outlives the engine.
    pub carve: Option<&'a Carveout<'d>>,
    /// Off unless TF_HEAT_HIGH and TF_HEAT_LOW are set.
    pub heat_gate: heat::Gate,
}

impl<'a, 'd> Engine<'a, 'd> {
    /// Loads the checkpoint into the Python engine's layouts and sizes
    /// the caches; `triton_dir` None: our own glue. Upstream: `init`
    /// (cuda_engine.zig:100). The engine comes back boxed — the head's
    /// raw self-pointer and the bound view need its address stable.
    pub fn init(
        ctx: &'d Context<'d>,
        model_dir: &std::path::Path,
        triton_dir: Option<&str>,
        opts: &Options<'a, 'd>,
    ) -> Result<Box<Engine<'a, 'd>>, Error> {
        let t0 = std::time::Instant::now();
        let c = Config::read(model_dir.to_str().unwrap_or_default())
            .map_err(|_| Error::BadConfig)?;
        if opts.segments < 1 || opts.segments > crate::cuda_segments::MAX {
            return Err(Error::BadSegments);
        }
        let slots = opts.context.unwrap_or(DEFAULT_CONTEXT) + MAX_ROWS;
        let max_len = slots.div_ceil(CHUNK_KEYS) * CHUNK_KEYS;
        let stream = Stream::init(ctx.d, true)?;
        let side = Side {
            s: Stream::init(ctx.d, true)?,
            fork: Event::init(ctx.d, false)?,
            join: Event::init(ctx.d, false)?,
        };
        let k = Kernels::load(ctx, triton_dir)?;
        let chunks = max_len / CHUNK_KEYS;
        let nch = match &k.triton {
            Some(t) => match t.smallest_const("_chunk", "NCH", chunks as i64) {
                Some(v) => v as usize,
                None => {
                    eprintln!(
                        "no captured attention kernel covers {chunks} chunks of 512 keys; capture one for this context"
                    );
                    return Err(Error::ContextUnsupported);
                }
            },
            None => chunks,
        };
        let mut headless = Engine {
            ctx,
            stream,
            side,
            k,
            w: weights::Weights {
                config: c,
                embed: weights::Embed {
                    w: 0,
                    s: 0,
                    b: 0,
                    n: 0,
                    k: 0,
                },
                blocks: Vec::new(),
                norm_f: 0,
                head: weights::QLinear {
                    w: 0,
                    s: 0,
                    b: 0,
                    n: 0,
                    k: 0,
                    npad: 0,
                },
                mtp: None,
                draft_head: None,
                draft_ids: 0,
                draft_count: 0,
                buffers: Vec::new(),
                named: Vec::new(),
            },
            b: Buffers::init(ctx.d, c, max_len, nch, opts.carveout)?,
            c,
            max_len,
            nch,
            pinned: HostBuffer::alloc(
                ctx.d,
                (PIN_SHARED + MAX_STREAMS * 64) * 4,
            )?,
            history: HostBuffer::alloc_mapped(ctx.d, (max_len + MAX_ROWS) * 4)?,
            history_dev: 0,
            serial: None,
            windows: [(); 2].map(|_| [const { None }; MAX_ROWS + 1]),
            done: {
                let mut v = Vec::with_capacity(LOOKAHEAD);
                for _ in 0..LOOKAHEAD {
                    v.push(Event::init(ctx.d, false)?);
                }
                v.try_into().map_err(|_| Error::Cuda(tensorfold_cuda::Error::Invalid)).unwrap()
            },
            copied: Event::init(ctx.d, false)?,
            sampled_ready: Event::init(ctx.d, false)?,
            pos: 0,
            parity: 0,
            prev_keep: 0,
            rows: 0,
            sampling: None,
            own: Seq {
                arena: None,
                carved: None,
                ptr: [0; 12],
                head: [0; 3],
                pos: 0,
                parity: 0,
                prev_keep: 0,
                rows: 0,
                head_pos: 0,
                sampling: None,
            },
            bound: std::ptr::null_mut(),
            head: None,
            load_seconds: 0.0,
            segments: opts.segments,
            seg: None,
            carve: opts.carveout,
            heat_gate: heat::Gate::default(),
        };
        headless.history_dev = headless.history.device()?;
        headless.b.init_shared(&headless.ops(), c)?;
        headless.own = Seq::view(&headless.b);
        headless.copied.record(&headless.stream)?;
        headless.set_sampling(opts.sampling)?;
        let mut e = Box::new(headless);
        // the bound view's pointer, after the move into the box
        e.bound = &mut e.own as *mut Seq<'a, 'd>;
        if opts.mtp {
            let head = Head::init(e.as_mut())?;
            e.head = Some(Box::new(head));
        }
        if opts.graphs {
            e.capture(opts.mtp)?;
        }
        e.heat_gate = heat::Gate::from_env().map_err(|_| {
            eprintln!(
                "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high"
            );
            Error::HeatBands
        })?;
        e.load_seconds = t0.elapsed().as_secs_f64();
        Ok(e)
    }

    /// The serial round (window plus feed) and, for drafted rounds, a
    /// verify window of each row count, each one graph. Upstream:
    /// `capture` (cuda_engine.zig:155).
    fn capture(&mut self, windows: bool) -> Result<(), Error> {
        self.reset()?;
        self.stream.synchronize()?;
        self.serial = Some(self.record(1, true)?);
        if windows {
            self.capture_windows()?;
        }
        Ok(())
    }

    /// The verify windows of the bound sequence's draw mode (greedy or
    /// sampled), each one graph. Upstream: `captureWindows`
    /// (cuda_engine.zig:163).
    pub fn capture_windows(&mut self) -> Result<(), Error> {
        let m = self.sampling.is_some() as usize;
        for r in 1..MAX_ROWS + 1 {
            self.windows[m][r] = Some(self.record(r, false)?);
        }
        Ok(())
    }

    fn record(&self, rows: usize, feed: bool) -> Result<Exec<'d>, Error> {
        graph::begin_capture(&self.stream, abi::CAPTURE_MODE_THREAD_LOCAL)?;
        if let Err(err) = self.record_body(rows, feed) {
            if let Ok(g) = graph::end_capture(&self.stream) {
                drop(g);
            }
            return Err(err);
        }
        let g = graph::end_capture(&self.stream)?;
        let exec = g.instantiate()?;
        exec.upload(&self.stream)?;
        self.stream.synchronize()?;
        Ok(exec)
    }

    fn record_body(&self, rows: usize, feed: bool) -> Result<(), Error> {
        self.forward().window(rows)?;
        if feed {
            return self
                .ops()
                .serial_feed(self.b.sampled, self.b.ids, self.b.meta, self.history_dev);
        }
        let mut sampled = [0u8; MAX_ROWS * 4];
        self.ops().download(&mut sampled[..rows * 4], self.b.sampled)?;
        // SAFETY: the pinned words' span, written as the engine's own
        // mut helpers write it (the pinned buffer is the engine's for
        // its whole life; the write here is the capture's own download
        // target).
        let words: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(
                self.pinned.slice().as_ptr() as *mut u32,
                self.pinned.slice().len() / 4,
            )
        };
        words[PIN_SAMPLED..PIN_SAMPLED + rows]
            .copy_from_slice(bytemuck_pod(&sampled[..rows * 4]));
        Ok(())
    }

    /// Upstream: `deinit` (cuda_engine.zig:191) — the port's Drop, the
    /// unwind in reverse.
    pub fn deinit(&mut self) {
        let _ = self.stream.synchronize();
        self.serial = None;
        for set in self.windows.iter_mut() {
            for g in set.iter_mut() {
                *g = None;
            }
        }
        self.seg = None;
        self.head = None;
        self.b.deinit();
        self.w.deinit();
        self.k.deinit();
    }

    /// The bound sequence's sampling: its rule on the device, which the
    /// target and the head draw by (null: greedy). Upstream:
    /// `setSampling` (cuda_engine.zig:212).
    pub fn set_sampling(&mut self, s: Option<Sampling>) -> Result<(), Error> {
        self.sampling = sampler::check(s);
        let x = match self.sampling {
            Some(x) => x,
            None => return Ok(()),
        };
        let r = sampler::rule(x);
        self.ops()
            .upload(self.b.rule, rule_bytes(&r))
    }

    /// A zeroed sequence for another stream (its own caches, state, head
    /// caches and settings). Upstream: `newSeq` (cuda_engine.zig:220) —
    /// the caller owns the box; free_seq takes it back.
    pub fn new_seq(&mut self) -> Result<*mut Seq<'a, 'd>, Error> {
        let head_sizes = self.with_head(|h, _| h.seq_sizes());
        let head: &[usize] = match &head_sizes {
            Some(s) => s,
            None => &[],
        };
        let s = Seq::init(self.ctx.d, self.c, self.max_len, head, &self.stream, self.carve)?;
        Ok(Box::into_raw(Box::new(s)))
    }

    /// The device bytes new_seq allocates. Upstream: `seqBytes`
    /// (cuda_engine.zig:230).
    pub fn seq_bytes(&mut self) -> usize {
        let head_sizes = self.with_head(|h, _| h.seq_sizes());
        match &head_sizes {
            Some(s) => Seq::bytes(self.c, self.max_len, s),
            None => Seq::bytes(self.c, self.max_len, &[]),
        }
    }

    /// Upstream: `freeSeq` (cuda_engine.zig:236).
    ///
    /// The raw sequence pointer is upstream's pointer graph: the
    /// caller owns the box (made by `new_seq`) and hands it back here,
    /// exactly once. The port keeps that invariant in this signature
    /// rather than an unsafe fn (the callers are the engine and the
    /// lane backend, both inside the documented scope).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // the raw shape is the documented invariant
    pub fn free_seq(&mut self, s: *mut Seq<'a, 'd>) {
        if std::ptr::eq(self.bound, s.cast()) {
            let own = &mut self.own as *mut Seq<'a, 'd>;
            self.bind(own);
        }
        // SAFETY: the caller's box, made by new_seq, dropped once.
        unsafe { drop(Box::from_raw(s)) };
    }

    /// Act on sequence `s` from here on: its buffers replace the bound
    /// one's, which keeps where it stands. Upstream: `bind`
    /// (cuda_engine.zig:243).
    ///
    /// The raw sequence pointer is upstream's `*state.Seq`: the own
    /// view or a caller's boxed sequence, alive for the call (the port
    /// keeps the invariant in this signature rather than an unsafe fn;
    /// the whole pointer graph is documented on the Engine).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // the raw shape is the documented invariant
    pub fn bind(&mut self, s: *mut Seq<'a, 'd>) {
        // SAFETY: bound points at the own view or a caller's boxed
        // sequence, both alive; the settings pass through plain fields.
        unsafe {
            let old = self.bound;
            // the settings pass in place; the seq's own buffers stay
            (*old).pos = self.pos;
            (*old).parity = self.parity;
            (*old).prev_keep = self.prev_keep;
            (*old).rows = self.rows;
            (*old).head_pos = self.head.as_ref().map_or(0, |h| h.pos);
            (*old).sampling = self.sampling;
            let b = &mut self.b;
            for (i, p) in (*s).ptr.iter().enumerate() {
                set_seq_field(b, i, *p);
            }
            self.pos = (*s).pos;
            self.parity = (*s).parity;
            self.prev_keep = (*s).prev_keep;
            self.rows = (*s).rows;
            self.sampling = (*s).sampling;
            let head_pos = (*s).head_pos;
            let head_ptrs = (*s).head;
            self.with_head(|h, _| h.bind_seq(head_ptrs, head_pos));
            self.bound = s;
        }
    }

    /// Captured graphs hold the own sequence's buffers; each set draws
    /// in one mode, picked by the bound sequence's. Upstream:
    /// `graphsBound` (cuda_engine.zig:257).
    pub fn graphs_bound(&self) -> bool {
        std::ptr::eq(self.bound, &self.own)
    }

    pub fn ops(&self) -> Ops<'_, 'd> {
        Ops {
            k: &self.k,
            s: &self.stream,
        }
    }

    pub fn forward(&self) -> Forward<'_, 'd> {
        let mut f = Forward::init(
            self.c,
            &self.w,
            &self.b,
            self.ops(),
            self.max_len,
            self.nch,
            self.sampling.is_some(),
        );
        if self.c.experts <= 128 && self.c.slots() <= 8 {
            f.side = Some(&self.side); // tf_plan_routed's bounds
        }
        f
    }

    /// Engine.reset: zeroed caches and state, position 0. Upstream:
    /// `reset` (cuda_engine.zig:273).
    pub fn reset(&mut self) -> Result<(), Error> {
        self.b.reset(&self.ops())?;
        self.pos = 0;
        self.parity = 0;
        self.prev_keep = 0;
        Ok(())
    }

    /// decode.prefill: a fresh state, the prompt in 2048-row chunks
    /// (absorbed by the head, in segments if set); the token. Upstream:
    /// `prefill` (cuda_engine.zig:281).
    pub fn prefill(&mut self, prompt: &[u32], head: bool) -> Result<u32, Error> {
        self.prefill_with(prompt, head, None)
    }

    /// prefill, stopping with Cancelled at the next chunk boundary once
    /// `cancel` says so. Upstream: `prefillWith` (cuda_engine.zig:286).
    pub fn prefill_with(
        &mut self,
        prompt: &[u32],
        head: bool,
        cancel: Option<&mut Cancel<'_>>,
    ) -> Result<u32, Error> {
        self.prefill_from(prompt, 0, head, cancel, None)
    }

    /// prefill from `from` (0 resets; else a restored state at a grid
    /// point); chunks stay on the zero-anchored grid. Upstream:
    /// `prefillFrom` (cuda_engine.zig:291).
    pub fn prefill_from(
        &mut self,
        prompt: &[u32],
        from: usize,
        head: bool,
        cancel: Option<&mut Cancel<'_>>,
        keep: Option<&mut Keep<'_>>,
    ) -> Result<u32, Error> {
        if prompt.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        if from != 0 && !crate::cuda_prompt_grid::resumable(from, prompt.len(), PREFILL_ROWS) {
            return Err(Error::BadReuse);
        }
        if prompt.len() + MAX_ROWS > self.max_len {
            return Err(Error::ContextFull);
        }
        if from == 0 {
            self.reset()?;
            if head {
                self.with_head(|h, e| h.reset(e)).ok_or(Error::NoMtpHead)??;
            }
        } else if self.pos != from {
            return Err(Error::BadReuse);
        }
        let waited = self.heat_gate.waited_s;
        if from == 0 && keep.is_none() && self.segments > 1 && prompt.len() > PREFILL_ROWS {
            self.segment_set()?;
            crate::cuda_segments::prefill(self, prompt, head, self.segments, cancel)?;
        } else {
            self.serial_chunks(prompt, from, head, cancel, keep)?;
        }
        if self.heat_gate.waited_s > waited {
            heat::note(self.heat_gate.waited_s - waited);
        }
        let mut tok = [0u8; 4];
        self.ops().download(&mut tok, self.b.p_sampled)?;
        self.stream.synchronize()?;
        Ok(u32::from_le_bytes(tok))
    }

    /// The segments' scratch and the engine's buffers, split into two
    /// disjoint borrows for the follow pass.
    pub(crate) fn split_seg_buffers(&mut self) -> (&mut crate::cuda_segments::Segments<'a, 'd>, &Buffers<'a, 'd>) {
        // SAFETY in the caller: both borrows live in one statement of
        // follow(); the fields they touch are disjoint (the segments'
        // scratch and the engine's buffers). The raw escape is that
        // split, made plain.
        let seg: *mut crate::cuda_segments::Segments<'a, 'd> =
            self.seg.as_mut().unwrap();
        let b: *const Buffers<'a, 'd> = &self.b;
        // SAFETY: the two borrows are disjoint by field.
        unsafe { (&mut *seg, &*b) }
    }

    /// The runner, its borrow escaped into the drive's raw use.
    pub(crate) fn seg_runner(&self) -> &Runner<'d> {
        // SAFETY: the runner is alive in e.seg through the drive.
        unsafe { &*self.seg.as_ref().unwrap().runner_ptr() }
    }

    /// The per-segment streams, as raw pointers (their borrows split).
    pub(crate) fn segment_streams(&mut self, parts: usize) -> [*const Stream<'d>; crate::cuda_segments::MAX] {
        let mut out = [std::ptr::null(); crate::cuda_segments::MAX];
        // SAFETY: the streams are alive in the runner through the
        // drive; the caller's parts stay within the runner's n.
        // SAFETY: the segments are alive in self.seg through the
        // drive; the streams come out of its runner.
        let runner: *const Runner<'d> =
            self.seg.as_ref().unwrap().runner_ptr();
        #[allow(clippy::needless_range_loop)] // upstream's own range loop
        for k in 0..parts {
            // SAFETY: the runner's side stream, alive through the
            // drive; the raw pointer erases the runner's borrow, the
            // stream lives as long as the runner does.
            let s: *const Stream<'_> =
                unsafe { (*runner).stream(&self.stream, k) };
            out[k] = s.cast::<Stream<'d>>();
        }
        out
    }

    /// Pinned words for `n` prompt ids on their way up (free once
    /// `copied` has completed). Upstream: `promptHost`
    /// (cuda_engine.zig:346).
    pub fn prompt_host(&mut self, n: usize) -> &mut [u32] {
        prompt_host(&mut self.pinned, n)
    }


    /// Heat bands, if set, before the chunk; a cancel ends the wait.
    /// Unset bands return without reading a zone. Upstream:
    /// `beforeChunk` (cuda_engine.zig:311).
    pub fn before_chunk(&mut self, cancel: Option<&mut Cancel<'_>>) -> Result<(), Error> {
        let mut stop = cancel.map(|c| heat::Stop {
            ptr: &mut c.check,
        });
        self.heat_gate
            .before_prompt_chunk(stop.as_mut().map(|s| s as &mut _))
            .map_err(Error::from)
    }

    /// The prompt's chunks one after another on the engine's stream,
    /// starting at `from`. Upstream: `serialChunks` (cuda_engine.zig:316).
    fn serial_chunks(
        &mut self,
        prompt: &[u32],
        from: usize,
        head: bool,
        mut cancel: Option<&mut Cancel<'_>>,
        mut keep: Option<&mut Keep<'_>>,
    ) -> Result<(), Error> {
        let mut s: usize = from;
        while s < prompt.len() {
            if Cancel::now(cancel.as_deref_mut()) {
                return Err(Error::Cancelled);
            }
            self.before_chunk(cancel.as_deref_mut())?;
            let end = crate::cuda_prompt_grid::end(s, prompt.len(), PREFILL_ROWS);
            let chunk = &prompt[s..end];
            self.copied.synchronize()?;
            let host = prompt_host(&mut self.pinned, chunk.len());
            host.copy_from_slice(chunk);
            let ops = Ops {
                k: &self.k,
                s: &self.stream,
            };
            ops.upload(self.b.p_ids, host.as_bytes())?;
            self.copied.record(&self.stream)?;
            self.forward().chunk(chunk.len(), self.pos)?;
            self.pos += chunk.len();
            let known = chunk.len().min(prompt.len() - 1 - s);
            if head && known > 0 {
                let hidden = self.b.p_hidden;
                let tokens: Vec<u32> = prompt[s + 1..s + 1 + known].to_vec();
                self.with_head(move |h, e| h.absorb(e, hidden, &tokens))
                    .ok_or(Error::NoMtpHead)??;
            }
            s = end;
            if let Some(k) = keep.as_deref_mut() {
                if s <= u32::MAX as usize && k.at.contains(&(s as u32)) {
                    (k.call)(s as u32);
                }
            }
        }
        Ok(())
    }

    /// Whole chunks a prompt call runs as staggered segments from here
    /// on; their streams and scratch are made now. Upstream:
    /// `setSegments` (cuda_engine.zig:339).
    pub fn set_segments(&mut self, n: usize) -> Result<(), Error> {
        if !(1..=crate::cuda_segments::MAX).contains(&n) {
            return Err(Error::BadSegments);
        }
        self.segments = n;
        if n > 1 {
            self.segment_set()?;
        }
        Ok(())
    }



    /// The segments' streams and scratch for calls of e.segments,
    /// remade when a call needs more of them. Upstream: `segmentSet`
    /// (cuda_engine.zig:351).
    fn segment_set(&mut self) -> Result<(), Error> {
        let n = self.segments;
        if let Some(s) = self.seg.as_mut() {
            if s.n >= n {
                return Ok(());
            }
            self.stream.synchronize()?;
            self.seg = None;
        }
        self.seg = Some(crate::cuda_segments::Segments::init(self, n)?);
        Ok(())
    }

    /// Engine.forward: a window of `rows` rows at pos whose first
    /// `ids.len` inputs come from the host (drafts are on the device).
    /// Upstream: `verify` (cuda_engine.zig:363).
    pub fn verify(
        &mut self,
        ids: &[u32],
        rows: usize,
    ) -> Result<(), Error> {
        if !(1..=MAX_ROWS).contains(&rows) || ids.len() > rows {
            return Err(Error::BadWindow);
        }
        if self.pos + rows > self.max_len {
            return Err(Error::ContextFull);
        }
        self.copied.synchronize()?;
        let meta = self.meta_row();
        {
            let host = pinned_u32_mut(&mut self.pinned);
            host[PIN_META..PIN_META + 4].copy_from_slice(&meta);
            host[PIN_IDS..PIN_IDS + ids.len()].copy_from_slice(ids);
            let ops = Ops {
                k: &self.k,
                s: &self.stream,
            };
            ops.upload(self.b.ids, host[PIN_IDS..PIN_IDS + ids.len()].as_bytes())?;
            ops.upload(self.b.meta, host[PIN_META..PIN_META + 4].as_bytes())?;
        }
        self.copied.record(&self.stream)?;
        let graph = self.windows[self.sampling.is_some() as usize][rows].take();
        match graph {
            Some(g) if self.graphs_bound() => {
                g.launch_on(&self.stream)?;
                self.windows[self.sampling.is_some() as usize][rows] = Some(g);
            }
            g => {
                if let Some(g) = g {
                    self.windows[self.sampling.is_some() as usize][rows] = Some(g);
                }
                self.forward().window(rows)?;
                let ops = Ops {
                    k: &self.k,
                    s: &self.stream,
                };
                let sampled = pinned_bytes_mut(&mut self.pinned, PIN_SAMPLED, rows);
                ops.download(sampled, self.b.sampled)?;
            }
        }
        self.sampled_ready.record(&self.stream)?;
        self.parity ^= 1;
        self.rows = rows;
        Ok(())
    }

    /// Several streams' windows in one forward (eager): each stream's
    /// rows read its own caches, state and rule. Upstream:
    /// `verifyShared` (cuda_engine.zig:386) — the per-segment Buffers
    /// views are upstream's struct copies; the port reads them
    /// uninitialized-but-owned and never drops them (the engine's `b`
    /// stays the owner).
    pub fn verify_shared(&mut self, parts: &[Shared<'a, 'd>]) -> Result<(), Error> {
        if parts.is_empty() || parts.len() > MAX_STREAMS {
            return Err(Error::BadWindow);
        }
        let mut total: usize = 0;
        for p in parts {
            if p.rows < 1 || p.ids.is_empty() || p.ids.len() > p.rows {
                return Err(Error::BadWindow);
            }
            total += p.rows;
        }
        if total > MAX_ROWS {
            return Err(Error::WindowTooWide);
        }
        self.copied.synchronize()?;
        // The MaybeUninit array never reads its (uninitialized)
        // contents; each filled slot stands on its own from its
        // ptr::read and is never dropped.
        let mut views: [std::mem::MaybeUninit<Buffers<'a, 'd>>; MAX_STREAMS] =
            // SAFETY: the array of MaybeUninit is itself just
            // uninitialized bytes; nothing reads a slot before it is
            // written.
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut rows_per: [usize; MAX_STREAMS] = [0; MAX_STREAMS];
        let mut sampled_per: [bool; MAX_STREAMS] = [false; MAX_STREAMS];
        for (k, p) in parts.iter().enumerate() {
            // SAFETY: bound points at the own view or a caller's boxed
            // sequence, both alive; the engine's bind writes the seq
            // fields through its own documented raw shape.
            self.bind(p.seq);
            if self.pos + p.rows > self.max_len {
                return Err(Error::ContextFull);
            }
            let meta = self.meta_row();
            {
                let host: &mut [u32] = pinned_u32_mut(&mut self.pinned);
                let slot = &mut host[PIN_SHARED + k * 64..PIN_SHARED + k * 64 + 64];
                slot[0..4].copy_from_slice(&meta);
                slot[4..4 + p.ids.len()].copy_from_slice(p.ids);
                let ops = Ops {
                    k: &self.k,
                    s: &self.stream,
                };
                ops.upload(self.b.ids, slot[4..4 + p.ids.len()].as_bytes())?;
                ops.upload(self.b.meta, slot[0..4].as_bytes())?;
            }
            // SAFETY: the read copies the bound sequence's fields into
            // a view the engine owns; the view is never dropped (the
            // engine's b keeps the arena's single ownership).
            views[k] = std::mem::MaybeUninit::new(unsafe { std::ptr::read(&self.b) });
            rows_per[k] = p.rows;
            sampled_per[k] = self.sampling.is_some();
        }
        // the views stand; the segments' references read them whole
        let mut row0: usize = 0;
        let mut segs_f: [Option<Seg<'_, 'd>>; MAX_STREAMS] = [None; MAX_STREAMS];
        for k in 0..parts.len() {
            // SAFETY: the view stands filled above.
            let b: &Buffers<'a, 'd> = unsafe { views[k].assume_init_ref() };
            segs_f[k] = Some(Seg {
                b: unsafe { std::ptr::read(&b) },
                row0,
                rows: rows_per[k],
                sampled: sampled_per[k],
            });
            row0 += rows_per[k];
        }
        self.copied.record(&self.stream)?;
        {
            let live: Vec<Seg<'_, 'd>> = segs_f[..parts.len()].iter().filter_map(|s| *s).collect();
            self.forward().window_segs(&live)?;
        }
        for (k, p) in parts.iter().enumerate() {
            let ops = Ops {
                k: &self.k,
                s: &self.stream,
            };
            let at = PIN_SHARED + k * 64 + 32;
            let sampled = pinned_bytes_mut(&mut self.pinned, at, p.rows);
            // SAFETY: the view still stands.
            let view = unsafe { views[k].assume_init_ref() };
            ops.download(sampled, view.sampled)?;
        }
        self.sampled_ready.record(&self.stream)?;
        for p in parts {
            self.bind(p.seq);
            self.parity ^= 1;
            self.rows = p.rows;
        }
        Ok(())
    }

    /// Part k's drawn ids from the last shared window (row r: the token
    /// at its position + r + 1). Upstream: `sharedTokens`
    /// (cuda_engine.zig:423).
    pub fn shared_tokens(&mut self, k: usize, rows: usize) -> Result<Vec<u32>, Error> {
        self.sampled_ready.synchronize()?;
        Ok(self.pinned_u32()[PIN_SHARED + k * 64 + 32..PIN_SHARED + k * 64 + 32 + rows].to_vec())
    }

    /// The window's meta row as the kernels read it: position, buffer
    /// parity, previous keep, a spare word. Upstream: `metaRow`
    /// (cuda_engine.zig:429).
    fn meta_row(&self) -> [u32; 4] {
        [self.pos as u32, self.parity as u32, self.prev_keep as u32, 0]
    }

    /// Engine.tokens: the last window's sampled ids (row r: the token at
    /// position pos + r + 1). Upstream: `tokens` (cuda_engine.zig:434).
    pub fn tokens(&mut self) -> Result<Vec<u32>, Error> {
        self.sampled_ready.synchronize()?;
        Ok(self.pinned_u32()[PIN_SAMPLED..PIN_SAMPLED + self.rows].to_vec())
    }

    /// Engine.commit: keep the first `keep` rows of the last window.
    /// Upstream: `commit` (cuda_engine.zig:440).
    pub fn commit(&mut self, keep: usize) -> Result<(), Error> {
        if keep < 1 || keep > self.rows {
            return Err(Error::BadKeep);
        }
        self.pos += keep;
        self.prev_keep = keep;
        Ok(())
    }

    /// The first serial window's inputs for the device-fed rounds: meta
    /// row and token. Upstream: `upload` (cuda_engine.zig:447).
    pub fn upload(&mut self, token: u32) -> Result<(), Error> {
        self.copied.synchronize()?;
        let meta = self.meta_row();
        {
            let host = pinned_u32_mut(&mut self.pinned);
            host[PIN_META..PIN_META + 4].copy_from_slice(&meta);
            host[PIN_IDS] = token;
            let ops = Ops {
                k: &self.k,
                s: &self.stream,
            };
            ops.upload(self.b.meta, host[PIN_META..PIN_META + 4].as_bytes())?;
            ops.upload(self.b.ids, host[PIN_IDS..PIN_IDS + 1].as_bytes())?;
        }
        self.copied.record(&self.stream)?;
        Ok(())
    }

    /// The token a serial round wrote for `position` (visible once that
    /// round's event has completed). Upstream: `written`
    /// (cuda_engine.zig:459).
    pub fn written(&self, position: usize) -> u32 {
        // SAFETY: the mapped history read once, as upstream's volatile.
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(self.history.slice().as_ptr().cast(), self.history.slice().len() / 4)
        };
        words[position]
    }

    /// One serial window of `token` at the current position; returns
    /// its sampled token and commits it. Upstream: `step`
    /// (cuda_engine.zig:464).
    pub fn step(&mut self, token: u32) -> Result<u32, Error> {
        self.verify(&[token], 1)?;
        let tok = self.tokens()?[0];
        self.commit(1)?;
        Ok(tok)
    }

    // -- the two named unsafe shapes ----------------------------------

    /// The one place the Engine↔Head pair is used as a pair.
    ///
    /// SAFETY: the head lives in a Box (a stable address) and no Head
    /// field aliases an Engine field; nothing replaces `head` while its
    /// borrow is live (only `deinit` takes it, after which no head
    /// method runs). The `Head::init` raw engine pointer and this
    /// helper are the crate's whole Head↔Engine bridge.
    pub(crate) fn with_head<R>(
        &mut self,
        f: impl for<'x> FnOnce(&mut Head<'d>, &mut Engine<'x, 'd>) -> R,
    ) -> Option<R> {
        let head = (&mut **self.head.as_mut()?) as *mut Head<'d>;
        // SAFETY: the invariants above.
        Some(f(unsafe { &mut *head }, self))
    }

    fn pinned_u32(&self) -> &[u32] {
        pinned_u32_of(&self.pinned)
    }


}

impl Drop for Engine<'_, '_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// The engine's seq fields by index, in seq_fields' order.
fn set_seq_field(b: &mut Buffers<'_, '_>, i: usize, p: u64) {
    match i {
        0 => b.k_cache = p,
        1 => b.v_cache = p,
        2 => b.ssm = p,
        3 => b.conv_base = p,
        4 => b.raw = p,
        5 => b.xc = p,
        6 => b.dt = p,
        7 => b.meta = p,
        8 => b.ids = p,
        9 => b.hidden = p,
        10 => b.sampled = p,
        _ => b.rule = p,
    }
}

/// The rule's own bytes, as the kernels read them.
fn rule_bytes(r: &sampler::Rule) -> &[u8] {
    // SAFETY: the extern-C rule's own bytes, read-only.
    unsafe { std::slice::from_raw_parts((r as *const sampler::Rule).cast(), std::mem::size_of::<sampler::Rule>()) }
}

/// Upstream: `seconds` (cuda_engine.zig:472) — the port's Instant.
pub fn seconds_since(t0: std::time::Instant) -> f64 {
    t0.elapsed().as_secs_f64()
}

/// The u32 slice's own bytes.
trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl AsBytes for [u32] {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: the slice's own bytes, read-only.
        unsafe { std::slice::from_raw_parts(self.as_ptr().cast(), self.len() * 4) }
    }
}

/// The pinned buffer's words (page-locked, 4-byte aligned; the view
/// never outlives the engine).
fn pinned_u32_of<'p>(pinned: &'p HostBuffer<'_>) -> &'p [u32] {
    // SAFETY: the aligned buffer's own words, read-only.
    unsafe { std::slice::from_raw_parts(pinned.slice().as_ptr().cast(), pinned.slice().len() / 4) }
}

/// The same, mutably, over the field the call site splits.
fn pinned_u32_mut<'p>(pinned: &'p mut HostBuffer<'_>) -> &'p mut [u32] {
    // SAFETY: the aligned buffer's own words.
    unsafe {
        std::slice::from_raw_parts_mut(pinned.slice_mut().as_mut_ptr().cast(), pinned.slice().len() / 4)
    }
}

/// Pinned words for `n` prompt ids on their way up (free once `copied`
/// has completed). Upstream: `promptHost` (cuda_engine.zig:346).
fn prompt_host<'p>(pinned: &'p mut HostBuffer<'_>, n: usize) -> &'p mut [u32] {
    &mut pinned_u32_mut(pinned)[PIN_PROMPT..PIN_PROMPT + n]
}

/// The pinned words' bytes at `at`, `n` words.
fn pinned_bytes_mut<'p>(pinned: &'p mut HostBuffer<'_>, at: usize, n: usize) -> &'p mut [u8] {
    // SAFETY: the words' own bytes, the span's.
    unsafe {
        std::slice::from_raw_parts_mut(
            (pinned.slice_mut().as_mut_ptr() as *mut u32).add(at).cast(),
            n * 4,
        )
    }
}

/// Four u32 words from sixteen bytes.
fn bytemuck_pod(bytes: &[u8]) -> &[u32] {
    // SAFETY: the local's own bytes, at the u32 alignment by construction.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 4) }
}
