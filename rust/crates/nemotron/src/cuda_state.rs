//! Device buffers of one Nemotron sequence: its caches and recurrent
//! state, and the scratch a window or chunk uses.
//!
//! Port of `zig/src/families/nemotron/cuda_state.zig` (311 lines) —
//! Zig's `inline for` over the field names becomes the explicit field
//! lists in the same order, and the Carveout is the cuda crate's
//! cross-platform name (the Linux form on Linux, the never-constructed
//! host form elsewhere).

use tensorfold_cuda::{Carveout, DeviceBuffer};

use crate::config::Config;
use crate::cuda_kernels::{self, Ops, Plan};
use crate::cuda_sampler;
use crate::error::Error;

pub use crate::cuda_sampler::Sampling;

pub const MAX_ROWS: usize = 16; // a verify window's rows (the row tile of the row-parallel kernels)
pub const PREFILL_ROWS: usize = 2048; // rows of a prompt chunk
pub const CHUNK_KEYS: usize = 512; // keys an attention chunk holds, at fixed absolute positions
const SHARED_WORDS: usize = 2 * MAX_ROWS + 8 * MAX_ROWS; // the shared expert's members, then (items, counts) a row count

/// Sub-allocations of one device allocation, 256-byte aligned like the
/// driver's own. Upstream: `Arena` (cuda_state.zig:12).
pub struct Arena<'d> {
    pub buf: DeviceBuffer<'d>,
    pub used: usize,
}

impl Arena<'_> {
    pub fn take(&mut self, bytes: usize) -> u64 {
        let at = self.used.next_multiple_of(256);
        self.used = at + bytes;
        debug_assert!(self.used <= self.buf.len);
        self.buf.ptr + at as u64
    }
}

/// A sequence's KV planes in the display carveout, given back when the
/// sequence goes. Upstream: `Carved` (cuda_state.zig:19).
pub struct Carved<'a, 'd> {
    /// Shared: the engine's buffers and its sequences take and give
    /// through one carveout (upstream's *Carveout).
    pub carve: &'a Carveout<'d>,
    pub kv: [DeviceBuffer<'d>; 2],
}

impl Carved<'_, '_> {
    /// Both KV planes of `n` bytes each from `carve`, or None when it is
    /// off or cannot hold both. Upstream: `take` (cuda_state.zig:24).
    pub fn take<'a, 'd>(
        carve: Option<&'a Carveout<'d>>,
        n: usize,
    ) -> Result<Option<Carved<'a, 'd>>, Error> {
        let cv = match carve {
            Some(cv) => cv,
            None => return Ok(None),
        };
        let k = match cv.take(n)? {
            Some(k) => k,
            None => return Ok(None),
        };
        match cv.take(n)? {
            Some(v) => Ok(Some(Carved { carve: cv, kv: [k, v] })),
            None => {
                cv.give(&k);
                Ok(None)
            }
        }
    }

    /// Upstream: `give` (cuda_state.zig:31).
    pub fn give(x: &Carved<'_, '_>) {
        x.carve.give(&x.kv[1]);
        x.carve.give(&x.kv[0]);
    }

    /// Arena bytes for `sizes` with the carved planes left out.
    /// Upstream: `arenaBytes` (cuda_state.zig:36).
    pub fn arena_bytes(x: Option<&Carved<'_, '_>>, sizes: &[usize]) -> usize {
        let mut total: usize = 0;
        for (i, n) in sizes.iter().enumerate() {
            total += if x.is_some() && i < 2 {
                0
            } else {
                n + 256
            };
        }
        total
    }
}

/// A sequence's own Buffers fields: cache and state planes (the first
/// seven), window io and its sampled rule. Upstream: `seq_fields`
/// (cuda_state.zig:43).
pub const SEQ_FIELDS: [&str; 12] = [
    "k_cache",
    "v_cache",
    "ssm",
    "conv_base",
    "raw",
    "xc",
    "dt",
    "meta",
    "ids",
    "hidden",
    "sampled",
    "rule",
];

/// Upstream: `seqSizes` (cuda_state.zig:45).
fn seq_sizes(c: Config, max_len: usize) -> [usize; SEQ_FIELDS.len()] {
    let w = MAX_ROWS;
    let nm = c.count(crate::config::Kind::Mamba);
    let na = c.count(crate::config::Kind::Attention);
    let kv = na * max_len * c.kv_heads * c.head_dim * 2;
    let cd = c.conv_dim();
    [
        kv,
        kv,
        nm * c.mamba_heads * c.mamba_head_dim * c.state * 4,
        nm * 3 * cd * 2,
        nm * 2 * w * cd * 2,
        nm * 2 * w * cd * 2,
        nm * 2 * w * c.mamba_heads * 4,
        16,
        w * 4,
        w * c.hidden * 2,
        w * 4,
        std::mem::size_of::<cuda_sampler::Rule>(),
    ]
}

const SCRATCH_COUNT: usize = 35;

/// Scratch windows and chunks share on one stream: window logits,
/// chunk io, activations, the expert plan. Upstream: `scratchSizes`
/// (cuda_state.zig:56).
fn scratch_sizes(c: Config, nch: usize) -> [usize; SCRATCH_COUNT] {
    let r = PREFILL_ROWS;
    let w = MAX_ROWS;
    let d = c.hidden;
    let ns = c.slots();
    let qd = c.heads * c.head_dim;
    let cd = c.conv_dim();
    let xd = c.inner();
    let pairs = r * ns;
    let items = cuda_kernels::max_items(pairs, c.experts + 2, 16);
    [
        w * c.vocab * 2,
        16,
        r * 4,
        r * d * 2,
        c.vocab * 2,
        16,
        r * d * 2,
        r * d * 2,
        r * d * 2,
        r * d * 2,
        r * d / 64 * 4,
        r * d * 2,
        r * c.proj_dim() * 2,
        r * cd * 2,
        r * xd * 2,
        r * xd * 2,
        r * xd / 64 * 4,
        r * c.qkv_dim() * 2,
        r * qd * 2,
        r * qd * 2,
        r * qd / 64 * 4,
        w * nch * qd * 4,
        w * nch * c.heads * 4,
        w * nch * c.heads * 4,
        6 * r * c.experts * 4,
        r * ns * 4,
        r * ns * 4,
        pairs * 4,
        items * 12,
        8,
        pairs * 4,
        pairs.div_ceil(1024) * (c.experts + 2) * 4,
        pairs * c.expert_width * 2,
        pairs * d * 4,
        SHARED_WORDS * 4,
    ]
}

/// One sequence: its own buffers (seq_fields, then its MTP head's) and
/// where it stands while another is bound. Upstream: `Seq`
/// (cuda_state.zig:73).
pub struct Seq<'a, 'd> {
    /// None: the engine's own buffers.
    pub arena: Option<Arena<'d>>,
    pub carved: Option<Carved<'a, 'd>>,
    pub ptr: [u64; SEQ_FIELDS.len()],
    /// The MTP head's seq_fields (cuda_mtp.zig).
    pub head: [u64; 3],
    pub pos: usize,
    pub parity: usize,
    pub prev_keep: usize,
    pub rows: usize,
    pub head_pos: usize,
    pub sampling: Option<Sampling>,
}

impl Seq<'_, '_> {
    /// The device bytes one sequence takes: its buffers for `max_len`
    /// cache rows and a head's `head` buffers. Upstream: `bytes`
    /// (cuda_state.zig:83).
    pub fn bytes(c: Config, max_len: usize, head: &[usize]) -> usize {
        let mut total: usize = 0;
        for n in seq_sizes(c, max_len) {
            total += n + 256;
        }
        for n in head {
            total += n + 256;
        }
        total
    }

    /// Cache and head buffers are zeroed on their owning stream; the KV
    /// pair goes to `carve` while it holds both. Upstream: `init`
    /// (cuda_state.zig:89).
    pub fn init<'a, 'd>(
        d: &'d tensorfold_cuda::Driver,
        c: Config,
        max_len: usize,
        head: &[usize],
        stream: &tensorfold_cuda::Stream<'d>,
        carve: Option<&'a Carveout<'d>>,
    ) -> Result<Seq<'a, 'd>, Error> {
        let sizes = seq_sizes(c, max_len);
        let mut carved = Carved::take(carve, sizes[0])?;
        let mut total = Carved::arena_bytes(carved.as_ref(), &sizes);
        for n in head {
            total += n + 256;
        }
        let buf = match DeviceBuffer::alloc(d, total) {
            Ok(b) => b,
            Err(e) => {
                if let Some(x) = carved.as_mut() {
                    Carved::give(x);
                }
                return Err(e.into());
            }
        };
        buf.fill8(0, Some(stream.handle))?;
        let mut a = Arena { buf, used: 0 };
        let mut s = Seq {
            arena: None,
            carved,
            ptr: [0; SEQ_FIELDS.len()],
            head: [0; 3],
            pos: 0,
            parity: 0,
            prev_keep: 0,
            rows: 0,
            head_pos: 0,
            sampling: None,
        };
        for (i, p) in s.ptr.iter_mut().enumerate() {
            *p = match &s.carved {
                Some(x) if i < 2 => x.kv[i].ptr,
                _ => a.take(sizes[i]),
            };
        }
        for (i, n) in head.iter().enumerate().take(3) {
            s.head[i] = a.take(*n);
        }
        s.arena = Some(a);
        Ok(s)
    }

    /// The engine's own buffers as a sequence. Upstream: `view`
    /// (cuda_state.zig:120) — the Buffers' own seq fields.
    pub fn view<'a, 'd>(b: &Buffers<'a, 'd>) -> Seq<'a, 'd> {
        Seq {
            arena: None,
            carved: None,
            ptr: [
                b.k_cache,
                b.v_cache,
                b.ssm,
                b.conv_base,
                b.raw,
                b.xc,
                b.dt,
                b.meta,
                b.ids,
                b.hidden,
                b.sampled,
                b.rule,
            ],
            head: b.head,
            pos: 0,
            parity: 0,
            prev_keep: 0,
            rows: 0,
            head_pos: 0,
            sampling: None,
        }
    }

    /// Upstream: `deinit` (cuda_state.zig:126) — the port's Drop frees
    /// the arena and gives the carved planes back.
    pub fn deinit(&mut self) {
        self.arena = None;
        if let Some(x) = self.carved.take() {
            Carved::give(&x);
        }
    }
}

impl Drop for Seq<'_, '_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// Upstream: `Buffers` (cuda_state.zig:131).
pub struct Buffers<'a, 'd> {
    pub arena: Arena<'d>,
    /// The own sequence's KV planes, when the display carveout holds
    /// them.
    pub carved: Option<Carved<'a, 'd>>,
    // sequence state (Engine.STATE in the Python engine)
    pub k_cache: u64,
    pub v_cache: u64,
    pub ssm: u64,
    pub conv_base: u64,
    pub raw: u64,
    pub xc: u64,
    pub dt: u64,
    pub state_bytes: [usize; 7],
    // a window's inputs and outputs
    pub meta: u64,
    pub ids: u64,
    pub hidden: u64,
    pub logits: u64,
    pub sampled: u64,
    // a prompt chunk's own inputs and outputs
    pub p_meta: u64,
    pub p_ids: u64,
    pub p_hidden: u64,
    pub p_logits: u64,
    pub p_sampled: u64,
    // scratch shared by windows and chunks (they never overlap on the
    // stream)
    pub emb: u64,
    pub h: [u64; 2],
    pub y: u64,
    pub xs: u64,
    pub delta: u64,
    pub proj: u64,
    pub p_xc: u64,
    pub sy: u64,
    pub g: u64,
    pub gxs: u64,
    pub qkv: u64,
    pub q: u64,
    pub att: u64,
    pub axs: u64,
    pub po: u64,
    pub pm: u64,
    pub pl: u64,
    pub part: u64,
    pub pick: u64,
    pub wts: u64,
    pub plan: Plan,
    pub act: u64,
    pub ymoe: u64,
    /// The sequence's sampled rule, as sample.cu reads it.
    pub rule: u64,
    /// The shared expert's fixed plans: 32 members, then (items,
    /// counts) for 1..16 rows.
    pub shared: u64,
    /// The MTP head's buffers (cuda_mtp.zig), as a sequence holds its
    /// own.
    pub head: [u64; 3],
}

impl<'a, 'd> Buffers<'a, 'd> {
    /// Sizes every buffer for `max_len` cache rows and `nch` attention
    /// chunk partials a row; the own sequence's first. Upstream:
    /// `init` (cuda_state.zig:177).
    pub fn init(
        d: &'d tensorfold_cuda::Driver,
        c: Config,
        max_len: usize,
        nch: usize,
        carve: Option<&'a Carveout<'d>>,
    ) -> Result<Buffers<'a, 'd>, Error> {
        let own = seq_sizes(c, max_len);
        let scratch = scratch_sizes(c, nch);
        let mut carved = Carved::take(carve, own[0])?;
        let mut total = Carved::arena_bytes(carved.as_ref(), &own);
        for n in scratch {
            total += n + 256;
        }
        let buf = match DeviceBuffer::alloc(d, total) {
            Ok(b) => b,
            Err(e) => {
                if let Some(x) = carved.as_mut() {
                    Carved::give(x);
                }
                return Err(e.into());
            }
        };
        let a = Arena { buf, used: 0 };
        let mut b = Buffers {
            arena: a,
            carved,
            k_cache: 0,
            v_cache: 0,
            ssm: 0,
            conv_base: 0,
            raw: 0,
            xc: 0,
            dt: 0,
            state_bytes: [0; 7],
            meta: 0,
            ids: 0,
            hidden: 0,
            logits: 0,
            sampled: 0,
            p_meta: 0,
            p_ids: 0,
            p_hidden: 0,
            p_logits: 0,
            p_sampled: 0,
            emb: 0,
            h: [0; 2],
            y: 0,
            xs: 0,
            delta: 0,
            proj: 0,
            p_xc: 0,
            sy: 0,
            g: 0,
            gxs: 0,
            qkv: 0,
            q: 0,
            att: 0,
            axs: 0,
            po: 0,
            pm: 0,
            pl: 0,
            part: 0,
            pick: 0,
            wts: 0,
            plan: Plan {
                members: 0,
                items: 0,
                counts: 0,
                rank: 0,
                hist: 0,
            },
            act: 0,
            ymoe: 0,
            rule: 0,
            shared: 0,
            head: [0; 3],
        };
        b.state_bytes = [
            own[0], own[1], own[2], own[3], own[4], own[5], own[6],
        ];
        let seq = [
            &mut b.k_cache,
            &mut b.v_cache,
            &mut b.ssm,
            &mut b.conv_base,
            &mut b.raw,
            &mut b.xc,
            &mut b.dt,
            &mut b.meta,
            &mut b.ids,
            &mut b.hidden,
            &mut b.sampled,
            &mut b.rule,
        ];
        for (i, p) in seq.into_iter().enumerate() {
            *p = match &b.carved {
                Some(x) if i < 2 => x.kv[i].ptr,
                _ => b.arena.take(own[i]),
            };
        }
        b.take_scratch(&scratch);
        Ok(b)
    }

    /// Its own scratch for another prompt segment, sharing `b`'s caches
    /// and state (follow); deinit frees the scratch. Upstream:
    /// `sibling` (cuda_state.zig:196).
    pub fn sibling(
        b: &Buffers<'a, 'd>,
        d: &'d tensorfold_cuda::Driver,
        c: Config,
        nch: usize,
    ) -> Result<Buffers<'a, 'd>, Error> {
        let scratch = scratch_sizes(c, nch);
        let mut total: usize = 0;
        for n in scratch {
            total += n + 256;
        }
        let mut s = Buffers {
            arena: Arena {
                buf: DeviceBuffer::alloc(d, total)?,
                used: 0,
            },
            // the owner gives its planes back
            carved: None,
            k_cache: b.k_cache,
            v_cache: b.v_cache,
            ssm: b.ssm,
            conv_base: b.conv_base,
            raw: b.raw,
            xc: b.xc,
            dt: b.dt,
            state_bytes: b.state_bytes,
            meta: b.meta,
            ids: b.ids,
            hidden: b.hidden,
            logits: b.logits,
            sampled: b.sampled,
            p_meta: b.p_meta,
            p_ids: b.p_ids,
            p_hidden: b.p_hidden,
            p_logits: b.p_logits,
            p_sampled: b.p_sampled,
            emb: b.emb,
            h: b.h,
            y: b.y,
            xs: b.xs,
            delta: b.delta,
            proj: b.proj,
            p_xc: b.p_xc,
            sy: b.sy,
            g: b.g,
            gxs: b.gxs,
            qkv: b.qkv,
            q: b.q,
            att: b.att,
            axs: b.axs,
            po: b.po,
            pm: b.pm,
            pl: b.pl,
            part: b.part,
            pick: b.pick,
            wts: b.wts,
            plan: b.plan,
            act: b.act,
            ymoe: b.ymoe,
            rule: b.rule,
            shared: b.shared,
            head: b.head,
        };
        s.take_scratch(&scratch);
        Ok(s)
    }

    /// A sibling takes `b`'s sequence buffers (they move when the
    /// engine binds another sequence). Upstream: `follow`
    /// (cuda_state.zig:207).
    pub fn follow(&mut self, b: &Buffers<'a, 'd>) {
        self.k_cache = b.k_cache;
        self.v_cache = b.v_cache;
        self.ssm = b.ssm;
        self.conv_base = b.conv_base;
        self.raw = b.raw;
        self.xc = b.xc;
        self.dt = b.dt;
        self.state_bytes = b.state_bytes;
    }

    /// Every scratch field, in scratchSizes' order. Upstream:
    /// `scratchPtrs` (cuda_state.zig:214) — the port assigns them in
    /// the same order.
    fn take_scratch(&mut self, s: &[usize; SCRATCH_COUNT]) {
        let mut it = s.iter();
        let mut next = || *it.next().unwrap();
        self.logits = self.arena.take(next());
        self.p_meta = self.arena.take(next());
        self.p_ids = self.arena.take(next());
        self.p_hidden = self.arena.take(next());
        self.p_logits = self.arena.take(next());
        self.p_sampled = self.arena.take(next());
        self.emb = self.arena.take(next());
        self.h[0] = self.arena.take(next());
        self.h[1] = self.arena.take(next());
        self.y = self.arena.take(next());
        self.xs = self.arena.take(next());
        self.delta = self.arena.take(next());
        self.proj = self.arena.take(next());
        self.p_xc = self.arena.take(next());
        self.sy = self.arena.take(next());
        self.g = self.arena.take(next());
        self.gxs = self.arena.take(next());
        self.qkv = self.arena.take(next());
        self.q = self.arena.take(next());
        self.att = self.arena.take(next());
        self.axs = self.arena.take(next());
        self.po = self.arena.take(next());
        self.pm = self.arena.take(next());
        self.pl = self.arena.take(next());
        self.part = self.arena.take(next());
        self.pick = self.arena.take(next());
        self.wts = self.arena.take(next());
        self.plan.members = self.arena.take(next());
        self.plan.items = self.arena.take(next());
        self.plan.counts = self.arena.take(next());
        self.plan.rank = self.arena.take(next());
        self.plan.hist = self.arena.take(next());
        self.act = self.arena.take(next());
        self.ymoe = self.arena.take(next());
        self.shared = self.arena.take(next());
    }

    /// The shared expert's plan at `rows` rows: halves `experts` and
    /// `experts + 1` over slots top_k and top_k + 1. Upstream:
    /// `sharedPlan` (cuda_state.zig:225).
    pub fn shared_plan(&self, rows: usize) -> Plan {
        let items = self.shared + (2 * MAX_ROWS + 8 * (rows - 1)) as u64 * 4;
        Plan {
            members: self.shared,
            items,
            counts: items + 6 * 4,
            rank: 0,
            hist: 0,
        }
    }

    /// Writes every row count's shared plan (pairs in row order, one
    /// item a half since rows <= 16). Upstream: `initShared`
    /// (cuda_state.zig:230).
    pub fn init_shared(&self, ops: &Ops<'_, '_>, c: Config) -> Result<(), Error> {
        let mut w = [0u32; SHARED_WORDS];
        for r in 0..MAX_ROWS {
            w[r] = (r * c.slots() + c.top_k) as u32;
            w[MAX_ROWS + r] = (r * c.slots() + c.top_k + 1) as u32;
        }
        for rows in 1..MAX_ROWS + 1 {
            let at = 2 * MAX_ROWS + 8 * (rows - 1);
            let n = rows as u32;
            w[at..at + 8].copy_from_slice(&[
                c.experts as u32,
                0,
                n,
                (c.experts + 1) as u32,
                MAX_ROWS as u32,
                n,
                2,
                2,
            ]);
        }
        ops.upload(self.shared, to_bytes(&w))?;
        ops.s.synchronize()?;
        Ok(())
    }

    /// Upstream: `deinit` (cuda_state.zig:244) — the port's Drop frees
    /// the arena and gives the carved planes back.
    pub fn deinit(&mut self) {
        if let Some(x) = self.carved.take() {
            Carved::give(&x);
        }
    }

    /// The cache and state planes' addresses, in seq_fields' order (the
    /// KV pair may live in the carveout). Upstream: `planes`
    /// (cuda_state.zig:248).
    pub fn planes(&self) -> [u64; 7] {
        [
            self.k_cache,
            self.v_cache,
            self.ssm,
            self.conv_base,
            self.raw,
            self.xc,
            self.dt,
        ]
    }

    /// Bytes of every cache and state plane together. Upstream:
    /// `stateBytes` (cuda_state.zig:252).
    pub fn state_bytes(&self) -> usize {
        self.state_bytes.iter().sum()
    }

    /// Engine.snapshot: a device copy of every cache and state plane,
    /// packed. Upstream: `snapshot` (cuda_state.zig:256).
    pub fn snapshot(&self, ops: &Ops<'_, 'd>) -> Result<DeviceBuffer<'d>, Error> {
        let copy = DeviceBuffer::alloc(ops.k.d, self.state_bytes())?;
        let mut at: usize = 0;
        for (p, n) in self.planes().iter().zip(self.state_bytes.iter()) {
            ops.copy(copy.ptr + at as u64, *p, *n)?;
            at += n;
        }
        Ok(copy)
    }

    /// Upstream: `restore` (cuda_state.zig:266).
    pub fn restore(&self, ops: &Ops<'_, '_>, saved: &DeviceBuffer<'_>) -> Result<(), Error> {
        let mut at: usize = 0;
        for (p, n) in self.planes().iter().zip(self.state_bytes.iter()) {
            ops.copy(*p, saved.ptr + at as u64, *n)?;
            at += n;
        }
        Ok(())
    }

    /// Engine.reset: every cache and state buffer zeroed, as a fresh
    /// request starts. Upstream: `reset` (cuda_state.zig:273).
    pub fn reset(&self, ops: &Ops<'_, '_>) -> Result<(), Error> {
        for (p, n) in self.planes().iter().zip(self.state_bytes.iter()) {
            ops.fill32(*p, 0, n / 4)?;
        }
        Ok(())
    }

    /// Bytes of the own sequence's planes in the display carveout.
    /// Upstream: `carvedBytes` (cuda_state.zig:280).
    pub fn carved_bytes(&self) -> usize {
        match &self.carved {
            Some(x) => x.kv[0].len + x.kv[1].len,
            None => 0,
        }
    }
}

impl Drop for Buffers<'_, '_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// The plan words as bytes (upstream's sliceAsBytes).
fn to_bytes(w: &[u32]) -> &[u8] {
    // SAFETY: the u32 words' own bytes, read-only.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() * 4) }
}
