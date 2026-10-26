//! Nemotron-H's forwards on CUDA in the Python engine's kernel order:
//! a verify window, and a prompt chunk.
//!
//! Port of `zig/src/families/nemotron/cuda_forward.zig` (342 lines).

use tensorfold_cuda::{Event, Stream};

use crate::config::Config;
use crate::cuda_dump::Dump;
use crate::cuda_glue as glue;
use crate::cuda_glue::Glue;
use crate::cuda_kernels::{self, Ops};
use crate::cuda_state::{Buffers, MAX_ROWS};
use crate::cuda_weights::{Moe, Weights};
use crate::error::Error;

/// Upstream: `Delta` (cuda_forward.zig:12).
#[derive(Clone, Copy, PartialEq)]
pub enum Delta {
    None,
    Dense,
    Moe,
}

/// A second stream the shared expert runs on while the routed experts
/// are chosen; fork and join are events. Upstream: `Side`
/// (cuda_forward.zig:15) — the port's Stream owns its CUDA object.
pub struct Side<'d> {
    pub s: Stream<'d>,
    pub fork: Event<'d>,
    pub join: Event<'d>,
}

/// A verify window supplies its stream buffers and starting row.
/// Upstream: `Seg` (cuda_forward.zig:18).
#[derive(Clone, Copy)]
pub struct Seg<'a, 'd> {
    pub b: &'a Buffers<'a, 'd>,
    pub row0: usize,
    pub rows: usize,
    pub sampled: bool,
}

/// A prompt chunk carries rows, position, residual delta and mixer
/// positions between blocks. Upstream: `Walk` (cuda_forward.zig:21).
#[derive(Clone)]
pub struct Walk {
    pub rows: usize,
    pub pos: usize,
    pub x: u64,
    pub delta: Delta,
    pub mj: usize,
    pub aj: usize,
}

impl Walk {
    pub fn new(rows: usize, pos: usize) -> Walk {
        Walk {
            rows,
            pos,
            x: 0,
            delta: Delta::None,
            mj: 0,
            aj: 0,
        }
    }
}

/// A window's kernel classes as a profile names them: each mark closes
/// the span since the one before. Upstream: `Class`
/// (cuda_forward.zig:24).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Class {
    Embed,
    Norm,
    MambaIn,
    ConvScan,
    Gnorm,
    MambaOut,
    Qkv,
    Attention,
    AttnO,
    Route,
    Plan,
    ExpertsUp,
    ExpertsDown,
    SharedWait,
    Head,
    Sample,
}

/// Timing events recorded between kernel classes (the window profile's;
/// None in every other forward). Upstream: `Marks` (cuda_forward.zig:27).
pub struct Marks<'m, 'd> {
    pub ev: &'m mut [Event<'d>],
    pub class: &'m mut [Class],
    pub n: usize,
}

impl Marks<'_, '_> {
    pub fn add(&mut self, s: &Stream<'_>, c: Class) -> Result<(), Error> {
        if self.n == self.ev.len() {
            return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // TooManyMarks
        }
        self.ev[self.n].record(s)?;
        self.class[self.n] = c;
        self.n += 1;
        Ok(())
    }
}

/// Upstream: `Forward` (cuda_forward.zig:40) — the port's Forward
/// carries its Ops by value (Zig copies the handle struct; the Ops is
/// two references) so no caller's temporary keeps it alive.
pub struct Forward<'a, 'd> {
    pub c: Config,
    pub w: &'a Weights<'d>,
    pub b: &'a Buffers<'a, 'd>,
    pub ops: Ops<'a, 'd>,
    pub glue: Glue<'a, 'd>,
    pub max_len: usize,
    pub nch: usize,
    /// The target draws by the bound sequence's rule (sample.cu);
    /// false: torch.argmax.
    pub sampled: bool,
    pub dump: Option<&'a Dump<'a, 'd>>,
    /// Decode MoE layers run the shared expert here (None: one stream).
    pub side: Option<&'a Side<'d>>,
    /// tf-cuda-test window-profile's events; None costs one check a
    /// class, no GPU work. Held as upstream holds it: a mutable
    /// pointer behind a const forward.
    pub marks: Option<&'a std::cell::RefCell<Marks<'a, 'd>>>,
}

impl<'a, 'd> Forward<'a, 'd> {
    /// Upstream: `init` (cuda_forward.zig:53).
    pub fn init(
        c: Config,
        w: &'a Weights<'d>,
        b: &'a Buffers<'a, 'd>,
        ops: Ops<'a, 'd>,
        max_len: usize,
        nch: usize,
        sampled: bool,
    ) -> Forward<'a, 'd> {
        let glue = Glue {
            set: ops.k.triton.as_ref(),
            f: &ops.k.glue,
            s: ops.s,
        };
        Forward {
            c,
            w,
            b,
            ops,
            glue,
            max_len,
            nch,
            sampled,
            dump: None,
            side: None,
            marks: None,
        }
    }

    /// Rows of logits draw their next tokens into `out`, row r at
    /// position META[0] + r + 1, by `rule` when sampled. Upstream:
    /// `sample` (cuda_forward.zig:58).
    fn sample(
        &self,
        logits: u64,
        rows: usize,
        meta: u64,
        out: u64,
        sampled: bool,
        rule: u64,
    ) -> Result<(), Error> {
        let v = self.c.vocab;
        if !sampled {
            return self.ops.torch().argmax(logits, v, v, out, rows);
        }
        self.ops.draw(logits, v, rule, meta, 0, out, rows, None, None)
    }

    fn mark(&self, c: Class) -> Result<(), Error> {
        if let Some(m) = &self.marks {
            m.borrow_mut().add(self.ops.s, c)?;
        }
        Ok(())
    }

    fn mshape(&self, rmax: usize) -> glue::Shape {
        let c = self.c;
        glue::Shape {
            proj: c.proj_dim(),
            xd: c.inner(),
            cd: c.conv_dim(),
            heads: c.mamba_heads,
            dh: c.mamba_head_dim,
            groups: c.groups,
            state: c.state,
            rmax,
        }
    }

    pub fn ashape(&self) -> glue::Attn {
        glue::Attn {
            nqkv: self.c.qkv_dim(),
            heads: self.c.heads,
            kv_heads: self.c.kv_heads,
            dim: self.c.head_dim,
            nch: self.nch,
        }
    }

    fn cache(&self, base: u64, layer: usize) -> u64 {
        base + layer as u64 * self.max_len as u64 * self.c.kv_heads as u64
            * self.c.head_dim as u64 * 2
    }

    /// Engine.norm: h = x + delta (dense, or the experts' slot-order
    /// sum), then the block's RMSNorm into y and xs. Upstream: `norm`
    /// (cuda_forward.zig:82).
    fn norm(&self, x: &mut u64, delta: Delta, w: u64, rows: usize, moe_f32: bool) -> Result<(), Error> {
        let b = self.b;
        let c = self.c;
        let next = if *x == b.h[0] { b.h[1] } else { b.h[0] };
        match delta {
            Delta::None => self.glue.add_rmsnorm(*x, None, w, *x, b.y, b.xs, rows, c.hidden, c.eps),
            Delta::Dense => {
                self.glue.add_rmsnorm(*x, Some(b.delta), w, next, b.y, b.xs, rows, c.hidden, c.eps)
            }
            Delta::Moe => self.glue.add_moe_norm(
                *x,
                b.ymoe,
                moe_f32,
                b.wts,
                w,
                next,
                b.y,
                b.xs,
                rows,
                c.hidden,
                c.eps,
                c.top_k,
                c.slots(),
            ),
        }?;
        if delta != Delta::None {
            *x = next;
        }
        if let Some(d) = self.dump {
            d.norm(*x, b.y, b.xs, rows, c.hidden)?;
        }
        self.mark(Class::Norm)
    }

    /// Engine._forward: rows tokens at meta's position; every row
    /// samples its next token into `sampled`. Upstream: `window`
    /// (cuda_forward.zig:97).
    pub fn window(&self, rows: usize) -> Result<(), Error> {
        let seg = Seg {
            b: self.b,
            row0: 0,
            rows,
            sampled: self.sampled,
        };
        self.window_segs(&[seg])
    }

    /// Shared matmuls, norms and experts retain each stream's mixer
    /// state. Upstream: `windowSegs` (cuda_forward.zig:102).
    pub fn window_segs(&self, segs: &[Seg<'_, '_>]) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let o = self.ops;
        let t = &self.glue;
        let ms = self.mshape(MAX_ROWS);
        let at = self.ashape();
        let d = c.hidden as u64;
        let mut rows: usize = 0;
        for s in segs {
            rows += s.rows;
        }
        if rows > MAX_ROWS {
            return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // WindowTooWide
        }
        for s in segs {
            t.embed(
                s.b.ids,
                self.w.embed.w,
                self.w.embed.s,
                self.w.embed.b,
                b.emb + s.row0 as u64 * d * 2,
                s.rows,
                c.hidden,
            )?;
        }
        self.mark(Class::Embed)?;
        let mut x = b.emb;
        let mut delta = Delta::None;
        let mut mj: usize = 0;
        let mut aj: usize = 0;
        for blk in &self.w.blocks {
            self.norm(&mut x, delta, blk.norm, rows, true)?;
            match blk.kind {
                crate::config::Kind::Mamba => {
                    let m = blk.mamba.unwrap();
                    let w = MAX_ROWS as u64;
                    o.dense(b.y, b.xs, m.in_proj, b.proj, rows)?;
                    self.mark(Class::MambaIn)?;
                    for s in segs {
                        let sb = s.b;
                        let raw = sb.raw + mj as u64 * 2 * w * c.conv_dim() as u64 * 2;
                        let xc = sb.xc + mj as u64 * 2 * w * c.conv_dim() as u64 * 2;
                        let dt = sb.dt + mj as u64 * 2 * w * c.mamba_heads as u64 * 4;
                        let ssm = sb.ssm
                            + mj as u64 * c.mamba_heads as u64
                                * c.mamba_head_dim as u64
                                * c.state as u64
                                * 4;
                        let base = sb.conv_base + mj as u64 * 3 * c.conv_dim() as u64 * 2;
                        let proj = b.proj + s.row0 as u64 * c.proj_dim() as u64 * 2;
                        t.conv(proj, base, raw, xc, m.conv_w, m.conv_b, sb.meta, s.rows, ms)?;
                        t.scan(
                            proj,
                            xc,
                            dt,
                            ssm,
                            m.a,
                            m.d,
                            m.dt_bias,
                            sb.meta,
                            b.sy + s.row0 as u64 * c.inner() as u64 * 2,
                            s.rows,
                            c.dt_min,
                            c.dt_max,
                            ms,
                        )?;
                    }
                    self.mark(Class::ConvScan)?;
                    t.group_rmsnorm(b.sy, m.gnorm, b.g, b.gxs, rows, c.inner(), c.groups, c.eps)?;
                    self.mark(Class::Gnorm)?;
                    o.dense(b.g, b.gxs, m.out_proj, b.delta, rows)?;
                    if m.out_rest != 0 {
                        o.rest_rows(
                            b.g,
                            c.inner(),
                            m.out_rest,
                            b.delta,
                            c.hidden,
                            false,
                            rows,
                            1,
                            0,
                            c.hidden,
                            c.inner(),
                        )?;
                    }
                    self.mark(Class::MambaOut)?;
                    delta = Delta::Dense;
                    mj += 1;
                }
                crate::config::Kind::Attention => {
                    let a = blk.attn.unwrap();
                    let qd = (c.heads * c.head_dim) as u64;
                    o.dense(b.y, b.xs, a.qkv, b.qkv, rows)?;
                    self.mark(Class::Qkv)?;
                    for s in segs {
                        let kc = self.cache(s.b.k_cache, aj);
                        let vc = self.cache(s.b.v_cache, aj);
                        let qkv = b.qkv + s.row0 as u64 * c.qkv_dim() as u64 * 2;
                        let r0 = s.row0 as u64;
                        t.kv_write(qkv, kc, vc, s.b.meta, s.rows, at)?;
                        t.attention(
                            qkv,
                            kc,
                            vc,
                            s.b.meta,
                            b.po + r0 * self.nch as u64 * qd * 4,
                            b.pm + r0 * self.nch as u64 * c.heads as u64 * 4,
                            b.pl + r0 * self.nch as u64 * c.heads as u64 * 4,
                            b.att + r0 * qd * 2,
                            b.axs + r0 * (qd / 64) * 4,
                            s.rows,
                            at,
                        )?;
                    }
                    self.mark(Class::Attention)?;
                    o.dense(b.att, b.axs, a.o, b.delta, rows)?;
                    if a.o_rest != 0 {
                        o.rest_rows(
                            b.att,
                            qd as usize,
                            a.o_rest,
                            b.delta,
                            c.hidden,
                            false,
                            rows,
                            1,
                            0,
                            c.hidden,
                            qd as usize,
                        )?;
                    }
                    self.mark(Class::AttnO)?;
                    delta = Delta::Dense;
                    aj += 1;
                }
                crate::config::Kind::Moe => {
                    self.experts(&blk.moe.unwrap(), rows, false)?;
                    delta = Delta::Moe;
                }
            }
        }
        self.norm(&mut x, delta, self.w.norm_f, rows, true)?;
        for s in segs {
            o.copy(s.b.hidden, b.y + s.row0 as u64 * d * 2, s.rows * c.hidden * 2)?;
        }
        o.dense(b.y, b.xs, self.w.head, b.logits, rows)?;
        self.mark(Class::Head)?;
        for s in segs {
            self.sample(
                b.logits + s.row0 as u64 * c.vocab as u64 * 2,
                s.rows,
                s.b.meta,
                s.b.sampled,
                s.sampled,
                s.b.rule,
            )?;
        }
        self.mark(Class::Sample)?;
        if let Some(dump) = self.dump {
            dump.tail(b.logits, b.sampled, rows, c.vocab)?;
        }
        Ok(())
    }

    /// Engine.moe / moe_rows: route each row's experts, group pairs by
    /// expert, then up (relu^2) and down. Upstream: `experts`
    /// (cuda_forward.zig:183).
    pub fn experts(&self, m: &Moe, rows: usize, prompt: bool) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let o = self.ops;
        let ex = m.experts;
        let pairs = rows * c.slots();
        if !prompt {
            if let Some(sd) = self.side {
                return self.forked(m, rows, sd);
            }
        }
        self.glue.route(
            b.y,
            m.router,
            m.bias,
            b.part,
            b.pick,
            b.wts,
            rows,
            c.hidden,
            c.experts,
            c.top_k,
            c.routed_scaling,
            c.norm_topk,
        )?;
        self.mark(Class::Route)?;
        let tile: usize = if prompt { 64 } else { 16 };
        o.plan(b.pick, pairs, ex.count, tile, b.plan)?;
        self.mark(Class::Plan)?;
        let items = cuda_kernels::max_items(pairs, ex.count, tile);
        if prompt {
            o.experts_prefill(
                true,
                b.y,
                c.hidden,
                c.slots(),
                ex.up,
                ex.dims / 64,
                ex.width / 32,
                b.plan,
                b.act,
                ex.width,
                items,
            )?;
            o.experts_prefill(
                false,
                b.act,
                ex.width,
                0,
                ex.down,
                ex.width / 64,
                ex.dims / 32,
                b.plan,
                b.ymoe,
                ex.dims,
                items,
            )?;
        } else {
            o.experts(
                true,
                b.y,
                c.hidden,
                c.slots(),
                ex.up,
                ex.dims / 64,
                ex.width / 32,
                b.plan,
                b.act,
                ex.width,
                items * (ex.width / 32),
            )?;
            self.mark(Class::ExpertsUp)?;
            o.experts(
                false,
                b.act,
                ex.width,
                0,
                ex.down,
                ex.width / 64,
                ex.dims / 32,
                b.plan,
                b.ymoe,
                ex.dims,
                items * (ex.dims / 32),
            )?;
            self.mark(Class::ExpertsDown)?;
        }
        self.shared_rest(m, rows, !prompt)
    }

    /// A lesson's change to the shared expert's down projection, past
    /// its codes: into its two halves' pair rows. Upstream:
    /// `sharedRest` (cuda_forward.zig:209).
    fn shared_rest(&self, m: &Moe, rows: usize, y_f32: bool) -> Result<(), Error> {
        let c = self.c;
        let ex = m.experts;
        for (h, r) in m.rest.iter().enumerate() {
            if *r != 0 {
                self.ops.rest_rows(
                    self.b.act,
                    ex.width,
                    *r,
                    self.b.ymoe,
                    ex.dims,
                    y_f32,
                    rows,
                    c.slots(),
                    c.top_k + h,
                    ex.dims,
                    ex.width,
                )?;
            }
        }
        Ok(())
    }

    /// The decode MoE with its shared halves on the side stream: their
    /// pairs are fixed, so they need no routing. Upstream: `forked`
    /// (cuda_forward.zig:216).
    fn forked(&self, m: &Moe, rows: usize, sd: &'a Side<'d>) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let o = self.ops;
        let ex = m.experts;
        let so = Ops {
            k: o.k,
            s: &sd.s,
        };
        sd.fork.record(o.s)?;
        sd.s.wait(&sd.fork)?;
        let sp = b.shared_plan(rows);
        so.experts(
            true,
            b.y,
            c.hidden,
            c.slots(),
            ex.up,
            ex.dims / 64,
            ex.width / 32,
            sp,
            b.act,
            ex.width,
            2 * (ex.width / 32),
        )?;
        so.experts(
            false,
            b.act,
            ex.width,
            0,
            ex.down,
            ex.width / 64,
            ex.dims / 32,
            sp,
            b.ymoe,
            ex.dims,
            2 * (ex.dims / 32),
        )?;
        self.shared_rest(m, rows, true)?;
        sd.join.record(&sd.s)?;
        self.glue.route(
            b.y,
            m.router,
            m.bias,
            b.part,
            b.pick,
            b.wts,
            rows,
            c.hidden,
            c.experts,
            c.top_k,
            c.routed_scaling,
            c.norm_topk,
        )?;
        self.mark(Class::Route)?;
        o.plan_routed(b.pick, rows, c.slots(), c.top_k, c.experts, 16, b.plan)?;
        self.mark(Class::Plan)?;
        let items = cuda_kernels::max_items(rows * c.top_k, c.experts, 16);
        o.experts(
            true,
            b.y,
            c.hidden,
            c.slots(),
            ex.up,
            ex.dims / 64,
            ex.width / 32,
            b.plan,
            b.act,
            ex.width,
            items * (ex.width / 32),
        )?;
        self.mark(Class::ExpertsUp)?;
        o.experts(
            false,
            b.act,
            ex.width,
            0,
            ex.down,
            ex.width / 64,
            ex.dims / 32,
            b.plan,
            b.ymoe,
            ex.dims,
            items * (ex.dims / 32),
        )?;
        self.mark(Class::ExpertsDown)?;
        o.s.wait(&sd.join)?;
        self.mark(Class::SharedWait)?;
        Ok(())
    }

    /// Engine.prefill_chunk: `rows` tokens in p_ids at positions pos..;
    /// commits them and samples the next token. Upstream: `chunk`
    /// (cuda_forward.zig:243).
    pub fn chunk(&self, rows: usize, pos: usize) -> Result<(), Error> {
        let mut w = Walk::new(rows, pos);
        self.chunk_begin(&mut w)?;
        for i in 0..self.w.blocks.len() {
            self.chunk_pre(&mut w, i)?;
            self.chunk_mixer(&mut w, i)?;
            self.chunk_post(&mut w, i)?;
        }
        self.chunk_finish(&mut w)
    }

    /// A chunk's embedding, from p_ids. Upstream: `chunkBegin`
    /// (cuda_forward.zig:255).
    pub fn chunk_begin(&self, w: &mut Walk) -> Result<(), Error> {
        let b = self.b;
        self.glue.embed(
            b.p_ids,
            self.w.embed.w,
            self.w.embed.s,
            self.w.embed.b,
            b.emb,
            w.rows,
            self.c.hidden,
        )?;
        w.x = b.emb;
        Ok(())
    }

    /// Block i up to its mixer: the norm, then the Mamba input
    /// projection or attention's q, k and v. Upstream: `chunkPre`
    /// (cuda_forward.zig:262).
    pub fn chunk_pre(&self, w: &mut Walk, i: usize) -> Result<(), Error> {
        let blk = self.w.blocks[i];
        let b = self.b;
        self.norm(&mut w.x, w.delta, blk.norm, w.rows, false)?;
        match blk.kind {
            crate::config::Kind::Mamba => {
                self.ops
                    .prefill_dense(b.y, blk.mamba.unwrap().in_proj, b.proj, w.rows)?;
            }
            crate::config::Kind::Attention => {
                self.ops
                    .prefill_dense(b.y, blk.attn.unwrap().qkv, b.qkv, w.rows)?;
                self.ops.fill32(b.p_meta, w.pos as u32, 4)?;
            }
            crate::config::Kind::Moe => {}
        }
        Ok(())
    }

    /// A block mixer reads its Mamba state or attention cache from
    /// earlier rows. Upstream: `chunkMixer` (cuda_forward.zig:277).
    pub fn chunk_mixer(&self, w: &mut Walk, i: usize) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let o = self.ops;
        match self.w.blocks[i].kind {
            crate::config::Kind::Mamba => {
                let m = self.w.blocks[i].mamba.unwrap();
                let ssm = b.ssm
                    + w.mj as u64 * c.mamba_heads as u64
                        * c.mamba_head_dim as u64
                        * c.state as u64
                        * 4;
                let base = b.conv_base + w.mj as u64 * 3 * c.conv_dim() as u64 * 2;
                self.glue.conv_rows(
                    b.proj,
                    base,
                    b.p_xc,
                    m.conv_w,
                    m.conv_b,
                    w.rows,
                    self.mshape(MAX_ROWS),
                )?;
                o.scan_rows(
                    b.proj,
                    b.p_xc,
                    ssm,
                    m.a,
                    m.d,
                    m.dt_bias,
                    b.sy,
                    w.rows,
                    c.proj_dim(),
                    c.mamba_heads,
                    c.mamba_head_dim,
                    c.conv_dim(),
                    c.groups,
                    c.dt_min,
                    c.dt_max,
                )?;
            }
            crate::config::Kind::Attention => {
                let kc = self.cache(b.k_cache, w.aj);
                let vc = self.cache(b.v_cache, w.aj);
                let qd = c.heads * c.head_dim;
                self.glue.kv_write(b.qkv, kc, vc, b.p_meta, w.rows, self.ashape())?;
                o.torch().copy_rows(b.qkv, c.qkv_dim() * 2, b.q, qd * 2, qd * 2, w.rows)?;
                let scale: f32 = (c.head_dim as f64).powf(-0.5) as f32;
                o.prefill_attention(b.q, kc, vc, b.att, w.pos, w.rows, c.heads, c.kv_heads, scale)?;
            }
            crate::config::Kind::Moe => {}
        }
        Ok(())
    }

    /// Block i after its mixer: Mamba gate norm and out projection,
    /// attention's out projection, or the experts. Upstream:
    /// `chunkPost` (cuda_forward.zig:303).
    pub fn chunk_post(&self, w: &mut Walk, i: usize) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let blk = self.w.blocks[i];
        match blk.kind {
            crate::config::Kind::Mamba => {
                let m = blk.mamba.unwrap();
                self.glue.group_rmsnorm(
                    b.sy,
                    m.gnorm,
                    b.g,
                    b.gxs,
                    w.rows,
                    c.inner(),
                    c.groups,
                    c.eps,
                )?;
                self.ops.prefill_dense(b.g, m.out_proj, b.delta, w.rows)?;
                if m.out_rest != 0 {
                    self.ops.rest_rows(
                        b.g,
                        c.inner(),
                        m.out_rest,
                        b.delta,
                        c.hidden,
                        false,
                        w.rows,
                        1,
                        0,
                        c.hidden,
                        c.inner(),
                    )?;
                }
                w.delta = Delta::Dense;
                w.mj += 1;
            }
            crate::config::Kind::Attention => {
                let a = blk.attn.unwrap();
                self.ops.prefill_dense(b.att, a.o, b.delta, w.rows)?;
                let qd = c.heads * c.head_dim;
                if a.o_rest != 0 {
                    self.ops.rest_rows(
                        b.att,
                        qd,
                        a.o_rest,
                        b.delta,
                        c.hidden,
                        false,
                        w.rows,
                        1,
                        0,
                        c.hidden,
                        qd,
                    )?;
                }
                w.delta = Delta::Dense;
                w.aj += 1;
            }
            crate::config::Kind::Moe => {
                self.experts(&blk.moe.unwrap(), w.rows, true)?;
                w.delta = Delta::Moe;
            }
        }
        Ok(())
    }

    /// The chunk's last norm, its rows' hidden states for the MTP head,
    /// and the token after its last row. Upstream: `chunkFinish`
    /// (cuda_forward.zig:330).
    pub fn chunk_finish(&self, w: &mut Walk) -> Result<(), Error> {
        let c = self.c;
        let b = self.b;
        let o = self.ops;
        let rows = w.rows;
        self.norm(&mut w.x, w.delta, self.w.norm_f, rows, false)?;
        o.copy(b.p_hidden, b.y, rows * c.hidden * 2)?;
        o.prefill_dense(b.y + (rows as u64 - 1) * c.hidden as u64 * 2, self.w.head, b.p_logits, 1)?;
        // sample_last: the chunk's last row is at pos + rows - 1
        o.fill32(b.p_meta, (w.pos + rows - 1) as u32, 4)?;
        self.sample(b.p_logits, 1, b.p_meta, b.p_sampled, self.sampled, b.rule)?;
        if let Some(dump) = self.dump {
            dump.tail(b.p_logits, b.p_sampled, 1, c.vocab)?;
        }
        Ok(())
    }
}
