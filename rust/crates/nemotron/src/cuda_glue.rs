//! Nemotron's glue on CUDA: our own kernels (nemotron_*.cu), or the
//! captured Triton set where one is loaded.
//!
//! Port of `zig/src/families/nemotron/cuda_glue.zig` (243 lines).

#![allow(clippy::too_many_arguments)] // the launch wrappers carry the kernels' parameters, upstream's own signatures

use tensorfold_cuda::{
    aot::Set, Args, Config as LaunchConfig, Dim3, Function, Module, Stream, launch,
};

use crate::cuda_triton::{divisor, Tri};

// Upstream re-exports the Triton glue's shape types (cuda_glue.zig:7-9).
pub use crate::cuda_triton::{Attn, Keyed, Shape};
use crate::error::Error;

/// Keys an attention chunk holds, at fixed absolute positions
/// (Python's CHUNK). Upstream: `chunk_keys` (cuda_glue.zig:12).
pub const CHUNK_KEYS: usize = 512;

/// Each attention block takes every chunk_blocks-th chunk, bounding
/// idle blocks at long windows. Upstream: `chunk_blocks`
/// (cuda_glue.zig:15).
pub const CHUNK_BLOCKS: usize = 64;

/// The native kernels, by their extern "C" names. Upstream: `Fns`
/// (cuda_glue.zig:18).
pub struct Fns<'d> {
    pub embed: Function<'d>,
    pub add_rmsnorm: Function<'d>,
    pub add_moe_norm: Function<'d>,
    pub concat_norms: Function<'d>,
    pub group_rmsnorm: Function<'d>,
    pub router: Function<'d>,
    pub topk: Function<'d>,
    pub conv: Function<'d>,
    pub conv_rows: Function<'d>,
    pub conv_commit: Function<'d>,
    pub scan: Function<'d>,
    pub kv_write: Function<'d>,
    pub attn_chunk: Function<'d>,
    pub attn_merge: Function<'d>,
    pub keyed: Function<'d>,
}

impl Fns<'_> {
    /// The modules of nemotron_norms, _route, _mamba, _attention and
    /// _keyed, in that order. Upstream: `resolve` (cuda_glue.zig:36).
    pub fn resolve<'d>(m: &[Module<'d>]) -> Result<Fns<'d>, Error> {
        Ok(Fns {
            embed: m[0].function("tf_nemo_embed")?,
            add_rmsnorm: m[0].function("tf_nemo_add_rmsnorm")?,
            add_moe_norm: m[0].function("tf_nemo_add_moe_norm")?,
            concat_norms: m[0].function("tf_nemo_concat_norms")?,
            group_rmsnorm: m[0].function("tf_nemo_group_rmsnorm")?,
            router: m[1].function("tf_nemo_router")?,
            topk: m[1].function("tf_nemo_topk")?,
            conv: m[2].function("tf_nemo_conv")?,
            conv_rows: m[2].function("tf_nemo_conv_rows")?,
            conv_commit: m[2].function("tf_nemo_conv_commit")?,
            scan: m[2].function("tf_nemo_scan")?,
            kv_write: m[3].function("tf_nemo_kv_write")?,
            attn_chunk: m[3].function("tf_nemo_attn_chunk")?,
            attn_merge: m[3].function("tf_nemo_attn_merge")?,
            keyed: m[4].function("tf_nemo_keyed_greedy")?,
        })
    }
}

fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// The router's K slices and the shared bytes its block stages: 16
/// rows and 16 experts of one slice, padded a word. Upstream:
/// `routerShape` (cuda_glue.zig:66).
pub fn router_shape(d: usize) -> (usize, u32) {
    let sk = divisor(d / 64, 6);
    (sk, (2 * 16 * (d / sk / 2 + 1) * 4) as u32)
}

/// Every launch of a window or prompt chunk's glue, on one stream.
/// Upstream: `Glue` (cuda_glue.zig:72).
pub struct Glue<'a, 'd> {
    /// A captured Triton set (GB10's qualified one), else our own
    /// kernels.
    pub set: Option<&'a Set<'d>>,
    pub f: &'a Fns<'d>,
    pub s: &'a Stream<'d>,
}

impl Glue<'_, '_> {
    fn tri(&self) -> Option<Tri<'_, '_>> {
        self.set.map(|x| Tri {
            set: x,
            s: self.s,
        })
    }

    fn go(
        &self,
        f: &Function<'_>,
        grid: [usize; 3],
        block: u32,
        shared: u32,
        a: &mut Args,
    ) -> Result<(), Error> {
        launch(
            f,
            LaunchConfig {
                grid: Dim3 {
                    x: grid[0] as u32,
                    y: grid[1] as u32,
                    z: grid[2] as u32,
                },
                block: Dim3 { x: block, y: 1, z: 1 },
                shared,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            self.s,
            a,
        )?;
        Ok(())
    }

    fn ptrs(a: &mut Args, xs: &[u64]) {
        for x in xs {
            a.add(*x);
        }
    }

    fn ints(a: &mut Args, xs: &[usize]) {
        for x in xs {
            a.add(*x as std::ffi::c_int);
        }
    }

    /// MLX 4-bit token rows, dequantized. Upstream: `embed`
    /// (cuda_glue.zig:95).
    pub fn embed(&self, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.embed(ids, w, s, b, out, rows, d);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[ids, w, s, b, out]);
        a.add(d as std::ffi::c_int);
        self.go(&self.f.embed, [rows, 1, 1], 256, 0, &mut a)
    }

    /// h = x + r (or x itself without r), y = rmsnorm(h) * w, xs = y's
    /// 64-group sums. Upstream: `addRmsnorm` (cuda_glue.zig:104).
    pub fn add_rmsnorm(&self, x: u64, r: Option<u64>, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.add_rmsnorm(x, r, w, h, y, xs, rows, d, eps);
        }
        if d > 4096 {
            return Err(Error::RowTooWide);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[x, r.unwrap_or(0), w, h, y, xs]);
        a.add(eps);
        a.add(d as std::ffi::c_int);
        self.go(&self.f.add_rmsnorm, [rows, 1, 1], 256, 0, &mut a)
    }

    /// The routed and shared expert sums in slot order, added to h,
    /// then its RMSNorm. Upstream: `addMoeNorm` (cuda_glue.zig:115).
    pub fn add_moe_norm(&self, h: u64, y: u64, y_f32: bool, wt: u64, w: u64, hn: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32, top_k: usize, slots: usize) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.add_moe_norm(h, y, y_f32, wt, w, hn, out, xs, rows, d, eps, top_k, slots);
        }
        if d > 4096 {
            return Err(Error::RowTooWide);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[h, y]);
        a.add(y_f32 as std::ffi::c_int);
        Self::ptrs(&mut a, &[wt, w, hn, out, xs]);
        a.add(eps);
        Self::ints(&mut a, &[d, top_k, slots]);
        self.go(&self.f.add_moe_norm, [rows, 1, 1], 256, 0, &mut a)
    }

    /// fp32 router logits over K slices, then each row's top-k and the
    /// shared halves' slots. Upstream: `route` (cuda_glue.zig:122).
    pub fn route(&self, x: u64, w: u64, bias: u64, part: u64, ids: u64, wts: u64, rows: usize, d: usize, e: usize, top_k: usize, scaling: f32, norm: bool) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.route(x, w, bias, part, ids, wts, rows, d, e, top_k, scaling, norm);
        }
        if e > 256 || top_k + 2 > 8 {
            return Err(Error::RouterTooWide);
        }
        let (sk, shared) = router_shape(d);
        let mut a = Args::new();
        Self::ptrs(&mut a, &[x, w, part]);
        Self::ints(&mut a, &[rows, d, e, sk]);
        self.go(&self.f.router, [cdiv(rows, 16), cdiv(e, 16), sk], 256, shared, &mut a)?;
        let mut b = Args::new();
        Self::ptrs(&mut b, &[part, bias, ids, wts]);
        b.add(rows as std::ffi::c_int);
        b.add(scaling);
        Self::ints(&mut b, &[e, sk, top_k, top_k + 2]);
        b.add(norm as std::ffi::c_int);
        self.go(&self.f.topk, [rows, 1, 1], 32, 0, &mut b)
    }

    /// The Mamba conv of a window: the last window's kept rows replay
    /// from RAW, then the window's rows. Upstream: `conv`
    /// (cuda_glue.zig:129).
    pub fn conv(&self, proj: u64, base: u64, raw: u64, xc: u64, cw: u64, cb: u64, meta: u64, rows: usize, m: Shape) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.conv(proj, base, raw, xc, cw, cb, meta, rows, m);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[proj, base, raw, xc, cw, cb, meta]);
        Self::ints(&mut a, &[rows, m.proj, m.xd, m.cd, m.rmax]);
        self.go(&self.f.conv, [cdiv(m.cd, 256), 1, 1], 256, 0, &mut a)
    }

    /// The Mamba scan of a window: the state lags a window; kept rows
    /// replay first in the same loop body. Upstream: `scan`
    /// (cuda_glue.zig:136).
    pub fn scan(&self, proj: u64, xc: u64, dt: u64, state: u64, a_: u64, dsk: u64, dtb: u64, meta: u64, y: u64, rows: usize, lo: f32, hi: f32, m: Shape) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.scan(proj, xc, dt, state, a_, dsk, dtb, meta, y, rows, lo, hi, m);
        }
        if m.state != 128 || m.dh % 32 != 0 {
            return Err(Error::ScanShape);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[proj, xc, dt, state, a_, dsk, dtb, meta, y]);
        a.add(rows as std::ffi::c_int);
        a.add(lo);
        a.add(hi);
        Self::ints(&mut a, &[m.proj, m.xd, m.cd, m.xd + m.cd, m.heads, m.dh, m.groups, m.rmax]);
        self.go(&self.f.scan, [m.heads, m.dh / 32, 1], 128, 0, &mut a)
    }

    /// Weight * RMSNorm over groups of xd / groups, and the 64-input
    /// group sums. Upstream: `groupRmsnorm` (cuda_glue.zig:143).
    pub fn group_rmsnorm(&self, x: u64, w: u64, out: u64, xs: u64, rows: usize, xd: usize, groups: usize, eps: f32) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.group_rmsnorm(x, w, out, xs, rows, xd, groups, eps);
        }
        if xd / groups > 1024 {
            return Err(Error::RowTooWide);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[x, w, out, xs]);
        a.add(eps);
        Self::ints(&mut a, &[xd, xd / groups]);
        self.go(&self.f.group_rmsnorm, [rows, groups, 1], 128, 0, &mut a)
    }

    /// A prompt chunk's conv rows in parallel, then BASE takes its
    /// last three inputs. Upstream: `convRows` (cuda_glue.zig:150).
    pub fn conv_rows(&self, proj: u64, base: u64, xc: u64, cw: u64, cb: u64, rows: usize, m: Shape) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.conv_rows(proj, base, xc, cw, cb, rows, m);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[proj, base, xc, cw, cb]);
        Self::ints(&mut a, &[rows, m.proj, m.xd, m.cd]);
        self.go(&self.f.conv_rows, [rows, cdiv(m.cd, 256), 1], 256, 0, &mut a)?;
        let mut b = Args::new();
        Self::ptrs(&mut b, &[proj, base]);
        Self::ints(&mut b, &[rows, m.proj, m.xd, m.cd]);
        self.go(&self.f.conv_commit, [cdiv(m.cd, 256), 1, 1], 256, 0, &mut b)
    }

    /// Each row's keys and values into the caches at meta's position.
    /// Upstream: `kvWrite` (cuda_glue.zig:159).
    pub fn kv_write(&self, qkv: u64, kc: u64, vc: u64, meta: u64, rows: usize, at: Attn) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.kv_write(qkv, kc, vc, meta, rows, at);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[qkv, kc, vc, meta]);
        Self::ints(&mut a, &[at.nqkv, at.heads * at.dim, at.kv_heads * at.dim]);
        self.go(&self.f.kv_write, [rows, 1, 1], 256, 0, &mut a)
    }

    /// 512-key chunks at absolute positions, merged in order (a row's
    /// bits ignore its window). Upstream: `attention` (cuda_glue.zig:166).
    pub fn attention(&self, qkv: u64, kc: u64, vc: u64, meta: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, at: Attn) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.attention(qkv, kc, vc, meta, po, pm, pl, out, xs, rows, at);
        }
        let grp = at.heads / at.kv_heads;
        if at.dim != 128 || grp > 16 {
            return Err(Error::AttentionShape);
        }
        let scale: f32 = (at.dim as f64).powf(-0.5) as f32;
        let mut a = Args::new();
        Self::ptrs(&mut a, &[qkv, kc, vc, meta, po, pm, pl]);
        Self::ints(&mut a, &[at.nqkv, at.heads, at.kv_heads, grp, CHUNK_KEYS, at.nch]);
        a.add(scale);
        self.go(
            &self.f.attn_chunk,
            [rows, at.kv_heads, at.nch.min(CHUNK_BLOCKS)],
            256,
            0,
            &mut a,
        )?;
        let mut b = Args::new();
        Self::ptrs(&mut b, &[po, pm, pl, meta, out, xs]);
        Self::ints(&mut b, &[at.heads, at.kv_heads, grp, CHUNK_KEYS, at.nch]);
        self.go(&self.f.attn_merge, [rows, at.kv_heads, 1], 256, 0, &mut b)
    }

    /// The MTP input [rmsnorm(e) * enorm | rmsnorm(h) * hnorm] and its
    /// group sums. Upstream: `concatNorms` (cuda_glue.zig:180).
    pub fn concat_norms(&self, e: u64, h: u64, we: u64, wh: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.concat_norms(e, h, we, wh, out, xs, rows, d, eps);
        }
        if d > 4096 {
            return Err(Error::RowTooWide);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[e, h, we, wh, out, xs]);
        a.add(eps);
        a.add(d as std::ffi::c_int);
        self.go(&self.f.concat_norms, [rows, 2, 1], 256, 0, &mut a)
    }

    /// A draw over candidate rows; our kernel takes the greedy rule
    /// (the head's draw), with or without its share. Upstream: `keyed`
    /// (cuda_glue.zig:189).
    pub fn keyed(&self, vals: u64, ids: u64, meta: u64, out: u64, seed: u64, fp: u64, prob: Option<u64>, offset: usize, rows: usize, count: usize, k: Keyed) -> Result<(), Error> {
        if let Some(t) = self.tri() {
            return t.keyed(vals, ids, meta, out, seed, fp, prob, offset, rows, count, k);
        }
        if !k.greedy || count > 32 {
            return Err(Error::KeyedRule);
        }
        let mut a = Args::new();
        Self::ptrs(&mut a, &[vals, ids, out, prob.unwrap_or(0)]);
        Self::ints(&mut a, &[count, k.k.min(count)]);
        self.go(&self.f.keyed, [rows, 1, 1], 32, 0, &mut a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `the router stages one slice of 16 rows and 16
    /// experts` (cuda_glue.zig:216).
    #[test]
    fn the_router_stages_one_slice_of_16_rows_and_16_experts() {
        let (sk, shared) = router_shape(2688);
        assert_eq!(6, sk);
        assert_eq!(2 * 16 * 225 * 4, shared);
    }
}
