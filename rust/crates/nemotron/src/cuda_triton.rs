//! Nemotron's Triton kernels from the captured cubins: each launch has
//! the Python wrapper's grid and constexprs.
//!
//! Port of `zig/src/families/nemotron/cuda_triton.zig` (119 lines).

#![allow(clippy::too_many_arguments)] // the launch wrappers carry the kernels' parameters, upstream's own signatures

use tensorfold_cuda::{aot::{self, Arg, Const, Set}, Stream};

use crate::error::Error;

fn int(name: &'static str, v: usize) -> Arg {
    aot::int(name, v as i32)
}

fn ci(name: &'static str, v: usize) -> Const {
    aot::ci(name, v as i64)
}

fn u(x: usize) -> u32 {
    x as u32
}

fn cdiv(a: usize, b: usize) -> u32 {
    u(a.div_ceil(b))
}

fn pow2(n: usize) -> usize {
    n.next_power_of_two()
}

/// glue._divisor: the largest K-slice count up to `want` dividing `n`.
/// Upstream: `divisor` (cuda_triton.zig:30).
pub fn divisor(n: usize, want: usize) -> usize {
    let mut g = want.min(n);
    while g > 1 {
        if n % g == 0 {
            return g;
        }
        g -= 1;
    }
    1
}

/// The captured set's launches on one stream. Upstream: `Tri`
/// (cuda_triton.zig:36) — the port's Stream owns its CUDA object, so
/// the launches borrow it.
pub struct Tri<'a, 'd> {
    pub set: &'a Set<'d>,
    pub s: &'a Stream<'d>,
}

impl Tri<'_, '_> {
    fn run(
        &self,
        name: &str,
        grid: [u32; 3],
        args: &[Arg],
        consts: &[Const],
    ) -> Result<(), Error> {
        Ok(self.set.run(self.s, name, grid, args, consts)?)
    }

    /// Upstream: `embed` (cuda_triton.zig:44).
    pub fn embed(&self, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) -> Result<(), Error> {
        self.run(
            "_embed",
            [u(rows), u(d / 64), 1],
            &[
                aot::ptr("IDS", "*i32", ids),
                aot::ptr("Wt", "*i32", w),
                aot::ptr("S", "*bf16", s),
                aot::ptr("B", "*bf16", b),
                aot::ptr("OUT", "*bf16", out),
            ],
            &[ci("D", d)],
        )
    }

    /// base.add_rmsnorm: h = x + r (or x itself without r), y =
    /// rmsnorm(h) * w, xs = y's 64-group sums. Upstream: `addRmsnorm`
    /// (cuda_triton.zig:49).
    pub fn add_rmsnorm(&self, x: u64, r: Option<u64>, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) -> Result<(), Error> {
        self.run(
            "_add_rmsnorm",
            [u(rows), 1, 1],
            &[
                aot::ptr("X", "*bf16", x),
                aot::ptr("R", "*bf16", r.unwrap_or(x)),
                aot::ptr("W", "*bf16", w),
                aot::ptr("H", "*bf16", if r.is_some() { h } else { x }),
                aot::ptr("Y", "*bf16", y),
                aot::ptr("XS", "*fp32", xs),
                aot::float("eps", eps),
            ],
            &[
                ci("D", d),
                ci("BLOCK", pow2(d)),
                ci("HAS_R", r.is_some() as usize),
            ],
        )
    }

    /// glue.add_moe_norm: the routed and shared expert sums in slot
    /// order, added to h, then its RMSNorm. Upstream: `addMoeNorm`
    /// (cuda_triton.zig:54).
    pub fn add_moe_norm(&self, h: u64, y: u64, y_f32: bool, wt: u64, w: u64, hn: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32, top_k: usize, slots: usize) -> Result<(), Error> {
        self.run(
            "_add_moe_norm",
            [u(rows), 1, 1],
            &[
                aot::ptr("H", "*bf16", h),
                aot::ptr("Y", if y_f32 { "*fp32" } else { "*bf16" }, y),
                aot::ptr("WT", "*fp32", wt),
                aot::ptr("W", "*bf16", w),
                aot::ptr("HN", "*bf16", hn),
                aot::ptr("OUT", "*bf16", out),
                aot::ptr("XS", "*fp32", xs),
                aot::float("eps", eps),
            ],
            &[
                ci("D", d),
                ci("BLOCK", pow2(d)),
                ci("NR", top_k),
                ci("NS", slots),
            ],
        )
    }

    /// glue.route: fp32 router partials over K slices, then each row's
    /// top-k and the shared halves' slots. Upstream: `route`
    /// (cuda_triton.zig:59).
    pub fn route(&self, x: u64, w: u64, bias: u64, part: u64, ids: u64, wts: u64, rows: usize, d: usize, e: usize, top_k: usize, scaling: f32, norm: bool) -> Result<(), Error> {
        let sk = divisor(d / 64, 6);
        self.run(
            "_router",
            [cdiv(rows, 16), cdiv(e, 16), u(sk)],
            &[
                aot::ptr("X", "*bf16", x),
                aot::ptr("W", "*bf16", w),
                aot::ptr("PART", "*fp32", part),
                int("R", rows),
            ],
            &[ci("D", d), ci("E", e), ci("SK", sk), ci("BE", 16)],
        )?;
        self.run(
            "_topk",
            [u(rows), 1, 1],
            &[
                aot::ptr("PART", "*fp32", part),
                aot::ptr("BIAS", "*fp32", bias),
                aot::ptr("IDX", "*i32", ids),
                aot::ptr("WT", "*fp32", wts),
                int("R", rows),
                aot::float("scaling", scaling),
            ],
            &[
                ci("E", e),
                ci("EP", pow2(e)),
                ci("SK", sk),
                ci("TOPK", top_k),
                ci("NS", top_k + 2),
                ci("NORM", norm as usize),
            ],
        )
    }

    /// mamba.conv: the last window's kept rows replay from RAW, then
    /// the window's rows. Upstream: `conv` (cuda_triton.zig:68).
    pub fn conv(&self, proj: u64, base: u64, raw: u64, xc: u64, cw: u64, cb: u64, meta: u64, rows: usize, m: Shape) -> Result<(), Error> {
        self.run(
            "_conv",
            [cdiv(m.cd, 256), 1, 1],
            &[
                aot::ptr("P", "*bf16", proj),
                aot::ptr("BASE", "*bf16", base),
                aot::ptr("RAW", "*bf16", raw),
                aot::ptr("XC", "*bf16", xc),
                aot::ptr("CW", "*fp32", cw),
                aot::ptr("CB", "*fp32", cb),
                aot::ptr("META", "*i32", meta),
                int("R", rows),
            ],
            &[
                ci("PROJ", m.proj),
                ci("XOFF", m.xd),
                ci("CD", m.cd),
                ci("RMAX", m.rmax),
                ci("BC", 256),
            ],
        )
    }

    /// mamba.scan: the state lags a window; kept rows replay first in
    /// the same loop body. Upstream: `scan` (cuda_triton.zig:73).
    pub fn scan(&self, proj: u64, xc: u64, dt: u64, state: u64, a: u64, dsk: u64, dtb: u64, meta: u64, y: u64, rows: usize, lo: f32, hi: f32, m: Shape) -> Result<(), Error> {
        self.run(
            "_scan",
            [u(m.heads), cdiv(m.dh, 32), 1],
            &[
                aot::ptr("P", "*bf16", proj),
                aot::ptr("XC", "*bf16", xc),
                aot::ptr("DT", "*fp32", dt),
                aot::ptr("S", "*fp32", state),
                aot::ptr("A", "*fp32", a),
                aot::ptr("DSK", "*fp32", dsk),
                aot::ptr("DTB", "*fp32", dtb),
                aot::ptr("META", "*i32", meta),
                aot::ptr("Y", "*bf16", y),
                int("R", rows),
                aot::float("lo", lo),
                aot::float("hi", hi),
            ],
            &[
                ci("PROJ", m.proj),
                ci("XD", m.xd),
                ci("CD", m.cd),
                ci("DTOFF", m.xd + m.cd),
                ci("H", m.heads),
                ci("DH", m.dh),
                ci("NG", m.groups),
                ci("DS", m.state),
                ci("RMAX", m.rmax),
                ci("BD", 32),
            ],
        )
    }

    /// Upstream: `groupRmsnorm` (cuda_triton.zig:77).
    pub fn group_rmsnorm(&self, x: u64, w: u64, out: u64, xs: u64, rows: usize, xd: usize, groups: usize, eps: f32) -> Result<(), Error> {
        self.run(
            "_group_rmsnorm",
            [u(rows), u(groups), 1],
            &[
                aot::ptr("X", "*bf16", x),
                aot::ptr("W", "*bf16", w),
                aot::ptr("OUT", "*bf16", out),
                aot::ptr("XS", "*fp32", xs),
                aot::float("eps", eps),
            ],
            &[ci("XD", xd), ci("GS", xd / groups)],
        )
    }

    /// mamba.conv_rows then commit_conv_rows: a chunk's rows in
    /// parallel, then BASE takes its last three inputs. Upstream:
    /// `convRows` (cuda_triton.zig:82).
    pub fn conv_rows(&self, proj: u64, base: u64, xc: u64, cw: u64, cb: u64, rows: usize, m: Shape) -> Result<(), Error> {
        self.run(
            "_conv_rows",
            [cdiv(rows, 16), cdiv(m.cd, 128), 1],
            &[
                aot::ptr("P", "*bf16", proj),
                aot::ptr("BASE", "*bf16", base),
                aot::ptr("XC", "*bf16", xc),
                aot::ptr("CW", "*fp32", cw),
                aot::ptr("CB", "*fp32", cb),
                int("R", rows),
            ],
            &[
                ci("PROJ", m.proj),
                ci("XOFF", m.xd),
                ci("CD", m.cd),
                ci("BR", 16),
                ci("BC", 128),
            ],
        )?;
        self.run(
            "_conv_commit",
            [cdiv(m.cd, 128), 1, 1],
            &[
                aot::ptr("P", "*bf16", proj),
                aot::ptr("BASE", "*bf16", base),
                int("R", rows),
            ],
            &[
                ci("PROJ", m.proj),
                ci("XOFF", m.xd),
                ci("CD", m.cd),
                ci("BC", 128),
            ],
        )
    }

    /// Upstream: `kvWrite` (cuda_triton.zig:89).
    pub fn kv_write(&self, qkv: u64, kc: u64, vc: u64, meta: u64, rows: usize, a: Attn) -> Result<(), Error> {
        self.run(
            "_kv_write",
            [u(rows), 1, 1],
            &[
                aot::ptr("QKV", "*bf16", qkv),
                aot::ptr("KC", "*bf16", kc),
                aot::ptr("VC", "*bf16", vc),
                aot::ptr("META", "*i32", meta),
                int("R", rows),
            ],
            &[
                ci("NQKV", a.nqkv),
                ci("QD", a.heads * a.dim),
                ci("KVD", a.kv_heads * a.dim),
            ],
        )
    }

    /// attention.attention: 512-key chunks at absolute positions,
    /// merged in order (a row's bits ignore its window). Upstream:
    /// `attention` (cuda_triton.zig:94).
    pub fn attention(&self, qkv: u64, kc: u64, vc: u64, meta: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, a: Attn) -> Result<(), Error> {
        let g = a.heads / a.kv_heads;
        let scale: f32 = (a.dim as f64).powf(-0.5) as f32;
        self.run(
            "_chunk",
            [u(rows), u(a.kv_heads), u(a.nch)],
            &[
                aot::ptr("QKV", "*bf16", qkv),
                aot::ptr("KC", "*bf16", kc),
                aot::ptr("VC", "*bf16", vc),
                aot::ptr("META", "*i32", meta),
                aot::ptr("PO", "*fp32", po),
                aot::ptr("PM", "*fp32", pm),
                aot::ptr("PL", "*fp32", pl),
            ],
            &[
                ci("NQKV", a.nqkv),
                ci("H", a.heads),
                ci("HK", a.kv_heads),
                ci("D", a.dim),
                ci("G", g),
                ci("CH", 512),
                ci("NCH", a.nch),
                aot::cf("SCALE", scale),
            ],
        )?;
        self.run(
            "_merge",
            [u(rows), u(a.kv_heads), 1],
            &[
                aot::ptr("PO", "*fp32", po),
                aot::ptr("PM", "*fp32", pm),
                aot::ptr("PL", "*fp32", pl),
                aot::ptr("META", "*i32", meta),
                aot::ptr("OUT", "*bf16", out),
                aot::ptr("XS", "*fp32", xs),
            ],
            &[
                ci("H", a.heads),
                ci("HK", a.kv_heads),
                ci("D", a.dim),
                ci("G", g),
                ci("CH", 512),
                ci("NCH", a.nch),
            ],
        )
    }

    /// glue.concat_norms: the MTP input [rmsnorm(e) * enorm |
    /// rmsnorm(h) * hnorm] and its group sums. Upstream: `concatNorms`
    /// (cuda_triton.zig:102).
    pub fn concat_norms(&self, e: u64, h: u64, we: u64, wh: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) -> Result<(), Error> {
        self.run(
            "_concat_norms",
            [u(rows), 2, 1],
            &[
                aot::ptr("E", "*bf16", e),
                aot::ptr("Hd", "*bf16", h),
                aot::ptr("WE", "*bf16", we),
                aot::ptr("WH", "*bf16", wh),
                aot::ptr("OUT", "*bf16", out),
                aot::ptr("XS", "*fp32", xs),
                aot::float("eps", eps),
            ],
            &[ci("D", d), ci("BLOCK", pow2(d))],
        )
    }

    /// sampler._keyed, one program a row: row r draws position
    /// META[0] + r + 1 + offset (without `prob`, PROB is OUT, as
    /// Python passes it). Upstream: `keyed` (cuda_triton.zig:107).
    pub fn keyed(&self, vals: u64, ids: u64, meta: u64, out: u64, seed: u64, fp: u64, prob: Option<u64>, offset: usize, rows: usize, count: usize, k: Keyed) -> Result<(), Error> {
        let share = match prob {
            Some(x) => aot::ptr("PROB", "*fp32", x),
            None => aot::ptr("PROB", "*i32", out),
        };
        self.run(
            "_keyed",
            [u(rows), 1, 1],
            &[
                aot::ptr("VALS", "*fp32", vals),
                aot::ptr("IDS", "*i64", ids),
                aot::ptr("META", "*i32", meta),
                aot::ptr("OUT", "*i32", out),
                aot::ptr("SEED", "*i64", seed),
                aot::ptr("FP", "*fp64", fp),
                share,
                aot::word("c1", 0x9E3779B97F4A7C15),
                aot::word("c2", 0xD1B54A32D192ED03),
                aot::word("m1", 0xBF58476D1CE4E5B9),
                aot::word("m2", 0x94D049BB133111EB),
                int("offset", offset),
            ],
            &[
                ci("C", count),
                ci("CP", pow2(count)),
                ci("K", k.k),
                ci("CUT", k.cut as usize),
                ci("GREEDY", k.greedy as usize),
                ci("WRITE_PROB", prob.is_some() as usize),
                ci("MINP", k.minp as usize),
                ci("CONF_T", k.conf_t as usize),
            ],
        )
    }
}

/// The Mamba layer's shapes the Triton glue takes (the state size
/// named, unlike glue_ref's). Upstream: `Tri.Shape` (cuda_triton.zig:65).
#[derive(Clone, Copy)]
pub struct Shape {
    pub proj: usize,
    pub xd: usize,
    pub cd: usize,
    pub heads: usize,
    pub dh: usize,
    pub groups: usize,
    pub state: usize,
    pub rmax: usize,
}

/// The attention layer's shapes the Triton glue takes. Upstream:
/// `Tri.Attn` (cuda_triton.zig:87).
#[derive(Clone, Copy)]
pub struct Attn {
    pub nqkv: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub nch: usize,
}

/// The rule a _keyed launch compiles in: top_k, the top-p cut, greedy
/// rank 0, the min-p cut, a draft's share at the request's temperature.
/// Upstream: `Keyed` (cuda_triton.zig:114).
#[derive(Clone, Copy)]
pub struct Keyed {
    pub k: usize,
    pub cut: bool,
    pub greedy: bool,
    pub minp: bool,
    pub conf_t: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `router K slices` (cuda_triton.zig:116).
    #[test]
    fn router_k_slices() {
        assert_eq!(6, divisor(42, 6));
        assert_eq!(4, divisor(28, 6));
    }
}
