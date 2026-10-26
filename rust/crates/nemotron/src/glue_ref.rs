//! Host references for the glue kernels (nemotron_*.cu): every output's
//! bits as the kernel computes them.
//!
//! Port of `zig/src/families/nemotron/glue_ref.zig` (418 lines) — the
//! exact fma/rounding chains, the thread-shaped reductions and the
//! online-softmax structure of the attention kernel, all carried op
//! for op.

#![allow(clippy::too_many_arguments)] // the reference kernels' parameter sets, upstream's own signatures

use crate::glue_math as m;

use m::{bf16_to_f32 as bf, f32_to_bf16 as tobf};

const NT: usize = 256; // the norm kernels' threads a row

/// An xor butterfly over n lanes (n / 2 down to 1), each lane adding
/// its partner to its own value. Upstream: `butterfly`
/// (glue_ref.zig:13).
fn butterfly<T: Copy + std::ops::Add<Output = T>, const N: usize>(v: [T; N]) -> T {
    let mut a = v;
    let mut off = N / 2;
    while off > 0 {
        let mut b = a;
        for l in 0..N {
            b[l] = a[l] + a[l ^ off];
        }
        a = b;
        off >>= 1;
    }
    a[0]
}

/// A row's sum of squares as `threads` threads take it: element i on
/// thread i % threads, warps summed in order. Upstream: `sumSquares`
/// (glue_ref.zig:25).
fn sum_squares<const THREADS: usize>(x: &[f32]) -> f32 {
    let mut lane = [0.0f32; THREADS];
    for (i, v) in x.iter().enumerate() {
        lane[i % THREADS] = m::fma(*v, *v, lane[i % THREADS]);
    }
    let mut ss = 0.0f32;
    for w in 0..THREADS / 32 {
        let mut warps = [0.0f32; 32];
        warps.copy_from_slice(&lane[w * 32..w * 32 + 32]);
        let s = butterfly::<f32, 32>(warps);
        ss = if w == 0 { s } else { ss + s };
    }
    ss
}

fn inv_rms(ss: f32, n: usize, eps: f32) -> f32 {
    1.0 / (ss / (n as f32) + eps).sqrt()
}

/// y = bf16((x * inv) * w), and each 64-group's sum of the stored
/// values in order. Upstream: `scaleStore` (glue_ref.zig:41).
fn scale_store(x: &[f32], inv: f32, w: &[u16], out: &mut [u16], xs: &mut [f32]) {
    for ((v, wv), o) in x.iter().zip(w).zip(out.iter_mut()) {
        *o = tobf((*v * inv) * bf(*wv));
    }
    for (g, s) in xs.iter_mut().enumerate() {
        let mut acc: f32 = 0.0;
        for y in &out[g * 64..g * 64 + 64] {
            acc += bf(*y);
        }
        *s = acc;
    }
}

/// tf_nemo_embed: MLX 4-bit rows, fma(q, scale, bias) rounded to bf16.
/// Upstream: `embed` (glue_ref.zig:51).
pub fn embed(ids: &[i32], w: &[u32], s: &[u16], b: &[u16], out: &mut [u16], d: usize) {
    for (r, id) in ids.iter().enumerate() {
        let tok = *id as usize;
        for i in 0..d {
            let q = ((w[tok * (d / 8) + i / 8] >> (4 * (i % 8))) & 0xF) as f32;
            let g = tok * (d / 64) + i / 64;
            out[r * d + i] = tobf(m::fma(q, bf(s[g]), bf(b[g])));
        }
    }
}

/// tf_nemo_add_rmsnorm: h = bf16(x + r) unless r is null, then y and
/// xs. Upstream: `addRmsnorm` (glue_ref.zig:63).
pub fn add_rmsnorm(
    x: &[u16],
    r: Option<&[u16]>,
    w: &[u16],
    h: &mut [u16],
    y: &mut [u16],
    xs: &mut [f32],
    rows: usize,
    d: usize,
    eps: f32,
) {
    let mut v = vec![0.0f32; d];
    for row in 0..rows {
        for i in 0..d {
            v[i] = bf(x[row * d + i]);
            if let Some(rr) = r {
                h[row * d + i] = tobf(v[i] + bf(rr[row * d + i]));
                v[i] = bf(h[row * d + i]);
            }
        }
        let g = d / 64;
        scale_store(
            &v,
            inv_rms(sum_squares::<NT>(&v), d, eps),
            w,
            &mut y[row * d..row * d + d],
            &mut xs[row * g..row * g + g],
        );
    }
}

/// tf_nemo_add_moe_norm: delta from the slots in order (routed by fma,
/// shared added), h, then y and xs. Upstream: `addMoeNorm`
/// (glue_ref.zig:79).
pub fn add_moe_norm(
    x: &[u16],
    y32: Option<&[f32]>,
    y16: Option<&[u16]>,
    wt: &[f32],
    w: &[u16],
    hn: &mut [u16],
    out: &mut [u16],
    xs: &mut [f32],
    rows: usize,
    d: usize,
    eps: f32,
    nr: usize,
    ns: usize,
) {
    let mut v = vec![0.0f32; d];
    for row in 0..rows {
        for i in 0..d {
            let mut routed: f32 = 0.0;
            let mut shared: f32 = 0.0;
            for s in 0..ns {
                let at = (row * ns + s) * d + i;
                let yv = match y32 {
                    Some(a) => a[at],
                    None => bf(y16.unwrap()[at]),
                };
                if s < nr {
                    routed = m::fma(wt[row * ns + s], yv, routed);
                } else {
                    shared += yv;
                }
            }
            hn[row * d + i] = tobf(bf(x[row * d + i]) + m::rbf(routed + shared));
            v[i] = bf(hn[row * d + i]);
        }
        let g = d / 64;
        scale_store(
            &v,
            inv_rms(sum_squares::<NT>(&v), d, eps),
            w,
            &mut out[row * d..row * d + d],
            &mut xs[row * g..row * g + g],
        );
    }
}

/// tf_nemo_concat_norms: [rmsnorm(e) * we | rmsnorm(h) * wh] and its
/// group sums. Upstream: `concatNorms` (glue_ref.zig:99).
pub fn concat_norms(
    e: &[u16],
    h: &[u16],
    we: &[u16],
    wh: &[u16],
    out: &mut [u16],
    xs: &mut [f32],
    rows: usize,
    d: usize,
    eps: f32,
) {
    let mut v = vec![0.0f32; d];
    for row in 0..rows {
        for part in 0..2 {
            let src = if part == 0 { e } else { h };
            for i in 0..d {
                v[i] = bf(src[row * d + i]);
            }
            let g = d / 64;
            let w = if part == 0 { we } else { wh };
            scale_store(
                &v,
                inv_rms(sum_squares::<NT>(&v), d, eps),
                w,
                &mut out[row * 2 * d + part * d..row * 2 * d + part * d + d],
                &mut xs[row * 2 * g + part * g..row * 2 * g + part * g + g],
            );
        }
    }
}

/// tf_nemo_group_rmsnorm: per group of gs, n = bf16(v * inv),
/// out = bf16(w * n), and the 64-group sums. Upstream: `groupRmsnorm`
/// (glue_ref.zig:111).
pub fn group_rmsnorm(
    x: &[u16],
    w: &[u16],
    out: &mut [u16],
    xs: &mut [f32],
    rows: usize,
    xd: usize,
    groups: usize,
    eps: f32,
) {
    let gs = xd / groups;
    let mut v = vec![0.0f32; gs];
    for row in 0..rows {
        for grp in 0..groups {
            let at = row * xd + grp * gs;
            for i in 0..gs {
                v[i] = bf(x[at + i]);
            }
            let inv = inv_rms(sum_squares::<128>(&v), gs, eps);
            for i in 0..gs {
                out[at + i] = tobf(bf(w[grp * gs + i]) * m::rbf(v[i] * inv));
            }
            for g in 0..gs / 64 {
                let mut acc: f32 = 0.0;
                for o in &out[at + g * 64..at + g * 64 + 64] {
                    acc += bf(*o);
                }
                xs[at / 64 + g] = acc;
            }
        }
    }
}

/// tf_nemo_router: PART[s][row][e], one fma chain over the slice's
/// inputs in order. Upstream: `router` (glue_ref.zig:129).
pub fn router(x: &[u16], w: &[u16], part: &mut [f32], rows: usize, d: usize, e: usize, sk: usize) {
    let per = d / sk;
    for s in 0..sk {
        for r in 0..rows {
            for ex in 0..e {
                let mut acc: f32 = 0.0;
                for k in s * per..(s + 1) * per {
                    acc = m::fma(bf(x[r * d + k]), bf(w[ex * d + k]), acc);
                }
                part[(s * rows + r) * e + ex] = acc;
            }
        }
    }
}

/// Sum slices in order, rank sigmoid scores plus bias (lowest id on
/// ties), then add shared slots. Upstream: `topk` (glue_ref.zig:139).
pub fn topk(
    part: &[f32],
    bias: &[f32],
    idx: &mut [i32],
    wt: &mut [f32],
    rows: usize,
    scaling: f32,
    e: usize,
    sk: usize,
    top_k: usize,
    ns: usize,
    norm: bool,
) {
    let mut score = [0.0f32; 256];
    let mut sel = [0.0f32; 256];
    for r in 0..rows {
        for ex in 0..e {
            let mut logit: f32 = 0.0;
            for s in 0..sk {
                logit += part[(s * rows + r) * e + ex];
            }
            score[ex] = m::sigmoid(logit);
            let v = score[ex] + bias[ex];
            sel[ex] = if v.is_nan() { f32::NEG_INFINITY } else { v };
        }
        let mut total: f32 = 0.0;
        let mut probs = [0.0f32; 8];
        let mut ids = [0i32; 8];
        for k in 0..top_k {
            let mut bi: usize = 0;
            for ex in 1..e {
                if sel[ex] > sel[bi] {
                    bi = ex;
                }
            }
            probs[k] = score[bi];
            ids[k] = bi as i32;
            total += score[bi];
            sel[bi] = f32::NEG_INFINITY;
        }
        for k in 0..ns {
            idx[r * ns + k] = if k < top_k {
                ids[k]
            } else {
                (e + k - top_k) as i32
            };
            wt[r * ns + k] = if k >= top_k {
                1.0
            } else if norm {
                (probs[k] / (total + 1e-20)) * scaling
            } else {
                probs[k] * scaling
            };
        }
    }
}

/// The conv taps in fma order, rounded to bf16, through SiLU, rounded
/// again. Upstream: `conv4` (glue_ref.zig:171).
fn conv4(bias: f32, w: [f32; 4], t0: f32, t1: f32, t2: f32, cur: f32) -> u16 {
    let mut acc = bias;
    acc = m::fma(w[0], t0, acc);
    acc = m::fma(w[1], t1, acc);
    acc = m::fma(w[2], t2, acc);
    acc = m::fma(w[3], cur, acc);
    tobf(m::silu(m::rbf(acc)))
}

/// The Mamba layer's shapes as the kernels take them (Tri.Shape's
/// fields). Upstream: `Shape` (glue_ref.zig:181).
#[derive(Clone, Copy)]
pub struct Shape {
    pub proj: usize,
    pub xd: usize,
    pub cd: usize,
    pub heads: usize,
    pub dh: usize,
    pub groups: usize,
    pub rmax: usize,
}

/// tf_nemo_conv: kept rows of RAW[1 - parity] replay, then the window's
/// rows; BASE commits after the last kept row. Upstream: `conv`
/// (glue_ref.zig:184).
pub fn conv(
    p: &[u16],
    base: &mut [u16],
    raw: &mut [u16],
    xc: &mut [u16],
    cw: &[f32],
    cb: &[f32],
    parity: usize,
    pk: usize,
    rows: usize,
    s: Shape,
) {
    let cd = s.cd;
    for ch in 0..cd {
        let mut t0 = bf(base[ch]);
        let mut t1 = bf(base[cd + ch]);
        let mut t2 = bf(base[2 * cd + ch]);
        let w = [cw[ch], cw[cd + ch], cw[2 * cd + ch], cw[3 * cd + ch]];
        for i in 0..pk + rows {
            let prev = i < pk;
            let cur = if prev {
                bf(raw[((1 - parity) * s.rmax + i) * cd + ch])
            } else {
                bf(p[(i - pk) * s.proj + s.xd + ch])
            };
            if !prev {
                let at = (parity * s.rmax + (i - pk)) * cd + ch;
                xc[at] = conv4(cb[ch], w, t0, t1, t2, cur);
                raw[at] = tobf(cur);
            }
            t0 = t1;
            t1 = t2;
            t2 = cur;
            if i + 1 == pk {
                base[ch] = tobf(t0);
                base[cd + ch] = tobf(t1);
                base[2 * cd + ch] = tobf(t2);
            }
        }
    }
}

/// tf_nemo_conv_rows then tf_nemo_conv_commit: a chunk's rows from its
/// own inputs or BASE, then BASE's new last three. Upstream:
/// `convRows` (glue_ref.zig:212).
pub fn conv_rows(
    p: &[u16],
    base: &mut [u16],
    xc: &mut [u16],
    cw: &[f32],
    cb: &[f32],
    rows: usize,
    s: Shape,
) {
    let cd = s.cd;
    for r in 0..rows {
        for ch in 0..cd {
            let mut t = [0.0f32; 4];
            for (j, tj) in t.iter_mut().enumerate() {
                let src = r as isize - 3 + j as isize;
                *tj = if src >= 0 {
                    bf(p[src as usize * s.proj + s.xd + ch])
                } else {
                    bf(base[(src + 3) as usize * cd + ch])
                };
            }
            xc[r * cd + ch] = conv4(
                cb[ch],
                [cw[ch], cw[cd + ch], cw[2 * cd + ch], cw[3 * cd + ch]],
                t[0],
                t[1],
                t[2],
                t[3],
            );
        }
    }
    for ch in 0..cd {
        let mut v = [0u16; 3];
        for (j, vj) in v.iter_mut().enumerate() {
            let src = rows as isize - 3 + j as isize;
            *vj = if src >= 0 {
                p[src as usize * s.proj + s.xd + ch]
            } else {
                base[(rows + j) * cd + ch]
            };
        }
        for j in 0..3 {
            base[j * cd + ch] = v[j];
        }
    }
}

/// tf_nemo_scan: per head and value row, the state replays the kept
/// rows, then the window's; y as the kernel sums it. Upstream: `scan`
/// (glue_ref.zig:233).
#[allow(unused_assignments)] // upstream's own `var dt_row: f32 = 0` (glue_ref.zig:238)
#[allow(clippy::needless_range_loop)] // upstream's own loops over the state's four lanes and tiles
pub fn scan(
    p: &[u16],
    xc: &[u16],
    dt: &mut [f32],
    state: &mut [f32],
    a: &[f32],
    dsk: &[f32],
    dtb: &[f32],
    y: &mut [u16],
    parity: usize,
    pk: usize,
    rows: usize,
    lo: f32,
    hi: f32,
    s: Shape,
) {
    const DS: usize = 128;
    for h in 0..s.heads {
        for d in 0..s.dh {
            let g = h / (s.heads / s.groups);
            let mut st = [0.0f32; DS];
            st.copy_from_slice(&state[(h * s.dh + d) * DS..(h * s.dh + d) * DS + DS]);
            let mut dt_row: f32 = 0.0;
            for i in 0..pk + rows {
                let prev = i < pk;
                let buf = if prev { 1 - parity } else { parity };
                let row = if prev { i } else { i - pk };
                let base = (buf * s.rmax + row) * s.cd;
                let dt_at = (buf * s.rmax + row) * s.heads + h;
                if prev {
                    dt_row = dt[dt_at];
                } else {
                    let v = bf(p[row * s.proj + s.xd + s.cd + h]) + dtb[h];
                    dt_row = m::softplus(v).min(hi).max(lo);
                    if d == 0 {
                        dt[dt_at] = dt_row;
                    }
                }
                let x = bf(xc[base + h * s.dh + d]);
                let da = m::exp(a[h] * dt_row);
                let xdt = x * dt_row;
                let mut outs = [0.0f32; 4];
                for (q, outq) in outs.iter_mut().enumerate() {
                    let mut acc = [0.0f32; 4];
                    for j in 0..8 {
                        for c in 0..4 {
                            let n = 4 * (4 * j + q) + c;
                            st[n] = m::fma(xdt, bf(xc[base + s.xd + g * DS + n]), st[n] * da);
                            acc[c] = m::fma(
                                st[n],
                                bf(xc[base + s.xd + s.groups * DS + g * DS + n]),
                                acc[c],
                            );
                        }
                    }
                    *outq = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                }
                if i + 1 == pk {
                    state[(h * s.dh + d) * DS..(h * s.dh + d) * DS + DS].copy_from_slice(&st);
                }
                if prev {
                    continue;
                }
                let out = (outs[0] + outs[1]) + (outs[2] + outs[3]);
                let yv = m::rbf(m::fma(x, dsk[h], out));
                let gz = m::rbf(m::silu(bf(p[row * s.proj + h * s.dh + d])));
                y[row * s.xd + h * s.dh + d] = tobf(gz * yv);
            }
        }
    }
}

/// The attention layer's shapes as the kernels take them (Tri.Attn's
/// fields). Upstream: `Attn` (glue_ref.zig:274).
#[derive(Clone, Copy)]
pub struct Attn {
    pub nqkv: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub nch: usize,
}

const TILE: usize = 64;
const CHUNK: usize = 512;

/// tf_nemo_attn_chunk then tf_nemo_attn_merge for rows at positions
/// pos.., caches already written through them. Upstream: `attention`
/// (glue_ref.zig:280).
#[allow(clippy::needless_range_loop)] // upstream's own loops over the tile's lanes
pub fn attention(
    qkv: &[u16],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    rows: usize,
    at: Attn,
    po: &mut [f32],
    pm: &mut [f32],
    pl: &mut [f32],
    out: &mut [u16],
    xs: &mut [f32],
) {
    const HD: usize = 128;
    let grp = at.heads / at.kv_heads;
    let scale: f32 = (at.dim as f64).powf(-0.5) as f32;
    let inf = f32::INFINITY;
    for r in 0..rows {
        for hk in 0..at.kv_heads {
            for g in 0..grp {
                let head = hk * grp + g;
                let limit = pos + r + 1;
                let nch = limit.div_ceil(CHUNK);
                for c in 0..nch {
                    let mut mx = -inf;
                    let mut den: f32 = 0.0;
                    let mut o = [0.0f32; HD];
                    let mut key0 = c * CHUNK;
                    while key0 < (c + 1) * CHUNK && key0 < limit {
                        let mut sc = [0.0f32; TILE];
                        let mut tile_m = -inf;
                        for (j, scj) in sc.iter_mut().enumerate() {
                            let key = key0 + j;
                            if key >= limit {
                                *scj = -inf;
                                continue;
                            }
                            let mut acc: f32 = 0.0;
                            for d in 0..HD {
                                acc = m::fma(
                                    bf(qkv[r * at.nqkv + head * HD + d]),
                                    bf(kc[(key * at.kv_heads + hk) * HD + d]),
                                    acc,
                                );
                            }
                            *scj = acc * scale;
                            tile_m = tile_m.max(*scj);
                        }
                        let active = tile_m != -inf;
                        let next = if active { mx.max(tile_m) } else { mx };
                        let alpha: f32 = if !active {
                            1.0
                        } else if mx == -inf {
                            0.0
                        } else {
                            m::exp(mx - next)
                        };
                        let mut p = [0.0f32; TILE];
                        for (j, pj) in p.iter_mut().enumerate() {
                            *pj = if active && key0 + j < limit {
                                m::exp(sc[j] - next)
                            } else {
                                0.0
                            };
                        }
                        let mut local = [0.0f32; 16];
                        for (jj, lj) in local.iter_mut().enumerate() {
                            let mut acc: f32 = 0.0;
                            for u in 0..4 {
                                acc += p[jj + 16 * u];
                            }
                            *lj = acc;
                        }
                        den = m::fma(den, alpha, butterfly::<f32, 16>(local));
                        mx = next;
                        for d in 0..HD {
                            let mut pv: f32 = 0.0;
                            for j in 0..TILE {
                                let key = key0 + j;
                                let v: f32 = if key < limit {
                                    bf(vc[(key * at.kv_heads + hk) * HD + d])
                                } else {
                                    0.0
                                };
                                pv = m::fma(p[j], v, pv);
                            }
                            o[d] = m::fma(o[d], alpha, pv);
                        }
                        key0 += TILE;
                    }
                    let base = (r * at.nch + c) * at.heads + head;
                    po[base * HD..base * HD + HD].copy_from_slice(&o);
                    pm[base] = mx;
                    pl[base] = den;
                }
                let mut mx = -inf;
                let mut den: f32 = 0.0;
                let mut o = [0.0f32; HD];
                for c in 0..nch {
                    let base = (r * at.nch + c) * at.heads + head;
                    let active = pl[base] > 0.0;
                    let next = if active { mx.max(pm[base]) } else { mx };
                    let a: f32 = if !active {
                        1.0
                    } else if mx == -inf {
                        0.0
                    } else {
                        m::exp(mx - next)
                    };
                    let b: f32 = if active { m::exp(pm[base] - next) } else { 0.0 };
                    for d in 0..HD {
                        o[d] = m::fma(o[d], a, po[base * HD + d] * b);
                    }
                    den = m::fma(den, a, pl[base] * b);
                    mx = next;
                }
                for d in 0..HD {
                    out[(r * at.heads + head) * HD + d] = tobf(o[d] / den);
                }
                for gi in 0..HD / 64 {
                    let mut acc: f32 = 0.0;
                    for j in 0..64 {
                        acc += bf(out[(r * at.heads + head) * HD + gi * 64 + j]);
                    }
                    xs[r * (at.heads * HD / 64) + head * (HD / 64) + gi] = acc;
                }
            }
        }
    }
}

/// tf_nemo_keyed_greedy: rank 0 by (value desc, id asc) and, given
/// `prob`, 1 / sum over the top k of e^(v - v0) in f64. Upstream:
/// `keyedGreedy` (glue_ref.zig:359).
pub fn keyed_greedy(
    vals: &[f32],
    ids: &[i64],
    out: &mut [i32],
    mut prob: Option<&mut [f32]>,
    rows: usize,
    count: usize,
    k: usize,
) {
    for r in 0..rows {
        let mut v = [f64::NEG_INFINITY; 32];
        let mut id = [1i64 << 40; 32];
        for i in 0..count {
            v[i] = vals[r * count + i] as f64;
            id[i] = ids[r * count + i];
        }
        let mut e = [0.0f64; 32];
        let mut top = f64::NEG_INFINITY;
        let mut rank = [0usize; 32];
        for i in 0..32 {
            rank[i] = 0;
            for j in 0..count {
                if v[j] > v[i] || (v[j] == v[i] && id[j] < id[i]) {
                    rank[i] += 1;
                }
            }
            if i < count && rank[i] == 0 {
                out[r] = id[i] as i32;
            }
            if i < count && rank[i] < k {
                top = top.max(v[i]);
            }
        }
        if let Some(p) = prob.as_mut() {
            for i in 0..count {
                e[i] = if rank[i] < k { (v[i] - top).exp() } else { 0.0 };
            }
            p[r] = (1.0 / butterfly::<f64, 32>(e)) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `one key: every head's output is that key's value,
    /// and the partials say one chunk of weight 1` (glue_ref.zig:384).
    #[test]
    fn one_key_every_heads_output_is_that_keys_value_and_the_partials_say_one_chunk_of_weight_1() {
        let a: Attn = Attn {
            nqkv: 32 * 128 + 2 * 2 * 128,
            heads: 32,
            kv_heads: 2,
            dim: 128,
            nch: 1,
        };
        let qkv: Vec<u16> = (0..a.nqkv)
            .map(|i| tobf((i % 7) as f32 - 3.0))
            .collect();
        let kv = 2 * 128;
        let mut kc = [0u16; 2 * 128];
        let mut vc = [0u16; 2 * 128];
        for i in 0..kv {
            kc[i] = tobf((i % 5) as f32 * 0.25);
            vc[i] = tobf((i % 11) as f32 - 5.0);
        }
        let mut po = [0.0f32; 32 * 128];
        let mut pm = [0.0f32; 32];
        let mut pl = [0.0f32; 32];
        let mut out = [0u16; 32 * 128];
        let mut xs = [0.0f32; 64];
        attention(
            &qkv, &kc, &vc, 0, 1, a, &mut po, &mut pm, &mut pl, &mut out, &mut xs,
        );
        for h in 0..32 {
            assert_eq!(1.0f32, pl[h]);
            assert_eq!(&vc[(h / 16) * 128..(h / 16) * 128 + 128], &out[h * 128..h * 128 + 128]);
        }
    }

    /// Upstream's `top-k picks by score plus bias, the lowest id on a
    /// tie, then the shared slots` (glue_ref.zig:409).
    #[test]
    fn top_k_picks_by_score_plus_bias_the_lowest_id_on_a_tie_then_the_shared_slots() {
        let part: [f32; 8] = [0.0, 2.0, -1.0, 2.0, 0.5, -3.0, 1.0, 2.0];
        let bias: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let mut idx = [0i32; 4];
        let mut wt = [0.0f32; 4];
        topk(&part, &bias, &mut idx, &mut wt, 1, 2.0, 8, 1, 2, 4, false);
        assert_eq!([6, 1, 8, 9], idx);
        assert_eq!(m::sigmoid(1.0) * 2.0, wt[0]);
        assert_eq!(1.0f32, wt[3]);
    }
}
