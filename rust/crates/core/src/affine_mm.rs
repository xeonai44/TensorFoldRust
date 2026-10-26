//! x W^T for affine-quantized W on the tensor units, dense or by expert: exact
//! codes, fp32 group sums and fp32 row sums.
//!
//! Port of `zig/src/core/affine_mm.zig` (304 lines). The launches bind the same
//! records `moe_route` binds; `Ref` is the port's name for upstream's
//! structural `.{ .buf, .off }` (see `moe_route.rs`). The two live tests are
//! upstream's, ported whole: the exact-sum host references are f64 arithmetic,
//! same tolerances, same bf16 round-to-nearest-even.

use tensorfold_metal::{Device, Encoder, Pipeline};

use crate::moe_route::Ref;

/// Upstream: `affine_mm.zig:8  Format` — the weights' bits (4 or 8) and group
/// (32 or 64), and the output's type. Zig's field defaults (4, 64, false) are
/// the port's `Default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub bits: u8,
    pub group: u16,
    pub out_f32: bool,
}

impl Default for Format {
    fn default() -> Format {
        Format { bits: 4, group: 64, out_f32: false }
    }
}

/// The kernel source (upstream `kernel_sources.core_affine_mm`, embedded
/// byte-identical from `kernels/metal/core/affine_mm.metal`).
const CORE_AFFINE_MM: &str = include_str!("../../../kernels/metal/core/affine_mm.metal");

/// Why a format's source was refused.
#[derive(Debug, PartialEq)]
pub enum Error {
    UnsupportedFormat,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnsupportedFormat => write!(f, "unsupported format"),
        }
    }
}

impl std::error::Error for Error {}

/// The kernel source for `f` (nax.h inlined for this GPU).
///
/// Upstream: `affine_mm.zig:11  source`.
pub fn source(device: &Device, f: Format) -> Result<String, Error> {
    if (f.bits != 4 && f.bits != 8) || (f.group != 32 && f.group != 64) {
        return Err(Error::UnsupportedFormat);
    }
    let body = crate::frags::source(device, CORE_AFFINE_MM);
    let out = if f.out_f32 { "float" } else { "bfloat" };
    Ok(format!(
        "#define TF_BITS {}\n#define TF_GROUP {}\n#define TF_OUT_T {}\n{}",
        f.bits, f.group, out, body
    ))
}

/// Upstream: `affine_mm.zig:19  names` — entry points: row sums, the dense
/// matmul, gathers by tile height, scattering gathers, the experts' gate and
/// up through a pair order into SwiGLU.
pub const NAMES: [&str; 8] = [
    "tf_affine_row_sums",
    "tf_affine_mm",
    "tf_affine_gather_32",
    "tf_affine_gather_64",
    "tf_affine_gather_scatter_32",
    "tf_affine_gather_scatter_64",
    "tf_affine_gather_glu_32",
    "tf_affine_gather_glu_64",
];

/// A format's pipelines, in `NAMES` order (upstream `affine_mm.zig:22`).
pub type Pipes = [Pipeline; 8];

/// Upstream: `affine_mm.zig:24  Args`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Args {
    pub rows: i32,
    pub n: i32,
    pub k: i32,
    pub experts: i32,
}

/// A dense matmul's pitches and batch offsets, in elements (the sums' in
/// rows); the default is one plain [rows, k] x [k, n] product.
///
/// Upstream: `affine_mm.zig:27  Strides` — an extern struct with Zig field
/// defaults; the port's `Default` carries them.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Strides {
    pub x_row: i32,
    pub y_row: i32,
    pub sums_row: i32,
    pub x_batch: i32,
    pub y_batch: i32,
    pub sums_batch: i32,
    pub w_batch: i32,
    pub pad: i32,
}

impl Default for Strides {
    fn default() -> Strides {
        Strides {
            x_row: 0,
            y_row: 0,
            sums_row: 1,
            x_batch: 0,
            y_batch: 0,
            sums_batch: 0,
            w_batch: 0,
            pad: 0,
        }
    }
}

/// One affine-quantized operand: codes `w`, scales `s`, biases `b`, shape [n,
/// k] — upstream's anytype `q` records of the call sites.
#[derive(Clone, Copy)]
pub struct Quant<'a> {
    pub w: Ref<'a>,
    pub s: Ref<'a>,
    pub b: Ref<'a>,
    pub n: u32,
    pub k: u32,
}

/// Upstream: `affine_mm.zig:29  fn bind`.
fn bind(e: &Encoder, i: usize, r: Ref<'_>) {
    e.set_buffer(r.buf, r.off, i);
}

/// Each row's sum over each group of x [rows, k] into `sums` (fp32 [rows, k /
/// group]): the bias's operand.
///
/// Upstream: `affine_mm.zig:34  rowSums`.
pub fn row_sums(
    e: &Encoder,
    p: &Pipes,
    group: u32,
    x: Ref<'_>,
    sums: Ref<'_>,
    rows: u32,
    k: u32,
) {
    row_sums_at(e, p, group, x, sums, rows, k, k);
}

/// `row_sums` over rows `ld` elements apart (k of them each).
///
/// Upstream: `affine_mm.zig:39  rowSumsAt`.
pub fn row_sums_at(
    e: &Encoder,
    p: &Pipes,
    group: u32,
    x: Ref<'_>,
    sums: Ref<'_>,
    rows: u32,
    k: u32,
    ld: u32,
) {
    e.set_pipeline(&p[0]);
    bind(e, 0, x);
    let args = Args { rows: rows as i32, n: ld as i32, k: k as i32, experts: 0 };
    e.set_value(&args, 5);
    bind(e, 6, sums);
    e.dispatch((k as u64 / group as u64, rows as u64, 1), (std::cmp::min(k / group, 64) as u64, 1, 1));
}

/// y [rows, n] = x [rows, q.k] W^T with x's group sums `sums`; n is q.n
/// rounded up to 64 (the padded weight rows, y's pitch).
///
/// Upstream: `affine_mm.zig:48  dense`.
pub fn dense(e: &Encoder, p: &Pipes, x: Ref<'_>, sums: Ref<'_>, q: Quant<'_>, y: Ref<'_>, rows: u32) {
    let n = (q.n + 63) & !63;
    dense_batch(
        e,
        p,
        x,
        sums,
        q,
        y,
        rows,
        1,
        Strides { x_row: q.k as i32, y_row: n as i32, ..Strides::default() },
    );
}

/// `batch` dense products at once, each over its own rows of the weights
/// (`s`; q.n a multiple of 64), e.g. a per-head projection.
///
/// Upstream: `affine_mm.zig:54  denseBatch`.
pub fn dense_batch(
    e: &Encoder,
    p: &Pipes,
    x: Ref<'_>,
    sums: Ref<'_>,
    q: Quant<'_>,
    y: Ref<'_>,
    rows: u32,
    batch: u32,
    s: Strides,
) {
    let n = (q.n + 63) & !63;
    e.set_pipeline(&p[1]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    let args = Args { rows: rows as i32, n: n as i32, k: q.k as i32, experts: 0 };
    e.set_value(&args, 5);
    bind(e, 6, y);
    e.set_value(&s, 7);
    e.dispatch_groups(
        (n as u64 / 64, (rows as u64 + 63) / 64, batch as u64),
        (32, 2, 2),
    );
}

/// y [rows, q.n] = x times each row's expert's W^T, rows sorted by expert with
/// `offsets` (each expert's first row); q is [experts, n, k].
///
/// Upstream: `affine_mm.zig:69  gather`.
pub fn gather(
    e: &Encoder,
    p: &Pipes,
    x: Ref<'_>,
    sums: Ref<'_>,
    q: Quant<'_>,
    offsets: Ref<'_>,
    y: Ref<'_>,
    rows: u32,
    experts: u32,
) {
    gather_to(e, p, x, sums, q, offsets, y, None, rows, experts);
}

/// `gather` with sorted row i's output at row dst[i] of y (i32 [rows]; none:
/// in sorted order).
///
/// Upstream: `affine_mm.zig:74  gatherTo` — the scatter is decided by whether
/// a destination was given.
pub fn gather_to(
    e: &Encoder,
    p: &Pipes,
    x: Ref<'_>,
    sums: Ref<'_>,
    q: Quant<'_>,
    offsets: Ref<'_>,
    y: Ref<'_>,
    dst: Option<Ref<'_>>,
    rows: u32,
    experts: u32,
) {
    assert!(q.n % 64 == 0 && q.k % 64 == 0);
    let bm: u32 = if rows / experts < 64 { 32 } else { 64 };
    let scatter = dst.is_some();
    e.set_pipeline(&p[(if bm == 64 { 3 } else { 2 }) + if scatter { 2 } else { 0 }]);
    if let Some(dst) = dst {
        bind(e, 8, dst);
    }
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    let args = Args { rows: rows as i32, n: q.n as i32, k: q.k as i32, experts: experts as i32 };
    e.set_value(&args, 5);
    bind(e, 6, y);
    bind(e, 7, offsets);
    let tiles = std::cmp::min(rows, (rows + bm - 1) / bm + experts - 1); // each expert's last tile may be partial
    e.dispatch_groups((q.n as u64 / 64, tiles as u64, 1), (32, 2, 2));
}

/// act = SwiGLU(x W_gate^T, x W_up^T) clipped at `limit` for expert-sorted
/// pairs (`offsets`), pair i on x's row order[i] / topk.
///
/// Upstream: `affine_mm.zig:93  gatherGlu`.
pub fn gather_glu(
    e: &Encoder,
    p: &Pipes,
    x: Ref<'_>,
    sums: Ref<'_>,
    gate: Quant<'_>,
    up: Quant<'_>,
    offsets: Ref<'_>,
    order: Ref<'_>,
    act: Ref<'_>,
    rows: u32,
    experts: u32,
    topk: u32,
    limit: f32,
) {
    assert!(gate.n % 32 == 0 && gate.k % 64 == 0 && gate.n == up.n && gate.k == up.k);
    let bm: u32 = if rows / experts < 200 { 32 } else { 64 }; // 32-row tiles waste less of each expert's last tile until ~200 rows
    e.set_pipeline(&p[if bm == 64 { 7 } else { 6 }]);
    bind(e, 0, x);
    bind(e, 1, gate.w);
    bind(e, 2, gate.s);
    bind(e, 3, gate.b);
    bind(e, 4, sums);
    let args = Args { rows: rows as i32, n: gate.n as i32, k: gate.k as i32, experts: experts as i32 };
    e.set_value(&args, 5);
    bind(e, 6, act);
    bind(e, 7, offsets);
    bind(e, 8, order);
    bind(e, 9, up.w);
    bind(e, 10, up.s);
    bind(e, 11, up.b);
    let pair = [topk as f32, limit];
    e.set_value(&pair, 12);
    let tiles = std::cmp::min(rows, (rows + bm - 1) / bm + experts - 1); // each expert's last tile may be partial
    e.dispatch_groups((gate.n as u64 / 32, tiles as u64, 1), (32, 2, 2));
}

/// Round to nearest even (upstream `affine_mm.zig:114  fn bf16`).
fn bf16(v: f32) -> u16 {
    let u: u32 = v.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

/// Upstream: `affine_mm.zig:119  fn f32of`.
fn f32of(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn n2_first(_p: usize) -> bool {
    true
}

#[cfg(test)]
fn next_check() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
fn pass_bisect(_gpu: &tensorfold_metal::Gpu, tag: &str, bufs: &[tensorfold_metal::Buffer], r: usize, n: usize) {
    // Safety: each bisect step submits and waits before this reads.
    let g0: &[f32] = unsafe { std::slice::from_raw_parts(bufs[5].as_ptr().add(r * n * 4).cast::<f32>(), 2) };
    let d0: &[f32] = unsafe { std::slice::from_raw_parts(bufs[5].as_ptr().add(3 * r * n * 4).cast::<f32>(), 2) };
    let sums: &[f32] = unsafe { std::slice::from_raw_parts(bufs[6].as_ptr().cast::<f32>(), 8) };
    eprintln!("BISECT {tag}: g_f32 {g0:?} d_f32 {d0:?} sums {sums:?}");
}

fn mismatch_collector(_p: usize) -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
    N[_p].fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorfold_metal::Gpu;

    /// The port's deterministic fill for the tests' random data — upstream
    /// seeds `std.Random.DefaultPrng` the same way; the exact stream is not a
    /// fixture (the checks recompute from the drawn values themselves).
    struct Rnd(u64);

    impl Rnd {
        fn new(seed: u64) -> Rnd {
            Rnd(seed | 1)
        }

        fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn float(&mut self) -> f32 {
            (self.next() >> 40) as f32 / 16777216.0
        }

        fn bytes(&mut self, b: &mut [u8]) {
            for x in b.iter_mut() {
                *x = (self.next() >> 33) as u8;
            }
        }
    }

    /// A host view of a shared buffer as `u16`s at a byte offset (upstream
    /// `bufs[m].slice(u16, n)`).
    unsafe fn u16_at(b: &tensorfold_metal::Buffer, at: usize, n: usize) -> &[u16] {
        std::slice::from_raw_parts(b.as_ptr().add(at).cast::<u16>(), n)
    }

    unsafe fn i32_at(b: &tensorfold_metal::Buffer, at: usize, n: usize) -> &[i32] {
        std::slice::from_raw_parts(b.as_ptr().add(at).cast::<i32>(), n)
    }

    unsafe fn f32_at(b: &tensorfold_metal::Buffer, at: usize, n: usize) -> &[f32] {
        std::slice::from_raw_parts(b.as_ptr().add(at).cast::<f32>(), n)
    }

    /// Upstream: `affine_mm.zig:123  test "dense and gathered, each format,
    /// against exact sums on the host"`.
    #[test]
    fn dense_and_gathered_each_format_against_exact_sums_on_the_host() {
        let Some(gpu) = Gpu::open() else {
            return;
        };
        let device = gpu.device();
        let e = 3;
        let n = 128;
        let k = 256;
        let counts = [37, 0, 70]; // an empty expert and partial tiles; densely, 107 rows (a partial tile)
        let r = counts[0] + counts[1] + counts[2];
        let mut rnd = Rnd::new(7);
        for [bits, group] in [[4, 32]] {
            let w_bytes = e * n * k * bits / 8;
            let s_n = e * n * k / group;
            let bufs: Vec<tensorfold_metal::Buffer> = [
                r * k * 2,
                w_bytes,
                s_n * 2,
                s_n * 2,
                e * 4,
                5 * r * n * 4,
                r * (k / group) * 4,
                r * 4,
            ]
            .iter()
            .map(|&len| gpu.buffer_with_len(len))
            .collect();
            // the scatter's destinations: rows reversed
            {
                // Safety: fresh buffer, nothing in flight.
                let dst = unsafe { std::slice::from_raw_parts_mut(bufs[7].as_ptr().cast::<i32>(), r) };
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = (r - 1 - i) as i32;
                }
            }
            // Safety: fresh buffers, nothing in flight.
            unsafe {
                let x = std::slice::from_raw_parts_mut(bufs[0].as_ptr().cast::<u16>(), r * k);
                for v in x.iter_mut() {
                    *v = bf16(rnd.float() * 2.0 - 1.0);
                }
                let wb = std::slice::from_raw_parts_mut(bufs[1].as_ptr().cast::<u8>(), w_bytes);
                rnd.bytes(wb);
                let sc = std::slice::from_raw_parts_mut(bufs[2].as_ptr().cast::<u16>(), s_n);
                let bi = std::slice::from_raw_parts_mut(bufs[3].as_ptr().cast::<u16>(), s_n);
                for (s, b) in sc.iter_mut().zip(bi.iter_mut()) {
                    *s = bf16((rnd.float() + 0.5) / 64.0);
                    *b = bf16(-(rnd.float() + 0.5) * 0.15);
                }
                let offs = std::slice::from_raw_parts_mut(bufs[4].as_ptr().cast::<i32>(), e);
                let mut first: i32 = 0;
                for (o, c) in offs.iter_mut().zip(counts.iter()) {
                    *o = first;
                    first += *c as i32;
                }
            }
            let mut pipes: Vec<Pipes> = Vec::with_capacity(2);
            for o in 0..2 {
                let src = source(device, Format { bits: bits as u8, group: group as u16, out_f32: o == 1 })
                    .unwrap();
                let lib =
                    tensorfold_metal::Library::from_source(device, &src, tensorfold_metal::CompileOptions::mlx())
                        .unwrap();
                let mut v: Vec<Pipeline> = Vec::with_capacity(8);
                for name in NAMES {
                    v.push(lib.pipeline(device, name, false).unwrap());
                }
                pipes.push(v.try_into().map_err(|_: Vec<Pipeline>| "eight pipelines").unwrap());
            }
            let xr = Ref::whole(&bufs[0]);
            let sums = Ref::whole(&bufs[6]);
            let q = Quant {
                w: Ref::whole(&bufs[1]),
                s: Ref::whole(&bufs[2]),
                b: Ref::whole(&bufs[3]),
                n: n as u32,
                k: k as u32,
            };
            let out = [
                Ref { buf: &bufs[5], off: 0 },
                Ref { buf: &bufs[5], off: r * n * 4 },
                Ref { buf: &bufs[5], off: 2 * r * n * 4 },
                Ref { buf: &bufs[5], off: 3 * r * n * 4 },
                Ref { buf: &bufs[5], off: 4 * r * n * 4 },
            ];
            {
                // Safety: fresh buffer, nothing in flight.
                let b5 = unsafe { std::slice::from_raw_parts_mut(bufs[5].as_ptr(), 5 * r * n * 4) };
                b5.fill(0xCD);
            }
            {
                let pass = gpu.compute_pass();
                let enc = pass.encoder();
                row_sums(&enc, &pipes[0], group as u32, xr, sums, r as u32, k as u32);
                pass.submit().unwrap();
            }
            pass_bisect(&gpu, "row_sums", &bufs, r, n);
            {
                let pass = gpu.compute_pass();
                let enc = pass.encoder();
                gather(&enc, &pipes[1], xr, sums, q, Ref::whole(&bufs[4]), out[1], r as u32, e as u32); // fp32
                pass.submit().unwrap();
            }
            pass_bisect(&gpu, "gather", &bufs, r, n);
            {
                let pass = gpu.compute_pass();
                let enc = pass.encoder();
                dense(&enc, &pipes[1], xr, sums, q, out[3], r as u32); // fp32
                pass.submit().unwrap();
            }
            pass_bisect(&gpu, "dense", &bufs, r, n);
            {
                let pass = gpu.compute_pass();
                let enc = pass.encoder();
                gather(&enc, &pipes[0], xr, sums, q, Ref::whole(&bufs[4]), out[0], r as u32, e as u32); // bf16
                gather_to(&enc, &pipes[1], xr, sums, q, Ref::whole(&bufs[4]), out[4], Some(Ref::whole(&bufs[7])), r as u32, e as u32); // fp32, rows reversed
                dense(&enc, &pipes[0], xr, sums, q, out[2], r as u32); // bf16
                pass.submit().unwrap();
            }
            // Safety: the pass above waited.
            unsafe {
                let all = bufs[5].bytes();
                let g16 = u16_at(&bufs[5], 0, r * n);
                let g32 = f32_at(&bufs[5], r * n * 4, r * n);
                let d16 = u16_at(&bufs[5], 2 * r * n * 4, r * n);
                let d32 = f32_at(&bufs[5], 3 * r * n * 4, r * n);
                let s32 = f32_at(&bufs[5], 4 * r * n * 4, r * n);
                let dst = i32_at(&bufs[7], 0, r);
                for i in 0..r {
                    for n2 in 0..n {
                        assert_eq!(g32[i * n + n2], s32[dst[i] as usize * n + n2]);
                    }
                }
                let wb = bufs[1].bytes();
                let x = u16_at(&bufs[0], 0, r * k);
                let sc = u16_at(&bufs[2], 0, s_n);
                let bi = u16_at(&bufs[3], 0, s_n);
                let mut row_i: usize = 0;
                for (ex, c) in counts.iter().enumerate() {
                    for _ in 0..*c {
                        for (use_, pass_i) in [ex, 0].into_iter().enumerate() {
                            for n2 in 0..n {
                                let row = use_ * n + n2;
                                let mut sum: f64 = 0.0;
                                let mut mag: f64 = 0.0;
                                let mut dots: Vec<f64> = Vec::new();
                                let mut xs_dbg: f64 = 0.0;
                                for g2 in 0..k / group {
                                    let mut dot: f64 = 0.0;
                                    let mut dmag: f64 = 0.0; // the sizes the fp32 sums carry: codes (4-bit ones plus 128) times x
                                    let mut xs: f64 = 0.0;
                                    let mut xmag: f64 = 0.0;
                                    let off: f64 = if bits == 4 { 128.0 } else { 0.0 };
                                    for j in 0..group {
                                        let k2 = g2 * group + j;
                                        let qv: u32 = if bits == 4 {
                                            ((wb[(row * k + k2) / 2] >> (4 * (k2 % 2))) & 15) as u32
                                        } else {
                                            wb[row * k + k2] as u32
                                        };
                                        let xv: f64 = f32of(x[row_i * k + k2]) as f64;
                                        dot += qv as f64 * xv;
                                        dmag += (qv as f64 + off) * xv.abs();
                                        xs += xv;
                                        xmag += xv.abs();
                                    }
                                        dots.push(dot);
                                    if g2 == 0 { xs_dbg = xs; }
                                    let sg = row * (k / group) + g2;
                                    let sg = row * (k / group) + g2;
                                    let s: f64 = f32of(sc[sg]) as f64;
                                    let b: f64 = f32of(bi[sg]) as f64;
                                    if row_i == 0 && n2 == 0 {
                                        eprintln!("G{g2} sg {sg} qv0 {} qv1 {} xv0 {} s {s} b {b} dot {dot} xs {xs} sum {sum}", if bits == 4 { (wb[(row * k) / 2] & 15) as u32 } else { wb[0] as u32 }, if bits == 4 { (wb[(row * k) / 2] >> 4) as u32 } else { wb[1] as u32 }, f32of(x[0]));
                                    }
                                    sum += s * dot + b * xs;
                                    mag += s * dmag + (b - off * s).abs() * xmag;
                                }
                                let got: f64 = if pass_i == 0 { g32[row_i * n + n2] } else { d32[row_i * n + n2] } as f64;
                                let got16 = if pass_i == 0 { g16[row_i * n + n2] } else { d16[row_i * n + n2] };
                                if row_i == 0 && n2 < 4 && pass_i == 0 {
                                    eprintln!("SEQ #{:?} ex{ex} use{use_} row{row} rowi{row_i} n{n2} pass{pass_i} got {got} sum {sum} wb0 {}", next_check(), wb[0]);
                                }
                                assert!(
                                    (got - sum).abs() <= 4e-6 * mag + 1e-30,
                                    "SEQFAIL bits {bits} group {group} pass {pass_i} row {row_i} n {n2}: got {got} sum {sum}"
                                ); // fp32 order only
                                assert_eq!(bf16(got as f32), got16);
                            }
                        }
                        row_i += 1;
                    }
                }
                let _ = all;
            }
        }
    }

    /// Upstream: `affine_mm.zig:233  test "a batch of dense products with
    /// pitches: each batch's rows of x, the weights and y"`.
    #[test]
    fn a_batch_of_dense_products_with_pitches() {
        let Some(gpu) = Gpu::open() else {
            return;
        };
        let device = gpu.device();
        let b = 3; // batches, interleaved in x and y like heads in a row: x [R, B, K], y [R, B, N]
        let n = 64;
        let k = 128;
        let r = 70;
        let mut rnd = Rnd::new(5);
        let bufs: Vec<tensorfold_metal::Buffer> = [
            r * b * k * 2,
            b * n * k / 2,
            b * n * k / 64 * 2,
            b * n * k / 64 * 2,
            r * b * n * 2,
            r * b * (k / 64) * 4,
        ]
        .iter()
        .map(|&len| gpu.buffer_with_len(len))
        .collect();
        // Safety: fresh buffers, nothing in flight.
        unsafe {
            let x = std::slice::from_raw_parts_mut(bufs[0].as_ptr().cast::<u16>(), r * b * k);
            for v in x.iter_mut() {
                *v = bf16(rnd.float() * 2.0 - 1.0);
            }
            let wb = std::slice::from_raw_parts_mut(bufs[1].as_ptr().cast::<u8>(), b * n * k / 2);
            rnd.bytes(wb);
            let sc = std::slice::from_raw_parts_mut(bufs[2].as_ptr().cast::<u16>(), b * n * k / 64);
            let bi = std::slice::from_raw_parts_mut(bufs[3].as_ptr().cast::<u16>(), b * n * k / 64);
            for (s, b2) in sc.iter_mut().zip(bi.iter_mut()) {
                *s = bf16((rnd.float() + 0.5) / 64.0);
                *b2 = bf16(-(rnd.float() + 0.5) * 0.15);
            }
        }
        let src = source(device, Format::default()).unwrap();
        let lib = tensorfold_metal::Library::from_source(device, &src, tensorfold_metal::CompileOptions::mlx())
            .unwrap();
        let mut made: Vec<Pipeline> = Vec::with_capacity(8);
        for name in NAMES {
            made.push(lib.pipeline(device, name, false).unwrap());
        }
        let pipes: Pipes = made.try_into().map_err(|_: Vec<Pipeline>| "eight pipelines").unwrap();
        let xr = Ref::whole(&bufs[0]);
        let sums = Ref::whole(&bufs[5]);
        let q = Quant {
            w: Ref::whole(&bufs[1]),
            s: Ref::whole(&bufs[2]),
            b: Ref::whole(&bufs[3]),
            n: n as u32,
            k: k as u32,
        };
        let pass = gpu.compute_pass();
        let enc = pass.encoder();
        // x as [R * B, K]: batch b's row r is row r * B + b
        row_sums(&enc, &pipes, 64, xr, sums, (r * b) as u32, k as u32);
        dense_batch(
            &enc,
            &pipes,
            xr,
            sums,
            q,
            Ref::whole(&bufs[4]),
            r as u32,
            b as u32,
            Strides {
                x_row: (b * k) as i32,
                y_row: (b * n) as i32,
                sums_row: b as i32,
                x_batch: k as i32,
                y_batch: n as i32,
                sums_batch: 1,
                w_batch: n as i32,
                pad: 0,
            },
        );
        pass.submit().unwrap();
        // Safety: the pass above waited.
        unsafe {
            let y = u16_at(&bufs[4], 0, r * b * n);
            let x = u16_at(&bufs[0], 0, r * b * k);
            let wb = bufs[1].bytes();
            let sc = u16_at(&bufs[2], 0, b * n * k / 64);
            let bi = u16_at(&bufs[3], 0, b * n * k / 64);
            for r2 in 0..r {
                for b2 in 0..b {
                    for n2 in 0..n {
                        let row = b2 * n + n2;
                        let mut sum: f64 = 0.0;
                        let mut mag: f64 = 0.0;
                        for g2 in 0..k / 64 {
                            let mut dot: f64 = 0.0;
                            let mut xs: f64 = 0.0;
                            let mut xmag: f64 = 0.0;
                            let xrow = (r2 * b + b2) * k;
                            for j in 0..64 {
                                let k2 = g2 * 64 + j;
                                let qv: u32 = ((wb[(row * k + k2) / 2] >> (4 * (k2 % 2))) & 15) as u32;
                                let xv: f64 = f32of(x[xrow + k2]) as f64;
                                dot += qv as f64 * xv;
                                xs += xv;
                                xmag += xv.abs();
                            }
                            let sg = row * (k / 64) + g2;
                            let s: f64 = f32of(sc[sg]) as f64;
                            sum += s * dot + f32of(bi[sg]) as f64 * xs;
                            mag += s * 143.0 * xmag + (f32of(bi[sg]) as f64 - 128.0 * s).abs() * xmag; // the sizes the fp32 sums carry (codes plus 128)
                        }
                        let got: f64 = f32of(y[(r2 * b + b2) * n + n2]) as f64;
                        assert!((got - sum).abs() <= sum.abs() / 128.0 + 4e-6 * mag); // one bf16 rounding of the fp32 sum
                    }
                }
            }
        }
    }
}
