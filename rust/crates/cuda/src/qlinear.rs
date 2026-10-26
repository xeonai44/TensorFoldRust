//! Quantized projections behind one call: a weight view per format,
//! with decode rows and prompt rows chosen per format.
//!
//! Port of `zig/src/cuda/qlinear.zig` (256 lines).

use crate::driver::{Driver, Error};
use crate::fp8;
use crate::launch::{launch, Args, Config};
use crate::memory::DeviceBuffer;
use crate::module::{Function, Module};
use crate::stream::Stream;
use crate::Dim3;

/// MLX affine 4-bit tiled for qmm: packed words, (kg, npad) scales and
/// biases, n outputs from k inputs. Upstream: `Affine4` (qlinear.zig:10).
#[derive(Clone, Copy)]
pub struct Affine4 {
    pub w: u64,
    pub s: u64,
    pub b: u64,
    pub n: usize,
    pub k: usize,
    pub npad: usize,
}

/// One projection's weights in any format this module serves. Upstream:
/// `Weight` (qlinear.zig:16).
#[derive(Clone, Copy)]
pub enum Weight {
    Affine4(Affine4),
    Fp8g(fp8::Weight),
}

impl Weight {
    /// Upstream: `outputs` (qlinear.zig:23).
    pub fn outputs(w: Weight) -> usize {
        match w {
            Weight::Affine4(q) => q.n,
            Weight::Fp8g(q) => q.n as usize,
        }
    }

    /// Upstream: `inputs` (qlinear.zig:30).
    pub fn inputs(w: Weight) -> usize {
        match w {
            Weight::Affine4(q) => q.k,
            Weight::Fp8g(q) => q.k as usize,
        }
    }
}

/// Rows in: bf16 `x` rows `ldx` apart (0: packed), their 64-group sums,
/// and fp8g's slice partials (Lane.partBytes). Upstream: `Rows`
/// (qlinear.zig:37).
#[derive(Clone, Copy)]
pub struct Rows {
    pub x: u64,
    pub ldx: usize,
    pub sums: u64,
    pub part: u64,
}

/// The kernels of every format a family loaded; a weight whose format
/// is not loaded is refused. Upstream: `Linear` (qlinear.zig:42) — the
/// port's single lifetime is the borrow (Zig's is a copied pointer
/// pair; `Stream<'d>` covaries into it).
pub struct Linear<'a> {
    pub affine4: Option<&'a Affine4Kernels<'a>>,
    pub fp8g: Option<&'a fp8::Lane<'a>>,
}

impl Linear<'_> {
    /// Decode rows: out (rows, n) bf16, each row's bits the same at
    /// every row count the format takes. Upstream: `decode`
    /// (qlinear.zig:46).
    pub fn decode(
        &self,
        s: &Stream,
        w: Weight,
        in_: Rows,
        out: u64,
        rows: usize,
    ) -> Result<(), Error> {
        match w {
            Weight::Affine4(q) => {
                let a = self.affine4.ok_or(Error::FormatNotLoaded)?;
                if in_.ldx != 0 && in_.ldx != q.k {
                    return Err(Error::RowStride);
                }
                a.decode(s, in_.x, in_.sums, q, out, rows)
            }
            Weight::Fp8g(q) => {
                let lane = self.fp8g.ok_or(Error::FormatNotLoaded)?;
                lane.matmul(
                    s,
                    in_.x,
                    if in_.ldx == 0 { q.k as usize } else { in_.ldx },
                    rows,
                    q,
                    out,
                    in_.part,
                )
            }
        }
    }

    /// Prompt rows: out (rows, n) bf16 by the format's prompt GEMM.
    /// Upstream: `prompt` (qlinear.zig:57).
    #[allow(clippy::too_many_arguments)] // the kernel's parameters, upstream's signature
    pub fn prompt(
        &self,
        s: &Stream,
        w: Weight,
        in_: Rows,
        out: u64,
        rows: usize,
    ) -> Result<(), Error> {
        match w {
            Weight::Affine4(q) => {
                let a = self.affine4.ok_or(Error::FormatNotLoaded)?;
                if in_.ldx != 0 && in_.ldx != q.k {
                    return Err(Error::RowStride);
                }
                a.prompt(s, in_.x, q, out, rows)
            }
            Weight::Fp8g(q) => {
                let lane = self.fp8g.ok_or(Error::FormatNotLoaded)?;
                lane.matmul(
                    s,
                    in_.x,
                    if in_.ldx == 0 { q.k as usize } else { in_.ldx },
                    rows,
                    q,
                    out,
                    in_.part,
                )
            }
        }
    }
}

/// The affine-4 instantiations' names (cuobjdump -symbols of the
/// qmm_group, qmm_prefill and lane_gemv fatbins). Upstream: `symbols`
/// (qlinear.zig:68).
pub mod symbols {
    pub const GROUP: &str = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const PREFILL: &str = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const GEMV: &str = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    pub const GEMV_SPLIT: &str = "tf_lane_gemv_split";
}

/// qmm_group.cu's Part and Parts, passed by value: four projections at
/// most, one used here. Upstream: `Part`/`Parts` (qlinear.zig:80).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Part {
    pub w: u64,
    pub scales: u64,
    pub biases: u64,
    pub out: u64,
    pub n: std::ffi::c_int,
    pub npad: std::ffi::c_int,
    pub sk: std::ffi::c_int,
    pub tiles: std::ffi::c_int,
    pub first: std::ffi::c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Parts {
    pub p: [Part; 4],
    pub count: std::ffi::c_int,
}

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs
/// share out. Upstream: `GemvPart` (qlinear.zig:82).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemvPart {
    pub w: u64,
    pub scales: u64,
    pub biases: u64,
    pub out: u64,
    pub n: std::ffi::c_int,
    pub npad: std::ffi::c_int,
    pub sk: std::ffi::c_int,
    pub tiles: std::ffi::c_int,
}

// Upstream: the comptime size assertions (qlinear.zig:85).
const _: () = assert!(std::mem::size_of::<Part>() == 56);
const _: () = assert!(std::mem::size_of::<Parts>() == 232);
const _: () = assert!(std::mem::size_of::<GemvPart>() == 48);

pub const GROUP_SMEM: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const SPLIT_SMEM: u32 = 17664; // LaneTile<64, 16, 64, 1, 4, 4>::SMEM: four stages, so more split CTAs fit an SM
pub const PREFILL_SMEM: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM

/// Split-K lane_gemv's fp32 slice partials and per-tile tickets:
/// splitK keeps tiles * sk under 384 and tiles under 192. Upstream:
/// `Split` (qlinear.zig:88).
pub struct Split<'d> {
    pub work: DeviceBuffer<'d>,
    pub tickets: DeviceBuffer<'d>,
}

impl<'d> Split<'d> {
    pub const TILES_MAX: usize = 192;
    const WORK_BYTES: usize = 384 * 16 * 64 * 4;

    /// Upstream: `init` (qlinear.zig:93) — the buffers then the
    /// zeroed tickets; a failure unwinds what stood (the port's
    /// buffers own, so their drops do).
    pub fn init(d: &'d Driver) -> Result<Split<'d>, Error> {
        let work = DeviceBuffer::alloc(d, Self::WORK_BYTES)?;
        let tickets = DeviceBuffer::alloc(d, Self::TILES_MAX * 4)?;
        // SAFETY: the tickets span zeroed on the device.
        let res =
            unsafe { (d.api.cuMemsetD32_v2.unwrap_unchecked())(tickets.ptr, 0, Self::TILES_MAX) };
        d.check(res, "cuMemsetD32")?;
        Ok(Split { work, tickets })
    }
}

/// qmm.split_k: K slices fixed by the weight's shape, never by the row
/// count. Upstream: `splitK` (qlinear.zig:105).
pub fn split_k(n: usize, k: usize) -> usize {
    let tiles = n.div_ceil(64);
    let groups = k / 64;
    let mut sk: usize = 1;
    while sk < 8 && tiles * sk < 192 && groups % (sk * 2) == 0 && groups / (sk * 2) >= 8 {
        sk *= 2;
    }
    sk
}

const ZERO_PART: Part = Part {
    w: 0,
    scales: 0,
    biases: 0,
    out: 0,
    n: 0,
    npad: 0,
    sk: 0,
    tiles: 0,
    first: 0,
};

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its
/// split-K form, qmm_prefill's prompt GEMM. Upstream: `Affine4Kernels`
/// (qlinear.zig:115).
pub struct Affine4Kernels<'d> {
    pub group_fn: Function<'d>,
    pub gemv_fn: Function<'d>,
    pub split_fn: Function<'d>,
    pub prefill_fn: Function<'d>,
    /// Resident lane_gemv CTAs: per SM times SMs.
    pub gemv_blocks: usize,
    /// Every GPU but GB10: lane_gemv's slices on CTAs of their own
    /// (one stream's decode() at a time).
    pub split: Option<Split<'d>>,
    /// GB10: the decode kernels launch with programmatic dependent
    /// launch.
    pub pdl: bool,
}

impl<'d> Affine4Kernels<'d> {
    /// From loaded qmm_group, qmm_prefill and lane_gemv modules; GB10
    /// decodes without the split form. Upstream: `resolve`
    /// (qlinear.zig:127) — the port takes the modules by borrow (Zig
    /// copies the handle value; the caller keeps the module alive).
    pub fn resolve(
        d: &'d Driver,
        group_mod: &Module<'d>,
        prefill_mod: &Module<'d>,
        gemv_mod: &Module<'d>,
        sms: usize,
        gb10: bool,
    ) -> Result<Affine4Kernels<'d>, Error> {
        let group_fn = group_mod.function(symbols::GROUP)?;
        let prefill_fn = prefill_mod.function(symbols::PREFILL)?;
        let gemv_fn = gemv_mod.function(symbols::GEMV)?;
        let split_fn = gemv_mod.function(symbols::GEMV_SPLIT)?;
        group_fn.allow_dynamic_shared(GROUP_SMEM)?;
        gemv_fn.allow_dynamic_shared(GROUP_SMEM)?;
        split_fn.allow_dynamic_shared(SPLIT_SMEM)?;
        prefill_fn.allow_dynamic_shared(PREFILL_SMEM)?;
        let gemv_blocks = (gemv_fn.occupancy(128, GROUP_SMEM as usize)?.max(1) as usize) * sms;
        Ok(Affine4Kernels {
            group_fn,
            gemv_fn,
            split_fn,
            prefill_fn,
            gemv_blocks,
            pdl: gb10,
            split: if gb10 { None } else { Some(Split::init(d)?) },
        })
    }

    /// Upstream: `deinit` (qlinear.zig:139) — the port's Drop frees
    /// the split's buffers; this ends their hold early.
    pub fn deinit(&mut self) {
        self.split = None;
    }

    /// qmm.matmul: x (rows, k) bf16 with group sums xs -> out (rows, n)
    /// bf16, qmm_group's tile-2 bits on every path. Upstream: `decode`
    /// (qlinear.zig:142).
    pub fn decode(
        &self,
        s: &Stream,
        x: u64,
        xs: u64,
        q: Affine4,
        out: u64,
        rows: usize,
    ) -> Result<(), Error> {
        if rows > 16 {
            return Err(Error::WindowTooWide);
        }
        let sk = split_k(q.n, q.k);
        if sk > 1 {
            if let Some(sp) = &self.split {
                return self.gemv_split(s, x, xs, q, out, rows, sk, sp);
            }
        }
        if sk > 1 {
            self.gemv(s, x, xs, q, out, rows, sk)
        } else {
            self.cluster(s, x, xs, q, out, rows, sk)
        }
    }

    /// Each CTA writes one K slice's partial from zero; the tile's
    /// last CTA sums them in slice order. Upstream: `gemvSplit`
    /// (qlinear.zig:148).
    #[allow(clippy::too_many_arguments)] // the kernel's parameters, upstream's signature
    pub fn gemv_split(
        &self,
        s: &Stream,
        x: u64,
        xs: u64,
        q: Affine4,
        out: u64,
        rows: usize,
        sk: usize,
        sp: &Split,
    ) -> Result<(), Error> {
        let tiles = q.n.div_ceil(64);
        if tiles > Split::TILES_MAX || tiles * sk * 16 * 64 * 4 > sp.work.len {
            return Err(Error::SplitTooWide);
        }
        let mut g: Args = Args::new();
        g.add(x);
        g.add(xs);
        g.add(GemvPart {
            w: q.w,
            scales: q.s,
            biases: q.b,
            out,
            n: q.n as std::ffi::c_int,
            npad: q.npad as std::ffi::c_int,
            sk: sk as std::ffi::c_int,
            tiles: tiles as std::ffi::c_int,
        });
        for v in [rows, q.k, q.k] {
            g.add(v as std::ffi::c_int);
        }
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        launch(
            &self.split_fn,
            Config {
                grid: Dim3 {
                    x: (tiles * sk) as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared: SPLIT_SMEM,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            s,
            &mut g,
        )
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in
    /// slice order, CTAs looping over the tiles. Upstream: `gemv`
    /// (qlinear.zig:158).
    #[allow(clippy::too_many_arguments)] // the kernel's parameters, upstream's signature
    pub fn gemv(
        &self,
        s: &Stream,
        x: u64,
        xs: u64,
        q: Affine4,
        out: u64,
        rows: usize,
        sk: usize,
    ) -> Result<(), Error> {
        let tiles = q.n.div_ceil(64);
        let mut g: Args = Args::new();
        g.add(x);
        g.add(xs);
        g.add(GemvPart {
            w: q.w,
            scales: q.s,
            biases: q.b,
            out,
            n: q.n as std::ffi::c_int,
            npad: q.npad as std::ffi::c_int,
            sk: sk as std::ffi::c_int,
            tiles: tiles as std::ffi::c_int,
        });
        for v in [rows, q.k, q.k] {
            g.add(v as std::ffi::c_int);
        }
        launch(
            &self.gemv_fn,
            Config {
                grid: Dim3 {
                    x: tiles.min(self.gemv_blocks) as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared: GROUP_SMEM,
                cluster: None,
                cluster_spread: false,
                pdl: self.pdl,
                cooperative: false,
            },
            s,
            &mut g,
        )
    }

    /// qmm_group's tile 2: a cluster of `sk` CTAs a column tile, its K
    /// slices summed over distributed shared memory. Upstream:
    /// `cluster` (qlinear.zig:169).
    #[allow(clippy::too_many_arguments)] // the kernel's parameters, upstream's signature
    pub fn cluster(
        &self,
        s: &Stream,
        x: u64,
        xs: u64,
        q: Affine4,
        out: u64,
        rows: usize,
        sk: usize,
    ) -> Result<(), Error> {
        let tiles = q.n.div_ceil(64);
        let mut parts = Parts {
            p: [ZERO_PART; 4],
            count: 1,
        };
        parts.p[0] = Part {
            w: q.w,
            scales: q.s,
            biases: q.b,
            out,
            n: q.n as std::ffi::c_int,
            npad: q.npad as std::ffi::c_int,
            sk: sk as std::ffi::c_int,
            tiles: tiles as std::ffi::c_int,
            first: 0,
        };
        let rows_t = rows.div_ceil(16);
        let clusters = rows_t * tiles; // one part: the cluster is its sk K slices of one tile
        let mut g: Args = Args::new();
        g.add(x);
        g.add(xs);
        g.add(parts);
        for v in [rows, q.k, q.k, rows_t, sk] {
            g.add(v as std::ffi::c_int);
        }
        launch(
            &self.group_fn,
            Config {
                grid: Dim3 {
                    x: (clusters * sk) as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared: GROUP_SMEM,
                cluster: if sk > 1 {
                    Some(Dim3 { x: sk as u32, y: 1, z: 1 })
                } else {
                    None
                },
                cluster_spread: false,
                pdl: self.pdl,
                cooperative: false,
            },
            s,
            &mut g,
        )
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one
    /// fp32 chain over K; bf16 out. Upstream: `prompt` (qlinear.zig:187).
    pub fn prompt(
        &self,
        s: &Stream,
        x: u64,
        q: Affine4,
        out: u64,
        rows: usize,
    ) -> Result<(), Error> {
        let rows_t = rows.div_ceil(128);
        let band = (12 << 20) / (128 * q.k * 2);
        let group = rows_t.min(band).max(1);
        let mut g: Args = Args::new();
        for v in [x, q.w, q.s, q.b, out] {
            g.add(v);
        }
        for v in [rows, q.n, q.k, q.npad, q.k, group] {
            g.add(v as std::ffi::c_int);
        }
        let grid = rows_t * q.n.div_ceil(128);
        launch(
            &self.prefill_fn,
            Config {
                grid: Dim3 {
                    x: grid as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: 256, y: 1, z: 1 },
                shared: PREFILL_SMEM,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            s,
            &mut g,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `split_k follows the Python shapes` (qlinear.zig:206).
    #[test]
    fn split_k_follows_the_python_shapes() {
        assert_eq!(2, split_k(10304, 2688));
        assert_eq!(8, split_k(2688, 4096));
        assert_eq!(2, split_k(4608, 2688));
        assert_eq!(4, split_k(2688, 5376));
        assert_eq!(1, split_k(131072, 2688));
    }

    /// Upstream's `a weight view reports its shape in every format`
    /// (qlinear.zig:214).
    #[test]
    fn a_weight_view_reports_its_shape_in_every_format() {
        let a: Weight = Weight::Affine4(Affine4 {
            w: 0,
            s: 0,
            b: 0,
            n: 10304,
            k: 2688,
            npad: 10304,
        });
        let f: Weight = Weight::Fp8g(fp8::Weight {
            codes: 0,
            scales: 0,
            n: 7168,
            k: 2560,
            npad: 7168,
        });
        assert_eq!(10304, Weight::outputs(a));
        assert_eq!(2560, Weight::inputs(f));
    }

    /// Upstream's `a format that is not loaded is refused before any
    /// launch` (qlinear.zig:221) — Zig's `undefined` stream stands as
    /// the test stub's, never reached: the refusal comes first.
    #[test]
    fn a_format_that_is_not_loaded_is_refused_before_any_launch() {
        let l: Linear = Linear {
            affine4: None,
            fp8g: None,
        };
        let w: Weight = Weight::Affine4(Affine4 {
            w: 0,
            s: 0,
            b: 0,
            n: 64,
            k: 64,
            npad: 64,
        });
        let s = crate::driver::stub::stub_stream();
        let in_: Rows = Rows {
            x: 0,
            ldx: 0,
            sums: 0,
            part: 0,
        };
        assert!(matches!(
            l.decode(&s, w, in_, 0, 1),
            Err(Error::FormatNotLoaded)
        ));
        assert!(matches!(
            l.prompt(&s, w, in_, 0, 1),
            Err(Error::FormatNotLoaded)
        ));
    }
}
