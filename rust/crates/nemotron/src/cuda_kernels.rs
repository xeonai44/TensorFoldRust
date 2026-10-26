//! Nemotron's CUDA kernels: our .cu fatbins with the Python wrappers'
//! launch logic, and a captured Triton set if given.
//!
//! Port of `zig/src/families/nemotron/cuda_kernels.zig` (296 lines) —

#![allow(clippy::too_many_arguments)] // the launch wrappers carry the kernels' parameters, upstream's own signatures
//! the 21-module registry (upstream's image order kept exactly), the
//! symbol resolutions, and `Ops`, every launch helper on one stream.

use tensorfold_cuda::{
    abi, grouped, kernels, launch, qlinear, Args, Config as LaunchConfig, Context, Dim3,
    Driver, Function, Module, Stream,
};
use tensorfold_cuda::aot::Set;

use crate::cuda_glue::Fns;
use crate::cuda_torch_ops;
use crate::error::Error;

/// Mangled names of the instantiations the copies in kernels/cuda
/// export (cuobjdump -symbols of each fatbin). Upstream: `sym`
/// (cuda_kernels.zig:9).
mod sym {
    pub const EXPERT_UP: &str = "_ZN10tf_experts13expert_kernelILi64ELi1ELi1ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const EXPERT_DOWN: &str = "_ZN10tf_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const PRE_UP: &str = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const PRE_DOWN: &str = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi3ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const PACK_EXPERTS: &str = "_ZN15tf_experts_pack11pack_kernelILi2EEEvPKjPKtS4_Pjiiii";
    pub const PATTN: &str = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    pub const SCAN_ROWS: &str = "_ZN12tf_scan_rows11scan_kernelEPK13__nv_bfloat16S2_PfPKfS5_S5_PS0_iiiiiiiiff";
}

pub const PRE_EXPERTS_SMEM: u32 = 38400; // Pre<64, 1, 2, 2, 4>: three stages of 800 uint4
pub const PATTN_SMEM: u32 = 65536; // eight 32-key slots of 128 dims

/// The 4-bit projections' shared pieces (cuda/qlinear.zig): split-K
/// scratch and a tiled weight. Upstream: the re-exports
/// (cuda_kernels.zig:22).
pub type Split<'d> = qlinear::Split<'d>;
pub type QLinear = qlinear::Affine4;

/// A layer's grouped expert tables ([E, N/32, K/64, 1, 288] int32
/// blocks). Upstream: `Experts` (cuda_kernels.zig:27).
#[derive(Clone, Copy)]
pub struct Experts {
    pub up: u64,
    pub down: u64,
    pub count: usize,
    pub width: usize,
    pub dims: usize,
}

/// Scratch the expert plan writes (cuda/grouped.zig, shared by the MoE
/// families). Upstream: the `Plan` re-export (cuda_kernels.zig:30).
pub type Plan = grouped::Plan;

/// Upstream: `Kernels` (cuda_kernels.zig:32) — the port's Modules own
/// their CUDA objects, so a failure unwinds them by the Vec's Drop.
pub struct Kernels<'d> {
    pub d: &'d Driver,
    pub mods: Vec<Module<'d>>,
    /// A captured Triton set for the glue (GB10's qualified one);
    /// None: our own glue kernels.
    pub triton: Option<Set<'d>>,
    pub glue: Fns<'d>,
    /// The 4-bit projections: qmm_group, lane_gemv and qmm_prefill.
    pub affine: qlinear::Affine4Kernels<'d>,
    pub expert_up: Function<'d>,
    pub expert_down: Function<'d>,
    pub router: grouped::Router<'d>,
    pub pre_up: Function<'d>,
    pub pre_down: Function<'d>,
    pub pack_experts: Function<'d>,
    pub pattn: Function<'d>,
    pub scan_rows: Function<'d>,
    pub pack_dense: Function<'d>,
    pub transpose16: Function<'d>,
    pub serial_feed: Function<'d>,
    pub plan_routed: Function<'d>,
    pub rest_rows: Function<'d>,
    pub draw: Function<'d>,
    pub draw_ids: Function<'d>,
    pub logprob_rows: Function<'d>,
    pub torch: cuda_torch_ops::Functions<'d>,
    /// Resident blocks the decode expert kernels fill: per SM times
    /// SMs.
    pub expert_blocks: [usize; 2],
    pub gb10: bool,
    /// The card has its own memory: checkpoint bytes reach it through
    /// page-locked slots.
    pub discrete: bool,
}

impl<'d> Kernels<'d> {
    /// Loads every module; `triton_dir`, if given, holds a captured
    /// aot.json and cubins for this GPU's glue. Upstream: `load`
    /// (cuda_kernels.zig:60) — the image order kept exactly.
    pub fn load(ctx: &Context<'d>, triton_dir: Option<&str>) -> Result<Kernels<'d>, Error> {
        let d = ctx.d;
        if !kernels::available {
            return Err(Error::Cuda(tensorfold_cuda::Error::BuiltWithoutKernels));
        }
        let images: [&[u8]; 21] = [
            &kernels::qmm_group.bytes,
            &kernels::qmm_prefill.bytes,
            &kernels::experts.bytes,
            &kernels::experts_prefill.bytes,
            &kernels::experts_pack.bytes,
            &kernels::prefill_attention.bytes,
            &kernels::scan_rows.bytes,
            &kernels::nemotron_ops.bytes,
            &kernels::torch_argmax.bytes,
            &kernels::torch_topk.bytes,
            &kernels::torch_pointwise.bytes,
            &kernels::torch_indexing.bytes,
            &kernels::torch_movement.bytes,
            &kernels::torch_nemotron_constants.bytes,
            &kernels::sample.bytes,
            &kernels::lane_gemv.bytes,
            &kernels::nemotron_norms.bytes,
            &kernels::nemotron_route.bytes,
            &kernels::nemotron_mamba.bytes,
            &kernels::nemotron_attention.bytes,
            &kernels::nemotron_keyed.bytes,
        ];
        let mut mods: Vec<Module<'d>> = Vec::with_capacity(images.len());
        for img in images {
            mods.push(Module::load(d, img)?);
        }
        // The resolutions in upstream's order (cuda_kernels.zig:73-92);
        // a failure unwinds the locals, the modules by the Vec's Drop
        let expert_up = mods[2].function(sym::EXPERT_UP)?;
        let expert_down = mods[2].function(sym::EXPERT_DOWN)?;
        let router = grouped::Router::resolve(&mods[2])?;
        let pre_up = mods[3].function(sym::PRE_UP)?;
        let pre_down = mods[3].function(sym::PRE_DOWN)?;
        let pack_experts = mods[4].function(sym::PACK_EXPERTS)?;
        let pattn = mods[5].function(sym::PATTN)?;
        let scan_rows = mods[6].function(sym::SCAN_ROWS)?;
        let pack_dense = mods[7].function("tf_pack_dense")?;
        let transpose16 = mods[7].function("tf_transpose_pad16")?;
        let serial_feed = mods[7].function("tf_serial_feed")?;
        let plan_routed = mods[7].function("tf_plan_routed")?;
        let rest_rows = mods[7].function("tf_rest_rows")?;
        let torch = cuda_torch_ops::Functions::resolve(&mods[8..14])?;
        let draw = mods[14].function("tf_draw")?;
        let draw_ids = mods[14].function("tf_draw_ids")?;
        let glue = Fns::resolve(&mods[16..21])?;
        let logprob_rows = mods[14].function("tf_logprob_rows")?;
        let triton = match triton_dir {
            Some(dir) => Some(Set::load(d, ctx.device, dir)?),
            None => None,
        };
        pre_up.allow_dynamic_shared(PRE_EXPERTS_SMEM)?;
        pre_down.allow_dynamic_shared(PRE_EXPERTS_SMEM)?;
        pattn.allow_dynamic_shared(PATTN_SMEM)?;
        let sms: usize = ctx.attribute(abi::DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
        let expert_blocks = [
            (expert_up.occupancy(128, 0)?.max(1) as usize) * sms,
            (expert_down.occupancy(128, 0)?.max(1) as usize) * sms,
        ];
        let major = ctx.attribute(abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = ctx.attribute(abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let gb10 = major == 12 && minor == 1;
        let discrete = ctx.attribute(abi::DEVICE_ATTRIBUTE_INTEGRATED)? == 0;
        let affine = qlinear::Affine4Kernels::resolve(d, &mods[0], &mods[1], &mods[15], sms, gb10)?;
        Ok(Kernels {
            d,
            mods,
            triton,
            glue,
            affine,
            expert_up,
            expert_down,
            router,
            pre_up,
            pre_down,
            pack_experts,
            pattn,
            scan_rows,
            pack_dense,
            transpose16,
            serial_feed,
            plan_routed,
            rest_rows,
            draw,
            draw_ids,
            logprob_rows,
            torch,
            expert_blocks,
            gb10,
            discrete,
        })
    }

    /// Upstream: `deinit` (cuda_kernels.zig:106) — the port's Drops free
    /// the split's buffers, the Triton set and the modules; this ends
    /// their hold early.
    pub fn deinit(&mut self) {
        self.affine.deinit();
        self.triton = None;
        self.mods.clear();
    }
}

impl Drop for Kernels<'_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// qmm.split_k (cuda/qlinear.zig). Upstream: the re-export
/// (cuda_kernels.zig:113).
pub use qlinear::split_k;

/// experts.max_items (cuda/grouped.zig). Upstream: the re-export
/// (cuda_kernels.zig:116).
pub use grouped::max_items;

/// Launch helpers on one stream; each mirrors the Python wrapper it
/// replaces. Upstream: `Ops` (cuda_kernels.zig:128) — the port's
/// Stream owns its CUDA object, so the ops borrow it.
#[derive(Clone, Copy)]
pub struct Ops<'a, 'd> {
    pub k: &'a Kernels<'d>,
    pub s: &'a Stream<'d>,
}

impl<'a, 'd> Ops<'a, 'd> {
    fn go(
        &self,
        f: &Function<'_>,
        grid: [usize; 3],
        block: u32,
        shared: u32,
        args: &mut Args,
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
            args,
        )?;
        Ok(())
    }

    /// The torch-op replacements on this stream. Upstream: `torch`
    /// (cuda_kernels.zig:137).
    pub fn torch(&self) -> cuda_torch_ops::Torch<'_, 'd> {
        cuda_torch_ops::Torch {
            f: &self.k.torch,
            s: self.s,
        }
    }

    /// sample.cu: row r of bf16 logits drawn at position meta[0] + r +
    /// 1 + offset, columns as `ids` token ids if given. Upstream:
    /// `draw` (cuda_kernels.zig:141).
    pub fn draw(&self, logits: u64, vocab: usize, rule: u64, meta: u64, offset: usize, out: u64, rows: usize, ids: Option<u64>, prob: Option<u64>) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(logits);
        a.add(vocab as u32);
        a.add(rule);
        a.add(meta);
        a.add(offset as i32);
        a.add(out);
        if let Some(x) = ids {
            a.add(x);
        }
        a.add(prob.unwrap_or(0));
        let f = if ids.is_some() { &self.k.draw_ids } else { &self.k.draw };
        self.go(f, [rows, 1, 1], 1024, 0, &mut a)
    }

    /// sample.cu: each bf16 logits row's lanes.logprob words at `out`,
    /// for its pick in `picks` and `k` best tokens. Upstream:
    /// `logprobRows` (cuda_kernels.zig:155).
    pub fn logprob_rows(&self, logits: u64, vocab: usize, picks: u64, k: u8, out: u64, rows: usize) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(logits);
        a.add(vocab as u32);
        a.add(picks);
        a.add(k);
        a.add(out);
        self.go(&self.k.logprob_rows, [rows, 1, 1], 1024, 0, &mut a)
    }

    /// The 4-bit projections on this stream (cuda/qlinear.zig's
    /// affine-4 paths, bits unchanged). Upstream: `dense`
    /// (cuda_kernels.zig:165).
    pub fn dense(&self, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) -> Result<(), Error> {
        Ok(self.k.affine.decode(self.s, x, xs, q, out, rows)?)
    }

    pub fn gemv_split(&self, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize, sp: &Split<'_>) -> Result<(), Error> {
        Ok(self.k.affine.gemv_split(self.s, x, xs, q, out, rows, sk, sp)?)
    }

    pub fn gemv(&self, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) -> Result<(), Error> {
        Ok(self.k.affine.gemv(self.s, x, xs, q, out, rows, sk)?)
    }

    pub fn cluster(&self, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) -> Result<(), Error> {
        Ok(self.k.affine.cluster(self.s, x, xs, q, out, rows, sk)?)
    }

    pub fn prefill_dense(&self, x: u64, q: QLinear, out: u64, rows: usize) -> Result<(), Error> {
        Ok(self.k.affine.prompt(self.s, x, q, out, rows)?)
    }

    /// nemotron_ops' rest: y (bf16, or fp32 when `y_f32`) += r x at
    /// rows `i * row_mul + row_add` for i < rows. Upstream: `restRows`
    /// (cuda_kernels.zig:175).
    pub fn rest_rows(&self, x: u64, x_stride: usize, r: u64, y: u64, y_stride: usize, y_f32: bool, rows: usize, row_mul: usize, row_add: usize, n: usize, k: usize) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(x);
        a.add(x_stride as std::ffi::c_int);
        a.add(r);
        a.add(y);
        a.add(y_stride as std::ffi::c_int);
        a.add(y_f32 as std::ffi::c_int);
        for v in [rows, row_mul, row_add, n, k] {
            a.add(v as std::ffi::c_int);
        }
        self.go(&self.k.rest_rows, [n.div_ceil(4), 1, 1], 128, 0, &mut a)
    }

    /// experts.route: pairs grouped by expert into items of at most
    /// `tile` pairs (16 decode, 64 prefill). Upstream: `plan`
    /// (cuda_kernels.zig:188).
    pub fn plan(&self, picks: u64, pairs: usize, count: usize, tile: usize, p: Plan) -> Result<(), Error> {
        Ok(self.k.router.route(self.s, picks, pairs, count, tile, p)?)
    }

    fn expert_args(
        x: u64,
        x_stride: usize,
        slots: usize,
        w: u64,
        kg: usize,
        nb: usize,
        p: Plan,
        out: u64,
        n: usize,
    ) -> Args {
        let mut a = Args::new();
        a.add(x);
        a.add(x_stride as std::ffi::c_int);
        a.add(slots as std::ffi::c_int);
        a.add(w);
        a.add(kg as std::ffi::c_int);
        a.add(nb as std::ffi::c_int);
        for v in [p.items, p.counts, p.members, out] {
            a.add(v);
        }
        a.add(n as std::ffi::c_int);
        a.add(0.0f32);
        a
    }

    /// experts.route's plan over the routed slots alone (rows <= 16,
    /// experts <= 128): the shared halves run apart. Upstream:
    /// `planRouted` (cuda_kernels.zig:197).
    pub fn plan_routed(&self, picks: u64, rows: usize, slots: usize, routed: usize, count: usize, tile: usize, p: Plan) -> Result<(), Error> {
        if rows > 16 || slots > 8 || count > 128 {
            return Err(Error::PlanTooWide);
        }
        let mut a = Args::new();
        a.add(picks);
        for v in [rows, slots, routed, count, tile] {
            a.add(v as std::ffi::c_int);
        }
        for v in [p.members, p.items, p.counts] {
            a.add(v);
        }
        self.go(&self.k.plan_routed, [1, 1, 1], 128, 0, &mut a)
    }

    /// experts.run (decode form): `up` takes token rows (relu^2, bf16
    /// out), else pair rows (fp32 out). Upstream: `experts`
    /// (cuda_kernels.zig:207).
    pub fn experts(&self, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_units: usize) -> Result<(), Error> {
        let grid = max_units.div_ceil(4);
        let grid = grid.min(self.k.expert_blocks[if up { 0 } else { 1 }]);
        if grid < 1 {
            return Ok(());
        }
        let mut a = Self::expert_args(x, x_stride, slots, w, kg, nb, p, out, n);
        let f = if up { &self.k.expert_up } else { &self.k.expert_down };
        self.go(f, [grid, 1, 1], 128, 0, &mut a)
    }

    /// experts.prefill: 64 pairs x 128 columns a CTA; `up` relu^2, else
    /// bf16 sums (epilogue 3). Upstream: `expertsPrefill`
    /// (cuda_kernels.zig:216).
    pub fn experts_prefill(&self, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_items: usize) -> Result<(), Error> {
        let grid = max_items * nb.div_ceil(4);
        if grid < 1 {
            return Ok(());
        }
        let mut a = Self::expert_args(x, x_stride, slots, w, kg, nb, p, out, n);
        let f = if up { &self.k.pre_up } else { &self.k.pre_down };
        self.go(f, [grid, 1, 1], 256, PRE_EXPERTS_SMEM, &mut a)
    }

    /// prefill_attention (head dim 128): q (rows, heads, 128) at
    /// positions p0.. against caches filled through them. Upstream:
    /// `prefillAttention` (cuda_kernels.zig:223).
    pub fn prefill_attention(&self, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) -> Result<(), Error> {
        let g = heads / kv_heads;
        let mut a = Args::new();
        for v in [q, kc, vc, out] {
            a.add(v);
        }
        for v in [p0, rows, heads, kv_heads, g] {
            a.add(v as std::ffi::c_int);
        }
        a.add(scale);
        self.go(
            &self.k.pattn,
            [rows.div_ceil(16), kv_heads * (g / 8), 1],
            256,
            PATTN_SMEM,
            &mut a,
        )
    }

    /// mamba.scan_rows: a chunk's scan through scan_rows.cu; `state`
    /// ends at the chunk's last row. Upstream: `scanRows`
    /// (cuda_kernels.zig:230).
    pub fn scan_rows(&self, proj: u64, xc: u64, state: u64, a_: u64, d_: u64, dtb: u64, y: u64, rows: usize, proj_w: usize, heads: usize, dh: usize, cd: usize, groups: usize, lo: f32, hi: f32) -> Result<(), Error> {
        let mut a = Args::new();
        for v in [proj, xc, state, a_, d_, dtb, y] {
            a.add(v);
        }
        for v in [rows, proj_w, heads * dh, cd, heads * dh + cd, dh, heads / groups, groups] {
            a.add(v as std::ffi::c_int);
        }
        a.add(lo);
        a.add(hi);
        self.go(&self.k.scan_rows, [heads, dh / 32, 1], 128, 0, &mut a)
    }

    /// Upstream: `copy` (cuda_kernels.zig:243).
    pub fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), Error> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: the device-to-device copy on this stream.
        let res = unsafe {
            (self.k.d.api.cuMemcpyDtoDAsync_v2.unwrap_unchecked())(dst, src, bytes, self.s.handle)
        };
        Ok(self.k.d.check(res, "cuMemcpyDtoDAsync")?)
    }

    /// Upstream: `fill32` (cuda_kernels.zig:247).
    pub fn fill32(&self, dst: u64, value: u32, words: usize) -> Result<(), Error> {
        // SAFETY: the device words set on this stream.
        let res = unsafe {
            (self.k.d.api.cuMemsetD32Async.unwrap_unchecked())(dst, value, words, self.s.handle)
        };
        Ok(self.k.d.check(res, "cuMemsetD32Async")?)
    }

    /// Upstream: `upload` (cuda_kernels.zig:251).
    pub fn upload(&self, dst: u64, bytes: &[u8]) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: the host bytes copied to the device on this stream.
        let res = unsafe {
            (self.k.d.api.cuMemcpyHtoDAsync_v2.unwrap_unchecked())(
                dst,
                bytes.as_ptr().cast(),
                bytes.len(),
                self.s.handle,
            )
        };
        Ok(self.k.d.check(res, "cuMemcpyHtoDAsync")?)
    }

    /// Upstream: `download` (cuda_kernels.zig:255).
    pub fn download(&self, dst: &mut [u8], src: u64) -> Result<(), Error> {
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: the device bytes copied to the host on this stream.
        let res = unsafe {
            (self.k.d.api.cuMemcpyDtoHAsync_v2.unwrap_unchecked())(
                dst.as_mut_ptr().cast(),
                src,
                dst.len(),
                self.s.handle,
            )
        };
        Ok(self.k.d.check(res, "cuMemcpyDtoHAsync")?)
    }

    /// One serial round's end on the device: its token feeds the next
    /// window, the window's meta advances. Upstream: `serialFeed`
    /// (cuda_kernels.zig:259).
    pub fn serial_feed(&self, sampled: u64, ids: u64, meta: u64, history: u64) -> Result<(), Error> {
        let mut a = Args::new();
        for v in [sampled, ids, meta, history] {
            a.add(v);
        }
        self.go(&self.k.serial_feed, [1, 1, 1], 1, 0, &mut a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `items follow the Python shapes` (cuda_kernels.zig:268).
    #[test]
    fn items_follow_the_python_shapes() {
        assert_eq!(135, max_items(328, 130, 64));
        assert_eq!(1154, max_items(16384, 130, 16));
        assert_eq!(8, max_items(8, 130, 16));
    }
}
