//! Launches of the torch-op replacements in kernels/cuda/torch_ops,
//! with their development launchers' geometry.
//!
//! Port of `zig/src/families/nemotron/cuda_torch_ops.zig` (151 lines).

#![allow(clippy::too_many_arguments)] // the launch wrappers carry the kernels' parameters, upstream's own signatures

use tensorfold_cuda::{Args, Config as LaunchConfig, Function, Module, Stream, Dim3, launch};

use crate::error::Error;

/// topk.cu's column tile, threads and workspace words a row (four
/// 8-bit digit passes, then ordered compaction). Upstream:
/// `topk_tile` and friends (cuda_torch_ops.zig:7).
const TOPK_TILE: usize = 4096;
const TOPK_THREADS: u32 = 256;

fn tiles(columns: usize) -> usize {
    columns.div_ceil(TOPK_TILE)
}

/// Scratch words `topk` needs for `rows` rows of `columns`. Upstream:
/// `topkScratchBytes` (cuda_torch_ops.zig:15).
pub fn topk_scratch_bytes(rows: usize, columns: usize) -> usize {
    rows * (tiles(columns) * 260 + 3) * 4
}

/// launch_shape.h's tf_launch_blocks: one block a `threads` elements,
/// at most 65535. Upstream: `blocks` (cuda_torch_ops.zig:20).
fn blocks(count: usize, threads: usize) -> u32 {
    count.div_ceil(threads).min(65535) as u32
}

/// Upstream: `Functions` (cuda_torch_ops.zig:24).
pub struct Functions<'d> {
    pub argmax: Function<'d>,
    pub hist: Function<'d>,
    pub digit: Function<'d>,
    pub count: Function<'d>,
    pub prefix: Function<'d>,
    pub compact: Function<'d>,
    pub to_f32: Function<'d>,
    pub lookup: Function<'d>,
    pub strided: Function<'d>,
    pub mamba_a: Function<'d>,
}

impl Functions<'_> {
    /// The modules in build order: argmax, topk, pointwise, indexing,
    /// movement, nemotron_constants. Upstream: `resolve`
    /// (cuda_torch_ops.zig:37).
    pub fn resolve<'d>(m: &[Module<'d>]) -> Result<Functions<'d>, Error> {
        Ok(Functions {
            argmax: m[0].function("tf_argmax_rows_i32_kernel")?,
            hist: m[1].function("tf_topk_f32_histogram_kernel")?,
            digit: m[1].function("tf_topk_f32_choose_digit_kernel")?,
            count: m[1].function("tf_topk_f32_count_kernel")?,
            prefix: m[1].function("tf_topk_f32_prefix_kernel")?,
            compact: m[1].function("tf_topk_f32_compact_kernel")?,
            to_f32: m[2].function("tf_bf16_to_f32_kernel")?,
            lookup: m[3].function("tf_lookup_ids_kernel")?,
            strided: m[4].function("tf_strided_copy_kernel")?,
            mamba_a: m[5].function("tf_nemotron_mamba_a_f32_kernel")?,
        })
    }
}

/// The torch-op replacements on one stream. Upstream: `Torch`
/// (cuda_torch_ops.zig:53) — the port's Stream owns its CUDA object,
/// so the handle borrows it.
pub struct Torch<'a, 'd> {
    pub f: &'a Functions<'d>,
    pub s: &'a Stream<'d>,
}

impl Torch<'_, '_> {
    fn go(
        &self,
        f: &Function<'_>,
        grid: [usize; 2],
        block: u32,
        args: &mut Args,
    ) -> Result<(), Error> {
        launch(
            f,
            LaunchConfig {
                grid: Dim3 {
                    x: grid[0] as u32,
                    y: grid[1] as u32,
                    z: 1,
                },
                block: Dim3 { x: block, y: 1, z: 1 },
                shared: 0,
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

    /// torch.argmax(rows of bf16 logits) as int32: the first maximum,
    /// a NaN first. Upstream: `argmax` (cuda_torch_ops.zig:62).
    pub fn argmax(
        &self,
        logits: u64,
        vocab: usize,
        ld: usize,
        out: u64,
        rows: usize,
    ) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(logits);
        a.add(out);
        for v in [rows, vocab, ld] {
            a.add(v as i64);
        }
        a.add(0i32); // dtype 0: bf16
        self.go(&self.f.argmax, [rows, 1], 1024, &mut a)
    }

    /// `.float()` of `count` bf16 values (exact). Upstream: `toF32`
    /// (cuda_torch_ops.zig:72).
    pub fn to_f32(&self, in_: u64, out: u64, count: usize) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(in_);
        a.add(out);
        a.add(count as u64);
        self.go(&self.f.to_f32, [blocks(count, 256) as usize, 1], 256, &mut a)
    }

    /// torch.topk(rows of fp32, k, sorted=False): values above the
    /// k-th in column order, then ties in column order. Upstream:
    /// `topk` (cuda_torch_ops.zig:80).
    pub fn topk(
        &self,
        in_: u64,
        columns: usize,
        rows: usize,
        k: usize,
        values: u64,
        indices: u64,
        scratch: u64,
    ) -> Result<(), Error> {
        let n = tiles(columns);
        let row_bytes: u64 = (columns * 4) as u64;
        let hist = scratch;
        let threshold = hist + (rows * n * 256 * 4) as u64;
        let remaining = threshold + (rows * 4) as u64;
        let greater = remaining + (rows * 4) as u64;
        let equal = greater + (rows * n * 4) as u64;
        let greater_prefix = equal + (rows * n * 4) as u64;
        let equal_prefix = greater_prefix + (rows * n * 4) as u64;
        let greater_total = equal_prefix + (rows * n * 4) as u64;
        let mut shift: u32 = 24;
        loop {
            let mut h = Args::new();
            for v in [in_, hist, threshold, columns as u64, n as u64, row_bytes, 4] {
                h.add(v);
            }
            h.add(shift);
            self.go(&self.f.hist, [n, rows], TOPK_THREADS, &mut h)?;
            let mut d = Args::new();
            for v in [hist, threshold, remaining, n as u64] {
                d.add(v);
            }
            d.add(k as u32);
            d.add(shift);
            self.go(&self.f.digit, [rows, 1], TOPK_THREADS, &mut d)?;
            if shift == 0 {
                break;
            }
            shift -= 8;
        }
        let mut c = Args::new();
        for v in [in_, threshold, greater, equal, columns as u64, n as u64, row_bytes, 4] {
            c.add(v);
        }
        self.go(&self.f.count, [n, rows], TOPK_THREADS, &mut c)?;
        let mut p = Args::new();
        for v in [greater, equal, greater_prefix, equal_prefix, greater_total, n as u64] {
            p.add(v);
        }
        self.go(&self.f.prefix, [rows, 1], 1, &mut p)?;
        let mut w = Args::new();
        for v in [
            in_,
            values,
            indices,
            threshold,
            greater_prefix,
            equal_prefix,
            greater_total,
            columns as u64,
            n as u64,
            row_bytes,
            4,
        ] {
            w.add(v);
        }
        w.add(k as u32);
        self.go(&self.f.compact, [n, rows], TOPK_THREADS, &mut w)
    }

    /// id_map[local]: draft-head columns to token ids; an out-of-range
    /// column sets `invalid`. Upstream: `lookup`
    /// (cuda_torch_ops.zig:118).
    pub fn lookup(
        &self,
        vocabulary: u64,
        size: usize,
        local: u64,
        global: u64,
        count: usize,
        invalid: u64,
    ) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(vocabulary);
        a.add(size as u64);
        a.add(local);
        a.add(global);
        a.add(count as u64);
        a.add(invalid);
        self.go(&self.f.lookup, [blocks(count, 256) as usize, 1], 256, &mut a)
    }

    /// `.contiguous()` of `rows` rows of `bytes` bytes, `src_ld` and
    /// `dst_ld` bytes apart. Upstream: `copyRows`
    /// (cuda_torch_ops.zig:130).
    pub fn copy_rows(
        &self,
        src: u64,
        src_ld: usize,
        dst: u64,
        dst_ld: usize,
        bytes: usize,
        count: usize,
    ) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(src);
        a.add(dst);
        for v in [count, 1, bytes, src_ld, bytes, dst_ld, bytes] {
            a.add(v as u64);
        }
        self.go(&self.f.strided, [count, 1], 256, &mut a)
    }

    /// A = -exp(A_log) from fp32 A_log (startup only). Upstream:
    /// `mambaA` (cuda_torch_ops.zig:139).
    pub fn mamba_a(&self, alog: u64, out: u64, count: usize) -> Result<(), Error> {
        let mut a = Args::new();
        a.add(alog);
        a.add(out);
        a.add(count as u64);
        self.go(&self.f.mamba_a, [blocks(count, 128) as usize, 1], 128, &mut a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `topk scratch matches topk.cu's workspace`
    /// (cuda_torch_ops.zig:148).
    #[test]
    fn topk_scratch_matches_topk_cus_workspace() {
        assert_eq!(16 * (32 * 260 + 3) * 4, topk_scratch_bytes(16, 131072));
        assert_eq!((8 * 260 + 3) * 4, topk_scratch_bytes(1, 32768));
    }
}
