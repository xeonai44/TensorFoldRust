//! Block-FP8 projections (128 x 128 e4m3 blocks, fp32 scales) on
//! fp8_lane.cu, chosen as the Python lane matmul chooses.
//!
//! Port of `zig/src/cuda/fp8.zig` (224 lines).

use crate::driver::{Driver, Error};
use crate::kernels;
use crate::launch::{launch, Args, Config};
use crate::module::{Function, Module};
use crate::stream::Stream;
use crate::Dim3;

pub const BLOCK: usize = 128; // a checkpoint scale's rows and inputs
pub const GROUP: usize = 64; // inputs a kernel stage reads, one column scale each
pub const FUSED_ROWS: usize = 256; // rows from which a projection's K slices meet in one block (linear.py FUSED_ROWS)
const BN: usize = 64;
const THREADS: u32 = 128; // WM 1 x WN 4 warps

/// The dynamic shared bytes of a BM-row tile (qmmf.cu Tile<FP8G, BM,
/// 64, 1, 4, 4>::SMEM). Upstream: `sharedBytes` (fp8.zig:22).
pub fn shared_bytes(bm: u32) -> u32 {
    let stage = (bm * GROUP as u32 * 2 + BN as u32 * GROUP as u32 + BN as u32 * 4)
        .div_ceil(128)
        * 128;
    let partials = (bm / 16) * 2 * 4 * THREADS * 4;
    (4 * stage).max(partials)
}

/// K slices for an (n, k) weight, from the shape alone so a row's bits
/// never depend on the row count (qmm.split_k). Upstream: `splitK`
/// (fp8.zig:29).
pub fn split_k(n: usize, k: usize) -> u32 {
    let tiles = n.div_ceil(BN);
    let groups = k / GROUP;
    let mut sk: usize = 1;
    while sk < 8 && tiles * sk < 192 && groups % (sk * 2) == 0 && groups / (sk * 2) >= 8 {
        sk *= 2;
    }
    sk as u32
}

/// The row tile: 16 or 32 rows, else 64-row tiles side by side
/// (qmm.bucket). Upstream: `bucket` (fp8.zig:38).
pub fn bucket(m: usize) -> u32 {
    if m <= 16 {
        16
    } else if m <= 32 {
        32
    } else {
        64
    }
}

/// Rows padded to whole 128-row blocks, as the kernel's 64-column
/// tiles read them. Upstream: `padded` (fp8.zig:43).
pub fn padded(n: usize) -> usize {
    n.div_ceil(BLOCK) * BLOCK
}

/// The k of byte `byte` of `lane` in a k32 step (fragment_index's
/// kin). Upstream: `kin` (fp8.zig:48).
fn kin(lane: usize, byte: usize) -> usize {
    let p = 16 * (byte / 4) + 4 * (lane % 4) + byte % 4;
    16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2)
}

/// e4m3 codes [n, k] in the kernel's order [npad/64][k/64][8][32][2][8],
/// rows past n zero (linear._fragment_order). Upstream: `packCodes`
/// (fp8.zig:54).
pub fn pack_codes(out: &mut [u8], codes: &[u8], n: usize, k: usize) {
    let npad = padded(n);
    debug_assert!(out.len() == npad * k && codes.len() == n * k && k % GROUP == 0);
    let mut i = 0;
    for t in 0..npad.div_ceil(64) {
        for g in 0..k / GROUP {
            for a in 0..8 {
                for lane in 0..32 {
                    for h in 0..2 {
                        for b in 0..8 {
                            let row = t * 64 + a * 8 + lane / 4;
                            let col = g * GROUP + h * 32 + kin(lane, b);
                            out[i] = if row < n { codes[row * k + col] } else { 0 };
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Block scales [ceil(n/128), k/128] as the kernel's tiles
/// [npad/64][k/64][64] of fp32, rows past n one (from_rows). Upstream:
/// `packScales` (fp8.zig:67).
pub fn pack_scales(out: &mut [f32], scale_inv: &[f32], n: usize, k: usize) {
    let npad = padded(n);
    let kg = k / GROUP;
    debug_assert!(
        out.len() == npad * kg
            && scale_inv.len() == n.div_ceil(BLOCK) * (k / BLOCK)
    );
    for t in 0..npad.div_ceil(64) {
        for g in 0..kg {
            for c in 0..64 {
                let row = t * 64 + c;
                out[(t * kg + g) * 64 + c] = if row < n {
                    scale_inv[(row / BLOCK) * (k / BLOCK) + g * GROUP / BLOCK]
                } else {
                    1.0
                };
            }
        }
    }
}

/// A packed projection on the device: codes and scale tiles. Upstream:
/// `Weight` (fp8.zig:78).
#[derive(Clone, Copy)]
pub struct Weight {
    pub codes: u64,
    pub scales: u64,
    pub n: u32,
    pub k: u32,
    pub npad: u32,
}

/// fp8_lane's instances, by row tile and cluster, then the fused
/// prompt kernel and the slice reduce. Upstream: `symbols` (fp8.zig:81)
/// — cuobjdump -symbols of the fp8_lane fatbin.
pub mod symbols {
    pub const LANE16: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const LANE16C: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const LANE32: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const LANE32C: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const LANE64: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const LANE64C: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const FUSED: &str = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const REDUCE: &str = "_ZN11tf_fp8_lane13reduce_kernelILb0EEEvPKfPvxif";
}

const LANE_SYMBOLS: [&str; 8] = [
    symbols::LANE16,
    symbols::LANE16C,
    symbols::LANE32,
    symbols::LANE32C,
    symbols::LANE64,
    symbols::LANE64C,
    symbols::FUSED,
    symbols::REDUCE,
];

/// What one call launches, from the shapes alone: tile, slices, cluster
/// or reduce (qmmf_cuda's choice). Upstream: `Plan` (fp8.zig:93).
#[derive(Clone, Copy)]
pub struct Plan {
    pub bm: u32,
    pub sk: u32,
    pub fused: bool,
    pub cluster: bool,
    pub reduce: bool,
}

/// Upstream: `plan` (fp8.zig:95).
pub fn plan(m: usize, n: usize, k: usize, clusters: bool) -> Plan {
    let sk = split_k(n, k);
    let fused = sk > 1 && m >= FUSED_ROWS;
    let cluster = !fused && sk > 1 && sk <= 8 && clusters;
    Plan {
        bm: if fused { 64 } else { bucket(m) },
        sk,
        fused,
        cluster,
        reduce: !fused && sk > 1 && !cluster,
    }
}

/// The fp8_lane kernels loaded for one family. Upstream: `Lane`
/// (fp8.zig:102).
pub struct Lane<'d> {
    pub mod_: Module<'d>,
    /// lane16, lane16c, lane32, lane32c, lane64, lane64c, fused,
    /// reduce.
    pub fns: [Function<'d>; 8],
    /// sm_90 on: slices meet in a cluster's shared memory.
    pub clusters: bool,
}

impl<'d> Lane<'d> {
    /// Upstream: `load` (fp8.zig:107) — the module first, then every
    /// instance; a failed resolution unloads (the port's Module owns,
    /// so the drop does).
    pub fn load(d: &'d Driver, capability_major: u32) -> Result<Lane<'d>, Error> {
        if !kernels::available {
            return Err(Error::BuiltWithoutKernels);
        }
        let module = Module::load(d, &kernels::fp8_lane.bytes)?;
        let mut fns: Vec<Function<'d>> = Vec::with_capacity(8);
        for s in LANE_SYMBOLS {
            fns.push(module.function(s)?);
        }
        let fns: [Function<'d>; 8] = fns
            .try_into()
            .unwrap_or_else(|_| unreachable!("eight of eight resolved"));
        for (bm, f) in [16u32, 16, 32, 32, 64, 64, 64].iter().zip(fns.iter()) {
            f.allow_dynamic_shared(shared_bytes(*bm))?;
        }
        Ok(Lane {
            mod_: module,
            fns,
            clusters: capability_major >= 9,
        })
    }

    /// Slice partials a call needs (fp32 [sk][m][n]) when it reduces
    /// them, else 0. Upstream: `partBytes` (fp8.zig:122).
    pub fn part_bytes(&self, m: usize, w: Weight) -> usize {
        let p = plan(m, w.n as usize, w.k as usize, self.clusters);
        if p.reduce {
            p.sk as usize * m * w.n as usize * 4
        } else {
            0
        }
    }

    /// y [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times the
    /// weight, fp32 sums; `part` holds partialBytes. Upstream:
    /// `matmul` (fp8.zig:128).
    #[allow(clippy::too_many_arguments)] // the kernel's parameters, upstream's signature
    pub fn matmul(
        &self,
        s: &Stream,
        x: u64,
        ldx: usize,
        m: usize,
        w: Weight,
        y: u64,
        part: u64,
    ) -> Result<(), Error> {
        let p = plan(m, w.n as usize, w.k as usize, self.clusters);
        let f = if p.fused {
            &self.fns[6]
        } else {
            match p.bm {
                16 => &self.fns[if p.cluster { 1 } else { 0 }],
                32 => &self.fns[if p.cluster { 3 } else { 2 }],
                _ => &self.fns[if p.cluster { 5 } else { 4 }],
            }
        };
        let rows_t = m.div_ceil(p.bm as usize);
        let band = std::cmp::max(
            1,
            std::cmp::min(rows_t, (12 << 20) / (p.bm as usize * w.k as usize * 2)),
        );
        let mut args: Args = Args::new();
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(1.0f32);
        args.add(y);
        args.add(if p.reduce { part } else { 0u64 });
        args.add(m as std::ffi::c_int);
        args.add(w.n as std::ffi::c_int);
        args.add(w.k as std::ffi::c_int);
        args.add(p.sk as std::ffi::c_int);
        args.add(w.npad as std::ffi::c_int);
        args.add(if m == 1 { w.k as std::ffi::c_int } else { ldx as std::ffi::c_int });
        args.add(band as std::ffi::c_int);
        launch(
            f,
            Config {
                grid: Dim3 {
                    x: (rows_t * (w.n as usize).div_ceil(BN)) as u32,
                    y: 1,
                    z: if p.fused { 1 } else { p.sk },
                },
                block: Dim3 { x: THREADS, y: 1, z: 1 },
                shared: shared_bytes(p.bm),
                cluster: if p.cluster {
                    Some(Dim3 { x: 1, y: 1, z: p.sk })
                } else {
                    None
                },
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            s,
            &mut args,
        )?;
        if p.reduce {
            let total: usize = m * w.n as usize;
            let mut r: Args = Args::new();
            r.add(part);
            r.add(y);
            r.add(total as i64);
            r.add(p.sk as std::ffi::c_int);
            r.add(1.0f32);
            launch(
                &self.fns[7],
                Config {
                    grid: Dim3 {
                        x: total.div_ceil(256) as u32,
                        y: 1,
                        z: 1,
                    },
                    block: Dim3 { x: 256, y: 1, z: 1 },
                    shared: 0,
                    cluster: None,
                    cluster_spread: false,
                    pdl: false,
                    cooperative: false,
                },
                s,
                &mut r,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `tile shared bytes, slices and row tiles as the
    /// Python host picks them` (fp8.zig:170).
    #[test]
    #[allow(clippy::identity_op)]
    fn tile_shared_bytes_slices_and_row_tiles_as_the_python_host_picks_them() {
        assert_eq!(25600, shared_bytes(16));
        assert_eq!(33792, shared_bytes(32));
        assert_eq!(50176, shared_bytes(64));
        // Kolibri 1's projections: qkv [7168, 2560], o [2560, 6144]
        assert_eq!(2, split_k(7168, 2560)); // values from qmm.split_k itself
        assert_eq!(8, split_k(2560, 6144));
        assert_eq!(4, split_k(512, 2560));
        assert_eq!(1, split_k(2560, 512));
        assert_eq!(1, split_k(32768, 2560));
        assert_eq!(16, bucket(1));
        assert_eq!(32, bucket(17));
        assert_eq!(64, bucket(300));
        let p = plan(300, 2560, 6144, true);
        assert!(p.fused && p.bm == 64 && !p.cluster && !p.reduce);
        let q = plan(3, 2560, 6144, true);
        assert!(!q.fused && q.cluster && q.bm == 16 && q.sk == 8);
        let r = plan(3, 2560, 6144, false);
        assert!(r.reduce && !r.cluster);
    }

    /// Upstream's `the code repack is a permutation that puts each byte
    /// where fragment_index says` (fp8.zig:191).
    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)] // upstream spells the coordinates out
    fn the_code_repack_is_a_permutation_that_puts_each_byte_where_fragment_index_says() {
        let n = 130;
        let k = 128;
        let codes: Vec<u8> = (0..n * k).map(|i| (i * 7 + 1) as u8).collect();
        let mut out = vec![0u8; padded(n) * k];
        pack_codes(&mut out, &codes, n, k);
        // byte (t=2, g=1, a=0, lane=9, h=1, b=5): row 128 + 9/4 = 130
        // -> padding; (t=0, g=1, a=3, lane=9, h=1, b=5): row 26
        let idx = ((((0 * 2 + 1) * 8 + 3) * 32 + 9) * 2 + 1) * 8 + 5;
        assert_eq!(codes[26 * k + 64 + 32 + kin(9, 5)], out[idx]);
        let zeros = out.iter().filter(|b| **b == 0).count();
        assert!(zeros >= (padded(n) - n) * k);
        // kin covers a k32 step once for each lane's quad: 32 inputs
        // from 4 lanes x 8 bytes
        let mut seen = [false; 32];
        for lane in 0..4 {
            for b in 0..8 {
                assert!(!seen[kin(lane, b)]);
                seen[kin(lane, b)] = true;
            }
        }
    }

    /// Upstream's `scale tiles: each 64-input group takes its 128 x 128
    /// block's scale, padded rows one` (fp8.zig:215).
    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)] // upstream spells the coordinates out
    fn scale_tiles_each_64_input_group_takes_its_128_x_128_blocks_scale_padded_rows_one() {
        let n = 130;
        let k = 256;
        let inv: [f32; 2 * 2] = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 256 * 4];
        pack_scales(&mut out, &inv, n, k);
        assert_eq!(2.0, out[(0 * 4 + 3) * 64 + 5]); // row 5, group 3 -> block (0, 1)
        assert_eq!(3.0, out[(2 * 4 + 0) * 64 + 1]); // row 129 -> block (1, 0)
        assert_eq!(1.0, out[(2 * 4 + 1) * 64 + 2]); // row 130: padding
    }
}
