//! experts.route's plan on experts.cu's kernels: routed pairs grouped
//! by expert into items, shared by MoE families.
//!
//! Port of `zig/src/cuda/grouped.zig` (77 lines).

use crate::driver::Error;
use crate::launch::{launch, Args, Config};
use crate::module::{Function, Module};
use crate::stream::Stream;
use crate::Dim3;

pub const SMALL: usize = 1024; // pairs the one-block plan takes; wider plans rank in blocks of 1024 pairs

/// experts.max_items: an item per used expert plus one per `tile` pairs
/// past its first. Upstream: `maxItems` (grouped.zig:14).
pub fn max_items(pairs: usize, experts: usize, tile: usize) -> usize {
    pairs.min(experts) + pairs / tile
}

/// Scratch the plan writes: members, items (expert, first, count),
/// counts, and the wide path's rank and hist. Upstream: `Plan`
/// (grouped.zig:18).
#[derive(Clone, Copy)]
pub struct Plan {
    pub members: u64,
    pub items: u64,
    pub counts: u64,
    pub rank: u64,
    pub hist: u64,
}

/// experts.cu's plan instances (cuobjdump -symbols of the experts
/// fatbin). Upstream: `symbols` (grouped.zig:22).
pub mod symbols {
    pub const PLAN: &str = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    pub const RANK: &str = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    pub const OFFSETS: &str = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    pub const SCATTER: &str = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
}

/// The plan kernels from a loaded experts module (kernels.experts).
/// Upstream: `Router` (grouped.zig:29) — the port borrows the module
/// (Zig copies the handle value; the caller keeps it loaded).
pub struct Router<'d> {
    pub plan_small: Function<'d>,
    pub plan_rank: Function<'d>,
    pub plan_offsets: Function<'d>,
    pub plan_scatter: Function<'d>,
}

impl<'d> Router<'d> {
    /// Upstream: `resolve` (grouped.zig:36).
    pub fn resolve(module: &Module<'d>) -> Result<Router<'d>, Error> {
        Ok(Router {
            plan_small: module.function(symbols::PLAN)?,
            plan_rank: module.function(symbols::RANK)?,
            plan_offsets: module.function(symbols::OFFSETS)?,
            plan_scatter: module.function(symbols::SCATTER)?,
        })
    }

    fn go(
        f: &Function,
        s: &Stream,
        grid: usize,
        block: u32,
        a: &mut Args,
    ) -> Result<(), Error> {
        launch(
            f,
            Config {
                grid: Dim3 {
                    x: grid as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: block, y: 1, z: 1 },
                shared: 0,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            s,
            a,
        )
    }

    /// experts.route: `pairs` int32 expert ids at `picks` grouped into
    /// items of at most `tile` pairs (16 decode). Upstream: `route`
    /// (grouped.zig:45).
    pub fn route(
        &self,
        s: &Stream,
        picks: u64,
        pairs: usize,
        count: usize,
        tile: usize,
        p: Plan,
    ) -> Result<(), Error> {
        let mut a: Args = Args::new();
        if pairs <= SMALL {
            a.add(picks);
            for v in [pairs, count, tile] {
                a.add(v as std::ffi::c_int);
            }
            for v in [p.members, p.items, p.counts] {
                a.add(v);
            }
            return Self::go(&self.plan_small, s, 1, 1024, &mut a);
        }
        let nblk = pairs.div_ceil(1024);
        a.add(picks);
        a.add(pairs as std::ffi::c_int);
        a.add(count as std::ffi::c_int);
        a.add(p.rank);
        a.add(p.hist);
        Self::go(&self.plan_rank, s, nblk, 1024, &mut a)?;
        let mut b: Args = Args::new();
        for v in [nblk, count, tile] {
            b.add(v as std::ffi::c_int);
        }
        for v in [p.hist, p.items, p.counts] {
            b.add(v);
        }
        Self::go(&self.plan_offsets, s, 1, 1024, &mut b)?;
        let mut c: Args = Args::new();
        c.add(picks);
        c.add(pairs as std::ffi::c_int);
        c.add(count as std::ffi::c_int);
        for v in [p.rank, p.hist, p.members] {
            c.add(v);
        }
        Self::go(&self.plan_scatter, s, pairs.div_ceil(256), 256, &mut c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `plan capacity as experts.max_items` (grouped.zig:73).
    #[test]
    fn plan_capacity_as_experts_max_items() {
        assert_eq!(7, max_items(7, 385, 16));
        assert_eq!(385 + 2100 / 64, max_items(2100, 385, 64));
        assert_eq!(21 + 1, max_items(21, 385, 16));
    }
}
