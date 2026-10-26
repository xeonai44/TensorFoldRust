//! A request's sampled rule as sample.cu reads it (the Metal engine's
//! draws), and the greedy head draw's _keyed rule.
//!
//! Port of `zig/src/families/nemotron/cuda_sampler.zig` (44 lines).

// Upstream: `pub const Sampling = @import("lanes").Sampling`
// (cuda_sampler.zig:7).
pub use tensorfold_lanes::Sampling;

use crate::cuda_triton::Keyed;

/// exact_sampling.MARGIN: candidates read beyond top_k. Upstream:
/// `margin` (cuda_sampler.zig:9).
pub const MARGIN: usize = 8;

/// The most candidates one _keyed program ranks. Upstream:
/// `max_candidates` (cuda_sampler.zig:11).
pub const MAX_CANDIDATES: usize = 256;

/// Candidates a draw takes from `vocab` columns. Upstream: `count`
/// (cuda_sampler.zig:14).
pub fn count(k: usize, vocab: usize) -> usize {
    vocab.min(k + MARGIN)
}

/// The request's rule as the engine runs it: null (or temperature 0)
/// decodes greedily. Upstream: `check` (cuda_sampler.zig:18).
pub fn check(s: Option<Sampling>) -> Option<Sampling> {
    let x = s?;
    if x.temperature > 0.0 {
        Some(x)
    } else {
        None
    }
}

/// The greedy head draw: the top 20 ranked, the argmax and its share
/// written (captured _keyed). Upstream: `greedy_draft`
/// (cuda_sampler.zig:22).
pub const GREEDY_DRAFT: Keyed = Keyed {
    k: 20,
    cut: false,
    greedy: true,
    minp: false,
    conf_t: false,
};

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near
/// window, ln(min_p) and top_k (0: off). Upstream: `Rule`
/// (cuda_sampler.zig:25).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Rule {
    pub seed: u64,
    pub inv_t: f32,
    pub top_p: f32,
    pub near: f32,
    pub min_log: f32,
    pub top_k: u32,
    pub pad: u32,
}

impl Default for Rule {
    fn default() -> Rule {
        Rule {
            seed: 0,
            inv_t: 0.0,
            top_p: 0.0,
            near: 0.0,
            min_log: 0.0,
            top_k: 0,
            pad: 0,
        }
    }
}

/// The rule the Metal engine's sample kernels read for `s`, as its host
/// fills cfg. Upstream: `rule` (cuda_sampler.zig:28).
pub fn rule(s: Sampling) -> Rule {
    Rule {
        seed: s.seed,
        inv_t: (1.0 / s.temperature.max(1e-6)) as f32,
        top_p: s.top_p as f32,
        near: 20.0,
        min_log: s.min_log() as f32,
        top_k: s.top_k,
        pad: 0,
    }
}

// The Rule is what sample.cu reads: its size fixed.
const _: () = assert!(std::mem::size_of::<Rule>() == 32);

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `rules as the Metal engine fills them`
    /// (cuda_sampler.zig:31).
    #[test]
    fn rules_as_the_metal_engine_fills_them() {
        let s = Sampling {
            seed: 1 << 63 | 5,
            temperature: 0.5,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.05,
        };
        let r = rule(check(Some(s)).unwrap());
        assert_eq!(1 << 63 | 5, r.seed);
        assert_eq!(2.0f32, r.inv_t);
        assert_eq!(s.min_log() as f32, r.min_log);
        assert_eq!(32, std::mem::size_of::<Rule>());
        assert!(check(Some(Sampling {
            seed: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
        }))
        .is_none());
        assert!(check(None).is_none());
        assert_eq!(28, count(20, 131072));
    }
}
