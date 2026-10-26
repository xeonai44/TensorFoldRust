//! decode.CopyIndex: copy drafts' search over a context, by each
//! token's positions (longest match, or the 8-token chain).
//!
//! Port of `zig/src/core/copy_index.zig` (159 lines) — upstream's
//! three tests mirrored, the fuzz test's random context driven by the
//! port's own xorshift (the contract holds for every context).

use std::collections::HashMap;

pub const MIN_MATCH: usize = 8;

/// A match's length, and where the tokens that followed it start.
/// Upstream: `Match` (copy_index.zig:8).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Match {
    pub n: usize,
    pub at: usize,
}

/// Upstream: `CopyIndex` (copy_index.zig:10).
pub struct CopyIndex {
    ctx: Vec<u32>,
    /// Each token's positions in ctx.
    seen: HashMap<u32, Vec<u32>>,
}

impl CopyIndex {
    /// Upstream: `init` (copy_index.zig:15).
    pub fn init(context: &[u32]) -> CopyIndex {
        let mut x = CopyIndex {
            ctx: Vec::new(),
            seen: HashMap::new(),
        };
        x.extend(context);
        x
    }

    /// Upstream: `extend` (copy_index.zig:29).
    pub fn extend(&mut self, tokens: &[u32]) {
        for t in tokens {
            self.seen.entry(*t).or_default().push(self.ctx.len() as u32);
            self.ctx.push(*t);
        }
    }

    /// The longest earlier match of the context's last tokens, up to
    /// `max_n` deep (the latest of equal length). Upstream: `longest`
    /// (copy_index.zig:39).
    pub fn longest(&self, max_n: usize) -> Match {
        let ctx = &self.ctx;
        let mut best = Match::default();
        if ctx.len() < 2 || max_n < 1 {
            return best;
        }
        let last = ctx.len() - 1;
        let list = match self.seen.get(&ctx[last]) {
            Some(l) => l,
            None => return best,
        };
        for i in (0..list.len()).rev() {
            let p = list[i] as usize;
            if p >= last {
                continue;
            }
            let mut n: usize = 1;
            while n < max_n && p >= n && ctx[p - n] == ctx[last - n] {
                n += 1;
            }
            if n > best.n {
                best = Match { n, at: p + 1 };
                if n == max_n {
                    break;
                }
            }
        }
        best
    }

    /// Up to `out.len` tokens that followed the latest earlier copy of
    /// the last 8 (none shorter than 8), copied out. Upstream: `chain`
    /// (copy_index.zig:61).
    pub fn chain<'o>(&self, out: &'o mut [u32]) -> &'o [u32] {
        let max_nodes = out.len();
        let ctx = &self.ctx;
        if ctx.len() < 2 * MIN_MATCH || max_nodes < 1 {
            return &[];
        }
        let key = &ctx[ctx.len() - MIN_MATCH..];
        let list = match self.seen.get(&key[MIN_MATCH - 1]) {
            Some(l) => l,
            None => return &[],
        };
        let mut best: &[u32] = &[];
        for i in (0..list.len()).rev() {
            let p = list[i] as usize; // the copy's last token
            if p + 1 < MIN_MATCH || p + 1 >= ctx.len() {
                continue;
            }
            if ctx[p + 1 - MIN_MATCH..p + 1] != *key {
                continue;
            }
            let cont = &ctx[p + 1..ctx.len().min(p + 1 + max_nodes)];
            if cont.len() > best.len() {
                best = cont;
                if best.len() == max_nodes {
                    break;
                }
            }
        }
        if best.len() < MIN_MATCH {
            return &[];
        }
        out[..best.len()].copy_from_slice(best);
        &out[..best.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `copies the continuation of the latest earlier
    /// match` (copy_index.zig:86).
    #[test]
    fn copies_the_continuation_of_the_latest_earlier_match() {
        let seq: Vec<u32> = (0..40).map(|i| (i % 20) as u32).collect();
        let x = CopyIndex::init(&seq);
        let mut buf = [0u32; 15];
        let got = x.chain(&mut buf);
        assert_eq!(15, got.len());
        assert_eq!(0, got[0]);
        assert_eq!(0, x.chain(&mut buf[..0]).len());
    }

    /// Plain scans of the whole context: what `longest` and `chain`
    /// must return (upstream's own reference, copy_index.zig:100).
    fn scan_longest(ctx: &[u32], max_n: usize) -> Match {
        let mut best = Match::default();
        if ctx.len() < 2 || max_n < 1 {
            return best;
        }
        let last = ctx.len() - 1;
        let mut p = last;
        while p > 0 {
            p -= 1;
            let mut n: usize = 0;
            while n < max_n && p >= n && ctx[p - n] == ctx[last - n] {
                n += 1;
            }
            if n > best.n {
                best = Match { n, at: p + 1 };
                if n == max_n {
                    break;
                }
            }
        }
        best
    }

    fn scan_chain(ctx: &[u32], max_nodes: usize) -> &[u32] {
        if ctx.len() < 2 * MIN_MATCH || max_nodes < 1 {
            return &[];
        }
        let key = &ctx[ctx.len() - MIN_MATCH..];
        let mut best: &[u32] = &[];
        let mut s = ctx.len() - MIN_MATCH;
        while s > 0 {
            s -= 1;
            if ctx[s..s + MIN_MATCH] != *key {
                continue;
            }
            let cont = &ctx[s + MIN_MATCH..ctx.len().min(s + MIN_MATCH + max_nodes)];
            if cont.len() > best.len() {
                best = cont;
                if best.len() == max_nodes {
                    break;
                }
            }
        }
        if best.len() < MIN_MATCH {
            &[]
        } else {
            best
        }
    }

    /// A tiny xorshift, as the fuzz test's generator.
    struct Xorshift(u64);
    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: u32, hi: u32) -> u32 {
            lo + (self.next() % (hi - lo + 1) as u64) as u32
        }
    }

    /// Upstream's `longest and chain match plain scans as the context
    /// grows` (copy_index.zig:134) — the index against the scan at
    /// every step, two hundred random contexts.
    #[test]
    fn longest_and_chain_match_plain_scans_as_the_context_grows() {
        let mut rng = Xorshift(7 | 1);
        for _ in 0..200 {
            let symbols = rng.range(1, 4);
            let mut x = CopyIndex::init(&[]);
            for _ in 0..rng.range(1, 300) {
                let t = (rng.next() % symbols as u64) as u32;
                x.extend(&[t]);
                let ctx = x.ctx.clone();
                for max_n in [1usize, 3, 8] {
                    let want = scan_longest(&ctx, max_n);
                    let got = x.longest(max_n);
                    assert_eq!(want.n, got.n);
                    if want.n > 0 {
                        assert_eq!(want.at, got.at);
                    }
                }
                let mut buf = [0u32; 20];
                for m in [1usize, 9, 20] {
                    let want = scan_chain(&ctx, m);
                    let got = x.chain(&mut buf[..m]);
                    assert_eq!(want, got);
                }
            }
        }
    }
}
