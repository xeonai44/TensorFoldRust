//! A head tree's shape for a stream's lanes: which of the head's ranked tokens each
//! lane takes, from the stream's odds.
//!
//! Port of `zig/src/core/lanes/shape.zig`.

/// Upstream: `shape.zig:5-6`.
pub const MAX_DEPTH: usize = 16;
pub const RANKS: usize = 4;

/// How often the target took the head's rank-r token at each depth when that lane
/// was verified (with a prior).
///
/// Upstream: `shape.zig:9  pub const Odds = struct { took, seen }`.
#[derive(Clone, Copy, Debug)]
pub struct Odds {
    took: [[f64; RANKS]; MAX_DEPTH],
    seen: [[f64; RANKS]; MAX_DEPTH],
}

impl Odds {
    /// Odds that start from `prior` (the chance for each rank, the same at every
    /// depth) worth `weight` lanes. Upstream: `init`.
    pub fn new(prior: [f64; RANKS], weight: f64) -> Odds {
        let mut o = Odds {
            took: [[0.0; RANKS]; MAX_DEPTH],
            seen: [[0.0; RANKS]; MAX_DEPTH],
        };
        for d in 0..MAX_DEPTH {
            for (r, &p) in prior.iter().enumerate() {
                o.took[d][r] = p * weight;
                o.seen[d][r] = weight;
            }
        }
        o
    }

    /// A verified lane at `depth` and `rank`: whether the target took it.
    /// Upstream: `observe`.
    pub fn observe(&mut self, depth: usize, rank: usize, took: bool) {
        if depth >= MAX_DEPTH || rank >= RANKS {
            return;
        }
        self.seen[depth][rank] += 1.0;
        if took {
            self.took[depth][rank] += 1.0;
        }
    }

    /// Upstream: `chance`.
    pub fn chance(&self, depth: usize, rank: usize) -> f64 {
        self.took[depth][rank] / self.seen[depth][rank]
    }
}

/// Draft lanes as a tree: each lane's parent lane (-1: the pending row), the head's
/// rank it takes, its depth (0: first).
///
/// Upstream: `shape.zig:36  pub const Shape = struct { parents, ranks, depths,
/// expected }`.
pub struct Shape {
    pub parents: Vec<i32>,
    pub ranks: Vec<u8>,
    pub depths: Vec<u8>,
    /// Tokens the lanes are expected to land (the bonus token not counted).
    pub expected: f64,
}

impl Shape {
    /// Levels the tree spans: its deepest lane's depth plus one. Upstream:
    /// `levels`.
    pub fn levels(&self) -> u32 {
        self.depths.iter().copied().max().unwrap_or(0) as u32 + 1
    }

    /// Upstream: `isChain`.
    pub fn is_chain(&self) -> bool {
        self.ranks.iter().all(|&r| r == 0)
    }
}

/// The odds a stream starts from: the head's chance by rank (any depth), worth a few
/// lanes until its own come in. Upstream: `shape.zig:62-63`.
pub const START_ODDS: [f64; RANKS] = [0.75, 0.08, 0.02, 0.01];
pub const START_WEIGHT: f64 = 4.0;

/// The lanes of the shape's chain (rank 0 from the pending row down), in depth
/// order; their count. Upstream: `shape.zig:66  pub fn chainLanes`.
pub fn chain_lanes(s: &Shape, out: &mut [usize]) -> usize {
    let mut n: usize = 0;
    let mut at: i32 = -1;
    while n < out.len() {
        let next = s
            .parents
            .iter()
            .zip(&s.ranks)
            .position(|(&p, &r)| p == at && r == 0);
        let next = match next {
            Some(i) => i,
            None => return n,
        };
        out[n] = next;
        at = next as i32;
        n += 1;
    }
    n
}

/// The heap never compares the candidate itself (the ordering rides the value and
/// the push sequence in the tuple), but `BinaryHeap` needs the bound; the derived
/// order over (parent, depth, rank, reach-as-bits, value-as-bits) is a stable
/// tiebreak no caller can observe.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    parent: i32,
    depth: u8,
    rank: u8,
    reach: u64,
    value: u64,
}

/// The `lanes` drafts with the most expected tokens, best first: a lane's value is
/// its chance of being reached and taken.
///
/// Upstream: `shape.zig:87  pub fn best`. The priority queue order (highest value
/// first, ties by insertion) is reproduced with a binary heap of `Reverse`-keyed
/// candidates; upstream's `std.PriorityQueue` pops the highest `value` with ties
/// broken by push order, which `BinaryHeap` on (value, seq) matches.
pub fn best(o: &Odds, lanes: usize, depth_cap: usize) -> Shape {
    let mut parents: Vec<i32> = Vec::new();
    let mut rs: Vec<u8> = Vec::new();
    let mut ds: Vec<u8> = Vec::new();
    // The heap pops the greatest (value, seq): highest value first, ties to the
    // earlier push (upstream's std.PriorityQueue keeps insertion order among
    // equals; Reverse makes the smaller seq the greater pair).
    // The value rides as its bits: positive floats order as their bits, so the
    // heap's pops match upstream's value-first, insertion-order-ties order.
    let mut queue: std::collections::BinaryHeap<(u64, std::cmp::Reverse<u64>, Candidate)> =
        std::collections::BinaryHeap::new();
    let cap = depth_cap.min(MAX_DEPTH);
    let mut expected: f64 = 0.0;
    let mut seq: u64 = 0;
    if lanes > 0 && cap > 0 {
        let first = Candidate { parent: -1, depth: 0, rank: 0, reach: 1.0f64.to_bits(), value: o.chance(0, 0).to_bits() };
        queue.push((o.chance(0, 0).to_bits(), std::cmp::Reverse(seq), first));
        seq += 1;
    }
    while parents.len() < lanes {
        let Some((_, _, c)) = queue.pop() else { break };
        let lane = parents.len() as i32;
        parents.push(c.parent);
        rs.push(c.rank);
        ds.push(c.depth);
        expected += f64::from_bits(c.value);
        if (c.rank as usize) + 1 < RANKS {
            let next = Candidate {
                parent: c.parent,
                depth: c.depth,
                rank: c.rank + 1,
                reach: c.reach,
                value: (f64::from_bits(c.reach) * o.chance(c.depth as usize, c.rank as usize + 1)).to_bits(),
            };
            queue.push((next.value, std::cmp::Reverse(seq), next));
            seq += 1;
        }
        if (c.depth as usize) + 1 < cap {
            let next = Candidate {
                parent: lane,
                depth: c.depth + 1,
                rank: 0,
                reach: c.value,
                value: (f64::from_bits(c.value) * o.chance(c.depth as usize + 1, 0)).to_bits(),
            };
            queue.push((next.value, std::cmp::Reverse(seq), next));
            seq += 1;
        }
    }
    Shape { parents, ranks: rs, depths: ds, expected }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors shape.zig:112: odds that only the head's first choice lands give a
    /// chain.
    #[test]
    fn only_first_choice_lands_gives_a_chain() {
        let s = best(&Odds::new([0.7, 0.0, 0.0, 0.0], 10.0), 5, MAX_DEPTH);
        assert!(s.is_chain());
        assert_eq!(s.parents, [-1, 0, 1, 2, 3]);
        assert!((s.expected - (0.7 + 0.49 + 0.343 + 0.2401 + 0.16807)).abs() < 1e-9);
    }

    /// Mirrors shape.zig:121: a second choice joins once it beats the chain's next
    /// lane, and lanes follow their parents.
    #[test]
    fn a_second_choice_joins_once_it_beats_the_chain() {
        let s = best(&Odds::new([0.6, 0.3, 0.0, 0.0], 10.0), 4, MAX_DEPTH);
        // values: d0r0 .6, d1r0 .36, d0r1 .3, d1r1 under the first .18, d2r0 .216,
        // d1r0 under the second .18
        assert_eq!(s.parents, [-1, 0, -1, 1]);
        assert_eq!(s.ranks, [0, 0, 1, 0]);
        assert_eq!(s.depths, [0, 1, 0, 2]);
        for (i, &p) in s.parents.iter().enumerate() {
            assert!(p < i as i32);
        }
    }

    /// Mirrors shape.zig:132: observed lanes move the odds from the prior.
    #[test]
    fn observed_lanes_move_the_odds() {
        let mut o = Odds::new([0.5, 0.1, 0.0, 0.0], 2.0);
        for _ in 0..8 {
            o.observe(0, 0, true);
        }
        o.observe(0, 1, false);
        assert!((o.chance(0, 0) - (1.0 + 8.0) / (2.0 + 8.0)).abs() < 1e-12);
        assert!((o.chance(0, 1) - 0.2 / 3.0).abs() < 1e-12);
        assert_eq!(o.chance(1, 0), 0.5);
    }
}
