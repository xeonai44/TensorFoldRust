//! A lone stream's lanes each round: the head tree, or the head's chain with
//! grafted copies, that lands the most tokens a round time.
//!
//! Port of `zig/src/core/lanes/plan_lanes.zig`.

use crate::shape::{self, Odds, Shape};
use crate::table::Table;

/// Lane counts the planner weighs for a head tree (each at most the cap).
/// Upstream: `plan_lanes.zig:8  pub const sizes`.
pub const SIZES: [u32; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

/// The pick that keeps the head's chain and grafts copies onto it.
/// Upstream: `plan_lanes.zig:11  pub const chain_grafts`.
pub const CHAIN_GRAFTS: usize = SIZES.len();

/// A stream's graft prospects this round: rows its suffix matches would graft, and
/// the share of grafted rows it keeps. Upstream: `plan_lanes.zig:14`.
#[derive(Clone, Copy)]
pub struct Grafts {
    pub rows: u32,
    pub hit: f64,
}

/// Upstream: `plan_lanes.zig:16  pub const Pick`.
pub struct Pick {
    pub choice: usize, // an index into `SIZES`, or `CHAIN_GRAFTS`
    pub tree: Option<Shape>, // the head tree to draft
    pub depth: u32, // the head's chain depth (chain_grafts)
    pub grafts: u32, // grafted rows the round's window keeps
    pub expected: f64, // tokens the round is expected to land, the bonus token included
}

impl Default for Pick {
    fn default() -> Self {
        Pick { choice: 0, tree: None, depth: 0, grafts: 0, expected: 0.0 }
    }
}

/// A pick's measured round times beyond its table price: the last few, their
/// median the pick's own correction. Upstream: `plan_lanes.zig:25  Measured`.
#[derive(Clone, Copy, Default)]
pub struct Measured {
    ms: [f64; 9],
    n: usize,
}

impl Measured {
    pub fn add(&mut self, x: f64) {
        self.ms[self.n % self.ms.len()] = x;
        self.n += 1;
    }

    pub fn median(&self) -> f64 {
        let k = self.n.min(self.ms.len());
        if k == 0 {
            return 0.0;
        }
        let mut xs = self.ms;
        xs[..k].sort_by(|a, b| a.partial_cmp(b).unwrap());
        if k % 2 == 1 {
            xs[k / 2]
        } else {
            (xs[k / 2 - 1] + xs[k / 2]) / 2.0
        }
    }
}

/// A round's price: the window by rows, a head level, and the stream's time beyond
/// both (chains and trees apart). Upstream: `plan_lanes.zig:44  Costs`.
pub struct Costs<'a> {
    pub window: &'a Table,
    pub head_ms: f64,
    pub over_ms: f64,
    pub tree_over_ms: Option<f64>,
    /// Each pick's own correction (none measured: the table's price).
    pub measured: Option<&'a [Measured; SIZES.len() + 1]>,
}

impl<'a> Costs<'a> {
    fn own(&self, choice: usize) -> f64 {
        match self.measured {
            Some(m) => m[choice].median(),
            None => 0.0,
        }
    }

    pub fn round(&self, rows: usize, steps: usize) -> Option<f64> {
        let w = self.window.get(rows as i64)?;
        Some(w + self.head_ms * steps as f64 + self.over_ms)
    }

    pub fn tree_round(&self, rows: usize, steps: usize) -> Option<f64> {
        let w = self.window.get(rows as i64)?;
        Some(w + self.head_ms * steps as f64 + self.tree_over_ms.unwrap_or(self.over_ms))
    }
}

fn levels(s: &Shape) -> usize {
    s.depths.iter().copied().map(|d| d as usize).max().unwrap_or(0) + usize::from(!s.depths.is_empty())
}

/// The head's chain of `depth` drafts: tokens it is expected to land (the bonus
/// token not counted). Upstream: `plan_lanes.zig  fn chainExpected`.
fn chain_expected(o: &Odds, depth: usize) -> f64 {
    let mut total: f64 = 0.0;
    let mut reach: f64 = 1.0;
    for d in 0..depth {
        reach *= o.chance(d.min(shape::MAX_DEPTH - 1), 0);
        total += reach;
    }
    total
}

/// A pick's table price (the window, its head levels, the stream's time beyond
/// them), for measuring its own correction. Upstream: `plan_lanes.zig:85`.
pub fn table_price(c: &Costs<'_>, p: &Pick) -> Option<f64> {
    if let Some(t) = &p.tree {
        return if t.is_chain() {
            c.round(1 + t.parents.len(), levels(t))
        } else {
            c.tree_round(1 + t.parents.len(), levels(t))
        };
    }
    c.round(1 + p.depth as usize + p.grafts as usize, p.depth as usize)
}

/// The most expected tokens a round time: a head tree of each size to `cap` lanes,
/// or the 1-4 draft chain with grafts. Upstream: `plan_lanes.zig:91  pub fn pick`.
pub fn pick(
    o: &Odds,
    c: &Costs<'_>,
    cap: u32,
    depth_cap: usize,
    grafts: Option<Grafts>,
    wider: bool,
) -> Pick {
    let mut best = Pick::default();
    let mut best_rate: f64 = -1.0;
    for (i, &n) in SIZES.iter().enumerate() {
        if n > cap {
            break;
        }
        let t = shape::best(o, n as usize, depth_cap);
        let priced = if t.is_chain() {
            c.round(1 + t.parents.len(), levels(&t))
        } else {
            c.tree_round(1 + t.parents.len(), levels(&t))
        };
        let cost = match priced {
            Some(p) => p + c.own(i),
            None => continue,
        };
        let rate = (1.0 + t.expected) / cost;
        if rate > best_rate {
            let expected = 1.0 + t.expected;
            best = Pick { choice: i, tree: Some(t), depth: 0, grafts: 0, expected };
            best_rate = rate;
        }
    }
    if let Some(g) = grafts {
        if g.rows > 0 {
            for d in 1..=(4.min(cap as usize)) {
                let rows = (g.rows as usize).min((cap as usize).saturating_sub(d)) as u32;
                let cost = match c.round(1 + d + rows as usize, d) {
                    Some(v) => v + c.own(CHAIN_GRAFTS),
                    None => continue,
                };
                let expected = 1.0 + chain_expected(o, d.min(depth_cap)) + g.hit * rows as f64;
                if expected / cost > best_rate {
                    best = Pick {
                        choice: CHAIN_GRAFTS,
                        tree: None,
                        depth: d as u32,
                        grafts: rows,
                        expected,
                    };
                    best_rate = expected / cost;
                }
            }
        }
    }
    if wider && best.choice + 1 < SIZES.len() && SIZES[best.choice + 1] <= cap {
        let t = shape::best(o, SIZES[best.choice + 1] as usize, depth_cap);
        let expected = 1.0 + t.expected;
        best = Pick { choice: best.choice + 1, tree: Some(t), depth: 0, grafts: 0, expected };
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors plan_lanes.zig:130: a cheap wide window picks the widest tree, a
    /// costly one a short chain.
    #[test]
    fn a_cheap_window_picks_the_widest_tree() {
        let mut flat = Table::new();
        let mut steep = Table::new();
        for r in 1..34 {
            flat.put(r, 10.0);
            steep.put(r, 5.0 + 2.0 * r as f64);
        }
        let o = Odds::new([0.8, 0.1, 0.03, 0.01], 10.0);
        let wide = pick(
            &o,
            &Costs { window: &flat, head_ms: 0.0, over_ms: 0.0, tree_over_ms: None, measured: None },
            32,
            16,
            None,
            false,
        );
        assert_eq!(wide.choice, SIZES.len() - 1);
        let narrow = pick(
            &o,
            &Costs { window: &steep, head_ms: 0.5, over_ms: 1.0, tree_over_ms: None, measured: None },
            32,
            16,
            None,
            false,
        );
        assert!(SIZES[narrow.choice] <= 4);
    }

    /// Mirrors plan_lanes.zig:144: a pick whose own rounds run past its table price
    /// loses to one that does not.
    #[test]
    fn a_pricey_pick_loses_to_a_cheap_one() {
        let mut flat = Table::new();
        for r in 1..34 {
            flat.put(r, 10.0);
        }
        let o = Odds::new([0.8, 0.1, 0.03, 0.01], 10.0);
        let mut measured = [Measured::default(); SIZES.len() + 1];
        for _ in 0..5 {
            measured[SIZES.len() - 1].add(30.0);
        }
        let p = pick(
            &o,
            &Costs { window: &flat, head_ms: 0.0, over_ms: 0.0, tree_over_ms: None, measured: Some(&measured) },
            32,
            16,
            None,
            false,
        );
        assert!(p.choice < SIZES.len() - 1);
        assert_eq!(measured[SIZES.len() - 1].median(), 30.0);
    }

    /// Mirrors plan_lanes.zig:157: grafts that keep landing win the round when they
    /// add more rows than the head would.
    #[test]
    fn grafts_that_keep_landing_win() {
        let mut costs = Table::new();
        for r in 1..34 {
            costs.put(r, 5.0 + 0.5 * r as f64);
        }
        let o = Odds::new([0.6, 0.1, 0.03, 0.01], 10.0);
        let p = pick(
            &o,
            &Costs { window: &costs, head_ms: 0.5, over_ms: 1.0, tree_over_ms: None, measured: None },
            32,
            16,
            Some(Grafts { rows: 20, hit: 0.95 }),
            false,
        );
        assert_eq!(p.choice, CHAIN_GRAFTS);
        assert!(p.grafts == 20 && p.depth >= 1);
    }
}
