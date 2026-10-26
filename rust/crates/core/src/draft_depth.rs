//! How deep a draft chain runs (cuda/draft_depth.py): a draft is
//! verified while its chance of landing pays for its row.
//!
//! Port of `zig/src/core/draft_depth.zig` (104 lines).

pub const MAX_ROWS: usize = 16;

/// Measured ms on this GPU: `verify[r]` for a window of r rows (index
/// 0 unused), `level` for one chain level. Upstream: `Costs`
/// (draft_depth.zig:8).
#[derive(Clone, Copy)]
pub struct Costs {
    pub verify: [f64; MAX_ROWS + 1],
    pub rows: usize,
    pub level: f64,
}

impl Default for Costs {
    fn default() -> Costs {
        Costs {
            verify: [0.0; MAX_ROWS + 1],
            rows: 0,
            level: 0.0,
        }
    }
}

impl Costs {
    /// Timed costs with each extra row's ms pooled so it never rises
    /// with rows, as shared experts make them. Upstream: `measured`
    /// (draft_depth.zig:14).
    pub fn measured(verify: &[f64], level: f64) -> Costs {
        let mut sums = [0.0f64; MAX_ROWS];
        let mut counts = [0.0f64; MAX_ROWS];
        let mut n: usize = 0;
        for (a, b) in verify[1..verify.len() - 1].iter().zip(verify[2..].iter()) {
            sums[n] = b - a;
            counts[n] = 1.0;
            n += 1;
            while n > 1 && sums[n - 2] * counts[n - 1] < sums[n - 1] * counts[n - 2] {
                sums[n - 2] += sums[n - 1];
                counts[n - 2] += counts[n - 1];
                n -= 1;
            }
        }
        let mut c = Costs {
            verify: [0.0; MAX_ROWS + 1],
            rows: verify.len() - 1,
            level: round4(level.max(0.0)),
        };
        c.verify[1] = verify[1];
        let mut at: usize = 2;
        for (s, k) in sums[..n].iter().zip(counts[..n].iter()) {
            let base = c.verify[at - 1];
            let mut i: usize = 0;
            while i < *k as usize {
                c.verify[at] = base + s / k * (i as f64 + 1.0);
                at += 1;
                i += 1;
            }
        }
        for v in &mut c.verify[1..at] {
            *v = round4(*v);
        }
        c
    }

    /// The ms the row carrying draft `k` adds to its window. Upstream:
    /// `row` (draft_depth.zig:44).
    pub fn row(&self, k: usize) -> f64 {
        (self.verify[k + 1] - self.verify[k]).max(0.0)
    }
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// The first draft, then draft k while P_k ** power >= tokens a ms
/// earned x its row's ms (or P_k >= `floor`). Upstream: `DepthRule`
/// (draft_depth.zig:54).
pub struct DepthRule {
    pub costs: Costs,
    pub most: usize,
    pub floor: Option<f64>,
    pub power: f64,
    pub weight: f64,
    pub tokens: f64,
    pub ms: f64,
}

impl DepthRule {
    /// Upstream: `init` (draft_depth.zig:63).
    pub fn init(costs: Costs, most: usize, floor: Option<f64>, power: f64) -> DepthRule {
        assert!((1..costs.rows).contains(&most));
        let ms = costs.verify[2] + costs.level;
        DepthRule {
            costs,
            most,
            floor,
            power,
            weight: 0.125,
            tokens: 2.0,
            ms,
        }
    }

    /// Tokens a ms the recent rounds earned, by the cost model.
    /// Upstream: `rate` (draft_depth.zig:69).
    pub fn rate(&self) -> f64 {
        self.tokens / self.ms
    }

    /// Verify draft `k` whose running confidence is `run` (every
    /// drafted window verifies its first draft). Upstream: `keep`
    /// (draft_depth.zig:74).
    pub fn keep(&self, k: usize, run: f64) -> bool {
        if k == 1 {
            return true;
        }
        if let Some(f) = self.floor {
            return run >= f;
        }
        run.powf(self.power) >= self.rate() * self.costs.row(k)
    }

    /// Draft level k + 1 only if a certain draft there would pay for
    /// its row and its level. Upstream: `more` (draft_depth.zig:81).
    pub fn more(&self, k: usize, run: f64) -> bool {
        if k >= self.most {
            return false;
        }
        if let Some(f) = self.floor {
            return run >= f;
        }
        run.powf(self.power) >= self.rate() * (self.costs.row(k + 1) + self.costs.level)
    }

    /// Follow a round from its modeled cost, never a clock, so every
    /// rank of a split model chooses alike. Upstream: `done`
    /// (draft_depth.zig:88).
    pub fn done(&mut self, tokens: usize, rows: usize, levels: usize) {
        self.tokens += self.weight * (tokens as f64 - self.tokens);
        self.ms +=
            self.weight * (self.costs.verify[rows] + levels as f64 * self.costs.level - self.ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `pooled costs never add more ms a row as rows grow`
    /// (draft_depth.zig:94).
    #[test]
    fn pooled_costs_never_add_more_ms_a_row_as_rows_grow() {
        let v = [
            0.0,
            9.85,
            12.44,
            14.72,
            16.92,
            18.51,
            20.11,
            21.7,
            23.03,
            24.36,
            25.69,
            26.95,
            28.22,
            29.33,
            30.36,
            31.22,
            32.08,
        ];
        let c = Costs::measured(&v, 0.6132);
        assert_eq!(16, c.rows);
        for k in 2..16 {
            assert!(c.row(k) <= c.row(k - 1) + 1e-9);
        }
        assert!((32.08 - c.verify[16]).abs() <= 1e-9);
        let mut r = DepthRule::init(c, 15, None, 2.0);
        assert!(r.keep(1, 0.0) && r.keep(2, 0.99) && !r.keep(2, 0.1));
        r.done(4, 4, 3);
        assert!(r.tokens > 2.0);
    }
}
