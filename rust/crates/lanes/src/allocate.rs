//! Shared rows by landing probability and forward costs between timed widths
//! (Python allocate.py, extend_costs).
//!
//! Port of `zig/src/core/lanes/allocate.zig`.

use crate::table::Table;

/// Upstream: `allocate.zig:6  pub const Cost = struct { width, ms }`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub width: u32,
    pub ms: f64,
}

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always
/// run.
///
/// Upstream: `allocate.zig:9  pub fn allocate`. Returns the counts, one a stream.
pub fn allocate(
    fixed: &[u32],
    probs: &[&[f64]],
    costs: &Table,
    overhead_ms: f64,
    max_rows: u64,
) -> Vec<u32> {
    let mut counts = vec![0u32; probs.len()];
    let mut best_counts = vec![0u32; probs.len()];
    let mut rows: u64 = fixed.iter().map(|&f| f as u64).sum();
    let mut expected: f64 = rows as f64;
    let timed = !costs.is_empty();
    let mut best = rate(costs, timed, rows, expected, overhead_ms);
    while rows < max_rows {
        // The likeliest next node, ties to the lowest stream (Python's heapq of
        // (-p, stream)).
        let mut pick: Option<usize> = None;
        let mut pick_p: f64 = 0.0;
        for (s, p) in probs.iter().enumerate() {
            if counts[s] as usize >= p.len() {
                continue;
            }
            let v = p[counts[s] as usize];
            if pick.is_none() || v > pick_p {
                pick = Some(s);
                pick_p = v;
            }
        }
        let s = match pick {
            Some(s) => s,
            None => break,
        };
        counts[s] += 1;
        rows += 1;
        expected += pick_p;
        let now = rate(costs, timed, rows, expected, overhead_ms);
        if now > best {
            best = now;
            best_counts.copy_from_slice(&counts);
        }
    }
    best_counts
}

fn rate(costs: &Table, timed: bool, total: u64, expected: f64, overhead_ms: f64) -> f64 {
    let cost: f64 = if timed {
        match costs.get(total as i64) {
            Some(ms) => ms,
            None => return -1.0,
        }
    } else {
        1.0
    };
    expected / (cost + overhead_ms)
}

/// A chain's nodes' chances from per-depth acceptance (the j-th lands if every
/// earlier one did).
///
/// Upstream: `allocate.zig:52  pub fn chainProbabilities`.
pub fn chain_probabilities(rates: &[f64], count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    let mut reach: f64 = 1.0;
    for j in 0..count {
        reach *= if j < rates.len() {
            rates[j]
        } else {
            rates[rates.len() - 1]
        };
        out.push(reach);
    }
    out
}

/// Costs for 1..=rows: timed widths as measured, interpolated between them, the
/// widest's ms a row past it.
///
/// Upstream: `allocate.zig:63  pub fn extendCosts`.
pub fn extend_costs(costs: &[Cost], rows: u32) -> Table {
    let mut out = Table::new();
    if costs.is_empty() {
        return out;
    }
    let mut measured: Vec<Cost> = costs.to_vec();
    measured.sort_by_key(|c| c.width);
    let widest = measured[measured.len() - 1];
    for total in 1..=rows {
        let value = if let Some(ms) = find(&measured, total) {
            ms
        } else if total > widest.width {
            widest.ms * total as f64 / widest.width as f64
        } else {
            let above = measured.iter().copied().find(|c| c.width > total);
            let below = measured.iter().copied().rev().find(|c| c.width < total);
            match (above, below) {
                (Some(above), Some(b)) => {
                    let span = (above.width - b.width) as f64;
                    b.ms + (above.ms - b.ms) * (total - b.width) as f64 / span
                }
                (Some(above), None) => above.ms,
                _ => unreachable!("a total between 1 and the widest has a cost"),
            }
        };
        out.put(total as i64, value);
    }
    out
}

fn find(costs: &[Cost], width: u32) -> Option<f64> {
    costs.iter().find(|c| c.width == width).map(|c| c.ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors allocate.zig:108 "allocate takes the widest round when no cost is
    /// timed".
    #[test]
    fn allocate_takes_the_widest_when_no_cost_is_timed() {
        let empty = Table::new();
        let probs: [&[f64]; 2] = [&[0.9, 0.5], &[0.8]];
        let counts = allocate(&[1, 1], &probs, &empty, 0.0, 8);
        assert_eq!(counts, [2, 1]);
    }

    /// Mirrors allocate.zig:117 "extendCosts interpolates and extrapolates".
    #[test]
    fn extend_costs_interpolates_and_extrapolates() {
        let t = extend_costs(
            &[Cost { width: 4, ms: 8.0 }, Cost { width: 2, ms: 4.0 }],
            6,
        );
        assert_eq!(t.get(1), Some(4.0));
        assert_eq!(t.get(3), Some(6.0));
        assert_eq!(t.get(6), Some(12.0));
    }
}
