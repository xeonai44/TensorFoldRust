//! More lanes for one stream: continuations of its context found by suffix match,
//! grafted onto the head's chain.
//!
//! Port of `zig/src/core/lanes/fill.zig`.

/// One proposed continuation of the pending token and the planner's trust in it.
/// Upstream: `fill.zig:6  pub const Proposal`.
#[derive(Clone)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    pub weight: f64,
}

/// Host drafts appended after a window's held drafts: each draft's token and its
/// parent draft (-1: the pending row). Upstream: `fill.zig:9  pub const Branches`.
pub struct Branches {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
}

/// Distinct continuations of the context's last n tokens (n from `longest` down to
/// `shortest`), most recent match first.
///
/// Upstream: `fill.zig:25  pub fn suffixCandidates`.
pub fn suffix_candidates(
    context: &[u32],
    longest: usize,
    shortest: usize,
    span: usize,
    most: usize,
) -> Vec<Proposal> {
    let mut out: Vec<Proposal> = Vec::new();
    if context.len() < shortest + 1 || shortest == 0 {
        return out;
    }
    let mut n = longest.min(context.len() - 1);
    while n >= shortest && out.len() < most {
        let tail = &context[context.len() - n..];
        let mut i = context.len() - n;
        while i > 0 && out.len() < most {
            i -= 1;
            if &context[i..i + n] != tail {
                continue;
            }
            let next = &context[(i + n).min(context.len())..(i + n + span).min(context.len())];
            if seen(&out, next) {
                continue;
            }
            out.push(Proposal {
                tokens: next.to_vec(),
                weight: n as f64,
            });
        }
        if n == 0 {
            break;
        }
        n -= 1;
    }
    out
}

fn seen(have: &[Proposal], next: &[u32]) -> bool {
    have.iter().any(|p| p.tokens == next)
}

/// The head's chain the branches graft onto: its drafts (level by level) and the
/// head's other best tokens at each level. Upstream: `fill.zig:53  pub const
/// Trunk`.
#[derive(Default, Clone, Copy)]
pub struct Trunk<'a> {
    pub tokens: &'a [u32],
    pub others: &'a [[u32; 3]],
}

/// Proposals grafted onto the trunk, heaviest first, leaving it past its end or at
/// a token the head ranked there.
///
/// Upstream: `fill.zig:56  pub fn graft`.
pub fn graft(trunk: Trunk<'_>, proposals: &[Proposal], budget: usize) -> Branches {
    let mut tokens: Vec<u32> = Vec::new();
    let mut parents: Vec<i32> = Vec::new();
    let mut order: Vec<usize> = (0..proposals.len()).collect();
    order.sort_by(|&a, &b| {
        proposals[b]
            .weight
            .partial_cmp(&proposals[a].weight)
            .unwrap()
    });
    let first = trunk.tokens.len();
    for pi in order {
        let mut at: i32 = -1;
        let mut on_trunk = true;
        for (level, &t) in proposals[pi].tokens.iter().enumerate() {
            if on_trunk && level < first && trunk.tokens[level] == t {
                at = level as i32;
                continue;
            }
            if let Some(d) = child(&tokens, &parents, first, at, t) {
                at = d;
                on_trunk = false;
                continue;
            }
            if on_trunk && level < first && !ranked(trunk, level, t) {
                break;
            }
            if tokens.len() == budget {
                break;
            }
            tokens.push(t);
            parents.push(at);
            at = (first + tokens.len() - 1) as i32;
            on_trunk = false;
        }
    }
    Branches { tokens, parents }
}

fn ranked(trunk: Trunk<'_>, level: usize, t: u32) -> bool {
    if level >= trunk.others.len() {
        return false;
    }
    trunk.others[level].contains(&t)
}

/// The draft under `parent` already carrying `token`, if the trie has one.
/// Upstream: `fill.zig:100  fn child`.
fn child(tokens: &[u32], parents: &[i32], first: usize, parent: i32, token: u32) -> Option<i32> {
    for (k, (&t, &p)) in tokens.iter().zip(parents).enumerate() {
        if p == parent && t == token {
            return Some((first + k) as i32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors fill.zig:105: suffix candidates find what followed earlier
    /// occurrences, longest match first.
    #[test]
    fn suffix_candidates_find_earlier_continuations() {
        let ctx = [1, 2, 3, 9, 8, 1, 2, 3, 7, 6, 2, 3];
        let got = suffix_candidates(&ctx, 4, 2, 3, 8);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].tokens, [7, 6, 2]);
        assert_eq!(got[1].tokens, [9, 8, 1]);
    }

    /// Mirrors fill.zig:115: graft without a trunk keeps shared prefixes once and
    /// respects the budget.
    #[test]
    fn graft_without_trunk_keeps_prefixes_once() {
        let ps = [
            Proposal { tokens: vec![5, 6, 7], weight: 3.0 },
            Proposal { tokens: vec![5, 6, 9], weight: 2.0 },
            Proposal { tokens: vec![4], weight: 1.0 },
        ];
        let b = graft(Trunk::default(), &ps, 8);
        assert_eq!(b.tokens, [5, 6, 7, 9, 4]);
        assert_eq!(b.parents, [-1, 0, 1, 1, -1]);
        let small = graft(Trunk::default(), &ps, 3);
        assert_eq!(small.tokens, [5, 6, 7]);
    }

    /// Mirrors fill.zig:127: graft extends the head's chain and leaves it only at
    /// the head's own ranked tokens.
    #[test]
    fn graft_extends_the_chain_at_ranked_tokens() {
        let trunk = Trunk {
            tokens: &[5, 6],
            others: &[[9, 10, 11], [12, 13, 14]],
        };
        let ps = [
            Proposal { tokens: vec![5, 6, 7, 8], weight: 4.0 },   // agrees then extends
            Proposal { tokens: vec![5, 12, 3], weight: 3.0 },    // leaves at level 1
            Proposal { tokens: vec![9, 4], weight: 2.0 },        // leaves at level 0
            Proposal { tokens: vec![7, 1], weight: 1.0 },        // unranked: dropped
        ];
        let b = graft(trunk, &ps, 16);
        assert_eq!(b.tokens, [7, 8, 12, 3, 9, 4]);
        assert_eq!(b.parents, [1, 2, 0, 4, -1, 6]);
    }
}
