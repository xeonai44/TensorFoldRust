//! Choose one LRU victim set only when it can satisfy both disk floors
//! and the learned-store cap.
//!
//! Port of `zig/src/core/learned_plan.zig` (45 lines): the planner behind
//! a learned write — the candidates (this identity's states, the other
//! identities' trees) ordered least recently used first, victims chosen
//! only while the need stands, and the whole plan refused when even every
//! candidate together cannot cover it (deleting nothing then).

/// One candidate: what its deletion frees on each rank and under the cap.
/// Upstream: `Candidate` (learned_plan.zig:3).
#[derive(Clone, Copy)]
pub struct Candidate {
    pub key: u64,
    pub local: u64,
    pub peer: u64,
    pub cap: u64,
    pub used: u64,
}

/// The write's needs across the ranks and the cap. Upstream: `Need`
/// (learned_plan.zig:4).
#[derive(Clone, Copy, Default)]
pub struct Need {
    pub local: u64,
    pub peer: u64,
    pub cap: u64,
}

fn fits(have: Need, need: Need) -> bool {
    have.local >= need.local && have.peer >= need.peer && have.cap >= need.cap
}

/// The victim keys, oldest first, or None when the need cannot be met.
/// Upstream: `choose` (learned_plan.zig:8).
pub fn choose(candidates: &[Candidate], need: Need) -> Option<Vec<u64>> {
    let mut total = Need::default();
    for c in candidates {
        total.local = total.local.saturating_add(c.local);
        total.peer = total.peer.saturating_add(c.peer);
        total.cap = total.cap.saturating_add(c.cap);
    }
    if !fits(total, need) {
        return None;
    }
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|x, y| {
        x.used
            .cmp(&y.used)
            .then_with(|| x.key.cmp(&y.key))
    });
    let mut victims: Vec<u64> = Vec::new();
    let mut have = Need::default();
    for c in &ordered {
        if fits(have, need) {
            break;
        }
        victims.push(c.key);
        have.local = have.local.saturating_add(c.local);
        have.peer = have.peer.saturating_add(c.peer);
        have.cap = have.cap.saturating_add(c.cap);
    }
    Some(victims)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's test (learned_plan.zig:37), case for case: both ranks
    /// use one victim set, and an impossible shortfall deletes nothing.
    #[test]
    fn both_ranks_use_one_victim_set_and_an_impossible_shortfall_deletes_nothing() {
        let items = [
            Candidate { key: 1, local: 10, peer: 5, cap: 10, used: 1 },
            Candidate { key: 2, local: 10, peer: 25, cap: 10, used: 2 },
            Candidate { key: 3, local: 50, peer: 0, cap: 50, used: 3 },
        ];
        let selected = choose(
            &items,
            Need { local: 20, peer: 30, cap: 0 },
        )
        .unwrap();
        assert_eq!(selected, vec![1, 2]);
        assert!(choose(&items, Need { local: 1, peer: 31, cap: 0 }).is_none());
        let empty = choose(&items, Need::default()).unwrap();
        assert!(empty.is_empty());
    }
}
