//! Where a prompt's prefill chunks start (Python's PrefillPlan and
//! message_markers): prompts that agree up to a start cut alike up to it.
//!
//! Port of `zig/src/server/chunk_plan.zig` (169 lines).

use crate::state::Text;
use serde_json::{json, Value};

/// Chunk starts: 0, then the first resume point `min_chunk` or more tokens
/// on, else `step` tokens on. Upstream: `Plan` (chunk_plan.zig:9).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// 0: the engine cuts prompts itself.
    pub step: u32,
    /// The special tokens that open any message.
    pub openers: Vec<u32>,
    /// The tokens that open an assistant message.
    pub assistant: Vec<u32>,
    pub min_chunk: u32,
}

impl Plan {
    pub fn new(step: u32) -> Plan {
        Plan {
            step,
            openers: Vec::new(),
            assistant: Vec::new(),
            min_chunk: 256,
        }
    }

    /// Resume points: every assistant message's start, and the second
    /// message's. Upstream: `points` (chunk_plan.zig:17).
    fn points(&self, ids: &[u32]) -> Vec<u32> {
        let mut found: Vec<u32> = Vec::new();
        if !self.openers.is_empty() {
            let mut seen = 0;
            for (i, &t) in ids.iter().enumerate() {
                if !self.openers.contains(&t) {
                    continue;
                }
                seen += 1;
                if seen == 2 {
                    found.push(i as u32);
                    break;
                }
            }
        }
        let k = self.assistant.len();
        if k > 0 && ids.len() >= k {
            for i in 0..ids.len() - k + 1 {
                if ids[i..i + k] == self.assistant[..] {
                    found.push(i as u32);
                }
            }
        }
        found.sort_unstable();
        let mut out: Vec<u32> = Vec::new();
        for q in found {
            if q > 0 && out.last().copied() != Some(q) {
                out.push(q);
            }
        }
        out
    }

    /// The starts after 0 of `ids`' chunks; empty with no step. Upstream:
    /// `starts` (chunk_plan.zig:36).
    pub fn starts(&self, ids: &[u32]) -> Vec<u32> {
        if self.step == 0 {
            return Vec::new();
        }
        let found = if (!self.openers.is_empty() || !self.assistant.is_empty()) && ids.len() > 1 {
            self.points(ids)
        } else {
            Vec::new()
        };
        let mut out: Vec<u32> = Vec::new();
        let mut last: usize = 0;
        let mut i = 0;
        loop {
            // Too close to the last start: merged into its chunk
            while i < found.len() && (found[i] as usize) < last + self.min_chunk as usize {
                i += 1;
            }
            let mut next = last + self.step as usize;
            if i < found.len() && (found[i] as usize) < next {
                next = found[i] as usize;
            }
            if next >= ids.len() {
                return out;
            }
            out.push(next as u32);
            last = next;
        }
    }
}

/// `starts` with `cut` too: a state kept just before a conversation's own
/// text reads only shared tokens. Upstream: `withCut` (chunk_plan.zig:50).
pub fn with_cut(starts: &[u32], cut: u32, len: usize, min_chunk: u32) -> Vec<u32> {
    if cut == 0 || cut as usize >= len {
        return starts.to_vec();
    }
    let mut out: Vec<u32> = starts
        .iter()
        .filter(|&s| s.abs_diff(cut) >= min_chunk)
        .copied()
        .collect();
    out.push(cut);
    out.sort_unstable();
    out
}

/// ``message_markers``: the openers and assistant header the chat template
/// renders, from probe conversations. Upstream: `markers`
/// (chunk_plan.zig:58) — four short exchanges rendered pairwise, the pieces
/// every message of a role starts with; a template that renders the last
/// reply differently gains its opener at the next marker past the
/// difference.
pub fn markers(text: &Text, step: u32) -> Plan {
    let render = |talk: &[Value], thinking: bool, generation: bool| -> Option<Vec<u32>> {
        let ids = render_probe(text, talk, thinking, generation)?;
        Some(ids)
    };
    let mut pieces: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
    let mut parted: Vec<(Vec<u32>, Vec<u32>, usize)> = Vec::new();
    let words = [
        ["Alpha", "Beta", "Gamma", "Delta"],
        ["one two", "three four", "five six", "seven eight"],
    ];
    for set in words {
        for thinking in [false, true] {
            let talk: Vec<Value> = set
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": text})
                })
                .collect();
            let r: Vec<Option<Vec<u32>>> = (1..=4)
                .map(|k| render(&talk[..k], thinking, false))
                .collect();
            let g1 = render(&talk[..1], thinking, true);
            let g3 = render(&talk[..3], thinking, true);
            let (r0, r1, r2, r3) = (&r[0], &r[1], &r[2], &r[3]);
            let pairs: [PiecePair; 5] = [
                (r0.as_ref(), r1.as_ref(), 0),
                (r1.as_ref(), r2.as_ref(), 1),
                (r2.as_ref(), r3.as_ref(), 0),
                (r0.as_ref(), g1.as_ref(), 0),
                (r2.as_ref(), g3.as_ref(), 0),
            ];
            for (before, after, role) in pairs {
                let (Some(before), Some(after)) = (before, after) else {
                    return Plan::new(step);
                };
                if after.len() > before.len() && after[..before.len()] == before[..] {
                    pieces[role].push(after[before.len()..].to_vec());
                } else {
                    parted.push((before.clone(), after.clone(), role));
                }
            }
        }
    }
    // Templates can render the last reply differently; new messages start
    // at the next opener past the difference
    let mut openers = openers_of(text, &pieces);
    for (before, after, role) in &parted {
        let split = before
            .iter()
            .zip(after)
            .position(|(x, y)| x != y)
            .unwrap_or(before.len().min(after.len()));
        let mut at: Option<usize> = None;
        let mut count = 0;
        for (i, &t) in after[split..].iter().enumerate() {
            if openers.contains(&t) {
                count += 1;
                at = Some(split + i);
            }
        }
        if count == 1 {
            if let Some(at) = at {
                pieces[*role].push(after[at..].to_vec());
            }
        }
    }
    if pieces[0].is_empty() || pieces[1].is_empty() {
        return Plan::new(step);
    }
    openers = openers_of(text, &pieces);
    let header = common(&pieces[1]);
    let user = common(&pieces[0]);
    let cut = (0..header.len()).find(|&i| i >= user.len() || user[i] != header[i]);
    let special = !header.is_empty() && text.tokenizer.is_special(header[0]);
    Plan {
        step,
        openers,
        assistant: match cut {
            Some(cut) if special => header[..cut + 1].to_vec(),
            _ => Vec::new(),
        },
        min_chunk: 256,
    }
}

/// One probe pair: the earlier and later renders of a message boundary.
type PiecePair<'a> = (Option<&'a Vec<u32>>, Option<&'a Vec<u32>>, usize);

/// One probe render: the normalized talk rendered, a dangling think opener
/// closed (prompt.zig:15's renderIds shape).
fn render_probe(text: &Text, talk: &[Value], thinking: bool, generation: bool) -> Option<Vec<u32>> {
    let normalized = crate::messages::normalize(Some(&Value::Array(talk.to_vec())), "system", false)
        .ok()?;
    let normalized = crate::messages::tool_arguments(normalized);
    let mut ids = tensorfold_core::render_ids(
        &text.template,
        &text.tokenizer,
        &normalized,
        tensorfold_core::RenderOptions {
            add_generation_prompt: generation,
            enable_thinking: thinking,
            reasoning_effort: None,
            tools: None,
        },
    )
    .ok()?;
    // A dangling think opener closes (prompt.zig:27-30)
    if !thinking && generation && !ids.is_empty() {
        let last = text.tokenizer.decode(&ids[ids.len() - 1..]);
        if crate::reply_text::py_strip(&last) == crate::reply_text::THINK_OPEN {
            if let Some(close) = text.tokenizer.special_token_id(crate::reply_text::THINK_MARKERS.close) {
                ids.push(close);
            }
        }
    }
    Some(ids)
}

/// The special tokens every piece of a role starts with. Upstream:
/// `openersOf` (chunk_plan.zig:121).
fn openers_of(text: &Text, pieces: &[Vec<Vec<u32>>]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for group in pieces {
        let Some(first) = group.first().and_then(|p| p.first()) else {
            continue;
        };
        if !text.tokenizer.is_special(*first) {
            continue;
        }
        if group.iter().all(|p| p.first() == Some(first)) && !out.contains(first) {
            out.push(*first);
        }
    }
    out.sort_unstable();
    out
}

/// The longest prefix every piece shares. Upstream: `common`
/// (chunk_plan.zig:137).
fn common(pieces: &[Vec<u32>]) -> Vec<u32> {
    let first = &pieces[0];
    let mut n = first.len();
    for p in pieces {
        n = n.min(p.len());
    }
    for i in 0..n {
        for p in pieces {
            if p[i] != first[i] {
                return first[..i].to_vec();
            }
        }
    }
    first[..n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The upstream test (chunk_plan.zig:146), assertion for assertion: a
    /// harness cut joins the starts, nearer ones go.
    #[test]
    fn a_harness_cut_joins_the_starts_nearer_ones_go() {
        assert_eq!(with_cut(&[4096, 4650], 4648, 4680, 256), [4096, 4648]);
        assert_eq!(with_cut(&[4096, 4650], 0, 4680, 256), [4096, 4650]);
        assert_eq!(with_cut(&[4096, 4650], 4680, 4680, 256), [4096, 4650]);
        assert_eq!(with_cut(&[4096], 600, 4680, 256), [600, 4096]);
    }

    /// The upstream test (chunk_plan.zig:153): chunk starts match
    /// PrefillPlan.chunks.
    #[test]
    fn chunk_starts_match_prefillplan_chunks() {
        let ids: Vec<u32> = vec![
            10, 1, 2, 10, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 10, 7, 9,
        ];
        let plan = Plan {
            step: 8,
            openers: vec![10],
            assistant: vec![10, 7],
            min_chunk: 3,
        };
        assert_eq!(plan.starts(&ids), [3, 11, 18]);
        let plan = Plan {
            step: 8,
            openers: vec![10],
            assistant: vec![10, 7],
            min_chunk: 4,
        };
        assert_eq!(plan.starts(&ids), [8, 16]);
        let talk: Vec<u32> = vec![
            10, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 10, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 10, 7, 1,
        ];
        assert_eq!(plan.starts(&talk), [8, 14, 22, 28]);
        assert_eq!(Plan::new(8).starts(&ids), [8, 16], "step alone cuts evenly");
        assert!(Plan::default().starts(&ids).is_empty(), "no step: no chunks");
    }
}
