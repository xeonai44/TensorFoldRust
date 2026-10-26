//! Copy proposals from a stream's own context (Python SuffixLookupProposer) behind
//! a small proposer interface.
//!
//! Port of `zig/src/core/lanes/proposer.zig`.

use crate::Error;
use std::collections::HashMap;

/// What the round loop asks of a proposer: a copied continuation, its evidence, and
/// how it fared. Upstream: `proposer.zig:6  pub const Proposer` (a vtable).
///
/// The port adds `state_fields` where upstream compares the vtable pointer to
/// recognise its own SuffixLookup in the trail (trail.zig:75) — the same
/// information, as a method.
pub trait Proposer {
    /// Tokens valid until the next `propose`.
    fn propose(&mut self, context: &[u32], max_draft: i64) -> Result<&[u32], Error>;
    fn last_match(&self) -> i64;
    fn observe(&mut self, proposed: i64, accepted: i64);
    /// The trail's proposer state, when this proposer reports one.
    fn state_fields<'s>(&'s self) -> Option<Vec<(String, crate::events::Value<'s>)>> {
        None
    }
}

/// Upstream: `proposer.zig:30  pub const max_ngram = 8`.
pub const MAX_NGRAM: usize = 8;

type Key = [u32; MAX_NGRAM];

/// Upstream: `proposer.zig:33  pub const Params`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub ngram: usize,
    pub min_match: i64,
    pub max_extension: i64,
    pub silence_rounds: i64,
    pub window: usize,
    pub confident_match: i64,
}

impl Default for Params {
    /// Upstream's field defaults (proposer.zig:34-40).
    fn default() -> Self {
        Params {
            ngram: 3,
            min_match: 6,
            max_extension: 64,
            silence_rounds: 16,
            window: 4,
            confident_match: 24,
        }
    }
}

/// Continuations backed by enough matching context; silent for a while after a run
/// of rejected proposals.
///
/// Upstream: `proposer.zig:43  pub const SuffixLookup`.
pub struct SuffixLookup {
    params: Params,
    index: HashMap<Key, Vec<u32>>,
    indexed: usize,
    last_key: Key,
    recent: Vec<i64>,
    silent_for: i64,
    out: Vec<u32>,
    proposals: i64,
    proposed_tokens: i64,
    judged_tokens: i64,
    accepted_tokens: i64,
    silenced_rounds: i64,
    last_match: i64,
    last_confident: bool,
}

impl SuffixLookup {
    /// Upstream: `init` (proposer.zig:60) — the params rules refuse, the window
    /// floors at one.
    pub fn new(params: Params) -> Result<SuffixLookup, Error> {
        if params.ngram < 1 || params.ngram > MAX_NGRAM {
            return Err(Error::BadNgram);
        }
        if params.min_match < params.ngram as i64 {
            return Err(Error::MinMatchBelowNgram);
        }
        let params = Params { window: params.window.max(1), ..params };
        Ok(SuffixLookup {
            params,
            index: HashMap::new(),
            indexed: 0,
            last_key: [u32::MAX; MAX_NGRAM],
            recent: Vec::new(),
            silent_for: 0,
            out: Vec::new(),
            proposals: 0,
            proposed_tokens: 0,
            judged_tokens: 0,
            accepted_tokens: 0,
            silenced_rounds: 0,
            last_match: 0,
            last_confident: false,
        })
    }

    fn key_at(&self, context: &[u32], end: usize) -> Key {
        let mut key = [u32::MAX; MAX_NGRAM];
        let n = self.params.ngram;
        key[..n].copy_from_slice(&context[end - n..end]);
        key
    }

    fn extend(&mut self, context: &[u32]) {
        let n = self.params.ngram;
        let mut position = self.indexed.max(n - 1);
        while position < context.len() {
            let key = self.key_at(context, position + 1);
            self.index.entry(key).or_default().push(position as u32);
            position += 1;
        }
        self.indexed = context.len();
    }

    fn match_length(&self, context: &[u32], end: usize) -> i64 {
        let len = context.len() as i64;
        let e = end as i64;
        let mut length: i64 = 0;
        let limit = self.params.max_extension.min(e);
        while length < limit
            && context[(e - 1 - length) as usize] == context[(len - 1 - length) as usize]
        {
            length += 1;
            if len - 1 - length < 0 {
                break;
            }
        }
        length
    }

    /// Python `propose`: the copied continuation (owned by the proposer until the
    /// next call), or empty.
    ///
    /// Upstream: `proposer.zig:133  pub fn propose`.
    pub fn propose(&mut self, context: &[u32], max_draft: i64) -> Result<&[u32], Error> {
        let n = self.params.ngram;
        self.out.clear();
        if max_draft <= 0 || context.len() < n + 1 {
            return Ok(&self.out);
        }
        if self.silent_for > 0 {
            self.silent_for -= 1;
            self.silenced_rounds += 1;
            return Ok(&self.out);
        }
        // The context changed underneath the index (a new request): rebuild it.
        let changed = self.indexed > context.len()
            || (self.indexed != 0 && self.key_at(context, self.indexed) != self.last_key);
        if changed {
            self.index.clear();
            self.indexed = 0;
        }
        self.extend(context);
        self.last_key = self.key_at(context, self.indexed);
        let Some(positions) = self.index.get(&self.key_at(context, context.len())) else {
            return Ok(&self.out);
        };
        if positions.is_empty() {
            return Ok(&self.out);
        }
        let mut best_end: i64 = -1;
        let mut best_len: i64 = 0;
        for i in (0..positions.len()).rev() {
            let end = positions[i] as usize + 1;
            if end >= context.len() {
                continue;
            }
            let length = self.match_length(context, end);
            if length > best_len {
                best_len = length;
                best_end = end as i64;
                if length >= self.params.max_extension {
                    break;
                }
            }
        }
        self.last_confident = false;
        self.last_match = best_len;
        if best_end < 0 || best_len < self.params.min_match {
            return Ok(&self.out);
        }
        self.last_confident = best_len >= self.params.confident_match;
        let from = best_end as usize;
        let to = context.len().min(from + max_draft as usize);
        self.out.extend_from_slice(&context[from..to]);
        if !self.out.is_empty() {
            self.proposals += 1;
            self.proposed_tokens += self.out.len() as i64;
        }
        Ok(&self.out)
    }

    /// Upstream: `proposer.zig:180  pub fn observe`.
    pub fn observe(&mut self, proposed: i64, accepted: i64) {
        self.judged_tokens += proposed;
        self.accepted_tokens += accepted;
        if proposed <= 0 {
            return;
        }
        // Upstream keeps a plain list, trimming the front by copying forward
        // (proposer.zig:184): the same order, the same window.
        self.recent.push(accepted);
        if self.recent.len() > self.params.window {
            let drop = self.recent.len() - self.params.window;
            self.recent.copy_within(drop.., 0);
            self.recent.truncate(self.params.window);
        }
        if self.recent.len() >= self.params.window && self.recent.iter().all(|&x| x == 0) {
            self.silent_for = self.params.silence_rounds;
            self.recent.clear();
        }
    }
}

impl Proposer for SuffixLookup {
    fn propose(&mut self, context: &[u32], max_draft: i64) -> Result<&[u32], Error> {
        SuffixLookup::propose(self, context, max_draft)
    }

    fn last_match(&self) -> i64 {
        self.last_match
    }

    fn observe(&mut self, proposed: i64, accepted: i64) {
        SuffixLookup::observe(self, proposed, accepted)
    }

    fn state_fields<'s>(&'s self) -> Option<Vec<(String, crate::events::Value<'s>)>> {
        // The trail's SuffixLookup listing (trail.zig:77).
        Some(vec![
            ("silent_for".into(), crate::events::Value::Int(self.silent_for)),
            ("recent".into(), crate::events::Value::I64s(&self.recent)),
            ("proposals".into(), crate::events::Value::Int(self.proposals)),
            ("proposed_tokens".into(), crate::events::Value::Int(self.proposed_tokens)),
            ("judged_tokens".into(), crate::events::Value::Int(self.judged_tokens)),
            ("accepted_tokens".into(), crate::events::Value::Int(self.accepted_tokens)),
            ("silenced_rounds".into(), crate::events::Value::Int(self.silenced_rounds)),
            ("last_match".into(), crate::events::Value::Int(self.last_match)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors proposer.zig:197: suffix lookup copies the continuation after a
    /// repeated n-gram.
    #[test]
    fn suffix_lookup_copies_the_continuation() {
        let mut s = SuffixLookup::new(Params { ngram: 2, min_match: 2, ..Params::default() }).unwrap();
        let ctx = [1, 2, 3, 4, 5, 9, 1, 2, 3];
        let got = s.propose(&ctx, 3).unwrap().to_vec();
        assert_eq!(got, [4, 5, 9]);
        assert_eq!(s.last_match, 3);
    }
}
