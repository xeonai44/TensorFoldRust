//! A host-only test backend whose tokens depend on all fed history, so a rollback
//! or position mistake changes them.
//!
//! Port of `zig/src/core/lanes/fake.zig` — the engine tests' target. The stream
//! cache is a plain history keyed by the stream's address, exactly like upstream's
//! pointer-keyed map; the draw paths destructure the fake's fields so the lane's
//! borrow and the token draw's stay disjoint.

use crate::accept;
use crate::backend::{
    Backend, DraftRequest, Features, Dtype, Feed, Space, Verified, Window,
};
use crate::stream::Stream;
use crate::Error;
use std::collections::HashMap;
use tensorfold_engine::{key_bits, LogprobRow, Sampling};

/// Upstream: `fake.zig:12  pub const vocab = 97`.
pub const VOCAB: u32 = 97;

/// The fake target's draw after `history` at `position` (keyed noise when sampled).
/// Upstream: `fake.zig:15  pub fn next`.
pub fn next(history: &[u32], s: Option<Sampling>, position: u64) -> u32 {
    let mut sum: u64 = 0;
    for &t in history {
        sum = sum.wrapping_add(t as u64);
    }
    let mut token =
        (sum.wrapping_mul(7).wrapping_add(history.len() as u64 * 3) % VOCAB as u64) as u32;
    if let Some(st) = s {
        token = (token + (key_bits(st.seed, position, 0) % 5) as u32) % VOCAB;
    }
    token
}

/// The fake target's row after `history` for `pick`, a function of the history as
/// logits are; best ids from the pick. Upstream: `fake.zig:21  pub fn rowAt`.
pub fn row_at(history: &[u32], pick: u32, count: u8) -> LogprobRow {
    let mut sum: u64 = 0;
    for &t in history {
        sum = sum.wrapping_add(t as u64);
    }
    let base = -((sum % 50) as f32) / 64.0;
    let mut r = LogprobRow { token: pick, logprob: base, count, ..LogprobRow::default() };
    for i in 0..count as usize {
        r.ids[i] = (pick + i as u32) % VOCAB;
        r.logprobs[i] = base - i as f32;
    }
    r
}

/// `targetNext` with the fake's controls taken by value/reference, so the callers
/// can hold a lane borrow at the same time. Upstream: `fake.zig:44  fn targetNext`.
fn target_next(
    cycle_after: Option<usize>,
    probe_prompt: usize,
    pattern: &[u32],
    answer_cycles: bool,
    history: &[u32],
    sampling: Option<Sampling>,
    position: u64,
) -> u32 {
    if let Some(start) = cycle_after {
        if history.len() >= probe_prompt {
            let reply = &history[probe_prompt..];
            if let Some(close) = reply.iter().position(|&t| t == 91) {
                if reply.len() == close + 1 {
                    return 92;
                }
                let after = reply.len() - close - 2;
                if answer_cycles {
                    let answer = [21u32, 22, 23];
                    return answer[after % 3];
                }
                let answer = [40u32, 41, 42, 96];
                return answer[after.min(3)];
            }
            if reply.len() >= start {
                return pattern[(reply.len() - start) % pattern.len()];
            }
        }
    }
    next(history, sampling, position)
}

#[derive(Default)]
struct Lane {
    /// The stream's cache: prompt and fed tokens.
    history: Vec<u32>,
    /// Drafts for the next round.
    held: Vec<u32>,
    /// The history length before the last verify.
    base: usize,
    /// The last verify's row tokens.
    rows: Vec<u32>,
}

/// Upstream: `fake.zig:40  pub const Fake`.
#[derive(Default)]
pub struct Fake {
    lanes: HashMap<usize, Lane>,
    /// Handle -> token.
    drawn: Vec<u32>,
    rounds: u64,
    pub cycle_after: Option<usize>,
    pub probe_prompt: usize,
    pub pattern: Vec<u32>,
    pub answer_cycles: bool,
    pub prefill_chunks: usize,
    pub prefill_count: usize,
    pub prefill_hook: Option<Box<dyn FnMut(usize)>>,
    /// Prefill refuses a sampled stream (Error::Backend("sampling refused")).
    pub refuse_sampled: bool,
}

impl Fake {
    pub fn new() -> Fake {
        Fake {
            pattern: vec![11, 12, 13],
            ..Fake::default()
        }
    }

    /// Upstream: `save` (fake.zig:157) — a copy of the stream's cache at `at`
    /// tokens (the prompt pass stands there), for core/prompt_cache.zig.
    ///
    /// # Safety
    ///
    /// `s` per the crate's invariants; the caller frees the result with
    /// `drop_saved`.
    pub unsafe fn save_for_test(&mut self, s: *mut Stream<'_>, at: u32) -> Result<*mut Vec<u32>, Error> {
        let lane = self.lanes.get_mut(&(s as usize)).ok_or(Error::NotAtMark)?;
        if lane.history.len() != at as usize {
            return Err(Error::NotAtMark);
        }
        Ok(Box::into_raw(Box::new(lane.history.clone())))
    }

    /// Upstream: `drop` (fake.zig:164).
    ///
    /// # Safety
    ///
    /// `saved` is a pointer `save_for_test` made and nothing else freed.
    pub unsafe fn drop_saved(&mut self, saved: *mut Vec<u32>) {
        drop(unsafe { Box::from_raw(saved) });
    }
}

impl Backend for Fake {
    /// Upstream: `prefill` (fake.zig:141).
    fn prefill(&mut self, s: *mut Stream<'_>) -> Result<(), Error> {
        // Safety: the stream pointer.
        let stream = unsafe { &mut *s };
        if self.refuse_sampled && stream.sampling.is_some() {
            return Err(Error::Backend("sampling refused".into()));
        }
        self.lanes.insert(s as usize, Lane::default());
        let chunks = if self.prefill_chunks == 0 { 1 } else { self.prefill_chunks };
        for chunk in 0..chunks {
            self.prefill_count += 1;
            if let Some(hook) = &mut self.prefill_hook {
                hook(chunk);
            }
            if stream.is_cancelled() {
                return Err(Error::Cancelled);
            }
        }
        let Lane { history, .. } = self.lanes.entry(s as usize).or_default();
        history.clear();
        stream.cached = 0;
        if !stream.reuse.saved.is_null() {
            // The kept state's tokens stand in for the prompt's first `at`.
            // Safety: the saved pointer is this fake's own save_for_test product.
            let kept: &mut Vec<u32> = unsafe { &mut *(stream.reuse.saved as *mut Vec<u32>) };
            history.extend_from_slice(kept);
            stream.cached = stream.reuse.at;
        }
        let marks = stream.reuse.marks;
        let has_hook = stream.reuse.hook.is_some();
        // The prompt's unread tail, collected so the mark hook can borrow the
        // stream while the history holds its own borrow of the fake's lanes.
        let tail: Vec<u32> = stream.prompt()[history.len()..].to_vec();
        let mut at = history.len();
        for t in tail {
            at += 1;
            history.push(t);
            if marks.contains(&(at as u32)) && has_hook {
                if let Some(hook) = &mut stream.reuse.hook {
                    // Safety: the mark hook re-enters the host with this stream.
                    hook(s, at as u32);
                }
            }
        }
        Ok(())
    }

    /// Upstream: `first` (fake.zig:170).
    ///
    /// Safety: `s` per the crate invariants.
    fn first(&mut self, s: *mut Stream<'_>, position: u64) -> Result<u64, Error> {
        // Safety: the stream pointer.
        let sampling = unsafe { (*s).sampling };
        let Fake {
            lanes, drawn, cycle_after, probe_prompt, pattern, answer_cycles, ..
        } = self;
        let lane = lanes.entry(s as usize).or_default();
        if position != lane.history.len() as u64 {
            return Err(Error::PositionMismatch);
        }
        let token = target_next(
            *cycle_after,
            *probe_prompt,
            pattern,
            *answer_cycles,
            &lane.history,
            sampling,
            position,
        );
        drawn.push(token);
        Ok(drawn.len() as u64 - 1)
    }

    /// Upstream: `firstRow` (fake.zig:174).
    ///
    /// Safety: `s` per the crate invariants.
    fn first_row(&mut self, s: *mut Stream<'_>) -> Result<LogprobRow, Error> {
        // Safety: the stream pointer.
        unsafe {
            let stream = &*s;
            let Fake {
                lanes,
                cycle_after,
                probe_prompt,
                pattern,
                answer_cycles,
                ..
            } = self;
            let lane = lanes.get(&(s as usize)).ok_or(Error::PositionMismatch)?;
            let pick = target_next(
                *cycle_after,
                *probe_prompt,
                pattern,
                *answer_cycles,
                &lane.history,
                stream.sampling,
                lane.history.len() as u64,
            );
            Ok(row_at(&lane.history, pick, stream.logprobs.unwrap()))
        }
    }

    /// Upstream: `queue` (fake.zig:179).
    ///
    /// Safety: `s` per the crate invariants.
    fn queue(&mut self, s: *mut Stream<'_>, feed: Feed, position: u64) -> Result<u64, Error> {
        // Safety: the stream pointer.
        let sampling = unsafe { (*s).sampling };
        let value = self.value(feed);
        let Fake {
            lanes, drawn, cycle_after, probe_prompt, pattern, answer_cycles, ..
        } = self;
        let lane = lanes.entry(s as usize).or_default();
        lane.history.push(value);
        if position != lane.history.len() as u64 {
            return Err(Error::PositionMismatch);
        }
        let token = target_next(
            *cycle_after,
            *probe_prompt,
            pattern,
            *answer_cycles,
            &lane.history,
            sampling,
            position,
        );
        drawn.push(token);
        Ok(drawn.len() as u64 - 1)
    }

    fn read(&mut self, handle: u64) -> Result<u32, Error> {
        self.drawn
            .get(handle as usize)
            .copied()
            .ok_or(Error::PositionMismatch)
    }

    /// Upstream: `verify` (fake.zig:176) — each row reads the cache and its own
    /// path from the pending row.
    fn verify(&mut self, windows: &[Window<'_>], out: &mut [Verified<'_>]) -> Result<(), Error> {
        self.rounds += 1;
        for (w, o) in windows.iter().zip(out.iter_mut()) {
            // Safety: the window's stream pointer.
            let (sampling, logprobs) = unsafe { ((*w.stream).sampling, (*w.stream).logprobs) };
            let Fake {
                lanes,
                cycle_after,
                probe_prompt,
                pattern,
                answer_cycles,
                ..
            } = self;
            let lane = lanes.entry(w.stream as usize).or_default();
            lane.base = lane.history.len();
            lane.rows.clear();
            lane.rows.push(w.pending);
            lane.rows.extend_from_slice(&lane.held[..w.held as usize]);
            lane.rows.extend_from_slice(w.tokens);
            o.drafts.copy_from_slice(&lane.rows[1..]);
            let parents = accept::row_parents(lane.rows.len(), w.parents);
            let mut sampled: Vec<u32> = Vec::with_capacity(lane.rows.len());
            let mut rows: Vec<LogprobRow> = Vec::new();
            for r in 0..lane.rows.len() {
                // Each row reads the cache and its own path from the pending row.
                lane.history.truncate(lane.base);
                let mut path: Vec<u32> = Vec::new();
                let mut at: i32 = r as i32;
                while at >= 0 {
                    path.insert(0, lane.rows[at as usize]);
                    at = parents[at as usize];
                }
                lane.history.extend_from_slice(&path);
                if w.positions[r] != lane.history.len() as u64 {
                    return Err(Error::PositionMismatch);
                }
                sampled.push(target_next(
                    *cycle_after,
                    *probe_prompt,
                    pattern,
                    *answer_cycles,
                    &lane.history,
                    sampling,
                    w.positions[r],
                ));
                if !o.rows.is_empty() {
                    rows.push(row_at(&lane.history, sampled[r], logprobs.unwrap()));
                }
            }
            o.sampled.copy_from_slice(&sampled);
            if !o.rows.is_empty() {
                o.rows.copy_from_slice(&rows);
            }
            lane.history.truncate(lane.base);
            // A chain keeps every row unless rolled back.
            lane.history.extend_from_slice(&lane.rows);
        }
        Ok(())
    }

    fn keep(&mut self, windows: &[Window<'_>], paths: &[&[u32]]) -> Result<(), Error> {
        for (w, path) in windows.iter().zip(paths.iter()) {
            let lane = self
                .lanes
                .get_mut(&(w.stream as usize))
                .ok_or(Error::PositionMismatch)?;
            lane.history.truncate(lane.base);
            for &r in path.iter() {
                lane.history.push(lane.rows[r as usize]);
            }
        }
        Ok(())
    }

    /// Upstream: `draft` (fake.zig:213) — the head reads the last verify's kept
    /// rows (a shared round drafts before its rollback).
    fn draft(&mut self, requests: &[DraftRequest<'_, '_>]) -> Result<(), Error> {
        for r in requests {
            // Safety: the stream pointer.
            let sampling = unsafe { (*r.stream).sampling };
            let value = self.value(r.first.unwrap_or(Feed::Value(0)));
            let first_pending = match r.first {
                Some(feed) => self.value(feed),
                None => r.follow[r.follow.len() - 1],
            };
            let Fake {
                lanes,
                cycle_after,
                probe_prompt,
                pattern,
                answer_cycles,
                ..
            } = self;
            let lane = lanes.entry(r.stream as usize).or_default();
            let mut guess: Vec<u32> = Vec::new();
            match r.rows {
                Some(path) => {
                    guess.extend_from_slice(&lane.history[..lane.base]);
                    for &row in path {
                        guess.push(lane.rows[row as usize]);
                    }
                }
                None => guess.extend_from_slice(&lane.history),
            }
            if r.position != guess.len() as u64 + 1 {
                return Err(Error::PositionMismatch);
            }
            lane.held.clear();
            guess.push(first_pending);
            for j in 0..r.depth {
                let at = r.position + j as u64;
                let mut t = target_next(
                    *cycle_after,
                    *probe_prompt,
                    pattern,
                    *answer_cycles,
                    &guess,
                    sampling,
                    at,
                );
                if (at.wrapping_mul(2654435761).wrapping_add(j as u64) % 5) == 0 {
                    t = (t + 1) % VOCAB; // a wrong guess now and then
                }
                lane.held.push(t);
                guess.push(t);
            }
            let _ = value;
        }
        Ok(())
    }

    /// A row's "state" is its token (one u32 a row, whatever the taps): enough for
    /// a drafter to rebuild the history. Upstream: `features` (fake.zig:243).
    fn features(
        &mut self,
        s: *mut Stream<'_>,
        _taps: &[u32],
        start: u64,
        count: u32,
    ) -> Result<Features, Error> {
        let lane = self
            .lanes
            .get(&(s as usize))
            .ok_or(Error::PositionMismatch)?;
        if start + count as u64 > lane.history.len() as u64 {
            return Err(Error::RowsNotHeld);
        }
        Ok(Features {
            space: Space::Host,
            dtype: Dtype::U32,
            buffer: lane.history.as_ptr() as u64,
            offset: start * 4,
            rows: count,
            row_bytes: 4,
            device: 0,
            ready: None,
        })
    }

    fn release(&mut self, s: *mut Stream<'_>) {
        self.lanes.remove(&(s as usize));
    }
}

impl Fake {
    fn value(&self, feed: Feed) -> u32 {
        match feed {
            Feed::Handle(h) => self.drawn[h as usize],
            Feed::Value(v) => v,
        }
    }
}
