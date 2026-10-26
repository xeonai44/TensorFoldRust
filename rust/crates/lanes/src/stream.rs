//! One stream (Python LaneStream) and the round loop's state for it (Python's
//! dicts keyed by stream id).
//!
//! Port of `zig/src/core/lanes/stream.zig`.

use crate::depth::State;
use crate::plan_lanes::{self, Measured};
use crate::proposer::Proposer;
use crate::shape::{self, Odds, Shape};
use crate::Error;
use tensorfold_engine::{LogprobRow, Sampling};

/// Upstream: `stream.zig:11  pub const Reason`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reason {
    #[default]
    None,
    Stop,
    Length,
    Error,
    Cancelled,
}

impl Reason {
    /// Upstream: `name` — "" for none, else the tag.
    pub fn name(self) -> &'static str {
        match self {
            Reason::None => "",
            Reason::Stop => "stop",
            Reason::Length => "length",
            Reason::Error => "error",
            Reason::Cancelled => "cancelled",
        }
    }
}

/// Upstream: `stream.zig:23  pub const Mode = enum { pipe, drain, verify, exit }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Pipe,
    Drain,
    Verify,
    Exit,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Pipe => "pipe",
            Mode::Drain => "drain",
            Mode::Verify => "verify",
            Mode::Exit => "exit",
        }
    }
}

/// Upstream: `stream.zig:25-27`.
pub const LOOP_MAX_PERIOD: usize = 8;
pub const LOOP_MIN_RUN: usize = 256;
pub const LOOP_WARM_IN: usize = 64;

/// Return the shortest exact cycle at the end of a combined token prefix.
/// Upstream: `stream.zig:30  fn cycleAt`.
fn cycle_at(head: &[u32], tail: &[u32], n: usize) -> Option<u32> {
    if n < LOOP_MIN_RUN + 1 {
        return None;
    }
    for period in 1..=LOOP_MAX_PERIOD {
        if n < LOOP_MIN_RUN + period {
            break;
        }
        if n - LOOP_MIN_RUN - period < LOOP_WARM_IN {
            break;
        }
        let mut matches = true;
        for k in 1..=LOOP_MIN_RUN {
            let right = n - k;
            let left = right - period;
            let right_token = if right < head.len() { head[right] } else { tail[right - head.len()] };
            let left_token = if left < head.len() { head[left] } else { tail[left - head.len()] };
            if right_token != left_token {
                matches = false;
                break;
            }
        }
        if matches {
            return Some(period as u32);
        }
    }
    None
}

/// Return the first token index in a candidate batch that would fire the guard.
/// Upstream: `stream.zig:52  pub fn loopCut`.
pub fn loop_cut(s: &Stream, tokens: &[u32]) -> Option<usize> {
    if !s.loop_guard || !s.think_open || s.think_end < 0 || s.loop_period.is_some() {
        return None;
    }
    for (i, &token) in tokens.iter().enumerate() {
        if token as i64 == s.think_end {
            return None;
        }
        if cycle_at(s.emitted(), tokens, s.emitted().len() + i + 1).is_some() {
            return Some(i);
        }
    }
    None
}

/// Return the detected period at the end of committed tokens, or None.
/// Upstream: `stream.zig:62  pub fn detectLoop`.
pub fn detect_loop(emitted: &[u32]) -> Option<u32> {
    cycle_at(emitted, &[], emitted.len())
}

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes
/// their tail). Upstream: `stream.zig:67  StopCheck`.
pub trait StopCheck {
    fn should_stop(&mut self, emitted: &[u32]) -> bool;
}

/// The host's answer to whether the stream's request was cancelled, asked between
/// prompt chunks. Upstream: `stream.zig:73  CancelCheck`.
pub trait CancelCheck {
    fn is_cancelled(&mut self) -> bool;
}

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the
/// chunk that ends there). Upstream: `stream.zig:87  MarkHook` — the port keeps the
/// same ctx+fn pair as a closure, because the hook re-enters the host with the
/// stream.
pub type MarkHook<'a> = Box<dyn FnMut(*mut Stream, u32) + 'a>;

/// A kept prompt state the backend restores before its prompt pass
/// (core/prompt_cache.zig — Phase 2), and where it keeps new ones. Upstream:
/// `stream.zig:79  Reuse`.
#[derive(Default)]
pub struct Reuse<'a> {
    /// The backend's copy of the state after `at` prompt tokens (null: start at 0).
    pub saved: *mut core::ffi::c_void,
    pub at: u32,
    /// Ascending: the pass cuts a chunk at each and calls `hook` there.
    pub marks: &'a [u32],
    pub hook: Option<MarkHook<'a>>,
}

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens
/// and parents on the host. Upstream: `stream.zig:93  Held`.
#[derive(Clone, Default)]
pub struct Held {
    pub count: u32,
    pub tokens: Option<Vec<u32>>,
    pub parents: Option<Vec<i32>>,
}

/// Upstream: `stream.zig:99  pub const Spec` — what a stream is asked to be.
#[derive(Default)]
pub struct Spec<'a> {
    pub id: &'a str,
    pub prompt: &'a [u32],
    pub max_new: u32,
    pub eos: &'a [u32],
    pub sampling: Option<Sampling>,
    pub drafts: bool,
    pub proposer: Option<Box<dyn Proposer + 'a>>,
    pub stop_check: Option<Box<dyn StopCheck + 'a>>,
    pub cancel_check: Option<Box<dyn CancelCheck + 'a>>,
    pub think_budget: u32,
    pub think_close: &'a [u32],
    pub think_end: i64,
    /// None: open when a budget is set (the server's rule).
    pub think_open: Option<bool>,
    pub loop_guard: bool,
    /// Where prefill chunks start after 0 (Python's PrefillPlan); empty: the
    /// backend's step.
    pub chunks: &'a [u32],
    pub reuse: Reuse<'a>,
    /// The target's log probabilities for each committed token, with this many best
    /// tokens.
    pub logprobs: Option<u8>,
}

impl Spec<'_> {
    /// The defaults upstream's field table implies (drafts on, thinking closed).
    pub fn new<'a>(id: &'a str, prompt: &'a [u32], max_new: u32) -> Spec<'a> {
        Spec {
            id,
            prompt,
            max_new,
            eos: &[],
            sampling: None,
            drafts: true,
            proposer: None,
            stop_check: None,
            cancel_check: None,
            think_budget: 0,
            think_close: &[],
            think_end: -1,
            think_open: None,
            loop_guard: false,
            chunks: &[],
            reuse: Reuse::default(),
            logprobs: None,
        }
    }
}

/// Upstream: `stream.zig:119  pub const Stream`.
pub struct Stream<'a> {
    pub id: &'a str,
    pub prompt_len: usize,
    pub max_new: u32,
    pub eos: &'a [u32],
    pub sampling: Option<Sampling>,
    pub drafts: bool,
    pub proposer: Option<Box<dyn Proposer + 'a>>,
    pub stop_check: Option<Box<dyn StopCheck + 'a>>,
    pub cancel_check: Option<Box<dyn CancelCheck + 'a>>,
    pub think_budget: u32,
    pub think_close: &'a [u32],
    pub think_end: i64,
    pub think_open: bool,
    pub loop_guard: bool,
    pub loop_period: Option<u32>,
    pub chunks: &'a [u32],
    pub logprobs: Option<u8>,
    /// With `logprobs`: one row a token of emitted(), in order.
    pub rows: Vec<LogprobRow>,
    pub reuse: Reuse<'a>,
    /// Prompt tokens the backend restored from `reuse` (its prompt pass started
    /// there).
    pub cached: u32,
    /// The backend's restore of `reuse` failed: it prefilled from 0.
    pub reuse_failed: bool,
    pub context: Vec<u32>,
    pub pending: Option<u32>,
    pub force: Vec<u32>,
    pub cache_len: u64,
    pub finished: bool,
    pub reason: Reason,
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Suffix-match branch rows verified (fill.zig).
    pub branch_rows: u64,
    /// Of those, rows on the accepted path.
    pub branch_accepted: u64,
    /// Recent share of grafted rows kept; a read of the chain that grafted none
    /// counts as a miss.
    pub graft_hit: f64,
    /// How often the target took the head's rank-r lane at each depth (head trees).
    pub odds: Option<Odds>,
    /// The shape of the held tree drafts.
    pub lanes: Option<Box<Shape>>,
    /// Tree lanes verified and taken.
    pub landed: [[[u32; 2]; shape::RANKS]; shape::MAX_DEPTH],
    /// The planner's picks: each tree size, then chain + grafts.
    pub picks: [u32; plan_lanes::SIZES.len() + 1],
    /// Grafted rows the planner left this round's window (None: no planner).
    pub graft_room: Option<u32>,
    /// Head levels drafted for the held window.
    pub held_levels: u32,
    /// The stream's round time beyond its window and head levels (ms, recent):
    /// chains, and trees.
    pub round_over: Option<f64>,
    pub tree_over: Option<f64>,
    /// Each pick's round times past its table price.
    pub measured: [Measured; plan_lanes::SIZES.len() + 1],
    /// The pick that shaped the held window, at its table price.
    pub priced: Option<(usize, f64)>,
    pub min_rows: u32,
    pub paused: bool,
    pub depth: Option<State>,
    pub mode: Option<Mode>,
    pub next: Option<Held>,
    pub inflight: Option<u64>,
    pub served: Option<u64>,
    pub granted: Option<u32>,
    pub copy_width: Option<u32>,
}

// Safety: a stream crosses threads exactly once, queued inside its job (the
// host's `submit` to the engine thread); from there the single engine thread
// drives it alone (the crate's invariants, lib.rs). The raw pointers it carries
// (the reuse state) are the host's own and unused until Phase 2's cache.
unsafe impl Send for Stream<'_> {}

impl<'a> Stream<'a> {
    /// Upstream: `init` (stream.zig:173).
    pub fn init(spec: Spec<'a>) -> Stream<'a> {
        let think_open = spec
            .think_open
            .unwrap_or(spec.think_budget > 0 || (spec.loop_guard && spec.think_end >= 0));
        let mut context = Vec::with_capacity(spec.prompt.len());
        context.extend_from_slice(spec.prompt);
        Stream {
            id: spec.id,
            prompt_len: spec.prompt.len(),
            max_new: spec.max_new,
            eos: spec.eos,
            sampling: spec.sampling,
            drafts: spec.drafts,
            proposer: if spec.drafts { spec.proposer } else { None },
            stop_check: spec.stop_check,
            cancel_check: spec.cancel_check,
            think_budget: spec.think_budget,
            think_close: spec.think_close,
            think_end: spec.think_end,
            think_open,
            loop_guard: spec.loop_guard,
            loop_period: None,
            chunks: spec.chunks,
            logprobs: spec.logprobs,
            rows: Vec::new(),
            reuse: spec.reuse,
            cached: 0,
            reuse_failed: false,
            context,
            pending: None,
            force: Vec::new(),
            cache_len: 0,
            finished: false,
            reason: Reason::None,
            rounds: 0,
            drafted: 0,
            accepted: 0,
            branch_rows: 0,
            branch_accepted: 0,
            graft_hit: 1.0,
            odds: None,
            lanes: None,
            landed: [[[0, 0]; shape::RANKS]; shape::MAX_DEPTH],
            picks: [0; plan_lanes::SIZES.len() + 1],
            graft_room: None,
            held_levels: 0,
            round_over: None,
            tree_over: None,
            measured: [Measured::default(); plan_lanes::SIZES.len() + 1],
            priced: None,
            min_rows: 0,
            paused: false,
            depth: None,
            mode: None,
            next: None,
            inflight: None,
            served: None,
            granted: None,
            copy_width: None,
        }
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    /// Upstream: `dropRoundState` (stream.zig:205).
    pub fn drop_round_state(&mut self) {
        self.drop_held();
        self.depth = None;
        self.mode = None;
        self.inflight = None;
        self.served = None;
        self.granted = None;
        self.copy_width = None;
    }

    /// Upstream: `dropHeld`.
    pub fn drop_held(&mut self) {
        self.next = None;
        self.lanes = None;
    }

    pub fn prompt(&self) -> &[u32] {
        &self.context[..self.prompt_len]
    }

    pub fn emitted(&self) -> &[u32] {
        &self.context[self.prompt_len..]
    }

    pub fn budget_left(&self) -> i64 {
        self.max_new as i64 - self.emitted().len() as i64
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks
    /// before each chunk. Upstream: `isCancelled`.
    pub fn is_cancelled(&mut self) -> bool {
        match &mut self.cancel_check {
            Some(c) => c.is_cancelled(),
            None => false,
        }
    }

    fn budget_active(&self) -> bool {
        self.think_open && self.think_budget > 0 && !self.think_close.is_empty()
    }

    /// The index the thinking budget replaces with `think_close[0]`, or None if the
    /// block closed first. Upstream: `thinkCut`.
    pub fn think_cut(&self, tokens: &[u32]) -> Option<usize> {
        if !self.budget_active() {
            return None;
        }
        let done = self.emitted().len();
        for (i, &t) in tokens.iter().enumerate() {
            if done + i + 1 >= self.think_budget as usize {
                return Some(i);
            }
            if t as i64 == self.think_end {
                return None;
            }
        }
        None
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    /// Upstream: `cutsNext`.
    pub fn cuts_next(&self) -> bool {
        self.budget_active() && self.emitted().len() + 1 >= self.think_budget as usize
    }

    /// Begin the thinking budget's close: its first token now, the rest forced
    /// after it. Upstream: `startClose`.
    pub fn start_close(&mut self) -> u32 {
        self.think_open = false;
        self.force.clear();
        self.force.extend_from_slice(&self.think_close[1..]);
        self.think_close[0]
    }

    /// Tokens a round may commit before the length limit or the thinking budget's
    /// cut. Upstream: `draftRoom`.
    pub fn draft_room(&self) -> i64 {
        let mut room = self.budget_left();
        if self.budget_active() {
            room = room.min(self.think_budget as i64 - self.emitted().len() as i64);
        }
        room
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Each held tree lane whose parent row the round reached: the target took it
    /// or not (its depth and rank). Upstream: `observeLanes`.
    pub fn observe_lanes(&mut self, l: &Shape, path: &[u32]) {
        let mut on = [false; 256];
        for &r in path {
            if (r as usize) < on.len() {
                on[r as usize] = true;
            }
        }
        for (i, (&par, (&d, &r))) in l
            .parents
            .iter()
            .zip(l.depths.iter().zip(l.ranks.iter()))
            .enumerate()
        {
            let parent_row: usize = if par < 0 { 0 } else { 1 + par as usize };
            if !on[parent_row] {
                continue;
            }
            let took = on[1 + i];
            if let Some(odds) = &mut self.odds {
                odds.observe(d as usize, r as usize, took);
            }
            if (d as usize) < shape::MAX_DEPTH && (r as usize) < shape::RANKS {
                self.landed[d as usize][r as usize][0] += 1;
                self.landed[d as usize][r as usize][1] += u32::from(took);
            }
        }
    }

    /// Append tokens, and with `logprobs` their rows, until the stream finishes;
    /// how many landed. Upstream: `commit` (stream.zig:302).
    pub fn commit(&mut self, tokens: &[u32], rows: &[LogprobRow]) -> Result<usize, Error> {
        if rows.len() != if self.logprobs.is_some() { tokens.len() } else { 0 } {
            return Err(Error::LogprobRowsMismatch);
        }
        let mut landed: usize = 0;
        for (i, &t) in tokens.iter().enumerate() {
            if self.finished {
                break;
            }
            self.context.push(t);
            if !rows.is_empty() {
                self.rows.push(rows[i]);
            }
            landed += 1;
            if t as i64 == self.think_end {
                self.think_open = false;
            }
            let fire = if self.loop_guard
                && self.think_open
                && self.think_end >= 0
                && self.loop_period.is_none()
            {
                detect_loop(self.emitted())
            } else {
                None
            };
            let stopped = self.stopped();
            if self.is_eos(t) || stopped {
                self.finished = true;
                self.reason = Reason::Stop;
            } else if let Some(period) = fire {
                self.loop_period = Some(period);
                if self.emitted().len() >= self.max_new as usize {
                    self.finished = true;
                    self.reason = Reason::Length;
                } else if self.think_close.is_empty() {
                    self.finished = true;
                    self.reason = Reason::Stop;
                } else {
                    self.think_open = false;
                    self.force.clear();
                    self.force.extend_from_slice(self.think_close);
                }
                break;
            } else if self.emitted().len() >= self.max_new as usize {
                self.finished = true;
                self.reason = Reason::Length;
            }
        }
        Ok(landed)
    }

    fn stopped(&mut self) -> bool {
        // The check re-enters the host with the emitted tail. The checker lives in
        // one field and the tail in another; the raw pointer keeps the two borrows
        // apart the way upstream's ctx pointer does (the crate's invariants hold:
        // the checker does not touch the stream).
        let Some(check) = self.stop_check.as_deref_mut() else {
            return false;
        };
        let check: *mut dyn StopCheck = check;
        let tail = &self.context[self.prompt_len..];
        // Safety: the checker is the stream's own; it must not re-enter the stream.
        unsafe { (*check).should_stop(tail) }
    }

    pub fn pop_force(&mut self) -> Option<u32> {
        if self.force.is_empty() {
            return None;
        }
        Some(self.force.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors stream.zig:348 "commit stops at eos and length".
    #[test]
    fn commit_stops_at_eos_and_length() {
        let mut s = Stream::init(Spec {
            id: "a",
            prompt: &[1, 2],
            max_new: 3,
            eos: &[9],
            ..Spec::new("a", &[1, 2], 3)
        });
        assert_eq!(s.commit(&[5, 9, 7], &[]).unwrap(), 2);
        assert_eq!(s.emitted(), [5, 9]);
        assert!(s.finished && s.reason == Reason::Stop);
    }

    /// Mirrors stream.zig:357: a logprob stream lands a row with each token, and
    /// refuses tokens without rows.
    #[test]
    fn logprob_streams_land_rows_and_refuse_without() {
        let mut s = Stream::init(Spec {
            id: "a",
            prompt: &[1, 2],
            max_new: 3,
            eos: &[9],
            logprobs: Some(0),
            ..Spec::new("a", &[1, 2], 3)
        });
        assert_eq!(s.commit(&[5], &[]).unwrap_err(), Error::LogprobRowsMismatch);
        let rows = [
            LogprobRow { token: 5, logprob: -0.5, ..Default::default() },
            LogprobRow { token: 9, logprob: -1.0, ..Default::default() },
            LogprobRow { token: 7, logprob: -2.0, ..Default::default() },
        ];
        assert_eq!(s.commit(&[5, 9, 7], &rows).unwrap(), 2);
        assert_eq!(s.rows.len(), 2);
        assert_eq!(s.rows[1].token, 9);
    }

    /// Mirrors stream.zig:368: the loop guard stops at the cap even when no cycle
    /// fires.
    #[test]
    fn loop_guard_stops_at_the_cap() {
        let tokens: Vec<u32> = (0..20).map(|i| 100 + i as u32).collect();
        let mut guarded = Stream::init(Spec {
            id: "guarded",
            prompt: &[1],
            max_new: 10,
            think_close: &[90, 91, 92],
            think_end: 91,
            loop_guard: true,
            ..Spec::new("guarded", &[1], 10)
        });
        assert_eq!(guarded.commit(&tokens, &[]).unwrap(), 10);
        assert!(guarded.finished && guarded.reason == Reason::Length);
    }

    /// Mirrors stream.zig:378: the loop guard fires, closes thinking, and latches
    /// once.
    #[test]
    fn loop_guard_fires_closes_and_latches() {
        let mut tokens: Vec<u32> = Vec::new();
        for i in 0..LOOP_WARM_IN {
            tokens.push(1000 + i as u32);
        }
        for _ in 0..257 {
            tokens.push(7);
        }
        let mut s = Stream::init(Spec {
            id: "loop",
            prompt: &[1],
            max_new: 400,
            think_close: &[90, 91, 92],
            think_end: 91,
            loop_guard: true,
            ..Spec::new("loop", &[1], 400)
        });
        assert_eq!(s.commit(&tokens, &[]).unwrap(), 321);
        assert_eq!(s.loop_period, Some(1));
        assert!(!s.finished && !s.think_open && s.force.len() == 3);
        while let Some(token) = s.pop_force() {
            s.commit(&[token], &[]).unwrap();
        }
        assert!(!s.finished && s.force.is_empty());
        s.commit(&[8], &[]).unwrap();
        assert!(!s.finished && s.loop_period == Some(1));
    }
}
