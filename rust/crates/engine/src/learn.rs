//! The Sliding/Living Weights learner contract.
//!
//! Port of the `Example`, `LearnRequest`, `LearnEvent`, `LearnSink`, `Learner` and
//! `LearnError` declarations of `zig/src/core/engine_api.zig` (lines 225-266). No engine
//! implements these yet (Phase 3); the contract is ported now so the server's
//! translation layer can be built against the final shape.

/// Upstream: `engine_api.zig:223  pub const LearnError = error{ Closed, Busy, Unsupported };`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearnError {
    Closed,
    Busy,
    Unsupported,
}

/// A worked example for the learner: token ids whose answer starts at `start` (rows from
/// start - 1 on predict it).
///
/// Upstream: `engine_api.zig:225  pub const Example = struct { ids, start }`.
#[derive(Clone, Copy, Debug)]
pub struct Example<'a> {
    pub ids: &'a [u32],
    pub start: u32,
}

/// One fact's lesson (Sliding Weights): answers, held-out answers, near misses, and
/// prompts that must not move.
///
/// Upstream: `engine_api.zig:228  pub const LearnRequest = struct { ... }`. The upstream
/// default `steps` is 60.
#[derive(Clone, Copy, Debug)]
pub struct LearnRequest<'a> {
    pub train: &'a [Example<'a>],
    pub held: &'a [Example<'a>],
    pub near: &'a [Example<'a>],
    /// Answers of every kind this lesson's change must leave as they are.
    pub keep: &'a [Example<'a>],
    /// Instead take the last lesson's change back, as if it never ran.
    pub undo: bool,
    /// Bounded steps this time at most.
    pub steps: u32,
    /// More steps on the last lesson's rows, which are not captured again.
    pub more: bool,
    /// The last lesson made a plain change of the model's weights, then checked.
    pub commit: bool,
    /// Every lesson's weight change written into the model's own shards.
    pub save: bool,
}

impl<'a> Default for LearnRequest<'a> {
    /// Mirrors the Zig defaults at `engine_api.zig:229` (`steps = 60`).
    fn default() -> Self {
        LearnRequest {
            train: &[],
            held: &[],
            near: &[],
            keep: &[],
            undo: false,
            steps: 60,
            more: false,
            commit: false,
            save: false,
        }
    }
}

/// A lesson's outcome in order, ending with `done` (a message says why it failed); slices
/// live only during the call.
///
/// Upstream: `engine_api.zig:241  pub const LearnEvent = union(enum) { learned, done }`.
#[derive(Clone, Copy, Debug)]
pub enum LearnEvent<'a> {
    Learned {
        recalled: bool,
        steps: u32,
        loss: f32,
    },
    Done {
        message: &'a str,
    },
}

/// Where a learn request's events go; called on the engine's thread, so it must return at
/// once.
///
/// Upstream: `engine_api.zig:248  pub const LearnSink = struct { ctx, event }`.
pub trait LearnSink {
    fn event(&mut self, event: LearnEvent<'_>);
}

/// A family's learner, which the lane host steps one bounded unit at a time while no
/// stream decodes.
///
/// Upstream: `engine_api.zig:255  pub const Learner = struct { ctx, begin, step, abort }`.
/// In Rust the ctx pointer disappears: an engine's `learn` implementation owns its
/// learner, and the stepping contract (`begin`/`step`/`abort`) becomes the implementor's
/// responsibility, documented here and exercised by the server tests in Phase 3.
pub struct Learner {
    /// Start a job; its events, `done` last, go to `sink` from later steps.
    pub begin: fn(request: &LearnRequest<'_>, sink: &mut dyn LearnSink) -> Result<(), LearnError>,
    /// One unit of the job: `(done, changed)`; `changed` when weights moved (kept prompt
    /// states are then dropped), done when it ended.
    pub step: fn() -> LearnerStep,
    /// End the open job now; it still sends `done`, with the reason.
    pub abort: fn(),
}

#[derive(Clone, Copy, Debug)]
pub struct LearnerStep {
    pub done: bool,
    pub changed: bool,
}
