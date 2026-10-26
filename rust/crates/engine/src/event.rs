//! The events a submitted request raises, and a finished reply's stats.
//!
//! Port of the `Reason`, `Stats` and `Event` declarations of
//! `zig/src/core/engine_api.zig`.

use crate::logprob::LogprobRow;

/// Upstream: `engine_api.zig:81  pub const Reason = enum { stop, length, cancelled, failed };`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reason {
    #[default]
    Stop,
    Length,
    Cancelled,
    Failed,
}

/// What a finished reply's rounds did, for its runtime fields and /metrics.
///
/// Upstream: `engine_api.zig:84  pub const Stats = struct { ... }`. The Zig defaults are
/// empty slices and zeros; Rust uses empty `Vec`s, which the engines fill per reply.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    pub min_rows: u32,
    pub prefill_widths: Vec<u32>,
    pub prefill_raised: Vec<bool>,
    /// The loop period that ended or interrupted the think block, if any.
    pub loop_period: Option<u32>,
    /// Prompt prefill duration measured by the engine host, in seconds.
    pub prefill_seconds: Option<f64>,
    /// The drafter's own counters as a JSON object, or empty.
    pub telemetry_json: String,
}

/// Why a reply ended, with its stats and the failure message when it failed.
///
/// Upstream folds this into `Event.finished`; Rust splits the terminal record out so the
/// borrow on `Event::Tokens`/`Event::Logprobs` rows stays simple. The upstream `message`
/// is a slice valid only during the call; here it is owned (a terminal record, delivered
/// once, so ownership is the cheaper contract).
#[derive(Clone, Debug, Default)]
pub struct Finished {
    pub reason: Reason,
    pub stats: Stats,
    pub message: String,
}

/// One event for a submitted request, in order.
///
/// Upstream: `engine_api.zig:100  pub const Event = union(enum) { prefilled, logprobs,
/// tokens, finished }`. The Zig rows are slices "valid only during the call"; the Rust
/// borrows carry exactly that lifetime.
#[derive(Clone, Debug)]
pub enum Event<'a> {
    /// The prompt is in the cache: that many of its tokens came from a kept prefix.
    Prefilled(u32),
    /// The next `Tokens` event's logprob rows, one a token (`logprobs` requests); valid
    /// only during the call.
    Logprobs(&'a [LogprobRow]),
    /// Tokens committed for this request, in order; valid only during the call.
    Tokens(&'a [u32]),
    /// The reply ended; nothing follows.
    Finished(Finished),
}
