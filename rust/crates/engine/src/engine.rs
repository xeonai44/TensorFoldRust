//! The engine a server drives, and what it reports about itself.
//!
//! Port of the `Info`, `PromptCachePlan`, `Status`, `Memory`, `Engine` and `Sink`
//! declarations of `zig/src/core/engine_api.zig`. The Zig vtable struct becomes a Rust
//! trait: the required slots stay required, and the optional slots (`keepalive`, `score`,
//! `learn`) become default methods returning "unsupported", so implementors grow into
//! them.

use crate::event::{Event, Reason};
use crate::learn::{LearnError, LearnRequest, LearnSink};
use crate::request::Request;
use crate::Id;

/// What the engine serves, fixed once it has loaded.
///
/// Upstream: `engine_api.zig:118  pub const Info = struct { ... }`.
#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub lanes: u32,
    /// Prompt plus reply tokens one request may use (0: no limit).
    pub context_window: u32,
    pub context_fitted: bool,
    /// What the engine enforces; the server refuses requests that need more.
    pub call_gates: bool,
    /// The engine observes loop_guard requests in its lane stream.
    pub loop_guard: bool,
    pub structures: bool,
    /// Requests may ask for `logprobs` (the engine's lanes give the target's rows).
    pub logprobs: bool,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine
    /// cuts prompts itself).
    pub prefill_step: u32,
    /// The engine keeps prompt states between requests, so `prefilled` counts the tokens
    /// it restored.
    pub prompt_cache: bool,
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    pub startup: String,
    /// A prompt-only request (max_tokens 0) keeps its end state: the server prefills each
    /// reply for the next turn.
    pub warm_turns: bool,
    /// The engine decodes plain whatever a request asks: no drafter loaded, or this chip
    /// is faster plain.
    pub plain_only: bool,
    /// The immutable retained-prefix plan applied at startup, not a process memory limit
    /// or current cache occupancy.
    pub prompt_cache_plan: Option<PromptCachePlan>,
    /// The engine decodes greedily only: the server refuses a request with a temperature
    /// before admitting it.
    pub greedy_only: bool,
}

impl Default for Info {
    /// Mirrors the Zig defaults at `engine_api.zig:118` (`lanes = 1`, zeros and false).
    fn default() -> Self {
        Info {
            name: "lanes".to_string(),
            lanes: 1,
            context_window: 0,
            context_fitted: false,
            call_gates: false,
            loop_guard: false,
            structures: false,
            logprobs: false,
            prefill_step: 0,
            prompt_cache: false,
            startup: String::new(),
            warm_turns: false,
            plain_only: false,
            prompt_cache_plan: None,
            greedy_only: false,
        }
    }
}

/// Upstream: `engine_api.zig:147  pub const PromptCachePlan = struct { ... }`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PromptCachePlan {
    pub source: PromptCacheSource,
    pub budget_bytes: u64,
    pub explicit_budget: bool,
    pub over_cap: Option<bool>,
    pub ram_bytes: Option<u64>,
    pub ready_footprint_bytes: Option<u64>,
    pub cap_bytes: Option<u64>,
    pub room_bytes: Option<u64>,
    pub margin_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptCacheSource {
    #[default]
    PhysicalFootprint,
    Explicit,
    MetalWorkingSet,
}

impl PromptCacheSource {
    /// The tag's name, the way `@tagName` prints it.
    pub fn source_name(&self) -> &'static str {
        match self {
            PromptCacheSource::PhysicalFootprint => "PhysicalFootprint",
            PromptCacheSource::Explicit => "Explicit",
            PromptCacheSource::MetalWorkingSet => "MetalWorkingSet",
        }
    }
}

/// Upstream: `engine_api.zig:194  pub const Memory = struct { ... }`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory {
    pub active: u64,
    pub cache: u64,
    pub peak: u64,
    pub pinned: u64,
}

/// Upstream: `engine_api.zig:208  pub const Status = struct { ... }`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Status {
    /// Requests in prefill or decode, and those waiting for a lane.
    pub running: u32,
    pub waiting: u32,
    pub decode_tokens_per_second: f64,
    pub prefill_tokens_per_second: f64,
    /// Requests that gave a lane up to a later one; None when the engine does not count
    /// them.
    pub preemptions: Option<u64>,
    pub warming: bool,
    /// Live streams written to the caller's buffer: tokens each holds.
    pub streams: usize,
    /// Generated tokens held by live streams, excluding their prompts.
    pub generation_tokens: u64,
}

/// Upstream: `engine_api.zig:223  pub const SubmitError = error{ Closed, Busy, InvalidSpans };`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    Closed,
    Busy,
    InvalidSpans,
}

/// Where a request's events go.
///
/// Upstream: `engine_api.zig:112  pub const Sink = struct { ctx, event }` — ported
/// as a boxed trait object the host owns until the finished event. The upstream
/// comment carries over verbatim as the contract: "Called on the engine's thread:
/// it must return at once and not re-enter the engine."
pub trait EventSink: Send {
    fn event(&mut self, id: Id, event: Event<'_>);
}

// Upstream: `engine_api.zig:198  pub const Explain` (a backend's words for its own
// refusals) — folded into this port as the `String` an `Engine::submit`/`open` failure
// carries; a dedicated hook returns with the first real backend (Phase 1).

/// The engine a server drives.
///
/// Upstream: `engine_api.zig:268  pub const Engine = struct { ctx, vtable }`. Required
/// slots: `info`, `submit`, `cancel`, `status`, `memory`. Optional slots, ported as
/// default methods: `keepalive` (Metal-only idle target; returns None until the Metal
/// backend arrives — upstream `engine_api.zig:281`), `score` (`engine_api.zig:283`) and
/// `learn` (`engine_api.zig:285`).
pub trait Engine: Send + Sync {
    fn info(&self) -> Info;

    /// `submit` takes ownership of the request and its sink: upstream's contract
    /// keeps both alive "until its `finished` event" (engine_api.zig:45), which
    /// the port expresses by moving them in — the engine thread holds them until
    /// the finished event has gone out, then drops them (ADR-016).
    fn submit(
        &self,
        id: Id,
        request: Box<Request<'static>>,
        sink: Box<dyn EventSink + Send>,
    ) -> Result<(), SubmitError>;

    /// Ends a request between rounds; its `finished` (cancelled) still arrives. Unknown
    /// ids are ignored.
    fn cancel(&self, id: Id);

    /// Live status; `stream_tokens` receives each live stream's token count.
    fn status(&self, stream_tokens: &mut [u32]) -> Status;

    /// Device memory in bytes, the peak since the last reset; None when the backend keeps
    /// no count.
    fn memory(&self, reset_peak: bool) -> Option<Memory>;

    /// The engine's queue as an idle keepalive target; None until a Metal engine exists.
    fn keepalive(&self) -> Option<()> {
        None // TODO(upstream: engine_api.zig:281, keepalive.zig) — Metal Phase 2.
    }

    /// One logit per label, and the vocabulary logsumexp; None/Unsupported until a family
    /// scores decisions.
    fn score(&self, _prompt: &[u32], _labels: &[u32], _logits: &mut [f64]) -> Result<f64, ScoreError> {
        Err(ScoreError::Unsupported)
    }

    /// Queue `request` for the family's learner; stays valid until `done`.
    fn learn(&self, _request: &LearnRequest, _sink: &mut dyn LearnSink) -> Result<(), LearnError> {
        Err(LearnError::Unsupported)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreError {
    Failed,
    Unsupported,
}

/// A request-reason helper mirroring `Reason::from` uses upstream servers make; kept
/// here so translations share one spelling.
impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Stop => "stop",
            Reason::Length => "length",
            Reason::Cancelled => "cancelled",
            Reason::Failed => "failed",
        }
    }
}
