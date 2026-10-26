//! What a request to the engine carries.
//!
//! Port of the `Request`-side declarations of `zig/src/core/engine_api.zig`
//! (`Stop`, `CallGate`, `Structure`, `Request`).

use crate::sampling::Sampling;

/// The request's stop strings, checked on the engine's thread after each committed token
/// (the server decodes the tail).
///
/// Upstream: `engine_api.zig:19  pub const Stop = struct { ctx, check }` — a ctx + function
/// pointer pair, ported here as an owned box: the request owns its check for as
/// long as it lives (upstream keeps the caller's ctx alive the same way). The
/// callback must not re-enter the engine.
pub trait StopCheck: Send {
    /// `emitted` is every token committed so far, in order; true ends the reply.
    fn should_stop(&mut self, emitted: &[u32]) -> bool;
}

/// tool_choice "required" or a named function: outside a think block the answer opens a
/// call to an offered tool.
///
/// Upstream: `engine_api.zig:25  pub const CallGate = struct { ... }`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CallGate<'a> {
    /// The token that opens a call, the template text before the name, and the mark that
    /// ends the name.
    pub opener: u32,
    pub lead: &'a str,
    pub tail: &'a str,
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    pub names: &'a [&'a str],
    pub think_open: Option<u32>,
    pub think_end: Option<u32>,
}

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
///
/// Upstream: `engine_api.zig:36  pub const Structure = struct { kind, text, after }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureKind {
    Json,
    JsonSchema,
    Regex,
    Choice,
    Grammar,
}

#[derive(Clone, Copy, Debug)]
pub struct Structure<'a> {
    pub kind: StructureKind,
    /// The schema (JSON text), regex, choices (a JSON array) or EBNF; empty for any JSON
    /// object.
    pub text: &'a str,
    /// With thinking on, the grammar starts after this token.
    pub after: Option<u32>,
}

/// Decoded intervals [start, end), sorted, nonempty and strictly separated.
///
/// Upstream encodes these as `[2]u32` pairs; `engine.rs` validates them exactly like
/// `zig/src/core/cache_modes.zig` does when a real engine arrives (Phase 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// A reply to decode.
///
/// Upstream: `engine_api.zig:46  pub const Request = struct { ... }`. The upstream comment
/// on the contract: "The request and every slice in it stay valid until its `finished`
/// event." In Rust the borrows in `Request<'a>` tie the request to the caller's scope;
/// an engine may not hold any of them beyond `Engine::submit` — it must copy what it
/// keeps, which is what the Zig engines do with their `Open`-time allocations.
///
/// No `Clone`/`Copy`: the `stop` field carries a unique callback borrower, exactly one
/// a request, the way the Zig ctx pointer is.
pub struct Request<'a> {
    pub prompt: &'a [u32],
    pub max_tokens: u32,
    pub sampling: Option<Sampling>,
    /// Tokens that end the reply as `stop` (the token is delivered); empty when the
    /// request ignores EOS.
    pub eos: &'a [u32],
    pub stop: Option<Box<dyn StopCheck + 'a>>,
    /// false: one token a round and no drafts, the serial reference drafted replies must
    /// equal.
    pub drafts: bool,
    /// Background work yields to foreground arrivals.
    pub background: bool,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then
    /// shared system blocks.
    pub history_len: u32,
    pub shared_prefixes: &'a [u32],
    /// Where the latest user message's text starts: a state kept before it lets an edit of
    /// that turn resume (0: none).
    pub rewind_len: u32,
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at
    /// `Info.prefill_step`); empty: the engine's own.
    pub chunks: &'a [u32],
    /// Decoded intervals [start,end), sorted, nonempty and strictly separated; an empty
    /// list uses prompt arithmetic.
    pub decode_spans: &'a [Span],
    /// Most reply tokens inside a think block before the engine writes `think_close`
    /// (0: no limit).
    pub think_budget: u32,
    pub think_close: &'a [u32],
    pub think_end: Option<u32>,
    /// The offered tools as JSON text, for drafters that propose a call's structure;
    /// empty: no tools.
    pub tools_json: &'a str,
    /// Set only when `Info` says the engine enforces it.
    pub call: Option<CallGate<'a>>,
    pub structure: Option<Structure<'a>>,
    /// Stop a short exact cycle while the think block is open.
    pub loop_guard: bool,
    /// Logprob rows for each reply token with this many best tokens (0..20), when
    /// `Info.logprobs`; None: none.
    pub logprobs: Option<u8>,
}

impl<'a> Default for Request<'a> {
    /// Mirrors the Zig struct's field defaults (`= true`, `= 0`, `= &.{}` etc.).
    fn default() -> Self {
        Request {
            prompt: &[],
            max_tokens: 0,
            sampling: None,
            eos: &[],
            stop: None,
            drafts: true,
            background: false,
            history_len: 0,
            shared_prefixes: &[],
            rewind_len: 0,
            chunks: &[],
            decode_spans: &[],
            think_budget: 0,
            think_close: &[],
            think_end: None,
            tools_json: "",
            call: None,
            structure: None,
            loop_guard: false,
            logprobs: None,
        }
    }
}
