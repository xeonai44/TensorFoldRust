//! The chat path: messages rendered, submitted to the engine, returned as an
//! OpenAI chat completion.
//!
//! Port of the core flow of `zig/src/server/chat.zig` — the Mailbox (chat.zig:65),
//! the prompt preparation (`prompt.zig`'s `prepare` + `renderIds`), the submit,
//! and the reply body the `/v1/chat/completions` route shapes. The reasoning-block
//! extraction (`reply_text.zig`), tool calls (`tool_*`) and logprob formatting are
//! Phase 2 in the map; the token flow, finish reasons and usage counts are the
//! Phase 1b gate. The streamed reply delivers the whole text in one content
//! chunk followed by the finish chunk; per-round chunks arrive with the
//! streaming sink in Phase 2.

use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::IntoResponse;
use serde_json::{json, Value};
use std::sync::{Arc, Condvar, Mutex};
use axum::response::sse::Event as SseEvent;
use tensorfold_engine::{Event, EventSink, Id, Reason, Request, Sampling, SubmitError};


/// The streaming channel's messages: each engine event, with the tokens as a
/// snapshot the SSE task decodes incrementally.
pub enum StreamEvent {
    Tokens(Vec<u32>),
    Prefilled(u32),
    Finished {
        reason: Reason,
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_tokens: usize,
        cached: u32,
        /// The reply's rounds: the runtime and speculative objects read
        /// them (chat.zig:520-547).
        stats: tensorfold_engine::Stats,
    },
}

/// The streaming Mailbox: the engine thread pushes events, the SSE task
/// receives them. Upstream: chat.zig's Mailbox with the sink calling into the
/// stream (chat.zig's StreamSink).
struct StreamMailbox {
    shared: Arc<(Mutex<MailboxInner>, Condvar)>,
    sender: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

impl EventSink for StreamMailbox {
    fn event(&mut self, _id: Id, event: Event<'_>) {
        let mut inner = self.shared.0.lock().unwrap();
        match event {
            Event::Prefilled(cached) => {
                inner.cached = Some(cached);
                let _ = self.sender.send(StreamEvent::Prefilled(cached));
            }
            Event::Logprobs(rows) => {
                inner.rows.extend_from_slice(rows);
            }
            Event::Tokens(tokens) => {
                inner.tokens.extend_from_slice(tokens);
                // The stream gets the batch — the Incremental decode's contract
                // (reply_text.zig:302 takes `more`, the new tokens only).
                let _ = self
                    .sender
                    .send(StreamEvent::Tokens(tokens.to_vec()));
            }
            Event::Finished(f) => {
                inner.finished = true;
                inner.reason = f.reason;
                let _ = self.sender.send(StreamEvent::Finished {
                    reason: f.reason,
                    prompt_tokens: inner.prompt_tokens.unwrap_or(0),
                    completion_tokens: inner.tokens.len(),
                    reasoning_tokens: 0, // set from the tokens by the caller
                    cached: inner.cached.unwrap_or(0),
                    stats: f.stats,
                });
            }
        }
        self.shared.1.notify_all();
    }
}

/// The events the engine thread hands this request, read by the request's own
/// thread. Upstream: chat.zig's Mailbox (chat.zig:65).
struct Mailbox {
    shared: Arc<(Mutex<MailboxInner>, Condvar)>,
}

struct MailboxInner {
    tokens: Vec<u32>,
    cached: Option<u32>,
    finished: bool,
    reason: Reason,
    prefill_seconds: Option<f64>,
    rounds: u64,
    drafted: u64,
    accepted: u64,
    message: String,
    prompt_tokens: Option<usize>,
    /// One row a token of `tokens`, for requests with logprobs
    /// (chat.zig:72).
    rows: Vec<tensorfold_engine::LogprobRow>,
    /// The reply's rounds whole: the runtime and speculative objects read
    /// the widths and counters the fields above leave out (chat.zig:520).
    stats: tensorfold_engine::Stats,
    /// When the prompt's prefill ended (the decode's start) and the first
    /// and last tokens arrived (chat.zig's Mailbox marks).
    prefilled: Option<std::time::Instant>,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
}

impl Mailbox {
    fn new() -> (Mailbox, Arc<(Mutex<MailboxInner>, Condvar)>) {
        let shared = Arc::new((
            Mutex::new(MailboxInner {
                tokens: Vec::new(),
                cached: None,
                finished: false,
                reason: Reason::Stop,
                prefill_seconds: None,
                rounds: 0,
                drafted: 0,
                accepted: 0,
                message: String::new(),
                prompt_tokens: None,
                rows: Vec::new(),
                stats: tensorfold_engine::Stats::default(),
                prefilled: None,
                first: None,
                last: None,
            }),
            Condvar::new(),
        ));
        (Mailbox { shared: Arc::clone(&shared) }, shared)
    }
}

impl EventSink for Mailbox {
    fn event(&mut self, _id: Id, event: Event<'_>) {
        let mut inner = self.shared.0.lock().unwrap();
        match event {
            Event::Prefilled(cached) => inner.cached = Some(cached),
            Event::Logprobs(_) => {}
            Event::Tokens(tokens) => inner.tokens.extend_from_slice(tokens),
            Event::Finished(f) => {
                inner.finished = true;
                inner.reason = f.reason;
                inner.prefill_seconds = f.stats.prefill_seconds;
                inner.rounds = f.stats.rounds;
                inner.drafted = f.stats.drafted;
                inner.accepted = f.stats.accepted;
                inner.message = f.message;
                inner.stats = f.stats;
            }
        }
        self.shared.1.notify_all();
    }
}

/// `ToolCallPolicy.finish`: the reply's calls parsed out of its content.
///
/// Upstream: `attachCalls` (openai.zig:309) - the one-call policy keeps the
/// first call and filters the content; the calls null the message's content
/// and mark the finish `tool_calls`.
pub fn attach_calls(
    reply: &mut Reply,
    tools: &[Value],
    policy: &crate::tool_stream::Policy,
) -> Option<Vec<Value>> {
    if tools.is_empty() {
        return None;
    }
    let parsed = crate::tool_parse::parse(&reply.content, tools, policy.max_calls());
    let Some(calls) = parsed.calls else {
        if policy.single {
            reply.content = policy.parsed_content(&parsed.content);
        }
        return None;
    };
    reply.content = policy.parsed_content(&parsed.content);
    reply.finish_reason = "tool_calls";
    let calls = if policy.single && calls.len() > 1 {
        calls.into_iter().take(1).collect()
    } else {
        calls
    };
    reply.tool_calls = Some(calls.clone());
    Some(calls)
}

/// The reply the routes shape. Upstream: chat.zig's Reply (chat.zig:43).
pub struct Reply {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Option<Vec<Value>>,
    pub finish_reason: &'static str,
    /// The switch the reply ran under and the effort it named (chat.zig:43).
    pub thinking: bool,
    pub effort: Option<String>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub failed: Option<crate::errors::Refused>,
    pub usage: Option<Value>,
    pub stop_sequence: Option<String>,
    /// ``choices[0].logprobs`` (``{"content": [...]}``) for a request that
    /// asked (chat.zig:58).
    pub logprobs: Option<Value>,
    /// The reply's runtime object: the timings, the token sha, the effort it
    /// ran under (chat.zig:513). The response extras carry it.
    pub runtime: Value,
    /// The draft rounds' counters (chat.zig:542). The response extras carry
    /// it.
    pub speculative: Value,
}

/// One request through the engine: render, submit, wait, decode. Upstream: the
/// chat flow around `Server.submit` (chat.zig:200+), with the body's own fields
/// over the server's defaults (upstream's `k in body` precedence).
/// What a stream hands its bridge beyond the events: the prompt's length,
/// the thinking switch, the request's stop strings with ignore_eos (the
/// finish strips the trailing eos tokens without it, chat.zig:474), and the
/// runtime object's inputs (chat.zig:513-547): the effort, the sampling
/// mode, the drafts flag, the engine's name, and when the request arrived.
pub struct StreamStart {
    /// The request's engine id — the bridge's cancel on a client
    /// disconnect keys it (upstream's `Generation.id`).
    pub id: u64,
    pub receiver: tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
    pub prompt_len: usize,
    pub thinking: bool,
    pub think_end: Option<u32>,
    pub stops: Vec<String>,
    pub ignore_eos: bool,
    pub effort: Option<String>,
    pub exact: bool,
    pub drafts: bool,
    pub engine: String,
    pub received: std::time::Instant,
    /// The engine's context window (the runtime's stamp fields).
    pub window: u32,
    /// The compaction object of the stamp when the reply compacted.
    pub compaction: Option<Value>,
}

/// Submit for streaming: the same prompt render and request build as
/// `generate`, but the sink pushes events through a channel the SSE task
/// reads. Returns the receiver and the rendered prompt's length.
pub fn submit_streaming(
    state: &AppState,
    raw: &Value,
) -> Result<StreamStart, crate::errors::Refused> {
    // The same prepared request the non-streaming path runs — through
    // compaction when the server compacts (openai.zig:465's compact.prepare):
    // the stream only differs in its mailbox
    let (prepared, compaction) = match state.config.compact_at {
        Some(at) => match crate::compact::prepare_compact(state, raw, at) {
            Ok((p, stamp)) => (p, stamp.object()),
            Err(r) => return Err(r),
        },
        None => (prepare(state, raw)?, Value::Null),
    };
    submit_prepared_streaming(state, prepared, compaction)
}

/// The streaming submit over an already-prepared request (the completions
/// route's path: its plan differs, the mailbox and the engine submit the
/// chat stream's own).
pub fn submit_prepared_streaming(
    state: &AppState,
    prepared: Prepared,
    compaction: Value,
) -> Result<StreamStart, crate::errors::Refused> {
    let Prepared {
        request,
        prompt_len,
        thinking,
        effort,
        exact,
        drafts,
        stops: stop_strings,
        ignore_eos,
        received: started,
        ..
    } = prepared;
    let text = state
        .text
        .as_ref()
        .ok_or_else(|| crate::errors::Refused::request(state.engineless_problem.clone()))?;
    let id = state.next_request_id();
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let mailbox = StreamMailbox {
        shared: Arc::new((
            Mutex::new(MailboxInner {
                tokens: Vec::new(),
                cached: None,
                finished: false,
                reason: Reason::Stop,
                prefill_seconds: None,
                rounds: 0,
                drafted: 0,
                accepted: 0,
                message: String::new(),
                // The prompt length for the finished event's usage.
                prompt_tokens: Some(prompt_len),
                rows: Vec::new(),
                stats: tensorfold_engine::Stats::default(),
                prefilled: None,
                first: None,
                last: None,
            }),
            Condvar::new(),
        )),
        sender,
    };
    state
        .engine
        .as_ref()
        .ok_or_else(|| crate::errors::Refused::other("the scheduler is closed"))?
        .submit(id, request, Box::new(mailbox))
        .map_err(|e| match e {
            // The engine's own failures (chat.zig:238-242)
            SubmitError::Busy => crate::errors::Refused::capacity("the engine is busy; retry shortly"),
            SubmitError::Closed => crate::errors::Refused::other("the scheduler is closed"),
            SubmitError::InvalidSpans => {
                crate::errors::Refused::request("the prompt span layout is invalid")
            }
        })?;
    Ok(StreamStart {
        id,
        receiver,
        prompt_len,
        thinking,
        think_end: text.think_end,
        stops: stop_strings,
        ignore_eos,
        effort,
        exact,
        drafts,
        engine: state
            .engine_info
            .as_ref()
            .map(|i| i.name.clone())
            .unwrap_or_default(),
        received: started,
        window: state
            .engine_info
            .as_ref()
            .map(|i| i.context_window)
            .unwrap_or(0),
        compaction: compaction.as_object().map(|_| compaction.clone()),
    })
}

/// The engine's stop check, here: the newest tokens' text holds a stop
/// string. Upstream: `StopHook` (chat.zig:571) — the tail decoded fresh each
/// check, its length the longest stop's characters plus eight
/// (`reply_text.Stops.tail`); a decode that cannot happen reads as no stop.
struct StopHook {
    strings: Vec<String>,
    tail: usize,
    text: Arc<crate::Text>,
}

impl StopHook {
    fn new(strings: Vec<String>, text: Arc<crate::Text>) -> StopHook {
        let view: Vec<&str> = strings.iter().map(String::as_str).collect();
        let tail = crate::reply_text::Stops { strings: &view }.tail();
        StopHook { strings, tail, text }
    }
}

impl tensorfold_engine::StopCheck for StopHook {
    fn should_stop(&mut self, emitted: &[u32]) -> bool {
        let start = emitted.len().saturating_sub(self.tail);
        let text = self.text.tokenizer.decode(&emitted[start..]);
        self.strings.iter().any(|s| text.contains(s.as_str()))
    }
}

/// A reply's token SHA: SHA-256 of the comma-joined ids, the first six
/// bytes as lowercase hex. Upstream: `tokenSha` (tokens.zig:101).
fn token_sha(tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            h.update(b",");
        }
        h.update(t.to_string().as_bytes());
    }
    let d = h.finalize();
    d.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// The reply's runtime and speculative objects (chat.zig:513-547): the
/// thinking and effort it ran under, the engine's name, the decode
/// timings from the prefill's end, the first token's wait, the token sha,
/// and the draft rounds' counters.
#[allow(clippy::too_many_arguments)]
fn runtime_objects(
    thinking: bool,
    effort: Option<&str>,
    engine: &str,
    exact: bool,
    drafts: bool,
    tokens: &[u32],
    stats: &tensorfold_engine::Stats,
    received: std::time::Instant,
    prefilled: Option<std::time::Instant>,
    first: Option<std::time::Instant>,
    finished: std::time::Instant,
    window: u32,
    used: usize,
) -> (Value, Value) {
    let decode_s = prefilled
        .map(|p| finished.duration_since(p).as_secs_f64().max(0.0))
        .unwrap_or(0.0);
    let decode_tokens = tokens.len().saturating_sub(1);
    let mut runtime = json!({
        "enable_thinking": thinking,
        "reasoning_effort": if !thinking {
            json!("none")
        } else {
            match effort {
                Some(e) => json!(e),
                None => Value::Null,
            }
        },
        "engine": engine,
        // The window and the prompt's use of it (compact.zig:106's stamp,
        // always on)
        "context_window": window,
        "context_used": used,
        // The reply's decode seconds, as tokens_per_second divides them
        "tokens_per_second": if decode_s > 0.0 {
            decode_tokens as f64 / decode_s
        } else {
            0.0
        },
        "seconds": decode_s,
        "prefill_seconds": stats.prefill_seconds,
        "prefill_widths": stats.prefill_widths,
        "prefill_raised": stats.prefill_raised,
        "time_to_first_token": first.map(|f| f.duration_since(received).as_secs_f64()),
        "sampling": if exact { "exact" } else { "greedy" },
        "drafts": drafts,
        "token_sha": token_sha(tokens),
        "min_rows": stats.min_rows,
    });
    if let Some(period) = stats.loop_period {
        runtime["loop"] = json!({"period": period});
    }
    let mut speculative = json!({
        "rounds": stats.rounds,
        "drafted": stats.drafted,
        "accepted": stats.accepted,
        "acceptance_rate": if stats.drafted > 0 {
            stats.accepted as f64 / stats.drafted as f64
        } else {
            0.0
        },
        "tokens_per_round": if stats.rounds > 0 {
            tokens.len() as f64 / stats.rounds as f64
        } else {
            0.0
        },
    });
    if !stats.telemetry_json.is_empty() {
        if let Ok(proposer) = serde_json::from_str::<Value>(&stats.telemetry_json) {
            speculative["proposer"] = proposer;
        }
    }
    (runtime, speculative)
}

/// `response_extras`: exact_mode, the matched stop, the runtime and the
/// draft counters, on the whole body and the streamed last chunk.
/// Upstream: `extras` (openai.zig:300).
pub(crate) fn response_extras(body: &mut Value, stop_sequence: Option<&str>, runtime: &Value, speculative: &Value) {
    body["exact_mode"] = json!("exact");
    if let Some(stop) = stop_sequence {
        body["stop_sequence"] = json!(stop);
    }
    body["tensorfold"] = runtime.clone();
    body["speculative"] = speculative.clone();
}

/// The reply to a body through whichever path the server runs (chat.zig:258's
/// `run`): compaction's when the server compacts, the plain generate
/// otherwise.
pub fn run(state: &AppState, body: &Value) -> Reply {
    match state.config.compact_at {
        Some(at) => crate::compact::run(state, body, at),
        None => generate(state, body),
    }
}

/// Upstream: `replyModel` (openai.zig:105) — the request's own model name
/// when it names a served id, else the served name. The chat bodies' `model`
/// and the Responses base's both read it.
pub fn reply_model(state: &AppState, body: &Value) -> String {
    if let Some(m) = body.get("model").and_then(Value::as_str) {
        if state.config.model_ids.iter().any(|known| known == m) {
            return m.to_string();
        }
    }
    state.config.served_name.clone()
}

/// The whole chat completion body the chat route sends and the Responses
/// route feeds its machine (openai.zig:376's `wholeBody`): the id, the
/// object, the created time, the model, the choice with the message and its
/// calls, the usage, and the response extras.
pub(crate) fn whole_body(
    id: &str,
    model: &str,
    created: u64,
    reply: &Reply,
    calls: Option<&[Value]>,
) -> Value {
    let mut whole = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    // The content nulls when the reply calls a tool
                    // (openai.zig:387)
                    "content": if calls.is_some() { Value::Null } else { Value::String(reply.content.clone()) },
                    "reasoning_content": if reply.reasoning.is_empty() {
                        Value::Null
                    } else {
                        Value::String(reply.reasoning.clone())
                    },
                    "tool_calls": if let Some(c) = calls { Value::Array(c.to_vec()) } else { Value::Null },
                },
                "finish_reason": reply.finish_reason,
                "logprobs": null,
            }],
            "usage": {
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "total_tokens": reply.prompt_tokens + reply.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": reply.cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": reply.reasoning_tokens},
            },
    });
    // response_extras (openai.zig:300): exact_mode, the matched stop, the
    // runtime and the draft counters
    response_extras(
        &mut whole,
        reply.stop_sequence.as_deref(),
        &reply.runtime,
        &reply.speculative,
    );
    // The logprobs ride the choice when the request asked
    // (openai.zig:395)
    if let Some(lp) = &reply.logprobs {
        whole["choices"][0]["logprobs"] = lp.clone();
    }
    whole
}

/// The reply with its usage and stop sequence filled (the OpenAI body's shape,
/// the Messages route's input). The finish_reason, prompt/completion counts,
/// reasoning and reasoning tokens come from generate; this adds what the
/// response bodies read.
pub fn generate_with(state: &AppState, body: &Value) -> Result<Reply, crate::errors::Refused> {
    let mut reply = run(state, body);
    if let Some(refused) = reply.failed.clone() {
        return Err(refused);
    }
    reply.usage = Some(json!({
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": reply.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reply.reasoning_tokens},
    }));
    Ok(reply)
}

/// The fields the routes pass the engine-side request checks: the body's
/// sampling-ish fields with the thinking fields beside them (openai.zig:139-
/// 155's passthrough set; the grammar fields and tool_call_required ride it
/// upstream too, with the features that read them).
const SAMPLING_KEYS: [&str; 16] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "priority",
    "draft",
    "thinking_budget",
    "ignore_eos",
    "stop",
    // The grammar fields ride the same passthrough (openai.zig:146)
    "response_format",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "structured_outputs",
];

/// The request's fields object: the sampling-ish passthrough with the
/// request's thinking fields (openai.zig:139-155).
fn fields_object(body: &Value, thinking_opts: &crate::fields::Thinking) -> Value {
    let mut fields = serde_json::Map::new();
    for k in SAMPLING_KEYS {
        if let Some(v) = body.get(k) {
            fields.insert(k.into(), v.clone());
        }
    }
    if let Some(e) = &thinking_opts.effort {
        fields.insert("reasoning_effort".into(), json!(e));
    }
    if let Some(on) = thinking_opts.enable {
        fields.insert("enable_thinking".into(), json!(on));
    }
    Value::Object(fields)
}

/// A request seed as the engine's u64: negative and huge ints wrap as two's
/// complement bits. Upstream: `seedBits` (server.zig:303).
fn seed_bits(v: Option<&Value>) -> u64 {
    match v {
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_i64().map(|i| i as u64))
            .unwrap_or(0),
        _ => 0,
    }
}

/// ``_resolve_sampling``: omitted or null fields keep the defaults; an
/// omitted seed is keyed to the prompt. Upstream: `resolveSampling`
/// (server.zig:135) — the server's default sampling merged under the
/// request's fields, the temperature gate (0 or less is greedy), the
/// greedy-only engine's refusal, the seed by bits or keyed to the prompt.
pub fn resolve_sampling(
    state: &AppState,
    fields: &Value,
    prompt_ids: &[u32],
) -> Result<Option<Sampling>, String> {
    let mut options = serde_json::Map::new();
    if let Some(defaults) = &state.config.default_sampling {
        for (k, v) in crate::fields::parse_numbers(defaults)?
            .as_object()
            .into_iter()
            .flatten()
        {
            if !v.is_null() {
                options.insert(k.clone(), v.clone());
            }
        }
    }
    for (k, v) in crate::fields::parse_numbers(fields)?
        .as_object()
        .into_iter()
        .flatten()
    {
        if !v.is_null() {
            options.insert(k.clone(), v.clone());
        }
    }
    let temp = options.get("temperature").and_then(Value::as_f64).unwrap_or(0.0);
    if temp <= 0.0 {
        return Ok(None);
    }
    if state.engine_info.as_ref().is_some_and(|i| i.greedy_only) {
        let defaults_temp = state
            .config
            .default_sampling
            .as_ref()
            .and_then(|d| d.get("temperature"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        return Err(if defaults_temp > 0.0 {
            "this engine decodes greedily only and this server samples by default: send temperature 0, or serve with --temperature 0".into()
        } else {
            "this engine decodes greedily only: send temperature 0 or leave it out".into()
        });
    }
    let seed = match options.get("seed") {
        Some(s) => seed_bits(Some(s)),
        None => tensorfold_engine::sample_seed_for(prompt_ids, state.config.seed_salt),
    };
    Ok(Some(Sampling {
        seed,
        temperature: temp,
        top_k: options
            .get("top_k")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .clamp(0, u32::MAX as i64) as u32,
        top_p: options.get("top_p").and_then(Value::as_f64).unwrap_or(1.0),
        min_p: options.get("min_p").and_then(Value::as_f64).unwrap_or(0.0),
    }))
}

/// The no-room refusal's message (chat.zig:166): the context limit's
/// wording, the fitted window's note, the prompt's count.
fn context_no_room(window: i64, fitted: bool, n: i64) -> String {
    let fitted = if fitted {
        ", the most this server's memory budget fits"
    } else {
        ""
    };
    format!(
        "{} {window} tokens{fitted}, but the rendered prompt has {n} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.",
        crate::errors::CONTEXT_LIMIT
    )
}

/// The reply-too-big refusal's message (chat.zig:168): the counts and the
/// two ways to fit.
fn context_too_big(window: i64, n: i64, limit: u32) -> String {
    format!(
        "{} {window} tokens, but the rendered prompt has {n} tokens and requests {limit} reply tokens, which exceeds the context window. Reduce the prompt to at most {} prompt tokens or request at most {} reply tokens, including chat template and thinking tokens.",
        crate::errors::CONTEXT_LIMIT,
        0.max(window - limit as i64),
        window - n
    )
}

/// (opener, lead, tail) of this template's tool calls. Upstream:
/// `Form` (server.zig:182) — the lead null when the opener alone is a token.
struct CallForm {
    opener: &'static str,
    lead: Option<String>,
    tail: Option<String>,
}

/// The form the rendered probe text shows. Upstream: `callForm`'s reading
/// (server.zig:194-201) — the last call opener before the probe's function
/// name (ties to the greater opener), the text between them the lead, the
/// one byte after the name the tail.
fn find_call_form(rendered: &str) -> Option<CallForm> {
    let at = rendered.rfind("tfprobe_fn")?;
    let mut best: Option<usize> = None;
    let mut opener: &str = "";
    for &(c, _) in crate::reply_text::CALLS.iter() {
        if let Some(s) = rendered[..at].rfind(c) {
            if best.is_none_or(|b: usize| s > b || (s == b && c > opener)) {
                best = Some(s);
                opener = c;
            }
        }
    }
    let s = best?;
    let end = at + "tfprobe_fn".len();
    Some(CallForm {
        opener,
        lead: Some(rendered[s + opener.len()..at].to_string()),
        tail: Some(if end < rendered.len() {
            rendered[end..end + 1].to_string()
        } else {
            String::new()
        }),
    })
}

/// This template's tool calls' form, read from a rendered probe call.
/// Upstream: `callForm` (server.zig:184) — an assistant tool call rendered
/// without thinking or a generation prompt; a render that fails reads as no
/// text; an opener the tokenizer knows alone counts when the probe shows no
/// name (server.zig:203).
fn call_form(text: &crate::Text) -> Option<CallForm> {
    let probe = json!([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "tfprobe_fn", "arguments": {}}}
        ]},
    ]);
    let probe: Vec<Value> = probe.as_array().cloned().unwrap_or_default();
    let ids = tensorfold_core::render_ids(
        &text.template,
        &text.tokenizer,
        &probe,
        tensorfold_core::RenderOptions {
            add_generation_prompt: false,
            enable_thinking: false,
            tools: Some(&[]),
            reasoning_effort: None,
        },
    )
    .unwrap_or_default();
    let rendered = text.tokenizer.decode(&ids);
    if find_call_form(&rendered).is_some() {
        return find_call_form(&rendered);
    }
    for &(c, _) in crate::reply_text::CALLS.iter() {
        if text.tokenizer.special_token_id(c).is_some() {
            return Some(CallForm {
                opener: c,
                lead: None,
                tail: None,
            });
        }
    }
    None
}

/// The call gate a request's tool_choice asks for. Upstream:
/// `checkFeatures`'s gate branch (server.zig:162-178) — the required choice
/// needs a template that marks calls (its opener a token the tokenizer
/// knows) and an engine that enforces gates; the gate names the offered
/// tools for the XML form, and the think block it must open past.
fn call_gate(
    state: &AppState,
    text: &crate::Text,
    fields: &Value,
    active_tools: &[Value],
) -> Result<Option<tensorfold_engine::CallGate<'static>>, crate::errors::Refused> {
    let required = fields
        .get("tool_call_required")
        .is_some_and(crate::pyrepr::truthy);
    if active_tools.is_empty() || !required {
        return Ok(None);
    }
    let form = call_form(text);
    let opener = form
        .as_ref()
        .and_then(|f| text.tokenizer.special_token_id(f.opener));
    if opener.is_none() {
        return Err(crate::errors::Refused::request(format!(
            "tool_choice \"required\" or a named function needs a chat template that marks tool calls ({} or <|tool_call>), and this one does not: send \"auto\"",
            crate::tool_parse::OPENER
        )));
    }
    if !state.engine_info.as_ref().is_some_and(|i| i.call_gates) {
        return Err(crate::errors::Refused::request(
            "tool_choice \"required\" or a named function is not supported by this engine yet: send \"auto\"",
        ));
    }
    let lead = form.as_ref().and_then(|f| f.lead.clone()).unwrap_or_default();
    // The XML form (a lead before the name) completes a name from the
    // offered tools; the single-token form needs none (server.zig:168)
    let names: Vec<String> = if form.as_ref().and_then(|f| f.lead.as_ref()).is_some() {
        active_tools
            .iter()
            .map(crate::tool_specs::tool_name)
            .collect()
    } else {
        Vec::new()
    };
    // The think block the answer opens past (server.zig:169-173): the
    // markers' open token, else the plain think opener
    let think_open = if !text.markers.open.is_empty() {
        text.tokenizer.encode(text.markers.open).first().copied()
    } else {
        text.tokenizer.special_token_id(crate::reply_text::THINK_OPEN)
    };
    let leaked_names: Vec<&'static str> = names
        .iter()
        .map(|n| Box::leak(n.clone().into_boxed_str()) as &'static str)
        .collect();
    Ok(Some(tensorfold_engine::CallGate {
        opener: opener.unwrap(),
        lead: Box::leak(lead.into_boxed_str()),
        tail: Box::leak(
            form.as_ref()
                .and_then(|f| f.tail.clone())
                .unwrap_or_default()
                .into_boxed_str(),
        ),
        names: Box::leak(leaked_names.into_boxed_slice()),
        think_open,
        think_end: text.think_end,
    }))
}

/// The reply-token limit: the request's max_tokens, else its
/// max_completion_tokens, else the server's default; 0 means the default,
/// less than 1 clamps to 1 (chat.zig:151's limit math over openai.zig:134-
/// 139's read).
fn max_tokens_of(state: &AppState, body: &Value) -> (bool, u32) {
    let first = body.get("max_tokens").filter(|v| crate::pyrepr::truthy(v));
    let max = first.or_else(|| body.get("max_completion_tokens"));
    let named = max.is_some();
    let named_value: Option<i64> = match max {
        None => None,
        Some(Value::Number(n)) => Some(n.as_i64().unwrap_or(if n.to_string().starts_with('-') {
            -1
        } else {
            i32::MAX as i64
        })),
        // a truthy non-number: upstream stores 0, which means the default
        Some(_) => Some(0),
    };
    let limit = match named_value {
        Some(v) if v != 0 => v,
        _ => state.config.default_max_tokens,
    };
    let limit = limit.max(1);
    (named, limit.min(u32::MAX as i64) as u32)
}

/// The rendered prompt's token count (anthropic.zig:47 countTokens).
pub fn count_tokens(state: &AppState, body: &Value) -> Result<usize, String> {
    let Some(text) = state.text.as_ref() else {
        return Err(state.engineless_problem.clone());
    };
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| crate::errors::Refused::request("messages must be a list"))?;
    // The thinking fields as the count renders them (anthropic.zig:51:
    // thinkingFields; the switch orelse the server's config, the effort
    // through effortFor)
    let levels: Vec<&str> = text.effort_levels.iter().map(String::as_str).collect();
    let thinking_opts = crate::fields::thinking_fields(body, &levels).map_err(crate::errors::Refused::request)?;
    let thinking = thinking_opts.enable.unwrap_or(state.config.enable_thinking);
    let effort = thinking_opts
        .effort
        .as_deref()
        .or(state.config.reasoning_effort.as_deref())
        .map(|e| crate::fields::coerce_effort(e, &levels).to_string());
    // The tools the template offers (anthropic.zig:53)
    let active_tools = crate::tool_specs::active(body.get("tools"), body.get("tool_choice"))?;
    tensorfold_core::render_ids(
        &text.template,
        &text.tokenizer,
        messages,
        tensorfold_core::RenderOptions {
            add_generation_prompt: true,
            enable_thinking: thinking,
            reasoning_effort: effort.as_deref(),
            tools: Some(&active_tools),
        },
    )
    .map_err(|e| e.to_string())
    .map(|ids| ids.len())
}

/// A request rendered and checked but not submitted. Upstream: chat.zig:131
/// `Prepared` — it holds what the reply's finish needs beside the request,
/// and (upstream's `preparing` aside, which the port's eager submit does not
/// carry) nothing else.
pub struct Prepared {
    pub request: Box<Request<'static>>,
    pub prompt_len: usize,
    pub thinking: bool,
    pub effort: Option<String>,
    /// A sampling config rode the request (the runtime's "exact").
    pub exact: bool,
    pub drafts: bool,
    /// The tools the template offered (the streamed bridge parses with
    /// them).
    pub tools: Vec<Value>,
    pub stops: Vec<String>,
    pub ignore_eos: bool,
    pub logprobs_count: Option<u8>,
    pub received: std::time::Instant,
}

/// The failed Reply every refusal path answers with.
pub(crate) fn failed_reply(refused: crate::errors::Refused) -> Reply {
    Reply {
        thinking: false,
        effort: None,
        reasoning: String::new(),
        content: String::new(),
        tool_calls: None,
        finish_reason: "stop",
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
        reasoning_tokens: 0,
        usage: None,
        stop_sequence: None,
        logprobs: None,
        runtime: Value::Null,
        speculative: Value::Null,
        failed: Some(refused),
    }
}

/// ``render_prompt_ids`` (prompt.zig:15): messages rendered, and a dangling
/// think opener closed — with thinking off and a generation prompt, a
/// template that ends its prompt on the opener gains the close token, so
/// the reply starts in its content (prompt.zig:27-30).
pub(crate) fn render_prompt(
    text: &crate::Text,
    messages: &[Value],
    thinking: bool,
    effort: Option<&str>,
    generation: bool,
    tools: &[Value],
) -> Result<Vec<u32>, crate::errors::Refused> {
    let mut ids = tensorfold_core::render_ids(
        &text.template,
        &text.tokenizer,
        messages,
        tensorfold_core::RenderOptions {
            add_generation_prompt: generation,
            enable_thinking: thinking,
            reasoning_effort: if thinking { effort } else { None },
            tools: if tools.is_empty() { None } else { Some(tools) },
        },
    )
    .map_err(|e| crate::errors::Refused::server(format!("the template: {e}")))?;
    if !thinking && generation && !ids.is_empty() {
        let last = text.tokenizer.decode(&ids[ids.len() - 1..]);
        // The opener the reply writes itself (prompt.zig:29): the separate
        // constant, not the markers' own open, which templates that use
        // these markers leave empty
        if crate::reply_text::py_strip(&last) == crate::reply_text::THINK_OPEN {
            if let Some(close) =
                text.tokenizer.special_token_id(crate::reply_text::THINK_MARKERS.close)
            {
                ids.push(close);
            }
        }
    }
    Ok(ids)
}

/// The history a prompt shares with the conversation without its generation
/// suffix (prompt.zig:104-108): the no-generation render's length when it
/// prefixes the prompt; a template without a suffix falls to the prompt
/// less one, so nothing retains everything.
fn history_len(prompt: &[u32], history: &[u32]) -> usize {
    if !history.is_empty() && history.len() < prompt.len() && prompt[..history.len()] == history[..]
    {
        return history.len();
    }
    if prompt.len() > 1 && prompt == history {
        return prompt.len() - 1;
    }
    0
}

/// Where the latest user message's text starts (prompt.zig:111): the prompt
/// and a probe there agree to it — the user's content swapped for the
/// invisible separator.
pub(crate) fn rewind_len(
    text: &crate::Text,
    messages: &[Value],
    tools: &[Value],
    prompt: &[u32],
    thinking: bool,
    effort: Option<&str>,
) -> usize {
    let Some(user_index) = (0..messages.len()).rev().find(|&i| {
        messages[i]
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|r| r == "user")
    }) else {
        return 0;
    };
    let mut list = messages.to_vec();
    let Some(user) = messages[user_index].as_object() else {
        return 0;
    };
    let mut probed = user.clone();
    probed.insert("content".into(), json!("\u{2063}probe"));
    list[user_index] = Value::Object(probed);
    let Ok(probe) = render_prompt(text, &list, thinking, effort, true, tools) else {
        return 0;
    };
    match prompt
        .iter()
        .zip(&probe)
        .position(|(a, b)| a != b)
        .unwrap_or(prompt.len().min(probe.len()))
    {
        i if i < prompt.len() => i,
        _ => 0,
    }
}

/// A reusable system prefix, found with a probe in place of the first user
/// message; zero below 512 tokens (prompt.zig:127).
fn system_prefix_len(
    text: &crate::Text,
    messages: &[Value],
    tools: &[Value],
    prompt: &[u32],
    thinking: bool,
    effort: Option<&str>,
) -> usize {
    let Some(first_user) = messages
        .iter()
        .position(|m| m.get("role").and_then(Value::as_str) == Some("user"))
    else {
        return 0;
    };
    let mut probe: Vec<Value> = messages[..first_user].to_vec();
    probe.push(json!({"role": "user", "content": "\u{2063}probe"}));
    let Ok(other) = render_prompt(text, &probe, thinking, effort, true, tools) else {
        return 0;
    };
    let shared = prompt
        .iter()
        .zip(&other)
        .take_while(|(a, b)| a == b)
        .count();
    if shared >= 512 {
        shared
    } else {
        0
    }
}

/// The template's generation prompt (prompt.zig:36): what a probe
/// conversation renders past its history, with and without thinking.
fn generation_prompt(
    text: &crate::Text,
    thinking: bool,
) -> Vec<u32> {
    let talk = vec![json!({"role": "user", "content": "x"})];
    let normalized = crate::messages::normalize(Some(&Value::Array(talk)), "system", false)
        .unwrap_or_default();
    let normalized = crate::messages::tool_arguments(normalized);
    let with = render_prompt(text, &normalized, thinking, None, true, &[]).unwrap_or_default();
    let without = render_prompt(text, &normalized, thinking, None, false, &[]).unwrap_or_default();
    if without.len() >= with.len() || with[..without.len()] != without[..] {
        return Vec::new();
    }
    with[without.len()..].to_vec()
}

/// Earlier replies' decoded rows: from each turn's generation prompt end
/// through its closing stop token (prompt.zig:52). The engine keys cached
/// decode states by them.
pub(crate) fn reply_spans(text: &crate::Text, prompt: &[u32]) -> Vec<[u32; 2]> {
    if text.eos.is_empty() {
        return Vec::new();
    }
    let openers = [
        generation_prompt(text, false),
        generation_prompt(text, true),
    ];
    let mut spans = Vec::new();
    let mut i = 0;
    while i < prompt.len() {
        let mut start: Option<usize> = None;
        let mut at_min = prompt.len();
        for o in &openers {
            if o.is_empty() {
                continue;
            }
            if let Some(at) = prompt[i..]
                .windows(o.len())
                .position(|w| w == o)
                .map(|at| at + i)
            {
                if at < at_min {
                    at_min = at;
                    start = Some(at + o.len());
                }
            }
        }
        let Some(first) = start else { break };
        let Some(end) = prompt[first..]
            .iter()
            .position(|t| text.eos.contains(t))
            .map(|e| e + first)
        else {
            break;
        };
        spans.push([first as u32, (end + 1) as u32]);
        i = end + 1;
    }
    spans
}

/// Render ``body`` and run every check that can refuse it. Upstream:
/// chat.zig:145's `prepare`.
pub(crate) fn prepare(state: &AppState, raw: &Value) -> Result<Prepared, crate::errors::Refused> {
    // ``parseNumbers`` first (openai.zig:120): a body that is not an
    // object fails there as Python's AttributeError does (the Other
    // kind, the bare message), before the messages read
    crate::fields::require_object(raw)?;
    crate::fields::parse_numbers(raw).map_err(crate::errors::Refused::request)?;
    // Upstream's normalize happens at the route's plan (openai.zig:126) —
    // a missing messages the normalize's own refusal (the non-empty
    // wording); compaction assembles its own already-normalized list and
    // hands it straight to the prepare below (compact.zig:176's
    // chat.prepare).
    let normalized = crate::messages::normalize(
        raw.get("messages"),
        "system",
        state
            .text
            .as_ref()
            .map(|t| t.needs_user_after_tool)
            .unwrap_or_default(),
    )
    .map_err(crate::errors::Refused::request)?;
    let normalized = crate::messages::tool_arguments(normalized);
    prepare_messages(state, raw, &normalized)
}

/// The prepared request over an already-normalized message list. Upstream:
/// chat.zig:145's prepare over its Input.
pub(crate) fn prepare_messages(
    state: &AppState,
    raw: &Value,
    normalized: &[Value],
) -> Result<Prepared, crate::errors::Refused> {
    let received = std::time::Instant::now();
    let Some(text) = state.text.as_ref() else {
        return Err(crate::errors::Refused::request(state.engineless_problem.clone()));
    };
    if state.engine.is_none() {
        return Err(crate::errors::Refused::request(state.engineless_problem.clone()));
    }
    // The body with its numeric fields as numbers (openai.zig:120's
    // parseNumbers read): everything downstream reads the coerced copy
    let body = match crate::fields::parse_numbers(raw) {
        Ok(b) => b,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    let body = &body;
    // The request body appended when a log is named (openai.zig:123) —
    // images redacted, background work skipped
    if let Some(path) = &state.config.request_log {
        crate::request_log::append(path, body);
    }
    // A malformed grammar, or one beside a required call, refused before
    // anything else (openai.zig:145's grammar.refusal)
    if let Err(m) = crate::grammar::refusal(body) {
        return Err(crate::errors::Refused::request(m));
    }
    let messages: &[Value] = normalized;
    let (named_max, max_tokens) = max_tokens_of(state, body);
    // The tools the template offers (tool_specs.zig:58's active): none for
    // "none", only the named one for a named function
    let active_tools = match crate::tool_specs::active(body.get("tools"), body.get("tool_choice")) {
        Ok(tools) => tools,
        Err(m) => return Err(crate::errors::Refused::other(m)),
    };
    // logprobs (openai.zig:124-125): the rows need each token's exact bytes;
    // 0.6.6's set — nonstreamed chat without tools, stop strings, structured
    // output or a thinking budget (openai.zig:152-158)
    let exact_bytes = body.get("logprobs").is_some_and(crate::pyrepr::truthy)
        && state.engine_info.as_ref().is_some_and(|i| i.logprobs)
        && text.tokenizer.token_bytes(0).is_some();
    let logprobs_count = match crate::fields::probability_options(body, exact_bytes) {
        Ok(count) => count,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    if logprobs_count.is_some()
        && (!active_tools.is_empty()
            || body.get("stop").is_some_and(crate::pyrepr::truthy)
            || body.get("thinking_budget").is_some_and(crate::pyrepr::truthy)
            || crate::grammar::FIELDS
                .iter()
                .any(|k| body.get(k).is_some_and(crate::pyrepr::truthy)))
    {
        return Err(crate::errors::Refused::request(
            "logprobs support nonstreamed chat without tools, stop strings, structured output or a thinking budget",
        ));
    }
    // The thinking fields (fields.zig:199's thinkingFields): the request's
    // switch and effort, the effort coerced against the template's levels.
    // The switch falls to the server's config; the effort to the server's
    // own (server.zig:130's effortFor).
    let levels: Vec<&str> = text.effort_levels.iter().map(String::as_str).collect();
    let thinking_opts = match crate::fields::thinking_fields(body, &levels) {
        Ok(t) => t,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    let thinking = thinking_opts.enable.unwrap_or(state.config.enable_thinking);
    let effort = thinking_opts
        .effort
        .as_deref()
        .or(state.config.reasoning_effort.as_deref())
        .map(|e| crate::fields::coerce_effort(e, &levels).to_string());
    // The request's fields object (openai.zig:139-155): the sampling-ish
    // passthrough with the thinking fields, and the required-call flag a
    // tool_choice that demands one sets (openai.zig:154)
    let mut fields = fields_object(body, &thinking_opts);
    if !active_tools.is_empty()
        && crate::tool_specs::choice_requires_call(body.get("tool_choice"))
    {
        fields["tool_call_required"] = json!(true);
    }
    let ids = render_prompt(text, messages, thinking, effort.as_deref(), true, &active_tools)?;
    let prompt_len = ids.len();
    if prompt_len == 0 {
        return Err(crate::errors::Refused::request("the rendered prompt is empty"));
    }
    // The prompt-cache fields (prompt.zig:104-125 + chat.zig:173-176): the
    // history a follow-up shares, where the latest user message starts (a
    // caching engine), the reusable system prefix with the shared-prefix
    // cuts
    let no_generation = render_prompt(text, messages, thinking, effort.as_deref(), false, &active_tools)?;
    let history = history_len(&ids, &no_generation);
    let rewind = if state.engine_info.as_ref().is_some_and(|i| i.prompt_cache) {
        rewind_len(text, messages, &active_tools, &ids, thinking, effort.as_deref())
    } else {
        0
    };
    let system_len = system_prefix_len(
        text,
        messages,
        &active_tools,
        &ids,
        thinking,
        effort.as_deref(),
    );
    let shared: Vec<u32> = [system_len as i64 - 2048, system_len as i64 - 512, system_len as i64]
        .into_iter()
        .filter(|cut| *cut >= 512)
        .map(|cut| cut as u32)
        .collect();
    // The prefill chunk starts (chunk_plan.zig:50's withCut): a cut just
    // before the conversation's own text, so a fresh session resumes its
    // whole harness (chat.zig:192)
    let chunk_starts = state.chunk_plan.starts(&ids);
    let cut = if state.chunk_plan.step > 0 && system_len > 0 {
        (system_len as u32).saturating_sub(1)
    } else {
        0
    };
    let chunks = crate::chunk_plan::with_cut(
        &chunk_starts,
        cut,
        ids.len(),
        state.chunk_plan.min_chunk,
    );
    // The earlier replies' spans, from the tokens alone (chat.zig:193):
    // cached states are keyed by them
    let spans: Vec<tensorfold_engine::Span> = reply_spans(text, &ids)
        .into_iter()
        .map(|[start, end]| tensorfold_engine::Span { start, end })
        .collect();
    finish_prepare(
        state,
        body,
        text,
        fields,
        ids,
        history,
        rewind,
        shared,
        chunks,
        spans,
        thinking,
        effort,
        active_tools,
        named_max,
        max_tokens,
        logprobs_count,
        received,
    )
}

/// A completion's prompt as the model reads it: token ids as given,
/// anything else as text. Upstream: `Prompt` (chat.zig:20) + `legacyPrompt`
/// (openai.zig:179).
pub enum CompletionPrompt {
    Text(String),
    Ids(Vec<u32>),
}

/// Upstream: `legacyPrompt` (openai.zig:179-192) — a non-empty all-int list
/// is ids (every value an int in the vocabulary's range), else text.
fn legacy_prompt(
    text: &crate::Text,
    prompt: Option<&Value>,
) -> Result<CompletionPrompt, crate::errors::Refused> {
    let Some(p) = prompt else {
        // A missing prompt reads as the empty text (openai.zig:180)
        return Ok(CompletionPrompt::Text(String::new()));
    };
    if let Some(list) = p.as_array() {
        if !list.is_empty() && all_ints(list) {
            let vocab = text.tokenizer.vocab_size() as i64;
            let mut out = Vec::with_capacity(list.len());
            for t in list {
                // Upstream reads only true ints: a bool reaches the ids
                // branch (allInts counts it) but refuses on the value
                let n = t.as_i64();
                if n.is_none_or(|n| n < 0 || n >= vocab) {
                    return Err(crate::errors::Refused::request(format!(
                        "prompt token ids must be integers in the valid range 0 to {}",
                        vocab - 1
                    )));
                }
                out.push(n.unwrap() as u32);
            }
            return Ok(CompletionPrompt::Ids(out));
        }
    }
    Ok(CompletionPrompt::Text(prompt_text(text, p)?))
}

/// Upstream: `allInts` (openai.zig:194) — bools count as ints here.
fn all_ints(items: &[Value]) -> bool {
    items
        .iter()
        .all(|t| t.is_i64() || t.is_u64() || t.is_boolean())
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    /// `allInts` (openai.zig:194): ints and bools count, strings and
    /// objects do not — the ids branch's gate before the range check.
    #[test]
    fn all_ints_counts_bools() {
        assert!(all_ints(&[json!(1), json!(2)]));
        assert!(all_ints(&[json!(true), json!(1)]));
        assert!(!all_ints(&[json!("a")]));
        assert!(!all_ints(&[json!({})]));
        assert!(all_ints(&[]));
    }
}

/// Upstream: `promptText` (openai.zig:199-213): a string is itself, null
/// empty, an all-int list decoded, a nested list joined on newlines,
/// anything else Python's text.
fn prompt_text(text: &crate::Text, p: &Value) -> Result<String, crate::errors::Refused> {
    match p {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(items) => {
            if all_ints(items) {
                let toks: Vec<u32> = items
                    .iter()
                    .map(|t| match t {
                        Value::Bool(b) => *b as u32,
                        _ => t.as_i64().unwrap_or(0).clamp(0, u32::MAX as i64) as u32,
                    })
                    .collect();
                return Ok(text.tokenizer.decode(&toks));
            }
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                parts.push(prompt_text(text, item)?);
            }
            Ok(parts.join("\n"))
        }
        _ => Ok(crate::tool_specs::py_str(p)),
    }
}

/// The completion route's plan: the body as a completion's input —
/// messages when a non-empty array came, else the legacy prompt — and the
/// same shared tail the chat plan runs. Upstream: openai.zig:119's `plan`
/// with `is_chat = false` over chat.zig:145's prepare.
pub(crate) fn prepare_completion(
    state: &AppState,
    raw: &Value,
) -> Result<Prepared, crate::errors::Refused> {
    let received = std::time::Instant::now();
    let text = state
        .text
        .as_ref()
        .ok_or_else(|| crate::errors::Refused::request(state.engineless_problem.clone()))?
        .clone();
    if state.engine.is_none() {
        return Err(crate::errors::Refused::request(state.engineless_problem.clone()));
    }
    crate::fields::require_object(raw)?;
    let body = match crate::fields::parse_numbers(raw) {
        Ok(b) => b,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    if let Some(path) = &state.config.request_log {
        crate::request_log::append(path, &body);
    }
    if let Err(m) = crate::grammar::refusal(&body) {
        return Err(crate::errors::Refused::request(m));
    }
    // The input (openai.zig:126-131): messages when a non-empty list, else
    // the legacy prompt
    let messages: Vec<Value> = match body.get("messages").and_then(Value::as_array) {
        Some(list) if !list.is_empty() => crate::messages::tool_arguments(
            crate::messages::normalize(
                Some(&Value::Array(list.clone())),
                "system",
                text.needs_user_after_tool,
            )
            .map_err(crate::errors::Refused::request)?,
        ),
        _ => Vec::new(),
    };
    let prompt = if messages.is_empty() {
        Some(legacy_prompt(&text, body.get("prompt"))?)
    } else {
        None
    };
    // logprobs: the rows describe the plain reply text — completions carry
    // none (openai.zig:124-158)
    let exact_bytes = body.get("logprobs").is_some_and(crate::pyrepr::truthy)
        && state.engine_info.as_ref().is_some_and(|i| i.logprobs)
        && text.tokenizer.token_bytes(0).is_some();
    let logprobs_count = match crate::fields::probability_options(&body, exact_bytes) {
        Ok(count) => count,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    if logprobs_count.is_some() {
        return Err(crate::errors::Refused::request(
            "logprobs are supported on /v1/chat/completions only",
        ));
    }
    let (named_max, max_tokens) = max_tokens_of(state, &body);
    // A prompt turns the thinking off (chat.zig:161); the messages path
    // reads the request's own switch
    let levels: Vec<&str> = text.effort_levels.iter().map(String::as_str).collect();
    let thinking_opts = match crate::fields::thinking_fields(&body, &levels) {
        Ok(t) => t,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    let thinking = if prompt.is_some() {
        false
    } else {
        thinking_opts.enable.unwrap_or(state.config.enable_thinking)
    };
    let effort = match &prompt {
        Some(_) => state
            .config
            .reasoning_effort
            .as_deref()
            .map(|e| crate::fields::coerce_effort(e, &levels).to_string()),
        None => thinking_opts
            .effort
            .as_deref()
            .or(state.config.reasoning_effort.as_deref())
            .map(|e| crate::fields::coerce_effort(e, &levels).to_string()),
    };
    let fields = fields_object(&body, &thinking_opts);
    // The render: a prompt's ids as given or the plain encode with the
    // specials (prompt.zig:79-86); messages through the template
    let (ids, history, rewind, system_len) = match &prompt {
        Some(CompletionPrompt::Text(t)) => (
            text.tokenizer.encode_special(t),
            0,
            0,
            0usize,
        ),
        Some(CompletionPrompt::Ids(v)) => (v.clone(), 0, 0, 0usize),
        None => {
            let ids = render_prompt(
                &text,
                &messages,
                thinking,
                effort.as_deref(),
                true,
                &[],
            )?;
            let prompt_len = ids.len();
            if prompt_len == 0 {
                return Err(crate::errors::Refused::request("the rendered prompt is empty"));
            }
            let no_generation = render_prompt(
                &text,
                &messages,
                thinking,
                effort.as_deref(),
                false,
                &[],
            )?;
            let history = history_len(&ids, &no_generation);
            let rewind = if state.engine_info.as_ref().is_some_and(|i| i.prompt_cache) {
                rewind_len(&text, &messages, &[], &ids, thinking, effort.as_deref())
            } else {
                0
            };
            let system_len = system_prefix_len(
                &text,
                &messages,
                &[],
                &ids,
                thinking,
                effort.as_deref(),
            );
            (ids, history, rewind, system_len)
        }
    };
    if ids.is_empty() {
        return Err(crate::errors::Refused::request("the rendered prompt is empty"));
    }
    // The shared cuts from the system prefix (empty for a prompt), the
    // chunk starts, the spans — the same as the chat plan's
    let shared: Vec<u32> = [system_len as i64 - 2048, system_len as i64 - 512, system_len as i64]
        .into_iter()
        .filter(|cut| *cut >= 512)
        .map(|cut| cut as u32)
        .collect();
    let chunk_starts = state.chunk_plan.starts(&ids);
    let cut = if state.chunk_plan.step > 0 && system_len > 0 {
        (system_len as u32).saturating_sub(1)
    } else {
        0
    };
    let chunks = crate::chunk_plan::with_cut(
        &chunk_starts,
        cut,
        ids.len(),
        state.chunk_plan.min_chunk,
    );
    let spans: Vec<tensorfold_engine::Span> = reply_spans(&text, &ids)
        .into_iter()
        .map(|[start, end]| tensorfold_engine::Span { start, end })
        .collect();
    finish_prepare(
        state,
        &body,
        &text,
        fields,
        ids,
        history,
        rewind,
        shared,
        chunks,
        spans,
        thinking,
        effort,
        // Completions carry no tools (openai.zig:127 reads them for is_chat)
        Vec::new(),
        named_max,
        max_tokens,
        logprobs_count,
        received,
    )
}

/// The prepared request's shared tail (chat.zig:161-242's second half): the
/// window's refusals and clamp, the thinking budget and the loop guard, the
/// call gate, the structured output, the sampling, the stops, and the
/// request build itself — shared by the chat and the completion plans.
#[allow(clippy::too_many_arguments)]
fn finish_prepare(
    state: &AppState,
    body: &Value,
    text: &Arc<crate::Text>,
    fields: Value,
    ids: Vec<u32>,
    history: usize,
    rewind: usize,
    shared: Vec<u32>,
    chunks: Vec<u32>,
    spans: Vec<tensorfold_engine::Span>,
    thinking: bool,
    effort: Option<String>,
    active_tools: Vec<Value>,
    named_max: bool,
    mut max_tokens: u32,
    logprobs_count: Option<u8>,
    received: std::time::Instant,
) -> Result<Prepared, crate::errors::Refused> {
    let prompt_len = ids.len();
    // The context window the engine enforces (chat.zig:164-170): the room
    // for the reply, the two refusals, the clamp
    let window = state
        .engine_info
        .as_ref()
        .map(|i| i.context_window as i64)
        .unwrap_or(0);
    if window > 0 {
        let n = prompt_len as i64;
        let room = window - n;
        if room < 1 {
            return Err(crate::errors::Refused::context_length(context_no_room(
                window,
                state.engine_info.as_ref().is_some_and(|i| i.context_fitted),
                n,
            )));
        }
        if named_max && max_tokens as i64 > room {
            return Err(crate::errors::Refused::context_length(context_too_big(
                window, n, max_tokens,
            )));
        }
        max_tokens = (max_tokens as i64).min(room) as u32;
    }
    // The thinking budget and the loop guard (chat.zig:198-204): the budget
    // counts thinking tokens before the close is forced; the loop guard
    // rides the same close tokens; a truthy budget that is not an int
    // reads as none, an unset one as the server's
    let loop_guard = state.config.loop_guard && !text.think_close.is_empty();
    let budget: i64 = match fields.get("thinking_budget") {
        Some(v) if crate::pyrepr::truthy(v) => v.as_i64().unwrap_or(0),
        _ => state.config.thinking_budget,
    };
    let mut think_budget = 0u32;
    let mut think_close: &'static [u32] = &[];
    let mut think_end_token = None;
    if (budget > 0 || loop_guard) && !text.think_close.is_empty() {
        if budget > 0 {
            think_budget = (budget as u64).min(u32::MAX as u64) as u32;
        }
        think_close = Box::leak(text.think_close.clone().into_boxed_slice());
        think_end_token = text.think_close_end;
    }
    // The call gate (server.zig:169's checkFeatures)
    let call = call_gate(state, text, &fields, &active_tools)?;
    // The structured output (server.zig:179-183): the spec on the fields
    // object; the engine must enforce structures
    let structure = match crate::grammar::spec(&fields) {
        Ok(None) => None,
        Ok(Some(spec)) => {
            if !state.engine_info.as_ref().is_some_and(|i| i.structures) {
                return Err(crate::errors::Refused::request(
                    "structured output (response_format and the guided_* fields) is not supported by this engine yet",
                ));
            }
            Some(tensorfold_engine::Structure {
                kind: spec.kind,
                text: Box::leak(spec.text.into_boxed_str()),
                after: if thinking { text.think_end } else { None },
            })
        }
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    // The sampling from the fields object (server.zig:135's
    // resolveSampling): the server's defaults under the request's, the
    // temperature gate, the seed by bits or keyed to the prompt
    let sampling = match resolve_sampling(state, &fields, &ids) {
        Ok(s) => s,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    // The stop strings end the reply at the engine (fields.zig:33's
    // stopOptions; chat.zig:233-234's hook); ignore_eos empties the end
    // tokens themselves (chat.zig:186).
    let (ignore_eos, stop_strings) = match crate::fields::stop_options(body) {
        Ok(v) => v,
        Err(m) => return Err(crate::errors::Refused::request(m)),
    };
    // draft: false turns the server's drafts off for this request
    // (chat.zig:181)
    let drafts = state.config.use_drafts
        && body.get("draft").and_then(Value::as_bool) != Some(false);
    let request: Box<Request<'static>> = Box::new(Request {
        prompt: Box::leak(ids.into_boxed_slice()),
        max_tokens,
        sampling,
        eos: Box::leak(
            if ignore_eos { Vec::new() } else { text.eos.clone() }.into_boxed_slice(),
        ),
        drafts,
        background: false,
        history_len: history as u32,
        shared_prefixes: Box::leak(shared.into_boxed_slice()),
        rewind_len: rewind as u32,
        chunks: Box::leak(chunks.into_boxed_slice()),
        decode_spans: Box::leak(spans.into_boxed_slice()),
        think_budget,
        think_close,
        think_end: think_end_token,
        tools_json: "",
        call,
        structure,
        loop_guard,
        // The rows are cut where the text is (chat.zig:207's admit)
        logprobs: match logprobs_count
            .map(|c| crate::logprobs::admit(text.think_end, thinking, think_budget, c))
        {
            Some(Ok(count)) => Some(count),
            Some(Err(m)) => return Err(crate::errors::Refused::request(m)),
            None => None,
        },
        stop: (!stop_strings.is_empty()).then(|| {
            Box::new(StopHook::new(stop_strings.clone(), Arc::clone(text)))
                as Box<dyn tensorfold_engine::StopCheck>
        }),
    });
    Ok(Prepared {
        request,
        prompt_len,
        thinking,
        effort,
        exact: sampling.is_some(),
        drafts,
        tools: active_tools,
        stops: stop_strings,
        ignore_eos,
        logprobs_count,
        received,
    })
}

/// The reply to a body: prepared, then submitted and finished. Upstream:
/// chat.zig's `generate` over its `prepare`.
fn generate(state: &AppState, raw: &Value) -> Reply {
    match prepare(state, raw) {
        Ok(prepared) => generate_prepared(state, prepared),
        Err(refused) => failed_reply(refused),
    }
}
pub(crate) fn generate_prepared(state: &AppState, prepared: Prepared) -> Reply {
    let started = prepared.received;
    let prompt_len = prepared.prompt_len;
    let thinking = prepared.thinking;
    let effort = prepared.effort.clone();
    let exact = prepared.exact;
    let drafts = prepared.drafts;
    let ignore_eos = prepared.ignore_eos;
    let stop_strings = prepared.stops.clone();
    let logprobs_count = prepared.logprobs_count;
    let fail = failed_reply;
    let Some(text) = state.text.as_ref() else {
        return fail(crate::errors::Refused::request(state.engineless_problem.clone()));
    };
    let Some(engine) = state.engine.as_ref() else {
        return fail(crate::errors::Refused::request(state.engineless_problem.clone()));
    };
    let (mailbox, shared) = Mailbox::new();
    if let Err(e) = engine.submit(state.next_request_id(), prepared.request, Box::new(mailbox)) {
        // The engine's own failures (chat.zig:238-242): the busy lane a
        // capacity, the closed scheduler an other, the span layout a request
        return fail(match e {
            SubmitError::Busy => crate::errors::Refused::capacity("the engine is busy; retry shortly"),
            SubmitError::Closed => crate::errors::Refused::other("the scheduler is closed"),
            SubmitError::InvalidSpans => {
                crate::errors::Refused::request("the prompt span layout is invalid")
            }
        });
    }
    // Wait for the finished event (the sink contract: it fired, then the engine
    // dropped the request and sink).
    let (inner, cond) = &*shared;
    let mut inner = inner.lock().unwrap();
    while !inner.finished {
        inner = cond.wait(inner).unwrap();
    }
    let tokens = inner.tokens.clone();
    let cached = inner.cached.unwrap_or(0) as usize;
    let reason = inner.reason;
    let (prefilled, first, last) = (inner.prefilled, inner.first, inner.last);
    let stats = inner.stats.clone();
    let rows = inner.rows.clone();
    drop(inner);
    let finished = std::time::Instant::now();
    // The request's metrics (server.zig's noteRequest shape): the counters
    // the /metrics scrape reads — the token counts, the latency, the round
    // telemetry the engine reports.
    state.metrics.note(
        prompt_len,
        cached,
        tokens.len(),
        stats.drafted,
        stats.accepted,
        stats.rounds,
        started.elapsed().as_secs_f64(),
        None,
        None,
        None,
        None,
    );
    // The runtime and speculative objects (chat.zig:513-547)
    let (runtime, speculative) = runtime_objects(
        thinking,
        effort.as_deref(),
        state
            .engine_info
            .as_ref()
            .map(|i| i.name.as_str())
            .unwrap_or_default(),
        exact,
        drafts,
        &tokens,
        &stats,
        started,
        prefilled,
        first,
        finished,
        state.engine_info.as_ref().map_or(0, |i| i.context_window),
        prompt_len,
    );
    let _ = last; // the metrics' queue-time read, upstream's gen.last_ns
    // The eos the model wrote ends the reply: the strip before the decode
    // (chat.zig:474), the stop strings hiding what they hit after it
    // (chat.zig:477).
    let content_tokens: &[u32] = if ignore_eos {
        &tokens
    } else {
        crate::reply_text::strip_trailing(&tokens, &text.eos)
    };
    let raw = text.tokenizer.decode(content_tokens);
    let stops_view: Vec<&str> = stop_strings.iter().map(String::as_str).collect();
    let stops_view = crate::reply_text::Stops { strings: &stops_view };
    let raw_visible = stops_view.visible(&raw, false).to_string();
    // The think block split (chat.zig:488's `if (g.thinking)`): reasoning apart
    // only when the request opened the block, else the reply is whole and the
    // Harmony channels parse stands in (chat.zig:495's parseHarmony).
    let (reasoning, content) = if thinking {
        let (r, answer) = crate::reply_text::split_thinking(&raw_visible, true, text.markers);
        let r = crate::reply_text::py_strip(&r);
        (r.to_string(), answer)
    } else {
        (String::new(), raw_visible)
    };
    // The tool calls attach at the routes (attach_calls, openai.zig:309), the
    // policy in hand: the reply here is the engine's raw answer.
    Reply {
        thinking,
        effort: effort.clone(),
        reasoning: reasoning.clone(),
        content,
        tool_calls: None,
        // chat.zig:547: the think end only counts when the request opened the
        // block; thinking off is no reasoning tokens.
        reasoning_tokens: if thinking {
            crate::reply_text::reasoning_count(&tokens, text.think_end)
        } else {
            0
        },
        usage: None,  // set by the caller (the OpenAI body shapes it)
        // The first stop the raw reply holds, ties in request order
        // (chat.zig:550's matched)
        stop_sequence: stops_view.matched(&raw).map(|s| s.to_string()),
        finish_reason: match reason {
            Reason::Stop | Reason::Cancelled => "stop",
            _ => "length",
        },
        prompt_tokens: prompt_len,
        cached_tokens: cached,
        completion_tokens: tokens.len(),
        // ``choices[0].logprobs`` (``{"content": [...]}``) for a request
        // that asked (chat.zig:552)
        logprobs: match logprobs_count {
            Some(_) => match crate::logprobs::value(text, thinking, content_tokens, &rows) {
                Ok(v) => Some(v),
                Err(m) => return fail(crate::errors::Refused::other(m)),
            },
            None => None,
        },
        runtime,
        speculative,
        failed: None,
    }
}

/// The `/v1/chat/completions` handler: the non-streaming reply in one body, the
/// streamed reply as SSE chunks (the whole text, then the finish, then the
/// usage's final chunk, as OpenAI's stream_options.usage shapes it).
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    if state.engine.is_none() || state.text.is_none() {
        // The none-backend's refusal, the same words every inference route uses.
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            crate::json_write::PythonJson(json!({
                "error": {"message": state.engineless_problem, "type": "engine_unavailable"}
            })),
        )
            .into_response();
    }
    // ``json.loads`` on the raw bytes, Python's own decoder and its error
    // messages (http_body.zig's readJson through json_parse.zig) — the
    // corpus pins them byte for byte
    let body: Value = match crate::json_parse::parse(&body) {
        Ok(v) => v,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                crate::json_write::PythonJson(json!({
                    "error": {"message": message}
                })),
            )
                .into_response();
        }
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    // parallel_tool_calls must be a boolean (openai.zig:164-166): false keeps
    // one call and filters the streamed prose.
    if let Some(p) = body.get("parallel_tool_calls") {
        if p.as_bool().is_none() {
            return (
                StatusCode::BAD_REQUEST,
                crate::json_write::PythonJson(json!({
                    "error": {"message": "parallel_tool_calls must be a boolean", "type": "invalid_request_error"}
                })),
            )
                .into_response();
        }
    }
    let policy = crate::tool_stream::Policy::new(
        body.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false),
    );
    // The tools the reply's calls parse against (upstream's input.tools:
    // tool_specs.zig:58's active — none for "none", only the named one for a
    // named function, so a choice that filters also gates the parse)
    let tools = match crate::tool_specs::active(body.get("tools"), body.get("tool_choice")) {
        Ok(tools) => tools,
        Err(m) => {
            return (
                StatusCode::BAD_REQUEST,
                crate::json_write::PythonJson(crate::errors::wrap_error(json!({"message": m}))),
            )
                .into_response()
        }
    };
    let mut reply = match tokio::task::block_in_place(|| generate_with(&state, &body)) {
        Ok(reply) => reply,
        Err(refused) => {
            let code =
                StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
            return (code, crate::json_write::PythonJson(crate::errors::wrap_error(crate::errors::error_body(
                &refused, "messages",
            ))))
                .into_response();
        }
    };
    // The reply's calls parsed out of its content (openai.zig:309)
    let calls = attach_calls(&mut reply, &tools, &policy);
    // The reply's id: the prefix then 32 random hex digits (openai.zig:220's
    // ids.make)
    let id = crate::ids::make("chatcmpl-", 32);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // The model the reply names: the request's own when it names a served
    // id, else the served name (openai.zig:174's plan.named)
    let model = reply_model(&state, &body);
    if stream {
        // The real stream (upstream chat.zig's streaming): the role chunk, one
        // content chunk a round through the incremental decode, the last chunk
        // with the finish reason and the usage, then [DONE].
        let start = match tokio::task::block_in_place(|| submit_streaming(&state, &body))
        {
            Ok(parts) => parts,
            // A context refusal reaches the client inside the stream the way
            // upstream's does (openai.zig:348 streamedContext): the stream
            // opens, the role chunk without tools, the error chunk, [DONE]
            Err(refused) if refused.kind == crate::errors::Kind::ContextLength => {
                let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<
                    Result<SseEvent, std::convert::Infallible>,
                >(8);
                std::thread::spawn(move || {
                    let send = |event: Result<SseEvent, std::convert::Infallible>| {
                        sse_tx.blocking_send(event)
                    };
                    if tools.is_empty() {
                        let _ = send(Ok(SseEvent::default()
                            .data(crate::json_write::dumps(&chunk_body(
                                &id, &model, created,
                                json!({"role": "assistant"}), None, true,
                            )))));
                    }
                    let _ = send(Ok(SseEvent::default()
                        .data(crate::json_write::dumps(&crate::errors::wrap_error(
                            crate::errors::error_body(&refused, "messages"),
                        )))));
                    let _ = send(Ok(SseEvent::default().data("[DONE]")));
                });
                return Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
                    .keep_alive(KeepAlive::default())
                    .into_response();
            }
            Err(refused) => {
                let code = StatusCode::from_u16(refused.status())
                    .unwrap_or(StatusCode::BAD_REQUEST);
                return (code, crate::json_write::PythonJson(crate::errors::wrap_error(crate::errors::error_body(
                    &refused, "messages",
                ))))
                    .into_response();
            }
        };
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<ChunkOut>(16);
        let text_arc = state.text.clone().expect("checked above");
        let id_for_stream = id.clone();
        let model_for_stream = model.clone();
        let tools = tools.clone();
        let policy = policy.clone();
        std::thread::spawn(move || {
            let separate_usage = body
                .get("stream_options")
                .and_then(Value::as_object)
                .is_some_and(|o| o.get("include_usage").is_some_and(crate::pyrepr::truthy));
            let _ = bridge(start, chunk_tx, id_for_stream, model_for_stream, created, text_arc, tools, policy, true, separate_usage, state);
        });
        // The chunks out as SSE data, [DONE] at the end (the chat surface's
        // own reading of the stream's end).
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<SseEvent, std::convert::Infallible>>(16);
        std::thread::spawn(move || {
            while let Some(out) = chunk_rx.blocking_recv() {
                let done = matches!(out, ChunkOut::Done);
                let event = match out {
                    ChunkOut::Chunk(body) => SseEvent::default().data(crate::json_write::dumps(&body)),
                    ChunkOut::Done => SseEvent::default().data("[DONE]"),
                };
                if sse_tx.blocking_send(Ok(event)).is_err() || done {
                    break;
                }
            }
        });
        Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let whole = whole_body(&id, &model, created, &reply, calls.as_deref());
        // The next turn's prompt prefilled in the background, from the
        // reply as its client got it (openai.zig:363's warmNext)
        let conversation = body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        crate::warm::reply(
            &state,
            &conversation,
            &tools,
            &crate::warm::message_of(&reply, calls.as_ref()),
            reply.thinking,
            reply.effort.as_deref(),
        );
        (StatusCode::OK, crate::json_write::PythonJson(whole)).into_response()
    }
}

/// The stream bridge: the engine's events to SSE chunks. The role chunk first
/// (upstream emits it before the sink's first call, chat.zig:477), one content
/// chunk a round through the incremental decode — the thinking split while
/// streaming holds partial markers, so reasoning deltas arrive before content
/// deltas — then the last chunk with the finish reason and the usage, and
/// [DONE]. Upstream: chat.zig's streaming loop over its StreamSink
/// (chat.zig:380-460).
type SendError = tokio::sync::mpsc::error::SendError<ChunkOut>;

/// The text a stream has sent: Python's `streamed = visible`, grown by its
/// delta while it only grows. Upstream: `Shown` (chat.zig:281) - the pointer
/// fast path of `after` rides on the append-only decode buffer; this port's
/// shown text is rebuilt each round, so the startsWith check decides alone.
#[derive(Default)]
struct Shown {
    text: String,
    chars: usize,
    extends: bool,
}

impl Shown {
    /// Python's `now[len(shown):]`; a rebuilt `now` re-sends from the
    /// character count. Upstream: `after` (chat.zig:289).
    fn after<'a>(&mut self, now: &'a str) -> &'a str {
        self.extends = now.starts_with(&self.text);
        if !self.extends {
            return crate::reply_text::after_chars(now, self.chars);
        }
        &now[self.text.len()..]
    }

    /// Upstream: `set` (chat.zig:297).
    fn set(&mut self, now: &str, delta: &str) {
        if self.extends {
            self.text.push_str(delta);
            self.chars += crate::reply_text::char_count(delta);
        } else {
            self.text.clear();
            self.text.push_str(now);
            self.chars = crate::reply_text::char_count(now);
        }
    }
}

/// What the stream bridge hands its consumer: one OpenAI-style chunk body
/// (the shape the chat route sends as SSE data and the Messages route
/// translates, upstream's `Out.event` payload), or the stream's end.
pub enum ChunkOut {
    Chunk(Value),
    Done,
}

/// The stream's own state: the filters, the streamer, what went out.
struct Bridge {
    sender: tokio::sync::mpsc::Sender<ChunkOut>,
    id: String,
    model: String,
    created: u64,
    thinking: bool,
    text: Arc<crate::Text>,
    tools: Vec<Value>,
    policy: crate::tool_stream::Policy,
    has_tools: bool,
    streamer: Option<crate::tool_stream::Streamer>,
    incremental: crate::reply_text::Incremental,
    shown: Shown,
    shown_reasoning: Shown,
    streamed_prose: bool,
    prose_sent: String,
    stops: Vec<String>,
    ignore_eos: bool,
    effort: Option<String>,
    exact: bool,
    drafts: bool,
    engine: String,
    received: std::time::Instant,
    prefilled: Option<std::time::Instant>,
    first: Option<std::time::Instant>,
    window: u32,
    compaction: Option<Value>,
    prompt_len: usize,
    /// The chat surface's chunks; a completion's carry `text`
    /// (openai.zig:257-273).
    is_chat: bool,
    /// ``stream_options.include_usage``: the usage a chunk of its own
    /// follows the stream with (openai.zig:523-533).
    separate_usage: bool,
}

impl Bridge {
    fn send(&self, body: Value) -> Result<(), SendError> {
        self.sender.blocking_send(ChunkOut::Chunk(body))
    }

    /// The stream's end: the consumer writes its own last line (the chat
    /// route's [DONE], the Messages route's message_stop).
    fn done(&self) -> Result<(), SendError> {
        self.sender.blocking_send(ChunkOut::Done)
    }

    fn chunk(&self, delta: Value, kind: Option<&str>) -> Result<(), SendError> {
        self.send(chunk_body(&self.id, &self.model, self.created, delta, kind, true))
    }

    /// The assistant role before the first prose of a tool-using stream.
    /// Upstream: `prose` (openai.zig:449).
    fn prose(&mut self) -> Result<(), SendError> {
        if self.streamed_prose {
            return Ok(());
        }
        self.streamed_prose = true;
        self.chunk(json!({"role": "assistant"}), None)
    }

    /// The one-call policy on a streamed delta: text filtered, tool_calls
    /// dropped; None when nothing is left. Upstream: `policyDelta`
    /// (openai.zig:436).
    fn policy_delta(&mut self, delta: Value) -> Option<Value> {
        if !self.policy.single {
            return crate::pyrepr::truthy(&delta).then_some(delta);
        }
        match delta {
            Value::String(s) => {
                let t = self.policy.text(&s);
                if t.is_empty() { None } else { Some(Value::String(t)) }
            }
            Value::Object(o) => {
                let content: Option<String> = o
                    .get("content")
                    .and_then(Value::as_str)
                    .map(|c| self.policy.text(c));
                let mut out = serde_json::Map::new();
                for (k, v) in o {
                    if k == "tool_calls" || (k == "content" && v.is_string()) {
                        continue;
                    }
                    out.insert(k, v);
                }
                if let Some(t) = content {
                    if !t.is_empty() {
                        out.insert("content".into(), Value::String(t));
                    }
                }
                if out.is_empty() { None } else { Some(Value::Object(out)) }
            }
            _ => None,
        }
    }

    /// The sink the streaming loop emits through: without tools the delta
    /// goes out as it is; with tools it is filtered, the prose preceded by
    /// the role chunk, its text remembered for the finish's `kept`.
    /// Upstream: `StreamSink.call` (openai.zig:418).
    fn sink(&mut self, delta: Value) -> Result<(), SendError> {
        if !self.has_tools {
            return self.chunk(delta, None);
        }
        let Some(d) = self.policy_delta(delta) else { return Ok(()) };
        self.prose()?;
        match &d {
            Value::String(s) => self.prose_sent.push_str(s),
            Value::Object(o) => {
                if let Some(Value::String(c)) = o.get("content") {
                    self.prose_sent.push_str(c);
                }
            }
            _ => {}
        }
        self.chunk(d, None)
    }

    /// One token batch: the incremental decode, the reasoning delta, the
    /// prose with the call markup hidden, the Streamer's deltas.
    /// Upstream: chat.zig:388-410.
    fn tokens(&mut self, tokens: &[u32]) -> Result<(), SendError> {
        let all = self
            .incremental
            .extend(tokens, |ids| self.text.tokenizer.decode(ids))
            .to_string();
        if self.incremental.pending() {
            return Ok(()); // a character still split across tokens (chat.zig:390)
        }
        // The stop strings hide what they hit while it streams (chat.zig:388)
        let view: Vec<&str> = self.stops.iter().map(String::as_str).collect();
        let view = crate::reply_text::Stops { strings: &view };
        let text = view.visible(&all, true).to_string();
        let (reasoning, answer) = if self.thinking {
            let (r, a) = crate::reply_text::split_thinking(&text, false, self.text.markers);
            (r, a)
        } else {
            (String::new(), text)
        };
        let piece = self.shown_reasoning.after(&reasoning);
        if !piece.is_empty() {
            let piece = piece.to_string();
            self.shown_reasoning.set(&reasoning, &piece);
            self.sink(json!({"reasoning_content": piece}))?;
        }
        // The call markup hides from the prose, the tail held back
        // (chat.zig:400-401)
        let shown = if self.has_tools {
            crate::reply_text::hide_tool_calls(&answer, false)
        } else {
            answer.clone()
        };
        let vis = crate::reply_text::held_back(&shown, self.has_tools);
        let delta = self.shown.after(vis);
        if !delta.is_empty() {
            let delta = delta.to_string();
            self.shown.set(vis, &delta);
            self.sink(Value::String(delta))?;
        }
        // The streamer reads the answer, never the reasoning (chat.zig:408)
        if let Some(c) = &mut self.streamer {
            let mut out = Vec::new();
            c.feed(&answer, &mut out);
            for d in out {
                self.sink(d)?;
            }
        }
        Ok(())
    }

    /// The finish: the closing call, the final prose and reasoning pieces,
    /// the end-of-stream calls, the last chunk with the finish reason and
    /// the usage. Upstream: chat.zig:470-510 and openai.zig:495-538.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        reason: Reason,
        prompt_len: usize,
        completion_tokens: usize,
        cached: u32,
        stats: &tensorfold_engine::Stats,
    ) -> Result<(), SendError> {
        let finished = std::time::Instant::now();
        // The eos the model wrote ends the reply: the strip before the
        // re-decode (chat.zig:474), the stop strings hiding what they hit
        // (chat.zig:477). What closes a call reads as markup the model wrote
        // — but only the model's own end token opens that reading
        // (chat.zig:343's eos check): a stop string, the length or
        // ignore_eos ends the reply with no call completion.
        let tokens = incremental_tokens(&self.incremental);
        let content_tokens: &[u32] = if self.ignore_eos {
            tokens
        } else {
            crate::reply_text::strip_trailing(tokens, &self.text.eos)
        };
        let raw = self.text.tokenizer.decode(content_tokens);
        let (visible, matched_stop) = {
            let view: Vec<&str> = self.stops.iter().map(String::as_str).collect();
            let view = crate::reply_text::Stops { strings: &view };
            (
                view.visible(&raw, false).to_string(),
                view.matched(&raw).map(str::to_string),
            )
        };
        let last_is_eos = tokens.last().is_some_and(|t| self.text.eos.contains(t));
        let close = if self.has_tools && last_is_eos {
            crate::tool_parse::close_call(&visible, &self.tools)
        } else {
            String::new()
        };
        let text = if close.is_empty() {
            visible
        } else {
            format!("{}{}", visible, close)
        };
        let (reasoning, content) = if self.thinking {
            let (r, a) = crate::reply_text::split_thinking(&text, true, self.text.markers);
            let r = crate::reply_text::py_strip(&r).to_string();
            (r, a)
        } else {
            (String::new(), text.clone())
        };
        // The reasoning the stream has not sent yet (chat.zig:497-500)
        if reasoning.starts_with(&self.shown_reasoning.text)
            && reasoning.len() > self.shown_reasoning.text.len()
        {
            let piece = reasoning[self.shown_reasoning.text.len()..].to_string();
            self.sink(json!({"reasoning_content": piece}))?;
        }
        // The prose the stream has not sent yet (chat.zig:502-504)
        let shown = if self.has_tools {
            crate::reply_text::finished_prose(&content)
        } else {
            content.clone()
        };
        if shown.starts_with(&self.shown.text) && shown.len() > self.shown.text.len() {
            let piece = shown[self.shown.text.len()..].to_string();
            self.sink(Value::String(piece))?;
        }
        // The closing call's deltas (chat.zig:505-507)
        if !close.is_empty() {
            if let Some(c) = &mut self.streamer {
                let fed: String = if self.thinking {
                    content.clone()
                } else {
                    text.clone()
                };
                let mut out = Vec::new();
                c.feed(&fed, &mut out);
                for d in out {
                    self.sink(d)?;
                }
            }
        }
        let mut finish_reason = match reason {
            Reason::Stop | Reason::Cancelled => "stop",
            _ => "length",
        };
        // The end-of-stream calls (openai.zig:495-520)
        if self.has_tools {
            let parsed =
                crate::tool_parse::parse(&content, &self.tools, self.policy.max_calls());
            let calls: Option<Vec<Value>> = parsed.calls.map(|c| {
                if self.policy.single && c.len() > 1 {
                    c.into_iter().take(1).collect()
                } else {
                    c
                }
            });
            let reply_content = self.policy.parsed_content(&parsed.content);
            // The tail the one-call filter held back (openai.zig:500)
            let tail = self.policy.flush();
            if !tail.is_empty() {
                self.prose()?;
                self.prose_sent.push_str(&tail);
                self.chunk(Value::String(tail), None)?;
            }
            let mut streamed_calls = self.streamer.as_ref().is_some_and(|s| s.streamed());
            if self.policy.single {
                streamed_calls = false;
            }
            let opening = calls.as_ref().filter(|_| !streamed_calls);
            if let Some(c) = opening {
                // The role before the calls, then each call's deltas
                self.chunk(json!({"role": "assistant"}), None)?;
                for d in crate::tool_parse::deltas(c) {
                    self.chunk(d, None)?;
                }
                finish_reason = "tool_calls";
            } else if !reply_content.is_empty() && !self.streamed_prose {
                self.chunk(Value::String(reply_content), None)?;
            } else if let Some(rest) = self.policy.kept(&reply_content, &self.prose_sent) {
                self.chunk(Value::String(rest.to_string()), None)?;
            }
        }
        let reasoning_tokens = if self.thinking {
            crate::reply_text::reasoning_count(
                incremental_tokens(&self.incremental),
                self.text.think_end,
            )
        } else {
            0
        };
        let mut last = chunk_body(
            &self.id,
            &self.model,
            self.created,
            json!({}),
            Some(finish_reason),
            self.is_chat,
        );
        let usage = json!({
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        });
        if !self.separate_usage {
            last["usage"] = usage.clone();
        }
        // The runtime and speculative objects (chat.zig:513-547), the
        // extras on the last chunk (openai.zig:521's extras call)
        let (runtime, speculative) = runtime_objects(
            self.thinking,
            self.effort.as_deref(),
            &self.engine,
            self.exact,
            self.drafts,
            incremental_tokens(&self.incremental),
            stats,
            self.received,
            self.prefilled,
            self.first,
            finished,
            self.window,
            self.prompt_len,
        );
        // The compaction object of the stamp (compact.zig:83's stamp, the
        // streamed shape)
        let mut runtime = runtime;
        if let Some(c) = &self.compaction {
            runtime["compaction"] = c.clone();
        }
        response_extras(
            &mut last,
            matched_stop.as_deref(),
            &runtime,
            &speculative,
        );
        self.send(last)?;
        if self.separate_usage {
            // The usage a chunk of its own: empty choices (openai.zig:525-
            // 532)
            let mut extra = chunk_body(
                &self.id,
                &self.model,
                self.created,
                Value::Null,
                None,
                self.is_chat,
            );
            extra["choices"] = json!([]);
            extra["usage"] = usage;
            self.send(extra)?;
        }
        self.done()?;
        Ok(())
    }
}

/// The stream bridge: the engine's events to SSE chunks. The role chunk first
/// when no tools are offered (a tool-using stream opens it before the first
/// prose, openai.zig:477), one chunk a round through the incremental decode,
/// the last chunk with the finish reason and the usage, then [DONE].
/// Upstream: chat.zig's streaming loop over its StreamSink (chat.zig:380-460).
/// The stream bridge: the engine's events to the chunk stream. The chat
/// route sends the chunks as SSE data; the Messages route translates them
/// (upstream: the Run's chunks through the Wire, anthropic.zig:101).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge(
    start: StreamStart,
    sender: tokio::sync::mpsc::Sender<ChunkOut>,
    id: String,
    model: String,
    created: u64,
    text: Arc<crate::Text>,
    tools: Vec<Value>,
    policy: crate::tool_stream::Policy,
    is_chat: bool,
    separate_usage: bool,
    state: Arc<crate::state::AppState>,
) -> Result<(), SendError> {
    let StreamStart {
        id: request_id,
        receiver,
        prompt_len,
        thinking,
        think_end: _,
        stops,
        ignore_eos,
        effort,
        exact,
        drafts,
        engine,
        received,
        window,
        compaction,
        ..
    } = start;
    let mut receiver = receiver;
    let result = bridge_inner(
        &mut receiver,
        sender,
        id,
        model,
        created,
        text,
        tools,
        policy,
        is_chat,
        separate_usage,
        prompt_len,
        thinking,
        stops,
        ignore_eos,
        effort,
        exact,
        drafts,
        engine,
        received,
        window,
        compaction,
    );
    if result.is_err() {
        // The consumer is gone (upstream: the sink's emit failing,
        // chat.zig:412's `Generation.cancel`): the engine's request
        // cancelled, its end waited for, and a cancellation the engine
        // reports counted as a client disconnect (chat.zig:426)
        if let Some(engine) = state.engine.as_ref() {
            engine.cancel(request_id);
        }
        while let Some(event) = receiver.blocking_recv() {
            if let StreamEvent::Finished { reason, .. } = event {
                if reason == Reason::Cancelled {
                    state.metrics.disconnected();
                }
                break;
            }
        }
    }
    result
}

/// The bridge's streaming loop, over the mailbox the shell still owns.
#[allow(clippy::too_many_arguments)]
fn bridge_inner(
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
    sender: tokio::sync::mpsc::Sender<ChunkOut>,
    id: String,
    model: String,
    created: u64,
    text: Arc<crate::Text>,
    tools: Vec<Value>,
    policy: crate::tool_stream::Policy,
    is_chat: bool,
    separate_usage: bool,
    prompt_len: usize,
    thinking: bool,
    stops: Vec<String>,
    ignore_eos: bool,
    effort: Option<String>,
    exact: bool,
    drafts: bool,
    engine: String,
    received: std::time::Instant,
    window: u32,
    compaction: Option<Value>,
) -> Result<(), SendError> {
    let has_tools = !tools.is_empty();
    let mut b = Bridge {
        sender,
        id,
        model,
        created,
        thinking,
        text,
        tools: tools.clone(),
        policy,
        has_tools,
        streamer: has_tools.then(|| crate::tool_stream::Streamer::new(&tools)),
        incremental: crate::reply_text::Incremental::default(),
        shown: Shown::default(),
        shown_reasoning: Shown::default(),
        streamed_prose: false,
        prose_sent: String::new(),
        stops,
        ignore_eos,
        effort,
        exact,
        drafts,
        engine,
        received,
        prefilled: None,
        first: None,
        window,
        compaction,
        prompt_len,
        is_chat,
        separate_usage,
    };
    if !has_tools {
        b.chunk(json!({"role": "assistant"}), None)?;
    }
    while let Some(event) = receiver.blocking_recv() {
        match event {
            StreamEvent::Prefilled(_) => {
                b.prefilled = Some(std::time::Instant::now());
            }
            StreamEvent::Tokens(tokens) => {
                let now = std::time::Instant::now();
                b.first.get_or_insert(now);
                b.tokens(&tokens)?
            }
            StreamEvent::Finished {
                reason,
                completion_tokens,
                cached,
                stats,
                ..
            } => {
                b.finish(reason, prompt_len, completion_tokens, cached, &stats)?;
                return Ok(());
            }
        }
    }
    // The channel closed without a finish (a cancel): the end alone.
    b.done()
}

/// The tokens the Incremental holds (the finished event's reasoning count).
fn incremental_tokens(inc: &crate::reply_text::Incremental) -> &[u32] {
    inc.tokens()
}

/// One streamed chunk. Upstream's reply chunks carry the same fields (the
/// route's `chunk_plan` shapes the delta; the whole-text delivery is the
/// Phase 1b shape).
fn chunk_body(
    id: &str,
    model: &str,
    created: u64,
    delta: Value,
    kind: Option<&str>,
    is_chat: bool,
) -> Value {
    // The chat chunk's delta (openai.zig:267): a bare string is prose and
    // wraps as {"content": text}, an empty one is {}, an object passes as
    // it is; a completion's chunk carries the prose in `text`
    // (openai.zig:271)
    let (object, choice) = if is_chat {
        let wrapped = match delta {
            Value::Object(_) => delta,
            Value::String(s) if !s.is_empty() => json!({"content": s}),
            _ => json!({}),
        };
        (
            "chat.completion.chunk",
            json!({
                "index": 0,
                "delta": wrapped,
                "finish_reason": kind.map(|k| Value::String(k.to_string())).unwrap_or(Value::Null),
            }),
        )
    } else {
        (
            "text_completion",
            json!({
                "index": 0,
                "text": match delta {
                    Value::String(s) => Value::String(s),
                    _ => Value::String(String::new()),
                },
                "finish_reason": kind.map(|k| Value::String(k.to_string())).unwrap_or(Value::Null),
                "logprobs": Value::Null,
            }),
        )
    };
    json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [choice],
    })
}

#[cfg(test)]
mod chunk_body_tests {
    use super::*;
    use serde_json::json;

    /// The chunk's delta shape (openai.zig:267): bare prose wraps as
    /// {"content": text} — the shape the Messages and Responses machines
    /// read — an empty string is {}, an object passes through.
    #[test]
    fn the_chunk_delta_wraps_prose() {
        let body = chunk_body("id", "m", 0, json!("hello"), None, true);
        assert_eq!(body["choices"][0]["delta"], json!({"content": "hello"}));
        let body = chunk_body("id", "m", 0, json!(""), None, true);
        assert_eq!(body["choices"][0]["delta"], json!({}));
        let body = chunk_body("id", "m", 0, json!({"role": "assistant"}), None, true);
        assert_eq!(body["choices"][0]["delta"], json!({"role": "assistant"}));
        let body = chunk_body("id", "m", 0, json!({}), Some("stop"), true);
        assert_eq!(body["choices"][0]["finish_reason"], json!("stop"));
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// resolveSampling (server.zig:135): the defaults under the request,
    /// the temperature gate, the seed, the greedy refusal.
    #[test]
    fn resolve_sampling_shapes() {
        let state = AppState::new(crate::Config::new("m".into()));
        let ids = [7u32, 8, 9];
        // No temperature: greedy
        let fields = json!({});
        assert!(resolve_sampling(&state, &fields, &ids).unwrap().is_none());
        // The temperature gate: 0 is greedy too
        let fields = json!({"temperature": 0});
        assert!(resolve_sampling(&state, &fields, &ids).unwrap().is_none());
        // A sampling request: its own fields, the seed keyed to the prompt
        let fields = json!({"temperature": "0.7", "top_p": 0.9, "top_k": "-5", "min_p": 0.1});
        let sampling = resolve_sampling(&state, &fields, &ids).unwrap().unwrap();
        assert_eq!(sampling.temperature, 0.7);
        assert_eq!(sampling.top_p, 0.9);
        assert_eq!(sampling.top_k, 0); // a negative top_k stores the zero it means
        assert_eq!(sampling.min_p, 0.1);
        assert_eq!(sampling.seed, tensorfold_engine::sample_seed_for(&ids, 0));
        // The request's own seed, by bits
        let fields = json!({"temperature": 1.0, "seed": -1});
        let sampling = resolve_sampling(&state, &fields, &ids).unwrap().unwrap();
        assert_eq!(sampling.seed, u64::MAX);
        // The server's defaults merge under the request
        let mut state = state;
        state.config.default_sampling = Some(json!({"temperature": 0.5, "top_p": 0.8}));
        let fields = json!({});
        let sampling = resolve_sampling(&state, &fields, &ids).unwrap().unwrap();
        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_p, 0.8);
        // The request wins where it names
        let fields = json!({"temperature": 0.3});
        let sampling = resolve_sampling(&state, &fields, &ids).unwrap().unwrap();
        assert_eq!(sampling.temperature, 0.3);
        assert_eq!(sampling.top_p, 0.8); // the default stays
        // A bad shape refuses through the fields object
        let fields = json!({"temperature": "hot"});
        assert_eq!(
            resolve_sampling(&state, &fields, &ids),
            Err("temperature must be a finite number or null".to_string())
        );
        // The greedy-only engine refuses, its message by the defaults'
        // temperature (server.zig:143-146)
        let mut greedy = AppState::new(crate::Config::new("m".into()));
        let mut info = tensorfold_engine::Info::default();
        info.greedy_only = true;
        greedy.engine_info = Some(info);
        let fields = json!({"temperature": 0.7});
        assert_eq!(
            resolve_sampling(&greedy, &fields, &ids),
            Err("this engine decodes greedily only: send temperature 0 or leave it out".to_string())
        );
        greedy.config.default_sampling = Some(json!({"temperature": 0.5}));
        assert_eq!(
            resolve_sampling(&greedy, &json!({}), &ids),
            Err("this engine decodes greedily only and this server samples by default: send temperature 0, or serve with --temperature 0".to_string())
        );
    }

    /// The call form a rendered probe shows (server.zig:194-201): the opener
    /// before the probe's name, the lead between them, the one byte after.
    #[test]
    fn the_call_form_a_probe_shows() {
        use crate::tool_parse::{CLOSER, OPENER};
        // The JSON shape: the name inside the opener's envelope
        let rendered = format!(
            "x<|im_start|>assistant\n{OPENER}{{\"name\": \"tfprobe_fn\", \"arguments\": {{}}}}{CLOSER}<|im_end|>"
        );
        let form = find_call_form(&rendered).expect("the probe's form");
        assert_eq!(form.opener, OPENER);
        assert_eq!(form.lead.as_deref(), Some("{\"name\": \""));
        // The one byte after the name: the quote that closes it
        assert_eq!(form.tail.as_deref(), Some("\""));
        // The XML shape: the name in a function tag
        let rendered = format!("a{OPENER}\n<function=tfprobe_fn>}}");
        let form = find_call_form(&rendered).expect("the probe's form");
        assert_eq!(form.lead.as_deref(), Some("\n<function="));
        assert_eq!(form.tail.as_deref(), Some(">"));
        // The Gemma envelope
        let rendered = "<|tool_call>{\"name\": \"tfprobe_fn\"}<tool_call|>";
        let form = find_call_form(&rendered).expect("the probe's form");
        assert_eq!(form.opener, "<|tool_call>");
        // The LAST opener before the name wins
        let rendered = format!("<|tool_call> far {OPENER}tfprobe_fn");
        let form = find_call_form(&rendered).expect("the probe's form");
        assert_eq!(form.opener, OPENER);
        // No name, or no call markup before it: no form
        assert!(find_call_form("no probe here").is_none());
        assert!(find_call_form("tfprobe_fn alone").is_none());
        // The name at the very end: the tail empty
        let rendered = format!("{OPENER}x tfprobe_fn");
        let form = find_call_form(&rendered).expect("the probe's form");
        assert_eq!(form.tail.as_deref(), Some(""));
    }

    /// The tool_choice required refusals, their exact words (server.zig:165,
    /// 168): the template-marks-calls refusal carries the openers upstream
    /// names.
    #[test]
    fn the_required_choice_refusals() {
        assert_eq!(
            crate::errors::Refused::request(format!(
                "tool_choice \"required\" or a named function needs a chat template that marks tool calls ({} or <|tool_call>), and this one does not: send \"auto\"",
                crate::tool_parse::OPENER
            )).message,
            format!(
                "tool_choice \"required\" or a named function needs a chat template that marks tool calls ({} or <|tool_call>), and this one does not: send \"auto\"",
                crate::tool_parse::OPENER
            )
        );
    }

    /// The context refusals' messages, exactly as chat.zig:166/168 build
    /// them: the wording clients match to compact.
    #[test]
    fn the_context_refusal_messages() {
        assert_eq!(
            context_no_room(32768, false, 33000),
            "This server's maximum context length is 32768 tokens, but the rendered prompt has 33000 tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation."
        );
        // The fitted window adds its note
        assert_eq!(
            context_no_room(32768, true, 33000),
            "This server's maximum context length is 32768 tokens, the most this server's memory budget fits, but the rendered prompt has 33000 tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation."
        );
        assert_eq!(
            context_too_big(32768, 30000, 4096),
            "This server's maximum context length is 32768 tokens, but the rendered prompt has 30000 tokens and requests 4096 reply tokens, which exceeds the context window. Reduce the prompt to at most 28672 prompt tokens or request at most 2768 reply tokens, including chat template and thinking tokens."
        );
        // The reply-budget math: window - limit and window - prompt
        assert_eq!(context_too_big(100, 80, 40).contains("at most 60 prompt tokens"), true);
        assert_eq!(context_too_big(100, 80, 40).contains("at most 20 reply tokens"), true);
    }

    /// history_len (prompt.zig:104): the no-generation render's length when
    /// it prefixes the prompt; a template with no generation suffix falls to
    /// the prompt less one.
    #[test]
    fn the_history_a_prompt_shares() {
        // A generation suffix: the history is the shared prefix
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let history: Vec<u32> = vec![1, 2, 3];
        assert_eq!(history_len(&prompt, &history), 3);
        // No shared prefix: nothing
        assert_eq!(history_len(&prompt, &[9, 9]), 0);
        // A template with no suffix: the history equals the prompt, falls
        // to the prompt less one
        assert_eq!(history_len(&prompt, &prompt), 4);
        // An empty or longer history never retains
        assert_eq!(history_len(&prompt, &[]), 0);
        assert_eq!(history_len(&prompt, &[1, 2, 3, 4, 5, 6, 7]), 0);
    }

    /// The token SHA: SHA-256 of the comma-joined ids, the first six bytes
    /// as lowercase hex (tokens.zig:101's shape, verified against the digest
    /// itself).
    #[test]
    fn the_token_sha() {
        assert_eq!(token_sha(&[1, 2, 3]), "8a6ae1512200");
        assert_eq!(token_sha(&[]), "e3b0c44298fc");
        assert_eq!(token_sha(&[42]), "73475cb40a56");
        assert_eq!(token_sha(&[151643, 151665, 11]), "265904f64992");
    }

    /// The runtime and speculative objects: the fields chat.zig:513-547
    /// puts, by name.
    #[test]
    fn the_runtime_and_speculative_objects() {
        let stats = tensorfold_engine::Stats {
            rounds: 4,
            drafted: 12,
            accepted: 9,
            min_rows: 2,
            prefill_widths: vec![2048, 512],
            prefill_raised: vec![true, false],
            loop_period: Some(7),
            prefill_seconds: Some(0.25),
            telemetry_json: String::new(),
        };
        let tokens = [11, 22, 33, 44, 45];
        let received = std::time::Instant::now();
        let (runtime, speculative) = runtime_objects(
            true,
            Some("high"),
            "metal",
            true,
            false,
            &tokens,
            &stats,
            received,
            Some(received), // no decode time: prefilled == finished
            Some(received),
            received,
            32768,
            100,
        );
        assert_eq!(runtime["enable_thinking"], true);
        assert_eq!(runtime["reasoning_effort"], "high");
        assert_eq!(runtime["engine"], "metal");
        assert_eq!(runtime["context_window"], 32768);
        assert_eq!(runtime["context_used"], 100);
        assert_eq!(runtime["tokens_per_second"], 0.0); // no decode seconds
        assert_eq!(runtime["seconds"], 0.0);
        assert_eq!(runtime["prefill_seconds"], 0.25);
        assert_eq!(runtime["prefill_widths"], json!([2048, 512]));
        assert_eq!(runtime["prefill_raised"], json!([true, false]));
        assert_eq!(runtime["sampling"], "exact");
        assert_eq!(runtime["drafts"], false);
        assert_eq!(runtime["token_sha"], token_sha(&tokens));
        assert_eq!(runtime["min_rows"], 2);
        assert_eq!(runtime["loop"], json!({"period": 7}));
        assert_eq!(speculative["rounds"], 4);
        assert_eq!(speculative["drafted"], 12);
        assert_eq!(speculative["accepted"], 9);
        assert_eq!(speculative["acceptance_rate"], 9.0 / 12.0);
        assert_eq!(speculative["tokens_per_round"], 5.0 / 4.0);
        // Thinking off labels the effort none; a greedy reply says so
        let (runtime, _) = runtime_objects(
            false,
            None,
            "metal",
            false,
            true,
            &tokens,
            &tensorfold_engine::Stats::default(),
            received,
            None,
            None,
            received,
            0,
            0,
        );
        assert_eq!(runtime["reasoning_effort"], "none");
        assert_eq!(runtime["sampling"], "greedy");
        assert_eq!(runtime["drafts"], true);
        assert!(runtime.get("loop").is_none());
        // The proposer rides when the telemetry parses
        let stats = tensorfold_engine::Stats {
            telemetry_json: r#"{"kind": "ngram"}"#.into(),
            ..tensorfold_engine::Stats::default()
        };
        let (_, speculative) = runtime_objects(
            true, None, "metal", true, true, &tokens, &stats,
            received, None, None, received, 0, 0,
        );
        assert_eq!(speculative["proposer"], json!({"kind": "ngram"}));
    }
}
