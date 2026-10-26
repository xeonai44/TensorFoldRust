//! After a chat reply, the next turn's shared prompt prefilled in the
//! background, so that turn resumes past the reply.
//!
//! Port of `zig/src/server/warm.zig` (70 lines). The pass is a
//! `max_tokens: 0` background request whose kept states the prompt cache
//! holds — the Info's `warm_turns` gates it (upstream only the Qwen 2.7
//! host sets it; the pinned family's reply is never prefilled ahead).

use crate::chat::Reply;
use crate::errors::Refused;
use crate::state::AppState;
use serde_json::{json, Value};

/// The next turn's message, the reply as its client got it
/// (openai.zig:367's warmNext): the content null when the reply calls a
/// tool, the calls carried then.
pub fn message_of(reply: &Reply, calls: Option<&Vec<Value>>) -> Value {
    let mut m = json!({"role": "assistant"});
    if let Some(c) = calls {
        m["content"] = Value::Null;
        m["tool_calls"] = Value::Array(c.clone());
    } else {
        m["content"] = Value::String(reply.content.clone());
    }
    m
}

/// Prefill the conversation, `message` (the reply as sent) and the next
/// turn's opening, on `warm_turns` engines. Upstream: `reply`
/// (warm.zig:32) — the gate first: engines that do not keep prompt-only
/// states' ends never prefill.
pub fn reply(
    state: &AppState,
    conversation: &[Value],
    tools: &[Value],
    message: &Value,
    thinking: bool,
    effort: Option<&str>,
) {
    if !state
        .engine_info
        .as_ref()
        .is_some_and(|i| i.warm_turns)
    {
        return;
    }
    if start(state, conversation, tools, message, thinking, effort).is_err() {
        // No background prefill: the next turn prefills from the start
        // (warm.zig:32's log line, quiet in the port)
    }
}

/// One warm pass. Upstream: `start` (warm.zig:36).
fn start(
    state: &AppState,
    conversation: &[Value],
    tools: &[Value],
    message: &Value,
    thinking: bool,
    effort: Option<&str>,
) -> Result<(), Refused> {
    let text = state
        .text
        .as_ref()
        .ok_or_else(|| Refused::request("no text"))?;
    let engine = state
        .engine
        .as_ref()
        .ok_or_else(|| Refused::request(state.engineless_problem.clone()))?;
    // The tokens every next turn shares: two renderings agree up to the
    // next turn's own text (warm.zig:58)
    let a = next(state, text, conversation, tools, message, "A", thinking, effort);
    let b = next(state, text, conversation, tools, message, "B", thinking, effort);
    let (prompt, messages) = match (a, b) {
        (Some((ids, list)), Some(other)) => {
            let n = ids
                .iter()
                .zip(&other.0)
                .position(|(x, y)| x != y)
                .unwrap_or(ids.len().min(other.0.len()));
            (ids[..n].to_vec(), list)
        }
        _ => return Ok(()),
    };
    if prompt.is_empty() {
        return Ok(());
    }
    if state.engine_info.as_ref().is_some_and(|i| {
        i.context_window > 0 && prompt.len() >= i.context_window as usize
    }) {
        return Ok(());
    }
    // The warm request (warm.zig:41): prompt-only, background, its whole
    // prompt the history, the rewind at the latest user message, the chunk
    // starts, the reply spans. The slices leak with the request's own
    // lifetime, the established pattern.
    let rewind = rewind_of(state, text, &messages, tools, &prompt, thinking, effort);
    let spans: Vec<tensorfold_engine::Span> = crate::chat::reply_spans(text, &prompt)
        .into_iter()
        .map(|[start, end]| tensorfold_engine::Span { start, end })
        .collect();
    let request: Box<tensorfold_engine::Request<'static>> =
        Box::new(tensorfold_engine::Request {
            prompt: Box::leak(prompt.to_vec().into_boxed_slice()),
            max_tokens: 0,
            sampling: None,
            eos: Box::leak(Vec::new().into_boxed_slice()),
            stop: None,
            drafts: false,
            background: true,
            history_len: prompt.len() as u32,
            shared_prefixes: &[],
            rewind_len: rewind as u32,
            chunks: Box::leak(state.chunk_plan.starts(&prompt).into_boxed_slice()),
            decode_spans: Box::leak(spans.into_boxed_slice()),
            think_budget: 0,
            think_close: Box::leak(Vec::new().into_boxed_slice()),
            think_end: None,
            tools_json: "",
            call: None,
            structure: None,
            loop_guard: false,
            logprobs: None,
        });
    // The pass's sink: nothing delivered; the request drops at its finished
    // event (warm.zig:18's Pass)
    let _ = engine.submit(0, request, Box::new(WarmSink));
    Ok(())
}

/// One next-turn rendering: the conversation plus the reply plus a user
/// turn of `text`. Upstream: the loop body of `shared` (warm.zig:63).
#[allow(clippy::too_many_arguments)]
fn next(
    _state: &AppState,
    text: &crate::Text,
    conversation: &[Value],
    tools: &[Value],
    message: &Value,
    user_text: &str,
    thinking: bool,
    effort: Option<&str>,
) -> Option<(Vec<u32>, Vec<Value>)> {
    let mut list = conversation.to_vec();
    list.push(message.clone());
    list.push(json!({"role": "user", "content": user_text}));
    let normalized = crate::messages::normalize(
        Some(&Value::Array(list.clone())),
        "system",
        text.needs_user_after_tool,
    )
    .ok()?;
    let normalized = crate::messages::tool_arguments(normalized);
    let ids = crate::chat::render_prompt(text, &normalized, thinking, effort, true, tools).ok()?;
    Some((ids, normalized))
}

/// The rewind position for the warm pass's rendering (warm.zig:41's
/// rewindLen over the warm messages).
#[allow(clippy::too_many_arguments)]
fn rewind_of(
    _state: &AppState,
    text: &crate::Text,
    messages: &[Value],
    tools: &[Value],
    prompt: &[u32],
    thinking: bool,
    effort: Option<&str>,
) -> usize {
    crate::chat::rewind_len(text, messages, tools, prompt, thinking, effort)
}

/// The warm pass's sink: nothing delivered; the request's leaked slices
/// live with the process (a prompt-only background pass has no reply).
struct WarmSink;

impl tensorfold_engine::EventSink for WarmSink {
    fn event(&mut self, _id: tensorfold_engine::Id, _event: tensorfold_engine::Event<'_>) {}
}
