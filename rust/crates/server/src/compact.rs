//! Compaction: a conversation over the server's threshold rebuilt as a
//! memory note and its kept tail, the note an ordinary greedy generation so
//! the same request compacts and answers the same way.
//!
//! Port of `zig/src/server/compact.zig` (539 lines). The message math is
//! pure and tested; the note's generation runs through the chat prepare over
//! the engine.

use crate::chat::{self, Prepared};
use crate::errors::{Kind, Refused};
use crate::state::AppState;
use crate::CompactAt;
use serde_json::{json, Value};

/// The line the memory note message starts with. Upstream: `marker_line`
/// (compact.zig:9).
pub const MARKER_LINE: &str = "Memory of the earlier conversation:";

/// Upstream: `note_task` (compact.zig:11).
const NOTE_TASK: &str = "Rewrite the memory note for this conversation so it keeps everything still needed.
Markdown sections: Goal, Constraints and preferences, Progress, Decisions, Next steps, Critical context.
Progress lists done, in progress and blocked. Decisions keep every earlier one and its reason.
Next steps stay in order. Critical context keeps exact names, paths, numbers and error text.
Under 800 words.";

/// Upstream: `split_task` (compact.zig:17).
const SPLIT_TASK: &str = "Summarize the cut part of this turn.
Keep the original request, the early progress, and what the kept reply needs in order to make sense.";

const FILES_HEAD: &str = "## Files read and changed\n";
const SPLIT_HEAD: &str = "## Kept reply context\n";

/// The compaction's stamp on a reply. Upstream: `Stamp` (compact.zig:31).
#[derive(Clone, Debug, Default)]
pub struct Stamp {
    pub window: i64,
    pub used: usize,
    pub compacted: bool,
    pub messages_compacted: usize,
    pub kept_messages: usize,
    pub note_tokens: usize,
    pub note: String,
}

impl Stamp {
    fn plain(window: i64, used: usize) -> Stamp {
        Stamp {
            window,
            used,
            ..Stamp::default()
        }
    }

    /// The compaction object the stamp puts (compact.zig:111-117); Null when
    /// the reply did not compact.
    pub fn object(&self) -> Value {
        if !self.compacted {
            return Value::Null;
        }
        json!({
            "messages_compacted": self.messages_compacted,
            "kept_messages": self.kept_messages,
            "note_tokens": self.note_tokens,
            "note": self.note,
        })
    }
}

/// One generated piece. Upstream: `Piece` (compact.zig:33).
struct Piece {
    text: String,
    tokens: usize,
}

/// The tokens reserved against the window before compaction. Upstream:
/// `reserveTokens` (compact.zig:44).
pub fn reserve_tokens(window: i64, at: CompactAt) -> i64 {
    match at {
        CompactAt::Auto => (window * 15 / 100).max(16384),
        CompactAt::Fraction(f) => 1.max(window - threshold(window, f)),
    }
}

/// The tokens the note's own generation may use. Upstream:
/// `noteBudgetTokens` (compact.zig:48).
pub fn note_budget_tokens(window: i64, at: CompactAt) -> i64 {
    8192.min((reserve_tokens(window, at) * 4 / 5).max(1))
}

/// The recent tail kept whole. Upstream: `keepTokens` (compact.zig:51).
pub fn keep_tokens(window: u32, keep_override: Option<u32>) -> usize {
    match keep_override {
        Some(k) => k as usize,
        None => 20000.min(window as usize / 4),
    }
}

/// Upstream: `threshold` (compact.zig:55).
fn threshold(window: i64, f: f64) -> i64 {
    let t = (f * window as f64).floor();
    if t <= 0.0 {
        return 0;
    }
    if t >= window as f64 {
        return window;
    }
    t as i64
}

/// Whether the prompt and its reply overflow the compaction point.
/// Upstream: `over` (compact.zig:61).
pub fn over(prompt: usize, max_out: i64, window: i64, at: CompactAt) -> bool {
    let sum = prompt as i64 + max_out.max(0);
    match at {
        CompactAt::Auto => sum > window - reserve_tokens(window, at),
        CompactAt::Fraction(f) => sum > threshold(window, f),
    }
}

fn role_of(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("")
}

fn content_of(m: &Value) -> &str {
    m.get("content").and_then(Value::as_str).unwrap_or("")
}

/// Upstream: `isLead` (compact.zig:347).
fn is_lead(m: &Value) -> bool {
    matches!(role_of(m), "system" | "developer")
}

/// Upstream: `isCut` (compact.zig:352).
fn is_cut(m: &Value) -> bool {
    matches!(role_of(m), "user" | "assistant")
}

/// Upstream: `prefixLen` (compact.zig:356).
fn prefix_len(msgs: &[Value]) -> usize {
    msgs.iter().take_while(|m| is_lead(m)).count()
}

/// The last marker message. Upstream: `findMarker` (compact.zig:361).
fn find_marker(msgs: &[Value]) -> Option<usize> {
    let mut found = None;
    for (i, m) in msgs.iter().enumerate() {
        if marker_body(m).is_some() {
            found = Some(i);
        }
    }
    found
}

/// The note a marker message carries. Upstream: `markerBody` (compact.zig:368).
fn marker_body(m: &Value) -> Option<&str> {
    if !matches!(role_of(m), "system" | "user" | "developer") {
        return None;
    }
    let c = content_of(m);
    let rest = c.strip_prefix(MARKER_LINE)?;
    Some(rest.strip_prefix('\n').unwrap_or(rest))
}

/// Where a kept assistant's turn began, mid-turn. Upstream: `splitIndex`
/// (compact.zig:379).
fn split_index(msgs: &[Value], prefix: usize, tail: usize, marker: Option<usize>) -> Option<usize> {
    if tail >= msgs.len() || role_of(&msgs[tail]) != "assistant" {
        return None;
    }
    let mut i = tail;
    while i > prefix {
        i -= 1;
        if marker == Some(i) {
            continue;
        }
        return match role_of(&msgs[i]) {
            "user" => Some(i),
            "assistant" => None,
            _ => continue,
        };
    }
    None
}

/// The next earlier cut point. Upstream: `nextCut` (compact.zig:391).
fn next_cut(msgs: &[Value], tail: usize, marker: Option<usize>) -> Option<usize> {
    let last = last_user(msgs, marker)?;
    let mut i = tail + 1;
    while i < msgs.len() {
        if marker != Some(i) && is_cut(&msgs[i]) {
            return (i <= last).then_some(i);
        }
        i += 1;
    }
    None
}

/// Upstream: `lastUser` (compact.zig:401).
fn last_user(msgs: &[Value], marker: Option<usize>) -> Option<usize> {
    (0..msgs.len())
        .rev()
        .find(|&i| marker != Some(i) && role_of(&msgs[i]) == "user")
}

/// Upstream: `countBetween` (compact.zig:410).
fn count_between(msgs: &[Value], from: usize, to: usize, marker: Option<usize>) -> usize {
    (from..to.min(msgs.len()))
        .filter(|i| marker != Some(*i))
        .count()
}

/// The messages of a range, the marker and the covered skipped. Upstream:
/// `collect` (compact.zig:224).
fn collect(msgs: &[Value], from: usize, to: usize, marker: Option<usize>, skip: usize) -> Vec<Value> {
    (from..to.min(msgs.len()))
        .filter(|i| *i >= skip && marker != Some(*i))
        .map(|i| msgs[i].clone())
        .collect()
}

/// The message's transcript line: role, content, its calls. Upstream:
/// `lineOf` (compact.zig:448).
fn line_of(m: &Value) -> String {
    let mut buf = format!("{}: {}", role_of(m), content_of(m));
    append_calls(&mut buf, m);
    buf
}

/// Upstream: `transcript` (compact.zig:457).
fn transcript(msgs: &[Value]) -> String {
    msgs.iter()
        .map(line_of)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Upstream: `appendCalls` (compact.zig:464).
fn append_calls(buf: &mut String, m: &Value) {
    let Some(calls) = m.get("tool_calls").and_then(Value::as_array) else {
        return;
    };
    for call in calls {
        let Some(function) = call.get("function") else { continue };
        buf.push_str(" call ");
        buf.push_str(function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default());
        if let Some(args) = function.get("arguments") {
            buf.push(' ');
            match args {
                Value::String(s) => buf.push_str(s),
                other => {
                    let text = crate::tool_parse::zig_stringify(other, true);
                    buf.push_str(&text);
                }
            }
        }
    }
}

/// The file paths a message's calls touched. Upstream: `pathsFromMsgs` /
/// `addArgs` (compact.zig:478/494).
fn paths_from_msgs(list: &mut Vec<String>, msgs: &[Value]) {
    for m in msgs {
        let Some(calls) = m.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for call in calls {
            if let Some(args) = call.get("function").and_then(|f| f.get("arguments")) {
                add_args(list, args);
            }
        }
    }
}

/// Upstream: `addArgs` (compact.zig:494) — a string parsed, the path keys
/// and the paths list.
fn add_args(list: &mut Vec<String>, args: &Value) {
    let mut obj = args.clone();
    if let Value::String(s) = args {
        match serde_json::from_str::<Value>(s) {
            Ok(v) if v.is_object() => obj = v,
            _ => return,
        }
    }
    if !obj.is_object() {
        return;
    }
    for k in ["path", "file_path", "filename", "file"] {
        if let Some(Value::String(v)) = obj.get(k) {
            if !v.is_empty() {
                add_unique(list, v);
            }
        }
    }
    if let Some(v) = obj.get("paths") {
        match v {
            Value::String(s) if !s.is_empty() => add_unique(list, s),
            Value::Array(items) => {
                for item in items {
                    if let Value::String(s) = item {
                        if !s.is_empty() {
                            add_unique(list, s);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The paths the note's own Files section listed. Upstream: `pathsFromNote`
/// (compact.zig:510).
fn paths_from_note(list: &mut Vec<String>, note: &str) {
    let Some(i) = note.find(FILES_HEAD) else { return };
    let mut rest = &note[i + FILES_HEAD.len()..];
    while !rest.is_empty() {
        let end = rest.find('\n').unwrap_or(rest.len());
        let line = &rest[..end];
        if let Some(p) = line.strip_prefix("- ") {
            let p = p.trim_matches(|c: char| c == ' ' || c == '\t');
            if !p.is_empty() {
                add_unique(list, p);
            }
        } else if !line.is_empty() {
            break;
        }
        if end == rest.len() {
            break;
        }
        rest = &rest[end + 1..];
    }
}

/// Upstream: `addUnique` (compact.zig:529).
fn add_unique(list: &mut Vec<String>, p: &str) {
    if !list.iter().any(|seen| seen == p) {
        list.push(p.to_string());
    }
}

/// The note without the sections the rebuild writes itself. Upstream:
/// `withoutExtra` (compact.zig:239).
fn without_extra(note: &str) -> &str {
    let mut end = note.len();
    for h in [FILES_HEAD, SPLIT_HEAD] {
        if let Some(i) = note.find(h) {
            let mut at = i;
            if at > 0 && note.as_bytes()[at - 1] == b'\n' {
                at -= 1;
            }
            if at > 0 && note.as_bytes()[at - 1] == b'\n' {
                at -= 1;
            }
            if at < end {
                end = at;
            }
        }
    }
    &note[..end]
}

/// The note with the split and the files sections written. Upstream:
/// `finishNote` (compact.zig:213).
fn finish_note(body: &str, split: &str, paths: &[String]) -> String {
    let mut buf = without_extra(body).to_string();
    if !split.is_empty() {
        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(SPLIT_HEAD);
        buf.push_str(split);
    }
    if !paths.is_empty() {
        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(FILES_HEAD);
        for p in paths {
            buf.push_str("- ");
            buf.push_str(p);
            buf.push('\n');
        }
    }
    buf
}

/// A plain role/content message. Upstream: `msg` (compact.zig:231).
fn msg(role: &str, content: &str) -> Value {
    json!({"role": role, "content": content})
}

/// The rebuilt conversation: the prefix, the note, the kept tail. Upstream:
/// `assemble` (compact.zig:183).
fn assemble(
    late_system: &str,
    msgs: &[Value],
    note: &str,
    split: &str,
    include_split: bool,
    tail: usize,
    marker: Option<usize>,
) -> Vec<Value> {
    let mut out = Vec::new();
    let prefix = prefix_len(msgs);
    for (i, m) in msgs[..prefix].iter().enumerate() {
        if marker == Some(i) {
            continue;
        }
        out.push(m.clone());
    }
    if !note.is_empty() {
        out.push(msg(late_system, &format!("{MARKER_LINE}\n{note}")));
    }
    if include_split && !split.is_empty() {
        out.push(msg("user", &format!("Earlier in this turn:\n{split}")));
    }
    for (i, m) in msgs[tail.min(msgs.len())..].iter().enumerate() {
        if marker == Some(tail + i) {
            continue;
        }
        out.push(m.clone());
    }
    if out.is_empty() {
        out.push(msg("user", ""));
    }
    out
}

// ---------------------------------------------------------------------------
// The counting and the generation (compact.zig:196-539)

/// The tokens a message costs in a transcript. Upstream: `countOne`
/// (compact.zig:441) — its line encoded.
fn count_one(text: &crate::state::Text, m: &Value) -> usize {
    text.tokenizer.encode(&line_of(m)).len()
}

/// Upstream: `countRange` (compact.zig:433).
fn count_range(
    text: &crate::state::Text,
    msgs: &[Value],
    from: usize,
    to: usize,
    marker: Option<usize>,
) -> usize {
    (from..to.min(msgs.len()))
        .filter(|i| marker != Some(*i))
        .map(|i| count_one(text, &msgs[i]))
        .sum()
}

/// Upstream: `countMsgs` (compact.zig:428).
fn count_msgs(text: &crate::state::Text, msgs: &[Value]) -> usize {
    count_range(text, msgs, 0, msgs.len(), None)
}

/// Where the kept tail starts: the earliest cut whose messages still fit
/// the keep budget. Upstream: `keptStart` (compact.zig:196).
fn kept_start(
    text: &crate::state::Text,
    msgs: &[Value],
    keep: usize,
    marker: Option<usize>,
) -> usize {
    let prefix = prefix_len(msgs);
    let starts: Vec<usize> = msgs
        .iter()
        .enumerate()
        .filter(|(i, m)| *i >= prefix && marker != Some(*i) && is_cut(m))
        .map(|(i, _)| i)
        .collect();
    if starts.is_empty() {
        return prefix;
    }
    let mut acc = 0;
    for g in (0..starts.len()).rev() {
        let from = starts[g];
        let to = starts.get(g + 1).copied().unwrap_or(msgs.len());
        acc += count_range(text, msgs, from, to, marker);
        if acc >= keep {
            return from;
        }
    }
    starts[0]
}

/// The room one note chunk may read. Upstream: `chunkRoom` (compact.zig:76).
fn chunk_room(window: i64, at: CompactAt) -> usize {
    let budget = note_budget_tokens(window, at);
    if window <= budget {
        1
    } else {
        (window - budget) as usize
    }
}

/// One note generation: the task, the previous note, the transcript.
/// Upstream: `oneNote` (compact.zig:149) — a greedy 1024-token reply with
/// thinking off; a context-length refusal retried measured, then given up.
fn one_note(
    state: &AppState,
    task: &str,
    prev: &str,
    msgs: &[Value],
) -> Result<Piece, Refused> {
    if msgs.is_empty() {
        return Ok(Piece {
            text: prev.to_string(),
            tokens: 0,
        });
    }
    let window: i64 = state
        .engine_info
        .as_ref()
        .map(|i| i.context_window as i64)
        .unwrap_or(0);
    let mut list = vec![msg("system", task)];
    if !prev.is_empty() {
        list.push(msg("user", &format!("Previous note:\n{prev}")));
    }
    list.push(msg(
        "user",
        &format!("Messages to add:\n{}", transcript(msgs)),
    ));
    let request_body = |max_tokens: u32| {
        json!({
            "messages": list,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false},
        })
    };
    let mut prepared = match chat::prepare(state, &request_body(1024)) {
        Ok(p) => p,
        Err(r) if r.kind == Kind::ContextLength => {
            // Measured: max_tokens 1 renders the transcript, the room is
            // what the window has past it (compact.zig:166-172)
            let mut fitted = match chat::prepare(state, &request_body(1)) {
                Ok(f) => f,
                Err(r2) if r2.kind == Kind::ContextLength => {
                    return Ok(Piece {
                        text: prev.to_string(),
                        tokens: 0,
                    })
                }
                Err(r2) => return Err(r2),
            };
            let room = window - fitted.prompt_len as i64;
            if room < 1 {
                return Ok(Piece {
                    text: prev.to_string(),
                    tokens: 0,
                });
            }
            fitted.request.max_tokens = 1024.min(room) as u32;
            // The note stays a drafted generation when replies pass
            // --no-drafts (compact.zig:175)
            fitted.request.drafts = true;
            fitted.drafts = true;
            fitted
        }
        Err(r) => return Err(r),
    };
    prepared.request.drafts = true;
    prepared.drafts = true;
    let reply = chat::generate_prepared(state, prepared);
    if let Some(r) = reply.failed {
        return Err(r);
    }
    Ok(Piece {
        text: reply.content,
        tokens: reply.completion_tokens,
    })
}

/// The note over chunks of messages that fit the room. Upstream: `update`
/// (compact.zig:130).
fn update(
    state: &AppState,
    text: &crate::state::Text,
    at: CompactAt,
    task: &str,
    prev: &str,
    msgs: &[Value],
) -> Result<Piece, Refused> {
    if msgs.is_empty() {
        return Ok(Piece {
            text: prev.to_string(),
            tokens: 0,
        });
    }
    let window: i64 = state
        .engine_info
        .as_ref()
        .map(|i| i.context_window as i64)
        .unwrap_or(0);
    let room = chunk_room(window, at);
    let mut note = prev.to_string();
    let mut tokens = 0;
    let mut i = 0;
    while i < msgs.len() {
        let mut j = i + 1;
        let mut n = count_msgs(text, &msgs[i..j]);
        while j < msgs.len() {
            let more = count_msgs(text, &msgs[j..j + 1]);
            if n + more > room {
                break;
            }
            n += more;
            j += 1;
        }
        let piece = one_note(state, task, &note, &msgs[i..j])?;
        note = piece.text;
        tokens += piece.tokens;
        i = j;
    }
    Ok(Piece { text: note, tokens })
}

/// The conversation rebuilt: the note over the compacted middle, the kept
/// tail, the retry that drops the split and cuts further back. Upstream:
/// `rebuild` (compact.zig:105).
fn rebuild(
    state: &AppState,
    raw: &Value,
    at: CompactAt,
    msgs: &[Value],
) -> Result<(Prepared, Stamp), Refused> {
    let Some(text) = state.text.as_ref() else {
        return Err(Refused::request(state.engineless_problem.clone()));
    };
    let window: i64 = state
        .engine_info
        .as_ref()
        .map(|i| i.context_window as i64)
        .unwrap_or(0);
    let marker = find_marker(msgs);
    let slot = state
        .config
        .compact_memory
        .as_deref()
        .and_then(|dir| crate::compact_mem::load_mem(dir, msgs));
    let hash_ok = slot.as_ref().is_some_and(|m| crate::compact_mem::hashes(msgs, m));
    let prev: &str = if hash_ok {
        slot.as_ref().map(|m| m.note.as_str()).unwrap_or_default()
    } else {
        marker
            .and_then(|i| marker_body(&msgs[i]))
            .unwrap_or_default()
    };
    let skip = if hash_ok {
        slot.as_ref().map(|m| m.covered).unwrap_or_default()
    } else {
        marker.map(|i| i + 1).unwrap_or_default()
    };
    let prefix = prefix_len(msgs);
    let mut tail = kept_start(
        text,
        msgs,
        keep_tokens(window as u32, state.config.compact_keep),
        marker,
    );
    if tail < prefix {
        tail = prefix;
    }
    let split_at = split_index(msgs, prefix, tail, marker);
    let middle_to = split_at.unwrap_or(tail);
    let middle = collect(msgs, prefix, middle_to, marker, skip);
    let mut model = update(state, text, at, NOTE_TASK, without_extra(prev), &middle)?;
    let mut tokens = model.tokens;
    let mut split_text = String::new();
    if let Some(s) = split_at {
        let part = collect(msgs, s, tail, marker, skip);
        if !part.is_empty() {
            let piece = update(state, text, at, SPLIT_TASK, "", &part)?;
            split_text = piece.text;
            tokens += piece.tokens;
        }
    }
    let mut include_split = !split_text.is_empty();
    let mut guard = msgs.len() + 2;
    while guard > 0 {
        guard -= 1;
        let mut paths = Vec::new();
        paths_from_note(&mut paths, prev);
        paths_from_msgs(&mut paths, &msgs[prefix..tail.min(msgs.len())]);
        let note = finish_note(&model.text, &split_text, &paths);
        let built = assemble(
            text.late_system,
            msgs,
            &note,
            &split_text,
            include_split,
            tail,
            marker,
        );
        let mut next = raw.clone();
        next["messages"] = Value::Array(built.clone());
        match chat::prepare_messages(state, &next, &built) {
            Ok(prepared) => {
                if let Some(dir) = state.config.compact_memory.as_deref() {
                    crate::compact_mem::store_mem(dir, msgs, tail, &note);
                }
                let used = prepared.prompt_len;
                return Ok((
                    prepared,
                    Stamp {
                        window,
                        used,
                        compacted: true,
                        messages_compacted: count_between(msgs, prefix, tail, marker),
                        kept_messages: count_between(msgs, tail, msgs.len(), marker),
                        note_tokens: tokens,
                        note,
                    },
                ));
            }
            Err(r) if r.kind == Kind::ContextLength => {
                if include_split {
                    include_split = false;
                    continue;
                }
                let Some(nxt) = next_cut(msgs, tail, marker) else {
                    return Err(r);
                };
                let dropped = collect(msgs, tail, nxt, marker, 0);
                let piece = update(
                    state,
                    text,
                    at,
                    NOTE_TASK,
                    without_extra(&model.text),
                    &dropped,
                )?;
                tokens += piece.tokens;
                model = piece;
                tail = nxt;
            }
            Err(r) => return Err(r),
        }
    }
    let prepared = chat::prepare(state, raw)?;
    let used = prepared.prompt_len;
    Ok((prepared, Stamp::plain(window, used)))
}

/// The compact-aware prepare (compact.zig:88): the plain one when the
/// request fits, the rebuild when it does not or the window refused it.
pub fn prepare_compact(
    state: &AppState,
    raw: &Value,
    at: CompactAt,
) -> Result<(Prepared, Stamp), Refused> {
    let window: i64 = state
        .engine_info
        .as_ref()
        .map(|i| i.context_window as i64)
        .unwrap_or(0);
    let has_msgs = raw.get("messages").and_then(Value::as_array).is_some();
    let can = has_msgs && window > 0;
    match chat::prepare(state, raw) {
        Ok(prepared) => {
            let used = prepared.prompt_len;
            let wanted = prepared.request.max_tokens as i64;
            if !can || !over(used, wanted, window, at) {
                return Ok((prepared, Stamp::plain(window, used)));
            }
        }
        Err(r) if r.kind == Kind::ContextLength && can => {}
        Err(r) => return Err(r),
    }
    // The normalized list upstream's plan handed compaction
    // (openai.zig:126's normalize, compact.zig:104's input.messages)
    let messages_value = raw
        .get("messages")
        .ok_or_else(|| Refused::request("messages must be a list"))?;
    let needs_user_after_tool = state
        .text
        .as_ref()
        .map(|t| t.needs_user_after_tool)
        .unwrap_or_default();
    let normalized =
        crate::messages::normalize(Some(messages_value), "system", needs_user_after_tool)
            .map_err(Refused::request)?;
    let normalized = crate::messages::tool_arguments(normalized);
    rebuild(state, raw, at, &normalized)
}

/// The stamp on a reply (compact.zig:83): the compaction object beside the
/// context fields the runtime already carries.
fn stamp_reply(reply: &mut crate::chat::Reply, s: &Stamp) {
    if !reply.runtime.is_object() {
        return;
    }
    reply.runtime["context_window"] = json!(s.window);
    reply.runtime["context_used"] = json!(s.used);
    let obj = s.object();
    if !obj.is_null() {
        reply.runtime["compaction"] = obj;
    }
}

/// Compaction's whole reply (compact.zig:78 run): the compact-aware
/// prepare, the generation, the stamp.
pub fn run(state: &AppState, raw: &Value, at: CompactAt) -> crate::chat::Reply {
    let (prepared, stamp) = match prepare_compact(state, raw, at) {
        Ok(ready) => ready,
        Err(r) => return crate::chat::failed_reply(r),
    };
    let mut reply = chat::generate_prepared(state, prepared);
    stamp_reply(&mut reply, &stamp);
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The thresholds (compact.zig:44-67).
    #[test]
    fn the_thresholds() {
        // auto reserves the bigger of 15% and 16384
        assert_eq!(reserve_tokens(100_000, CompactAt::Auto), 16384);
        assert_eq!(reserve_tokens(200_000, CompactAt::Auto), 30_000);
        assert_eq!(note_budget_tokens(200_000, CompactAt::Auto), 8192);
        assert_eq!(note_budget_tokens(30_000, CompactAt::Auto), 8192);
        assert_eq!(note_budget_tokens(10_000, CompactAt::Auto), 8192);
        // a fraction keeps the window minus its threshold
        assert_eq!(reserve_tokens(100_000, CompactAt::Fraction(0.9)), 10_000);
        // keep: the override, else min(20000, a quarter)
        assert_eq!(keep_tokens(100_000, None), 20_000);
        assert_eq!(keep_tokens(40_000, None), 10_000);
        assert_eq!(keep_tokens(100_000, Some(5_000)), 5_000);
        // over: the auto reserve, the fraction's threshold
        // The auto reserve leaves 83_616 of a 100_000-token window
        assert!(!over(80_000, 3_000, 100_000, CompactAt::Auto));
        assert!(over(80_000, 4_000, 100_000, CompactAt::Auto));
        assert!(over(80_000, 4_000, 100_000, CompactAt::Fraction(0.8)));
        assert!(over(83_617, 0, 100_000, CompactAt::Auto));
        assert!(!over(83_616, 0, 100_000, CompactAt::Auto));
        // a negative reply budget counts as none
        assert!(!over(90_000, -5, 100_000, CompactAt::Fraction(0.9)));
    }

    /// The message indexes (compact.zig:347-418).
    #[test]
    fn the_message_indexes() {
        let msgs = vec![
            json!({"role": "system", "content": "lead"}),
            json!({"role": "developer", "content": "dev"}),
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "second"}),
            json!({"role": "assistant", "content": "a2"}),
        ];
        assert_eq!(prefix_len(&msgs), 2);
        assert_eq!(find_marker(&msgs), None);
        // The marker: the last message carrying the line
        let marked = [
            msgs.clone(),
            vec![
                json!({"role": "system", "content": format!("{MARKER_LINE}\nthe note")}),
                json!({"role": "user", "content": "after"}),
            ],
        ]
        .concat();
        assert_eq!(find_marker(&marked), Some(6));
        assert_eq!(marker_body(&marked[6]), Some("the note"));
        assert_eq!(marker_body(&msgs[0]), None, "the line must lead");
        // The split: a kept assistant's turn start
        assert_eq!(split_index(&msgs, 2, 5, None), Some(4));
        assert_eq!(split_index(&msgs, 2, 4, None), None, "the kept user message has no split");
        // The next earlier cut
        assert_eq!(next_cut(&msgs, 3, None), Some(4));
        assert_eq!(next_cut(&msgs, 5, None), None, "no cut before the last user");
        assert_eq!(last_user(&msgs, None), Some(4));
        assert_eq!(count_between(&msgs, 2, 6, None), 4);
        assert_eq!(count_between(&msgs, 2, 6, Some(3)), 3);
        // collect skips the covered and the marker
        let got = collect(&msgs, 2, 6, Some(3), 0);
        assert_eq!(got.len(), 3);
        let got = collect(&msgs, 0, 6, None, 4);
        assert_eq!(got.len(), 2);
    }

    /// The transcript lines with calls (compact.zig:448-476).
    #[test]
    fn the_transcript_lines() {
        let m = json!({"role": "assistant", "content": "hi", "tool_calls": [
            {"function": {"name": "read", "arguments": "{\"path\": \"/tmp/x\"}"}},
            {"function": {"name": "write", "arguments": {"path": "/tmp/y", "paths": ["/a", "/b"]}}},
        ]});
        assert_eq!(
            line_of(&m),
            "assistant: hi call read {\"path\": \"/tmp/x\"} call write {\"path\":\"/tmp/y\",\"paths\":[\"/a\",\"/b\"]}"
        );
        assert_eq!(transcript(&[msg("user", "a"), msg("assistant", "b")]), "user: a\nassistant: b");
    }

    /// The paths a note and a conversation's calls list (compact.zig:478-530).
    #[test]
    fn the_paths() {
        let mut list = Vec::new();
        paths_from_note(&mut list, &format!("before\n{FILES_HEAD}- /one\n- /two\n-  /spaced \nafter"));
        assert_eq!(list, ["/one", "/two", "/spaced"]);
        let calls = vec![json!({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "r", "arguments": "{\"file_path\": \"/f\", \"paths\": \"/s\"}"}},
            {"function": {"name": "w", "arguments": {"paths": ["/a", "/a", ""]}}},
        ]})];
        paths_from_msgs(&mut list, &calls);
        assert_eq!(list, ["/one", "/two", "/spaced", "/f", "/s", "/a"]);
    }

    /// The note's own sections written and stripped (compact.zig:213/239).
    #[test]
    fn the_note_finishes_without_its_extras() {
        let note = "the body\n\n## Files read and changed\n- /x\n";
        assert_eq!(without_extra(note), "the body");
        let done = finish_note(
            "the body",
            "the split",
            &["/x".to_string()],
        );
        assert_eq!(
            done,
            "the body\n\n## Kept reply context\nthe split\n\n## Files read and changed\n- /x\n"
        );
        // and the extras strip again
        assert_eq!(without_extra(&done), "the body");
        // a body alone with a trailing blank line before the head
        assert_eq!(without_extra("the body\n\n## Kept reply context\ns"), "the body");
        assert_eq!(finish_note("", "", &[]), "");
        assert_eq!(finish_note("", "", &["/x".into()]), "## Files read and changed\n- /x\n");
    }

    /// The assembled conversation (compact.zig:183).
    #[test]
    fn the_rebuilt_conversation() {
        let msgs = vec![
            json!({"role": "system", "content": "lead"}),
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "kept"}),
        ];
        let built = assemble("user", &msgs, "the note", "the split", true, 3, None);
        assert_eq!(built.len(), 4);
        assert_eq!(built[0]["role"], "system");
        assert_eq!(built[1]["role"], "user");
        assert_eq!(built[1]["content"], format!("{MARKER_LINE}\nthe note"));
        assert_eq!(built[2]["content"], "Earlier in this turn:\nthe split");
        assert_eq!(built[3]["content"], "kept");
        // without the split
        let built = assemble("system", &msgs, "n", "", false, 3, None);
        assert_eq!(built.len(), 3);
        // the marker drops everywhere
        let marked = [
            msgs[..2].to_vec(),
            vec![json!({"role": "system", "content": MARKER_LINE})],
            msgs[2..].to_vec(),
        ]
        .concat();
        let built = assemble("system", &marked, "n", "", false, 2, Some(2));
        // The prefix, the note, and the two tail messages past the marker
        assert_eq!(built.len(), 4);
        assert!(built.iter().all(|m| m["content"].as_str() != Some(MARKER_LINE)));
        // an empty conversation keeps one user message
        let built = assemble("system", &[], "", "", false, 0, None);
        assert_eq!(built.len(), 1);
        assert_eq!(built[0]["role"], "user");
    }
}
