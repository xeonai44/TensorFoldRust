//! Tool calls in a finished reply, every family's markup, as OpenAI
//! tool_calls (`tools.parse_tool_calls_from_content`).
//!
//! Port of `zig/src/server/tool_parse.zig` (501 lines): the envelope
//! finding for every family (Qwen, the custom XML tags, Gemma, DSML), the
//! payload parsing chain (JSON, Qwen's XML function block, GLM, Gemma 4,
//! DSML invokes), the bare-JSON reply body, and the streaming half beside
//! this in `deltas` (tool_streaming.zig, 46 lines).

use crate::reply_text::{is_space_byte, py_strip};
use crate::tool_params::{self, Schemas};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

use crate::pyrepr::truthy;

/// The Qwen tool call markup: the exact bytes from upstream (hex
/// 3c746f6f6c5f63616c6c3e, 11 bytes) and its closer (hex
/// 3c2f746f6f6c5f63616c6c3e, 12 bytes).
pub const OPENER: &str = "\x3c\x74\x6f\x6f\x6c\x5f\x63\x61\x6c\x6c\x3e";
pub const CLOSER: &str = "\x3c\x2f\x74\x6f\x6f\x6c\x5f\x63\x61\x6c\x6c\x3e";

/// The DSML structured block, upstream tool_parse.zig:23-28.
const DSML_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
const DSML_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
const INVOKE_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}invoke name=\"";
const INVOKE_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}invoke>";
const PARAM_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}parameter name=\"";
const PARAM_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}parameter>";

/// GLM's markers (the bytes of tool_parse.zig:364-385): the name separator
/// (9), the key end (10), the value start (11) and the value end (12).
const GLM_KV: &str = "\x3c\x61\x72\x67\x5f\x6b\x65\x79\x3e";
const GLM_KE: &str = "\x3c\x2f\x61\x72\x67\x5f\x6b\x65\x79\x3e";
const GLM_VS: &str = "\x3c\x61\x72\x67\x5f\x76\x61\x6c\x75\x65\x3e";
const GLM_VE: &str = "\x3c\x2f\x61\x72\x67\x5f\x76\x61\x6c\x75\x65\x3e";

/// Upstream: `Error` (tool_parse.zig:17) - what Python raises, at the port
/// one shape: the unit. Where upstream's `error.Invalid` separates a raise
/// from a non-call (`null`), this type carries the distinction.
pub type Invalid = ();

/// One parsed call. Upstream: `Call` (tool_parse.zig:18).
struct Call {
    name: String,
    arguments: Value,
}

/// (start, end, payload) of each tool-call block. Upstream: `Envelope`
/// (tool_parse.zig:19).
#[derive(Clone, Debug)]
struct Envelope {
    start: usize,
    end: usize,
    payload: String,
}

/// What `parse` returns: the content without the calls' markup, and the
/// OpenAI-format calls (None when the reply makes none). Upstream:
/// `Parsed` (tool_parse.zig:21).
pub struct Parsed {
    pub content: String,
    pub calls: Option<Vec<Value>>,
}

/// The offered tools by lowercase name: the original spelling.
/// Upstream: parse's `known` (tool_parse.zig:100).
type Known = HashMap<String, String>;

/// Upstream: `findCI` (tool_parse.zig:30) - ASCII case-insensitive.
fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    let hay_lower = hay[from..].to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    hay_lower.find(&needle_lower).map(|at| at + from)
}

/// Upstream: `find` (tool_parse.zig:35).
fn find(hay: &str, needle: &str, from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..].find(needle).map(|at| at + from)
}

/// Python's `\w` on one byte, non-ASCII bytes counting as letters.
/// Upstream: `word` (tool_parse.zig:41).
fn word(ch: u8) -> bool {
    ch.is_ascii_alphanumeric() || ch == b'_' || ch >= 0x80
}

/// Upstream: `allSpace` (tool_parse.zig:47).
fn all_space(s: &str) -> bool {
    py_strip(s).is_empty()
}

/// (start, end, payload) of each tool-call block in order; a block inside an
/// earlier one is part of it. Upstream: `envelopes` (tool_parse.zig:52) -
/// the four families, sorted by start then end then payload, the overlaps
/// dropped.
fn envelopes(text: &str) -> Vec<Envelope> {
    let mut found: Vec<Envelope> = Vec::new();
    let mut pos: usize = 0;
    // Qwen's markup, any case
    while let Some(s) = find_ci(text, OPENER, pos) {
        let Some(e) = find_ci(text, CLOSER, s + 11) else { break };
        found.push(Envelope {
            start: s,
            end: e + 12,
            payload: py_strip(&text[s + 11..e]).to_string(),
        });
        pos = e + 12;
    }
    pos = 0;
    // The custom XML family: <NAME:tool_call> ... </NAME:tool_call>, any case
    while let Some(s) = text[pos..].find('<').map(|at| at + pos) {
        pos = s + 1;
        let bytes = text.as_bytes();
        let mut j = s + 1;
        if j >= bytes.len() || !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
            continue;
        }
        j += 1;
        while j < bytes.len() && (word(bytes[j]) || bytes[j] == b'.' || bytes[j] == b'-') {
            j += 1;
        }
        if !text[j..].to_ascii_lowercase().starts_with(":tool_call>") {
            continue;
        }
        let closer = format!("</{}:tool_call>", &text[s + 1..j]);
        let Some(e) = find_ci(text, &closer, j + 11) else { continue };
        found.push(Envelope {
            start: s,
            end: e + closer.len(),
            payload: py_strip(&text[j + 11..e]).to_string(),
        });
        pos = e + closer.len();
    }
    pos = 0;
    // Gemma's markup, exact case
    while let Some(s) = find(text, "<|tool_call>", pos) {
        let Some(e) = find(text, "<tool_call|>", s + 12) else { break };
        found.push(Envelope {
            start: s,
            end: e + 12,
            payload: py_strip(&text[s + 12..e]).to_string(),
        });
        pos = e + 12;
    }
    pos = 0;
    // DSML's structured block, exact case; the payload keeps its wrapper
    while let Some(s) = find(text, DSML_OPEN, pos) {
        let Some(e) = find(text, DSML_CLOSE, s + DSML_OPEN.len()) else { break };
        found.push(Envelope {
            start: s,
            end: e + DSML_CLOSE.len(),
            payload: text[s..e + DSML_CLOSE.len()].to_string(),
        });
        pos = e + DSML_CLOSE.len();
    }
    found.sort_by(|x, y| {
        (x.start, x.end, x.payload.as_bytes()).cmp(&(y.start, y.end, y.payload.as_bytes()))
    });
    let mut kept: Vec<Envelope> = Vec::with_capacity(found.len());
    for env in found {
        if kept.is_empty() || env.start >= kept.last().unwrap().end {
            kept.push(env);
        }
    }
    kept
}

/// `parse_tool_calls_from_content`: the reply's content and its calls (None
/// when it makes none). Upstream: `parse` (tool_parse.zig:98) - a one-call
/// reply (`max_calls`) drops what it will not take; unlimited replies keep a
/// malformed call as text.
pub fn parse(text: &str, tools: &[Value], max_calls: Option<usize>) -> Parsed {
    if tools.is_empty() {
        return Parsed {
            content: text.to_string(),
            calls: None,
        };
    }
    let mut known: Known = HashMap::new();
    for tool in tools {
        let name = crate::tool_specs::tool_name(tool);
        known.insert(name.to_ascii_lowercase(), name);
    }
    let schemas = tool_params::schemas(tools);
    let envs = envelopes(text);
    if envs.is_empty() {
        if let Some(calls) = bare_calls(text, &known, max_calls) {
            return Parsed {
                content: String::new(),
                calls: Some(calls),
            };
        }
        return Parsed {
            content: text.to_string(),
            calls: None,
        };
    }
    let mut calls: Vec<Value> = Vec::new();
    let mut residue = String::new();
    let mut cursor: usize = 0;
    for env in &envs {
        residue.push_str(&text[cursor..env.start]);
        cursor = env.end;
        if let Some(m) = max_calls {
            if calls.len() >= m {
                continue;
            }
        }
        let parsed: Option<Vec<Call>> = if env.payload.starts_with(DSML_OPEN) {
            let inner = &env.payload[DSML_OPEN.len()..env.payload.len() - DSML_CLOSE.len()];
            dsml_calls(inner).ok()
        } else {
            match parse_payload(&env.payload, Some(&schemas), max_calls.is_some()) {
                Ok(one) => one.map(|c| vec![c]),
                Err(_) => None,
            }
        };
        let good = parsed.as_ref().is_some_and(|list| {
            !list.is_empty() && list.iter().all(|c| call_name(&c.name, &known).is_some())
        });
        if !good {
            // a malformed or unoffered call stays text when calls are
            // unlimited, and drops out of a one-call reply
            if max_calls.is_none() {
                residue.push_str(&text[env.start..env.end]);
            }
            continue;
        }
        for c in parsed.unwrap() {
            if max_calls.is_none() || calls.len() < max_calls.unwrap() {
                calls.push(openai_call(
                    &call_name(&c.name, &known).unwrap(),
                    &c.arguments,
                ));
            }
        }
    }
    residue.push_str(&text[cursor..]);
    Parsed {
        content: py_strip(&residue).to_string(),
        calls: if calls.is_empty() { None } else { Some(calls) },
    }
}

/// What closes the call `text` ends inside (the model's end token came
/// before its markup did). Upstream: `closeCall` (tool_parse.zig:154) - the
/// opener `envelopes` leaves without a closer, so the one appended pairs
/// with it; the strict reading keeps only a whole call to an offered tool.
pub fn close_call(text: &str, tools: &[Value]) -> String {
    let Some(start) = last_open(text) else {
        return String::new();
    };
    let block = &text[start + 11..];
    let tail = arguments_close(py_strip(block));
    let closed = format!("{OPENER}{block}{tail}{CLOSER}");
    if parse(&closed, tools, Some(1)).calls.is_none() {
        return String::new();
    }
    format!("{tail}{CLOSER}")
}

/// The last opener `envelopes` would leave unmatched. Upstream: the while
/// loop of `closeCall` (tool_parse.zig:157-159).
fn last_open(text: &str) -> Option<usize> {
    let mut pos = 0;
    let mut start = None;
    while let Some(s) = find_ci(text, OPENER, pos) {
        match find_ci(text, CLOSER, s + 11) {
            Some(e) => pos = e + 12,
            None => {
                start = Some(s);
                break;
            }
        }
    }
    start
}

/// What a call's arguments lack to close. Upstream: `argumentsClose`
/// (tool_parse.zig:169).
fn arguments_close(body: &str) -> String {
    if body.to_ascii_lowercase().starts_with("<function=") {
        return if body.to_ascii_lowercase().ends_with("</function>") {
            String::new()
        } else {
            "</function>".into()
        };
    }
    if body.is_empty() || (body.as_bytes()[0] != b'{' && body.as_bytes()[0] != b'[') {
        return String::new();
    }
    match tool_params::closed_json(body) {
        // The caller strips the body, so the closed form starts with it
        Some(closed) => closed.get(body.len()..).unwrap_or("").to_string(),
        None => String::new(),
    }
}

/// The offered tool's spelling, or None when the request did not declare
/// that name. Upstream: `callName` (tool_parse.zig:177).
fn call_name(name: &str, known: &Known) -> Option<String> {
    known.get(&py_strip(name).to_ascii_lowercase()).cloned()
}

/// `_openai_tool_call`: the call under `name`, arguments as compact JSON.
/// Upstream: `openaiCall` (tool_parse.zig:183) - the id from `ids.make`
/// (tool_parse.zig:188), 24 random hex digits after `call_`.
fn openai_call(name: &str, arguments: &Value) -> Value {
    json!({
        "id": crate::ids::make("call_", 24),
        "type": "function",
        "function": {
            "name": name,
            "arguments": zig_stringify(arguments, true),
        },
    })
}

/// The whole reply as bare JSON calls, or None. Upstream: `bareCalls`
/// (tool_parse.zig:194) - a fenced ```json block counts; JSON that is not a
/// call (a structured answer) is the reply's content, not calls.
fn bare_calls(text: &str, known: &Known, max_calls: Option<usize>) -> Option<Vec<Value>> {
    let stripped = strip_fence(py_strip(text));
    let first = *stripped.as_bytes().first()?;
    if first != b'[' && first != b'{' {
        return None;
    }
    let value: Value = serde_json::from_str(stripped).ok()?;
    let items: Vec<Value> = match value {
        Value::Array(list) => list,
        one => vec![one],
    };
    let mut calls: Vec<Value> = Vec::new();
    for item in items {
        if let Some(m) = max_calls {
            if calls.len() >= m {
                break;
            }
        }
        if !item.is_object() {
            return None;
        }
        // Re-parsed through its document, the way upstream round-trips the
        // item; any compact spelling reads the same
        let doc = serde_json::to_string(&item).ok()?;
        match parse_payload(&doc, None, max_calls.is_some()) {
            Err(_) => {
                // a structured answer is the reply's content
                max_calls?;
                continue;
            }
            Ok(None) => return None,
            Ok(Some(c)) => {
                // bare JSON naming a tool the request did not offer is a
                // structured answer, the reply's content
                let name = known
                    .get(&py_strip(&c.name).to_ascii_lowercase())
                    .cloned()?;
                calls.push(openai_call(&name, &c.arguments));
            }
        }
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// `_strip_json_fence`: the inside of a ```json fence, else the text.
/// Upstream: `stripFence` (tool_parse.zig:223).
fn strip_fence(text: &str) -> &str {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_space_byte(bytes[i]) {
        i += 1;
    }
    if !text[i..].starts_with("```") {
        return text;
    }
    i += 3;
    if text[i..].to_ascii_lowercase().starts_with("json") {
        i += 4;
    }
    let mut at = i;
    while let Some(f) = find(text, "```", at) {
        if all_space(&text[f + 3..]) {
            return py_strip(&text[i..f]);
        }
        at = f + 1;
    }
    text
}

/// `_parse_tool_call_payload`: one call from a block; None when it is not
/// one, Err where Python raises. Upstream: `payload` (tool_parse.zig:237).
fn parse_payload(
    block: &str,
    schemas: Option<&Schemas>,
    complete: bool,
) -> Result<Option<Call>, Invalid> {
    if block.starts_with("call:") || block.starts_with(':') {
        if let Some(c) = gemma_call(block)? {
            return Ok(Some(c));
        }
    }
    let decoded: Option<Value> = serde_json::from_str(block).ok();
    if let Some(v) = &decoded {
        if let Value::Array(list) = v {
            for item in list {
                let doc = serde_json::to_string(item).unwrap_or_default();
                if let Some(c) = parse_payload(&doc, None, complete)? {
                    return Ok(Some(c));
                }
            }
            return Ok(None);
        }
        if let Value::Object(obj) = v {
            let holder: &Map<String, Value> = match obj.get("function") {
                Some(Value::Object(f)) => f,
                _ => obj,
            };
            let name_keys: &[&str] = if std::ptr::eq(holder, obj) {
                &["name", "tool", "function", "call"]
            } else {
                &["name", "tool", "function"]
            };
            let name = name_keys
                .iter()
                .find_map(|k| holder.get(*k).filter(|v| truthy(v)))
                .map(|v| py_strip(&crate::tool_specs::py_str(v)).to_string())
                .unwrap_or_default();
            let mut arguments: Option<Value> = None;
            for k in ["arguments", "args", "parameters"] {
                if holder.contains_key(k) {
                    arguments = holder.get(k).cloned();
                    break;
                }
            }
            let args = arguments.unwrap_or_else(|| {
                // The loose arguments: the holder's other keys
                let mut loose = Map::new();
                for (k, v) in holder {
                    if !["name", "tool", "function", "call", "type"].contains(&k.as_str()) {
                        loose.insert(k.clone(), v.clone());
                    }
                }
                Value::Object(loose)
            });
            if name.is_empty() {
                return Err(());
            }
            let object = json_object(&args)?;
            if complete && !tool_params::finite(&object) {
                return Err(());
            }
            return Ok(Some(Call { name, arguments: object }));
        }
    }
    if let Some(c) = function_block(block, schemas, complete)? {
        return Ok(Some(c));
    }
    if decoded.is_none() {
        let lead = block.trim_start_matches([' ', '\t', '\r', '\n', '\x0b', '\x0c']);
        let first = lead.as_bytes().first();
        if lead.is_empty() || !matches!(first, Some(b'{') | Some(b'[') | Some(b'<')) {
            return Ok(glm_call(block, schemas, complete));
        }
    }
    Ok(None)
}

/// `_tool_json_object`: arguments as an object, Err where Python raises.
/// Upstream: `jsonObject` (tool_parse.zig:287).
fn json_object(v: &Value) -> Result<Value, Invalid> {
    match v {
        Value::Object(_) => Ok(v.clone()),
        Value::Null => Ok(Value::Object(Map::new())),
        Value::String(s) => {
            let doc = py_strip(s);
            if doc.is_empty() {
                return Ok(Value::Object(Map::new()));
            }
            match serde_json::from_str::<Value>(doc) {
                Ok(Value::Object(parsed)) => Ok(Value::Object(parsed)),
                _ => Err(()),
            }
        }
        _ => Err(()),
    }
}

/// <function=NAME> <parameter=K>V</parameter> </function> (Qwen3-Coder's
/// XML). Upstream: `functionBlock` (tool_parse.zig:306).
fn function_block(
    block: &str,
    schemas: Option<&Schemas>,
    complete: bool,
) -> Result<Option<Call>, Invalid> {
    let bytes = block.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_space_byte(bytes[i]) {
        i += 1;
    }
    if !block[i..].to_ascii_lowercase().starts_with("<function=") {
        return Ok(None);
    }
    i += 10;
    let name_start = i;
    while i < bytes.len() && bytes[i] != b'>' && !is_space_byte(bytes[i]) {
        i += 1;
    }
    if i == name_start || i >= bytes.len() || bytes[i] != b'>' {
        return Ok(None);
    }
    let name = &block[name_start..i];
    let body_start = i + 1;
    // The last </function> with only space after it
    let mut at = body_start;
    let mut close = None;
    while let Some(f) = find_ci(block, "</function>", at) {
        at = f + 1;
        if all_space(&block[f + 11..]) {
            close = Some(f);
        }
    }
    let Some(close) = close else { return Ok(None) };
    let body = py_strip(&block[body_start..close]);
    let params = parameter_blocks(body);
    if complete && !all_space(&params.residue) {
        return Ok(None);
    }
    let mut arguments = Map::new();
    for p in &params.items {
        let key = py_strip(p.key);
        let schema = match schemas {
            Some(s) => tool_params::schema_of(s, name, key),
            None => Value::Object(Map::new()),
        };
        arguments.insert(
            key.to_string(),
            tool_params::decode(p.value, &schema, true),
        );
    }
    Ok(Some(Call {
        name: name.to_string(),
        arguments: Value::Object(arguments),
    }))
}

/// One parameter block of a function body. Upstream: `Param`
/// (tool_parse.zig:332).
struct Param<'a> {
    key: &'a str,
    value: &'a str,
}

/// The parameter blocks of a function body, and the rest. Upstream:
/// `parameterBlocks` (tool_parse.zig:336) - <parameter=K> value
/// </parameter>, any case.
struct Params<'a> {
    items: Vec<Param<'a>>,
    residue: String,
}

fn parameter_blocks(body: &str) -> Params<'_> {
    let mut items: Vec<Param> = Vec::new();
    let mut residue = String::new();
    let mut pos: usize = 0;
    let mut kept: usize = 0;
    let bytes = body.as_bytes();
    while let Some(s) = find_ci(body, "<parameter=", pos) {
        pos = s + 1;
        let mut i = s + 11;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'>' && !is_space_byte(bytes[i]) {
            i += 1;
        }
        if i == key_start || i >= bytes.len() || bytes[i] != b'>' {
            continue;
        }
        let key = &body[key_start..i];
        i += 1;
        if i < bytes.len() && bytes[i] == b'\n' {
            i += 1;
        }
        let Some(e) = find_ci(body, "</parameter>", i) else { break };
        let mut value = &body[i..e];
        if value.ends_with('\n') {
            value = &value[..value.len() - 1];
        }
        items.push(Param { key, value });
        residue.push_str(&body[kept..s]);
        kept = e + 12;
        pos = kept;
    }
    residue.push_str(&body[kept..]);
    Params { items, residue }
}

/// GLM's NAME, then its key and value pairs - values decoded as Qwen's
/// (without Python spellings). Upstream: `glmCall` (tool_parse.zig:363); it
/// cannot raise, so the port returns the Option alone.
fn glm_call(block: &str, schemas: Option<&Schemas>, complete: bool) -> Option<Call> {
    let cut = find(block, GLM_KV, 0).unwrap_or(block.len());
    let name = py_strip(&block[..cut]);
    if name.is_empty() {
        return None;
    }
    // any one-line name, spaces included: parse keeps a name the request
    // did not declare as text
    if name.bytes().any(|ch| matches!(ch, b'<' | b'>' | b'\n' | b'\r')) {
        return None;
    }
    let rest = &block[cut..];
    let mut pairs: Vec<Param> = Vec::new();
    let mut residue = String::new();
    let mut pos: usize = 0;
    let mut kept: usize = 0;
    'outer: while let Some(s) = find(rest, GLM_KV, pos) {
        pos = s + 1;
        let mut key_end_at = s + 9;
        while let Some(ke) = find(rest, GLM_KE, key_end_at) {
            key_end_at = ke + 1;
            let mut j = ke + 10;
            while j < rest.len() && is_space_byte(rest.as_bytes()[j]) {
                j += 1;
            }
            if !rest[j..].starts_with(GLM_VS) {
                continue;
            }
            let v_start = j + 11;
            let Some(ve) = find(rest, GLM_VE, v_start) else {
                continue 'outer;
            };
            pairs.push(Param {
                key: &rest[s + 9..ke],
                value: &rest[v_start..ve],
            });
            residue.push_str(&rest[kept..s]);
            kept = ve + 12;
            pos = kept;
            continue 'outer;
        }
    }
    residue.push_str(&rest[kept..]);
    if complete && !all_space(&residue) {
        return None;
    }
    let mut arguments = Map::new();
    for p in &pairs {
        let key = py_strip(p.key);
        let schema = match schemas {
            Some(s) => tool_params::schema_of(s, name, key),
            None => Value::Object(Map::new()),
        };
        arguments.insert(key.to_string(), tool_params::decode(p.value, &schema, false));
    }
    Some(Call {
        name: name.to_string(),
        arguments: Value::Object(arguments),
    })
}

/// Gemma 4's call:NAME{key:value,...} (or bare :NAME{...}): strings between
/// the fence markers, keys bare. Upstream: `gemmaCall` (tool_parse.zig:402).
fn gemma_call(block: &str) -> Result<Option<Call>, Invalid> {
    let bytes = block.as_bytes();
    let mut i = 0;
    if block.starts_with("call:") {
        i = 4;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return Ok(None);
    }
    i += 1;
    let name_start = i;
    while i < bytes.len() && (word(bytes[i]) || bytes[i] == b'.' || bytes[i] == b'-') {
        i += 1;
    }
    if i == name_start {
        return Ok(None);
    }
    let name = &block[name_start..i];
    while i < bytes.len() && is_space_byte(bytes[i]) {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return Ok(None);
    }
    let mut end = block.len();
    if block.ends_with("}\n") {
        end -= 1;
    }
    if end <= i || bytes[end - 1] != b'}' {
        return Ok(None);
    }
    let inner = &block[i..end];
    let fence = "<|\"|>";
    // The strings between the fences, marked so the bare keys can be
    // quoted without touching them
    let mut strings: Vec<&str> = Vec::new();
    let mut marked = String::new();
    let mut pos: usize = 0;
    while let Some(s) = find(inner, fence, pos) {
        let Some(e) = find(inner, fence, s + 5) else { break };
        marked.push_str(&inner[pos..s]);
        marked.push_str(&format!("\x00{}\x00", strings.len()));
        strings.push(&inner[s + 5..e]);
        pos = e + 5;
    }
    marked.push_str(&inner[pos..]);
    let src = marked.as_bytes();
    // Bare keys after { or , become quoted ones
    let mut keyed = String::new();
    let mut p: usize = 0;
    let mut copied: usize = 0;
    while p < src.len() {
        if p != 0 && !matches!(src[p - 1], b'{' | b',') {
            p += 1;
            continue;
        }
        let mut j = p;
        while j < src.len() && is_space_byte(src[j]) {
            j += 1;
        }
        if j >= src.len() || !(src[j].is_ascii_alphabetic() || src[j] == b'_') {
            p += 1;
            continue;
        }
        let k_start = j;
        while j < src.len() && (word(src[j]) || src[j] == b'-') {
            j += 1;
        }
        let key = &marked[k_start..j];
        while j < src.len() && is_space_byte(src[j]) {
            j += 1;
        }
        if j >= src.len() || src[j] != b':' {
            p += 1;
            continue;
        }
        keyed.push_str(&marked[copied..p]);
        keyed.push_str(&format!("\"{key}\":"));
        copied = j + 1;
        p = j;
        p += 1;
    }
    keyed.push_str(&marked[copied..]);
    // The strings return, quoted
    let mut text = keyed;
    for (idx, value) in strings.iter().enumerate() {
        let mark = format!("\x00{idx}\x00");
        let quoted = serde_json::to_string(value).unwrap_or_default();
        text = text.replace(&mark, &quoted);
    }
    Ok(Some(Call {
        name: name.to_string(),
        arguments: json_object(&Value::String(text))?,
    }))
}

/// Every invoke of a DSML block, or Err when anything in it is not a
/// well-formed invoke. Upstream: `dsmlCalls` (tool_parse.zig:457).
fn dsml_calls(block: &str) -> Result<Vec<Call>, Invalid> {
    let mut calls: Vec<Call> = Vec::new();
    let mut at: usize = 0;
    let mut pos: usize = 0;
    while let Some(s) = find(block, INVOKE_OPEN, pos) {
        pos = s + 1;
        let name_start = s + INVOKE_OPEN.len();
        let Some(q) = block[name_start..].find('"').map(|at| at + name_start) else {
            continue;
        };
        if !block[q..].starts_with("\">") {
            continue;
        }
        let Some(e) = find(block, INVOKE_CLOSE, q + 2) else { continue };
        if !all_space(&block[at..s]) {
            return Err(());
        }
        at = e + INVOKE_CLOSE.len();
        pos = at;
        let body = &block[q + 2..e];
        let mut arguments = Map::new();
        let mut p: usize = 0;
        let mut kept: usize = 0;
        let mut residue = String::new();
        while let Some(ps) = find(body, PARAM_OPEN, p) {
            p = ps + 1;
            let ns = ps + PARAM_OPEN.len();
            let Some(nq) = body[ns..].find('"').map(|at| at + ns) else {
                continue;
            };
            let flag = &body[nq..];
            let is_string = flag.starts_with("\" string=\"true\">");
            if !is_string && !flag.starts_with("\" string=\"false\">") {
                continue;
            }
            let v_start = nq + if is_string {
                "\" string=\"true\">".len()
            } else {
                "\" string=\"false\">".len()
            };
            let Some(pe) = find(body, PARAM_CLOSE, v_start) else { continue };
            let value = &body[v_start..pe];
            residue.push_str(&body[kept..ps]);
            kept = pe + PARAM_CLOSE.len();
            p = kept;
            if is_string {
                arguments.insert(body[ns..nq].to_string(), Value::String(value.to_string()));
            } else {
                match serde_json::from_str::<Value>(value) {
                    Ok(v) => {
                        arguments.insert(body[ns..nq].to_string(), v);
                    }
                    Err(_) => return Err(()),
                }
            }
        }
        residue.push_str(&body[kept..]);
        if !all_space(&residue) {
            return Err(());
        }
        calls.push(Call {
            name: py_strip(&block[name_start..q]).to_string(),
            arguments: Value::Object(arguments),
        });
    }
    if calls.is_empty() || !all_space(&block[at..]) {
        return Err(());
    }
    Ok(calls)
}

/// Zig's JSON writer (json_write.zig:15 `stringify`), which the tool
/// surfaces lean on for byte-exact arguments: the compact or wide
/// separators, floats by Python's repr with NaN and Infinity spelled as
/// JavaScript does (json_write.zig:89 `floatJson`), strings as `serde`
/// writes them - the six short escapes and \uXXXX controls, the rest raw.
pub(crate) fn zig_stringify(v: &Value, compact: bool) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_nan() {
                    "NaN".into()
                } else if f.is_infinite() {
                    if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
                } else {
                    crate::pyrepr::float_repr(f)
                }
            }
        }
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(|i| zig_stringify(i, compact)).collect();
            format!("[{}]", parts.join(if compact { "," } else { ", " }))
        }
        Value::Object(o) => {
            let sep = if compact { ":" } else { ": " };
            let parts: Vec<String> = o
                .iter()
                .map(|(k, val)| {
                    format!(
                        "{}{sep}{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        zig_stringify(val, compact)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(if compact { "," } else { ", " }))
        }
    }
}

/// `stream_tool_call_deltas`: each call's opening delta, then its arguments
/// in one more. Upstream: `deltas` (tool_streaming.zig:9) - the id, type and
/// name when truthy, else their fallbacks.
pub fn deltas(calls: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        let Some(function) = call.get("function") else { continue };
        let Some(function_object) = function.as_object() else { continue };
        let id = call
            .get("id")
            .filter(|v| truthy(v))
            .map(crate::tool_specs::py_str)
            .unwrap_or_else(|| format!("call_{index}"));
        let kind = call
            .get("type")
            .filter(|v| truthy(v))
            .map(crate::tool_specs::py_str)
            .unwrap_or_else(|| "function".into());
        let name = function_object
            .get("name")
            .filter(|v| truthy(v))
            .map(crate::tool_specs::py_str)
            .unwrap_or_default();
        let head = json!({
            "index": index,
            "id": id,
            "type": kind,
            "function": {"name": name, "arguments": ""},
        });
        out.push(wrap_calls(head));
        let args = function_object
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !args.is_empty() {
            let more = json!({
                "index": index,
                "function": {"arguments": args},
            });
            out.push(wrap_calls(more));
        }
    }
    out
}

/// `{"tool_calls": [one]}`: one streamed call delta.
/// Upstream: `wrapCalls` (tool_streaming.zig:39).
pub fn wrap_calls(one: Value) -> Value {
    json!({"tool_calls": [one]})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "get_weather"}}),
            json!({"type": "function", "function": {"name": "lookup_time"}}),
        ]
    }

    #[test]
    fn parses_the_qwen_markup() {
        let text = format!(
            "The weather: {}{}{} done.",
            OPENER,
            r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#,
            CLOSER
        );
        let tools = tools();
        let parsed = parse(&text, &tools, None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert!(parsed.content.contains("The weather:"), "{}", parsed.content);
        assert!(!parsed.content.contains("get_weather"), "{}", parsed.content);
        assert!(parsed.content.ends_with("done."), "{}", parsed.content);
    }

    #[test]
    fn parses_the_gemma_markup() {
        let text = r#"<|tool_call>{"name": "lookup_time", "arguments": {"tz": "UTC"}}<tool_call|>"#;
        let tools = tools();
        let parsed = parse(text, &tools, None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "lookup_time");
        assert!(parsed.content.trim().is_empty());
    }

    #[test]
    fn parses_the_custom_xml_markup_any_case() {
        let text = "One <wx:TOOL_CALL>{\"tool\": \"get_weather\", \"args\": {\"city\": \"Rome\"}}</wx:TOOL_CALL> two";
        let parsed = parse(&text, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Rome"}"#);
        assert!(parsed.content.starts_with("One"), "{}", parsed.content);
        assert!(parsed.content.ends_with("two"), "{}", parsed.content);
    }

    #[test]
    fn parses_the_qwen_xml_function_block() {
        let text = format!(
            "{OPENER}\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n{CLOSER}"
        );
        let parsed = parse(&text, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    #[test]
    fn parses_glm_markup() {
        let text = format!(
            "{OPENER}get_weather{GLM_KV}city{GLM_KE}{GLM_VS}Paris{GLM_VE}{CLOSER}"
        );
        let parsed = parse(&text, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    #[test]
    fn parses_gemma_call_markup() {
        let text = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>, days:3}<tool_call|>";
        let parsed = parse(&text, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Paris","days":3}"#);
    }

    #[test]
    fn parses_dsml_invokes() {
        let invoke = "<\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\"><\u{ff5c}DSML\u{ff5c}parameter name=\"city\" string=\"true\">Paris</\u{ff5c}DSML\u{ff5c}parameter></\u{ff5c}DSML\u{ff5c}invoke>";
        let text = format!("<\u{ff5c}DSML\u{ff5c}tool_calls>{invoke}</\u{ff5c}DSML\u{ff5c}tool_calls>");
        let parsed = parse(&text, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    #[test]
    fn parses_bare_json_calls_and_the_json_fence() {
        let fenced =
            "```json\n[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}]\n```";
        let parsed = parse(fenced, &tools(), None);
        let calls = parsed.calls.expect("the reply calls a tool");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert!(parsed.content.is_empty());
        // A single call object works too
        let parsed = parse(
            r#"{ "name": "get_weather", "arguments": { } }"#,
            &tools(),
            None,
        );
        assert!(parsed.calls.is_some());
        // JSON that names an unoffered tool is a structured answer: content
        let parsed = parse(r#"{ "name": "not_offered", "x": 1 }"#, &tools(), None);
        assert!(parsed.calls.is_none());
        assert!(parsed.content.contains("not_offered"));
    }

    #[test]
    fn undeclared_tools_stay_text() {
        let text = format!(
            r#"{OPENER}{{"name": "unknown_fn", "arguments": {{}}}}{CLOSER}"#
        );
        let parsed = parse(&text, &tools(), None);
        assert!(parsed.calls.is_none());
        assert!(parsed.content.contains("unknown_fn"), "{}", parsed.content);
    }

    #[test]
    fn no_tools_means_no_calls() {
        let parsed = parse("plain text", &tools(), None);
        assert!(parsed.calls.is_none());
        assert_eq!(parsed.content, "plain text");
        let empty: Vec<Value> = vec![];
        let text = format!("{OPENER}{{}}{CLOSER}");
        let parsed = parse(&text, &empty, None);
        assert!(parsed.calls.is_none());
    }

    #[test]
    fn a_one_call_reply_drops_the_second_and_the_unparsed() {
        let text = format!(
            "one {OPENER}{{\"name\": \"get_weather\", \"arguments\": {{}}}}{CLOSER} two {OPENER}{{\"name\": \"lookup_time\"}}{CLOSER}"
        );
        let parsed = parse(&text, &tools(), Some(1));
        let calls = parsed.calls.expect("one call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // The second call drops out of the reply entirely
        assert!(
            !parsed.content.contains("lookup_time"),
            "{}",
            parsed.content
        );
        // Unlimited keeps a malformed call as text
        let malformed = format!("m {OPENER}not json at all{CLOSER} e");
        let parsed = parse(&malformed, &tools(), None);
        assert!(parsed.calls.is_none());
        assert_eq!(parsed.content, malformed.trim());
        // The one-call reply drops it, leaving the spaces around it
        let parsed = parse(&malformed, &tools(), Some(1));
        assert!(parsed.calls.is_none());
        assert_eq!(parsed.content, "m  e");
    }

    #[test]
    fn close_call_completes_the_open_block() {
        let tools = vec![json!({"type": "function", "function": {"name": "lookup"}})];
        let open = format!(
            "{OPENER}\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n"
        );
        // Nothing closes a finished call
        assert_eq!(close_call(&format!("{open}{CLOSER}"), &tools), "");
        // The one the text ends inside closes: </function> then the
        // closer, no newline between (tool_parse.zig:165)
        let closing = close_call(&open, &tools);
        assert_eq!(closing, format!("</function>{CLOSER}"));
        // An unoffered function does not
        let unoffered = format!("{OPENER}\n<function=nope>\n<parameter=x>\n1\n");
        assert_eq!(close_call(&unoffered, &tools), "");
        // The JSON-envelope body closes the same way
        let json_open = format!("{OPENER}{{\"name\": \"lookup\", \"arguments\": {{\"q\": \"a\"");
        assert_eq!(close_call(&json_open, &tools), format!("}}}}{CLOSER}"));
    }

    #[test]
    fn deltas_carry_the_opening_and_the_arguments() {
        let calls = vec![json!({
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": r#"{"city": "Paris"}"#},
        })];
        let d = deltas(&calls);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            d[1]["tool_calls"][0]["function"]["arguments"],
            r#"{"city": "Paris"}"#
        );
        // Missing id and type fall back
        let calls = vec![json!({"function": {"name": "x", "arguments": ""}})];
        let d = deltas(&calls);
        assert_eq!(d[0]["tool_calls"][0]["id"], "call_0");
        assert_eq!(d[0]["tool_calls"][0]["type"], "function");
        // A falsy id falls back too
        let calls = vec![json!({"id": "", "type": "function", "function": {"name": "x", "arguments": "y"}})];
        let d = deltas(&calls);
        assert_eq!(d[0]["tool_calls"][0]["id"], "call_0");
    }

    #[test]
    fn the_arguments_stringify_like_zig() {
        assert_eq!(
            zig_stringify(&json!({"a": 1, "b": [true, null]}), true),
            r#"{"a":1,"b":[true,null]}"#
        );
        assert_eq!(zig_stringify(&json!({"a": 1, "b": 2}), false), r#"{"a": 1, "b": 2}"#);
        assert_eq!(zig_stringify(&json!({}), true), "{}");
        assert_eq!(zig_stringify(&json!([]), false), "[]");
        // Floats by Python's repr
        assert_eq!(zig_stringify(&json!({"k": 5.0}), true), r#"{"k":5.0}"#);
        assert_eq!(zig_stringify(&json!([1e16]), true), "[1e+16]");
        assert_eq!(zig_stringify(&json!("é\nx"), true), "\"é\\nx\"");
    }
}
