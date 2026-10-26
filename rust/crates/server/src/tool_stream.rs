//! Tool calls while they are written: XML calls as argument deltas, and
//! the one-call policy's text filter.
//!
//! Port of `zig/src/server/tool_stream.zig` (404 lines): the Streamer that
//! turns a growing reply into tool_call deltas (its XML argument fragments
//! matching what the finished parser produces), the TextFilter that hides
//! call markup from streamed prose, and the `parallel_tool_calls: false`
//! policy built on it.

use crate::reply_text::{after_chars, char_count, find_tag, is_space_byte, py_lstrip, py_rstrip, py_strip};
use crate::tool_params::{self, Schemas};
use crate::tool_parse::{wrap_calls, zig_stringify, CLOSER, OPENER};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// `\s*<TAG=([^>\n]+)>\n?` at the start of `rest`: the name and the bytes
/// matched. Upstream: `opening` (tool_stream.zig:20).
fn opening<'a>(rest: &'a str, tag: &str) -> Option<(&'a str, usize, bool)> {
    let b = rest.as_bytes();
    let mut i = 0;
    while i < b.len() && is_space_byte(b[i]) {
        i += 1;
    }
    if !rest[i..].starts_with(tag) {
        return None;
    }
    i += tag.len();
    let start = i;
    while i < b.len() && b[i] != b'>' && b[i] != b'\n' {
        i += 1;
    }
    if i == start || i >= b.len() || b[i] != b'>' {
        return None;
    }
    let name = &rest[start..i];
    i += 1;
    let newline = i < b.len() && b[i] == b'\n';
    Some((name, i + usize::from(newline), newline))
}

/// `\s*TEXT` at the start of `rest`: bytes matched. Upstream: `closing`
/// (tool_stream.zig:31).
fn closing(rest: &str, text: &str) -> Option<usize> {
    let b = rest.as_bytes();
    let mut i = 0;
    while i < b.len() && is_space_byte(b[i]) {
        i += 1;
    }
    if rest[i..].starts_with(text) {
        Some(i + text.len())
    } else {
        None
    }
}

/// Bytes of `rest` that cannot be the start of a `tail`-byte closer (Python
/// slices characters). Upstream: `trimTail` (tool_stream.zig:195).
fn trim_tail(rest: &str, tail: usize) -> usize {
    let chars = char_count(rest);
    if chars <= tail {
        return 0;
    }
    rest.len() - after_chars(rest, chars - tail).len()
}

/// The schema types whose values stream whole: everything but strings.
/// Upstream: `typedSchema` (tool_stream.zig:199).
fn typed_schema(schema: &Value) -> bool {
    matches!(
        schema.get("type").and_then(Value::as_str),
        Some("array" | "object" | "boolean" | "integer" | "number" | "null")
    )
}

/// The stream state. Upstream: `Streamer`'s `state` enum
/// (tool_stream.zig:34).
#[derive(Clone, Copy, PartialEq)]
enum State {
    Outside,
    Call,
    Function,
    Value,
    Closing,
    Abort,
}

/// Streams Qwen's XML calls as JSON argument deltas matching the final
/// parser. Upstream: `Streamer` (tool_stream.zig:37).
pub struct Streamer {
    schemas: Schemas,
    known: HashMap<String, String>,
    current: String,
    value_schema: Value,
    typed: bool,
    buffer: String,
    pos: usize,
    state: State,
    index: i64,
    args_open: bool,
    lead: bool,
    held: String,
    streamed: bool,
    searched: usize,
}

impl Streamer {
    /// Upstream: `init` (tool_stream.zig:38) - the offered tools' schemas
    /// and their names, lowercased to the spelling the model writes.
    pub fn new(tools: &[Value]) -> Streamer {
        let mut known: HashMap<String, String> = HashMap::new();
        for tool in tools {
            let Some(obj) = tool.as_object() else { continue };
            let spec = match obj.get("function") {
                Some(Value::Object(f)) => f,
                _ => obj,
            };
            let name = match spec.get("name") {
                Some(n) if crate::pyrepr::truthy(n) => crate::tool_specs::py_str(n),
                _ => String::new(),
            };
            if name.is_empty() {
                continue;
            }
            known.insert(name.to_ascii_lowercase(), name);
        }
        Streamer {
            schemas: tool_params::schemas(tools),
            known,
            current: String::new(),
            value_schema: Value::Object(Map::new()),
            typed: false,
            buffer: String::new(),
            pos: 0,
            state: State::Outside,
            index: -1,
            args_open: false,
            lead: false,
            held: String::new(),
            streamed: false,
            searched: 0,
        }
    }

    /// Whether any delta went out: the parse cannot take them back.
    /// Upstream: the `streamed` field (tool_stream.zig:48).
    pub fn streamed(&self) -> bool {
        self.streamed
    }

    /// The deltas the reply's `text` (its whole answer so far) adds.
    /// Upstream: `feed` (tool_stream.zig:70).
    pub fn feed(&mut self, text: &str, out: &mut Vec<Value>) {
        while self.state != State::Abort {
            let rest = &text[self.pos.min(text.len())..];
            match self.state {
                State::Outside => {
                    // The answer is only ever extended, so prose already
                    // searched holds no opener (tool_stream.zig:65)
                    if text.len() < self.searched {
                        self.searched = self.pos;
                    }
                    let from = (self.searched.saturating_sub(OPENER.len() - 1))
                        .max(self.pos)
                        .min(text.len());
                    let Some(j) = find_tag(text, from, OPENER) else {
                        self.searched = text.len();
                        return;
                    };
                    self.pos = j + OPENER.len();
                    self.state = State::Call;
                }
                State::Call => {
                    let Some((raw, m_len, _)) = opening(rest, "<function=") else {
                        // A reply that is not writing XML: after 256
                        // characters, or at any prose lead, the search ends
                        let trimmed = py_strip(rest);
                        if char_count(rest) > 256
                            || (!trimmed.is_empty() && trimmed.as_bytes()[0] != b'<')
                        {
                            self.state = State::Abort;
                        }
                        return;
                    };
                    let lower = py_strip(raw).to_ascii_lowercase();
                    let Some(name) = self.known.get(&lower).cloned() else {
                        // A function the request did not offer
                        self.state = State::Abort;
                        return;
                    };
                    self.current = name.clone();
                    self.index += 1;
                    self.streamed = true;
                    let head = json!({
                        "index": self.index,
                        "id": crate::ids::make("call_", 24),
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    });
                    out.push(wrap_calls(head));
                    self.args_open = false;
                    self.pos += m_len;
                    self.state = State::Function;
                }
                State::Function => {
                    if let Some((raw, m_len, newline)) = opening(rest, "<parameter=") {
                        let key = py_strip(raw);
                        self.value_schema = tool_params::schema_of(&self.schemas, &self.current, key);
                        self.typed = typed_schema(&self.value_schema);
                        self.buffer.clear();
                        let quoted = serde_json::to_string(key).unwrap_or_default();
                        let fragment = format!(
                            "{}{quoted}{}",
                            if self.args_open { "," } else { "{" },
                            if self.typed { ":" } else { ":\"" },
                        );
                        out.push(self.args(&fragment));
                        self.args_open = true;
                        self.pos += m_len;
                        self.state = State::Value;
                        self.lead = !newline;
                        self.held = String::new();
                        continue;
                    }
                    if let Some(n) = closing(rest, "</function>") {
                        out.push(self.args(if self.args_open { "}" } else { "{}" }));
                        self.pos += n;
                        self.state = State::Closing;
                        continue;
                    }
                    return;
                }
                State::Value => {
                    let tail = "</parameter>";
                    let end = find_tag(rest, 0, tail);
                    let chunk = match end {
                        Some(e) => &rest[..e],
                        None => &rest[..trim_tail(rest, tail.len())],
                    };
                    self.pos += chunk.len();
                    let mut piece = format!("{}{}", self.held, chunk);
                    if self.lead && !piece.is_empty() {
                        if piece.as_bytes()[0] == b'\n' {
                            piece = piece[1..].to_string();
                        }
                        self.lead = false;
                    }
                    let keep: &str = if end.is_some() {
                        // the framing newline
                        self.held = String::new();
                        if piece.ends_with('\n') {
                            &piece[..piece.len() - 1]
                        } else {
                            &piece
                        }
                    } else {
                        // its last newline may be the framing one
                        let k = py_rstrip(&piece);
                        self.held = piece[k.len()..].to_string();
                        k
                    };
                    if self.typed {
                        self.buffer.push_str(keep);
                    } else if !keep.is_empty() {
                        out.push(self.args(&esc(keep)));
                    }
                    if end.is_none() {
                        return;
                    }
                    if self.typed {
                        let v = tool_params::decode(&self.buffer, &self.value_schema, true);
                        out.push(self.args(&zig_stringify(&v, false)));
                    } else {
                        out.push(self.args("\""));
                    }
                    self.pos += tail.len();
                    self.state = State::Function;
                }
                State::Closing => {
                    let Some(n) = closing(rest, CLOSER) else { return };
                    self.pos += n;
                    self.state = State::Outside;
                }
                State::Abort => return,
            }
        }
    }

    /// `{"tool_calls": [{"index", "function": {"arguments"}}]}`: one
    /// argument fragment. Upstream: `args` (tool_stream.zig:56).
    fn args(&self, fragment: &str) -> Value {
        wrap_calls(json!({
            "index": self.index,
            "function": {"arguments": fragment},
        }))
    }
}

/// The JSON string quoting without its outer quotes. Upstream: `esc`
/// (tool_stream.zig:64).
fn esc(text: &str) -> String {
    let quoted = serde_json::to_string(text).unwrap_or_default();
    if quoted.len() >= 2 {
        quoted[1..quoted.len() - 1].to_string()
    } else {
        quoted
    }
}

/// Python's identifier shape, widened: letters, digits, `.`, `-` and any
/// non-ASCII. Upstream: `nameLike` (tool_stream.zig:200).
fn name_like(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || !(b[0].is_ascii_alphabetic() || b[0] == b'_') {
        return false;
    }
    b[1..]
        .iter()
        .all(|&ch| ch.is_ascii_alphanumeric() || matches!(ch, b'_' | b'.' | b'-') || ch >= 0x80)
}

/// `(?:NAME:)?tool_call`, any case. Upstream: `callTag` (tool_stream.zig:272).
fn call_tag(name: &str) -> bool {
    if name.eq_ignore_ascii_case("tool_call") {
        return true;
    }
    let Some(colon) = name.find(':') else { return false };
    name_like(&name[..colon]) && name[colon + 1..].eq_ignore_ascii_case("tool_call")
}

/// Hides tool envelopes from streamed prose, holding only a partial tag
/// between deltas. Upstream: `TextFilter` (tool_stream.zig:207).
#[derive(Clone, Default)]
pub struct TextFilter {
    pending: String,
    hidden: Option<String>,
}

impl TextFilter {
    /// Upstream: `feed` (tool_stream.zig:211) - the close searched case by
    /// case, the opener held back when it may still come.
    pub fn feed(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let mut out = String::new();
        while !self.pending.is_empty() {
            let p = self.pending.clone();
            if let Some(name) = self.hidden.clone() {
                let close = format!("</{name}>");
                match find_ci(&p, &close) {
                    Some(end) => {
                        self.pending = p[end + close.len()..].to_string();
                        self.hidden = None;
                        continue;
                    }
                    None => {
                        // Hold back what may be the start of the closer
                        let keep = char_count(&close) - 1;
                        let chars = char_count(&p);
                        let tail = if chars > keep {
                            after_chars(&p, chars - keep)
                        } else {
                            &p[..]
                        };
                        self.pending = tail.to_string();
                        break;
                    }
                }
            }
            let Some(start) = p.find('<') else {
                out.push_str(&p);
                self.pending.clear();
                break;
            };
            out.push_str(&p[..start]);
            let rest = &p[start..];
            let Some(end) = rest.find('>') else {
                // The tag may still be writing; a shape that cannot be a
                // call tag emits its `<` and moves on
                let name = &rest[1..];
                let colon = name.find(':');
                let possible = name.is_empty()
                    || name_like(name)
                    || (colon.is_some()
                        && name_like(&name[..colon.unwrap()])
                        && "tool_call"
                            .to_ascii_lowercase()
                            .starts_with(&name[colon.unwrap() + 1..].to_ascii_lowercase()));
                if possible {
                    self.pending = rest.to_string();
                    break;
                }
                out.push('<');
                self.pending = rest[1..].to_string();
                continue;
            };
            let name = &rest[1..end];
            if call_tag(name) {
                self.hidden = Some(name.to_ascii_lowercase());
            } else {
                out.push_str(&rest[..end + 1]);
            }
            self.pending = rest[end + 1..].to_string();
        }
        out
    }

    /// What the filter held back once the reply ends: nothing when a block
    /// stayed hidden. Upstream: `finish` (tool_stream.zig:264).
    pub fn finish(&mut self) -> String {
        let tail = if self.hidden.is_some() {
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        };
        if self.hidden.is_some() {
            self.pending.clear();
        }
        tail
    }
}

/// Upstream: `findCI` (tool_parse.zig:30) - ASCII case-insensitive from 0.
fn find_ci(hay: &str, needle: &str) -> Option<usize> {
    hay.to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

/// `parallel_tool_calls: false` keeps one call and hides call markup from
/// streamed prose. Upstream: `Policy` (tool_stream.zig:279).
#[derive(Clone)]
pub struct Policy {
    /// The one-call reply: parse takes the first call only.
    pub single: bool,
    filter: TextFilter,
}

impl Policy {
    pub fn new(single: bool) -> Policy {
        Policy {
            single,
            filter: TextFilter::default(),
        }
    }

    /// Upstream: `maxCalls` (tool_stream.zig:283).
    pub fn max_calls(&self) -> Option<usize> {
        if self.single {
            Some(1)
        } else {
            None
        }
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter
    /// content), or passes it through. Upstream: `text`
    /// (tool_stream.zig:288).
    pub fn text(&mut self, s: &str) -> String {
        if self.single {
            self.filter.feed(s)
        } else {
            s.to_string()
        }
    }

    /// Upstream: `flush` (tool_stream.zig:292).
    pub fn flush(&mut self) -> String {
        if self.single {
            self.filter.finish()
        } else {
            String::new()
        }
    }

    /// A whole string through the fresh filter. Upstream: `content`
    /// (tool_stream.zig:296).
    pub fn content(&self, s: &str) -> String {
        if !self.single {
            return s.to_string();
        }
        let mut f = TextFilter::default();
        let head = f.feed(s);
        let tail = f.finish();
        format!("{head}{tail}")
    }

    /// `parsed_content`: a parsed reply's content, whole envelopes the
    /// parser left as text kept and the rest filtered, so an unclosed
    /// envelope stays hidden. Upstream: `parsedContent`
    /// (tool_stream.zig:303).
    pub fn parsed_content(&self, s: &str) -> String {
        if !self.single {
            return s.to_string();
        }
        let mut out = String::new();
        let mut cursor: usize = 0;
        while let Some((start, end)) = whole_envelope(s, cursor) {
            out.push_str(&self.content(&s[cursor..start]));
            out.push_str(&s[start..end]);
            cursor = end;
        }
        out.push_str(&self.content(&s[cursor..]));
        out
    }

    /// What the filter held back that the reply keeps as text (a malformed
    /// call): `final` past the prose `sent`. Upstream: `kept`
    /// (tool_stream.zig:316).
    pub fn kept<'a>(&self, final_text: &'a str, sent: &str) -> Option<&'a str> {
        if !self.single {
            return None;
        }
        let shown = py_lstrip(sent); // the parsed content starts stripped
        if !final_text.starts_with(shown) {
            return None;
        }
        let rest = &final_text[shown.len()..];
        if py_strip(rest).is_empty() {
            None
        } else {
            Some(rest)
        }
    }
}

/// The first whole envelope from `from` (`<((?:NAME:)?tool_call)>.*?</>`,
/// any case). Upstream: `wholeEnvelope` (tool_stream.zig:327).
fn whole_envelope(s: &str, from: usize) -> Option<(usize, usize)> {
    let mut at = from;
    while let Some(start) = s[at..].find('<').map(|x| x + at) {
        at = start + 1;
        let close = s[start..].find('>').map(|x| x + start)?;
        if !call_tag(&s[start + 1..close]) {
            continue;
        }
        let closer = format!("</{}>", &s[start + 1..close]);
        let Some(end) = find_ci_from(s, &closer, close + 1) else {
            continue;
        };
        return Some((start, end + closer.len()));
    }
    None
}

/// Upstream: `findCI` (tool_parse.zig:30) - ASCII case-insensitive, from a
/// position.
fn find_ci_from(hay: &str, needle: &str, from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..]
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
        .map(|at| at + from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parse::close_call;

    fn lookup_tools() -> Vec<Value> {
        vec![json!({
            "type": "function",
            "function": {"name": "lookup", "parameters": {"properties": {"n": {"type": "integer"}}}},
        })]
    }

    /// The deltas one streamer sends while the reply grows through `texts`.
    /// Upstream: `streamed` (tool_stream.zig:390).
    fn streamed(tools: &[Value], texts: &[&str]) -> Vec<Value> {
        let mut s = Streamer::new(tools);
        let mut out = Vec::new();
        for t in texts {
            s.feed(t, &mut out);
        }
        out
    }

    /// The arguments a client joins from streamed deltas. Upstream:
    /// `sentArguments` (tool_stream.zig:398).
    fn sent_arguments(deltas: &[Value]) -> String {
        let mut out = String::new();
        for d in deltas {
            for one in d["tool_calls"].as_array().into_iter().flatten() {
                out.push_str(one["function"]["arguments"].as_str().unwrap_or(""));
            }
        }
        out
    }

    /// Upstream: the test at tool_stream.zig:339, assertion for assertion -
    /// the call its end token left open completes as its closing markup
    /// completes it.
    #[test]
    fn a_streamed_call_its_end_token_left_open_completes_as_its_closing_markup_completes_it() {
        let tools = lookup_tools();
        let open = format!(
            "{OPENER}\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n"
        );
        let sent = sent_arguments(&streamed(&tools, &[&open]));
        let close = close_call(&open, &tools);
        let ended = streamed(&tools, &[&open, &format!("{open}{close}")]);
        let marked = streamed(
            &tools,
            &[&open, &format!("{open}</function>\n{CLOSER}")],
        );
        // the arguments sent before the end, closed by closedJson, are the
        // ones the client ends with
        let closed = tool_params::closed_json(&sent).expect("closable");
        assert_eq!(closed, sent_arguments(&ended));
        // and the deltas match, the heads (index 0) apart - their ids are
        // random
        assert_eq!(marked.len(), ended.len());
        for (x, y) in marked[1..].iter().zip(&ended[1..]) {
            assert_eq!(serde_json::to_string(x).unwrap(), serde_json::to_string(y).unwrap());
        }
    }

    /// Upstream: the test at tool_stream.zig:354 - a streamed call cut
    /// inside a value stays open; what it sent does not close.
    #[test]
    fn a_streamed_call_cut_inside_a_value_stays_open() {
        let tools = vec![json!({"type": "function", "function": {"name": "lookup"}})];
        let open = format!(
            "{OPENER}\n<function=lookup>\n<parameter=q>\nfast cars and more"
        );
        assert_eq!(close_call(&open, &tools), "");
        // its opening delta went out and cannot be taken back
        let sent = sent_arguments(&streamed(&tools, &[&open]));
        assert_eq!(sent, r#"{"q":"fast c"#);
        assert!(tool_params::closed_json(&sent).is_none());
    }

    /// Upstream: the test at tool_stream.zig:367 - a one-call reply drops
    /// an unoffered call and still hides an unclosed one.
    #[test]
    fn a_one_call_reply_drops_an_unoffered_call_and_still_hides_an_unclosed_one() {
        let tools = vec![json!({"type": "function", "function": {"name": "lookup"}})];
        let single = Policy::new(true);
        let text = format!(
            "Launching. {OPENER}<function=launch><parameter=when>now</function>{CLOSER}"
        );
        // what attachCalls makes of the reply: the parse, then the policy's
        // content
        let r = crate::tool_parse::parse(&text, &tools, single.max_calls());
        assert!(r.calls.is_none());
        assert_eq!(r.content, "Launching.");
        assert_eq!(single.parsed_content(&r.content), "Launching.");
        assert_eq!(
            single.parsed_content(&format!("Launching. {OPENER}{{\"name\"")),
            "Launching. "
        );
        // a stream that sent the prose before the block ends with the block
        let kept = single.kept(&text, "\nLaunching. ").unwrap();
        assert_eq!(kept, &text["Launching. ".len()..]);
        assert!(single.kept("Launching.", "Launching. ").is_none());
        let parallel = Policy::new(false);
        assert!(parallel.kept(&text, "Launching. ").is_none());
    }

    #[test]
    fn the_filter_hides_the_markup_and_holds_a_partial_tag() {
        let mut f = TextFilter::default();
        // A whole envelope hides, its close releases the tail
        assert_eq!(f.feed(&format!("a {OPENER}call text{CLOSER} b")), "a  b");
        assert_eq!(f.feed(" tail"), " tail");
        // A partial opener stays pending until it cannot be one
        let mut f = TextFilter::default();
        assert_eq!(f.feed("x <to"), "x ");
        assert_eq!(f.feed("ol_call> y"), "");
        assert_eq!(f.finish(), "");
        // A tag that cannot be a call passes through
        let mut f = TextFilter::default();
        assert_eq!(f.feed("a <b> c"), "a <b> c");
        assert_eq!(f.finish(), "");
        let single = Policy::new(true);
        assert_eq!(single.max_calls(), Some(1));
    }
}
