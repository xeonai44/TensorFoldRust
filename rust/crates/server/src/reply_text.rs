//! Reply text as it streams: think blocks, tool markup held back, stop strings.
//!
//! Port of `zig/src/server/reply_text.zig`. The think markers here are upstream's
//! own bytes (reply_text.zig:7): Qwen's prompt opens the think block, so the
//! marker's `open` is empty and `close` is ``; the opener a reply writes
//! itself (Kolibri's template leaves the block to it) is `<think>`.

/// Upstream: reply_text.zig:5  pub const Markers = struct { open, close }.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Markers {
    pub open: &'static str,
    pub close: &'static str,
}

/// Qwen's think block. Upstream: `think_markers` (reply_text.zig:7).
pub const THINK_MARKERS: Markers = Markers { open: "", close: "</think>" };

/// The opener a reply writes itself when its template leaves the think block to
/// it. Upstream: `think_open` (reply_text.zig:9).
pub const THINK_OPEN: &str = "<think>";

/// Gemma 4's thought channel. Upstream: `channel_markers` (reply_text.zig:10).
pub const CHANNEL_MARKERS: Markers = Markers {
    open: "<|channel>thought",
    close: "<channel|>",
};

/// (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's
/// DSML block. Upstream: `calls` (reply_text.zig:12).
pub const CALLS: [(&str, &str); 3] = [
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call>", "<tool_call|>"),
    (
        "<\u{ff5c}DSML\u{ff5c}tool_calls>",
        "</\u{ff5c}DSML\u{ff5c}tool_calls>",
    ),
];

/// `text.find(tag, from)` led by a scan for the tag's first byte.
/// Upstream: `findTag` (reply_text.zig:19).
pub fn find_tag(text: &str, from: usize, tag: &str) -> Option<usize> {
    let (text, tag) = (text.as_bytes(), tag.as_bytes());
    if tag.is_empty() {
        return if from <= text.len() { Some(from) } else { None };
    }
    let mut at = from;
    while at < text.len() {
        let i = text[at..].iter().position(|&b| b == tag[0])? + at;
        if text[i..].starts_with(tag) {
            return Some(i);
        }
        at = i + 1;
    }
    None
}

fn find(text: &str, tag: &str) -> Option<usize> {
    find_tag(text, 0, tag)
}

/// Upstream: `isChannel` (reply_text.zig:34).
pub fn is_channel(m: Markers) -> bool {
    m.close == CHANNEL_MARKERS.close
}

/// How many bytes at the end of `text` could begin `tag`.
/// Upstream: `partialTag` (reply_text.zig:39).
pub fn partial_tag(text: &str, tag: &str) -> usize {
    let (text, tag) = (text.as_bytes(), tag.as_bytes());
    if tag.is_empty() {
        return 0;
    }
    let mut k = (tag.len() - 1).min(text.len());
    while k > 0 {
        if text.ends_with(&tag[..k]) {
            return k;
        }
        k -= 1;
    }
    0
}

fn trim_newlines(s: &str) -> &str {
    s.trim_start_matches('\n')
}

/// (reasoning, answer) of a thinking reply; while it streams, a tail that could
/// begin a marker is held back. Upstream: `splitThinking`
/// (reply_text.zig:53).
pub fn split_thinking(full: &str, finished: bool, m: Markers) -> (String, String) {
    let mut text = full;
    if !m.open.is_empty() && !text.starts_with(m.open) {
        if let Some(start) = find(text, m.open) {
            if start > 0 {
                let mut prefix = &text[..start];
                let hold = partial_tag(prefix, m.open);
                prefix = &prefix[..prefix.len() - hold]; // a stray partial opener
                let (reasoning, answer) = split_thinking(&text[start..], finished, m);
                return (reasoning, format!("{}{}", prefix, answer));
            }
        }
        if !finished {
            let mut held = partial_tag(text, m.open).max(partial_tag(text, m.close));
            for &(opener, _) in CALLS.iter() {
                held = held.max(partial_tag(text, opener));
            }
            return (String::new(), text[..text.len() - held].to_string());
        }
        return (String::new(), text.to_string());
    }
    if !m.open.is_empty() {
        text = trim_newlines(&text[m.open.len()..]);
    } else if text.starts_with(THINK_OPEN) {
        text = trim_newlines(&text[THINK_OPEN.len()..]);
    } else if !finished && THINK_OPEN.starts_with(text) {
        return (String::new(), String::new()); // held while the opener may still come
    }
    if let Some(end) = find(text, m.close) {
        return (
            text[..end].to_string(),
            trim_newlines(&text[end + m.close.len()..]).to_string(),
        );
    }
    let mut call: Option<usize> = None;
    for &(opener, _) in CALLS.iter() {
        if let Some(at) = find(text, opener) {
            call = Some(call.map_or(at, |c: usize| c.min(at)));
        }
    }
    if let Some(at) = call {
        // A call written before the block closes: held while it may still close,
        // the answer if the reply ended.
        if !finished {
            return (text[..at].to_string(), String::new());
        }
        for &(opener, closer) in CALLS.iter() {
            if text[at..].starts_with(opener) && find(&text[at..], closer).is_some() {
                return (text[..at].to_string(), text[at..].to_string());
            }
        }
        return (text.to_string(), String::new());
    }
    let mut held = 0;
    if !finished {
        held = partial_tag(text, m.close);
        for &(opener, _) in CALLS.iter() {
            held = held.max(partial_tag(text, opener));
        }
    }
    (text[..text.len() - held].to_string(), String::new())
}

/// Hides tool-call blocks and partial opening tags while streaming, so calls
/// arrive as deltas instead. Upstream: `hideToolCalls` (reply_text.zig:97).
pub fn hide_tool_calls(text: &str, finished: bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pos: usize = 0;
    loop {
        let mut start: Option<usize> = None;
        let mut which = 0;
        for (i, &(opener, _)) in CALLS.iter().enumerate() {
            if let Some(at) = find_tag(text, pos, opener) {
                if start.is_none_or(|s: usize| at < s) {
                    start = Some(at);
                    which = i;
                }
            }
        }
        let Some(start) = start else {
            let tail = &text[pos..];
            let mut held = 0;
            if !finished {
                for &(opener, _) in CALLS.iter() {
                    held = held.max(partial_tag(tail, opener));
                }
            }
            if pos == 0 {
                return tail[..tail.len() - held].to_string();
            }
            out.push_str(&tail[..tail.len() - held]);
            return out;
        };
        out.push_str(&text[pos..start]);
        let opener = CALLS[which].0;
        let Some(end) = find_tag(text, start + opener.len(), CALLS[which].1) else {
            return out;
        };
        pos = end + CALLS[which].1.len();
    }
}

/// With tools, a stream holds trailing whitespace back until more text comes.
/// Upstream: `heldBack` (reply_text.zig:133).
pub fn held_back(text: &str, tools: bool) -> &str {
    if tools {
        let stripped = py_strip(text);
        let start = stripped.as_ptr() as usize - text.as_ptr() as usize;
        &text[..start + stripped.len()]
    } else {
        text
    }
}

/// `hideToolCalls` of a finished reply; once a call is hidden, the prose's
/// trailing whitespace stays unsent too. Upstream: `finishedProse`
/// (reply_text.zig:138).
pub fn finished_prose(text: &str) -> String {
    let prose = hide_tool_calls(text, true);
    if prose.len() != text.len() {
        let stripped = py_strip(&prose);
        let start = stripped.as_ptr() as usize - prose.as_ptr() as usize;
        prose[..start + stripped.len()].to_string()
    } else {
        prose
    }
}

/// A thinking reply's reasoning tokens: through its close, else all of them.
/// Upstream: `reasoningCount` (reply_text.zig:182).
pub fn reasoning_count(tokens: &[u32], think_end: Option<u32>) -> usize {
    let Some(end) = think_end else { return 0 };
    match tokens.iter().position(|&t| t == end) {
        Some(i) => i + 1,
        None => tokens.len(),
    }
}

/// Python's `s[n:]` with `n` counted in characters.
/// Upstream: `afterChars` (reply_text.zig:188).
pub fn after_chars(s: &str, n: usize) -> &str {
    let mut seen: usize = 0;
    for (i, b) in s.bytes().enumerate() {
        if b & 0xc0 == 0x80 {
            continue;
        }
        if seen == n {
            return &s[i..];
        }
        seen += 1;
    }
    ""
}

/// Upstream: `charCount` (reply_text.zig:199).
pub fn char_count(s: &str) -> usize {
    s.bytes().filter(|&b| b & 0xc0 != 0x80).count()
}

/// Python's `str.strip()` for the whitespace replies hold.
/// Upstream: `pyStrip` (reply_text.zig:206).
pub fn py_strip(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut start: usize = 0;
    let mut end = bytes.len();
    while start < end {
        let n = space_at(s, start);
        if n == 0 {
            break;
        }
        start += n;
    }
    while end > start {
        let mut back: usize = 1;
        while back < 4 && end - back > start && bytes[end - back] & 0xc0 == 0x80 {
            back += 1;
        }
        if space_at(&s[..end], end - back) != back {
            break;
        }
        end -= back;
    }
    &s[start..end]
}

/// `str.lstrip()`. Upstream: `pyLstrip` (reply_text.zig:224).
pub fn py_lstrip(s: &str) -> &str {
    let stripped = py_strip(s);
    let start = stripped.as_ptr() as usize - s.as_ptr() as usize;
    &s[start..]
}

/// `str.rstrip()`. Upstream: `pyRstrip` (reply_text.zig:229).
pub fn py_rstrip(s: &str) -> &str {
    let stripped = py_strip(s);
    if stripped.is_empty() {
        return "";
    }
    let start = stripped.as_ptr() as usize - s.as_ptr() as usize;
    &s[..start + stripped.len()]
}

/// Python's `\s` on one byte. Upstream: `isSpaceByte` (reply_text.zig:236).
pub fn is_space_byte(ch: u8) -> bool {
    ch == b' ' || (0x09..=0x0d).contains(&ch) || (0x1c..=0x1f).contains(&ch)
}

/// The whitespace at `i`: its UTF-8 length when the codepoint is Python's
/// `\s`, else 0. Upstream: `spaceAt` (reply_text.zig:240).
fn space_at(s: &str, i: usize) -> usize {
    let bytes = s.as_bytes();
    let c = bytes[i];
    if c == b' ' || (0x09..=0x0d).contains(&c) || (0x1c..=0x1f).contains(&c) {
        return 1;
    }
    if c < 0x80 {
        return 0;
    }
    let n = match std::str::from_utf8(&bytes[i..]) {
        Ok(rest) => rest.chars().next().map_or(0, |ch| ch.len_utf8()),
        Err(e) => {
            if e.valid_up_to() == 0 {
                return 0;
            }
            let ch = std::str::from_utf8(&bytes[i..i + e.valid_up_to()])
                .ok()
                .and_then(|r| r.chars().next());
            ch.map_or(0, |c| c.len_utf8())
        }
    };
    if i + n > bytes.len() || n == 0 {
        return 0;
    }
    let cp = match std::str::from_utf8(&bytes[i..i + n]).ok().and_then(|r| r.chars().next()) {
        Some(c) => c as u32,
        None => return 0,
    };
    match cp {
        0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000 => n,
        _ => 0,
    }
}

/// Request stop strings: the visible text ends before the first, and a possible
/// start is held while streaming. Upstream: `Stops` (reply_text.zig:254).
pub struct Stops<'a> {
    pub strings: &'a [&'a str],
}

impl<'a> Stops<'a> {
    /// Upstream: `visible` (reply_text.zig:257).
    pub fn visible<'t>(&self, text: &'t str, partial: bool) -> &'t str {
        let mut first: Option<usize> = None;
        for &stop in self.strings {
            if let Some(at) = find(text, stop) {
                first = Some(first.map_or(at, |f: usize| f.min(at)));
            }
        }
        if let Some(at) = first {
            return &text[..at];
        }
        if !partial {
            return text;
        }
        let mut held: usize = 0;
        for &stop in self.strings {
            let mut size: usize = 1;
            while size < stop.len() && size <= text.len() {
                if text.as_bytes().ends_with(&stop.as_bytes()[..size]) {
                    held = held.max(size);
                }
                size += 1;
            }
        }
        &text[..text.len() - held]
    }

    /// The first stop the text holds, ties in request order.
    /// Upstream: `matched` (reply_text.zig:275).
    pub fn matched(&self, text: &str) -> Option<&'a str> {
        let mut best: Option<usize> = None;
        let mut which: &str = "";
        for &stop in self.strings {
            if let Some(at) = find(text, stop) {
                if best.is_none_or(|b: usize| at < b) {
                    best = Some(at);
                    which = stop;
                }
            }
        }
        if best.is_some() {
            Some(which)
        } else {
            None
        }
    }

    /// Tokens the engine's stop check decodes: the longest stop's length plus 8.
    /// Upstream: `tail` (reply_text.zig:287).
    pub fn tail(&self) -> usize {
        let mut longest = 0;
        for &stop in self.strings {
            longest = longest.max(char_count(stop));
        }
        longest + 8
    }
}

/// The trailing tokens that are the model's own end tokens, off the reply.
/// Upstream: `stripTrailing` (reply_text.zig:194).
pub fn strip_trailing<'t>(tokens: &'t [u32], stops: &[u32]) -> &'t [u32] {
    let mut end = tokens.len();
    while end > 0 && stops.contains(&tokens[end - 1]) {
        end -= 1;
    }
    &tokens[..end]
}

/// Decodes only new tokens with the previous chunk as context, so characters
/// split across tokens stay whole. Upstream: `Incremental`
/// (reply_text.zig:296).
#[derive(Default)]
pub struct Incremental {
    tokens: Vec<u32>,
    text: String,
    prefix: usize,
    read: usize,
}

impl Incremental {
    /// Upstream: `extend` (reply_text.zig:302) — `decode` yields the text of a
    /// token slice; only whole new characters append to the stream.
    pub fn extend<F>(&mut self, more: &[u32], decode: F) -> &str
    where
        F: Fn(&[u32]) -> String,
    {
        self.tokens.extend_from_slice(more);
        let before = decode(&self.tokens[self.prefix..self.read]);
        let after = decode(&self.tokens[self.prefix..]);
        let nb = char_count(&before);
        if char_count(&after) > nb && !after.ends_with("\u{fffd}") {
            self.text.push_str(after_chars(&after, nb));
            self.prefix = self.read;
            self.read = self.tokens.len();
        }
        &self.text
    }

    /// A character is still split across tokens, so streaming waits for the
    /// rest. Upstream: `pending` (reply_text.zig:316).
    pub fn pending(&self) -> bool {
        !self.tokens.is_empty() && self.read < self.tokens.len()
    }

    /// Every token the decoder holds (the finished event's reads).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors reply_text.zig:321 "splits and holds".
    #[test]
    fn trailing_end_tokens_strip() {
        // Mirrors reply_text.zig:194's shape: every trailing eos off
        assert_eq!(strip_trailing(&[1, 2, 7, 7], &[7]), &[1, 2]);
        assert_eq!(strip_trailing(&[7, 7], &[7]), &[] as &[u32]);
        assert_eq!(strip_trailing(&[1, 7, 2, 7], &[7]), &[1, 7, 2]);
        assert_eq!(strip_trailing(&[], &[7]), &[] as &[u32]);
        // The stop strings never strip tokens: only the model's own end
        assert_eq!(strip_trailing(&[1, 2], &[7]), &[1, 2]);
    }

    #[test]
    fn splits_and_holds() {
        let s = split_thinking(PLAN_INPUT, true, THINK_MARKERS);
        assert_eq!(s.0, "plan");
        assert_eq!(s.1, "Hi");
        let held = split_thinking("plan</thi", false, THINK_MARKERS);
        assert_eq!(held.0, "plan");
        let hidden = hide_tool_calls(HIDE_INPUT, false);
        assert_eq!(hidden, "a b");
        let stops = Stops { strings: &["END"] };
        assert_eq!(stops.visible("abEN", true), "ab");
        assert_eq!(py_strip(" \u{3000}x\u{a0}\n"), "x");
        assert_eq!(after_chars("h\u{e9}lo", 2), "lo");
    }

    /// Mirrors reply_text.zig:337: a reply that writes its own think opener
    /// keeps the tag out of its reasoning.
    #[test]
    fn a_reply_writing_its_own_opener_keeps_the_tag_out() {
        let reply = REPLY_INPUT;
        let mut sent = String::new();
        for n in 1..=reply.len() {
            let s = split_thinking(&reply[..n], false, THINK_MARKERS);
            assert!(!s.0.contains('<'), "reasoning at {}: {:?}", n, s.0);
            assert!(s.0.starts_with(&sent), "monotonic at {}", n);
            sent = s.0;
        }
        let tail = TAIL_INPUT;
        for text in [reply, tail] {
            let s = split_thinking(text, true, THINK_MARKERS);
            assert_eq!(s.0, "Plan.\n");
            assert_eq!(s.1, "Done.");
        }
    }
}

// The tests' fixed inputs, byte-precise (built from upstream's own test bytes).
#[cfg(test)]
const PLAN_INPUT: &str = "plan</think>\n\nHi";
#[cfg(test)]
const HIDE_INPUT: &str = "a <tool_call>x</tool_call>b<tool";
#[cfg(test)]
const REPLY_INPUT: &str = "<think>\nPlan.\n</think>\n\nDone.";
#[cfg(test)]
const TAIL_INPUT: &str = "Plan.\n</think>\n\nDone.";
