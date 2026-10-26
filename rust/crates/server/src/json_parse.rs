//! ``json.loads`` on request bytes, with CPython's decoder and scanner
//! error messages.
//!
//! Port of `zig/src/server/json_parse.zig` (336 lines): the parse every
//! inference route reads its body through — the BOM dropped, the UTF-8
//! codec errors worded as Python's ``UnicodeDecodeError``, the scanner's
//! failures in Python's ``msg: line L column C (char N)`` format (the
//! corpus pins them byte for byte), ``-0`` canonicalized to the int ``0``
//! (CPython's own reading), the 4300-digit limit, the 900-deep recursion
//! answer. The values land in serde_json: ints that do not fit its u64/
//! i64 fall to f64 and the literals ``NaN``/``Infinity`` read as null —
//! upstream holds any width int and the non-finite floats whole, the
//! map's named number-edge gap.

use serde_json::{Map, Number, Value};

const MAX_DEPTH: usize = 900; // nesting past this answers as Python's RecursionError does

/// Parse one JSON document out of `bytes` (a leading BOM dropped), the
/// error as Python words it. Upstream: `parse` (json_parse.zig:17).
pub fn parse(bytes: &[u8]) -> Result<Value, String> {
    let text = bytes
        .strip_prefix("\u{feff}".as_bytes())
        .unwrap_or(bytes);
    let offset = bytes.len() - text.len();
    if let Some(message) = utf8_error(text, offset) {
        return Err(message);
    }
    let mut p = Parser { s: text, pos: 0 };
    let value = p.value(p.skip(0), 0)?;
    let end = p.skip(p.pos);
    if end != text.len() {
        return Err(p.fail("Extra data", end));
    }
    Ok(value)
}

/// ``json.loads`` on a Python str: a leading BOM is refused, as for text
/// that was already decoded. Upstream: `parseText` (json_parse.zig:33).
pub fn parse_text(text: &[u8]) -> Result<Value, String> {
    if text.starts_with("\u{feff}".as_bytes()) {
        return Err("Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)".into());
    }
    parse(text)
}

/// CPython's UnicodeDecodeError text for the first bad sequence, or None
/// when `s` decodes. Upstream: `utf8Error` (json_parse.zig:39).
fn utf8_error(s: &[u8], offset: usize) -> Option<String> {
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if c < 0x80 {
            i += 1;
            continue;
        }
        let need: usize = match c {
            0xc2..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf4 => 3,
            _ => return Some(decode_error(c, offset + i, offset + i + 1, "invalid start byte")),
        };
        for k in 1..=need {
            if i + k >= s.len() {
                return Some(decode_error(
                    c,
                    offset + i,
                    offset + s.len(),
                    "unexpected end of data",
                ));
            }
            let b = s[i + k];
            let (lo, hi): (u8, u8) = if k == 1 {
                match c {
                    0xe0 => (0xa0, 0xbf),
                    0xf0 => (0x90, 0xbf),
                    0xf4 => (0x80, 0x8f),
                    _ => (0x80, 0xbf),
                }
            } else {
                (0x80, 0xbf)
            };
            if b < lo || b > hi {
                return Some(decode_error(
                    c,
                    offset + i,
                    offset + i + k,
                    "invalid continuation byte",
                ));
            }
        }
        i += need + 1;
    }
    None
}

/// Upstream: `decodeError` (json_parse.zig:66).
fn decode_error(byte: u8, start: usize, end: usize, reason: &str) -> String {
    if end - start == 1 {
        format!("'utf-8' codec can't decode byte 0x{byte:02x} in position {start}: {reason}")
    } else {
        format!("'utf-8' codec can't decode bytes in position {start}-{}: {reason}", end - 1)
    }
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip(&self, at: usize) -> usize {
        let mut i = at;
        while i < self.s.len() && matches!(self.s[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        i
    }

    /// Records Python's ``msg: line L column C (char N)`` for a failure at
    /// byte `at`. Upstream: `fail` (json_parse.zig:84).
    fn fail(&self, msg: &str, at: usize) -> String {
        let mut chars = 0;
        let mut line = 1;
        let mut last_newline: Option<usize> = None;
        for &b in &self.s[..at] {
            if b & 0xc0 == 0x80 {
                continue; // a continuation byte is no character of its own
            }
            if b == b'\n' {
                line += 1;
                last_newline = Some(chars);
            }
            chars += 1;
        }
        let column = match last_newline {
            Some(n) => chars - n,
            None => chars + 1,
        };
        format!("{msg}: line {line} column {column} (char {chars})")
    }

    fn syntax(&mut self, msg: &str, at: usize) -> String {
        self.fail(msg, at)
    }

    /// Upstream: `value` (json_parse.zig:106).
    fn value(&mut self, at: usize, depth: usize) -> Result<Value, String> {
        if at >= self.s.len() {
            return Err(self.syntax("Expecting value", at));
        }
        match self.s[at] {
            b'"' => {
                let text = self.string(at + 1)?;
                Ok(Value::String(text))
            }
            b'{' => {
                if depth >= MAX_DEPTH {
                    return Err(self.recursion("object"));
                }
                self.object(at + 1, depth + 1)
            }
            b'[' => {
                if depth >= MAX_DEPTH {
                    return Err(self.recursion("array"));
                }
                self.array(at + 1, depth + 1)
            }
            b'n' => self.word(at, "null", Value::Null),
            b't' => self.word(at, "true", Value::Bool(true)),
            b'f' => self.word(at, "false", Value::Bool(false)),
            // NaN and Infinity parse as Python's json does; the port's
            // Value holds no non-finite number, so they read as null (the
            // map's named gap — nothing pins them)
            b'N' => self.word(at, "NaN", Value::Null),
            b'I' => self.word(at, "Infinity", Value::Null),
            b'-' => {
                if self.s[at..].starts_with(b"-Infinity") {
                    self.pos = at + 9;
                    Ok(Value::Null)
                } else {
                    self.number_at(at)
                }
            }
            _ => self.number_at(at),
        }
    }

    /// Upstream: `recursion` (json_parse.zig:134).
    fn recursion(&mut self, kind: &str) -> String {
        format!("maximum recursion depth exceeded while decoding a JSON {kind} from a unicode string")
    }

    fn word(&mut self, at: usize, text: &str, v: Value) -> Result<Value, String> {
        if !self.s[at..].starts_with(text.as_bytes()) {
            return Err(self.syntax("Expecting value", at));
        }
        self.pos = at + text.len();
        Ok(v)
    }

    /// Upstream: `numberAt` (json_parse.zig:145).
    fn number_at(&mut self, start: usize) -> Result<Value, String> {
        let s = self.s;
        let mut i = start;
        if i < s.len() && s[i] == b'-' {
            i += 1;
        }
        if i < s.len() && (b'1'..=b'9').contains(&s[i]) {
            i += 1;
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
        } else if i < s.len() && s[i] == b'0' {
            i += 1;
        } else {
            return Err(self.syntax("Expecting value", start));
        }
        let mut is_float = false;
        if i + 1 < s.len() && s[i] == b'.' && s[i + 1].is_ascii_digit() {
            is_float = true;
            i += 2;
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i + 1 < s.len() && (s[i] == b'e' || s[i] == b'E') {
            let e_start = i;
            i += 1;
            if i + 1 < s.len() && (s[i] == b'-' || s[i] == b'+') {
                i += 1;
            }
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
            if s[i - 1].is_ascii_digit() {
                is_float = true;
            } else {
                i = e_start;
            }
        }
        self.pos = i;
        let text = std::str::from_utf8(&s[start..i]).unwrap_or_default();
        if is_float {
            let f: f64 = text.parse().unwrap_or(f64::NAN);
            return Ok(Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null));
        }
        // CPython's canonical int: -0 reads as 0, a width past its 4300
        // digit limit the limit's own message (json_parse.zig:174-181)
        let digits = if text.starts_with('-') { text.len() - 1 } else { text.len() };
        if digits > 4300 {
            return Err(format!(
                "Exceeds the limit (4300 digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
            ));
        }
        let canonical = if text == "-0" { "0" } else { text };
        if let Ok(u) = canonical.parse::<u64>() {
            return Ok(Value::Number(Number::from(u)));
        }
        if let Ok(n) = canonical.parse::<i64>() {
            return Ok(Value::Number(Number::from(n)));
        }
        // Past serde's widths: the float (upstream holds any width whole —
        // the map's named number-edge gap)
        let f: f64 = canonical.parse().unwrap_or(f64::NAN);
        Ok(Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null))
    }

    /// Upstream: `string` (json_parse.zig:183) — the escapes, the control
    /// characters, the surrogate pairs; a lone surrogate reads as the
    /// replacement character (upstream wtf-8 encodes it — the port's
    /// String cannot hold one).
    fn string(&mut self, begin: usize) -> Result<String, String> {
        let s = self.s;
        let mut out = String::new();
        let mut i = begin;
        loop {
            let chunk_start = i;
            while i < s.len() && s[i] != b'"' && s[i] != b'\\' && s[i] >= 0x20 {
                i += 1;
            }
            if i >= s.len() {
                return Err(self.syntax("Unterminated string starting at", begin - 1));
            }
            if s[i] < 0x20 {
                return Err(self.syntax("Invalid control character at", i));
            }
            if s[i] == b'"' {
                if out.is_empty() {
                    self.pos = i + 1;
                    // No escapes: the input slice itself
                    return Ok(String::from_utf8_lossy(&s[chunk_start..i]).into_owned());
                }
                out.push_str(&String::from_utf8_lossy(&s[chunk_start..i]));
                self.pos = i + 1;
                return Ok(out);
            }
            out.push_str(&String::from_utf8_lossy(&s[chunk_start..i]));
            i += 1; // the backslash
            if i >= s.len() {
                return Err(self.syntax("Unterminated string starting at", begin - 1));
            }
            let c = s[i];
            if c != b'u' {
                let mapped: u8 = match c {
                    b'"' => b'"',
                    b'\\' => b'\\',
                    b'/' => b'/',
                    b'b' => 8,
                    b'f' => 12,
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    _ => return Err(self.syntax("Invalid \\escape", i - 1)),
                };
                out.push(mapped as char);
                i += 1;
                continue;
            }
            let u_at = i;
            i += 1;
            if i + 4 >= s.len() {
                return Err(self.syntax("Invalid \\uXXXX escape", u_at));
            }
            let mut code: u32 = match hex4(&s[i..i + 4]) {
                Some(c) => c as u32,
                None => return Err(self.syntax("Invalid \\uXXXX escape", u_at)),
            };
            i += 4;
            if (0xd800..=0xdbff).contains(&code)
                && i + 6 < s.len()
                && s[i] == b'\\'
                && s[i + 1] == b'u'
            {
                if let Some(low) = hex4(&s[i + 2..i + 6]) {
                    if (0xdc00..=0xdfff).contains(&(low as u32)) {
                        code = 0x10000 + ((code - 0xd800) << 10) + ((low as u32) - 0xdc00);
                        i += 6;
                    }
                }
            }
            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
        }
    }

    /// Upstream: `object` (json_parse.zig:239) — a repeated key keeps its
    /// first place and its last value, as dict() does.
    fn object(&mut self, after_brace: usize, depth: usize) -> Result<Value, String> {
        let mut o = Map::new();
        let mut i = self.skip(after_brace);
        if i < self.s.len() && self.s[i] == b'}' {
            self.pos = i + 1;
            return Ok(Value::Object(o));
        }
        loop {
            if i >= self.s.len() || self.s[i] != b'"' {
                return Err(self.syntax("Expecting property name enclosed in double quotes", i));
            }
            let key = self.string(i + 1)?;
            i = self.skip(self.pos);
            if i >= self.s.len() || self.s[i] != b':' {
                return Err(self.syntax("Expecting ':' delimiter", i));
            }
            i = self.skip(i + 1);
            let v = self.value(i, depth)?;
            o.insert(key, v);
            i = self.skip(self.pos);
            if i < self.s.len() && self.s[i] == b'}' {
                self.pos = i + 1;
                return Ok(Value::Object(o));
            }
            if i >= self.s.len() || self.s[i] != b',' {
                return Err(self.syntax("Expecting ',' delimiter", i));
            }
            let comma = i;
            i = self.skip(i + 1);
            if i < self.s.len() && self.s[i] == b'}' {
                return Err(self.syntax("Illegal trailing comma before end of object", comma));
            }
        }
    }

    /// Upstream: `array` (json_parse.zig:265).
    fn array(&mut self, after_bracket: usize, depth: usize) -> Result<Value, String> {
        let mut out = Vec::new();
        let mut i = self.skip(after_bracket);
        if i < self.s.len() && self.s[i] == b']' {
            self.pos = i + 1;
            return Ok(Value::Array(out));
        }
        loop {
            let v = self.value(i, depth)?;
            out.push(v);
            i = self.skip(self.pos);
            if i < self.s.len() && self.s[i] == b']' {
                self.pos = i + 1;
                return Ok(Value::Array(out));
            }
            if i >= self.s.len() || self.s[i] != b',' {
                return Err(self.syntax("Expecting ',' delimiter", i));
            }
            let comma = i;
            i = self.skip(i + 1);
            if i < self.s.len() && self.s[i] == b']' {
                return Err(self.syntax("Illegal trailing comma before end of array", comma));
            }
        }
    }
}

fn hex4(s: &[u8]) -> Option<u16> {
    let text = std::str::from_utf8(s).ok()?;
    u16::from_str_radix(text, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus's pinned messages (zig/tests/server/golden/errors/), byte
    /// for byte, and upstream's own parser tests (json_parse.zig:320).
    #[test]
    fn the_corpus_messages() {
        // not-json: {nope
        assert_eq!(
            parse(b"{nope").unwrap_err(),
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        );
        // trailing-comma: the corpus's spaced body (char 7), and
        // upstream's own unspaced table case (char 6)
        assert_eq!(
            parse(b"{\"a\": 1,}").unwrap_err(),
            "Illegal trailing comma before end of object: line 1 column 8 (char 7)"
        );
        assert_eq!(
            parse(b"{\"a\":1,}").unwrap_err(),
            "Illegal trailing comma before end of object: line 1 column 7 (char 6)"
        );
        // bad-utf8
        assert_eq!(
            parse(b"{\"messages\": \"\xff\"}").unwrap_err(),
            "'utf-8' codec can't decode byte 0xff in position 14: invalid start byte"
        );
        // The upstream test table (json_parse.zig:323-325)
        assert_eq!(
            parse(b"").unwrap_err(),
            "Expecting value: line 1 column 1 (char 0)"
        );
        assert_eq!(
            parse(b"[1,\n 2,\n x]").unwrap_err(),
            "Expecting value: line 3 column 2 (char 9)"
        );
    }

    /// The types and the canonical numbers.
    #[test]
    fn the_values() {
        assert_eq!(parse(b"{\"a\": [1, -0, true, null]}").unwrap(), serde_json::json!({"a": [1, 0, true, null]}));
        assert_eq!(parse(b"-0").unwrap(), serde_json::json!(0));
        assert_eq!(parse(b"2.5").unwrap(), serde_json::json!(2.5));
        // A repeated key keeps its first place and its last value
        let v = parse(b"{\"a\": 1, \"a\": 2}").unwrap();
        let o = v.as_object().unwrap();
        assert_eq!(o.len(), 1);
        assert_eq!(o.get("a"), Some(&serde_json::json!(2)));
        // Extra data after the document
        assert_eq!(
            parse(b"1 2").unwrap_err(),
            "Extra data: line 1 column 3 (char 2)"
        );
        // The escapes and the pairs
        assert_eq!(
            parse(b"\"a\\n\\ud83d\\ude00\\\"b\"").unwrap(),
            serde_json::json!("a\n\u{1f600}\"b")
        );
        // Unterminated and control characters
        assert_eq!(
            parse(b"{\"a\": \"x").unwrap_err(),
            "Unterminated string starting at: line 1 column 7 (char 6)"
        );
        assert_eq!(
            parse(b"\"a\nb\"").unwrap_err(),
            "Invalid control character at: line 1 column 3 (char 2)"
        );
        // The BOM drops
        assert_eq!(parse(b"\xef\xbb\xbf{}").unwrap(), serde_json::json!({}));
        assert_eq!(
            parse_text(b"\xef\xbb\xbf{}").unwrap_err(),
            "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)"
        );
    }
}
