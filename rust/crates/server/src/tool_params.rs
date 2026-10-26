//! Typed tool parameters: XML parameter text decoded by the offered schema
//! (``tool_parameters.py``).
//!
//! Port of `zig/src/server/tool_params.zig` (381 lines): the offered tools'
//! parameter schemas by name, the partial-JSON closer, and the decoder that
//! reads a parameter's text as the schema's type — including Python's own
//! spellings (``None``, ``True``, quoted strings, numbers, lists, tuples and
//! dicts) when the caller allows them.

use crate::pyrepr::{float_repr, truthy};
use crate::reply_text::py_strip;
use serde_json::{Map, Value};

/// Each offered tool's parameter schemas by lowercase name.
///
/// Upstream: `Schemas` (tool_params.zig:10).
pub type Schemas = std::collections::HashMap<String, Map<String, Value>>;

/// Each offered tool's parameter schemas: the ``parameters`` (else
/// ``input_schema``) properties whose values are objects, kept by the tool's
/// lowercased name.
///
/// Upstream: `schemas` (tool_params.zig:12).
pub fn schemas(tools: &[Value]) -> Schemas {
    let mut out = Schemas::new();
    for tool in tools {
        let Some(tool_obj) = tool.as_object() else { continue };
        let function = match tool_obj.get("function") {
            Some(Value::Object(f)) => f,
            _ => tool_obj,
        };
        let mut parameters = Value::Object(Map::new());
        if let Some(p) = function.get("parameters") {
            if truthy(p) {
                parameters = p.clone();
            }
        }
        if !truthy(&parameters) {
            if let Some(p) = function.get("input_schema") {
                if truthy(p) {
                    parameters = p.clone();
                }
            }
        }
        let mut properties = Value::Object(Map::new());
        if let Value::Object(p) = &parameters {
            if let Some(v) = p.get("properties") {
                if truthy(v) {
                    properties = v.clone();
                }
            }
        }
        let mut kept = Map::new();
        if let Value::Object(props) = &properties {
            for (k, v) in props {
                if v.is_object() {
                    kept.insert(k.clone(), v.clone());
                }
            }
        }
        let name = function
            .get("name")
            .map(crate::tool_specs::py_str)
            .unwrap_or_default();
        out.insert(name.to_ascii_lowercase(), kept);
    }
    out
}

/// The schema of ``key`` in tool ``name``, or an empty one.
///
/// Upstream: `schemaOf` (tool_params.zig:40).
pub fn schema_of(s: &Schemas, name: &str, key: &str) -> Value {
    let props = s.get(&name.to_ascii_lowercase());
    let schema = props.and_then(|p| p.get(key));
    match schema {
        Some(v) => v.clone(),
        None => Value::Object(Map::new()),
    }
}

/// The typed kinds a schema's ``type`` can name. Upstream: `Kind`
/// (tool_params.zig:46).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Array,
    Object,
    Boolean,
    Integer,
    Number,
    Null,
}

/// Upstream: `kindOf` (tool_params.zig:48).
fn kind_of(schema: &Value) -> Option<Kind> {
    let t = schema.get("type")?;
    match t.as_str()? {
        "array" => Some(Kind::Array),
        "object" => Some(Kind::Object),
        "boolean" => Some(Kind::Boolean),
        "integer" => Some(Kind::Integer),
        "number" => Some(Kind::Number),
        "null" => Some(Kind::Null),
        _ => None,
    }
}

/// Upstream: `valid` (tool_params.zig:54) — a number counts as an integer
/// only in its integer spellings.
fn valid(kind: Kind, v: &Value) -> bool {
    match kind {
        Kind::Array => v.is_array(),
        Kind::Object => v.is_object(),
        Kind::Boolean => v.is_boolean(),
        Kind::Integer => v.is_i64() || v.is_u64(),
        Kind::Number => v.is_number(),
        Kind::Null => v.is_null(),
    }
}

/// ``json.dumps(v, allow_nan=False)`` succeeds: no NaN or infinity anywhere.
///
/// Upstream: `finite` (tool_params.zig:66).
pub fn finite(v: &Value) -> bool {
    match v {
        Value::Number(n) if n.is_f64() => n.as_f64().is_none_or(f64::is_finite),
        Value::Array(items) => items.iter().all(finite),
        Value::Object(o) => o.values().all(finite),
        _ => true,
    }
}

/// ``closed_json``: the arrays and objects ``text`` left open, closed; None
/// unless it only stops short of them.
///
/// Upstream: `closedJson` (tool_params.zig:80) — strings and their escapes
/// tracked so brackets inside them do not close; a mismatched closer makes
/// the whole text hopeless.
pub fn closed_json(text: &str) -> Option<String> {
    let mut closers: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for &ch in text.as_bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else {
                escaped = ch == b'\\';
                in_string = ch != b'"';
            }
        } else if ch == b'"' {
            in_string = true;
        } else if ch == b'[' || ch == b'{' {
            closers.push(if ch == b'[' { b']' } else { b'}' });
        } else if (ch == b']' || ch == b'}') && closers.pop() != Some(ch) {
            return None;
        }
    }
    if closers.is_empty() || in_string {
        return None;
    }
    closers.reverse();
    let tail = String::from_utf8(closers).ok()?;
    Some(format!("{}{}", crate::reply_text::py_rstrip(text), tail))
}

/// ``decode_parameter``: the schema's type when the text spells one, else the
/// text; ``python`` also reads Python's spelling.
///
/// Upstream: `decode` (tool_params.zig:106) — the text as JSON first, then
/// the closed form for the collective kinds, then (when allowed) the Python
/// literal; each candidate must parse, be finite, and match the kind.
pub fn decode(value: &str, schema: &Value, python: bool) -> Value {
    let Some(kind) = kind_of(schema) else {
        return Value::String(value.to_string());
    };
    let closed = if kind == Kind::Array || kind == Kind::Object {
        closed_json(value)
    } else {
        None
    };
    for candidate in [Some(value), closed.as_deref()] {
        let Some(text) = candidate else { continue };
        if let Ok(parsed) = serde_json::from_str::<Value>(text) {
            if !finite(&parsed) {
                continue;
            }
            return if valid(kind, &parsed) {
                parsed
            } else {
                Value::String(value.to_string())
            };
        }
    }
    if !python {
        return Value::String(value.to_string());
    }
    let Some(parsed) = python_literal(value) else {
        return Value::String(value.to_string());
    };
    if !finite(&parsed) {
        return Value::String(value.to_string());
    }
    if valid(kind, &parsed) {
        parsed
    } else {
        Value::String(value.to_string())
    }
}

/// ``_python_literal``: True, None, quoted strings, numbers, lists, tuples
/// and dicts; None if ``text`` is not one.
///
/// Upstream: `pythonLiteral` (tool_params.zig:125) — the short keywords
/// case-insensitive, everything else the recursive-descent parser below,
/// and the whole text must be consumed.
pub fn python_literal(text: &str) -> Option<Value> {
    let stripped = py_strip(text);
    if stripped.len() <= 5 {
        let lower = stripped.to_ascii_lowercase();
        if lower == "true" {
            return Some(Value::Bool(true));
        }
        if lower == "false" {
            return Some(Value::Bool(false));
        }
        if lower == "none" || lower == "null" {
            return Some(Value::Null);
        }
    }
    let mut p = Literal { s: stripped.as_bytes(), i: 0 };
    let v = p.expr(0)?;
    p.space();
    if p.i == p.s.len() {
        Some(v)
    } else {
        None
    }
}

/// The literal parser: one ``Value`` out of Python's spellings, position by
/// position. Upstream: `Literal` (tool_params.zig:143). Where upstream
/// reports errors this returns None; the parse fails soft and the caller
/// falls back to the text.
struct Literal<'a> {
    s: &'a [u8],
    i: usize,
}

impl Literal<'_> {
    /// Upstream: `space` (tool_params.zig:150) — the four space characters,
    /// not Python's wider strip set.
    fn space(&mut self) {
        while self.i < self.s.len()
            && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.i += 1;
        }
    }

    /// Upstream: `peek` (tool_params.zig:154).
    fn peek(&mut self) -> Option<u8> {
        self.space();
        if self.i < self.s.len() {
            Some(self.s[self.i])
        } else {
            None
        }
    }

    /// Upstream: `expr` (tool_params.zig:159) — the depth cap, the bracket
    /// kinds, the string prefixes, then the keywords and the number.
    fn expr(&mut self, depth: usize) -> Option<Value> {
        if depth > 200 {
            return None;
        }
        let c = self.peek()?;
        match c {
            b'[' => return self.sequence(b']', depth),
            b'(' => return self.sequence(b')', depth),
            b'{' => return self.dict(depth),
            b'\'' | b'"' => return self.strings(),
            b'r' | b'R' | b'u' | b'U'
                if self.i + 1 < self.s.len()
                    && matches!(self.s[self.i + 1], b'\'' | b'"') =>
            {
                return self.strings();
            }
            _ => {}
        }
        if self.s[self.i..].starts_with(b"None") && !self.word_at(self.i + 4) {
            self.i += 4;
            return Some(Value::Null);
        }
        if self.s[self.i..].starts_with(b"True") && !self.word_at(self.i + 4) {
            self.i += 4;
            return Some(Value::Bool(true));
        }
        if self.s[self.i..].starts_with(b"False") && !self.word_at(self.i + 5) {
            self.i += 5;
            return Some(Value::Bool(false));
        }
        self.number()
    }

    /// Upstream: `wordAt` (tool_params.zig:176).
    fn word_at(&self, at: usize) -> bool {
        at < self.s.len()
            && (self.s[at].is_ascii_alphanumeric() || self.s[at] == b'_')
    }

    /// Upstream: `sequence` (tool_params.zig:185) — a list, a tuple with a
    /// trailing comma, or the parenthesized single value itself.
    fn sequence(&mut self, close: u8, depth: usize) -> Option<Value> {
        self.i += 1;
        let mut items: Vec<Value> = Vec::new();
        let mut commas = 0usize;
        loop {
            if self.peek() == Some(close) {
                self.i += 1;
                if close == b')' && items.len() == 1 && commas == 0 {
                    return Some(items.remove(0)); // (x) is x
                }
                return Some(Value::Array(items));
            }
            items.push(self.expr(depth + 1)?);
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    commas += 1;
                }
                Some(c) if c == close => {}
                _ => return None,
            }
        }
    }

    /// Upstream: `dict` (tool_params.zig:204) — keys stringified the JSON way
    /// (bools as true/false, floats by repr), sets rejected.
    fn dict(&mut self, depth: usize) -> Option<Value> {
        self.i += 1;
        let mut o = Map::new();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Some(Value::Object(o));
        }
        loop {
            let key = self.expr(depth + 1)?;
            if self.peek() != Some(b':') {
                return None; // a set is no JSON value
            }
            self.i += 1;
            let v = self.expr(depth + 1)?;
            let name = match key {
                Value::String(s) => s,
                Value::Number(n) => {
                    if n.is_f64() {
                        float_repr(n.as_f64()?)
                    } else {
                        n.to_string()
                    }
                }
                Value::Bool(b) => if b { "true".into() } else { "false".into() },
                Value::Null => "null".into(),
                _ => return None,
            };
            o.insert(name, v);
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                        return Some(Value::Object(o));
                    }
                }
                Some(b'}') => {
                    self.i += 1;
                    return Some(Value::Object(o));
                }
                _ => return None,
            }
        }
    }

    /// One or more adjacent string literals, joined as Python's parser joins
    /// them. Upstream: `strings` (tool_params.zig:243).
    fn strings(&mut self) -> Option<Value> {
        let mut out: Vec<u8> = Vec::new();
        let mut any = false;
        while let Some(c) = self.peek() {
            let mut raw = false;
            if matches!(c, b'r' | b'R' | b'u' | b'U') {
                if self.i + 1 >= self.s.len()
                    || !matches!(self.s[self.i + 1], b'\'' | b'"')
                {
                    break;
                }
                raw = matches!(c, b'r' | b'R');
                self.i += 1;
            } else if !matches!(c, b'\'' | b'"') {
                break;
            }
            self.string(&mut out, raw)?;
            any = true;
        }
        if !any {
            return None;
        }
        Some(Value::String(String::from_utf8(out).ok()?))
    }

    /// Upstream: `string` (tool_params.zig:260) — triple quotes, the raw
    /// prefixes, and every escape Python's parser accepts; a lone
    /// surrogate (which upstream keeps as WTF-8) fails the string here
    /// instead, so the whole literal falls back to text.
    fn string(&mut self, out: &mut Vec<u8>, raw: bool) -> Option<()> {
        let q = self.s[self.i];
        let triple =
            self.i + 2 < self.s.len() && self.s[self.i + 1] == q && self.s[self.i + 2] == q;
        self.i += if triple { 3 } else { 1 };
        loop {
            if self.i >= self.s.len() {
                return None;
            }
            let c = self.s[self.i];
            if c == q
                && (!triple
                    || (self.i + 2 < self.s.len()
                        && self.s[self.i + 1] == q
                        && self.s[self.i + 2] == q))
            {
                self.i += if triple { 3 } else { 1 };
                return Some(());
            }
            if c == b'\n' && !triple {
                return None;
            }
            if c != b'\\' {
                out.push(c);
                self.i += 1;
                continue;
            }
            if self.i + 1 >= self.s.len() {
                return None;
            }
            let e = self.s[self.i + 1];
            if raw {
                out.extend_from_slice(&self.s[self.i..self.i + 2]);
                self.i += 2;
                continue;
            }
            self.i += 2;
            match e {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'r' => out.push(b'\r'),
                b'\\' => out.push(b'\\'),
                b'\'' => out.push(b'\''),
                b'"' => out.push(b'"'),
                b'a' => out.push(7),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'v' => out.push(11),
                b'0'..=b'7' => {
                    let mut v: u32 = (e - b'0') as u32;
                    let mut n = 1;
                    while n < 3 && self.i < self.s.len() && (b'0'..=b'7').contains(&self.s[self.i])
                    {
                        v = v * 8 + (self.s[self.i] - b'0') as u32;
                        self.i += 1;
                        n += 1;
                    }
                    append_code(out, v)?;
                }
                b'x' | b'u' | b'U' => {
                    let n: usize = if e == b'x' { 2 } else if e == b'u' { 4 } else { 8 };
                    if self.i + n > self.s.len() {
                        return None;
                    }
                    let hex = std::str::from_utf8(&self.s[self.i..self.i + n]).ok()?;
                    let v = u32::from_str_radix(hex, 16).ok()?;
                    self.i += n;
                    append_code(out, v)?;
                }
                b'\n' => {} // the line continuation
                _ => {
                    out.push(b'\\');
                    out.push(e);
                }
            }
        }
    }

    /// Upstream: `number` (tool_params.zig:331) — the sign toggling, the
    /// Python int spellings (0x/0o/0b, underscores), the leading-zero rule,
    /// complex rejected, floats by ``float(text)``. Where upstream keeps an
    /// int of any length, a value past 64 bits fails the literal here and
    /// the caller falls back to the text.
    fn number(&mut self) -> Option<Value> {
        let mut negative = false;
        while let Some(c) = self.peek() {
            if c == b'-' {
                negative = !negative;
            } else if c != b'+' {
                break;
            }
            self.i += 1;
        }
        let start = self.i;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            let kept = c.is_ascii_alphanumeric()
                || c == b'_'
                || c == b'.'
                || ((c == b'+' || c == b'-')
                    && self.i > start
                    && matches!(self.s[self.i - 1], b'e' | b'E'));
            if !kept {
                break;
            }
            self.i += 1;
        }
        let text = &self.s[start..self.i];
        if text.is_empty() {
            return None;
        }
        // The other int base: 0x, 0o or 0b, digits with underscores.
        if text.len() > 2 && text[0] == b'0' && matches!(text[1], b'x' | b'X' | b'o' | b'O' | b'b' | b'B') {
            let base: u32 = match text[1].to_ascii_lowercase() {
                b'x' => 16,
                b'o' => 8,
                _ => 2,
            };
            let digits: Vec<u8> = text[2..].iter().filter(|&b| *b != b'_').copied().collect();
            let v = i128::from_str_radix(&String::from_utf8_lossy(&digits), base).ok()?;
            return int_value(negative, v);
        }
        if !text.iter().any(|&b| matches!(b, b'.' | b'e' | b'E')) {
            let digits: Vec<u8> = text.iter().filter(|&&b| b != b'_').copied().collect();
            if digits.iter().any(|b| !b.is_ascii_digit()) {
                return None;
            }
            // Python's leading-zero rule: 007 is not an int literal.
            if digits.len() > 1 && digits[0] == b'0' && digits.iter().any(|&b| b != b'0') {
                return None;
            }
            let digits = String::from_utf8(digits).ok()?;
            let trimmed = digits.trim_start_matches('0');
            if trimmed.is_empty() {
                return Some(Value::Number(0u64.into())); // -0 is 0
            }
            if negative {
                let v: i64 = format!("-{trimmed}").parse().ok()?;
                return Some(Value::Number(v.into()));
            }
            let v: u64 = trimmed.parse().ok()?;
            return Some(Value::Number(v.into()));
        }
        if text.iter().any(|&b| matches!(b, b'j' | b'J')) {
            return None; // complex
        }
        let f = py_float_bytes(text)?;
        if text
            .iter()
            .any(|&b| b"infaINFA".contains(&b))
            && !text[0].is_ascii_digit()
            && text[0] != b'.'
        {
            return None;
        }
        let signed = if negative { -f } else { f };
        Some(Value::Number(serde_json::Number::from_f64(signed)?))
    }
}

/// Upstream: `appendCode` (tool_params.zig:324) — the codepoint as its
/// UTF-8 bytes, past 0x10ffff invalid.
fn append_code(out: &mut Vec<u8>, v: u32) -> Option<()> {
    let c = char::from_u32(v)?;
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
    Some(())
}

/// ``-v`` when negative, as a JSON number. Upstream prints the decimal text
/// (tool_params.zig:351); values past 64 bits fall back to the text here.
fn int_value(negative: bool, v: i128) -> Option<Value> {
    let signed = if negative { -v } else { v };
    if let Ok(i) = i64::try_from(signed) {
        return Some(Value::Number(i.into()));
    }
    if !negative {
        if let Ok(u) = u64::try_from(v) {
            return Some(Value::Number(u.into()));
        }
    }
    None
}

/// ``float(text)``, or None. Upstream: `pyFloat` (fields.zig:137) — the
/// underscore digits, the inf/infinity/nan spellings, the 100-char cap;
/// its strip set is the nine ASCII characters (fields.zig:103), narrower
/// than reply_text's.
pub fn py_float(text: &str) -> Option<f64> {
    py_float_bytes(text.as_bytes())
}

fn py_float_bytes(text: &[u8]) -> Option<f64> {
    // The nine-character strip (fields.zig:103), both ends.
    const STRIP: &[u8] = b" \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f";
    let mut start = 0;
    let mut end = text.len();
    while start < end && STRIP.contains(&text[start]) {
        start += 1;
    }
    while end > start && STRIP.contains(&text[end - 1]) {
        end -= 1;
    }
    let mut sign = 1.0f64;
    let mut s: &[u8] = &text[start..end];
    if !s.is_empty() && (s[0] == b'-' || s[0] == b'+') {
        if s[0] == b'-' {
            sign = -1.0;
        }
        s = &s[1..];
    }
    if s.eq_ignore_ascii_case(b"inf") || s.eq_ignore_ascii_case(b"infinity") {
        return Some(sign * f64::INFINITY);
    }
    if s.eq_ignore_ascii_case(b"nan") {
        return Some(sign * f64::NAN);
    }
    if s.len() > 100 {
        return None;
    }
    let e_at = s
        .iter()
        .position(|&b| b == b'e' || b == b'E')
        .unwrap_or(s.len());
    let mantissa = &s[..e_at];
    // The parts around the dot; no dot means an empty fraction (fields.zig:153).
    let (whole, frac) = match mantissa.iter().position(|&b| b == b'.') {
        Some(d) => (&mantissa[..d], &mantissa[d + 1..]),
        None => (mantissa, &mantissa[mantissa.len()..]),
    };
    if whole.is_empty() && frac.is_empty() {
        return None;
    }
    let mut list = String::new();
    if !whole.is_empty() && !digits_into(whole, &mut list) {
        return None;
    }
    if list.is_empty() {
        list.push('0');
    }
    list.push('.');
    if !frac.is_empty() && !digits_into(frac, &mut list) {
        return None;
    }
    list.push('0');
    if e_at < s.len() {
        let mut e = &s[e_at + 1..];
        list.push('e');
        if !e.is_empty() && (e[0] == b'-' || e[0] == b'+') {
            list.push(e[0] as char);
            e = &e[1..];
        }
        if !digits_into(e, &mut list) {
            return None;
        }
    }
    list.parse::<f64>().ok().map(|f| sign * f)
}

/// Digits with single underscores between them, as Python's int() and
/// float() accept. Upstream: `digitsInto` (fields.zig:172).
fn digits_into(s: &[u8], out: &mut String) -> bool {
    if s.is_empty() || !s[0].is_ascii_digit() || !s[s.len() - 1].is_ascii_digit() {
        return false;
    }
    for i in 0..s.len() {
        let ch = s[i];
        if ch == b'_' {
            if s[i - 1] == b'_' {
                return false;
            }
            continue;
        }
        if !ch.is_ascii_digit() {
            return false;
        }
        out.push(ch as char);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The upstream test (tool_params.zig:368), assertion for assertion.
    #[test]
    fn decode_parameters() {
        let int_schema = json!({"type": "integer"});
        assert_eq!(decode("5", &int_schema, true), json!(5));
        assert_eq!(decode("5.5", &int_schema, true), json!("5.5"));
        let arr = json!({"type": "array"});
        let closed = decode("[1, [2", &arr, true);
        assert_eq!(closed.as_array().map(Vec::len), Some(2));
        let py = decode("['a', None, True]", &arr, true);
        assert_eq!(py, json!(["a", null, true]));
        assert_eq!(
            decode(" None ", &json!({"type": "null"}), true),
            Value::Null
        );
    }

    #[test]
    fn schemas_keep_object_properties_by_lowercase_name() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "Get_Weather",
                "parameters": {"properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "bad": "not-an-object",
                }},
            },
        })];
        let s = schemas(&tools);
        let city = schema_of(&s, "get_weather", "city");
        assert_eq!(city, json!({"type": "string"}));
        let days = schema_of(&s, "GET_WEATHER", "days");
        assert_eq!(days, json!({"type": "integer"}));
        // A property whose value is not an object is not kept
        assert_eq!(schema_of(&s, "get_weather", "bad"), json!({}));
        assert_eq!(schema_of(&s, "get_weather", "missing"), json!({}));
        assert_eq!(schema_of(&s, "no_such_tool", "city"), json!({}));
        // The anthropic spelling of the parameters block also feeds it
        let anthropic = vec![json!({
            "name": "Lookup",
            "input_schema": {"properties": {"tz": {"type": "string"}}},
        })];
        let s = schemas(&anthropic);
        assert_eq!(schema_of(&s, "lookup", "tz"), json!({"type": "string"}));
    }

    #[test]
    fn closed_json_only_closes_what_stopped_short() {
        assert_eq!(closed_json("[1, [2").as_deref(), Some("[1, [2]]"));
        assert_eq!(closed_json("{").as_deref(), Some("{}"));
        assert_eq!(closed_json("[]"), None); // nothing open
        assert_eq!(closed_json("\"abc"), None); // ends inside a string
        assert_eq!(closed_json("[1}"), None); // the wrong closer
        // Brackets inside strings do not close
        assert_eq!(closed_json("{\"a\": \"][\"").as_deref(), Some("{\"a\": \"][\"}"));
        // Escaped quotes do not end the string, so this one is stuck open
        assert_eq!(closed_json("[\"a\\\"]"), None);
        assert_eq!(
            closed_json("[\"a\\\\\"]"),
            None // balanced: nothing left to close
        );
    }

    #[test]
    fn decode_without_python_needs_the_json_spelling() {
        let arr = json!({"type": "array"});
        assert_eq!(decode("[1, 2]", &arr, false), json!([1, 2]));
        assert_eq!(
            decode("['a', None]", &arr, false),
            json!("['a', None]")
        );
        // The partial array still closes
        assert_eq!(decode("[1, [2", &arr, false), json!([1, [2]]));
    }

    #[test]
    fn decode_validates_against_the_kind() {
        assert_eq!(
            decode("true", &json!({"type": "boolean"}), true),
            json!(true)
        );
        assert_eq!(
            decode("yes", &json!({"type": "boolean"}), true),
            json!("yes")
        );
        assert_eq!(
            decode("1.5", &json!({"type": "number"}), true),
            json!(1.5)
        );
        assert_eq!(
            decode("2", &json!({"type": "number"}), true),
            json!(2)
        );
        // An int text for a float kind keeps its int self, as Python's
        // json.loads does
        assert_eq!(decode("{}", &json!({"type": "object"}), true), json!({}));
        assert_eq!(decode("5", &Value::Null, true), json!("5"));
    }

    #[test]
    fn python_literals_of_every_shape() {
        assert_eq!(python_literal(" True "), Some(json!(true)));
        assert_eq!(python_literal("FALSE"), Some(json!(false)));
        assert_eq!(python_literal(" none "), Some(Value::Null));
        assert_eq!(python_literal("null"), Some(Value::Null));
        assert_eq!(python_literal("5"), Some(json!(5)));
        assert_eq!(python_literal("-0"), Some(json!(0)));
        assert_eq!(python_literal("1_000"), Some(json!(1000)));
        assert_eq!(python_literal("0x1F"), Some(json!(31)));
        assert_eq!(python_literal("-0o17"), Some(json!(-15)));
        assert_eq!(python_literal("0b101"), Some(json!(5)));
        assert_eq!(python_literal("1.5e3"), Some(json!(1500.0)));
        // Python rejects these
        assert_eq!(python_literal("007"), None);
        assert_eq!(python_literal("5j"), None);
        assert_eq!(python_literal("inf"), None);
        assert_eq!(python_literal("r'x"), None);
        assert_eq!(python_literal("'a\nb'"), None);
        assert_eq!(python_literal("Truer"), None);
        assert_eq!(python_literal("5 5"), None);
    }

    #[test]
    fn python_strings_and_their_escapes() {
        assert_eq!(python_literal("'a' \"b\""), Some(json!("ab"))); // adjacent joined
        assert_eq!(python_literal("'''a\nb'''"), Some(json!("a\nb")));
        assert_eq!(python_literal("r'\\n'"), Some(json!("\\n")));
        assert_eq!(python_literal("u'\\u00e9'"), Some(json!("é")));
        assert_eq!(python_literal("'\\x41'"), Some(json!("A")));
        assert_eq!(python_literal("'\\101'"), Some(json!("A"))); // octal
        assert_eq!(python_literal("'a\\\nb'"), Some(json!("ab"))); // line continuation
        assert_eq!(python_literal("'\\q'"), Some(json!("\\q"))); // kept as written
        assert_eq!(python_literal("''"), Some(json!("")));
    }

    #[test]
    fn python_sequences_and_dicts() {
        assert_eq!(python_literal("(5)"), Some(json!(5))); // (x) is x
        assert_eq!(python_literal("(5,)"), Some(json!([5])));
        assert_eq!(python_literal("[1, 2,]"), Some(json!([1, 2])));
        assert_eq!(python_literal("[1, [2]]"), Some(json!([1, [2]])));
        assert_eq!(
            python_literal("{'a': 1, 'b': True}"),
            Some(json!({"a": 1, "b": true}))
        );
        assert_eq!(
            python_literal("{'k': 2.5, 'j': 1e16}"),
            Some(json!({"k": 2.5, "j": 1e16}))
        );
        assert_eq!(python_literal("{1: 'x'}"), Some(json!({"1": "x"})));
        assert_eq!(python_literal("{'a': 1,}"), Some(json!({"a": 1})));
        // A set is no JSON value
        assert_eq!(python_literal("{1, 2}"), None);
        // The depth cap: 201 nested brackets parse, the 202nd does not
        let deep = format!("{}{}", "[".repeat(201), "]".repeat(201));
        assert_eq!(
            python_literal(&deep).map(|v| v.to_string()),
            Some(deep.clone())
        );
        let over = format!("{}{}", "[".repeat(202), "]".repeat(202));
        assert_eq!(python_literal(&over), None);
    }

    #[test]
    fn float_text_as_python_reads_it() {
        assert_eq!(py_float("5"), Some(5.0));
        assert_eq!(py_float(" 5.5 "), Some(5.5));
        assert_eq!(py_float("-5"), Some(-5.0));
        assert_eq!(py_float("1_000.5"), Some(1000.5));
        assert_eq!(py_float(".5"), Some(0.5));
        assert_eq!(py_float("5."), Some(5.0));
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float("1E-3"), Some(0.001));
        assert_eq!(py_float("+2"), Some(2.0));
        assert_eq!(py_float("inf"), Some(f64::INFINITY));
        assert_eq!(py_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(py_float("nan").is_some_and(f64::is_nan));
        // Not numbers
        assert_eq!(py_float(""), None);
        assert_eq!(py_float("."), None);
        assert_eq!(py_float("5x"), None);
        assert_eq!(py_float("1e"), None);
        assert_eq!(py_float("1__0"), None);
        assert_eq!(py_float("_1"), None);
    }
}
