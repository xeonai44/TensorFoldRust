//! ``json.dumps`` byte for byte: Python's separators, escapes and float
//! repr.
//!
//! Port of `zig/src/server/json_write.zig` (170 lines): every JSON body the
//! server writes goes through here, so the port's replies match the Zig
//! engine's byte for byte — the frozen corpus
//! (`zig/tests/server/golden/**`) carries the reference bytes. The chat
//! reply bodies ride axum's `Json` no more; `PythonJson` below writes
//! these bytes with the `application/json` header, and the SSE events carry
//! them as their `data:` payload.

use serde_json::Value;
use std::fmt::Write as _;

/// The writer's options. Upstream: `Options` (json_write.zig:5) —
/// `ascii` is `ensure_ascii`, `compact` the `separators=(",", ":")` shape.
#[derive(Clone, Copy)]
pub struct Options {
    pub ascii: bool,
    pub compact: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            ascii: true,
            compact: false,
        }
    }
}

/// The server's JSON body: `json.dumps(v)` — spaced, ASCII-safe.
pub fn dumps(v: &Value) -> String {
    dumps_with(v, &Options::default())
}

/// Upstream: `write` (json_write.zig:9).
pub fn dumps_with(v: &Value, o: &Options) -> String {
    let mut out = String::new();
    write(&mut out, v, o);
    out
}

fn write(out: &mut String, v: &Value, o: &Options) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        // Python's ints hold any width; the port's u64/i64 write their
        // exact decimals, the rest fall to the float (a parser-side
        // limit, not a writer one)
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(f) = n.as_f64() {
                out.push_str(&float_json(f));
            }
        }
        Value::String(s) => write_string(out, s, o),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(if o.compact { "," } else { ", " });
                }
                write(out, item, o);
            }
            out.push(']');
        }
        // The insertion order the port's serde_json preserves is the
        // order upstream's ordered Object holds
        Value::Object(obj) => {
            out.push('{');
            for (i, (k, item)) in obj.iter().enumerate() {
                if i > 0 {
                    out.push_str(if o.compact { "," } else { ", " });
                }
                write_string(out, k, o);
                out.push_str(if o.compact { ":" } else { ": " });
                write(out, item, o);
            }
            out.push('}');
        }
    }
}

/// Upstream: `writeString` (json_write.zig:37) — the short escapes, the
/// `\u00xx` of the controls, `ensure_ascii`'s `\uxxxx` and the surrogate
/// pairs, DEL escaped with the rest of the ASCII-unsafe.
fn write_string(out: &mut String, s: &str, o: &Options) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            _ => {
                let code = c as u32;
                if code >= 0x20 && !(o.ascii && code >= 0x7f) {
                    out.push(c);
                } else if code < 0x10000 {
                    let _ = write!(out, "\\u{:04x}", code);
                } else {
                    // The astral planes as their surrogate pair
                    let v = code - 0x10000;
                    let _ = write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (v >> 10),
                        0xdc00 + (v & 0x3ff)
                    );
                }
            }
        }
    }
    out.push('"');
}

/// Python's json float text: repr, with NaN and Infinity spelled as
/// JavaScript does. Upstream: `floatJson` (json_write.zig:86). The port's
/// serde_json numbers cannot hold NaN or Infinity — the spellings stay
/// for the parity of the rule.
pub fn float_json(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    crate::pyrepr::float_repr(f)
}

/// A JSON body written the way upstream writes every one: the bytes with
/// the `application/json` type. Pairs with a status the way `Json` does —
/// `(StatusCode::BAD_REQUEST, PythonJson(v))`.
pub struct PythonJson(pub Value);

impl axum::response::IntoResponse for PythonJson {
    fn into_response(self) -> axum::response::Response {
        let body = dumps(&self.0);
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Upstream's `float repr` test (json_write.zig:139), case for case.
    #[test]
    fn float_repr() {
        let cases: [(f64, &str); 11] = [
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (-2.5, "-2.5"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
        ];
        for (f, want) in cases {
            assert_eq!(float_json(f), want, "{f}");
        }
    }

    /// Upstream's `dumps` test (json_write.zig:152), byte for byte: the
    /// default spacing and ensure_ascii, the compact non-ASCII shape.
    #[test]
    fn dumps_the_upstream_case() {
        // Upstream's input: é, a newline, 😀 and DEL; -0 written from the
        // int it canonicalizes to (the port's parser turns the literal
        // "-0" into the float -0.0 — a parse-side gap, the writer below)
        let r: Value = serde_json::from_str(
            r#"{"a": [1, 2.50, "é\n😀\u007f"], "b": {}}"#,
        )
        .unwrap();
        let mut r = r;
        r["c"] = json!(0);
        assert_eq!(
            dumps(&r),
            "{\"a\": [1, 2.5, \"\\u00e9\\n\\ud83d\\ude00\\u007f\"], \"b\": {}, \"c\": 0}"
        );
        // ensure_ascii off keeps the text raw, DEL included; the newline
        // still escapes
        assert_eq!(
            dumps_with(
                &r,
                &Options {
                    ascii: false,
                    compact: true
                }
            ),
            "{\"a\":[1,2.5,\"é\\n😀\u{7f}\"],\"b\":{},\"c\":0}"
        );
    }

    /// The parse-side `-0` gap, asserted as the port's current behavior:
    /// serde_json holds the literal "-0" as the float -0.0, which dumps
    /// as Python dumps a float zero; upstream's parser canonicalizes to
    /// the int 0. A request echoing its own "-0" back diverges until the
    /// request parser grows upstream's int rule (see the map's gaps).
    #[test]
    fn the_minus_zero_parse_gap() {
        let parsed: Value = serde_json::from_str("-0").unwrap();
        assert_eq!(dumps(&parsed), "-0.0");
        assert_eq!(dumps(&json!(0)), "0");
    }

    /// The escapes: the short table, the controls, DEL, the BMP and the
    /// astral plane (writeString's every branch).
    #[test]
    fn the_escapes() {
        let v = json!({
            "quote": "a\"b", "back": "a\\b", "controls": "\u{1}\u{1f}",
            "del": "\u{7f}", "bmp": "é", "astral": "😀",
        });
        assert_eq!(
            dumps(&v),
            "{\"quote\": \"a\\\"b\", \"back\": \"a\\\\b\", \"controls\": \"\\u0001\\u001f\", \"del\": \"\\u007f\", \"bmp\": \"\\u00e9\", \"astral\": \"\\ud83d\\ude00\"}"
        );
        // Without ensure_ascii the text passes through
        let o = Options {
            ascii: false,
            compact: false,
        };
        assert_eq!(
            dumps_with(&json!({"bmp": "é", "del": "\u{7f}"}), &o),
            "{\"bmp\": \"é\", \"del\": \"\u{7f}\"}"
        );
    }

    /// A frozen corpus body, byte for byte: the whole chain — the fields,
    /// the spacing, the escapes — against
    /// `zig/tests/server/golden/responses/refuse-include.json`.
    #[test]
    fn the_frozen_corpus_body() {
        let body = json!({
            "error": {
                "message": "include is not supported (reasoning.encrypted_content): reasoning comes as text in its reasoning item, and nothing is encrypted",
                "type": "invalid_request_error",
                "param": null,
                "code": null,
            }
        });
        assert_eq!(
            dumps(&body),
            "{\"error\": {\"message\": \"include is not supported (reasoning.encrypted_content): reasoning comes as text in its reasoning item, and nothing is encrypted\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": null}}"
        );
        // The 404 with the quoted id (get-missing.json)
        let body = json!({
            "error": {"message": "no stored response has id 'resp_missing'",
                      "type": "invalid_request_error", "param": "response_id", "code": null}
        });
        assert_eq!(
            dumps(&body),
            "{\"error\": {\"message\": \"no stored response has id 'resp_missing'\", \"type\": \"invalid_request_error\", \"param\": \"response_id\", \"code\": null}}"
        );
    }
}
