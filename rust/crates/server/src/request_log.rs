//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one
//! JSON line, images redacted, for replays.
//!
//! Port of `zig/src/server/request_log.zig` (40 lines).

use serde_json::Value;
use std::io::Write;

/// The body with every image part's URL replaced, the rest kept.
///
/// Upstream: `redact` (request_log.zig:10) — an object whose type is
/// image_url keeps its shape, the URL the redaction text; everything else
/// rebuilt as it stood.
fn redact(v: &Value) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        Value::Object(o) => {
            let image_url = o.get("type").and_then(Value::as_str) == Some("image_url");
            if image_url {
                let mut copy = o.clone();
                copy.insert(
                    "image_url".into(),
                    serde_json::json!({"url": "<redacted>"}),
                );
                Value::Object(copy)
            } else {
                Value::Object(
                    o.iter().map(|(k, item)| (k.clone(), redact(item))).collect(),
                )
            }
        }
        other => other.clone(),
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are
/// not client traffic). Upstream: `append` (request_log.zig:24) — the
/// failure modes all quiet (the request goes on whatever the log did).
pub fn append(path: &std::path::Path, body: &Value) {
    if body.get("priority").and_then(Value::as_str) == Some("background") {
        return;
    }
    let line = match serde_json::to_string(&redact(body)) {
        Ok(line) => line,
        Err(_) => return,
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
    else {
        return;
    };
    let _ = writeln!(file, "{line}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The redaction: image URLs replaced, the rest kept (request_log.zig:10).
    #[test]
    fn the_image_urls_redact() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                ]},
            ],
        });
        let redacted = redact(&body);
        assert_eq!(
            redacted["messages"][0]["content"][1]["image_url"]["url"],
            "<redacted>"
        );
        assert_eq!(redacted["messages"][0]["content"][0]["text"], "look");
        assert_eq!(redacted["model"], "m");
        // A nested image deep inside
        let deep = json!({"a": {"type": "image_url", "image_url": {"url": "u"}, "type2": "image_url"}});
        assert_eq!(redact(&deep)["a"]["image_url"]["url"], "<redacted>");
    }

    /// The append: background skipped, each body one JSON line
    /// (request_log.zig:24).
    #[test]
    fn the_log_appends_one_line_each() {
        let dir = std::env::temp_dir().join(format!(
            "tf-reqlog-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("log.jsonl");
        std::fs::create_dir_all(&dir).unwrap();
        append(&path, &json!({"a": 1}));
        append(&path, &json!({"priority": "background"})); // skipped
        append(&path, &json!({"b": 2}));
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "the background body skipped");
        assert!(lines[0].contains("\"a\":1"));
        assert!(lines[1].contains("\"b\":2"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
