//! `capabilities --json`: what this build is, serves, and takes.
//!
//! Port of `zig/src/server/cli.zig:289  pub fn capabilities` (the document's shape) and
//! `server/engines.zig:10` (what fills it). The Zig document's keys stay; the engine
//! field names this port ("rust"), and the flags list only what this build honours —
//! upstream's rule: a flag the table does not carry is refused, and the document lists
//! the native flags the table has (`cli.zig:288`).

use serde_json::{json, Value};

/// The flags this build honours, with their value choices where the table has them.
fn serve_flags() -> Value {
    let mut flags = serde_json::Map::new();
    for name in [
        "--host",
        "--port",
        "--name",
        "--alias",
        "--api-key",
        "--api-key-file",
        "--metrics-open",
        "--context",
        "--max-tokens",
        "--temperature",
        "--top-p",
        "--top-k",
        "--min-p",
        "--thinking",
        "--no-thinking",
        "--thinking-budget",
        "--loop-guard",
        "--no-drafts",
        "--parallel",
        "--dashboard",
        "--keep-warm",
    ] {
        flags.insert(name.to_string(), json!({}));
    }
    // --reasoning-effort: low, medium, high or xhigh (the table's choices).
    flags.insert(
        "--reasoning-effort".to_string(),
        json!({"values": ["low", "medium", "high", "xhigh"]}),
    );
    Value::Object(flags)
}

/// The environment variables this build reads (the env list, the same field upstream
/// fills with its honoured variables).
fn env() -> Vec<&'static str> {
    vec![
        "TENSORFOLD_API_KEY",
        "TENSORFOLD_REQUEST_LOG",
        "TENSORFOLD_SEED_SALT",
    ]
}

pub fn document() -> Value {
    json!({
        "schema": 1,
        "engine": "rust",
        "version": env!("CARGO_PKG_VERSION"),
        "chip": Value::Null,
        "backends": [],
        "families": {},
        "serve": {"flags": serve_flags()},
        "env": env(),
    })
}

/// Print the document (with --json, the only way upstream prints it; anything else is
/// the usage line and exit 2, mirroring `server/main.zig:38-45`).
pub fn run(json: bool) -> i32 {
    if !json {
        eprintln!("usage: tensorfold-native capabilities --json | models | info MODEL | pull REPO[@REVISION] | serve MODEL [flags]");
        return 2;
    }
    println!("{}", document());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_holds_the_upstream_keys() {
        let doc = document();
        assert_eq!(doc["schema"], 1);
        assert_eq!(doc["engine"], "rust");
        assert!(doc["chip"].is_null());
        assert_eq!(doc["backends"].as_array().unwrap().len(), 0);
        assert_eq!(doc["families"].as_object().unwrap().len(), 0);
        // The reasoning-effort choices, the one flag with values in Phase 0.
        assert_eq!(
            doc["serve"]["flags"]["--reasoning-effort"]["values"],
            json!(["low", "medium", "high", "xhigh"])
        );
        // The usage line is the document's sibling: nothing prints without --json.
        assert!(run(false) == 2);
    }
}
