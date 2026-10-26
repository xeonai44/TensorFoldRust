//! The live status line's /health copy: open requests, how many wait,
//! decode and prefill tokens a second.
//!
//! Port of `zig/src/server/live.zig`'s snapshot (the terminal line's
//! remaining half rides with the CLI's serve loop, which the port's axum
//! serve does not draw).

use serde_json::{json, Value};
use tensorfold_engine::Engine;

/// Python's ``round(x, 1)``.
fn round1(x: f64) -> f64 {
    // The shortest one-decimal text, parsed back (Zig's `{d:.1}` round-trip
    // shape)
    let text = format!("{x:.1}");
    text.parse().unwrap_or(x)
}

/// /health's ``live``: open requests, how many wait, decode and prefill
/// tokens a second. Upstream: `snapshot` (live.zig:16).
pub fn snapshot(engine: &dyn Engine) -> Value {
    let mut tokens: [u32; 64] = [0; 64];
    let status = engine.status(&mut tokens);
    json!({
        "connections": status.running + status.waiting,
        "waiting": status.waiting,
 "decode_tokens_per_second": round1(status.decode_tokens_per_second),
        "prefill_tokens_per_second": round1(status.prefill_tokens_per_second),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// round1: one decimal, the round-trip shape (live.zig:8).
    #[test]
    fn rounds_to_one_decimal() {
        assert_eq!(round1(142.0), 142.0);
        assert_eq!(round1(142.123), 142.1);
        assert_eq!(round1(1210.46), 1210.5);
        assert_eq!(round1(0.0), 0.0);
    }
}
