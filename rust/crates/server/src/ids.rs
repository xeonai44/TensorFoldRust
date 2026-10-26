//! Random ids: ``uuid4().hex`` prefixes for replies, calls and items.
//!
//! Port of `zig/src/server/ids.zig` (23 lines). Upstream reads its bytes from
//! the platform random source (`log.io().random`); this port reads them from
//! `getrandom`, the same source underneath. Where the OS source is unavailable
//! (upstream asserts it never is), a time/process/counter mix keeps ids unique
//! and the shape intact.

use std::sync::atomic::{AtomicU64, Ordering};

/// ``prefix`` then ``n`` random lowercase hex digits.
///
/// Upstream: `make` (ids.zig:6) — each byte feeds two digits, high nibble
/// first, the pairing kept exact.
pub fn make(prefix: &str, n: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let n = n.min(64); // the upstream buffer is 32 bytes, so 64 digits at most
    let mut bytes = [0u8; 32];
    if getrandom::fill(&mut bytes).is_err() {
        fallback(&mut bytes);
    }
    let mut out = String::with_capacity(prefix.len() + n);
    out.push_str(prefix);
    for i in 0..n {
        let byte = bytes[i / 2];
        let nibble = (byte >> (4 * (1 - (i % 2) as u32))) & 0xf;
        out.push(HEX[nibble as usize] as char);
    }
    out
}

/// The entropy of last resort: nanoseconds, the process id and a counter,
/// mixed a byte at a time.
fn fallback(bytes: &mut [u8; 32]) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id() as u64;
    let mut state = now ^ (pid << 1) ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for b in bytes.iter_mut() {
        state = state.rotate_left(13) ^ (state >> 7);
        state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        *b = (state >> 56) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prefix_then_hex_digits() {
        let id = make("call_", 24);
        assert_eq!(id.len(), 24 + "call_".len());
        assert!(id.starts_with("call_"));
        assert!(id["call_".len()..]
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }

    #[test]
    fn ids_differ() {
        assert_ne!(make("call_", 24), make("call_", 24));
    }
}
