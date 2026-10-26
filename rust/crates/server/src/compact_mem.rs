//! The compaction note file and the hash of the messages a note already
//! covers.
//!
//! Port of `zig/src/server/compact_mem.zig` (86 lines).

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

/// The stored note and what it covers. Upstream: `Mem` (compact_mem.zig:7).
pub struct Mem {
    pub note: String,
    pub covered: usize,
    pub hash: String,
}

fn role_of(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("")
}

fn first_role<'m>(msgs: &'m [Value], role: &str) -> Option<&'m Value> {
    msgs.iter().find(|m| role_of(m) == role)
}

/// The conversation's key: the first system and first user messages,
/// compact JSON. Upstream: `conversationKey` (compact_mem.zig:11) — the
/// first 16 hex of the SHA-256.
pub fn conversation_key(msgs: &[Value]) -> [u8; 16] {
    let mut h = Sha256::new();
    if let Some(m) = first_role(msgs, "system") {
        h.update(crate::tool_parse::zig_stringify(m, true).as_bytes());
    }
    h.update(b"\n");
    if let Some(m) = first_role(msgs, "user") {
        h.update(crate::tool_parse::zig_stringify(m, true).as_bytes());
    }
    let d = h.finalize();
    // The first sixteen hex characters of the digest (compact_mem.zig:18's
    // hex[0..16])
    let hex: String = d.iter().map(|b| format!("{b:02x}")).collect();
    let mut out = [0u8; 16];
    out.copy_from_slice(&hex.as_bytes()[..16]);
    out
}

fn prefix_hash(msgs: &[Value]) -> String {
    let bytes = crate::tool_parse::zig_stringify(&Value::Array(msgs.to_vec()), true);
    let d = Sha256::digest(bytes.as_bytes());
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether the stored hash still matches the messages the note covered.
/// Upstream: `hashes` (compact_mem.zig:26).
pub fn hashes(msgs: &[Value], mem: &Mem) -> bool {
    if mem.covered == 0 || mem.covered > msgs.len() || mem.hash.is_empty() {
        return false;
    }
    prefix_hash(&msgs[..mem.covered]) == mem.hash
}

fn read_named(dir: &Path, key: &[u8; 16], ext: &str) -> Option<String> {
    let name = format!("{}{}", String::from_utf8_lossy(key), ext);
    std::fs::read_to_string(dir.join(name)).ok()
}

fn write_named(dir: &Path, key: &[u8; 16], ext: &str, data: &str) {
    let name = format!("{}{}", String::from_utf8_lossy(key), ext);
    let tmp = format!("{}.tmp", name);
    let path = dir.join(name);
    let tmp_path = dir.join(tmp);
    if std::fs::write(&tmp_path, data).is_ok() {
        let _ = std::fs::rename(&tmp_path, &path);
    }
}

/// The stored note for this conversation, if one is. Upstream: `loadMem`
/// (compact_mem.zig:31).
pub fn load_mem(dir: &Path, msgs: &[Value]) -> Option<Mem> {
    let key = conversation_key(msgs);
    let note = read_named(dir, &key, ".md")?;
    let js = match read_named(dir, &key, ".json") {
        Some(js) => js,
        None => {
            return Some(Mem {
                note,
                covered: 0,
                hash: String::new(),
            })
        }
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&js) else {
        return Some(Mem {
            note,
            covered: 0,
            hash: String::new(),
        });
    };
    let covered = parsed
        .get("covered")
        .and_then(Value::as_i64)
        .filter(|n| *n > 0)
        .map(|n| n as usize)
        .unwrap_or(0);
    let hash = parsed
        .get("hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Some(Mem {
        note,
        covered,
        hash,
    })
}

/// The note stored with what it covers. Upstream: `storeMem`
/// (compact_mem.zig:49) — the atomic rename the port keeps.
pub fn store_mem(dir: &Path, msgs: &[Value], covered: usize, note: &str) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let key = conversation_key(msgs);
    let n = covered.min(msgs.len());
    let hash = prefix_hash(&msgs[..n]);
    let body = format!("{{\"covered\":{n},\"hash\":\"{hash}\"}}");
    write_named(dir, &key, ".md", note);
    write_named(dir, &key, ".json", &body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "you are helpful"}),
            json!({"role": "user", "content": "hello there"}),
            json!({"role": "assistant", "content": "hi"}),
        ]
    }

    /// The key: the first system and user, the hash of the covered prefix
    /// (compact_mem.zig:11/26/49).
    #[test]
    fn the_key_and_the_covered_hash() {
        let key = conversation_key(&msgs());
        assert_eq!(key.len(), 16);
        assert!(key.iter().all(|b| b.is_ascii_hexdigit()));
        // The key reads the first of each role only: a longer conversation
        // with the same heads keys the same
        let longer = [
            msgs(),
            vec![json!({"role": "user", "content": "more"})],
        ]
        .concat();
        assert_eq!(conversation_key(&longer), key);
        // The covered hash matches the exact prefix
        let mem = Mem {
            note: String::new(),
            covered: 2,
            hash: prefix_hash(&msgs()[..2]),
        };
        assert!(hashes(&msgs(), &mem));
        let bad = Mem {
            covered: 2,
            hash: "deadbeef".into(),
            ..mem
        };
        assert!(!hashes(&msgs(), &bad));
        // Out of range or empty never hashes
        assert!(!hashes(&msgs(), &Mem {
            note: String::new(),
            covered: 0,
            hash: "x".into(),
        }));
        assert!(!hashes(&msgs(), &Mem {
            note: String::new(),
            covered: 9,
            hash: "x".into(),
        }));
    }

    /// The store round-trips: the note and its covered hash
    /// (compact_mem.zig:31/49).
    #[test]
    fn the_note_store_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "tf-compact-mem-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let m = msgs();
        // Nothing stored: nothing
        assert!(load_mem(&dir, &m).is_none());
        store_mem(&dir, &m, 2, "the note");
        let mem = load_mem(&dir, &m).expect("the stored note");
        assert_eq!(mem.note, "the note");
        assert_eq!(mem.covered, 2);
        assert!(hashes(&m, &mem));
        // A conversation whose covered prefix changed beyond the key's
        // heads: the same key, the hash no longer matches
        store_mem(&dir, &m, 3, "the note");
        let diverged = [
            json!({"role": "system", "content": "you are helpful"}),
            json!({"role": "user", "content": "hello there"}),
            json!({"role": "assistant", "content": "hi, edited"}),
        ];
        let mem = load_mem(&dir, &diverged).expect("the same key: the same heads");
        assert!(!hashes(&diverged, &mem), "a changed body breaks the hash");
        let mem = load_mem(&dir, &m).expect("the stored note");
        assert!(hashes(&m, &mem));
        // A note stored without its json (covered 0) never hashes
        std::fs::write(
            dir.join(format!("{}.md", String::from_utf8_lossy(&conversation_key(&m)))),
            "bare note",
        )
        .ok();
        std::fs::remove_file(
            dir.join(format!("{}.json", String::from_utf8_lossy(&conversation_key(&m)))),
        )
        .ok();
        let mem = load_mem(&dir, &m).expect("the bare note loads");
        assert_eq!(mem.note, "bare note");
        assert_eq!(mem.covered, 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
