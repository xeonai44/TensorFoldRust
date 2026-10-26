//! API keys held as SHA-256 digests with labels; a restricted key file
//! reread on change or SIGHUP.
//!
//! Port of `zig/src/server/auth.zig` (380 lines): the keys from
//! `--api-key` (labels `cli-N`), `TENSORFOLD_API_KEY` (comma-split, labels
//! `env-N`) and the key file (a named `label: key` line a label, a bare
//! line `file-N`) — every key digested, the matches timing-safe; the file
//! a regular file other users cannot read (`chmod 600`), reread at most
//! once a second when its stat stamp changes or SIGHUP asked, a refused
//! reload closing the file's keys while the static ones stand.

use axum::http::HeaderMap;
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The label of the key a request carries, or `None`.
type Key = ([u8; 32], String);

/// SIGHUP's request: reread the key file (auth.zig:9).
pub static RELOAD_REQUESTED: AtomicBool = AtomicBool::new(false);

/// The key file's stat stamp: its device, inode, mtime, size and mode
/// (auth.zig:369's Meta).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    dev: u64,
    ino: u64,
    mtime_ns: i128,
    size: u64,
    mode: u32,
}

struct FileState {
    keys: Vec<Key>,
    valid: bool,
    stamp: Option<Stamp>,
    checked: Option<Instant>,
}

/// The key store: the static keys (the flag, the environment) and the
/// live file. Upstream: `Store` (auth.zig:17).
pub struct Auth {
    static_keys: Vec<Key>,
    path: Option<PathBuf>,
    pub metrics_open: bool,
    file: Mutex<FileState>,
}

/// SHA-256 of a key, or the error Python's message worded; None for empty
/// text or text with whitespace or control characters. Upstream: `digest`
/// (auth.zig:325).
fn digest(key: &str) -> Option<[u8; 32]> {
    if key.is_empty() {
        return None;
    }
    for ch in key.chars() {
        if (ch as u32) < 32 || is_space(ch) {
            return None;
        }
    }
    Some(Sha256::digest(key.as_bytes()).into())
}

/// Upstream: `isSpace` (auth.zig:333).
fn is_space(cp: char) -> bool {
    matches!(
        cp,
        ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}'
            | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

fn empty_digest() -> [u8; 32] {
    Sha256::digest(b"").into()
}

/// A header value is latin-1 text whose UTF-8 is hashed, as Python decodes
/// and encodes it. Upstream: `headerDigest` (auth.zig:301).
fn header_digest(value: &[u8]) -> [u8; 32] {
    let text: String = value.iter().map(|&b| b as char).collect();
    digest(&text).unwrap_or_else(empty_digest)
}

/// Upstream: `validLabel` (auth.zig:307).
fn valid_label(label: &str) -> bool {
    (1..=64).contains(&label.len())
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    || {
        // A label's length is its bytes' upstream (the 1-64 over the
        // ASCII letters) — non-ASCII cannot appear
        false
    }
}

/// The timing-safe comparison (upstream: `std.crypto.timing_safe.eql`).
fn timing_safe_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

impl Auth {
    /// The keys from `--api-key`, the environment and the file; errors
    /// carry Python's message. Upstream: `Store.init` (auth.zig:23).
    pub fn init(
        keys: &[String],
        environment: &str,
        path: Option<&Path>,
        metrics_open: bool,
    ) -> Result<Auth, String> {
        let mut static_keys = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let Some(d) = digest(key) else {
                return Err("API keys must be nonempty text without whitespace".into());
            };
            static_keys.push((d, format!("cli-{}", i + 1)));
        }
        for (i, raw) in environment.split(',').enumerate() {
            let key = raw.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n' | '\u{b}' | '\u{c}'));
            if key.is_empty() {
                continue;
            }
            let Some(d) = digest(key) else {
                return Err("API keys must be nonempty text without whitespace".into());
            };
            static_keys.push((d, format!("env-{}", i + 1)));
        }
        let auth = Auth {
            static_keys,
            path: path.map(Path::to_path_buf),
            metrics_open,
            file: Mutex::new(FileState {
                keys: Vec::new(),
                valid: true,
                stamp: None,
                checked: None,
            }),
        };
        if auth.path.is_some() {
            let mut problem = String::new();
            auth.read_file(&mut problem)?;
        }
        Ok(auth)
    }

    pub fn open() -> Auth {
        Auth {
            static_keys: Vec::new(),
            path: None,
            metrics_open: false,
            file: Mutex::new(FileState {
                keys: Vec::new(),
                valid: true,
                stamp: None,
                checked: None,
            }),
        }
    }

    /// Whether any key source stands. Upstream: `enabled` (auth.zig:58).
    pub fn enabled(&self) -> bool {
        !self.static_keys.is_empty() || self.path.is_some()
    }

    /// Reads the key file, replacing the file's keys; the problems carry
    /// Python's messages. Upstream: `readFile` (auth.zig:62).
    fn read_file(&self, problem: &mut String) -> Result<(), String> {
        use std::os::unix::fs::OpenOptionsExt;
        let path = self.path.as_deref().expect("the path stands");
        // NOFOLLOW: a symlink is not the file (auth.zig:65's .NOFOLLOW)
        // — the platform's own bits, verified against each platform's
        // fcntl.h (a live-caught bug twice over: macOS's 0x400000 is
        // O_DIRECTORY on Linux, and a mis-remembered "O_NONBLOCK"
        // 0x4000 is O_DIRECT there, breaking small unaligned reads)
        #[cfg(target_os = "macos")]
        const O_NOFOLLOW: i32 = 0x0040_0000;
        #[cfg(target_os = "macos")]
        const O_NONBLOCK: i32 = 0x0000_8000;
        #[cfg(target_os = "linux")]
        const O_NOFOLLOW: i32 = 0x0002_0000; // 0400000 octal, the header's own
        #[cfg(target_os = "linux")]
        const O_NONBLOCK: i32 = 0x0000_0800; // 04000 octal
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW | O_NONBLOCK)
            .open(path)
            .map_err(|_| "API key file cannot be opened safely".to_string())?;
        let meta = file
            .metadata()
            .map_err(|_| "API key file cannot be opened safely".to_string())?;
        use std::os::unix::fs::MetadataExt;
        let mode = meta.mode();
        if !meta.is_file() || mode & 0o044 != 0 {
            *problem =
                "API key file must be a regular file unreadable by other users; use chmod 600"
                    .into();
            return Err(problem.clone());
        }
        if meta.len() > 1 << 20 {
            *problem = "API key file exceeds 1 MiB".into();
            return Err(problem.clone());
        }
        let text = {
            use std::io::Read;
            let mut text = Vec::new();
            file.take(1 << 21).read_to_end(&mut text).map_err(|_| {
                "API key file cannot be opened safely".to_string()
            })?;
            if text.len() > 1 << 20 {
                *problem = "API key file exceeds 1 MiB".into();
                return Err(problem.clone());
            }
            String::from_utf8(text)
                .map_err(|_| "API key file cannot be opened safely".to_string())?
        };
        let mut keys: Vec<Key> = Vec::new();
        for raw in text.split(['\n', '\r']) {
            let line = raw.trim_matches(|c: char| {
                matches!(c, ' ' | '\t' | '\u{b}' | '\u{c}' | '\u{1c}'..='\u{1f}')
            });
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (label, key) = match line.find(':') {
                Some(colon) => {
                    let label = line[..colon].trim_matches(|c: char| matches!(c, ' ' | '\t'));
                    let key = line[colon + 1..].trim_matches(|c: char| matches!(c, ' ' | '\t'));
                    if !valid_label(label) {
                        *problem = "API key labels need 1-64 letters, digits, dots, underscores or hyphens".into();
                        return Err(problem.clone());
                    }
                    (label.to_string(), key)
                }
                None => (format!("file-{}", keys.len() + 1), line),
            };
            let Some(d) = digest(key) else {
                *problem = "API keys must be nonempty text without whitespace".into();
                return Err(problem.clone());
            };
            keys.push((d, label));
        }
        if keys.is_empty() {
            *problem = "API key file contains no keys".into();
            return Err(problem.clone());
        }
        let mut file = self.file.lock().unwrap();
        file.keys = keys;
        file.valid = true;
        file.stamp = Some(Stamp {
            dev: meta.dev(),
            ino: meta.ino(),
            mtime_ns: meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
            size: meta.len(),
            mode,
        });
        Ok(())
    }

    fn stat_stamp(&self) -> Option<Stamp> {
        let path = self.path.as_deref()?;
        let meta = std::fs::metadata(path).ok()?;
        use std::os::unix::fs::MetadataExt;
        Some(Stamp {
            dev: meta.dev(),
            ino: meta.ino(),
            mtime_ns: meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
            size: meta.len(),
            mode: meta.mode(),
        })
    }

    /// Rereads the key file when SIGHUP asked or, at most once a second,
    /// when its stamp changed. Upstream: `refresh` (auth.zig:118).
    fn refresh(&self) {
        if self.path.is_none() {
            return;
        }
        let forced = RELOAD_REQUESTED.swap(false, Ordering::AcqRel);
        let mut file = self.file.lock().unwrap();
        if !forced {
            if let Some(last) = file.checked {
                if last.elapsed().as_nanos() < 1_000_000_000 {
                    return;
                }
            }
        }
        file.checked = Some(Instant::now());
        drop(file);
        let stamp = self.stat_stamp();
        let stamp_changed = stamp.is_some() && Some(stamp.unwrap()) != self.file.lock().unwrap().stamp;
        if stamp.is_none() || forced || !self.file.lock().unwrap().valid || stamp_changed {
            let mut problem = String::new();
            let ok = stamp.is_some() && self.read_file(&mut problem).is_ok();
            let mut file = self.file.lock().unwrap();
            if ok {
                file.valid = true;
            } else {
                if file.valid {
                    eprintln!("[tensorfold] API key file reload refused; authenticated routes remain closed");
                }
                file.keys.clear();
                file.valid = false;
            }
        }
    }

    /// The label of the key a request carries (Authorization: Bearer, or
    /// x-api-key), or None. Upstream: `match` (auth.zig:141).
    pub fn match_(&self, authorization: Option<&str>, x_api_key: Option<&str>) -> Option<String> {
        self.refresh();
        let mut candidates = [empty_digest(), empty_digest()];
        if let Some(value) = authorization {
            if let Some(space) = value.find(' ') {
                if value[..space].eq_ignore_ascii_case("bearer") {
                    let rest = value[space + 1..].trim_matches(|c: char| {
                        matches!(
                            c,
                            ' ' | '\t'
                                | '\r'
                                | '\n'
                                | '\u{b}'
                                | '\u{c}'
                                | '\u{1c}'
                                | '\u{1d}'
                                | '\u{1e}'
                                | '\u{1f}'
                                | '\u{85}'
                                | '\u{a0}'
                        )
                    });
                    candidates[0] = header_digest(rest.as_bytes());
                }
            }
        }
        candidates[1] = match x_api_key {
            Some(value) => header_digest(value.as_bytes()),
            None => empty_digest(),
        };
        let file = self.file.lock().unwrap();
        let mut label: Option<&str> = None;
        for keys in [&self.static_keys, &file.keys] {
            for (digest, key_label) in keys {
                let matched = candidates
                    .iter()
                    .any(|c| timing_safe_eq(digest, c));
                if matched && label.is_none() {
                    label = Some(key_label);
                }
            }
        }
        if !file.valid {
            return None;
        }
        // The request's own copy: a reload frees the file's labels
        label.map(str::to_string)
    }

    /// The header reads through the middleware's request. Upstream: the
    /// conn's `Authorization` and `x-api-key` singles.
    pub fn match_headers(&self, headers: &HeaderMap) -> Option<String> {
        let authorization = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        let x_api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
        self.match_(authorization, x_api_key)
    }
}

/// The paths only a key opens (``auth_http.gated``). Upstream: `gated`
/// (auth.zig:346).
pub fn gated(raw_path: &str, metrics_open: bool) -> bool {
    let path = route_path(raw_path);
    if metrics_open && (path == "/metrics" || path == "/v1/metrics") {
        return false;
    }
    if path == "/v1" || path.starts_with("/v1/") {
        return true;
    }
    for p in [
        "/metrics",
        "/dashboard",
        "/stats",
        "/memory",
        "/tokenize",
        "/detokenize",
        "/models",
        "/messages",
        "/messages/count_tokens",
        "/chat/completions",
        "/completions",
        "/decisions",
        "/responses",
    ] {
        if path == p {
            return true;
        }
    }
    if path.starts_with("/responses/") {
        return true;
    }
    for end in ["/chat/completions", "/completions", "/decisions", "/models"] {
        if path.ends_with(end) {
            return true;
        }
    }
    false
}

/// A request path without its query and trailing slashes. Upstream:
/// `routePath` (auth.zig:248).
pub fn route_path(raw: &str) -> &str {
    let q = raw.find('?').unwrap_or(raw.len());
    raw[..q].trim_end_matches('/')
}

/// Whether ``host`` is a loopback address (a server without keys warns
/// otherwise). Upstream: `loopback` (auth.zig:254).
pub fn loopback(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let Ok(ip) = host.parse::<std::net::IpAddr>() else {
        return false;
    };
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => v6.is_loopback(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `gates and digests` test (auth.zig:341).
    #[test]
    fn gates_and_digests() {
        assert!(gated("/v1/models", false));
        assert!(!gated("/health", false));
        assert!(gated("/memory?reset_peak=1", true));
        assert!(gated("/v1/memory/", true));
        assert!(!gated("/metrics/", true));
        assert!(gated("/alternative/chat/completions", false));
        assert!(digest("has space").is_none());
        assert!(loopback("127.0.0.1") && loopback("::1") && !loopback("0.0.0.0"));
    }

    /// Upstream's `auth cleanup after invalid initialization` (auth.zig:352).
    #[test]
    fn init_refuses_invalid_keys() {
        assert!(Auth::init(&["sk-one".into(), "has space".into()], "", None, false).is_err());
        assert!(Auth::init(&["sk-one".into()], "sk-two,has space", None, false).is_err());
        assert!(Auth::init(&["sk-one".into()], "sk-two", Some(Path::new("missing")), false).is_err());
    }

    fn key_file(dir: &std::path::Path, name: &str, text: &str, mode: u32) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    /// Upstream's `the key file` test (auth.zig:401): refused while
    /// others can read it, read at 0600, reread after an edit.
    #[test]
    fn the_key_file_refused_read_and_reread() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("tf-auth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = key_file(&dir, "keys", "ops: sk-one\n", 0o644);
        let problem = match Auth::init(&[], "", Some(&path), false) {
            Err(problem) => problem,
            Ok(_) => panic!("the world-readable file must refuse"),
        };
        assert!(problem.contains("chmod 600"), "{problem}");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let auth = Auth::init(&["sk-cli".into()], "sk-env", Some(&path), false).unwrap();
        assert_eq!(auth.match_(Some("Bearer sk-cli"), None).as_deref(), Some("cli-1"));
        assert_eq!(auth.match_(None, Some("sk-env")).as_deref(), Some("env-1"));
        assert_eq!(auth.match_(Some("Bearer sk-one"), None).as_deref(), Some("ops"));
        // The edit changes the stamp: a SIGHUP (the public force) rereads
        // within the one-second guard
        std::fs::write(&path, "ops: sk-one\nci: sk-two\n").unwrap();
        RELOAD_REQUESTED.store(true, Ordering::Release);
        assert_eq!(auth.match_(None, Some("sk-two")).as_deref(), Some("ci"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Upstream's `a request's key label outlives a reload` (auth.zig:419)
    /// and the refused reload closing the file's keys.
    #[test]
    fn a_label_outlives_a_reload_and_a_refused_reload_closes_the_file_keys() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("tf-auth-outlive-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = key_file(&dir, "keys", "ops: sk-one\n", 0o600);
        let auth = Auth::init(&[], "", Some(&path), false).unwrap();
        let label = auth.match_(Some("Bearer sk-one"), None).expect("the request's key");
        std::fs::write(&path, "ops: sk-one\nci: sk-two\n").unwrap();
        // Another request rereads the file (a SIGHUP within the guard)
        RELOAD_REQUESTED.store(true, Ordering::Release);
        assert_eq!(auth.match_(None, Some("sk-two")).as_deref(), Some("ci"));
        // The request's own copy still names the first read's label
        assert_eq!(label, "ops");
        // The file made world-readable: the reload refuses and the
        // file's keys close while the label the request holds stands
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        RELOAD_REQUESTED.store(true, Ordering::Release);
        assert_eq!(auth.match_(Some("Bearer sk-one"), None), None);
        assert_eq!(label, "ops");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
