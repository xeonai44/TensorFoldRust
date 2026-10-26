//! The Hugging Face cache on disk: where checkpoints live, their sizes, and
//! which family serves them.
//!
//! Port of `zig/src/cli/hub.zig` (203 lines). The layout is the same one
//! 0.6.6 uses (upstream's contract): `HF_HUB_CACHE/models--org--name/`
//! with `blobs/`, `snapshots/{sha}/` (symlinks into blobs) and `refs/main`.

use serde_json::Value;
use std::path::{Path, PathBuf};

/// Upstream: hub.zig:6  pub const GiB.
pub const GIB: u64 = 1 << 30;

/// One family the engine registry serves, with a display title.
/// Upstream: hub.zig:9  pub const Family.
#[derive(Clone, Copy, Debug)]
pub struct Family {
    pub model_type: &'static str,
    pub title: &'static str,
    /// The weight formats the gate entries name (the registry's own field).
    #[allow(dead_code)]
    pub formats: &'static [&'static str],
}

/// The port's registry: the families this build serves (the Zig build's own
/// `engines.families` — upstream's native_engines registry).
const FAMILIES: [Family; 1] = [Family {
    model_type: "qwen3_5",
    title: "Qwen3.5-2B",
    formats: &["mlx-affine-4bit-group64"],
}];

/// The family that serves `model_type` from the engine registry, or None when
/// this build cannot serve it. Upstream: `family` (hub.zig:19).
pub fn family(model_type: &str) -> Option<Family> {
    FAMILIES.iter().copied().find(|f| f.model_type == model_type)
}

/// HF_HUB_CACHE, or HF_HOME/hub, or the default cache under the home
/// directory. Upstream: `cacheDir` (hub.zig:33).
pub fn cache_dir() -> PathBuf {
    if let Ok(over) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(over);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/huggingface/hub")
}

/// The repo's cache folder name: models--org--name.
/// Upstream: `repoDirName` (hub.zig:47).
pub fn repo_dir_name(repo: &str) -> Option<String> {
    let slash = repo.find('/')?;
    if slash == 0 || slash + 1 >= repo.len() {
        return None;
    }
    Some(format!(
        "models--{}--{}",
        &repo[..slash],
        &repo[slash + 1..]
    ))
}

/// A git sha: 40 hex characters, the only strings ever joined into a cache
/// path. Upstream: `hexSha` (hub.zig:53).
pub fn hex_sha(text: &str) -> bool {
    text.len() == 40
        && text
            .chars()
            .all(|ch| ch.is_ascii_hexdigit())
}

/// The cached snapshot serving `repo`: refs/main's revision, else the newest
/// snapshot with a config.json. Upstream: `cachedSnapshot` (hub.zig:63).
pub fn cached_snapshot(hub: &Path, repo: &str) -> Option<PathBuf> {
    let dir_name = repo_dir_name(repo)?;
    let root = hub.join(dir_name);
    let ref_path = root.join("refs/main");
    if let Ok(text) = std::fs::read_to_string(&ref_path) {
        let rev = text.trim();
        // Only a 40-hex git sha is ever joined into a cache path.
        if hex_sha(rev) {
            let snapshot = root.join("snapshots").join(rev);
            if snapshot.is_dir() {
                return Some(snapshot);
            }
        }
    }
    // Else the newest snapshot with a config.json.
    let snapshots = root.join("snapshots");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&snapshots).ok()?.flatten() {
        let snapshot = entry.path();
        if !snapshot.is_dir() || !snapshot.join("config.json").is_file() {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if let Ok(mtime) = meta.modified() {
            if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                newest = Some((mtime, snapshot));
            }
        }
    }
    newest.map(|(_, p)| p)
}

/// `owner/name` shape, without consulting the filesystem.
/// Upstream: `isRepoIdLike` (hub.zig:94).
pub fn is_repo_id_like(text: &str) -> bool {
    let Some(slash) = text.find('/') else { return false };
    if text.rfind('/') != Some(slash) || slash == 0 || slash + 1 == text.len() {
        return false;
    }
    for part in [&text[..slash], &text[slash + 1..]] {
        if part.len() > 96 || !part.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.') {
            return false;
        }
    }
    true
}

/// One integer field from the checkpoint's config.json.
/// Upstream: `configInt` (hub.zig:106).
pub fn config_int(dir: &Path, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value.get(key).and_then(Value::as_u64)
}

/// config.json's model_type, "unknown" when unreadable.
/// Upstream: `modelType` (hub.zig:116).
pub fn model_type(dir: &Path) -> String {
    model_type_of(dir).unwrap_or_else(|| "unknown".into())
}

fn model_type_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value
        .get("model_type")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// config.json's quantization: bits and group size, when the checkpoint is
/// MLX-affine quantized. Upstream: `quantization` (hub.zig:126).
pub fn quantization(dir: &Path) -> Option<(u64, u64)> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let q = value.get("quantization")?;
    let bits = q.get("bits").and_then(Value::as_u64)?;
    let group = q
        .get("group_size")
        .and_then(Value::as_u64)
        .unwrap_or(32);
    Some((bits, group))
}

/// Bytes of the files under `dir`, symlinks followed.
/// Upstream: `sizeOf` (hub.zig:139) — statFile follows the cache's symlinks
/// into blobs.
pub fn size_of(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    fn walk(dir: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_file() => {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        *total += meta.len();
                    }
                }
                Ok(t) if t.is_symlink() => {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        *total += meta.len();
                    }
                }
                Ok(t) if t.is_dir() => walk(&path, total),
                _ => {}
            }
        }
    }
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() => walk(&path, &mut total),
            _ => {
                if let Ok(meta) = std::fs::metadata(&path) {
                    total += meta.len();
                }
            }
        }
    }
    total
}

/// The checkpoint directory for `model`, or None with `problem` set (no
/// downloads: the cache must hold it). Upstream: `resolve` (hub.zig:23).
///
/// `model` resolves to a local directory (with `~/` expansion), else a
/// Hugging Face cache entry by repo id — the same resolution both builds'
/// serve paths run.
pub fn resolve(model: &str) -> Result<PathBuf, String> {
    // `~/` expansion (upstream's first step).
    let expanded = if let Some(rest) = model.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(model)
    };
    if expanded.is_dir() {
        return Ok(expanded);
    }
    if !is_repo_id_like(model) {
        return Err(format!(
            "{} is neither a directory nor a Hugging Face repo id (owner/name)",
            model
        ));
    }
    let hub_root = cache_dir();
    match cached_snapshot(&hub_root, model) {
        Some(snapshot) => Ok(snapshot),
        None => Err(format!(
            "{} is not in the Hugging Face cache; the native engine does not download: run tensorfold pull {} first",
            model, model
        )),
    }
}

/// models--org--name back to org/name. Upstream: models.zig's
/// `repoFromDirName`.
pub fn repo_from_dir_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("models--")?;
    let sep = rest.find("--")?;
    if sep == 0 || sep + 2 >= rest.len() {
        return None;
    }
    Some(format!("{}/{}", &rest[..sep], &rest[sep + 2..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors hub.zig:168: family detection matches the registry and refuses
    /// unknown types.
    #[test]
    fn family_detection() {
        assert_eq!(family("qwen3_5").unwrap().title, "Qwen3.5-2B");
        assert!(family("gemma4").is_none());
        assert!(family("unknown").is_none());
    }

    /// Mirrors hub.zig:175: the cache layout helpers on a synthetic cache.
    #[test]
    fn cache_layout_reads_a_synthetic_cache() {
        let hub = std::env::temp_dir().join(format!("tf-hub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&hub);
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = hub.join("models--Org--Name/snapshots").join(sha);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(hub.join("models--Org--Name/refs")).unwrap();
        std::fs::write(
            snapshot.join("config.json"),
            r#"{"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}, "max_position_embeddings": 262144}"#,
        )
        .unwrap();
        std::fs::write(hub.join("models--Org--Name/refs/main"), format!("{}\n", sha)).unwrap();
        let weights = snapshot.join("weights.safetensors");
        std::fs::write(&weights, vec![b'x'; 4096]).unwrap();
        let found = cached_snapshot(&hub, "Org/Name").unwrap();
        assert!(found.ends_with(sha));
        assert_eq!(model_type(&found), "qwen3_5");
        let (bits, group) = quantization(&found).unwrap();
        assert_eq!((bits, group), (4, 64));
        assert_eq!(config_int(&found, "max_position_embeddings"), Some(262144));
        assert!(size_of(&found) >= 4096);
        // A ref that is not a sha is never joined.
        std::fs::create_dir_all(hub.join("models--Org--Bad/refs")).unwrap();
        std::fs::write(hub.join("models--Org--Bad/refs/main"), "not-a-sha").unwrap();
        assert!(cached_snapshot(&hub, "Org/Bad").is_none());
        assert!(cached_snapshot(&hub, "Org/Other").is_none());
        assert!(repo_dir_name("Org/Name").is_some());
        assert!(hex_sha(sha));
        assert!(!hex_sha("short"));
        assert!(is_repo_id_like("Org/Name"));
        assert!(!is_repo_id_like("just-a-word"));
        assert_eq!(
            repo_from_dir_name("models--Org--Name").unwrap(),
            "Org/Name"
        );
        assert!(repo_from_dir_name("models--no-sep").is_none());
        let _ = std::fs::remove_dir_all(&hub);
    }
}

/// The name a server answers to: --name, else the repo's or directory's last
/// part. Port of the old hub.zig:61 `servedName` (kept beside its callers).
pub fn served_name(name: &str, model: &str, dir: &str) -> String {
    if !name.is_empty() {
        return name.to_string();
    }
    let source = if is_repo_id_like(model) {
        model.trim_end_matches('/')
    } else {
        dir.trim_end_matches('/')
    };
    Path::new(source)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| source.to_string())
}
