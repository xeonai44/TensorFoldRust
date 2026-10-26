//! The `models` command: the checkpoints on disk that a registered family can
//! serve.
//!
//! Port of `zig/src/cli/models.zig` (66 lines).

use crate::hub;
use std::io::Write;
use std::path::PathBuf;

/// Lists each cache checkpoint whose model_type a family serves. Returns the
/// process exit code. Upstream: `run` (models.zig:7).
pub fn run(out: &mut dyn Write, override_cache: Option<&std::path::Path>) -> i32 {
    let root = override_cache.map(PathBuf::from).unwrap_or_else(hub::cache_dir);
    let Ok(entries) = std::fs::read_dir(&root) else {
        let _ = writeln!(
            out,
            "no checkpoints on disk: the Hugging Face cache at {} does not exist yet; run: tensorfold pull OWNER/NAME",
            root.display()
        );
        return 0;
    };
    let mut found: usize = 0;
    let mut repos: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            hub::repo_from_dir_name(&name)
        })
        .collect();
    repos.sort();
    for repo in repos {
        let Some(snapshot) = hub::cached_snapshot(&root, &repo) else { continue };
        let model_type = hub::model_type(&snapshot);
        let Some(family) = hub::family(&model_type) else { continue };
        let _ = writeln!(out, "{}: {} ({})", repo, family.title, family.model_type);
        found += 1;
    }
    if found == 0 {
        let _ = writeln!(
            out,
            "no checkpoints on disk that a registered Zig family can serve; run: tensorfold pull OWNER/NAME"
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors models.zig's test: serveable checkpoints listed, others skipped.
    #[test]
    fn lists_serveable_checkpoints_only() {
        let root = std::env::temp_dir().join(format!("tf-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let flash = root.join("models--Org--Flash/snapshots").join(sha);
        std::fs::create_dir_all(root.join("models--Org--Flash/refs")).unwrap();
        std::fs::create_dir_all(&flash).unwrap();
        std::fs::write(root.join("models--Org--Flash/refs/main"), sha).unwrap();
        std::fs::write(
            flash.join("config.json"),
            r#"{"model_type": "qwen3_5"}"#,
        )
        .unwrap();
        let draft = root.join("models--Org--Draft/snapshots/rev2");
        std::fs::create_dir_all(&draft).unwrap();
        std::fs::write(draft.join("config.json"), r#"{"model_type": "gemma4"}"#).unwrap();
        // Point the cache at the test root.
        let mut out = Vec::new();
        let code = run(&mut out, Some(&root));
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Org/Flash: Qwen3.5-2B (qwen3_5)"), "{}", text);
        assert!(!text.contains("Org/Draft"), "{}", text);
        let _ = std::fs::remove_dir_all(&root);
    }
}
