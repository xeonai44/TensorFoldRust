//! The `info` command: one checkpoint's family, format, context, weights size
//! and memory floor.
//!
//! Port of `zig/src/cli/info.zig` (84 lines).

use crate::hub;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Kept back for kernels, activations and buffers that grow with long prompts.
/// Upstream: info.zig:8  pub const runtime_margin = 2 * GiB.
pub const RUNTIME_MARGIN: u64 = 2 * hub::GIB;

/// Prints what a family would serve from `model` (a directory or a cached repo
/// id). Returns the process exit code. Upstream: `run` (info.zig:11).
pub fn run(out: &mut dyn Write, override_cache: Option<&std::path::Path>, model: &str) -> i32 {
    let dir: PathBuf = if Path::new(model).is_dir() {
        PathBuf::from(model)
    } else if hub::is_repo_id_like(model) {
        let root = override_cache.map(PathBuf::from).unwrap_or_else(hub::cache_dir);
        match hub::cached_snapshot(&root, model) {
            Some(snapshot) => snapshot,
            None => {
                let _ = writeln!(
                    out,
                    "{} is not in the Hugging Face cache; run: tensorfold pull {}",
                    model, model
                );
                return 1;
            }
        }
    } else {
        let _ = writeln!(
            out,
            "{} is neither a directory nor a Hugging Face repo id (owner/name)",
            model
        );
        return 1;
    };
    let model_type = hub::model_type(&dir);
    let Some(family) = hub::family(&model_type) else {
        let _ = writeln!(
            out,
            "{}: no registered Zig family serves model_type {}; the 0.6 line may: tensorfold@0.6",
            model, model_type
        );
        return 1;
    };
    let weights = hub::size_of(&dir);
    let _ = writeln!(out, "family: {} ({})", family.title, family.model_type);
    match hub::quantization(&dir) {
        Some((bits, group)) => {
            let _ = writeln!(out, "format: {}-bit, group {}", bits, group);
        }
        None => {
            let _ = writeln!(out, "format: full precision");
        }
    }
    if let Some(context) = hub::config_int(&dir, "max_position_embeddings") {
        let _ = writeln!(out, "context: {}", context);
    }
    let _ = writeln!(
        out,
        "weights: {:.2} GiB ({} bytes)",
        weights as f64 / hub::GIB as f64,
        weights
    );
    let _ = writeln!(
        out,
        "memory floor: {:.2} GiB (weights plus a {:.2} GiB runtime margin)",
        (weights + RUNTIME_MARGIN) as f64 / hub::GIB as f64,
        RUNTIME_MARGIN as f64 / hub::GIB as f64
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors info.zig's test: family, format, context, weights and floor
    /// for a cached checkpoint; the refusal for a family the build cannot serve.
    #[test]
    fn prints_family_format_context_weights_and_floor() {
        let root = std::env::temp_dir().join(format!("tf-info-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let flash = root.join("models--Org--Flash/snapshots/rev1");
        std::fs::create_dir_all(root.join("models--Org--Flash/refs")).unwrap();
        std::fs::create_dir_all(&flash).unwrap();
        std::fs::write(root.join("models--Org--Flash/refs/main"), "rev1").unwrap();
        std::fs::write(
            flash.join("config.json"),
            r#"{"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}, "max_position_embeddings": 262144}"#,
        )
        .unwrap();
        std::fs::write(flash.join("weights.safetensors"), vec![b'x'; 8192]).unwrap();
        let mut out = Vec::new();
        assert_eq!(run(&mut out, Some(&root), "Org/Flash"), 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("family: Qwen3.5-2B (qwen3_5)"), "{}", text);
        assert!(text.contains("format: 4-bit, group 64"), "{}", text);
        assert!(text.contains("context: 262144"), "{}", text);
        assert!(text.contains("weights:"), "{}", text);
        assert!(text.contains("memory floor: 2.00 GiB"), "{}", text);
        let draft = root.join("models--Org--Draft/snapshots/rev2");
        std::fs::create_dir_all(root.join("models--Org--Draft/refs")).unwrap();
        std::fs::create_dir_all(&draft).unwrap();
        std::fs::write(root.join("models--Org--Draft/refs/main"), "rev2").unwrap();
        std::fs::write(draft.join("config.json"), r#"{"model_type": "gemma4"}"#).unwrap();
        let mut refused = Vec::new();
        assert_eq!(run(&mut refused, Some(&root), "Org/Draft"), 1);
        let text = String::from_utf8(refused).unwrap();
        assert!(
            text.contains("no registered Zig family serves model_type gemma4"),
            "{}",
            text
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
