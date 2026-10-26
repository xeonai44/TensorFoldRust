//! A model folder's config files: what a family gates on and what a server defaults to.
//!
//! Upstream reads config.json with the family's own typed config (for example
//! `zig/src/families/qwen3_5/config.zig`) and takes its sampling defaults from
//! generation_config.json ("Sampling defaults come from the checkpoint's
//! generation_config.json, then serve flags and request fields override them" — README).
//! The typed per-family configs port with the families in Phase 1; this module owns the
//! folder-level reads they will share.

use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ConfigError {
    Io(PathBuf, std::io::Error),
    /// The config is not a JSON object.
    BadShape,
    /// No `model_type` field, so no family can gate on it.
    NoModelType,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(path, err) => write!(f, "{}: {}", path.display(), err),
            ConfigError::BadShape => f.write_str("config.json is not a JSON object"),
            ConfigError::NoModelType => f.write_str("config.json names no model_type"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// The model's config.json plus its optional generation_config.json.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// The family gate: `config.json`'s `model_type` (engine_api.zig:160 "model_type").
    pub model_type: String,
    /// The whole config.json, for the family's typed read.
    pub json: Value,
    /// generation_config.json when the folder holds one; the server's sampling defaults.
    pub generation: Option<Value>,
}

impl ModelConfig {
    pub fn open(dir: &Path) -> Result<ModelConfig, ConfigError> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| ConfigError::Io(path.clone(), e))?;
        let json: Value =
            serde_json::from_str(&text).map_err(|_| ConfigError::BadShape)?;
        if !json.is_object() {
            return Err(ConfigError::BadShape);
        }
        let model_type = json
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or(ConfigError::NoModelType)?
            .to_string();
        let generation = {
            let path = dir.join("generation_config.json");
            match std::fs::read_to_string(&path) {
                Ok(text) => serde_json::from_str(&text).ok(),
                Err(_) => None,
            }
        };
        Ok(ModelConfig {
            model_type,
            json,
            generation,
        })
    }

    /// A sampling default from generation_config.json, when the field is there.
    /// Upstream's precedence: checkpoint defaults, then serve flags, then request
    /// fields; the server keeps that order when it merges.
    pub fn generation_number(&self, field: &str) -> Option<f64> {
        self.generation
            .as_ref()
            .and_then(|g| g.get(field))
            .and_then(Value::as_f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_model_type_and_generation_defaults() {
        let dir = std::env::temp_dir().join("tf-config-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "qwen3_5", "hidden_size": 2048}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.6, "top_p": 0.95}"#,
        )
        .unwrap();
        let cfg = ModelConfig::open(&dir).unwrap();
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.json["hidden_size"], 2048);
        assert_eq!(cfg.generation_number("temperature"), Some(0.6));
        assert_eq!(cfg.generation_number("top_k"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
