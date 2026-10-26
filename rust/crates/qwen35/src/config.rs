//! Admission for the Qwen3.5-2B MLX affine checkpoint: pinned geometry and formats,
//! tied or untied head; other geometries require qualification.
//!
//! Port of `zig/src/families/qwen3_5/config.zig`. Every admission rule stays: the
//! pinned dimensions, the 3:1 linear/full layer pattern, the rotary layout
//! (theta 1e7, partial rotary 0.25, mrope sections [11, 11, 10], interleaved), the
//! 4-bit affine group-64 quantization with no unknown fields, and the tiedness
//! agreement between the root and the language config.

use serde_json::Value;

// Upstream: config.zig:4-18 — the only admitted geometry, as consts.
pub const HIDDEN: usize = 2048;
pub const VOCAB: usize = 248320;
pub const LAYERS: usize = 24;
pub const INTERMEDIATE: usize = 6144;
pub const LINEAR_HEADS: usize = 16;
pub const LINEAR_DIM: usize = 128;
pub const CONV_DIM: usize = 6144;
pub const CONV_TAPS: usize = 4;
pub const QUERY_HEADS: usize = 8;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 256;
pub const ROTARY_DIM: usize = 64;
pub const GROUP: usize = 64;
pub const EPS: f32 = 1e-6;
pub const THETA: f32 = 10000000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub context: usize,
    /// Whether the head is the packed input embedding (true) or a separate lm_head
    /// projection (false).
    pub tied: bool,
}

/// Upstream: config.zig:34  pub fn linear(index) bool — three linear attention
/// layers then one full, by index mod 4.
pub fn linear(index: usize) -> bool {
    index % 4 != 3
}

fn object(v: &Value) -> Result<&serde_json::Map<String, Value>, crate::Error> {
    v.as_object().ok_or(crate::Error::BadQwenConfig)
}

fn field<'a>(o: &'a serde_json::Map<String, Value>, key: &str) -> Result<&'a Value, crate::Error> {
    o.get(key).ok_or(crate::Error::BadQwenConfig)
}

fn number(v: &Value) -> Result<f64, crate::Error> {
    v.as_f64().ok_or(crate::Error::BadQwenConfig)
}

fn integer(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, crate::Error> {
    let v = field(o, key)?;
    // Upstream: `.integer < 1` refuses — only whole dimensions 1 or more.
    let n = v.as_i64().ok_or(crate::Error::BadQwenConfig)?;
    if n < 1 {
        return Err(crate::Error::BadQwenConfig);
    }
    Ok(n as usize)
}

fn expect_string(
    o: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<(), crate::Error> {
    let v = field(o, key)?;
    if v.as_str() != Some(expected) {
        return Err(crate::Error::UnsupportedQwenConfig);
    }
    Ok(())
}

fn expect_bool(
    o: &serde_json::Map<String, Value>,
    key: &str,
    expected: bool,
) -> Result<(), crate::Error> {
    let v = field(o, key)?;
    if v.as_bool() != Some(expected) {
        return Err(crate::Error::UnsupportedQwenConfig);
    }
    Ok(())
}

fn flag(o: &serde_json::Map<String, Value>, key: &str) -> Result<bool, crate::Error> {
    field(o, key)?
        .as_bool()
        .ok_or(crate::Error::BadQwenConfig)
}

fn quantization(v: &Value) -> Result<(), crate::Error> {
    let o = object(v)?;
    if integer(o, "bits")? != 4 || integer(o, "group_size")? != GROUP {
        return Err(crate::Error::UnsupportedQwenQuantization);
    }
    if let Some(mode) = o.get("mode") {
        if mode.as_str() != Some("affine") {
            return Err(crate::Error::UnsupportedQwenQuantization);
        }
    }
    for key in o.keys() {
        if key != "bits" && key != "group_size" && key != "mode" {
            return Err(crate::Error::UnsupportedQwenQuantization);
        }
    }
    Ok(())
}

/// Parse the checkpoint's `config.json` bytes (upstream `parse`, config.zig:90).
pub fn parse(bytes: &[u8]) -> Result<Config, crate::Error> {
    let root_value: Value =
        serde_json::from_slice(bytes).map_err(|_| crate::Error::BadQwenConfig)?;
    let root = object(&root_value)?;
    expect_string(root, "model_type", "qwen3_5")?;
    let text = object(field(root, "text_config")?)?;
    expect_string(text, "model_type", "qwen3_5_text")?;
    // The head's tiedness: the language config decides; a root-level copy must agree.
    let tied = flag(text, "tie_word_embeddings")?;
    if let Some(v) = root.get("tie_word_embeddings") {
        let root_tied = v.as_bool().ok_or(crate::Error::BadQwenConfig)?;
        if root_tied != tied {
            return Err(crate::Error::BadQwenConfig);
        }
    }
    expect_bool(text, "attention_bias", false)?;
    expect_bool(text, "attn_output_gate", true)?;
    expect_string(text, "hidden_act", "silu")?;
    expect_string(text, "mamba_ssm_dtype", "float32")?;
    // Upstream: config.zig:107-114 — the dims table, each exact.
    for (key, value) in [
        ("hidden_size", HIDDEN),
        ("vocab_size", VOCAB),
        ("num_hidden_layers", LAYERS),
        ("intermediate_size", INTERMEDIATE),
        ("linear_num_key_heads", LINEAR_HEADS),
        ("linear_num_value_heads", LINEAR_HEADS),
        ("linear_key_head_dim", LINEAR_DIM),
        ("linear_value_head_dim", LINEAR_DIM),
        ("linear_conv_kernel_dim", CONV_TAPS),
        ("num_attention_heads", QUERY_HEADS),
        ("num_key_value_heads", KV_HEADS),
        ("head_dim", HEAD_DIM),
        ("full_attention_interval", 4),
    ] {
        if integer(text, key)? != value {
            return Err(crate::Error::UnsupportedQwenGeometry);
        }
    }
    if number(field(text, "rms_norm_eps")?)? != 1e-6 {
        return Err(crate::Error::UnsupportedQwenConfig);
    }
    let types = field(text, "layer_types")?;
    let kinds = types
        .as_array()
        .ok_or(crate::Error::UnsupportedQwenGeometry)?;
    if kinds.len() != LAYERS {
        return Err(crate::Error::UnsupportedQwenGeometry);
    }
    for (i, kind) in kinds.iter().enumerate() {
        let expected = if linear(i) { "linear_attention" } else { "full_attention" };
        if kind.as_str() != Some(expected) {
            return Err(crate::Error::UnsupportedQwenGeometry);
        }
    }
    let rope = object(field(text, "rope_parameters")?)?;
    expect_string(rope, "rope_type", "default")?;
    expect_bool(rope, "mrope_interleaved", true)?;
    if number(field(rope, "rope_theta")?)? != THETA as f64
        || number(field(rope, "partial_rotary_factor")?)? != 0.25
    {
        return Err(crate::Error::UnsupportedQwenRotary);
    }
    let sections = field(rope, "mrope_section")?;
    let sections = sections
        .as_array()
        .ok_or(crate::Error::UnsupportedQwenRotary)?;
    if sections.len() != 3 {
        return Err(crate::Error::UnsupportedQwenRotary);
    }
    for (s, expected) in sections.iter().zip([11i64, 11, 10]) {
        if s.as_i64() != Some(expected) {
            return Err(crate::Error::UnsupportedQwenRotary);
        }
    }
    let quant = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
        .ok_or(crate::Error::UnsupportedQwenQuantization)?;
    quantization(quant)?;
    if let Some(q) = root.get("quantization_config") {
        quantization(q)?;
    }
    let context = integer(text, "max_position_embeddings")?;
    if context > 262144 {
        return Err(crate::Error::UnsupportedQwenConfig);
    }
    Ok(Config { context, tied })
}

impl Config {
    /// Upstream: `Config.read` (config.zig:25) — read `dir`'s config.json.
    pub fn read(dir: &std::path::Path) -> Result<Config, crate::Error> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|_| crate::Error::BadQwenConfig)?;
        parse(&bytes)
    }
}

/// Upstream: config.zig:139 — the fixture the admission tests run against, the
/// pinned checkpoint's own config.json shape.
#[cfg(test)]
const FIXTURE: &str = r#"{
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": true,
    "use_cache": true,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [
        11,
        11,
        10
      ],
      "rope_type": "default",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25
    }
  },
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
  },
  "quantization_config": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
  },
  "tie_word_embeddings": true
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors config.zig:224 "admit the affine 2B geometry and text rotary layout":
    /// the fixture admits with its context, and each compact edit the upstream test
    /// makes refuses with the error it names.
    #[test]
    fn admits_the_affine_2b_geometry_and_text_rotary_layout() {
        let config = parse(FIXTURE.as_bytes()).unwrap();
        assert_eq!(config.context, 262144);
        let parsed: Value = serde_json::from_str(FIXTURE).unwrap();
        let compact = serde_json::to_string(&parsed).unwrap();
        let changes: &[(&str, &str, crate::Error)] = &[
            ("\"hidden_size\":2048", "\"hidden_size\":1024", crate::Error::UnsupportedQwenGeometry),
            ("\"bits\":4", "\"bits\":8", crate::Error::UnsupportedQwenQuantization),
            ("\"group_size\":64", "\"group_size\":32", crate::Error::UnsupportedQwenQuantization),
            ("\"mode\":\"affine\"", "\"mode\":\"mxfp4\"", crate::Error::UnsupportedQwenQuantization),
            ("\"rope_theta\":10000000", "\"rope_theta\":100000", crate::Error::UnsupportedQwenRotary),
            ("\"mrope_section\":[11,11,10]", "\"mrope_section\":[16,8,8]", crate::Error::UnsupportedQwenRotary),
            ("\"max_position_embeddings\":262144", "\"max_position_embeddings\":524288", crate::Error::UnsupportedQwenConfig),
        ];
        for (good, bad, err) in changes {
            let target = compact.replace(good, bad);
            // The upstream test checks the good text is present before replacing;
            // a failed replace means the fixture drifted, so fail loudly here.
            assert_ne!(target, compact, "fixture drifted: {}", good);
            assert_eq!(&parse(target.as_bytes()).unwrap_err(), err);
        }
    }

    /// Mirrors config.zig:249: untied heads admit, a root/language tiedness
    /// disagreement is malformed.
    #[test]
    fn admits_untied_heads_and_refuses_a_tiedness_disagreement() {
        // Both declarations flipped: the language config and the root-level copy
        // agree on untied.
        let untied = FIXTURE.replace("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
        let config = parse(untied.as_bytes()).unwrap();
        assert!(!config.tied);
        assert_eq!(config.context, 262144);
        // Only the root flipped: text_config keeps the tie, the root disagrees ->
        // malformed export.
        let mismatch = FIXTURE.replace("\"tie_word_embeddings\": true\n}", "\"tie_word_embeddings\": false\n}");
        assert_eq!(
            parse(mismatch.as_bytes()).unwrap_err(),
            crate::Error::BadQwenConfig
        );
    }

    /// The layer pattern: three linear then one full, six times over 24 layers.
    #[test]
    fn linear_layers_run_three_then_one() {
        for i in 0..LAYERS {
            assert_eq!(linear(i), i % 4 != 3);
        }
        assert_eq!(linear(3), false);
        assert_eq!(linear(23), false);
        assert_eq!(linear(22), true);
    }
}
