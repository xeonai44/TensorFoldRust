//! Nemotron-H dimensions from config.json, one Config for every
//! backend, checked against the shapes our kernels serve.
//!
//! Port of `zig/src/families/nemotron/config.zig` (344 lines). The
//! refusal reasons are byte-exact (upstream's tests read them); the
//! quantization entries read as MLX's `nn.quantize` reads them.

use crate::error::Error;
use serde_json::Value;
use std::fmt::Write as _;

/// Upstream: `Kind` (config.zig:4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Mamba,
    Moe,
    Attention,
}

pub const MAX_LAYERS: usize = 64;

/// The MTP head's acceptance at depth 1, 2, ... given the ones before
/// it, until a stream has its own (model.py draft_prior). Upstream:
/// `draft_prior` (config.zig:8).
pub const DRAFT_PRIOR: [f64; 8] = [0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5];

/// Why a check refused the checkpoint, kept for the caller's log line
/// (tests read it instead). Upstream: `Why` (config.zig:12).
pub struct Why {
    buf: [u8; 320],
    len: usize,
}

impl Why {
    pub fn new() -> Why {
        Why {
            buf: [0; 320],
            len: 0,
        }
    }

    /// Keep the reason, cut short if it is long. Upstream: `set`
    /// (config.zig:17).
    pub fn set(&mut self, args: std::fmt::Arguments<'_>) {
        // A fixed buffer: format into a stack string, then keep what
        // fits (upstream's fixed Io.Writer).
        let mut scratch = [0u8; 320];
        let mut w = Wrap { buf: &mut scratch, at: 0 };
        let _ = write!(w, "{args}");
        self.len = w.at.min(320);
        self.buf[..self.len].copy_from_slice(&scratch[..self.len]);
    }

    pub fn text(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

impl Default for Why {
    fn default() -> Why {
        Why::new()
    }
}

/// A tiny fmt::Write over a fixed buffer: the truncation of upstream's
/// `.fixed` writer.
struct Wrap<'a> {
    buf: &'a mut [u8],
    at: usize,
}

impl std::fmt::Write for Wrap<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        let take = bytes.len().min(self.buf.len() - self.at);
        self.buf[self.at..self.at + take].copy_from_slice(&bytes[..take]);
        self.at += take;
        Ok(())
    }
}

/// The JSON value printed as the config's own text (upstream's
/// `std.json.fmt`).
fn json_text(v: &Value) -> String {
    v.to_string()
}

/// Upstream: `Config` (config.zig:28).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    /// The per-block kinds; only `[0..layers]` is read (Zig leaves the
    /// tail undefined — the port fills it with a harmless first value).
    pub kinds: [Kind; MAX_LAYERS],
    pub mamba_heads: usize,
    pub mamba_head_dim: usize,
    pub groups: usize,
    pub state: usize,
    pub conv_kernel: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_width: usize,
    pub shared_width: usize,
    pub routed_scaling: f32,
    pub norm_topk: bool,
    pub eps: f32,
    /// time_step_limit: the Mamba dt clamp, (0, inf) when the config
    /// sets none.
    pub dt_min: f32,
    pub dt_max: f32,
    pub group_size: usize,
    pub bits: usize,
    pub eos: [u32; 4],
    pub eos_count: usize,
}

impl Config {
    /// Upstream: `inner` (config.zig:55).
    pub fn inner(self) -> usize {
        self.mamba_heads * self.mamba_head_dim
    }

    /// Upstream: `convDim` (config.zig:59).
    pub fn conv_dim(self) -> usize {
        self.inner() + 2 * self.groups * self.state
    }

    /// Upstream: `projDim` (config.zig:62).
    pub fn proj_dim(self) -> usize {
        self.inner() + self.conv_dim() + self.mamba_heads
    }

    /// Upstream: `qkvDim` (config.zig:66).
    pub fn qkv_dim(self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    /// A token's expert pairs: its routed experts, then the shared
    /// expert's two halves. Upstream: `slots` (config.zig:71).
    pub fn slots(self) -> usize {
        self.top_k + 2
    }

    /// Upstream: `count` (config.zig:76).
    pub fn count(self, kind: Kind) -> usize {
        self.kinds[..self.layers]
            .iter()
            .filter(|k| **k == kind)
            .count()
    }

    /// Upstream: `isEos` (config.zig:82).
    pub fn is_eos(self, token: u32) -> bool {
        self.eos[..self.eos_count].contains(&token)
    }

    /// `dir`/config.json. Upstream: `read` (config.zig:88).
    pub fn read(dir: &str) -> Result<Config, Error> {
        let path = std::path::Path::new(dir).join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|_| Error::BadConfig)?;
        if text.len() > 1 << 22 {
            return Err(Error::BadConfig);
        }
        let mut why = Why::new();
        parse(&text, &mut why).inspect_err(|_| {
            if !why.text().is_empty() {
                eprintln!("{}: {}", path.display(), String::from_utf8_lossy(why.text()));
            }
        })
    }
}

/// Upstream: `int` (config.zig:101).
fn int(obj: &serde_json::Map<String, Value>, key: &str) -> Result<usize, Error> {
    match obj.get(key).and_then(|v| v.as_i64()) {
        Some(v) => Ok(v as usize),
        None => {
            eprintln!("config.json has no {key}");
            Err(Error::BadConfig)
        }
    }
}

/// Upstream: `number` (config.zig:109).
fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64().ok_or(Error::BadConfig)
}

/// Upstream: `float` (config.zig:117) — uncalled there too; kept for
/// the family's later tranches that read config floats by key.
#[allow(dead_code)]
fn float(obj: &serde_json::Map<String, Value>, key: &str) -> Result<f32, Error> {
    number(obj.get(key).ok_or(Error::BadConfig)?).map(|f| f as f32)
}

/// Upstream: `kindOf` (config.zig:121).
fn kind_of(name: &str) -> Result<Kind, Error> {
    match name {
        "mamba" | "M" => Ok(Kind::Mamba),
        "moe" | "E" => Ok(Kind::Moe),
        "attention" | "*" => Ok(Kind::Attention),
        _ => Err(Error::UnsupportedBlock),
    }
}

/// The quantization block as MLX reads it: "quantization", else the
/// "quantization_config" that mirrors it. Upstream: `quantBlock`
/// (config.zig:129).
fn quant_block(o: &serde_json::Map<String, Value>) -> Option<&serde_json::Map<String, Value>> {
    for key in ["quantization", "quantization_config"] {
        if let Some(v) = o.get(key) {
            if let Some(m) = v.as_object() {
                if !m.is_empty() {
                    return Some(m);
                }
            }
        }
    }
    None
}

/// A quantization block's own fields; every other key names a module.
/// Upstream: `blockField` (config.zig:137).
fn block_field(key: &str) -> bool {
    ["bits", "group_size", "mode", "quant_method"].contains(&key)
}

/// MLX's affine mode, the one an entry has when it names none.
/// Upstream: `affine` (config.zig:143).
fn affine(o: &serde_json::Map<String, Value>) -> bool {
    match o.get("mode") {
        None => true,
        Some(Value::Null) => true,
        Some(m) => match m.as_str() {
            Some(s) => s.is_empty() || s.eq_ignore_ascii_case("affine"),
            None => false,
        },
    }
}

/// A positive integer field, `default` when absent or null, 0 when
/// unreadable. Upstream: `width` (config.zig:149).
fn width(o: &serde_json::Map<String, Value>, key: &str, default: usize) -> usize {
    match o.get(key) {
        None => default,
        Some(Value::Null) => default,
        Some(v) => match v.as_i64() {
            Some(i) if i > 0 => i as usize,
            _ => 0,
        },
    }
}

/// The checkpoint's bits and group size; a per-module entry at any
/// other width is refused, as the kernels read one. Upstream:
/// `quantization` (config.zig:159).
fn quantization(c: &mut Config, o: &serde_json::Map<String, Value>, why: &mut Why) -> Result<(), Error> {
    let q = match quant_block(o) {
        Some(q) => q,
        None => {
            why.set(format_args!("config.json declares no quantization; the native Nemotron kernels read MLX affine checkpoints"));
            return Err(Error::UnsupportedQuantization);
        }
    };
    if let Some(m) = q.get("quant_method") {
        let ok = m.is_null()
            || m.as_str()
                .is_some_and(|s| s.eq_ignore_ascii_case("mlx") || s.eq_ignore_ascii_case("affine"));
        if !ok {
            why.set(format_args!(
                "config.json's quant_method is {}; the native Nemotron kernels read MLX affine checkpoints",
                json_text(m)
            ));
            return Err(Error::UnsupportedQuantization);
        }
    }
    if !affine(q) {
        why.set(format_args!(
            "config.json quantizes in mode {}; the native Nemotron kernels read MLX affine checkpoints",
            json_text(q.get("mode").unwrap())
        ));
        return Err(Error::UnsupportedQuantization);
    }
    c.bits = width(q, "bits", 0);
    c.group_size = width(q, "group_size", 64);
    if c.bits == 0 || c.group_size == 0 {
        why.set(format_args!("config.json's quantization has no readable bits or group_size"));
        return Err(Error::UnsupportedQuantization);
    }
    for (key, value) in q {
        if !block_field(key) {
            module(&*c, key, value, why)?;
        }
    }
    Ok(())
}

/// A per-module entry as nn.quantize reads it: true, or a dict at the
/// checkpoint's width, passes; anything else is refused. Upstream:
/// `module` (config.zig:183) — the exact fall-through shape: bool
/// false (unless learned), an empty dict and null all fall to the
/// unquantized refusal; other JSON values pass silently.
fn module(c: &Config, path: &str, v: &Value, why: &mut Why) -> Result<(), Error> {
    match v {
        Value::Bool(on) => {
            if *on || learned(path) {
                return Ok(());
            }
        }
        Value::Object(m) => {
            if !m.is_empty() {
                // a dict's missing bits or group_size take
                // to_quantized's affine defaults, not the top level's
                let bits = width(m, "bits", 4);
                let group = width(m, "group_size", 64);
                if affine(m) && bits == c.bits && group == c.group_size {
                    return Ok(());
                }
                if !affine(m) || bits == 0 || group == 0 {
                    why.set(format_args!(
                        "config.json quantizes {path} as {}; the native Nemotron kernels read every matrix at {}-bit in groups of {}",
                        json_text(v),
                        c.bits,
                        c.group_size
                    ));
                } else {
                    why.set(format_args!(
                        "config.json quantizes {path} at {bits}-bit in groups of {group}; the native Nemotron kernels read every matrix at the checkpoint's {}-bit in groups of {}",
                        c.bits,
                        c.group_size
                    ));
                }
                return Err(Error::MixedQuantization);
            }
        }
        Value::Null => {}
        _ => return Ok(()),
    }
    why.set(format_args!(
        "config.json leaves {path} unquantized; the native Nemotron kernels read every matrix at {}-bit in groups of {}",
        c.bits,
        c.group_size
    ));
    Err(Error::MixedQuantization)
}

/// An output projection Sliding Weights wrote in bf16: the loader reads
/// it as codes plus a residual. Upstream: `learned` (config.zig:206).
fn learned(path: &str) -> bool {
    if !path.starts_with("backbone.layers.") {
        return false;
    }
    [".mixer.shared_experts.down_proj", ".mixer.out_proj", ".mixer.o_proj"]
        .iter()
        .any(|leaf| path.ends_with(leaf))
}

/// config.json's text; a refusal's reason goes to `why`. Upstream:
/// `parse` (config.zig:213).
pub fn parse(text: &str, why: &mut Why) -> Result<Config, Error> {
    let parsed: Value = serde_json::from_str(text).map_err(|_| Error::BadConfig)?;
    let o = parsed
        .as_object()
        .ok_or(Error::BadConfig)?;
    if o.get("model_type").and_then(|v| v.as_str()) != Some("nemotron_h") {
        return Err(Error::NotNemotronH);
    }
    let mut c = Config {
        hidden: int(o, "hidden_size")?,
        vocab: int(o, "vocab_size")?,
        layers: 0,
        kinds: [Kind::Mamba; MAX_LAYERS],
        mamba_heads: int(o, "mamba_num_heads")?,
        mamba_head_dim: int(o, "mamba_head_dim")?,
        groups: int(o, "n_groups")?,
        state: int(o, "ssm_state_size")?,
        conv_kernel: int(o, "conv_kernel")?,
        heads: int(o, "num_attention_heads")?,
        kv_heads: int(o, "num_key_value_heads")?,
        head_dim: 0,
        experts: int(o, "n_routed_experts")?,
        top_k: int(o, "num_experts_per_tok")?,
        expert_width: int(o, "moe_intermediate_size")?,
        shared_width: int(o, "moe_shared_expert_intermediate_size")?,
        routed_scaling: match o.get("routed_scaling_factor") {
            Some(v) => number(v)? as f32,
            None => 1.0,
        },
        norm_topk: true,
        eps: match o.get("layer_norm_epsilon") {
            Some(v) => number(v)? as f32,
            None => 1e-5,
        },
        dt_min: 0.0,
        dt_max: f32::INFINITY,
        group_size: 64,
        bits: 4,
        eos: [0; 4],
        eos_count: 0,
    };
    c.head_dim = match o.get("head_dim") {
        Some(v) => v.as_i64().ok_or(Error::BadConfig)? as usize,
        None => c.hidden / c.heads,
    };
    if let Some(v) = o.get("norm_topk_prob") {
        c.norm_topk = v.as_bool().unwrap_or(true);
    }
    // hybrid_override_pattern (one character a block) wins over
    // layers_block_type, as the Python loader reads them
    if let Some(v) = o.get("hybrid_override_pattern") {
        if let Some(pattern) = v.as_str() {
            if !pattern.is_empty() {
                if pattern.len() > MAX_LAYERS {
                    return Err(Error::BadConfig);
                }
                for (i, ch) in pattern.bytes().enumerate() {
                    c.kinds[i] = kind_of(std::str::from_utf8(&[ch]).unwrap_or(""))?;
                }
                c.layers = pattern.len();
            }
        }
    }
    if c.layers == 0 {
        let blocks = o
            .get("layers_block_type")
            .and_then(|v| v.as_array())
            .ok_or(Error::BadConfig)?;
        if blocks.len() > MAX_LAYERS {
            return Err(Error::BadConfig);
        }
        for (i, b) in blocks.iter().enumerate() {
            c.kinds[i] = kind_of(b.as_str().ok_or(Error::BadConfig)?)?;
        }
        c.layers = blocks.len();
    }
    match o.get("eos_token_id") {
        Some(v) if v.as_i64().is_some() => {
            c.eos[0] = v.as_i64().unwrap() as u32;
            c.eos_count = 1;
        }
        Some(Value::Array(a)) => {
            for x in a.iter().take(4) {
                c.eos[c.eos_count] = x.as_i64().ok_or(Error::BadConfig)? as u32;
                c.eos_count += 1;
            }
        }
        _ => {}
    }
    if let Some(v) = o.get("time_step_limit") {
        if let Some(a) = v.as_array() {
            if a.len() == 2 {
                c.dt_min = number(&a[0])? as f32;
                c.dt_max = number(&a[1])? as f32;
            }
        }
    }
    quantization(&mut c, o, why)?;
    Ok(c)
}

/// The shapes our kernels serve (Nemotron 3.5 Lightning 30B-A3B, 4-bit
/// groups of 64): Metal's generated sources, CUDA's captured cubins.
/// Upstream: `checkShapes` (config.zig:271).
pub fn check_shapes(c: Config) -> Result<(), Error> {
    let ok = c.hidden == 2688
        && c.vocab == 131072
        && c.mamba_heads == 64
        && c.mamba_head_dim == 64
        && c.groups == 8
        && c.state == 128
        && c.conv_kernel == 4
        && c.heads == 32
        && c.kv_heads == 2
        && c.head_dim == 128
        && c.experts == 128
        && c.top_k == 6
        && c.expert_width == 1856
        && c.shared_width == 3712
        && c.group_size == 64
        && c.bits == 4;
    if !ok {
        eprintln!("this Nemotron-H's shapes differ from the kernels built for Nemotron 3.5 Lightning 30B-A3B");
        return Err(Error::UnsupportedShapes);
    }
    Ok(())
}

#[cfg(test)]
const LIGHTNING: &str = r#"{"model_type": "nemotron_h", "hidden_size": 2688, "vocab_size": 131072, "mamba_num_heads": 64,
 "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
 "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": 128, "num_experts_per_tok": 6,
 "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712, "routed_scaling_factor": 2.5,
 "layer_norm_epsilon": 1e-05, "eos_token_id": [2, 11], "layers_block_type": ["mamba", "moe", "attention"],
 "quantization": {"group_size": 64, "bits": 4}}"#;

/// The Lightning config with its quantization block's text replaced by
/// `block`. Upstream: `withQuantization` (config.zig:292).
#[cfg(test)]
fn with_quantization(block: &str) -> String {
    LIGHTNING.replace(
        "\"quantization\": {\"group_size\": 64, \"bits\": 4}",
        block,
    )
}

/// parse() must refuse the Lightning config with quantization `block`,
/// its reason naming each of `words`. Upstream: `expectRefused`
/// (config.zig:297).
#[cfg(test)]
fn expect_refused(block: &str, err: Error, words: &[&str]) {
    let text = with_quantization(block);
    let mut why = Why::new();
    let got = match parse(&text, &mut why) {
        Err(e) => e,
        Ok(_) => panic!("the config passed, wanted {err:?}"),
    };
    assert_eq!(
        std::mem::discriminant(&got),
        std::mem::discriminant(&err),
        "refused as {got:?}, wanted {err:?}"
    );
    let text = String::from_utf8_lossy(why.text());
    for word in words {
        assert!(
            text.contains(word),
            "refusal \"{text}\" does not name \"{word}\""
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `per-module quantization entries read as MLX reads
    /// them, any other width refused` (config.zig:308).
    #[test]
    fn per_module_quantization_entries_read_as_mlx_reads_them_any_other_width_refused() {
        let fc1 = "backbone.layers.1.mixer.switch_mlp.fc1";
        for block in [
            "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\", \"lm_head\": true}".to_string(),
            format!("\"quantization\": {{\"bits\": 4, \"{fc1}\": {{\"group_size\": 64, \"bits\": 4}}, \"lm_head\": {{\"mode\": \"affine\"}}}}"),
            "\"quantization_config\": {\"group_size\": 64, \"bits\": 4, \"quant_method\": \"mlx\", \"oq_note\": \"metadata\"}".to_string(),
        ] {
            let text = with_quantization(&block);
            let mut why = Why::new();
            check_shapes(parse(&text, &mut why).unwrap()).unwrap();
        }
        expect_refused(&format!("\"quantization\": {{\"group_size\": 64, \"bits\": 4, \"{fc1}\": {{\"group_size\": 64, \"bits\": 8}}}}"), Error::MixedQuantization, &[fc1, "8-bit in groups of 64"]);
        expect_refused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"backbone.embeddings\": {\"bits\": 6}}", Error::MixedQuantization, &["backbone.embeddings", "6-bit"]);
        expect_refused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"group_size\": 32}}", Error::MixedQuantization, &["lm_head", "4-bit in groups of 32"]);
        expect_refused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": false}", Error::MixedQuantization, &["lm_head", "unquantized"]);
        expect_refused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {}}", Error::MixedQuantization, &["lm_head", "unquantized"]);
        expect_refused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"mode\": \"mxfp8\"}}", Error::MixedQuantization, &["lm_head", "mxfp8"]);
        expect_refused("\"quantization\": {\"group_size\": 32, \"bits\": 4, \"mode\": \"mxfp4\"}", Error::UnsupportedQuantization, &["mxfp4"]);
        expect_refused("\"quantization_config\": {\"quant_method\": \"fp8\", \"weight_block_size\": [128, 128]}", Error::UnsupportedQuantization, &["fp8"]);
        expect_refused("\"quantization\": {\"group_size\": 64}", Error::UnsupportedQuantization, &["bits"]);
        expect_refused("\"tie_word_embeddings\": false", Error::UnsupportedQuantization, &["no quantization"]);
    }

    /// Upstream's `parse the Lightning config` (config.zig:332).
    #[test]
    fn parse_the_lightning_config() {
        let mut why = Why::new();
        let c = parse(LIGHTNING, &mut why).unwrap();
        assert_eq!(3, c.layers);
        assert_eq!(Kind::Attention, c.kinds[2]);
        assert_eq!(10304, c.proj_dim());
        assert_eq!(6144, c.conv_dim());
        assert_eq!(4608, c.qkv_dim());
        assert_eq!(8, c.slots());
        assert!(c.is_eos(11) && !c.is_eos(3));
        assert_eq!(f32::INFINITY, c.dt_max);
        check_shapes(c).unwrap();
    }
}
