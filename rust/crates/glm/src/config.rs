//! GLM-5.3-Flash's shape from config.json's text_config: the engine's
//! kernels are generated for exactly this shape.
//!
//! Port of `zig/src/families/glm/config.zig` (287 lines) — the
//! pinned-shape Config with its parallel splits, parse's exact-shape
//! refusal, and all five of upstream's tests mirrored.

use serde_json::Value;

pub const MAX_LAYERS: usize = 46;

/// Upstream: `Kind` (glm/config.zig:5).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Kda,
    Mla,
}

/// List entries a decode attention block takes: the same at every
/// window width, so every row's sums keep one order. Upstream:
/// `attn_span` (glm/config.zig:8).
pub const ATTN_SPAN: u32 = 128;

/// Upstream: `Config` (glm/config.zig:10) — the defaults are the
/// exactly-pinned GLM-5.3-Flash shape.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: u32,
    pub layers: u32,
    /// Backbone layers loaded and run: all, or the first few for a
    /// check that must fit (GLM_LAYERS).
    pub run: u32,
    pub vocab: u32,
    pub eps: f32,
    pub dense_layers: u32,
    /// This Mac's dense MLP rows: all, or its half in TP2.
    pub dense_inter: u32,
    pub experts: u32,
    /// Routed experts [lo, hi) this Mac holds: all, or its half in
    /// expert-parallel mode.
    pub own: [u32; 2],
    /// Each routed expert's intermediate rows [lo, hi) this Mac holds
    /// (half by rows).
    pub inter: [u32; 2],
    pub topk: u32,
    pub moe_inter: u32,
    pub routed_scale: f32,
    pub swiglu_limit: f32,
    /// This Mac's KDA heads: all, or its share in TP2.
    pub kda_heads: u32,
    pub kda_dim: u32,
    pub conv: u32,
    pub lower_bound: f32,
    /// This Mac's MLA heads: all, or its share in TP2.
    pub mla_heads: u32,
    pub nope: u32,
    pub v_dim: u32,
    pub q_lora: u32,
    pub kv_lora: u32,
    pub i_heads: u32,
    pub i_dim: u32,
    pub i_topk: u32,
    pub kpool: u32,
    pub hc: u32,
    pub hc_eps: f32,
    pub sinkhorn: u32,
    pub mtp: u32,
    /// Tensor parallel: the Macs splitting the attention heads.
    pub tp: u32,
    pub tp_rank: u32,
    pub eos: [u32; 3],
    pub eos_n: u32,
    /// Layer i is MLA (sparse attention) when true; KDA otherwise.
    pub mla: [bool; MAX_LAYERS],
}

impl Default for Config {
    fn default() -> Config {
        Config {
            hidden: 4096,
            layers: 45,
            run: 45,
            vocab: 154880,
            eps: 1e-5,
            dense_layers: 3,
            dense_inter: 12288,
            experts: 288,
            own: [0, 288],
            inter: [0, 2048],
            topk: 8,
            moe_inter: 2048,
            routed_scale: 2.5,
            swiglu_limit: 10.0,
            kda_heads: 64,
            kda_dim: 128,
            conv: 4,
            lower_bound: -5.0,
            mla_heads: 64,
            nope: 256,
            v_dim: 256,
            q_lora: 1536,
            kv_lora: 512,
            i_heads: 32,
            i_dim: 128,
            i_topk: 2048,
            kpool: 4,
            hc: 4,
            hc_eps: 1e-6,
            sinkhorn: 20,
            mtp: 1,
            tp: 1,
            tp_rank: 0,
            eos: [154820, 154827, 154829],
            eos_n: 3,
            mla: default_mla(),
        }
    }
}

fn default_mla() -> [bool; MAX_LAYERS] {
    let mut s = [false; MAX_LAYERS];
    let mut l = 3;
    while l < 45 {
        s[l] = true;
        l += 4;
    }
    s[45] = true; // the MTP layer
    s
}

/// Why parse or a split refused.
#[derive(Debug, PartialEq)]
pub enum ConfigError {
    BadConfig,
    NotGlm53Flash,
    BadLayerCount,
    BadExpertSplit,
    BadHeadSplit,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::BadConfig => write!(f, "config.json does not hold this field"),
            ConfigError::NotGlm53Flash => {
                write!(f, "the checkpoint is not GLM-5.3-Flash's shape")
            }
            ConfigError::BadLayerCount => write!(f, "the layer count is out of range"),
            ConfigError::BadExpertSplit => write!(f, "the expert split does not divide the shape"),
            ConfigError::BadHeadSplit => write!(f, "the head split does not divide the shape"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Upstream: `kind` (glm/config.zig:51).
    pub fn kind(&self, i: u32) -> Kind {
        if self.mla[i as usize] {
            Kind::Mla
        } else {
            Kind::Kda
        }
    }

    /// Expert parallel by rows: this Mac holds part of every routed
    /// expert's intermediate rows. Upstream: `byRows` (glm/config.zig:54).
    pub fn by_rows(&self) -> bool {
        self.inter[1] - self.inter[0] != self.moe_inter
    }

    /// Upstream: `isMoe` (glm/config.zig:59).
    pub fn is_moe(&self, i: u32) -> bool {
        i >= self.dense_layers
    }

    /// Upstream: `kdaWidth` (glm/config.zig:63).
    pub fn kda_width(&self) -> u32 {
        self.kda_heads * self.kda_dim
    }

    /// The stacked KDA input projection's rows: q, k, v, f_a, g_a, b.
    /// Upstream: `kdaProj` (glm/config.zig:67).
    pub fn kda_proj(&self) -> u32 {
        3 * self.kda_width() + 2 * self.kda_dim + self.kda_heads
    }

    /// The stacked MLA projection of the layer input: q_a, kv_a, the
    /// indexer's keys and head weights. Upstream: `xProj` (glm/config.zig:72).
    pub fn x_proj(&self) -> u32 {
        self.q_lora + self.kv_lora + self.i_dim + self.i_heads
    }

    /// The stacked projection of the normed q_a: q_b and the
    /// indexer's queries. Upstream: `qrProj` (glm/config.zig:77).
    pub fn qr_proj(&self) -> u32 {
        self.mla_heads * self.nope + self.i_heads * self.i_dim
    }

    /// Keys a sparse row attends: index_topk from whole blocks plus
    /// the tail of its partial block. Upstream: `keyWidth`
    /// (glm/config.zig:82).
    pub fn key_width(&self) -> u32 {
        self.i_topk + self.kpool - 1
    }

    /// A decode row's key list in blocks of `ATTN_SPAN`, each block's
    /// softmax its own, combined in block order. Upstream:
    /// `attnBlocks` (glm/config.zig:87).
    pub fn attn_blocks(&self) -> u32 {
        self.key_width().div_ceil(ATTN_SPAN)
    }

    /// The head's rows [lo, hi) this Mac computes: the whole
    /// vocabulary, or its half in TP2. Upstream: `vocabPart`
    /// (glm/config.zig:92).
    pub fn vocab_part(&self) -> [u32; 2] {
        let n = self.vocab / self.tp;
        [
            self.tp_rank * n,
            if self.tp_rank + 1 == self.tp {
                self.vocab
            } else {
                (self.tp_rank + 1) * n
            },
        ]
    }

    /// Upstream: `countKind` (glm/config.zig:98).
    pub fn count_kind(&self, k: Kind) -> u32 {
        (0..self.layers).filter(|&i| self.kind(i) == k).count() as u32
    }

    /// Dense index of layer i among the layers of its kind (state
    /// arenas are per kind). Upstream: `kindIndex` (glm/config.zig:104).
    pub fn kind_index(&self, i: u32) -> u32 {
        let mine = self.kind(i);
        (0..i).filter(|&j| self.kind(j) == mine).count() as u32
    }
}

/// The checkpoint's text_config, refused unless it is GLM-5.3-Flash's
/// shape (the kernels are built for it). Upstream: `parse`
/// (glm/config.zig:138).
pub fn parse(json: &str) -> Result<Config, ConfigError> {
    let parsed: Value = serde_json::from_str(json).map_err(|_| ConfigError::BadConfig)?;
    let root = parsed.as_object().ok_or(ConfigError::BadConfig)?;
    let t = match root.get("text_config").and_then(|v| v.as_object()) {
        Some(t) => t,
        None => root,
    };
    let mut c = Config::default();
    let want = c.clone();
    let mut la_error = false;
    let int = |t: &serde_json::Map<String, Value>, key: &str| -> Result<i64, ConfigError> {
        t.get(key)
            .and_then(|v| v.as_i64())
            .ok_or(ConfigError::BadConfig)
    };
    let float = |t: &serde_json::Map<String, Value>, key: &str| -> Result<f64, ConfigError> {
        t.get(key)
            .and_then(|v| v.as_f64())
            .ok_or(ConfigError::BadConfig)
    };
    c.hidden = int(t, "hidden_size")? as u32;
    c.layers = int(t, "num_hidden_layers")? as u32;
    c.run = c.layers;
    c.vocab = int(t, "vocab_size")? as u32;
    c.eps = float(t, "rms_norm_eps")? as f32;
    c.dense_layers = int(t, "first_k_dense_replace")? as u32;
    c.dense_inter = int(t, "intermediate_size")? as u32;
    c.experts = int(t, "n_routed_experts")? as u32;
    c.own = [0, c.experts];
    c.inter = [0, c.moe_inter];
    c.topk = int(t, "num_experts_per_tok")? as u32;
    c.moe_inter = int(t, "moe_intermediate_size")? as u32;
    c.routed_scale = float(t, "routed_scaling_factor")? as f32;
    c.swiglu_limit = float(t, "swiglu_limit")? as f32;
    c.mla_heads = int(t, "num_attention_heads")? as u32;
    c.nope = int(t, "qk_nope_head_dim")? as u32;
    c.v_dim = int(t, "v_head_dim")? as u32;
    c.q_lora = int(t, "q_lora_rank")? as u32;
    c.kv_lora = int(t, "kv_lora_rank")? as u32;
    c.i_heads = int(t, "index_n_heads")? as u32;
    c.i_dim = int(t, "index_head_dim")? as u32;
    c.i_topk = int(t, "index_topk")? as u32;
    c.kpool = int(t, "index_kpool")? as u32;
    c.hc = int(t, "hc_mult")? as u32;
    c.hc_eps = float(t, "hc_eps")? as f32;
    c.sinkhorn = int(t, "hc_sinkhorn_iters")? as u32;
    c.mtp = int(t, "num_nextn_predict_layers")? as u32;
    let la = t
        .get("linear_attn_config")
        .and_then(|v| v.as_object())
        .ok_or(ConfigError::BadConfig)?;
    c.kda_heads = int(la, "num_heads")? as u32;
    c.kda_dim = int(la, "head_dim")? as u32;
    c.conv = int(la, "short_conv_kernel_size")? as u32;
    c.lower_bound = float(la, "gate_lower_bound")? as f32;
    c.mla = [false; MAX_LAYERS];
    let layers = la
        .get("full_attn_layers")
        .and_then(|v| v.as_array())
        .ok_or(ConfigError::BadConfig)?;
    for v in layers {
        let i = v.as_i64().unwrap_or(-1);
        if !(0..MAX_LAYERS as i64).contains(&i) {
            return Err(ConfigError::BadConfig);
        }
        c.mla[i as usize] = true;
    }
    if c.layers as usize + c.mtp as usize > MAX_LAYERS {
        return Err(ConfigError::BadConfig);
    }
    if c.mtp > 0 {
        c.mla[c.layers as usize] = true;
    }
    if let Some(v) = t.get("eos_token_id").and_then(|v| v.as_array()) {
        if v.len() <= 3 {
            c.eos_n = v.len() as u32;
            for (i, e) in v.iter().enumerate() {
                c.eos[i] = e.as_i64().unwrap_or(0) as u32;
            }
        }
    }
    let rope = t.get("qk_rope_head_dim");
    let plain = t.get("n_group").and_then(|v| v.as_i64()).unwrap_or(1) == 1
        && t.get("topk_group").and_then(|v| v.as_i64()).unwrap_or(1) == 1;
    let _ = &mut la_error;
    let same = c.hidden == want.hidden
        && c.layers == want.layers
        && c.vocab == want.vocab
        && c.dense_layers == want.dense_layers
        && c.dense_inter == want.dense_inter
        && c.experts == want.experts
        && c.topk == want.topk
        && c.moe_inter == want.moe_inter
        && c.mla_heads == want.mla_heads
        && c.nope == want.nope
        && c.v_dim == want.v_dim
        && c.q_lora == want.q_lora
        && c.kv_lora == want.kv_lora
        && c.i_heads == want.i_heads
        && c.i_dim == want.i_dim
        && c.i_topk == want.i_topk
        && c.kpool == want.kpool
        && c.hc == want.hc
        && c.sinkhorn == want.sinkhorn
        && c.kda_heads == want.kda_heads
        && c.kda_dim == want.kda_dim
        && c.conv == want.conv
        && c.mla == want.mla
        && plain
        && (rope.is_none() || rope.and_then(|v| v.as_i64()) == Some(0));
    if !same {
        return Err(ConfigError::NotGlm53Flash);
    }
    Ok(c)
}

/// The first `n` backbone layers only (the MTP layer stays): a check
/// whose weights must fit where the whole model does not. Upstream:
/// `subset` (glm/config.zig:202).
pub fn subset(c: &mut Config, n: u32) -> Result<(), ConfigError> {
    if n == 0 || n > c.layers {
        return Err(ConfigError::BadLayerCount);
    }
    c.run = n;
    Ok(())
}

/// Expert parallel by rows over `ranks` Macs: rank r holds every
/// routed expert's intermediate rows [r n, (r + 1) n). Upstream:
/// `splitRows` (glm/config.zig:208).
pub fn split_rows(c: &mut Config, rank: u32, ranks: u32) -> Result<(), ConfigError> {
    if ranks == 0 || rank >= ranks || c.moe_inter % (64 * ranks) != 0 {
        return Err(ConfigError::BadExpertSplit);
    }
    let n = c.moe_inter / ranks;
    c.own = [0, c.experts];
    c.inter = [rank * n, (rank + 1) * n];
    Ok(())
}

/// Tensor parallel over `ranks` Macs: rank r runs its share of every
/// layer's KDA and MLA heads and dense MLP rows. Upstream:
/// `splitHeads` (glm/config.zig:216).
pub fn split_heads(c: &mut Config, rank: u32, ranks: u32) -> Result<(), ConfigError> {
    if ranks == 0 || rank >= ranks || c.tp != 1 {
        return Err(ConfigError::BadHeadSplit);
    }
    if c.kda_heads % ranks != 0 || c.mla_heads % ranks != 0 || c.dense_inter % (64 * ranks) != 0 {
        return Err(ConfigError::BadHeadSplit);
    }
    c.kda_heads /= ranks;
    c.mla_heads /= ranks;
    c.dense_inter /= ranks;
    c.tp = ranks;
    c.tp_rank = rank;
    Ok(())
}

/// Expert parallel over `ranks` Macs: rank r holds routed experts
/// [r * experts / ranks, (r + 1) * experts / ranks). Upstream:
/// `split` (glm/config.zig:227).
pub fn split(c: &mut Config, rank: u32, ranks: u32) -> Result<(), ConfigError> {
    if ranks == 0 || rank >= ranks || c.experts % ranks != 0 {
        return Err(ConfigError::BadExpertSplit);
    }
    let per = c.experts / ranks;
    c.own = [rank * per, (rank + 1) * per];
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `GLM-5.3-Flash's layer kinds: 34 KDA and 11 MLA in
    /// the backbone, the MTP layer MLA` (glm/config.zig:234).
    #[test]
    fn glm_53_flash_layer_kinds_34_kda_and_11_mla_in_the_backbone_the_mtp_layer_mla() {
        let c = Config::default();
        assert_eq!(34, c.count_kind(Kind::Kda));
        assert_eq!(11, c.count_kind(Kind::Mla));
        assert_eq!(Kind::Mla, c.kind(3));
        assert_eq!(Kind::Mla, c.kind(43));
        assert_eq!(Kind::Kda, c.kind(44));
        assert_eq!(Kind::Mla, c.kind(45));
        assert_eq!(24896, c.kda_proj());
        assert_eq!(2208, c.x_proj());
        assert_eq!(20480, c.qr_proj());
        assert_eq!(10, c.kind_index(43));
    }

    /// Upstream's `a layer subset keeps the MTP layer's place and
    /// refuses an empty or oversized count` (glm/config.zig:248).
    #[test]
    fn a_layer_subset_keeps_the_mtp_layers_place_and_refuses_an_empty_or_oversized_count() {
        let mut c = Config::default();
        subset(&mut c, 8).unwrap();
        assert_eq!(8, c.run);
        assert_eq!(45, c.layers);
        assert_eq!(Err(ConfigError::BadLayerCount), subset(&mut c, 0));
        assert_eq!(Err(ConfigError::BadLayerCount), subset(&mut c, 46));
    }

    /// Upstream's `by rows, two ranks hold every expert's halves`
    /// (glm/config.zig:257).
    #[test]
    fn by_rows_two_ranks_hold_every_experts_halves() {
        let mut c = Config::default();
        split_rows(&mut c, 1, 2).unwrap();
        assert_eq!([0, 288], c.own);
        assert_eq!([1024, 2048], c.inter);
        assert!(c.by_rows());
        assert!(!Config::default().by_rows());
    }

    /// Upstream's `TP2: each rank runs half the KDA heads; its
    /// projection keeps f_a and g_a whole` (glm/config.zig:266).
    #[test]
    fn tp2_each_rank_runs_half_the_kda_heads_its_projection_keeps_fa_and_ga_whole() {
        let mut c = Config::default();
        split_heads(&mut c, 1, 2).unwrap();
        assert_eq!(32, c.kda_heads);
        assert_eq!(4096, c.kda_width());
        assert_eq!(12576, c.kda_proj());
        assert_eq!([77440, 154880], c.vocab_part());
        assert_eq!(12288, c.qr_proj());
        assert_eq!(Err(ConfigError::BadHeadSplit), split_heads(&mut c, 0, 2));
        let mut d = Config::default();
        assert_eq!(Err(ConfigError::BadHeadSplit), split_heads(&mut d, 2, 2));
    }

    /// Upstream's `two ranks hold the routed experts' halves`
    /// (glm/config.zig:279).
    #[test]
    fn two_ranks_hold_the_routed_experts_halves() {
        let mut c = Config::default();
        split(&mut c, 0, 2).unwrap();
        assert_eq!([0, 144], c.own);
        split(&mut c, 1, 2).unwrap();
        assert_eq!([144, 288], c.own);
        assert_eq!(Err(ConfigError::BadExpertSplit), split(&mut c, 2, 2));
        assert_eq!(Err(ConfigError::BadExpertSplit), split(&mut c, 0, 5));
    }
}
