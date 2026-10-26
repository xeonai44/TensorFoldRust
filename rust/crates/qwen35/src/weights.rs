//! Views into one shared checkpoint buffer: a tied head aliases the packed input
//! embedding; an untied head is its own projection.
//!
//! Port of `zig/src/families/qwen3_5/weights.zig`. The linear views mirror the
//! affine 4-bit layout the kernels read: `weight` packs 8 nibbles a u32 word
//! (inputs/8 words a row), `scales` and `biases` are bf16 per 64-input group.

use crate::config;
use tensorfold_metal::checkpoint::Checkpoint;

/// The checkpoint's Tensor view, re-exported the way upstream does at weights.zig:6.
pub use tensorfold_metal::checkpoint::Tensor;

/// Upstream: weights.zig:6-17 — the view types.
pub type Linear = LinearView;
#[derive(Clone)]
pub struct LinearView {
    pub weight: Tensor,
    pub scales: Tensor,
    pub biases: Tensor,
    pub inputs: usize,
    pub outputs: usize,
}

#[derive(Clone)]
pub struct Delta {
    pub qkv: Linear,
    pub z: Linear,
    pub a: Linear,
    pub b: Linear,
    pub out: Linear,
    pub conv: Tensor,
    pub norm: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
}

#[derive(Clone)]
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

/// Upstream's union is inline (`union(enum)`) and so is this: the mixers sit once a
/// layer on the model, and boxing them would buy nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum Mixer {
    Delta(Delta),
    Attention(Attention),
}

#[derive(Clone)]
pub struct Layer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub mixer: Mixer,
}

#[derive(Clone)]
pub struct Weights {
    pub embedding: Linear,
    /// The checkpoint's own output projection when it ships one, else None (the head
    /// is the packed embedding).
    pub separate_head: Option<Linear>,
    pub norm: Tensor,
    pub blocks: [Layer; config::LAYERS],
}

impl Weights {
    /// The output projection the forward's logits read.
    ///
    /// Upstream: weights.zig:27  pub fn head(self) Linear.
    pub fn head(&self) -> Linear {
        self.separate_head.clone().unwrap_or_else(|| self.embedding.clone())
    }
}

/// Upstream: weights.zig:32  fn expect — dtype, rank and exact shape, else the
/// tensor is unexpected.
fn expect(ckpt: &Checkpoint, name: &str, dtype: tensorfold_safetensors::DType, shape: &[usize]) -> Result<Tensor, crate::Error> {
    let t = ckpt.get(name).map_err(|_| crate::Error::UnexpectedQwenTensor)?;
    let matches = t.dtype == dtype
        && t.rank == shape.len()
        && t.shape[..t.rank] == *shape;
    if !matches {
        return Err(crate::Error::UnexpectedQwenTensor);
    }
    Ok(t)
}

fn tensor(
    ckpt: &Checkpoint,
    prefix: &str,
    suffix: &str,
    dtype: tensorfold_safetensors::DType,
    shape: &[usize],
) -> Result<Tensor, crate::Error> {
    let name = format!("{}{}", prefix, suffix);
    expect(ckpt, &name, dtype, shape)
}

/// Upstream: weights.zig:46  fn projection — the affine triple with its dimensions.
fn projection(
    ckpt: &Checkpoint,
    prefix: &str,
    suffix: &str,
    outputs: usize,
    inputs: usize,
) -> Result<Linear, crate::Error> {
    let name = format!("{}{}", prefix, suffix);
    Ok(Linear {
        weight: tensor(ckpt, &name, ".weight", tensorfold_safetensors::DType::U32, &[outputs, inputs / 8])?,
        scales: tensor(ckpt, &name, ".scales", tensorfold_safetensors::DType::BF16, &[outputs, inputs / config::GROUP])?,
        biases: tensor(ckpt, &name, ".biases", tensorfold_safetensors::DType::BF16, &[outputs, inputs / config::GROUP])?,
        inputs,
        outputs,
    })
}

/// Upstream: weights.zig:58  pub fn load.
pub fn load(ckpt: &Checkpoint, tied: bool) -> Result<Weights, crate::Error> {
    let ships_head = ckpt.has("language_model.lm_head.weight");
    if ships_head == tied {
        // Upstream's log names the disagreement; the error carries it.
        return Err(crate::Error::QwenHeadTiednessMismatch);
    }
    let embedding = projection(ckpt, "language_model.model.", "embed_tokens", config::VOCAB, config::HIDDEN)?;
    let separate_head = if ships_head {
        Some(projection(ckpt, "language_model.", "lm_head", config::VOCAB, config::HIDDEN)?)
    } else {
        None
    };
    let norm = tensor(
        ckpt,
        "language_model.model.",
        "norm.weight",
        tensorfold_safetensors::DType::BF16,
        &[config::HIDDEN],
    )?;
    let mut blocks: Vec<Layer> = Vec::with_capacity(config::LAYERS);
    for i in 0..config::LAYERS {
        let p = format!("language_model.model.layers.{}.", i);
        let input_norm = tensor(ckpt, &p, "input_layernorm.weight", tensorfold_safetensors::DType::BF16, &[config::HIDDEN])?;
        let post_norm = tensor(ckpt, &p, "post_attention_layernorm.weight", tensorfold_safetensors::DType::BF16, &[config::HIDDEN])?;
        let gate = projection(ckpt, &p, "mlp.gate_proj", config::INTERMEDIATE, config::HIDDEN)?;
        let up = projection(ckpt, &p, "mlp.up_proj", config::INTERMEDIATE, config::HIDDEN)?;
        let down = projection(ckpt, &p, "mlp.down_proj", config::HIDDEN, config::INTERMEDIATE)?;
        let mixer = if config::linear(i) {
            Mixer::Delta(Delta {
                qkv: projection(ckpt, &p, "linear_attn.in_proj_qkv", config::CONV_DIM, config::HIDDEN)?,
                z: projection(ckpt, &p, "linear_attn.in_proj_z", config::HIDDEN, config::HIDDEN)?,
                a: projection(ckpt, &p, "linear_attn.in_proj_a", config::LINEAR_HEADS, config::HIDDEN)?,
                b: projection(ckpt, &p, "linear_attn.in_proj_b", config::LINEAR_HEADS, config::HIDDEN)?,
                out: projection(ckpt, &p, "linear_attn.out_proj", config::HIDDEN, config::HIDDEN)?,
                conv: tensor(ckpt, &p, "linear_attn.conv1d.weight", tensorfold_safetensors::DType::BF16, &[config::CONV_DIM, config::CONV_TAPS, 1])?,
                norm: tensor(ckpt, &p, "linear_attn.norm.weight", tensorfold_safetensors::DType::BF16, &[config::LINEAR_DIM])?,
                a_log: tensor(ckpt, &p, "linear_attn.A_log", tensorfold_safetensors::DType::F32, &[config::LINEAR_HEADS])?,
                dt_bias: tensor(ckpt, &p, "linear_attn.dt_bias", tensorfold_safetensors::DType::BF16, &[config::LINEAR_HEADS])?,
            })
        } else {
            Mixer::Attention(Attention {
                q: projection(ckpt, &p, "self_attn.q_proj", 2 * config::HIDDEN, config::HIDDEN)?,
                k: projection(ckpt, &p, "self_attn.k_proj", config::KV_HEADS * config::HEAD_DIM, config::HIDDEN)?,
                v: projection(ckpt, &p, "self_attn.v_proj", config::KV_HEADS * config::HEAD_DIM, config::HIDDEN)?,
                out: projection(ckpt, &p, "self_attn.o_proj", config::HIDDEN, config::HIDDEN)?,
                q_norm: tensor(ckpt, &p, "self_attn.q_norm.weight", tensorfold_safetensors::DType::BF16, &[config::HEAD_DIM])?,
                k_norm: tensor(ckpt, &p, "self_attn.k_norm.weight", tensorfold_safetensors::DType::BF16, &[config::HEAD_DIM])?,
            })
        };
        blocks.push(Layer { input_norm, post_norm, gate, up, down, mixer });
    }
    if blocks.len() != config::LAYERS {
        return Err(crate::Error::UnsupportedQwenGeometry);
    }
    let mut block_iter = blocks.into_iter();
    let blocks: [Layer; config::LAYERS] = std::array::from_fn(|_| block_iter.next().unwrap());
    Ok(Weights { embedding, separate_head, norm, blocks })
}
