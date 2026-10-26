//! Qwen pipelines use the existing row decoder's fixed arithmetic, including on GPUs
//! without tensor units.
//!
//! Port of `zig/src/families/qwen3_5/kernels.zig` and the generated registry it
//! reads (`zig/kernels/metal/qwen3_5/kernels.zig`). Upstream compiles these sources
//! at run time from strings embedded at build time — the same runtime compiler and
//! MLX compile options `mx.fast.metal_kernel` uses (`kernels.zig:57`), which is
//! deliberate: the numerics must match the captured fixtures. The port keeps the
//! whole path: the sources here are the same `.metal` files, byte-identical under
//! `kernels/metal/qwen3_5/`, embedded with `include_str!` where upstream has
//! `@embedFile`.
//!
//! Upstream's registry is generated code walked at comptime; the port's is a frozen
//! static table (decision record: docs/DECISIONS.md, ADR-005). The config gate pins
//! the geometry to one exact shape, so the table's launch geometries are constants
//! by construction — and the `columns`/`threads` values below are the generated
//! file's own, verbatim.

use tensorfold_metal::{CompileOptions, Device, Library, Pipeline};

/// Upstream: the generated `kernels.zig` `Kernel = struct { key, function, source,
/// columns, threads }` — the 15 generated kernels, verbatim keys, function names and
/// launch geometries.
pub struct KernelDef {
    pub key: &'static str,
    pub function: &'static str,
    pub source: &'static str,
    pub columns: usize,
    pub threads: usize,
}

/// Upstream: `kernel_sources.qwen35` (`kernels/metal/qwen3_5/kernels.zig` `all`).
pub const ALL: [KernelDef; 15] = [
    KernelDef {
        key: "qmm_6144_2048",
        function: "custom_kernel_qwen35_qmm_6144_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_6144_2048.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "qmm_2048_2048",
        function: "custom_kernel_qwen35_qmm_2048_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_2048_2048.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "qmm_16_2048",
        function: "custom_kernel_qwen35_qmm_16_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_16_2048.metal"),
        columns: 16,
        threads: 256,
    },
    KernelDef {
        key: "qmm_4096_2048",
        function: "custom_kernel_qwen35_qmm_4096_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_4096_2048.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "qmm_512_2048",
        function: "custom_kernel_qwen35_qmm_512_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_512_2048.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "qmm_2048_6144",
        function: "custom_kernel_qwen35_qmm_2048_6144_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_2048_6144.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "qmm_248320_2048",
        function: "custom_kernel_qwen35_qmm_248320_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/qmm_248320_2048.metal"),
        columns: 32,
        threads: 256,
    },
    KernelDef {
        key: "norm",
        function: "custom_kernel_qwen35_norm_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/norm.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "norm_nores",
        function: "custom_kernel_qwen35_norm_nores_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/norm_nores.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "gdn_pre",
        function: "custom_kernel_qwen35_gdn_pre_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/gdn_pre.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "gdn_chain",
        function: "custom_kernel_qwen35_gdn_chain__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_float_int32_tc_bfloat16_t_float",
        source: include_str!("../../../kernels/metal/qwen3_5/gdn_chain.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "gdn_post",
        function: "custom_kernel_qwen35_gdn_post_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/gdn_post.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "mlp_act",
        function: "custom_kernel_qwen35_mlp_act_bfloat16_t_bfloat16_t_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/mlp_act.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "attn_partial",
        function: "custom_kernel_qwen35_attn_partial_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_float_float_float",
        source: include_str!("../../../kernels/metal/qwen3_5/attn_partial.metal"),
        columns: 0,
        threads: 0,
    },
    KernelDef {
        key: "attn_merge",
        function: "custom_kernel_qwen35_attn_merge_float_float_float_int32_t_bfloat16_t",
        source: include_str!("../../../kernels/metal/qwen3_5/attn_merge.metal"),
        columns: 0,
        threads: 0,
    },
];

/// Upstream: `kernels.zig:6  layout_names` — the fixed arithmetic in one shared
/// library (`kernel_sources.qwen35_layout`).
const LAYOUT_NAMES: [&str; 5] = [
    "qwen35_embed",
    "qwen35_head_norm",
    "qwen35_queries",
    "qwen35_keys",
    "qwen35_attention_gate",
];

/// Upstream: `kernel_sources.qwen35_layout = @embedFile("qwen3_5/layout.metal")`.
const LAYOUT: &str = include_str!("../../../kernels/metal/qwen3_5/layout.metal");

/// Upstream: `kernels.zig:8  pub const total`.
pub const TOTAL: usize = ALL.len() + LAYOUT_NAMES.len();

/// One projection kernel's launch geometry. Upstream: `kernels.zig:39
/// pub const Projection = struct { pipeline, columns, threads }`.
#[derive(Clone)]
pub struct Projection {
    pub pipeline: Pipeline,
    pub columns: usize,
    pub threads: usize,
}

/// The family's compiled pipelines, keyed the way upstream's comptime registry keys
/// them. Built once at model load; clones retain.
#[derive(Clone)]
pub struct Kernels {
    pipelines: Vec<Pipeline>,
}

impl Kernels {
    /// Compile every kernel at run time with MLX's options (upstream `load`,
    /// kernels.zig:53). A compile failure is the error its message names — never a
    /// guess at a missing kernel.
    pub fn load(device: &Device) -> Result<Kernels, String> {
        let options = CompileOptions::mlx();
        let mut pipelines = Vec::with_capacity(TOTAL);
        for def in &ALL {
            let library = Library::from_source(device, def.source, options)?;
            let pipeline = library.pipeline(device, def.function, false)?;
            pipelines.push(pipeline);
        }
        let layout = Library::from_source(device, LAYOUT, options)?;
        for name in LAYOUT_NAMES {
            pipelines.push(layout.pipeline(device, name, false)?);
        }
        Ok(Kernels { pipelines })
    }

    /// Upstream: `byKey` — a pipeline by name, or None when the registry has none.
    pub fn by_key(&self, key: &str) -> Option<Pipeline> {
        for (def, pipeline) in ALL.iter().zip(&self.pipelines) {
            if def.key == key {
                return Some(pipeline.clone());
            }
        }
        for (&name, pipeline) in LAYOUT_NAMES.iter().zip(&self.pipelines[ALL.len()..]) {
            if name == key {
                return Some(pipeline.clone());
            }
        }
        None
    }

    /// The pipeline a key names; the known keys are compile-time constants in the
    /// forward passes, so a miss is a programming error surfaced as a family error.
    pub fn get(&self, key: &str) -> Result<Pipeline, crate::Error> {
        self.by_key(key)
            .ok_or_else(|| crate::Error::QwenKernel(format!("no Qwen3.5 kernel {}", key)))
    }

    /// The projection kernel for an (outputs, inputs) pair (upstream `projection`,
    /// kernels.zig:23): the qmm registry entry whose key names those dimensions.
    pub fn projection(&self, outputs: usize, inputs: usize) -> Option<Projection> {
        for (def, pipeline) in ALL.iter().zip(&self.pipelines) {
            if def.columns > 0 {
                let dims = dimensions(def.key)?;
                if dims == [outputs, inputs] {
                    return Some(Projection {
                        pipeline: pipeline.clone(),
                        columns: def.columns,
                        threads: def.threads,
                    });
                }
            }
        }
        None
    }
}

/// Upstream: `kernels.zig:41  fn dimensions(key)` — "qmm_N_K" to [N, K].
fn dimensions(key: &str) -> Option<[usize; 2]> {
    let tail = key.strip_prefix("qmm_")?;
    let at = tail.find('_')?;
    Some([tail[..at].parse().ok()?, tail[at + 1..].parse().ok()?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    /// The registry's keys parse into the dimensions `projection` looks up, and the
    /// projection table covers every linear the weights load: one per distinct
    /// (outputs, inputs) the family's pinned geometry names.
    #[test]
    fn the_qmm_keys_name_their_dimensions() {
        assert_eq!(dimensions("qmm_6144_2048"), Some([6144, 2048]));
        assert_eq!(dimensions("qmm_248320_2048"), Some([248320, 2048]));
        assert_eq!(dimensions("norm"), None);
        let mut seen = std::collections::HashSet::new();
        for def in &ALL {
            if def.columns > 0 {
                let dims = dimensions(def.key).unwrap();
                seen.insert(dims);
            }
        }
        // Every projection the weights loader asks for, from weights.zig's shapes.
        for dims in [
            [config::VOCAB, config::HIDDEN],        // embedding and head
            [config::INTERMEDIATE, config::HIDDEN], // gate, up
            [config::HIDDEN, config::INTERMEDIATE],  // down
            [config::CONV_DIM, config::HIDDEN],      // delta qkv
            [config::HIDDEN, config::HIDDEN],        // delta z, out; attention out
            [config::LINEAR_HEADS, config::HIDDEN],  // delta a, b
            [2 * config::HIDDEN, config::HIDDEN],    // attention q (fused gates)
            [config::KV_HEADS * config::HEAD_DIM, config::HIDDEN], // attention k, v
        ] {
            assert!(seen.contains(&dims), "no qmm kernel for {:?}", dims);
        }
    }
}
