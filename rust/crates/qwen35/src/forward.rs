//! Shared row projections with segment-local recurrence and causal attention,
//! encoded through the native Metal queue.
//!
//! Port of `zig/src/families/qwen3_5/forward.zig`. The encode is one command buffer
//! a forward: the rows of every segment run through the same kernels with the same
//! fixed arithmetic, and the GPU work commits before any host state moves.

use crate::config;
use crate::model::Model;
use crate::state::{Cache, Scratch};
use crate::weights::{Linear, Tensor};
use tensorfold_metal::{Buffer, Encoder as PassEncoder};

/// Upstream: forward.zig:9  pub const Segment = struct { cache, rows }.
pub struct Segment<'a> {
    pub cache: &'a mut Cache,
    pub rows: usize,
}

/// Upstream: forward.zig:10  pub const Head = enum { last, all }.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Head {
    Last,
    All,
}

/// Upstream: forward.zig:12  const Encoder — the model-bound encode helpers.
struct Encoder<'m, 'p> {
    e: PassEncoder<'p>,
    m: &'m Model,
}

impl<'m, 'p> Encoder<'m, 'p> {
    /// Upstream: `pipe(key)` — the kernel registry by key (its comptime get).
    fn pipe(&self, key: &str) -> Result<(), crate::Error> {
        self.e.set_pipeline(&self.m.kernels.get(key)?);
        Ok(())
    }

    /// Upstream: `tensor(t, slot)` — a weight view at its buffer offset.
    fn tensor(&self, t: &Tensor, slot: usize) {
        self.e.set_buffer(&t.buffer, t.offset, slot);
    }

    /// Upstream: `run(grid, group)`.
    fn run(&self, grid: [usize; 3], group: [usize; 3]) {
        self.e.dispatch(
            (grid[0] as u64, grid[1] as u64, grid[2] as u64),
            (group[0] as u64, group[1] as u64, group[2] as u64),
        );
    }

    /// Upstream: `projection(linear, x, offset, y, y_offset, rows)` — the affine
    /// qmm with its fixed grid arithmetic (forward.zig:25).
    fn projection(
        &self,
        linear: &Linear,
        x: &Buffer,
        offset: usize,
        y: &Buffer,
        y_offset: usize,
        rows: usize,
    ) -> Result<(), crate::Error> {
        let p = self
            .m
            .kernels
            .projection(linear.outputs, linear.inputs)
            .ok_or_else(|| {
                crate::Error::QwenKernel(format!(
                    "no Qwen3.5 projection for {}x{}",
                    linear.outputs, linear.inputs
                ))
            })?;
        self.e.set_pipeline(&p.pipeline);
        self.e.set_buffer(x, offset, 0);
        let dims = [rows as i32, linear.inputs as i32];
        self.e.set_value(&dims, 1);
        self.tensor(&linear.weight, 2);
        self.tensor(&linear.scales, 3);
        self.tensor(&linear.biases, 4);
        let one = [1.0f32];
        self.e.set_value(&one, 5);
        self.e.set_buffer(y, y_offset, 6);
        self.run(
            [p.threads * linear.outputs.div_ceil(p.columns), rows.div_ceil(8), 1],
            [p.threads, 1, 1],
        );
        Ok(())
    }

    /// Upstream: `norm(h, residual, weight, x, rows)` — the RMS norm, with or
    /// without the residual add (forward.zig:37).
    fn norm(
        &self,
        h: &Buffer,
        residual: Option<&Buffer>,
        weight: &Tensor,
        x: &Buffer,
        rows: usize,
    ) -> Result<(), crate::Error> {
        if let Some(r) = residual {
            self.pipe("norm")?;
            self.e.set_buffer(h, 0, 0);
            self.e.set_buffer(r, 0, 1);
            self.tensor(weight, 2);
            let eps = [config::EPS];
            self.e.set_value(&eps, 3);
            self.e.set_buffer(h, 0, 4);
            self.e.set_buffer(x, 0, 5);
        } else {
            self.pipe("norm_nores")?;
            self.e.set_buffer(h, 0, 0);
            self.tensor(weight, 1);
            let eps = [config::EPS];
            self.e.set_value(&eps, 2);
            self.e.set_buffer(x, 0, 3);
        }
        self.run([128, rows, 1], [128, 1, 1]);
        Ok(())
    }
}

/// One forward over every segment's rows (upstream `run`, forward.zig:57): embed,
/// per-layer mixer passes, the final norm and the head; then the commit.
#[allow(clippy::too_many_arguments)]
pub fn run(
    m: &Model,
    s: &mut Scratch,
    segments: &mut [Segment<'_>],
    tokens: &[u32],
    record: bool,
    head: Head,
) -> Result<(), crate::Error> {
    let mut total: usize = 0;
    if tokens.is_empty() || tokens.len() > s.rows || (record && tokens.len() > crate::state::BATCH_ROWS) {
        return Err(crate::Error::QwenRowsExceeded);
    }
    for seg in segments.iter() {
        if seg.rows == 0 || seg.cache.len + seg.rows > seg.cache.capacity {
            return Err(crate::Error::QwenContextFull);
        }
        total += seg.rows;
    }
    if total != tokens.len()
        || (head == Head::All && total > crate::state::BATCH_ROWS)
        || (!record && segments.len() != 1)
    {
        return Err(crate::Error::InvalidQwenSegments);
    }
    for id in tokens {
        if *id as usize >= config::VOCAB {
            return Err(crate::Error::InvalidQwenToken);
        }
    }
    // The host writes the ids and the conv windows before any GPU work (the ids and
    // windows buffers are the encode's first reads).
    let ids: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(s.ids.as_ptr() as *mut u32, s.ids.len() / 4)
    };
    ids[..total].copy_from_slice(tokens);
    let windows: &mut [i32] = unsafe {
        std::slice::from_raw_parts_mut(s.windows.as_ptr() as *mut i32, s.windows.len() / 4)
    };
    let mut base: usize = 0;
    for seg in segments.iter() {
        for r in 0..seg.rows {
            for tap in 0..config::CONV_TAPS {
                windows[(base + r) * config::CONV_TAPS + tap] = (r + tap) as i32;
            }
        }
        base += seg.rows;
    }

    // The encode: one command buffer over every kernel (upstream builds it inside an
    // autorelease pool; the pass owns its own here).
    let pass = m.gpu.compute_pass();
    let enc = Encoder { e: pass.encoder(), m };
    enc.pipe("qwen35_embed")?;
    enc.e.set_buffer(&s.ids, 0, 0);
    enc.tensor(&m.weights.embedding.weight, 1);
    enc.tensor(&m.weights.embedding.scales, 2);
    enc.tensor(&m.weights.embedding.biases, 3);
    enc.e.set_buffer(&s.h, 0, 4);
    enc.run([1024, total, 1], [256, 1, 1]);
    for (i, block) in m.weights.blocks.iter().enumerate() {
        let residual = if i == 0 { None } else { Some(&s.r) };
        enc.norm(&s.h, residual, &block.input_norm, &s.x, total)?;
        match &block.mixer {
            crate::weights::Mixer::Delta(delta) => {
                delta_pass(&enc, s, delta, i, segments, record, total)?
            }
            crate::weights::Mixer::Attention(attention) => {
                attention_pass(&enc, s, attention, i, segments, total)?
            }
        }
        enc.norm(&s.h, Some(&s.r), &block.post_norm, &s.x, total)?;
        enc.projection(&block.gate, &s.x, 0, &s.gate, 0, total)?;
        enc.projection(&block.up, &s.x, 0, &s.up, 0, total)?;
        enc.pipe("mlp_act")?;
        enc.e.set_buffer(&s.gate, 0, 0);
        enc.e.set_buffer(&s.up, 0, 1);
        enc.e.set_buffer(&s.act, 0, 2);
        enc.run([config::INTERMEDIATE, total, 1], [256, 1, 1]);
        enc.projection(&block.down, &s.act, 0, &s.r, 0, total)?;
    }
    enc.norm(&s.h, Some(&s.r), &m.weights.norm, &s.x, total)?;
    match head {
        Head::All => enc.projection(&m.weights.head(), &s.x, 0, &s.logits, 0, total)?,
        Head::Last => {
            enc.projection(
                &m.weights.head(),
                &s.x,
                (total - 1) * config::HIDDEN * 2,
                &s.logits,
                0,
                1,
            )?;
        }
    }
    // Upstream: e.end(); cb.commit(); cb.wait(); and the failure read.
    pass
        .submit()
        .map_err(|message| {
            eprintln!("Qwen forward: {}", message);
            crate::Error::GpuFailed
        })?;

    // The host commit phase: each segment's logits and state (forward.zig:114).
    let mut base: usize = 0;
    for seg in segments.iter_mut() {
        let logit_row = if head == Head::All {
            base + seg.rows - 1
        } else {
            0
        };
        crate::state::copy_bytes(
            &seg.cache.logits,
            0,
            &s.logits,
            logit_row * config::VOCAB * 2,
            config::VOCAB * 2,
        );
        seg.cache.commit(s, base, seg.rows, record);
        base += seg.rows;
    }
    Ok(())
}

/// The Gated DeltaNet pass over one layer's segments (upstream `delta`,
/// forward.zig:123).
#[allow(clippy::too_many_arguments)]
fn delta_pass(
    enc: &Encoder<'_, '_>,
    s: &Scratch,
    d: &crate::weights::Delta,
    layer: usize,
    segments: &[Segment<'_>],
    record: bool,
    total: usize,
) -> Result<(), crate::Error> {
    enc.projection(&d.qkv, &s.x, 0, &s.qkv, 0, total)?;
    enc.projection(&d.z, &s.x, 0, &s.z, 0, total)?;
    enc.projection(&d.a, &s.x, 0, &s.a, 0, total)?;
    enc.projection(&d.b, &s.x, 0, &s.b, 0, total)?;
    let mut base: usize = 0;
    let snapshot = s.snapshots[layer]
        .as_ref()
        .expect("linear layers hold snapshots");
    for seg in segments {
        let cache = &seg.cache.blocks[layer];
        let crate::state::Mixer::Delta(cache) = cache else {
            unreachable!("linear layers hold delta caches");
        };
        enc.pipe("gdn_pre")?;
        enc.e.set_buffer(&s.qkv, base * config::CONV_DIM * 2, 0);
        enc.e.set_buffer(&cache.conv, 0, 1);
        enc.tensor(&d.conv, 2);
        enc.e.set_buffer(&s.windows, base * config::CONV_TAPS * 4, 3);
        enc.e.set_buffer(&s.a, base * config::LINEAR_HEADS * 2, 4);
        enc.e.set_buffer(&s.b, base * config::LINEAR_HEADS * 2, 5);
        enc.tensor(&d.a_log, 6);
        enc.tensor(&d.dt_bias, 7);
        for (slot, buffer) in [&s.q, &s.k, &s.v].iter().enumerate() {
            enc.e.set_buffer(buffer, base * config::HIDDEN * 2, 8 + slot);
        }
        enc.e.set_buffer(&s.g, base * config::LINEAR_HEADS * 4, 11);
        enc.e.set_buffer(&s.beta, base * config::LINEAR_HEADS * 2, 12);
        enc.e.set_buffer(&snapshot.conv, base * crate::state::CONV_BYTES, 13);
        enc.run([32, 3 * config::LINEAR_HEADS, seg.rows], [32, 1, 1]);
        enc.pipe("gdn_chain")?;
        for (slot, buffer) in [&s.q, &s.k, &s.v].iter().enumerate() {
            enc.e.set_buffer(buffer, base * config::HIDDEN * 2, slot);
        }
        enc.e.set_buffer(&s.g, base * config::LINEAR_HEADS * 4, 3);
        enc.e.set_buffer(&s.beta, base * config::LINEAR_HEADS * 2, 4);
        enc.e.set_buffer(&cache.recurrence, 0, 5);
        let dims = [seg.rows as i32, record as i32];
        enc.e.set_value(&dims, 6);
        enc.e.set_buffer(&s.y, base * config::HIDDEN * 2, 7);
        let snapshot_at = if record { base * crate::state::DELTA_BYTES } else { 0 };
        enc.e.set_buffer(&snapshot.recurrence, snapshot_at, 8);
        enc.run([32, config::LINEAR_DIM, config::LINEAR_HEADS], [32, 4, 1]);
        base += seg.rows;
    }
    enc.pipe("gdn_post")?;
    enc.e.set_buffer(&s.y, 0, 0);
    enc.e.set_buffer(&s.z, 0, 1);
    enc.tensor(&d.norm, 2);
    let eps = [config::EPS];
    enc.e.set_value(&eps, 3);
    enc.e.set_buffer(&s.mix, 0, 4);
    enc.run([32, config::LINEAR_HEADS, total], [32, 1, 1]);
    enc.projection(&d.out, &s.mix, 0, &s.r, 0, total)?;
    Ok(())
}

/// The full-attention pass over one layer's segments (upstream `attention`,
/// forward.zig:168).
fn attention_pass(
    enc: &Encoder<'_, '_>,
    s: &Scratch,
    a: &crate::weights::Attention,
    layer: usize,
    segments: &[Segment<'_>],
    total: usize,
) -> Result<(), crate::Error> {
    enc.projection(&a.q, &s.x, 0, &s.qraw, 0, total)?;
    enc.projection(&a.k, &s.x, 0, &s.k, 0, total)?;
    enc.projection(&a.v, &s.x, 0, &s.v, 0, total)?;
    enc.pipe("qwen35_head_norm")?;
    enc.e.set_buffer(&s.qraw, 0, 0);
    enc.tensor(&a.q_norm, 1);
    enc.e.set_buffer(&s.q, 0, 2);
    let dims = [
        total as u32,
        config::QUERY_HEADS as u32,
        (2 * config::HIDDEN) as u32,
        (2 * config::HEAD_DIM) as u32,
    ];
    enc.e.set_value(&dims, 3);
    enc.run(
        [64 * total * config::QUERY_HEADS, 1, 1],
        [64, 1, 1],
    );
    enc.pipe("qwen35_head_norm")?;
    enc.e.set_buffer(&s.k, 0, 0);
    enc.tensor(&a.k_norm, 1);
    enc.e.set_buffer(&s.knorm, 0, 2);
    let dims = [
        total as u32,
        config::KV_HEADS as u32,
        (config::KV_HEADS * config::HEAD_DIM) as u32,
        config::HEAD_DIM as u32,
    ];
    enc.e.set_value(&dims, 3);
    enc.run([64 * total * config::KV_HEADS, 1, 1], [64, 1, 1]);
    let mut base: usize = 0;
    for seg in segments {
        let cache = &seg.cache.blocks[layer];
        let crate::state::Mixer::Attention(cache) = cache else {
            unreachable!("full layers hold attention caches");
        };
        enc.pipe("qwen35_queries")?;
        enc.e.set_buffer(&s.q, 0, 0);
        enc.e.set_buffer(&s.queries, 0, 1);
        let dims = [seg.cache.len as u32, total as u32, base as u32, 0];
        enc.e.set_value(&dims, 2);
        enc.run(
            [config::HEAD_DIM, config::QUERY_HEADS, seg.rows],
            [256, 1, 1],
        );
        enc.pipe("qwen35_keys")?;
        enc.e.set_buffer(&s.knorm, 0, 0);
        enc.e.set_buffer(&s.v, 0, 1);
        enc.e.set_buffer(&cache.keys, 0, 2);
        enc.e.set_buffer(&cache.values, 0, 3);
        let dims = [
            seg.cache.len as u32,
            seg.cache.capacity as u32,
            base as u32,
            0,
        ];
        enc.e.set_value(&dims, 4);
        enc.run(
            [config::HEAD_DIM, config::KV_HEADS, seg.rows],
            [256, 1, 1],
        );
        let nch = (seg.cache.len + seg.rows).div_ceil(128);
        let dims = [
            seg.cache.len as i32,
            seg.rows as i32,
            seg.cache.capacity as i32,
            nch as i32,
            0,
            total as i32,
            base as i32,
            0,
        ];
        enc.pipe("attn_partial")?;
        enc.e.set_buffer(&s.queries, 0, 0);
        enc.e.set_buffer(&cache.keys, 0, 1);
        enc.e.set_buffer(&cache.values, 0, 2);
        let scale = [1.0f32 / 16.0];
        enc.e.set_value(&scale, 3);
        enc.e.set_value(&dims, 4);
        enc.e.set_buffer(&s.pm, 0, 5);
        enc.e.set_buffer(&s.pl, 0, 6);
        enc.e.set_buffer(&s.po, 0, 7);
        enc.run([512, nch, config::KV_HEADS * seg.rows], [512, 1, 1]);
        enc.pipe("attn_merge")?;
        enc.e.set_buffer(&s.pm, 0, 0);
        enc.e.set_buffer(&s.pl, 0, 1);
        enc.e.set_buffer(&s.po, 0, 2);
        enc.e.set_value(&dims, 3);
        enc.e.set_buffer(&s.y, base * config::HIDDEN * 2, 4);
        enc.run([32, config::QUERY_HEADS, seg.rows], [32, 1, 1]);
        base += seg.rows;
    }
    enc.pipe("qwen35_attention_gate")?;
    enc.e.set_buffer(&s.y, 0, 0);
    enc.e.set_buffer(&s.qraw, 0, 1);
    enc.e.set_buffer(&s.mix, 0, 2);
    enc.run([config::HIDDEN, total, 1], [256, 1, 1]);
    enc.projection(&a.out, &s.mix, 0, &s.r, 0, total)?;
    Ok(())
}
