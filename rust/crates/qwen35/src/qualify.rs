//! Admit wider native lanes only after real-weight forward rows and committed caches
//! match one-row execution on this GPU.
//!
//! Port of `zig/src/families/qwen3_5/qualify.zig`. The check is upstream's own
//! local exactness gate, run at every Metal backend open (backend.zig:26): wider
//! lanes are admitted only when their rows, committed states and keep rolls equal
//! one-row execution on the same weights. The port keeps the same fixed token
//! sequence and the same comparisons, so a Rust engine that passes this check makes
//! the same admission the Zig engine would.

use crate::config;
use crate::forward::{self, Head, Segment};
use crate::model::Model;
use crate::state::{self, Cache, Scratch};
use std::sync::Arc;

/// Upstream: qualify.zig:9  pub fn equal — two committed caches agree on length,
/// logits, recurrence, conv and the used KV rows.
pub fn equal(a: &Cache, b: &Cache) -> Result<(), crate::Error> {
    if a.len != b.len {
        return Err(crate::Error::QwenCacheLengthMismatch);
    }
    let a_logits = a.logits.host_bytes(0, config::VOCAB * 2);
    let b_logits = b.logits.host_bytes(0, config::VOCAB * 2);
    if a_logits != b_logits {
        return Err(crate::Error::QwenCommittedLogitsMismatch);
    }
    for (x, y) in a.blocks.iter().zip(&b.blocks) {
        match (x, y) {
            (state::Mixer::Delta(x), state::Mixer::Delta(y)) => {
                let xr = x.recurrence.host_bytes(0, state::DELTA_BYTES);
                let yr = y.recurrence.host_bytes(0, state::DELTA_BYTES);
                let xc = x.conv.host_bytes(0, state::CONV_BYTES);
                let yc = y.conv.host_bytes(0, state::CONV_BYTES);
                if xr != yr || xc != yc {
                    return Err(crate::Error::QwenRecurrenceMismatch);
                }
            }
            (state::Mixer::Attention(x), state::Mixer::Attention(y)) => {
                // Compare each used KV row, head by head, at each cache's own
                // capacity stride (qualify.zig:18).
                for (ab, bb) in [(&x.keys, &y.keys), (&x.values, &y.values)] {
                    for h in 0..config::KV_HEADS {
                        let ao = h * a.capacity * config::HEAD_DIM * 2;
                        let bo = h * b.capacity * config::HEAD_DIM * 2;
                        let bytes = a.len * config::HEAD_DIM * 2;
                        let left = ab.host_bytes(ao, bytes);
                        let right = bb.host_bytes(bo, bytes);
                        if left != right {
                            return Err(crate::Error::QwenAttentionCacheMismatch);
                        }
                    }
                }
            }
            _ => unreachable!("mixers pair by layer kind"),
        }
    }
    Ok(())
}

/// Upstream: qualify.zig:30  pub fn check — the wide scratch's two-stream recorded
/// round must equal one-row execution on the same caches, and a keep roll must equal
/// a fresh prefix.
pub fn check(m: &Model, wide: &mut Scratch) -> Result<(), crate::Error> {
    let gpu = Arc::clone(&m.gpu);
    let mut solo = Scratch::new(Arc::clone(&gpu), 1, 64);
    let mut caches: Vec<Cache> = Vec::with_capacity(4);
    for i in 0..4 {
        caches.push(Cache::new(Arc::clone(&gpu), 64 + i));
    }
    // Upstream: qualify.zig:40 — the fixed token sequence.
    let ids: [u32; 32] = [
        825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373, 825, 264, 15352, 4829, 9944, 13, 42,
        264, 271, 279, 15352, 854, 13, 4829, 825, 22373, 9944, 264, 271, 42, 13, 279,
    ];
    for (i, cache) in caches.iter_mut().enumerate() {
        let rows = if i % 2 == 0 { 1 } else { 3 };
        let segment = Segment { cache, rows };
        forward::run(m, wide, &mut [segment], &ids[..rows], false, Head::Last)?;
    }
    {
        // Two streams' caches in one recorded round: the wide scratch's batch of
        // two segments (the borrow checker gets the same split upstream's pointer
        // arithmetic implies — distinct caches).
        let (front, _) = caches.split_at_mut(2);
        let (cache_a, rest) = front.split_at_mut(1);
        let segment_a = Segment { cache: &mut cache_a[0], rows: 16 };
        let segment_b = Segment { cache: &mut rest[0], rows: 16 };
        forward::run(m, wide, &mut [segment_a, segment_b], &ids, true, Head::All)?;
    }
    for stream in 0..2 {
        for r in 0..16 {
            let segment = Segment {
                cache: &mut caches[stream + 2],
                rows: 1,
            };
            forward::run(
                m,
                &mut solo,
                &mut [segment],
                &ids[stream * 16 + r..][..1],
                false,
                Head::All,
            )?;
            let actual = wide
                .logits
                .host_bytes((stream * 16 + r) * config::VOCAB * 2, config::VOCAB * 2);
            let expected = solo.logits.host_bytes(0, config::VOCAB * 2);
            if actual != expected {
                return Err(crate::Error::QwenWindowLogitsMismatch);
            }
        }
        equal(&caches[stream], &caches[stream + 2])?;
    }
    caches[0].keep(wide, &[0, 1, 2])?;
    caches[1].keep(wide, &[0])?;
    for stream in 0..2 {
        let mut fresh = Cache::new(Arc::clone(&gpu), 64);
        let prefix = if stream == 0 { 1 } else { 3 };
        let kept = if stream == 0 { 3 } else { 1 };
        for r in 0..prefix {
            let segment = Segment { cache: &mut fresh, rows: 1 };
            forward::run(m, &mut solo, &mut [segment], &ids[r..][..1], false, Head::Last)?;
        }
        for r in 0..kept {
            let segment = Segment { cache: &mut fresh, rows: 1 };
            forward::run(
                m,
                &mut solo,
                &mut [segment],
                &ids[stream * 16 + r..][..1],
                false,
                Head::All,
            )?;
        }
        equal(&caches[stream], &fresh)?;
    }
    Ok(())
}
