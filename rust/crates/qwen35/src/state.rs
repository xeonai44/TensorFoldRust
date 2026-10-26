//! Per-stream committed recurrence and KV state, with a shared pool of window
//! snapshots for partial acceptance.
//!
//! Port of `zig/src/families/qwen3_5/state.zig`. The buffers are shared+untracked
//! (upstream `state.zig:5`); the host copies between them happen only while no
//! command buffer is in flight — the forwards wait for completion before commit.

use crate::config;
use tensorfold_metal::{Buffer, Gpu};

/// Upstream: state.zig:6-9 — the state sizes and the scratch's bounds.
pub const DELTA_BYTES: usize = config::LINEAR_HEADS * config::LINEAR_DIM * config::LINEAR_DIM * 4;
pub const CONV_BYTES: usize = (config::CONV_TAPS - 1) * config::CONV_DIM * 2;
pub const WINDOW_ROWS: usize = 16;
pub const BATCH_ROWS: usize = 32;

/// A pool of device buffers freed together. Upstream: state.zig:11  Storage.
pub struct Storage {
    gpu: std::sync::Arc<Gpu>,
    buffers: Vec<Buffer>,
}

impl Storage {
    pub(crate) fn new(gpu: std::sync::Arc<Gpu>) -> Storage {
        Storage {
            gpu,
            buffers: Vec::new(),
        }
    }

    /// Upstream: `alloc(bytes)` — max(bytes, 16), shared+untracked.
    pub fn alloc(&mut self, bytes: usize) -> Buffer {
        let buffer = self
            .gpu
            .buffer_with_options(bytes.max(16), Gpu::shared_untracked());
        self.buffers.push(buffer.clone());
        buffer
    }
}

/// Upstream: state.zig:29-31.
#[derive(Clone)]
pub struct Delta {
    pub recurrence: Buffer,
    pub conv: Buffer,
}

#[derive(Clone)]
pub struct Attention {
    pub keys: Buffer,
    pub values: Buffer,
}

#[derive(Clone)]
pub enum Mixer {
    Delta(Delta),
    Attention(Attention),
}

/// One stream's committed state. Upstream: state.zig:33  Cache.
pub struct Cache {
    /// The cache's buffer pool, held for ownership: the blocks' buffers keep their
    /// own retains; this drops them all together (upstream's `memory.deinit()`).
    #[allow(dead_code)]
    memory: Storage,
    pub capacity: usize,
    pub len: usize,
    last: Option<Last>,
    pub blocks: [Mixer; config::LAYERS],
    pub logits: Buffer,
}

#[derive(Clone, Copy)]
struct Last {
    start: usize,
    base: usize,
    rows: usize,
}

impl Cache {
    /// Upstream: `init` (state.zig:41) — the committed recurrence starts zeroed, the
    /// KV rows are capacity-sized.
    pub fn new(gpu: std::sync::Arc<Gpu>, capacity: usize) -> Cache {
        let mut memory = Storage { gpu, buffers: Vec::new() };
        let logits = memory.alloc(config::VOCAB * 2);
        let mut blocks: Vec<Mixer> = Vec::with_capacity(config::LAYERS);
        for i in 0..config::LAYERS {
            if config::linear(i) {
                let mut recurrence = memory.alloc(DELTA_BYTES);
                recurrence.as_mut_ptr_bytes()[..DELTA_BYTES].fill(0);
                let mut conv = memory.alloc(CONV_BYTES);
                conv.as_mut_ptr_bytes()[..CONV_BYTES].fill(0);
                blocks.push(Mixer::Delta(Delta { recurrence, conv }));
            } else {
                let bytes = capacity * config::KV_HEADS * config::HEAD_DIM * 2;
                blocks.push(Mixer::Attention(Attention {
                    keys: memory.alloc(bytes),
                    values: memory.alloc(bytes),
                }));
            }
        }
        let mut block_iter = blocks.into_iter();
        let blocks: [Mixer; config::LAYERS] = std::array::from_fn(|_| block_iter.next().unwrap());
        Cache {
            memory,
            capacity,
            len: 0,
            last: None,
            blocks,
            logits,
        }
    }

    /// Upstream: `commit` (state.zig:64) — after a forward: the last row's recurrence
    /// and conv become the committed state, and a recorded round remembers where its
    /// rows started.
    pub fn commit(&mut self, scratch: &Scratch, base: usize, rows: usize, record: bool) {
        for (block, snapshot) in self.blocks.iter().zip(&scratch.snapshots) {
            if let Mixer::Delta(delta) = block {
                let snapshot = snapshot.as_ref().expect("linear layers hold snapshots");
                let saved = if record { base + rows - 1 } else { 0 };
                copy_bytes(&delta.recurrence, 0, &snapshot.recurrence, saved * DELTA_BYTES, DELTA_BYTES);
                copy_bytes(&delta.conv, 0, &snapshot.conv, (base + rows - 1) * CONV_BYTES, CONV_BYTES);
            }
        }
        self.last = if record {
            Some(Last { start: self.len, base, rows })
        } else {
            None
        };
        self.len += rows;
    }

    /// A restored state stands at `at`: no recorded round, the length it
    /// holds (snapshot.zig:104's restore tail).
    pub fn clear_last(&mut self) {
        self.last = None;
    }

    /// Upstream: `keep` (state.zig:77) — roll back to a recorded round's kept prefix
    /// path: the recurrence and conv of the path's last row become the state, the
    /// logits of that row become the committed logits.
    pub fn keep(&mut self, scratch: &Scratch, path: &[u32]) -> Result<(), crate::Error> {
        let Some(last) = self.last else {
            return Err(crate::Error::NothingToKeep);
        };
        if path.is_empty() || path.len() > last.rows {
            return Err(crate::Error::NothingToKeep);
        }
        for (i, row) in path.iter().enumerate() {
            if *row != i as u32 {
                return Err(crate::Error::UnsupportedQwenTree);
            }
        }
        for (block, snapshot) in self.blocks.iter().zip(&scratch.snapshots) {
            if let Mixer::Delta(delta) = block {
                let snapshot = snapshot.as_ref().expect("linear layers hold snapshots");
                let src = last.base + path.len() - 1;
                copy_bytes(&delta.recurrence, 0, &snapshot.recurrence, src * DELTA_BYTES, DELTA_BYTES);
                copy_bytes(&delta.conv, 0, &snapshot.conv, src * CONV_BYTES, CONV_BYTES);
            }
        }
        self.len = last.start + path.len();
        let row = last.base + path.len() - 1;
        copy_bytes(&self.logits, 0, &scratch.logits, row * config::VOCAB * 2, config::VOCAB * 2);
        self.last = None;
        Ok(())
    }
}

/// The shared window scratch every forward runs in. Upstream: state.zig:95 Scratch.
pub struct Scratch {
    /// The scratch's buffer pool, held for ownership (as `Cache`'s).
    #[allow(dead_code)]
    memory: Storage,
    pub rows: usize,
    pub ids: Buffer,
    pub windows: Buffer,
    pub dims: Buffer,
    pub h: Buffer,
    pub x: Buffer,
    pub r: Buffer,
    pub qkv: Buffer,
    pub z: Buffer,
    pub a: Buffer,
    pub b: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub g: Buffer,
    pub beta: Buffer,
    pub y: Buffer,
    pub mix: Buffer,
    pub qraw: Buffer,
    pub knorm: Buffer,
    pub queries: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub act: Buffer,
    pub pm: Buffer,
    pub pl: Buffer,
    pub po: Buffer,
    pub logits: Buffer,
    pub snapshots: [Option<Delta>; config::LAYERS],
}

impl Scratch {
    /// Upstream: `init` (state.zig:127) — every buffer at the forward's row bound,
    /// the logits at the batch bound, and one snapshot pair per linear layer.
    pub fn new(gpu: std::sync::Arc<Gpu>, rows: usize, capacity: usize) -> Scratch {
        let mut memory = Storage { gpu, buffers: Vec::new() };
        let mut snapshot_list: Vec<Option<Delta>> = Vec::with_capacity(config::LAYERS);
        let ids = memory.alloc(rows * 4);
        let windows = memory.alloc(rows * config::CONV_TAPS * 4);
        let dims = memory.alloc(8 * 4);
        let hidden = rows * config::HIDDEN * 2;
        let h = memory.alloc(hidden);
        let x = memory.alloc(hidden);
        let r = memory.alloc(hidden);
        let qkv_inter = memory.alloc(rows * config::INTERMEDIATE * 2);
        let z = memory.alloc(hidden);
        let a = memory.alloc(rows * config::LINEAR_HEADS * 2);
        let b = memory.alloc(rows * config::LINEAR_HEADS * 2);
        let q = memory.alloc(hidden);
        let k = memory.alloc(hidden);
        let v = memory.alloc(hidden);
        let beta = memory.alloc(rows * config::LINEAR_HEADS * 2);
        let g = memory.alloc(rows * config::LINEAR_HEADS * 4);
        let y = memory.alloc(hidden);
        let mix = memory.alloc(hidden);
        let qraw = memory.alloc(rows * 2 * config::HIDDEN * 2);
        let knorm = memory.alloc(rows * config::KV_HEADS * config::HEAD_DIM * 2);
        let queries = memory.alloc(hidden);
        let gate = memory.alloc(rows * config::INTERMEDIATE * 2);
        let up = memory.alloc(rows * config::INTERMEDIATE * 2);
        let act = memory.alloc(rows * config::INTERMEDIATE * 2);
        let chunks = capacity.div_ceil(128);
        let pm = memory.alloc(rows * config::QUERY_HEADS * chunks * 4);
        let pl = memory.alloc(rows * config::QUERY_HEADS * chunks * 4);
        let po = memory.alloc(rows * config::QUERY_HEADS * chunks * config::HEAD_DIM * 4);
        let logits = memory.alloc(BATCH_ROWS * config::VOCAB * 2);
        for i in 0..config::LAYERS {
            if config::linear(i) {
                snapshot_list.push(Some(Delta {
                    recurrence: memory.alloc(BATCH_ROWS * DELTA_BYTES),
                    conv: memory.alloc(rows * CONV_BYTES),
                }));
            } else {
                snapshot_list.push(None);
            }
        }
        let mut snapshot_iter = snapshot_list.into_iter();
        let snapshots: [Option<Delta>; config::LAYERS] =
            std::array::from_fn(|_| snapshot_iter.next().unwrap());
        Scratch {
            memory,
            rows,
            ids,
            windows,
            dims,
            h,
            x,
            r,
            qkv: qkv_inter,
            z,
            a,
            b,
            q,
            k,
            v,
            g,
            beta,
            y,
            mix,
            qraw,
            knorm,
            queries,
            gate,
            up,
            act,
            pm,
            pl,
            po,
            logits,
            snapshots,
        }
    }
}

/// Copy `bytes` between two shared buffers at offsets, the port of upstream's
/// `@memcpy` on `contents()` (state.zig:69, commit.zig's keep).
///
/// Safety contract, same as upstream's: both buffers are shared storage, the copy
/// happens on the host while no command buffer that reads them is in flight (every
/// forward waits for completion before committing), and the ranges stay in bounds.
pub(crate) fn copy_bytes(dst: &Buffer, dst_at: usize, src: &Buffer, src_at: usize, bytes: usize) {
    debug_assert!(dst_at + bytes <= dst.len());
    debug_assert!(src_at + bytes <= src.len());
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr().add(src_at),
            dst.as_ptr().add(dst_at),
            bytes,
        );
    }
}

