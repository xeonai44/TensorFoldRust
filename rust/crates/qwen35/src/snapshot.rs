//! Qwen3.5's prompt states for the prompt cache: the linear layers' delta
//! recurrence and conv, and the attention KV rows — one kept state one set
//! of fresh buffers, copied on the host (the state's buffers are shared
//! memory, and the pass commits before the host reads).
//!
//! Port of `zig/src/families/nemotron/snapshot.zig` (144 lines) for this
//! family's state layout.

use crate::config;
use crate::state::{copy_bytes, Cache, Mixer, CONV_BYTES, DELTA_BYTES};
use crate::Error;
use std::sync::Arc;
use tensorfold_metal::{Buffer, Gpu};

/// One kept state: `at` prompt tokens of one stream.
pub struct Snap {
    pub at: u32,
    /// The linear layers' recurrence and conv (fixed bytes).
    linear: Vec<(Buffer, Buffer)>,
    /// The attention layers' key and value rows for [0, at).
    attention: Vec<(Buffer, Buffer)>,
}

/// The bytes a state at `at` tokens takes. Upstream: `bytes`
/// (snapshot.zig:30).
pub fn bytes(at: u32) -> u64 {
    let row = at as usize * config::KV_HEADS * config::HEAD_DIM * 2;
    let mut total = 0usize;
    for i in 0..config::LAYERS {
        if config::linear(i) {
            total += DELTA_BYTES + CONV_BYTES;
        } else {
            total += 2 * row;
        }
    }
    total as u64
}

/// The stream's state after `at` prompt tokens into new buffers (its pass
/// stands there). Upstream: `save` (snapshot.zig:70).
pub fn save(gpu: &Arc<Gpu>, cache: &Cache, at: u32) -> Result<Snap, Error> {
    let mut memory = crate::state::Storage::new(Arc::clone(gpu));
    let mut linear = Vec::new();
    let mut attention = Vec::new();
    let row = at as usize * config::KV_HEADS * config::HEAD_DIM * 2;
    for block in &cache.blocks {
        match block {
            Mixer::Delta(delta) => {
                let recurrence = memory.alloc(DELTA_BYTES);
                copy_bytes(&recurrence, 0, &delta.recurrence, 0, DELTA_BYTES);
                let conv = memory.alloc(CONV_BYTES);
                copy_bytes(&conv, 0, &delta.conv, 0, CONV_BYTES);
                linear.push((recurrence, conv));
            }
            Mixer::Attention(att) => {
                let keys = memory.alloc(row);
                copy_bytes(&keys, 0, &att.keys, 0, row);
                let values = memory.alloc(row);
                copy_bytes(&values, 0, &att.values, 0, row);
                attention.push((keys, values));
            }
        }
    }
    Ok(Snap {
        at,
        linear,
        attention,
    })
}

/// Make the live state `snap`'s: the pass continues from its position.
/// Upstream: `restore` (snapshot.zig:104) — the committed state replaced,
/// the rows the state holds the cache's length.
pub fn restore(cache: &mut Cache, snap: &Snap) -> Result<(), Error> {
    if snap.at as usize > cache.capacity {
        return Err(Error::InvalidQwenPrompt);
    }
    let mut li = 0;
    let mut ai = 0;
    let row = snap.at as usize * config::KV_HEADS * config::HEAD_DIM * 2;
    for block in &mut cache.blocks {
        match block {
            Mixer::Delta(delta) => {
                let (recurrence, conv) = &snap.linear[li];
                copy_bytes(&delta.recurrence, 0, recurrence, 0, DELTA_BYTES);
                copy_bytes(&delta.conv, 0, conv, 0, CONV_BYTES);
                li += 1;
            }
            Mixer::Attention(att) => {
                let (keys, values) = &snap.attention[ai];
                copy_bytes(&att.keys, 0, keys, 0, row);
                copy_bytes(&att.values, 0, values, 0, row);
                ai += 1;
            }
        }
    }
    cache.len = snap.at as usize;
    cache.clear_last();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bytes() grows with the attention rows, the linear state fixed.
    #[test]
    fn the_state_sizes() {
        let linear_layers = (0..config::LAYERS).filter(|i| config::linear(*i)).count();
        let attention_layers = config::LAYERS - linear_layers;
        let fixed = linear_layers * (DELTA_BYTES + CONV_BYTES);
        let row = config::KV_HEADS * config::HEAD_DIM * 2;
        assert_eq!(bytes(0) as usize, fixed);
        assert_eq!(bytes(1024) as usize, fixed + attention_layers * 2 * 1024 * row);
        // Bigger than the nemotron test's numbers but the same shape
        assert!(bytes(4096) > bytes(1024));
    }
}

// ---------------------------------------------------------------------------
// The prompt cache's family face (metal.zig:162-164's Cached for this
// family's backend).

use tensorfold_lanes::prompt_cache::{Saved, Snapshots, StoreError};

/// The backend behind the store's saves: the raw pointer is the ADR-013
/// documented topology — the save runs inside the backend's own prefill
/// call, between chunks (the host commit done), and only reads the
/// stream's cache. Upstream: metal.zig:162's `S = nemotron.snapshot.Cached`.
pub struct Cached {
    pub backend: *mut crate::backend::Metal,
}

// Safety: the backend pointer is the leaked static backend (the same one
// the lane core holds); the store moves to the engine thread once, before
// any use.
unsafe impl Send for Cached {}

impl Snapshots for Cached {
    fn bytes(&self, _at: u32) -> u64 {
        bytes(_at)
    }

    /// Upstream: `snapSave` (snapshot.zig:112) — the stream's cache, its
    /// pass standing at `at`, copied out.
    ///
    /// Safety: the backend pointer per the ADR-013 topology; the save
    /// reads the stream's cache between chunks.
    fn save(
        &mut self,
        owner: *mut core::ffi::c_void,
        at: u32,
    ) -> Result<Saved, StoreError> {
        let backend = unsafe { &*self.backend };
        let key = owner as usize;
        let cache = backend
            .cache_of_key(key)
            .ok_or(StoreError::NotAtMark)?;
        let snap = save(&backend.model.gpu, cache, at).map_err(|_| StoreError::CopyFailed)?;
        Ok(Box::new(snap))
    }

    /// Upstream: `snapRestore` (snapshot.zig:118) — `error.BackendRestores`:
    /// the prompt pass restores, before its first chunk.
    fn restore(
        &mut self,
        _owner: *mut core::ffi::c_void,
        _saved: &Saved,
    ) -> Result<(), StoreError> {
        Err(StoreError::CopyFailed)
    }

    fn drop_saved(&mut self, saved: Saved) {
        drop(saved);
    }
    fn saved_ptr(saved: &Saved) -> *mut core::ffi::c_void {
        let snap = saved.downcast_ref::<Snap>().expect("the family's state");
        snap as *const Snap as *mut core::ffi::c_void
    }
}

/// The Reuse's saved state, restored into a cache: the backend's prefill
/// path (backend.zig:292-298 for this family).
///
/// # Safety
///
/// The pointer is the store's own Snap, alive while the entry is (the
/// host's reported() discipline); the cache is the pass's fresh one.
pub unsafe fn restore_reuse(
    cache: &mut Cache,
    saved: *mut core::ffi::c_void,
) -> u32 {
    let snap = unsafe { &*(saved as *const Snap) };
    if snap.at as usize <= cache.capacity && restore(cache, snap).is_ok() {
        snap.at
    } else {
        0
    }
}
