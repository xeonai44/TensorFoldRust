//! Nemotron CUDA prompt states for the shared cache: the engine's
//! caches and MTP head, copied on its stream.
//!
//! Port of `zig/src/families/nemotron/cuda_reuse.zig` (75 lines) — the
//! cache's anyopaque pairs become raw pointers under the host cache's
//! own invariant (the saved copy the host owns, restored here).

use tensorfold_cuda::DeviceBuffer;

use crate::cuda_engine::Engine;
use crate::error::Error;

/// What one kept state holds. The token position lives beside the
/// device copy. Upstream: `Copy` (cuda_reuse.zig:9).
pub struct Copy {
    pub state: DeviceBuffer<'static>,
    pub head: Option<DeviceBuffer<'static>>,
    pub pos: usize,
    pub parity: usize,
    pub prev_keep: usize,
    pub rows: usize,
    pub head_pos: usize,
}

/// The engine and head a cache save copies. The native host keeps one
/// of these for the loaded model. Upstream: `Target`
/// (cuda_reuse.zig:20) — the port carries the engine as the raw
/// bridge `with_head` documents (the target lives beside the engine).
pub struct Target<'a, 'd> {
    pub e: *mut Engine<'a, 'd>,
}

/// Bytes a save at any position takes: the caches are sized for the
/// whole window, not for `at`. Upstream: `bytes` (cuda_reuse.zig:26).
pub fn bytes(ptr: *mut core::ffi::c_void, _at: u32) -> u64 {
    // SAFETY: the caller's target, alive for the call (the host
    // cache's own invariant over its saved handles).
    let t: &Target<'_, '_> = unsafe { &*(ptr.cast::<Target<'_, '_>>()) };
    // SAFETY: the engine behind the target's raw bridge, alive for
    // the call.
    let e: &Engine<'_, '_> = unsafe { &*t.e };
    let mut n: u64 = e.b.state_bytes() as u64;
    if let Some(h) = &e.head {
        // SAFETY: the head in its Box, read-only here.
        n += snapshot_bytes(h.as_ref(), e) as u64;
    }
    n
}

fn snapshot_bytes<'d>(h: &crate::cuda_mtp::Head<'d>, e: &Engine<'d, 'd>) -> usize {
    h.snapshot_bytes(e)
}

/// Copy the live state after `at` prompt tokens. The pass is standing
/// there. Upstream: `save` (cuda_reuse.zig:35).
///
/// The saved copy comes back as a raw pointer the HOST owns; `drop`
/// takes it back.
///
/// # Safety
///
/// `ptr` points at a live Target; the engine stands at `at` with no
/// round in flight.
pub unsafe fn save(
    ptr: *mut core::ffi::c_void,
    _owner: *mut core::ffi::c_void,
    at: u32,
) -> Result<*mut core::ffi::c_void, Error> {
    // SAFETY: the caller's target.
    let t: &mut Target<'_, '_> = unsafe { &mut *ptr.cast::<Target<'_, '_>>() };
    // SAFETY: the engine behind the target's raw bridge.
    let e: &mut Engine<'_, '_> = unsafe { &mut *t.e };
    if e.pos != at as usize {
        return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // NotAtMark
    }
    if let Some(h) = &e.head {
        // SAFETY: the head's position read-only.
        if h.as_ref().pos != at as usize {
            return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // NotAtMark
        }
    }
    let state = e.b.snapshot(&e.ops())?;
    let mut head = None;
    let mut head_pos = 0;
    if e.head.is_some() {
        let saved = e
            .with_head(|h, e| h.snapshot(e))
            .ok_or(Error::NoMtpHead)??;
        head_pos = e.head.as_ref().unwrap().pos;
        head = Some(saved);
    }
    let copy = Box::new(Copy {
        state,
        head,
        pos: e.pos,
        parity: e.parity,
        prev_keep: e.prev_keep,
        rows: e.rows,
        head_pos,
    });
    Ok(Box::into_raw(copy).cast())
}

/// Make the bound sequence `saved`'s. The next prompt chunk starts at
/// its position. Upstream: `restore` (cuda_reuse.zig:54).
///
/// # Safety
///
/// `ptr` points at a live Target; `saved` is a copy this module's
/// `save` made (the host cache's ownership), not yet dropped.
pub unsafe fn restore(
    ptr: *mut Target<'_, '_>,
    saved: *mut core::ffi::c_void,
    _at: u32,
) -> Result<(), Error> {
    // SAFETY: the caller's target, alive for the call.
    let ptr: &mut Target<'_, '_> = unsafe { &mut *ptr };
    // SAFETY: the engine behind the target's raw bridge.
    let e: &mut Engine<'_, '_> = unsafe { &mut *ptr.e };
    // SAFETY: the host cache's saved copy, alive until it drops it.
    let copy: &mut Copy = unsafe { &mut *saved.cast::<Copy>() };
    let has_head = e.head.is_some();
    if has_head && copy.head.is_none() {
        return Err(Error::NoMtpHead); // checked before any copy: nothing half restored
    }
    e.b.restore(&e.ops(), &copy.state)?;
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = copy.prev_keep;
    e.rows = copy.rows;
    if has_head {
        let head_pos = copy.head_pos;
        let head_state = copy.head.as_ref().unwrap();
        e.with_head(|h, e| h.restore(e, head_state, head_pos))
            .ok_or(Error::NoMtpHead)??;
    }
    Ok(())
}

/// Free a kept state's device copy. Upstream: `drop`
/// (cuda_reuse.zig:69).
///
/// # Safety
///
/// `ptr` points at a live Target; `saved` is a copy this module made
/// and is dropped exactly once.
pub unsafe fn drop_saved(saved: *mut core::ffi::c_void) {
    // SAFETY: the host cache's saved copy, dropped once (the buffers
    // inside free with it).
    unsafe {
        let copy: Box<Copy> = Box::from_raw(saved.cast::<Copy>());
        drop(copy);
    }
}
