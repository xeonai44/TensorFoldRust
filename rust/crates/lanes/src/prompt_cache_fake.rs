//! A host-memory snapshot family shared by the prompt cache's tests and
//! the learned-disk fault tests.
//!
//! Port of `zig/src/core/prompt_cache_fake.zig` (208 lines): its live
//! state is a position and a running sum of the prompt's tokens; the
//! learned slots (`learned()` upstream, the `disk` flag here) write a
//! state as one file under the imprint directory, the single-rank helpers
//! checking and reclaiming it. The peer pair (`paired()` upstream — the
//! fabric's rank 1 with its reservations and ACK failures) is the fabric
//! tranche's; the single-rank fault cases that need only the write, the
//! checked deletion and the reclaim ride here.

#![cfg(test)]

use crate::prompt_cache::{
    single_forget_checked, single_reclaim, Plan, Saved, Snapshots, Store, StoreError,
};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/// A family over host memory for tests: its live state is a position
/// and a running sum of the prompt's tokens. Upstream:
/// `prompt_cache_fake.zig`'s `Fake` — the store's vtable points at the
/// same object the test drives, so the port's fake shares its state
/// through a handle.
#[derive(Clone, Default)]
pub struct Fake(pub(crate) Rc<RefCell<FakeInner>>);

#[derive(Default)]
pub struct FakeInner {
    /// The learned slots stand (upstream's `learned()`/`paired()`
    /// vtables).
    pub disk: bool,
    pub at: u32,
    pub sum: u64,
    pub fail_save: bool,
    pub fail_restore: bool,
    pub live: usize,
    pub fail_write: bool,
    pub fail_delete: bool,
    pub writes: usize,
    pub disk_forgets: usize,
}

/// The fake's saved state: a position and the running sum.
pub struct State {
    at: u32,
    sum: u64,
}

impl Snapshots for Fake {
    fn bytes(&self, _at: u32) -> u64 {
        100 + _at as u64
    }
    fn save(
        &mut self,
        _owner: *mut core::ffi::c_void,
        at: u32,
    ) -> Result<Saved, StoreError> {
        let mut f = self.0.borrow_mut();
        if f.fail_save {
            return Err(StoreError::CopyFailed);
        }
        if at != f.at {
            return Err(StoreError::NotAtMark);
        }
        f.live += 1;
        Ok(Box::new(State { at: f.at, sum: f.sum }))
    }
    fn restore(
        &mut self,
        _owner: *mut core::ffi::c_void,
        saved: &Saved,
    ) -> Result<(), StoreError> {
        let mut f = self.0.borrow_mut();
        if f.fail_restore {
            return Err(StoreError::CopyFailed);
        }
        let st = saved.downcast_ref::<State>().expect("the fake's state");
        f.at = st.at;
        f.sum = st.sum;
        Ok(())
    }
    fn drop_saved(&mut self, saved: Saved) {
        self.0.borrow_mut().live -= 1;
        drop(saved);
    }
    fn saved_ptr(saved: &Saved) -> *mut core::ffi::c_void {
        let st = saved.downcast_ref::<State>().expect("the fake's state");
        st as *const State as *mut core::ffi::c_void
    }
    fn writes_learned(&self) -> bool {
        self.0.borrow().disk
    }
    fn reads_learned(&self) -> bool {
        self.0.borrow().disk
    }
    fn write(
        &mut self,
        saved: &Saved,
        dir: &str,
        key: u64,
    ) -> Option<Result<(), StoreError>> {
        let mut f = self.0.borrow_mut();
        if !f.disk {
            return None;
        }
        f.writes += 1;
        if f.fail_write {
            return Some(Err(StoreError::WriteFailed));
        }
        let st = saved.downcast_ref::<State>().expect("the fake's state");
        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&st.at.to_le_bytes());
        data.extend_from_slice(&st.sum.to_le_bytes());
        let path = format!("{dir}/{key:016x}.bin");
        Some(
            std::fs::write(&path, &data)
                .map(|_| ())
                .map_err(|_| StoreError::WriteFailed),
        )
    }
    fn read(&mut self, dir: &str, key: u64, at: u32) -> Option<Result<Saved, StoreError>> {
        let mut f = self.0.borrow_mut();
        if !f.disk {
            return None;
        }
        let path = format!("{dir}/{key:016x}.bin");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => return Some(Err(StoreError::ReadFailed)),
        };
        if data.len() != 12 {
            return Some(Err(StoreError::ReadFailed));
        }
        let st = State {
            at: u32::from_le_bytes(data[0..4].try_into().unwrap()),
            sum: u64::from_le_bytes(data[4..12].try_into().unwrap()),
        };
        if st.at != at {
            return Some(Err(StoreError::ReadFailed));
        }
        f.live += 1;
        Some(Ok(Box::new(st)))
    }
    fn forget(&mut self, dir: &str, key: u64) {
        let mut f = self.0.borrow_mut();
        if !f.disk {
            return;
        }
        f.disk_forgets += 1;
        let _ = std::fs::remove_file(format!("{dir}/{key:016x}.bin"));
    }
    fn forget_checked(&mut self, dir: &str, key: u64) -> Option<Result<(), StoreError>> {
        let mut f = self.0.borrow_mut();
        if !f.disk {
            return None;
        }
        if f.fail_delete {
            return Some(Err(StoreError::DeleteFailed));
        }
        let checked = single_forget_checked(dir, key);
        // Upstream's checkedForgetFn counts the forget and unlinks again
        // through forgetFn (the idempotent second pass)
        f.disk_forgets += 1;
        let _ = std::fs::remove_file(format!("{dir}/{key:016x}.bin"));
        Some(checked)
    }
    fn reclaim(&mut self, dir: &str, key: u64) -> Option<Result<u64, StoreError>> {
        let f = self.0.borrow();
        if !f.disk {
            return None;
        }
        Some(single_reclaim(dir, key))
    }
}

/// The sum a fresh pass over `prompt` gives. Upstream: `fresh`.
pub fn fresh(prompt: &[u32]) -> u64 {
    let mut sum: u64 = 0;
    for &t in prompt {
        sum = sum.wrapping_mul(31).wrapping_add(t as u64);
    }
    sum
}

impl Fake {
    /// The learned-disk slots stand (upstream's `learned()` and
    /// `paired()` — one flag here, the peer pair the fabric tranche's).
    pub fn learned(self) -> Fake {
        self.0.borrow_mut().disk = true;
        self
    }

    /// A prompt pass from `plan.from`, keeping at each mark; returns the
    /// sum the pass gives. Upstream: `Fake.pass`.
    pub fn pass(&mut self, s: &mut Store<Fake>, prompt: &[u32], plan: &Plan) -> u64 {
        if plan.from == 0 {
            // Upstream's pass reset keeps the live count, the failure flags
            // and the write counter (prompt_cache_fake.zig:181) — and the
            // port's disk flag, which rides where upstream's vtable
            // choice does
            let (live, fail_save, fail_restore, fail_write, writes, disk) = {
                let f = self.0.borrow();
                (
                    f.live,
                    f.fail_save,
                    f.fail_restore,
                    f.fail_write,
                    f.writes,
                    f.disk,
                )
            };
            *self.0.borrow_mut() = FakeInner {
                live,
                fail_save,
                fail_restore,
                fail_write,
                writes,
                disk,
                ..FakeInner::default()
            };
        }
        let mut mi = 0;
        for &t in &prompt[plan.from as usize..] {
            let at_now = {
                let mut f = self.0.borrow_mut();
                f.sum = f.sum.wrapping_mul(31).wrapping_add(t as u64);
                f.at += 1;
                f.at
            };
            if mi < plan.marks.len() && plan.marks[mi] == at_now {
                s.keep(prompt, at_now, std::ptr::null_mut(), &[], &[]);
                mi += 1;
            }
        }
        self.0.borrow().sum
    }
}

