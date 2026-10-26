//! A peer's disk reservation covers its snapshot half until rank 0 has
//! committed or removed the shared index entry.
//!
//! Port of `zig/src/families/glm/learned_pair.zig` (117 lines) —
//! upstream's three tests mirrored (the third's mkdir/ftruncate side
//! over std::fs, learned_dirs' own total/removeOther underneath).

use tensorfold_lanes::learned_dirs;
use tensorfold_lanes::learned_disk::{Admission, Result_};

/// Upstream: `State` (learned_pair.zig:4).
pub struct State {
    pub admission: Admission,
    pub cap: u64,
    pub pending: Option<Pending>,
}

#[derive(Clone, Copy)]
pub struct Pending {
    pub key: u64,
    pub bytes: u64,
}

/// Why the reservation refused.
#[derive(Debug, PartialEq)]
pub enum PairError {
    PeerDiskBusy,
    PeerDiskFull,
    PeerDiskOutOfStep,
}

impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairError::PeerDiskBusy => write!(f, "a peer reservation is in flight"),
            PairError::PeerDiskFull => write!(f, "the peer's disk share is full"),
            PairError::PeerDiskOutOfStep => write!(f, "the peer reservation is out of step"),
        }
    }
}

impl std::error::Error for PairError {}

impl State {
    /// Upstream: `need` (learned_pair.zig:8) — the bytes the write
    /// still needs (u64::MAX while it cannot tell).
    pub fn need(&self, dir: &str, payload: u64, extra: u64) -> u64 {
        if self.pending.is_some() || self.admission.waiting(dir) {
            return u64::MAX;
        }
        let bytes = match payload.checked_add(extra) {
            Some(b) => b,
            None => return u64::MAX,
        };
        let Some(floor_need) = self.admission.shortfall(dir, bytes) else {
            return u64::MAX;
        };
        let used = match learned_dirs::total(learned_dirs::root_of(dir)) {
            Ok(u) => u,
            Err(_) => return u64::MAX,
        };
        let with = match used.checked_add(bytes) {
            Some(w) => w,
            None => return u64::MAX,
        };
        floor_need.max(with.saturating_sub(self.cap))
    }

    /// Upstream: `reserve` (learned_pair.zig:15).
    pub fn reserve(&mut self, dir: &str, key: u64, payload: u64, extra: u64) -> Result<(), PairError> {
        if self.pending.is_some() {
            return Err(PairError::PeerDiskBusy);
        }
        let bytes = payload
            .checked_add(extra)
            .ok_or(PairError::PeerDiskFull)?;
        if self.need(dir, payload, extra) != 0 {
            return Err(PairError::PeerDiskFull);
        }
        if self.admission.reserve(dir, bytes) != Result_::Ready {
            return Err(PairError::PeerDiskFull);
        }
        self.pending = Some(Pending { key, bytes });
        Ok(())
    }

    /// Upstream: `finish` (learned_pair.zig:22) — rollback also
    /// accepts no pending operation (the reply may have been lost).
    pub fn finish(&mut self, dir: &str, key: u64, success: bool) -> Result<(), PairError> {
        let Some(pending) = self.pending else {
            return Ok(());
        };
        if pending.key != key {
            return Err(PairError::PeerDiskOutOfStep);
        }
        self.admission.finish(dir, pending.bytes, success);
        self.pending = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `a peer floor or in-flight reservation prevents
    /// writes and failure backs off until space changes`
    /// (learned_pair.zig:29).
    #[test]
    fn a_peer_floor_or_in_flight_reservation_prevents_writes_and_failure_backs_off_until_space_changes() {
        use std::cell::Cell;
        use std::sync::atomic::{AtomicU64, Ordering};
        static FREE_BYTES: AtomicU64 = AtomicU64::new(109);
        static TICK: AtomicU64 = AtomicU64::new(10);
        fn free(_: &str) -> Option<u64> {
            Some(FREE_BYTES.load(Ordering::Relaxed))
        }
        fn clock() -> Option<u64> {
            Some(TICK.load(Ordering::Relaxed))
        }
        let mut s = State {
            admission: Admission::with_readers(free, clock),
            cap: u64::MAX,
            pending: None,
        };
        s.admission.floor = 100;
        assert_eq!(1, s.need(".", 8, 2));
        assert_eq!(Err(PairError::PeerDiskFull), s.reserve(".", 1, 8, 2));
        assert!(s.pending.is_none());
        FREE_BYTES.store(110, Ordering::Relaxed);
        s.reserve(".", 1, 8, 2).unwrap();
        assert_eq!(u64::MAX, s.need(".", 1, 0));
        assert_eq!(Err(PairError::PeerDiskOutOfStep), s.finish(".", 2, true));
        s.finish(".", 1, false).unwrap();
        assert_eq!(0, s.admission.reserved);
        assert_eq!(Err(PairError::PeerDiskFull), s.reserve(".", 1, 8, 2));
        FREE_BYTES.store(120, Ordering::Relaxed);
        s.reserve(".", 1, 8, 2).unwrap();
        s.finish(".", 1, true).unwrap();
        let _ = Cell::new(0); // the Cell import keeps the atomic-only shape honest
    }

    /// Upstream's `lost reservation acknowledgement rolls back
    /// idempotently and each rank enforces its own cap`
    /// (learned_pair.zig:57).
    #[test]
    fn lost_reservation_acknowledgement_rolls_back_idempotently_and_each_rank_enforces_its_own_cap() {
        fn free(_: &str) -> Option<u64> {
            Some(1 << 30)
        }
        fn clock() -> Option<u64> {
            Some(100)
        }
        let mut s = State {
            admission: Admission::with_readers(free, clock),
            cap: 9,
            pending: None,
        };
        s.admission.floor = 0;
        assert_eq!(1, s.need(".", 8, 2));
        assert_eq!(Err(PairError::PeerDiskFull), s.reserve(".", 1, 8, 2));
        s.cap = 100;
        s.reserve(".", 1, 8, 2).unwrap();
        // The caller cannot know whether the reserve reply arrived, so
        // rollback also accepts no pending operation.
        s.finish(".", 1, false).unwrap();
        s.finish(".", 1, false).unwrap();
        assert_eq!(0, s.admission.reserved);
        s.admission.set_retry_at(0);
        s.reserve(".", 2, 8, 2).unwrap();
        assert_eq!(Err(PairError::PeerDiskOutOfStep), s.finish(".", 1, false));
        s.finish(".", 2, true).unwrap();
    }

    /// Upstream's `each rank includes actual other-identity files in
    /// its independent cap before reservation` (learned_pair.zig:81) —
    /// the dirs created over std::fs, learned_dirs' own total and
    /// removeOther underneath.
    #[test]
    fn each_rank_includes_actual_other_identity_files_in_its_independent_cap_before_reservation() {
        use std::sync::atomic::AtomicBool;
        fn free(_: &str) -> Option<u64> {
            Some(1 << 30)
        }
        fn clock() -> Option<u64> {
            Some(100)
        }
        fn make(dir: &std::path::Path, id: u64) -> std::path::PathBuf {
            let d = dir.join(format!("{id:016}"));
            std::fs::create_dir(&d).unwrap();
            d
        }
        let root_tmp = tempfile_dir("tf-glm-pair");
        let root = root_tmp.to_string_lossy().into_owned();
        let current = make(&root_tmp, 1);
        let other = make(&root_tmp, 2);
        // the other identity's state.bin: 96 bytes on disk
        std::fs::write(other.join("state.bin"), vec![0u8; 96]).unwrap();
        let current = current.to_string_lossy().into_owned();
        let mut rank0 = State {
            admission: Admission::with_readers(free, clock),
            cap: 200,
            pending: None,
        };
        rank0.admission.floor = 0;
        let mut rank1 = State {
            admission: Admission::with_readers(free, clock),
            cap: 100,
            pending: None,
        };
        rank1.admission.floor = 0;
        assert_eq!(0, rank0.need(&current, 8, 2));
        assert_eq!(6, rank1.need(&current, 8, 2));
        assert_eq!(Err(PairError::PeerDiskFull), rank1.reserve(&current, 3, 8, 2));
        assert!(rank1.pending.is_none());
        learned_dirs::remove_other(&root, &current, 2).unwrap();
        rank1.reserve(&current, 3, 8, 2).unwrap();
        rank1.finish(&current, 3, true).unwrap();
        assert_eq!(0, rank1.admission.reserved);
        let _ = AtomicBool::new(false);
        std::fs::remove_dir_all(&root_tmp).ok();
    }

    fn tempfile_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
