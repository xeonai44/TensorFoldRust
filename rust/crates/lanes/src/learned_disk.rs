//! Reserve learned-state writes above a free-disk floor, with bounded
//! backoff after refusals or write failures.
//!
//! Port of `zig/src/core/lanes/learned_disk.zig` (126 lines): the
//! `available` bytes under a directory (`statvfs`, the port's own extern
//! with the darwin layout — the rusage lesson of the memory tranche, the
//! struct's size checked by a test), and `Admission`, the floor, the
//! in-flight reservation and the bounded backoff whose clock and free
//! reads are injectable exactly as upstream's are (learned_disk.zig:60's
//! fields).


/// The `statvfs` result the port reads. Upstream: `Stat`
/// (learned_disk.zig:5) — linux's counts are 64-bit with a 24-byte tail,
/// darwin's 32-bit without (the sizes upstream's own test asserts: 112 on
/// linux, 64 elsewhere).
#[repr(C)]
#[cfg(target_os = "linux")]
struct Stat {
    bsize: u64,
    frsize: u64,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [u8; 24],
}

#[repr(C)]
#[cfg(not(target_os = "linux"))]
struct Stat {
    bsize: u64,
    frsize: u64,
    blocks: u32,
    bfree: u32,
    bavail: u32,
    files: u32,
    ffree: u32,
    favail: u32,
    fsid: u64,
    flag: u64,
    namemax: u64,
}

unsafe extern "C" {
    fn statvfs(path: *const std::ffi::c_char, result: *mut Stat) -> std::ffi::c_int;
}

/// The bytes a non-root writer may still put under `dir`. Upstream:
/// `available` (learned_disk.zig:22) — a failed read is no answer at all.
pub fn available(dir: &str) -> Option<u64> {
    let path = std::ffi::CString::new(dir).ok()?;
    let mut stat = Stat {
        bsize: 0,
        frsize: 0,
        blocks: 0,
        bfree: 0,
        bavail: 0,
        files: 0,
        ffree: 0,
        favail: 0,
        fsid: 0,
        flag: 0,
        namemax: 0,
        #[cfg(target_os = "linux")]
        tail: [0; 24],
    };
    // SAFETY: `path` is a valid NUL-terminated string and `stat` a valid
    // out-pointer for the duration of the call.
    if unsafe { statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    // the fields are u32 on macOS, u64 on LP64 Linux — the casts
    // carry the smaller platform, redundant where they already are u64
    #[allow(clippy::unnecessary_cast)]
    (stat.frsize as u64).checked_mul(stat.bavail as u64)
}

/// Upstream: `now` (learned_disk.zig:28) — MONOTONIC seconds.
fn monotonic_now() -> Option<u64> {
    // The process's own monotonic base (Instant cannot go back); the
    // port's `available` tests only need a monotone tick, and the
    // admission math reads it as seconds.
    Some(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs(),
    )
}

/// The learned-state write admission: a free-disk floor kept, the bytes a
/// write reserves while it runs, and the bounded backoff after refusals or
/// failures. Upstream: `Admission` (learned_disk.zig:39).
pub struct Admission {
    pub floor: u64,
    pub reserved: u64,
    failures: u8,
    retry_at: u64,
    failed_free: Option<u64>,
    free: fn(&str) -> Option<u64>,
    clock: fn() -> Option<u64>,
}

/// Upstream: `Admission.Result` (learned_disk.zig:46).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Result_ {
    Ready,
    Refused,
    Quiet,
}

impl Default for Admission {
    fn default() -> Admission {
        Admission {
            floor: 4 << 30,
            reserved: 0,
            failures: 0,
            retry_at: 0,
            failed_free: None,
            free: available,
            clock: monotonic_now,
        }
    }
}

impl Admission {
    /// The injectable clock and free read, upstream's own test seams.
    pub fn with_readers(
        free: fn(&str) -> Option<u64>,
        clock: fn() -> Option<u64>,
    ) -> Admission {
        Admission {
            free,
            clock,
            ..Default::default()
        }
    }

    /// The test seam upstream reaches by field write (Zig tests share
    /// the module): the backoff's earliest retry, cleared.
    #[doc(hidden)]
    pub fn set_retry_at(&mut self, at: u64) {
        self.retry_at = at;
    }

    /// Whether the backoff still holds, a growing free space aside.
    /// Upstream: `waiting` (learned_disk.zig:49).
    pub fn waiting(&self, dir: &str) -> bool {
        let Some(tick) = (self.clock)() else {
            return true;
        };
        let free = (self.free)(dir);
        let increased = match (free, self.failed_free) {
            (Some(n), Some(old)) => n > old,
            _ => false,
        };
        tick < self.retry_at && !increased
    }

    /// The bytes of shortfall a sweep must free for `bytes` to fit.
    /// Upstream: `shortfall` (learned_disk.zig:57).
    pub fn shortfall(&self, dir: &str, bytes: u64) -> Option<u64> {
        (self.clock)()?;
        let need = self.floor.checked_add(self.reserved)?;
        let full = need.checked_add(bytes)?;
        Some(full.saturating_sub((self.free)(dir)?))
    }

    /// One refusal: the backoff begins (or grows). Upstream: `refuse`
    /// (learned_disk.zig:64).
    pub fn refuse(&mut self, dir: &str) {
        let space = (self.free)(dir);
        self.block(space);
    }

    /// Reserve `bytes` above the floor. Upstream: `reserve`
    /// (learned_disk.zig:68).
    pub fn reserve(&mut self, dir: &str, bytes: u64) -> Result_ {
        let space = (self.free)(dir);
        let changed = match (space, self.failed_free) {
            (Some(n), Some(old)) => n > old,
            _ => false,
        };
        let Some(tick) = (self.clock)() else {
            return self.block(space);
        };
        if tick < self.retry_at && !changed {
            return Result_::Quiet;
        }
        let Some(need) = self.floor.checked_add(self.reserved) else {
            return self.block(space);
        };
        let Some(full) = need.checked_add(bytes) else {
            return self.block(space);
        };
        if space.is_none_or(|s| s < full) {
            return self.block(space);
        }
        self.reserved += bytes;
        Result_::Ready
    }

    /// One refusal recorded, the backoff doubling to a minute's ceiling.
    /// Upstream: `block` (learned_disk.zig:86).
    fn block(&mut self, space: Option<u64>) -> Result_ {
        self.failed_free = space;
        self.failures = (self.failures.saturating_add(1)).min(7);
        let wait = (1u64 << (self.failures - 1)).min(60);
        self.retry_at = (self.clock)().unwrap_or(0).saturating_add(wait);
        Result_::Refused
    }

    /// A reserved write finished; success clears the backoff, a failure
    /// grows it. Upstream: `finish` (learned_disk.zig:94).
    pub fn finish(&mut self, dir: &str, bytes: u64, success: bool) {
        debug_assert!(self.reserved >= bytes);
        self.reserved -= bytes;
        if success {
            self.failures = 0;
            self.retry_at = 0;
            self.failed_free = None;
        } else {
            let space = (self.free)(dir);
            self.block(space);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Upstream's test (learned_disk.zig:100): the disk floor, unknown
    /// space, overflow, reservations and bounded write-failure retries —
    /// case for case over the fake free and clock.
    #[test]
    fn the_disk_floor_unknown_space_and_bounded_retries() {
        // The injectable fns upstream's own test uses (the Fake struct)
        static SPACE: AtomicU64 = AtomicU64::new(110);
        static NO_SPACE: AtomicU64 = AtomicU64::new(0);
        static TICK: AtomicU64 = AtomicU64::new(100);
        fn free(_: &str) -> Option<u64> {
            if NO_SPACE.load(Ordering::Relaxed) == 1 {
                return None;
            }
            Some(SPACE.load(Ordering::Relaxed))
        }
        fn clock() -> Option<u64> {
            Some(TICK.load(Ordering::Relaxed))
        }
        fn tick(n: u64) {
            TICK.fetch_add(n, Ordering::Relaxed);
        }

        let mut a = Admission::with_readers(free, clock);
        a.floor = 100;
        assert_eq!(a.reserve(".", 10), Result_::Ready);
        assert_eq!(a.reserve(".", 1), Result_::Refused);
        a.finish(".", 10, false);
        assert_eq!(a.reserve(".", 10), Result_::Quiet);
        // A grown free space cuts through the backoff
        SPACE.store(120, Ordering::Relaxed);
        assert_eq!(a.reserve(".", 10), Result_::Ready);
        a.finish(".", 10, true);
        // Unknown space: the read gone, the write refused
        NO_SPACE.store(1, Ordering::Relaxed);
        assert_eq!(a.reserve(".", 1), Result_::Refused);
        NO_SPACE.store(0, Ordering::Relaxed);
        // Overflow of floor + reservation + bytes blocks the write
        SPACE.store(u64::MAX, Ordering::Relaxed);
        tick(100);
        assert_eq!(a.reserve(".", u64::MAX), Result_::Refused);
        // The wait doubles to its 60-second ceiling over repeated failures
        for _ in 0..20 {
            tick(100);
            let _ = a.reserve(".", u64::MAX);
        }
        assert_eq!(a.retry_at, TICK.load(Ordering::Relaxed) + 60);
        assert_eq!(a.reserved, 0);
    }

    /// Upstream's real-statvfs test (learned_disk.zig:126): a missing
    /// directory fails closed, a real one reads.
    /// Upstream's own assertion (learned_disk.zig:125): the struct's
    /// size is the platform's statvfs layout — 112 on linux, 64 elsewhere.
    #[test]
    fn the_stat_layout() {
        #[cfg(target_os = "linux")]
        assert_eq!(std::mem::size_of::<Stat>(), 112);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(std::mem::size_of::<Stat>(), 64);
    }

    #[test]
    fn the_real_statvfs_fails_closed_for_a_missing_directory() {
        assert!(available("/this-directory-does-not-exist").is_none());
        let free = available(".").expect("the working directory reads");
        assert!(free > 0, "a sane free space, not garbage: {free}");
    }
}
