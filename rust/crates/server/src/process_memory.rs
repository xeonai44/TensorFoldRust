//! Kernel process footprint readings; device allocator counters have a
//! separate scope and reset policy.
//!
//! Port of `zig/src/server/process_memory.zig` (77 lines) — Darwin's
//! rusage_info_v4, the two fields the memory route publishes.

/// Upstream: `Snapshot` (process_memory.zig:4).
pub struct Snapshot {
    pub physical_footprint_bytes: u64,
    pub lifetime_peak_physical_footprint_bytes: u64,
}

/// One kernel sample, including peaks between HTTP polls; None on
/// unsupported platforms or failed reads. Upstream: `read`
/// (process_memory.zig:51) — the struct's ABI laid out to match
/// rusage_info_v4, the field offsets the same.
pub fn read() -> Option<Snapshot> {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct RUsageInfoV4([u64; 43]);
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut core::ffi::c_void) -> i32;
            fn getpid() -> i32;
        }
        // The ABI's indices: phys_footprint at 9, lifetime_max_phys_
        // footprint at 30 (rusage_info_v4's field order, upstream's extern
        // struct the same count: uuid, times, wakeups, pageins, wired,
        // resident, phys_footprint, ... lifetime_max_phys_footprint).
        let mut info = RUsageInfoV4([0; 43]);
        let ok = unsafe {
            proc_pid_rusage(getpid(), 4, (&mut info as *mut RUsageInfoV4).cast())
        };
        (ok == 0).then_some(Snapshot {
            physical_footprint_bytes: info.0[9],
            lifetime_peak_physical_footprint_bytes: info.0[30],
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reading on this Mac (the upstream test's shape, memory_read).
    #[test]
    fn the_kernel_reading() {
        if cfg!(not(target_os = "macos")) {
            return;
        }
        let snap = read().expect("macOS reads its rusage");
        assert!(snap.physical_footprint_bytes > 0);
        assert!(snap.lifetime_peak_physical_footprint_bytes >= snap.physical_footprint_bytes);
    }
}
