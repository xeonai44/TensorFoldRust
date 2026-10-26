//! The prompt cache's budget from memory: what 70% of RAM leaves past the
//! server's footprint once loaded, less a margin.
//!
//! Port of `zig/src/native/cache_fit.zig` (99 lines). The budget plan the
//! family that arms a store computes at open (upstream's nemotron host,
//! metal.zig:155); this port's qwen35 serve arms none (the map's tranche
//! twenty-six), so the module waits for that family — its tests keep the
//! plan verified.

#![allow(dead_code)]

/// Upstream: `GiB` / `SHARE_PERCENT` / `MARGIN` (cache_fit.zig:5-8).
pub const GIB: u64 = 1 << 30;
pub const SHARE_PERCENT: u64 = 70;
pub const MARGIN: u64 = 2 * GIB;

/// The cache's budget, what fits under the cap, and the cap itself (bytes).
/// Upstream: `Fit` (cache_fit.zig:11).
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub budget: u64,
    pub room: u64,
    pub cap: u64,
}

/// The budget for `total` bytes of RAM and a footprint of `ready` bytes
/// once loaded: `gib` when given and it fits (or `over`), else all that
/// fits; a given budget past the room is refused. Upstream: `fit`
/// (cache_fit.zig:15).
pub fn fit(total: u64, ready: u64, gib: Option<f64>, over: bool) -> Result<Fit, String> {
    let cap = total / 100 * SHARE_PERCENT;
    let room = cap.saturating_sub(ready).saturating_sub(MARGIN);
    let want = match gib {
        Some(g) if g > 0.0 => (g * GIB as f64) as u64,
        Some(_) => 0,
        None => room,
    };
    if want > room && !over {
        return Err("CacheOverCap".into());
    }
    Ok(Fit {
        budget: want,
        room,
        cap,
    })
}

/// This machine's RAM (None where the OS does not say). Upstream: `ram`
/// (cache_fit.zig:25) — macOS's `hw.memsize`.
pub fn ram() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctl(
                name: *const i32,
                namelen: u32,
                oldp: *mut core::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut core::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mib: [i32; 2] = [6 /* CTL_HW */, 24 /* HW_MEMSIZE */];
        let mut value: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        let ok = unsafe {
            sysctl(
                mib.as_ptr(),
                2,
                (&mut value as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (ok == 0).then_some(value)
    }
    #[cfg(target_os = "linux")]
    {
        // The std's own read on Linux: /proc/meminfo's MemTotal (kB)
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kb.checked_mul(1024)
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "linux")))]
    {
        None
    }
}

/// This process's physical footprint as macOS counts it, Metal buffers
/// included (rusage_info_v4; None elsewhere). Upstream: `footprint`
/// (cache_fit.zig:32).
pub fn footprint() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct RUsageInfoV4([u64; 43]);
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut core::ffi::c_void) -> i32;
            fn getpid() -> i32;
        }
        let mut info = RUsageInfoV4([0; 43]);
        let ok = unsafe {
            proc_pid_rusage(getpid(), 4, (&mut info as *mut RUsageInfoV4).cast())
        };
        (ok == 0).then_some(info.0[9])
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// GiB with one decimal, for log lines and messages. Upstream: `gibs`
/// (cache_fit.zig:86).
pub fn gibs(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

/// The cache plan: what 70% of RAM leaves past this process, or `gib` if
/// it fits (else refused unless `over`). Upstream: `budget`
/// (cache_fit.zig:37) — the plan the Info reports and the store keeps.
pub fn budget(
    gib: Option<f64>,
    over: bool,
    spare: u64,
) -> Result<tensorfold_engine::PromptCachePlan, String> {
    use tensorfold_engine::{PromptCacheSource, PromptCachePlan};
    let total = ram().unwrap_or(0);
    let ready = footprint().unwrap_or(0);
    if total == 0 || ready == 0 {
        let b = match gib {
            Some(g) if g > 0.0 => (g * GIB as f64) as u64,
            _ => spare,
        };
        eprintln!(
            "prompt cache: {b} bytes for kept prompt states (no memory reading)"
        );
        return Ok(PromptCachePlan {
            source: if gib.is_some() {
                PromptCacheSource::Explicit
            } else {
                PromptCacheSource::MetalWorkingSet
            },
            budget_bytes: b,
            explicit_budget: gib.is_some(),
            ..PromptCachePlan::default()
        });
    }
    let f = fit(total, ready, gib, over).map_err(|_| {
        // The room without a given budget never refuses
        let left = fit(total, ready, None, false)
            .map(|f| f.room)
            .unwrap_or(0);
        format!(
            "--prompt-cache-gib {} would take this server past 70% of this Mac's memory: it holds {} GiB once loaded, 70% of {} GiB is {} GiB, and {} GiB stays free for prompt buffers, so {} GiB is left for kept prompt states. Leave --prompt-cache-gib out to use that, pass a smaller one, or add --prompt-cache-over-cap to keep {} GiB anyway.",
            gib.unwrap_or_default(),
            gibs(ready),
            gibs(total),
            gibs(total / 100 * SHARE_PERCENT),
            MARGIN >> 30,
            gibs(left),
            gib.unwrap_or_default(),
        )
    })?;
    eprintln!(
        "prompt cache: {} GiB from {} GiB free under the 70% cap ({} GiB in use once loaded, {} GiB of RAM, {} GiB kept for prompt buffers)",
        gibs(f.budget),
        gibs(f.room),
        gibs(ready),
        gibs(total),
        MARGIN >> 30,
    );
    Ok(PromptCachePlan {
        source: PromptCacheSource::PhysicalFootprint,
        budget_bytes: f.budget,
        explicit_budget: gib.is_some(),
        over_cap: Some(f.budget > f.room),
        ram_bytes: Some(total),
        ready_footprint_bytes: Some(ready),
        cap_bytes: Some(f.cap),
        room_bytes: Some(f.room),
        margin_bytes: Some(MARGIN),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The upstream test (cache_fit.zig:89), assertion for assertion: the
    /// default budget is what 70% of RAM leaves past the footprint at
    /// ready and the margin.
    #[test]
    fn the_default_budget_is_what_70_percent_leaves() {
        let ram256: u64 = 256 * GIB;
        let f = fit(ram256, 172 * GIB, None, false).unwrap(); // this tip on a 256 GB M5 Ultra
        assert_eq!(f.cap, ram256 / 100 * 70);
        assert_eq!(f.budget, f.cap - 174 * GIB);
        assert!(gibs(f.budget) > 5.0 && gibs(f.budget) < 5.3);
        assert_eq!(fit(ram256, 179 * GIB, None, false).unwrap().budget, 0); // nothing left: no cache
        assert_eq!(fit(192 * GIB, 172 * GIB, None, false).unwrap().budget, 0); // a 192 GB Mac: past the cap already
    }

    /// The upstream test (cache_fit.zig:96): a given budget that fits is
    /// kept, a larger one refused unless the override.
    #[test]
    fn a_given_budget_kept_or_refused() {
        let ram256: u64 = 256 * GIB;
        assert_eq!(fit(ram256, 172 * GIB, Some(4.0), false).unwrap().budget, 4 * GIB);
        assert_eq!(fit(ram256, 172 * GIB, Some(0.0), false).unwrap().budget, 0);
        assert!(fit(ram256, 172 * GIB, Some(8.0), false).is_err());
        assert!(fit(ram256, 172 * GIB, Some(16.0), false).is_err());
        assert_eq!(fit(ram256, 172 * GIB, Some(16.0), true).unwrap().budget, 16 * GIB);
        assert_eq!(fit(512 * GIB, 172 * GIB, Some(64.0), false).unwrap().budget, 64 * GIB); // a 512 GB Mac has room
    }
}
