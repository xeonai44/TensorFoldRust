//! CUDA memory admission arithmetic and host-pool validation, without
//! a device.
//!
//! Port of `zig/src/native/cuda_memory.zig` (170 lines) — upstream's
//! four tests mirrored (the meminfo live read runs on the box, where
//! /proc/meminfo stands).

/// What /proc/meminfo says: MemTotal and MemAvailable in bytes (the
/// page cache counts as available). Upstream: `MemInfo`
/// (cuda_memory.zig:6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
}

/// Why the admission refused.
#[derive(Debug, PartialEq)]
pub enum MemoryError {
    /// A variable's text is not a number in range.
    Invalid,
    /// Unified memory without a readable /proc/meminfo.
    HostMemoryUnavailable,
    /// No stream fits.
    NoStream,
    /// A fixed --parallel count the room cannot hold.
    TooMany,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::Invalid => write!(f, "a memory variable is not a number in range"),
            MemoryError::HostMemoryUnavailable => write!(f, "the host memory cannot be read"),
            MemoryError::NoStream => write!(f, "no stream fits"),
            MemoryError::TooMany => write!(f, "the fixed stream count exceeds the room"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Upstream: `meminfo` (cuda_memory.zig:8).
pub fn meminfo(text: &str) -> Option<MemInfo> {
    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;
    for row in text.split('\n') {
        // the tokenizer skips blank rows (a trailing newline is not one)
        let mut words = row.split([':', ' ', '\t']).filter(|w| !w.is_empty());
        let key = match words.next() {
            Some(k) => k,
            None => continue,
        };
        let is_total = key == "MemTotal";
        if !is_total && key != "MemAvailable" {
            continue;
        }
        let kib: u64 = words.next()?.parse().ok()?;
        if words.next()? != "kB" || words.next().is_some() {
            return None;
        }
        let bytes = kib.checked_mul(1024)?;
        if is_total {
            if total.is_some() {
                return None;
            }
            total = Some(bytes);
        } else {
            if available.is_some() {
                return None;
            }
            available = Some(bytes);
        }
    }
    let t = total?;
    let a = available?;
    if t > 0 && a <= t {
        Some(MemInfo { total: t, available: a })
    } else {
        None
    }
}

/// CUDA allocation room after the reserve and optional process-memory
/// cap. Upstream: `Pool` (cuda_memory.zig:34).
#[derive(Clone, Copy, Debug)]
pub struct Pool {
    pub free: u64,
    pub total: u64,
    pub reserve: u64,
    pub limit: Option<u64>,
    pub unified: bool,
}

impl Pool {
    /// Bytes the engine may still allocate on top of `held`. Upstream:
    /// `room` (cuda_memory.zig:42).
    pub fn room(&self, held: u64) -> u64 {
        let left = self.free.saturating_sub(self.reserve);
        match self.limit {
            Some(cap) => left.min(cap.saturating_sub(held)),
            None => left,
        }
    }
}

/// The reserve's default share and floor, and the least GiB a variable
/// may set: unified memory is the OS's too. Upstream: `Floor`
/// (cuda_memory.zig:49).
pub struct Floor {
    pub least: f64,
    pub at_least: u64,
    pub share: u64,
}

const GIB: f64 = (1u64 << 30) as f64;

/// Upstream: `floor` (cuda_memory.zig:51).
pub fn floor(unified: bool) -> Floor {
    if unified {
        Floor {
            least: 2.0,
            at_least: 4 << 30,
            share: 10,
        }
    } else {
        Floor {
            least: 0.5,
            at_least: 1 << 30,
            share: 32,
        }
    }
}

/// TENSORFOLD_MEMORY_RESERVE_GIB (from the floor's least up to the
/// pool's size), else the floor's share of the pool. Upstream:
/// `reserveBytes` (cuda_memory.zig:56).
pub fn reserve_bytes(text: Option<&str>, total: u64, unified: bool) -> Result<u64, MemoryError> {
    let f = floor(unified);
    let t = text.unwrap_or("").trim();
    if t.is_empty() {
        return Ok(f.at_least.max(total / f.share));
    }
    let g: f64 = t.parse().map_err(|_| MemoryError::Invalid)?;
    if g < f.least {
        return Err(MemoryError::Invalid);
    }
    let bytes = scaled_bytes(g)?;
    if bytes <= total {
        Ok(bytes)
    } else {
        Err(MemoryError::Invalid)
    }
}

/// TENSORFOLD_CUDA_MEMORY_LIMIT_GB in bytes; None when unset. Upstream:
/// `limitBytes` (cuda_memory.zig:67).
pub fn limit_bytes(text: Option<&str>) -> Result<Option<u64>, MemoryError> {
    let t = match text {
        None => return Ok(None),
        Some(t) => t.trim(),
    };
    if t.is_empty() {
        return Ok(None);
    }
    let g: f64 = t.parse().map_err(|_| MemoryError::Invalid)?;
    Ok(Some(scaled_bytes(g)?))
}

fn scaled_bytes(g: f64) -> Result<u64, MemoryError> {
    let bytes = g * GIB;
    let upper: f64 = 18446744073709551616.0; // 2^64
    if !(bytes.is_finite() && bytes >= 1.0 && bytes < upper) {
        return Err(MemoryError::Invalid);
    }
    Ok(bytes as u64)
}

/// Free streams use no room; admit all asked when they fit, refusing
/// zero or an oversized fixed count. Upstream: `admit`
/// (cuda_memory.zig:82).
pub fn admit(
    room: u64,
    stream: u64,
    free: u32,
    asked: u32,
    fixed: bool,
) -> Result<u32, MemoryError> {
    let more: u32 = if stream == 0 {
        asked
    } else {
        // upstream: std.math.cast(room / stream) orelse maxInt — the
        // checked division IS upstream's own arithmetic
        match room.checked_div(stream) {
            Some(q) => u32::try_from(q).unwrap_or(u32::MAX),
            None => u32::MAX,
        }
    };
    let fits = free.saturating_add(more);
    if fits == 0 {
        return Err(MemoryError::NoStream);
    }
    if fits >= asked {
        return Ok(asked);
    }
    if fixed {
        Err(MemoryError::TooMany)
    } else {
        Ok(fits)
    }
}

/// Kept prompt states' budget: --prompt-cache-gib (0: none), else 16
/// GiB, never past what the streams leave of `room`. Upstream:
/// `cacheBytes` (cuda_memory.zig:91).
pub fn cache_bytes(asked: Option<f64>, room: u64, streams: u64) -> u64 {
    let want: u64 = match asked {
        Some(g) if g > 0.0 => (g * GIB) as u64,
        Some(_) => 0,
        None => 16 << 30,
    };
    room.saturating_sub(streams).min(want)
}

/// Integrated GPUs require validated host counts; discrete devices use
/// CUDA's counts. Upstream: `counts` (cuda_memory.zig:97).
pub fn counts(
    unified: bool,
    card: MemInfo,
    text: Option<&str>,
) -> Result<MemInfo, MemoryError> {
    if unified {
        return meminfo(text.ok_or(MemoryError::HostMemoryUnavailable)?)
            .ok_or(MemoryError::HostMemoryUnavailable);
    }
    Ok(card)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `the memory plan: reserve, cap, and admission that
    /// fails closed` (cuda_memory.zig:102).
    #[test]
    fn the_memory_plan_reserve_cap_and_admission_that_fails_closed() {
        let g: u64 = 1 << 30;
        assert_eq!(Ok(4 * g), reserve_bytes(None, 24 * g, true)); // unified: at least 4 GiB
        assert_eq!(Ok(12 * g), reserve_bytes(Some(""), 120 * g, true)); // a tenth
        assert_eq!(Ok(3 * g), reserve_bytes(Some("3"), 120 * g, true));
        assert_eq!(Err(MemoryError::Invalid), reserve_bytes(Some("1"), 120 * g, true));
        assert_eq!(Err(MemoryError::Invalid), reserve_bytes(Some("200"), 120 * g, true));
        assert_eq!(Ok(g), reserve_bytes(None, 24 * g, false)); // a 24 GiB card: at least 1 GiB
        assert_eq!(Ok(3 * g), reserve_bytes(None, 96 * g, false)); // a 32nd
        assert_eq!(Ok(g / 2), reserve_bytes(Some("0.5"), 24 * g, false));
        assert_eq!(Err(MemoryError::Invalid), reserve_bytes(Some("0.25"), 24 * g, false));
        assert_eq!(Ok(None), limit_bytes(None));
        assert_eq!(Ok(Some(40 * g)), limit_bytes(Some("40")));
        assert_eq!(Err(MemoryError::Invalid), limit_bytes(Some("0")));
        let p = Pool {
            free: 70 * g,
            total: 96 * g,
            reserve: 10 * g,
            limit: None,
            unified: false,
        };
        assert_eq!(60 * g, p.room(20 * g));
        let capped = Pool {
            free: 70 * g,
            total: 96 * g,
            reserve: 10 * g,
            limit: Some(30 * g),
            unified: false,
        };
        assert_eq!(10 * g, capped.room(20 * g)); // the cap less what the engine holds
        assert_eq!(Ok(8), admit(60 * g, g, 0, 8, false));
        assert_eq!(Ok(5), admit(5 * g + 1, g, 0, 8, false)); // auto serves what fits
        assert_eq!(Err(MemoryError::TooMany), admit(5 * g, g, 0, 8, true)); // a fixed --parallel refuses
        assert_eq!(Err(MemoryError::NoStream), admit(g - 1, g, 0, 8, false));
        assert_eq!(Ok(2), admit(2 * g, g + g / 2, 1, 4, false)); // one stream on the engine's own buffers
        assert_eq!(Ok(1), admit(g - 1, g, 1, 8, false));
        assert_eq!(Err(MemoryError::TooMany), admit(2 * g, g, 1, 4, true));
    }

    /// Upstream's `the prompt cache: 16 GiB unless the flag sizes it,
    /// inside what the streams leave` (cuda_memory.zig:129).
    #[test]
    fn the_prompt_cache_16_gib_unless_the_flag_sizes_it_inside_what_the_streams_leave() {
        let g: u64 = 1 << 30;
        assert_eq!(16 * g, cache_bytes(None, 60 * g, 8 * g));
        assert_eq!(10 * g, cache_bytes(None, 18 * g, 8 * g)); // the streams' room comes first
        assert_eq!(4 * g, cache_bytes(Some(4.0), 60 * g, 8 * g));
        assert_eq!(52 * g, cache_bytes(Some(64.0), 60 * g, 8 * g));
        assert_eq!(0, cache_bytes(Some(0.0), 60 * g, 8 * g));
        assert_eq!(0, cache_bytes(None, 6 * g, 8 * g));
    }

    /// Upstream's `meminfo: total and available in bytes`
    /// (cuda_memory.zig:139).
    #[test]
    fn meminfo_total_and_available_in_bytes() {
        let m = meminfo("MemTotal:       131072 kB\nMemFree:          1024 kB\nMemAvailable:    65536 kB\n").unwrap();
        assert_eq!(131072 * 1024, m.total);
        assert_eq!(65536 * 1024, m.available);
        assert!(meminfo("MemTotal: 5 kB\n").is_none());
    }

    /// Upstream's `unified admission refuses missing or malformed host
    /// memory` (cuda_memory.zig:146).
    #[test]
    fn unified_admission_refuses_missing_or_malformed_host_memory() {
        let card = MemInfo {
            total: 128 << 30,
            available: 100 << 30,
        };
        for text in [
            None,
            Some(""),
            Some("MemTotal: 131072 kB\n"),
            Some("MemTotal: bad kB\nMemAvailable: 1024 kB\n"),
            Some("MemTotal: 131072 MB\nMemAvailable: 1024 MB\n"),
            Some("MemTotal: 0 kB\nMemAvailable: 0 kB\n"),
            Some("MemTotal: 1024 kB\nMemAvailable: 2048 kB\n"),
            Some("MemTotal: 18014398509481984 kB\nMemAvailable: 1024 kB\n"),
            Some("MemTotal: 131072 kB\nMemTotal: 131072 kB\nMemAvailable: 1024 kB\n"),
        ] {
            assert_eq!(
                Err(MemoryError::HostMemoryUnavailable),
                counts(true, card, text)
            );
        }
        assert_eq!(Ok(card), counts(false, card, None));
        assert_eq!(
            Ok(MemInfo {
                total: 131072 * 1024,
                available: 65536 * 1024
            }),
            counts(true, card, Some("MemTotal: 131072 kB\nMemAvailable: 65536 kB\n"))
        );
    }

    /// Upstream's `memory limit refuses an oversized finite value and
    /// the rounded u64 boundary` (cuda_memory.zig:165).
    #[test]
    fn memory_limit_refuses_an_oversized_finite_value_and_the_rounded_u64_boundary() {
        assert_eq!(Err(MemoryError::Invalid), limit_bytes(Some("1e300")));
        assert_eq!(Err(MemoryError::Invalid), limit_bytes(Some("17179869184")));
        assert_eq!(Err(MemoryError::Invalid), limit_bytes(Some("17179869183.9999995"))); // parses to 2^34 GiB: 2^64 bytes
        assert_eq!(
            Ok(Some(u64::MAX - 2047)),
            limit_bytes(Some("17179869183.9999980926513671875"))
        );
    }

    /// Upstream's live read (cuda_memory.zig:405) — Linux only, the
    /// host's own /proc/meminfo parses whole.
    #[cfg(target_os = "linux")]
    #[test]
    fn this_hosts_proc_meminfo_reads_whole_and_parses() {
        let text = match std::fs::read_to_string("/proc/meminfo") {
            Ok(t) => t,
            Err(_) => return, // no procfs: the read refuses upstream too
        };
        assert!(meminfo(&text).is_some());
    }
}
