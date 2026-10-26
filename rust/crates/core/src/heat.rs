//! Opt-in wait before a prompt chunk when the hottest thermal zone is
//! above a band.
//!
//! Port of `zig/src/core/heat.zig` (274 lines) — the bands, the wait
//! loop, the zone gathering over `thermal_zone*/temp`, and upstream's
//! four tests mirrored.

use std::ffi::c_char;

pub const DEFAULT_ROOT: &str = "/sys/class/thermal";

/// Upstream: `Bands` (heat.zig:6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    pub high_c: f64,
    pub low_c: f64,
}

/// Why the bands or the wait refused.
#[derive(Debug, PartialEq)]
pub enum HeatError {
    /// One band set, or low above high.
    HeatBands,
    /// No readable thermal zone.
    NoZone,
    /// A bad reading.
    Read,
    /// The wait ended by the stop.
    Cancelled,
}

impl std::fmt::Display for HeatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeatError::HeatBands => write!(
                f,
                "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high"
            ),
            HeatError::NoZone => write!(f, "no readable thermal zone"),
            HeatError::Read => write!(f, "the zone read failed"),
            HeatError::Cancelled => write!(f, "the wait was cancelled"),
        }
    }
}

impl std::error::Error for HeatError {}

/// Asked after each pause: true ends the wait with Cancelled (the
/// request was cancelled meanwhile). Upstream: `Stop` (heat.zig:9).
pub struct Stop<'a> {
    pub ptr: &'a mut dyn FnMut() -> bool,
}

impl Stop<'_> {
    fn check(&mut self) -> bool {
        (self.ptr)()
    }
}

/// Upstream: `Gate` (heat.zig:11).
pub struct Gate {
    pub bands: Option<Bands>,
    pub waited_s: f64,
    /// Set after the first unreadable zone, so later chunks skip the
    /// read and the log.
    pub no_zone: bool,
    pub root: String,
}

impl Default for Gate {
    fn default() -> Gate {
        Gate {
            bands: None,
            waited_s: 0.0,
            no_zone: false,
            root: DEFAULT_ROOT.to_string(),
        }
    }
}

impl Gate {
    /// TF_HEAT_HIGH and TF_HEAT_LOW in degrees, both or neither
    /// (neither: no read); TF_HEAT_ROOT moves the zones. Upstream:
    /// `fromEnv` (heat.zig:19).
    pub fn from_env() -> Result<Gate, HeatError> {
        let root = env_span("TF_HEAT_ROOT").unwrap_or(DEFAULT_ROOT.to_string());
        Ok(Gate {
            bands: bands_from(
                env_span("TF_HEAT_HIGH").as_deref(),
                env_span("TF_HEAT_LOW").as_deref(),
            )?,
            waited_s: 0.0,
            no_zone: false,
            root,
        })
    }

    /// `read` returns the gathered maximum in celsius. `sleep_fn`
    /// stands in for the 2 s pause. Upstream: `waitUntil` (heat.zig:25).
    pub fn wait_until(
        &mut self,
        read: &mut dyn FnMut() -> Result<f64, HeatError>,
        sleep_fn: &mut dyn FnMut(),
        stop: Option<&mut Stop<'_>>,
    ) -> Result<(), HeatError> {
        let bands = match self.bands {
            Some(b) => b,
            None => return Ok(()),
        };
        if self.no_zone {
            return Ok(());
        }
        let mut waiting = false;
        let mut stop = stop;
        loop {
            let max_c = match read() {
                Ok(t) => t,
                Err(HeatError::NoZone) => {
                    self.no_zone = true;
                    note_no_zone(&self.root);
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            if allow(bands, max_c, waiting) {
                return Ok(());
            }
            waiting = true;
            sleep_fn();
            self.waited_s += 2.0;
            if let Some(s) = stop.as_deref_mut() {
                if s.check() {
                    return Err(HeatError::Cancelled);
                }
            }
        }
    }

    /// The real pause, on this process's own hottest zone
    /// (`gathered_max` takes several readings), until `stop` says so.
    /// Upstream: `beforePromptChunk` (heat.zig:47).
    pub fn before_prompt_chunk(&mut self, stop: Option<&mut Stop<'_>>) -> Result<(), HeatError> {
        if self.bands.is_none() {
            return Ok(());
        }
        let root = self.root.clone();
        let mut read = || -> Result<f64, HeatError> {
            let dir = std::fs::read_dir(&root).map_err(|_| HeatError::NoZone)?;
            hottest(&root, dir)
        };
        let mut sleep = || {
            std::thread::sleep(std::time::Duration::from_millis(2000));
        };
        self.wait_until(&mut read, &mut sleep, stop)
    }
}

impl From<std::io::Error> for HeatError {
    fn from(_: std::io::Error) -> HeatError {
        HeatError::Read
    }
}

/// Env text is degrees. Both missing means unset. Upstream: `bandsFrom`
/// (heat.zig:69).
pub fn bands_from(high: Option<&str>, low: Option<&str>) -> Result<Option<Bands>, HeatError> {
    if high.is_none() && low.is_none() {
        return Ok(None);
    }
    let hi = high.ok_or(HeatError::HeatBands)?;
    let lo = low.ok_or(HeatError::HeatBands)?;
    let bands = Bands {
        high_c: degrees(hi)?,
        low_c: degrees(lo)?,
    };
    if bands.low_c > bands.high_c {
        return Err(HeatError::HeatBands);
    }
    Ok(Some(bands))
}

/// Upstream: `degrees` (heat.zig:78).
pub fn degrees(text: &str) -> Result<f64, HeatError> {
    text.trim()
        .parse::<f64>()
        .map_err(|_| HeatError::HeatBands)
}

/// A zone file is millidegrees. 94000 is 94.0 celsius. Upstream:
/// `zoneCelsius` (heat.zig:83).
pub fn zone_celsius(text: &str) -> Result<f64, HeatError> {
    let milli: i64 = text.trim().parse().map_err(|_| HeatError::Read)?;
    Ok(milli as f64 / 1000.0)
}

/// Run when idle and at or under high, or when already waiting and at
/// or under low. Upstream: `allow` (heat.zig:89).
pub fn allow(bands: Bands, max_c: f64, waiting: bool) -> bool {
    if waiting {
        return max_c <= bands.low_c;
    }
    max_c <= bands.high_c
}

/// Upstream: `gatheredMax` (heat.zig:94).
pub fn gathered_max(xs: &[f64]) -> f64 {
    let mut m = xs[0];
    for x in &xs[1..] {
        m = m.max(*x);
    }
    m
}

/// Hottest `thermal_zone*/temp` under `dir`. Other names are ignored.
/// No readable zone is an error. Upstream: `hottest` (heat.zig:101).
pub fn hottest(_root: &str, dir: std::fs::ReadDir) -> Result<f64, HeatError> {
    let mut max: Option<f64> = None;
    for e in dir {
        let e = e.map_err(|_| HeatError::Read)?;
        let name = e.file_name();
        let name = name.to_string_lossy().into_owned();
        if !name.starts_with("thermal_zone") {
            continue;
        }
        let text = std::fs::read_to_string(e.path().join("temp"));
        let c = match text {
            Ok(t) => zone_celsius(&t),
            Err(_) => continue,
        };
        let c = match c {
            Ok(c) => c,
            Err(_) => continue,
        };
        max = Some(match max {
            Some(m) => m.max(c),
            None => c,
        });
    }
    max.ok_or(HeatError::NoZone)
}

/// Upstream: `note` (heat.zig:117).
pub fn note(seconds: f64) {
    if cfg!(test) {
        return;
    }
    eprintln!("heat_wait_s {seconds:.1}");
}

/// Test builds count the warning. A real process logs it.
static NO_ZONE_NOTES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn note_no_zone(root: &str) {
    NO_ZONE_NOTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if cfg!(test) {
        return;
    }
    eprintln!(
        "no readable thermal zone under {root}; running without a heat wait"
    );
}

fn env_span(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

#[allow(unused)]
unsafe extern "C" {
    fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    fn unsetenv(name: *const c_char) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `bands parse as degrees and a lone value is refused`
    /// (heat.zig:138).
    #[test]
    fn bands_parse_as_degrees_and_a_lone_value_is_refused() {
        assert!((bands_from(None, None).unwrap()).is_none());
        assert_eq!(Some(HeatError::HeatBands), bands_from(Some("92"), None).err());
        assert_eq!(Some(HeatError::HeatBands), bands_from(None, Some("88")).err());
        assert_eq!(Some(HeatError::HeatBands), bands_from(Some("88"), Some("92")).err());
        let bands = bands_from(Some("92"), Some(" 88\n")).unwrap().unwrap();
        assert_eq!(92.0, bands.high_c);
        assert_eq!(88.0, bands.low_c);
        assert_eq!(94.0, zone_celsius("94000\n").unwrap());
    }

    /// Upstream's `the chunk runs under the high band, and a hot
    /// reading waits until the low band` (heat.zig:149) — the real
    /// wait_until driven by scripted reads and sleeps (the script
    /// shared behind a RefCell, as Zig's shared mutable context).
    #[test]
    fn the_chunk_runs_under_the_high_band_and_a_hot_reading_waits_until_the_low_band() {
        use std::cell::RefCell;
        let bands = Bands {
            high_c: 92.0,
            low_c: 88.0,
        };
        assert!(allow(bands, 92.0, false));
        assert!(!allow(bands, 92.5, false));
        assert!(!allow(bands, 90.0, true));
        assert!(allow(bands, 88.0, true));
        assert_eq!(94.0, gathered_max(&[80.0, 94.0, 91.0]));

        struct Script {
            temps: Vec<f64>,
            i: usize,
            sleeps: u32,
        }
        let hot = RefCell::new(Script {
            temps: vec![95.0, 93.0, 88.0],
            i: 0,
            sleeps: 0,
        });
        let mut gate = Gate {
            bands: Some(bands),
            ..Gate::default()
        };
        let mut read = || -> Result<f64, HeatError> {
            let mut s = hot.borrow_mut();
            let t = s.temps[s.i];
            if s.i + 1 < s.temps.len() {
                s.i += 1;
            }
            Ok(t)
        };
        let mut sleep_fn = || {
            hot.borrow_mut().sleeps += 1;
        };
        gate.wait_until(&mut read, &mut sleep_fn, None).unwrap();
        assert_eq!(2, hot.borrow().sleeps);
        assert_eq!(4.0, gate.waited_s);

        let cool = RefCell::new(Script {
            temps: vec![90.0],
            i: 0,
            sleeps: 0,
        });
        let mut idle = Gate {
            bands: Some(bands),
            ..Gate::default()
        };
        let mut read = || -> Result<f64, HeatError> {
            let s = cool.borrow();
            Ok(s.temps[s.i])
        };
        let mut sleep_fn = || {
            cool.borrow_mut().sleeps += 1;
        };
        idle.wait_until(&mut read, &mut sleep_fn, None).unwrap();
        assert_eq!(0, cool.borrow().sleeps);

        // bands unset: a failing read is never asked for
        let mut off = Gate::default();
        let mut read = || -> Result<f64, HeatError> { Err(HeatError::Read) };
        let mut sleep_fn = || {};
        off.wait_until(&mut read, &mut sleep_fn, None).unwrap();
        assert_eq!(0.0, off.waited_s);

        // a cancel ends the wait after the pause it came in
        let stuck = RefCell::new(Script {
            temps: vec![95.0, 95.0, 95.0],
            i: 0,
            sleeps: 0,
        });
        let mut hot_gate = Gate {
            bands: Some(bands),
            ..Gate::default()
        };
        let mut read = || -> Result<f64, HeatError> {
            let mut s = stuck.borrow_mut();
            let t = s.temps[s.i];
            if s.i + 1 < s.temps.len() {
                s.i += 1;
            }
            Ok(t)
        };
        let mut sleep_fn = || {
            stuck.borrow_mut().sleeps += 1;
        };
        let mut stop = Stop {
            ptr: &mut || true,
        };
        assert_eq!(
            Some(HeatError::Cancelled),
            hot_gate
                .wait_until(&mut read, &mut sleep_fn, Some(&mut stop))
                .err()
        );
        assert_eq!(1, stuck.borrow().sleeps);
    }

    /// Upstream's `TF_HEAT_HIGH and TF_HEAT_LOW set the bands
    /// together, and TF_HEAT_ROOT the zones' folder` (heat.zig:206).
    #[test]
    fn tf_heat_high_and_tf_heat_low_set_the_bands_together_and_tf_heat_root_the_zones_folder() {
        let keys = ["TF_HEAT_HIGH", "TF_HEAT_LOW", "TF_HEAT_ROOT"];
        for k in keys {
            // SAFETY: the test's own environment, single-threaded.
            unsafe { std::env::remove_var(k) };
        }
        assert!((Gate::from_env().unwrap()).bands.is_none());
        // SAFETY: the test's own environment, single-threaded.
        unsafe { std::env::set_var("TF_HEAT_HIGH", "70") };
        assert_eq!(Some(HeatError::HeatBands), Gate::from_env().err());
        // SAFETY: the test's own environment, single-threaded.
        unsafe { std::env::set_var("TF_HEAT_LOW", "60") };
        // SAFETY: the test's own environment, single-threaded.
        unsafe { std::env::set_var("TF_HEAT_ROOT", "core-root") };
        let both = Gate::from_env().unwrap();
        assert_eq!(70.0, both.bands.unwrap().high_c);
        assert_eq!(60.0, both.bands.unwrap().low_c);
        assert_eq!("core-root", both.root);
        // SAFETY: the test's own environment, single-threaded.
        unsafe { std::env::remove_var("TF_HEAT_ROOT") };
        assert_eq!(DEFAULT_ROOT, Gate::from_env().unwrap().root);
        for k in keys {
            // SAFETY: the test's own environment, single-threaded.
            unsafe { std::env::remove_var(k) };
        }
    }

    /// Upstream's `no readable zone logs once and the chunk runs`
    /// (heat.zig:223). The no-zone note counts live in the port as the
    /// counter itself (upstream reads its private static back).
    #[test]
    fn no_readable_zone_logs_once_and_the_chunk_runs() {
        let mut probe_reads = 0u32;
        let mut probe_sleeps = 0u32;
        let mut gate = Gate {
            bands: Some(Bands {
                high_c: 92.0,
                low_c: 88.0,
            }),
            root: "missing-thermal".to_string(),
            ..Gate::default()
        };
        let notes = NO_ZONE_NOTES.load(std::sync::atomic::Ordering::Relaxed);
        {
            let mut read = || -> Result<f64, HeatError> {
                probe_reads += 1;
                Err(HeatError::NoZone)
            };
            let mut sleep = || {
                probe_sleeps += 1;
            };
            gate.wait_until(&mut read, &mut sleep, None).unwrap();
            gate.wait_until(&mut read, &mut sleep, None).unwrap();
        }
        assert!(gate.no_zone);
        assert_eq!(1, probe_reads);
        assert_eq!(0, probe_sleeps);
        assert_eq!(0.0, gate.waited_s);
        assert_eq!(
            notes + 1,
            NO_ZONE_NOTES.load(std::sync::atomic::Ordering::Relaxed)
        );

        let notes = NO_ZONE_NOTES.load(std::sync::atomic::Ordering::Relaxed);
        let mut missing = Gate {
            bands: Some(Bands {
                high_c: 92.0,
                low_c: 88.0,
            }),
            root: "no/such/thermal/root".to_string(),
            ..Gate::default()
        };
        missing.before_prompt_chunk(None).unwrap();
        assert_eq!(
            notes + 1,
            NO_ZONE_NOTES.load(std::sync::atomic::Ordering::Relaxed)
        );
        missing.before_prompt_chunk(None).unwrap();
        assert_eq!(
            notes + 1,
            NO_ZONE_NOTES.load(std::sync::atomic::Ordering::Relaxed)
        );
        assert!(missing.no_zone);
        assert_eq!(0.0, missing.waited_s);

        // no zones at all is the NoZone error
        let dir = std::env::temp_dir().join("tf-heat-empty-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            Some(HeatError::NoZone),
            hottest(dir.to_str().unwrap(), std::fs::read_dir(&dir).unwrap()).err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Upstream's `hottest zone is the maximum millidegree reading`
    /// (heat.zig:262).
    #[test]
    fn hottest_zone_is_the_maximum_millidegree_reading() {
        let dir = std::env::temp_dir().join("tf-heat-zones-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, data) in [
            ("thermal_zone0", "80000\n"),
            ("thermal_zone1", "94000\n"),
            ("other", "99000\n"),
        ] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
            std::fs::write(dir.join(name).join("temp"), data).unwrap();
        }
        std::fs::write(dir.join("thermal_zone2"), "not-a-zone").unwrap();
        assert_eq!(
            94.0,
            hottest(dir.to_str().unwrap(), std::fs::read_dir(&dir).unwrap()).unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
