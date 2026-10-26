//! The CUDA driver opened at run time: libcuda.so.1's versioned entry
//! points in one table, and checked calls.
//!
//! Port of `zig/src/cuda/driver.zig` (84 lines): the platform's own
//! dlopen/dlsym (upstream's std.DynLib), every entry point resolved by
//! its exact symbol — a miss refuses the whole driver — and every call
//! checked with the driver's own error name and text. NO_READY is a
//! state, not a failure; code 2 is the out-of-memory; the rest the
//! generic CUDA failure.

use crate::abi::{self, Api};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

/// Why a driver call failed. Upstream: driver.zig's ``Error``.
#[derive(Debug)]
pub enum Error {
    DriverUnavailable,
    MissingSymbol(String),
    CudaFailed { what: String, code: i32 },
    OutOfDeviceMemory,
    NotReady,
    NotFound,
    Invalid,
    Segments,
    BuiltWithoutKernels,
    FormatNotLoaded,
    RowStride,
    WindowTooWide,
    SplitTooWide,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DriverUnavailable => write!(f, "no CUDA driver library"),
            Error::MissingSymbol(symbol) => write!(f, "the driver has no {symbol}"),
            Error::CudaFailed { what, code } => write!(f, "{what} failed ({code})"),
            Error::OutOfDeviceMemory => write!(f, "the device is out of memory"),
            Error::NotReady => write!(f, "the driver is not ready"),
            Error::NotFound => write!(f, "not found"),
            Error::Invalid => write!(f, "invalid"),
            Error::Segments => write!(f, "the segment count is out of range"),
            Error::BuiltWithoutKernels => write!(f, "built without kernels"),
            Error::FormatNotLoaded => write!(f, "the format is not loaded"),
            Error::RowStride => write!(f, "the row stride does not fit the weight"),
            Error::WindowTooWide => write!(f, "the window is too wide"),
            Error::SplitTooWide => write!(f, "the split is too wide"),
        }
    }
}

impl std::error::Error for Error {}

/// RTLD_NOW: resolve every symbol at load, lazily nothing.
const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

/// The opened driver: the library and its resolved API table. Upstream:
/// ``Driver`` (driver.zig:8).
pub struct Driver {
    lib: *mut c_void,
    pub api: Api,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Driver {
    /// libcuda.so.1 by its platform name. Upstream: `open` (driver.zig:13).
    pub fn open() -> Result<Driver, Error> {
        Driver::open_path("libcuda.so.1")
    }

    /// Resolves every field of the API by its exact name; a missing
    /// symbol refuses the whole driver. Upstream: `openPath`
    /// (driver.zig:17) — the load, the resolution, and cuInit.
    pub fn open_path(path: &str) -> Result<Driver, Error> {
        let name = CString::new(path).map_err(|_| Error::DriverUnavailable)?;
        // SAFETY: the platform's dlopen over the NUL-terminated path.
        let lib = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return Err(Error::DriverUnavailable);
        }
        let lookup = |symbol: &str| -> Option<*mut c_void> {
            let name = CString::new(symbol).ok()?;
            // SAFETY: the handle stands open through the resolution.
            let raw = unsafe { dlsym(lib, name.as_ptr()) };
            (!raw.is_null()).then_some(raw)
        };
        // SAFETY: each symbol resolved to the exact exported entry point.
        let api = match unsafe { Api::resolve(&lookup) } {
            std::result::Result::Ok(api) => api,
            std::result::Result::Err(problem) => {
                // SAFETY: the handle opened above.
                unsafe { dlclose(lib) };
                return Err(problem);
            }
        };
        let d = Driver { lib, api };
        // SAFETY: the driver initialized with no flags.
        let res = unsafe { (d.api.cuInit.unwrap_unchecked())(0) };
        match res {
            abi::SUCCESS => Ok(d),
            code => {
                let what = "cuInit";
                eprintln!("{what}: {} ({code})", d.error_name(code));
                Err(d.map_result(code, what))
            }
        }
    }

    /// Logs a failed call with the driver's own name and text for the
    /// code; NOT_READY is a state, not a failure. Upstream: `check`
    /// (driver.zig:35).
    pub fn check(&self, res: abi::Result, what: &str) -> Result<(), Error> {
        if res == abi::SUCCESS {
            return Ok(());
        }
        if res == abi::ERROR_NOT_READY {
            return Err(Error::NotReady);
        }
        eprintln!("{}: {} ({res}) {}", what, self.error_name(res), self.error_text(res));
        Err(self.map_result(res, what))
    }

    fn map_result(&self, res: abi::Result, what: &str) -> Error {
        match res {
            2 => Error::OutOfDeviceMemory,
            abi::ERROR_NOT_FOUND => Error::NotFound,
            code => Error::CudaFailed {
                what: what.into(),
                code,
            },
        }
    }

    pub fn error_name(&self, res: abi::Result) -> String {
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: the driver's own error-name lookup.
        let ok = unsafe { (self.api.cuGetErrorName.unwrap_unchecked())(res, &mut s) };
        if ok != abi::SUCCESS || s.is_null() {
            return "CUDA_ERROR_UNKNOWN_CODE".into();
        }
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }

    pub fn error_text(&self, res: abi::Result) -> String {
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: the driver's own error-text lookup.
        let ok = unsafe { (self.api.cuGetErrorString.unwrap_unchecked())(res, &mut s) };
        if ok != abi::SUCCESS || s.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }

    /// The driver's CUDA version as 1000 * major + 10 * minor. Upstream:
    /// `version` (driver.zig:51).
    pub fn version(&self) -> Result<c_int, Error> {
        let mut v: c_int = 0;
        // SAFETY: the version read into a c_int.
        let res = unsafe { (self.api.cuDriverGetVersion.unwrap_unchecked())(&mut v) };
        self.check(res, "cuDriverGetVersion")?;
        Ok(v)
    }

    pub fn device_count(&self) -> Result<c_int, Error> {
        let mut n: c_int = 0;
        // SAFETY: the count read into a c_int.
        let res = unsafe { (self.api.cuDeviceGetCount.unwrap_unchecked())(&mut n) };
        self.check(res, "cuDeviceGetCount")?;
        Ok(n)
    }

    /// Device `ordinal`'s compute capability as 10 * major + minor, read
    /// without making a context. Upstream: `capability` (driver.zig:60).
    pub fn capability(&self, ordinal: c_int) -> Result<u32, Error> {
        let mut dev: abi::Device = 0;
        // SAFETY: the ordinal resolved into a device handle.
        let res = unsafe { (self.api.cuDeviceGet.unwrap_unchecked())(&mut dev, ordinal) };
        self.check(res, "cuDeviceGet")?;
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        // SAFETY: the attribute reads.
        let res = unsafe {
            (self.api.cuDeviceGetAttribute.unwrap_unchecked())(
                &mut major,
                abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        };
        self.check(res, "cuDeviceGetAttribute")?;
        // SAFETY: the attribute reads.
        let res = unsafe {
            (self.api.cuDeviceGetAttribute.unwrap_unchecked())(
                &mut minor,
                abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        };
        self.check(res, "cuDeviceGetAttribute")?;
        Ok((10 * major + minor) as u32)
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        if self.lib.is_null() {
            return; // the test stub: no handle at all
        }
        // SAFETY: the handle opened at construction, closed once.
        unsafe { dlclose(self.lib) };
    }
}

#[cfg(test)]
pub(crate) mod stub {
    use super::*;
    use std::sync::OnceLock;

    /// A driver with no library and no entry points: tests whose spans
    /// never call the driver construct against it (upstream's
    /// `.{ .d = undefined, ... }` shape, made safe).
    pub fn stub() -> &'static Driver {
        static STUB: OnceLock<Driver> = OnceLock::new();
        STUB.get_or_init(|| Driver {
            lib: std::ptr::null_mut(),
            api: Api::none(),
        })
    }

    /// A module over the stub: never loaded, its drop safe.
    pub fn stub_module() -> crate::module::Module<'static> {
        crate::module::Module {
            d: stub(),
            handle: std::ptr::null_mut(),
        }
    }

    /// A function over the stub: never resolved.
    pub fn stub_function() -> crate::module::Function<'static> {
        crate::module::Function {
            d: stub(),
            handle: std::ptr::null_mut(),
        }
    }

    /// A stream over the stub: never touched, its drop safe (the null
    /// handle stands for Zig's `undefined` in tests whose calls never
    /// reach the driver — qlinear.zig:248's refusal test).
    pub fn stub_stream() -> crate::stream::Stream<'static> {
        crate::stream::Stream {
            d: stub(),
            handle: std::ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loader on this platform: macOS has no libcuda (the
    /// DriverUnavailable path); a Linux box with the toolkit resolves
    /// every symbol against the stub, and the stub's own cuInit refusal
    /// exercises the error mapping — every outcome short of
    /// MissingSymbol is the contract holding.
    #[test]
    fn the_opens() {
        match Driver::open_path("libcuda.so.1") {
            Ok(_) => {
                // A real GPU: the init succeeded
            }
            Err(Error::DriverUnavailable) => {
                // macOS, or a Linux box without the toolkit: the library
                // itself absent
            }
            Err(Error::CudaFailed { code, .. }) => {
                // The stub library: every symbol resolved, cuInit
                // refused with the stub's own code (999 on CUDA 12+)
                eprintln!("the stub refused cuInit with {code}");
                assert_ne!(code, 500, "not a NOT_FOUND masquerading");
            }
            Err(other) => panic!("an unexpected outcome: {other:?}"),
        }
        // A path that cannot exist: the unavailable path
        assert!(matches!(
            Driver::open_path("/no/such/libcuda.so"),
            Err(Error::DriverUnavailable)
        ));
    }
}
