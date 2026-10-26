//! NCCL opened at run time for tensor-parallel ranks: communicators
//! and the collectives the lane engines use.
//!
//! Port of `zig/src/cuda/nccl.zig` (71 lines) — its own error set and
//! its own dlopen'd API table, the fields named by their exact symbols
//! the way the driver's are.

#![allow(non_upper_case_globals)] // the attribute names are upstream's own

use std::ffi::{c_char, c_int, c_void, CStr, CString};

pub type Result_ = c_int;
pub type Comm = *mut c_void;

/// Upstream: `UniqueId` (nccl.zig:8).
#[repr(C)]
pub struct UniqueId {
    pub internal: [u8; 128],
}

/// Upstream: `DataType` (nccl.zig:10) — the C enum values.
#[derive(Clone, Copy)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum DataType {
    i8 = 0,
    u8 = 1,
    i32 = 2,
    u32 = 3,
    i64 = 4,
    u64 = 5,
    f16 = 6,
    f32 = 7,
    f64 = 8,
    bf16 = 9,
}

/// Upstream: `RedOp` (nccl.zig:11).
#[derive(Clone, Copy)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum RedOp {
    sum = 0,
    prod = 1,
    max = 2,
    min = 3,
    avg = 4,
}

/// Upstream: `Error` (nccl.zig:13).
#[derive(Debug)]
pub enum Error {
    LibraryUnavailable,
    MissingSymbol,
    NcclFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::LibraryUnavailable => write!(f, "no NCCL library"),
            Error::MissingSymbol => write!(f, "the library is missing a symbol"),
            Error::NcclFailed => write!(f, "NCCL failed"),
        }
    }
}

impl std::error::Error for Error {}

/// Device pointers are declared as u64: the same register class as
/// `void*` on every LP64 target.
pub type DevicePtr = u64;
pub type Stream = *mut c_void;

/// Upstream: `Api` (nccl.zig:18).
#[allow(non_snake_case, clippy::type_complexity)]
pub struct Api {
    pub ncclGetVersion: Option<unsafe extern "C" fn(*mut c_int) -> Result_>,
    pub ncclGetUniqueId: Option<unsafe extern "C" fn(*mut UniqueId) -> Result_>,
    pub ncclCommInitRank: Option<unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> Result_>,
    pub ncclCommInitAll: Option<unsafe extern "C" fn(*mut Comm, c_int, *const c_int) -> Result_>,
    pub ncclCommDestroy: Option<unsafe extern "C" fn(Comm) -> Result_>,
    pub ncclGetErrorString: Option<unsafe extern "C" fn(Result_) -> *const c_char>,
    pub ncclAllReduce: Option<
        unsafe extern "C" fn(DevicePtr, DevicePtr, usize, DataType, RedOp, Comm, Stream) -> Result_,
    >,
    pub ncclAllGather:
        Option<unsafe extern "C" fn(DevicePtr, DevicePtr, usize, DataType, Comm, Stream) -> Result_>,
    pub ncclBroadcast: Option<
        unsafe extern "C" fn(DevicePtr, DevicePtr, usize, DataType, c_int, Comm, Stream) -> Result_,
    >,
    pub ncclGroupStart: Option<unsafe extern "C" fn() -> Result_>,
    pub ncclGroupEnd: Option<unsafe extern "C" fn() -> Result_>,
}

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

/// The opened library: the handle and its resolved API table. Upstream:
/// `Library` (nccl.zig:32).
pub struct Library {
    lib: *mut c_void,
    pub api: Api,
}

impl Library {
    /// Upstream: `open` (nccl.zig:36).
    pub fn open() -> Result<Library, Error> {
        Library::open_path("libnccl.so.2")
    }

    /// Upstream: `openPath` (nccl.zig:40) — the load, then every field
    /// by its exact name; a missing symbol refuses the whole library.
    #[allow(clippy::missing_transmute_annotations)] // the dlsym pointer becomes the field's own Option<fn>, the same one-step table build abi.rs performs
    pub fn open_path(path: &str) -> Result<Library, Error> {
        let name = CString::new(path).map_err(|_| Error::LibraryUnavailable)?;
        // SAFETY: the platform's dlopen over the NUL-terminated path.
        let lib = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return Err(Error::LibraryUnavailable);
        }
        let lookup = |symbol: &str| -> Option<*mut c_void> {
            let name = CString::new(symbol).ok()?;
            // SAFETY: the handle stands open through the resolution.
            let raw = unsafe { dlsym(lib, name.as_ptr()) };
            (!raw.is_null()).then_some(raw)
        };
        macro_rules! sym {
            ($name:ident) => {{
                let raw = lookup(stringify!($name)).ok_or_else(|| {
                    eprintln!("{path} has no {}", stringify!($name));
                    Error::MissingSymbol
                })?;
                // SAFETY: the symbol resolved to the library's own
                // exported entry point.
                Some(unsafe { std::mem::transmute::<*mut c_void, _>(raw) })
            }};
        }
        let api = Api {
            ncclGetVersion: sym!(ncclGetVersion),
            ncclGetUniqueId: sym!(ncclGetUniqueId),
            ncclCommInitRank: sym!(ncclCommInitRank),
            ncclCommInitAll: sym!(ncclCommInitAll),
            ncclCommDestroy: sym!(ncclCommDestroy),
            ncclGetErrorString: sym!(ncclGetErrorString),
            ncclAllReduce: sym!(ncclAllReduce),
            ncclAllGather: sym!(ncclAllGather),
            ncclBroadcast: sym!(ncclBroadcast),
            ncclGroupStart: sym!(ncclGroupStart),
            ncclGroupEnd: sym!(ncclGroupEnd),
        };
        Ok(Library { lib, api })
    }

    /// Upstream: `close` (nccl.zig:54).
    pub fn close(&mut self) {
        // SAFETY: the handle opened at construction, closed once.
        unsafe { dlclose(self.lib) };
        self.lib = std::ptr::null_mut();
    }

    /// Upstream: `check` (nccl.zig:58) — the library's own error string
    /// logged for the code.
    pub fn check(&self, r: Result_, what: &str) -> Result<(), Error> {
        if r == 0 {
            return Ok(());
        }
        // SAFETY: the library's own error-string lookup.
        let s = unsafe { (self.api.ncclGetErrorString.unwrap_unchecked())(r) };
        let text = if s.is_null() {
            "?".to_string()
        } else {
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        };
        eprintln!("{what}: {text} ({r})");
        Err(Error::NcclFailed)
    }

    /// NCCL's version as 10000 * major + 100 * minor + patch. Upstream:
    /// `version` (nccl.zig:65).
    pub fn version(&self) -> Result<c_int, Error> {
        let mut v: c_int = 0;
        // SAFETY: the version written by the library.
        let res = unsafe { (self.api.ncclGetVersion.unwrap_unchecked())(&mut v) };
        self.check(res, "ncclGetVersion")?;
        Ok(v)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.lib.is_null() {
            self.close();
        }
    }
}

// Upstream: ncclCommInitAll's third parameter is ?[*]const c_int — a
// nullable device list; the port asserts the UniqueId layout the
// rank call passes by value.
const _: () = assert!(std::mem::size_of::<UniqueId>() == 128);
