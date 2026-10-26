//! cuBLASLt opened at run time: descriptors, layouts, heuristics and
//! matmul, with fp32 compute only (no TF32 or fast modes).
//!
//! Port of `zig/src/cuda/cublaslt.zig` (163 lines) — its own error set
//! and its own dlopen'd API table, the fields named by their exact
//! symbols the way the driver's are.

#![allow(non_upper_case_globals)] // the attribute names are upstream's own

use std::ffi::{c_char, c_int, c_void, CStr, CString};

pub type Status = c_int;
pub type Handle = *mut c_void;
pub type MatmulDesc = *mut c_void;
pub type Layout = *mut c_void;
pub type Preference = *mut c_void;

/// Upstream: `Algo` (cublaslt.zig:12).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Algo {
    pub data: [u64; 8],
}

/// Upstream: `Heuristic` (cublaslt.zig:14).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Heuristic {
    pub algo: Algo,
    pub workspace_size: usize,
    pub state: Status,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

// Upstream: the comptime size assertions (cublaslt.zig:160).
const _: () = assert!(std::mem::size_of::<Algo>() == 64);
const _: () = assert!(std::mem::size_of::<Heuristic>() == 96);

/// Upstream: `DataType` (cublaslt.zig:22) — the C enum values.
#[derive(Clone, Copy)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum DataType {
    f32 = 0,
    f16 = 2,
    bf16 = 14,
}

pub const compute_32f: c_int = 68;

/// Upstream: `Op` (cublaslt.zig:24).
#[derive(Clone, Copy)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum Op {
    n = 0,
    t = 1,
}

/// Upstream: `Order` (cublaslt.zig:25).
#[derive(Clone, Copy)]
#[repr(i32)]
pub enum Order {
    Col = 0,
    Row = 1,
}

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
pub const layout_order: c_int = 1;
pub const pref_max_workspace_bytes: c_int = 1;

/// Upstream: `Error` (cublaslt.zig:32).
#[derive(Debug)]
pub enum Error {
    LibraryUnavailable,
    MissingSymbol,
    CublasFailed,
    NoAlgorithm,
    Invalid,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::LibraryUnavailable => write!(f, "no cuBLASLt library"),
            Error::MissingSymbol => write!(f, "the library is missing a symbol"),
            Error::CublasFailed => write!(f, "cuBLASLt failed"),
            Error::NoAlgorithm => write!(f, "no algorithm for the shape"),
            Error::Invalid => write!(f, "invalid"),
        }
    }
}

impl std::error::Error for Error {}

type CPtr = *const c_void;
/// Device pointers are declared as u64: the same register class as
/// `void*` on every LP64 target (upstream: cublaslt.zig:34).
pub type DevicePtr = u64;
pub type Stream = *mut c_void;

/// Upstream: `Api` (cublaslt.zig:38).
#[allow(non_snake_case, clippy::type_complexity)]
pub struct Api {
    pub cublasLtCreate: Option<unsafe extern "C" fn(*mut Handle) -> Status>,
    pub cublasLtDestroy: Option<unsafe extern "C" fn(Handle) -> Status>,
    pub cublasLtGetVersion: Option<unsafe extern "C" fn() -> usize>,
    pub cublasLtGetStatusName: Option<unsafe extern "C" fn(Status) -> *const c_char>,
    pub cublasLtMatmulDescCreate:
        Option<unsafe extern "C" fn(*mut MatmulDesc, c_int, DataType) -> Status>,
    pub cublasLtMatmulDescDestroy: Option<unsafe extern "C" fn(MatmulDesc) -> Status>,
    pub cublasLtMatmulDescSetAttribute:
        Option<unsafe extern "C" fn(MatmulDesc, c_int, CPtr, usize) -> Status>,
    pub cublasLtMatrixLayoutCreate:
        Option<unsafe extern "C" fn(*mut Layout, DataType, u64, u64, i64) -> Status>,
    pub cublasLtMatrixLayoutDestroy: Option<unsafe extern "C" fn(Layout) -> Status>,
    pub cublasLtMatrixLayoutSetAttribute:
        Option<unsafe extern "C" fn(Layout, c_int, CPtr, usize) -> Status>,
    pub cublasLtMatmulPreferenceCreate: Option<unsafe extern "C" fn(*mut Preference) -> Status>,
    pub cublasLtMatmulPreferenceDestroy: Option<unsafe extern "C" fn(Preference) -> Status>,
    pub cublasLtMatmulPreferenceSetAttribute:
        Option<unsafe extern "C" fn(Preference, c_int, CPtr, usize) -> Status>,
    pub cublasLtMatmulAlgoGetHeuristic: Option<
        unsafe extern "C" fn(
            Handle,
            MatmulDesc,
            Layout,
            Layout,
            Layout,
            Layout,
            Preference,
            c_int,
            *mut Heuristic,
            *mut c_int,
        ) -> Status,
    >,
    pub cublasLtMatmul: Option<
        unsafe extern "C" fn(
            Handle,
            MatmulDesc,
            CPtr,
            DevicePtr,
            Layout,
            DevicePtr,
            Layout,
            CPtr,
            DevicePtr,
            Layout,
            DevicePtr,
            Layout,
            *const Algo,
            DevicePtr,
            usize,
            Stream,
        ) -> Status,
    >,
}

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

/// The opened library: the handle and its resolved API table. Upstream:
/// `Library` (cublaslt.zig:56).
pub struct Library {
    lib: *mut c_void,
    pub api: Api,
}

impl Library {
    /// The CUDA 13 soname first, then the unversioned link name.
    /// Upstream: `open` (cublaslt.zig:61).
    pub fn open() -> Result<Library, Error> {
        for path in ["libcublasLt.so.13", "libcublasLt.so"] {
            return match Library::open_path(path) {
                Ok(l) => Ok(l),
                Err(Error::LibraryUnavailable) => continue,
                Err(e) => return Err(e),
            };
        }
        Err(Error::LibraryUnavailable)
    }

    /// Upstream: `openPath` (cublaslt.zig:71) — the load, then every
    /// field by its exact name; a missing symbol refuses the whole
    /// library.
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
            cublasLtCreate: sym!(cublasLtCreate),
            cublasLtDestroy: sym!(cublasLtDestroy),
            cublasLtGetVersion: sym!(cublasLtGetVersion),
            cublasLtGetStatusName: sym!(cublasLtGetStatusName),
            cublasLtMatmulDescCreate: sym!(cublasLtMatmulDescCreate),
            cublasLtMatmulDescDestroy: sym!(cublasLtMatmulDescDestroy),
            cublasLtMatmulDescSetAttribute: sym!(cublasLtMatmulDescSetAttribute),
            cublasLtMatrixLayoutCreate: sym!(cublasLtMatrixLayoutCreate),
            cublasLtMatrixLayoutDestroy: sym!(cublasLtMatrixLayoutDestroy),
            cublasLtMatrixLayoutSetAttribute: sym!(cublasLtMatrixLayoutSetAttribute),
            cublasLtMatmulPreferenceCreate: sym!(cublasLtMatmulPreferenceCreate),
            cublasLtMatmulPreferenceDestroy: sym!(cublasLtMatmulPreferenceDestroy),
            cublasLtMatmulPreferenceSetAttribute: sym!(cublasLtMatmulPreferenceSetAttribute),
            cublasLtMatmulAlgoGetHeuristic: sym!(cublasLtMatmulAlgoGetHeuristic),
            cublasLtMatmul: sym!(cublasLtMatmul),
        };
        Ok(Library { lib, api })
    }

    /// Upstream: `close` (cublaslt.zig:85).
    pub fn close(&mut self) {
        // SAFETY: the handle opened at construction, closed once.
        unsafe { dlclose(self.lib) };
        self.lib = std::ptr::null_mut();
    }

    /// Upstream: `check` (cublaslt.zig:89) — the library's own status
    /// name logged for the code.
    pub fn check(&self, s: Status, what: &str) -> Result<(), Error> {
        if s == 0 {
            return Ok(());
        }
        // SAFETY: the library's own status-name lookup.
        let n = unsafe { (self.api.cublasLtGetStatusName.unwrap_unchecked())(s) };
        let name = if n.is_null() {
            "?".to_string()
        } else {
            unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned()
        };
        eprintln!("{what}: {name} ({s})");
        Err(Error::CublasFailed)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.lib.is_null() {
            self.close();
        }
    }
}

/// One row-major bf16 product D[m,n] = X[m,k] . W[n,k]^T (a torch
/// Linear), fp32 accumulate, bf16 or fp32 out. Upstream: `Linear`
/// (cublaslt.zig:98).
pub struct Linear<'a> {
    lt: &'a Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,
}

impl<'a> Linear<'a> {
    /// Picks the heuristic's first algorithm for this shape once;
    /// `workspace_limit` bounds what it may ask for. Upstream: `init`
    /// (cublaslt.zig:110) — the errdefer chain unwinds exactly what
    /// stood (the port's `unwind` runs it on the failure path, the
    /// Drop on scope exit).
    pub fn init(
        lt: &'a Library,
        m: u64,
        n: u64,
        k: u64,
        out: DataType,
        workspace_limit: usize,
    ) -> Result<Linear<'a>, Error> {
        let mut handle: Handle = std::ptr::null_mut();
        let mut desc: MatmulDesc = std::ptr::null_mut();
        let mut a: Layout = std::ptr::null_mut();
        let mut b: Layout = std::ptr::null_mut();
        let mut c: Layout = std::ptr::null_mut();
        let mut pref: Preference = std::ptr::null_mut();
        // The chain below mirrors upstream statement by statement
        // (cublaslt.zig:113-137); on any failure `unwind` destroys
        // exactly what stood (the errdefer chain).
        let res = unsafe { (lt.api.cublasLtCreate.unwrap_unchecked())(&mut handle) };
        lt.check(res, "cublasLtCreate")?;
        let res = unsafe {
            (lt.api.cublasLtMatmulDescCreate.unwrap_unchecked())(
                &mut desc,
                compute_32f,
                DataType::f32,
            )
        };
        if let Err(e) = lt.check(res, "cublasLtMatmulDescCreate") {
            Linear::unwind(lt, handle, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        let ta: c_int = Op::t as c_int;
        let tb: c_int = Op::n as c_int;
        let res = unsafe {
            (lt.api.cublasLtMatmulDescSetAttribute.unwrap_unchecked())(
                desc,
                desc_transa,
                (&ta as *const c_int).cast(),
                std::mem::size_of::<c_int>(),
            )
        };
        if let Err(e) = lt.check(res, "set transa") {
            Linear::unwind(lt, handle, desc, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        let res = unsafe {
            (lt.api.cublasLtMatmulDescSetAttribute.unwrap_unchecked())(
                desc,
                desc_transb,
                (&tb as *const c_int).cast(),
                std::mem::size_of::<c_int>(),
            )
        };
        if let Err(e) = lt.check(res, "set transb") {
            Linear::unwind(lt, handle, desc, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        // column-major view: D^T[n,m] = W[n,k] (stored k x n,
        // transposed) . X^T[k,m]
        let res = unsafe {
            (lt.api.cublasLtMatrixLayoutCreate.unwrap_unchecked())(&mut a, DataType::bf16, k, n, k as i64)
        };
        if let Err(e) = lt.check(res, "layout W") {
            Linear::unwind(lt, handle, desc, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        let res = unsafe {
            (lt.api.cublasLtMatrixLayoutCreate.unwrap_unchecked())(&mut b, DataType::bf16, k, m, k as i64)
        };
        if let Err(e) = lt.check(res, "layout X") {
            Linear::unwind(lt, handle, desc, a, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        let res = unsafe {
            (lt.api.cublasLtMatrixLayoutCreate.unwrap_unchecked())(&mut c, out, n, m, n as i64)
        };
        if let Err(e) = lt.check(res, "layout D") {
            Linear::unwind(lt, handle, desc, a, b, std::ptr::null_mut(), std::ptr::null_mut());
            return Err(e);
        }
        let res = unsafe { (lt.api.cublasLtMatmulPreferenceCreate.unwrap_unchecked())(&mut pref) };
        if let Err(e) = lt.check(res, "cublasLtMatmulPreferenceCreate") {
            Linear::unwind(lt, handle, desc, a, b, c, std::ptr::null_mut());
            return Err(e);
        }
        let limit: u64 = workspace_limit as u64;
        let res = unsafe {
            (lt.api.cublasLtMatmulPreferenceSetAttribute.unwrap_unchecked())(
                pref,
                pref_max_workspace_bytes,
                (&limit as *const u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        if let Err(e) = lt.check(res, "set workspace") {
            Linear::unwind(lt, handle, desc, a, b, c, pref);
            return Err(e);
        }
        let mut found: [Heuristic; 1] = [Heuristic {
            algo: Algo { data: [0; 8] },
            workspace_size: 0,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }];
        let mut count: c_int = 0;
        let res = unsafe {
            (lt.api.cublasLtMatmulAlgoGetHeuristic.unwrap_unchecked())(
                handle,
                desc,
                a,
                b,
                c,
                c,
                pref,
                1,
                found.as_mut_ptr(),
                &mut count,
            )
        };
        if let Err(e) = lt.check(res, "cublasLtMatmulAlgoGetHeuristic") {
            Linear::unwind(lt, handle, desc, a, b, c, pref);
            return Err(e);
        }
        if count < 1 || found[0].state != 0 {
            Linear::unwind(lt, handle, desc, a, b, c, pref);
            return Err(Error::NoAlgorithm);
        }
        Ok(Linear {
            lt,
            handle,
            desc,
            a,
            b,
            c,
            pref,
            algo: found[0].algo,
            workspace_need: found[0].workspace_size,
        })
    }

    /// The errdefer chain's unwind: the non-null handles destroyed in
    /// the reverse of the build order.
    fn unwind(
        lt: &Library,
        handle: Handle,
        desc: MatmulDesc,
        a: Layout,
        b: Layout,
        c: Layout,
        pref: Preference,
    ) {
        // SAFETY: each non-null handle destroyed once.
        unsafe {
            if !pref.is_null() {
                _ = (lt.api.cublasLtMatmulPreferenceDestroy.unwrap_unchecked())(pref);
            }
            if !c.is_null() {
                _ = (lt.api.cublasLtMatrixLayoutDestroy.unwrap_unchecked())(c);
            }
            if !b.is_null() {
                _ = (lt.api.cublasLtMatrixLayoutDestroy.unwrap_unchecked())(b);
            }
            if !a.is_null() {
                _ = (lt.api.cublasLtMatrixLayoutDestroy.unwrap_unchecked())(a);
            }
            if !desc.is_null() {
                _ = (lt.api.cublasLtMatmulDescDestroy.unwrap_unchecked())(desc);
            }
            if !handle.is_null() {
                _ = (lt.api.cublasLtDestroy.unwrap_unchecked())(handle);
            }
        }
    }

    /// Upstream: `deinit` (cublaslt.zig:140) — pref, c, b, a, desc,
    /// handle.
    pub fn deinit(self_: &mut Linear) {
        Linear::unwind(
            self_.lt,
            self_.handle,
            self_.desc,
            self_.a,
            self_.b,
            self_.c,
            self_.pref,
        );
        self_.pref = std::ptr::null_mut();
        self_.c = std::ptr::null_mut();
        self_.b = std::ptr::null_mut();
        self_.a = std::ptr::null_mut();
        self_.desc = std::ptr::null_mut();
        self_.handle = std::ptr::null_mut();
    }

    /// D = X . W^T on `stream`; the workspace must hold
    /// `workspace_need` bytes. Upstream: `run` (cublaslt.zig:146).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // the pointers ride through to cuBLASLt, never dereferenced here
    pub fn run(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        d: DevicePtr,
        workspace: DevicePtr,
        workspace_len: usize,
        stream: Stream,
    ) -> Result<(), Error> {
        if workspace_len < self.workspace_need {
            return Err(Error::Invalid);
        }
        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        // SAFETY: the matmul over the picked algorithm.
        let res = unsafe {
            (self.lt.api.cublasLtMatmul.unwrap_unchecked())(
                self.handle,
                self.desc,
                (&one as *const f32).cast(),
                w,
                self.a,
                x,
                self.b,
                (&zero as *const f32).cast(),
                d,
                self.c,
                d,
                self.c,
                &self.algo,
                workspace,
                workspace_len,
                stream,
            )
        };
        self.lt.check(res, "cublasLtMatmul")
    }
}

impl Drop for Linear<'_> {
    fn drop(&mut self) {
        Linear::deinit(self);
    }
}
