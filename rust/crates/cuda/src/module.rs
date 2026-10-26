//! Loaded GPU code: cubin, fatbin or PTX images, their kernels by symbol
//! name and their device globals.
//!
//! Port of `zig/src/cuda/module.zig` (83 lines).

use crate::abi;
use crate::driver::{Driver, Error};
use std::ffi::CString;

/// Upstream: ``Module`` (module.zig:7).
pub struct Module<'d> {
    pub d: &'d Driver,
    pub handle: abi::Module,
}

impl<'d> Module<'d> {
    /// A cubin or fatbin image (8-byte aligned); the driver picks the
    /// SASS for this device and copies what it needs. Upstream: `load`
    /// (module.zig:12).
    pub fn load(d: &'d Driver, image: &[u8]) -> Result<Module<'d>, Error> {
        if image.is_empty() {
            eprintln!(
                "empty GPU image: this binary was built without kernels (-Dnvcc or -Dfatbins)"
            );
            return Err(Error::Invalid);
        }
        if image.as_ptr() as usize % 8 != 0 {
            return Err(Error::Invalid);
        }
        let mut m: abi::Module = std::ptr::null_mut();
        // SAFETY: the image a driver-owned module loads from.
        let res =
            unsafe { (d.api.cuModuleLoadData.unwrap_unchecked())(&mut m, image.as_ptr().cast()) };
        d.check(res, "cuModuleLoadData")?;
        Ok(Module { d, handle: m })
    }

    /// PTX text JIT-compiled by the driver; its error log is written into
    /// `log` on failure. Upstream: `loadPtx` (module.zig:23).
    pub fn load_ptx(d: &'d Driver, ptx: &str, log: &mut [u8]) -> Result<Module<'d>, Error> {
        log.fill(0);
        let text = CString::new(ptx).map_err(|_| Error::Invalid)?;
        let mut opts = [abi::JIT_ERROR_LOG_BUFFER, abi::JIT_ERROR_LOG_BUFFER_SIZE];
        let mut vals = [
            log.as_mut_ptr().cast::<core::ffi::c_void>(),
            log.len() as *mut core::ffi::c_void,
        ];
        let mut m: abi::Module = std::ptr::null_mut();
        // SAFETY: the PTX JIT-compiled with the error-log buffers wired.
        let res = unsafe {
            (d.api.cuModuleLoadDataEx.unwrap_unchecked())(
                &mut m,
                text.as_ptr().cast(),
                opts.len() as std::ffi::c_uint,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        d.check(res, "cuModuleLoadDataEx")?;
        Ok(Module { d, handle: m })
    }

    /// A kernel by its exact (mangled or extern "C") symbol; the function
    /// lives as long as the module. Upstream: `function` (module.zig:41).
    pub fn function(&self, name: &str) -> Result<Function<'d>, Error> {
        let name = CString::new(name).map_err(|_| Error::Invalid)?;
        let mut f: abi::Function = std::ptr::null_mut();
        // SAFETY: the symbol resolved out of the loaded module.
        let res =
            unsafe { (self.d.api.cuModuleGetFunction.unwrap_unchecked())(&mut f, self.handle, name.as_ptr()) };
        if let Err(e) = self.d.check(res, "cuModuleGetFunction") {
            eprintln!("kernel symbol not in module: {name:?}");
            return Err(e);
        }
        Ok(Function { d: self.d, handle: f })
    }

    pub fn global(&self, name: &str) -> Result<Global, Error> {
        let name = CString::new(name).map_err(|_| Error::Invalid)?;
        let mut g = Global { ptr: 0, len: 0 };
        // SAFETY: the module's device global resolved.
        let res = unsafe {
            (self.d.api.cuModuleGetGlobal_v2.unwrap_unchecked())(
                &mut g.ptr,
                &mut g.len,
                self.handle,
                name.as_ptr(),
            )
        };
        self.d.check(res, "cuModuleGetGlobal")?;
        Ok(g)
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        let Some(unload) = self.d.api.cuModuleUnload else {
            return; // a stub driver: nothing was ever loaded
        };
        // SAFETY: the module unloaded once.
        unsafe { unload(self.handle) };
    }
}

/// Upstream: `Global` (module.zig:49).
pub struct Global {
    pub ptr: abi::DevicePtr,
    pub len: usize,
}

/// Upstream: ``Function`` (module.zig:58).
pub struct Function<'d> {
    pub d: &'d Driver,
    pub handle: abi::Function,
}

impl<'d> Function<'d> {
    pub fn attribute(&self, a: std::ffi::c_int) -> Result<std::ffi::c_int, Error> {
        let mut v: std::ffi::c_int = 0;
        // SAFETY: the function's attribute read.
        let res = unsafe { (self.d.api.cuFuncGetAttribute.unwrap_unchecked())(&mut v, a, self.handle) };
        self.d.check(res, "cuFuncGetAttribute")?;
        Ok(v)
    }

    pub fn set_attribute(
        &self,
        a: std::ffi::c_int,
        value: std::ffi::c_int,
    ) -> Result<(), Error> {
        // SAFETY: the function's attribute set.
        let res = unsafe { (self.d.api.cuFuncSetAttribute.unwrap_unchecked())(self.handle, a, value) };
        self.d.check(res, "cuFuncSetAttribute")
    }

    /// Resident blocks an SM holds of this kernel at `threads` a block
    /// and `shared` dynamic bytes. Upstream: `occupancy` (module.zig:66).
    pub fn occupancy(&self, threads: u32, shared: usize) -> Result<u32, Error> {
        let mut n: std::ffi::c_int = 0;
        // SAFETY: the occupancy calculated for the block shape.
        let res = unsafe {
            (self.d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor.unwrap_unchecked())(
                &mut n,
                self.handle,
                threads as std::ffi::c_int,
                shared,
            )
        };
        self.d.check(res, "cuOccupancyMaxActiveBlocksPerMultiprocessor")?;
        Ok(n.max(0) as u32)
    }

    /// Dynamic shared memory above 48 KiB must be opted into per kernel
    /// before launch. Upstream: `allowDynamicShared` (module.zig:72).
    pub fn allow_dynamic_shared(&self, bytes: u32) -> Result<(), Error> {
        self.set_attribute(abi::FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as std::ffi::c_int)
    }
}
