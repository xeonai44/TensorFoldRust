//! One device's primary context, the one CUDA's runtime and PyTorch
//! share, made current on the calling thread.
//!
//! Port of `zig/src/cuda/context.zig` (65 lines).

use crate::abi;
use crate::driver::{Driver, Error};

/// Upstream: `MemInfo` (context.zig:53).
pub struct MemInfo {
    pub free: usize,
    pub total: usize,
}

/// Upstream: ``Context`` (context.zig:8).
pub struct Context<'d> {
    pub d: &'d Driver,
    pub device: abi::Device,
    pub handle: abi::Context,
}

impl<'d> Context<'d> {
    /// Upstream: `init` (context.zig:13).
    pub fn init(d: &'d Driver, ordinal: std::ffi::c_int) -> Result<Context<'d>, Error> {
        let mut dev: abi::Device = 0;
        // SAFETY: the ordinal resolved into a device handle.
        let res = unsafe { (d.api.cuDeviceGet.unwrap_unchecked())(&mut dev, ordinal) };
        d.check(res, "cuDeviceGet")?;
        let mut ctx: abi::Context = std::ptr::null_mut();
        // SAFETY: the primary context retained.
        let res = unsafe { (d.api.cuDevicePrimaryCtxRetain.unwrap_unchecked())(&mut ctx, dev) };
        d.check(res, "cuDevicePrimaryCtxRetain")?;
        // SAFETY: the context made current on this thread.
        let res = unsafe { (d.api.cuCtxSetCurrent.unwrap_unchecked())(ctx) };
        if d.check(res, "cuCtxSetCurrent").is_err() {
            // SAFETY: the retain undone on the failure path.
            unsafe { (d.api.cuDevicePrimaryCtxRelease_v2.unwrap_unchecked())(dev) };
            return Err(Error::CudaFailed {
                what: "cuCtxSetCurrent".into(),
                code: res,
            });
        }
        Ok(Context { d, device: dev, handle: ctx })
    }

    /// Waits for all work, then drops this retain; resources made in it
    /// must be released first. Upstream: `deinit` (context.zig:26).
    pub fn deinit(&mut self) {
        // SAFETY: the drains before the release.
        unsafe { (self.d.api.cuCtxSynchronize.unwrap_unchecked())() };
        // SAFETY: this retain released.
        unsafe { (self.d.api.cuDevicePrimaryCtxRelease_v2.unwrap_unchecked())(self.device) };
    }

    /// Makes the context current on another thread before it calls the
    /// driver. Upstream: `makeCurrent` (context.zig:32).
    pub fn make_current(&self) -> Result<(), Error> {
        // SAFETY: the context made current on the calling thread.
        let res = unsafe { (self.d.api.cuCtxSetCurrent.unwrap_unchecked())(self.handle) };
        self.d.check(res, "cuCtxSetCurrent")
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        // SAFETY: the context's work drained.
        let res = unsafe { (self.d.api.cuCtxSynchronize.unwrap_unchecked())() };
        self.d.check(res, "cuCtxSynchronize")
    }

    pub fn attribute(&self, a: std::ffi::c_int) -> Result<std::ffi::c_int, Error> {
        let mut v: std::ffi::c_int = 0;
        // SAFETY: the attribute read.
        let res = unsafe {
            (self.d.api.cuDeviceGetAttribute.unwrap_unchecked())(&mut v, a, self.device)
        };
        self.d.check(res, "cuDeviceGetAttribute")?;
        Ok(v)
    }

    /// Compute capability as 10 * major + minor (GB10: 121). Upstream:
    /// `capability` (context.zig:44).
    pub fn capability(&self) -> Result<u32, Error> {
        let major = self.attribute(abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.attribute(abi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((10 * major + minor) as u32)
    }

    pub fn name(&self, buf: &mut [u8]) -> Result<String, Error> {
        if buf.len() < 2 {
            return Err(Error::Invalid);
        }
        // SAFETY: the name read into the caller's buffer.
        let res = unsafe {
            (self.d.api.cuDeviceGetName.unwrap_unchecked())(
                buf.as_mut_ptr(),
                buf.len() as std::ffi::c_int,
                self.device,
            )
        };
        self.d.check(res, "cuDeviceGetName")?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    pub fn mem_info(&self) -> Result<MemInfo, Error> {
        let mut m = MemInfo { free: 0, total: 0 };
        // SAFETY: the memory info read.
        let res = unsafe {
            (self.d.api.cuMemGetInfo_v2.unwrap_unchecked())(&mut m.free, &mut m.total)
        };
        self.d.check(res, "cuMemGetInfo")?;
        Ok(m)
    }
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        self.deinit();
    }
}
