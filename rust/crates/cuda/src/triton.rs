//! Triton kernels compiled ahead of time: a cubin plus its metadata,
//! launched the way Triton 3.7's own launcher does.
//!
//! Port of `zig/src/cuda/triton.zig` (93 lines).

use crate::abi;
use crate::driver::{Driver, Error};
use crate::launch::{self, Args, Config, Dim3};
use crate::module::{Function, Module};
use crate::stream::Stream;

/// The launch facts Triton writes into a kernel's metadata JSON (field
/// names as Triton spells them). Upstream: `Meta` (triton.zig:13).
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub name: String,
    pub num_warps: u32,
    pub num_ctas: u32,
    pub shared: u32,
    pub global_scratch_size: u32,
    pub global_scratch_align: u32,
    pub profile_scratch_size: u32,
    pub profile_scratch_align: u32,
    pub launch_cooperative_grid: bool,
    pub launch_pdl: bool,
}

impl Meta {
    /// The default metadata's own values (Triton's own defaults):
    /// num_ctas 1, the aligns 1.
    pub const DEFAULT_META_CTAS: u32 = 1;
}

/// Upstream: `parseMeta` (triton.zig:27) — the unknown fields ignored.
pub fn parse_meta(json: &str) -> Result<Meta, Error> {
    #[derive(serde::Deserialize)]
    struct MetaJson {
        name: String,
        num_warps: u32,
        #[serde(default = "default_one")]
        num_ctas: u32,
        #[serde(default)]
        shared: u32,
        #[serde(default)]
        global_scratch_size: u32,
        #[serde(default = "default_one")]
        global_scratch_align: u32,
        #[serde(default)]
        profile_scratch_size: u32,
        #[serde(default = "default_one")]
        profile_scratch_align: u32,
        #[serde(default)]
        launch_cooperative_grid: bool,
        #[serde(default)]
        launch_pdl: bool,
    }
    fn default_one() -> u32 {
        1
    }
    let v: MetaJson = serde_json::from_str(json).map_err(|_| Error::Invalid)?;
    Ok(Meta {
        name: v.name,
        num_warps: v.num_warps,
        num_ctas: v.num_ctas,
        shared: v.shared,
        global_scratch_size: v.global_scratch_size,
        global_scratch_align: v.global_scratch_align,
        profile_scratch_size: v.profile_scratch_size,
        profile_scratch_align: v.profile_scratch_align,
        launch_cooperative_grid: v.launch_cooperative_grid,
        launch_pdl: v.launch_pdl,
    })
}

/// Upstream: `Scratch` (triton.zig:91).
#[derive(Clone, Copy, Default)]
pub struct Scratch {
    pub global: abi::DevicePtr,
    pub profile: abi::DevicePtr,
}

/// Upstream: ``Kernel`` (triton.zig:31).
pub struct Kernel<'d> {
    pub module: Module<'d>,
    pub function: Function<'d>,
    pub meta: Meta,
}

impl<'d> Kernel<'d> {
    /// Loads the cubin and, as Triton's loader does, opts the kernel
    /// into its dynamic shared memory above 48 KiB. Upstream: `load`
    /// (triton.zig:35).
    pub fn load(
        d: &'d Driver,
        device: abi::Device,
        cubin: &[u8],
        meta: Meta,
        name: &str,
    ) -> Result<Kernel<'d>, Error> {
        if meta.name != name {
            return Err(Error::Invalid);
        }
        let m = Module::load(d, cubin)?;
        let function = match m.function(name) {
            Ok(f) => f,
            Err(e) => {
                let handle = m.handle;
                std::mem::forget(m); // the module unloads its own below
                let _ = handle;
                return Err(e);
            }
        };
        if meta.shared > 49152 {
            // SAFETY: the device's opt-in shared-memory limit read.
            let res = unsafe {
                (d.api.cuDeviceGetAttribute.unwrap_unchecked())(
                    &mut 0,
                    abi::DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    device,
                )
            };
            if res == abi::SUCCESS {
                // SAFETY: the limit read properly.
                let mut optin: std::ffi::c_int = 0;
                let res = unsafe {
                    (d.api.cuDeviceGetAttribute.unwrap_unchecked())(
                        &mut optin,
                        abi::DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        device,
                    )
                };
                if d.check(res, "cuDeviceGetAttribute").is_ok() && optin > 49152 {
                    // SAFETY: the cache config preferred to shared.
                    let res = unsafe {
                        (d.api.cuFuncSetCacheConfig.unwrap_unchecked())(
                            function.handle,
                            abi::FUNC_CACHE_PREFER_SHARED,
                        )
                    };
                    if d.check(res, "cuFuncSetCacheConfig").is_err() {
                        drop(m);
                        return Err(Error::Invalid);
                    }
                    let static_shared = function
                        .attribute(abi::FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
                        .unwrap_or(0);
                    let _ = function
                        .set_attribute(
                            abi::FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            optin - static_shared,
                        );
                }
            }
        }
        Ok(Kernel { module: m, function, meta })
    }

    /// The launch geometry Triton uses for a grid: x times num_ctas, 32 *
    /// num_warps threads, metadata shared bytes. Upstream: `config`
    /// (triton.zig:57).
    pub fn config(&self, grid: Dim3) -> Config {
        let m = &self.meta;
        Config {
            grid: Dim3 { x: grid.x * m.num_ctas, y: grid.y, z: grid.z },
            block: Dim3 { x: 32 * m.num_warps, ..Dim3::DEFAULT },
            shared: m.shared,
            cluster: (m.num_ctas != 1).then_some(Dim3 { x: m.num_ctas, ..Dim3::DEFAULT }),
            cluster_spread: m.num_ctas != 1,
            pdl: m.launch_pdl,
            cooperative: m.launch_cooperative_grid,
        }
    }

    /// `args`: non-constexpr arguments in order (scratch pointers appended
    /// here); `divisible16` args must be multiples of 16. Upstream:
    /// `launchOn` (triton.zig:63).
    pub fn launch_on(
        &self,
        grid: Dim3,
        stream: &Stream,
        args: &mut Args,
        scratch: Scratch,
        divisible16: &[u32],
    ) -> Result<(), Error> {
        let m = &self.meta;
        for &i in divisible16 {
            let v = args.integer(i as usize).ok_or(Error::Invalid)?;
            if v % 16 != 0 {
                return Err(Error::Invalid);
            }
        }
        if m.global_scratch_size > 0 && scratch.global == 0 {
            return Err(Error::Invalid);
        }
        if m.profile_scratch_size > 0 && scratch.profile == 0 {
            return Err(Error::Invalid);
        }
        if m.num_ctas == 16 {
            self.function.set_attribute(
                abi::FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )?;
        }
        args.add(scratch.global);
        args.add(scratch.profile);
        launch::launch(&self.function, self.config(grid), stream, args)
    }

    /// Bytes of global scratch a launch of `grid` needs (Triton: blocks
    /// * num_ctas * size). Upstream: `globalScratchBytes` (triton.zig:79).
    pub fn global_scratch_bytes(&self, grid: Dim3) -> usize {
        grid.x as usize * grid.y as usize * grid.z as usize
            * self.meta.num_ctas as usize
            * self.meta.global_scratch_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Triton metadata parse: Triton's own field spellings, the
    /// unknown ignored, the defaults (num_ctas 1, the aligns 1).
    #[test]
    fn meta_parses() {
        let m = parse_meta(
            r#"{"name": "k", "num_warps": 4, "num_ctas": 2, "shared": 1024,
                 "global_scratch_size": 256, "launch_pdl": true, "unknown": 1}"#,
        )
        .unwrap();
        assert_eq!(m.name, "k");
        assert_eq!(m.num_warps, 4);
        assert_eq!(m.num_ctas, 2);
        assert_eq!(m.shared, 1024);
        assert_eq!(m.global_scratch_size, 256);
        assert!(m.launch_pdl);
        assert!(!m.launch_cooperative_grid);
        let m = parse_meta(r#"{"name": "k", "num_warps": 1}"#).unwrap();
        assert_eq!(m.num_ctas, 1, "the default num_ctas");
        assert_eq!(m.global_scratch_align, 1, "the default align");
        assert!(parse_meta("not json").is_err());
    }

    /// The launch geometry Triton uses (triton.zig:57): x times num_ctas,
    /// 32 * num_warps threads, the cluster and its spread.
    #[test]
    #[allow(clippy::identity_op)] // upstream spells the z=1 factor out
    fn the_launch_geometry() {
        let k = Kernel {
            module: crate::driver::stub::stub_module(),
            function: crate::driver::stub::stub_function(),
            meta: Meta {
                name: "k".into(),
                num_warps: 4,
                num_ctas: 2,
                shared: 1024,
                global_scratch_size: 256,
                launch_pdl: true,
                ..Default::default()
            },
        };
        let c = k.config(Dim3 { x: 3, y: 1, z: 1 });
        assert_eq!(c.grid.x, 6, "x times num_ctas");
        assert_eq!(c.block.x, 128, "32 * num_warps");
        assert_eq!(c.shared, 1024);
        assert_eq!(c.cluster, Some(Dim3 { x: 2, y: 1, z: 1 }));
        assert!(c.cluster_spread);
        assert!(c.pdl);
        // The global scratch: blocks * num_ctas * size
        assert_eq!(
            k.global_scratch_bytes(Dim3 { x: 2, y: 2, z: 1 }),
            2 * 2 * 1 * 2 * 256
        );
    }
}
