//! Pipelines: the functions a metallib exposes, compiled for this device.
//!
//! Port of `zig/src/metal/library.zig` and `zig/src/metal/types.zig`'s pipeline side.
//! Upstream's kernel registries — the comptime name-keyed tables that map a kernel key
//! to its pipeline — are deliberately not ported as comptime; the Rust replacement is a
//! runtime registry compiled once per family (decision record: docs/DECISIONS.md,
//! ADR-005), which the first family engine builds in Phase 1.

use crate::device::Device;

/// The kernels a build embeds, opened as this device's library.
pub struct Kernels {
    library: metal::Library,
}

impl Kernels {
    /// Load the embedded metallib (upstream: the embedded module's bytes passed to
    /// `mtl.library`).
    pub fn from_builtin(device: &Device) -> Result<Kernels, String> {
        Ok(Kernels {
            library: device.library_from_builtin()?,
        })
    }

    /// One named kernel as a runnable pipeline. Unknown names are an error, never a
    /// guess — the same refusal the comptime registry enforced at compile time
    /// (`families/nemotron/kernels.zig:126  fn index`).
    pub fn pipeline(&self, device: &Device, name: &str) -> Result<Pipeline, String> {
        objc::rc::autoreleasepool(|| {
            let function = self
                .library
                .get_function(name, None)
                .map_err(|_| format!("no kernel named {}", name))?;
            let state = device.compute_pipeline_state(&function)?;
            Ok(Pipeline(state))
        })
    }
}

/// A compiled compute pipeline bound to one kernel function. The state inside is the
/// `metal` crate's own wrapper (clones retain); `Gpu::dispatch` is the one that
/// binds it.
#[derive(Clone)]
pub struct Pipeline(pub(crate) metal::ComputePipelineState);

impl Pipeline {
    /// Adopt a pipeline the runtime-compile path produced (`library::Library`).
    pub(crate) fn from_state(state: metal::ComputePipelineState) -> Pipeline {
        Pipeline(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Gpu;
    use crate::kernels_builtin;

    /// Upstream `runtime_tests.metal`'s own first test, run the way its Zig runtime
    /// tests do: fill two buffers, dispatch `tf_vadd`, read the third back.
    #[test]
    fn vadd_round_trips() {
        if !kernels_builtin() {
            eprintln!("skipping: no Metal toolchain in this build");
            return;
        }
        let Some(gpu) = Gpu::open() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = Kernels::from_builtin(gpu.device()).expect("the builtin metallib loads");
        let vadd = kernels
            .pipeline(gpu.device(), "tf_vadd")
            .expect("runtime_tests.metal holds tf_vadd");
        let n = 1024u32;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
        let c = gpu.buffer_with_len(n as usize * 4);
        let a_bytes = bytemuck_lite(&a);
        let b_bytes = bytemuck_lite(&b);
        let n_bytes = gpu.buffer_with_data(&n.to_le_bytes());
        let ba = gpu.buffer_with_data(&a_bytes);
        let bb = gpu.buffer_with_data(&b_bytes);
        gpu.dispatch(&vadd, &[&ba, &bb, &c, &n_bytes], (n as u64, 1, 1), (64, 1, 1));
        let got = f32_slice(c.bytes());
        for i in 0..n as usize {
            assert_eq!(got[i], 3.0 * i as f32, "row {}", i);
        }
    }

    fn bytemuck_lite(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn f32_slice(bytes: &[u8]) -> &[f32] {
        assert_eq!(bytes.len() % 4, 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
    }
}
