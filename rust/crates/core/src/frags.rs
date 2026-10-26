//! The prompt kernels' fragment layout: nax.h inlined for this GPU, and
//! checked on it.
//!
//! Port of `zig/src/core/frags.zig` (83 lines). Upstream `source` takes an
//! allocator; the port's `String` allocation cannot fail, so the signature is
//! `(device, text)`. Upstream `check` takes a device and a queue; the port's
//! metal layer couples them in one `Gpu`, so it takes that (the seam noted in
//! place).

use tensorfold_metal::{Buffer, CompileOptions, Device, Gpu, Library};

/// nax.h, embedded byte-identical (upstream `kernel_sources.nax`,
/// `@embedFile("nax.h")`).
const NAX: &str = include_str!("../../../kernels/metal/nax.h");

/// A prompt-kernel source with nax.h inlined; a GPU without tensor units takes
/// nax.h's simdgroup-matrix layout.
///
/// Upstream: `frags.zig:7  source`.
pub fn source(device: &Device, text: &str) -> String {
    let src = text.replace("#include \"../nax.h\"", NAX);
    if device.tensor_units() {
        return src;
    }
    format!("#define TF_SIMD_FRAGS 1\n{src}")
}

/// The op both ways the prompt kernels use it (a transposed right operand, a
/// plain one) through nax.h's fragments.
///
/// Upstream: `frags.zig:15  check_source`, verbatim.
const CHECK_SOURCE: &str = "#include \"../nax.h\"
using namespace tfp;
[[kernel]] void tf_frag_check(const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]],
    const device bfloat* E [[buffer(2)]], device float* C [[buffer(3)]], uint lane [[thread_index_in_simdgroup]]) {
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> a, b0, b1, e0, e1;
  frag_get(a, A, 16, 0, 0, home);
  frag_get_t(b0, B, 16, 0, 0, home);
  frag_get_t(b1, B, 16, 16, 0, home);
  frag_get(e0, E, 32, 0, 0, home);
  frag_get(e1, E, 32, 0, 16, home);
  frag<float> c0 = frag<float>(0), c1 = frag<float>(0), d0 = frag<float>(0), d1 = frag<float>(0);
  mma_16x32<false, true>(c0, c1, a, b0, b1);
  mma_16x32<false, false>(d0, d1, a, e0, e1);
  frag_put(c0, C, 32, 0, 0, home);
  frag_put(c1, C, 32, 0, 16, home);
  frag_put(d0, C + 512, 32, 0, 0, home);
  frag_put(d1, C + 512, 32, 0, 16, home);
}
";

/// Why the fragment check failed.
#[derive(Debug)]
pub enum FragsError {
    /// Upstream: `error.GpuFailed` — the command buffer failed.
    GpuFailed(String),
    /// Upstream: `error.FragLayout` — the layout is wrong on this GPU; carries
    /// the device's name.
    FragLayout(String),
}

impl std::fmt::Display for FragsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragsError::GpuFailed(message) => write!(f, "fragment check failed: {message}"),
            FragsError::FragLayout(device) => {
                write!(f, "prompt kernels: nax.h's fragment layout is wrong on {device}")
            }
        }
    }
}

impl std::error::Error for FragsError {}

/// Small integers a check reads (exact products and sums in any order).
///
/// Upstream: `frags.zig:38  value`.
fn value(i: usize, salt: usize) -> f32 {
    ((i * 7 + salt * 3) % 5) as i32 as f32 - 2.0
}

/// A host view of a shared buffer's bytes as `u16`s (upstream
/// `buffer.slice(u16, n)` over the contents).
///
/// Safety: the buffer must outlive the view and no in-flight GPU work may
/// write it (the check waits for completion before it reads).
unsafe fn u16_slice(buffer: &Buffer, n: usize) -> &[u16] {
    std::slice::from_raw_parts(buffer.as_ptr().cast::<u16>(), n)
}

/// nax.h's layout checked on this GPU, A B^T and A E against the host's sums:
/// [`FragsError::FragLayout`] when it is wrong.
///
/// Upstream: `frags.zig:43  check` — takes a device and its queue; the port's
/// `Gpu` couples them, so it takes the one handle.
pub fn check(gpu: &Gpu) -> Result<(), FragsError> {
    let device = gpu.device();
    let text = source(device, CHECK_SOURCE);
    let lib = Library::from_source(device, &text, CompileOptions::mlx())
        .map_err(FragsError::GpuFailed)?;
    let pipe = lib
        .pipeline(device, "tf_frag_check", false)
        .map_err(FragsError::GpuFailed)?;
    let sizes = [16 * 16 * 2, 32 * 16 * 2, 16 * 32 * 2, 2 * 16 * 32 * 4];
    let mut bufs: Vec<Buffer> = Vec::with_capacity(4);
    for &len in &sizes {
        bufs.push(gpu.buffer_with_len(len));
    }
    for m in 0..3 {
        // Safety: the buffer is fresh and no GPU work is in flight; each fill
        // happens before the pass below is encoded.
        let view = unsafe { std::slice::from_raw_parts_mut(bufs[m].as_ptr().cast::<u16>(), sizes[m] / 2) };
        for (i, x) in view.iter_mut().enumerate() {
            *x = (value(i, m).to_bits() >> 16) as u16;
        }
    }
    let pass = gpu.compute_pass();
    let enc = pass.encoder();
    enc.set_pipeline(&pipe);
    for (j, b) in bufs.iter().enumerate() {
        enc.set_buffer(b, 0, j);
    }
    enc.dispatch((32, 1, 1), (32, 1, 1));
    pass.submit().map_err(FragsError::GpuFailed)?;
    // Safety: the pass above waited for completion.
    let out = unsafe { std::slice::from_raw_parts(bufs[3].as_ptr().cast::<f32>(), 1024) };
    for i in 0..16 {
        for j in 0..32 {
            let mut c: f32 = 0.0;
            let mut d: f32 = 0.0;
            for k in 0..16 {
                c += value(i * 16 + k, 0) * value(j * 16 + k, 1);
                d += value(i * 16 + k, 0) * value(k * 32 + j, 2);
            }
            if out[i * 32 + j] != c || out[512 + i * 32 + j] != d {
                return Err(FragsError::FragLayout(device.name()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The port's own live gate (upstream checks this at glm's engine open, so
    /// frags.zig itself has no test): nax.h's layout holds on this GPU, A B^T
    /// and A E exact against the host's sums. Skipped without a GPU, like
    /// upstream's `Device.init() catch return error.SkipZigTest`.
    #[test]
    fn nax_h_layout_holds_on_this_gpu() {
        let Some(gpu) = Gpu::open() else {
            return;
        };
        check(&gpu).unwrap();
    }
}
