//! Runtime kernel compilation: what `mx.fast.metal_kernel` compiles with, the port of
//! `zig/src/metal/library.zig`'s fromSource half.
//!
//! Upstream compiles the generated family kernels from embedded source at open time
//! (`families/qwen3_5/kernels.zig:57  mtl.Library.fromSource(device, s.source,
//! mtl.CompileOptions.mlx())`) — deliberately the same runtime compiler and options
//! MLX uses, so the numerics match what the captured fixtures were produced with.
//! The port keeps that path: sources are embedded (`include_str!` where upstream has
//! `@embedFile`) and compiled at open with the options below.

use crate::device::Device;
use crate::objc::{self, Id};
use crate::pipeline::Pipeline;
use metal::foreign_types::ForeignType;
use ::objc::{class, msg_send, sel, sel_impl};

/// MTLCompileOptions for a run-time compile; `mlx()` gives what
/// mx.fast.metal_kernel uses on this macOS.
///
/// Upstream: `metal/library.zig:15  pub const CompileOptions = struct { ... }` —
/// `math_mode = .safe`, `math_functions = .fast` (Metal's defaults), and MLX's language
/// version for the OS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    /// MTLMathMode: safe = 0, relaxed = 1, fast = 2.
    pub math_mode: isize,
    /// MTLMathFloatingPointFunctions: fast = 0, precise = 1.
    pub math_functions: isize,
    /// MTLLanguageVersion raw value: (major << 16) + minor.
    pub language: usize,
}

impl CompileOptions {
    /// Safe math, fast fp32 functions (Metal's default), and MLX's language version
    /// for this OS.
    pub fn mlx() -> CompileOptions {
        CompileOptions {
            math_mode: 0,
            math_functions: 0,
            language: mlx_language(macos_major()),
        }
    }
}

/// The host macOS's major version, the input to the MLX language-version rule.
///
/// Upstream reads NSProcessInfo (`metal/objc.zig` `osVersion().major`); the port runs
/// `sw_vers` — the same source build.zig's pipeline uses — once per compile. A missing
/// answer falls back to macOS 15 (the oldest macOS upstream's table gives metal3.2), so
/// a failed probe shows up as a compile refusal, never as a guess at newer syntax.
fn macos_major() -> u64 {
    if let Ok(out) = std::process::Command::new("sw_vers").arg("-productVersion").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(major) = text.trim().split('.').next().and_then(|p| p.parse().ok()) {
            return major;
        }
    }
    15
}

/// MLX's custom-kernel language version on macOS `major`: 4.1 from 27, 4.0 on 26,
/// 3.2 on 15, else 3.1.
///
/// Upstream: `metal/library.zig:27  pub fn mlxLanguage`.
pub fn mlx_language(major: u64) -> usize {
    if major >= 27 {
        language_version(4, 1)
    } else if major >= 26 {
        language_version(4, 0)
    } else if major >= 15 {
        language_version(3, 2)
    } else {
        language_version(3, 1)
    }
}

/// Upstream: `metal/types.zig:49  pub fn languageVersion(major, minor) usize`.
pub fn language_version(major: u64, minor: u64) -> usize {
    ((major as usize) << 16) + minor as usize
}

/// A library compiled at run time from source.
///
/// Upstream: `metal/library.zig:42  pub const Library = struct { id }` — the fromSource
/// half; the fromFile/fromBytes halves belong to the build-time metallib path.
pub struct Library {
    id: Id,
}

// Safety: the Library owns one ObjC reference; the object is only touched through
// this handle on whatever thread opened the engine, like upstream's mtl.Library.
unsafe impl Send for Library {}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { msg_send![self.id, release] }
    }
}

impl Library {
    /// Compile `source` on `device` with `options`.
    ///
    /// Upstream: `metal/library.zig:63  pub fn fromSource` — MTLCompileOptions gets
    /// setMathMode:, setMathFloatingPointFunctions: and setLanguageVersion:, then
    /// newLibraryWithSource:options:error:.
    pub fn from_source(device: &Device, source: &str, options: CompileOptions) -> Result<Library, String> {
        unsafe {
            let source_string = objc::string(source);
            let opts: Id = msg_send![class!(MTLCompileOptions), new];
            if opts.is_null() {
                let _: () = msg_send![source_string, release];
                return Err("cannot allocate MTLCompileOptions".to_string());
            }
            let _: () = msg_send![opts, setMathMode: options.math_mode];
            let _: () = msg_send![
                opts,
                setMathFloatingPointFunctions: options.math_functions
            ];
            let _: () = msg_send![opts, setLanguageVersion: options.language];
            let mut err: Id = std::ptr::null_mut();
            let id: Id = msg_send![
                device.id(),
                newLibraryWithSource: source_string options: opts error: &mut err
            ];
            let _: () = msg_send![opts, release];
            let _: () = msg_send![source_string, release];
            if id.is_null() {
                let message = if !err.is_null() {
                    let desc: Id = msg_send![err, localizedDescription];
                    crate::objc::nsstring_text(desc)
                } else {
                    "unknown compile error".to_string()
                };
                return Err(format!("source ({} bytes): {}", source.len(), message));
            }
            Ok(Library { id })
        }
    }

    /// One named function of this library as a runnable pipeline.
    ///
    /// Upstream: `metal/library.zig:95  Pipeline.init` — the descriptor path
    /// (`MTLComputePipelineDescriptor`, `setSupportIndirectCommandBuffers:`,
    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`), which is
    /// the only creation path the engine uses, including for run-time-compiled
    /// kernels. Unknown names are an error, never a guess — the same refusal the
    /// comptime registry enforced. `indirect` lets indirect command buffers use the
    /// pipeline (upstream passes false from the family loads).
    pub fn pipeline(&self, device: &Device, name: &str, indirect: bool) -> Result<Pipeline, String> {
        unsafe {
            let name_string = objc::string(name);
            let function: Id = msg_send![self.id, newFunctionWithName: name_string];
            let _: () = msg_send![name_string, release];
            if function.is_null() {
                return Err(format!("no function {} in the library", name));
            }
            let desc: Id = msg_send![class!(MTLComputePipelineDescriptor), new];
            if desc.is_null() {
                let _: () = msg_send![function, release];
                return Err("cannot allocate MTLComputePipelineDescriptor".to_string());
            }
            let _: () = msg_send![desc, setComputeFunction: function];
            let _: () = msg_send![desc, setSupportIndirectCommandBuffers: indirect];
            let mut err: Id = std::ptr::null_mut();
            let pipeline: Id = msg_send![
                device.id(),
                newComputePipelineStateWithDescriptor: desc
                options: 0usize
                reflection: std::ptr::null_mut::<std::ffi::c_void>()
                error: &mut err
            ];
            let _: () = msg_send![desc, release];
            let _: () = msg_send![function, release];
            if pipeline.is_null() {
                let message = if !err.is_null() {
                    let desc: Id = msg_send![err, localizedDescription];
                    objc::nsstring_text(desc)
                } else {
                    "unknown pipeline error".to_string()
                };
                return Err(format!("pipeline {}: {}", name, message));
            }
            // Adopt the retained object through the crate's own constructor: the
            // owned wrappers are repr(transparent) over the object pointer, and
            // `from_ptr` is the interface foreign-types generates for exactly this.
            let state = metal::ComputePipelineState::from_ptr(pipeline as *mut metal::MTLComputePipelineState);
            Ok(Pipeline::from_state(state))
        }
    }
}

// The device's raw id, for the message sends above (the typed wrapper keeps its
// field private; this is the crate's own seam, matching upstream's `device.id` field).
impl Device {
    pub(crate) fn id(&self) -> Id {
        // Safety: the owned wrapper is repr(transparent) over the object pointer
        // (foreign-types), so `as_ptr` is the ObjC id itself.
        use metal::foreign_types::ForeignType;
        self.0.as_ptr() as Id
    }
}
