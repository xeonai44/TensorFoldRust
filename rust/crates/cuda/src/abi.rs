//! CUDA driver API types and entry points as NVIDIA documents them
//! (CUDA 12.0+ ABI), declared by hand.
//!
//! Port of `zig/src/cuda/abi.zig` (222 lines): the result codes, the
//! opaque handles, the attribute enums as the ABI's own integers, the
//! launch structs at their documented sizes, and ``Api`` — every entry
//! point the runtime calls, each field the exact exported symbol, so
//! the loader resolves by name and a miss refuses the whole driver
//! (upstream: "Each field is the exact exported symbol").

pub type Result = std::ffi::c_int;
pub type Device = std::ffi::c_int;
/// CUdeviceptr.
pub type DevicePtr = u64;

pub type Context = *mut core::ffi::c_void;
pub type Module = *mut core::ffi::c_void;
pub type Function = *mut core::ffi::c_void;
pub type Stream = *mut core::ffi::c_void;
pub type Event = *mut core::ffi::c_void;
pub type Graph = *mut core::ffi::c_void;
pub type GraphNode = *mut core::ffi::c_void;
pub type GraphExec = *mut core::ffi::c_void;
pub type Kernel = *mut core::ffi::c_void;

pub const SUCCESS: Result = 0;
pub const ERROR_NOT_FOUND: Result = 500;
pub const ERROR_NOT_READY: Result = 600;

pub const STREAM_NON_BLOCKING: std::ffi::c_uint = 1;
pub const EVENT_DISABLE_TIMING: std::ffi::c_uint = 2;
pub const HOST_ALLOC_PORTABLE: std::ffi::c_uint = 1;
pub const HOST_ALLOC_DEVICEMAP: std::ffi::c_uint = 2;
pub const HOST_REGISTER_DEVICEMAP: std::ffi::c_uint = 2;
/// The range is device or I/O memory, not RAM.
pub const HOST_REGISTER_IOMEMORY: std::ffi::c_uint = 4;

// CUctxSynchronize's capture modes
pub const CAPTURE_MODE_GLOBAL: std::ffi::c_int = 0;
pub const CAPTURE_MODE_THREAD_LOCAL: std::ffi::c_int = 1;
pub const CAPTURE_MODE_RELAXED: std::ffi::c_int = 2;

// CUstreamCaptureStatus
pub const CAPTURE_STATUS_NONE: std::ffi::c_int = 0;
pub const CAPTURE_STATUS_ACTIVE: std::ffi::c_int = 1;
pub const CAPTURE_STATUS_INVALIDATED: std::ffi::c_int = 2;

/// CUdevice_attribute's values the runtime reads.
pub const DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: std::ffi::c_int = 1;
pub const DEVICE_ATTRIBUTE_WARP_SIZE: std::ffi::c_int = 10;
pub const DEVICE_ATTRIBUTE_CLOCK_RATE: std::ffi::c_int = 13;
pub const DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: std::ffi::c_int = 16;
pub const DEVICE_ATTRIBUTE_INTEGRATED: std::ffi::c_int = 18;
pub const DEVICE_ATTRIBUTE_L2_CACHE_SIZE: std::ffi::c_int = 38;
pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: std::ffi::c_int = 75;
pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: std::ffi::c_int = 76;
pub const DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: std::ffi::c_int = 81;
pub const DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: std::ffi::c_int = 97;

/// CUfunction_attribute's values the runtime reads and sets.
pub const FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: std::ffi::c_int = 0;
pub const FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: std::ffi::c_int = 1;
pub const FUNC_ATTRIBUTE_CONST_SIZE_BYTES: std::ffi::c_int = 2;
pub const FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: std::ffi::c_int = 3;
pub const FUNC_ATTRIBUTE_NUM_REGS: std::ffi::c_int = 4;
pub const FUNC_ATTRIBUTE_PTX_VERSION: std::ffi::c_int = 5;
pub const FUNC_ATTRIBUTE_BINARY_VERSION: std::ffi::c_int = 6;
pub const FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: std::ffi::c_int = 8;
pub const FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: std::ffi::c_int = 14;

pub const FUNC_CACHE_PREFER_SHARED: std::ffi::c_int = 1;

// CUlaunchAttributeID's values
pub const LAUNCH_ATTRIBUTE_COOPERATIVE: std::ffi::c_uint = 2;
pub const LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: std::ffi::c_uint = 4;
pub const LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: std::ffi::c_uint = 5;
pub const LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: std::ffi::c_uint = 6;
pub const LAUNCH_ATTRIBUTE_PRIORITY: std::ffi::c_uint = 8;

pub const CLUSTER_SCHEDULING_SPREAD: std::ffi::c_int = 1;

/// CUjit_option's input/output log buffers.
pub const JIT_INFO_LOG_BUFFER: std::ffi::c_int = 3;
pub const JIT_INFO_LOG_BUFFER_SIZE: std::ffi::c_int = 4;
pub const JIT_ERROR_LOG_BUFFER: std::ffi::c_int = 5;
pub const JIT_ERROR_LOG_BUFFER_SIZE: std::ffi::c_int = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dim3 {
    pub x: std::ffi::c_uint,
    pub y: std::ffi::c_uint,
    pub z: std::ffi::c_uint,
}

impl Dim3 {
    pub const DEFAULT: Dim3 = Dim3 { x: 1, y: 1, z: 1 };
}

/// CUlaunchAttributeValue: a 64-byte union aligned for its pointer members.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub union LaunchAttributeValue {
    pub pad: [u8; 64],
    pub align8: *mut core::ffi::c_void,
    pub int: std::ffi::c_int,
    pub cluster_dim: [std::ffi::c_uint; 3],
}

impl Default for LaunchAttributeValue {
    fn default() -> Self {
        LaunchAttributeValue { pad: [0; 64] }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LaunchAttribute {
    pub id: std::ffi::c_uint,
    /// The ABI's own padding to the 8-byte-aligned value.
    pub pad: [u8; 4],
    pub value: LaunchAttributeValue,
}

impl LaunchAttribute {
    pub fn new(id: std::ffi::c_uint) -> LaunchAttribute {
        LaunchAttribute {
            id,
            pad: [0; 4],
            value: LaunchAttributeValue::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    pub grid_x: std::ffi::c_uint,
    pub grid_y: std::ffi::c_uint,
    pub grid_z: std::ffi::c_uint,
    pub block_x: std::ffi::c_uint,
    pub block_y: std::ffi::c_uint,
    pub block_z: std::ffi::c_uint,
    pub shared_bytes: std::ffi::c_uint,
    pub stream: Stream,
    pub attrs: *const LaunchAttribute,
    pub num_attrs: std::ffi::c_uint,
}

/// CUDA_KERNEL_NODE_PARAMS_v2.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelNodeParams {
    pub func: Function,
    pub grid_x: std::ffi::c_uint,
    pub grid_y: std::ffi::c_uint,
    pub grid_z: std::ffi::c_uint,
    pub block_x: std::ffi::c_uint,
    pub block_y: std::ffi::c_uint,
    pub block_z: std::ffi::c_uint,
    pub shared_bytes: std::ffi::c_uint,
    pub params: *const *mut core::ffi::c_void,
    pub extra: *const *mut core::ffi::c_void,
    pub kern: Kernel,
    pub ctx: Context,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExecUpdateResultInfo {
    pub result: std::ffi::c_uint,
    pub error_node: GraphNode,
    pub error_from_node: GraphNode,
}

/// The driver's entry points, each field the exact exported symbol
/// (upstream: abi.zig:143's ``Api``). The macro is upstream's comptime
/// field loop: one table declares the type and the loader that resolves
/// every field by its own name.
macro_rules! cuda_api {
    ($($name:ident: unsafe extern "C" fn($($arg:ty),*) -> $ret:ty,)*) => {
        #[allow(clippy::type_complexity)]
        #[repr(C)]
        pub struct Api {
            $(pub $name: std::option::Option<unsafe extern "C" fn($($arg),*) -> $ret>,)*
        }

        impl Api {
            /// Every entry point absent: the test stub's shape (a driver
            /// whose spans never call it).
            #[cfg(test)]
            pub fn none() -> Api {
                Api { $($name: std::option::Option::None,)* }
            }

            /// Resolves every field by its exact name; a missing symbol
            /// refuses the whole driver (upstream: driver.zig's field loop).
            /// # Safety
            ///
            /// Each symbol must come from a real CUDA driver library
            /// opened by the caller — the resolved table calls into it.
            pub unsafe fn resolve(
                lookup: &dyn Fn(&str) -> Option<*mut core::ffi::c_void>,
            ) -> std::result::Result<Api, crate::driver::Error> {
                $(let $name = match lookup(stringify!($name)) {
                    std::option::Option::Some(symbol) => std::option::Option::Some(
                        // SAFETY: the symbol comes from the opened driver
                        // library, its address the ABI's own entry point.
                        unsafe {
                            std::mem::transmute::<
                                *mut core::ffi::c_void,
                                unsafe extern "C" fn($($arg),*) -> $ret,
                            >(symbol)
                        },
                    ),
                    std::option::Option::None => {
                        return std::result::Result::Err(
                            crate::driver::Error::MissingSymbol(stringify!($name).into()),
                        )
                    }
                };)*
                std::result::Result::Ok(Api { $($name,)* })
            }
        }
    };
}

cuda_api! {
    cuInit: unsafe extern "C" fn(std::ffi::c_uint) -> Result,
    cuDriverGetVersion: unsafe extern "C" fn(*mut std::ffi::c_int) -> Result,
    cuDeviceGet: unsafe extern "C" fn(*mut Device, std::ffi::c_int) -> Result,
    cuDeviceGetCount: unsafe extern "C" fn(*mut std::ffi::c_int) -> Result,
    cuDeviceGetName: unsafe extern "C" fn(*mut u8, std::ffi::c_int, Device) -> Result,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut std::ffi::c_int, std::ffi::c_int, Device) -> Result,
    cuDeviceTotalMem_v2: unsafe extern "C" fn(*mut usize, Device) -> Result,
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut Context, Device) -> Result,
    cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(Device) -> Result,
    cuCtxSetCurrent: unsafe extern "C" fn(Context) -> Result,
    cuCtxGetCurrent: unsafe extern "C" fn(*mut Context) -> Result,
    cuCtxSynchronize: unsafe extern "C" fn() -> Result,
    cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> Result,
    cuMemAlloc_v2: unsafe extern "C" fn(*mut DevicePtr, usize) -> Result,
    cuMemFree_v2: unsafe extern "C" fn(DevicePtr) -> Result,
    cuMemHostAlloc: unsafe extern "C" fn(*mut *mut core::ffi::c_void, usize, std::ffi::c_uint) -> Result,
    cuMemHostGetDevicePointer_v2: unsafe extern "C" fn(*mut DevicePtr, *mut core::ffi::c_void, std::ffi::c_uint) -> Result,
    cuMemFreeHost: unsafe extern "C" fn(*mut core::ffi::c_void) -> Result,
    cuMemHostRegister_v2: unsafe extern "C" fn(*mut core::ffi::c_void, usize, std::ffi::c_uint) -> Result,
    cuMemHostUnregister: unsafe extern "C" fn(*mut core::ffi::c_void) -> Result,
    cuMemcpyHtoD_v2: unsafe extern "C" fn(DevicePtr, *const core::ffi::c_void, usize) -> Result,
    cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut core::ffi::c_void, DevicePtr, usize) -> Result,
    cuMemcpyDtoD_v2: unsafe extern "C" fn(DevicePtr, DevicePtr, usize) -> Result,
    cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(DevicePtr, *const core::ffi::c_void, usize, Stream) -> Result,
    cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(*mut core::ffi::c_void, DevicePtr, usize, Stream) -> Result,
    cuMemcpyDtoDAsync_v2: unsafe extern "C" fn(DevicePtr, DevicePtr, usize, Stream) -> Result,
    cuMemsetD8_v2: unsafe extern "C" fn(DevicePtr, u8, usize) -> Result,
    cuMemsetD32_v2: unsafe extern "C" fn(DevicePtr, std::ffi::c_uint, usize) -> Result,
    cuMemsetD8Async: unsafe extern "C" fn(DevicePtr, u8, usize, Stream) -> Result,
    cuMemsetD32Async: unsafe extern "C" fn(DevicePtr, std::ffi::c_uint, usize, Stream) -> Result,
    cuStreamCreate: unsafe extern "C" fn(*mut Stream, std::ffi::c_uint) -> Result,
    cuStreamDestroy_v2: unsafe extern "C" fn(Stream) -> Result,
    cuStreamSynchronize: unsafe extern "C" fn(Stream) -> Result,
    cuStreamWaitEvent: unsafe extern "C" fn(Stream, Event, std::ffi::c_uint) -> Result,
    cuStreamQuery: unsafe extern "C" fn(Stream) -> Result,
    cuEventCreate: unsafe extern "C" fn(*mut Event, std::ffi::c_uint) -> Result,
    cuEventDestroy_v2: unsafe extern "C" fn(Event) -> Result,
    cuEventRecord: unsafe extern "C" fn(Event, Stream) -> Result,
    cuEventSynchronize: unsafe extern "C" fn(Event) -> Result,
    cuEventQuery: unsafe extern "C" fn(Event) -> Result,
    cuEventElapsedTime: unsafe extern "C" fn(*mut f32, Event, Event) -> Result,
    cuModuleLoadData: unsafe extern "C" fn(*mut Module, *const core::ffi::c_void) -> Result,
    cuModuleLoadDataEx: unsafe extern "C" fn(*mut Module, *const core::ffi::c_void, std::ffi::c_uint, *mut std::ffi::c_int, *mut *mut core::ffi::c_void) -> Result,
    cuModuleUnload: unsafe extern "C" fn(Module) -> Result,
    cuModuleGetFunction: unsafe extern "C" fn(*mut Function, Module, *const std::ffi::c_char) -> Result,
    cuModuleGetGlobal_v2: unsafe extern "C" fn(*mut DevicePtr, *mut usize, Module, *const std::ffi::c_char) -> Result,
    cuFuncGetAttribute: unsafe extern "C" fn(*mut std::ffi::c_int, std::ffi::c_int, Function) -> Result,
    cuFuncSetAttribute: unsafe extern "C" fn(Function, std::ffi::c_int, std::ffi::c_int) -> Result,
    cuFuncSetCacheConfig: unsafe extern "C" fn(Function, std::ffi::c_int) -> Result,
    cuOccupancyMaxActiveBlocksPerMultiprocessor: unsafe extern "C" fn(*mut std::ffi::c_int, Function, std::ffi::c_int, usize) -> Result,
    cuLaunchKernel: unsafe extern "C" fn(Function, std::ffi::c_uint, std::ffi::c_uint, std::ffi::c_uint, std::ffi::c_uint, std::ffi::c_uint, std::ffi::c_uint, std::ffi::c_uint, Stream, *const *mut core::ffi::c_void, *const *mut core::ffi::c_void) -> Result,
    cuLaunchKernelEx: unsafe extern "C" fn(*const LaunchConfig, Function, *const *mut core::ffi::c_void, *const *mut core::ffi::c_void) -> Result,
    cuStreamBeginCapture_v2: unsafe extern "C" fn(Stream, std::ffi::c_int) -> Result,
    cuStreamEndCapture: unsafe extern "C" fn(Stream, *mut Graph) -> Result,
    cuStreamIsCapturing: unsafe extern "C" fn(Stream, *mut std::ffi::c_int) -> Result,
    cuGraphCreate: unsafe extern "C" fn(*mut Graph, std::ffi::c_uint) -> Result,
    cuGraphDestroy: unsafe extern "C" fn(Graph) -> Result,
    cuGraphAddKernelNode_v2: unsafe extern "C" fn(*mut GraphNode, Graph, *const GraphNode, usize, *const KernelNodeParams) -> Result,
    cuGraphKernelNodeGetParams_v2: unsafe extern "C" fn(GraphNode, *mut KernelNodeParams) -> Result,
    cuGraphKernelNodeSetParams_v2: unsafe extern "C" fn(GraphNode, *const KernelNodeParams) -> Result,
    cuGraphAddDependencies: unsafe extern "C" fn(Graph, *const GraphNode, *const GraphNode, usize) -> Result,
    cuGraphGetNodes: unsafe extern "C" fn(Graph, *mut GraphNode, *mut usize) -> Result,
    cuGraphInstantiateWithFlags: unsafe extern "C" fn(*mut GraphExec, Graph, u64) -> Result,
    cuGraphUpload: unsafe extern "C" fn(GraphExec, Stream) -> Result,
    cuGraphLaunch: unsafe extern "C" fn(GraphExec, Stream) -> Result,
    cuGraphExecDestroy: unsafe extern "C" fn(GraphExec) -> Result,
    cuGraphExecUpdate_v2: unsafe extern "C" fn(GraphExec, Graph, *mut ExecUpdateResultInfo) -> Result,
    cuGraphExecKernelNodeSetParams_v2: unsafe extern "C" fn(GraphExec, GraphNode, *const KernelNodeParams) -> Result,
    cuGetErrorName: unsafe extern "C" fn(Result, *mut *const std::ffi::c_char) -> Result,
    cuGetErrorString: unsafe extern "C" fn(Result, *mut *const std::ffi::c_char) -> Result,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's comptime assertions (abi.zig:216-221), as a test: the
    /// structs at their documented sizes — the launch ABI's own shapes.
    #[test]
    fn the_launch_structs_at_their_documented_sizes() {
        assert_eq!(std::mem::size_of::<LaunchAttribute>(), 72);
        assert_eq!(
            std::mem::offset_of!(LaunchAttribute, value),
            8,
            "the value at its 8-byte alignment"
        );
        assert_eq!(std::mem::size_of::<LaunchConfig>(), 56);
        assert_eq!(std::mem::offset_of!(LaunchConfig, stream), 32);
        assert_eq!(std::mem::size_of::<KernelNodeParams>(), 72);
        assert_eq!(std::mem::offset_of!(KernelNodeParams, params), 40);
        assert_eq!(std::mem::size_of::<ExecUpdateResultInfo>(), 24);
        // The attribute value's union padding: the full 64 bytes
        assert_eq!(std::mem::size_of::<LaunchAttributeValue>(), 64);
    }
}
