//! TensorFold's CUDA runtime: the driver, cuBLASLt and NCCL through
//! dlopen, our kernels and Triton's, no Python.
//!
//! Port of `zig/src/cuda/root.zig` (42 lines).

#![allow(non_snake_case)]
pub mod abi;
pub mod aot;
pub mod carveout;
pub mod context;
pub mod cublaslt;
pub mod driver;
pub mod fp8;
pub mod graph;
pub mod grouped;
pub mod kernels;
pub mod launch;
pub mod memory;
pub mod module;
pub mod nccl;
pub mod qlinear;
pub mod segments;
pub mod stream;
pub mod triton;

pub use context::Context;
pub use carveout::Carveout;
pub use carveout::CarveoutError;
pub use driver::{Driver, Error};
pub use graph::{begin_capture, capture_status, end_capture, Exec, Graph, Node};
pub use launch::{launch, Args, Config};
pub use memory::{usage, DeviceBuffer, HostBuffer, Usage};
pub use module::{Function, Module};
pub use stream::{Event, Stream};
pub use triton::{Kernel, Meta, Scratch};

// Dim3 is the ABI's own (launch re-exports it)
pub use abi::Dim3;
