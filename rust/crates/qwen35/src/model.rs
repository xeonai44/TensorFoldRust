//! A native Qwen3.5-2B model: validated config, shared affine checkpoint views and
//! Metal pipelines.
//!
//! Port of `zig/src/families/qwen3_5/model.zig`. Load order mirrors upstream's
//! `Model.load`: the config gate, the device and queue, the Metal-resident
//! checkpoint (only the `language_model.` namespace — upstream `addFileSelected`
//! with that selection, so a vision tower never loads), the weight views, and the
//! run-time kernel compile.

use crate::config::Config;
use crate::kernels::Kernels;
use crate::weights;
use std::path::Path;
use std::sync::Arc;
use tensorfold_core::checkpoint;
use tensorfold_metal::checkpoint::Checkpoint;
use tensorfold_metal::Gpu;

/// Upstream: model.zig:10  pub const Model.
pub struct Model {
    pub gpu: Arc<Gpu>,
    pub config: Config,
    pub checkpoint: Checkpoint,
    pub weights: weights::Weights,
    pub kernels: Kernels,
}

impl Model {
    /// Upstream: `load` (model.zig:19).
    pub fn load(dir: &Path) -> Result<Model, crate::Error> {
        let config = Config::read(dir)?;
        let gpu = Arc::new(Gpu::open().ok_or(crate::Error::NoMetalDevice)?);
        let mut checkpoint = Checkpoint::new();
        // Upstream: shards.shardFiles then addFileSelected(device, path, "",
        // "language_model.") for each — the checkpoint's own shard list (index map
        // else the model*.safetensors glob), name order, with the missing/leftover
        // refusals that list carries.
        let shard_paths =
            checkpoint::shard_files(dir).map_err(|_| crate::Error::BadQwenConfig)?;
        for path in &shard_paths {
            // The selected namespace's refusals (missing tensors, duplicates, short
            // reads) are all UnexpectedQwenTensor here, the way upstream's
            // Model.load funnels its checkpoint errors into one error set.
            checkpoint
                .add_file_selected(&gpu, path, "", Some("language_model."))
                .map_err(|_| crate::Error::UnexpectedQwenTensor)?;
        }
        let weights = weights::load(&checkpoint, config.tied)?;
        let kernels = Kernels::load(gpu.device()).map_err(crate::Error::QwenKernel)?;
        Ok(Model {
            gpu,
            config,
            checkpoint,
            weights,
            kernels,
        })
    }

    /// Upstream: `residentBytes` is read through the checkpoint; expose the same
    /// number for the startup line (Phase 1b's engine).
    pub fn resident_bytes(&self) -> usize {
        self.checkpoint.resident_bytes()
    }
}
