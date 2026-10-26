//! Our .cu kernels as fatbins built at build time with each Python
//! extension's nvcc flags, embedded in the binary.
//!
//! Port of `zig/src/cuda/kernels.zig` (71 lines). The images come from
//! the crate's build script — the port of `zig/build/cuda.zig`'s
//! pipeline: `TENSORFOLD_NVCC` compiles them (`-Dnvcc`),
//! `TENSORFOLD_FATBINS` embeds prebuilt ones (`-Dfatbins`), and
//! neither builds host-only — `available` false and every image empty,
//! exactly upstream's `options.with_kernels` (cuda.zig:61).
//!
//! The blobs are aligned as upstream aligns its embedded bytes
//! (`align(16)`, kernels.zig:7) because `Module::load` reads the image
//! at its alignment — `include_bytes!` gives no alignment guarantee,
//! so each image sits in an aligned `Fatbin` and the callers pass
//! `&image.bytes`.

#![allow(non_upper_case_globals)] // the blob names are upstream's own

/// False in host-only builds (no nvcc, no prebuilt fatbins); every
/// image is then empty. Upstream: `available` (kernels.zig:13) — the
/// build's `with_kernels`.
pub const available: bool = cfg!(kernels);

/// An embedded fatbin image at upstream's alignment. Upstream: the
/// `align(16)` blob wrapper (kernels.zig:6). The generic field lets the
/// literal build as the sized `Fatbin<[u8; N]>` and the reference
/// unsize to the slice-tailed form.
#[repr(align(16))]
pub struct Fatbin<B: ?Sized = [u8]> {
    pub bytes: B,
}

#[cfg(kernels)]
macro_rules! fatbin {
    ($file:literal) => {
        &Fatbin {
            bytes: *include_bytes!(concat!(env!("OUT_DIR"), "/", $file)),
        }
    };
}

#[cfg(not(kernels))]
macro_rules! fatbin {
    ($file:literal) => {
        &Fatbin { bytes: [] }
    };
}

/// The fatbin images: empty in host-only builds (upstream: the
/// `if (available)` blobs, kernels.zig:15-38).
pub static gdn: &Fatbin = fatbin!("fatbin_gdn");
pub static probe: &Fatbin = fatbin!("fatbin_probe");
pub static qmm_group: &Fatbin = fatbin!("fatbin_qmm_group");
pub static qmm_prefill: &Fatbin = fatbin!("fatbin_qmm_prefill");
pub static experts: &Fatbin = fatbin!("fatbin_experts");
pub static experts_prefill: &Fatbin = fatbin!("fatbin_experts_prefill");
pub static experts_pack: &Fatbin = fatbin!("fatbin_experts_pack");
pub static prefill_attention: &Fatbin = fatbin!("fatbin_prefill_attention");
pub static scan_rows: &Fatbin = fatbin!("fatbin_scan_rows");
pub static nemotron_ops: &Fatbin = fatbin!("fatbin_nemotron_ops");
pub static lane_gemv: &Fatbin = fatbin!("fatbin_lane_gemv");
pub static sample: &Fatbin = fatbin!("fatbin_sample");
pub static fp8_lane: &Fatbin = fatbin!("fatbin_fp8_lane");
pub static torch_argmax: &Fatbin = fatbin!("fatbin_torch_argmax");
pub static torch_topk: &Fatbin = fatbin!("fatbin_torch_topk");
pub static torch_pointwise: &Fatbin = fatbin!("fatbin_torch_pointwise");
pub static torch_indexing: &Fatbin = fatbin!("fatbin_torch_indexing");
pub static torch_movement: &Fatbin = fatbin!("fatbin_torch_movement");
pub static torch_nemotron_constants: &Fatbin = fatbin!("fatbin_torch_nemotron_constants");
pub static nemotron_norms: &Fatbin = fatbin!("fatbin_nemotron_norms");
pub static nemotron_route: &Fatbin = fatbin!("fatbin_nemotron_route");
pub static nemotron_mamba: &Fatbin = fatbin!("fatbin_nemotron_mamba");
pub static nemotron_attention: &Fatbin = fatbin!("fatbin_nemotron_attention");
pub static nemotron_keyed: &Fatbin = fatbin!("fatbin_nemotron_keyed");

/// Symbols in the gdn image as cuobjdump lists them for the built
/// fatbin (named namespace tf_gdn). Upstream: `gdn_symbols`
/// (kernels.zig:41).
pub mod gdn_symbols {
    pub const replay_bf16: &str =
        "_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxiPKiiS5_iPfiii";
    pub const replay_f32: &str = "_ZN6tf_gdn13replay_kernelIfLi8ELi4EEEvPKxiPKiiS4_iPfiii";
}

/// One gdn tree_kernel instantiation. Upstream: `TreeVariant`
/// (kernels.zig:52).
#[derive(Clone, Copy)]
pub struct TreeVariant {
    pub slots: u32,
    pub r: u32,
    pub warps: u32,
    pub chain: bool,
    pub symbol: &'static str,
}

/// Every instantiation gdn.cu exports; tests resolve each one so a
/// wrong name cannot hide behind an unused path. Upstream:
/// `gdn_variants` (kernels.zig:47) — the comptime `tree` names spelled
/// out.
pub const GDN_VARIANTS: [TreeVariant; 8] = [
    TreeVariant { slots: 0, r: 8, warps: 4, chain: true, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li0ELi8ELi4ELb1EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 1, r: 8, warps: 4, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li1ELi8ELi4ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 2, r: 8, warps: 2, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li2ELi8ELi2ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 2, r: 4, warps: 4, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li2ELi4ELi4ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 4, r: 2, warps: 4, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li4ELi2ELi4ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 8, r: 2, warps: 4, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li8ELi2ELi4ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 16, r: 2, warps: 2, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li16ELi2ELi2ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
    TreeVariant { slots: 32, r: 2, warps: 1, chain: false, symbol: "_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li32ELi2ELi1ELb0EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b" },
];

/// The tree_kernel instantiation gdn.cu's dispatch_tree launches for
/// bf16 keys on this GPU. Upstream: `treeVariant` (kernels.zig:61).
pub fn tree_variant(slots: u32, streams: u32, major: i32, minor: i32, sms: i32) -> TreeVariant {
    let wide = streams >= 2 && major == 12 && minor == 0 && sms >= 96;
    if slots == 0 {
        return GDN_VARIANTS[0];
    }
    if wide && slots <= 1 {
        return GDN_VARIANTS[1];
    }
    if wide && slots <= 2 {
        return GDN_VARIANTS[2];
    }
    if slots <= 2 {
        return GDN_VARIANTS[3];
    }
    if slots <= 4 {
        return GDN_VARIANTS[4];
    }
    if slots <= 8 {
        return GDN_VARIANTS[5];
    }
    if slots <= 16 {
        return GDN_VARIANTS[6];
    }
    GDN_VARIANTS[7]
}
