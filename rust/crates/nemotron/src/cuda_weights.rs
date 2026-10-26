//! Nemotron-H weights on the GPU in the Python engine's layouts:
//! tiled projections, packed expert blocks, fp32 Mamba.
//!
//! Port of `zig/src/families/nemotron/cuda_weights.zig` (445 lines).
//! The port's loader carries its own 2-d tensor view (`Part`) where
//! upstream aliases core.checkpoint.Tensor for both mapped tensors and
//! host-made ones — the checkpoint's own tensors flow in and convert
//! at the reads — and a lesson's kept host buffers live in a `Held`
//! threaded beside the loader's borrows, staged as upstream's `held`
//! list is. Upstream stages every host copy through one reused buffer
//! (`host`); the port's calls carry their own — the uploads are
//! byte-identical either way.

#![allow(clippy::too_many_arguments)] // the loader's calls carry the kernels' parameters, upstream's own signatures

use tensorfold_core::affine4_host as host4;
use tensorfold_core::checkpoint::Checkpoint;
use tensorfold_cuda::{launch, Args, Config as LaunchConfig, DeviceBuffer, Dim3};
use tensorfold_safetensors::{DType, Tensor as Mapped};

use crate::config::{Config, Kind};
use crate::cuda_kernels::{self, Ops};
pub use crate::cuda_kernels::QLinear;
use crate::cuda_source::Source;
use crate::draft_ids;
use crate::error::Error;

pub const MTP_FILE: &str = "mtp-4bit.safetensors";

/// The token table as stored (MLX words, scales, biases): a row
/// lookup, not a matmul. Upstream: `Embed` (cuda_weights.zig:20).
#[derive(Clone, Copy)]
pub struct Embed {
    pub w: u64,
    pub s: u64,
    pub b: u64,
    pub n: usize,
    pub k: usize,
}

/// A `*_rest` is a lesson's change past its projection's 4-bit codes
/// ([out, in] bf16), added after it; 0 when none. Upstream: `Mamba`
/// (cuda_weights.zig:22).
#[derive(Clone, Copy)]
pub struct Mamba {
    pub in_proj: QLinear,
    pub out_proj: QLinear,
    pub conv_w: u64,
    pub conv_b: u64,
    pub a: u64,
    pub d: u64,
    pub dt_bias: u64,
    pub gnorm: u64,
    pub out_rest: u64,
}

/// Upstream: `Attention` (cuda_weights.zig:23) — the shared expert's
/// down projection rest in its two halves, as experts E and E + 1
/// split its columns.
#[derive(Clone, Copy)]
pub struct Attention {
    pub qkv: QLinear,
    pub o: QLinear,
    pub o_rest: u64,
}

/// Upstream: `MoE` (cuda_weights.zig:25).
#[derive(Clone, Copy)]
pub struct Moe {
    pub router: u64,
    pub bias: u64,
    pub experts: cuda_kernels::Experts,
    pub rest: [u64; 2],
}

/// Upstream: `Block` (cuda_weights.zig:26) — the one mixer the kind
/// names is set; the others stay None (Zig leaves them undefined,
/// never read).
#[derive(Clone, Copy)]
pub struct Block {
    pub kind: Kind,
    pub norm: u64,
    pub mamba: Option<Mamba>,
    pub attn: Option<Attention>,
    pub moe: Option<Moe>,
}

/// Upstream: `Mtp` (cuda_weights.zig:27).
#[derive(Clone, Copy)]
pub struct Mtp {
    pub enorm: u64,
    pub hnorm: u64,
    pub eh_proj: QLinear,
    pub attn_norm: u64,
    pub attn: Attention,
    pub moe_norm: u64,
    pub moe: Moe,
    pub final_norm: u64,
}

/// A loaded tensor by the Python engine's dotted name, for digests
/// against its weights. Upstream: `Named` (cuda_weights.zig:30).
pub struct Named {
    pub name: String,
    pub ptr: u64,
    pub len: usize,
}

/// Upstream: `Weights` (cuda_weights.zig:32) — the port's Drop frees
/// the buffers; the names' strings with them.
pub struct Weights<'d> {
    pub config: Config,
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm_f: u64,
    pub head: QLinear,
    pub mtp: Option<Mtp>,
    pub draft_head: Option<QLinear>,
    pub draft_ids: u64,
    pub draft_count: usize,
    pub buffers: Vec<DeviceBuffer<'d>>,
    pub named: Vec<Named>,
}

impl Weights<'_> {
    /// Upstream: `deinit` (cuda_weights.zig:46) — the port's Drop.
    pub fn deinit(&mut self) {
        self.buffers.clear();
        self.named.clear();
        self.blocks.clear();
        self.mtp = None;
        self.draft_head = None;
    }
}

impl Drop for Weights<'_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// The loader's 2-d tensor view over any bytes — upstream aliases
/// core.checkpoint.Tensor for both mapped tensors and host-made ones;
/// the port keeps the checkpoint's own for mapped reads and this
/// inline form where the loader makes parts.
#[derive(Clone, Copy)]
pub struct Part<'a> {
    pub dtype: DType,
    pub d0: usize,
    pub d1: usize,
    pub bytes: &'a [u8],
}

fn part_of<'t>(t: &Mapped<'t>) -> Part<'t> {
    Part {
        dtype: t.dtype,
        d0: t.dim(0),
        d1: t.dim(1),
        bytes: t.data,
    }
}

/// A 2-d tensor over host memory. Upstream: `made`
/// (cuda_weights.zig:391).
fn made(dtype: DType, d0: usize, d1: usize, bytes: &[u8]) -> Part<'_> {
    Part {
        dtype,
        d0,
        d1,
        bytes,
    }
}

/// A lesson's projection made 4-bit on the host: the parts `three`
/// would read, and the bf16 rest [n, k] they miss. Upstream: `Learned`
/// (cuda_weights.zig:383) — the rest borrows the Held that keeps it.
struct Learned<'h> {
    parts: [Part<'h>; 3],
    rest: &'h [u16],
    n: usize,
    k: usize,
}

/// Host buffers a lesson's made parts read, kept until the load's
/// uploads finish (upstream's `held` list — the port moves the slices
/// in whole; their heap data, what the parts point into, never moves).
pub struct Held {
    u32s: Vec<Box<[u32]>>,
    u16s: Vec<Box<[u16]>>,
}

impl Held {
    pub fn new() -> Held {
        Held {
            u32s: Vec::new(),
            u16s: Vec::new(),
        }
    }

    /// SAFETY (the caller's contract): the returned slice is used only
    /// while this Held stays alive — the box it reads is owned here
    /// and its data never moves, so `held`'s own borrow may end and
    /// start again around it.
    fn hold_u32<'h>(&mut self, v: Box<[u32]>) -> &'h [u32] {
        let (ptr, len) = (v.as_ptr(), v.len());
        self.u32s.push(v);
        // SAFETY: the Held owns the box for as long as the contract
        // keeps it alive.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// The same contract as `hold_u32`.
    fn hold_u16<'h>(&mut self, v: Box<[u16]>) -> &'h [u16] {
        let (ptr, len) = (v.as_ptr(), v.len());
        self.u16s.push(v);
        // SAFETY: the Held owns the box for as long as the contract
        // keeps it alive.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

}

impl Default for Held {
    fn default() -> Held {
        Held::new()
    }
}

/// Upstream: `Loader` (cuda_weights.zig:56).
pub struct Loader<'a, 'd, 'w, 's> {
    pub ops: &'a Ops<'a, 'd>,
    pub w: &'w mut Weights<'d>,
    pub src: &'s mut Source<'a, 'd>,
    scratch: DeviceBuffer<'d>,
    pub learned: usize,
}

impl<'a, 'd> Loader<'a, 'd, '_, '_> {
    /// Upstream: `alloc` (cuda_weights.zig:66).
    fn alloc(&mut self, name: &str, bytes: usize) -> Result<u64, Error> {
        let b = DeviceBuffer::alloc(self.ops.k.d, bytes)?;
        let ptr = b.ptr;
        self.w.buffers.push(b);
        self.w.named.push(Named {
            name: name.to_string(),
            ptr,
            len: bytes,
        });
        Ok(ptr)
    }

    /// Device scratch of at least `bytes`; growing waits for the stream
    /// so pending packs keep their inputs. Upstream: `tmp`
    /// (cuda_weights.zig:74).
    fn tmp(&mut self, bytes: usize) -> Result<u64, Error> {
        if self.scratch.len < bytes {
            self.src.flush()?;
            self.ops.s.synchronize()?;
            self.scratch = DeviceBuffer::alloc(self.ops.k.d, bytes)?;
        }
        Ok(self.scratch.ptr)
    }

    /// Upstream: `raw` (cuda_weights.zig:92).
    fn raw(&mut self, name: &str, t: &Mapped<'_>) -> Result<u64, Error> {
        let ptr = self.alloc(name, t.data.len())?;
        self.src.upload(ptr, t.data)?;
        Ok(ptr)
    }

    /// `.float()` of a bf16 tensor, exact: each bf16's bits in the
    /// high half of a u32. Upstream: `widened` (cuda_weights.zig:99) —
    /// staged into the reused host buffer; the port returns the owned
    /// words.
    fn widened(&mut self, t: &Mapped<'_>) -> Result<Vec<u32>, Error> {
        if t.dtype != DType::BF16 {
            return Err(Error::UnexpectedTensor);
        }
        let n = t.data.len() / 2;
        let mut words = vec![0u32; n];
        let input = self.src.view(t.data)?;
        for (i, o) in words.iter_mut().enumerate() {
            let bits = input[2 * i] as u32 | ((input[2 * i + 1] as u32) << 8);
            *o = bits << 16;
        }
        Ok(words)
    }

    /// Upstream: `widen` (cuda_weights.zig:108).
    fn widen(&mut self, name: &str, t: &Mapped<'_>) -> Result<u64, Error> {
        let host = self.widened(t)?;
        let bytes = words_bytes(&host);
        let ptr = self.alloc(name, bytes.len())?;
        self.ops.upload(ptr, bytes)?;
        Ok(ptr)
    }

    /// qmm_fast.tile of row-stacked parts (one projection, or q, k, v
    /// as attention stacks them). Upstream: `tile` (cuda_weights.zig:116).
    fn tile(&mut self, name: &str, parts: &[[Part<'_>; 3]]) -> Result<QLinear, Error> {
        let mut n: usize = 0;
        let k8 = parts[0][0].d1;
        for pt in parts {
            if pt[0].dtype != DType::U32
                || pt[1].dtype != DType::BF16
                || pt[2].dtype != DType::BF16
            {
                return Err(Error::UnsupportedQuantization);
            }
            if pt[0].d1 != k8 || pt[1].d1 * 8 != k8 {
                return Err(Error::UnexpectedTensor);
            }
            n += pt[0].d0;
        }
        let k = k8 * 8;
        let kg = k / 64;
        let npad = n.div_ceil(128) * 128;
        let sizes = [n * k8 * 4, n * kg * 2, n * kg * 2];
        let base = self.tmp(sizes[0] + sizes[1] + sizes[2])?;
        let mut at = [0, sizes[0], sizes[0] + sizes[1]];
        for pt in parts {
            for j in 0..3 {
                self.src.upload(base + at[j] as u64, pt[j].bytes)?;
                at[j] += pt[j].bytes.len();
            }
        }
        let q = QLinear {
            w: self.alloc(&format!("{name}.weight"), npad * k / 2)?,
            s: self.alloc(&format!("{name}.scales"), kg * npad * 2)?,
            b: self.alloc(&format!("{name}.biases"), kg * npad * 2)?,
            n,
            k,
            npad,
        };
        let total: i64 = (npad / 64 * kg * 512) as i64;
        self.src.flush()?;
        let mut a = Args::new();
        a.add(base);
        for v in [n, k / 8, kg] {
            a.add(v as std::ffi::c_int);
        }
        a.add(q.w);
        a.add(total);
        launch(
            &self.ops.k.pack_dense,
            LaunchConfig {
                grid: Dim3 {
                    x: (total as usize).div_ceil(256) as u32,
                    y: 1,
                    z: 1,
                },
                block: Dim3 { x: 256, y: 1, z: 1 },
                shared: 0,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            self.ops.s,
            &mut a,
        )?;
        for (out, off) in [(q.s, sizes[0]), (q.b, sizes[0] + sizes[1])] {
            let mut t = Args::new();
            t.add(base + off as u64);
            for v in [n, kg, npad] {
                t.add(v as std::ffi::c_int);
            }
            t.add(out);
            let cells = kg * npad;
            launch(
                &self.ops.k.transpose16,
                LaunchConfig {
                    grid: Dim3 {
                        x: cells.div_ceil(256) as u32,
                        y: 1,
                        z: 1,
                    },
                    block: Dim3 { x: 256, y: 1, z: 1 },
                    shared: 0,
                    cluster: None,
                    cluster_spread: false,
                    pdl: false,
                    cooperative: false,
                },
                self.ops.s,
                &mut t,
            )?;
        }
        Ok(q)
    }

    /// A projection a lesson changed (bf16 in the shards) as 4-bit
    /// words, scales and biases, and the rest they miss. Upstream:
    /// `lesson` (cuda_weights.zig:173).
    fn lesson<'h>(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        pre: &str,
        mid: &str,
    ) -> Result<Option<Learned<'h>>, Error> {
        // a missing name is an error (upstream's `try`); only a
        // non-bf16 weight means no lesson
        let w = ck.get(&format!("{pre}{mid}.weight"))?;
        if w.dtype != DType::BF16 {
            return Ok(None);
        }
        if w.rank() != 2 || w.dim(1) % host4::GROUP != 0 {
            return Err(Error::UnexpectedTensor);
        }
        let n = w.dim(0);
        let k = w.dim(1);
        let count = n * k;
        let mut values = vec![0.0f32; count];
        {
            let input = self.src.view(w.data)?;
            for (i, v) in values.iter_mut().enumerate() {
                let bits = input[2 * i] as u32 | ((input[2 * i + 1] as u32) << 8);
                *v = host4::f32_of(bits as u16);
            }
        }
        let mut words = vec![0u32; count / 8].into_boxed_slice();
        let mut scales = vec![0u16; count / host4::GROUP].into_boxed_slice();
        let mut biases = vec![0u16; count / host4::GROUP].into_boxed_slice();
        host4::quantize(&values, &mut words, &mut scales, &mut biases);
        let mut back = vec![0.0f32; count];
        host4::dequantize(&words, &scales, &biases, &mut back);
        let mut left = vec![0u16; count].into_boxed_slice();
        for (r, (v, q)) in left.iter_mut().zip(values.iter().zip(back.iter())) {
            *r = host4::bf16_of(v - q);
        }
        let words = held.hold_u32(words);
        let scales = held.hold_u16(scales);
        let biases = held.hold_u16(biases);
        let rest = held.hold_u16(left);
        self.learned += 1;
        Ok(Some(Learned {
            parts: [
                made(DType::U32, n, k / 8, words_bytes(words)),
                made(DType::BF16, n, k / host4::GROUP, halves_bytes(scales)),
                made(DType::BF16, n, k / host4::GROUP, halves_bytes(biases)),
            ],
            rest,
            n,
            k,
        }))
    }

    /// Columns [first, first + width) of a learned rest. Upstream:
    /// `rest` (cuda_weights.zig:203).
    fn rest(&mut self, name: &str, l: &Learned<'_>, first: usize, width: usize) -> Result<u64, Error> {
        let mut host = vec![0u8; l.n * width * 2];
        for r in 0..l.n {
            let dst = &mut host[r * width * 2..(r + 1) * width * 2];
            let src = &l.rest[r * l.k + first..r * l.k + first + width];
            dst.copy_from_slice(unsafe {
                // SAFETY: the rest slice's own bytes, read-only.
                std::slice::from_raw_parts(src.as_ptr().cast(), width * 2)
            });
        }
        let ptr = self.alloc(&format!("{name}.rest"), host.len())?;
        self.ops.upload(ptr, &host)?;
        Ok(ptr)
    }

    /// A one-part projection; `rest` takes a lesson's change when its
    /// weight is bf16 (else that is refused). Upstream: `dense`
    /// (cuda_weights.zig:212).
    fn dense(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        prefixes: &[&str],
    ) -> Result<QLinear, Error> {
        self.dense_rest(held, ck, name, prefixes, None)
    }

    /// Upstream: `denseRest` (cuda_weights.zig:217).
    fn dense_rest(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        prefixes: &[&str],
        rest_out: Option<&mut u64>,
    ) -> Result<QLinear, Error> {
        if let Some(out) = rest_out {
            if let Some(l) = self.lesson(held, ck, prefixes[0], "")? {
                *out = self.rest(name, &l, 0, l.k)?;
                return self.tile(name, &[l.parts]);
            }
        }
        let mut parts: Vec<[Part<'_>; 3]> = Vec::with_capacity(prefixes.len());
        for pre in prefixes {
            parts.push(three(ck, pre, "")?);
        }
        self.tile(name, &parts)
    }

    /// weights.fold_shared then experts.make: the shared expert's
    /// halves become experts E and E + 1. Upstream: `experts`
    /// (cuda_weights.zig:228).
    fn experts(
        &mut self,
        _held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        pre: &str,
        c: Config,
        shared_down: Option<[Part<'_>; 3]>,
    ) -> Result<cuda_kernels::Experts, Error> {
        let s_up = three(ck, pre, "shared_experts.up_proj")?;
        if s_up[0].d0 != 2 * c.expert_width {
            return Err(Error::SharedExpertWidth);
        }
        let e = c.experts + 2;
        let down = three(ck, pre, "switch_mlp.fc2")?;
        let shared_down = match shared_down {
            Some(p) => p,
            None => three(ck, pre, "shared_experts.down_proj")?,
        };
        let up = self.pack_experts(
            &format!("{name}.up"),
            &three(ck, pre, "switch_mlp.fc1")?,
            &s_up,
            false,
            e,
            c.expert_width,
            c.hidden,
        )?;
        let down = self.pack_experts(
            &format!("{name}.down"),
            &down,
            &shared_down,
            true,
            e,
            c.hidden,
            c.expert_width,
        )?;
        Ok(cuda_kernels::Experts {
            up,
            down,
            count: e,
            width: c.expert_width,
            dims: c.hidden,
        })
    }

    /// One expert matrix (n outputs of k inputs) with the shared halves
    /// appended; `split` halves the shared columns. Upstream:
    /// `packExperts` (cuda_weights.zig:240).
    fn pack_experts(
        &mut self,
        name: &str,
        routed: &[Part<'_>; 3],
        shared: &[Part<'_>; 3],
        split: bool,
        e: usize,
        n: usize,
        k: usize,
    ) -> Result<u64, Error> {
        let kg = k / 64;
        let sizes = [e * n * (k / 8) * 4, e * n * kg * 2, e * n * kg * 2];
        let base = self.tmp(sizes[0] + sizes[1] + sizes[2])?;
        let at = [0, sizes[0], sizes[0] + sizes[1]];
        for j in 0..3 {
            if routed[j].bytes.len() + shared[j].bytes.len() != sizes[j] {
                return Err(Error::UnexpectedTensor);
            }
            self.src.upload(base + at[j] as u64, routed[j].bytes)?;
            if !split {
                self.src.upload(base + at[j] as u64 + routed[j].bytes.len() as u64, shared[j].bytes)?;
            }
        }
        // the output's allocation runs while the reads queued above go on
        let nb = n / 32;
        let ptr = self.alloc(name, e * nb * kg * 288 * 4)?;
        if split {
            for j in 0..3 {
                let row = shared[j].d0;
                let half = row / 2;
                let mut host = vec![0u8; shared[j].bytes.len()];
                let input = self.src.view(shared[j].bytes)?;
                for h in 0..2usize {
                    for r in 0..n {
                        let dst = &mut host[(h * n + r) * half * 2..(h * n + r + 1) * half * 2];
                        dst.copy_from_slice(
                            &input[r * row * 2 + h * half * 2..r * row * 2 + (h + 1) * half * 2],
                        );
                    }
                }
                self.ops.upload(base + at[j] as u64 + routed[j].bytes.len() as u64, &host)?;
            }
        }
        self.src.flush()?;
        let mut a = Args::new();
        for v in [base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64, ptr] {
            a.add(v);
        }
        for v in [n, k / 8, kg, nb] {
            a.add(v as std::ffi::c_int);
        }
        launch(
            &self.ops.k.pack_experts,
            LaunchConfig {
                grid: Dim3 {
                    x: kg as u32,
                    y: nb as u32,
                    z: e as u32,
                },
                block: Dim3 { x: 288, y: 1, z: 1 },
                shared: 0,
                cluster: None,
                cluster_spread: false,
                pdl: false,
                cooperative: false,
            },
            self.ops.s,
            &mut a,
        )?;
        Ok(ptr)
    }

    /// Upstream: `mamba` (cuda_weights.zig:268).
    fn mamba(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        pre: &str,
    ) -> Result<Mamba, Error> {
        let mut m = Mamba {
            in_proj: zero_qlinear(),
            out_proj: zero_qlinear(),
            conv_w: 0,
            conv_b: 0,
            a: 0,
            d: 0,
            dt_bias: 0,
            gnorm: 0,
            out_rest: 0,
        };
        m.in_proj = self.dense(held, ck, &format!("{name}.in_proj"), &[&format!("{pre}in_proj")])?;
        let mut out_rest = 0u64;
        m.out_proj = self.dense_rest(
            held,
            ck,
            &format!("{name}.out_proj"),
            &[&format!("{pre}out_proj")],
            Some(&mut out_rest),
        )?;
        m.out_rest = out_rest;
        let conv = ck.get(&format!("{pre}conv1d.weight"))?;
        if conv.rank() != 3 || conv.dim(1) != 4 || conv.dim(2) != 1 || conv.dtype != DType::BF16 {
            return Err(Error::UnexpectedTensor);
        }
        let ch = conv.dim(0);
        // the taps as the kernel reads them: each bf16's bits in a u32's
        // high half, tap t of channel c at taps[t * ch + c]
        let mut taps = vec![0u32; ch * 4];
        {
            let input = self.src.view(conv.data)?;
            for c in 0..ch {
                for t in 0..4 {
                    let bits = input[(c * 4 + t) * 2] as u32
                        | ((input[(c * 4 + t) * 2 + 1] as u32) << 8);
                    taps[t * ch + c] = bits << 16;
                }
            }
        }
        let m_conv_w = self.alloc(&format!("{name}.conv_w"), taps.len() * 4)?;
        self.ops.upload(m_conv_w, words_bytes(&taps))?;
        m.conv_w = m_conv_w;
        m.conv_b = self.widen(&format!("{name}.conv_b"), &ck.get(&format!("{pre}conv1d.bias"))?)?;
        let alog = self.widened(&ck.get(&format!("{pre}A_log"))?)?;
        let src = self.tmp(alog.len() * 4)?;
        self.ops.upload(src, words_bytes(&alog))?;
        m.a = self.alloc(&format!("{name}.a"), alog.len() * 4)?;
        self.ops.torch().mamba_a(src, m.a, alog.len())?;
        m.d = self.widen(&format!("{name}.d"), &ck.get(&format!("{pre}D"))?)?;
        m.dt_bias = self.widen(&format!("{name}.dt_bias"), &ck.get(&format!("{pre}dt_bias"))?)?;
        m.gnorm = self.raw(&format!("{name}.gnorm"), &ck.get(&format!("{pre}norm.weight"))?)?;
        Ok(m)
    }

    /// Upstream: `attention` (cuda_weights.zig:290).
    fn attention(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        pre: &str,
    ) -> Result<Attention, Error> {
        let qkv = self.dense(
            held,
            ck,
            &format!("{name}.qkv"),
            &[
                &format!("{pre}q_proj"),
                &format!("{pre}k_proj"),
                &format!("{pre}v_proj"),
            ],
        )?;
        let mut o_rest = 0u64;
        let o = self.dense_rest(
            held,
            ck,
            &format!("{name}.o"),
            &[&format!("{pre}o_proj")],
            Some(&mut o_rest),
        )?;
        Ok(Attention {
            qkv,
            o,
            o_rest,
        })
    }

    /// Upstream: `moe` (cuda_weights.zig:301).
    fn moe(
        &mut self,
        held: &mut Held,
        ck: &Checkpoint,
        name: &str,
        pre: &str,
        c: Config,
    ) -> Result<Moe, Error> {
        let router = ck.expect(&format!("{pre}gate.weight"), DType::BF16, &[c.experts, c.hidden])?;
        let bias = ck.expect(
            &format!("{pre}gate.e_score_correction_bias"),
            DType::F32,
            &[c.experts],
        )?;
        let mut m = Moe {
            router: 0,
            bias: 0,
            experts: cuda_kernels::Experts {
                up: 0,
                down: 0,
                count: 0,
                width: 0,
                dims: 0,
            },
            rest: [0, 0],
        };
        m.router = self.raw(&format!("{name}.router"), &router)?;
        m.bias = self.raw(&format!("{name}.bias"), &bias)?;
        let down = self.lesson(held, ck, pre, "shared_experts.down_proj")?;
        if let Some(l) = &down {
            for (h, r) in m.rest.iter_mut().enumerate() {
                *r = self.rest(
                    &format!("{name}.shared_down{h}"),
                    l,
                    h * c.expert_width,
                    c.expert_width,
                )?;
            }
        }
        m.experts = self.experts(
            held,
            ck,
            &format!("{name}.experts"),
            pre,
            c,
            down.map(|l| l.parts),
        )?;
        Ok(m)
    }
}

fn zero_qlinear() -> QLinear {
    QLinear {
        w: 0,
        s: 0,
        b: 0,
        n: 0,
        k: 0,
        npad: 0,
    }
}

/// The words' own bytes, read-only.
fn words_bytes(w: &[u32]) -> &[u8] {
    // SAFETY: the u32 slice's own bytes, read-only.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() * 4) }
}

/// The halves' own bytes, read-only.
fn halves_bytes(h: &[u16]) -> &[u8] {
    // SAFETY: the u16 slice's own bytes, read-only.
    unsafe { std::slice::from_raw_parts(h.as_ptr().cast(), h.len() * 2) }
}

/// A quantized matrix's words, scales and biases ("{pre}{mid}.weight"
/// ...), refused unless 4-bit with bf16 scales. Upstream: `three`
/// (cuda_weights.zig:372).
fn three<'t>(ck: &'t Checkpoint, pre: &str, mid: &str) -> Result<[Part<'t>; 3], Error> {
    let mut out: [Part<'_>; 3] = [made(DType::U32, 0, 0, &[]); 3];
    for (i, s) in ["weight", "scales", "biases"].iter().enumerate() {
        let t = ck.get(&format!("{pre}{mid}.{s}"))?;
        out[i] = part_of(&t);
    }
    if out[0].dtype != DType::U32 || out[1].dtype != DType::BF16 || out[2].dtype != DType::BF16 {
        return Err(Error::UnsupportedQuantization);
    }
    Ok(out)
}

/// weights.load: the model folder's checkpoint (and its MTP head when
/// `with_mtp`) on the GPU, leftovers refused. Upstream: `load`
/// (cuda_weights.zig:323).
pub fn load<'d>(
    ops: &Ops<'_, 'd>,
    dir: &std::path::Path,
    c: Config,
    with_mtp: bool,
) -> Result<Weights<'d>, Error> {
    if c.group_size != 64 || c.bits != 4 {
        return Err(Error::UnsupportedQuantization);
    }
    let mut w = Weights {
        config: c,
        embed: Embed {
            w: 0,
            s: 0,
            b: 0,
            n: 0,
            k: 0,
        },
        blocks: Vec::new(),
        norm_f: 0,
        head: zero_qlinear(),
        mtp: None,
        draft_head: None,
        draft_ids: 0,
        draft_count: 0,
        buffers: Vec::new(),
        named: Vec::new(),
    };
    let mut src = Source::init(ops)?;
    let mut held = Held::new();
    let scratch = DeviceBuffer::alloc(ops.k.d, 1 << 20)?;
    let load = || -> Result<(), Error> {
        let mut loader = Loader {
            ops,
            w: &mut w,
            src: &mut src,
            scratch,
            learned: 0,
        };
        let ck = Checkpoint::open(dir).map_err(|_| Error::FileNotFound)?;
        loader.src.add(&ck)?;
        loader.w.blocks = Vec::with_capacity(c.layers);
        for (i, kind) in c.kinds[..c.layers].iter().copied().enumerate() {
            let name = format!("blocks.{i}");
            let norm = ck.expect(
                &format!("backbone.layers.{i}.norm.weight"),
                DType::BF16,
                &[c.hidden],
            )?;
            let mut b = Block {
                kind,
                norm: 0,
                mamba: None,
                attn: None,
                moe: None,
            };
            b.norm = loader.raw(&format!("{name}.norm"), &norm)?;
            let pre = format!("backbone.layers.{i}.mixer.");
            match kind {
                Kind::Mamba => {
                    b.mamba =
                        Some(loader.mamba(&mut held, &ck, &format!("{name}.mamba"), &pre)?);
                }
                Kind::Attention => {
                    b.attn =
                        Some(loader.attention(&mut held, &ck, &format!("{name}.attn"), &pre)?);
                }
                Kind::Moe => {
                    b.moe = Some(loader.moe(&mut held, &ck, &format!("{name}.moe"), &pre, c)?);
                }
            }
            loader.w.blocks.push(b);
        }
        let emb = ck.get("backbone.embeddings.weight")?;
        let w_ptr = loader.raw("embed.weight", &emb)?;
        let s_ptr = loader.raw("embed.scales", &ck.get("backbone.embeddings.scales")?)?;
        let b_ptr = loader.raw("embed.biases", &ck.get("backbone.embeddings.biases")?)?;
        loader.w.embed = Embed {
            w: w_ptr,
            s: s_ptr,
            b: b_ptr,
            n: emb.dim(0),
            k: emb.dim(1) * 8,
        };
        let norm_f = ck.expect("backbone.norm_f.weight", DType::BF16, &[c.hidden])?;
        loader.w.norm_f = loader.raw("norm_f", &norm_f)?;
        loader.w.head = loader.dense(&mut held, &ck, "head", &["lm_head"])?;
        if ck.unused() != 0 {
            return Err(Error::UnusedCheckpointTensors);
        }
        if loader.learned > 0 {
            eprintln!(
                "{} output projections carry what lessons taught this model",
                loader.learned
            );
        }
        if with_mtp {
            load_mtp(&mut loader, &mut held, &ck, dir, c)?;
        }
        loader.src.flush()?;
        loader.ops.s.synchronize()?;
        Ok(())
    };
    load()?;
    let _ = held;
    Ok(w)
}

/// Upstream: `loadMtp` (cuda_weights.zig:360).
fn load_mtp(
    loader: &mut Loader<'_, '_, '_, '_>,
    held: &mut Held,
    main: &Checkpoint,
    dir: &std::path::Path,
    c: Config,
) -> Result<(), Error> {
    let ck = {
        let mut head = Checkpoint::empty();
        head.add(dir, MTP_FILE).map_err(|_| Error::FileNotFound)?;
        head
    };
    loader.src.add(&ck)?;
    let enorm = ck.get("layers.0.enorm.weight")?;
    let hnorm = ck.get("layers.0.hnorm.weight")?;
    let attn_norm = ck.get("layers.0.norm.weight")?;
    let moe_norm = ck.get("layers.1.norm.weight")?;
    let final_norm = ck.get("layers.1.final_layernorm.weight")?;
    let mtp = Mtp {
        enorm: loader.raw("mtp.enorm", &enorm)?,
        hnorm: loader.raw("mtp.hnorm", &hnorm)?,
        eh_proj: loader.dense(held, &ck, "mtp.eh_proj", &["layers.0.eh_proj"])?,
        attn_norm: loader.raw("mtp.attn_norm", &attn_norm)?,
        attn: loader.attention(held, &ck, "mtp.attn", "layers.0.mixer.")?,
        moe_norm: loader.raw("mtp.moe_norm", &moe_norm)?,
        moe: loader.moe(held, &ck, "mtp.moe", "layers.1.mixer.", c)?,
        final_norm: loader.raw("mtp.final_norm", &final_norm)?,
    };
    loader.w.mtp = Some(mtp);
    if ck.unused() != 0 {
        return Err(Error::UnusedMtpTensors);
    }
    let ids = draft_ids::load(c.vocab)?;
    // head_rows: the vocabulary head's rows for the draft ids, tiled as
    // a head of their own
    let parts = three(main, "lm_head", "")?;
    let mut total: usize = 0;
    for t in &parts {
        total += ids.len() * (t.bytes.len() / t.d0);
    }
    let mut host = vec![0u8; total];
    let mut head_parts: [Part<'_>; 3] = [made(DType::U32, 0, 0, &[]); 3];
    // the fill pass first: the parts borrow the staging while later
    // fills would want it mutable
    let rows: [usize; 3] = [
        parts[0].bytes.len() / parts[0].d0,
        parts[1].bytes.len() / parts[1].d0,
        parts[2].bytes.len() / parts[2].d0,
    ];
    let mut at: usize = 0;
    for (t, row) in parts.iter().zip(rows) {
        let mut whole = vec![0u8; t.bytes.len()];
        loader.src.read(&mut whole, t.bytes)?;
        for (r, id) in ids.iter().enumerate() {
            host[at + r * row..at + r * row + row]
                .copy_from_slice(&whole[*id as usize * row..*id as usize * row + row]);
        }
        at += ids.len() * row;
    }
    let mut at: usize = 0;
    for (out, (t, row)) in head_parts.iter_mut().zip(parts.iter().zip(rows)) {
        *out = Part {
            dtype: t.dtype,
            d0: ids.len(),
            d1: t.d1,
            bytes: &host[at..at + ids.len() * row],
        };
        at += ids.len() * row;
    }
    let draft_head = loader.tile("draft_head", &[head_parts])?;
    loader.w.draft_head = Some(draft_head);
    let mut wide = vec![0i64; ids.len()];
    for (x, id) in wide.iter_mut().zip(ids.iter()) {
        *x = *id as i64;
    }
    loader.w.draft_ids = loader.alloc("draft_ids", ids.len() * 8)?;
    // SAFETY: the i64 slice's own bytes, read-only.
    let bytes = unsafe { std::slice::from_raw_parts(wide.as_ptr().cast(), ids.len() * 8) };
    loader.ops.upload(loader.w.draft_ids, bytes)?;
    loader.w.draft_count = ids.len();
    Ok(())
}
