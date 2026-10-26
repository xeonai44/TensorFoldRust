//! A MoE layer's route on Metal in two launches for 1-16 rows
//! (kernels/metal/core/moe_route.metal): sigmoid + bias routers.
//!
//! Port of `zig/src/core/moe_route.zig` (74 lines). Upstream's launches take
//! anytype records shaped `.{ .buf, .off }`; the port names that shape once —
//! [`Ref`] below — and the family layers alias it (upstream's own alias is
//! `moe.zig:9  const Ref = wts.Ref`, one structural shape everywhere).

use tensorfold_metal::{Buffer, Encoder, Pipeline};

/// Upstream: `moe_route.zig:6  Shape` — the router's shape: hidden width,
/// experts (a multiple of 4, at most 512), picks a row, rows a launch at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: u32,
    pub experts: u32,
    pub topk: u32,
    pub max_rows: u32,
    pub out_bf16: bool,
}

impl Shape {
    /// Zig's struct init `.{ .hidden = …, .experts = …, .topk = … }` with the
    /// defaulted `max_rows` and `out_bf16` (upstream's field defaults).
    pub const fn new(hidden: u32, experts: u32, topk: u32) -> Shape {
        Shape { hidden, experts, topk, max_rows: 16, out_bf16: false }
    }
}

/// Why a router source was refused.
#[derive(Debug, PartialEq)]
pub enum Error {
    UnsupportedRouterShape,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnsupportedRouterShape => write!(f, "unsupported router shape"),
        }
    }
}

impl std::error::Error for Error {}

/// The kernel source (upstream `kernel_sources.core_moe_route`, embedded
/// byte-identical from `kernels/metal/core/moe_route.metal`).
const CORE_MOE_ROUTE: &str = include_str!("../../../kernels/metal/core/moe_route.metal");

/// The kernel source with `shape`'s constants.
///
/// Upstream: `moe_route.zig:10  source` — the four defines then the source,
/// `TF_OUT_T bfloat` only when the route asks for bf16 picks.
pub fn source(s: Shape) -> Result<String, Error> {
    if s.experts % 4 != 0 || s.experts > 512 || s.hidden % 64 != 0 || s.max_rows > 16 || s.topk * s.max_rows > 512 {
        return Err(Error::UnsupportedRouterShape);
    }
    let out = if s.out_bf16 { "#define TF_OUT_T bfloat\n" } else { "" };
    Ok(format!(
        "#define TF_K {}\n#define TF_E {}\n#define TF_TOPK {}\n#define TF_MAXR {}\n{}{}",
        s.hidden, s.experts, s.topk, s.max_rows, out, CORE_MOE_ROUTE
    ))
}

/// Upstream: `moe_route.zig:16  names` — the two entry points.
pub const NAMES: [&str; 2] = ["tf_route_logits", "tf_route_select"];

/// A buffer at an offset — the shape every kernel bind in the core Metal
/// modules takes (upstream's structural `.{ .buf, .off }` records; its own
/// name for them is `weights.zig:7  Ref`).
#[derive(Clone, Copy)]
pub struct Ref<'a> {
    pub buf: &'a Buffer,
    pub off: usize,
}

impl<'a> Ref<'a> {
    /// `.{ .buf = b, .off = 0 }` — the whole-buffer ref.
    pub fn whole(buf: &'a Buffer) -> Ref<'a> {
        Ref { buf, off: 0 }
    }
}

/// Upstream: `moe_route.zig:18  fn bind` — a ref to a buffer index.
fn bind(e: &Encoder, i: usize, r: Ref<'_>) {
    e.set_buffer(r.buf, r.off, i);
}

/// Upstream: `moe_route.zig:21  logits` — logits [rows, experts] fp32 from
/// bf16 rows `x` and the router packed by `pack` (a 4-expert block a
/// threadgroup).
pub fn logits(
    e: &Encoder,
    pipe: &Pipeline,
    s: Shape,
    x: Ref<'_>,
    packed_router: Ref<'_>,
    out: Ref<'_>,
    rows: u32,
) {
    e.set_pipeline(pipe);
    bind(e, 0, x);
    bind(e, 1, packed_router);
    let rows = rows as i32;
    e.set_value(&rows, 2);
    bind(e, 3, out);
    e.dispatch_groups(
        (s.experts as u64 / 4, (rows as u64 + 3) / 4, 1),
        (256, 1, 1),
    );
}

/// Upstream: `moe_route.zig:24  Args` — the selection's inline values.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Args {
    pub rows: i32,
    pub lo: i32,
    pub hi: i32,
    pub scale: f32,
}

/// Where the selection writes (upstream `moe_route.zig:26  Outputs(B)`):
/// picks and weights, this Mac's picked experts with their members, both Macs'
/// pick lists, counts, word. Upstream's `B` is its comptime buffer view; the
/// port's is always [`Ref`] (the one structural shape, named above).
pub struct Outputs<'a> {
    pub pick: Ref<'a>,
    pub wts: Ref<'a>,
    pub ids: Ref<'a>,
    pub members: Ref<'a>,
    pub count: Ref<'a>,
    pub mine: Ref<'a>,
    pub theirs: Ref<'a>,
    pub counts: Ref<'a>,
    pub word: Ref<'a>,
}

/// Upstream: `moe_route.zig:32  select` — the top-k by sigmoid + bias (ties
/// to the lower id), weights normalized and scaled, and the groups for experts
/// `own`; the nine outputs bind at indices 3..12 in their order.
pub fn select(
    e: &Encoder,
    pipe: &Pipeline,
    logits_in: Ref<'_>,
    bias: Ref<'_>,
    scale: f32,
    rows: u32,
    own: [u32; 2],
    out: &Outputs<'_>,
) {
    e.set_pipeline(pipe);
    bind(e, 0, logits_in);
    bind(e, 1, bias);
    let args = Args {
        rows: rows as i32,
        lo: own[0] as i32,
        hi: own[1] as i32,
        scale,
    };
    e.set_value(&args, 2);
    for (i, r) in [
        out.pick, out.wts, out.ids, out.members, out.count, out.mine, out.theirs, out.counts,
        out.word,
    ]
    .into_iter()
    .enumerate()
    {
        bind(e, 3 + i, r);
    }
    e.dispatch_groups((1, 1, 1), (512, 1, 1));
}

/// Upstream: `moe_route.zig:41  pack` — a [experts, hidden] bf16 router
/// repacked for `logits`:
/// packed[q][m][i][tm][tn] = router[4q + tn][32i + 4m + tm].
pub fn pack(input: &[u16], out: &mut [u16], experts: usize, hidden: usize) {
    let mut o: usize = 0;
    for q in 0..experts / 4 {
        for m in 0..8 {
            for i in 0..hidden / 32 {
                for tm in 0..4 {
                    for tn in 0..4 {
                        out[o] = input[(4 * q + tn) * hidden + 32 * i + 4 * m + tm];
                        o += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream: `moe_route.zig:47  test "the router's shape is checked and
    /// the pack is a permutation"`.
    #[test]
    fn the_routers_shape_is_checked_and_the_pack_is_a_permutation() {
        let src = source(Shape::new(4096, 288, 8)).unwrap();
        assert!(src.starts_with("#define TF_K 4096\n#define TF_E 288\n#define TF_TOPK 8\n#define TF_MAXR 16\n"));
        assert_eq!(Err(Error::UnsupportedRouterShape), source(Shape::new(4096, 290, 8)));
        let e = 8;
        let k = 64;
        let input: Vec<u16> = (0..(e * k)).map(|i| i as u16).collect();
        let mut out = vec![0u16; e * k];
        pack(&input, &mut out, e, k);
        let mut seen = vec![false; e * k];
        for &v in &out {
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
        assert_eq!(1 * k + 0, out[1] as usize); // q 0, m 0, i 0, tm 0, tn 1: expert 1, k 0
        assert_eq!(0 * k + 1, out[4] as usize); // tm 1, tn 0: expert 0, k 1
    }
}
