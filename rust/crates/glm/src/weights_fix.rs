//! GLM-5.3-Flash weights rewritten as they load: packed mixes and
//! router, conv taps, fp32 norms, column slices.
//!
//! Port of `zig/src/families/glm/weights_fix.zig` (81 lines) — the
//! fixes are the exact index arithmetic of upstream's switch, the
//! tensors read whole into host memory and rewritten into their place.

/// One tensor's bytes on disk. Upstream: `Src` (weights_fix.zig:5).
#[derive(Clone, Copy)]
pub struct Src {
    pub shard: u32,
    pub off: u64,
    pub len: u64,
    pub dtype: tensorfold_safetensors::DType,
    pub shape: [usize; 5],
    pub rank: u8,
}

/// Upstream: `Fix` (weights_fix.zig:6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fix {
    HcPack,
    RouterPack,
    Conv,
    ToF32,
    TransposeBf16,
    Cols,
}

/// A tensor read whole into host memory and rewritten into its place
/// (small: mixes, router, conv taps, norms). Upstream: `Transform`
/// (weights_fix.zig:8) — the port's `dst` is a mutable slice and the
/// reads come through a reader over the shards.
pub struct Transform<'a> {
    pub src: Src,
    pub dst: &'a mut [u8],
    pub fix: Fix,
    pub extra: [Option<Src>; 3],
    pub arg: [usize; 3],
}

impl<'a> Transform<'a> {
    /// Upstream: `.{ .src = ..., .dst = ..., .fix = ... }` with the
    /// defaults.
    pub fn new(src: Src, dst: &'a mut [u8], fix: Fix) -> Transform<'a> {
        Transform {
            src,
            dst,
            fix,
            extra: [None; 3],
            arg: [0, 0, 0],
        }
    }
}

/// Why a transform refused.
#[derive(Debug, PartialEq)]
pub enum TransformError {
    ShortRead,
    MissingTensor,
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::ShortRead => write!(f, "the tensor's read ran short"),
            TransformError::MissingTensor => write!(f, "a part of the tensor is missing"),
        }
    }
}

impl std::error::Error for TransformError {}

/// A shard read: the exact span of a tensor. Upstream: `readAll`
/// (weights_fix.zig:11) over the fd pread.
pub fn read_span<R>(read: R, dest: &mut [u8], at: u64) -> Result<(), TransformError>
where
    R: Fn(&mut [u8], u64) -> Result<usize, ()>,
{
    let mut done = 0;
    while done < dest.len() {
        let at = at + done as u64;
        let n = read(&mut dest[done..], at).map_err(|_| TransformError::ShortRead)?;
        if n == 0 {
            return Err(TransformError::ShortRead);
        }
        done += n;
    }
    Ok(())
}

/// Upstream: `transform` (weights_fix.zig:15) — the fix's exact index
/// arithmetic over the raw bytes.
pub fn transform<R>(
    alloc: &mut dyn FnMut(usize) -> Option<Vec<u8>>,
    fds: &[R],
    t: Transform<'_>,
) -> Result<(), TransformError>
where
    R: Fn(&mut [u8], u64) -> Result<usize, ()>,
{
    let mut raw = alloc(t.src.len as usize).ok_or(TransformError::ShortRead)?;
    read_span(|b, at| fds[t.src.shard as usize](b, at), &mut raw, t.src.off)?;
    let in16: &[u16] = {
        // SAFETY: the raw bytes are the tensor's own, read whole and
        // bf16-aligned by construction (safetensors' payload even).
        unsafe { std::slice::from_raw_parts(raw.as_ptr().cast(), raw.len() / 2) }
    };
    match t.fix {
        Fix::HcPack => {
            // [og 6][tm 4][i 16][sgn 8][lane 32][tn 4] -> [og][sgn][lane][i][tm][tn]
            let out = as_u16(t.dst);
            let mut o = 0;
            for og in 0..6 {
                for sgn in 0..8 {
                    for lane in 0..32 {
                        for i in 0..16 {
                            for tm in 0..4 {
                                for tn in 0..4 {
                                    out[o] = in16[(og * 4 + tm) * 16384 + i * 1024 + sgn * 128 + lane * 4 + tn];
                                    o += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        Fix::RouterPack => {
            // packed[q][thrM][i][tm][tn] = router[4q + tn][32i + 4thrM + tm]
            let out = as_u16(t.dst);
            let e = t.src.shape[0];
            let kk = t.src.shape[1];
            let mut o = 0;
            for q in 0..e / 4 {
                for thr in 0..8 {
                    for i in 0..kk / 32 {
                        for tm in 0..4 {
                            for tn in 0..4 {
                                out[o] = in16[(4 * q + tn) * kk + 32 * i + 4 * thr + tm];
                                o += 1;
                            }
                        }
                    }
                }
            }
        }
        Fix::Conv => {
            // [q | k | v][channel][tap] bf16 -> [tap][3 * channels] fp32,
            // channels [arg 0, arg 0 + arg 1) in TP2 — each part's bytes
            // read into its own buffer and rewritten in place, one
            // part a turn (upstream's per-part stack buffer)
            let out = as_f32(t.dst);
            let c = if t.arg[1] != 0 { t.arg[1] } else { t.src.shape[0] };
            let c0 = t.arg[0];
            let taps = t.src.shape[2];
            let parts = [Some(t.src), t.extra[0], t.extra[1]];
            for (part, ps) in parts.iter().enumerate() {
                let Some(p) = ps else {
                    return Err(TransformError::MissingTensor);
                };
                let raw = if part == 0 {
                    raw.clone()
                } else {
                    let mut b = alloc(p.len as usize).ok_or(TransformError::ShortRead)?;
                    read_span(|bb, at| fds[p.shard as usize](bb, at), &mut b, p.off)?;
                    b
                };
                // SAFETY: the part's own bytes, bf16 by construction.
                let v: &[u16] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr().cast(), raw.len() / 2) };
                for ch in 0..c {
                    for tap in 0..taps {
                        out[tap * 3 * c + part * c + ch] =
                            f32::from_bits((v[(c0 + ch) * taps + tap] as u32) << 16);
                    }
                }
            }
        }
        Fix::ToF32 => {
            let out = as_f32(t.dst);
            for (x, o) in in16.iter().zip(out.iter_mut()) {
                *o = f32::from_bits((*x as u32) << 16);
            }
        }
        Fix::Cols => {
            // each stored row's bytes [arg 1, arg 1 + arg 2) of its arg 0
            let rows = t.src.len as usize / t.arg[0];
            for rr in 0..rows {
                let src = &raw[rr * t.arg[0] + t.arg[1]..rr * t.arg[0] + t.arg[1] + t.arg[2]];
                t.dst[rr * t.arg[2]..rr * t.arg[2] + t.arg[2]].copy_from_slice(src);
            }
        }
        Fix::TransposeBf16 => {
            // [r][c] -> [c][r]
            let out = as_u16(t.dst);
            let rr = t.src.shape[0];
            let cc = t.src.shape[1];
            for r in 0..rr {
                for c in 0..cc {
                    out[c * rr + r] = in16[r * cc + c];
                }
            }
        }
    }
    Ok(())
}

fn as_u16(dst: &mut [u8]) -> &mut [u16] {
    // SAFETY: the transform's own destination, bf16-aligned by
    // construction (the load's placement).
    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len() / 2) }
}

fn as_f32(dst: &mut [u8]) -> &mut [f32] {
    // SAFETY: the transform's own destination, fp32-aligned by
    // construction.
    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len() / 4) }
}
