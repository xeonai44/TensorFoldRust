//! The glue kernels' scalar math on the host, op for op as
//! glue_math.cuh runs it: IEEE adds, muls, fmas, divs only.
//!
//! Port of `zig/src/families/nemotron/glue_math.zig` (121 lines) —
//! bit-exact f32 work; the hex-float constants are spelled as the exact
//! f32s they name (Zig parses hex floats; the values are the nearest
//! f32 to each, unchanged).

/// Upstream: `fma` (glue_math.zig:5).
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Upstream: `bf16ToF32` (glue_math.zig:9).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// cvt.rn.bf16.f32: round to nearest even, any NaN to 0x7fff.
/// Upstream: `f32ToBf16` (glue_math.zig:14).
pub fn f32_to_bf16(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fff;
    }
    let u = v.to_bits();
    // NaN left above; the largest finite u + 0x8000 stays in u32
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// The value rounded through bf16. Upstream: `rbf` (glue_math.zig:21).
pub fn rbf(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(v))
}

/// glue_math.cuh's own truncated log2(e), not std's LOG2_E.
#[allow(clippy::approx_constant)] // upstream's own 0x1.715476p+0 (glue_math.zig:25)
const L2E: f32 = 1.442695;
const LN2_HI: f32 = 0.69314575; // 15 significant bits: j * ln2_hi is exact for |j| <= 255 — 0x1.62e4p-1
const LN2_LO: f32 = 1.4286068e-06; // 0x1.7f7d1cp-20
const SHIFTER: f32 = 1.2582912e+07; // adding it rounds to an integer — 0x1.8p+23

/// e^x within one ulp: x = j ln2 + f, a degree-7 Taylor polynomial in
/// f, then 2^j in two exact factors. Upstream: `exp` (glue_math.zig:31).
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 104.0 {
        return f32::INFINITY;
    }
    if x < -104.0 {
        return 0.0;
    }
    let j = fma(x, L2E, SHIFTER) - SHIFTER;
    let mut f = fma(j, -LN2_HI, x);
    f = fma(j, -LN2_LO, f);
    let mut p: f32 = 0.0001984127; // 1/7! .. 1/3!, each the nearest float
    p = fma(p, f, 0.0013888889);
    p = fma(p, f, 0.008333334);
    p = fma(p, f, 0.041666668);
    p = fma(p, f, 0.16666667);
    p = fma(p, f, 0.5);
    p = fma(p, f, 1.0);
    p = fma(p, f, 1.0);
    let i: i32 = j as i32;
    let a: f32 = f32::from_bits(if i > 0 { 0x7f000000 } else { 0x02000000 });
    let b: f32 = f32::from_bits(if i > 0 {
        (i << 23) as u32
    } else {
        ((i + 250) << 23) as u32
    });
    (p * a) * b
}

/// ln(1 + u) for u in [0, 1], within one ulp: 2 atanh(t / (2 + t)) with
/// the quotient carried in two floats. Upstream: `log1pUnit`
/// (glue_math.zig:53).
pub fn log1p_unit(u: f32) -> f32 {
    let hi = u > 0.5;
    let t = if hi { (u - 1.0) * 0.5 } else { u }; // exact for u in (0.5, 1]: ln(1 + u) = ln2 + ln(1 + t)
    let s = 2.0 + t;
    let err = t - (s - 2.0); // 2 + t = s + err exactly
    let zh = t / s;
    let zl = (fma(-zh, s, t) - zh * err) / s; // t / (2 + t) = zh + zl to second order
    let z2 = zh * zh;
    let mut p: f32 = 0.09090909; // 1/11 .. 1/3, each the nearest float
    p = fma(p, z2, 0.11111111);
    p = fma(p, z2, 0.14285715);
    p = fma(p, z2, 0.2);
    p = fma(p, z2, 0.33333334);
    let zz = zh + zh;
    let r = zz + fma(zz * z2, p, zl + zl);
    if !hi {
        return r;
    }
    LN2_HI + (LN2_LO + r)
}

/// 1 / (1 + e^-x). Upstream: `sigmoid` (glue_math.zig:73).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// x / (1 + e^-x): SiLU with one rounding fewer than x * sigmoid(x).
/// Upstream: `silu` (glue_math.zig:78).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

/// max(v, 0) + ln(1 + e^-|v|), the dt softplus, without log(1 + e)'s
/// loss for small e. Upstream: `softplus` (glue_math.zig:83).
pub fn softplus(v: f32) -> f32 {
    v.max(0.0) + log1p_unit(exp(-v.abs()))
}

/// The test helper's ulp distance (upstream: `ulps`, glue_math.zig:87)
/// — frexp and ldexp hand-written, std's absent for f32 here.
#[cfg(test)]
pub(crate) fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    if w == 0.0 || !w.is_finite() {
        return if got == w { 0.0 } else { f64::INFINITY };
    }
    let e = frexp32(w).1;
    let ulp = 2.0f64.powi((e - 24).max(-149));
    (got as f64 - want).abs() / ulp
}

#[cfg(test)]
fn frexp32(w: f32) -> (f32, i32) {
    let mut w = w;
    let mut e: i32 = 0;
    while w != 0.0 && w.abs() < f32::MIN_POSITIVE {
        w *= 4294967296.0; // 2^32: exact, twice reaches every subnormal
        e -= 32;
    }
    let b = w.to_bits();
    let biased = ((b >> 23) & 0xff) as i32;
    let m = f32::from_bits((b & 0x807f_ffff) | ((126u32) << 23));
    (m, biased - 126 + e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `exp stays within 1 ulp of the f64 exp, including
    /// subnormal and overflowing results` (glue_math.zig:95).
    #[test]
    fn exp_stays_within_1_ulp_of_the_f64_exp_including_subnormal_and_overflowing_results() {
        let mut worst: f64 = 0.0;
        let mut x: f32 = -103.9;
        while x < 88.72 {
            worst = worst.max(ulps(exp(x), (x as f64).exp()));
            x += 0.000377;
        }
        assert!(worst <= 1.0);
        assert_eq!(f32::INFINITY, exp(88.73));
        assert_eq!(1.0f32, exp(0.0));
        assert_eq!(1.0f32, exp(-0.0));
        assert_eq!(0.0f32, exp(-200.0));
        assert!(exp(f32::NAN).is_nan());
    }

    /// Upstream's `log1pUnit stays within 1 ulp of the f64 log1p over
    /// [0, 1]` (glue_math.zig:107).
    #[test]
    fn log1p_unit_stays_within_1_ulp_of_the_f64_log1p_over_0_1() {
        let mut worst: f64 = 0.0;
        let mut u: f32 = 0.0;
        while u <= 1.0 {
            worst = worst.max(ulps(log1p_unit(u), (u as f64).ln_1p()));
            u += 0.0000037;
        }
        for v in [
            1e-30f32,
            1e-10,
            5.9604645e-8,
            3.861785e-3,
            0.49958834,
            0.5,
            0.50000006,
            1.0,
        ] {
            worst = worst.max(ulps(log1p_unit(v), (v as f64).ln_1p()));
        }
        assert!(worst <= 1.0);
        assert_eq!(0.0f32, log1p_unit(0.0));
    }

    /// Upstream's `bf16 rounding is round to nearest even`
    /// (glue_math.zig:116).
    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(0x3f80u16, f32_to_bf16(1.0));
        assert_eq!(0x3f80u16, f32_to_bf16(f32::from_bits(0x3f808000))); // a tie rounds to even
        assert_eq!(0x3f82u16, f32_to_bf16(f32::from_bits(0x3f818000)));
        assert_eq!(0x7fffu16, f32_to_bf16(f32::NAN));
    }
}
