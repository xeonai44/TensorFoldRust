//! MLX's affine 4-bit layout at group 64 on the host: 8 codes a u32
//! word, low nibble first, bf16 scale and bias.
//!
//! Port of `zig/src/core/affine4_host.zig` (58 lines).

/// Upstream: `group` (affine4_host.zig:4).
pub const GROUP: usize = 64;

/// Upstream: `f32of` (affine4_host.zig:6).
pub fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The nearest bf16, ties to even; NaN stays NaN. Upstream: `bf16of`
/// (affine4_host.zig:10) — the NaN keeps its top bits with the quiet
/// bit forced, unlike glue_math's cvt.rn form.
pub fn bf16_of(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// Row-major values from codes, scales and biases: scale * q + bias in
/// f32 (the rows' width a multiple of 64). Upstream: `dequantize`
/// (affine4_host.zig:18).
pub fn dequantize(words: &[u32], scales: &[u16], biases: &[u16], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let q: u32 = (words[i / 8] >> (4 * (i % 8))) & 0xf;
        *o = f32_of(scales[i / GROUP]) * q as f32 + f32_of(biases[i / GROUP]);
    }
}

/// Codes, scales and biases for row-major values: each group's minimum
/// is its bias, its range over 15 its scale. Upstream: `quantize`
/// (affine4_host.zig:26).
pub fn quantize(values: &[f32], words: &mut [u32], scales: &mut [u16], biases: &mut [u16]) {
    words.fill(0);
    for g in 0..values.len() / GROUP {
        let v = &values[g * GROUP..g * GROUP + GROUP];
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        scales[g] = bf16_of((hi - lo) / 15.0);
        biases[g] = bf16_of(lo);
        let s = f32_of(scales[g]);
        let b = f32_of(biases[g]);
        for (i, x) in v.iter().copied().enumerate() {
            let q: u32 = if s == 0.0 {
                0
            } else {
                ((x - b) / s).round().clamp(0.0, 15.0) as u32
            };
            words[(g * GROUP + i) / 8] |= q << (4 * ((g * GROUP + i) % 8));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `dequantized codes land within half a step, a flat
    /// group exactly on its bias; bf16 ties go to even`
    /// (affine4_host.zig:43).
    #[test]
    fn dequantized_codes_land_within_half_a_step_a_flat_group_exactly_on_its_bias_bf16_ties_go_to_even() {
        let mut values = [0.0f32; 2 * GROUP];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 * 0.37).sin() * 0.05;
        }
        values[GROUP..].fill(0.02);
        let mut words = [0u32; 2 * GROUP / 8];
        let mut scales = [0u16; 2];
        let mut biases = [0u16; 2];
        quantize(&values, &mut words, &mut scales, &mut biases);
        let mut back = [0.0f32; 2 * GROUP];
        dequantize(&words, &scales, &biases, &mut back);
        for (i, (v, b)) in values.iter().zip(back.iter()).enumerate() {
            assert!((v - b).abs() <= f32_of(scales[i / GROUP]) * 0.51 + 1e-3);
        }
        assert_eq!(0x3f80u16, bf16_of(1.0));
        assert_eq!(0x3f80u16, bf16_of(f32::from_bits(0x3f808000))); // a tie rounds to even
        assert_eq!(0x3f82u16, bf16_of(f32::from_bits(0x3f818000)));
        assert!(f32_of(bf16_of(f32::NAN)).is_nan());
    }
}
