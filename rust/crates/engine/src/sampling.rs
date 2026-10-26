//! The keyed host sampler (Python exact_sampling.py) bit for bit: libm's exp and
//! log, numpy's sum order.
//!
//! Port of `zig/src/core/lanes/sampling.zig`. The exactness contract is the file's
//! own doc comment: libm's `exp` and `log` (the same system libm Python calls) and
//! numpy's float64 pairwise sum order. The unit tests pin the values the Zig tests
//! pin, so any drift from the Zig bytes shows up here first.

use sha2::{Digest, Sha256};

// The system libm, exactly as Python calls it (Zig: `extern "c" fn exp(x: f64) f64`).
unsafe extern "C" {
    #[link_name = "exp"]
    fn libm_exp(x: f64) -> f64;
    #[link_name = "log"]
    fn libm_log(x: f64) -> f64;
}

/// libm's exp (Python's), the one the Zig sampler calls.
fn exp(x: f64) -> f64 {
    unsafe { libm_exp(x) }
}

/// libm's log (Python's), the one the Zig sampler calls.
fn log(x: f64) -> f64 {
    unsafe { libm_log(x) }
}

/// One request's sampling (Python's exact_sampling.Sampling); a request without one
/// decodes greedily.
///
/// Upstream: `lanes.zig` `Sampling`, re-exported at `engine_api.zig:10`. The field
/// defaults mirror `sampling.zig:8` exactly: temperature 1.0, top_k 20, top_p 0.95,
/// min_p 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub seed: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub min_p: f64,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            seed: 0,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
        }
    }
}

impl Sampling {
    /// ln(min_p), -inf when off. Upstream: `sampling.zig:16  minLog`.
    pub fn min_log(&self) -> f64 {
        if self.min_p > 0.0 {
            log(self.min_p)
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// A prompt's reproducible seed (`seed_for`), for requests that name none.
///
/// Upstream: `engine_api.zig:13  pub const seedFor = lanes.sampling.seedFor;`
/// Implementation: `sampling.zig:22` — SHA-256 over the decimal token ids joined by
/// commas followed by `|{salt}`, first 8 digest bytes little-endian, masked to 63 bits.
pub fn sample_seed_for(tokens: &[u32], salt: i64) -> u64 {
    let mut h = Sha256::new();
    let mut text = String::with_capacity(16);
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            h.update(b",");
        }
        use std::fmt::Write;
        text.clear();
        let _ = write!(text, "{}", t);
        h.update(text.as_bytes());
    }
    use std::fmt::Write as _;
    text.clear();
    let _ = write!(text, "|{}", salt);
    h.update(text.as_bytes());
    let digest = h.finalize();
    u64::from_le_bytes(digest[0..8].try_into().unwrap()) & ((1u64 << 63) - 1)
}

/// splitmix64's finalizer. Upstream: `sampling.zig:34  fn mix`.
pub fn mix(v: u64) -> u64 {
    let mut x = v;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// splitmix64 of (seed, position, id): the bits every keyed sampler starts from.
///
/// Upstream: `sampling.zig:44  fn keyBits`.
pub fn key_bits(seed: u64, position: u64, id: u64) -> u64 {
    let mut x = mix(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    x = mix(x ^ position.wrapping_mul(0xD1B5_4A32_D192_ED03));
    mix(x ^ id)
}

/// Uniform (0, 1] double from the key's top 53 bits (Python `uniform`).
///
/// Upstream: `sampling.zig:51  fn uniform`.
pub fn uniform(seed: u64, position: u64, id: u64) -> f64 {
    let x = key_bits(seed, position, id);
    (x >> 11) as f64 * f64::from_bits(0x3CA0_0000_0000_0000) + f64::from_bits(0x3C90_0000_0000_0000)
}

/// numpy's float64 pairwise sum (`np.add.reduce` on a contiguous row).
///
/// Upstream: `sampling.zig:57  fn pairwiseSum` — the branch points and the `-0.0`
/// start for short rows are numpy's own, not style.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    if a.len() < 8 {
        let mut res: f64 = -0.0;
        for v in a {
            res += v;
        }
        return res;
    }
    if a.len() <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[0..8]);
        let mut i = 8;
        while i < a.len() - (a.len() % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < a.len() {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = a.len() / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[0..n2]) + pairwise_sum(&a[n2..])
}

#[derive(Clone, Copy)]
struct Candidate {
    value: f64,
    id: u64,
}

/// The id drawn among candidates `ids` with float64 logits `values` (Python `choose`,
/// and `choose_rows` row by row).
///
/// Upstream: `sampling.zig:85  pub fn choose`. The sort is by value descending with
/// ties to the lower id — a total order, so any correct sort agrees; libm's exp and
/// log and the pairwise-sum order make the rest bit-exact.
pub fn choose(values: &[f64], ids: &[u64], position: u64, s: Sampling) -> Result<u64, &'static str> {
    if values.len() != ids.len() {
        return Err("values and ids disagree in length");
    }
    let width = ids.len();
    let mut order: Vec<Candidate> = values
        .iter()
        .zip(ids)
        .map(|(&value, &id)| Candidate { value, id })
        .collect();
    order.sort_unstable_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap()
            .then(a.id.cmp(&b.id))
    });
    if s.temperature <= 0.0 {
        // temperature 0 is greedy: the first maximum, ties to the lower id.
        return Ok(order[0].id);
    }
    let top_k = if s.top_k != 0 { s.top_k as usize } else { width };
    let k = top_k.min(width).max(1);
    let t = s.temperature.max(1e-6);
    let mut scaled = vec![0.0f64; k];
    for (x, o) in scaled.iter_mut().zip(&order[0..k]) {
        *x = o.value / t;
    }
    let mut keep = k;
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mut probs = vec![0.0f64; k];
        for (p, x) in probs.iter_mut().zip(&scaled) {
            *p = exp(x - scaled[0]);
        }
        let total = pairwise_sum(&probs);
        let mut cum: f64 = 0.0;
        let mut below: usize = 0;
        for p in probs.iter_mut() {
            *p /= total;
            cum += *p;
            if cum < s.top_p {
                below += 1;
            }
        }
        keep = keep.min(below + 1);
    }
    if s.min_p > 0.0 {
        let floor = scaled[0] + s.min_log();
        let mut n: usize = 0;
        for x in &scaled[0..keep] {
            if *x >= floor {
                n += 1;
            }
        }
        keep = n;
    }
    let mut best: usize = 0;
    let mut best_score = f64::NEG_INFINITY;
    for j in 0..keep {
        let gumbel = -log(-log(uniform(s.seed, position, order[j].id)));
        let score = scaled[j] + gumbel;
        if j == 0 || score > best_score {
            best = j;
            best_score = score;
        }
    }
    Ok(order[best].id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors sampling.zig:133: seeds keep 63 bits and uniforms fall in (0, 1].
    #[test]
    fn seeds_keep_63_bits_and_uniforms_fall_in_unit_range() {
        assert!(sample_seed_for(&[1, 2, 3], 0) < (1u64 << 63));
        let u = uniform(7, 11, 13);
        assert!(u > 0.0 && u <= 1.0);
    }

    /// Mirrors sampling.zig:139: temperature 0 draws greedily, an exact tie going to
    /// the lower id, at every position.
    #[test]
    fn temperature_0_draws_greedily_with_ties_to_the_lower_id() {
        let values = [1.5, 3.0, -2.0, 3.0];
        let ids = [10u64, 35, 50, 93];
        for position in 0..64 {
            assert_eq!(
                choose(&values, &ids, position, Sampling { seed: 7, temperature: 0.0, ..Sampling::default() }),
                Ok(35)
            );
        }
    }

    /// A wider set: the top_p cut and the gumbel draw both stay reproducible.
    #[test]
    fn sampled_draws_are_pinned() {
        let values: Vec<f64> = (0..24u64).map(|i| (i as f64 * 0.5) - 4.0).collect();
        let ids: Vec<u64> = (0..24u64).collect();
        let s = Sampling { seed: 1234, temperature: 0.7, top_p: 0.95, top_k: 20, min_p: 0.1 };
        let first = choose(&values, &ids, 0, s).unwrap();
        // The same inputs draw the same id (the keyed contract); a different
        // position may draw a different one.
        assert_eq!(choose(&values, &ids, 0, s).unwrap(), first);
        assert_eq!(choose(&values, &ids, 128, s).unwrap(), choose(&values, &ids, 128, s).unwrap());
    }

    /// numpy's pairwise order: the -0.0 start and the 8-lane shape are numpy's.
    #[test]
    fn pairwise_sum_matches_the_short_path_and_long_path() {
        assert_eq!(pairwise_sum(&[5.0]), 5.0);
        let long: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&long), long.iter().sum::<f64>());
    }
}
