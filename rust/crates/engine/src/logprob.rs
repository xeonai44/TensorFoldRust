//! A committed token's log probability under the target's raw distribution, with its
//! best tokens (OpenAI logprobs).
//!
//! Port of `zig/src/core/lanes/logprob.zig`.

/// The most alternatives a row carries (OpenAI's `top_logprobs` limit).
///
/// Upstream: `logprob.zig:4  pub const max_top = 20;`
pub const MAX_TOP: usize = 20;

/// The words a backend writes for one row: the log normalizer, the pick's logit, the best
/// ids, their logits.
///
/// Upstream: `logprob.zig:7  pub const words = 2 + 2 * max_top;`
pub const ROW_WORDS: usize = 2 + 2 * MAX_TOP;

/// The words a backend writes for one row.
///
/// Upstream: `logprob.zig:9  pub const Row = struct { ... }`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogprobRow {
    /// The committed token.
    pub token: u32,
    /// Its log probability; NaN when it was forced and is not among the best `count`.
    pub logprob: f32,
    /// How many of the `ids`/`logprobs` entries are meaningful.
    pub count: u8,
    /// Best ids, logit descending, ties by the lower id.
    pub ids: [u32; MAX_TOP],
    /// Their log probabilities.
    pub logprobs: [f32; MAX_TOP],
}

impl LogprobRow {
    /// One row from a backend's words: `pick` is the token whose logit the backend
    /// gathered, `count` best ids follow.
    ///
    /// Upstream: `logprob.zig` `Row.fromWords`.
    pub fn from_words(w: &[u32], pick: u32, count: u8) -> LogprobRow {
        assert!(count as usize <= MAX_TOP && w.len() >= ROW_WORDS);
        let lse = f32::from_bits(w[0]);
        let mut r = LogprobRow {
            token: pick,
            logprob: f32::from_bits(w[1]) - lse,
            count,
            ids: [0; MAX_TOP],
            logprobs: [0.0; MAX_TOP],
        };
        for i in 0..count as usize {
            r.ids[i] = w[2 + i];
            r.logprobs[i] = f32::from_bits(w[2 + MAX_TOP + i]) - lse;
        }
        r
    }

    /// A forced token's row at this position: its log probability among the best, else NaN.
    ///
    /// Upstream: `logprob.zig` `Row.forToken`.
    pub fn for_token(&self, token: u32) -> LogprobRow {
        if token == self.token {
            return *self;
        }
        let mut out = *self;
        out.token = token;
        out.logprob = f32::NAN;
        for i in 0..self.count as usize {
            if self.ids[i] == token {
                out.logprob = self.logprobs[i];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the Zig `fromWords`/`forToken` contract with a hand-built word row.
    #[test]
    fn from_words_and_for_token() {
        // lse = bits of 1.0, pick logit = bits of 2.0, one best id 7 with bits of 1.5.
        let mut w = [0u32; ROW_WORDS];
        w[0] = 1.0f32.to_bits();
        w[1] = 2.0f32.to_bits();
        w[2] = 7;
        w[2 + MAX_TOP] = 1.5f32.to_bits();
        let row = LogprobRow::from_words(&w, 9, 1);
        assert_eq!(row.token, 9);
        assert_eq!(row.count, 1);
        assert_eq!(row.ids[0], 7);
        let forced = row.for_token(7);
        assert_eq!(forced.token, 7);
        assert!(!forced.logprob.is_nan());
        assert!(row.for_token(11).logprob.is_nan());
    }
}
