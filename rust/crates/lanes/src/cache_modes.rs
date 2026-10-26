//! Canonical decoded intervals are ordered, nonempty and strictly separated.
//!
//! Port of `zig/src/core/cache_modes.zig`. The engine contract's `decode_spans`
//! validation (engine_api.zig:292) runs this; the serial host extends a reply's
//! spans through it.

use crate::Error;

/// A half-open decoded-row interval. Upstream: `cache_modes.zig:4  pub const Span
/// = [2]u32`.
pub type Span = [u32; 2];

/// Reject empty, reversed, overlapping, adjacent or unsorted intervals; the empty
/// list is valid. Upstream: `validate` (cache_modes.zig:7).
pub fn validate(spans: &[Span]) -> Result<(), Error> {
    let mut previous: u32 = 0;
    for (i, span) in spans.iter().enumerate() {
        if span[0] >= span[1] || (i > 0 && span[0] <= previous) {
            return Err(Error::Backend("the decoded intervals are not canonical".into()));
        }
        previous = span[1];
    }
    Ok(())
}

/// Compare canonical interval lists in linear time, clipped to consumed rows
/// before `at`. Upstream: `equal` (cache_modes.zig:16).
pub fn equal(a: &[Span], b: &[Span], at: u32) -> bool {
    let mut i: usize = 0;
    loop {
        let left = i < a.len() && a[i][0] < at;
        let right = i < b.len() && b[i][0] < at;
        if !left || !right {
            return left == right;
        }
        if a[i][0] != b[i][0] || a[i][1].min(at) != b[i][1].min(at) {
            return false;
        }
        i += 1;
    }
}

/// Copy a canonical map through `at`, without retaining later rows or borrowed
/// storage. Upstream: `prefix` (cache_modes.zig:27).
pub fn prefix(spans: &[Span], at: u32) -> Vec<Span> {
    let mut n: usize = 0;
    while n < spans.len() && spans[n][0] < at {
        n += 1;
    }
    let mut out: Vec<Span> = spans[..n].to_vec();
    if n > 0 {
        out[n - 1][1] = out[n - 1][1].min(at);
    }
    out
}

/// Clip prompt modes, then append consumed reply rows and coalesce the one
/// possible adjacent interval. Upstream: `reply` (cache_modes.zig:36).
pub fn reply(spans: &[Span], prompt: u32, stands: u32) -> Vec<Span> {
    debug_assert!(stands >= prompt);
    let mut n: usize = 0;
    while n < spans.len() && spans[n][0] < prompt {
        n += 1;
    }
    let merge = n > 0 && spans[n - 1][1] >= prompt;
    let append = stands > prompt && !merge;
    let mut out: Vec<Span> = Vec::with_capacity(n + usize::from(append));
    out.extend_from_slice(&spans[..n]);
    if n > 0 {
        out[n - 1][1] = if merge {
            stands
        } else {
            out[n - 1][1].min(prompt)
        };
    }
    if append {
        out.push([prompt, stands]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors cache_modes.zig:49: canonical spans reject alternate spellings
    /// before storage.
    #[test]
    fn canonical_spans_reject_alternate_spellings() {
        validate(&[]).unwrap();
        validate(&[[0, 2], [3, 100]]).unwrap();
        for bad in [
            &[[2, 2]][..],
            &[[3, 2]],
            &[[1, 4], [3, 5]],
            &[[1, 4], [4, 5]],
            &[[7, 8], [1, 3]],
        ] {
            assert!(validate(bad).is_err());
        }
    }

    /// Mirrors cache_modes.zig:56: canonical equality and owned prefixes ignore
    /// future rows and lookahead.
    #[test]
    fn equality_and_prefixes_ignore_future_rows() {
        let spans = [[2, 10], [20, 30]];
        let got = prefix(&spans, 7);
        assert_eq!(got, [[2, 7]]);
        assert!(equal(&spans, &[[2, 99]], 7));
        assert!(!equal(&spans, &[[2, 6]], 7));
        assert!(equal(&[[7, 20]], &[], 7));
        assert!(equal(&[[0, 1]], &[], 0));
    }

    /// Mirrors cache_modes.zig:67: reply extension clips future prompt modes and
    /// coalesces only its boundary.
    #[test]
    fn reply_extension_clips_and_coalesces() {
        for stands in [10u32, 11, 13] {
            let merged = reply(&[[2, 10], [12, 30]], 10, stands);
            validate(&merged).unwrap();
            assert_eq!(merged, [[2, stands]]);
            let extended = reply(&[[2, 5], [10, 30]], 10, stands);
            validate(&extended).unwrap();
            assert_eq!(extended.len(), if stands == 10 { 1 } else { 2 });
            if stands > 10 {
                assert_eq!(extended[1], [10, stands]);
            }
        }
    }
}
