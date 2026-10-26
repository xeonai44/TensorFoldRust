//! Prompt chunks on the zero-anchored grid: a pass that resumes or
//! keeps states cuts where a cache-off pass does.
//!
//! Port of `zig/src/families/nemotron/cuda_prompt_grid.zig` (28 lines).

/// The end of the chunk starting at `at`: the next multiple of `step`,
/// never past `len`. Upstream: `end` (cuda_prompt_grid.zig:5).
pub fn end(at: usize, len: usize, step: usize) -> usize {
    len.min(at - at % step + step)
}

/// A kept state a pass may start from: a grid point inside the prompt.
/// Upstream: `resumable` (cuda_prompt_grid.zig:10).
pub fn resumable(from: usize, len: usize, step: usize) -> bool {
    from > 0 && from < len && from % step == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `a resumed or kept pass cuts the cache-off grid,
    /// and only grid points resume` (cuda_prompt_grid.zig:14).
    #[test]
    fn a_resumed_or_kept_pass_cuts_the_cache_off_grid_and_only_grid_points_resume() {
        let step = 2048;
        let mut cuts = [0usize; 3];
        let mut at: usize = 0;
        for c in cuts.iter_mut() {
            *c = end(at, 5000, step);
            at = *c;
        }
        assert_eq!([2048, 4096, 5000], cuts);
        assert_eq!(4096, end(2048, 5000, step)); // resumed at 2048: the same later cuts
        assert_eq!(2048, end(100, 5000, step)); // an off-grid start returns to the grid
        assert_eq!(1000, end(0, 1000, step));
        assert!(resumable(2048, 5000, step) && resumable(4096, 5000, step));
        assert!(
            !resumable(0, 5000, step)
                && !resumable(3000, 5000, step)
                && !resumable(4096, 4096, step)
        );
    }
}
