//! Window rows as chains or trees, the accepted path and tree trimming (Python
//! lane_tree, sanitize_tree).
//!
//! Port of `zig/src/core/lanes/accept.zig`.

/// Each row's parent row (row 0 is the pending token): a chain when `parents`
/// (drafts' parents) is None.
///
/// Upstream: `accept.zig:6  pub fn rowParents`.
pub fn row_parents(rows: usize, parents: Option<&[i32]>) -> Vec<i32> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        if r == 0 {
            out.push(-1);
        } else if let Some(p) = parents {
            let q = p[r - 1];
            out.push(if q < 0 { 0 } else { q + 1 });
        } else {
            out.push(r as i32 - 1);
        }
    }
    out
}

/// Each row's depth (its distance from row 0); parents must precede their children.
///
/// Upstream: `accept.zig:20  pub fn depths`.
pub fn depths(parents: &[i32]) -> Result<Vec<u32>, crate::Error> {
    let mut out = Vec::with_capacity(parents.len());
    for (row, &q) in parents.iter().enumerate() {
        if q < 0 {
            out.push(0);
        } else {
            if q >= row as i32 {
                return Err(crate::Error::ParentAfterChild);
            }
            out.push(out[q as usize] + 1);
        }
    }
    Ok(out)
}

/// Upstream: `accept.zig:34  pub fn isChain`.
pub fn is_chain(parents: &[i32]) -> bool {
    parents
        .iter()
        .enumerate()
        .all(|(row, &q)| q == row as i32 - 1)
}

/// Rows of the accepted path from row 0: follow the first child (by row) whose
/// token is the target's pick.
///
/// Upstream: `accept.zig:40  pub fn acceptPath`.
pub fn accept_path(tokens: &[u32], parents: &[i32], picks: &[u32]) -> Vec<u32> {
    let mut path = vec![0u32];
    loop {
        let at = *path.last().unwrap() as usize;
        let want = picks[at];
        let mut next: Option<u32> = None;
        for (row, &q) in parents.iter().enumerate() {
            if q == at as i32 && tokens[row] == want {
                next = Some(row as u32);
                break;
            }
        }
        match next {
            Some(row) => path.push(row),
            None => return path,
        }
    }
}

/// Upstream: `accept.zig:59  pub const Tree`.
pub struct Tree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
}

/// The first `budget` nodes, dropping orphaned subtrees and nodes whose parents do
/// not precede them.
///
/// Upstream: `accept.zig:62  pub fn sanitizeTree`.
pub fn sanitize_tree(tokens: &[u32], parents: &[i32], budget: usize) -> Tree {
    let n = budget.min(tokens.len().min(parents.len()));
    let mut kept = vec![-1i32; n];
    let mut out_t: Vec<u32> = Vec::new();
    let mut out_p: Vec<i32> = Vec::new();
    for i in 0..n {
        let q = parents[i];
        if q >= 0 && (q >= i as i32 || kept[q as usize] < 0) {
            continue;
        }
        kept[i] = out_t.len() as i32;
        out_t.push(tokens[i]);
        out_p.push(if q < 0 { -1 } else { kept[q as usize] });
    }
    Tree { tokens: out_t, parents: out_p }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors accept.zig:81 "accept path follows matching children".
    #[test]
    fn accept_path_follows_matching_children() {
        let parents = [-1i32, 0, 0, 1];
        let path = accept_path(&[5, 6, 7, 8], &parents, &[7, 9, 9, 9]);
        assert_eq!(path, [0, 2]);
        assert_eq!(row_parents(3, None), [-1, 0, 1]);
    }

    /// Mirrors accept.zig:92 "sanitize drops orphans".
    #[test]
    fn sanitize_drops_orphans() {
        let t = sanitize_tree(&[1, 2, 3, 4], &[-1, 0, 5, 2], 4);
        assert_eq!(t.tokens, [1, 2]);
        assert_eq!(t.parents, [-1, 0]);
    }
}
