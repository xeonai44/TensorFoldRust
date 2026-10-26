//! Speed-up mode's slot commands: rank 0 sends each slot operation
//! before it runs it; rank 1 runs the same, in order.
//!
//! Port of `zig/src/families/glm/mirror.zig`'s pure half (mirror.zig:14-66,
//! 200-240): the command word codecs — window and draft — with
//! upstream's two tests mirrored whole. The follow/apply halves ride
//! with the engine and slots tranche (they need Engine and Slots).

/// Upstream: `Kind` (mirror.zig:10).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Kind {
    Begin = 1,
    Chunk,
    Window,
    Keep,
    Draft,
    Release,
    Save,
    Restore,
    Drop,
    Persist,
    Load,
    Forget,
    DiskNeed,
    DiskReclaim,
    DiskReserve,
    DiskFinish,
    DiskOtherNext,
    DiskOtherBytes,
    DiskOtherRemove,
    DiskOtherUsed,
}

impl Kind {
    /// Upstream: `std.enums.fromInt` — None for an out-of-set word.
    pub fn from_u32(v: u32) -> Option<Kind> {
        if (1..=20).contains(&v) {
            Some(unsafe { std::mem::transmute::<u32, Kind>(v) })
        } else {
            None
        }
    }
}

/// Why a command codec refused.
#[derive(Debug, PartialEq)]
pub enum CommandError {
    CommandOutOfStep,
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandError::CommandOutOfStep => write!(f, "the command is out of step"),
        }
    }
}

impl std::error::Error for CommandError {}

/// One window in a window command (mirror.zig's `Win` via slots_mod):
/// its slot, pending token, held count and host drafts, the tokens
/// pointing into the command's words.
#[derive(Clone, Copy)]
pub struct Win<'w> {
    pub slot: u32,
    pub pending: u32,
    pub held: u32,
    pub tokens: &'w [u32],
}

/// One stream's head request in a draft command (mirror.zig's `Draft`
/// via slots_mod): its slot, whether it absorbs the prompt's last row,
/// its depth and its next tokens, the follow pointing into the words.
#[derive(Clone, Copy)]
pub struct Draft<'w> {
    pub slot: u32,
    pub prompt: bool,
    pub depth: u32,
    pub follow: &'w [u32],
}

/// A window command: rank 0's last window digest, then each window's
/// slot, pending token, held count and host drafts. Upstream:
/// `windowWords` (mirror.zig:19).
pub fn window_words(out: &mut Vec<u32>, digest: u64, wins: &[Win<'_>]) {
    out.clear();
    out.extend_from_slice(&[digest as u32, (digest >> 32) as u32, wins.len() as u32]);
    for w in wins {
        out.extend_from_slice(&[w.slot, w.pending, w.held, w.tokens.len() as u32]);
        out.extend_from_slice(w.tokens);
    }
}

/// A window command's windows into `out` (their drafts point into
/// `words`) and rank 0's digest. Upstream: `readWindows`
/// (mirror.zig:25).
pub fn read_windows<'w>(words: &'w [u32], out: &mut [Win<'w>]) -> Result<(u64, usize), CommandError> {
    if words.len() < 3 || words[2] == 0 || words[2] as usize > out.len() {
        return Err(CommandError::CommandOutOfStep);
    }
    let n = words[2] as usize;
    let mut at: usize = 3;
    for w in out[..n].iter_mut() {
        if at + 4 > words.len() || at + 4 + words[at + 3] as usize > words.len() {
            return Err(CommandError::CommandOutOfStep);
        }
        *w = Win {
            slot: words[at],
            pending: words[at + 1],
            held: words[at + 2],
            tokens: &words[at + 4..at + 4 + words[at + 3] as usize],
        };
        at += 4 + words[at + 3] as usize;
    }
    if at != words.len() {
        return Err(CommandError::CommandOutOfStep);
    }
    Ok((words[0] as u64 | (words[1] as u64) << 32, n))
}

/// A draft command: each stream's slot, whether it absorbs the
/// prompt's last row, its depth and its next tokens. Upstream:
/// `draftWords` (mirror.zig:42).
pub fn draft_words(out: &mut Vec<u32>, reqs: &[Draft<'_>]) {
    out.clear();
    out.push(reqs.len() as u32);
    for r in reqs {
        out.extend_from_slice(&[r.slot, r.prompt as u32, r.depth, r.follow.len() as u32]);
        out.extend_from_slice(r.follow);
    }
}

/// A draft command's streams into `out` (their tokens point into
/// `words`). Upstream: `readDrafts` (mirror.zig:51).
pub fn read_drafts<'w>(words: &'w [u32], out: &mut [Draft<'w>]) -> Result<usize, CommandError> {
    if words.is_empty() || words[0] == 0 || words[0] as usize > out.len() {
        return Err(CommandError::CommandOutOfStep);
    }
    let n = words[0] as usize;
    let mut at: usize = 1;
    for d in out[..n].iter_mut() {
        if at + 4 > words.len() || at + 4 + words[at + 3] as usize > words.len() {
            return Err(CommandError::CommandOutOfStep);
        }
        *d = Draft {
            slot: words[at],
            prompt: words[at + 1] != 0,
            depth: words[at + 2],
            follow: &words[at + 4..at + 4 + words[at + 3] as usize],
        };
        at += 4 + words[at + 3] as usize;
    }
    if at != words.len() {
        return Err(CommandError::CommandOutOfStep);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `draft commands carry every stream's head request`
    /// (mirror.zig:200).
    #[test]
    fn draft_commands_carry_every_streams_head_request() {
        let words = vec![];
        let reqs = [
            Draft { slot: 3, prompt: true, depth: 2, follow: &[9] },
            Draft { slot: 1, prompt: false, depth: 0, follow: &[4, 5] },
        ];
        let mut words = words;
        draft_words(&mut words, &reqs);
        let mut back: [Draft; 16] = [Draft { slot: 0, prompt: false, depth: 0, follow: &[] }; 16];
        let n = read_drafts(&words, &mut back).unwrap();
        assert_eq!(2, n);
        for (a, b) in reqs.iter().zip(back[..n].iter()) {
            assert_eq!(a.slot, b.slot);
            assert_eq!(a.prompt, b.prompt);
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.follow, b.follow);
        }
        assert_eq!(
            Err(CommandError::CommandOutOfStep),
            read_drafts(&words[..words.len() - 1], &mut back)
        );
    }

    /// Upstream's `window commands carry every window's rows and the
    /// digest` (mirror.zig:218).
    #[test]
    fn window_commands_carry_every_windows_rows_and_the_digest() {
        let wins = [
            Win { slot: 2, pending: 11, held: 2, tokens: &[] },
            Win { slot: 0, pending: 12, held: 0, tokens: &[5, 6, 7] },
        ];
        let mut words = vec![];
        window_words(&mut words, 0x0123456789abcdef, &wins);
        // the read windows point into `words` (their tokens): the
        // borrow ends at the block's close before the mutation
        {
            let mut back: [Win; 16] = [Win { slot: 0, pending: 0, held: 0, tokens: &[] }; 16];
            let (digest, n) = read_windows(&words, &mut back).unwrap();
            assert_eq!(0x0123456789abcdef, digest);
            assert_eq!(2, n);
            for (a, b) in wins.iter().zip(back[..2].iter()) {
                assert_eq!(a.slot, b.slot);
                assert_eq!(a.pending, b.pending);
                assert_eq!(a.held, b.held);
                assert_eq!(a.tokens, b.tokens);
            }
            assert_eq!(
                Err(CommandError::CommandOutOfStep),
                read_windows(&words[..words.len() - 1], &mut back)
            );
        }
        words[2] = 0;
        let mut back: [Win; 16] = [Win { slot: 0, pending: 0, held: 0, tokens: &[] }; 16];
        assert_eq!(Err(CommandError::CommandOutOfStep), read_windows(&words, &mut back));
    }
}
