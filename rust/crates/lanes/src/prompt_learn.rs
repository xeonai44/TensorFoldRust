//! --learn: a shared cut's kept state written to disk above the free-disk
//! floor and within the cap, on every rank.
//!
//! Port of `zig/src/core/prompt_learn.zig` (146 lines) — the single-rank
//! flow: upstream's `peer_*` slots (the two-Mac fabric's rank 1: its needs,
//! its reservations, its half of the deletions) are the fabric tranche's,
//! so the peer need reads 0 and its reservation steps wait; everything
//! else is the flow whole — the pending cleanup, the payload arithmetic,
//! the victim plan across this identity's states and the others' trees,
//! the reserve, the journal, the write, the index, and the bounded
//! backoff on every refusal.

use crate::learned_plan::{self, Candidate, Need};
use crate::learned_dirs;
use crate::prompt_cache::{Snapshots, Store, StoreError};
use crate::prompt_imprint;

impl<F: Snapshots> Store<F> {
    /// A shared cut's state written to disk once (--learn): later
    /// sessions read it back, after a restart too. Upstream: `learn`
    /// (prompt_learn.zig:10).
    pub fn learn(&mut self, entry: u64, starts: &[u32]) {
        if self.imprint.is_none() {
                       return;
        }
        let Some(index) = self.entries.iter().position(|e| e.id == entry) else {
            return;
        };
        if !self.entries[index].decode_spans.is_empty() {
            // Learned states are prompt arithmetic only: their key holds
            // no span map
            return;
        }
        if !self.family.writes_learned() {
                       return;
        }
        let (dir, root, cap) = {
            let im = self.imprint.as_ref().unwrap();
            (im.dir.clone(), im.root.clone(), im.cap)
        };
        let tokens = self.entries[index].tokens.clone();
        let at = self.entries[index].at;
        let bytes = self.entries[index].bytes;
        let key = prompt_imprint::key_of(&tokens);
        if self
            .imprint
            .as_ref()
            .is_some_and(|im| im.admission.waiting(&dir))
        {
            return;
        }
        if self.finish_pending().is_err() {
            self.imprint.as_mut().unwrap().admission.refuse(&dir);
            return;
        }
        let (has_key, total) = {
            let im = self.imprint.as_ref().unwrap();
            (im.has(key), im.total())
        };
        if has_key || bytes > cap {
                       return;
        }
        let payload_bytes =
            bytes.saturating_add(256 + (tokens.len() + starts.len()) as u64 * 4);
        let scratch = match self.imprint.as_ref().unwrap().index_scratch_bytes() {
            Ok(n) => n,
            // Upstream: the index unreadable stops the write outright
            Err(_) => return,
        };
        let disk_bytes = payload_bytes.saturating_add(scratch);
        if self
            .imprint
            .as_ref()
            .is_some_and(|im| im.admission.waiting(&dir))
        {
            return;
        }
        let local_need = match self
            .imprint
            .as_ref()
            .unwrap()
            .admission
            .shortfall(&dir, disk_bytes)
        {
            Some(n) => n,
            None => {
                self.imprint.as_mut().unwrap().admission.refuse(&dir);
                return;
            }
        };
        // (The peer's need of `disk_bytes - bytes` extra: the fabric
        // tranche's rank 1)
        let others = match self.other_candidates() {
            Ok(o) => o,
            Err(()) => {
                self.imprint.as_mut().unwrap().admission.refuse(&dir);
                return;
            }
        };
        let own_dir_bytes = match learned_dirs::bytes(&dir) {
            Ok(n) => n,
            Err(_) => return,
        };
        let other_bytes = match learned_dirs::total_others(&root, &dir) {
            Ok(n) => n,
            Err(_) => return,
        };
        let used = total.max(own_dir_bytes) + other_bytes;
        let cap_need = used.saturating_add(payload_bytes).saturating_sub(cap);
        let metas: Vec<(u64, u64, u64)> = {
            let im = self.imprint.as_ref().unwrap();
            im.metas
                .iter()
                .map(|m| (m.key, m.bytes, m.used))
                .collect()
        };
        let mut candidates: Vec<Candidate> = Vec::with_capacity(metas.len() + others.len());
        for (i, (meta_key, meta_bytes, meta_used)) in metas.iter().enumerate() {
            let local = match self.family.reclaim(&dir, *meta_key) {
                Some(Ok(n)) => n,
                Some(Err(_)) => return,
                None => 0,
            };
            candidates.push(Candidate {
                key: i as u64,
                local,
                peer: 0,
                cap: *meta_bytes,
                used: meta_used.saturating_add(1 << 63),
            });
        }
        for (i, other) in others.iter().enumerate() {
            candidates.push(Candidate {
                key: (metas.len() + i) as u64,
                local: other.local,
                peer: other.peer,
                cap: other.local,
                used: other.used,
            });
        }
        let current_count = metas.len();
        let victims = match learned_plan::choose(
            &candidates,
            Need {
                local: local_need,
                peer: 0,
                cap: cap_need,
            },
        ) {
            Some(v) => v,
            None => {
                self.imprint.as_mut().unwrap().admission.refuse(&dir);
                learning_paused(self.imprint.as_ref().unwrap().admission.floor);
                return;
            }
        };
        let keys: Vec<u64> = victims
            .iter()
            .map(|&victim| {
                if (victim as usize) < current_count {
                    metas[victim as usize].0
                } else {
                    others[victim as usize - current_count].id
                }
            })
            .collect();
        for (victim, victim_key) in victims.iter().zip(keys.iter()) {
            if (*victim as usize) < current_count {
                if self.unlearn_checked(*victim_key).is_err() {
                    self.imprint.as_mut().unwrap().admission.refuse(&dir);
                    return;
                }
            } else {
                // (The peer's half of another identity's deletion: the
                // fabric tranche's)
                if learned_dirs::remove_other(&root, &dir, *victim_key).is_err() {
                    self.imprint.as_mut().unwrap().admission.refuse(&dir);
                    return;
                }
            }
        }
        match learned_dirs::total_others(&root, &dir) {
            Ok(n) => self.imprint.as_mut().unwrap().others = n,
            Err(_) => return,
        }
        match self.imprint.as_mut().unwrap().admission.reserve(&dir, disk_bytes) {
            crate::learned_disk::Result_::Quiet => return,
            crate::learned_disk::Result_::Refused => {
                learning_paused(self.imprint.as_ref().unwrap().admission.floor);
                return;
            }
            crate::learned_disk::Result_::Ready => {}
        }
        let success = self.learn_write(entry, key, at, &tokens, starts, bytes, &dir, disk_bytes);
        self.imprint
            .as_mut()
            .unwrap()
            .admission
            .finish(&dir, disk_bytes, success);
    }

    /// The write itself, under the reservation: the journal, the family's
    /// files, the index, the journal cleared. Upstream: `learn`'s second
    /// half (prompt_learn.zig:108-122).
    #[allow(clippy::too_many_arguments)]
    fn learn_write(
        &mut self,
        entry: u64,
        key: u64,
        at: u32,
        tokens: &[u32],
        starts: &[u32],
        bytes: u64,
        dir: &str,
        _disk_bytes: u64,
    ) -> bool {
        let Some(index) = self.entries.iter().position(|e| e.id == entry) else {
            return false;
        };
        if self.imprint.as_ref().unwrap().begin_write(key).is_err() {
            return false;
        }
        // The family's write, over the entry's saved state
        let written = {
            let family = &mut self.family;
            let saved = &self.entries[index].saved;
            family.write(saved, dir, key)
        };
        let Some(written) = written else {
            // The family keeps nothing on disk after all: the journal
            // clears and the reservation finishes
            let _ = self.finish_pending();
            return false;
        };
        if let Err(problem) = written {
            if self.finish_pending().is_err() {
                note("learned cleanup remains pending".into());
            }
            note(format!(
                "learning {} tokens failed ({}); retry delayed",
                at, problem
            ));
            return false;
        }
        if let Err(problem) = self
            .imprint
            .as_mut()
            .unwrap()
            .add(key, at, tokens, starts, bytes)
        {
            if self.finish_pending().is_err() {
                note("learned cleanup remains pending".into());
            }
            note(format!(
                "indexing {} learned tokens failed ({}); retry delayed",
                at, problem
            ));
            return false;
        }
        if self.imprint.as_ref().unwrap().clear_pending().is_err() {
            return false;
        }
        note(format!("learned {} tokens to disk", at));
        true
    }

    /// Learned state `key` forgotten: its files (the family's), then its
    /// index record. Upstream: `unlearn` (prompt_learn.zig:127).
    pub fn unlearn(&mut self, key: u64) {
        if let Err(problem) = self.unlearn_checked(key) {
            note(format!("forgetting a learned state failed ({problem})"));
        }
    }

    /// Upstream: `unlearnChecked` (prompt_learn.zig:130).
    fn unlearn_checked(&mut self, key: u64) -> Result<(), StoreError> {
        let dir = self.imprint.as_ref().map(|im| im.dir.clone()).unwrap_or_default();
        match self.family.forget_checked(&dir, key) {
            Some(result) => result?,
            None => self.family.forget(&dir, key),
        }
        self.imprint
            .as_mut()
            .unwrap()
            .remove(key)
            .map_err(|_| StoreError::DeleteFailed)
    }

    /// The crash journal's cleanup: a pending write whose key the index
    /// never gained removes its half-written files. Upstream:
    /// `finishPending` (prompt_learn.zig:134).
    fn finish_pending(&mut self) -> Result<(), ()> {
        let im = self.imprint.as_ref().ok_or(())?;
        let dir = im.dir.clone();
        let key = im.pending_write().map_err(|_| ())?;
        let Some(key) = key else {
            return Ok(());
        };
        if !self.imprint.as_ref().unwrap().has(key) {
            match self.family.forget_checked(&dir, key) {
                Some(Ok(())) => {}
                // An unchecked cleanup refuses (prompt_learn.zig:139's
                // UncheckedLearnedCleanup)
                _ => return Err(()),
            }
        }
        self.imprint
            .as_mut()
            .unwrap()
            .clear_pending()
            .map_err(|_| ())
    }

    /// The other identities as plan candidates. Upstream:
    /// `otherCandidates` (prompt_learn.zig:142) — the peer's other rank
    /// walks beside the directories; the fabric tranche's.
    fn other_candidates(&self) -> Result<Vec<Other>, ()> {
        let im = self.imprint.as_ref().ok_or(())?;
        let mut out = Vec::new();
        let mut cursor: Option<u64> = None;
        loop {
            let local = learned_dirs::next(&im.root, &im.dir, cursor).map_err(|_| ())?;
            let Some(id) = local else {
                break;
            };
            let own_bytes = learned_dirs::other_bytes(&im.root, &im.dir, id).map_err(|_| ())?;
            let used = learned_dirs::used(&im.root, id).map_err(|_| ())?;
            out.push(Other {
                id,
                local: own_bytes,
                peer: 0,
                used,
            });
            cursor = Some(id);
        }
        Ok(out)
    }
}

struct Other {
    id: u64,
    local: u64,
    peer: u64,
    used: u64,
}

/// Upstream: `learningPaused` (prompt_learn.zig:124).
fn learning_paused(floor: u64) {
    note(format!(
        "learning paused: free disk cannot preserve the {} MiB floor",
        floor >> 20
    ));
}

/// A line on the engine's log; tests stay quiet. Upstream: `note`
/// (prompt_cache.zig:509).
pub(crate) fn note(message: String) {
    if cfg!(test) {
        return;
    }
    eprintln!("prompt cache: {message}");
}

#[cfg(test)]
mod tests {
    //! The learned-disk fault suite (learned_fault_test.zig), the
    //! single-rank cases: the null clock, the cap-planned sweep, the
    //! missing half's reclaim, the pending crash's recovery, the torn
    //! used stamps. The peer-specific case (the uncertain reserve ACK —
    //! `pairReserve`'s lost acknowledgment) and the unlink/admission
    //! cases ride with the fabric tranche and the disk module's own tests.

    use super::*;
    use crate::prompt_cache::Rules;
    use crate::prompt_cache_fake::Fake;
    use crate::prompt_imprint::Imprint;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static TICK: AtomicU64 = AtomicU64::new(100);
    static NULL_CLOCK: AtomicBool = AtomicBool::new(false);

    fn disk_clock() -> Option<u64> {
        (!NULL_CLOCK.load(Ordering::Relaxed)).then(|| TICK.load(Ordering::Relaxed))
    }
    fn disk_free(_: &str) -> Option<u64> {
        Some(1 << 30)
    }

    fn scratch(suffix: &str) -> String {
        let path = std::env::temp_dir()
            .join(format!("tf-learn-fault-{suffix}-{}", std::process::id()));
        path.to_str().unwrap().to_string()
    }

    fn rm_tree(root: &str) {
        // The tests' scratch trees: two levels, the identity files first
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if std::fs::remove_file(&path).is_ok() {
                continue;
            }
            if let Ok(inner) = std::fs::read_dir(&path) {
                for f in inner.flatten() {
                    let _ = std::fs::remove_file(f.path());
                }
            }
            let _ = std::fs::remove_dir(&path);
        }
        let _ = std::fs::remove_dir(root);
    }

    fn put(dir: &str, name: &str, count: usize) {
        let path = std::path::Path::new(dir).join(name);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .expect("the scratch file");
        f.set_len(count as u64).expect("the scratch size");
    }

    /// One store over the disk-slot fake (standing at the keep's mark,
    /// upstream's `.{ .at = 5 }`), its imprint under `root`.
    fn learned_store(
        root: &str,
        identity: u64,
        cap: u64,
        fake: &Fake,
        floor: u64,
        free: fn(&str) -> Option<u64>,
    ) -> Store<Fake> {
        fake.0.borrow_mut().at = 5;
        let mut im = Imprint::open(root, identity, cap).expect("the imprint opens");
        im.admission = crate::learned_disk::Admission::with_readers(free, disk_clock);
        im.admission.floor = floor;
        Store::new(
            fake.clone(),
            Rules {
                min_prompt: 0,
                min_gap: 1,
                lookahead: 1,
                ..Default::default()
            },
            1 << 20,
        )
        .with_imprint(im)
    }

    const PROMPT: [u32; 8] = [7, 7, 7, 7, 7, 7, 9, 10];

    /// Upstream's `clock` case (learned_fault_test.zig:36): a null clock
    /// cannot authorize eviction or either half of a write.
    #[test]
    fn a_null_clock_cannot_authorize_eviction_or_either_half_of_a_write() {
        let root = scratch("clock");
        let fake = Fake::default().learned();
        let mut s = learned_store(&root, 1, 440, &fake, 0, disk_free);
        {
            let im = s.imprint_mut().unwrap();
            im.add(7, 1, &[1, 2], &[], 105).unwrap();
        }
        NULL_CLOCK.store(true, Ordering::Relaxed);
        let _ = s.lookup(&PROMPT, 7, &[5], &[], &[]);
        assert!(s.keep(&PROMPT, 5, std::ptr::null_mut(), &[], &[]));
        NULL_CLOCK.store(false, Ordering::Relaxed);
        assert!(s.imprint().unwrap().has(7));
        assert_eq!(fake.0.borrow().disk_forgets, 0);
        assert_eq!(fake.0.borrow().writes, 0);
        rm_tree(&root);
    }

    /// Upstream's `others` case (learned_fault_test.zig:80): other-identity
    /// cap pressure is planned before a current-identity write.
    #[test]
    fn other_identity_cap_pressure_is_planned_before_a_current_identity_write() {
        let root = scratch("others");
        {
            let old = Imprint::open(&root, 1, 1 << 20).unwrap();
            put(&old.dir, "state.bin", 300);
        }
        let fake = Fake::default().learned();
        let mut s = learned_store(&root, 2, 440, &fake, 0, disk_free);
        assert_eq!(s.imprint().unwrap().others, 308);
        let _ = s.lookup(&PROMPT, 7, &[5], &[], &[]);
        assert!(s.keep(&PROMPT, 5, std::ptr::null_mut(), &[], &[]));
        assert_eq!(
            learned_dirs::other_bytes(&root, &s.imprint().unwrap().dir, 1).unwrap(),
            0,
            "the other identity swept"
        );
        assert_eq!(fake.0.borrow().writes, 1);
        assert!(s.imprint().unwrap().has(crate::prompt_imprint::key_of(&PROMPT[..6])));
        rm_tree(&root);
    }

    /// Upstream's `missing` case (learned_fault_test.zig:133): a missing
    /// indexed half cannot promise physical reclaim or destroy a valid
    /// victim.
    #[test]
    fn a_missing_indexed_half_cannot_promise_physical_reclaim() {
        static AVAILABLE: AtomicU64 = AtomicU64::new(0);
        fn space_free(_: &str) -> Option<u64> {
            Some(AVAILABLE.load(Ordering::Relaxed))
        }
        let root = scratch("missing");
        let fake = Fake::default().learned();
        let mut s = learned_store(&root, 1, 1 << 20, &fake, 100, space_free);
        {
            let im = s.imprint_mut().unwrap();
            im.add(7, 1, &[1, 2], &[], 100).unwrap();
            im.add(8, 1, &[3, 4], &[], 100).unwrap();
            put(&im.dir, &format!("{:016x}.bin", 8u64), 100);
        }
        let scratch_bytes = s.imprint().unwrap().index_scratch_bytes().unwrap();
        AVAILABLE.store(100 + 105 + 256 + 24 + scratch_bytes - 150, Ordering::Relaxed);
        let _ = s.lookup(&PROMPT, 7, &[5], &[], &[]);
        assert!(s.keep(&PROMPT, 5, std::ptr::null_mut(), &[], &[]));
        let im = s.imprint().unwrap();
        assert!(im.has(7) && im.has(8));
        assert_eq!(fake.0.borrow().disk_forgets, 0);
        assert_eq!(fake.0.borrow().writes, 0);
        assert_eq!(crate::prompt_cache::single_reclaim(&im.dir, 8).unwrap(), 100);
        rm_tree(&root);
    }

    /// Upstream's `pending` case (learned_fault_test.zig:185): a failed
    /// rollback deletion survives a restart and recovers before cap
    /// admission. (Upstream drives the failure through the peer write;
    /// the port's single-rank equivalent is the family's own failed write
    /// with a failed checked deletion.)
    #[test]
    fn a_failed_rollback_deletion_survives_restart_and_recovers() {
        let root = scratch("pending");
        let key = crate::prompt_imprint::key_of(&PROMPT[..6]);
        for round in 0..2 {
            let fail = round == 0;
            let fake = Fake::default().learned();
            fake.0.borrow_mut().fail_write = fail;
            fake.0.borrow_mut().fail_delete = fail;
            let mut s = learned_store(&root, 1, 400, &fake, 0, disk_free);
            let _ = s.lookup(&PROMPT, 7, &[5], &[], &[]);
            assert!(s.keep(&PROMPT, 5, std::ptr::null_mut(), &[], &[]));
            assert_eq!(fake.0.borrow().writes, 1);
            let im = s.imprint().unwrap();
            if round == 0 {
                // The write failed, the cleanup failed too: the journal
                // stands, the index never gained the key
                assert_eq!(im.pending_write().unwrap(), Some(key));
                assert!(!im.has(key));
            } else {
                // The restart recovered: the journal cleared before the
                // cap admission, the write went through
                assert_eq!(im.pending_write().unwrap(), None);
                assert!(im.has(key));
            }
            assert_eq!(im.admission.reserved, 0);
        }
        rm_tree(&root);
    }

    /// Upstream's `used` case (learned_fault_test.zig:213): the store
    /// learns after old zero-byte and torn used stamps under cap
    /// pressure — a torn recency hint reads oldest, never invalid.
    #[test]
    fn the_store_learns_after_old_zero_byte_and_torn_used_stamps() {
        for length in 0..8 {
            let root = scratch(&format!("used-{length}"));
            {
                let old = Imprint::open(&root, 1, 1 << 20).unwrap();
                put(&old.dir, "state.bin", 300);
                put(&old.dir, "used", length);
            }
            let fake = Fake::default().learned();
            let mut s = learned_store(&root, 2, 440, &fake, 0, disk_free);
            let _ = s.lookup(&PROMPT, 7, &[5], &[], &[]);
            assert!(s.keep(&PROMPT, 5, std::ptr::null_mut(), &[], &[]));
            assert_eq!(fake.0.borrow().writes, 1, "length {length}");
            assert!(
                s.imprint()
                    .unwrap()
                    .has(crate::prompt_imprint::key_of(&PROMPT[..6])),
                "length {length}"
            );
            assert_eq!(
                learned_dirs::other_bytes(&root, &s.imprint().unwrap().dir, 1).unwrap(),
                0,
                "length {length}"
            );
            rm_tree(&root);
        }
    }
}

