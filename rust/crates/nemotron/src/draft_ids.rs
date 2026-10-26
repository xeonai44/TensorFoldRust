//! Nemotron's draft vocabulary: the Python package's frequency-ranked
//! token ids the MTP head drafts among.
//!
//! Port of `zig/src/families/nemotron/draft_ids.zig` (32 lines) — the
//! ids embedded from the file beside the module (upstream's generated
//! `nemotron_draft_ids` import).

/// Whitespace-separated ids, each below `vocab`, a multiple of 64 of
/// them (the draft head is tiled by 64 rows). Upstream: `parse`
/// (draft_ids.zig:5).
pub fn parse(text: &str, vocab: usize) -> Result<Vec<u32>, crate::error::Error> {
    use crate::error::Error;
    let mut ids: Vec<u32> = Vec::new();
    for tok in text.split([' ', '\n', '\r', '\t']) {
        if tok.is_empty() {
            continue;
        }
        let v: u32 = tok.parse().map_err(|_| Error::BadConfig)?;
        if v as usize >= vocab {
            return Err(Error::DraftIdPastVocabulary);
        }
        ids.push(v);
    }
    if ids.is_empty() || ids.len() % 64 != 0 {
        return Err(Error::DraftIdsNotTiled);
    }
    Ok(ids)
}

/// The draft vocabulary (draft_ids.txt beside this file), embedded at
/// build time. Upstream: `load` (draft_ids.zig:19).
pub fn load(vocab: usize) -> Result<Vec<u32>, crate::error::Error> {
    parse(include_str!("draft_ids.txt"), vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `draft ids parse and refuse` (draft_ids.zig:23).
    #[test]
    fn draft_ids_parse_and_refuse() {
        let mut text = [0u8; 64 * 3];
        for i in 0..64 {
            let cell = &mut text[i * 3..i * 3 + 3];
            write_cell(cell, i);
        }
        let text = String::from_utf8(text.to_vec()).unwrap();
        let ids = parse(&text, 64).unwrap();
        assert_eq!(63, ids[63]);
        assert!(matches!(parse(&text, 10), Err(crate::error::Error::DraftIdPastVocabulary)));
        assert!(matches!(parse("1 2 3", 10), Err(crate::error::Error::DraftIdsNotTiled)));
    }

    /// `{d:>2} `: the id right-aligned in two columns, a space after.
    fn write_cell(cell: &mut [u8], i: usize) {
        let s = format!("{i:>2} ");
        cell.copy_from_slice(s.as_bytes());
    }
}
