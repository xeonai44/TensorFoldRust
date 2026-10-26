//! Milliseconds by an integer key (rows or drafts): the int -> float dicts the
//! depth rule reads in Python.
//!
//! Port of `zig/src/core/lanes/table.zig`.

/// A table of `f64` slots keyed by small non-negative integers (rows, drafts).
///
/// Upstream: `table.zig:5  pub const Table = struct { slots: []?f64 }`.
#[derive(Default)]
pub struct Table {
    slots: Vec<Option<f64>>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Upstream: `get`.
    pub fn get(&self, key: i64) -> Option<f64> {
        if key < 0 || key as usize >= self.slots.len() {
            return None;
        }
        self.slots[key as usize]
    }

    /// Set `key` (growing the table): keys are small non-negative ints.
    /// Upstream: `put`.
    pub fn put(&mut self, key: i64, value: f64) {
        debug_assert!(key >= 0);
        let at = key as usize;
        if at >= self.slots.len() {
            self.slots.resize(at + 1, None);
        }
        self.slots[at] = Some(value);
    }

    /// Python's `not costs`: no key set. Upstream: `empty`.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|slot| slot.is_none())
    }

    /// The slots in key order, for the trail's table listing.
    pub fn slots(&self) -> &[Option<f64>] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors table.zig:37 "table get and put".
    #[test]
    fn table_get_and_put() {
        let mut t = Table::new();
        assert!(t.is_empty());
        t.put(3, 1.5);
        assert_eq!(t.get(3), Some(1.5));
        assert_eq!(t.get(2), None);
        assert_eq!(t.get(-1), None);
        assert!(!t.is_empty());
    }
}
