//! TensorFold's tokenizer for Hugging Face tokenizer.json files.
//!
//! Upstream contract (`zig/src/core/tokenizer/tokenizer.zig:1`): "encode equals
//! tokenizers' encode(text, add_special_tokens=False)". The Zig engine reimplements the
//! `tokenizers` library by hand — including an Oniguruma-compatible backtracking regex
//! — because it must load with no Python. This port uses the `tokenizers` crate itself,
//! which *is* that reference implementation (decision record: docs/DECISIONS.md,
//! ADR-003). The Phase 1 exactness gate compares its outputs against the Zig engine's on
//! every qualified model's tokenizer.json before the first engine lands; if any drift
//! appears, the hand-rolled port becomes the fallback and the crate goes.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum TokenizerError {
    /// `path/tokenizer.json` and `path` itself are both absent or unopenable.
    Open(PathBuf),
    /// The file is not a tokenizer.json this build accepts.
    Parse(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Open(path) => {
                write!(f, "no tokenizer.json at {}", path.display())
            }
            TokenizerError::Parse(message) => write!(f, "tokenizer.json: {}", message),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// The tokenizer a server and an engine share.
use tokenizers::Decoder;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab", &self.inner.get_vocab_size(false))
            .finish()
    }
}

impl Tokenizer {
    /// Loads `path/tokenizer.json`, or `path` itself when it names a .json file.
    ///
    /// Upstream: `tokenizer.zig:210`.
    pub fn open(path: &Path) -> Result<Tokenizer, TokenizerError> {
        let direct = path
            .extension()
            .map(|e| e == "json")
            .unwrap_or(false);
        let file = if direct {
            path.to_path_buf()
        } else {
            path.join("tokenizer.json")
        };
        if !file.is_file() {
            return Err(TokenizerError::Open(file));
        }
        let inner = tokenizers::Tokenizer::from_file(&file)
            .map_err(|e| TokenizerError::Parse(e.to_string()))?;
        Ok(Tokenizer { inner })
    }

    /// Encode text as token ids: `tokenizers`' `encode(text, add_special_tokens=False)`,
    /// the exactness contract's reference behavior.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// Encode with the post-processor's specials (`encode(text, add_special_tokens=True)`).
    /// Upstream: `encodeFn` (hf_text.zig:225) — the hand-rolled prefix/suffix
    /// read from tokenizer.json's post_processor; the crate runs that very
    /// processor, the reference behavior the Zig side imitates.
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// Decode ids to text. Upstream: the one shared `decodeFn` (hf_text.zig:233)
    /// — special tokens stay as their own text (`skip_special_tokens=false`).
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner
            .decode(ids, false)
            .unwrap_or_else(|_| "\u{fffd}".into())
    }

    /// The vocabulary size, for engines that allocate embedding tables.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// A special token's id: added tokens and the vocab, `None` when the tokenizer
    /// knows no such piece. Upstream: `hf_text.zig`'s `specialTokenId`.
    pub fn special_token_id(&self, piece: &str) -> Option<u32> {
        self.inner.token_to_id(piece)
    }

    /// A token's vocabulary piece (the id's own string), or None when the
    /// tokenizer knows no such piece. Upstream: `tokenString`
    /// (hf_text.zig:240).
    pub fn token_string(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Whether a token is one of the tokenizer's added special tokens.
    /// Upstream: `isSpecial` (hf_text.zig:215) — the piece in the specials
    /// set; the crate's added-tokens decoder carries the flag.
    pub fn is_special(&self, id: u32) -> bool {
        self.inner
            .get_added_tokens_decoder()
            .get(&id)
            .is_some_and(|t| t.special)
    }

    /// Exact, possibly incomplete UTF-8 bytes of one token; `None` when the
    /// tokenizer is context-dependent (no decoder to run on a piece alone).
    /// Upstream: `tokenBytes` (tokenizer.zig:198) — a decoder local to the
    /// token, the special its own text; the port runs the crate's decoder on
    /// the piece (the byte-level alphabet maps each char to its byte).
    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let piece = self.inner.id_to_token(id)?;
        let decoder = self.inner.get_decoder()?;
        let text = decoder.decode(vec![piece]).ok()?;
        Some(text.chars().map(|c| c as u8).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny BPE tokenizer.json written inline, the way HF saves them: a BPE with one
    /// merge, plus a TemplateProcessing post-processor that adds `<s>` and `</s>` —
    /// so the test proves encode's `add_special_tokens=false` skips what a request
    /// must not see.
    const TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null, "padding": null,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "added_tokens": [
            {"id": 4, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 5, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [4], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [5], "tokens": ["</s>"]}
            }
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false,
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "<s>": 4, "</s>": 5},
            "merges": ["a b"]
        }
    }"#;

    fn fixture(tag: &str) -> Tokenizer {
        // One directory a test: the tokenizer tests run in parallel and share /tmp.
        let dir = std::env::temp_dir().join(format!("tf-tokenizer-test-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        Tokenizer::open(&dir).unwrap()
    }

    #[test]
    fn encode_adds_no_special_tokens() {
        let t = fixture("encode");
        let ids = t.encode("ab");
        assert_eq!(ids, vec![3]); // the merge, not <s> ... </s>
        assert_eq!(t.decode(&ids), "ab");
    }

    /// `encode_special` wraps with the post-processor's specials (hf_text.zig:225's
    /// prefix/suffix around the encode).
    #[test]
    fn encode_special_wraps_the_processors_specials() {
        let t = fixture("special");
        assert_eq!(t.encode_special("ab"), vec![4, 3, 5]);
    }

    /// Decode keeps flagged specials as their own text (hf_text.zig:233's
    /// `skip_special_tokens=false`).
    #[test]
    fn decode_keeps_specials() {
        let t = fixture("decode");
        assert_eq!(t.decode(&[4, 3, 5]), "<s>ab</s>");
    }

    #[test]
    fn open_accepts_a_direct_json_path() {
        let t = fixture("path");
        assert!(t.vocab_size() > 0);
        let nowhere = std::env::temp_dir()
            .join(format!("tf-tokenizer-test-{}-path", std::process::id()))
            .join("nowhere.json");
        assert!(Tokenizer::open(&nowhere).is_err());
    }
}
