# AGENTS.md

Working commands for this workspace:

```
cargo build                 # must succeed with no new warnings
cargo test                  # all suites green (66 tests); GPU tests run on the
                            # Metal device when present, skip otherwise
cargo test -p tensorfold-qwen35 --test qualify -- --nocapture
                            # the Phase 1a gate: the pinned checkpoint's real weights
                            # through the qualify exactness check (QWEN35_MODEL_DIR
                            # overrides /Volumes/llm/models/Qwen3.5-2B-MLX-4bit)
cargo test -p tensorfold-lanes --test engine -- --nocapture
                            # the lane core's exactness on the fake target
cargo test -p tensorfold-qwen35 --test lanes -- --nocapture
                            # the Phase 1b gate: the lane core driving the Metal
                            # backend on the pinned checkpoint
cargo build --release && cargo test --release -p tensorfold-native --test serve_e2e -- --nocapture
                            # the end-to-end gate: the release binary serving the
                            # pinned checkpoint over HTTP
cargo test --release -p tensorfold-native --test golden -- --nocapture
                            # the golden harness: the Zig engine (the upstream
                            # clone at /Volumes/llm/tensorfold, built with its
                            # pinned Zig) and this port answer byte-identically
cargo clippy --workspace    # must be clean before any work is called done
./target/debug/tensorfold-native capabilities --json   # smoke: prints the capabilities document
```

The upstream Zig sources live at `/Volumes/llm/tensorfold` (a full clone of
`ashhart/TensorFold`). Treat them as the specification. The pinned test checkpoint
lives at `/Volumes/llm/models/Qwen3.5-2B-MLX-4bit` (hash-verified at download).

## Rules for porting work

1. **Never guess.** Every ported declaration carries a comment citing the Zig source it
   mirrors, in the form `Upstream: file.zig:LINE  decl`. If you cannot find the
   declaration to cite, find it before writing anything.
2. **Every stub is marked.** A placeholder carries `TODO(upstream: file.zig:LINE)` and
   names the phase in `docs/PORTING-PLAN.md` that fills it. No silent stubs.
3. **Kernels are never edited.** `kernels/metal/` is upstream's, byte-identical, and
   stays that way; the build pipeline compiles it. If a kernel needs a change, the
   change belongs upstream, and the copy updates from there.
4. **Tests mirror upstream tests.** When a Zig test pins behaviour (a golden value, a
   refusal, an exact message), the Rust test pins the same thing and says which Zig
   test it mirrors. Where upstream has no test, write one and say so.
5. **Keep the map current.** When a module's status changes (planned → ported), update
   `docs/ZIG-TO-RUST-MAP.md` in the same change.
6. **Decisions are recorded.** Any non-obvious choice gets an entry in
   `docs/DECISIONS.md` — context, decision, consequences — before it lands.
7. **Exactness gates come before features.** A phase is done when its exit criteria in
   `docs/PORTING-PLAN.md` pass, not when its code compiles.

## Conventions

* Rust edition 2024, stable toolchain (see `rust-toolchain.toml`).
* Crate names: `tensorfold-*`; the binary is `tensorfold-native`, upstream's name.
* No `unsafe` outside the two places that need it (the mmap'd shard borrows in
  `tensorfold-core`, the buffer contents in `tensorfold-metal`); each carries a
  safety comment.
* Module layout mirrors the Zig layout so cross-referencing is mechanical; see
  `docs/ZIG-TO-RUST-MAP.md` for the table.

## The lanes crate's pointer invariants

`tensorfold-lanes` mirrors upstream's pointer topology under documented invariants
(its `lib.rs`): one thread owns the engine; streams are boxed, live, and distinct
from admission to release. Any change touching stream ownership, the arena resets
(`step`, `add_stream`), or the Backend trait's contracts must re-read that
documentation and keep the engine tests green.
