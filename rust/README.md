# TensorFold (Rust port)

A Rust port of [TensorFold](https://github.com/ashhart/TensorFold), the LLM inference
engine that serves language models from a native binary on Apple Silicon and NVIDIA
GPUs. This port starts from upstream's 1.0.4 Zig engine (see `docs/ZIG-TO-RUST-MAP.md`
for what the port covers and what still waits).

**Status: the port serves.** Phase 1b is complete end to end: the engine
contract, the checkpoint/tokenizer readers, the Qwen3.5-2B Metal family, the full
lane round loop with the qwen35 backend behind it, the lane host (the engine
thread), the chat template (minijinja + the Python string-method shim), and the
chat route — the binary answers `/v1/chat/completions` on the real pinned
checkpoint, greedy and sampled, streaming included:

```
tensorfold-native serve /Volumes/llm/models/Qwen3.5-2B-MLX-4bit --name local-model
curl -fsS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

The exactness gates all pass: upstream's own qualify check on the real weights,
upstream's engine tests mirrored case by case (drafted == plain == shared ==
serial), the end-to-end serve gate (`cargo test --release -p tensorfold-native
--test serve_e2e`), and — the final referee — **the golden harness**
(`cargo test --release -p tensorfold-native --test golden`): the upstream Zig
engine and this port, built on the same machine, both serve the pinned
checkpoint, and eight probes (greedy, sampled with derived and explicit seeds,
multi-turn) answer **byte-identically**. Phase 2 (tools, reasoning bodies,
anthropic, pull, more families, CUDA) is scoped in `docs/ZIG-TO-RUST-MAP.md`.

## Layout

```
crates/
  engine/    tensorfold-engine       the contract a server drives (engine_api.zig)
  safetensors/ tensorfold-safetensors the format's rules, dtype by dtype
  core/      tensorfold-core         namespace, tokenizer, config readers
  metal/     tensorfold-metal        Metal host layer: device, encoder, runtime
                                      kernel compilation, Metal-resident checkpoint
  lanes/     tensorfold-lanes         the lane round loop (stream, windows, depth
                                      rule, planner, proposer, engine, fake)
  qwen35/    tensorfold-qwen35       the Qwen3.5-2B family (config gate, weights,
                                      kernels, state, forwards, qualify, backend)
  server/    tensorfold-server       HTTP server: config, auth, routes
  cli/       tensorfold-native       the binary (serve, capabilities)
kernels/
  metal/     upstream's Metal kernels, copied byte-identical (249 files)
docs/        architecture, porting plan, decision records, Zig→Rust map
```

## Build and test

```
cargo build                # debug build (needs Xcode's Metal toolchain for kernels;
                           # without it the build still succeeds, GPU tests skip)
cargo test                 # 70 tests, including real-GPU exactness gates
cargo build --release     # then the end-to-end gate:
cargo test --release -p tensorfold-native --test serve_e2e -- --nocapture
cargo clippy --workspace   # clean
./target/debug/tensorfold-native capabilities --json

# The Phase 1a gate: the pinned checkpoint on the GPU (QWEN35_MODEL_DIR overrides the
# model directory; without a model or Metal device the test skips with a notice):
cargo test -p tensorfold-qwen35 --test qualify -- --nocapture

# The Phase 1b gates: the lane round loop's exactness on the fake target, and the
# real-model gate where the lane core drives the Metal backend:
cargo test -p tensorfold-lanes --test engine -- --nocapture
cargo test -p tensorfold-qwen35 --test lanes -- --nocapture
```

The pinned checkpoint is `mlx-community/Qwen3.5-2B-MLX-4bit` at revision
`93760be4f1f69842a46bc13dbdc0f19e291392a3`, downloaded with per-file hash
verification (sha256 for LFS blobs, git-blob sha1 for the rest — upstream `pull`'s
own contract).

Serving a model requires a complete local model directory (a `config.json`, a
`tokenizer.json` and safetensors shards). Until the lane core lands in Phase 1b the
binary still refuses to serve — the same words the upstream engine-less build uses —
so behaviour is honest at every step.

## How the port works

The port follows one rule: **never guess**. Every ported declaration cites its Zig
source (`file.zig:line`), every stub marks the phase that fills it, and upstream's
kernel sources are copied verbatim and compiled by a `build.rs` that mirrors
`build.zig`'s pipeline (same `xcrun metal` flags, same macOS-version-dependent Metal
language standard). Read `docs/DECISIONS.md` for the choices behind the layout, the
crates substituted for hand-rolled Zig code, and the exactness gates every phase must
pass before the next begins.

## License

Apache-2.0, like upstream. This is a derivative work of TensorFold; see `LICENSE` and
`NOTICE`.
