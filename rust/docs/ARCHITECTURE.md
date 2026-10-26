# Architecture

The port mirrors upstream's architecture. The server never touches a GPU; it drives an
`Engine` through a narrow contract, and engines live behind that contract. Everything
below cites the Zig sources it ports; `docs/ZIG-TO-RUST-MAP.md` tracks status per
module.

## Crates and their upstream sources

| Crate | Upstream Zig | What it holds |
|---|---|---|
| `tensorfold-engine` | `src/core/engine_api.zig`, `src/core/lanes/sampling.zig`, `lanes/logprob.zig` | The contract: `Engine` trait, `Request`/`Event`/`Sink`, `Open`/`Backend`, the keyed sampler (`uniform`, numpy-order `pairwise_sum`, `choose` on libm), the sampling seed bits. |
| `tensorfold-safetensors` | `src/core/safetensors.zig` | The format's rules as upstream applies them: dtypes (incl. F8_E8M0), the rank ceiling, the shape/offset refusals, and `parseHeaderPrefix`'s namespace selection — shared by the host namespace and the Metal-resident shards (ADR-010). |
| `tensorfold-core` | `src/core/checkpoint.zig`, `src/core/tokenizer/`, the config reads | The host namespace (index weight map, leftovers refused) over the safetensors crate; tokenizer wrapper over the `tokenizers` crate; `config.json`/`generation_config.json` reads. |
| `tensorfold-metal` | `src/metal/` + `build.zig`'s pipeline + `src/core/checkpoint_metal.zig` | Device, queue, buffers; the compute encoder; runtime kernel compilation with MLX's options (ADR-011); the build-time metallib pipeline; the Metal-resident checkpoint (one buffer a shard, selected namespace, 8-thread pread). |
| `tensorfold-qwen35` | `src/families/qwen3_5/` + `kernels/metal/qwen3_5/` | The family: config admission gate, weight views, the frozen kernel registry (ADR-005), state (caches, scratch, snapshots), the forwards, `draw`, and the qualify exactness check — its integration test is the Phase 1a gate. |
| `tensorfold-server` | `src/server/` skeleton: `server.zig` (Config), `auth.zig`, model discovery | axum routes: `/v1/models` (window extensions when the engine reports a finite one), auth, metrics stub, and the engine-less inference refusals. |
| `tensorfold-native` | `src/server/main.zig`, `server/cli.zig`, `hub.zig` (served names), `serve.zig` front half, `native/none.zig` | The binary: `capabilities --json` (the same document shape), `serve MODEL [flags]` up to the engine open, the `NoneBackend` refusal. |

Dependency direction (arrows point at what a crate uses):

```
tensorfold-native ──► tensorfold-server ──► tensorfold-engine
       │                    │                    ▲
       └──────────► tensorfold-qwen35 ───────────┤
                          │  ▲                  │
              tensorfold-metal ──► tensorfold-safetensors
                          ▲        ▲
              tensorfold-core ─────┘ (namespace + tokenizer + shard list)
```

`kernels/metal/` is not a crate: it is upstream's kernel tree, byte-identical,
embedded by the family registries (`include_str!`) and compiled at runtime, plus the
`runtime_tests.metal` metallib the build compiles for the metal crate's own tests.

## The engine contract

Port of `engine_api.zig` — the seam the whole product hangs on:

* **`Engine` trait** (engine_api.zig:268): required `info`, `submit`, `cancel`,
  `status`, `memory`; the Zig vtable's optional slots (`keepalive`, `score`, `learn`)
  are default methods returning "unsupported" until a family implements them.
* **`EventSink`** (engine_api.zig:112): the port of `Sink`. The upstream contract
  carries over verbatim: events fire **on the engine's thread**, the sink **must return
  at once and must not re-enter the engine**. Borrowed rows (`Event::Tokens`,
  `Event::Logprobs`) are valid only during the call — Rust's lifetimes enforce what
  the Zig comment states.
* **`Request<'a>`** (engine_api.zig:46): all request fields, including the stop-string
  callback (`StopCheck` trait for the ctx+fn pair), call gates, grammar structures,
  thinking budget and logprobs. The engine may not keep any borrow past `submit`.
* **`Open`/`Backend`/`Opened`** (engine_api.zig:162-192, engines.zig, none.zig): what
  the server asks at startup. `NoneBackend` is the engine-less build, with upstream's
  refusal message.
* **Sampling bits** (lanes/sampling.zig): `sample_seed_for` (SHA-256 over
  `"t1,t2,...|salt"`, first 8 digest bytes LE, masked to 63 bits) and the splitmix64
  `mix`/`key_bits`, ported bit-exactly and pinned by the same tests upstream pins.

Phase 0 runs the contract engine-less, exactly like upstream's parity-suite build
(`zig/tests/server/build.sh`). The first real engine (Phase 1) implements the trait for
the Qwen3.5-2B family on Metal.

## Execution model

Upstream: a hand-rolled HTTP server, one thread a connection (server.zig:217), driving
an engine that runs its own lane-host thread (lane_host.zig). The port keeps the
split but swaps the HTTP half for axum/tokio (ADR-004): the async layer only
translates requests and events; GPU work never runs on a tokio worker. The Phase 1
engine runs its own thread — the `LaneHost` shape — and bridges to the async layer
through the `EventSink`; that bridge is the first Phase 1 deliverable, so no inference
code grows inside the async runtime by accident.

## The kernel pipeline

`build.rs` ports `build.zig`'s pipeline (lines 70-97, 246-256):

* `mlxMetalStd`: macOS 27 → `metal4.1`, 26 → `metal4.0`, 15 → `metal3.2`, older →
  `metal3.1`.
* `xcrun -sdk macosx metal <flags> -c SRC -o X.air`, then
  `xcrun -sdk macosx metallib -o X.metallib X.air`, with upstream's flags for
  hand-written kernels (`-fmetal-math-mode=safe -fmetal-math-fp32-functions=fast`).
* The metallib is embedded in the binary (`include_bytes!`), the way upstream
  `@embedFile`s its metallibs (build.zig:99).
* Without the Metal toolchain the build still succeeds: a placeholder metallib is
  embedded, `kernels_builtin()` returns false, GPU tests skip with a notice, and
  engine opens refuse — the same "engine-less build" posture upstream has.

Phase 0 compiles `runtime_tests.metal` (upstream's own runtime test kernels) and runs
its `tf_vadd` on the GPU in the metal crate's tests. Phase 1 extends the pipeline to
the shared `ops/` and `core/` kernels, then the family metallibs, in the same order
upstream's build declares them.

## What is deliberately not ported yet

See `docs/PORTING-PLAN.md` for the phases; the short list: the comptime kernel
registries (replaced by a runtime registry, ADR-005), the lane core and engines
(Phase 1), the chat/Anthropic translation layers, compaction, hub/cache management
and pull (Phase 2), Sliding/Living Weights and the two-Mac fabric (Phase 3).
