# Zig → Rust map

Status per module. Line counts are the upstream Zig files (host code, excluding
tests) at the ported revision. Update this file in the same change that moves a
module's status.

## Ported (Phase 0)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/engine_api.zig` | 360 | `crates/engine/src/{engine,event,request,learn,backend}.rs` | The whole contract: Engine trait, Request/Event/Sink, Open/Backend, learner types. Cites line by line. |
| `src/core/lanes/sampling.zig` (seed parts) | ~40 | `crates/engine/src/sampling.rs` | `sample_seed_for`, `mix`, `key_bits` bit-exact; tests pin the same values. |
| `src/core/safetensors.zig` (parse rules) | ~260 | `crates/safetensors/src/{safetensors,dtype}.rs` | DType incl. F8_E8M0, size table, rank-5 ceiling, shape/offset refusals, `parse_header_prefix`; the same test cases. |
| `src/core/checkpoint.zig` (reads) | ~180 | `crates/core/src/checkpoint.rs` | `shard_files` (weight map, plain glob, name order, missing/leftover refusals) plus the host namespace over it. |
| `src/core/tokenizer/tokenizer.zig` (load contract) | load entry | `crates/core/src/tokenizer.rs` | Wrapper over the `tokenizers` crate (ADR-003); encode = `add_special_tokens=false`. The rest of `core/tokenizer/` (~7K lines) is **not ported** — the gate in Phase 1b decides whether it ever is. |
| `config.json`/`generation_config.json` reads | scattered | `crates/core/src/config.rs` | Folder-level reads; the family's typed config ports with the family (done — see qwen35). |
| `src/metal/` (essentials) | ~350 | `crates/metal/src/{device,pipeline}.rs`, `build.rs` | Device, queue, buffers, pipelines, dispatch; the build-time kernel pipeline with upstream's flags and MSL version rule. |
| `kernels/metal/runtime_tests.metal` | 38 | `kernels/metal/` (copied) | Compiled and `tf_vadd` runs on the GPU in tests. The whole `kernels/metal/` tree (249 files) is copied byte-identical. |
| `src/server/server.zig` (Config) | config half | `crates/server/src/lib.rs` | The Config fields; timeouts arrive with `http_conn.zig` (Phase 2). |
| `src/server/auth.zig` | ~100 | `crates/server/src/auth.rs` | Bearer check, key file, the same 401 refusal. |
| Model discovery routes | `routes.zig` part | `crates/server/src/routes.rs` | `/v1/models` with `context_length`/`max_model_len` when the window is finite; `/health`; `/metrics` stub. |
| `src/server/main.zig` + `server/cli.zig` (surface) | main + table | `crates/cli/src/main.rs`, `capabilities.rs` | `--version`, `capabilities --json` (same shape), usage line, exit-2 refusals. |
| `src/server/hub.zig` (served names) | 65 | `crates/cli/src/hub.rs` | `isRepoId`, `servedName`, with upstream's tests. Cache-side waits for Phase 2. |
| `src/server/serve.zig` (front half) | 85-125 | `crates/cli/src/serve.rs` | Keys from flags/file/env, the loopback warning, model ids trimmed and deduped, seed salt, the engine open refusal. |
| `src/native/none.zig` | 16 | `crates/engine/src/backend.rs` | `NoneBackend`, exact refusal message. |

## Ported (Phase 1a)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/checkpoint_metal.zig` | 220 | `crates/metal/src/checkpoint.rs` | Metal-resident shards: selected namespace, [min begin, max end) region, 8-thread pread, one buffer a shard, offset views; upstream's test mirrored. |
| `src/metal/library.zig` (fromSource + CompileOptions) | ~75 | `crates/metal/src/library.rs` | Runtime compile with MLX's options (`safe` math, `fast` fp32 functions, OS-mapped language version), the descriptor-path pipeline creation (ADR-011). |
| `src/metal/command.zig` (ComputeEncoder half) | ~60 | `crates/metal/src/encoder.rs` | ComputePass + Encoder: set_pipeline/set_buffer/set_value/dispatch/submit with the failure read. |
| `src/metal/objc.zig` (string, ownership) | ~90 | `crates/metal/src/objc.rs` | Alloc+init NSStrings (ADR-012), error text reads. |
| `src/core/lanes/sampling.zig` (sampler) | ~100 | `crates/engine/src/sampling.rs` | `uniform`, `pairwise_sum` (numpy order), `choose` (gumbel keyed draws) over libm `exp`/`log`; Sampling's defaults; upstream's tests mirrored. |
| `src/core/lanes/logprob.zig` | 47 | `crates/engine/src/logprob.rs` | `LogprobRow`, `from_words`, `for_token`. |
| `src/families/qwen3_5/config.zig` | 260 | `crates/qwen35/src/config.rs` | The full admission gate (pinned geometry, layer pattern, rotary, quantization, tiedness), upstream's fixture and both its tests mirrored. |
| `src/families/qwen3_5/weights.zig` | 105 | `crates/qwen35/src/weights.rs` | The affine view types and the whole load, tied head aliasing included. |
| `src/families/qwen3_5/kernels.zig` + `kernels/metal/qwen3_5/kernels.zig` | 70 + 19 | `crates/qwen35/src/kernels.rs` | The frozen 15-entry registry (ADR-005's realization) + the 5 layout kernels; runtime compile per entry; `dimensions` from the qmm keys. |
| `src/families/qwen3_5/state.zig` | 157 | `crates/qwen35/src/state.rs` | Storage/Cache/Scratch: window_rows 16, batch_rows 32, zeroed recurrence, commit and keep rolls, snapshots. |
| `src/families/qwen3_5/forward.zig` | 227 | `crates/qwen35/src/forward.rs` | One command buffer a forward: embed, per-layer delta/attention, norm, MLP, head; the exact grid arithmetic. |
| `src/families/qwen3_5/model.zig` | 46 | `crates/qwen35/src/model.rs` | Load order verbatim; `language_model.` namespace selection. |
| `src/families/qwen3_5/qualify.zig` | 65 | `crates/qwen35/src/qualify.rs` + `tests/qualify.rs` | The wide-vs-solo exactness check, run on the real pinned weights on the GPU (the Phase 1a gate). |
| `src/families/qwen3_5/backend.zig` (draw half) | ~40 | `crates/qwen35/src/backend.rs` | `value`, `draw` (top-k insertion loop verbatim), upstream's candidate-selection test mirrored over the full vocabulary. |

## Ported (Phase 1b, part 1)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/lanes/table.zig` | 46 | `crates/lanes/src/table.rs` | Slots by small int keys; upstream's test mirrored. |
| `src/core/lanes/events.zig` | 161 | `crates/lanes/src/events.rs` | The recorder's Value/Log; the JSON line pinned byte for byte. |
| `src/core/lanes/accept.zig` | 99 | `crates/lanes/src/accept.rs` | rowParents, depths, acceptPath, sanitizeTree; both tests mirrored. |
| `src/core/lanes/allocate.zig` | 124 | `crates/lanes/src/allocate.rs` | Allocation by expected tokens a ms; extendCosts; both tests mirrored. |
| `src/core/lanes/shape.zig` | 139 | `crates/lanes/src/shape.rs` | Odds with prior; lane trees by value; all three tests mirrored. |
| `src/core/lanes/fill.zig` | 140 | `crates/lanes/src/fill.rs` | Suffix candidates and grafts; all three tests mirrored. |
| `src/core/lanes/plan_lanes.zig` | 173 | `crates/lanes/src/plan_lanes.rs` | The lane planner's pricing; all three tests mirrored. |
| `src/core/lanes/proposer.zig` | 205 | `crates/lanes/src/proposer.rs` | SuffixLookup behind the Proposer trait; upstream's test mirrored. |
| `src/core/lanes/depth.zig` | 271 | `crates/lanes/src/depth.rs` | The depth rule: rates, costs, probes, budgets; the pinned test mirrored. |
| `src/core/lanes/config.zig` | 138 | `crates/lanes/src/config.rs` | Model facts → round constants; the widths test mirrored. |
| `src/core/lanes/stream.zig` | 393 | `crates/lanes/src/stream.rs` | The stream state machine, commit/stop/loop-guard; all four tests mirrored. |
| `src/core/lanes/backend.zig` | 180 | `crates/lanes/src/backend.rs` | The vtable as a trait; the optional slots as default methods; WallClock. |
| `src/core/lanes/windows.zig` | 238 | `crates/lanes/src/windows.rs` | plan/copyProposal/allocate/build/tokens; who and probe. |
| `src/core/lanes/engine.zig` | 599 | `crates/lanes/src/engine.rs` | The round loop: addStream/step/takeTurns/familyRound/roundStreams/conclude/askDrafts/pipelinedRound (ADR-013/014). |
| `src/core/lanes/fake.zig` | 272 | `crates/lanes/src/fake.rs` | The history-dependent test backend; save/drop for the reuse tests. |
| `src/core/lanes/engine_test.zig` | 220 | `crates/lanes/tests/engine.rs` | All six exactness cases, mirrored. |
| `src/core/lanes/trail.zig` | 96 | `crates/lanes/src/trail.rs` | The recorder's events, field for field. |
| `src/families/qwen3_5/backend.zig` (vtable half) | ~180 | `crates/qwen35/src/backend.rs` | prefill/first/queue/read/verify/keep/draft/release on the lanes Backend; the recorded shared-round verify with keep rolls. |

## Ported (Phase 2, fifty-second tranche — the GLM snapshots, mirrors and weight fixes)

The GLM family's second tranche: the checksummed snapshot envelope
(core, never ported — the learned halves' and the peer's underpinning),
the versioned GLM half-file over it, the speed-up-mode command codecs,
and the weight-load rewrite kernels.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/snapshot_file.zig` | 66 | `crates/core/src/snapshot_file.rs` | Learned snapshots carry a SHA-256 of their entire header and payload; exact length is required on admission — `write` (header, payload, digest, fsync), `read` (the exact size, the digest checked, the payload pread at the header's length). Upstream's one test mirrored over std::fs: bit flips, header changes, truncation, legacy files and extra bytes all refused. |
| `families/glm/snapshot_file.zig` | 45 | `crates/glm/src/snapshot_file.rs` | The GLM learned halves' versioned envelope: the 24-byte Head (magic GLMT, version 1, position, bytes), the atomic `.part`-then-rename write with the unlink on failure, the exact-head read (a changed position or bytes refused before the digest). Upstream's test mirrored. |
| `families/glm/mirror.zig` (the codec half) | 14-66, 200-240 | `crates/glm/src/mirror.rs` | Speed-up mode's slot commands, the word codecs: **`windowWords`/`readWindows`** (rank 0's digest, then each window's slot, pending, held count and host drafts — the drafts pointing into the words) and **`draftWords`/`readDrafts`** (each stream's slot, prompt-absorb flag, depth and follow), every out-of-step length refused. The 20-command `Kind`. Both upstream tests mirrored. The follow/apply halves (the engine's rank-1 loop, apply's command dispatch over Slots and the learned-disk commands) ride with the engine and slots tranche. |
| `families/glm/weights_fix.zig` | 81 | `crates/glm/src/weights_fix.rs` | The weight-load rewrite kernels, index arithmetic exact: **`hc_pack`** ([og 6][tm 4][i 16][sgn 8][lane 32][tn 4] -> [og][sgn][lane][i][tm][tn] — the HC indexer's packed mix), **`router_pack`** (packed[q][thrM][i][tm][tn] = router[4q+tn][32i+4thrM+tm]), **`conv`** (the [q|k|v][channel][tap] bf16 taps -> [tap][3·channels] fp32, the TP2 channel slice across three part tensors), `to_f32`, `cols` (the column slice), `transpose_bf16`. The port's `read_span`/reader abstraction replaces the fd preads; the conv's per-part buffers hold in their own scope (the borrow checker restructured upstream's stack buffer per part — the arithmetic unchanged). |

Gates: Mac 335 passed / 0 failed (the two snapshot suites, the mirror
codecs' two, the earlier GLM tests and the whole workspace), clippy
clean on both crates; box 328 / 0 — **the half-file test green with
real fsync'd files** on the box's ext4.

## Ported (Phase 2, fifty-first tranche — the GLM family's foundation)

The GLM-5.3-Flash family's first tranche: the exactly-pinned config
and the peer-disk reservation — both pure, both with upstream's own
tests. The `--device`/`--segments` flags (the CUDA path's own, the
fifty-first's follow-up) land here too.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/cli.zig`'s --device/--segments | 255-266 | `crates/cli/src/serve.rs` (Linux-cfg'd flags) + serve_cuda | Upstream's own parse and refusals ("a GPU ordinal from 0", "a count from 1"); the named-refusal lines in serve_cuda's messages now name `--device`/`--segments` when the flags set them, exactly upstream's `named` rule. |
| `families/glm/config.zig` | 287 | `crates/glm/src/config.rs` | The exactly-pinned shape: the 45-layer backbone (34 KDA, 11 MLA, the MTP layer), the projection arithmetic (kdaProj 24896 = 3·width + 2·dim + heads; xProj; qrProj), keyWidth/attnBlocks, vocabPart (the TP2 half), countKind/kindIndex; **`parse`'s exact-shape refusal** (every field against the defaults, `n_group`/`topk_group` 1, `qk_rope_head_dim` 0 or absent) and the splits — `subset` (a check's first-n layers), **`splitRows`** (expert parallel by rows, 64-row multiples), **`splitHeads`** (TP2's half KDA heads — f_a and g_a whole), `split` (routed experts' halves). All five tests mirrored. |
| `families/glm/learned_pair.zig` | 117 | `crates/glm/src/learned_pair.rs` | A peer's disk reservation: `need` (u64::MAX while it cannot tell — pending, backoff, or a failed read), `reserve` over the Admission's floor and shortfall and the rank's own cap (the learned dirs' total included), `finish`'s idempotent rollback (a lost reply accepted). All three tests mirrored — the third's real identity dirs over learned_dirs' own total/removeOther. |

**Live-caught port bug (T27's learned_dirs):** Zig's
`std.fs.path.dirname(".")` is null, so `rootOf(".")` falls to "." — but
Rust's `Path::parent(".")` returns `Some("")`, so the port's
`root_of(".")` handed `total` an empty path that failed NotFound, and
the pair tests' `need` over "." returned u64::MAX. The pair test
caught it (it needs `need(".", ...) == 1`); the fix is upstream's own
empty-parent rule. Also: the Admission's `set_retry_at` test seam
upstream reaches by field write (Zig tests share the module) became a
`#[doc(hidden)]` setter.

Gates: Mac 331 passed / 0 failed (323 + the five config tests and
three pair tests, less the one shared lane count), clippy clean on
both crates; box 324 / 0 — **the box's real-identity-dir pair test
green** (the rank-cap rule over learned_dirs' own files).

## Ported (Phase 2, fiftieth tranche — the CUDA serve path)

The Linux branch of `tensorfold serve`: the whole native/cuda.zig open
half — the memory plan, the family open, the admission, the cache, the
lane host — wired behind the same server the Metal family runs on.
**The Linux binary now compiles the full CUDA path** (the box's
workspace build and its 316 tests green with it in). With it every
file of the CUDA target is ported: `native/cuda.zig` (open + Host),
`native/cuda_memory.zig`, `families/nemotron` whole, and the serve
branch that drives them.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `native/cuda.zig` (the open half + Host) | 298-403 | `crates/cli/src/serve_cuda.rs` + serve.rs's Linux branch | **openWith**: the window (the model's, the flag inside it — Negative/NoNative/PastNative in upstream's words), the device and the segments from TF_CUDA_DEVICE/TF_CUDA_SEGMENTS (the flag tranche wires --device/--segments; the variables carry them, `setting`'s flag side None), the device-count refusal, the context, the capability, the captured-kernels pick (TENSORFOLD_CUDA_KERNELS=native, a captured folder, or the missing-aot.json refusal), **the pool** (unified requires the validated meminfo; the reserve and limit with their variable problems), the weights-vs-room refusal in upstream's full format, the family open, the streams (**`admit`**: auto serves what fits, a fixed --parallel refuses, NoStream names the window and the room), the cache budget, and **the startup line** — CUDA sm_X device N (name): model X GiB; N streams at once, ... — upstream's own words. **Host.serve**: the lane config from the facts, the engine over the **bound backend** (the context made current on the calling thread — the port's Bound wrapper over the trait, upstream's vtable wrappers' whole), the host with the device-memory read, the cache, the lone driver, and the same HTTP wiring the Metal path runs. |
| `native/cuda.zig` (the memory read) | 131-134 | `crates/cli/src/serve_cuda.rs`'s read_memory | The device counters as /v1/memory: active/cache/peak/pinned over `cuda.usage`. |
| `native/cuda.zig` (the tests) | 405-441 | native_decisions.rs's tests | The /proc/meminfo live read (Linux, the box — green), the chip classes, the families, the variables, the windows. The #291 cancel integration test is the port's lanes engine.rs test's own — the fake backend's tranche already covers the flow. |

Supporting: the lanes crate's **LaneHost gained `with_memory`** —
upstream's memoryFn (lane_host.zig:217), None by default exactly as
the port had it; the CUDA path sets it. The Box's clippy (1.98, first
install) flagged three of the older tranches' Linux-only lines and one
of the port's own (uring's manual Default → derive; direct_io's CStr
byte literals → c""; learned_disk's platform-typed casts allowed) —
all fixed; the box's touched-crate clippy is clean.

Gates: Mac 323 passed / 0 failed, clippy clean on every crate touched;
box 316 / 0 and the workspace builds with the full CUDA path; the
meminfo live read and all six admission tests green.

## Ported (Phase 2, forty-ninth tranche — the native family interface)

The family's `open` whole — the Loaded that owns the boxed engine, its
lanes backend (through the raw bridge), the reuse target and the
carveout — plus the prompt-cache snapshots bridge and the two-word
refusal explanations. With it, the nemotron family's port is whole
from config to native open: every file of
`zig/src/families/nemotron/cuda*.zig` now has its Rust counterpart.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/cuda_native.zig` | 121 | `crates/nemotron/src/cuda_native.rs` | **`Loaded::open`**: the carveout first (TF_CUDA_CARVEOUT with the MiB option, the card from TF_DRM_CARD, the refusal and the startup lines exact), the engine (graphs on, the segments), the head and **both draw modes' graph sets** — the greedy from Engine::init's capture, the sampled with a placeholder rule (each stream uploads its own at prefill), the head captured under each — then the lanes over the engine and the measurement. The Loaded answers the native server's questions: the backend, the facts, the rows (16 drafting, 1 not), the stream bytes, the free-streams constant, the reuse target, and the lone driver. `Snaps` bridges the target into the lanes store's family face (the raw copy in the store's box, the NotAtMark and CopyFailed mapped), and **`explain`** carries the two refusals in words (the context window, the GPU out of memory). |

The port's Loaded owns everything through the one bridge shape the
tranche before established (the boxed engine, the Cuda and the target
holding its address; the carveout dropped after the engine). The
`lone` upstream exposes as a function pointer the port calls as a
method — the same run, the Loaded in scope.

Gates: Mac 323 passed / 0 failed, clippy clean on the crate; box
316 / 0. The serve wiring (the Linux branch that opens this Loaded
behind the LaneHost, as the Metal family's does on macOS) is the last
remaining piece of the CUDA path.

## Ported (Phase 2, forty-eighth tranche — the lone driver and the memory admission)

The lone-stream driver and the CUDA memory plan — the family's own
`run` over its backend, and the pure arithmetic the native open gates
every request through. **Live on the Ubuntu box: all six of the
admission suite's tests green, including the host's own /proc/meminfo
reading whole and parsing.**

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/cuda_lone.zig` | 45 | `crates/nemotron/src/cuda_lone.rs` | A lone drafted stream as `tensorfold run` decodes it: the opening (the prompt pass, its first token committed), the rounds loop (a copied chain or the head's drafts verified, the depth rule reading each level's confidence, the commit with its landed count), the hand-over when yield says so — the whole decode until finished or handed. |
| `native/cuda_memory.zig` | 170 | `crates/nemotron/src/cuda_memory.rs` | The memory plan whole: **`meminfo`** (MemTotal/MemAvailable in bytes, the malformed and duplicated rows refused, `0`/`>`-than-total refused — upstream's refusal matrix mirrored test by test), `Pool::room` (the reserve, then the cap less what the engine holds), `floor` (unified: at least 4 GiB, a tenth; discrete: 1 GiB, a 32nd), **`reserveBytes`/`limitBytes`** (the floor's least up to the pool's size; the 2^64 boundary case exact), **`admit`** (free streams use no room; auto serves what fits; a fixed --parallel refuses; zero refuses), `cacheBytes` (16 GiB else the flag, never past the streams' room), `counts` (unified requires the validated host counts). Five upstream tests mirrored, plus the Linux live /proc/meminfo read. |
| `native/cuda.zig` (the decision half) | 33-98 | `crates/nemotron/src/native_decisions.rs` | The open's pure decisions: **`chip`** (TF_CUDA_DEVICE's ordinal, its capability as the gate's class name), `envNumber` (unset or empty the default, anything but a whole number refused — the unit error named), **`contextWindow`** (the family default inside the model's window, 0 for the model's, never past it — Negative/NoNative/PastNative), `setting` (the flag, else the variable, else the default; the problem line naming the variable), **`modelContext`** (config.json's max_position_embeddings, text_config's first), `weightBytes` (the safetensors' bytes). Four upstream tests mirrored (chip class, variables, windows) plus the text_config-first rule pinned. |
| `families/nemotron/cuda_lanes.zig` (the engine shape) | — | (refactored) | Cuda now holds the engine as the Loaded's raw bridge — upstream's `e: *Engine` made plain, with the same invariant `with_head` documents — so the family's Loaded can own both the boxed engine and its backend (the native open, the next tranche, boxes them together). The accessors carry the SAFETY notes; the borrow-splitting sites (the pinned span's download beside the engine's fields) read through the bridge in one statement each. |

Live-caught while porting: the first meminfo port returned None on a
trailing blank row (the tests caught it — upstream's tokenizer skips
them); the `Costs` move in `depthRule`'s callers needed `Copy` (upstream
copies by value); and the engine-behind-Cuda borrows that the Loaded
could not hold (the self-referential shape) resolved by the raw bridge,
exactly upstream's pointer.

Gates: Mac 323 passed / 0 failed (310 + cuda_memory's five and the
decisions' four, less overlaps), clippy clean on every crate touched;
box 316 / 0 — **the live /proc/meminfo read green**.

## Ported (Phase 2, forty-seventh tranche — the drafts and lane-backend layers)

The draft-decode machinery and the Backend over the engine — the
family's whole compute stack above the engine, plus the two core
modules it draws on.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/draft_depth.zig` | 104 | `crates/core/src/draft_depth.rs` | The depth rule: `Costs::measured` with the **pooling pass** (each extra row's ms pooled so it never rises with rows — the adjacent-sum merging), round4, `row`; `DepthRule` (keep/more by P^power against tokens-a-ms times the row's cost, or a floor), the modeled-cost `done`. The test mirrored. |
| `src/core/copy_index.zig` | 159 | `crates/core/src/copy_index.rs` | Copy drafts' search over a context: the token-positions map, `longest` (latest of equal length), **`chain`** (the continuation of the latest earlier 8-match, none shorter than 8), and all three of upstream's tests mirrored — including the 200-context fuzz against plain scans, driven by the port's xorshift (the contract holds for every context). |
| `families/nemotron/cuda_drafts.zig` | 142 | `crates/nemotron/src/cuda_drafts.rs` | `Rounds` (the copied chain or the head's chain verified in one window; the accept count as the longest matching prefix), `queue`/`depth` (the rule reads each level's confidence before drafting the next), `decode`'s loop (the eos stop inside the sampled window, the count clamp), and **`Drafter`** (the given-17-costs shape or a measurement here; greedy calibration power 2, every draft depth). |
| `families/nemotron/cuda_decode.zig` | 71 | `crates/nemotron/src/cuda_decode.rs` | prefill + serial_decode: the drafter, the serial-graph loop (`lookahead` deep, `done` events a ring, the history read by position, the parity/keep bookkeeping), or the eager steps. |
| `families/nemotron/cuda_lanes.zig` | 332 | `crates/nemotron/src/cuda_lanes.rs` | The Backend over the engine: the stream→lane map (the engine's own sequence while free — its rounds replay the graphs — else a new seq), `facts`/`measure`, **`prefillFn`** (the reuse restore or a fresh pass, the keep hook at the marks, the first token's logprob row from the mapped words), `first/queue/read`, **`verifyFn`** (positions checked, the window verified, the held drafts downloaded past the sampled, the logprob rows read from the mapped slots) and **`verifyShared`** (Engine.verifyShared over the parts; per-part ids staged as raw spans, the logprob kernel at each part's first row, the pinned drafts and picks read back), `keepFn/draftFn/release`, `handOver`. The vtable's errors ride the lanes `Error::Backend` with the family's own words — upstream's anyerror over the vtable. |
| `families/nemotron/cuda_costs.zig` | 139 | `crates/nemotron/src/cuda_costs.rs` | The timing harness: fastest-of-seven over widths and a full chain (the port without the cost cache retimes once when the fresh costs drift, keeping the faster run — upstream's cached-reference shape follows the native cache tranche). |
| `families/nemotron/cuda_reuse.zig` | 75 | `crates/nemotron/src/cuda_reuse.rs` | The shared-cache state: `bytes/save/restore/drop` over the engine (the head's KV beside the target's, the NotAtMark and no-head-first refusals), the host cache's raw saved copies under its own invariant. |

Supporting: the lanes crate re-exports `LogprobRow` (lanes.zig:35) and
the row's words; the engine's `with_head` settled on its final HRTB
(the head at the engine's `'d`, the engine's own borrow free — the
snapshot flows that live past it compile); `Rounds` carries the engine
as the same documented raw bridge when its borrows cross.

Live-caught while porting: `Costs::measured`'s first port moved
`costs` before reading `verify[2]` (a plain E0382, fixed); the lanes'
`ids[k]` staging in verifyShared could not borrow the whole array
across the parts' fills (restructured as raw spans); and the cancel
adapter's borrow of the stream's check had to end explicitly before
the check moved back (the None-assigns).

Gates: Mac 313 passed / 0 failed (310 + draft_depth's and
copy_index's three tests), clippy clean on every crate touched; box
305 / 0.

## Ported (Phase 2, forty-sixth tranche — the engine)

The CUDA engine whole — upstream's `Engine`, its `Head` and its
staggered `Segments`, the heart of the family's host side (~890 lines).

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/cuda_engine.zig` | 475 | `crates/nemotron/src/cuda_engine.rs` | The engine: `init` (the context rounded to chunk keys, the NCH the captured Triton set covers or the plain count, the weights, the buffers with the shared plans, the boxed engine so the head's view keeps its address), `capture/record/captureWindows` (the serial graph with its feed, the per-row-count verify windows of the bound draw mode), `setSampling`, `newSeq/seqBytes/freeSeq`, **`bind`** (the old bound sequence keeps where it stands; the engine's seq fields take the new one's), `graphsBound`, `reset`, **`prefillFrom`** (the empty/off-grid/full-context refusals, the heat-gated chunks — segmented when set, else serial — the Keep callbacks at the grid points, the sampled token down), `beforeChunk`, `serialChunks`, `setSegments/segmentSet`, **`verify`** (the pinned meta row and ids uploaded past `copied`, the graph launch when the own sequence is bound, else the eager window), **`verifyShared`** (up to eight sequences' rows through one window — each binds, its slot uploads, its Buffers view read whole and never dropped, the eager `windowSegs`, the per-part downloads, the parity and rows set per bound), `sharedTokens/metaRow/tokens/commit/upload/written/step`. |
| `families/nemotron/cuda_mtp.zig` | 304 | `crates/nemotron/src/cuda_mtp.rs` | The MTP head: the one-allocation arena over the 24 fields in upstream's order, the pinned words with the level positions, **`level`** (0 absorbs, 1 drafts after, j>1 extends the chain; the meta advancing a word a level), **`sample`** (the stream's rule over the draft ids, or the greedy path: toF32, topk, lookup, the zeroed-fp `_keyed`), `capture/record` (levels 0/1 at every kept-row count, later levels at one row), `begin/launch` (the graphs picked by the bound mode and keep), `chain`, `confidence/drafts/draw`, **`absorb`** (a chunk's rows into the head's cache, keys and values only). |
| `families/nemotron/cuda_segments.zig` | 110 | `crates/nemotron/src/cuda_segments.rs` | The staggered prompt: `Segments` (the runner and up to three sibling scratch sets), **`prefill`** (the per-chunk ids staged and uploaded ahead of the fork, each segment's Forward reconstructed on its stream for every hook — the Hooks hold the buffer views, the walks and the stream pointers, and reach the engine through the drive's raw bridge), the last segment's hidden, logits and sampled copied back. |
| `cuda/segments.zig` (the fam side) | — | `crates/cuda/src/segments.rs` (extended) | `Fam` gained its family's error type — upstream's anytype stage passes the family's errors through; the port's trait is `Fam<E: From<Error>>` so the hooks' engine calls carry the family's Error whole. |

The port's two named unsafe shapes, each with its invariant in one
place: **`with_head`** (the head lives in a stable Box; no Head↔Engine
field pair is written through both references at once; nothing replaces
`head` while its borrow is live) and **`bound`** (the raw `*mut Seq` to
the own view or a caller's boxed sequence, the caller keeping the
address until `free_seq`). The segments' drive adds the third — the
Hooks' raw engine pointer, alive for the drive's scope. Everything
else is plain: the Forward now carries its Ops by value (Zig copies the
handle struct), the splits over the pinned words and the siblings are
free functions over the fields.

Live-caught while porting: the first `verify_shared` read its views
interleaved with the fills (the borrow checker pinned a real
aliasing — restructured into a fill pass and a parts pass); the
`end_capture` lifetime tied to the stream's borrow rather than its
driver (an engine storing graphs in itself could not compile — the
T39 signature corrected); and the segments' `known` first missed the
chunk-row min (upstream's `@min(seg[k].walk.rows, ...)`).

Gates: Mac 310 passed / 0 failed, clippy clean on the crate; box
302 / 0.

## Ported (Phase 2, forty-fifth tranche — the heat gate and the carveout's shared shape)

The Engine's two prerequisites: the thermal gate (core/heat.zig, with
its whole test suite), and the Carveout reshaped as the shared
allocator upstream's pointer graph always said it was — the piece the
Engine, its buffers and its sequences all take and give through.
**Live on the Ubuntu box: the port's `hottest()` read the machine's
own thermal zones — 54.0 C, the real hottest zone — and the
`from_env()` gate read its unset bands.**

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/heat.zig` | 274 | `crates/core/src/heat.rs` | The whole gate: TF_HEAT_HIGH/TF_HEAT_LOW (both or neither, low at or under high) with TF_HEAT_ROOT moving the zones; the wait loop — run at or under high, or once waiting at or under low — with its 2 s pause, its waited-seconds tally, and the stop that ends a wait as Cancelled after the pause it came in; the no-zone latch (read once, logged once); `hottest()` over `thermal_zone*/temp` in millidegrees. All five of upstream's tests mirrored: the bands' parsing and refusals, the hot-scripted wait through the REAL `wait_until` (a first draft of the port's test drove a copy of the loop — caught on read-back and rewritten), the env-var gate, the no-zone latch, the maximum-zone gather. The test's scripted reads share their script behind a RefCell, Zig's shared-mutable context made plain. |
| `cuda/carveout.zig` (the shared shape) | — | `crates/cuda/src/carveout.rs` | The Carveout's take/give run over a shared borrow now, the spans behind a RefCell: upstream's engine, its buffers and its sequences all mutate one *Carveout (a resource allocator's shared mutable state), and the port's exclusive &mut could not express that. cuda_state's Carved follows — `&Carveout` — and the host-only stub matches. The Engine's seqs and buffers now share one carveout the way upstream's do. |

Gates: Mac 310 passed / 0 failed (305 + heat's five), clippy clean on
every crate touched; box 302 / 0 (297 + the five), the live thermal
read above.

## Ported (Phase 2, forty-fourth tranche — the forward layer)

The computation keystone: Nemotron-H's forwards on CUDA, the kernel
order the Python engine's profile names. Every launch it makes — the
tf_nemo glue fifteen, the affine-4 projections, the expert plan and
kernels, scan_rows, pattn, sample's draws, argmax — was already
symbol-verified against the real fatbins (the T41 live gate); this
layer composes them in upstream's exact order.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/cuda_forward.zig` | 342 | `crates/nemotron/src/cuda_forward.rs` | The Forward whole: **`windowSegs`** (up to four sequences' rows through one shared set of scratch — each segment's own caches, conv state and meta, the double-buffered norm, the mixer indices per block kind, the head and per-segment draws), **`experts`** (route, plan, up relu^2, down; prompt takes the prefill kernels at 64 pairs a tile), **`forked`** (the decode MoE's shared halves on the side stream with the fixed precomputed plan while the routed pairs plan on the main one — fork, join, shared_wait), and the chunk path **`chunkBegin/chunkPre/chunkMixer/chunkPost/chunkFinish`** (the staggered segments' Forward hooks run these: the norm, the Mamba in-projection, conv_rows and scan_rows over the chunk, the attention's kv-write + the q copy + prefill attention, the group-rms and out-projection, the experts at prompt tiles, the last norm, p_hidden, the chunk's last row's logits and sample_last). The Marks classes (the window profile's spans), Walk (a chunk's residual delta and mixer positions), Seg/Side (a verify window's segment, the second stream with its fork and join). Upstream holds the marks and dump as mutable pointers behind a const forward — the port carries them as a RefCell and a Cell'd counter, the same interior mutability named. |

Gates: Mac 305 passed / 0 failed, clippy clean on the crate; box
297 / 0 (with the T41 symbols' launches all this composes verified
live earlier). cuda_mtp's Head reads the Engine throughout — it lands
with the engine tranche.

## Ported (Phase 2, forty-third tranche — the weight loader)

The whole GPU weight load: `cuda_weights.zig` byte for byte — the
tiled projections (qmm_fast.tile's pack_dense + transpose16 pair), the
packed expert blocks (the shared expert's halves as experts E and E+1,
the split columns staged on the host), the fp32 Mamba (the conv taps
transposed, A_log through the -exp kernel), the lesson path (a bf16
output projection quantized on the host with affine4_host, its missed
remainder kept as the rest) — plus the eager-forward dump helper.
**Live on the Ubuntu box: every kernel the loader launches —
tf_pack_dense, tf_transpose_pad16 and the experts-pack instantiation —
resolved in the real fatbins.**

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/cuda_weights.zig` | 445 | `crates/nemotron/src/cuda_weights.rs` | The loader whole: `tile` (row-stacked parts packed dense then transposed 16 — the scratch staged per part, the pack's grid from npad/64·kg·512 words), `lesson`/`rest`/`denseRest` (a bf16 `{pre}{mid}.weight` quantized on the host, scales/biases and the [n,k] rest the codes miss, columns [first, first+width) staged for the shared halves), `experts`/`packExperts` (the routed experts with the shared appended, `split` halving the shared columns on the host — `fold_shared` then `experts.make`), `mamba` (in_proj/out_proj, the conv taps as the kernel reads them, A_log widened then -exp'd on the device, gnorm raw), `attention` (q,k,v stacked through one tile, the o projection's rest), `moe` (the gate's weight and bias, the shared down's lesson becoming experts E and E+1's rest), `three` (the quantized triple, u32 words with bf16 scales and biases), and `load`/`loadMtp` (every block by its kind, the embeddings as a row table, lm_head, leftovers refused — and the MTP head: enorm/hnorm/eh_proj/the attention layer/the MoE layer, then the vocabulary head's rows for the draft ids gathered and tiled as a head of their own, the ids as i64). The port carries its own 2-d `Part` view where upstream aliases core.checkpoint.Tensor for mapped and host-made tensors alike, and a `Held` keeps a lesson's host parts under an explicit contract (the boxes never move; the loader's borrows end and start around them). |
| `families/nemotron/cuda_dump.zig` | 35 | `crates/nemotron/src/cuda_dump.rs` | The eager forward's raw dumps — each norm's h, y and xs named `{i:0>3}_*` as the oracle names them, then logits and sampled. |

Supporting: the core Checkpoint's `get`/`expect` became shared borrows
with interior used-marking — upstream's get takes no exclusivity any
reader can observe, and the loader reads several mapped tensors at
once; `Checkpoint::empty` for the MTP head's `add`-grown namespace (the
upstream `.{ .gpa, .io }` init); and `From<MissingTensor>` in the
family's Error.

Live-caught while porting: `lesson`'s missing name is upstream's `try`
— an error, not the absence of a lesson (the port first returned None,
wrong); the conv-tap fill first written wrong (caught on read-back
before it ever compiled); and the staging borrows restructured —
upstream stages through one reused host buffer, the port's calls carry
their own, the uploads byte-identical either way (the head-rows gather
split into a fill pass and a parts pass for the same reason).

Gates: Mac 305 passed / 0 failed, clippy clean on every crate this
tranche touched; box 297 / 0; the loader's launch symbols resolved
against the real fatbins.

## Ported (Phase 2, forty-second tranche — the read staging and state layers)

The checkpoint's read side, the quantize-dequantize math the loader
stages with, and the nemotron source/sampler/state layers. **Live on
the Ubuntu box: the port's hand-rolled io_uring — every layout from
the kernel's own `linux/io_uring.h` (SQE 64, CQE 16, params 120), the
x86_64 syscall numbers, the mmap offsets — submitted two reads and
completed both with the user_data tags and bytes intact** (the test
std's own IoUring carries upstream, since Rust's std has none); and
direct_io's O_DIRECT test ran against `/proc/self/exe`, the reads'
bytes equal to the buffered reference's.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/checkpoint.zig` (the Checkpoint half) | 57-122 | `crates/core/src/checkpoint.rs` (extended) | `get` aligned with upstream: marks the name used and refuses naming it; **`expect`** (dtype and shape checked, the log line), **`add`** (one more file — the MTP head beside the model's), **`unused`** (the loader's leftover refusal, the first five logged), and **`files`** (the paths with their mapped bytes, for the CUDA source's direct reads). The port's `open` was already the `openModel` shape; only tests called the old `get`. |
| `src/core/direct_io.zig` | 151 | `crates/core/src/direct_io.rs` | Positional reads that skip the page cache: the aligned-span read with the **short-read retry from the last 4 KiB boundary**, the pread injected as a trait exactly as upstream's fn parameter, Linux's O_DIRECT tried then fallen back, and `read_fd` — the raw-descriptor entry the source's queued reads take (upstream copies the File value; the port's File owns its close). All three tests mirrored: span/fits, the Script retry, and the Linux `/proc` byte-equality (live on the box). |
| `src/core/affine4_host.zig` | 58 | `crates/core/src/affine4_host.rs` | MLX's affine 4-bit on the host: 8 codes a u32 low-nibble-first, bf16 scale and bias (NaN keeps its bits with the quiet bit forced — glue_math's cvt.rn form differs, as upstream's two do), dequantize, and quantize (each group's minimum its bias, range over 15 its scale). The test mirrored. |
| (std.os.linux.IoUring, the piece cuda_source imports) | — | `crates/nemotron/src/uring.rs` | The ring hand-rolled: setup through the raw syscall (425), the three mmaps its params point to, SQE preparation at the submission tail, enter (426) with GETEVENTS for waits, the CQ drained under acquire/release — the layouts size-asserted against the header's own. **Live on the box: the ring test.** Off Linux a never-constructed host form stands the names in (upstream's IoUring lives in std.os.linux and the source catches its absence). |
| `families/nemotron/cuda_source.zig` | 241 | `crates/nemotron/src/cuda_source.rs` | Checkpoint reads through 64-MiB O_DIRECT slots, four in flight: page-locked on a discrete card with a copy event each (settle waits the event, not the stream), the ring when the kernel has one (a short or failed read finished by a plain read of the same span) **else each read on its own thread, reaped oldest-first by the asked order** (the port's thread body through a Send wrapper and a free fn — Rust 2021's disjoint closure captures would move the raw field, a live-caught shape). `upload` (mapped bytes queue, others flush first), `flush`, `read`, and `view` (a slot's span, valid until the next call). |
| `families/nemotron/cuda_sampler.zig` | 44 | `crates/nemotron/src/cuda_sampler.rs` | The request's rule as sample.cu reads it: margin 8, `count`, the temperature-0 `check`, the greedy head draw's Keyed, the extern Rule (size 32 asserted), and `rule`'s fills. The test mirrored. |
| `families/nemotron/cuda_state.zig` | 311 | `crates/nemotron/src/cuda_state.rs` | One sequence's device buffers: the 256-aligned Arena, the **Carved** KV pair in the display carveout (both planes or none, given back in reverse), the twelve seq fields and the thirty-five scratch fields in `scratchSizes'` own order, the shared expert's precomputed plans (`sharedPlan`'s items at 2·16 + 8·(rows-1) words, `initShared`'s halves over the routed slots), snapshot/restore/reset over the seven planes, sibling/follow for a second prompt segment. Zig's `inline for` field walks became the explicit lists in the same order; the Carveout is the cuda crate's cross-platform name. |

Supporting: safetensors' Tensor gained `rank`/`dim` (safetensors.zig:112-118),
lanes re-exports `Sampling` (lanes.zig:33), the cuda crate re-exports
`CarveoutError`, and the Carveout name exists off Linux as a
never-constructed host form (upstream's compiles only there — Zig's
laziness never analyzes it on other hosts — while the family's state
layer names the type everywhere).

Gates: Mac 305 passed / 0 failed (301 + this tranche's four portable
tests), clippy clean on every crate this tranche touched; box 297 / 0
(291 + four portable + **the two Linux-only tests live**: the
`/proc` byte-equality and the io_uring ring).

## Ported (Phase 2, forty-first tranche — the Nemotron family's foundation and launch layers)

The `tensorfold-nemotron` crate lands: the first tranche of
`zig/src/families/nemotron`'s CUDA half — the foundation (config, draft
ids, the glue's host math and references) and the launch layers (torch
ops, Triton dispatch, glue, kernels). **Live on the Ubuntu box: every
one of the 40 kernel symbols the tranche's registry names — the two
expert kernels, the two prefill experts, pack, pattn, scan_rows, the
five nemotron_ops extern-Cs, the three sample kernels, all ten torch-op
replacements, and the fifteen tf_nemo glue kernels — resolved in the
real fatbins** through `cuobjdump -symbols`, the driverless form of
upstream's "a wrong name cannot hide behind an unused path"; and
draft_ids.txt validated (49,152 ids below vocab 131,072, tiled by 64).

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/config.zig` | 344 | `crates/nemotron/src/config.rs` | The admission gate whole: the dimension reads, hybrid_override_pattern (one character a block, "M"/"E"/"*") over layers_block_type, eos int-or-array, time_step_limit, and **the quantization entries read as MLX's nn.quantize reads them** — true, a dict at the checkpoint's width, the learned bf16 projections; null and empty dicts fall to the unquantized refusal while other JSON values pass silently — every refusal reason byte-exact (both upstream tests mirrored over the Lightning fixture, the ten refusal cases naming their words). `checkShapes` pins Nemotron 3.5 Lightning 30B-A3B. |
| `families/nemotron/draft_ids.zig` + draft_ids.txt | 32 | `crates/nemotron/src/draft_ids.rs` + the file beside it | Whitespace-separated ids each below vocab, a multiple of 64 (DraftIdPastVocabulary, DraftIdsNotTiled); the file embedded with `include_str!` where upstream's build generates a module. Upstream's test mirrored (the 64-id `{d:>2} ` grid). |
| `families/nemotron/glue_math.zig` | 121 | `crates/nemotron/src/glue_math.rs` | The glue kernels' scalar math op for op: fma, bf16 round-trips (cvt.rn, NaN to 0x7fff), and the **within-one-ulp exp and log1pUnit** — the j·ln2 split, the degree-7 and atanh polynomials with every constant the nearest f32 (Zig's hex floats spelled as their exact shortest round-trips) — plus sigmoid/silu/softplus over them. All three tests mirrored, the ulp harness frexp/ldexp hand-written. |
| `families/nemotron/glue_ref.zig` | 418 | `crates/nemotron/src/glue_ref.rs` | Host references for every glue kernel, bit-exact: the thread-shaped reductions (butterfly, sumSquares at 256/128 threads, warp order), scaleStore's group sums, embed's MLX 4-bit rows, the norms (add/moe/concat/group), the router's slice fma chains, top-k by score-plus-bias with lowest-id ties, the conv's fma taps through SiLU twice-rounded, the scan's state replay (softplus-dt clamp, 2·4·8 lane structure, exp(a·dt)), **the attention's online-softmax chunk/tile structure** (α-rescaling, the 16-lane butterfly denominator), and keyed_greedy's f64 rank with the share's sum. Both tests mirrored (the one-key attention, the top-k tie). |
| `families/nemotron/cuda_prompt_grid.zig` | 28 | `crates/nemotron/src/cuda_prompt_grid.rs` | end/resumable — the zero-anchored grid; upstream's test mirrored. |
| `families/nemotron/cuda_torch_ops.zig` | 151 | `crates/nemotron/src/cuda_torch_ops.rs` | The torch-op replacement launches: argmax (bf16, NaN first), toF32, **topk's four 8-bit digit passes then ordered compaction** with its workspace arithmetic, lookup, strided copies, mambaA — the development launchers' geometry; upstream's workspace test mirrored. |
| `families/nemotron/cuda_triton.zig` | 119 | `crates/nemotron/src/cuda_triton.rs` | The captured Triton set's launches, each with the Python wrapper's grid and constexprs: the thirteen wrappers (embed, the norms, route/topk, conv/scan/group-rms, kv-write, the 512-key chunk/merge pair, concat, **keyed with its splitmix/pcg constants**), the Shape/Attn/Keyed types at module scope, and `divisor`. Upstream's slices test mirrored. |
| `families/nemotron/cuda_glue.zig` | 243 | `crates/nemotron/src/cuda_glue.rs` | Every glue launch on one stream, ours or the captured set's (the `tri()` dispatch per call): the fifteen tf_nemo kernels with their grids, shared-memory stages (the router's 16×16 slice padded a word) and shape refusals (RowTooWide, RouterTooWide, ScanShape, AttentionShape, KeyedRule); chunk_keys 512 / chunk_blocks 64; upstream's router-shape test mirrored. |
| `families/nemotron/cuda_kernels.zig` | 296 | `crates/nemotron/src/cuda_kernels.rs` | The registry: the 21-module image list in upstream's exact order, every symbol resolution (the mangled instantiations and the extern-C names), the shared opt-ins (38400 prefill experts, 65536 pattn), the SM counts into expert_blocks, GB10/integrated detection, the Triton set's load, and `Ops` — all the launch helpers (draw/logprob with the ids-optional pair, the affine-4 delegations, restRows, the routed plan with its bounds, experts/prefill experts, pattn, scan_rows, the copy/fill/upload/download stream ops, serialFeed); upstream's items test mirrored. |

The crate's one `Error` enum carries the family's error names (the
config refusals, the draft-id tiles, the shape refusals) with the CUDA
runtime's inside — Zig's anyerror has no boundary. Note for the record:
upstream's `float` helper (config.zig:117) is uncalled there too; the
port keeps it for the later tranches, allowed dead. Gates: Mac 301
passed / 0 failed (288 + this tranche's 13), clippy clean; box 291 / 0;
golden 2/2 and E2E green in the workspace run.

## Ported (Phase 2, fortieth tranche — the fatbin build pipeline)

`zig/build/cuda.zig`'s kernel pipeline onto cargo, and `kernels.rs` flipped
from its host-only shape to the embedded images. The `kernels/cuda` sources
themselves are copied into the tree byte-identical (39 files, like the
metal tree before them). **Live on the Ubuntu box: all 24 fatbins built
through nvcc 13.3** (torch's own flags, each extension's `extra_cuda_cflags`,
`-gencode` for sm 121), and **every mangled name the port's symbol tables
claim resolved in the real fatbins** — gdn's 2 replay + 8 tree kernels,
qmm_group's tile-2, qmm_prefill's prompt GEMM, lane_gemv and its split,
experts' 4 plan kernels, fp8_lane's 8 qmmf instances — through
`cuobjdump -symbols`, the driverless form of upstream's runtime resolution
test (runtime_tests.zig:419, which needs a GPU). The prebuilt path embedded
the same 24 real fatbins; the missing-file refusal says upstream's own
"missing or empty CUDA fatbin"; an unknown `TENSORFOLD_SM` surfaces nvcc's
own fatal text per kernel.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `zig/build/cuda.zig` (the fatbin pipeline) | ~110 of 229 | `crates/cuda/build.rs` | The `kernels` table whole — all 24 entries with the flags their Python extension passes (torch_ops: no contraction, no flush to zero; the nemotron glue: IEEE div/sqrt for bit-for-bit host references; gdn's/prefill_attention's/scan_rows' `--fmad=false`) — `torch_flags` verbatim (torch 2.13's cpp_extension set incl. `--expt-relaxed-constexpr`, `-std=c++20`), and `fatbin()`'s exact command: `nvcc -fatbin <torch flags> <kernel flags> -gencode=arch=compute_<sm><a>,code=sm_<sm><a> per target -MD -MF <dep> -o <out> <src>`, the 24 compiles in parallel. `TENSORFOLD_NVCC`/`TENSORFOLD_FATBINS`/`TENSORFOLD_SM` mirror `-Dnvcc`/`-Dfatbins`/`-Dsm` (default 121); neither of the first two builds host-only. The nvcc version text is an input of every fatbin (cuda.zig:90), so it rides each kernel's stamp; the depfile's headers feed the stamp the way upstream's build consumes it; the source tree is watched coarsely and each kernel's stamp decides whether nvcc runs — Zig's per-step laziness inside cargo's one script. `checkDistFatbins`'s missing-or-empty refusal guards the prebuilt set, every file copied into OUT_DIR for `include_bytes!`. |
| `zig/src/cuda/kernels.zig` (the flip) | 71 | `crates/cuda/src/kernels.rs` | `available` is `cfg!(kernels)` — the build's `with_kernels` — and each image is an aligned `Fatbin`: upstream embeds at `align(16)` and `Module::load` reads the image at its alignment, while `include_bytes!` guarantees none, so each blob sits in a `#[repr(align(16))]` wrapper (its generic field building as the sized `[u8; N]` and the reference unsizing to the slice tail) and the callers pass `&image.bytes`. Host-only keeps every image empty and the module's logic — the gdn symbol tables, the 8 spelled-out tree instantiations, `treeVariant`'s dispatch — unchanged from the thirty-ninth tranche. |

The one consumer so far (fp8's `Lane::load`) now passes
`&kernels::fp8_lane.bytes`; the host-only refusal before `Module::load`
stays. The box's two Linux-only unused imports in `carveout.rs` (invisible
on macOS, where the cfg'd modules don't compile) are gone. Gates: Mac 288
passed / 0 failed, clippy clean on this tranche's files; box 278 / 0
host-only and 28 / 0 with the real fatbins embedded; golden 2/2 and E2E
green.

## Ported (Phase 2, thirty-ninth tranche — the CUDA segments, quantized projections and runtime libraries)

The staggered-prompt core and the five CUDA modules behind it, plus cuBLASLt
and NCCL — the last of `zig/src/cuda` except the fatbin build pipeline and
the family CUDA halves. All pure-logic tests run green on both platforms;
**cuBLASLt's whole 15-symbol table resolved live on the Ubuntu box** against
the real `libcublasLt.so.13` (CUDA 13.6.0.2, `cublasLtGetVersion` answering
130600); NCCL is not installed there, so its `open()` refusal is the
verified outcome — exactly upstream's host behavior. A live-caught port
bug: fp8's `shared_bytes` wrote `4 * max(stage, partials)` where upstream
has `@max(4 * stage, partials)` — upstream's own test values (50176 for a
64-row tile) caught it before anything shipped.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/stagger.zig` | 180 | `crates/lanes/src/stagger.rs` | The backend-neutral stagger whole: `rows`/`start` (near-equal, first ones longer), `next` (the fewest calls of two segments of up to `step`, each `min_rows` or more), `chunked`/`chunkRows` (whole serial chunks cut exactly where serial chunks fall), and `drive` — the hook order itself (begins; per layer, per segment in row order: pre, mixer, post, with k waiting on k-1's post-mixer signal ahead of pre or of the mixer). `drive` runs over a `Hooks` trait (Zig's `anytype`). Upstream's three tests mirrored whole, the third (the queue simulation that checks every dependency holds and nothing deadlocks) through the port's own driver of the same queues. |
| `src/cuda/segments.zig` | 323 | `crates/cuda/src/segments.rs` | Staggered prompt segments on CUDA streams: the `Runner` (side streams, handoff/join/fork events made once and reused — the port's `Stream` owns its CUDA object, so `Lane` borrows it), `Ops` and `Fam` as traits (Zig's `anytype` pair), and `stage` — fork, drive, join, with the error path joining what was queued and draining every lane before buffers are reused. **`handoff[k]` is segment k's mixer event (waited by k+1), one event a pair serving every layer.** Both simulation tests mirrored: the stream simulator (a record marks its stream's position, a wait takes the event's latest record — the port's xorshift picks the orders, the contract holds for every one) proving two calls with reused events never deadlock and every dependency in `Check` holds over 200 seeds × 4 widths, and the missing-handoff test proving the check bites. The port was written against the full upstream text after a first draft guessed the adapter shape — the rewrite follows the source line by line. |
| `src/cuda/kernels.zig` | 71 | `crates/cuda/src/kernels.rs` | Host-only shape: `available = false`, every fatbin image empty — exactly upstream's no-nvcc build, which the fatbin-pipeline tranche flips to the embedded images. The gdn symbol tables (replay pair, the 8 tree_kernel instantiations spelled out from upstream's comptime `tree`) and `treeVariant`'s GPU dispatch (wide = streams ≥ 2 on 12.0 with ≥ 96 SMs) as array selection. |
| `src/cuda/fp8.zig` | 224 | `crates/cuda/src/fp8.rs` | Block-FP8 projections on fp8_lane.cu: `sharedBytes`/`splitK`/`bucket`/`padded` pure and tested against upstream's own values, `kin` + `packCodes`/`packScales` (the kernel's fragment order, rows past n zero/one — both permutation tests mirrored), the `Plan` chooser (fused/cluster/reduce as qmmf_cuda picks), and `Lane` — the 8 instances by exact mangled symbol, the row-tile shared opt-ins, `partBytes`, and `matmul`'s launch shape (the z-dim the slice count when not fused, the cluster of `sk` when clustering, the reduce kernel after). `load` refuses `BuiltWithoutKernels` in host-only builds. |
| `src/cuda/qlinear.zig` | 256 | `crates/cuda/src/qlinear.rs` | Quantized projections behind one call: the `Weight` view per format (affine-4 or fp8g) with `Linear` dispatching decode/prompt rows per format — a format that is not loaded refused before any launch (the port's test stub standing in for Zig's `undefined` stream); the affine-4 kernels: `splitK` by shape alone (upstream's Python-shape test values), the split-K partials/tickets buffers, `gemvSplit`/`gemv`/`cluster`/`prompt` with their exact `Args` orders (the `Part`/`Parts`/`GemvPart` extern structs size-asserted 56/232/48) and launch shapes (the cluster's `sk` CTAs, the prefill's 12-MiB band). |
| `src/cuda/grouped.zig` | 77 | `crates/cuda/src/grouped.rs` | experts.route's plan on experts.cu: `maxItems` (an item a used expert plus one a `tile` pairs past its first — upstream's capacity test), the four plan kernels by exact mangled symbol, and `route` — the one-block plan under 1024 pairs, else rank (blocks of 1024) → offsets → scatter, each launch's `Args` in upstream's order. |
| `src/cuda/cublaslt.zig` | 163 | `crates/cuda/src/cublaslt.rs` | cuBLASLt dlopen'd with **its own error set and API table**, fields by exact symbol: the row-major bf16 product as a column-major view (D^T = W·X^T, transa=t/transb=n, fp32 compute only), the descriptor/layout/preference chain built and unwound exactly (the port's `unwind` the errdefer chain, Drop the same), the heuristic's first algorithm picked once with the workspace bound, `Algo`/`Heuristic` size-asserted 64/96, and `run`'s `Invalid` when the workspace falls short. **Live on the box: all 15 symbols resolve against the real 13.6 library, the version call answering 130600.** |
| `src/cuda/nccl.zig` | 71 | `crates/cuda/src/nccl.rs` | NCCL dlopen'd for tensor-parallel ranks, the same shape: its own error set, the 11 collectives/communicators by exact symbol, `UniqueId` (128 bytes asserted), and the version call. Not installed on the box — the `LibraryUnavailable` path verified there, upstream's host behavior on a machine without NCCL. |

Driver `Error` gains the upstream error names this layer raises (`Segments`,
`BuiltWithoutKernels`, `FormatNotLoaded`, `RowStride`, `WindowTooWide`,
`SplitTooWide`). Note for the record: the Mac's clippy moved to 1.96 between
tranches and its new lints (`items_after_test_module`, bool `assert_eq!`,
needless-range) now flag a handful of pre-existing lines in older tranches'
files — this tranche's own files are clean; the sweep of the older hits is
its own follow-up.

## Ported (Phase 2, thirty-eighth tranche — the CUDA Triton, aot and carveout layers)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/cuda/triton.zig` | 93 | `crates/cuda/src/triton.rs` | Triton kernels ahead of time: the Meta (Triton's own field spellings, the unknown ignored — parsed and tested), the Kernel load (the >48-KiB dynamic-shared opt-in the Triton launcher does: the device's opt-in limit read, the cache config preferred shared, the static shared subtracted), the launch geometry (x times num_ctas, 32 × num_warps threads, the cluster and its spread when num_ctas ≠ 1), `launch_on`'s validations (the divisible-16 args, the scratch the metadata demands, the non-portable cluster size at num_ctas 16), and the global-scratch arithmetic. The launch-geometry test caught its own bug: the test's Meta never set `global_scratch_size` (the port's test, fixed by setting it — the geometry itself was right). |
| `src/cuda/aot.zig` | 210 | `crates/cuda/src/aot.rs` | The captured-set layer whole: `pick` (explicit or the binary's `share/tensorfold/cuda/sm<cap>` with aot.json), the aot.json shapes (the kernel's fn/hash/name/num_warps/num_ctas/shared/pdl/params/consts — serde with the defaults Triton's exporter leaves out), and **the specialization rules** — the meat, all pure and tested: a pointer's 16-divisibility must match what the capture recorded, an int Triton folded only ever the value 1, the type names Triton's own (`*bf16`, `fp32`), the constexprs equal by int and by bits, the runtime count exactly the params. Plus `find`'s diagnostic log (every argument printed the way upstream prints it), `smallest_const` (the smallest at or above the floor — tested over variants), and `run`'s packing in the variant's own param order. Four tests over every rule. |
| `src/cuda/carveout.zig` | 232 | `crates/cuda/src/carveout.rs` | GB10's display memory as device memory: the pure placement whole and portable — `round_bytes` (whole framebuffer rows), `requested` (the flag/`on`/`mib` option rule with the row-floor error), and **`Spans`** (first-fit 256-byte-aligned placement, the given-back room retaken) with **upstream's own five tests** green on both platforms; the DRM dumb-buffer shapes and ioctls (drm_mode.h's constants and layouts asserted where compiled — Linux; a live-caught **visibility bug** on the box: the extern fns private to the `drm` module, the glob import resolving nothing, the error pinning it). The `Carveout` itself (open/register-DEVICEMAP-first/map/take/give, the close in reverse) is Linux-only and compiles against the box's glibc. |

## Ported (Phase 2, thirty-seventh tranche — the CUDA module, launch and graph layers)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/cuda/module.zig` | 83 | `crates/cuda/src/module.rs` | Loaded GPU code: the cubin/fatbin image (its 8-byte alignment and the empty-image refusal — upstream's own "built without kernels" line), PTX through `cuModuleLoadDataEx` with the JIT error log wired into the caller's buffer, the kernel by its exact symbol (the not-in-module log), the device globals, and the Function's attributes/occupancy/48-KiB dynamic-shared opt-in. Unload on Drop. |
| `src/cuda/launch.zig` | 143 | `crates/cuda/src/launch.rs` | The argument packing whole: `Args` — up to 48 values in 2048 bytes at their **C alignments**, the pointer array rebuilt on each `pointers()` (a moved Args never handing out stale addresses), `integer`'s 4/8-byte reads; `Config` (every dimension non-zero, a block at most 1024 threads) and `launch` — the plain path through `cuLaunchKernel`, the extended through `cuLaunchKernelEx` with the attributes in the ABI's order (PDL, cooperative, the cluster dimension, the spread preference). Upstream's own test ported: the offsets [0, 8, 12, 16, 24] with the values read back through the packed pointers — green on both platforms. |
| `src/cuda/graph.zig` | 136 | `crates/cuda/src/graph.rs` | The whole graph surface: capture (begin/end/status on a stream), the node-by-node `Graph` (addKernel/setKernel/depend/nodeCount/nodes/instantiate), `Exec` (upload, launch, the in-place kernel update, the topology-preserving `update` with the driver's verdict through `ExecUpdateResultInfo`), and `nodeParams`' plain-config refusal (cluster/PDL/cooperative are launch-only, never graph nodes). |

## Ported (Phase 2, thirty-sixth tranche — the CUDA driver layer)

The `tensorfold-cuda` crate lands (the port of `zig/src/cuda`'s driver
half), **compile- and load-verified live on the Ubuntu box**: the stub
`libcuda.so.1` resolved all 70+ entry points by name, its `cuInit`
answered with a real CUDA error code through the mapped error path, and
**every upstream kernel source compiled** — all 18 `.cu` files to PTX
for sm_90 on CUDA 13.3.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/cuda/abi.zig` | 222 | `crates/cuda/src/abi.rs` | The ABI table whole: the types, the result codes, the attribute constants, the launch structs — and `Api`, every entry point declared with its exact exported symbol. Upstream's comptime field loop is a declarative macro: one table declares the struct and the resolver, a missing symbol refusing the whole driver. The comptime size assertions port as a test (LaunchAttribute 72 / value at 8, LaunchConfig 56, KernelNodeParams 72, ExecUpdateResultInfo 24) — green on both platforms. |
| `src/cuda/driver.zig` | 84 | `crates/cuda/src/driver.rs` | The dlopen'd driver: the platform's own `dlopen`/`dlsym` (upstream's std.DynLib — the port's hand-rolled convention), RTLD_NOW, every symbol resolved, `cuInit` at the end; the checked calls with the driver's own error name and text — NOT_READY a state, code 2 the out-of-memory, the rest the generic failure. The loader test runs on both platforms: macOS's absent library the `DriverUnavailable` path; the box's stub the resolution plus the live `cuInit` refusal (code 34, the stub's own) through `CudaFailed` — an outcome other than `MissingSymbol` is the contract holding. |
| `src/cuda/context.zig` | 65 | `crates/cuda/src/context.rs` | One device's primary context, retained, made current; the capability read, the name, the memory info. |
| `src/cuda/stream.zig` | 79 | `crates/cuda/src/stream.rs` | Streams (non-blocking their flag) and events (timing or not): ordering, waits, `done`'s NOT_READY-is-not-done, the elapsed milliseconds. |
| `src/cuda/memory.zig` | 253 | `crates/cuda/src/memory.rs` | Device memory and pinned host memory, each owned by one value: the usage counters (device, host, the peak since the last reset) with their lock, the borrowed span, the span arithmetic, the blocking copies (upload's own sync) and the async ones, the byte and word fills. Upstream's three tests ported whole — the counts and the peak, the borrowed span's free leaving everything alone (its `undefined` driver made safe through a None-entry-point stub), and the peak reset's read-store pair proven atomic against a concurrent allocation (the hook's port). The **first test-caught port bug** in this crate: the tests share global counters and run in parallel — the guard upstream's serial test runner never needed. |
| `kernels/cuda/*.cu` (18 files) | 3630 | not ported — CUDA C, shipped to the box | **All 18 compile**: `nvcc -arch=sm_90 --expt-relaxed-constexpr -ptx` green on CUDA 13.3 — a live catch confirming the flag: `gdn.cu` fails without `--expt-relaxed-constexpr` (a constexpr host call from a global), and upstream's own `build/cuda.zig:47` passes exactly that flag. The fatbin pipeline (build.zig's `-code=compute_90,sm_90` fatbins) is the next tranche beside the module/launch/graph layers. |

## Ported (Phase 2, thirty-fifth tranche — the port on Linux, toward CUDA)

The port now **builds and fully tests on Linux** (the Ubuntu box at
192.168.1.55 — CUDA 13.3 toolchain, 32 cores; the working directory
`/home/ganzoubuntu/development/tensorfoldrust`, the upstream sources at
`…/tensorfold-zig`), the prerequisite for the CUDA crate the toolchain
verifies. 20 suites / 247 passed / 0 failed there, all gates green on the
Mac after the same tree.

| The work | Upstream | Rust | Notes |
|---|---|---|---|
| The compile-time family selection | `native.backends` is target-selected (metal.zig's path on macOS, the CUDA hosts on Linux) | `crates/metal` + `crates/qwen35` empty crates off macOS (`#![cfg]`), their deps target-conditional, the Metal serve tail and the family's integration tests gated | A Linux build's `serve` refuses a `qwen3_5` checkpoint with **exactly upstream's own message** ("the native engine has no backend for … the Python engine 0.6.6 may serve them"), verified live on the box. |
| The Linux FFI shapes | `learned_disk.zig:5` (linux's counts 64-bit, a 24-byte tail; the sizes upstream's test asserts) | `learned_disk.rs`'s `Stat` cfg'd per platform | Upstream's own `@sizeOf(Stat)` assertion ported: 112 on linux, 64 elsewhere — green on both. |
| The Linux RAM read | `std.process.totalSystemMemory`'s read there | `cache_fit.rs`'s `ram()` reads `/proc/meminfo`'s `MemTotal` on Linux | The prompt-cache budget plan works on the box; `footprint` stays macOS-only exactly as upstream (null elsewhere). |
| **The live-caught O-flag bug** | auth.zig:65's `.NOFOLLOW` | the key file's open flags cfg'd **from each platform's own fcntl.h** | Caught twice over on the box: macOS's `0x400000` is O_DIRECTORY on Linux, and a mis-remembered "O_NONBLOCK" `0x4000` there is **O_DIRECT**, breaking small unaligned reads with EINVAL — the tests pinned it, the box's headers settled it (`__O_NOFOLLOW 0400000` = 0x20000, `O_NONBLOCK 04000` = 0x800). |
| The next tranche's ground | `src/cuda/` (~2.8K) + `kernels/cuda/*.cu` (~3.6K) | — | The CUDA 13.3 toolchain on the box (nvcc + NVRTC + the libcuda **stub** — compile-verified, no GPU) is the target for the `tensorfold-cuda` crate; the upstream sources shipped to the box for the kernel-ABI work. |

## Ported (Phase 2, thirty-fourth tranche — the single-stream download whole)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `cli/pull_file.zig` | 152 | `pull.rs`'s `download_one` + `link_into_snapshot` | The single-stream download at full fidelity, replacing the port's simplified baseline: **the range resume** — the LFS files download into a `.incomplete` beside the blob, a restart sending `Range: bytes=N-` with **the on-disk prefix read back and fed into the same streaming sha256** (the check spanning the whole file, nothing allocated whole), a hub that answers a range with the whole file restarting from zero; **the git blob sha1** for the small files (`sha1("blob SIZE\0" + bytes)` against the tree's oid — the port had checked nothing for them, a fidelity catch); the size check; the rename into blobs with the `MiB` line upstream prints. And `linkIntoSnapshot` fixed to the upstream shape: **the relative `../../blobs/NAME` symlink** (the port had linked to absolute paths — the cache is movable upstream) and the **traversal refusal** (`snapshotPathOk` — the port had linked a hostile tree path anywhere; a security catch). Seven tests over a local HTTP server: the whole download verified and renamed, the interrupted prefix resuming with exactly one ranged request, the rangeless restart, the git sha1 pass and refuse, the short-body size refusal, upstream's path table, and the relative link's read-back. |

## Ported (Phase 2, thirty-third tranche — the 16-way piecewise download)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `cli/pull_parts.zig` | 182 | `crates/cli/src/pull_parts.rs` | The ranged 16-way whole: files at least 64 MiB as 32-MiB pieces on one shared queue over up to 16 connections — each piece resuming within itself after a reset or stall (five attempts, the fetch `done` offset), its marker going down only once every byte is written, a restart skipping the marked pieces; a 206's `Content-Range` first byte verified against the ask (the hub answering a range with the whole file `RangeUnsupported`, the caller's single-stream fallback); the post-assembly sha256 digest. Tested over a **local range server** (a test-spawned HTTP listener over an in-memory file, its requests recorded): the pieces assemble byte-identical, the markers clean, one request a piece; a restart with a marker standing fetches only the missing pieces (the requests counted); the rangeless hub refuses; `discard` restarts a file; a short file refuses its digest. Upstream's own piece table asserted. |
| `cli/pull.zig:100-155 fetchBig` | 60 | `pull.rs`'s `fetch_big` | The caller wired: the large LFS entries collected with their staging files (`.ranges` beside the blob), the ranged fetch, the `RangeUnsupported` fallback to one stream a file, **each file's sha256 verified before it becomes a blob** — a mismatch discarding and fetching the file once more, a second mismatch refused ("the bytes do not match the hub's sha256 twice"), the whole staged file renamed into blobs, the "MiB in N pieces" line. With it: `download_retrying`, the single-stream path gaining upstream's retry shape (pull_file.zig:26 — five attempts, the wait doubling from 250 ms, a sha mismatch restarting once). The single-stream **mid-file range resume** (pull_file.zig's `.partial` + the on-disk prefix feeding the same digest) remains the named next piece. |

## Ported (Phase 2, thirty-second tranche — the key gate and the reply counter)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/server/auth.zig` | 380 | `crates/server/src/auth.rs` | The digest store whole: every key SHA-256'd, the matches timing-safe; the sources — `--api-key` (labels `cli-N`), `TENSORFOLD_API_KEY` comma-split (labels `env-N`, **always merged** — the port had read it only when no flag key stood), and the key file (`label: key` named lines with the label rules, bare lines `file-N`, comments skipped, ≤1 MiB, UTF-8, **a regular file others cannot read** — the chmod-600 refusal — opened NOFOLLOW); the file **reread at most once a second when its stat stamp changes or SIGHUP asked**, a refused reload closing the file's keys while the static ones stand, a request's label outliving the reload. Upstream's four auth tests ported whole (the gates/digests table, the init refusals, the file's refusal/read/reread, the label's outliving). |
| `server.zig:249 authenticate` + `:257 countReply` | 40 | `routes.rs`'s `auth_and_count` middleware | The dispatch-level gate the per-handler checks became: OPTIONS passes (the CORS preflight), the exact `gated()` path set (the `/metrics` exception under `--metrics-open`, the `endsWith` tails), the 401 — the corpus's own body (`param: null`, `code: invalid_api_key`, the `WWW-Authenticate: Bearer` header, the connection closed) and the Messages paths Anthropic's own shape — and **every reply counted** under its key's label or `unauthenticated` (the `requests_total` family, never wired before this tranche). The per-handler auth checks swept away (20 sites), the handlers back to their own business. |
| `auth.zig:16 installHangup` | 6 | the serve's SIGHUP handler | SIGHUP flags the store's reread (the handler only sets the atomic — async-signal-safe). |

## Ported (Phase 2, thirty-first tranche — the client-disconnect cancellation)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `chat.zig:423 Generation.cancel` + the gone checks | ~30 | `chat.rs`'s bridge shell + the routes | A streaming client that leaves mid-reply cancels the engine's request: the SSE consumer's drop closes the bridge's chunk channel, and the bridge's failing send — upstream's sink-emit failure — cancels the request, waits for the engine's own end, and counts the disconnection when the engine reports the cancellation (`chat.zig:426`'s `disconnected` on `.cancelled` with no own reason). The **request ids made unique** first: the port had hardcoded id 1 for every submit, so concurrent streams collided in the cancel queue — a latent bug the tranche caught — now a per-server counter keys each. Wired on all four streaming surfaces (chat, Messages, Responses, completions). The E2E probes the corpus's cancel case live: a socket leaves mid-reply, `client_disconnections_total` counts it, and a fresh request answers (the lane freed). |

## Ported (Phase 2, thirtieth tranche — the parse errors and the dispatch tail)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/server/json_parse.zig` | 336 | `crates/server/src/json_parse.rs` | ``json.loads`` on the request bytes with CPython's own decoder and scanner wordings: the BOM dropped (or refused for already-decoded text), the UTF-8 codec errors (``'utf-8' codec can't decode byte 0x… in position N``, the continuation and end-of-data cases), every scanner failure in Python's ``msg: line L column C (char N)`` format — Expecting value, the property-name and ':'/',' delimiters, the Illegal trailing comma, Extra data, Unterminated string, the control characters, the \escape and \uXXXX forms, the 4300-digit limit, the 900-deep recursion answer — and **``-0`` canonicalized to the int ``0``** (the number-edge gap's first half closed). The values land in serde_json: ints past its widths fall to f64 and the NaN/Infinity literals read as null (the remaining named gap — nothing pins them). The corpus's pinned messages byte-verified against the running Zig server through the golden harness. |
| `openai.zig:120` the chat/completions body read | — | `chat.rs`'s `chat_completions` + `completions.rs` | The routes read raw bytes through the new parser — the port's **extractor gap closed** for both surfaces: a malformed body answers Python's own message (the bare Other-kind body, exactly the corpus's `{"error": {"message": …}}`), a non-object body Python's AttributeError (`fields::require_object`: ``'list'/'NoneType'/'str' object has no attribute 'get'``, upstream fields.zig:26's kind and wording), a missing or empty messages the normalize's own **non-empty** wording (the port's `prepare` had short-circuited it with a divergent "must be a list" — a word-for-word catch). |
| `auth.zig:248 routePath` + routes.zig's dispatch tail | 12 | `routes.rs`'s fallback | The odd paths the status group pins: the query stripped and trailing slashes trimmed — an empty route and `/health` the health route, **any** route ending in `/models` the models one (`/x/models` serves!), `/v1/health` the unknown path. Router tests over every shape. |
| `status_routes.zig:26` the health reset | 2 | `routes.rs`'s health | `?reset_peak=1` resets the engine's peak before the counters read. |

## Ported (Phase 2, twenty-ninth tranche — the legacy completions)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `openai.zig`'s `is_chat=false` half | ~150 | `crates/server/src/completions.rs` + `chat.rs`'s `prepare_completion` | The whole legacy surface: the plan (`legacyPrompt` — token ids as given with the vocabulary's own range refusal, bools reaching the ids branch and refusing on the value; anything else `promptText`: a string itself, null empty, an all-int list decoded, a nested list joined on newlines, anything else Python's text), messages riding the template when a non-empty array came, a prompt's ids the plain encode **with the specials** (prompt.zig:81) or as given, thinking off for a prompt (chat.zig:161), no tools (openai.zig:127 reads them for chat alone), logprobs refused (the model's own refusal first on Metal — the completions-only wording behind an engine that reports rows). The shared tail extracted (`finish_prepare`: the window refusals and clamp, the budget, the call gate, the structure, the sampling, the stops, the request build) — the chat plan and the completion plan one tail, upstream's one prepare. The whole reply `wholeBody`'s else branch (`text_completion`, the choice's `text`, `logprobs: null`, the `cmpl-` id) with the same extras; the stream the bridge's completion shape (`chunk_body`'s `is_chat`: the prose in `text`, the role chunk never opening) and the context refusal inside the stream. |
| `openai.zig:523-533` the usage chunk | 11 | `chat.rs`'s bridge | `stream_options.include_usage`: the usage a chunk of its own — empty choices after the last — for the completion **and the chat** stream (the port's chat path had always inlined it; upstream's one mechanism serves both routes, the corpus's usage-chunk case pins the shape). |
| `prompt_cache_test.zig`-style live parity | — | `golden.rs` | The golden harness grew a completions leg: both servers, the same greedy text and prompt-token count — the raw prompt's reply (the think markers passing through as text, the completions' own behavior) byte-equal through the real engine. The E2E battery probes the corpus's cases live: the text/ids/number/nested/messages shapes, the empty-prompt and range refusals, the streamed chunks, the usage chunk on both surfaces. |

## Ported (Phase 2, twenty-eighth tranche — the Store-side learn flow)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `core/prompt_learn.zig` | 146 | `crates/lanes/src/prompt_learn.rs` | The learn flow whole, single-rank: the pending cleanup first (an unchecked cleanup refuses, upstream's UncheckedLearnedCleanup), the payload arithmetic (the state's bytes, the 256-byte and token/start allowances, the index's scratch beside), the disk shortfall, the victim plan across this identity's metas (their reclaims, the `1<<63` recency offset that sweeps the other identities first) and the others' trees, the deletions, the reserve, the journal, the family's write, the index, the journal cleared — every refusal the bounded backoff's. The peer half (`peer_need`/`peer_reserve`/`peer_finish`, the paired vtable's ACK failures) is the fabric tranche's; the flow's shape leaves its seams. |
| `prompt_cache.zig`'s learned slots + `recall` | ~110 | `prompt_cache.rs` | The `Snapshots` learned slots — `write`/`read`/`forget`/`forget_checked`/`reclaim`, each an optional the null vtable entry is, with the presence probes the null checks are — and the single-rank helpers (`single_reclaim`/`single_forget_checked`, the `{key}.bin` file). The Store's `imprint` (owned where upstream borrows the host's, `imprint`/`imprint_mut` the observation); `keep`'s learn calls on both shared paths, the cut's `starts` carried; and `recall` in the lookup — a learned state longer than memory's read back as a shared entry, the span-carrying requests refused it, the unreadable one unlearned and forgotten, the touch after the read. |
| `core/prompt_cache_fake.zig` | 208 | `crates/lanes/src/prompt_cache_fake.rs` | The shared test family, extracted from the store's tests and grown its learned half: the state file `{key}.bin`, the counted writes and forgets, the failed write and failed checked deletion, the learned flag riding where upstream's vtable choice does — the pass reset keeping it (upstream's reset list, prompt_cache_fake.zig:181). |
| `learned_fault_test.zig` | 241 | `prompt_learn.rs`'s tests | Five of the eight, assertion for assertion: the null clock (no eviction, no half of a write), the other-identity cap pressure planned before the write (the sweep), the missing indexed half (its reclaim cannot promise, the valid victim survives), the failed rollback deletion surviving the restart (the journal stands, the recovery before the cap admission — the failures driven through the family's own write and checked deletion, upstream's peer write aside), and the eight torn `used` stamps (the store learns over each). The peer-ACK case is the fabric tranche's; the unlink and admission cases already ride their own modules' tests. |
| `prompt_cache_test.zig:290/:320` | 85 | `prompt_cache.rs`'s tests | The two end-to-end cases: a learned harness state outlives its store — a fresh session on a **new** store resumes it from disk, the pass equal to a fresh one (`from 5` through the recall); and past the cap the least recently used goes, its file with it, while a file lost behind the store's back unlearns its state — the next pass learning it again. |

## Ported (Phase 2, twenty-seventh tranche — the learned-disk machinery)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `core/lanes/learned_dirs.zig` | 123 | `crates/lanes/src/learned_dirs.rs` | The bounded-name identity directories: `next` in id order (a malformed tree an error, not a skip), the acknowledged `unlink` (a directory refuses, an absent file fine), `bytes` (a non-file in the tree a type error), `fileBytes`, `otherBytes`/`totalOthers`/`total`, the `used` recency hint (a torn read oldest, an I/O error refused), `removeOther`. Upstream's unlink test ported. |
| `core/lanes/learned_disk.zig` | 126 | `crates/lanes/src/learned_disk.rs` | `available` — the port's own `statvfs` extern with the darwin struct shape (the rusage lesson: the layout tested against a real read) — and the `Admission`: the free-disk floor, the in-flight reservation, the bounded backoff (failures doubling to a 60-second ceiling, a grown free space cutting through), the clock and free reads injectable exactly as upstream's test seams. Upstream's floor/overflow/unknown-space/backoff test ported case for case, and the fails-closed real-statvfs test. |
| `core/learned_plan.zig` | 45 | `crates/lanes/src/learned_plan.rs` | The victim planner: one LRU set that must satisfy both disk floors and the cap together — an impossible shortfall deletes nothing. Upstream's test ported case for case. |
| `core/prompt_imprint.zig` | 392 | `crates/lanes/src/prompt_imprint.rs` | The Imprint: the append-only index (records little-endian, a torn tail ends the read, duplicates folded, the load bounds upstream's own), the crash journal (`pending` — 48 bytes, the "IMPRINT1" magic, a Sha256 checksum, `begin_write`'s atomic part-rename, a second write refused while one stands), the LRU (`touch`, `victim`, `best`'s prefix-and-starts match with `same_below`'s exact-`at` rule), the cross-identity sweep (least recently opened first, `fits` under the cap), the open stamp (root/clock + dir/used, each through a part-rename), `os_build` (sysctl kern.osversion), `default_root`, `make_path` (0700 each level). Upstream's three imprint tests ported whole — the reopen (the `best` orderings assertion for assertion), the LRU, the sweep (the 108-byte identity arithmetic included) — plus the two I/O fault tests, and the ownership test in the form Rust's ownership makes automatic. **The keys**: `key_of` is Wyhash (seed 0x1e47) over the tokens' little-endian bytes through the `wyhash-final4` crate — the reference port with the reference secret, the algorithm Zig's `std.hash.Wyhash` is built on, so the learned trees stay byte-compatible between the engines; the definitive Zig-side cross-check is the named gap below. |

The follow-up tranche (twenty-eight) landed the Store-side flow: the
`Snapshots` learned slots, `prompt_learn.zig`, the lookup's `recall`, the
fake family's disk side, and the fault suite's single-rank cases — the
`peer_*` pair still the fabric tranche's.

## Ported (Phase 2, twenty-sixth tranche — the live-parity fixes)

The tranche's gate: both servers — the Zig binary and the port — on the
pinned checkpoint, every observable surface diffed live. What it caught
and fixed, each verified against the running Zig server afterward:

| The divergence | Upstream | Rust | The fix |
|---|---|---|---|
| The prompt cache on the pinned family | qwen35.zig arms no store: a rerun's `cached_tokens` stays 0, /v1/memory has no plan section, no cache lines | The tranche-21 wiring: 5,358 cached tokens on a rerun, the plan, the nemotron serve lines | The disarm — `with_cache`, the budget plan and the `prompt_cache` Info flag all went from this family's serve; `prompt_cache.rs`, `snapshot.rs`, `cache_fit.rs` stay as the ported machinery for the family that arms one, their tests green. The E2E now asserts the upstream behavior (a rerun reports 0, the greedy reply identical) |
| The serve log | `[tensorfold]` lines on stdout: the chunk-plan line (server.zig:114, thousands commas) then the serving line of serve.zig:150 — `serving {name} at http://{host}:{port}/v1 (sampling: {s}; drafts: {d}; context: {c}; loaded in {t}s)` — nothing while loading | stderr lines in the port's own format, the loading/loaded lines, the cache lines | Upstream's exact two lines on stdout, the `commas` formatter, `shownSampling` (the defaults from generation_config.json under the flags, main.zig:104), the load time in the serving line; the loading/loaded lines went. The E2E reads stdout now |
| The unknown paths | `{"error": {"message": "unknown path {p}"}}` (routes.zig:82) | `{"error": {"message": "unknown route", "type": "invalid_request_error"}}`, and axum's bare bodies beyond the registered routes | Upstream's exact body from the gated routes and a router-level fallback carrying the original path — unmatched paths answer upstream's words |
| The /metrics render | metrics.zig:97-158's order, family for family: the kv_cache ratio and perc gauges (a pool-0 line when no streams), the keeps-gated cached and prefix families, the process gauges, the device gauges when the engine counts, the decode help's whole sentence, disconnections last | The port's own order, the kv_cache families absent, `plain_only` wrong | The render rewritten in upstream's order with every family; `/metrics` now byte-identical between the two servers (202 lines, only the live process gauges apart) |
| The prefill step | qwen35's own `const chunk = 128` (qwen35.zig:52) — the step the chunk plan floors to | The nemotron path's device rule (2048 here) | The family's constant, `Options::DEFAULT_CHUNK`; the chunk-plan line now says 128 as the Zig server's does |
| `plain_only` | qwen35 never sets it — "drafts: on" in the serving line | `plain_only: true` ("the 2B checkpoint has no draft head") | Unset, as upstream's Info leaves it |

## Ported (Phase 2, twenty-fifth tranche — the JSON writer)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/server/json_write.zig` | 170 | `crates/server/src/json_write.rs` + `PythonJson` | ``json.dumps`` byte for byte: the `", "`/`": "` separators, the short escape table, the `\u00xx` of the controls and DEL, `ensure_ascii`'s `\uxxxx` and the astral planes' surrogate pairs, and the float text — Python's `repr` with the NaN/Infinity spellings (the port's `pyrepr::float_repr` already was the rule; `float_json` wraps it). Every JSON reply now carries these bytes: `PythonJson`, a responder with the `application/json` type, replaces axum's `Json` at every route, and every SSE event's `data:` payload is pre-serialized through the same writer. Upstream's own test cases ported verbatim (the float repr table, the spaced/compact dumps case), a corpus body byte test, and — the tranche's gate — a **live byte-parity test in the golden harness**: both servers under the same name, the models route, the tokenize ids and a frozen corpus refusal body all byte for byte, the corpus's own bytes the reference. |
| `native/qwen35.zig:38-56` the window rule | 19 | `crates/cli/src/serve.rs` | **A live byte-caught port bug**: the port served with the checkpoint's raw window in the `Info` (262,144) whatever `--context` said, and silently clamped an oversized flag; upstream serves under the flag or the native window, refuses one outside its range with "--context must fit the Qwen checkpoint's positive context window", and the backend's capacity and the reported window are the same one. The models route, the context refusals' wording and the memory snapshot all read the served window now. |
| `status_routes.zig:37` the models `created` | 2 | `routes.rs`'s models | The other byte-caught one: `created` is the clock read per models request, not a constant 0. |

## Ported (Phase 2, twenty-fourth tranche — the Responses API)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/server/responses_translate.zig` | 344 | `crates/server/src/responses_translate.rs` | A Responses request as the chat completion that runs it: the input items as chat messages (a turn's reasoning, text and calls one assistant message, a call's output a tool message, the call output's text parts joined), the offered tools, the tool choice (none/auto/required/a function/`allowed_tools` filtering by name), `text.format` as a `response_format` (text/json_object/json_schema), and the echo object the Response carries — every field with its default. Every refusal wording mirrored char for char against the frozen corpus (`zig/tests/server/golden/responses/`): the four refused fields, `include` (each entry pyStr'd, an object's keys, a string's characters), the truncation, the metadata bounds (16 pairs, keys 64, values 512), the input shapes, the item and content-part types in Python's quotes, the missing parent. Six tests. |
| `src/server/responses.zig` | 486 | `crates/server/src/responses.rs` + the routes | The whole surface: the in-memory store (1,024 entries, 256 MiB, put replaces by id and evicts oldest, a stored response's messages the added ones plus its output items translated back), the chain walk for `previous_response_id` (each read entry moves newest, the missing parent the exact refusal), GET/DELETE on the collection and its ids (the 404s naming the id with `param: response_id`, the empty id for the collection, the unknown-path refusal for a GET on the collection and a POST on an id), and the Reply machine — reasoning, message and function_call items in order, their `response.*` events with counting sequence numbers, the item close on a kind change, the usage mapped, incomplete on a length with its reason, an all-reasoning completion still a message, the store put on a finish, `response.failed` on an error chunk or a stream that ends early (nothing stored). The streamed path rides the chat bridge; the context refusal streams created, in_progress, failed. Ten tests, plus the router-level 404s in `routes.rs` and live E2E probes (the whole object, the stored chain with GET/DELETE/orphan, the streamed events, the refusal shape). |
| `openai.zig:105 replyModel` | 8 | `crates/server/src/chat.rs` `reply_model` | The model a reply names: the request's own when it names a served id, else the served name. The chat route's whole and streamed bodies had hardcoded the first served id — right for a one-id server, wrong under aliases; both now read it, and the Responses base does too. |
| `openai.zig:376 wholeBody` | 30 | `crates/server/src/chat.rs` `whole_body` | Extracted from the chat route's inline construction: the Responses route feeds its machine the very body the chat route sends (choices, usage, the extras) — one whole-body builder both share, as upstream's one `wholeBody` serves both. |
| `openai.zig:267` the chunk's delta wrap | 2 | `crates/server/src/chat.rs` `chunk_body` | **A live-caught, port-wide bug**: upstream wraps a bare string delta as `{"content": text}` at the chunk body; the port passed it raw, so every streamed prose chunk since the real-streaming tranche carried `"delta": "text"` where upstream sends `"delta": {"content": "text"}`. The whole-reply golden couldn't see it (the wire shape, not the content, was wrong) and the E2E's loose `contains` didn't pin it — but both translated surfaces starve: the Messages machine and the Responses machine read `delta.content` and got nothing, so an Anthropic or Responses stream opened and closed with no text. The wrap restored, a unit test locks the shape, and the E2E now asserts the wire shape on the chat stream and probes a Messages stream live for the first time. |
| `messages.zig:137 textPart` | 4 | `crates/server/src/messages.rs` `text_part` | Shared by the Anthropic and Responses translations as upstream shares it; the Anthropic translation's private copy folded in. |

## Ported (Phase 2, twenty-third tranche — the tokenize routes)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/server/tokens.zig` | 114 | `crates/server/src/tokens.rs` + the routes in `routes.rs` | vLLM's /tokenize and /detokenize (both under /v1 too). Tokenize: a body with messages runs the chat route's own machinery for the ids it would run (normalize, tool arguments, the active tools, the thinking fields, the template render); a plain prompt encodes, `add_special_tokens` honored. `return_token_strs` carries the pieces beside the ids. Detokenize: one prompt a request (the nested single-prompt shape un-nests), every id an integer then in the vocabulary's range. `max_model_len` the engine's window when it reports one. Tests: the validations and the flags; the E2E gate probes both live (a round trip, the messages render, the token strings, a special id). |
| `hf_text.zig:225 encodeFn` | 14 | `crates/core/src/tokenizer.rs` `encode_special` | The add-specials encode: upstream hand-reads the post-processor's specials into a prefix/suffix (TemplateProcessing, Bert/Roberta, Sequence recursed); the port runs that very processor — the crate's `encode(text, add_special_tokens=True)`, the reference behavior upstream imitates (ADR-003). Tested: the wrap on the fixture's template. |
| `hf_text.zig:233 decodeFn` | 8 | `crates/core/src/tokenizer.rs` `decode` | A fidelity gap the tranche caught: upstream's one shared decode keeps flagged specials as their own text (`skip_special_tokens=false`); the port's dropped them. Invisible on the pinned family — its think markers are `special: false` added tokens, and the eos is stripped before any reply decode — but a raw special id (an eos on /detokenize) came back empty. The decode keeps specials now; the golden harness still byte-identical. |

## Ported (Phase 2, twenty-second tranche — the remaining server surfaces)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `status_routes.zig` | 175 | `routes.rs`'s health + dashboard + stats | /health upgraded from the Phase 0 stub to the status line's copy: status ok, the model and its ids, the lane bound, warming, the memory counters, the live line (the connections, the waits, the decode/prefill rates). /dashboard the opt-in one-file page (dashboard.html copied byte-identical, served when --dashboard); /stats its JSON poll — the counts, the rates, the running tokens, the memory. The metrics' counters gain the totals read. |
| `live.zig` | 40 | `crates/server/src/live.rs` | /health's live: the connections (running + waiting), the waits, the decode and prefill rates rounded to one decimal (Python's round(x, 1), tested). The terminal line's other half rides with the CLI's serve loop, which the port's axum serve does not draw. |
| `memory_routes.zig` + `process_memory.zig` | 170 | `memory_routes.rs` + `process_memory.rs` | The authenticated sizing snapshot: the schema, the window and its fitted flag, the process footprint (rusage_info_v4, the indices verified by a live reading — the port's first guess at the ABI's field order failed its own test, the fix the same field upstream's extern struct names); the device counters (the engine's Memory); the prompt-cache plan's every field. Every unknown omitted. Two tests (the omission rule, the live reading). |
| `request_log.zig` | 40 | `request_log.rs` | Each chat body appended as one JSON line, every image part's URL redacted, the background work skipped. Wired at the prepare (openai.zig:123), the Config's TENSORFOLD_REQUEST_LOG already set. Two tests (the redaction, the one-line append). |
| `fields.zig:280 validateModalities` | 16 | `fields.rs` `validate_modalities` | Text in, text out: the six media fields refuse, the modalities array its one text shape passes. Tested over every field and the shapes. |
| `warm.zig` + openai.zig:363 warmNext | 85 | `warm.rs` | The next turn's prompt prefilled in the background after a whole reply: the message as its client got it (the content null when calls), the shared tokens found by two renderings agreeing up to the next turn's own text, the warm request (prompt-only, background, its whole prompt the history, the rewind, the chunk starts, the reply spans). The Reply gains the thinking and effort it carries upstream (chat.zig:43). The Info's warm_turns gates it — the pinned family never reports it (upstream only the Qwen 2.7 host does), so the port's reply is never prefilled ahead either; the machinery is ready for the family that sets it. |
| `decisions.zig` | 443 | `decisions.rs` + the route | The whole /v1/decisions route: the unknown-field refusal (the first unknown named, sorted), the version check, the temperature (a number above 0), the kwargs (enable_thinking false or unset), the ids (blank refused, repeats named); the wordings — a choice's 2-to-26 alphabetised options (names not blank or line-broken, repeats folded), a score's 2-to-10 levels, a yes_no pair; the prompt rendered with thinking off; the answer position's open think block refused (`THINK_OPEN` searched — not the channel markers' empty open, a live-caught bug: `rfind("")` matches everywhere); the context fit; the labels one fresh token each; the scoring through the engine hook (the port's Metal host refuses `Unsupported`, exactly the pinned family's upstream behavior — "decision scoring is not supported by this engine yet"). Seven tests: the validations, every wording, the think check, the softmax and label mass. |
| `server.zig`'s engine keepalive | — | not ported | The idle GPU ticker (`core/keepalive.zig`, armed by --keep-warm) needs the engine's keepaliveTarget the Metal backend does not report in this port; the flag's serve-level handling (the keep-warm seconds in the Config) is present. |

## Ported (Phase 2, twenty-first tranche
## Ported (Phase 2, twenty-first tranche — the retained-prefix store live)

> **Erratum (tranche twenty-six):** this tranche wired the store onto the
> pinned family, citing upstream's *nemotron* path (metal.zig:154-166) —
> but upstream's qwen3_5 host (qwen35.zig, 61 lines) arms **no** store at
> all: it serves cache-less, its reruns report no cached tokens, its
> /v1/memory carries no plan. The tranche-twenty-six fixes below disarmed
> the port's wiring to match; every entry in this table stays as the
> faithful port of the machinery, waiting for the family that arms one.

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `families/nemotron/snapshot.zig` | 144 | `crates/qwen35/src/snapshot.rs` | The kept state for this family's layout: the linear layers' delta recurrence and conv (fixed bytes) plus the attention layers' key and value rows for [0, at), copied on the host (the state's buffers shared memory, the pass's commit synchronous); `bytes` (the storage a state takes), save, restore (the committed state replaced, the rows the cache's length), the `Cached` Snapshots face (save through the backend's per-stream cache, `restore` = `error.BackendRestores` — the pass restores itself), and `restore_reuse` — the backend's own restore from the Reuse's pointer. The size test mirrors upstream's. |
| `families/nemotron/backend.zig:292-310` | 20 | `crates/qwen35/src/backend.rs`'s prefill | The reuse path: a kept prefix restores into the fresh cache before the first chunk (the resumed position the stream's `cached`, a failed restore `reuse_failed` from 0); each mark the pass stands at fires the hook (the cache keeps the state there). The cache registers before the loop — the hook's save reads it mid-pass (the ADR-013 topology: the pass's mutable use and the hook's between-chunk reads do not overlap). |
| `lane_host.zig:41/60-83/260/339-345/384` | 50 | `crates/lanes/src/lane_host.rs` | The host's store (`with_cache`, behind an Arc's uncontended lock for the hook's re-entry), the Job's entry/marks/kept0/started, the submit's lookup (the Reuse the Spec carries — the saved state's pointer, the marks leaked to 'static, the kept closure), the kept hook (the report first, the store's keep), the reported (the restore stood or failed; the borrow releases), the prefilled moment's report, the end paths' (`take_cancels`'s both, `deliver`'s finish, `fail_all`). **A live-caught deadlock**: the admission held the job's mutex through the prompt pass while the hook re-entered the same job — the Request's facts are all Copy, so the admission reads them under one short lock, drops the guard, and the pass runs unlocked. |
| `native/cache_fit.zig` | 99 | `crates/cli/src/cache_fit.rs` | The budget plan: fit (the 70% cap, the room past the footprint and the 2 GiB margin, the given-budget refusal unless `over`), the RAM and footprint readings (sysctl's hw.memsize, proc_pidrusage's ri_phys_footprint), the plan the Info reports. Both upstream tests mirrored assertion for assertion. |
| `native/metal.zig:42-45/152/154-166` | 16 | `crates/cli/src/serve.rs` | The prefill step (M5-or-later with a 48 GiB working set 8192, else 2048 — the step the chunk plan and the planned marks floor to), the Info's `prefill_step` and `prompt_cache_plan`, the store constructed over the backend's snapshots with the Metal rules (lookahead 1, planned), a 0-byte budget running the host without it. The device's name and working-set cap read through the backend. |
| `cli.zig:69-70/243` | — | the serve flags | `--prompt-cache-gib`, `--prompt-cache-over-cap`. |

The E2E gate probes the cache live: a 5,400-token conversation's rerun **resumes 5,358 cached tokens** and answers identically to the un-cached first run — the store's core invariant (a resumed pass equals a fresh one) held on the real Metal lanes.

## Ported (Phase 2, twentieth tranche
## Ported (Phase 2, twentieth tranche — the retained-prefix store's core)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `core/prompt_cache.zig` | 528 | `crates/lanes/src/prompt_cache.rs` (new) | The exact-prompt-reuse Store whole: the Snapshots contract (the four required methods — bytes, save, restore, drop — plus the optional charged/spare/trim/reuses; the port a trait with the optional ones defaulted, the Send bounds off as the store lives on the host's own thread), the Rules (lookahead, planned, grid, min_gap, min_prompt, warm), the Entry with its **stable id** — upstream hands pointers around while the port's Vec shifts on eviction, so the lookups and the victim choice carry ids instead (a faithful equivalent of the pointer identity, the one place the port's shapes differ), find (a kept token prefix with matching canonical spans through its position, leaving at least one row), keptNear (the rewind served by a state under min_gap before it), begin/beginRewind (restore the longest prefix, plan marks; a failed restore prefills from the start and drops the state), lookup/lookupRewind (the endpoint's and rewind's bytes weighed, the shared cuts noted, the room reserved before the pass), reserve (the evictions made before it), fitting/fittingRewind (the budget's selection), resumed, marks/marksRewind (the history, the rewind however near, the stable prefix with the last prompt, the shared blocks, min_gap apart, past the resume point, before the end), keep (the same state again no copy, spare storage first, the trim inside the budget, superseded-first eviction), victimBut/before (superseded oldest-first, the rewind last, shared only as LRU), clear, forget. |
| `core/prompt_cache_test.zig` + `prompt_cache_rewind_test.zig` | 774 | the test module | **Nineteen tests**, the upstream core and rewind suites mirrored assertion for assertion: the growing conversation resuming each turn, the lookahead keying, the planned chunk starts, the grid, the eviction order and the refused past-budget state, the failed copy and restore, the marks' every rule, min_prompt, the shared system cut outliving cold prompts, the rewind edit-resume, the one-state budget's endpoint priority, the planned floors, the warm endpoint, the grid rewind, the evict-rewind-last, the shared first turn, the refused count, the min_gap resume, the warm pass's keep. The fake family shares its state through a handle the way upstream's vtable pointer does (the pass reset keeps the failure flags — a first port bug the tests caught). The learned-disk and peer suites (prompt_imprint, --learn, the paired vtables) ride later with that layer. |

Not yet wired: the lane host's `cache` hooks (lookupRewind at admission, the Job's entry/marks, the keep through the pass — lane_host.zig:339), the family-side Snapshots for the Metal state (families/nemotron/snapshot.zig's equivalent in the qwen35 backend), cache_fit's budget plan, and the serve flags (`--prompt-cache-gib`, on by default upstream — the pinned server currently runs without the store, a divergence this tranche's core readies closing).

## Ported (Phase 2, nineteenth tranche
## Ported (Phase 2, nineteenth tranche — prompt caching's request fields)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `chunk_plan.zig` | 169 | `crates/server/src/chunk_plan.rs` (new) | The prefill chunk starts: the Plan (the step, the openers, the assistant header, the min chunk), the resume points, the starts (the first marker `min_chunk` or more on, else step tokens on — too-near markers merged into their chunk), withCut (a harness cut joined, nearer starts dropped). The message markers read from probe conversations at open (four exchanges rendered pairwise, the pieces every message of a role starts with, a template that renders the last reply differently gaining its opener past the difference, the assistant header common with a special first token). Both upstream tests mirrored assertion for assertion. |
| `prompt.zig:15-30` — renderIds + the dangling close | — | `chat.rs` `render_prompt` | The rendered prompt with a dangling think opener closed: with thinking off and a generation prompt, a template ending on the opener the reply writes itself (upstream's separate `think_open` constant, reply_text.zig:8) gains the close token. **The golden harness caught a real bug here**: the port's first check compared against the markers' own open — upstream's `think_markers.open` is genuinely the empty string (templates that use these markers leave the opening to the template) — so every whitespace-ending prompt matched the empty constant and gained a spurious close token, diverging the prompt by one byte and the reply by a leading newline; the byte-exact comparison caught it immediately. |
| `prompt.zig:104-125` — history and rewind | 25 | `chat.rs` `history_len`/`rewind_len` | The no-generation render's length when it prefixes the prompt (a template with no suffix falls to the prompt less one, so nothing retains everything); where the latest user message starts — its content swapped for the invisible separator, the first difference — a caching engine only (the Info's `prompt_cache`). history_len tested. |
| `prompt.zig:127-134` + `chat.zig:173-177` — the system prefix and the shared cuts | 10 | `chat.rs` `system_prefix_len` + the request build | The reusable system prefix (a probe in place of the first user message, 512 tokens at least), with the shared-prefix cuts [len-2048, len-512, len] the request carries. |
| `prompt.zig:36-82` — generation prompts and reply spans | 25 | `chat.rs` `generation_prompt`/`reply_spans` | The template's generation suffix (with and without thinking, probed once a request), and the earlier replies' spans — from each turn's generation prompt end through its closing stop token — onto `Request.decode_spans`, which the lane core validates (the construction always canonical). |
| `chat.zig:192/197` — the chunks and the tools json | 4 | the request build | The prefill starts with the harness cut (the system prefix less one, so a fresh session resumes its whole harness); the offered tools as compact JSON on `Request.tools_json` (the port sent an empty string). |
| The request's cache fields | — | the request build | `history_len`, `rewind_len`, `shared_prefixes`, `chunks`, `decode_spans` all set — every field of upstream's request build that the port left defaulted. The engine's retained-prefix store that consumes them (`core/prompt_cache.zig`, 528 lines) is the lane-side feature they await, as the lane host's doc notes. |
| `server.zig:113`/`hf_text.zig:215` | — | `state.rs` `with_engine`/`core` `is_special` | The chunk plan read once at open when the engine prefill-steps (the pinned Metal engine steps 0, so its plan is the default — the code path complete, the pin unchanged); a token's added-special check, the markers probe's gate. |

## Ported (Phase 2, eighteenth tranche
## Ported (Phase 2, eighteenth tranche — compaction)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `compact.zig:44-80` — the thresholds | 40 | `compact.rs` | reserveTokens (auto: the bigger of 15% and 16384; a fraction: the window minus its threshold), noteBudgetTokens (≤8192, four fifths of the reserve), keepTokens (the override else min(20000, a quarter)), threshold (floored, clamped), over (the sum over the reserve or the fraction's threshold). Tested, the boundary sums included. |
| `compact.zig:105-180` — the rebuild | 80 | `compact.rs` `rebuild` | The marker (the last "Memory of the earlier conversation:" message) and the stored note by conversation hash; the skip the covered prefix; the kept tail from keptStart (the earliest cut whose messages fit the keep budget) clamped past the prefix; the split at a kept assistant's turn start; the middle collected; the note and the split each generated through `update`; the retry loop — the paths (the note's own Files section and the conversation's call arguments) read, the note finished (the body without the sections the rebuild writes, the split head, the files head), the conversation assembled (the prefix sans the marker, the note under the template's late-system role, the split as a user message, the tail), re-prepared — a context-length refusal first drops the split, then cuts further back through nextCut with the dropped piece noted in; the store written on success. Upstream's `preparing` release has no port shape (noted with the split). |
| `compact.zig:130-181` — the note generation | 55 | `compact.rs` `update`/`one_note` | Chunks of messages that fit the window minus the note budget, each one a greedy 1024-token generation with thinking off through the shared prepare — the previous note carried, the transcript with the calls' arguments; a context-length refusal retried measured (max_tokens 1, the room past the rendered transcript), then given up keeping the previous note; the note stays a drafted generation when replies pass --no-drafts. |
| `compact.zig:196-211` — the kept tail | 16 | `compact.rs` `kept_start`/`count_range` | The messages counted by their transcript lines' tokens (role, content, the calls), the earliest cut holding the keep budget. |
| `compact.zig:213-237/239/347-530` — the message math | 90 | `compact.rs` | The pure half: prefixLen, findMarker, markerBody, splitIndex, nextCut, lastUser, countBetween, collect, lineOf/transcript/appendCalls, pathsFromNote/pathsFromMsgs/addArgs, addUnique, withoutExtra, finishNote, assemble (the late-system role, the empty user turn). Six tests, the upstream shapes. |
| `compact.zig:83/88/78` — the stamp, the prepare, the run | 25 | `compact.rs` `stamp_reply`/`prepare_compact`/`run` | The stamp (the context fields beside the runtime's, the compaction object when compacted); the compact-aware prepare (the plain one kept when the request fits, the rebuild when it does not or the window refused it); the run (prepare, generate, stamp). The routes: the chat whole path and generate_with go through `chat::run` (the compact-aware entry, chat.zig:258), the stream through `prepare_compact` with the stamp's object riding the bridge (openai.zig:465/521). The chat route had bypassed `run` — the plain generate called directly — found and fixed while testing the live probe. |
| `compact_mem.zig` | 86 | `compact_mem.rs` | The whole file: the conversation key (the first system and first user, compact JSON, the first sixteen hex characters), the covered-prefix hash, the load (the note .md, the covered .json, the bare note without it), the store (the atomic rename). Two tests, the store round-tripped under a temp directory. |
| `server.zig:305` — lateSystem | 6 | `state.rs` `late_system` | The template probe: a system message past the first exchange renders its text — the role the note assembles under ("system" or "user"), read once at open. |
| `serve.zig:119-121`/`cli.zig:243-252` — the flags | 8 | `cli/src/serve.rs` | --compact-at (auto or a fraction in (0, 1]), --compact-keep, --compact-memory. |

The E2E gate probes compaction live: a second server with the fraction forced low and the keep small answers a twenty-turn conversation with its middle compacted — messages_compacted > 0, the kept tail small, the note the model wrote with its own generation counted, the note store's files written, and the reply answered from the rebuilt conversation (29s, the note a real generation).

## Ported (Phase 2, seventeenth tranche
## Ported (Phase 2, seventeenth tranche — the prepare/generate split)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `chat.zig:131 Prepared` + `chat.zig:145 prepare` | — | `chat.rs` `Prepared`/`prepare`/`generate_prepared` | **The structural split compaction needs** (compact runs prepare, discards, re-prepares): the port's `generate` was one monolithic fn; now `prepare` — every check, the render, the context window, the gates, the budget, the request build — answers a `Prepared` (the request plus what the finish reads: the thinking, effort, sampling mode, drafts, stops, the logprobs count, the received instant), and `generate_prepared` submits and finishes it. `generate` is the thin pair upstream's `chat.generate` is. Upstream's `preparing` count (the queue admission compaction releases) has no port shape yet — noted. |
| `chat.zig:258` — the stream through the same prepare | — | `chat.rs` `submit_streaming` | The stream's own duplicated request-build (the checks, the render, the gates — half of submit_streaming) collapses into the shared `prepare`; the stream differs only in its mailbox, as upstream's does. |
| `compact.zig:106-110` — the stamp's context fields | 5 | `chat.rs` `runtime_objects` | The runtime object carries `context_window` and `context_used` on every reply — the part of compaction's stamp that is always on; the compaction object itself arrives with the compaction tranche. The `StreamStart`/`Bridge` carry the window for the streamed last chunk. Tested. |

## Ported (Phase 2, sixteenth tranche
## Ported (Phase 2, sixteenth tranche — structured output)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `grammar.zig` | 101 | `crates/server/src/grammar.rs` (new) | The whole file: the six field names; `spec` — response_format's three names (json_object the bare JSON kind; json_schema needing `json_schema.schema`, a string parsed with the decoder's bare message, a bool the empty schema or null, an object kept as Zig's wide JSON text; anything but text/json_object/json_schema refused with the value's Python repr), the guided fields by name, structured_outputs' sorted keys with the unknown refused by name and the first sorted known key winning; `refusal` — a grammar beside a required call names its field. The full shape and refusal matrix tested, the messages character-for-character. |
| `openai.zig:145-146` | 2 | `chat.rs` both request paths | The refusal before anything else; the grammar fields ride the fields object's passthrough (the spec reads them there, as checkFeatures' `f` does). |
| `server.zig:179-183` — checkFeatures' grammar branch | 5 | `chat.rs` both request paths | The spec on the fields object; the engine must report `structures` (nothing in the upstream tree sets it, so the refusal is the live behavior, as with the call gate); the Structure onto `Request.structure` — the leaked text, `after` the think-close token when thinking rides. |
| `openai.zig:155` — the logprobs structured check | — | `chat.rs` | The exclusions now count the grammar fields' presence, closing the gap the logprobs tranche left. |

The E2E gate probes both refusals live: `response_format: {"type": "json_object"}` refuses with the not-supported-yet message (the pinned Metal engine, exactly as upstream), and a malformed `response_format` refuses with the grammar's own words before the engine check.

## Ported (Phase 2, fifteenth tranche
## Ported (Phase 2, fifteenth tranche — the thinking budget and the loop guard)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server.zig:104-110` — the think close tokens | 7 | `state.rs` `Text.think_close`/`think_close_end` | The tokens that close a think block — the newline, the plain think-close token (the literal verified by hexdump; the channel markers' close does not stand in), the two newlines after — read once at open, empty without the token. |
| `chat.zig:198-204` — the budget and the guard | 7 | `chat.rs` both request paths | The budget from the fields (a truthy non-int reads as none, an unset one as the server's configured default), the loop guard only when the close tokens exist; the close and its end token ride the request only when a budget or a guard asks; the budget clamped to the u32 the engine takes. |
| `chat.zig:207` — the admit's budget | — | `chat.rs` | The logprobs admission now reads the real budget (it took a hard 0; the logprobs exclusion still refuses a truthy thinking_budget first, exactly as upstream orders them). |
| The engine's side | — | `lanes` (was ported) | The stream's budget forcing — the close begun at the budget, the rest forced after it — and the loop guard's think detection; the lane host passes the request's fields through (the `Request.think_budget`/`think_close`/`think_end`/`loop_guard` fields existed unused). |

The E2E gate now probes the budget live: a `thinking_budget: 8` request answers with its reasoning tokens bounded (the forced close), the reply continuing — the whole chain through the real Metal lanes.

## Ported (Phase 2, fourteenth tranche
## Ported (Phase 2, fourteenth tranche — logprobs)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `tokenizer.zig:198 tokenBytes` | 14 | `crates/core/src/tokenizer.rs` `token_bytes` | A token's exact, possibly incomplete UTF-8 bytes; None without a decoder (the context-dependent case). The byte-level alphabet maps each char to its byte, the port runs the crate's decoder on the piece. |
| `fields.zig:265 probabilityOptions` | 16 | `crates/server/src/fields.rs` `probability_options` | n must be 1; logprobs a bool (present-and-non-null, false unset); top_logprobs 0-20 requiring logprobs: true; the unsupported refusal. The full shape matrix tested. |
| `logprobs.zig` | 55 | `crates/server/src/logprobs.rs` (new) | The whole file: `admit` — the rows cut where the text is, thinking needing the end-of-thinking token, the budget refusal — and `value` — after the first end-of-thinking token and dropped framing newlines, a row a token (the rows out of step or a forced token failing the reply), each `{token, logprob, bytes, top_logprobs}` with the token's own bytes whole. `admit` tested. |
| `openai.zig:124/150-158` — the request gates | 10 | `chat.rs` both request paths | The exact_bytes gate (logprobs asked, the engine reports them, the tokenizer gives bytes); the exclusions — nonstreamed chat without tools, stop strings, structured output or a thinking budget — with the streamed path refusing them outright. |
| `chat.zig:72/92/207/552` — the rows and the value | — | `chat.rs` | The Mailbox collects the Logprobs events (the stream mailbox drops them, as streams refuse); the request carries the admitted count; the finish builds `choices[0].logprobs` over the eos-stripped tokens, its failures other-kind; the whole body carries the field when the request asked (openai.zig:395). |

The pinned model's live behavior, asserted by the E2E probe: **the Metal lanes have no first-row producer upstream (only the CUDA lanes and the fake test backend implement one), so Metal-served models refuse logprobs** — "logprobs are not supported by this model or backend" — and the port refuses identically (the engine reports `logprobs: false` on Metal, exactly as upstream's Info defaults). The value formatting rides backends that report rows.

## Ported (Phase 2, thirteenth tranche
## Ported (Phase 2, thirteenth tranche — tool_choice and the call gate)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| The render's tools | — | `chat.rs` all three render paths | **A render gap fixed**: the core's `RenderOptions.tools` existed but the server never passed it — templates rendered without the offered tool definitions, so a request with tools prompted a model that had never seen them. All three paths (generate, streaming, count_tokens) now render with `tool_specs.active`'s list. |
| `tool_specs.zig:58 active` — the choice filters | — | `tool_specs.rs` (was ported) + the wiring | The active list now flows everywhere upstream's `input.tools` does: the render, the fields flag, the gate, and **the parse** — `attachCalls` and the streamed bridge parse calls against the filtered list, so `tool_choice: "none"` leaves a reply's call markup as text and a named function gates the parse to itself. The chat and Messages routes compute the list; its refusals (the tools-shape and unknown-tool messages) are `other`-kind. |
| `openai.zig:154` — tool_call_required | 2 | `chat.rs` both request paths | The fields object carries the flag a required or named choice sets, with tools offered. |
| `server.zig:184-203 callForm` | 20 | `chat.rs` `call_form`/`find_call_form` | The template's tool-call form read from a rendered probe (an assistant tool call, no thinking, no generation prompt): the last call opener before the probe's function name — ties to the greater opener — the text between them the lead, the one byte after the name the tail; a failed render reads as no text; an opener the tokenizer knows alone counts when the probe shows no name. The pure finder tested over the XML, JSON and Gemma probe shapes. |
| `server.zig:162-178 checkFeatures`'s gate branch | 17 | `chat.rs` `call_gate` | The required choice's gate: the template must mark calls (its opener a token the tokenizer knows — the refusal names the openers upstream names, verified by hexdump), the engine must enforce gates (the second refusal — `Info.call_gates` is false everywhere in the upstream tree, so the gate never sets in practice, exactly as upstream); the gate names the offered tools for the XML form, the think block it opens past (the markers' open token, else the plain think opener), and rides `Request.call`. |

Not yet: the gate's lane-side consumers ride unported paths (lane_host.zig:393's `loneFits` condition — the lone-drafting path; serial_host.zig:185's refusal — no serial host in the port), and the generation-side forcing itself does not exist in the upstream tree either (the gate is server wiring awaiting engine support, which is why `call_gates` is false everywhere). checkFeatures' grammar branch (structures) remains with the structured-output work.

## Ported (Phase 2, twelfth tranche
## Ported (Phase 2, twelfth tranche — the refusals and the context window)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `errors.zig` | 62 | `crates/server/src/errors.rs` (new) | The refusal kinds and their statuses (request 400 with the type, context_length 400 with param and code, capacity 503, other 400 message-only, server 500), the context limit's wording, `Refused` as the failure currency (the Cx port), `errorBody`/`wrapError` with the param ("messages" here; "prompt" rides the completions surface when it ports). Tested. |
| The failure channel | — | `chat.rs`/`anthropic.rs` | `Reply.failed`, `generate_with` and `submit_streaming` carry the structured refusal; the routes shape the bodies from the kind — the string-matching that mapped three messages to statuses is gone. |
| `chat.zig:238-242` — the engine's own failures | 5 | both request paths | The messages upstream sends: the busy lane a capacity ("the engine is busy; retry shortly" — the port said less), the closed scheduler an other ("the scheduler is closed" — the port said "the engine closed" at a 503), the span layout a request ("the prompt span layout is invalid" — the port paraphrased). |
| `chat.zig:164-170` — the context window | 10 | both request paths | The room for the reply after the render: no room the context refusal with the fitted window's note; a named reply over the room the second refusal; the limit clamped to the room. `max_tokens_of` now returns whether the request named a limit (openai.zig:134's input.max_tokens), which the second refusal reads. Both messages as testable builders, asserted character-for-character. |
| `openai.zig:348/461` — streamedContext | 12 | both stream branches | A context refusal on a stream reaches the client inside it, upstream's way: the chat surface opens the SSE, the role chunk when no tools, the wrapped error chunk, [DONE]; the Messages surface message_start, then the typed error event, and no message_stop. |
| `anthropic.zig:74-76` — the Messages error status | — | `anthropic.rs` | The refusal's status by its kind (the capacity 503 upstream maps; the port string-matched). |

A deliberate divergence, noted: upstream opens the SSE before the prepare-time checks, so a non-context prepare refusal on a stream is dropped after the headers (openai.zig's `unsent` through the opened wire, anthropic.zig:88's `if (wire.opened) return`); the port runs the checks before the stream opens and answers the clean 400/503 the refusal names — the client sees the error instead of a truncated stream.

## Ported (Phase 2, eleventh tranche
## Ported (Phase 2, eleventh tranche — the sampling chain)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `fields.zig:56-90` — parseNumbers, toInt, toFloat, digitRun | 50 | `crates/server/src/fields.rs` | The body with its numeric fields as numbers: the integer fields (max_tokens, max_completion_tokens, seed, thinking_budget, top_k) as ints — an integral float or a numeric string counts, `pyIntText`'s digit runs and leading zeros read as Python's int() does — the rest as finite floats; min_p between 0 and 1; a negative top_k the zero it means; the one refusal message per shape; the stop validation re-run as upstream's side effect. An integer past 64 bits refuses where upstream keeps its i128 text (the same documented deviation the seed parsing carries). Tested, the upstream test's numbers half mirrored. |
| `openai.zig:120` — parseNumbers at the route | 1 | `chat.rs` `generate`/`submit_streaming` | The coerced body at both entries: everything downstream reads the copy, as upstream's plan hands every later read the coerced body. |
| `openai.zig:134-139` + `chat.zig:151` — the max_tokens read | 8 | `chat.rs` `max_tokens_of` | max_tokens first (truthy), else max_completion_tokens; 0 means the default; under 1 clamps to 1; the u32 cap — the port read max_tokens alone as an unsigned field and turned 0 into 1 instead of the default. |
| `openai.zig:139-155` — the fields object | 15 | `chat.rs` `SAMPLING_KEYS`/`fields_object` | The sampling-ish passthrough (temperature, top_p, top_k, min_p, seed, priority, draft, thinking_budget, ignore_eos, stop) with the request's thinking fields beside them; the grammar fields and tool_call_required ride it upstream with the features that read them. |
| `server.zig:135-160` — resolveSampling, merge, number, seedBits | 30 | `chat.rs` `resolve_sampling`/`seed_bits` | The server's default sampling (`--temperature`'s config) parsed and merged under the request's fields — the port never honored it: a server default temperature now samples every request; the temperature gate (0 or less greedy); the greedy-only engine's refusal, its message by the defaults' temperature; the seed by bits or keyed to the prompt; the clamps. Tested: the merge, the gate, the seed, both refusals. |

Not yet: the context-window refusals and the limit clamp (chat.zig:164-170's room math — the whole context_length refusal chain is unported), the Generation's server-side `stopHit` (noted before), `validateModalities`/`probabilityOptions` (the logprobs request checks), the grammar fields and `tool_call_required` (the call-gate path, with `Request.call`), and the request's `thinking_budget` (the think-close machinery).

## Ported (Phase 2, tenth tranche
## Ported (Phase 2, tenth tranche — the response extras and the effort chain)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `fields.zig`'s efforts + coerceEffort + effortLevels + thinkingFields | 65 | `crates/server/src/fields.rs` (new) | The fields.zig home: the seven efforts and their order, `coerce_effort` (the nearest level the template names, ties higher; xhigh and none stay), `effort_levels` (the quoted efforts in the template, by strength), and `thinking_fields` — the effort from the body or the kwargs validated and coerced, the switch from the kwargs' `enable_thinking`, else its DeepSeek `{"type": ...}` spelling, the effort's own enable, the enable clearing a none effort. The upstream test's effort half (fields.zig:308-310) mirrored, with the levels, switch and kwargs shapes. |
| `fields.zig`'s stopOptions | — | `crates/server/src/fields.rs` | Moved from chat.rs to its upstream home; the tests with it. |
| `server.zig:102/130` — effortLevels at open, effortFor | — | `crates/server/src/state.rs` `Text.effort_levels`, `chat.rs`/`count_tokens` | The template's effort levels read once at open; the effort chain — the request's effort, else the server's configured one, coerced — now reaches the RENDER (`RenderOptions.reasoning_effort` was in the core contract but the server never passed it: a request naming an effort rendered without it). |
| `chat.zig:161` — the thinking switch | — | `chat.rs` `generate`/`submit_streaming`/`count_tokens` | The switch read through `thinking_fields` (the kwargs' `enable_thinking`, the DeepSeek spelling, the effort's enable) orelse the server's config — the port read a top-level `thinking` field the upstream surface does not. The Anthropic surface's translated kwargs now take effect. |
| `tokens.zig:101 tokenSha` | 14 | `chat.rs` `token_sha` | SHA-256 of the comma-joined ids, the first six bytes as lowercase hex; sha2 joined the server's deps. Tested against the digests themselves. |
| `chat.zig:513-547` — the runtime and speculative objects | 40 | `chat.rs` `runtime_objects` | The `tensorfold` object — enable_thinking, reasoning_effort (none when thinking off), the engine's name, tokens_per_second and seconds from the prefill's end, prefill_seconds, prefill_widths, prefill_raised, time_to_first_token from the request's arrival, sampling exact/greedy, drafts, token_sha, min_rows, the loop period — and the `speculative` counters — rounds, drafted, accepted, acceptance_rate, tokens_per_round, the proposer telemetry when it parses. One builder, both paths. Tested. |
| `openai.zig:300/521` — response_extras | 8 | `chat.rs` `response_extras` | exact_mode, the matched stop, the runtime and the speculative counters: on the whole body (wholeBody's `r.extras`) and the streamed last chunk, top level both. |
| `chat.zig`'s Mailbox marks + Stats | — | `chat.rs` `MailboxInner`/`StreamEvent::Finished`/`Bridge` | The full `Stats` kept (the widths, raised, min_rows, loop period and telemetry the inner fields left out), the prefill and first/last-token instants marked; the stream's timings stamped at the bridge (its `received` through `StreamStart`). The metrics' note now carries the real drafted/accepted/rounds (it counted zeros). |
| `chat.zig:181` — the draft override | — | `chat.rs` both request builds | `draft: false` turns the server's drafts off for the request. |

Not yet: the Generation's server-side `stopHit` (noted last tranche), the sampling object's field set (`resolveSampling`'s passthrough shape — the port reads the numeric fields directly; `temperature` as an integer, `priority` and the grammar fields ride it upstream), and `parseNumbers`/`probabilityOptions`/`validateModalities` — the rest of fields.zig's request checks.

## Ported (Phase 2, ninth tranche
## Ported (Phase 2, ninth tranche — the stop strings and the Messages stream's tool blocks)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `fields.zig`'s stopOptions | 23 | `crates/server/src/chat.rs` `stop_options` | ignore_eos must be a bool; stop a string, a list of nonempty strings, or null — the one refusal message for every wrong shape. Tested. |
| `chat.zig`'s StopHook | 15 | `crates/server/src/chat.rs` `StopHook` | The engine's stop check (`Request.stop`, `StopCheck`): the emitted tail decoded fresh each check, its length the longest stop's characters plus eight; true ends the reply with the engine's own `Reason::Stop`. |
| `chat.zig:186/233` + the request builds | — | `chat.rs` `generate`/`submit_streaming` | Both request builds: the hook when strings exist, `ignore_eos` emptying the eos list. |
| `reply_text.zig:194 stripTrailing` | 8 | `crates/server/src/reply_text.rs` `strip_trailing` | The model's own end tokens off the collected reply before the decode (chat.zig:474) — the port emitted them into the text before. Tested. |
| `chat.zig:388/477` — the stops in the streamed and finished text | — | `chat.rs` `Bridge::tokens`/`Bridge::finish` | The streamed text through `Stops.visible(all, true)` (the partial stop held back); the finish re-decodes the eos-stripped tokens and takes `visible(raw, false)`. The matched stop rides the last chunk. `Reply.stop_sequence` set from the raw (chat.zig:550). |
| `chat.zig:343`'s eos check on closeCall | — | `chat.rs` `Bridge::finish` | A call the model's end token left open completes only when that last token is one of the model's eos (a stop string, the length or ignore_eos ends with no completion). |
| `openai.zig:220/302` — the ids and the stop_sequence extra | ~10 | `chat.rs`/`anthropic.rs` | The reply ids by `ids.make` (`chatcmpl-`/`msg_` + 32 hex — the port's `{pid}` shape was a divergence); `stop_sequence` on the last chunk and the whole body when matched — the part of `response_extras` the surfaces read. The extras' rest (exact_mode, the runtime object, the speculative counters) still awaits the mailbox's stats. |
| `anthropic.zig:101 Wire` + the chunking layer | — | `chat.rs` `ChunkOut` | The stream bridge now emits the OpenAI-style chunk bodies (`ChunkOut::Chunk`) and its end (`Done`); the chat route sends them as SSE data with `[DONE]`, the Messages route translates them — upstream's layering (the Run's chunks through the Wire), where the port's two bridges each re-implemented the stream before. |
| `anthropic.zig:134-328 Reply` | 195 | `crates/server/src/anthropic.rs` `Reply` | The typed-event state machine, ported whole: message_start with the base message (the id by `ids.make`), the thinking block's empty signature_delta at its close, the text/thinking blocks at their first delta, the tool_use blocks — the calls assembled from the chat deltas by index, each block opening at its id and name, the arguments as input_json_delta, completion when they close as a JSON object, a second call waiting, the text that arrives while a block is open held for its close — the error chunk, the usage and the matched stop captured, `finishOf`'s reasons (tool_use/max_tokens/end_turn) and `stop_sequence`'s override, message_delta and message_stop. Two tests pin the event sequences. |

Not yet: the extras' rest (exact_mode, the `tensorfold` runtime object with its timings and token sha, the `speculative` counters — the mailbox's widths/raised/min_rows and the first-token timing still to thread), and the Generation's own server-side `stopHit` (chat.zig:368 — the port's engine-side hook ends the reply on its own thread, which the Mailbox design reads the same way).

## Ported (Phase 2, eighth tranche — the tool bodies complete)
## Ported (Phase 2, eighth tranche — the tool bodies complete)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/tool_params.zig` | 381 | `crates/server/src/tool_params.rs` | The offered tools' parameter schemas by lowercase name; the partial-JSON closer (`closedJson`, strings and escapes tracked); the decoder (`decode`) that reads a parameter's text as the schema's type — the JSON spelling, then the closed form, then Python's own (the recursive-descent `Literal` parser: True/None keywords case-insensitive, adjacent strings joined, raw prefixes, triple quotes, every escape including octal and \u/\U, Python's int spellings with underscores and the leading-zero rule, complex rejected, floats by `pyFloat`); `finite` (json.dumps allow_nan=False); the 9 upstream tests mirrored plus the literal, closer and float coverage. |
| `tool_stream.zig`'s Streamer | 200 | `crates/server/src/tool_stream.rs` | The XML-call streamer: the opener search with its `searched` bookkeeping, the `<function=NAME>` head delta (id from `ids.make`), the `<parameter=K>` openings that emit the arguments JSON incrementally, string values escaped as they arrive (`esc`), typed values buffered whole then decoded by the schema and stringified wide (Zig's non-compact writer — `zig_stringify`), the `</function>` close and the envelope's closer; the 256-char and prose-lead aborts; the three upstream tests mirrored exactly (the open-call completion, the cut value, the one-call policy). |
| `tool_stream.zig`'s TextFilter + Policy | 130 | `crates/server/src/tool_stream.rs` | The streamed-prose filter (the close held back by characters, the partial opener held, `(?:NAME:)?tool_call` any case) and `parallel_tool_calls: false` on it: `maxCalls`, `text`, `flush`, `content`, `parsedContent` (whole envelopes kept, the rest filtered), `kept`. |
| `server/tool_parse.zig` | 501 | `crates/server/src/tool_parse.rs` | Rewritten at full fidelity: the four envelope families (Qwen, the custom `<NAME:tool_call>` tags any case, Gemma, DSML — the markers verified by hexdump, never by display), `parse` with `max_calls` (a one-call reply drops what it will not take), the full payload chain (`gemmaCall`'s fenced strings and bare keys, the JSON holder shape with the truthy name keys, `functionBlock`'s parameter blocks, GLM's name/key/value markers, `dsmlCalls`), `bareCalls` (the fenced json reply body; a structured answer stays content), `closeCall`/`argumentsClose` (what closes the call the text ends inside), `deltas` (the id, type and name when truthy), `wrapCalls`, and `zig_stringify` (Zig's JSON writer: compact or wide, floats by Python's repr, NaN and Infinity as JavaScript). 14 tests. |
| `server/tool_streaming.zig` | 46 | `crates/server/src/tool_parse.rs` | (Was already ported; the truthiness of the id, type and name now matches `tool_streaming.zig:9` exactly.) |
| `server/ids.zig` | 23 | `crates/server/src/ids.rs` | `make`: the prefix then n random lowercase hex digits, the high-nibble-first pairing kept; getrandom the entropy source, a time/process/counter mix the never-happening fallback. The call ids across tool_parse now match the upstream shape (24 hex digits, not `call_{index}`). |
| `json.zig:47 truthy` + `json_write.zig floatRepr` | ~60 | `crates/server/src/pyrepr.rs` | Python truthiness of a JSON value; `repr(float)` — the shortest round-trip digits, fixed between 1e-4 and 1e16, else the signed exponent form. Both tested. |
| `chat.zig`'s streaming loop | ~80 | `crates/server/src/chat.rs` | The bridge reworked: the `Shown` tracking (the char fallback when the visible text rebuilds), the call markup hidden from streamed prose with the tail held back, the Streamer fed per batch (never the reasoning), the finish's closing call (`closeCall`), the final prose and reasoning pieces, and the end-of-stream calls per `openai.zig:495-520` — the role before them when they were not streamed, the one-call filter's tail, the whole-content and `kept` fallbacks. The last chunk's finish reason becomes `tool_calls`. |
| `openai.zig`'s sink + policyDelta + attachCalls | ~110 | `crates/server/src/chat.rs` | `StreamSink.call`'s port: without tools the delta passes through; with tools the one-call policy filters (text through the filter, `tool_calls` keys dropped, empty dropped), the role chunk opens lazily before the first prose (`prose`), the prose remembered for `kept`. `attachCalls` (openai.zig:309) at both routes: the parse with the policy's `maxCalls`, the content through `parsedContent`, the calls truncated to one, the finish reason `tool_calls`; the chat body nulls the content when the reply calls (openai.zig:387) and carries `tool_calls` only then. `parallel_tool_calls` must be a boolean (openai.zig:164). The Messages route attaches from its translated body (anthropic_translate.zig:248's `parallel_tool_calls`). |
| `fields.zig`'s pyFloat | 35 | `crates/server/src/tool_params.rs` | `float(text)`: the nine-character strip, the inf/infinity/nan spellings, the underscore digits, the 100-char cap; `digitsInto` beside it. |

Not yet in this tranche: the Stops wiring of the streamed path (chat.zig:389's `stops.visible` and the StopHook into the engine's `Request.stop` — the port's bridge has neither, so a stop string streams as prose and the engine runs past it; the non-streaming path detects the match but does not stop generation either), and the Anthropic stream's `tool_use` blocks (anthropic.zig:292's content_block_start over the call deltas — the port's typed SSE shapes text and thinking blocks only).

## Ported (Phase 2, seventh tranche — tool-call bodies)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/tool_parse.zig` | 501 | `crates/server/src/tool_parse.rs` | The envelope finding (the Qwen and Gemma markups, verified against upstream's own byte-exact hex), the JSON payload parsing (function/name/arguments/args/parameters keys, the loose fallback), the parse flow (envelope → payload → callName against the offered tools → the OpenAI tool_calls shape, the markup stays in the content when the call is malformed or unoffered); 5 tests. |
| `server/tool_streaming.zig` | 46 | `crates/server/src/tool_parse.rs` `deltas` | Each call's opening delta (id/type/function name), then its arguments in one more. |
| The chat response's tool_calls | ~15 | `crates/server/src/chat.rs` | The generate path: after the reply, when the request offered tools, the markup becomes `tool_calls` in the message and `finish_reason: "tool_calls"`. |
| The Messages response's tool_use | ~15 | `crates/server/src/anthropic.rs` | The tool calls as `tool_use` blocks (type/id/name/input), `stop_reason: "tool_use"`. |
| `server/tool_params.zig` | 381 | — | Deferred: the parameter schema validation and structured-output strictness. |

## Ported (Phase 2, sixth tranche — the Anthropic typed SSE streaming)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `anthropic.zig`'s Reply (134-375) — the typed Messages events | ~240 | `crates/server/src/anthropic.rs` `stream_messages` + `sse_bridge` | message_start, content_block_start (text/thinking at their first delta), content_block_delta (the per-round deltas through the incremental decode + the thinking split's streaming holds), content_block_stop, message_delta (stop_reason + the Anthropic usage shape), message_stop. Tool_use blocks arrive with the tool-call bodies (the remaining Phase 2 tranche). Verified against the Zig engine: 3 streaming probes MATCH. |

## Ported (Phase 2, fifth tranche — the HF cache resolution and the metrics scrape)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/hub.zig`'s `resolve` | ~45 | `crates/cli/src/hub.rs` `resolve` | MODEL resolves to a local directory (with `~/` expansion), else a HF cache entry by repo id — the same resolution pull fills. Live-tested: `serve mlx-community/Qwen3.5-2B-MLX-4bit` loads the pulled checkpoint from the cache and answers. |
| `server/metrics.zig` | 359 | `crates/server/src/metrics.rs` | The request counters and histograms, the Prometheus text render (family/gauge/histogram with cumulative buckets +Inf/sum/count), the vLLM aliases; the /metrics route reads the engine status at scrape time. The process footprint gauges (macOS task_info) and the device-memory counters (CUDA) arrive with their backends. Live-tested: a request counts (prompt_tokens_total, generation_tokens_total, latency). |
| `server.zig`'s noteRequest call site | ~4 | `crates/server/src/chat.rs` | generate() reports the token counts and the wall latency to the metrics. |

## Ported (Phase 2, fourth tranche — the CLI's model management)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `cli/hub.zig` | 203 | `crates/cli/src/hub.rs` | The HF cache layout: cacheDir (HF_HUB_CACHE/HF_HOME/default), repoDirName, hexSha (only shas join paths), cachedSnapshot (refs/main or newest with config.json), isRepoIdLike, configInt, modelType, quantization, sizeOf (symlinks followed); upstream's tests mirrored. |
| `cli/models.zig` | 66 | `crates/cli/src/models.rs` | Lists cached checkpoints the port's registry serves; the test mirrors upstream's own. |
| `cli/info.zig` | 84 | `crates/cli/src/info.rs` | One checkpoint's family, format, context, weights and memory floor (the 2 GiB runtime margin); the test mirrors upstream's. |
| `cli/pull.zig` + `pull_file.zig` | ~430 | `crates/cli/src/pull.rs` | The pull: revision resolve, tree list, the family check on config.json before any weight moves, per-file sha256-verified download into the HF cache (blobs + snapshot symlinks + refs/main). The 16-way piecewise ranged download (pull_parts.zig) is deferred — the correctness contract (verify, then blob) is the same. Live-tested: the full 1.6 GiB pinned checkpoint pulls, `models` lists it, `info` reads it. |
| `cli/hub.zig`'s `servedName` | 8 | `crates/cli/src/hub.rs` `served_name` | Kept beside its callers (the serve path). |

## Ported (Phase 2, third tranche — real streaming)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `reply_text.zig`'s Incremental (296-319) | ~24 | `crates/server/src/reply_text.rs` | Decodes only new tokens with the previous chunk as context, so characters split across tokens stay whole. |
| `chat.zig`'s StreamSink + streaming loop (380-460) | ~80 | `crates/server/src/chat.rs` | The role chunk first, one content chunk a round through the incremental decode (with the thinking split's streaming holds), the last chunk with finish_reason + usage, then [DONE]. The Mailbox pushes the *batch* through a tokio channel; a bridge thread does the incremental decode and the delta computation. |
| `openai.zig`'s chunk (250-275) | ~25 | `crates/server/src/chat.rs` `chunk_body` | The chat.completion.chunk body: id/object/created/model/choices with the delta and finish_reason, the usage on the last chunk. |

Verified against the Zig engine: three streaming probes with identical concatenated deltas, identical finish reasons, identical usage chunks (`MATCH` on all three).

## Ported (Phase 2, second tranche — the Anthropic Messages surface)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/tool_specs.zig` | 97 | `crates/server/src/tool_specs.rs` | pyStr/nameText, choiceDisables/choiceRequiresCall, active (none/named-function rules); 3 tests. |
| `server/pyrepr.zig` | 104 | `crates/server/src/pyrepr.rs` | Python's repr of a JSON value (the refusal messages' quoting). |
| `server/anthropic_translate.zig` | 293 | `crates/server/src/anthropic_translate.rs` | Messages → chat: system text, content blocks, tool_use→tool_calls, tool_result→tool messages, thinking switches, output_config/effort/format, context_management, tool_choice mapping; 3 tests. Verified against the Zig server: 4 probes MATCH (text, system, multi-turn, blocks) + count_tokens MATCH. |
| `server/anthropic.zig` (non-streaming) | ~180 | `crates/server/src/anthropic.rs` | The `/v1/messages` route (translate → engine → content blocks + Anthropic's usage shape + stop_reason mapping) and `/v1/messages/count_tokens`. The typed SSE events (message_start/content_block_*/message_stop) are the next tranche with the streaming chunk plan. |
| `chat.zig`'s think_end rule | ~3 | `crates/server/src/chat.rs` | chat.zig:547: reasoning counts only when the request opened the block — found by the first Anthropic golden probe (thinking_tokens 16 vs 0), ported exactly. |

## Ported (Phase 2, first tranche — the chat surface's text machinery)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/messages.zig` | 221 | `crates/server/src/messages.rs` | normalize (leading system/developer merged, late-system role, the placeholder user turn for query-demanding templates) + toolArguments (string args as objects, bad under `_invalid_arguments`); upstream's 4 tests mirrored + the tool-args contract. |
| `server/reply_text.zig` (core) | ~200 | `crates/server/src/reply_text.rs` | Markers with upstream's exact bytes, splitThinking (streaming holds), hideToolCalls, pyStrip/Lstrip/Rstrip (Python `\s` semantics incl. the unicode whitespace codepoints), Stops (visible/matched/tail), reasoningCount; upstream's 2 tests mirrored. |
| `chat.zig`'s normalize + thinking split | ~30 | `crates/server/src/chat.rs` | Messages normalized before the template render (chat.zig's prepare); the think split only when the request opened the block (chat.zig:488's `g.thinking`); `reasoning_content` + `completion_tokens_details.reasoning_tokens` in the body (openai.zig:388/295). |
| `server.zig:89-104,318`'s one-time reads | ~20 | `crates/server/src/state.rs` `Text::read` | The markers by the tokenizer's knowledge of the channel close; `needs_user_after_tool` from the template source; `think_end` through the special tokens. |
| `model_text.zig`'s source probe | ~3 | `crates/core/src/template.rs` `source` | The template's raw source, for the user-query demand read. |

## Ported (Phase 1b closure — the golden gates)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `server/hf_text.zig` `eosIds` | ~25 | `crates/cli/src/serve.rs` `eos_ids` | The model config's eos_token_id plus the tokenizer config's eos_token through the special tokens (hf_text.zig:126). Found by the golden harness. |
| `server/server.zig:140-158` | ~18 | `crates/server/src/chat.rs` | The sampling rule: the request's seed (seedBits' low 64) else the prompt's reproducible seed with the salt; top_p 1.0, min_p 0, top_k 0 defaults. |
| The golden harness itself | — | `crates/cli/tests/golden.rs` | Both servers on the same machine and weights: eight probes, byte-identical. |

## Ported (Phase 1b, part 2 — the serving path)

| Upstream Zig | Lines | Rust | Notes |
|---|---|---|---|
| `src/core/cache_modes.zig` | 79 | `crates/lanes/src/cache_modes.rs` | validate/equal/prefix/reply; upstream's three tests mirrored. |
| `src/core/lane_host.zig` | 549 | `crates/lanes/src/lane_host.rs` | The engine thread: job queue foreground-first, admission with the prefill rate, cancel hooks, token delivery with the 2s decode rate, fail-all/close-all; ADR-016's ownership shape. |
| `src/core/template/*` (14 modules) | ~4.6K | `crates/core/src/template.rs` | minijinja (ADR-015) + the Python string-method shim; the pinned 153-line template renders (tested). `render_ids` composes the tokenizer. |
| `server/chat.zig`'s core (Mailbox, submit, reply body) | ~200 | `crates/server/src/chat.rs` | The Mailbox sink, prompt render with the thinking switch, submit-and-wait, the OpenAI completion body and SSE chunks. Tools/reasoning/logprobs bodies: Phase 2. |
| `server/serve.zig`'s engine wiring + startup | ~80 | `crates/cli/src/serve.rs` | Model load, Metal backend with `--parallel`'s stream bound, LaneHost, `AppState::with_engine`, the startup line at the listener's `on_listen` (serve.zig:144's shape). |
| The serve gate itself | — | `crates/cli/tests/serve_e2e.rs` | The release binary serving the pinned checkpoint: greedy text, streamed chunks, model discovery — the port's end-to-end exactness pin. |

## Planned

| Upstream Zig | Lines | Phase | Where it lands |
|---|---|---|---|
| `core/serial_host.zig` + its tests | ~380 | 1b-2 | `tensorfold-lanes` (the serial reference host) |
| `core/prompt_cache.zig`, `prompt_imprint.zig` | ~600 | 2 | `tensorfold-lanes` |
| `server/chat.zig`'s tools/reasoning/logprobs bodies, `messages.zig`, `reply_text.zig`, `tool_*.zig` | ~2K | 2 | `tensorfold-server` |
| The lane-host reuse tests (`lane_host_reuse_test.zig`) | ~250 | 2 | `crates/lanes/tests` (with prompt_cache) |
| `core/lane_host.zig`, `core/{cache_modes,prompt_cache}.zig` | ~2.5K | 1b | `tensorfold-lanes` |
| `src/server/chat.zig`, `model_text.zig`, templates | ~8K | 1b-2 | `tensorfold-server` |
| `src/families/nemotron/`, `glm/`, `flashnext/`, `qwen27/`, `kimi_k3/` | ~38.5K | 2 | family crates (Metal first, then their `cuda_*` halves) |
| `src/cuda/`, `src/hip/` | ~4K | 2 | `tensorfold-cuda` (the DynLib loading maps to `cudarc`-style dynamic loading, mirroring `cuda_source.zig`'s) |
| `cli/pull_parts.zig`'s ranged 16-way piecewise download | 182 | 2 | `crates/cli/src/pull.rs` (the single-stream verified download is the correctness baseline) |
| `src/server/compact*.zig`, `anthropic*.zig`, responses, tools | ~10K | 2 | `tensorfold-server` |
| `src/core/prompt_cache.zig`, `prompt_imprint.zig`, keepalive | ~4K | 2-3 | `tensorfold-engine`/server |
| Sliding/Living Weights (`families/*/train*.zig`, `slide*.zig`) | ~8K | 3 | family crates behind the already-ported learner contract |
| `src/fabric/`, `src/cluster/` (two-Mac serving, MCDMA) | ~12.3K | 3 | `tensorfold-fabric` (unsafe shared-memory windows, documented) |
| `src/metal/ane.zig`, `icb.zig`, `residency.zig`, `sync.zig` | ~700 | 1b-3 | `tensorfold-metal`, with the engines that use them |
| `build.zig` remaining kernel sets + codegen tools (`tools/zig/gen_*.py`) | build graph | 1b-2 | `build.rs` / a codegen step (ADR-005) |

## Not ported on purpose

| Upstream Zig | Why |
|---|---|
| `core/tokenizer/{model,steps,regex,unicode}.zig` (~7K) | The `tokenizers` crate is the reference; Phase 1b's gate decides (ADR-003). |
| Hand-rolled HTTP, JSON writer, listener sockets | axum/tokio/serde (ADRs-002/004); the exactness-sensitive parts of JSON (ids in replies) port with the translation layers in Phase 1b-2. |
| `zig/tests/*` harnesses (19K) | Replaced by the golden harness design in the plan; the pinning tests were mirrored case by case where they exist (see safetensors, hub, sampling, config, backend, qualify tests). |

## Known open gaps

| The gap | What differs | The plan |
|---|---|---|
| The Wyhash cross-check | `key_of` runs the `wyhash-final4` reference — the algorithm and secret Zig's `std.hash.Wyhash` is built on — so the learned trees stay byte-compatible; the definitive check (one Zig program hashing the same token slices beside the Rust one) waits for a Zig toolchain on this Mac (the /tmp purge took the last, and the pinned binary predates the loss). No live surface pins the keys today: the trees are the family-that-arms-a-store's own. | A five-line zig test beside the imprint tests at the toolchain's return; the end-to-end check rides the family that arms a store. |
| The request parser's number edges | ``-0`` now canonicalizes to the int ``0`` (the tranche-thirty parser); the remaining edge: upstream holds ints at any width (Python's) and the NaN/Infinity literals as floats, the port's json_parse falls past-width ints to f64 and reads the non-finite literals as null — a request echoing its own huge int back (the Responses echo, metadata) writes the float's text where upstream writes the int's. | The corpus freezes none of it; the true fix is the arbitrary-precision Value (a named refactor). |
| The Messages route's malformed-body replies | The Messages route reads its body with axum's `Json` extractor, whose rejection wording and bytes are axum's own; every other route reads raw (the responses route its own "not JSON" wording, the chat and completions routes Python's decoder messages, byte-verified). With a well-formed body every route agrees. | The Messages route moves to the raw read when its corpus cases pin the wording. |
| The streamed paths' warm prefill | `openai.zig`'s warmNext fires on both the whole and the streamed path; the port fires it on the whole paths only. Inert for the pinned family (no `warm_turns`), so no behavior differs today. | Ride it in the bridge when a family sets `warm_turns`. |
| The engineless refusal's order | Upstream refuses engine-less inside the plan, after the body's own translation checks; the port's routes check the engine first (the chat route's Phase 0 convention). With an engine — every golden, corpus and E2E run — the orders agree. | Keep the convention; a none-backend build's parity is the only divergent case. |
| Axum's routing edges | DELETE on a non-responses path: upstream routes every DELETE through the responses store (the null id's `None` 404); the port registers the responses paths only. Unknown paths answer upstream's `unknown path {p}` exactly (the tranche-twenty-six fallback); a method mismatch on a registered path still gets axum's bare 405 where upstream writes the unknown-path body. | The dispatcher's catch-alls are a routing-layer tranche if the corpus grows DELETE cases beyond the responses paths. |
