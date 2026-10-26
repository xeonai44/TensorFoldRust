# Porting plan

Estimates assume a senior engineer or two, porting upstream file by file with the
rules in AGENTS.md. "Done" means the exit criteria pass, not that code compiles.

## Phase 0 — the skeleton (COMPLETE)

**Scope:** workspace, the engine contract, checkpoint/tokenizer/config readers, the
Metal build pipeline and host layer, the HTTP server skeleton, the CLI.

**Exit criteria (all met):**
- [x] `cargo build`, `cargo test`, `cargo clippy --workspace` green on macOS with
      Xcode's Metal toolchain.
- [x] An upstream kernel (`runtime_tests.metal`'s `tf_vadd`) compiles through the
      ported pipeline and runs on the GPU with correct results.
- [x] The safetensors parser accepts and refuses the exact cases upstream's
      `safetensors.zig` tests pin (F8 spellings, rank ceiling, shape and offset
      refusals).
- [x] A checkpoint opens through `model.safetensors.index.json`, the plain
      `model*.safetensors` glob, and refuses missing and leftover shards.
- [x] `capabilities --json` prints upstream's document shape; `serve MODEL` walks the
      config gate, the tokenizer load, and refuses with none.zig's message.
- [x] `kernels/metal/` verified byte-identical to upstream.
- [x] Sampling seed bits (`sample_seed_for`, `mix`, `key_bits`) ported bit-exactly.

## Phase 1a — the Qwen3.5-2B family's Metal engine (COMPLETE)

**Scope:** the whole `families/qwen3_5` model layer — config admission, Metal-resident
checkpoint, weight views, the runtime-compiled kernel registry, state, forward
passes, the keyed draw, and the qualify exactness check — plus the metal-crate
ground it needs (runtime `Library::from_source` with MLX's options, the compute
encoder, `checkpoint_metal`) and the exact sampler (`uniform`, `pairwise_sum`,
`choose`) in the engine crate.

**Exit criteria (all met, on the pinned checkpoint
`mlx-community/Qwen3.5-2B-MLX-4bit` @ 93760be4, hash-verified on download):**
- [x] `Model::load` reads the real checkpoint end to end: the admission gate passes
      on the pinned config.json, 1.6 GiB goes resident in Metal buffers, and all 20
      kernels (15 generated + 5 layout) compile from source at runtime with MLX's
      compile options.
- [x] The qualify check passes on the real weights on the GPU
      (`cargo test -p tensorfold-qwen35 --test qualify`): wider lanes' rows,
      committed caches and keep rolls equal one-row execution — upstream's own
      admission test (qualify.zig), mirrored.
- [x] The config gate's refusal table matches upstream's fixture tests case by
      case; the draw's candidate selection matches the shared sampler over the
      full 248320-word vocabulary, for every top-k cut and position upstream pins.
- [x] The sampler (`choose`) runs on libm's `exp`/`log` with numpy's pairwise-sum
      order, its tests pinning the values the Zig tests pin.

## Phase 1b — the lane core and the serving path (COMPLETE, golden gates green)

**Part 1 (complete):** the whole `core/lanes` round loop, ported as
`tensorfold-lanes`, and the qwen35 backend's vtable half wired to it:

| Upstream Zig | Rust |
|---|---|
| `table.zig` (46) | `lanes/src/table.rs` |
| `events.zig` (161) | `lanes/src/events.rs` — the recorder's JSON lines, keys sorted, floats as bits |
| `accept.zig` (99) | `lanes/src/accept.rs` |
| `allocate.zig` (124) | `lanes/src/allocate.rs` |
| `shape.zig` (139) | `lanes/src/shape.rs` |
| `fill.zig` (140) | `lanes/src/fill.rs` |
| `plan_lanes.zig` (173) | `lanes/src/plan_lanes.rs` |
| `proposer.zig` (205) | `lanes/src/proposer.rs` |
| `depth.zig` (271) | `lanes/src/depth.rs` |
| `config.zig` (138) | `lanes/src/config.rs` |
| `stream.zig` (393) | `lanes/src/stream.rs` |
| `backend.zig` (180) | `lanes/src/backend.rs` — the vtable as a trait, the optional slots as defaults |
| `windows.zig` (238) | `lanes/src/windows.rs` |
| `engine.zig` (599) | `lanes/src/engine.rs` |
| `fake.zig` (272) | `lanes/src/fake.rs` |
| `engine_test.zig` (220) | `lanes/tests/engine.rs` |
| `families/qwen3_5/backend.zig` (219) | `qwen35/src/backend.rs` — prefill/first/queue/read/verify/keep/draft/release + the Phase 1a draw |

**Part 1 exit criteria (all met):**
- [x] The engine tests pass, every case mirrored from engine_test.zig: drafted
      rounds == one-token rounds == the serial target (greedy and sampled);
      shared rounds == solo; the thinking budget's forced close identical drafted,
      plain and shared; logprob rows following their tokens on every path; the
      loop guard matching drafted and plain, reporting its period, latching once,
      and answering after close.
- [x] The per-module tests mirror upstream's own (table, events' sorted-key JSON
      line pinned byte for byte, accept, allocate, shape's odds and trees,
      plan_lanes' pricing, proposer's suffix copy, depth's tokens-a-ms, config's
      widths, stream's commit/eos/length/loop-guard cases).
- [x] The real-model gate (`cargo test -p tensorfold-qwen35 --test lanes`): the
      lane core driving the qwen35 Metal backend on the pinned checkpoint — two
      streams through shared rounds commit the same tokens as each commits alone,
      and both equal the solo run.

**Part 2 (COMPLETE):** the host and the serving surface, end to end:

| Upstream Zig | Rust |
|---|---|
| `core/cache_modes.zig` (79) | `lanes/src/cache_modes.rs` — span validation, prefix/reply; all three tests mirrored |
| `core/lane_host.zig` (549) | `lanes/src/lane_host.rs` — the engine thread: job queue (foreground first), admission, cancel hooks, delivery, decode-rate status, fail-all and close-all (ADR-016's ownership) |
| `core/template/*` (~4.6K) | `core/src/template.rs` — minijinja + the Python string-method shim (ADR-015); the pinned 153-line template renders |
| `server/model_text.zig` + `prompt.zig`'s render | `core/src/template.rs`'s `render_ids` — the template and tokenizer composed |
| `server/chat.zig`'s core flow (Mailbox, submit, reply) | `server/src/chat.rs` — the Mailbox sink, prompt render, submit, the OpenAI body; SSE chunks |
| `serve.zig`'s engine wiring | `cli/src/serve.rs` — model load, Metal backend, lane host, `AppState::with_engine`, the startup line at `on_listen` |

**Part 2 exit criteria (all met):**
- [x] `serve` on the pinned checkpoint answers `/v1/chat/completions`, greedy and
      sampled, over HTTP and streaming SSE (`cargo test --release -p
      tensorfold-native --test serve_e2e` pins the gate: the release binary, the
      real model, a greedy reply with text and usage counts, streamed chunks
      ending in `[DONE]`, and model discovery with the window extensions).
- [x] The lane host's exactness holds on the real model: the E2E gate runs
      through the same lane core whose shared-vs-solo and drafted-vs-plain
      equality the part-1 gates proved.
- [x] The golden harness vs the upstream Zig binary
      (`crates/cli/tests/golden.rs`): the Zig engine (built from the upstream
      clone with the pinned Zig 0.17.0) and the Rust port both serve the pinned
      checkpoint; eight probes — greedy, sampled (derived and explicit seeds) and
      multi-turn — assert byte-identical replies, identical completion counts and
      identical prompt counts. All eight MATCH.
- [x] The tokenizer gate (ADR-003), folded into the golden harness: every probe's
      `prompt_tokens` matches, proving the minijinja+shim render and the
      `tokenizers` crate produce the same token sequence as upstream's
      hand-rolled Jinja and tokenizer on the pinned model.
- [x] The eos rule (hf_text.zig:126 `eosIds`) and the sampling rule
      (server.zig:140-158, `seedBits`) — both found by the golden harness's
      first DIFFER and ported exactly: the model config's eos_token_id plus the
      tokenizer config's eos_token through the special tokens; the request's seed
      as its low 64 bits, else the prompt's reproducible seed with the salt.

**Phase 1b is closed.** Phase 2 is underway — its first tranche (the chat
surface's text machinery: messages normalization, the reasoning split, stop
strings; `messages.zig` and `reply_text.zig` ported with upstream's tests
mirrored, wired into the chat route, and the golden harness still green
against the Zig engine after the change). The remaining Phase 2 items are
scoped in `docs/ZIG-TO-RUST-MAP.md` — the Anthropic Messages surface
(translate + route + count_tokens) and real streaming (per-token deltas
through the incremental decode, the thinking split's streaming holds, the
finish/usage chunk) are now ported too, both verified against the Zig engine
on live probes. Remaining: the other families, and CUDA —
the Triton, aot and carveout layers are
ported: the ahead-of-time Triton kernels
(their metadata, the >48-KiB shared
opt-in, the cluster geometry, the scratch
arithmetic), the captured sets — aot.json
parsed, the specialization rules tested
over every shape (the divisibility, the
folded 1, the type names, the bits) —
and the GB10 display carveout: the
first-fit span placement with upstream's
own tests green on both platforms, the
DRM shapes Linux-only and compiled on
the box (a visibility bug caught live:
the extern fns private, the glob import
empty).
the module, launch and graph layers are
ported over the same driver table: the
cubin/PTX loading with the JIT error
log, the kernel launches with the
C-aligned argument packing (upstream's
own alignment test green on both
platforms), and the whole graph surface
— capture, node-by-node graphs, the
executables with their in-place updates.
The stub library's error paths remain
the box's live gate until real GPU
hardware.
the CUDA driver layer is ported and
live-verified: the ABI table resolved
against the box's stub library (all
entry points, `cuInit`'s real refusal
code through the mapped error), the
loader, the contexts, the streams, the
device and pinned memory with upstream's
three tests whole — and all 18 upstream
kernel sources compiled to PTX on CUDA
13.3, the build's own
`--expt-relaxed-constexpr` flag caught
live (gdn.cu refuses without it, the
build file passes exactly it).
the ground for CUDA is laid: the port
builds and fully tests on Linux now (the
Ubuntu box, 20 suites / 247 green), the
same compile-time family selection
upstream's native registry does (Metal on
macOS, the Linux build refusing a Metal
checkpoint with upstream's own message,
verified live), the Linux statvfs and RAM
shapes ported, and a live-caught O-flag
bug fixed against the box's own headers
(the mis-remembered constant was O_DIRECT,
breaking small reads with EINVAL).
The CUDA toolchain there (13.3, with the
libcuda stub) is ready for the
`tensorfold-cuda` crate — compile-verified,
execution pending real GPU hardware.
The single-stream download is ported at
full fidelity: the `.incomplete` range
resume with the on-disk prefix feeding the
same digest, the rangeless restart, the
git blob sha1 for the small files (the
port had checked nothing for them), and
the snapshot link made upstream's own —
the relative `../../blobs` symlink and
the traversal refusal (a security catch).
All tested over a local HTTP server with
the requests counted.
The 16-way piecewise download is ported:
the pieces over one shared queue, a
piece's marker letting a restart fetch
only the rest, the resume within a piece,
the 206 verified against the ask, and the
caller's whole `fetchBig` — each file's
sha256 before it becomes a blob, the
once-more refetch, the rangeless fallback
— the single-stream path gaining the
retry shape besides. All tested over a
local range server, its requests counted.
The key gate is ported whole: the digest
store (SHA-256, timing-safe, the labels),
the three sources (the flags, the
environment always merged, the restricted
key file — chmod 600, named and bare lines,
live), the SIGHUP and the once-a-second
stat reread with a refused reload closing
the file's keys. The dispatch-level gate
replaces the port's per-handler checks:
OPTIONS passes, the gated path set exact,
the 401 the corpus's own body (the Messages
paths Anthropic's shape), and every reply
counted under its key's label — the
requests_total family never wired before.
The client-disconnect cancellation is
ported: a streaming client that leaves
mid-reply cancels the engine's request —
the request ids made unique first (the
port had hardcoded id 1 for every submit,
concurrent streams colliding in the cancel
queue, a latent bug the tranche caught) —
the engine's own end waited for, the
disconnection counted when the engine
reports the cancellation. Wired on all
four streaming surfaces, probed live by
the E2E's cancel case (the corpus's own).
The parse errors are ported: the request
bodies read through `json.loads`'s own
decoder — Python's scanner wordings (the
line/column/char format, the codec errors,
the trailing commas, the digit limit), `-0`
canonicalized to `0` — and the corpus's
pinned messages are byte-verified against
the running Zig server, five bodies side
by side. The chat and completions routes
read raw (the extractor gap closed); a
non-object body answers Python's
AttributeError, a missing messages the
normalize's own non-empty wording — two
word-for-word catches. The dispatch tail
routes the odd paths (an empty route and
`/health` the health route, any route
ending in `/models` the models one), and
`?reset_peak=1` resets the engine's peak.
The legacy completions surface is ported: the
prompt as the model reads it (ids as given with
the vocabulary's range refusal, anything else
as text — the specials added, the nested
joined), messages riding the template, thinking
off for a prompt, the `text_completion` reply
and its streamed chunks in the completion's own
shape — and the usage a chunk of its own on
both the completion and the chat stream (a live
gap the corpus pinned). The golden harness
compares both servers' completion text
byte-equal; the E2E probes the corpus's cases
live. The last engineless stub is gone.
The learn flow is ported whole: the Store's
learned slots and the single-rank helpers,
`keep` learning its shared cuts, the lookup
recalling a learned state from disk (a fresh
store on a new server resuming it, the pass
equal to a fresh one), the journal's crash
recovery, the victim plan — the fault suite's
single-rank cases and the two end-to-end
upstream tests green. The `peer_*` pair (the
two-Mac fabric's rank 1) is the named seam,
with the fabric tranche.
The learned-disk machinery is ported: the
identity directories, the disk admission (the
floor, the reservation, the bounded backoff),
the victim planner, and the Imprint itself —
the append-only index, the crash journal, the
LRU across identities, the sweep — upstream's
own tests ported whole, the keys hashed
through the wyhash-final4 reference (the
algorithm and secret of Zig's own Wyhash), the
trees byte-compatible. The Store-side learn
flow — the write slots, the recall path, the
fault suite's eight cases — is the named next
tranche.
The live-parity tranche: both servers on the
pinned checkpoint, every observable surface
diffed. It caught the biggest port bug yet —
the tranche-21 prompt cache was wired onto the
pinned family citing upstream's *nemotron*
path, but upstream's qwen3_5 host arms no
store at all: its reruns report no cached
tokens, its /v1/memory carries no plan. The
disarm matches it (the machinery stays for the
family that arms one); the E2E now asserts the
upstream behavior. With it: upstream's own
serve lines on stdout (the chunk-plan line,
the serving line's format, the sampling from
generation_config under the flags), the
unknown-path refusal everywhere, the metrics
render in upstream's order with the kv_cache
families (byte-identical now, 202 lines), the
family's own 128-token prefill step, and the
drafts flag as upstream leaves it. All
verified live against the running Zig server.
The JSON writer is ported: every reply now carries
`json.dumps`'s own bytes — the spacing, the
escapes, the float repr — through a `PythonJson`
responder at every route and the SSE payloads
through the same writer. The golden harness grew
the byte-parity gate: both servers under one name,
the static routes byte for byte, a frozen corpus
body the reference. Porting it caught two live
port bugs the field-wise golden could never see:
the serve ran under the checkpoint's raw window
whatever `--context` said (upstream serves under
the flag, refuses one out of range), and the
models route stamped `created: 0` where upstream
reads the clock per request.
The Responses API is ported whole: the translation
(items to chat messages, tools, the choice, the format,
the echo object), the store (GET, DELETE,
previous_response_id chains, the eviction bounds), and
the Reply machine — the typed `response.*` events
streamed or assembled whole, every refusal the frozen
corpus's own words. The E2E gate probes it live: the
whole object, the stored chain, the streamed events.
Porting it caught a port-wide wire bug: the chat
bridge's prose chunks carried `"delta": "text"` where
upstream wraps `{"content": text}` (openai.zig:267) —
the whole-reply golden couldn't see the shape, and
both translated surfaces (Messages, Responses) starved
of streamed text. The wrap restored, a test locks the
shape, and the E2E now probes a Messages stream live
for the first time. The chat route's `model` also
gains upstream's replyModel rule (the request's own
name when it names a served id).
The tokenize routes are ported: /tokenize and /detokenize (both
under /v1 too) — the ids the chat route would run (its own
render on messages, the plain encode on a prompt, the
add-specials flag honored), and the ids back to text; the E2E
gate probes both live. Two fidelity finds came with them: the
add-specials encode (upstream hand-rolls the post-processor's
prefix/suffix; the port runs the crate's own processor, the
reference behavior upstream imitates — ADR-003), and the
shared decode's keep-specials (upstream's one decode keeps
flagged specials as their own text; the port dropped them —
invisible on the pinned family, whose think markers are
unspecial added tokens and whose eos strips before decode, but
a raw special id on /detokenize came back empty).
The remaining server surfaces are ported: the /health
upgrade (the status line's counts), the /dashboard page and its
/stats poll, the /v1/memory sizing snapshot (the process
footprint, the device counters, the cache plan), the request
log with image URLs redacted, the modalities refusal, the
next-turn warm prefill (gated on the family's warm_turns),
and the /v1/decisions route whole — the checks, the wordings,
the label tokens, the engine scoring refusal on the pinned
Metal engine. The E2E gate probes the status routes live.
The retained-prefix store is live: the family-side snapshots
(this family's delta and KV state, the restore into the fresh
cache before the first chunk, the mark hooks through the pass),
the lane host's whole cache flow (the lookup at admission, the
keep through the re-entrant hook, the restore's report, the
budget plan from the machine's memory — the 70% cap, the
footprint, the margin), and the serve flags. The E2E gate
probes it live: a long conversation's rerun resumes 5,358
cached tokens and answers identically — on by default, as
upstream is. A live-caught deadlock in the admission (the job's
mutex held through the pass while the hook re-entered it) is
fixed by the Copy-facts restructure.
The prompt-caching request fields are wired: the history a
follow-up shares with its conversation, where the latest user
message starts (a caching engine), the reusable system prefix
with its shared-prefix cuts, the earlier replies' spans the
engine keys cached decode states by, the offered tools as JSON,
and the prefill chunk starts — the message markers read from
probe conversations at open when the engine prefill-steps, a
harness cut joined so a fresh session resumes its whole harness.
The rendered prompt closes a dangling think opener exactly as
upstream does — a distinction the golden harness enforced byte
for byte when the port first compared against the wrong
constant.
Compaction is ported end to end: the thresholds
(the auto reserve, the fraction's, the keep budget, the note
budget), the message math (the marker, the kept tail's cuts,
the mid-turn split, the transcript lines, the note's own
sections), the note generated through the model in chunks
with the measured retry, the rebuild with its drop-the-split
then cut-further-back loop, the stored note by conversation
key with its covered-prefix hash, the late-system template
probe for the note's role, the routes through the compact-
aware run on both surfaces, and the serve flags. The E2E gate
probes it live: a twenty-turn conversation rebuilt as a note
and a kept tail, the store written, the reply answered.
Structured output is wired: the grammar fields ride the fields
object, the spec chain (response_format's three names with the
schema shapes and their refusal messages, the guided fields,
structured_outputs' sorted keys) validated at both request
entries, a grammar beside a required call refused, and the
Structure onto the request when an engine reports structures —
which nothing in the upstream tree does, so the refusal is the
live behavior, probed end to end.
The thinking budget is wired end to end: the server reads the
close tokens once at open (the newline, the think-close token,
the two that follow), the request carries the budget — the
request's own, a truthy non-int none, an unset one the server's
default — with the close forced at it through the engine's own
budget machinery, and the loop guard now rides the same tokens
where the server enables it. The Anthropic surface's
budget_tokens flows through the same field; the E2E gate probes
a budgeted request live.
logprobs are ported end to end: the request gates (n, the
boolean, the top count, the exact-bytes requirement, the
nonstreamed-without-tools exclusions), the engine's rows collected
a token, and the response's logprobs.content — each row's token,
logprob, its exact bytes and the top ids it asked for — with the
admissions the thinking template and the reply's shape demand. The
pinned model refuses them exactly as upstream's Metal does (the
CUDA lanes are the backend that produces rows), asserted live.
tool_choice now flows end to end: the offered tools reach the
render (the core contract had them; the server never passed them),
the choice filters them everywhere upstream's input.tools goes —
the render, the call parse (a "none" reply's call markup stays
text), the required-call flag — and the call gate's request path is
wired: the template's tool-call form read from a rendered probe,
the required choice's two refusals exactly as upstream words them,
the gate onto Request.call. As upstream, no engine reports
call_gates yet, so the gate stays unset in practice.
The refusals are now upstream's shape end to end: a
structured refusal (the errors.zig kinds — request, context_length,
capacity, other, server) carries through every path, so the busy lane
refuses at 503 with its exact message and the context window enforces
both its refusals — no room for a reply, or a reply that overflows it —
with the wording clients match to compact, the reply limit clamped to
the room. A context refusal on a stream arrives inside it, the OpenAI
way: the stream opens, the error chunk, [DONE]; the Messages way:
message_start, the typed error event, and no message_stop.
The sampling chain is now upstream's: the numeric fields
coerced at the entry (strings and integral floats included, the
refusals with the one message each), the fields object passthrough,
and resolveSampling — the server's default sampling finally merging
under the request's (a serve-time temperature now samples), the
greedy-only refusal, the seed by bits or keyed to the prompt, the
max_tokens fallback order (max_tokens else max_completion_tokens, 0 the
default).
The response extras are now whole: the `tensorfold` runtime
object (the thinking and effort it ran under, the decode timings, the
token sha) and the `speculative` counters ride the whole body and the
streamed last chunk, exact_mode beside them. The effort chain that
feeds the runtime — the template's effort levels, the coerced effort,
the thinking switch through the kwargs and its DeepSeek spelling — now
reaches the render (the core contract had `reasoning_effort`; the
server never passed it), and the request's draft override turns the
server's drafts off. fields.rs is the fields.zig home: stopOptions
moved there with its tests, the thinking and effort chain beside them.
The stop strings now wire end to end: the request's options validated
(stop a string or a list, ignore_eos), the StopHook ends the reply at
the engine, the streamed and finished texts strip what they hit, the
eos strip before the decode, the matched stop on the reply and the last
chunk. The Messages stream now carries its tool_use blocks: the chat
bridge emits the OpenAI-style chunks and the anthropic Reply machine
(upstream's whole state machine, tool_use blocks with input_json_delta
included) translates them — the two surfaces no longer re-implement the
stream each its own way.
The tool-call bodies are now complete: the parse at full fidelity (all four
envelope families, the payload chain through JSON, Qwen XML, GLM, Gemma 4
and DSML, the bare-JSON reply, closeCall), the typed parameter decoding
(tool_params: the schema-driven decode with Python's literal spellings),
the in-flight XML-call streamer, the one-call policy
(parallel_tool_calls: false) with its prose filter, and the wiring —
attachCalls at both routes (the chat body nulls the content when the reply
calls, tool_calls streamed or at the end) and the streamed prose with the
markup hidden. The CLI's model management (`pull`, `models`, `info`),
the HF cache resolution in `serve`, the `/metrics` scrape, and the
Anthropic typed SSE streaming (message_start → content_block_* →
message_stop, verified byte-identical against the Zig engine on three
live probes) are now ported.

## Phase 2 — breadth: families, CUDA, the full server

**Scope:** the rest of the qualified families (Nemotron 3.5 Lightning, GLM-5.3-Flash,
Flash Next, Qwen3.8-27B + DFlash2 drafter); the CUDA backend (`src/cuda/`, `families/
*/cuda_*.zig`); hub management (`models`, `info`, `pull` with sha256 verification,
16-way piecewise download — `cli/hub.zig`, `cli/pull*.zig`); the chat/Anthropic
translation layers (`server/anthropic*.zig`, `server/chat.zig`); context compaction
(`server/compact*.zig`); prompt caching (`core/prompt_cache.zig`,
`core/prompt_imprint.zig`); metrics and memory diagnostics routes.

**Exit criteria:** each family passes its own golden gate before the next starts;
the pull path refuses (never guesses) any model no build serves, upstream's rule.

**Estimate:** 3-6 months after Phase 1.

## Phase 3 — the hard corners

**Scope:** Sliding/Living Weights (`core/engine_api.zig`'s learner contract is already
ported; the family learners — `families/nemotron/train*.zig`, `slide*.zig` — are
not); the two-Mac fabric (`fabric/`, `cluster/`, MCDMA, the GLM two-Ultra setup);
the ANE experiments (`metal/ane.zig`); speculative-decoding depth work.

**Estimate:** 2-3 months after Phase 2.

## Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Tokenizer drift (`tokenizers` crate vs the hand-rolled Zig engine) | A single token difference breaks exact decoding | ADR-003's Phase 1b gate: compare on the pinned tokenizer.json before any generation lands; fallback is porting the Zig tokenizer (the contract is pinned by tests) |
| Numerics drift | The product's promise is bit-exact replies | Kernels port verbatim and unchanged; the real-weight qualify gate already runs on every test; golden gates run against the Zig binary on the same machine |
| comptime registry semantics | Upstream's kernel selection is compile-time keyed | Realized in Phase 1a as the frozen registry (ADR-005); upstream regenerating kernels re-freezes the table, tracked in the map |
| ObjC ownership (autoreleased vs retained) | An over-release crashes; an under-release leaks GPU-adjacent memory | ADR-012: mirror upstream's `objc.zig` ownership exactly (alloc/init paths), proven by the qualify test's clean teardown |
| Zig std semantics (new `std.Io` reader/writer) | No direct Rust equivalent | Conceptual mapping only — std::io::Read/Write — done case by case at port time, never bulk-translated |
| Volume (150K lines total) | Throughput, not difficulty | The phases only pull in what their exit criteria need; crates substitution already removed ~25K lines of infra |

## Upstream sync

Upstream moves (1,080 commits at port time). Before each phase's exit gate, diff
`/Volumes/llm/tensorfold` against the revision the port cites and port any changes in
the files the phase touched. Kernels sync by `diff -rq` (they must stay
byte-identical); Rust code syncs by re-reading the cited declarations.
