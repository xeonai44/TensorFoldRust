# Decision records

Numbered in the order they were made. Each cites the upstream source it decides about.

## ADR-001: Mirror the upstream layout; a crate a subsystem

**Context:** The Zig engine is 122K lines in clear subsystems (`src/core`, `src/server`,
`src/metal`, `src/families/...`). A port could re-idiomatize the layout or keep it.

**Decision:** The workspace mirrors upstream: module boundaries, module names, and
even file names follow the Zig tree (`engine_api.zig` → `crates/engine/src/engine.rs`),
and every declaration cites `file.zig:line`.

**Consequences:** Cross-referencing during the port is mechanical; upstream changes can
be diffed onto the port; new engineers can navigate by the map in
`docs/ZIG-TO-RUST-MAP.md`. The cost is that the layout serves the port, not Rust
taste — acceptable for a project whose correctness budget is consumed by exactness,
not aesthetics.

## ADR-002: Hybrid fidelity — kernels verbatim, infrastructure by crates

**Context:** Upstream hand-rolls everything: an HTTP server on raw POSIX sockets, a
JSON writer, a tokenizer with its own Oniguruma-compatible regex engine, a safetensors
parser. ~25K lines of that is infrastructure the Rust ecosystem already ships as the
reference implementation.

**Decision:** Port the engine-specific code faithfully; substitute established crates
where the crate **is** the reference or the format owner: `tokenizers` (ADR-003),
`axum`/`tokio` (ADR-004), `serde_json`, `clap`, `sha2`, `memmap2`. Keep the
safetensors parser hand-ported (ADR-007). Copy the GPU kernels byte-identical
(`kernels/metal/`), never edit them, and compile them through the same pipeline.

**Consequences:** Roughly 25K lines of upstream scope disappear; the remaining port is
engine logic. Each substitution carries an exactness gate (see the risk register in
`docs/PORTING-PLAN.md`) — if a substitute drifts from the Zig behaviour, the fallback
is porting the Zig code, and the contract stays pinned by tests either way.

## ADR-003: Tokenizer: the `tokenizers` crate, gated

**Context:** Upstream's tokenizer (`core/tokenizer/`) reimplements Hugging Face's
`tokenizers` because it must load with no Python — its own doc comment says "encode
equals tokenizers' encode(text, add_special_tokens=False)". The Rust port has the
reference implementation itself available.

**Decision:** Use the `tokenizers` crate (the very library whose behaviour defines the
contract), wrapped in `crates/core/src/tokenizer.rs` with the load semantics from
`tokenizer.zig:210` ("`path/tokenizer.json`, or `path` itself when it names a .json
file"). Gate: Phase 1 compares its encodes against the Zig engine on every qualified
model's `tokenizer.json` before any inference lands. If any drift appears, port the
Zig tokenizer (including its regex engine) instead — the fallback is well-scoped
because the wrapper, not callers, changes.

**Consequences:** ~7K lines of upstream scope dropped provisionally; the exactness
burden moves to a gate the porting plan enforces. Known wrinkle already hit: the
crate's deserializer is strict about tokenizer.json shapes (it rejected a hand-written
fixture that was invalid as a *real* tokenizer.json — which is the crate doing its
job; see the tokenizer tests' fixture notes).

## ADR-004: HTTP: axum/tokio, engines stay off the async runtime

**Context:** Upstream: hand-rolled HTTP, one thread a connection (server.zig:217), no
async anywhere. A port could hand-roll threads the same way.

**Decision:** Use `axum` + `tokio` for the HTTP surface; keep every GPU operation off
the async runtime. Engines run their own thread (upstream's `lane_host.zig` shape) and
the async layer bridges through the `EventSink` contract. The bridge is Phase 1 work,
built before any inference code exists, so no engine code ever grows inside tokio.

**Consequences:** SSE streaming, graceful shutdown, and connection handling come for
free; the contract's "sink must return at once" rule maps cleanly to a channel push.
Risk: tokio's ubiquity inviting async into engine code — mitigated by the bridge being
the only crossing point, and by the engine crate having no tokio dependency at all.

## ADR-005: comptime kernel registries → a runtime registry

**Context:** Upstream's kernel dispatch is comptime-keyed: name lists walked with
`inline for` + `comptime std.mem.eql` into indices (`families/nemotron/kernels.zig:
126`), kernel names built with `std.fmt.comptimePrint`, Metal source concatenated and
parameterized at compile time (`families/glm/kernels.zig:265`). Zig has no runtime
equivalent need; Rust has no comptime.

**Decision:** A runtime registry, built once per family at engine open: the kernel
table (name → pipeline) is a `HashMap` built from the same name lists, reading the
generated source registries (`zig/bf16_topk_sources.zig` and siblings) as data. Unknown
names refuse, exactly like the comptime lookup refusing at compile time. Where upstream
instantiates parameterized source at comptime, the port does the same string surgery
at build time in `build.rs` (or a codegen step), so the compiled metallibs match.

**Consequences:** Slight startup cost (build pipelines at open — upstream already
compiles kernels at open on a thread: `families/nemotron/model.zig:58`); identical
refusal semantics; no proc-macro machinery in the hot path. The registries are the
one place the port changes shape deliberately; everything around them keeps upstream's
naming so the generated files stay diffable.

## ADR-006: Metal bindings: the `metal` crate

**Context:** Rust's Metal story has two candidates: the `metal` crate (used by candle's
Metal backend, mlx-rs, burn) and the `objc2-metal` bindings.

**Decision:** The `metal` crate. It is the battle-tested path for exactly this use case
(LLM inference on Metal), its API maps 1:1 to the MTL selectors upstream calls, and
the autorelease-pool pattern matches upstream's `mtl.objc.Pool` idiom.

**Consequences:** Phase 1's ICB/residency/fence work (`src/metal/icb.zig`,
`residency.zig`, `sync.zig`) will need the crate's lower-level escape hatches
(`msg_send!`); that is the crate's normal usage and was the plan even with the
alternative. Known transitive wart: the `block` 0.1.6 dependency emits a
future-incompatibility notice from the crate graph — upstream of us, monitored, not
ours to patch.

## ADR-007: safetensors: hand-ported parser, not the `safetensors` crate

**Context:** The `safetensors` crate exists and would parse the format. But upstream's
`safetensors.zig` applies TensorFold-specific rules the crate does not: the rank-5
ceiling, exact dtype spellings including `F8_E8M0`, and offset/shape arithmetic
refusals the upstream tests pin case by case.

**Decision:** Port the parser from `safetensors.zig` (it is ~250 lines with its rules)
over `memmap2`, with the same refusals and the same test cases.

**Consequences:** The accepted-file set matches the Zig engine exactly, which is what
"leftovers refused" and the loaders behind it rely on. The cost is owning a parser for
a frozen format — small, bounded, and it made the F8_E8M0 dtype and the rank ceiling
non-issues from day one.

## ADR-008: Binary and document parity: `tensorfold-native`, `capabilities --json`

**Context:** Switch tooling and scripts around upstream invoke
`tensorfold-native capabilities --json` and parse a specific document shape
(`server/cli.zig:289`).

**Decision:** Same binary name; same document keys in the same order (hence
`serde_json` with `preserve_order`); `"engine": "rust"` marks the port; the flags list
carries only flags this build honours — upstream's own rule that a flag the table does
not carry is refused (`cli.zig:18`).

**Consequences:** Anything scripted against the Zig binary's interface works against
the port to the extent the port implements it, and fails loudly where it does not —
which is the honest failure mode for a partial port.

## ADR-009: The none-backend is a real component

**Context:** Upstream has an engine-less build whose server runs for tests and parity
suites (`native/none.zig`, `zig/tests/server/build.sh`), and whose open refusal is a
user-facing message pointing at the Python engine.

**Decision:** `NoneBackend` is a first-class `Backend` implementation with upstream's
exact refusal message, and Phase 0's `serve` path runs through it rather than around
it.

**Consequences:** The server, CLI, and contract get exercised in their real shapes
from day one, and Phase 1 swaps a family backend in without touching the server or
CLI — the seam upstream already proved.

## ADR-010: The safetensors rules get their own crate

**Context:** Phase 1a's Metal-resident checkpoint (`checkpoint_metal.zig`'s port)
needs the safetensors header parser and dtype rules, but the only crate holding them
(`tensorfold-core`) also carries the tokenizer and its dependency tree.

**Decision:** `tensorfold-safetensors` holds `dtype.rs` and `safetensors.rs`
(including the `parseHeaderPrefix` port), matching upstream's placement where
`safetensors.zig` is one self-contained module; `tensorfold-core` re-exports it, and
`tensorfold-metal` depends on the small crate, not the tokenizer-carrying one.

**Consequences:** The Metal layer's dependency graph stays proportional to what it
does; the two reading layers (host namespace and Metal-resident shards) provably
share one set of admission rules — the same property upstream gets from one
`safetensors.zig`.

## ADR-011: Runtime kernel compilation ports as runtime compilation

**Context:** Upstream compiles the qwen3_5 kernels from embedded source at engine
open (`Library.fromSource` with `CompileOptions.mlx()`), the same runtime compiler and
options MLX's `mx.fast.metal_kernel` uses — a deliberate exactness choice, since the
captured fixtures were produced under that compiler. A port could instead build the
metallibs at build time with `xcrun`.

**Decision:** Port the runtime path as-is: sources embedded (`include_str!` where
upstream `@embedFile`s), compiled at open with the three MTLCompileOptions setters
upstream sets (`setMathMode: safe`, `setMathFloatingPointFunctions: fast`, and the
OS-mapped `setLanguageVersion:`). And — a Phase 1a discovery recorded here because
it bit — pipeline creation follows upstream's **descriptor path**
(`MTLComputePipelineDescriptor` + `newComputePipelineStateWithDescriptor:...`),
not the direct `newComputePipelineStateWithFunction:` selector, which throws on
this toolchain for runtime-compiled MSL 4.1 functions. Upstream never used the
direct selector; the port now mirrors it exactly.

**Consequences:** The numerics surface is the same compiler MLX used, which is what
the exactness gates measure against. Compile time lands at engine open (~0.5s for
the 20 kernels, measured), exactly where upstream pays it.

## ADR-012: ObjC ownership mirrors upstream's objc.zig, exactly

**Context:** Two Phase 1a crashes taught the same lesson. First, pipeline creation
threw (ADR-011's half); second, the qualify test segfaulted at teardown — an
over-release: `stringWithUTF8String:` returns an **autoreleased** NSString, but the
port released it the way upstream's callers release their strings.

**Decision:** Every ObjC object the port creates uses upstream's ownership rules
verbatim: `objc.string` is `alloc` + `initWithBytes:length:encoding:` (retained, so
callers own it), pipelines come from `new`-allocated descriptors the caller releases,
and libraries from the retained `newLibraryWithSource:` result. When in doubt, read
`metal/objc.zig` and copy its pattern rather than reaching for the idiomatic
convenience constructor.

**Consequences:** Reference counting stays balanced by construction; the qualify
integration test doubles as the leak/crash gate for the whole runtime-compile path
(a clean exit is part of its pass condition).

## ADR-005 note (Phase 1a): the runtime registry, realized

ADR-005's replacement for the comptime kernel registries now exists for qwen3_5:
`crates/qwen35/src/kernels.rs` holds the generated `kernels.zig`'s 15 entries plus
the 5 layout kernels as a frozen static table (`columns`/`threads` verbatim), with
`by_key`/`projection` lookups and the same refusal on unknown names. Because the
config gate pins the geometry to one exact shape, the launch geometries are
constants; if upstream regenerates its kernels, the table re-freezes from the new
`kernels.zig` — tracked in `docs/ZIG-TO-RUST-MAP.md`.

## ADR-013: The lane core keeps upstream's pointer topology, as documented unsafe

**Context:** `zig/src/core/lanes` is a pointer-linked graph: the engine holds
`[]*Stream`, the depth rule reaches a stream's state through `Who.state`, the rule
calls the engine back through `Probe`, and windows carry `*Stream` into the
backend. The exactness contract pins the round loop's decisions, and re-architecting
the topology to satisfy the borrow checker would change the very logic the gates
measure.

**Decision:** `tensorfold-lanes` mirrors the topology with raw pointers under the
same invariants upstream runs by, written once in the crate's documentation:
one thread owns the engine; a stream is live, boxed, and distinct from admission
until release; pointers into a stream are only used while that holds. Every unsafe
block names the field it touches; the stream-taking public functions are `unsafe`
with `# Safety` sections; the whole crate carries the pointer-deref lint allow with
that justification. The `Backend` trait is the port of upstream's vtable — its
stream parameters carry the invariant instead of the vtable's implicit contract.

**Consequences:** The core compiles to the same shape as upstream's, and the engine
tests (mirrored case by case from engine_test.zig) are the behavioural proof. The
host (part 2) must hold streams boxed and drive the engine from one thread — the
same discipline upstream's LaneHost already has.

## ADR-014: The engine's arena is unbounded by an accessor, not by leak

**Context:** Upstream's round loop allocates its round temporaries in an arena
reset every step, and its products (window slices, sampled rows, the concluded
path) outlive any single borrow of the engine — a shape Rust's borrow checker
cannot express with a plain field.

**Decision:** The engine owns a `bumpalo` arena reached through
`Engine::arena()`, an accessor returning `&'static Bump` under documented
invariants: the engine is single-threaded, the arena resets only between rounds,
and every slice dies within the round that allocated it (the round's temporaries
never escape `step`). The same accessor shape backs the decision log's arena.

**Consequences:** The round functions keep upstream's structure (allocate, call,
consume) without per-round `Vec` churn; the reset between rounds is the
scratch-free guarantee. The invariant is enforced by review of the two reset
sites (`step`, `add_stream`) and by the tests, which run hundreds of rounds
through the fake and the real backend.

## ADR-015: The chat template renders with minijinja, with a Python string-method shim

**Context:** Upstream ships its own Jinja-subset engine (`zig/src/core/template/`, ~4.6K
lines) because the native binary must render `chat_template.jinja` with no Python.
The pinned checkpoint's 153-line template uses macros, `namespace()`,
`raise_exception`, `tojson`, and — the hard part — Python string *methods*:
`content.startswith(...)`, `content.split(...)[0].rstrip(...).split(...)[-1]`.

**Decision:** Render with `minijinja` — the same reference-implementation argument as
the `tokenizers` crate (ADR-003): upstream's engine exists only for the
no-Python constraint, and minijinja is a full Jinja2 implementation. The one gap —
Python string methods called as methods, which Jinja2 supports and minijinja does
not — is closed by a source shim at template load (`template.rs`'s
`patch_python_string_methods`): `.startswith(x)`/`.endswith(x)` become
`is startingwith(x)`/`is endingwith(x)` tests, and `.split(x)`/`.lstrip(x)`/
`.rstrip(x)` become the equivalent filters with the operand parenthesized around
each call that an index follows, so chains evaluate in Python's order:
`content.split(a)[0].rstrip(b)` → `((content|split(a))[0]|rstrip(b))`. The shim is
string-literal aware (methods inside quotes stay) and tested against the pinned
template itself, which renders byte-identically on the load path.

**Consequences:** The pinned model family's templates render without modification;
templates using Python methods the shim doesn't cover will fail loudly at load
(the template's refusal, not a guess). When more families arrive in Phase 2, the
shim grows their methods the same way.

## ADR-016: `Engine::submit` takes ownership; the engine's methods take `&self`

**Context:** Upstream's contract keeps the request and its sink alive "until its
`finished` event" (engine_api.zig:45) through raw ctx pointers, and the engine
itself is a ctx+vtable the HTTP threads share — the submit may arrive on any
thread while the engine thread steps rounds.

**Decision:** The port expresses both facts with ownership: `submit` takes
`Box<Request<'static>>` and `Box<dyn EventSink + Send>` — the engine holds them
until the finished event has gone out, then drops them (the chat route leaks the
rendered prompt slices into the request the same way). Every `Engine` method takes
`&self`: the implementor's interior mutability is the port of upstream's ctx
pointer (the lane host's mutexes), so the server can hold the engine behind an
`Arc`.

**Consequences:** No lifetime unsoundness at the seam; the chat route's Mailbox is
a plain `EventSink + Send`. Requests' borrowed pieces (stop checks, call gates,
structures) are boxed at the server's edge in Phase 2 as those features port.
