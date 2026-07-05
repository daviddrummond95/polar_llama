# Local MLX Backend — Implementation Plan (authoritative)

**Status: this document supersedes the earlier "LocalMLXProvider" handover doc in full.**
Where the two disagree, this document wins. All agents implement against the API,
file map, and acceptance definitions below.

---

## 0. Reframing: what the old doc got wrong

Two of the old doc's central premises are false, and both cut in our favor.

**1. "LocalMLXProvider implements the same provider contract" — impossible, and unnecessary.**
The provider contract in this repo is Rust-only: `#[async_trait] pub trait ModelClient`
at `src/model_client/mod.rs:126`, dispatched through the closed `enum Provider`
(`src/model_client/mod.rs:56`) and the `create_client` factory (`src/model_client/mod.rs:351`).
Every implementor is a per-request HTTP client; batching lives *above* the trait in
`run_batch` (`src/model_client/mod.rs:397`, `futures::buffered` at `:414`, concurrency from
`POLAR_LLAMA_MAX_CONCURRENCY` at `:34-36`, default 64). There is no Python provider
abstraction anywhere. A Python MLX generation loop cannot implement a Rust async trait,
and we will **not** add a `Provider::LocalMLX` variant — the enum stays closed. Instead:

- The **server** path needs zero Rust changes, because `OpenAIClient::api_endpoint`
  already honors `OPENAI_BASE_URL` (`src/model_client/openai.rs:58`). Pointing the
  existing async fan-out at `mlx_lm.server` / vllm-mlx *is* the provider integration.
- The **in-process** path bypasses Rust entirely: a pure-Python `map_batches` UDF over
  the whole column, mirroring the existing plugin→`map_batches` pattern at
  `polar_llama/__init__.py:443-531`.

**2. "Build a continuous-batching scheduler and hybrid batched KV cache" — already shipped upstream.**
Verified against a real mlx-lm **0.31.3** wheel:

- `BatchGenerator` (`mlx_lm/generate.py`) is a continuous-batching scheduler with
  per-request `max_tokens` / `stop` / `uids`, and `insert(..., caches=...)` for
  per-request prompt caches. It sets the Metal wired limit itself.
- `batch_generate(prompts, prompt_caches=..., return_prompt_caches=...)` is the
  one-shot convenience wrapper.
- Left-padded `BatchKVCache` and `BatchRotatingKVCache` (with merge/extend/extract)
  live in `mlx_lm/models/cache.py`; `_make_cache` auto-converts a model's
  `make_cache()` output into batched caches — including Gemma 4's hybrid layout.
- Gemma 4 (`gemma4.py` / `gemma4_text.py`: hybrid 5:1 sliding/full attention,
  `sliding_window=512`, KV-shared later layers) and Qwen 3.5 (`qwen3_5.py`) are
  supported model families upstream.

So the scheduler, the batched hybrid KV cache, and the model ports the old doc
budgeted weeks for are **zero work**. Our job is a thin, correct adapter and the
correctness rules around prefix reuse. What upstream does *not* have (and we
therefore cut or defer, §3): a `BatchQuantizedKVCache` (`RotatingKVCache.to_quantized`
raises `NotImplementedError`), batched speculative decoding (spec-decode is
single-sequence only), and the fictional "extract nested E2B draft from E4B" API.

---

## 1. Public API (frozen — all agents conform to this exact shape)

```python
pl.col("prompt").llama.inference_local(
    model,                  # positional: HF repo id or local path
    *,
    system=None,            # separate arg — the prefix-cache key basis (§4)
    engine="server",        # "server" (default) | "in_process"
    base_url=None,          # server engine only; default http://localhost:8080
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    stop=None,              # str | list[str] | None
) -> pl.Expr                # String completions, ORIGINAL ROW ORDER
```

- Namespace method on `LlamaNamespace` (`polar_llama/__init__.py:1151-1152`), named
  `inference_local` — the repo convention is `inference_*` (`inference` :1176,
  `inference_async` :1210); there is no `complete_*` anywhere. Keyword-only args,
  delegating to a module-level function, exactly like every existing method.
- `system` is a **separate argument**, never concatenated into the prompt by callers.
  This structurally enforces immutable-prefix-first: the rendered system prefix is
  the cache key (§4). It maps onto the existing `system_prompt` plumbing that
  `inference_async` already has (`polar_llama/__init__.py:331`).
- Per-row failures return an **error-JSON string** in that row, mirroring
  `create_error_response` (`src/model_client/mod.rs:335`):
  `{"_error": <type>, "_details": <msg>, "_raw": <optional>}`. A bad row never
  kills the batch — same contract as the HTTP path (`mod.rs:379-389`).

### Module layout (fixed by `polar_llama/local/__init__.py`, already landed)

The packaging agent's `polar_llama/local/__init__.py` lazily re-exports these names
from these sibling modules; implementors must use exactly these paths:

| Module (`polar_llama/local/`) | Exports |
|---|---|
| `engine.py` | `LocalEngine` (Protocol), `FakeEngine`, `MlxBatchEngine` |
| `expr.py` | `inference_local` (module-level function the namespace delegates to) |
| `prefix_cache.py` | `PrefixCache` |
| `server_backend.py` | `start_server_backend` |

Import rules: **nothing** under `polar_llama/local/` may import `mlx`/`mlx_lm` at
module top level. Use `polar_llama.local.require_mlx()` (already implemented in
`local/__init__.py:61`) inside functions that need it; it raises the helpful
`pip install polar-llama[local]` ImportError. `uv run python -c "import polar_llama"`
must keep working on a machine with no mlx.

---

## 2. Task board

Tiers: **VD** = very difficult, **M** = moderate, **R** = routine.
**[GPU]** = final validation blocked on this machine's GPU + real model weights
(logic still built and unit-tested against `FakeEngine` first).

| # | Task | Tier | Owner / files | Notes |
|---|------|------|---------------|-------|
| T0 | This plan (source of truth) | M | `docs/local_mlx_backend_plan.md` (this file) | Done. |
| T1 | `engine="server"` routing | M | INTEGRATION agent: `polar_llama/__init__.py` (namespace method + module fn stub delegating to `local/expr.py`); ENGINE agent: `polar_llama/local/expr.py` | Set `os.environ["OPENAI_BASE_URL"]` (read by Rust at `openai.rs:58`), default `base_url` to `http://localhost:8080`, dispatch to `inference_async(provider="openai", model=..., system_prompt=system)`. Set a dummy `OPENAI_API_KEY` if unset (local servers ignore it; `mod.rs:233` reads it unconditionally). Document that `OPENAI_BASE_URL` is process-global (§6, caveat). |
| T2 | Engine seam: `LocalEngine` Protocol + `FakeEngine` + singleton registry | M | ENGINE agent: `polar_llama/local/engine.py` | `generate(prompts: list[str], params: GenerationParams) -> list[str]`, row-aligned. `FakeEngine` is deterministic (`"echo:" + prompt` / canned map), imports no mlx. Registry: process-global dict keyed `(model, engine)` behind a `threading.Lock`; loads once (§6, threading). |
| T3 | `MlxBatchEngine` wrapping `mlx_lm` `BatchGenerator`/`batch_generate` | **VD** [GPU] | ENGINE agent: `polar_llama/local/engine.py` | Thin adapter only — no scheduler of our own. `require_mlx()` inside `__init__`. Lifecycle: `load()` once, `close()` in `finally` (restores Metal wired limit). Sampling/stop plumbed per-request via `BatchGenerator` uids. |
| T4 | Chat-template rendering + prefix split | M | ENGINE agent: `polar_llama/local/templates.py` (new) | `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Render `(system, user)` → full prompt AND the rendered immutable prefix string; pure function, CI-testable with a fake tokenizer. |
| T5 | `PrefixCache`: EXACT/EXTEND reuse with token-boundary verification | **VD** [GPU] | ENGINE agent: `polar_llama/local/prefix_cache.py` | Rules in §4. Classification logic is pure Python → fully CI-testable; only the mlx cache-object handling is GPU-blocked. |
| T6 | UDF glue: `map_batches` expr, row-order preservation, per-row error JSON | M | ENGINE agent: `polar_llama/local/expr.py` | Mirror `polar_llama/__init__.py:443` pattern. UDF may be called **per-morsel** under the streaming engine, so it must be re-entrant against the T2 singleton; never assume it sees the whole column at once. `return_dtype=pl.String`. |
| T7 | Packaging: `[local]` optional extra | R | PACKAGING agent: `pyproject.toml`, `polar_llama/local/__init__.py` (owned; landed) | `local = ["mlx-lm>=0.31.3"]` with `sys_platform == 'darwin' and platform_machine == 'arm64'` markers. |
| T8 | CI: fake-engine tests on ubuntu-latest | R | CI agent: `.github/workflows/CI.yml` (~:110 `linux_tests`); TEST agent: `tests/test_local_*.py` (new files only) | No mlx, no GPU, no network. MLX's CPU backend exists but we do **not** rely on it in CI. |
| T9 | Acceptance benchmarks: parity, TTFT warm/cold, throughput, memory | M [GPU] | BENCH agent: `benchmarks/local_mlx/` (new) | Definitions in §5. Runs only on this M4 Pro via `uv run python ...`. |
| T10 | User docs | R | DOCS agent: `docs/local_mlx_backend.md` (referenced by `local/__init__.py:14`), README section | Include licensing note (§6). |
| T11 | Deferred-items ledger | R | This file, §3 | Quantized batch KV, Metal kernel, batched spec decode: cut with rationale. |

Dependency spine: T2 → {T3, T5, T6} → T9; T1 is independent and ships first; T4 feeds T5.

---

## 3. Build vs. buy — decisions

**(a) SHIP NOW — `engine="server"` (buy).** Zero Rust changes: `OPENAI_BASE_URL`
(`openai.rs:58`) points the existing `run_batch` fan-out (`mod.rs:397`, 64-way
`buffered`) at `mlx_lm.server` or vllm-mlx. We inherit retries-by-row error JSON,
concurrency control, and the whole tested HTTP path for free. The server does its
own continuous batching. This is the low-risk path and the default. `start_server_backend`
(`local/server_backend.py`) is a convenience that spawns/health-checks `mlx_lm.server`
as a subprocess; using an already-running server is equally supported via `base_url`.

**(b) SCAFFOLD — `engine="in_process"` (thin build).** Wrap — do not reimplement —
`BatchGenerator`/`batch_generate`. Value over the server path: no HTTP/serialization
overhead, `prompt_caches=`/`return_prompt_caches=` access for real prefix reuse
(§4), and no separate process to manage. All scheduling, padding, and batched hybrid
KV stay upstream's problem.

**(c) CUT / DEFER — with upstream justification:**

- **`BatchQuantizedKVCache`: CUT.** It does not exist upstream, and
  `RotatingKVCache.to_quantized` raises `NotImplementedError` — for Gemma 4's hybrid
  layout there is nothing to quantize the sliding layers into. The motivation is also
  gone: 4-of-5 Gemma-4 layers are capped at 512 tokens and later layers share KV, so
  KV is not the memory bottleneck on 24 GB (§5). Revisit only if upstream ships it.
- **Custom Metal kernel: CUT.** `BatchGenerator` already manages the wired limit and
  uses mlx's fused kernels. The old doc's perf gap this was meant to close was
  extrapolated from vllm-mlx numbers measured on an **M4 Max** (≈2× this machine's
  memory bandwidth); those 4.3×/5.8× figures are ceilings, not our targets.
- **Batched speculative decoding: CUT.** Upstream spec decode is single-sequence only
  (not in `BatchGenerator`), and the old doc's "extract a nested E2B draft from E4B"
  API **does not exist** in mlx-lm. Building batched spec decode is an upstream
  contribution, not a polar-llama feature.

---

## 4. Prefix-reuse correctness rules (`PrefixCache`, T5)

The unit of reuse is the **rendered chat-template prefix**: the exact string produced
by `apply_chat_template` for the system turn + template preamble, up to (and not
including) the first character that varies per row. `system` being a separate API
argument guarantees this prefix is identifiable without parsing user text.

1. **Cache key** = `(model_id, tokenizer/template revision, rendered_prefix_string)`.
   Never key on the raw `system` arg alone — template changes must invalidate.
2. **Token-boundary verification (BPE hazard).** Storing `prefix_tokens =
   tokenize(rendered_prefix)` is not enough: BPE may merge across the prefix/suffix
   boundary. On every lookup, tokenize the **full** prompt and verify
   `full_tokens[:len(prefix_tokens)] == prefix_tokens`. On mismatch → treat as MISS
   (full prefill). Never slice a cache at a token position not verified this way.
3. **Classify EXACT / EXTEND / MISS only — never TRIM.**
   - EXACT: cached token seq == needed prefix → reuse cache as-is.
   - EXTEND: cached seq is a strict prefix of the needed tokens → reuse and prefill
     the remainder (upstream `merge`/`extend` on batch caches supports this).
   - Anything requiring trimming a *longer* cached sequence down → **MISS**.
     Rationale: mlx-lm issue **#980** — the hybrid (sliding-window) cache trim path
     is broken; it was closed "outdated" via a server-side mitigation and no trim
     fix was merged. We structurally avoid the trim path rather than trusting it.
4. **Replicate, don't share (v1).** `BatchGenerator.insert(..., caches=...)` gives
   each sequence its own cache; sharing one prefix's KV across B rows in a batch
   means B physical copies. Accept this: for Gemma 4 the replicated prefix KV is
   cheap (sliding-512 caps 4/5 layers, later layers KV-shared), and replication is
   trivially correct. True cross-sequence sharing = deferred, upstream-shaped work.
5. **Eviction:** LRU by key, capped count (default 4 prefixes) — caches are
   host-memory-visible mlx arrays; unbounded growth is an OOM vector.

### Parity + latency acceptance DEFINITIONS (T9 measures these; agree on them now)

- **Parity:** greedy decoding (`temperature=0.0`) with fixed `mx.random.seed(0)`.
  For each row: in-process batched output vs single-sequence `mlx_lm.generate`
  reference. Left-padding changes numerics, so exact token match is NOT the bar.
  Pass = **divergence rate ≤ 2% of rows** diverging within the first 64 generated
  tokens, AND for diverging rows the first-divergent-step max-abs logit delta
  < 1e-2 (i.e. a genuine near-tie, not a bug). A prefix-cache-reused row must be
  parity-checked against the *same* row run cold — EXACT-hit reuse must be
  token-identical under greedy (no padding difference applies there).
- **TTFT, per row, under continuous batching:** time from `BatchGenerator.insert()`
  of that row to its first emitted token, measured at steady state (batch ≥ 8 rows
  in flight). **Cold** = prefix MISS; **warm** = EXACT/EXTEND hit. Report the
  warm/cold ratio per model; do not compare against the old doc's numbers.
- **Acceptance criterion 4 of the old doc (throughput target) is VOID** pending
  empirical re-derivation on this machine (§5) — it was derived from wrong memory
  math and M4 Max benchmarks.

---

## 5. Memory budget on this machine (M4 Pro, 24 GB) — corrected

The old doc's Q4 weight figures (1.3 / 2.5 / 5.0 GB) are wrong — it quantized
"effective" (PLE-offloaded) parameter counts. **Real measured 4-bit footprints:**

| Model | Q4 weights | Old doc said |
|---|---|---|
| Gemma 4 E2B | **2.51 GB** | 1.3 GB |
| Gemma 4 E4B | **5.82 GB** | 2.5 GB |
| Ornith-1.0-9B | **5.2 GB** | 5.0 GB |

Usable GPU budget: macOS wires ≲ ~75% of unified memory to Metal → ~16–17 GB
ceiling; leave ~2 GB for Polars + Python + framework overhead → plan against
**~14 GB** for weights + KV + activations.

KV per sequence (fp16): `2 × n_kv_heads × head_dim × min(seq_len, window) × 2 bytes
× n_layers_effective`. For Gemma 4, 4/5 of layers cap at `min(seq_len, 512)` and
later full-attention layers **share** KV, so per-sequence KV grows sub-linearly in
context and is dominated by a handful of full layers. Order of magnitude at 4k
context: tens of MB per sequence, not GB — hence the quantized-KV cut (§3c).

**Honest batch-size expectations (to be confirmed by T9, not promised):**

| Model | Weights | Headroom for KV+act | Expected concurrent seqs @ ~2–4k ctx |
|---|---|---|---|
| E2B (2.51 GB) | 2.5 GB | ~11.5 GB | 64+ (scheduler-limited, not memory-limited) |
| E4B (5.82 GB) | 5.8 GB | ~8 GB | ~24–48 |
| Ornith-9B (5.2 GB) | 5.2 GB | ~9 GB | ~16–32 (standard KV, no sliding cheapness — verify arch) |

These are planning numbers only. **T9 must re-derive acceptance criterion 4
empirically** on this machine before any throughput number appears in user docs.

---

## 6. Omissions in the old doc — now owned tasks

1. **Chat templates.** Raw-string prompting silently mis-formats instruct models.
   `templates.py` (T4) always renders via `tokenizer.apply_chat_template(...,
   add_generation_prompt=True)`; the rendered prefix feeds `PrefixCache`.
2. **Sampling + stop plumbing.** `temperature`/`top_p` → `mlx_lm` `make_sampler`;
   `stop` and `max_tokens` are **per-request** via `BatchGenerator` uids. The API
   defaults to greedy (`temperature=0.0`) for reproducibility.
3. **Model lifecycle.** First use downloads from HF hub to its standard cache.
   Registry (T2): process-global `(model, engine)` → engine singleton; explicit
   `close()`; document that swapping models in one process holds both until closed.
4. **GIL + Polars threading.** Polars calls `map_batches` UDFs from its own threads,
   possibly per-morsel and concurrently under the streaming engine. The engine
   singleton lives behind a lock; `generate()` calls are serialized through it
   (a queue, effectively). Two morsels never race the Metal context. `FakeEngine`
   tests assert re-entrancy (concurrent calls from a ThreadPool).
5. **Cancellation + cleanup.** `try/finally` around generation; `finally` calls
   `BatchGenerator.close()` so a Ctrl-C mid-query restores the Metal wired limit
   instead of leaving the GPU wedged.
6. **Per-row error isolation.** Engine catches per-row exceptions and returns the
   `create_error_response`-shaped JSON (`mod.rs:335`): `{"_error", "_details",
   "_raw"?}` — downstream tooling already parses this shape.
   *Server-path caveat:* `OPENAI_BASE_URL` is process-global env; two concurrent
   queries with different `base_url`s race. Document; do not "fix" with Rust changes.
7. **Packaging.** `[local]` extra (T7), darwin/arm64 markers, lazy import guards
   (already enforced by `local/__init__.py`). CI never installs the extra.
8. **Licensing.** Gemma 4 weights ship under the **Gemma Terms of Use** (gated,
   use-restricted) — polar-llama must not bundle or auto-accept; docs link the ToU.
   **Ornith-1.0-9B is MIT** (post-trained on Gemma4/Qwen3.5) → use it as the
   default model in examples/benchmarks. Citation hygiene for docs: prompt-cache
   paper is **Gim et al., arXiv:2311.04934** (not "Gill"); the "MLX leads llama.cpp
   by 20–87%" figure is **arXiv:2601.19139 (vllm-mlx)**, not 2511.05502; the Excel
   integration is **lablup/mlxcel**; vllm-mlx is waybarrios/vllm-mlx (Apache-2.0),
   benchmarked on M4 Max — treat its 4.3×/5.8× as ceilings.
9. **CI without GPU.** Tests run on ubuntu-latest only (`.github/workflows/CI.yml`
   ~:110); macOS runners only build wheels. Therefore every piece of local-backend
   *logic* — template rendering, prefix classification (EXACT/EXTEND/MISS incl. BPE
   boundary fallback), registry locking, error-JSON shape, row order, UDF wiring —
   is exercised through `FakeEngine` with no mlx import. `MlxBatchEngine` itself is
   covered by T9 on this machine only, marked `@pytest.mark.skipif(not
   is_mlx_available(), ...)`.

---

## 7. Definition of done

1. `engine="server"`: end-to-end completions through the existing Rust fan-out
   against a local `mlx_lm.server`, no Rust diff, row order preserved. **[GPU-machine
   verification; logic testable with any OpenAI-compatible stub server]**
2. `engine="in_process"`: `FakeEngine` test suite green on Linux CPU;
   `MlxBatchEngine` passes the §5 parity + TTFT definitions on this M4 Pro. **[GPU]**
3. `uv run python -c "import polar_llama"` and `import polar_llama.local` succeed
   without mlx installed; helpful ImportError only when `engine="in_process"` is used.
4. Throughput acceptance number re-derived empirically (T9) and recorded here,
   replacing the old doc's void criterion 4.
