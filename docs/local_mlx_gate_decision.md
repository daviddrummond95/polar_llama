# Local MLX Backend: Build-vs-Buy Gate

**Status:** the hybrid batched blocker (**#1384**) is now **fixed & verified**
(see RESOLVED below) — a runtime patch makes `in_process` batching correct on
hybrid Gemma. Remaining gate factors: build (in_process) beats sequential 1.34×
on Qwen3-8B but is memory-bound on 24 GB; buy (`mlx_lm.server`) OOMs when
batching big shared prompts; both named target checkpoints need mlx-vlm to load.
**Harness:** `benchmarks/local_mlx_gate.py`
**Target machine:** Apple M4 Pro, 24 GB unified memory, macOS 26.x (this class
of machine *is* the deployment target — no cloud GPU proxy is acceptable).

## Real-hardware run — 2026-07-04 (M4 Pro, this machine)

Ran the harness on real target-class models. Two of the design's open
questions are now **empirically resolved**, and a blocking upstream bug was
**reproduced**.

### Resolved architecture questions
- **Ornith-1.0-9B is a Qwen 3.5 derivative** (`config.json` → `model_type:
  qwen3_5`, `Qwen3_5ForConditionalGeneration`, text `qwen3_5_text`). mlx-lm
  ships `qwen3_5`. Smallest MLX quant is 8-bit (~9 GB); no 4-bit published yet.
- **Gemma 4 E4B is multimodal** (`Gemma4ForConditionalGeneration`, with
  `audio_config` + image tokens). `mlx_lm.load` rejects it — *"Received 126
  parameters not in model"* (the vision/audio tower). It needs **mlx-vlm**, not
  mlx-lm's text loader. There is **no text-only `-lm` quant for Gemma 4**; the
  closest loadable stand-in is `gemma-3n-E4B-it-lm-4bit` (same PLE +
  sliding-window hybrid lineage, text-only), which is what was benchmarked.

### BLOCKING upstream bug (reproduced) — batched gen is broken for hybrid
On `mlx-community/gemma-3n-E4B-it-lm-4bit`, **every batched path crashes**:

```
mlx_lm/models/gemma3n.py:133   keys, values = cache.state
ValueError: too many values to unpack (expected 2)
```

The batched cache (`BatchKVCache` / `BatchRotatingKVCache`) exposes a `.state`
with more than two elements, but gemma3n's attention hardcodes a 2-tuple
unpack. This breaks **both** candidate engines:
- **build** — in-process `batch_generate` (with *and* without prefix cache): ✗
- **buy** — `mlx_lm.server` also batches internally (`_generate` →
  `batch_generator.next()` → `model(cache=prompt_cache)` → same line 133): ✗
  (server returns HTTP 200 then its generation thread dies)

Only **single-sequence** `mlx_lm.generate` works on the hybrid model. The tiny
full-attention Qwen-0.5B batched fine (earlier smoke), so the bug is specific
to the **sliding-window / hybrid** cache path — a *more severe* cousin of
issue #980 (that was cache *reuse*; this is a first-time batched forward).

### Measured numbers (only `sequential` is valid for hybrid)
`gemma-3n-E4B-it-lm-4bit`, 32 rows, max_tokens 128, shared 5,135-char system prompt:

| mode | ok | wall_s | eff tok/s | TTFT p50 | peak GB | notes |
|---|---|---|---|---|---|---|
| sequential | ✓ | 56.5 | 11.5 | 1.435 s | 4.66 | ~46 s (32 × 1.435) is **repeated prefill of the shared prompt** |
| in_process | ✗ | — | — | — | — | crash: gemma3n `cache.state` unpack |
| server | ✗* | (65.5) | — | — | — | *false-positive `ok`; server batch worker crashed, wall_s is hang+errors — discard* |

The `sequential` result is itself the argument for prefix caching: **82 % of
wall-clock is re-prefilling the same 5 KB system prompt 32 times.** That is the
win the in-process/server prefix cache is meant to capture — and it is exactly
what the upstream bug currently blocks for these models.

### Full-attention datapoint (Qwen3-8B — where batching actually runs)
Both named targets are multimodal (unloadable text-only), so build-vs-buy was
measured on `mlx-community/Qwen3-8B-4bit` — the closest loadable relative of
Ornith (which is itself `qwen3_5`). Full attention ⇒ no #1384. 16 rows, 128
max tokens, shared 5 KB system prompt, 2048 output tokens each (apples-to-apples):

| path | wall_s | eff tok/s | peak GB | outcome |
|---|---|---|---|---|
| sequential (batch 1) | 101.5 | 20.2 | 5.2 | ok |
| **in_process** (build, batched, no prefix cache) | **75.6** | **27.1** | 9.6 | **1.34× faster** |
| server (buy, batches internally) | — | — | — | **OOM at batch 16 (full prompt)** |
| batch 32 (in_process) | — | — | — | **OOM** |

Findings:
- **Build beats sequential 1.34×** — real but modest. Prefix caching (the big
  lever that would collapse ~59 s of repeated prefill) is *disabled* here
  because replicating the prefix KV B× **OOMs** at these batch sizes; so batching
  only amortizes decode. On M4 Pro (½ the bandwidth of the M4 Max in the vllm-mlx
  paper) the headline 3–4× targets are not reached without a working prefix cache.
- **The 24 GB memory wall dominates, not compute.** batch-32 OOMs; batch-16
  in_process fits at 9.6 GB (weights 5 GB + full-attention KV nearly doubles it).
- **Buy is not a free pass on memory.** `mlx_lm.server` batches internally, so at
  batch 16 with the full prompt it OOMs too — no valid server number at 8B on
  24 GB. (Server batch-size caps / queueing would be needed.)
- **Server fairness bug found & fixed in the harness.** `inference_async(system_prompt=…)`
  only sends the system prompt when `cache=True`; the benchmark's earlier
  `run_server` silently sent ~53-token prompts. `run_server` now calls the
  *shipped* `polar_llama.local.server_backend.inference_local_server`, which
  builds an explicit `[system, user]` array (confirmed: server saw 1,339-token
  prompts). The shipped path was already correct; the benchmark was not.

### Patch spike + upstream status (attempted 2026-07-04)
Tried the "small vendored patch" idea end-to-end:
- `gemma3n.py:133` `keys, values = cache.state` → `cache.state[:2]` (batched
  caches return a 4-tuple `(k, v, offset, left_padding)`; non-batched return a
  2-tuple). This **fixes the crash** — batched generation now runs.
- **But output is garbage.** Greedy parity, chat-templated, on
  `gemma-3n-E4B-it-lm-4bit`:
  - sequential (reference): `"Paris"` / a correct primary-colors list
  - batched (patched): `"Okay.\n\nItItItIt…"` / `"OkayOkayOkay…"`
  The KV-**shared** layers read `cache.state[:2]` (sliced keys/values) without
  the offset/left-padding alignment the batched path needs — the classic
  "silently wrong output" hybrid-cache failure. A correct fix is real
  shared-KV/batched-cache work, **not a one-liner**, and guessing risks
  plausible-looking wrong answers.
- **This is a known OPEN upstream bug:**
  [ml-explore/mlx-lm#1384](https://github.com/ml-explore/mlx-lm/issues/1384)
  — *"Gemma 3n: mlx_lm.server returns garbage output (shared-KV layers
  incompatible with batched cache)"* (opened 2026-06-11). **mlx-lm 0.31.3 is
  the latest published version**, so no upgrade fixes it today.

### RESOLVED — #1384 fixed & verified (2026-07-04)
A correct fix now exists. **The root cause was not the `.state` geometry — it was
offset aliasing:** batch caches store `offset` as a mutable `mx.array`, and
`update_and_fetch`'s `self.offset += L` mutates that object in place, so keys are
RoPE'd at `p` and queries at `p+L` on every layer (sequential caches use an
immutable `int`, which hid it). The fix snapshots the offset
(`mx.array(cache.offset)`) and threads each layer's post-`update_and_fetch`
`(keys, values)` to the KV-shared layers — mirroring mlx-lm's own `gemma4_text`.

Deliverables (uncommitted): `patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch`
(upstream-PR candidate), `polar_llama/local/_mlx_patches.py` (guarded runtime
monkeypatch — no fork), `benchmarks/validate_1384_fix.py` (token-parity
validator), `docs/mlx_lm_1384_fix.md` (root cause + PR writeup).

**Verified independently + via the validator against a PRISTINE 0.31.3:** hybrid
gemma3n batched == sequential, token-identical (3/4 exact + 1 exact-tie argmax
flip at a 0.0000-nat logprob gap); Qwen control 4/4 (no regression); 640-token
sliding-window rotation identical; `mlx_lm.server` concurrent requests correct.
Was 0/4 (garbage) before.

### Hybrid batched — first real numbers (post-fix, 2026-07-04)
With the runtime patch, the gemma-3n E4B 3-way finally runs batched + prefix
cache (32 rows, 128 max tokens, shared 5 KB prompt):

| mode | wall_s | eff tok/s | peak GB |
|---|---|---|---|
| sequential | 56.8 | 11.7 | 4.66 |
| in_process (batched + replicated prefix cache) | 43.4 | 14.8 | 7.15 |

**1.31×** — modest (short EOS-terminated outputs make this prefill-bound, and the
harness's prefix-cache replication doesn't fully collapse the repeated prefill),
but two things matter: (1) it is now **correct** (was crash/garbage), and (2)
Gemma's sliding-window KV lets the replicated prefix cache **fit at batch 32
(7.15 GB)** where full-attention Qwen OOM'd at batch 32 — the architectural point
the design made. Deeper prefix-cache efficiency and larger batches are the next
optimization, not a blocker.

### Revised recommendation
1. **Hybrid Gemma E-series (E2B/E4B): the batched crash is FIXED** (see RESOLVED)
   — apply `_mlx_patches` at engine init and `in_process` batching produces
   correct output on hybrid models; land the upstream PR so the patch isn't
   permanent. Remaining for these models: multimodal checkpoints (need mlx-vlm to
   load text-only), prefix-cache reuse (#980), and the 24 GB memory wall (prefix
   replication OOMs — the case for batched quantized KV).
2. **Full-attention / Qwen-family (incl. Ornith-9B if it loads as text):**
   batching works; the buy-leaning preview holds. Reconfirm on
   `Ornith-1.0-9B-8bit` (resolve whether its `Qwen3_5ForConditionalGeneration`
   checkpoint loads text-only in mlx-lm, or also needs mlx-vlm).
3. The `engine="server"` default should **detect/guard models whose batched
   path is broken** and fall back to sequential, rather than returning silent
   errors.

Raw results: `benchmarks/results_gemma3n_e4b_lm_nocache.json` (sequential
valid); `results_gemma3n_e4b_lm_server.json` (server, invalid — see above).

## What is being decided

`pl.col("prompt").llama.inference_local(model, ...)` ships with two engines:

- **`engine="server"` (buy, DEFAULT):** the existing Rust async fan-out
  (`run_batch`, default 64-way concurrency) pointed at a local
  OpenAI-compatible server (`mlx_lm.server` or vllm-mlx) via the
  `OPENAI_BASE_URL` override that already exists in
  `src/model_client/openai.rs`. **Zero Rust changes; zero new inference code
  to maintain.** The maintenance surface is "keep an env var working."
- **`engine="in_process"` (build):** a pure-Python `map_batches` UDF wrapping
  `mlx_lm.batch_generate` / `BatchGenerator` behind the `LocalEngine`
  protocol in `polar_llama/local/`. Maintenance surface: engine singleton +
  lock semantics under the streaming engine, prompt-cache replication,
  per-row error containment, tracking mlx-lm's fast-moving batch API.

The in-process path only earns its maintenance surface if it is *meaningfully*
faster or lighter than the server path on the real workload shape (many rows,
one long shared system prompt, short outputs).

## Decision rule

Run `benchmarks/local_mlx_gate.py` on **both** target models at
`--rows 64 --max-tokens 256` (and once at `--rows 256` to stress the fan-out):

1. **Parity gate (hard):** both paths must return one completion per row, in
   original row order, with no error-JSON rows under normal operation. Any
   parity failure disqualifies that path outright.
2. **Buy threshold:** if the server path's total wall-clock is within **~20%**
   of in-process (`server_wall / in_process_wall <= 1.20`) on both target
   models, **prefer buy**: ship `engine="server"` as the only documented
   engine and keep `in_process` experimental/undocumented.
   The harness prints this ratio directly (`GATE:` line).
3. **Build threshold:** if in-process is **>=1.5x** faster wall-clock *or*
   saves **>=25% peak memory** (relevant on 24 GB with a ~5.8 GB E4B resident),
   the build path earns its keep and both engines ship documented.
4. **Grey zone (1.2x–1.5x):** default to buy; revisit only if users hit the
   server path's operational annoyances (managing a sidecar process, port
   conflicts) in practice.
5. **Effective tokens/s is the metric.** Output tokens divided by *total*
   wall-clock including all prefill. Decode-only tok/s flatters the
   in-process path (its win is batched prefill + replicated prefix caches)
   and must not be used for the gate.

Rationale for the 20% number: the server path costs ~0 marginal maintenance
(it reuses the entire existing provider stack, retries, error JSON,
concurrency env), so a small wall-clock premium is cheaper than owning a
second inference code path. HTTP + JSON overhead per request is milliseconds;
the real variable is whether the server's own batching/prefix caching keeps
up with hand-replicated `prompt_caches`.

## Smoke-test findings (mlx-lm 0.31.3, verified on this machine, 2026-07-04)

Run in an isolated venv against `mlx-community/Qwen2.5-0.5B-Instruct-4bit`:

- **`batch_generate` signature (actual):**
  `batch_generate(model, tokenizer, prompts: List[List[int]], prompt_caches:
  Optional[List[List[Any]]] = None, max_tokens: Union[int, List[int]] = 128,
  verbose: bool = False, return_prompt_caches: bool = False, **kwargs) ->
  BatchResponse`. Prompts are **token IDs, not strings** — passing strings
  fails with an unhelpful `ZeroDivisionError`. `MlxBatchEngine` must tokenize
  (`apply_chat_template(..., tokenize=True)`) before calling.
- **`BatchResponse`** has `.texts`, `.stats`, `.caches`. With
  `return_prompt_caches=True` the caches surface on **`.caches`** (there is
  no `.prompt_caches` attribute despite the parameter name). `.stats` is a
  `BatchStats(prompt_tokens, prompt_tps, prompt_time, generation_tokens,
  generation_tps, generation_time, peak_memory)` — use `generation_tokens`
  for effective-throughput math instead of re-tokenizing outputs.
- **Prefix-cache replication works:** warm one system-prefix cache
  (`max_tokens=1, return_prompt_caches=True`), `copy.deepcopy` it per row,
  pass via `prompt_caches=`. Output parity with the uncached run confirmed
  (`res_cached.texts == res_uncached.texts` on the 3-prompt smoke).
  Each cache entry is a per-layer `list` of `KVCache` objects (24 layers for
  the 0.5B model).
- **`BatchGenerator`** confirmed:
  `__init__(model, *, max_tokens=128, stop_tokens=None, sampler=None,
  logits_processors=None, completion_batch_size=32, prefill_batch_size=8,
  prefill_step_size=2048, max_kv_size=None, stream=None)` and
  `insert(prompts, max_tokens=None, caches=None, all_tokens=None,
  samplers=None, logits_processors=None, state_machines=None)` — i.e.
  per-request `max_tokens` and cache injection exist as claimed. (Note:
  per-request `stop` is at the generator level via `stop_tokens`, not per
  `insert`; `inference_local(stop=...)` should therefore be batch-uniform per
  `BatchGenerator` instance or enforced by post-truncation.)
- **`mlx_lm.models.cache`** exposes `BatchKVCache`, `BatchRotatingKVCache`,
  `RotatingKVCache` as claimed.
- **Packaging landmine:** mlx-lm 0.31.3 is **incompatible with
  transformers>=5** (import crashes in `AutoTokenizer.register` —
  `AttributeError: 'str' object has no attribute '__module__'`). The
  `[local]` extra must pin `transformers<5` (or require the mlx-lm version
  that fixes this) or every user's first import fails.
- Tiny-model sanity numbers (0.5B 4-bit, 3 prompts, 16 tokens): load ~30 s
  cold-download / <1 s warm, batch of 3 in ~0.24 s, peak ~1.0 GB. These
  validate the harness only; they say nothing about the gate.
- **Full 3-way harness run completed** on the tiny model (16 rows,
  max_tokens=48, `mlx_lm.server` on localhost, results in
  `benchmarks/results_tiny_qwen05b_3way.json`):

  | mode | wall_s | eff tok/s | ttft p50 | peak GB |
  |---|---|---|---|---|
  | sequential | 5.73 | 82.0 | 0.270 s | 1.21 |
  | in_process (replicated caches) | 5.13 | 99.7 | n/a | 1.61 |
  | server (existing Rust fan-out) | **3.65** | ~290* | n/a | server-side |

  The server path was 0.71x in-process wall-clock — i.e. **faster**, not
  merely within 20%, because `mlx_lm.server` runs its own continuous
  batching under polar_llama's 64-way fan-out while the in-process path
  pays a Python round of tokenize + deepcopy per row. (*server tok/s is
  approximate: completions token-counted client-side, and response lengths
  differ between paths.) A 0.5B model maximally flatters HTTP overhead
  relative to compute, so this does NOT settle the gate — but it is
  directional evidence for buy, and it proves the entire mode-(b) plumbing
  (OPENAI_BASE_URL -> Rust fan-out -> local server) works with zero code
  changes today.

## What must be measured on the real models (open items)

1. **Gemma 4 E4B 4-bit (~5.82 GB resident)** — `--rows 64 --max-tokens 256`,
   all three modes, with `mlx_lm.server` for mode (b). Record the `GATE:`
   ratio, TTFT p50, and peak memory. KV should be cheap (4-of-5 layers
   sliding-window capped at 512 + KV-shared later layers), so
   `completion_batch_size=32` defaults likely hold on 24 GB — verify, don't
   assume.
2. **Ornith-1.0-9B 4-bit (~5.2 GB resident)** — same protocol. **Open
   question:** Ornith is post-trained on Gemma4/Qwen3.5 lineage and its exact
   inherited architecture (Gemma-4-hybrid with sliding+full attention vs
   Qwen-3.5) is unconfirmed. `mlx_lm.load` resolves architecture from
   `config.json`'s `model_type` field to the matching `mlx_lm.models.<type>`
   module (mlx-lm ships both `gemma4`/`gemma4_text` and `qwen3_5`), so the
   check is: download the model card / config and read `model_type` *before*
   assuming KV-cache behaviour. If Gemma-4-hybrid, note that
   `RotatingKVCache.to_quantized` raises `NotImplementedError` and there is
   no `BatchQuantizedKVCache` — quantized-KV memory savings are NOT available
   for the sliding-window layers; budget accordingly.
3. **Hybrid prefix-cache trim risk (mlx-lm #980):** the hybrid-model
   prefix-cache *trim* path was broken and the issue was closed "outdated"
   via a server-side mitigation, with no merged trim fix. The in-process
   replication strategy sidesteps trimming (deepcopy per row, never trim),
   but this must be re-verified on Gemma 4 specifically: run the harness's
   parity check (`--no-prefix-cache` vs default) on E4B and diff completions.
4. **vllm-mlx warm-TTFT claim:** vllm-mlx (waybarrios/vllm-mlx, Apache-2.0)
   advertises paged prefix caching; whether it shows real warm-TTFT gains on
   Gemma 4's hybrid attention needs **one measurement, not an assumption**:
   run mode (b) twice against a vllm-mlx server (cold then warm, same system
   prompt) and compare wall-clock. Note its published 4.3x/5.8x figures were
   measured on an **M4 Max** (~2x this machine's memory bandwidth, per
   arXiv:2601.19139) — treat them as ceilings, not targets, on the M4 Pro.
5. **Server-mode memory:** the harness cannot see the server process's peak
   memory; record it from `mlx_lm.server` logs / Activity Monitor when
   running the real gate, since the build threshold has a memory arm.
6. **Speculative decoding is off the table** for batching: mlx-lm's
   speculative path is single-sequence only, and the "extract nested E2B
   draft from E4B" API does not exist. Do not factor it into either path.

## Non-goals / guardrails (settled)

- No Rust `Provider::LocalMLX` variant. The provider contract
  (`ModelClient`, `src/model_client/mod.rs`) is Rust-only, per-request HTTP;
  a Python MLX loop cannot implement it, and the server path already covers
  the "local model behind the existing fan-out" case with zero Rust changes.
- Do not rebuild continuous batching, batched KV, or left-padded caches —
  mlx-lm ships all of them (`BatchGenerator`, `BatchKVCache`,
  `BatchRotatingKVCache`, `_make_cache`); the build path is glue, not an
  engine.
- All in-process logic stays CI-testable on Linux CPU via the `LocalEngine`
  protocol + `FakeEngine`; mlx imports stay guarded inside
  `polar_llama/local/` and never run at import time on CI.
- Per-row failures return error JSON mirroring `create_error_response`
  (`{"_error": ..., "_details": ...}`), never a batch abort — parity with
  the remote providers' contract.

## How to run the gate

```bash
# 1. server for mode (b) (separate terminal):
uv run --with mlx-lm --with 'transformers<5' \
    python -m mlx_lm server --model mlx-community/gemma-4-E4B-it-4bit --port 8080

# 2. the 3-way gate:
uv run --with mlx-lm --with 'transformers<5' \
    python benchmarks/local_mlx_gate.py \
    --model mlx-community/gemma-4-E4B-it-4bit \
    --rows 64 --max-tokens 256 \
    --base-url http://127.0.0.1:8080 \
    --json gate_e4b.json
```

The `GATE:` line in the output applies the decision rule directly.

## References

- Prompt Cache: Gim et al., arXiv:2311.04934.
- vllm-mlx measurements (M4 Max; "MLX leads llama.cpp 20–87%"):
  arXiv:2601.19139.
- mlx-lm hybrid prefix-cache trim: mlx-lm issue #980 (closed "outdated",
  server-side mitigation only).
- mlxcel: lablup/mlxcel.
