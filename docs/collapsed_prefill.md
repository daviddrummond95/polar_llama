# Collapsed shared-prefix prefill for the in-process MLX backend

**TL;DR** On the gate workload (32 rows sharing one ~5 KB system prompt,
gemma-3n E4B 4-bit, M4 Pro), the "replicated prompt_caches" batched path was
not collapsing the shared-prompt prefill at all — it re-prefilled the full
~1,295-token prompt for every row. `polar_llama.local.collapsed_prefill`
computes the shared token prefix **once** and batches only the per-row
suffixes on top, dropping the column from **56.6 s → 5.5 s (10.4× vs
sequential, 7.9× vs the naive batched path)** with **exact greedy token
parity on every row** (32/32 gemma-3n, 8/8 Qwen control). No mlx-lm
monkeypatch is needed; the mechanism composes public mlx-lm 0.31.x APIs.

## Root cause: why the old path did not collapse

`benchmarks/local_mlx_gate.py::run_in_process` warmed a cache from the
system prompt and passed it (deep-copied per row) to `batch_generate`
together with the **full** per-row prompts:

```python
sys_prefix = tokenizer.apply_chat_template([system], tokenize=True)
warm  = batch_generate(model, tok, [sys_prefix], max_tokens=1, return_prompt_caches=True)
caches = [deepcopy(warm.caches[0]) for _ in rows]
batch_generate(model, tok, chats, prompt_caches=caches)   # chats = FULL prompts
```

Three independent defects, all verified against mlx-lm 0.31.3 source:

1. **The warm prefix was ~empty.** Gemma's chat template has no system
   role — it folds the system prompt into the *first user turn*. Rendering
   `[{"role": "system", ...}]` alone therefore produces just `[<bos>]`:
   **one token**. The "shared system-prompt cache" that was replicated 32
   times contained essentially nothing, which is why `in_process` only
   reached 1.31× (43.4 s): every row still prefilled all ~1,295 prompt
   tokens. (For templates that do render a system section, a text-rendered
   prefix still isn't guaranteed to be a *token-level* prefix of the full
   prompt — BPE can merge across the seam; see
   `polar_llama/local/prefix_cache.py::verify_token_boundary`.)

2. **`batch_generate` never skips cached tokens.** In
   `BatchGenerator.insert → insert_segments → PromptProcessingBatch.prompt`,
   every prompt token passed in is prefilled *on top of* the supplied
   cache's existing offset. `prompt_caches` means "KV history to continue
   from", not "prefix to deduplicate against". Passing full prompts with a
   warm prefix cache is double work by construction — the caller must pass
   only the suffix.

3. **The warm cache was contaminated.** With `max_tokens=1`,
   `GenerationBatch.__init__` steps the model on the last prompt token and
   the first `next()` steps it again on the sampled token before
   `finish_reason` fires, so the extracted cache holds
   `prefix + 1 generated token` (offset P+1). Prefilling the full prompt on
   top yields the context `[<bos>, t_gen, <bos>, …full prompt…]` at shifted
   RoPE positions — silently wrong context that still produces
   plausible-looking output. The gate benchmark only checked row *counts*,
   not content, so this went unnoticed.

## The fix (`polar_llama/local/collapsed_prefill.py`)

Follows the `prefix_cache.py` invariants (immutable-prefix-first,
token-boundary safety, no-trim), but enforces the token boundary **by
construction** instead of by post-hoc check:

1. Tokenize every row's **full** prompt (chat template applied to
   system + user).
2. The shared prefix is the **token-level longest common prefix (LCP)** of
   those token lists (`common_token_prefix_len` / `plan_collapsed_prefill`).
   Nothing is re-tokenized, so there is no BPE seam to verify, and the
   definition is template-agnostic — for gemma the LCP swallows `<bos>`,
   the folded system prompt, *and* the common head of the user turn
   (1,258 of ~1,295 tokens on the gate workload). The prefix is clamped to
   `min(row length) − 1` so every row keeps a non-empty suffix.
3. `prefill_prompt_cache` computes the prefix KV **once** at batch size 1:
   `mlx_lm.models.cache.make_prompt_cache(model)` + chunked `model()` calls
   — the exact shape of upstream sequential prefill, except *all* prefix
   tokens go into the cache (the next token comes from the suffix, not from
   sampling). This avoids defect 3 entirely.
4. `collapsed_batch_generate` hands `batch_generate` the per-row
   **suffixes** with the prefix cache attached per row. mlx-lm natively
   supports "batching with history": `KVCache.merge` /
   `RotatingKVCache.merge` combine per-sequence caches into
   `BatchKVCache` / `BatchRotatingKVCache` with per-row true offsets, and
   ragged suffixes are right-padded during prefill, then rolled into left
   padding by `cache.finalize()` (`dynamic_roll`), so per-row positions and
   masks line up without any caller-side padding logic.

Two deliberate design points:

- **Zero-copy replication.** `Batch*KVCache.merge` copies each
  per-sequence cache's KV into freshly allocated batch arrays and never
  mutates its inputs, so the *same* warm cache object is attached to every
  row (`replicate_prefix_caches(..., clone_per_row=False)`), avoiding B
  deep copies. (Physical per-lane replication inside `merge` is inherent
  to batched attention; what collapses is the prefix *compute*.)
- **No-trim invariant preserved.** Reuse is strictly EXACT/EXTEND — the
  cached prefix is by construction a true token prefix of every row — so
  the broken hybrid-cache trim path (mlx-lm #980) is never reachable.

The hybrid gemma-3n batched path additionally requires the #1384 runtime
patch (`polar_llama/local/_mlx_patches.py`) exactly as before; the collapse
mechanism itself is orthogonal to it.

## Measured results (M4 Pro 24 GB, mlx-lm 0.31.3, greedy)

`benchmarks/validate_collapsed_prefill.py --rows 32 --max-tokens 128`
(full JSON: `benchmarks/results_collapsed_prefill.json`).

**gemma-3n E4B 4-bit (hybrid), 32 rows × ~5 KB shared system prompt,
max_tokens=128** — LCP 1,258 of ~1,295 tokens/row, suffixes 37–39 tokens:

| mode            | wall s | eff tok/s | prompt tokens prefetched | peak GB |
|-----------------|-------:|----------:|-------------------------:|--------:|
| sequential      |  56.57 |      11.2 |                  ~41,468 |    5.00 |
| naive batched   |  43.34 |      15.3 |          41,436 measured |    7.68 |
| **collapsed**   | **5.46** | **121.6** |       **2,438 measured** | 6.00 |

- **10.36× vs sequential, 7.93× vs naive batched**; measured prefill
  collapse 17.0× (prefix prefilled once in 1.33 s).
- Parity: **32/32 rows exact token MATCH** vs the sequential greedy
  reference (0 near-ties, 0 mismatches, tie-eps 0.1 nats).
- Reproducibility: `collapsed_batch_generate` text == the instrumented
  parity-drive text on 32/32 rows.
- Notably the *naive batched* path agreed with sequential on only 29/32
  rows (floating-point near-tie argmax flips in batched kernels — the
  known-benign behaviour from the #1384 validation). The collapsed path
  matched 32/32: the prefix KV is computed with the same batch-1 kernels
  as the sequential path, so it is bit-identical, and only the ~38-token
  suffixes run batched.

**Qwen2.5-0.5B-Instruct-4bit (full-attention control), 8 rows,
max_tokens=64**: 8/8 exact MATCH; 2.76 s → 0.68 s (**4.05×** vs
sequential, 3.02× vs naive batched), prefill 10,713 → 1,592 tokens.

## Correctness argument

- **Positions.** The prefix cache holds exactly the LCP tokens at positions
  `[0, L)` (offset asserted after warm prefill). `merge` carries per-row
  true offsets, so suffix token `j` of row `i` is roped at position
  `L + j` — identical to a cold prefill of the full prompt. The
  right-pad → `finalize()` roll is mlx-lm's own ragged-batch mechanism, the
  same one the naive batched path uses.
- **Token identity.** Suffixes are literal slices `full[L:]` of the same
  tokenization the sequential reference consumes; `prefix + suffix ==
  full` is asserted row-by-row in the unit tests and holds by slicing.
- **Empirically:** greedy token streams are identical to sequential on
  every row of both a hybrid (sliding-window + shared-KV) model and a
  full-attention control, under the same near-tie-tolerant gate used for
  the #1384 fix — and in fact stricter (zero near-ties needed).

## Upstream-worthiness

Two observations generalize beyond polar_llama:

1. `batch_generate`'s `prompt_caches` docs don't say that prompt tokens are
   prefilled on top of the cache (i.e. callers must pass suffixes). An
   upstream docs clarification — or a convenience that takes full prompts
   plus a shared prefix cache and slices internally — would prevent this
   whole failure class.
2. A `batch_generate`-level "shared prefix" option (tokenize, LCP, warm
   once, batch suffixes — what `collapsed_batch_generate` does) is small,
   uses only existing public machinery (`merge`/`prepare`/`finalize`), and
   gave 7.9× over the naive batched path here. Worth proposing upstream.

Also worth noting upstream: a prompt cache returned by
`batch_generate(..., max_tokens=1, return_prompt_caches=True)` contains the
first *generated* token's KV in addition to the prompt (defect 3 above), so
it is not a clean prompt-only cache; `make_prompt_cache` + manual prefill is
the reliable way to warm a prefix.

## Files

- `polar_llama/local/collapsed_prefill.py` — mechanism (CI-safe import; mlx
  only inside functions).
- `benchmarks/validate_collapsed_prefill.py` — parity gate + benchmark
  (exit code reflects parity; speedups only reported alongside a passing
  gate).
- `benchmarks/results_collapsed_prefill.json` — the measured run above.
- `tests/test_collapsed_prefill.py` — CI-safe unit tests for the
  planner/replication logic (no mlx).
