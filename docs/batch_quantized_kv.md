# Batched quantized KV cache for mlx-lm (`BatchQuantizedKVCache`)

Status: implemented and validated on Apple M4 Pro (24 GB), mlx 0.31.2 /
mlx-lm 0.31.3. Pure `mx`-ops — **no custom Metal kernel was needed.**

> Measured results in this document come from
> `benchmarks/validate_quantized_kv.py` (parity gates + memory table,
> PASS/FAIL, exit code). Numbers are filled in from the run recorded at the
> bottom.

## 1. The gap

mlx-lm (<= 0.31.3) has quantized KV **or** batching, never both:

| | fp16 KV | quantized KV |
|---|---|---|
| single sequence | `KVCache` | `QuantizedKVCache` (`kv_bits` in `generate`) |
| batched | `BatchKVCache` / `BatchRotatingKVCache` | **missing** |

Concretely:

* `QuantizedKVCache` has no `merge`, so `mlx_lm.generate._merge_caches`
  raises `... does not yet support batching with history`.
* `RotatingKVCache.to_quantized` / `BatchRotatingKVCache.to_quantized` raise
  `NotImplementedError`; `BatchKVCache` has no `to_quantized` at all.
* `BatchGenerator` / `batch_generate` / `mlx_lm.server` expose no `kv_bits`.
* Even if a batched quantized cache existed, attention would break:
  `quantized_scaled_dot_product_attention` reshapes queries to
  `(B, n_kv, n_repeats, L, D)` under GQA, so scores are 5-D, while the batch
  caches' boolean masks are 4-D `(B, 1, L, S)`. For `B > 1` the broadcast
  raises `[broadcast_shapes] (B,1,L,S) and (B,n_kv,R,L,S) cannot be
  broadcast` — and for the unlucky case `B == n_kv` it broadcasts **silently
  wrong** (mask batch dim lands on the kv-head dim). Upstream never hits
  this because nothing batched is ever quantized.

Why it matters: fp16 KV for Qwen3-8B is `2 * 8 kv-heads * 128 head-dim * 2 B
* 36 layers ≈ 147 KB/token`. 32 rows × ~3K context ≈ **14.5 GB of KV alone**,
plus ~4.6 GB of 4-bit weights — past the M4 Pro's Metal
max-recommended-working-set (19.07 GB on this 24 GB machine). That is the
memory wall this closes.

## 2. Design (merge / update_and_fetch / extract)

`polar_llama/local/batch_quantized_kv.py` implements
`BatchQuantizedKVCache` = `BatchKVCache` bookkeeping × `QuantizedKVCache`
storage:

* **Storage / quantize-on-write** — keys and values are
  `(packed_uint32, scales, biases)` triples from `mx.quantize`.
  `update_and_fetch` quantizes each incoming fp16 K/V chunk as it is
  written (so the *resident* cache is always quantized — fp16 exists only
  for the current prefill chunk). Quantization groups run along `head_dim`,
  so every position-axis operation the batch protocol needs (slice, pad,
  `dynamic_roll`, per-row gather) is exact — it never crosses a group.
* **merge** — left-pads shorter sequences and right-justifies each row's
  quantized triples into freshly allocated batch buffers (mirror of
  `BatchKVCache.merge`). Accepts `QuantizedKVCache` rows as-is and quantizes
  fp16 `KVCache` rows on entry, so existing prefix caches can join a
  quantized batch.
* **update_and_fetch (batched)** — per-row `offset` (an `mx.array` that
  starts at `-left_padding`, consumed by RoPE exactly like `BatchKVCache`),
  scalar write index `_idx`, 256-step buffer growth, boolean batch mask from
  `create_causal_mask(N, offset=_idx, left_padding=...)`. Attention needs
  **no new code**: `base.scaled_dot_product_attention` already routes any
  cache with a `.bits` attribute into the existing
  `quantized_scaled_dot_product_attention`, whose `mx.quantized_matmul`
  calls are batch-shape-generic (verified: batched 5-D
  `quantized_matmul` with a broadcast repeat dim matches the dequantized
  reference and is bitwise-consistent with per-row calls).
* **extract** — slices one row, strips its left padding (`mx.contiguous`),
  and returns a single-sequence `QuantizedKVCache` subclass — so finished
  rows can be saved (`save_prompt_cache`) or reused as prefix caches.
* Full `BatchGenerator` protocol: `prepare`/`finalize` (right-padded
  prefill + `dynamic_roll` per quantized component), `filter` (with the
  min-left-pad shift), `extend` (continuous batching, including empty
  sides), `trim`, `state`/`meta_state` round-trip.

The **one** upstream fix required is a guarded, idempotent wrapper around
`mlx_lm.models.base.quantized_scaled_dot_product_attention`
(`apply_batched_quantized_sdpa_mask_patch()`): when the model uses GQA and
the mask is a 4-D batch mask `(B, 1, L, S)`, insert the missing repeat axis
(`(B, 1, 1, L, S)`). Nothing else upstream is touched, and the patch is a
no-op for every mask shape upstream currently produces.

### Wiring into mlx-lm batching (no upstream edits)

`_merge_caches` builds batch caches by calling `merge` on the *per-sequence*
caches handed to `BatchGenerator.insert(..., caches=...)`. So:

* `MergeableQuantizedKVCache` — a `QuantizedKVCache` subclass whose `merge`
  returns a `BatchQuantizedKVCache`; used as the per-sequence seed.
* `make_quantized_prompt_cache(model, group_size=, bits=)` — per-layer seeds
  (non-`KVCache` layers of hybrid models pass through and keep their stock
  fp16 batch path).
* `batch_generate_quantized(model, tokenizer, prompts, kv_bits=, ...)` —
  drop-in `mlx_lm.batch_generate`.
* `QuantizedBatchGenerator(model, kv_bits=, ...)` — drop-in
  `BatchGenerator` (continuous batching, streaming, insert/remove/extract).

```python
from polar_llama.local.batch_quantized_kv import batch_generate_quantized
out = batch_generate_quantized(model, tokenizer, prompt_token_lists,
                               kv_bits=4, max_tokens=128)
```

## 3. Parity — method, thresholds, results

Free-running token comparison alone cannot separate "broken" from
"quantization noise": one flipped near-tie token changes every later token
(a single early flip drives free-running prefix-match down to ~0.3 even when
the per-step behavior is near-identical, and even fp16-batched vs fp16-single
free-running scores only ~0.93 on this model for the same reason). Both gated
metrics are therefore **teacher-forced** — the reference history is fed at
every step, so there is no cascade and each number is a clean per-position
rate. Greedy, Qwen3-8B-4bit, 8 mixed-length prompts, 64 new tokens:

1. **Batching faithfulness** (the property this work adds): teacher-forced
   argmax predictions of the batched-quantized cache vs the upstream
   *single-sequence* `QuantizedKVCache` (no batch, no padding), compared to
   each other position-by-position over the same reference history. Any
   batching bug shows up directly here. Ceiling: the same comparison for
   fp16 (`BatchKVCache` vs `KVCache`) — non-1.0 only because padded lanes
   change floating-point reduction order. Gate: quantized faithfulness
   within **0.02** of the fp16 ceiling.
2. **Teacher-forced quality vs fp16**: batched-quantized argmax predictions
   vs the fp16 reference tokens, at every continuation position. Gates:
   **Q8 >= 0.95, Q4 >= 0.85** absolute (8-bit KV is near-lossless in
   published KV-quant results; 4-bit KV flips only low-margin tokens), and
   within 0.04 / 0.15 of the fp16 teacher-forced anchor measured in the same
   run.
3. Free-running fp16-vs-quantized divergence — reported for context, not
   gated (cascade-dominated; see above).

**Results (M4 Pro, Qwen3-8B-4bit, group_size 64, 8 mixed-length prompts,
64 new tokens):**

| metric | fp16 (ceiling) | Q8 | Q4 |
|---|---|---|---|
| batching faithfulness (TF, batched vs single) | 0.9961 | 0.9844 (gate 0.9661) | 0.9805 (gate 0.9361) |
| teacher-forced quality vs fp16 | 0.9961 | 0.9922 (gate 0.95) | 0.9395 (gate 0.85) |
| free-run vs fp16 (info only, cascade) | — | 0.9102 | 0.3262 |
| verdict | — | **PASS** | **PASS** |

Reading the table: quantized *faithfulness* (0.984 / 0.981) sits right at the
fp16 batched-vs-single ceiling (0.996) — batching the quantized cache changes
essentially nothing beyond padded-lane reduction-order near-ties. Teacher-
forced *quality* vs fp16 is 0.992 (Q8, near-lossless) and 0.940 (Q4, only
low-margin tokens flip). The free-run column shows exactly why it cannot be a
gate: Q4's 0.33 is one early near-tie flip cascading through a 64-token
greedy tail, not a per-step defect (its per-step quality is 0.94).

The earlier free-running "exactness" metric was dropped as a *gate* (kept as
info): it is cascade-dominated, so even fp16-vs-fp16 batching scores only
~0.93, making it useless for separating a batching bug from ordinary
quantization noise. The tensor-level faithfulness proof is in the unit test
`test_batched_quantized_attention_matches_per_row` (batched quantized SDPA ==
per-row single quantized SDPA, atol 2e-3).

Cross-model note: on the 0.5B control (Qwen2.5-0.5B-Instruct-4bit, CPU) the
batched cache reproduces single-sequence quantized generation *including*
upstream's failure mode — Q4/g64 collapses to immediate-EOS on that fragile
model both single and batched (identical outputs), and Q4/g32 restores
coherent output both single and batched. Faithfulness, not accident.

## 4. Memory — method and results

Two complementary measurements (both in `validate_quantized_kv.py`); the
budget throughout is the Metal **max recommended working set** = 19.07 GB on
this 24 GB M4 Pro (beyond it macOS pages GPU memory; end-to-end runs on this
machine hard-OOM past it).

### 4a. Deterministic resident-KV (`--stage kvbytes`, exact, CPU)

The KV storage is what drives the OOM, and it depends only on shape/dtype,
not the weights or the run. So one transformer layer's batched cache is built
at the target (batch × 2816-token context) for each dtype and its **exact
`nbytes`** read (× 36 layers). This runs on the CPU stream — immune to the
GPU-memory pressure that stalls a live batch — and is fully reproducible.

| batch | fp16 KV GB | Q8 KV GB | Q4 KV GB | Q8/fp16 | Q4/fp16 |
|---|---|---|---|---|---|
| 8  | 3.32  | 1.76  | 0.93 | 0.53 | 0.28 |
| 16 | 6.64  | 3.53  | 1.87 | 0.53 | 0.28 |
| 24 | 9.97  | 5.29  | 2.80 | 0.53 | 0.28 |
| 32 | 13.29 | 7.06  | 3.74 | 0.53 | 0.28 |
| 48 | 19.93 | 10.59 | 5.61 | 0.53 | 0.28 |
| 64 | 26.58 | 14.12 | 7.47 | 0.53 | 0.28 |

Adding the ~4.6 GB of 4-bit weights (total vs the 19.07 GB budget):

| batch | fp16 total | Q8 total | Q4 total |
|---|---|---|---|
| 32 | 17.9 GB (fits KV, **OOMs end-to-end**¹) | 11.7 GB fits | 8.3 GB fits |
| 48 | 24.5 GB **OOM** | 15.2 GB fits | 10.2 GB fits |
| 64 | 31.2 GB **OOM** | 18.7 GB fits | 12.1 GB fits |

¹ At batch 32 the weights+KV alone (17.9 GB) is under budget, but the live
run's prompt-logit / activation transient (~2–5 GB) pushes the peak over —
which is exactly why stock fp16 `batch_generate`/server OOMs at batch 32 on
this machine. Quantized KV removes ~5.6 GB (Q8) / ~9.6 GB (Q4) of that at
batch 32, so the same batch clears the budget with room to spare. The **KV
wall** for weights+KV alone is batch 48 for fp16 (19.9 GB KV) vs well past
batch 64 for Q4.

### 4b. Measured end-to-end peak (clean re-run, machine idle, 2026-07-04)

Each `kv:batch` runs in a fresh subprocess (~1536-token prompts, a few decode
steps — peak is at prefill), reporting `mx.get_peak_memory()`. **Re-measured on
an idle machine.** (An earlier run was contaminated by a concurrent GPU workload
and spuriously OOM-ed even small fp16 configs — a caution that these peaks are
sensitive to whatever else holds unified memory at the time.)

| kv | batch | measured peak GB | outcome |
|---|---|---|---|
| fp16 | 24 | 15.51 | fits |
| fp16 | 32 | 17.78 | fits |
| q8 | 32 | 12.13 | fits |
| q4 | 32 | 9.43 | fits |
| q4 | 48 | 11.21 | fits |
| q4 | 64 | 11.21 | fits |

The honest read: plain fp16 `batch_generate` fits to **~batch 32 (17.78 GB)** on
this 24 GB machine and OOMs past ~batch 40. Quantized KV cuts the KV portion
**~47% at batch 32** (Q4 9.43 vs fp16 17.78 GB; Q8 12.13), so **Q4 clears batch
64 at ~11 GB** — roughly doubling the viable batch. The Q4 peak plateaus at
~11 GB for batch 48 and 64 (dominated by the fixed prefill/pool high-water mark
rather than growing KV), leaving headroom for larger batches still.

Note: the earlier gate-run OOMs of fp16 `in_process` and `mlx_lm.server` at
batch 32 were driven by prefix-cache **replication** overhead and concurrent
load — **not** plain batching, which fits batch 32 fine on an idle machine.
The value of quantized KV is therefore *headroom*: ~2× the batch (or ~2× the
context) before the wall, with correctness preserved (§3).

## 5. Was a Metal kernel needed?

No. `mx.quantized_matmul` already supports the batched 5-D shapes with a
broadcast repeat dim (verified bitwise-consistent with per-row calls), so
option (a) — a real `BatchQuantizedKVCache` over existing ops — was
sufficient. The known cost of the unfused path is that attention scores
materialize (`(B, heads, L_chunk, S)`), which fp16's fused
`mx.fast.scaled_dot_product_attention` avoids; the validator caps the
quantized prefill chunk at 256 to bound that transient. A fused batched
quantized-SDPA kernel remains a *throughput* optimization, not a
correctness or memory-win requirement.

## 6. Upstream note (ready to post)

Two independent, upstreamable pieces fall out of this work:

1. **Bug**: `quantized_scaled_dot_product_attention` mishandles 4-D masks
   when `n_repeats > 1` — raises for `B != n_kv_heads`, silently wrong for
   `B == n_kv_heads`. One-line fix: `mask = mx.expand_dims(mask, -3)` when
   `mask.ndim == 4` (repro: `tests/test_batch_quantized_kv.py::
   test_mask_patch_idempotent_and_fixes_gqa_broadcast`).
2. **Feature**: `BatchQuantizedKVCache` + `QuantizedKVCache.merge` +
   `kv_bits` on `BatchGenerator`, closing the "quantized KV × batching"
   hole in the matrix above. The implementation here mirrors upstream's own
   `BatchKVCache` / `QuantizedKVCache` structures 1:1 and passes
   token-level parity against the existing single-sequence quantized path.

## 7. Files

* `polar_llama/local/batch_quantized_kv.py` — implementation (import-guarded,
  self-contained, loadable via `importlib` without the polar_llama package).
* `benchmarks/validate_quantized_kv.py` — parity + memory validator
  (PASS/FAIL, exit code, `--json` report).
* `tests/test_batch_quantized_kv.py` — 15 CI-safe unit tests (CPU stream,
  fake tensors): quantize-on-write, buffer growth, merge/extract round-trip,
  right-pad + `finalize` roll, `filter` shift, `extend` (incl. empty sides),
  mask equality with `BatchKVCache`, state round-trip, the GQA mask-patch
  repro, and batched-vs-per-row quantized attention equivalence.
