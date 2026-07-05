# mlx-lm #1384: Gemma 3n batched generation fix

Upstream issue: <https://github.com/ml-explore/mlx-lm/issues/1384>
Affected: `mlx-lm <= 0.31.3` (bug still present on `main` as of 2026-07-04)
Model family: Gemma 3n (hybrid architecture with KV-shared top layers), e.g.
`mlx-community/gemma-3n-E4B-it-lm-4bit`.

Symptom: sequential generation (`mlx_lm.generate`) is correct, but **batched**
generation — `mlx_lm.batch_generate`, and `mlx_lm.server` which batches
internally — either crashes (`too many values to unpack (expected 2)`) or,
once the crash is papered over, **silently produces garbage** ("ItItItIt…",
"OkayOkayOkay…").

## TL;DR root cause

There are **two independent bugs**, both in
`mlx_lm/models/gemma3n.py::Gemma3nAttention.__call__`, and both only
triggered by the batched caches (`BatchKVCache` / `BatchRotatingKVCache`):

1. **The loud one (crash).** The KV-shared layers read
   `keys, values = cache.state`. `state` is the *serialization* API: the
   non-batched caches return a 2-tuple, but the batched caches return a
   4-tuple `(keys, values, offset, left_padding)` → unpack crash.

2. **The silent one (garbage), and the actual heart of #1384.** The attention
   ropes **keys before** `cache.update_and_fetch(...)` and **queries after**
   it, both nominally with the same `offset = cache.offset`:

   ```python
   offset = cache.offset            # <-- captures the cache's offset OBJECT
   keys = self.rope(keys, offset=offset)
   keys, values = cache.update_and_fetch(keys, values)   # does: self.offset += L
   ...
   queries = self.rope(queries, offset=offset)           # !!! reads offset AFTER +=
   ```

   For the non-batched caches `cache.offset` is a Python `int` — immutable —
   so the captured local keeps the pre-update value and queries and keys agree.

   For the **batched** caches `cache.offset` is an `mx.array` (one offset per
   sequence), and `update_and_fetch` advances it **with `+=`, which mutates
   the same `mx.array` Python object in place**. MLX operations capture the
   array's current value at *call* time, so:

   - keys are roped with the pre-update offset `p` (correct), but
   - queries are roped with the post-update offset `p + L` (wrong by `L`).

   Every layer therefore computes attention with query positions shifted by
   the chunk length relative to its keys — +21 for a 21-token prompt at
   prefill, +1 at every decode step — on **all** layers, not just the
   KV-shared ones. The model degenerates into token repetition.

   This is easy to verify:

   ```python
   a = mx.array([0]); b = a; a += 21   # b is now array([21]) — same object!
   ```

   Notably, `gemma4_text.py` (the newer sibling architecture, whose batched
   path works) guards against exactly this with an easily-overlooked
   `offset = mx.array(cache.offset)` snapshot.

Both bugs are invisible in sequential mode, which is why the model card
"works" and the issue title blames only the shared-KV layers.

## The fix

`patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch` (applies with `patch -p1`
inside an mlx-lm checkout or site-packages). Two changes, mirroring the design
mlx-lm itself adopted for `gemma4_text.py`:

1. **Snapshot the RoPE offset** in the concrete (non-shared) branch:

   ```python
   offset = mx.array(cache.offset) if cache is not None else 0
   ```

   `mx.array(...)` snapshots the pre-update value, so the queries roped after
   `update_and_fetch` use the same positions as the keys. (In-place `+=` only
   re-points the *cache's* wrapper; the snapshot keeps the original buffer.)

2. **Thread shared KV through the layer loop instead of reading
   `cache.state`.** Each concrete layer returns the exact `(keys, values)` it
   attended over (the return value of `update_and_fetch`) plus the offset it
   used; `LanguageModel.__call__` stores these per layer (`intermediates`) and
   hands them to the KV-shared layers via `layer_idx_to_cache_idx`. The shared
   layers use them directly and never touch `cache.state`. This is
   cache-type-agnostic, so batched caches (per-sequence offsets, left padding,
   rotation) need no special-casing — the alignment work was already done by
   the concrete layer's `update_and_fetch` and by the (shared) attention mask.

   As a bonus this makes the shared layers' query positions *correct*: the old
   code used the post-update `cache.offset` (a `+L` shift baked into the
   sequential path too). Empirically that quirk was benign in sequential mode
   — greedy outputs are unchanged token-for-token after the fix.

### Why this alignment argument is airtight

A KV-shared layer must attend over exactly the same `(keys, values)` tensor
that its source layer attended over, with queries roped at the same positions
its source layer used — the mask for the layer type is shared and already
consistent with that geometry (left padding, per-sequence offsets, sliding
window, rotation). Passing the source layer's actual tensors and offset makes
this true *by construction*, for every cache implementation, instead of trying
to re-derive the geometry from cache internals via `state`.

## Validation (all on Apple Silicon GPU, greedy `temp=0.0`, chat-templated)

`benchmarks/validate_1384_fix.py` — sequential (`stream_generate`) vs batched
(`BatchGenerator`, the `mlx_lm.server` machinery), token-level comparison.

**Before the fix** (pristine 0.31.3): `batch_generate` crashes; with the naive
`cache.state[:2]` band-aid:

```
seq: 'Paris'                    bat: 'Okay.\n\nItItItItItItIt...'
seq: 'The three primary...'     bat: 'OkayOkayOkayOkayOkay...'
```

**After the fix** (`mlx-community/gemma-3n-E4B-it-lm-4bit`, B=4, 32 tokens):

```
[MATCH]    p0 'Paris'
[MATCH]    p1 'The three primary colors are:\n\n*   **Red**\n*   **Yellow**\n*   **Blue**'
[MATCH]    p2 '2 + 2 = 4'
[NEAR-TIE] p3 diverges at token 31/32: seq ' or' vs bat ' and'
           batched logprobs: ' or' = -0.75, ' and' = -0.75  (exact bf16 tie)
PASS
```

3/4 prompts are token-identical over the full generation; the 4th is
identical for 31 tokens and flips its final token on an **exact logprob tie**
(-0.75 vs -0.75 nats in the batched distribution) — expected bf16 kernel
noise for padded batches, not misalignment.

Additional evidence:

- **B=1 batched (padding-free) is token-identical to sequential on 4/4
  prompts** — the batched cache path is exact when no cross-sequence padding
  noise exists.
- **Rotation stress:** 640-token generation (sliding window 512 exceeded,
  `BatchRotatingKVCache` rotates) — batched output is **token-identical to
  sequential for all 640 tokens**.
- **Control (no regression):** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
  (full attention) — 4/4 token-identical, before and after.
- **Sequential unchanged:** greedy sequential outputs are byte-identical to
  pristine mlx-lm on all test prompts.
- **`mlx_lm.server` smoke test:** concurrent chat completions return
  `'Paris\n'` and the correct primary-colors list (garbage before the fix).

Reproduce:

```bash
python benchmarks/validate_1384_fix.py               # patched install
python benchmarks/validate_1384_fix.py --monkeypatch # pristine install + runtime patch
```

## Runtime workaround in polar_llama (no mlx-lm fork needed)

`polar_llama/local/_mlx_patches.py` provides
`apply_gemma3n_batched_shared_kv_patch()`: a guarded, idempotent monkeypatch
that applies the same fix at runtime. It no-ops if mlx-lm is not installed or
if upstream has already fixed the shared-KV path (detected by the
`cache.state` read in the attention source). It is **not** applied
automatically; call it before loading a Gemma 3n model. Validated PASS against
pristine mlx-lm 0.31.3 (see above).

---

## Ready-to-post upstream PR description

> **Fix Gemma 3n batched generation (shared-KV layers + RoPE offset aliasing) — fixes #1384**
>
> Batched generation (`batch_generate`, `mlx_lm.server`) produced crashes or
> garbage for Gemma 3n while sequential generation was fine. Two independent
> bugs in `models/gemma3n.py`, both only triggered by the batch caches:
>
> 1. **Shared-KV layers read `cache.state`**, which returns a 2-tuple for
>    `KVCache`/`RotatingKVCache` but a 4-tuple
>    `(keys, values, offset, left_padding)` for `BatchKVCache`/
>    `BatchRotatingKVCache` → "too many values to unpack".
>
> 2. **Query RoPE offset aliasing.** The attention captures
>    `offset = cache.offset`, ropes keys, calls `update_and_fetch` (which does
>    `self.offset += L` — an *in-place* mutation when `offset` is an
>    `mx.array`, i.e. for the batch caches), and only then ropes the queries
>    with the same `offset` variable. Keys get the pre-update positions,
>    queries get post-update (+L) positions, on **every** layer → garbage.
>    With the non-batched caches `offset` is an immutable `int`, which is why
>    sequential generation never showed this. (`gemma4_text.py` already
>    guards against this with `offset = mx.array(cache.offset)`.)
>
> Fix, mirroring the KV-sharing design of `gemma4_text.py`:
>
> - snapshot the offset in the non-shared branch
>   (`offset = mx.array(cache.offset)`),
> - thread each concrete layer's post-`update_and_fetch` `(keys, values)` and
>   its RoPE offset through the layer loop, and have the KV-shared layers
>   consume those directly instead of reading `cache.state`. This is correct
>   for every cache type by construction (padding/rotation alignment is
>   whatever the source layer's `update_and_fetch` + shared mask produced).
>
> Validation (Apple Silicon, greedy, chat template,
> `mlx-community/gemma-3n-E4B-it-lm-4bit`):
>
> - batched B=4 vs sequential: token-identical on 3/4 prompts over 32 tokens;
>   the 4th flips only its final token on an exact bf16 logprob tie
>   (-0.75 vs -0.75).
> - batched B=1 vs sequential: token-identical 4/4.
> - 640-token generation through sliding-window rotation
>   (`BatchRotatingKVCache`): token-identical to sequential for all 640 tokens.
> - greedy sequential *token* outputs byte-identical to `main` on all test
>   prompts (the offset snapshot also corrects a latent `+L` shared-layer
>   query-RoPE error in sequential mode, shifting shared-layer *logits*
>   slightly — a second, quieter fix; greedy decoding is unaffected).
> - control `Qwen2.5-0.5B-Instruct-4bit`: batched vs sequential token-identical
>   4/4 (no regression).
> - `mlx_lm.server` now returns correct completions for concurrent Gemma 3n
>   requests.
