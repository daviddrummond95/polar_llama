# Upstream-readiness review — mlx-lm #1384 Gemma 3n batched fix

**Reviewer:** Opus (moderate upstream-readiness pass, read/reason only; no models run)
**Date:** 2026-07-04
**Patch:** `patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch`
**Target:** `mlx_lm/models/gemma3n.py` (stock mlx-lm 0.31.3 — latest published)
**Verdict:** **MERGE-READY** (patch logic). Two minor, non-blocking wording/cosmetic suggestions for the PR description below.

---

## 1. Does it apply? — YES, cleanly

Verified against a copy of the pristine 0.31.3 `gemma3n.py` (confirmed byte-identical to the venv's `.pristine` backup) in an isolated scratchpad tree. The venv site-packages was never modified.

- `patch -p1 --dry-run` → exit 0, `patching file 'mlx_lm/models/gemma3n.py'`
- `git apply --check -p1` → OK
- Real `patch -p1 --verbose`: all **8 hunks succeeded at their nominal line numbers** (120, 129, 163, 334, 343, 376, 501, 515) — **zero fuzz, zero offset**.
- `python -m py_compile` on the applied file → OK.
- The applied result is **line-for-line identical** (in all three changed methods) to `polar_llama/local/_mlx_patches.py`, i.e. the exact code the human operator already validated PASS against pristine. So the diff maintainers would merge produces the validated behavior.

The diff is well-formed unified format with correct `a/` `b/` prefixes and accurate hunk headers.

## 2. Is it minimal / in-style? — YES

- Only `gemma3n.py` is touched. No unrelated edits, no reformatting churn, no import changes, no touched base classes.
- Every added line is load-bearing for the fix: two new `__call__` params (`shared_kv`, `offset`) on `Gemma3nAttention` and `Gemma3nDecoderLayer`, the `mx.array(cache.offset)` snapshot, the 3-tuple returns, and the `intermediates` threading in `LanguageModel.__call__`.
- Style matches mlx-lm conventions (type hints, comment idiom, `mx.array(...)` snapshot). Comments are a touch verbose (they cite issue #1384 and explain the aliasing), but informative — a maintainer may trim, not block.

## 3. Faithfulness to `gemma4_text.py` — YES, this is the decisive merge argument

The patch is a faithful port of the KV-sharing design already merged upstream in `gemma4_text.py`:
- `offset = mx.array(cache.offset) if cache is not None else 0` — identical snapshot idiom (gemma4_text line 256).
- Attention returns `self.o_proj(output), (keys, values), offset`; decoder returns `h, shared_kv, offset` — identical 3-tuple contract (gemma4_text lines 276 / 389).
- `intermediates = [(None, None)] * len(self.layers)`, populate per concrete layer, hand to shared layers — identical mechanism (gemma4_text lines 552–573).

The only structural divergence is idiomatic to gemma3n's existing code: gemma3n reuses its pre-existing `layer_idx_to_cache_idx` map plus a runtime guard (`if i >= first_kv_shared_layer_idx and c is not None`) to select the source layer's intermediates, whereas gemma4_text precomputes a `previous_kvs` list in `__init__` and reads it unconditionally. Both are correct; gemma3n's choice avoids adding new `__init__` state and reuses the map it already maintains. Not a blocker.

Because gemma4_text (the accepted upstream design) uses exactly the pre-update snapshot for shared-layer query RoPE, the pristine gemma3n behavior it replaces (post-update `cache.offset`) is the anomaly, not the patch.

## 4. Root cause — CONFIRMED

Both diagnosed bugs hold on inspection of pristine lines 130–160:
1. **Crash:** shared branch does `keys, values = cache.state` (line 133) — a 2-tuple for `KVCache`/`RotatingKVCache`, a 4-tuple `(keys, values, offset, left_padding)` for the batch caches → unpack error. Fixed by threading `shared_kv` instead of reading `state`.
2. **The real killer:** concrete branch captures `offset = cache.offset` (line 138), ropes keys at that offset, then `cache.update_and_fetch` advances the offset, then ropes queries with the same `offset` variable (line 152). For batch caches `cache.offset` is a mutable `mx.array` and the `+=` mutates the aliased object in place → queries roped at `p+L` vs keys at `p`, on **every** layer → garbage. Sequential `int` offset is copied by value, hiding it. Fixed by `mx.array(cache.offset)` snapshot. This matches the writeup and the `a = mx.array([0]); b = a; a += 21` demonstration.

## 5. Sequential behavior-change risk — IMPORTANT NUANCE (not a defect, but the PR wording needs care)

The fix **does** change one thing in the sequential (non-batched) path, and a maintainer will want this stated precisely:

- Pristine shared-KV layers roped their **queries** with the *post-update* `cache.offset` (= `p + L`, because the concrete source layer already advanced that shared cache earlier in the loop). The current tokens are actually at positions `p .. p+L-1`, so this was a latent `+L` position error in the shared (top) layers — present in sequential too, just never crashed. The patch ropes shared-layer queries at the source layer's **pre-update** offset `p` (the correct position), matching gemma4_text.
- Consequence: shared-layer sequential **logits are not bit-identical** to pristine. The writeup's own bonus note (docs §"The fix", pt 2) is candid about this; but the summary bullet ("Sequential unchanged: greedy sequential outputs are **byte-identical**") and the PR-description bullet ("sequential outputs unchanged vs main (byte-identical)") are slightly stronger than what was measured. What was empirically shown is: **greedy (temp=0) token outputs are unchanged token-for-token** on the tested prompts. Under sampling, or on other prompts, the shifted shared-layer logits could in principle diverge from pristine.
- The concrete-layer change (`int` → `mx.array(int)` scalar offset into `rope`) is genuinely output-neutral (same positions; gemma4_text already relies on this).

**Recommendation (non-blocking):** reword the PR bullet from "sequential outputs byte-identical to main" to "greedy sequential token outputs unchanged on the tested prompts (the fix additionally corrects a latent `+L` RoPE-position error in the shared layers, so shared-layer logits shift slightly)." Framed this way it is a *strengthening* argument for merge (it fixes a second, quieter bug), not a regression.

## 6. Red-flag / edge-case sweep — CLEAN

- **Shared base classes:** none touched. The changed return signatures of `Gemma3nAttention.__call__` / `Gemma3nDecoderLayer.__call__` are model-internal; the only callers are within `gemma3n.py` (all updated). Generic mlx-lm machinery calls `Model.__call__`, never a decoder layer directly. Safe.
- **Per-layer state / memory:** `intermediates` holds references to each concrete layer's post-`update_and_fetch` `(keys, values)` — tensors the cache already owns — for the duration of one forward, then drops them. No extra peak memory; identical pattern to gemma4_text.
- **`num_kv_shared_layers = 0` / non-shared models:** `first_kv_shared_layer_idx == num_hidden_layers`, so the `i >= first_kv_shared_layer_idx` guard is never true; every layer computes fresh KV; `intermediates` is written but never read for sharing. Correct. (The unconditional `shared_full_idx`/`shared_sliding_idx` computation in `__init__` predates this patch and is unchanged.)
- **`cache is None` (e.g. cacheless forward / perplexity):** loop passes `shared_kv=None, offset=None`; attention's else branch overwrites `offset = 0` and recomputes KV. This exactly preserves pristine cacheless behavior — the patch is a strict no-op on that path. (Note: gemma4_text *does* share KV even cacheless; gemma3n stays conservative and matches its own prior behavior. Fine.)
- **KV content for shared layers is unchanged from pristine:** pristine read `cache.state` of `cache[shared_full/sliding_idx]`; the patch reads that same concrete layer's post-`update_and_fetch` `(keys, values)`, which *is* that cache's state. Only the offset handling differs. So the fix is surgical.
- **`offset` consumers:** `offset` feeds only `self.rope(...)`. Masks are built separately from `cache` in `LanguageModel.__call__`; `sdpa` receives `cache=`, not `offset`. No downstream code expects an `int` offset. Safe with `mx.array` offsets (batch and sequential alike).

## 7. Minor cosmetic nits (optional, non-blocking)

- Variable naming: gemma3n uses `kv` (singular) in `h, kv, offset = layer(...)` / `intermediates[i] = (kv, offset)`, whereas gemma4_text uses `kvs`. Purely cosmetic cross-file inconsistency.
- Comments referencing `#1384` are fine for a PR but a maintainer may prefer them trimmed to one line.

---

## Manifest

- **Verdict:** MERGE-READY (patch logic requires no changes).
- **Applies cleanly:** YES — `patch -p1` and `git apply --check -p1` both clean; 8/8 hunks, no fuzz/offset; applied file byte-matches the validated monkeypatch; py_compile OK. Venv left pristine.
- **Top 3 review points:**
  1. Faithful, minimal port of the already-merged `gemma4_text.py` KV-sharing design (snapshot offset + thread per-layer intermediates); root cause (state-unpack crash + `mx.array` offset aliasing) confirmed on pristine source.
  2. The "sequential byte-identical" claim is slightly overstated — the fix also corrects a latent `+L` shared-layer query-RoPE error, so shared-layer sequential *logits* shift (greedy *tokens* were unchanged in testing). Reword the PR bullet accordingly; it reads as a second fix, not a regression.
  3. Edge cases clean: no base classes touched, memory-safe intermediates, correct for `num_kv_shared_layers=0` / non-shared / `cache=None`, and general across cache types by construction. Optional cosmetics: `kv`→`kvs` naming, trim `#1384` comments.
