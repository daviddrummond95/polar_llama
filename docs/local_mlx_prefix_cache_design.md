# Local MLX Backend: Prefix-Cache Layer & Parity Harness

Design notes for `polar_llama/local/prefix_cache.py` and
`polar_llama/local/parity.py`, the correctness-critical layer under
`pl.col("prompt").llama.inference_local(..., engine="in_process")`.

The failure mode of a prefix cache is not a crash — it is **silently wrong
output**: KV reused at the wrong positions produces fluent, plausible,
incorrect generations. Every design decision below exists to make that
failure structurally impossible or loudly detected.

## Where this sits

- `engine="server"` (default) routes the existing Rust async fan-out at a
  local OpenAI-compatible server via the existing `OPENAI_BASE_URL`
  override (`src/model_client/openai.rs`). No caching logic of ours runs;
  the server owns its KV.
- `engine="in_process"` is a pure-Python `map_batches` UDF wrapping
  mlx-lm's **existing** continuous batching (`BatchGenerator`,
  `batch_generate(prompts, prompt_caches=..., return_prompt_caches=...)`,
  left-padded `BatchKVCache`). We do not rebuild batching or KV; this
  layer only decides *when prefix KV may be reused* and *proves the reuse
  is sound*.
- All logic is injected-tokenizer / injected-engine, so it runs in CI
  (ubuntu, no mlx, no GPU) against `FakeTokenizer` / `FakeEngine`.
  `mlx`/`mlx_lm` are imported only inside `require_mlx()` and guarded
  real-run paths (`pip install 'polar-llama[local]'`).

## Invariant 1: token-level prefix, not text-level

RoPE binds each cached K/V entry to an **absolute position**. Reuse is
therefore valid only when the cached tokens are a true prefix of the
tokens the model would see for the new prompt. BPE tokenization is
context-dependent: a merge can span the prefix/suffix seam, so a raw-text
prefix match does **not** imply token identity. Example (real shape,
toy vocab): with `"walking"` in the vocab,

```
encode("walk")          == [walk]
encode("walk" + "ing")  == [walking]      # merge across the seam
```

`verify_token_boundary(tokenizer, prefix, suffix)` implements the exact
check `encode(prefix+suffix)[:len(encode(prefix))] == encode(prefix)` and
must pass **per row** (merges depend on the suffix's leading characters)
before any reuse. The default prompt template ends the prefix at a
turn-boundary newline, which reduces seam merges but is not relied upon —
the check always runs. A failing row falls back to a cold prefill; a
batch never mixes cached and uncached position bases within one arm.

## Invariant 2: no trim, ever

mlx-lm's prefix-cache **trim** path is broken for hybrid
(sliding-window / rotating) caches — mlx-lm issue #980, closed
"outdated" via a server-side mitigation with no merged trim fix. Gemma 4
class models are exactly this hybrid shape (5:1 sliding/full,
`sliding_window=512`, KV-shared later layers), so trimming is the one
operation guaranteed to hit the broken path on the models we care about.

Policy: `classify(cached, new)` returns only `EXACT | EXTEND | DIVERGE`.
The trim-tempting case — cached prefix **longer** than the new prompt —
is `DIVERGE`. `PrefixStore` then either recomputes (`on_longer_cache=
"skip"`, default: never wrong, only slower) or raises
`TrimNotSupportedError` (`"raise"`, for callers that want loud failure).
The safe direction is the only direction: EXACT reuse, or EXTEND a
*shorter* cached prefix (boundary-checked at the extension seam).

## Invariant 3: immutable prefix first, structurally

The public API takes `system=` separately from the per-row prompt.
`assemble_prompt(system, few_shot, user)` renders through two separate
callables — `render_prefix(system, few_shot)` never sees `user`, and
`render_suffix(user)` never sees the shared parts — so per-row content
cannot leak into the cache-key prefix by construction. The full prompt is
always exactly `prefix + suffix`.

`PrefixKey` hashes the **rendered prefix text** (sha256), not tokens:
token ids vary with tokenizer revision and encode flags, while the
rendered text is the stable identity. Correctness never rests on the
hash — `get_exact` compares the full stored text on hit.

## Batch replication and the memory trade-off

`batch_generate(prompt_caches=[...])` takes one cache per row, and decode
mutates each row's cache in place, so a shared prefix must be
**replicated**: `PrefixEntry.replicate_into_batch(n)` returns `n`
independent, row-aligned copies (clone function injected; deepcopy for
fakes, state-level clone for MLX caches).

Cost: on full-attention layers this is `B ×` the prefix KV. Worst case
(9B dense-attention model, long prefix, B=64) that is real memory on a
24 GB machine. Mitigations, in order:

- **Gemma 4 makes this cheap by architecture**: 4 of 5 layers are
  sliding-window capped at 512 tokens regardless of prefix length, and
  later layers share KV — only the sparse full-attention layers pay `B ×
  prefix`. (Model weights at 4-bit: E2B ≈ 2.51 GB, E4B ≈ 5.82 GB,
  9B ≈ 5.2 GB — budget KV against what remains of 24 GB.)
- Cap effective batch width via the engine's admission control rather
  than replicating for the whole column at once.
- `PrefixStore` is a small LRU (`max_entries=8`) so idle prefixes don't
  pin KV.

There is no upstream `BatchQuantizedKVCache` (and
`RotatingKVCache.to_quantized` raises `NotImplementedError`), so KV
quantization is **not** an available mitigation for the batched path.

## Parity harness (`parity.py`)

Parity is defined as runnable code, greedy (`temperature=0.0`) with a
fixed seed. Divergence metric: `1 − lcp/max(len)` over token sequences —
greedy decode is autoregressive, so everything after the first flipped
token is conditioned on a different history; longest-common-prefix
measures "how soon did trajectories split" without double-counting.

| Check | Threshold | Why |
|---|---|---|
| batched vs sequential (fake, CI) | **0.0** | Deterministic engine; any drift is a harness/store bug. |
| batched vs sequential (real MLX) | **0.02** mean per-row rate | Left-padded batched decode (`BatchKVCache`) changes fp16 reduction order; near-tied greedy argmaxes can flip, and one early flip diverges that row's remainder. 2% mean is a regression tripwire sized so a handful of tie-flips in a batch passes but any systematic positional bug (which diverges *every* row, rate → ~1.0) fails immediately. It is not a correctness license. |
| with-cache vs without-cache | **0.0**, byte-identical strings | Cache reuse replays the same tokens at the same absolute positions; *any* difference means a violated invariant (seam, trim, alignment). This is the check that catches silent corruption, so it gets zero tolerance on both fake and real engines. |
| TTFT warm vs cold | reported, not thresholded | Measured **from row admission** (immediately before that row's engine call), not batch start — batch-start timing credits the cache with queueing time it never saved. Warm timing excludes the one-time shared prefill (amortized by design). |

Run: `uv run python -m polar_llama.local.parity` (CI-safe fakes; exits
non-zero on failure). `--real MODEL` is the guarded real-model entry.

Per-row failures return the JSON payload shape of
`create_error_response` in `src/model_client/mod.rs`
(`{"_error": ..., "_details": ..., "_raw"?}`) instead of failing the
batch, keeping error handling uniform across the Rust and local backends.

## BLOCKED-ON-GPU

Logic is fully CI-tested now (`tests/test_local_prefix_cache.py`, marker
`local`). Pending real-hardware validation on this M4 Pro / 24 GB:

1. Wire `_run_real_parity_suite` to the `MlxBatchEngine` and **measure**
   the real batched-vs-sequential divergence for Gemma 4 E2B/E4B and
   Ornith-1.0-9B; pin the observed number (expected ≪ 0.02) as the
   regression threshold.
2. Confirm with-cache == without-cache exactly on hybrid-cache models
   (the class where #980 lived).
3. Measure warm-vs-cold TTFT and replicated-KV memory at B ∈ {8, 32, 64}.
4. Validate the MLX cache clone function preserves state exactly
   (round-trip via cache `.state`).

## References

- Prompt-cache position-binding analysis: Gim et al., arXiv:2311.04934.
- MLX-vs-llama.cpp throughput (20–87% lead) and vllm-mlx speedups:
  arXiv:2601.19139 (waybarrios/vllm-mlx, Apache-2.0). Those numbers were
  measured on an M4 **Max** (~2× this machine's memory bandwidth) —
  treat 4.3×/5.8× as ceilings here, not targets.
- Broken hybrid trim path: mlx-lm issue #980 (closed "outdated",
  server-side mitigation only, no merged trim fix).
- Batched KV / continuous batching used as-is: mlx-lm ≥ 0.31
  (`BatchGenerator`, `batch_generate`, `BatchKVCache`,
  `BatchRotatingKVCache`, `_make_cache`).
