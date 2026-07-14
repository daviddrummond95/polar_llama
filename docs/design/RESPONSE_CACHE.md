# Duplicate collapsing and a persistent response cache (issue #77)

`inference_async(..., dedupe=True)` / `inference_messages(..., dedupe=True)`
collapse exact-duplicate rows within one run before sending them to the
backend. `response_cache=...` adds a persistent, cross-job store on top,
reusing `polar_llama/checkpoint.py`'s Parquet-part store verbatim.

The feature is **DataFrame-shaped**, exactly like checkpointing (#75): the
API stays a lazy Polars expression, composes in `with_columns`/`LazyFrame`
pipelines, and works under both the default and streaming collect engines.

## Naming: four different "caches"

This library now has four kwargs that are all, loosely, "a cache". They are
deliberately kept separate rather than overloaded onto one `cache=` kwarg:

| Kwarg              | What it caches                                                          |
| ------------------ | ------------------------------------------------------------------------ |
| `cache=`            | Provider prompt-prefix caching (transport-level; changes how a request is *sent*, never what is asked for). |
| `checkpoint=`       | Resume the *same* job after a crash.                                     |
| `dedupe=`           | Collapse exact-duplicate rows *within one run*, zero I/O.                |
| `response_cache=`   | Reuse results *across* runs/processes for identical requests.            |

`response_cache=` **implies** `dedupe=True` silently: computing content keys
to consult the store already does all the grouping work, so serving a cache
without also collapsing in-batch duplicates would be pure waste.

## Mechanism

Both features share one Python module, `polar_llama/dedup.py`, and one
`map_batches` UDF wrapper, `deduped_expr` (mirrors
`checkpoint.checkpointed_expr`'s shape exactly). Per batch the UDF:

1. Computes a per-row content key (`polar_llama.keys.content_key`) — the
   exact same primitive checkpointing uses, extracted grouping logic
   included (see below).
2. If a `ResponseCacheStore` is attached, loads its index (TTL-filtered).
3. Splits rows into hits (served from the store) and pending, deduplicating
   pending rows by key so identical inputs are only requested once per
   batch — even with no store at all, this is `dedupe=True`'s entire
   contract.
4. Runs the caller-supplied `run_pending` callback **once**, over every
   unique pending input in the batch (no chunking — see "Why no
   `flush_every`" below).
5. Fans each result back out to every row that shared its key, in original
   order.
6. If a store is attached, persists only the `ok=True` results.

### Shared collapse bookkeeping: `polar_llama/keys.py`

The hit/pending/fan-out bookkeeping used to live inline in
`checkpoint.checkpointed_expr`'s `_udf`. It has been extracted to
`polar_llama.keys.plan_collapse` / `polar_llama.keys.fan_out` (pure, no I/O)
so `dedup.py` can reuse it byte-for-byte, and `checkpoint.py` has been
refactored to call the same helpers instead of duplicating the logic.
`tests/test_checkpointing.py` passes unmodified against the refactor — this
was verified as a behavior-preservation gate before anything else was built.

### Why no `flush_every` chunking

Checkpointing chunks pending rows and flushes to disk after each chunk so a
crash loses at most `flush_every` rows' worth of duplicate spend — there is
a durability contract for a long-running, resumable job. Dedupe has no such
contract: a "batch" here is transient (one `map_batches` invocation), not a
resumable job, and the Rust inference plugin already parallelizes internally
across the rows it's given. So `deduped_expr` sends every unique pending row
to `run_pending` in one call — simpler and at least as fast.

### Why only `ok=True` is ever persisted

A transient failure (rate limit, timeout, provider outage) must never be
replayed cross-job as if it were a real answer. `deduped_expr` always passes
`serve_failed=False` to `plan_collapse` (dedupe/response_cache never treats
a stored failure as "done") and only appends `ok=True` rows to the store —
a failed row simply recomputes on the next run, exactly like an uncached
call would.

## The persistent store: `ResponseCacheStore`

A thin subclass of `polar_llama.checkpoint.CheckpointStore` living in
`dedup.py`. It reuses the parent's on-disk layout (append-only Parquet
parts, atomic `os.replace` writes, fingerprint filtering, latest-`ts`-wins
read-side dedup) verbatim, and adds:

- **TTL-on-read filtering** (`load_index(ttl_seconds=...)`): an entry older
  than `now - ttl_seconds` is treated as absent (pending, not a hit).
  Enforced lazily on every read — an expired entry sits inert on disk until
  reclaimed.
- **`prune(now=None)`**: compaction. Reads every part, drops expired and
  superseded entries (keeping the latest `ts` per key across **all**
  fingerprints sharing the directory, so other jobs'/configs' entries
  survive), rewrites the survivors as one new part, deletes the old parts.
  Never required for correctness — pure housekeeping.
- **`clear()`**: deletes every part file (full invalidation). `_meta.json`
  is left as-is.

`ResponseCache.clear()` / `.prune()` are the user-facing entry points; they
construct a `ResponseCacheStore` with `fingerprint=None` (an administrative
handle not tied to any one run's config) and delegate.

**Fingerprint**: identical to checkpointing's
`polar_llama.keys.config_fingerprint` — the same `_request_fingerprint`
helper in `polar_llama/__init__.py` (extracted from what used to be
duplicated inline in `inference_async`/`inference_messages`) computes it for
both the checkpoint path and the dedupe/response_cache path. Since content
keys embed the fingerprint, a shared cache directory can safely hold entries
from many different configs (model, schema, endpoint, ...) simultaneously —
their key spaces are disjoint, exactly the property checkpointing already
relies on.

## Stats: `DedupeStats`

Aggregates (`rows_total`, `rows_null`, `cache_hits`, `rows_collapsed`,
`calls_made`) don't fit a pure-expression return value, so they're surfaced
via a caller-passed mutable accumulator:

```python
stats = DedupeStats()
out = df.with_columns(r=inference_async(pl.col("p"), dedupe=True, dedupe_stats=stats))
print(stats.hit_rate, stats.rows_collapsed, stats.calls_made)
```

The UDF `+=`s into it (lock-guarded, so it's safe across concurrently
collecting LazyFrames) once per batch at collect time. Rejected
alternatives: a module-level "last run" global (races between concurrent
runs); returning a tuple from the expression (breaks the pure-expression
contract); log-only (not programmatic — though a `logging.debug` summary
line is still emitted per batch for free observability).

`DedupeStats` is deliberately **not** folded into `CacheMetrics`
(`polar_llama/types.py`) — that type is token/provider-shaped (prompt
caching); `DedupeStats` is call/row-shaped. Kept separate, documented in
both docstrings.

## Interop rules

1. **`response_cache=` + `checkpoint=`** → raises `ValueError`. Two
   persistent stores for one expression; point `checkpoint=` at same-job
   resume, `response_cache=` at cross-job reuse. Composing them (a layered
   cache → checkpoint → compute lookup) is deferred, not needed for v1.
2. **`dedupe=True` + `checkpoint=`** → allowed, silently subsumed (no
   warning). The checkpoint UDF already collapses duplicates per batch on
   its own.
3. **`response_cache=` + `usage=True`** → raises `ValueError`. Same
   envelope problem as `checkpoint=` + `usage=True` (see
   `docs/design/CHECKPOINTING.md`): `usage=True` wraps every row in
   `{"response": ..., "usage": {...}}`, and the store's `ok`/`fail`
   classification (`checkpoint._is_error_row`) only recognizes a top-level
   `{"_error": ...}` shape — under the envelope a failed row would be
   misclassified as `ok=True` and cached. A served row's cached `usage`
   would also misreport spend (stale tokens/latency from whenever it was
   first computed).
4. **`dedupe=True` + `usage=True`** (no store) → allowed. Fanned-back
   duplicate rows carry the **same** `usage` struct as the row that was
   actually computed — the raw envelope string is exactly what fans out, so
   this falls out naturally and truthfully reflects "what the model
   returned for this content". **Documented consequence:**
   `SUM(cost_usd)` over-counts actual spend by the collapse factor;
   `dedupe_stats.calls_made` is the true-spend signal, or
   `df.unique(subset=[...])` before summing. (Zeroing duplicate usage was
   rejected: it requires rewriting envelopes post-hoc, makes per-row usage
   lie about what the response cost to produce, and breaks row-identity
   with an uncached run.)
5. **`dedupe=`/`response_cache=` + provider `cache=`** → compose freely.
   Provider prompt caching is transport-level and already excluded from
   `config_fingerprint` (`polar_llama/keys.py`). Fewer unique calls simply
   means fewer opportunities for the provider's own cache to hit.
6. **`inference_stream`** → out of scope; no `dedupe=`/`response_cache=`
   kwargs.

## Testing

`tests/test_dedupe_cache.py` mirrors `tests/test_checkpointing.py`'s
layering: unit tests against `deduped_expr` with a counting stub (no
network), an end-to-end test against a local mock OpenAI-compatible HTTP
server, and a gated real-API proof skipped unless `OPENAI_API_KEY` is set.
`tests/test_checkpointing.py` was run unmodified against the `keys.py`
extraction/`checkpoint.py` refactor as a behavior-preservation gate — all
pre-existing tests pass with zero edits.
