# Resumable batch checkpointing (issue #75)

`inference_async(..., checkpoint=...)` and `inference_messages(..., checkpoint=...)`
persist completed rows to a sidecar store so a large run that dies mid-way
(rate limit, network error, provider outage) can be re-run and resume, paying
only for the rows that had not yet completed.

The feature is **DataFrame-shaped**: the API stays a lazy Polars expression
(`df.with_columns(out=inference_async(col, checkpoint="run.ckpt"))`), composes
in `with_columns` / `LazyFrame` pipelines, and works under both the default and
streaming collect engines.

## Mechanism

Checkpointing is implemented entirely in Python (`polar_llama/checkpoint.py`) as
a `map_batches` UDF wrapped around the *input* expression. Per batch the UDF:

1. Computes a per-row **content key** (see Hashing).
2. Loads the store index (`key -> (ok, result_raw)`) once.
3. Partitions rows into *hits* (already `ok`, or `ok=False` when
   `retry_failed=False`) and *pending*.
4. Runs the existing Rust inference plugin on the pending rows, in chunks of
   `flush_every`, via a nested eager `select`.
5. Appends each chunk's results to the store **before** the next chunk runs, so
   a crash loses at most one chunk of duplicate spend.
6. Reassembles the batch in original row order and returns a `Utf8` series; the
   existing struct-decode `map_batches` runs downstream unchanged, so structured
   output is byte-identical whether a row came from the store or the API.

"Persist as they finish" is per-*chunk*, not per-row (the Rust fan-out returns a
whole call's results at once). `flush_every` (default 100) bounds worst-case
re-spend on crash. True per-row persistence would need a Rust-side completion
callback — a possible follow-up, not required to meet the acceptance criteria.

## Store

`checkpoint="run.ckpt"` names a **directory** of append-only Parquet parts:

```
run.ckpt/
  _meta.json                 # {"format_version": 1, "fingerprint": "...", ...}
  part-<uuid4>.parquet       # one per flush
```

Each flush writes `part-<uuid>.parquet.tmp` then `os.replace()` (atomic on
POSIX) — a crash mid-write leaves a `.tmp` that readers ignore, and unique
filenames make concurrent writers on the same store benign. Resume loads
`part-*.parquet` and dedupes per `key` (latest `ts`, preferring `ok=True`).
Part schema: `key, fingerprint, result_raw, ok, error, ts`. Polars reads/writes
Parquet natively, so there is **no new dependency** (SQLite was rejected: results
round-trip through JSON strings anyway, and #77 wants to scan the store as a
DataFrame, which Parquet gives for free).

## Hashing (shared with issue #77)

`polar_llama/keys.py` holds the reusable primitives so #77 (content-hash
response caching) can import them without pulling in store I/O:

- `config_fingerprint(...)` — sha256 of canonical JSON over every
  request-shaping parameter: `symbol` (`inference_async` vs
  `inference_messages`), provider, model, `response_schema`,
  `response_model_name`, `system_prompt`, and `extra` (which carries the
  **endpoint** base-URL override, so a run against real OpenAI never returns a
  stale hit for a run routed to the local-MLX server via `OPENAI_BASE_URL`).
- `content_key(row_input, fingerprint)` — `sha256(fingerprint || 0x00 || input)`.

Hashing uses `hashlib.sha256`, **not** `pl.Expr.hash()` (Polars' hash is seed-
and version-unstable, which would silently break resume across upgrades). The
key embeds the fingerprint, so changing prompt/model/params/schema/endpoint
makes every old entry inert automatically — invalidation is a property of the
key, not a separate check.

## Error handling / retry

Failed rows are stored `ok=False` and retried by default (`retry_failed=True`).
With a `response_model` the Rust plugin embeds `{"_error", "_details", "_raw"}`
into the row string on failure; with no schema, a failed request surfaces as a
bare `None`. A successful *plain-text* row is the model's raw output and is
**never** reinterpreted as an error even if it happens to be JSON containing an
`_error` key (that would misclassify a good row and re-request it every resume).
`retry_failed=False` returns the stored error verbatim through the existing
`_error`/`_details`/`_raw` struct decode.

## Acceptance criteria coverage

- **Kill at 50% → resume ≈ one full pass**: verified with a `flush_every=1`
  counting stub (exactly-once-per-row, no double-spend).
- **Invalidate on prompt/model/param change**: the key embeds the fingerprint.
- **Works with structured outputs and tool-use**: the struct-decode path is
  unchanged; stored raw strings round-trip identically.
