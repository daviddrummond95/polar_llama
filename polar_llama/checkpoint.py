"""Resumable batch checkpointing for `inference_async` / `inference_messages`.

Design summary (issue #75; see `docs/design/CHECKPOINTING.md` for the full
write-up):

Checkpointing is implemented entirely in Python, as a `map_batches` UDF
wrapped around the *input* expression. The public API stays a lazy Polars
expression -- `checkpointed_expr` returns `expr.map_batches(...)`, so it
composes in `with_columns` / `LazyFrame` pipelines exactly like the
un-checkpointed path, and works under both the default and streaming
collect engines (a store keyed by content hash is idempotent no matter how
many batches the engine happens to split a column into).

Inside the UDF, per batch it receives, the wrapper:

1. Canonicalizes each row to a hashable string and computes its content key
   (`polar_llama.keys.content_key`).
2. Loads the on-disk store index (`key -> (ok, result_raw)`) once.
3. Splits rows into "hit" (already stored -- and, if `ok=False`, only a hit
   when `retry_failed=False`) and "pending", deduplicating pending rows by
   key so identical inputs are only requested once per batch.
4. Runs the caller-supplied `run_pending` callback (the real Rust plugin
   expression, invoked eagerly on a chunk of up to `flush_every` unique
   pending rows).
5. Appends each chunk's results to the store as an atomic new Parquet part
   file *before* moving on to the next chunk -- a crash loses at most the
   in-flight chunk.
6. Reassembles the full batch (hits + fresh results, fanned back out to
   every row that shared a duplicate key) in original order.

Honest limitation: persistence is per-*chunk*, not per-row -- the wrapped
inference call returns a whole chunk's results at once, so there's no way to
persist a finer grain without Rust-side support for a per-row completion
callback (explicitly deferred; not needed to meet the "kill at 50%, resume,
re-spend at most ~flush_every rows" acceptance bar). `flush_every` (default
100) bounds the worst case.

Usage/cost accounting interop (issue #76): `inference_async`/`inference_messages`
raise `ValueError` if both `checkpoint=...` and `usage=True` are passed
together. Reason: `usage=True` wraps every row in a JSON envelope
(`{"response": ..., "usage": {...}}`, see `src/model_client/mod.rs`), but
`_is_error_row` below only recognizes a failed row by a TOP-LEVEL
`{"_error": ...}` shape -- under the envelope, a failed row's error object is
nested one level deeper (`{"response": {"_error": ...}, "usage": {...}}`),
so a stored envelope-wrapped error row would be misclassified as `ok=True`
and never retried on resume. Combining the two is designed to be supportable
(the store already persists the raw plugin string -- whatever it is -- so a
resumed usage=True run would naturally re-serve prior tokens/latency and
recompute `cost_usd` at decode time against whatever price table is active
at resume, which is desirable for invoicing), but making it correct requires
threading a `has_usage: bool` through this module (a "look one level under
`response`" branch in `_is_error_row`) and adding `"usage": bool` to the
checkpoint fingerprint (`polar_llama/keys.py::config_fingerprint`'s `extra`)
so a usage=False store can never be resumed as usage=True or vice versa
(rows would otherwise fail envelope decode). Deferred as follow-up work;
tracked in the issue #76 design doc.

Known risk: nesting an eager `DataFrame.select(...)` of a plugin expression
inside a `map_batches` UDF could, in principle, contend with Polars' rayon
thread pool on some versions if the plugin itself dispatched rayon work. It
doesn't here -- the inference plugins block on their own Tokio runtime, not
rayon, and the inner query is a single elementwise expression over a fresh
DataFrame -- so this is believed low-risk. If a deadlock is ever observed,
the fix is to run the inner `.select(...)` on a dedicated `threading.Thread`
and `.join()` it (that detaches the call from whatever rayon context the
outer `map_batches` UDF is invoked under), which is why `run_pending` is
passed in as a plain, thread-safe callable rather than something this module
calls directly.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import polars as pl

from polar_llama.keys import content_key

_META_FILENAME = "_meta.json"
_FORMAT_VERSION = 1

_PART_SCHEMA = {
    "key": pl.Utf8,
    "fingerprint": pl.Utf8,
    "result_raw": pl.Utf8,
    "ok": pl.Boolean,
    "error": pl.Utf8,
    "ts": pl.Datetime("us", "UTC"),
}


@dataclass
class Checkpoint:
    """Configuration for resumable batch checkpointing.

    Parameters
    ----------
    path : str or Path
        Sidecar directory holding checkpoint parts + metadata. Created (and
        any missing parent directories) if it doesn't already exist.
    flush_every : int
        Number of *unique* pending rows sent to the underlying inference
        call -- and flushed to disk -- per chunk. Bounds the amount of
        duplicate spend lost to a crash to (at most) one chunk's worth of
        rows. Default 100.
    retry_failed : bool
        Whether rows previously stored with `ok=False` (the Rust plugin's
        in-band `{"_error": ...}` response, for either `api_error` or
        `validation_failed`) are re-attempted on resume. Default True.
        When False, the stored error row is returned exactly as stored --
        which still flows through the normal downstream
        `_error`/`_details`/`_raw` struct decoding when a `response_model`
        is set, so a "skip-failed-rows" resume looks identical to a fresh
        run that happened to fail those rows.
    on_mismatch : {"restart", "error"}
        What to do when an existing store's fingerprint doesn't match the
        current run's configuration (model/prompt/schema/etc. changed).
        `"restart"` (default) is nearly a no-op: content keys already embed
        the fingerprint, so old entries can never match new rows regardless
        -- every row is simply treated as pending, and old part files are
        left untouched (harmless dead weight, not deleted). `"error"` raises
        immediately instead, for callers who want to catch an accidental
        config drift rather than silently re-spend.
    """

    path: Union[str, Path]
    flush_every: int = 100
    retry_failed: bool = True
    on_mismatch: str = "restart"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.on_mismatch not in ("restart", "error"):
            raise ValueError(
                "Checkpoint.on_mismatch must be 'restart' or 'error', got "
                f"{self.on_mismatch!r}"
            )
        if self.flush_every < 1:
            raise ValueError("Checkpoint.flush_every must be >= 1")


def _is_error_row(raw: Optional[str], has_schema: bool) -> Tuple[bool, Optional[str]]:
    """Detect a failed row cheaply, for both of the Rust plugin's two shapes.

    `has_schema` (a `response_model` was set): the plugin always embeds
    `{"_error": ..., "_details": ..., ["_raw": ...]}` into the row string on
    failure (`create_error_response`, `src/model_client/mod.rs`), for both
    `api_error` and `validation_failed` -- so a failed row is detectable from
    the raw string alone. Only in this mode do we interpret an in-band
    `{"_error": ...}` object as failure.

    `not has_schema` (plain text): the plugin's non-schema fetch path
    (`fetch_data_generic` / `fetch_data_generic_enhanced`, `errors_as_json =
    false`) reports a failed request as a bare `None` for that row -- the exact
    same value a checkpoint-unaware caller would see today. A *successful*
    plain-text row is the model's raw output, which may legitimately be a JSON
    object that happens to contain an `_error` key; interpreting that as a
    failure would misclassify a good row, store it `ok=False`, and re-request
    it on every resume (breaking idempotency). So on the no-schema path only a
    bare `None` output is a failure -- never the row's own text. Since every
    row reaching this check already had a non-null *input* (null inputs are
    filtered out before `run_pending` is ever called), a `None` output
    unambiguously means the request failed, never "nothing to compute".

    Failed rows are surfaced as `ok=False` so the store retries them by
    default and never mistakes them for a completed row.

    Returns
    -------
    (is_error, error_type) -- `error_type` is the `_error` field's value
    (e.g. `"api_error"` or `"validation_failed"`) when a schema was present,
    `"request_failed"` for the no-schema `None` case, else `None`.
    """
    if raw is None:
        return True, "request_failed"
    if not has_schema:
        # Plain-text success is the model's own output; never reinterpret it.
        return False, None
    stripped = raw.lstrip()
    if not stripped.startswith("{") or '"_error"' not in stripped:
        return False, None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False, None
    if isinstance(obj, dict) and "_error" in obj:
        return True, obj.get("_error")
    return False, None


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class CheckpointStore:
    """Parquet-directory sidecar store for a single checkpointed run.

    Layout::

        <path>/
          _meta.json             {"format_version": 1, "fingerprint": ..., ...}
          part-<uuid4>.parquet   one per flush, append-only, atomic write

    Parquet has no append mode, so each flush writes a brand-new part file
    (`part-<uuid>.parquet.tmp` then `os.replace()`, atomic on POSIX) --
    a crash mid-write leaves only an orphaned `.tmp` file that readers ignore
    (the glob `part-*.parquet` never matches a `.tmp` suffix). This also
    makes two concurrent writers on the same store directory benign: unique
    filenames mean no locking is needed, and read-side dedup (keep the
    latest `ts`; ties prefer `ok=True`) resolves any overlap.
    """

    def __init__(
        self,
        path: Union[str, Path],
        fingerprint: str,
        on_mismatch: str = "restart",
        fingerprint_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.on_mismatch = on_mismatch
        self.path.mkdir(parents=True, exist_ok=True)
        self._reconcile_meta(fingerprint_inputs)

    @property
    def meta_path(self) -> Path:
        return self.path / _META_FILENAME

    def _reconcile_meta(self, fingerprint_inputs: Optional[Dict[str, Any]]) -> None:
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
            stored_fp = meta.get("fingerprint")
            if stored_fp is not None and stored_fp != self.fingerprint:
                if self.on_mismatch == "error":
                    raise ValueError(
                        f"Checkpoint store at {self.path} was written under a "
                        f"different run configuration (stored fingerprint "
                        f"{stored_fp!r} != current {self.fingerprint!r}). Pass "
                        "on_mismatch='restart' to treat this as a fresh run "
                        "(old entries are inert -- their keys embed the old "
                        "fingerprint and can never match new rows) or use a "
                        "different checkpoint path."
                    )
                # on_mismatch == "restart": fall through and overwrite
                # _meta.json below. Old part files are left in place; their
                # keys can never collide with new keys (both embed their
                # respective fingerprint), so they are simply inert weight,
                # not a correctness risk.
        self._write_meta(fingerprint_inputs)

    def _write_meta(self, fingerprint_inputs: Optional[Dict[str, Any]]) -> None:
        meta: Dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "fingerprint": self.fingerprint,
        }
        if fingerprint_inputs is not None:
            meta["fingerprint_inputs"] = fingerprint_inputs
        tmp = self.path / (_META_FILENAME + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp, self.meta_path)

    def _part_paths(self) -> List[Path]:
        return sorted(self.path.glob("part-*.parquet"))

    def load_index(self) -> Dict[str, Tuple[bool, str]]:
        """Load the store into `key -> (ok, result_raw)`, deduped.

        Among entries sharing a key, the latest `ts` wins; exact ties prefer
        `ok=True` (a fresh success over a stale failure written at the same
        instant).
        """
        parts = self._part_paths()
        if not parts:
            return {}

        frames = []
        for p in parts:
            try:
                frames.append(pl.read_parquet(p))
            except Exception:
                # Defensive: a corrupt/truncated file should never exist
                # under the `part-*.parquet` (non-`.tmp`) name given the
                # atomic-rename write path, but skip rather than crash if
                # one somehow does.
                continue
        if not frames:
            return {}

        df = pl.concat(frames, how="vertical_relaxed")
        # Entries are only ever meaningful for the current fingerprint --
        # this also makes stale entries from a superseded config inert even
        # under on_mismatch="restart", with no explicit deletion needed.
        df = df.filter(pl.col("fingerprint") == self.fingerprint)
        if df.height == 0:
            return {}

        # Sort ascending by (key, ts, ok); `False < True` means an ok=True
        # row sorts *after* an ok=False row at the same timestamp, so taking
        # the last row per key picks the successful one on exact ties.
        df = df.sort(["key", "ts", "ok"])
        df = df.group_by("key", maintain_order=True).tail(1)

        index: Dict[str, Tuple[bool, str]] = {}
        for row in df.iter_rows(named=True):
            index[row["key"]] = (bool(row["ok"]), row["result_raw"])
        return index

    def append(
        self,
        keys: List[str],
        raws: List[Optional[str]],
        oks: List[bool],
        errors: List[Optional[str]],
    ) -> None:
        """Atomically append one new part file holding these rows."""
        if not keys:
            return
        ts = _now_utc()
        df = pl.DataFrame(
            {
                "key": keys,
                "fingerprint": [self.fingerprint] * len(keys),
                "result_raw": raws,
                "ok": oks,
                "error": errors,
                "ts": [ts] * len(keys),
            },
            schema=_PART_SCHEMA,
        )
        part_name = f"part-{uuid.uuid4().hex}.parquet"
        final_path = self.path / part_name
        tmp_path = self.path / (part_name + ".tmp")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, final_path)


def checkpointed_expr(
    expr: pl.Expr,
    *,
    run_pending: Callable[[pl.Series], pl.Series],
    fingerprint: str,
    checkpoint: Checkpoint,
    canonicalize: Optional[Callable[[Any], str]] = None,
    fingerprint_inputs: Optional[Dict[str, Any]] = None,
    has_schema: bool = False,
) -> pl.Expr:
    """Wrap `expr` (the raw input column) with resumable checkpointing.

    Parameters
    ----------
    expr
        The expression producing each row's *input* to inference (a Utf8
        prompt column, or a message-array column for `inference_messages`).
        Not the inference *output* -- this wrapper computes content keys
        from the input and calls `run_pending` to obtain output for
        whichever rows aren't already in the store.
    run_pending
        Callable that performs the actual inference: given a `pl.Series` of
        pending rows' (canonicalized-for-hashing, but original-dtype)
        inputs, returns a same-length, same-order `pl.Series` of raw output
        strings (pre-struct-decode -- exactly what the bare Rust plugin
        expression would return). Must be safe to call multiple times, once
        per chunk of at most `checkpoint.flush_every` unique pending rows.
    fingerprint
        Run-configuration fingerprint (`polar_llama.keys.config_fingerprint`)
        embedded into every content key so a config change can never
        silently reuse stale results.
    checkpoint
        `Checkpoint` settings (store path, chunk size, retry-failed policy).
    canonicalize
        Optional function mapping a raw row value (str, or a native
        List(Struct) row as a list of dicts) to the string that gets hashed
        for its content key. `None` means the row value is already a
        hashable string as-is (the `inference_async` case). Only affects
        *hashing* -- `run_pending` always receives the original row value,
        not the canonicalized form, so the wrapped call sees exactly what it
        would have without checkpointing.
    fingerprint_inputs
        Optional human-readable dict recorded in `_meta.json` purely for
        debuggability (not used for invalidation -- `fingerprint` alone is
        authoritative).

    Returns
    -------
    A lazy `pl.Expr` (`expr.map_batches(...)`) that can be used exactly like
    the un-wrapped inference expression -- in `with_columns`, chained into a
    `LazyFrame`, and under either the default or streaming collect engine.
    """

    def _udf(s: pl.Series) -> pl.Series:
        store = CheckpointStore(
            checkpoint.path,
            fingerprint,
            checkpoint.on_mismatch,
            fingerprint_inputs=fingerprint_inputs,
        )
        index = store.load_index()

        dtype = s.dtype
        raw_inputs: List[Any] = s.to_list()
        n = len(raw_inputs)

        keys: List[Optional[str]] = [None] * n
        for i, v in enumerate(raw_inputs):
            if v is None:
                continue
            hash_source = canonicalize(v) if canonicalize is not None else v
            keys[i] = content_key(hash_source, fingerprint)

        results: List[Optional[str]] = [None] * n
        # Ordered de-dup of pending rows by key: identical inputs (including
        # duplicates already inside this one batch) are only ever sent to
        # `run_pending` once; every row sharing that key gets the same
        # result fanned back out below.
        unique_pending_keys: List[str] = []
        unique_pending_inputs: List[Any] = []
        seen_pending: Dict[str, int] = {}
        key_to_positions: Dict[str, List[int]] = {}

        for i, key in enumerate(keys):
            if key is None:
                continue  # null row: passes through as null, never stored
            hit = index.get(key)
            if hit is not None:
                ok, stored_raw = hit
                if ok or not checkpoint.retry_failed:
                    results[i] = stored_raw
                    continue
            key_to_positions.setdefault(key, []).append(i)
            if key not in seen_pending:
                seen_pending[key] = len(unique_pending_keys)
                unique_pending_keys.append(key)
                unique_pending_inputs.append(raw_inputs[i])

        flush_every = checkpoint.flush_every
        n_unique = len(unique_pending_keys)
        for start in range(0, n_unique, flush_every):
            chunk_keys = unique_pending_keys[start : start + flush_every]
            chunk_inputs = unique_pending_inputs[start : start + flush_every]

            chunk_series = pl.Series(chunk_inputs, dtype=dtype)
            chunk_result_series = run_pending(chunk_series)
            chunk_raws: List[Optional[str]] = chunk_result_series.to_list()

            oks: List[bool] = []
            errors: List[Optional[str]] = []
            for key, raw in zip(chunk_keys, chunk_raws):
                is_err, err_type = _is_error_row(raw, has_schema)
                oks.append(not is_err)
                errors.append(err_type)
                for pos in key_to_positions[key]:
                    results[pos] = raw

            # Flush *this chunk* before moving to the next one: a crash
            # loses at most the in-flight chunk's worth of duplicate spend.
            store.append(chunk_keys, chunk_raws, oks, errors)

        return pl.Series(s.name, results, dtype=pl.Utf8)

    return expr.map_batches(_udf, return_dtype=pl.Utf8)
