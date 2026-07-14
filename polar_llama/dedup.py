"""In-run duplicate collapsing and a persistent cross-job response cache
(issue #77).

Two related but distinct features, both built on the content-hash primitives
in `polar_llama/keys.py`:

- **In-run dedupe** (`dedupe=True`, no store): identical rows *within one
  batch* are sent to the backend once and the single result is fanned back
  out to every row that asked for it. Zero I/O beyond the inference calls
  themselves.
- **Persistent response cache** (`response_cache=...`): a directory-backed
  store -- a thin, feature-flavored subclass of
  `polar_llama.checkpoint.CheckpointStore` -- that lets identical requests
  from *different* runs (even different processes) reuse a prior result.
  Implies in-run dedupe for free, since computing content keys to consult
  the store already does all the grouping work.

Both are implemented as a single `map_batches` UDF wrapper (`deduped_expr`,
mirroring `checkpoint.checkpointed_expr`'s shape exactly): the public API
stays a lazy `pl.Expr`, so it composes in `with_columns` / `LazyFrame`
pipelines and works under both the default and streaming collect engines.

Key differences from checkpointing (deliberate, not oversights):

- No chunked `flush_every` loop -- there is no crash-durability contract for
  dedupe (a "batch" here is transient, not a resumable job), and the Rust
  inference plugin already parallelizes internally, so one call over all
  pending rows is both simpler and at least as fast.
- Only `ok=True` results are ever persisted to a response cache. A
  transient failure must never be replayed cross-job as if it were a real
  answer -- it simply recomputes on the next run, exactly like an
  uncached call would.
- `ResponseCacheStore` reuses `CheckpointStore`'s on-disk layout, atomic
  writes, and read-side dedup verbatim; it only adds TTL-on-read filtering
  and manual invalidation (`prune()`, `clear()`).

See `docs/design/RESPONSE_CACHE.md` for the full design write-up and the
interop rules with `checkpoint=`, `usage=`, and provider `cache=`.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import polars as pl

from polar_llama.checkpoint import CheckpointStore, _is_error_row, _now_utc
from polar_llama.keys import CollapsePlan, fan_out, plan_collapse

logger = logging.getLogger(__name__)


# ============================================================================
# TTL parsing
# ============================================================================

_TTL_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _parse_ttl(
    ttl: Optional[Union[str, int, float, datetime.timedelta]],
) -> Optional[float]:
    """Normalize a `ResponseCache.ttl` value to seconds (or `None` = never expires).

    Accepts a `datetime.timedelta`, a plain number of seconds (`int`/`float`),
    or a string of the form `"<number><unit>"` where unit is one of
    `s`/`m`/`h`/`d` (e.g. `"30m"`, `"24h"`, `"7d"`).

    Examples
    --------
    >>> _parse_ttl(None) is None
    True
    >>> _parse_ttl(90)
    90.0
    >>> _parse_ttl(datetime.timedelta(minutes=1.5))
    90.0
    >>> _parse_ttl("30m")
    1800.0
    >>> _parse_ttl("24h")
    86400.0
    >>> _parse_ttl("7d")
    604800.0
    """
    if ttl is None:
        return None
    if isinstance(ttl, datetime.timedelta):
        return ttl.total_seconds()
    if isinstance(ttl, bool):  # bool is an int subclass -- reject explicitly
        raise TypeError(f"Unsupported ttl type: {type(ttl)!r}")
    if isinstance(ttl, (int, float)):
        return float(ttl)
    if isinstance(ttl, str):
        s = ttl.strip().lower()
        if len(s) >= 2:
            unit = s[-1]
            number_part = s[:-1]
            if unit in _TTL_UNIT_SECONDS:
                try:
                    value = float(number_part)
                except ValueError:
                    pass
                else:
                    return value * _TTL_UNIT_SECONDS[unit]
        raise ValueError(
            f"Invalid ttl string {ttl!r}: expected a number followed by "
            "'s' (seconds), 'm' (minutes), 'h' (hours), or 'd' (days), "
            "e.g. '30m', '24h', '7d'."
        )
    raise TypeError(f"Unsupported ttl type: {type(ttl)!r}")


# ============================================================================
# Public config / stats types
# ============================================================================


@dataclass
class ResponseCache:
    """Configuration for a persistent, cross-job response cache.

    Parameters
    ----------
    path : str or Path
        Sidecar directory holding the cache's Parquet parts + metadata
        (same on-disk layout as `polar_llama.checkpoint.Checkpoint` --
        see `ResponseCacheStore`). Created if it doesn't already exist.
    ttl : str, int, float, timedelta, or None
        How long a cached entry stays servable. `None` (default) means
        entries never expire. A number is seconds; a string is
        `"<number><unit>"` with unit `s`/`m`/`h`/`d` (e.g. `"24h"`).
        Enforced lazily on read (`ResponseCacheStore.load_index`) -- an
        expired entry sits inert on disk until `prune()` reclaims it.
    on_mismatch : {"restart", "error"}
        What to do when the store's recorded fingerprint doesn't match the
        current run's configuration. Same semantics as
        `polar_llama.checkpoint.Checkpoint.on_mismatch`: content keys embed
        the fingerprint already, so `"restart"` (default) is nearly a
        no-op -- old entries simply can never match new keys.

    Passing `response_cache=` to `inference_async`/`inference_messages`
    implies in-run dedupe (`dedupe=True`) automatically -- computing content
    keys to consult the store already does the grouping work, so serving a
    cache without also collapsing in-batch duplicates would be wasteful.
    """

    path: Union[str, Path]
    ttl: Optional[Union[str, int, float, datetime.timedelta]] = None
    on_mismatch: str = "restart"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.on_mismatch not in ("restart", "error"):
            raise ValueError(
                "ResponseCache.on_mismatch must be 'restart' or 'error', got "
                f"{self.on_mismatch!r}"
            )
        _parse_ttl(self.ttl)  # validate eagerly -- fail at construction, not at collect

    def clear(self) -> None:
        """Delete every stored entry (full invalidation). `_meta.json` is left as-is."""
        ResponseCacheStore.for_admin(self.path).clear()

    def prune(self, now: Optional[datetime.datetime] = None) -> int:
        """Compact the store: drop expired + superseded entries, rewrite as one part.

        Expired entries (per `self.ttl`) and stale duplicates (an older
        entry superseded by a newer one under the same key) are dropped;
        the remainder is rewritten as a single new part file and the old
        part files are deleted. Safe to call at any time -- it is pure
        housekeeping, never required for correctness (TTL is already
        enforced lazily on every read).

        Returns
        -------
        int
            Number of entries retained after compaction.
        """
        return ResponseCacheStore.for_admin(
            self.path, ttl_seconds=_parse_ttl(self.ttl)
        ).prune(now=now)


@dataclass
class DedupeStats:
    """Mutable accumulator for in-run dedupe / response-cache aggregates.

    Pass a fresh (or `.reset()`) instance as `dedupe_stats=` to
    `inference_async`/`inference_messages`; it is updated in place (under an
    internal lock, so it's safe to share across concurrently-collecting
    LazyFrames) once per batch the underlying `map_batches` UDF processes.
    Values only populate at *collect* time -- reading them before `.collect()`
    (or `with_columns`, for an eager DataFrame) sees zeros. Re-collecting the
    same LazyFrame accumulates again; call `.reset()` first if that's not
    what you want.

    Attributes
    ----------
    rows_total : int
        Every row seen across all batches (including null rows).
    rows_null : int
        Null input rows (never hashed, never sent, never counted below).
    cache_hits : int
        Rows served from a persistent `response_cache` store.
    rows_collapsed : int
        Non-null, non-cache-hit rows that shared a content key with another
        row in the same batch and so were folded into that row's single
        call (i.e. computed rows minus unique calls made).
    calls_made : int
        Unique requests actually sent to the backend.
    """

    rows_total: int = 0
    rows_null: int = 0
    cache_hits: int = 0
    rows_collapsed: int = 0
    calls_made: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def saved_calls(self) -> int:
        """Total calls avoided: persistent cache hits plus in-run collapses."""
        return self.cache_hits + self.rows_collapsed

    @property
    def hit_rate(self) -> float:
        """`saved_calls / max(non-null rows seen, 1)`."""
        denom = max(self.rows_total - self.rows_null, 1)
        return self.saved_calls / denom

    def estimated_savings(self, avg_cost_per_call: float) -> float:
        """Estimate USD saved: `saved_calls * avg_cost_per_call`."""
        return self.saved_calls * avg_cost_per_call

    def reset(self) -> None:
        """Zero every counter (thread-safe)."""
        with self._lock:
            self.rows_total = 0
            self.rows_null = 0
            self.cache_hits = 0
            self.rows_collapsed = 0
            self.calls_made = 0


# ============================================================================
# Persistent store: thin subclass of CheckpointStore
# ============================================================================


class ResponseCacheStore(CheckpointStore):
    """`CheckpointStore` + TTL-on-read filtering + manual invalidation.

    Reuses the parent's Parquet-part layout, atomic writes, fingerprint
    filtering, and latest-`ts`-wins read-side dedup verbatim -- cross-job
    reuse needs exactly what checkpoint resume already built. See the
    `polar_llama.checkpoint.CheckpointStore` docstring for the on-disk
    layout.
    """

    def __init__(
        self,
        path: Union[str, Path],
        fingerprint: Optional[str],
        on_mismatch: str = "restart",
        fingerprint_inputs: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            path, fingerprint, on_mismatch, fingerprint_inputs=fingerprint_inputs
        )
        self.ttl_seconds = ttl_seconds

    @classmethod
    def for_admin(
        cls, path: Union[str, Path], ttl_seconds: Optional[float] = None
    ) -> "ResponseCacheStore":
        """Construct a store for `clear()`/`prune()` -- not tied to any run's
        fingerprint (`fingerprint=None`), so `_meta.json` is left untouched.
        """
        return cls(path, fingerprint=None, ttl_seconds=ttl_seconds)

    def load_index(
        self, ttl_seconds: Optional[float] = None
    ) -> Dict[str, Tuple[bool, str]]:
        """Load the index, TTL-filtered.

        `ttl_seconds` overrides `self.ttl_seconds` when explicitly passed;
        otherwise the store's own configured TTL applies.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        return super().load_index(ttl_seconds=effective_ttl)

    def prune(self, now: Optional[datetime.datetime] = None) -> int:
        """Drop expired + superseded entries, compact to a single part file.

        Reads every part, drops entries older than `self.ttl_seconds` (if
        set), keeps only the latest `ts` per key -- across **all**
        fingerprints, so other jobs' (or other configs') entries sharing
        this directory survive -- and rewrites the survivors as one new
        part before deleting the old parts. Returns the number of entries
        kept.
        """
        parts = self._part_paths()
        if not parts:
            return 0

        frames = []
        for p in parts:
            try:
                frames.append(pl.read_parquet(p))
            except Exception:
                continue  # same defensive skip as load_index
        if not frames:
            return 0

        df = pl.concat(frames, how="vertical_relaxed")

        if self.ttl_seconds is not None:
            cutoff = (now or _now_utc()) - datetime.timedelta(seconds=self.ttl_seconds)
            df = df.filter(pl.col("ts") >= cutoff)

        if df.height == 0:
            for p in parts:
                p.unlink(missing_ok=True)
            return 0

        # Keep latest ts per key (ties prefer ok=True), same rule as
        # load_index -- but across every fingerprint present in this
        # directory, not just self.fingerprint.
        df = df.sort(["key", "ts", "ok"])
        df = df.group_by("key", maintain_order=True).tail(1)

        part_name = f"part-{uuid.uuid4().hex}.parquet"
        final_path = self.path / part_name
        tmp_path = self.path / (part_name + ".tmp")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, final_path)

        for p in parts:
            p.unlink(missing_ok=True)

        return df.height

    def clear(self) -> None:
        """Delete every part file (full invalidation). `_meta.json` untouched."""
        for p in self._part_paths():
            p.unlink(missing_ok=True)


# ============================================================================
# The expression wrapper
# ============================================================================


def deduped_expr(
    expr: pl.Expr,
    *,
    run_pending: Callable[[pl.Series], pl.Series],
    fingerprint: str,
    canonicalize: Optional[Callable[[Any], str]] = None,
    store: Optional[ResponseCacheStore] = None,
    ttl_seconds: Optional[float] = None,
    stats: Optional[DedupeStats] = None,
    has_schema: bool = False,
) -> pl.Expr:
    """Wrap `expr` (the raw input column) with in-run dedupe (+ optional
    persistent response cache).

    Parameters
    ----------
    expr
        The expression producing each row's *input* to inference -- same
        contract as `checkpoint.checkpointed_expr`.
    run_pending
        Callable performing the actual inference: given a `pl.Series` of
        the batch's unique pending inputs, returns a same-length, same-order
        `pl.Series` of raw output strings. Unlike checkpointing, this is
        called **once per batch** (no chunking) -- there is no
        crash-durability contract here, and the Rust plugin already
        parallelizes internally.
    fingerprint
        Run-configuration fingerprint, embedded in every content key.
    canonicalize
        Optional row-value-to-hashable-string mapper (see `plan_collapse`).
    store
        A `ResponseCacheStore`, or `None` for in-run-only dedupe (zero I/O
        beyond the inference calls themselves).
    ttl_seconds
        TTL passed to `store.load_index()` each batch. Ignored if `store`
        is `None`.
    stats
        Optional `DedupeStats` accumulator, updated (lock-guarded) once per
        batch.
    has_schema
        Whether a `response_model` schema was requested -- passed through to
        `checkpoint._is_error_row` so only genuinely successful results are
        ever persisted to `store` (a failed row must recompute next run,
        never be cached as if it were a real answer).

    Returns
    -------
    A lazy `pl.Expr` (`expr.map_batches(...)`), usable exactly like the
    un-wrapped inference expression.
    """

    def _udf(s: pl.Series) -> pl.Series:
        dtype = s.dtype
        raw_inputs: List[Any] = s.to_list()
        n = len(raw_inputs)
        n_null = sum(1 for v in raw_inputs if v is None)

        index = store.load_index(ttl_seconds=ttl_seconds) if store is not None else None

        # response_cache/dedupe never serves a stored failure -- a transient
        # error must recompute, never be replayed as a real answer.
        plan: CollapsePlan = plan_collapse(
            raw_inputs,
            fingerprint,
            canonicalize=canonicalize,
            index=index,
            serve_failed=False,
        )

        n_pending = len(plan.unique_pending_keys)
        if n_pending:
            pending_series = pl.Series(plan.unique_pending_inputs, dtype=dtype)
            result_series = run_pending(pending_series)
            chunk_raws: List[Optional[str]] = result_series.to_list()

            fan_out(plan, plan.unique_pending_keys, chunk_raws)

            if store is not None:
                ok_keys: List[str] = []
                ok_raws: List[Optional[str]] = []
                ok_flags: List[bool] = []
                ok_errors: List[Optional[str]] = []
                for key, raw in zip(plan.unique_pending_keys, chunk_raws):
                    is_err, _err_type = _is_error_row(raw, has_schema)
                    if not is_err:
                        ok_keys.append(key)
                        ok_raws.append(raw)
                        ok_flags.append(True)
                        ok_errors.append(None)
                store.append(ok_keys, ok_raws, ok_flags, ok_errors)

        # Non-null, non-cache-hit rows that shared a pending key with at
        # least one other row -- i.e. rows folded into another row's call.
        rows_collapsed = sum(
            len(positions) - 1 for positions in plan.key_to_positions.values()
        )

        if stats is not None:
            with stats._lock:
                stats.rows_total += n
                stats.rows_null += n_null
                stats.cache_hits += plan.n_hits
                stats.rows_collapsed += rows_collapsed
                stats.calls_made += n_pending

        logger.debug(
            "polar_llama.dedup: rows_total=%d rows_null=%d cache_hits=%d "
            "rows_collapsed=%d calls_made=%d",
            n,
            n_null,
            plan.n_hits,
            rows_collapsed,
            n_pending,
        )

        return pl.Series(s.name, plan.results, dtype=pl.Utf8)

    return expr.map_batches(_udf, return_dtype=pl.Utf8)
