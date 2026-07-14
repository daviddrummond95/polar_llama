"""
Deterministic run manifests for Polar Llama (issue #85).

A `RunManifest` is a small, JSON-serializable audit record of *what a run
asked for* -- provider, model, prompt/schema content (as hashes, not text),
sampling params, and the fingerprint the runtime itself used for dedupe /
checkpointing -- with a `manifest_id` computed deterministically from those
fields. Two runs with identical configuration get the same `manifest_id`
even if they ran hours apart and produced different `created_at` timestamps
or different token counts.

Design summary
---------------

1. **Reuse, not reimplementation.** The manifest's `config_fingerprint` is
   computed by calling `polar_llama.keys.request_fingerprint` -- the exact
   function `inference_async`/`inference_messages` call for the #75
   checkpoint path and the #77 dedupe/response-cache path (moved here
   verbatim from `polar_llama/__init__.py`, aliased there as
   `_request_fingerprint` so those call sites are untouched). A manifest's
   `config_fingerprint` is therefore *guaranteed* to equal the runtime
   fingerprint for the same (symbol, provider, model, schema, system_prompt)
   -- not a parallel hash that could drift out of sync.

   Likewise `checkpoint_id`/`response_cache_id` are read straight out of a
   `Checkpoint`/`ResponseCache` store's `_meta.json` (`polar_llama.checkpoint`
   / `polar_llama.dedup`, issues #75/#77) rather than recomputed, and
   `dedupe_stats` is a snapshot of a `polar_llama.dedup.DedupeStats`
   (issue #76's `USAGE_DTYPE` shape is reused unmodified for
   `aggregate_usage`, aggregated straight off a `usage=True` output column).

2. **Determinism hash excludes anything the runtime can't pin down.**
   `manifest_id` is a sha256 over a canonical-JSON subset of fields
   (`RunManifest.hashed_fields()`) -- the same canonicalization
   `config_fingerprint` uses (`sort_keys=True, separators=(",", ":"),
   ensure_ascii=False`). Explicitly EXCLUDED: `manifest_id` itself,
   `created_at` (the one field two identical runs are *permitted* to
   differ on), `aggregate_usage`/`dedupe_stats` (nondeterministic output,
   not request-shaping), and `checkpoint_id`/`response_cache_id` (identify
   a *resume mechanism*, not the request -- same philosophy as
   `keys.config_fingerprint` excluding `cache_*`).

   FLAG: `polar_llama_version` and `endpoint` (the provider base-URL
   override) ARE in the hash -- a library upgrade or a different endpoint
   is a config change for audit purposes, matching `config_fingerprint`'s
   own inclusion of `endpoint` in its `extra`.

3. **Params / server-path caveat.** `inference_async`/`inference_messages`
   do not forward `temperature`/`max_tokens`/`seed` to hosted providers
   today -- those only exist on the local-MLX path
   (`.llama.inference_local` / `inference_local`). So `params={}` on a
   hosted-path manifest means "provider server-side defaults apply", which
   this library cannot pin or verify; the manifest guarantees what the
   *client* asked for, not what the provider actually used. `seed` is
   recorded for audit even though it, too, is not forwarded on hosted
   paths today. This caveat applies to `replay()` as well: replaying a
   manifest re-issues the *client-visible* request, not a guarantee of
   identical provider-side sampling.

4. **Prompt-hash vs. replay tension -- verify-then-replay.** The manifest
   stores sha256 HASHES of `system_prompt` / `prompt_template` / the
   response schema, never their text, so a manifest is safe to share as an
   audit artifact without leaking prompt IP. This means `replay()` cannot
   reconstruct a call from the manifest alone by design: the caller must
   re-supply the prompt/schema, and `replay()` verifies (by hash) that what
   was supplied matches what the manifest recorded *before* re-running --
   catching silent drift instead of silently replaying against changed
   content. Callers who want fully self-contained replay can opt in via
   `build_manifest(..., store_texts=True)`, which embeds the plaintext in
   `RunManifest.texts` (itself excluded from the determinism hash, since
   the hashes above already cover it).

No Rust changes -- this module is plain Python (dataclasses + stdlib
`hashlib`/`json`), mirroring `polar_llama/hitl.py` / `polar_llama/codebook.py`.
`manifest.py` imports only `polar_llama.checkpoint`, `polar_llama.dedup`,
`polar_llama.keys`, stdlib, and `polars` at module scope -- none of those
import the `polar_llama` package root, so importing `manifest.py` from
`polar_llama/__init__.py` creates no cycle. The one thing that *does* live
in `__init__.py` is `_pydantic_to_json_schema` (schema serialization) and
`inference_async`/`inference_messages` (for `replay`'s default execution
path) -- both are imported lazily, inside function bodies, to avoid a
circular import.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
)

import polars as pl

from polar_llama.checkpoint import Checkpoint, _META_FILENAME
from polar_llama.dedup import DedupeStats, ResponseCache
from polar_llama.keys import (
    _DEFAULT_MODEL,
    _DEFAULT_PROVIDER,
    config_fingerprint,
    endpoint_fingerprint_input,
    request_fingerprint,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

MANIFEST_FORMAT_VERSION = 1

#: Field names that participate in `manifest_id` (see `RunManifest.hashed_fields`).
#: Order matches the dataclass declaration; irrelevant for hashing (the
#: canonical JSON dump sorts keys) but kept in sync for readability.
_HASHED_FIELD_NAMES: Tuple[str, ...] = (
    "format_version",
    "polar_llama_version",
    "symbol",
    "provider",
    "model",
    "config_fingerprint",
    "system_prompt_hash",
    "prompt_template_hash",
    "response_model_name",
    "response_schema_hash",
    "params",
    "seed",
    "endpoint",
    # NOTE: row_count is deliberately NOT hashed. manifest_id is a stable
    # identity of the request *configuration*, so a manifest built before a run
    # (df=None -> row_count=None) and after (row_count=N), or two runs over
    # different batch sizes with byte-identical config, must share a
    # manifest_id. row_count is recorded as excluded metadata instead.
)


class ManifestIntegrityError(Exception):
    """A saved manifest's `manifest_id` doesn't match its own content.

    Raised by `load_manifest` after recomputing the hash over the loaded
    determinism fields (`RunManifest.hashed_fields()`) -- signals the file
    was hand-edited or corrupted after `save_manifest` wrote it.
    """


class ManifestMismatchError(Exception):
    """`replay(..., verify=True)` found the supplied inputs diverge from the manifest.

    Raised only for CONTENT mismatches (system_prompt / prompt_template /
    response_model, or the recomputed `config_fingerprint`) -- environment
    drift (endpoint, `polar_llama` version) is reported as a `UserWarning`
    instead, since it doesn't necessarily mean the *request* changed.
    """


def _hash_text(text: Optional[str]) -> Optional[str]:
    """sha256 hex digest of `text`, or `None` if `text` is `None`."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_id(hashed: Dict[str, Any]) -> str:
    """sha256 hex digest of `hashed`'s canonical JSON.

    Byte-identical canonicalization to `polar_llama.keys.config_fingerprint`
    (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`) -- nested
    dicts (e.g. `params`) are canonicalized by the same `sort_keys=True` pass,
    no separate handling needed.
    """
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """A deterministic, shareable audit record of one inference run's configuration.

    See the module docstring for the full design rationale. Construct via
    `build_manifest(...)`, not directly -- `manifest_id`/`created_at` must be
    computed consistently, which `build_manifest` does for you.

    Attributes
    ----------
    format_version : int
        `MANIFEST_FORMAT_VERSION` at build time. `load_manifest` refuses to
        load a manifest with a newer format version than this release knows.
    polar_llama_version : str
        `importlib.metadata.version("polar-llama")` at build time. Part of
        the determinism hash: a library upgrade is a config change for audit
        purposes (matches `config_fingerprint`'s inclusion of `endpoint`).
    symbol : str
        `"inference_async"` or `"inference_messages"` -- which Rust plugin
        entry point the run used.
    provider, model : str
        Normalized exactly like `polar_llama.keys.config_fingerprint`: a
        caller-supplied `None` is recorded as the library's fixed sentinel
        (`"openai"` / `"default"`), never resolved to "whatever today's
        actual default is".
    config_fingerprint : str
        `polar_llama.keys.request_fingerprint`'s output for this run's
        (symbol, provider, model, schema, system_prompt, endpoint) --
        identical to the fingerprint the #75 checkpoint / #77 dedupe paths
        compute for the same configuration.
    system_prompt_hash, prompt_template_hash, response_schema_hash : str or None
        sha256 hex digest of the corresponding text/JSON-schema-string, or
        `None` if that input wasn't used. Hashes, not text, by design --
        see the module docstring's "prompt-hash vs. replay tension" note.
    response_model_name : str or None
        `response_model.__name__`, or `None`.
    params : dict
        Client-forwarded sampling params (e.g. `temperature`, `max_tokens`,
        as recorded by the caller). `{}` on hosted paths today -- see the
        module docstring's params/server-path caveat.
    seed : int or None
        Recorded for audit; not forwarded on hosted paths today (see caveat).
    endpoint : str or None
        `polar_llama.keys.endpoint_fingerprint_input(provider)` at build
        time -- the active base-URL override, if any.
    row_count : int or None
        `df.height` at build time, or `None` if no `df` was supplied.
    manifest_id : str
        sha256 hex digest of the canonical JSON of every field above.
        EXCLUDED from its own computation, obviously.
    created_at : str
        ISO-8601 UTC timestamp. The ONLY field two manifests built from
        identical configuration are permitted to differ on. EXCLUDED from
        `manifest_id`.
    aggregate_usage : dict or None
        `{"input_tokens", "output_tokens", "cached_tokens", "cost_usd",
        "rows_with_usage"}` summed from a `usage=True` output column (issue
        #76), or `None` if `usage_column` wasn't supplied to
        `build_manifest`. EXCLUDED from `manifest_id` (output-nondeterministic).
    checkpoint_id, response_cache_id : str or None
        The fingerprint recorded in `<path>/_meta.json` for a `Checkpoint`
        (#75) / `ResponseCache` (#77) store, or `None` if not supplied / the
        store has no meta file yet. EXCLUDED from `manifest_id` -- identifies
        a resume mechanism, not the request itself.
    dedupe_stats : dict or None
        Snapshot of a `polar_llama.dedup.DedupeStats` instance
        (`rows_total`, `rows_null`, `cache_hits`, `rows_collapsed`,
        `calls_made`), or `None`. EXCLUDED from `manifest_id`.
    texts : dict or None
        Opt-in plaintext `{"system_prompt": ..., "prompt_template": ...}`,
        only populated when `build_manifest(..., store_texts=True)`.
        EXCLUDED from `manifest_id` (the hashes above already cover this
        content; storing plaintext is purely a replay convenience).
    """

    # ---- determinism-hashed fields (all enter manifest_id) ----
    format_version: int
    polar_llama_version: str
    symbol: str
    provider: str
    model: str
    config_fingerprint: str
    system_prompt_hash: Optional[str]
    prompt_template_hash: Optional[str]
    response_model_name: Optional[str]
    response_schema_hash: Optional[str]
    params: Dict[str, Any]
    seed: Optional[int]
    endpoint: Optional[str]
    row_count: Optional[int]

    # ---- identity + metadata (EXCLUDED from manifest_id) ----
    manifest_id: str
    created_at: str
    aggregate_usage: Optional[Dict[str, Any]] = None
    checkpoint_id: Optional[str] = None
    response_cache_id: Optional[str] = None
    dedupe_stats: Optional[Dict[str, Any]] = None
    texts: Optional[Dict[str, str]] = None

    def hashed_fields(self) -> Dict[str, Any]:
        """The subset of fields that determine `manifest_id`.

        Excludes `manifest_id` itself and every metadata field
        (`created_at`, `aggregate_usage`, `checkpoint_id`,
        `response_cache_id`, `dedupe_stats`, `texts`).
        """
        return {name: getattr(self, name) for name in _HASHED_FIELD_NAMES}

    def save(self, path: Union[str, Path]) -> Path:
        """See `save_manifest`."""
        return save_manifest(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RunManifest":
        """See `load_manifest`."""
        return load_manifest(path)


def _read_store_fingerprint(path: Union[str, Path]) -> Optional[str]:
    """Read the `fingerprint` recorded in `<path>/_meta.json`, or `None`.

    Reuses the exact `_meta.json` convention `polar_llama.checkpoint`
    (issue #75) writes and `polar_llama.dedup.ResponseCacheStore` (issue
    #77) inherits unmodified -- see `CheckpointStore._write_meta`. Returns
    `None` for a missing directory, missing/corrupt meta file, or a meta
    file with no `fingerprint` key (the admin-access case, `fingerprint=None`).
    """
    meta_path = Path(path) / _META_FILENAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return meta.get("fingerprint")


def _aggregate_usage(df: pl.DataFrame, usage_column: str) -> Dict[str, Any]:
    """Sum a `usage=True` output column's `usage` sub-struct (issue #76).

    Requires `usage_column` to be a `Struct{..., usage: USAGE_DTYPE}` column
    -- i.e. the run used `usage=True`. Null-safe: a row whose envelope is
    null (a null input row) contributes 0 to every sum and isn't counted in
    `rows_with_usage`.
    """
    if usage_column not in df.columns:
        raise ValueError(
            f"build_manifest: usage_column={usage_column!r} not found in df "
            f"columns {df.columns!r}"
        )
    dtype = df.schema[usage_column]
    field_names = [f.name for f in dtype.fields] if isinstance(dtype, pl.Struct) else []
    if "usage" not in field_names:
        raise ValueError(
            f"build_manifest: column {usage_column!r} has no 'usage' struct "
            "field -- aggregate_usage requires the run to have used "
            "usage=True (issue #76); got dtype "
            f"{dtype!r}"
        )

    usage_field = pl.col(usage_column).struct.field("usage")
    agg = df.select(
        input_tokens=usage_field.struct.field("input_tokens").sum(),
        output_tokens=usage_field.struct.field("output_tokens").sum(),
        cached_tokens=usage_field.struct.field("cached_tokens").sum(),
        cost_usd=usage_field.struct.field("cost_usd").sum(),
        rows_with_usage=usage_field.is_not_null().sum(),
    )
    row = agg.row(0, named=True)
    return {
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cached_tokens": row["cached_tokens"],
        "cost_usd": row["cost_usd"],
        "rows_with_usage": row["rows_with_usage"],
    }


def build_manifest(
    df: Optional[pl.DataFrame] = None,
    *,
    symbol: str = "inference_async",
    provider: Optional[Any] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    response_model: Optional[Type["BaseModel"]] = None,
    prompt_template: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    usage_column: Optional[str] = None,
    checkpoint: Optional[Union[str, Path, Checkpoint]] = None,
    response_cache: Optional[Union[str, Path, ResponseCache]] = None,
    dedupe_stats: Optional[DedupeStats] = None,
    store_texts: bool = False,
) -> RunManifest:
    """Build a `RunManifest` for a run's configuration.

    Purely post-hoc / descriptive -- this never calls the inference plugin
    itself. Call it after (or independently of) `inference_async`/
    `inference_messages`, passing the *same* configuration you passed (or
    intend to pass) to them, so `config_fingerprint` matches the runtime
    fingerprint (`polar_llama.keys.request_fingerprint`) exactly.

    Parameters
    ----------
    df : polars.DataFrame, optional
        The materialized output (or input) DataFrame -- supplies
        `row_count` (`df.height`). Required if `usage_column` is given.
    symbol : {"inference_async", "inference_messages"}
        Which entry point this manifest describes.
    provider, model : optional
        Passed through exactly like `inference_async`/`inference_messages`:
        a `polar_llama.Provider` (or anything duck-typed to `str()`) is
        accepted for `provider`; `None` normalizes to the library's fixed
        sentinel (never "whatever the current default happens to be") --
        reusing `polar_llama.keys`'s normalization, not reimplementing it.
    system_prompt : str, optional
        Hashed into `system_prompt_hash`; the plaintext is discarded unless
        `store_texts=True`.
    response_model : Type[BaseModel], optional
        Hashed via the same `_pydantic_to_json_schema` serialization
        `inference_async`/`inference_messages` use, so a manifest's
        `response_schema_hash` changes exactly when a real run's schema
        would.
    prompt_template : str, optional
        A user-supplied prompt template string, hashed into
        `prompt_template_hash`. Not otherwise interpreted -- `polar_llama`
        has no template-application step of its own; this is a place to
        record what template (if any) produced the rows sent to
        `inference_async`/`inference_messages`.
    params : dict, optional
        Client-forwarded sampling params (e.g. `{"temperature": 0.2}`).
        Recorded as given -- `{}` if omitted. See the module docstring's
        params/server-path caveat: hosted paths don't forward these today,
        so this documents *intent*, not necessarily *effect*.
    seed : int, optional
        Recorded for audit; see the module docstring's caveat.
    usage_column : str, optional
        Name of a `usage=True` output column in `df` (issue #76 --
        `Struct{response, usage: USAGE_DTYPE}`). When given, `df` must also
        be given, and `aggregate_usage` is computed by summing the column's
        `usage` sub-struct. Raises `ValueError` if the column doesn't have
        that shape.
    checkpoint : str, Path, or Checkpoint, optional
        A checkpoint store (issue #75) whose `<path>/_meta.json` fingerprint
        is read into `checkpoint_id`. `None` if the store has no meta file
        yet (e.g. an empty/new directory).
    response_cache : str, Path, or ResponseCache, optional
        Same, for a response-cache store (issue #77) -> `response_cache_id`.
    dedupe_stats : DedupeStats, optional
        A populated `polar_llama.DedupeStats` (issue #77) snapshotted into
        `dedupe_stats` (`rows_total`/`rows_null`/`cache_hits`/
        `rows_collapsed`/`calls_made`).
    store_texts : bool, optional
        When True, embeds the plaintext `system_prompt`/`prompt_template`
        into `RunManifest.texts` for self-contained replay (see the module
        docstring's "prompt-hash vs. replay tension" note). Default False --
        manifests are hash-only (no prompt IP) unless you opt in.

    Returns
    -------
    RunManifest
    """
    if symbol not in ("inference_async", "inference_messages"):
        raise ValueError(
            "build_manifest: symbol must be 'inference_async' or "
            f"'inference_messages', got {symbol!r}"
        )

    # Duck-typed Provider -> str, exactly like inference_async/inference_messages.
    if provider is not None and not isinstance(provider, str):
        provider = provider.as_str() if hasattr(provider, "as_str") else str(provider)

    response_schema_str: Optional[str] = None
    response_model_name: Optional[str] = None
    if response_model is not None:
        # Lazy import: `_pydantic_to_json_schema` lives in
        # `polar_llama/__init__.py`; importing it at module scope here would
        # be circular (see module docstring).
        from polar_llama import _pydantic_to_json_schema

        schema = _pydantic_to_json_schema(response_model)
        response_schema_str = json.dumps(schema)
        response_model_name = response_model.__name__

    kwargs = {
        "provider": provider,
        "model": model,
        "response_schema": response_schema_str,
        "response_model_name": response_model_name,
    }
    fingerprint, fingerprint_inputs = request_fingerprint(symbol, kwargs, system_prompt)
    endpoint = fingerprint_inputs["endpoint"]

    row_count = df.height if df is not None else None

    aggregate_usage = None
    if usage_column is not None:
        if df is None:
            raise ValueError("build_manifest: usage_column requires df to be provided")
        aggregate_usage = _aggregate_usage(df, usage_column)

    checkpoint_id = None
    if checkpoint is not None:
        ckpt_path = checkpoint.path if isinstance(checkpoint, Checkpoint) else Path(checkpoint)
        checkpoint_id = _read_store_fingerprint(ckpt_path)

    response_cache_id = None
    if response_cache is not None:
        rc_path = (
            response_cache.path if isinstance(response_cache, ResponseCache) else Path(response_cache)
        )
        response_cache_id = _read_store_fingerprint(rc_path)

    dedupe_stats_snapshot = None
    if dedupe_stats is not None:
        dedupe_stats_snapshot = {
            "rows_total": dedupe_stats.rows_total,
            "rows_null": dedupe_stats.rows_null,
            "cache_hits": dedupe_stats.cache_hits,
            "rows_collapsed": dedupe_stats.rows_collapsed,
            "calls_made": dedupe_stats.calls_made,
        }

    texts = None
    if store_texts:
        texts = {}
        if system_prompt is not None:
            texts["system_prompt"] = system_prompt
        if prompt_template is not None:
            texts["prompt_template"] = prompt_template

    try:
        polar_llama_version = importlib.metadata.version("polar-llama")
    except importlib.metadata.PackageNotFoundError:
        polar_llama_version = "0+unknown"

    hashed_fields: Dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "polar_llama_version": polar_llama_version,
        "symbol": symbol,
        "provider": provider if provider is not None else _DEFAULT_PROVIDER,
        "model": model if model is not None else _DEFAULT_MODEL,
        "config_fingerprint": fingerprint,
        "system_prompt_hash": _hash_text(system_prompt),
        "prompt_template_hash": _hash_text(prompt_template),
        "response_model_name": response_model_name,
        "response_schema_hash": _hash_text(response_schema_str),
        "params": params or {},
        "seed": seed,
        "endpoint": endpoint,
        # row_count intentionally excluded -- see _HASHED_FIELD_NAMES. This dict
        # must stay in sync with that tuple so the load-time integrity re-check
        # (RunManifest.hashed_fields) recomputes the same manifest_id.
    }
    manifest_id = _manifest_id(hashed_fields)
    created_at = datetime.now(timezone.utc).isoformat()

    return RunManifest(
        **hashed_fields,
        manifest_id=manifest_id,
        created_at=created_at,
        # row_count is recorded metadata, deliberately NOT part of hashed_fields
        # (so it never enters manifest_id) -- passed explicitly here.
        row_count=row_count,
        aggregate_usage=aggregate_usage,
        checkpoint_id=checkpoint_id,
        response_cache_id=response_cache_id,
        dedupe_stats=dedupe_stats_snapshot,
        texts=texts,
    )


def with_manifest_id(
    df: pl.DataFrame, manifest: RunManifest, *, column: str = "manifest_id"
) -> pl.DataFrame:
    """Attach `manifest.manifest_id` to every row of `df` as a constant Utf8 column.

    Pure bookkeeping -- `df.with_columns(pl.lit(manifest.manifest_id).alias(column))`
    -- so a result DataFrame (e.g. written to Parquet) carries a self-describing
    link back to the manifest that produced it. Every other column is untouched.
    """
    return df.with_columns(pl.lit(manifest.manifest_id).alias(column))


def save_manifest(manifest: RunManifest, path: Union[str, Path]) -> Path:
    """Write `manifest` as an indented, sorted-keys JSON sidecar at `path`.

    Atomic write (`<path>.tmp` + `os.replace`), the same pattern
    `polar_llama.checkpoint.CheckpointStore._write_meta` uses -- a crash
    mid-write leaves only an orphaned `.tmp` file, never a truncated `path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(manifest)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def load_manifest(path: Union[str, Path]) -> RunManifest:
    """Load a `RunManifest` from `path`, verifying its integrity.

    Recomputes `manifest_id` over the loaded determinism fields
    (`RunManifest.hashed_fields()`) and raises `ManifestIntegrityError` if it
    doesn't match the stored `manifest_id` -- a hand-edited or corrupted
    file is detected, not silently trusted. Raises `ValueError` if the
    file's `format_version` is newer than this release's
    `MANIFEST_FORMAT_VERSION` (an old `polar_llama` reading a manifest
    written by a newer one).
    """
    path = Path(path)
    payload = json.loads(path.read_text())

    format_version = payload.get("format_version")
    if format_version is None or format_version > MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"manifest at {path} has format_version={format_version!r}, newer "
            f"than this polar-llama's MANIFEST_FORMAT_VERSION="
            f"{MANIFEST_FORMAT_VERSION}; upgrade polar-llama to load it."
        )

    manifest = RunManifest(**payload)
    recomputed = _manifest_id(manifest.hashed_fields())
    if recomputed != manifest.manifest_id:
        raise ManifestIntegrityError(
            f"manifest at {path} failed its integrity check: stored "
            f"manifest_id {manifest.manifest_id!r} does not match the hash "
            f"recomputed over its determinism fields ({recomputed!r}) -- the "
            "file may have been hand-edited or corrupted."
        )
    return manifest


def _verify_replay_inputs(
    manifest: RunManifest,
    *,
    system_prompt: Optional[str],
    response_model: Optional[Type["BaseModel"]],
    prompt_template: Optional[str],
) -> None:
    """Raise `ManifestMismatchError` naming every diverging CONTENT field.

    Environment drift (endpoint, `polar_llama` version) is a separate,
    non-fatal `UserWarning` -- see `replay`.
    """
    mismatches = []

    if _hash_text(system_prompt) != manifest.system_prompt_hash:
        mismatches.append("system_prompt")

    if _hash_text(prompt_template) != manifest.prompt_template_hash:
        mismatches.append("prompt_template")

    response_schema_str: Optional[str] = None
    response_model_name: Optional[str] = None
    if response_model is not None:
        from polar_llama import _pydantic_to_json_schema

        response_schema_str = json.dumps(_pydantic_to_json_schema(response_model))
        response_model_name = response_model.__name__

    if _hash_text(response_schema_str) != manifest.response_schema_hash:
        mismatches.append("response_model (schema)")
    if response_model_name != manifest.response_model_name:
        mismatches.append("response_model (name)")

    # Belt-and-suspenders: recompute config_fingerprint the same way
    # `config_fingerprint` itself does, holding `endpoint` fixed at the
    # manifest's recorded value (not the current environment's) so this
    # stays a pure CONTENT check -- environment drift is checked/warned on
    # separately below.
    recomputed_fingerprint = config_fingerprint(
        symbol=manifest.symbol,
        provider=manifest.provider,
        model=manifest.model,
        response_schema=response_schema_str,
        response_model_name=response_model_name,
        system_prompt=system_prompt,
        extra={"endpoint": manifest.endpoint},
    )
    if recomputed_fingerprint != manifest.config_fingerprint:
        mismatches.append("config_fingerprint")

    if mismatches:
        raise ManifestMismatchError(
            f"replay: supplied inputs diverge from manifest "
            f"{manifest.manifest_id!r} in: {', '.join(sorted(set(mismatches)))}"
        )

    # Environment drift -- warn, don't raise (see module docstring / class docstring).
    current_endpoint = endpoint_fingerprint_input(manifest.provider)
    if current_endpoint != manifest.endpoint:
        warnings.warn(
            "replay: current endpoint "
            f"({current_endpoint!r}) differs from the manifest's recorded "
            f"endpoint ({manifest.endpoint!r}); replay may hit a different "
            "backend than the original run.",
            UserWarning,
            stacklevel=3,
        )
    try:
        current_version = importlib.metadata.version("polar-llama")
    except importlib.metadata.PackageNotFoundError:
        current_version = None
    if current_version is not None and current_version != manifest.polar_llama_version:
        warnings.warn(
            f"replay: current polar-llama version ({current_version}) "
            "differs from the manifest's recorded version "
            f"({manifest.polar_llama_version}); provider defaults or "
            "library behavior may have changed since the original run.",
            UserWarning,
            stacklevel=3,
        )


def replay(
    manifest: RunManifest,
    df: pl.DataFrame,
    input_column: str,
    *,
    system_prompt: Optional[str] = None,
    response_model: Optional[Type["BaseModel"]] = None,
    prompt_template: Optional[str] = None,
    verify: bool = True,
    _inference_fn: Optional[Callable[..., pl.Expr]] = None,
) -> pl.DataFrame:
    """Re-run the request a `RunManifest` describes, against re-supplied row data.

    The manifest never stores row data (or prompt/schema plaintext, unless
    `build_manifest(..., store_texts=True)` was used) -- see the module
    docstring's "prompt-hash vs. replay tension" note. This is
    **verify-then-replay**: the caller re-supplies `df`/`input_column` (the
    rows to run) and, unless `manifest.texts` covers it, the
    `system_prompt`/`prompt_template`/`response_model` too; `replay` proves
    (by hash) that what was supplied matches the manifest before re-running.

    Parameters
    ----------
    manifest : RunManifest
        The manifest describing the run to replay.
    df : polars.DataFrame
        Row data to run -- NOT re-derived from the manifest (it stores no
        rows). Must have `row_count == manifest.row_count` in spirit, but
        this is not enforced (re-running on a different row set is a
        legitimate use, e.g. replaying on newly-arrived rows).
    input_column : str
        Column of `df` holding the text (or, for an `inference_messages`
        manifest, the JSON message array) to send.
    system_prompt, response_model, prompt_template : optional
        Re-supplied inputs, verified against `manifest.system_prompt_hash` /
        `manifest.response_schema_hash`+`response_model_name` /
        `manifest.prompt_template_hash`. If `manifest.texts` is present
        (i.e. `store_texts=True` at build time) and `system_prompt` /
        `prompt_template` are omitted here, they are auto-filled from it --
        fully self-contained replay for callers who opted in.
    verify : bool, optional
        When True (default), raises `ManifestMismatchError` (naming every
        diverging field) if the supplied content doesn't hash-match the
        manifest; warns (doesn't raise) on endpoint/version drift. Set False
        to skip all of this and just re-issue the call.
    _inference_fn : callable, optional
        Test seam: the `inference_async`/`inference_messages`-shaped
        callable to invoke instead of resolving one from
        `manifest.symbol`. Default (`None`) lazily imports
        `polar_llama.inference_async`/`inference_messages` and picks the one
        matching `manifest.symbol`.

    Returns
    -------
    polars.DataFrame
        `df` with a `response` column (the reconstructed call's output) and
        a `manifest_id` column (via `with_manifest_id`) appended. Row count
        and every other original column are preserved.

    Notes
    -----
    Only what `inference_async`/`inference_messages` themselves accept is
    replayed: `provider`, `model`, `system_prompt` (async only --
    `inference_messages` has no such parameter; messages carry their own
    system turn), `response_model`, and `usage` (set to
    `manifest.aggregate_usage is not None`, matching whether the original
    run used `usage=True`). `params`/`seed` are NOT forwarded -- see the
    module docstring's params/server-path caveat: on hosted paths there is
    nothing to forward them *to*, so replay against current server defaults
    is the best this can do. Row-level data (this function's `df`) is never
    reconstructed from the manifest; the manifest never stored it.
    """
    system_prompt = (
        manifest.texts.get("system_prompt")
        if system_prompt is None and manifest.texts is not None
        else system_prompt
    )
    prompt_template = (
        manifest.texts.get("prompt_template")
        if prompt_template is None and manifest.texts is not None
        else prompt_template
    )

    if verify:
        _verify_replay_inputs(
            manifest,
            system_prompt=system_prompt,
            response_model=response_model,
            prompt_template=prompt_template,
        )

    if _inference_fn is not None:
        fn = _inference_fn
    else:
        # Lazy import -- see module docstring's circular-import note.
        from polar_llama import inference_async, inference_messages

        fn = inference_async if manifest.symbol == "inference_async" else inference_messages

    # build_manifest stores the keys.py sentinel "default" when model was None.
    # Forwarding it literally would make replay call a nonexistent model named
    # "default" instead of re-resolving the real provider default the original
    # run used, so map it back to None. (No real model is named "default".)
    # The provider "openai" sentinel is left as-is: unlike "default" it IS a
    # valid provider string that routes exactly where a None provider would.
    call_kwargs: Dict[str, Any] = {
        "provider": manifest.provider,
        "model": None if manifest.model == _DEFAULT_MODEL else manifest.model,
        "response_model": response_model,
        "usage": manifest.aggregate_usage is not None,
    }
    if manifest.symbol == "inference_async":
        call_kwargs["system_prompt"] = system_prompt

    out = df.with_columns(fn(pl.col(input_column), **call_kwargs).alias("response"))
    return with_manifest_id(out, manifest)
