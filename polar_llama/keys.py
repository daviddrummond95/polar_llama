"""Reusable content-hashing primitives.

These are the shared building blocks behind two features:

- Issue #75 (resumable checkpointing, `polar_llama/checkpoint.py`): a row is
  "done" if its content key is already present in the checkpoint store.
- Issue #77 (dedupe / persistent cross-job cache): rows -- possibly from
  different DataFrames, even different processes -- are considered the same
  request iff they produce the same content key.

Kept in a standalone module (not inside `checkpoint.py`) so #77 can import
just these pure functions without pulling in any checkpoint store I/O.

Hashing uses `hashlib.sha256` over canonical JSON, deliberately **not**
`pl.Expr.hash()` / `pl.Series.hash()` -- Polars' hash is seed- and
version-dependent (it is explicitly documented as unstable across Polars
versions and even process runs), which would silently break resume: a row
computed and stored under one Polars version could hash differently after an
upgrade and never be recognized as already-done.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import polars as pl

# Provider -> the env var that overrides its request endpoint. The Rust client
# reads these per request (openai.rs / anthropic.rs / groq.rs), so the same
# provider/model/prompt can be routed to a *different* backend (e.g. real
# OpenAI vs the local-MLX server, which sets OPENAI_BASE_URL) between runs.
# That endpoint is therefore a request-shaping input and must enter the
# fingerprint, or a resume could return results computed against another host.
_PROVIDER_BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "groq": "GROQ_BASE_URL",
}


def endpoint_fingerprint_input(provider: Optional[str]) -> Optional[str]:
    """The provider's active base-URL override (from the environment), or None.

    Folded into :func:`config_fingerprint` via ``extra`` so a checkpoint keyed
    on a run against one endpoint never yields a stale hit for a run against a
    different endpoint. Providers without a base-URL override (gemini, bedrock)
    return None.
    """
    if not provider:
        return None
    env = _PROVIDER_BASE_URL_ENV.get(provider.lower())
    return os.environ.get(env) if env else None


#: Normalization applied to `provider`/`model` when not explicitly passed --
#: matters for the fingerprint, since two runs of "whatever the library
#: default happens to be" must hash identically. Documented consequence: if a
#: future release changes the library's default provider/model, checkpoints
#: created against the old default will look unchanged (same fingerprint)
#: even though the *effective* request changed. Callers who care should pass
#: `model=` explicitly.
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "default"


def config_fingerprint(
    *,
    symbol: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    response_schema: Optional[str] = None,
    response_model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Compute a sha256 hex digest of every request-shaping parameter.

    Parameters
    ----------
    symbol
        Which Rust plugin entry point is being invoked (``"inference_async"``
        vs. ``"inference_messages"``) -- these must never share a key space
        since they interpret the row input differently.
    provider, model
        Passed through as given; ``None`` is normalized to a fixed sentinel
        (see module docstring) rather than being resolved to the library's
        actual current default, so the fingerprint is a pure function of the
        caller-visible arguments.
    response_schema
        The already-serialized JSON schema string (from
        ``_pydantic_to_json_schema`` / ``tools_to_response_model``). Any
        change to a Pydantic ``response_model`` -- or a tool-use schema --
        changes this string and therefore invalidates old checkpoint
        entries, satisfying "a schema change forces recompute".
    response_model_name
        Included in addition to the schema so that two schemas which
        happen to serialize identically but come from differently-named
        models are still distinguished (defensive; schema equality already
        implies this in practice).
    system_prompt
        Included -- unlike ``cache_*`` kwargs -- because it changes the
        actual content sent to the model.
    extra
        Escape hatch for future request-shaping kwargs that should
        participate in the fingerprint without changing this function's
        signature.

    Notes
    -----
    Explicitly **excluded**: all ``cache_*`` kwargs (``cache``,
    ``cache_strategy``, ``cache_ttl``, ``cache_key``, ``cache_min_tokens``).
    Provider prompt caching is a transport-level optimization -- it changes
    how a request is *sent*, never what is being asked for -- so toggling it
    must not invalidate a checkpoint or the persistent cache.

    Examples
    --------
    >>> a = config_fingerprint(symbol="inference_async", provider="openai", model="gpt-4o-mini")
    >>> b = config_fingerprint(symbol="inference_async", provider="openai", model="gpt-4o-mini")
    >>> a == b
    True
    >>> c = config_fingerprint(symbol="inference_async", provider="openai", model="gpt-4o")
    >>> a == c
    False
    >>> # None (library default) normalizes to a fixed sentinel, not "whatever
    >>> # the current default happens to be":
    >>> d = config_fingerprint(symbol="inference_async")
    >>> e = config_fingerprint(symbol="inference_async", provider=None, model=None)
    >>> d == e
    True
    """
    payload = {
        "symbol": symbol,
        "provider": provider if provider is not None else _DEFAULT_PROVIDER,
        "model": model if model is not None else _DEFAULT_MODEL,
        "response_schema": response_schema,
        "response_model_name": response_model_name,
        "system_prompt": system_prompt,
        "extra": extra or {},
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_key(row_input: str, fingerprint: str) -> str:
    """Compute the per-row content key: ``sha256(fingerprint \\0 row_input)``.

    The fingerprint is embedded in every key, so a config change (which
    produces a different fingerprint) automatically produces disjoint keys
    -- old checkpoint entries simply never match new rows, with no explicit
    invalidation/deletion step required.

    Examples
    --------
    >>> k1 = content_key("hello", "fp1")
    >>> k2 = content_key("hello", "fp1")
    >>> k1 == k2
    True
    >>> k3 = content_key("hello", "fp2")
    >>> k1 == k3
    False
    """
    h = hashlib.sha256()
    h.update(fingerprint.encode("utf-8"))
    h.update(b"\x00")
    h.update(row_input.encode("utf-8"))
    return h.hexdigest()


def content_keys(s: pl.Series, fingerprint: str) -> pl.Series:
    """Row-wise ``content_key`` over a Utf8 series. Null rows stay null.

    This is the reusable primitive #77 is expected to call directly to group
    rows by content across an entire DataFrame (or across DataFrames /
    processes, since the key only depends on the fingerprint + row text).
    """

    def _key(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return content_key(v, fingerprint)

    return s.map_elements(_key, return_dtype=pl.Utf8)


def canonicalize_messages_input(v: Any) -> str:
    """Canonicalize a `inference_messages` row to a compact JSON string.

    `inference_messages` accepts two input shapes for the same logical
    content: a JSON-encoded string (as produced by
    ``combine_messages``/``string_to_message``) or a native
    ``List(Struct{role, content})`` column. Both must hash to the same
    content key for identical conversations, so this parses whichever shape
    arrived and re-serializes it deterministically (sorted keys, no
    whitespace).

    Falls back to hashing the raw string as-is if it isn't valid JSON
    (defensive -- should not happen for well-formed message-array input).
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    else:
        # Already a native Python value (list of dicts) from a
        # List(Struct) column via `Series.to_list()`.
        parsed = v
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
