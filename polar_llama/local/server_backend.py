"""``engine="server"`` backend for ``pl.col(...).llama.inference_local(...)``.

This is the "buy" path described in ``docs/local_mlx_backend.md``: instead of
building a new provider/HTTP stack for local models, it retargets Polar
Llama's *existing* async Rust fan-out (``inference_async`` / ``inference_messages``
in ``polar_llama/__init__.py``, backed by the ``ModelClient`` trait in
``src/model_client/mod.rs``) at a local OpenAI-compatible server by pointing
the ``OPENAI_BASE_URL`` environment variable (read per-request in
``src/model_client/openai.rs::api_endpoint``) at ``http://localhost:PORT``
instead of ``https://api.openai.com``.

No Rust changes are required. This module is a thin adapter: it does not
open sockets, retry, or parse HTTP responses itself -- all of that is the
existing OpenAI provider code path's job.

Launch a local server first, e.g.::

    # mlx_lm.server (ships with mlx-lm, Apple Silicon only)
    pip install mlx-lm
    mlx_lm.server --model mlx-community/gemma-4-e2b-it-4bit --port 8080

    # or vllm-mlx (https://github.com/waybarrios/vllm-mlx)
    pip install vllm-mlx
    vllm-mlx serve mlx-community/gemma-4-e2b-it-4bit --port 8080

Then, in Python::

    import polars as pl
    from polar_llama.local.server_backend import inference_local_server

    df = pl.DataFrame({"prompt": ["Summarize photosynthesis in one sentence."]})
    df = df.with_columns(
        answer=inference_local_server(
            pl.col("prompt"),
            model="mlx-community/gemma-4-e2b-it-4bit",
            base_url="http://localhost:8080",
        )
    )

This module never imports ``mlx`` / ``mlx_lm`` -- it only talks to whatever
OpenAI-compatible HTTP server you already started, so it works even without
the ``[local]`` extra installed and on non-Apple-Silicon machines.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import polars as pl

from polar_llama import (
    Provider,
    combine_messages,
    inference_async,
    inference_messages,
    string_to_message,
)
from polar_llama.utils import parse_into_expr

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

__all__ = [
    "set_local_endpoint",
    "inference_local_server",
    "start_server_backend",
]

# ASSUMPTION (documented per task instructions): the Provider enum is a
# closed Rust set (OpenAI/Anthropic/Gemini/Groq/Bedrock) with no dedicated
# "local"/"openai-compatible" variant. Since any server that implements
# POST /v1/chat/completions in the OpenAI request/response shape (mlx_lm.server,
# vllm-mlx, llama.cpp server, LM Studio, ...) is indistinguishable from OpenAI
# itself at the wire level, we reuse `Provider.OPENAI` and only change where
# it points via `OPENAI_BASE_URL`. This is exactly the mechanism
# `src/model_client/openai.rs::api_endpoint` already supports.
_LOCAL_PROVIDER = Provider.OPENAI


def set_local_endpoint(base_url: str) -> None:
    """Point the existing OpenAI provider path at a local server.

    Sets the ``OPENAI_BASE_URL`` environment variable, which
    ``src/model_client/openai.rs`` reads on every request (so this takes
    effect immediately, including for calls already built into a not-yet
    -collected ``LazyFrame``).

    Parameters
    ----------
    base_url : str
        e.g. ``"http://localhost:8080"`` for a local ``mlx_lm.server`` or
        ``vllm-mlx`` instance. A trailing ``/v1`` is not needed -- the Rust
        client appends ``/v1/chat/completions`` itself.

    Notes
    -----
    This mutates process-wide environment state (there is no per-call/
    thread-local scoping in the Rust client), so it is not safe to use this
    to juggle multiple *different* local endpoints concurrently from the
    same process. For that, manage ``OPENAI_BASE_URL`` yourself around each
    call instead.
    """
    if not base_url or not base_url.strip():
        raise ValueError(
            "base_url must be a non-empty URL, e.g. 'http://localhost:8080'"
        )
    os.environ["OPENAI_BASE_URL"] = base_url.strip().rstrip("/")


def _broadcast_system_column(prompt_expr: pl.Expr, system: str) -> pl.Expr:
    """Build a String column of ``system`` repeated to match ``prompt_expr``'s
    length, mirroring the precedent in ``tag_taxonomy`` (polar_llama/__init__.py)
    for turning a single Python string into a per-row column before feeding it
    into the ``string_to_message`` / ``combine_messages`` Rust plugins.
    """
    return prompt_expr.map_batches(
        lambda s: pl.Series(s.name, [system] * len(s), dtype=pl.Utf8),
        return_dtype=pl.Utf8,
    )


def inference_local_server(
    expr: "IntoExpr",
    *,
    model: str,
    system: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[Union[str, Sequence[str]]] = None,
    usage: bool = False,
    price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> pl.Expr:
    """``engine="server"`` implementation of ``.llama.inference_local(...)``.

    Routes ``expr`` through the existing ``inference_async`` /
    ``inference_messages`` Rust plugins with ``provider="openai"``, after
    pointing ``OPENAI_BASE_URL`` at a local OpenAI-compatible server. Returns
    String completions **in the original row order** -- the same contract as
    every other ``inference_*`` function in this library, because ordering
    is handled entirely by the existing Rust fan-out (each request tracks its
    row index and results are re-assembled positionally; see
    ``src/expressions.rs``).

    Parameters
    ----------
    expr : polars.Expr
        The prompt (user-turn) text expression.
    model : str
        The model name/path exactly as your local server expects it (e.g.
        ``"mlx-community/gemma-4-e2b-it-4bit"``).
    system : str, optional
        A system prompt, kept as a separate immutable prefix rather than
        concatenated into the user text -- this is the prefix a local
        server or in-process KV cache uses to decide what can be reused
        across rows. When provided, each row becomes a 2-message
        ``[system, user]`` array via ``string_to_message`` +
        ``combine_messages`` and is sent through ``inference_messages``.
        When omitted, ``expr`` is sent as a single user turn through
        ``inference_async``.
    base_url : str, optional
        If given, calls :func:`set_local_endpoint` for you. If omitted, an
        already-configured ``OPENAI_BASE_URL`` is reused as-is; if neither is
        set, this raises rather than silently falling through to the real
        ``https://api.openai.com``.
    max_tokens, temperature, top_p, stop
        Accepted for API parity with ``engine="in_process"``. **Known
        limitation / assumption:** the current Rust ``OpenAIClient``
        (``src/model_client/openai.rs::format_request_body``) intentionally
        does not put sampling parameters in the request body (some hosted
        models, e.g. o-series/gpt-5, reject them), so these are not yet
        forwarded to your local server's request body. Your server's own
        defaults apply. A non-default value triggers a one-time warning
        rather than being silently dropped. Forwarding these will require
        extending ``InferenceKwargs``/``format_request_body`` in Rust --
        tracked as follow-up work, out of scope for this thin adapter.
    usage : bool, optional
        Issue #76: forwarded as-is to ``inference_async``/``inference_messages``,
        so the same JSON-envelope mechanism applies unchanged -- the local
        server's own reported ``usage`` block (if any) is parsed by the Rust
        OpenAI client exactly like a real OpenAI response. Cost resolution
        uses ``provider="openai"`` pricing, which will typically MISS for a
        local model name (giving a null ``cost_usd`` plus a one-time
        warning) since local models aren't in the packaged price table --
        pass ``price_table=`` to register a cost for your local model if you
        want a non-null ``cost_usd``. When True, the return dtype changes
        from ``Utf8`` to ``Struct{response: Utf8, usage: USAGE_DTYPE}``
        (see ``inference_async``'s ``usage`` parameter).
    price_table : dict, str, or Path, optional
        Per-call cost-resolution override (``usage=True`` only).

    Returns
    -------
    polars.Expr
        String completions, one per input row, in original row order
        (or a usage-struct column when ``usage=True``; see above).
    """
    if base_url is not None:
        set_local_endpoint(base_url)
    elif not os.environ.get("OPENAI_BASE_URL"):
        raise ValueError(
            'inference_local_server(engine="server") needs a local '
            "OpenAI-compatible endpoint. Pass base_url=... or set "
            "OPENAI_BASE_URL yourself before calling this (see "
            "docs/local_mlx_backend.md) -- otherwise this would silently "
            "fall through to the real https://api.openai.com."
        )

    if max_tokens != 512 or temperature != 0.0 or top_p != 1.0 or stop is not None:
        warnings.warn(
            "inference_local_server: max_tokens/temperature/top_p/stop are "
            'accepted for API parity with engine="in_process" but are not '
            "yet forwarded to the request body by the Rust OpenAI client "
            "(src/model_client/openai.rs). Your local server's own defaults "
            "will be used instead.",
            stacklevel=2,
        )

    prompt_expr = parse_into_expr(expr)

    if system is None:
        return inference_async(
            prompt_expr,
            provider=_LOCAL_PROVIDER,
            model=model,
            usage=usage,
            price_table=price_table,
        )

    system_expr = _broadcast_system_column(prompt_expr, system)
    messages_expr = combine_messages(
        string_to_message(system_expr, message_type="system"),
        string_to_message(prompt_expr, message_type="user"),
    )
    return inference_messages(
        messages_expr,
        provider=_LOCAL_PROVIDER,
        model=model,
        usage=usage,
        price_table=price_table,
    )


# Alias matching the name `polar_llama/local/__init__.py` lazily re-exports
# (`_LAZY_ATTRS["start_server_backend"] = "server_backend"`). Kept as a plain
# alias -- not a separate implementation -- so there is exactly one code path
# to maintain.
start_server_backend = inference_local_server
