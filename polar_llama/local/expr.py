"""Polars expression entry point for the local inference backend.

Public shape (conformed to by all agents)::

    pl.col("prompt").llama.inference_local(
        model, *, system=None, engine="server", base_url=None,
        max_tokens=512, temperature=0.0, top_p=1.0, stop=None,
    ) -> pl.Expr  # String completions, in ORIGINAL ROW ORDER

Two backends:

* ``engine="server"`` (default, low-risk): delegates to
  :func:`polar_llama.local.server_backend.inference_local_server`, which routes
  through the existing async fan-out to a local OpenAI-compatible endpoint by
  setting ``OPENAI_BASE_URL``. That backend is owned by a separate agent and is
  imported lazily so this module imports cleanly even before it exists.
* ``engine="in_process"``: a pure-Python ``map_batches`` UDF over the column
  that builds prompts (system prefix + row text), fetches the process-global
  singleton engine via :func:`polar_llama.local.engine.get_engine`, and returns
  a String Series in the original row order. No Rust involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import polars as pl

from polar_llama.local.engine import (
    generate_chunked,
    generate_chunked_with_usage,
    get_engine,
)

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


# Prompt sections are joined immutable-prefix-first (system before user text) so
# the system string forms a stable prefix -- the basis for prefix caching on the
# real engine.
_PROMPT_SEPARATOR = "\n\n"


def _parse_into_expr(expr: "IntoExpr") -> pl.Expr:
    """Parse an input into an expression (str -> column name)."""
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _build_prompt(text: Optional[str], system: Optional[str]) -> Optional[str]:
    """Combine the (optional) system prefix with the row text.

    Returns ``None`` for null input so the output row stays null.
    """
    if text is None:
        return None
    if system:
        return system + _PROMPT_SEPARATOR + text
    return text


def inference_local(
    expr: "IntoExpr",
    *,
    model: str,
    system: Optional[str] = None,
    engine: str = "server",
    base_url: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    usage: bool = False,
    price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> pl.Expr:
    """Run local inference over a text column, returning String completions.

    Parameters
    ----------
    expr
        The text expression (or column name) to complete.
    model
        Model identifier (an OpenAI-compatible model name for ``engine="server"``
        or an mlx-lm model path/repo for ``engine="in_process"``).
    system
        Optional system prompt. Kept as a SEPARATE argument so it forms an
        immutable prefix (the prefix-cache key basis).
    engine
        ``"server"`` (default) routes through the existing async fan-out to a
        local OpenAI-compatible endpoint. ``"in_process"`` uses the in-process
        mlx engine via a ``map_batches`` UDF (guarded behind the ``[local]``
        extra).
    base_url
        Base URL of the local OpenAI-compatible server (``engine="server"``
        only).
    max_tokens, temperature, top_p, stop
        Sampling parameters.
    usage
        Issue #76: enable per-row usage/cost accounting. When True, the
        return dtype changes from ``Utf8`` to
        ``Struct{response: Utf8, usage: USAGE_DTYPE}``.
        For ``engine="in_process"``, tokens are whitespace counts of the
        prompt/completion (see ``LocalEngine.generate_with_usage``),
        ``latency_ms`` is measured with ``time.perf_counter()``, and
        ``cost_usd`` is ALWAYS ``0.0`` (no network cost for a model running
        on this machine). For ``engine="server"``, this is forwarded as-is
        to ``inference_local_server`` -- see its ``usage`` parameter for the
        (non-zero-cost-by-default) semantics there.
    price_table
        Per-call cost-resolution override (``usage=True``, ``engine="server"``
        only -- ignored for ``engine="in_process"``, whose cost is always 0.0).

    Returns
    -------
    polars.Expr
        A String expression of completions, in the original row order (or a
        usage-struct column when ``usage=True``; see above).
    """
    if engine == "server":
        # Imported lazily: this backend is owned by a separate agent and may not
        # exist yet. Import inside the function so this module always imports.
        from polar_llama.local.server_backend import inference_local_server

        return inference_local_server(
            expr,
            model=model,
            system=system,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            usage=usage,
            price_table=price_table,
        )

    if engine in ("in_process", "mlx", "mlx_batch", "fake"):
        return _inference_local_in_process(
            expr,
            model=model,
            system=system,
            engine=engine,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            usage=usage,
        )

    raise ValueError(f"Unknown engine {engine!r}; expected 'server' or 'in_process'.")


def _inference_local_in_process(
    expr: "IntoExpr",
    *,
    model: str,
    system: Optional[str],
    engine: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    usage: bool = False,
) -> pl.Expr:
    parsed = _parse_into_expr(expr)
    stop_list: Optional[List[str]] = list(stop) if stop else None

    if usage:
        # Imported lazily (not at module top-level) so this module keeps
        # importing cleanly even in the unlikely event of a future circular
        # import; by the time `inference_local` is called, `polar_llama`'s
        # top-level package is always already fully initialized (it can only
        # reach this lazily-imported module through its own __init__).
        from polar_llama import USAGE_DTYPE

        full_dtype = pl.Struct({"response": pl.Utf8, "usage": USAGE_DTYPE})

        def _udf_with_usage(series: pl.Series) -> pl.Series:
            texts = series.to_list()

            prompts: List[str] = []
            positions: List[int] = []
            for idx, text in enumerate(texts):
                prompt = _build_prompt(text, system)
                if prompt is None:
                    continue
                positions.append(idx)
                prompts.append(prompt)

            out: List[Optional[Dict[str, Any]]] = [None] * len(texts)

            if prompts:
                local_engine = get_engine(model, engine)
                completions = generate_chunked_with_usage(
                    local_engine,
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                )
                for pos, (text_out, row_usage) in zip(positions, completions):
                    out[pos] = {
                        "response": text_out,
                        "usage": {
                            "input_tokens": row_usage.get("input_tokens"),
                            "output_tokens": row_usage.get("output_tokens"),
                            "cached_tokens": None,
                            "latency_ms": row_usage.get("latency_ms"),
                            # No network cost for a model running on this
                            # machine -- always 0.0, never looked up.
                            "cost_usd": 0.0,
                        },
                    }

            return pl.Series(series.name, out, dtype=full_dtype)

        return parsed.map_batches(_udf_with_usage, return_dtype=full_dtype)

    def _udf(series: pl.Series) -> pl.Series:
        texts = series.to_list()

        # Build prompts only for non-null rows; remember their positions so the
        # completions can be scattered back into the original order with nulls
        # preserved.
        prompts: List[str] = []
        positions: List[int] = []
        for idx, text in enumerate(texts):
            prompt = _build_prompt(text, system)
            if prompt is None:
                continue
            positions.append(idx)
            prompts.append(prompt)

        out: List[Optional[str]] = [None] * len(texts)

        if prompts:
            # Fetch the process-global singleton engine (never construct one per
            # morsel). Under the streaming engine this UDF may be called once per
            # morsel; get_engine loads the model exactly once per process.
            local_engine = get_engine(model, engine)
            completions = generate_chunked(
                local_engine,
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_list,
            )
            for pos, completion in zip(positions, completions):
                out[pos] = completion

        return pl.Series(series.name, out, dtype=pl.String)

    return parsed.map_batches(_udf, return_dtype=pl.String)
