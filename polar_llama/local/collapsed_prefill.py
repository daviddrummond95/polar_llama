"""Collapsed shared-prefix prefill for the in-process local (MLX) backend.

Problem this module solves
--------------------------
``mlx_lm.batch_generate(..., prompt_caches=...)`` does **not** skip prompt
tokens that are already in the cache: ``BatchGenerator.insert`` prefills
every prompt token it is given *on top of* whatever the cache already
contains (verified against mlx-lm 0.31.3, ``generate.py``:
``insert_segments`` -> ``PromptProcessingBatch.prompt``). A caller who warms
a shared-prefix cache and then passes the FULL prompts therefore

  1. re-prefills the shared prefix once per row (no compute collapse), and
  2. silently corrupts the context: the cached prefix occupies positions
     ``[0, P)`` and the full prompt is appended at ``[P, P + full)``, so the
     model attends over the prefix twice at shifted RoPE positions.

A second, subtler trap: deriving the prefix by rendering the system message
alone through the chat template. Templates are free to fold the system
prompt into the first user turn -- Gemma's template does exactly this, so
``apply_chat_template([system], tokenize=True)`` renders to just ``[<bos>]``
(ONE token) and the "shared prefix cache" is empty in practice. Even for
templates with a real system section, the rendered-prefix tokens are not
guaranteed to be a token-level prefix of the full prompt (BPE merges across
the seam; see ``polar_llama.local.prefix_cache.verify_token_boundary``).

The mechanism
-------------
This module follows the invariants of ``polar_llama.local.prefix_cache``
(token-boundary safety, immutable-prefix-first, no-trim) but enforces the
token-boundary invariant *by construction* instead of by post-hoc check:

  1. Tokenize every row's FULL prompt (template applied to system + user).
  2. The shared prefix is the token-level longest common prefix (LCP) of
     those token lists. By construction it is a true token prefix of every
     row, whatever the template did -- there is no seam to verify.
  3. Prefill the LCP ONCE into a single-sequence prompt cache
     (``mlx_lm.models.cache.make_prompt_cache`` + chunked ``model()`` calls,
     the exact shape of upstream ``generation`` prefill -- except we prefill
     ALL prefix tokens because the next token comes from the suffix, not
     from sampling).
  4. Hand ``batch_generate`` the per-row SUFFIXES with the prefix cache
     attached per row. mlx-lm natively supports "batching with history":
     ``KVCache.merge`` / ``RotatingKVCache.merge`` combine per-sequence
     caches into ``Batch*KVCache`` with per-row true offsets, and ragged
     suffixes are right-padded during prefill then rolled into left padding
     by ``cache.finalize()``. No monkeypatch is required -- only public
     mlx-lm APIs.

Cache replication and the no-trim invariant
-------------------------------------------
``Batch*KVCache.merge`` (mlx-lm 0.31.x) COPIES each per-sequence cache's KV
into freshly allocated batch arrays and never mutates its inputs, so the
same warm cache object can be attached to every row (``clone_per_row=False``,
the default) -- this avoids B deep copies of the prefix KV. Physical
replication into the batch arrays still happens inside ``merge`` (batched
attention needs per-lane KV); what this module collapses is the prefix
*compute*, which is the dominant cost. Reuse here is strictly
EXACT/EXTEND -- the cached prefix is always a true token prefix of every
row -- so the broken trim path (mlx-lm #980) is never taken.

CI safety: no ``mlx`` / ``mlx_lm`` import at module level. All planning
logic (:func:`common_token_prefix_len`, :func:`plan_collapsed_prefill`) is
pure Python and testable without Apple silicon.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

__all__ = [
    "DEFAULT_MIN_PREFIX_TOKENS",
    "DEFAULT_PREFILL_STEP_SIZE",
    "CollapsePlan",
    "CollapsedBatchResponse",
    "common_token_prefix_len",
    "plan_collapsed_prefill",
    "prefill_prompt_cache",
    "replicate_prefix_caches",
    "collapsed_batch_generate",
]

# Below this many shared tokens the warm-up pass costs more than it saves.
DEFAULT_MIN_PREFIX_TOKENS = 8
# Matches mlx-lm's default prompt-processing chunk size.
DEFAULT_PREFILL_STEP_SIZE = 2048


# ---------------------------------------------------------------------------
# Pure planning logic (CI-safe, no mlx)
# ---------------------------------------------------------------------------


def common_token_prefix_len(prompts: Sequence[Sequence[int]]) -> int:
    """Length of the token-level longest common prefix across all rows.

    Token-level LCP of the FULL tokenizations is the only prefix definition
    that is correct by construction: it cannot suffer BPE cross-seam merges
    (nothing is re-tokenized) and it is agnostic to how the chat template
    laid out the system prompt (Gemma folds it into the first user turn).
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    first = prompts[0]
    lcp = len(first)
    for row in prompts[1:]:
        n = min(lcp, len(row))
        i = 0
        while i < n and row[i] == first[i]:
            i += 1
        lcp = i
        if lcp == 0:
            break
    return lcp


@dataclass(frozen=True)
class CollapsePlan:
    """A validated plan splitting a batch at the shared-prefix seam.

    Invariant: for every row ``i``,
    ``list(prefix_tokens) + list(suffixes[i])`` equals the original full
    prompt, and every suffix is non-empty (the batch machinery needs at
    least one token per row to kick off generation).
    """

    prefix_tokens: tuple
    suffixes: tuple  # tuple of tuples, one per row

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_tokens)

    @property
    def rows(self) -> int:
        return len(self.suffixes)

    @property
    def naive_prefill_tokens(self) -> int:
        """Prompt tokens a non-collapsed batch would prefill."""
        return sum(self.prefix_len + len(s) for s in self.suffixes)

    @property
    def collapsed_prefill_tokens(self) -> int:
        """Prompt tokens actually computed with the collapse (prefix once)."""
        return self.prefix_len + sum(len(s) for s in self.suffixes)

    @property
    def saved_prefill_tokens(self) -> int:
        return self.naive_prefill_tokens - self.collapsed_prefill_tokens

    def suffix_lists(self) -> List[List[int]]:
        """Fresh mutable per-row suffix token lists for ``batch_generate``."""
        return [list(s) for s in self.suffixes]


def plan_collapsed_prefill(
    prompts: Sequence[Sequence[int]],
    *,
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS,
) -> Optional[CollapsePlan]:
    """Build a :class:`CollapsePlan`, or ``None`` when collapsing is not
    worthwhile (shared prefix shorter than ``min_prefix_tokens``).

    The prefix is clamped to ``min(len(row)) - 1`` so every row keeps a
    non-empty suffix even if some rows are identical.
    """
    if min_prefix_tokens < 1:
        raise ValueError("min_prefix_tokens must be >= 1")
    rows = [list(p) for p in prompts]
    if not rows:
        raise ValueError("prompts must be non-empty")
    for i, r in enumerate(rows):
        if len(r) == 0:
            raise ValueError(f"prompt {i} is empty")

    lcp = common_token_prefix_len(rows)
    lcp = min(lcp, min(len(r) for r in rows) - 1)
    if lcp < min_prefix_tokens:
        return None

    prefix = tuple(rows[0][:lcp])
    suffixes = tuple(tuple(r[lcp:]) for r in rows)
    return CollapsePlan(prefix_tokens=prefix, suffixes=suffixes)


def replicate_prefix_caches(
    prefix_cache: List[Any], n: int, *, clone_per_row: bool = False
) -> List[List[Any]]:
    """Row-aligned prompt-cache list for ``batch_generate(prompt_caches=...)``.

    ``clone_per_row=False`` (default) attaches the SAME cache object to every
    row. Safe with mlx-lm 0.31.x because ``Batch*KVCache.merge`` copies the
    per-sequence KV into new batch arrays and never mutates its inputs (the
    per-row caches are read exactly once, at merge time). Pass
    ``clone_per_row=True`` for a defensive deep copy per row (B x prefix-KV
    extra memory and copy time) if running against an unvetted mlx-lm.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if clone_per_row:
        return [copy.deepcopy(prefix_cache) for _ in range(n)]
    return [prefix_cache for _ in range(n)]


# ---------------------------------------------------------------------------
# MLX-backed mechanism (imports guarded inside functions)
# ---------------------------------------------------------------------------


def prefill_prompt_cache(
    model: Any,
    prefix_tokens: Sequence[int],
    *,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
) -> List[Any]:
    """Compute the KV cache for ``prefix_tokens`` ONCE (batch size 1).

    Mirrors upstream sequential prefill (chunked forward passes, eval of the
    cache state per chunk) with one deliberate difference: upstream keeps the
    last prompt token back for the first decode step, whereas here ALL
    prefix tokens go into the cache -- the token that follows the prefix is
    the first *suffix* token, supplied later by the batch machinery.

    Returns a per-layer list of single-sequence caches
    (``make_prompt_cache(model)``) whose offset is exactly
    ``len(prefix_tokens)``.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import make_prompt_cache

    tokens = list(prefix_tokens)
    if not tokens:
        raise ValueError("prefix_tokens must be non-empty")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be >= 1")

    caches = make_prompt_cache(model)
    y = mx.array(tokens)
    with mx.stream(generation_stream):
        while y.size > 0:
            n = min(prefill_step_size, y.size)
            model(y[:n][None], cache=caches)
            mx.eval([c.state for c in caches])
            mx.clear_cache()
            y = y[n:]

    # Cheap sanity check: every integer-offset cache must sit exactly at the
    # end of the prefix. (Batched offsets are arrays; single-sequence
    # KVCache/RotatingKVCache offsets are ints.)
    for i, c in enumerate(caches):
        off = getattr(c, "offset", None)
        if isinstance(off, int) and off != len(tokens):
            raise RuntimeError(
                f"prefix prefill inconsistency: cache {i} offset {off} != "
                f"prefix length {len(tokens)}"
            )
    return caches


@dataclass
class CollapsedBatchResponse:
    """Result of :func:`collapsed_batch_generate`.

    ``texts``/``stats``/``response`` mirror ``mlx_lm.batch_generate``'s
    ``BatchResponse``. ``stats.prompt_tokens`` counts only the tokens the
    batch machinery prefilled (the suffixes); the one-time prefix prefill is
    reported separately via ``plan.prefix_len`` and ``prefix_prefill_s``.
    """

    texts: List[str]
    stats: Any
    plan: Optional[CollapsePlan]
    used_collapse: bool
    prefix_prefill_s: float
    response: Any


def collapsed_batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    *,
    max_tokens: Any = 128,
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    clone_per_row: bool = False,
    **kwargs: Any,
) -> CollapsedBatchResponse:
    """``mlx_lm.batch_generate`` with the shared token prefix computed once.

    ``prompts`` are FULL per-row token id lists (chat template already
    applied). The shared prefix is derived, prefilled once, and only the
    per-row suffixes are handed to ``batch_generate`` together with the
    replicated prefix cache. Greedy outputs match the sequential reference
    up to floating-point near-ties (see
    ``benchmarks/validate_collapsed_prefill.py``).

    Falls back to a plain ``batch_generate`` call when the shared prefix is
    shorter than ``min_prefix_tokens``.

    ``kwargs`` are forwarded to ``batch_generate`` / ``BatchGenerator``
    (e.g. ``sampler``, ``prefill_batch_size``, ``completion_batch_size``).
    """
    from mlx_lm import batch_generate

    rows = [list(p) for p in prompts]
    plan = plan_collapsed_prefill(rows, min_prefix_tokens=min_prefix_tokens)

    if plan is None:
        resp = batch_generate(
            model,
            tokenizer,
            rows,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            **kwargs,
        )
        return CollapsedBatchResponse(
            texts=resp.texts,
            stats=resp.stats,
            plan=None,
            used_collapse=False,
            prefix_prefill_s=0.0,
            response=resp,
        )

    t0 = time.perf_counter()
    prefix_cache = prefill_prompt_cache(
        model, plan.prefix_tokens, prefill_step_size=prefill_step_size
    )
    prefix_prefill_s = time.perf_counter() - t0

    caches = replicate_prefix_caches(
        prefix_cache, plan.rows, clone_per_row=clone_per_row
    )
    resp = batch_generate(
        model,
        tokenizer,
        plan.suffix_lists(),
        prompt_caches=caches,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        **kwargs,
    )
    return CollapsedBatchResponse(
        texts=resp.texts,
        stats=resp.stats,
        plan=plan,
        used_collapse=True,
        prefix_prefill_s=prefix_prefill_s,
        response=resp,
    )
