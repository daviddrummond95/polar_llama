"""Prefix-cache layer for the in-process local (MLX) backend.

Correctness-critical module. The invariants enforced here exist because a
violated prefix cache does not crash -- it silently produces wrong output:

1. **Token-boundary invariant.** KV reuse is only valid for a TRUE
   token-level prefix of the rendered prompt (RoPE binds each KV entry to
   an absolute position). BPE is context-dependent, so a raw-text prefix
   match does NOT guarantee token identity: ``encode(prefix + suffix)`` may
   merge across the seam. Every reuse must pass
   :func:`verify_token_boundary` (or the row falls back to a cold prefill).

2. **No-trim invariant.** mlx-lm's trim path for hybrid
   (sliding-window / rotating) caches is broken upstream (mlx-lm issue
   #980, closed "outdated" without a merged fix). The only safe operations
   against a cached prefix are EXACT reuse and EXTEND (the cached prefix is
   a true prefix of the new one). A cached prefix that is LONGER than the
   new prompt is classified DIVERGE and is never trimmed -- we recompute
   (``on_longer_cache="skip"``) or raise (``"raise"``), configurable per
   store.

3. **Immutable-prefix-first.** The public API takes ``system`` (and
   optional few-shot turns) separately from the per-row ``user`` text, and
   :func:`assemble_prompt` renders them through two *separate* callables so
   per-row content structurally cannot leak into the cache-key prefix.

This module is pure Python and CI-safe: it never imports ``mlx`` at module
level. Tokenizers, engines and cache-clone functions are injected so all
logic is testable with fakes (see ``polar_llama/local/parity.py``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple

__all__ = [
    "MatchClass",
    "PromptParts",
    "PrefixKey",
    "PrefixEntry",
    "PrefixStore",
    "PrefixCacheError",
    "TrimNotSupportedError",
    "TokenBoundaryError",
    "assemble_prompt",
    "classify",
    "verify_token_boundary",
    "error_payload",
    "require_mlx",
]

_MLX_INSTALL_HINT = (
    "the in-process local backend requires the optional [local] extra: "
    "pip install 'polar-llama[local]' (Apple Silicon + macOS only)"
)


# ---------------------------------------------------------------------------
# Errors and error payloads
# ---------------------------------------------------------------------------


class PrefixCacheError(RuntimeError):
    """Base class for prefix-cache violations."""


class TrimNotSupportedError(PrefixCacheError):
    """Raised when reuse would require trimming a longer cached prefix.

    Trimming is deliberately unsupported: the upstream trim path for hybrid
    sliding-window caches is broken (mlx-lm #980) and a wrong trim yields
    silently corrupted generations. Recompute instead.
    """


class TokenBoundaryError(PrefixCacheError):
    """Raised when a text-level prefix match fails the token-level check."""


def error_payload(error_type: str, details: str, raw: Optional[str] = None) -> str:
    """Per-row error JSON, mirroring ``create_error_response`` in
    ``src/model_client/mod.rs`` so downstream parsing is uniform across the
    Rust and local backends. Row failures must return this payload instead
    of killing the batch."""
    obj: dict = {"_error": error_type, "_details": details}
    if raw is not None:
        obj["_raw"] = raw
    return json.dumps(obj)


def require_mlx() -> Any:
    """Import and return ``mlx_lm`` or raise a helpful ImportError.

    Never called at module import time; only from code paths that actually
    need a real MLX engine.
    """
    try:
        import mlx_lm  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only sans mlx
        raise ImportError(_MLX_INSTALL_HINT) from exc
    return mlx_lm


# ---------------------------------------------------------------------------
# Prompt assembly: immutable prefix first, structurally enforced
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptParts:
    """A rendered prompt split at the cache seam.

    ``prefix`` is the immutable, shared part (system + few-shot) -- the
    basis of the cache key. ``suffix`` is the per-row part. The full prompt
    is always exactly ``prefix + suffix``; nothing may be inserted between
    them, or cached KV positions would no longer line up.
    """

    prefix: str
    suffix: str

    @property
    def full(self) -> str:
        return self.prefix + self.suffix


def _default_render_prefix(
    system: Optional[str], few_shot: Optional[Sequence[Tuple[str, str]]]
) -> str:
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}\n")
    for user_turn, assistant_turn in few_shot or ():
        parts.append(f"<|user|>\n{user_turn}\n<|assistant|>\n{assistant_turn}\n")
    return "".join(parts)


def _default_render_suffix(user: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n"


def assemble_prompt(
    system: Optional[str],
    few_shot: Optional[Sequence[Tuple[str, str]]],
    user: str,
    *,
    render_prefix: Callable[[Optional[str], Optional[Sequence[Tuple[str, str]]]], str] = _default_render_prefix,
    render_suffix: Callable[[str], str] = _default_render_suffix,
) -> PromptParts:
    """Render a prompt with the immutable prefix first.

    Immutability is enforced *structurally*: ``render_prefix`` never sees
    the per-row ``user`` text and ``render_suffix`` never sees
    ``system``/``few_shot``, so no template bug can leak row content into
    the cache-key prefix. When using a real chat template, pass callables
    that apply the model's template to the respective message slices --
    both must operate on the same rendered-token stream (the prefix render
    must be a literal string prefix of rendering the full conversation).

    Note the default template ends the prefix at a ``\\n`` turn boundary,
    which reduces (but does NOT eliminate) BPE cross-seam merges; callers
    must still run :func:`verify_token_boundary` before any KV reuse.
    """
    if not isinstance(user, str):
        raise TypeError(f"user must be str, got {type(user).__name__}")
    prefix = render_prefix(system, few_shot)
    suffix = render_suffix(user)
    return PromptParts(prefix=prefix, suffix=suffix)


# ---------------------------------------------------------------------------
# Cache keys and classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixKey:
    """Cache key: (model, sha256 of the rendered-prefix TEXT).

    Keyed on text, not tokens, on purpose: token ids depend on tokenizer
    revision and encode flags, and the token-boundary check is performed
    separately at reuse time. Hashing the exact rendered text makes the key
    deterministic and cheap while EXACT classification still compares full
    strings (no reliance on hash uniqueness for correctness).
    """

    model: str
    digest: str

    @classmethod
    def from_text(cls, model: str, prefix_text: str) -> "PrefixKey":
        digest = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
        return cls(model=model, digest=digest)


class MatchClass(Enum):
    """Relationship between a cached prefix and a new prompt's prefix."""

    EXACT = "exact"  # identical text: reuse as-is
    EXTEND = "extend"  # cached is a strict text-prefix of new: safe extend
    DIVERGE = "diverge"  # anything else, INCLUDING cached-longer-than-new


def classify(cached_prefix_text: str, new_prefix_text: str) -> MatchClass:
    """Classify reuse safety of a cached prefix against a new prefix.

    Only EXACT and EXTEND permit KV reuse (and EXTEND additionally requires
    :func:`verify_token_boundary` on the extension seam). The case where
    the *cached* text is longer -- i.e. the new prompt is a prefix of the
    cache -- is deliberately DIVERGE, not "trim": trimming is the broken
    path (mlx-lm #980) and is never taken.
    """
    if cached_prefix_text == new_prefix_text:
        return MatchClass.EXACT
    if new_prefix_text.startswith(cached_prefix_text):
        return MatchClass.EXTEND
    return MatchClass.DIVERGE


def verify_token_boundary(tokenizer: Any, prefix: str, suffix: str) -> bool:
    """True iff ``prefix``/``suffix`` split on a real token boundary.

    Checks ``encode(prefix + suffix)[:len(encode(prefix))] == encode(prefix)``.
    This is the load-bearing BPE check: greedy/BPE tokenization is
    context-dependent, so a merge can span the seam (e.g. a vocab entry
    ``"walking"`` swallowing ``"walk" + "ing"``), in which case cached KV
    computed for ``encode(prefix)`` does not correspond to any prefix of
    the tokens actually fed to the model and reuse would be silently wrong.

    ``tokenizer`` needs only an ``encode(text) -> Sequence[int]`` method.
    It must not append terminal special tokens (a leading BOS added
    consistently to both encodings is harmless: it appears at the same
    positions in both and the prefix comparison still holds).

    An empty prefix trivially passes (cold start, nothing reused).
    """
    if prefix == "":
        return True
    prefix_tokens = list(tokenizer.encode(prefix))
    full_tokens = list(tokenizer.encode(prefix + suffix))
    return full_tokens[: len(prefix_tokens)] == prefix_tokens


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass
class PrefixEntry:
    """A computed prompt cache for one (model, prefix)."""

    key: PrefixKey
    prefix_text: str
    token_count: int
    cache_obj: Any
    # How to produce an independent per-row copy of cache_obj. Injected so
    # fakes can use deepcopy while a real MLX engine supplies a state-level
    # clone of its layer-cache list. Independence matters: batch_generate
    # mutates each row's cache in place during decode.
    clone: Callable[[Any], Any] = copy.deepcopy

    def replicate_into_batch(self, n: int) -> list:
        """Return ``n`` row-aligned, independent copies of the cache.

        Suitable for ``batch_generate(..., prompt_caches=[...])``: element
        ``i`` belongs to row ``i``. NOTE the memory trade-off: replication
        costs B x prefix-KV on full-attention layers (mitigated on Gemma 4
        class models where 4 of 5 layers are sliding-window capped at 512
        tokens and later layers share KV).
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return [self.clone(self.cache_obj) for _ in range(n)]


class PrefixStore:
    """Process-wide store of computed prefix caches, keyed per (model, prefix).

    Thread-safe: under the streaming engine a map_batches UDF may run
    per-morsel, so lookups/inserts take a lock. Small LRU (``max_entries``)
    because each entry pins prefix-length KV in memory.

    ``on_longer_cache`` sets the no-trim policy when the best cached
    candidate is longer than the new prefix (the case trimming would
    "solve"): ``"skip"`` (default) treats it as a miss and recomputes --
    never wrong, only slower; ``"raise"`` raises
    :class:`TrimNotSupportedError` for callers that want loud failure.
    """

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        max_entries: int = 8,
        on_longer_cache: str = "skip",
    ) -> None:
        if on_longer_cache not in ("skip", "raise"):
            raise ValueError(
                f'on_longer_cache must be "skip" or "raise", got {on_longer_cache!r}'
            )
        self._tokenizer = tokenizer
        self._max_entries = max_entries
        self._on_longer_cache = on_longer_cache
        self._entries: "OrderedDict[PrefixKey, PrefixEntry]" = OrderedDict()
        self._lock = threading.Lock()

    # -- basic ops ---------------------------------------------------------

    def put(
        self,
        model: str,
        prefix_text: str,
        cache_obj: Any,
        token_count: int,
        *,
        clone: Callable[[Any], Any] = copy.deepcopy,
    ) -> PrefixEntry:
        key = PrefixKey.from_text(model, prefix_text)
        entry = PrefixEntry(
            key=key,
            prefix_text=prefix_text,
            token_count=token_count,
            cache_obj=cache_obj,
            clone=clone,
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return entry

    def get_exact(self, model: str, prefix_text: str) -> Optional[PrefixEntry]:
        key = PrefixKey.from_text(model, prefix_text)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                # Guard against (astronomically unlikely) hash collision:
                # correctness never rests on the digest alone.
                if entry.prefix_text != prefix_text:
                    return None
                self._entries.move_to_end(key)
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- reuse decision ----------------------------------------------------

    def match(
        self, model: str, new_prefix_text: str
    ) -> Tuple[MatchClass, Optional[PrefixEntry]]:
        """Find the best reusable entry for ``new_prefix_text``.

        Returns ``(EXACT, entry)`` on an exact hit; ``(EXTEND, entry)`` when
        a cached prefix is a strict text-prefix of the new one (longest such
        wins; caller must still pass :func:`verify_token_boundary` on
        ``(entry.prefix_text, extension)`` before reuse); otherwise
        ``(DIVERGE, None)``.

        No-trim policy: if the only relationship available is a cached
        prefix that is LONGER than the new one, we do not trim it --
        per ``on_longer_cache`` we either treat it as a miss ("skip") or
        raise TrimNotSupportedError ("raise").
        """
        exact = self.get_exact(model, new_prefix_text)
        if exact is not None:
            return (MatchClass.EXACT, exact)

        best: Optional[PrefixEntry] = None
        saw_longer_candidate = False
        with self._lock:
            for entry in self._entries.values():
                if entry.key.model != model:
                    continue
                cls = classify(entry.prefix_text, new_prefix_text)
                if cls is MatchClass.EXTEND:
                    if best is None or len(entry.prefix_text) > len(best.prefix_text):
                        best = entry
                elif cls is MatchClass.DIVERGE and entry.prefix_text.startswith(
                    new_prefix_text
                ):
                    # The trim-tempting case: cache is longer than the new
                    # prompt. Never trimmed (mlx-lm #980).
                    saw_longer_candidate = True

        if best is not None:
            return (MatchClass.EXTEND, best)
        if saw_longer_candidate and self._on_longer_cache == "raise":
            raise TrimNotSupportedError(
                "a cached prefix is longer than the requested prefix; trimming "
                "is unsupported (broken upstream for hybrid caches, mlx-lm "
                "#980) -- recompute the prefix instead"
            )
        return (MatchClass.DIVERGE, None)

    def checked_batch_caches(
        self,
        model: str,
        prefix_text: str,
        suffixes: Sequence[str],
        *,
        tokenizer: Any = None,
    ) -> Optional[list]:
        """Return per-row cache copies for a batch, or None on a safe miss.

        Convenience wrapper enforcing both invariants for the common
        one-shared-prefix batch: requires an EXACT hit and a passing
        token-boundary check for EVERY row's suffix (BPE merges depend on
        the suffix's leading characters, so this is per-row, not
        per-batch). Any boundary failure disables reuse for the whole batch
        rather than mixing cached and uncached position bases.
        """
        tok = tokenizer if tokenizer is not None else self._tokenizer
        if tok is None:
            raise ValueError("a tokenizer is required for the token-boundary check")
        cls, entry = self.match(model, prefix_text)
        if cls is not MatchClass.EXACT or entry is None:
            return None
        for suffix in suffixes:
            if not verify_token_boundary(tok, prefix_text, suffix):
                return None
        return entry.replicate_into_batch(len(suffixes))
