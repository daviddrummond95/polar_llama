"""In-process local embedding engines for polar-llama (issue #83).

This module defines the ``LocalEmbeddingEngine`` seam used by
:func:`embedding_local`, the fully-offline counterpart of
:func:`polar_llama.embedding_async`. It mirrors the design of
:mod:`polar_llama.local.engine` (the completion-side seam) but keeps a
SEPARATE registry, since embedding engines return row-aligned vectors
(``List[Optional[List[float]]]``) rather than strings.

Two implementations are provided:

* :class:`FakeEmbeddingEngine` -- a deterministic, dependency-free engine
  used by CI. It never imports ``mlx``/``mlx_embeddings`` and derives each
  vector from a SHA-256 hash of the input text, so identical text always
  produces an identical vector (useful for asserting nearest-neighbor
  behavior in tests without a real model).
* :class:`MlxEmbeddingEngine` -- the real Apple-silicon path, wrapping
  ``mlx_embeddings.load``/``generate``. Every ``mlx``/``mlx_embeddings``
  import is deferred to call time and guarded by
  :func:`_require_mlx_embeddings`, so importing this module never pulls
  in mlx.

A process-global singleton registry keyed by ``(model, engine)`` ensures each
model is loaded at most once per process, mirroring
:func:`polar_llama.local.engine.get_engine`.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import polars as pl

from polar_llama.local.engine import iter_chunks

# Default number of texts handed to the embedding engine in a single call.
# Smaller than the completion-side DEFAULT_CHUNK_SIZE (256): embedding calls
# are cheaper per item but the batch matrix (texts x seq_len) still grows
# memory linearly, so a conservative default keeps peak memory bounded.
DEFAULT_EMBED_CHUNK_SIZE = 64

# Default model for `engine="in_process"` / `engine="mlx"`: a small, widely
# used BGE conversion (384 dims, ~65MB bf16 download from the HF Hub, cached
# under ~/.cache/huggingface after the first use). See docs/LOCAL_EMBEDDINGS.md.
DEFAULT_LOCAL_EMBED_MODEL = "mlx-community/bge-small-en-v1.5-bf16"

# Sentinel substring understood by :class:`FakeEmbeddingEngine`. Any text
# containing this marker raises inside ``embed``; that failure is caught per
# row and surfaced as ``None`` (a null embedding), demonstrating that a
# single bad row does not abort the batch. Mirrors
# ``polar_llama.local.engine.FAKE_FAIL_MARKER``.
FAKE_FAIL_MARKER = "<<FAIL>>"


# ---------------------------------------------------------------------------
# The embedding engine seam
# ---------------------------------------------------------------------------
@runtime_checkable
class LocalEmbeddingEngine(Protocol):
    """Protocol for a local, in-process embedding engine.

    Implementations must return one vector per input text, in the same order
    as the input (row-aligned). A failure affecting a single text must be
    surfaced as ``None`` in the output list (a null embedding row) rather
    than raised, so one bad row cannot abort the batch.
    """

    def embed(
        self, texts: List[str], *, normalize: bool = True
    ) -> List[Optional[List[float]]]:
        """Return one embedding vector (or ``None``) per text, in order."""
        ...


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    """L2-normalize ``vec``; a zero (or near-zero) norm is left unchanged."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


# ---------------------------------------------------------------------------
# FakeEmbeddingEngine -- deterministic, no mlx, for CI
# ---------------------------------------------------------------------------
class FakeEmbeddingEngine:
    """Deterministic embedding engine for CI and unit tests. Never imports mlx.

    Each text is hashed with SHA-256; the first ``dim`` bytes of the digest
    become the (unnormalized) vector components. This is deterministic
    (identical text -> identical vector, so a duplicated document is its own
    nearest neighbor in tests) and distinct per text (different text -> a
    different vector, with overwhelming probability), without needing any
    real model.

    A text containing :data:`FAKE_FAIL_MARKER` (or matching the optional
    ``fail_on`` predicate) raises internally; that failure is caught per row
    and surfaced as ``None``, mirroring :class:`polar_llama.local.engine.FakeEngine`.
    """

    def __init__(
        self,
        model: str = "fake",
        *,
        dim: int = 8,
        fail_on: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self._fail_on = fail_on

    def embed(
        self, texts: List[str], *, normalize: bool = True
    ) -> List[Optional[List[float]]]:
        results: List[Optional[List[float]]] = []
        for text in texts:
            try:
                results.append(self._one(text, normalize=normalize))
            except Exception:  # noqa: BLE001 -- isolate per-row failure
                results.append(None)
        return results

    def _one(self, text: str, *, normalize: bool) -> List[float]:
        if FAKE_FAIL_MARKER in text:
            raise ValueError(f"simulated failure for text: {text!r}")
        if self._fail_on is not None and self._fail_on(text):
            raise ValueError(f"simulated failure for text: {text!r}")

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [float(b) / 255.0 for b in digest[: self.dim]]
        if normalize:
            vec = _l2_normalize(vec)
        return vec


# ---------------------------------------------------------------------------
# mlx_embeddings guard (never imported at module scope)
# ---------------------------------------------------------------------------
def _require_mlx_embeddings() -> None:
    """Ensure ``mlx``/``mlx_embeddings`` are importable, else raise a helpful error."""
    try:
        import mlx  # noqa: F401
        import mlx_embeddings  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The in-process local embedding engine requires the optional "
            "'mlx' and 'mlx-embeddings' packages, which are only available "
            "on Apple silicon. Install them with: pip install 'polar-llama[local]'"
        ) from exc


# ---------------------------------------------------------------------------
# MlxEmbeddingEngine -- the real Apple-silicon path (requires [local] extra)
# ---------------------------------------------------------------------------
class MlxEmbeddingEngine:
    """Real in-process embedding engine wrapping ``mlx_embeddings``.

    NOTE: This path requires Apple silicon + the ``[local]`` extra and cannot
    be exercised in CI (no ``mlx``, no GPU). All ``mlx``/``mlx_embeddings``
    imports are deferred to call time and guarded via
    :func:`_require_mlx_embeddings`, so importing this module is always safe.

    ``embed()`` calls ``mlx_embeddings.generate(model, tokenizer, texts)`` and
    reads the pooled per-text vectors off ``output.text_embeds`` (BGE-style
    models already L2-normalize this output; ``normalize=True`` re-normalizes
    defensively so the guarantee holds regardless of model). On any batch-call
    failure, falls back to a per-text loop so a single bad row degrades to a
    ``None`` entry instead of aborting the whole chunk.
    """

    def __init__(self, model: str) -> None:
        self.model_id = model
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_lock = threading.Lock()

    # -- lazy, once-only model load ----------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            _require_mlx_embeddings()
            from mlx_embeddings import load  # type: ignore

            self._model, self._tokenizer = load(self.model_id)
            self._loaded = True

    def embed(
        self, texts: List[str], *, normalize: bool = True
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []

        self._ensure_loaded()

        try:
            return self._embed_batch(texts, normalize=normalize)
        except Exception:  # noqa: BLE001 -- fall back to per-text isolation
            return self._embed_one_by_one(texts, normalize=normalize)

    def _embed_batch(
        self, texts: List[str], *, normalize: bool
    ) -> List[Optional[List[float]]]:
        from mlx_embeddings import generate  # type: ignore

        output = generate(self._model, self._tokenizer, texts)
        vectors = self._extract_vectors(output)

        results: List[Optional[List[float]]] = []
        for row in vectors:
            floats = [float(x) for x in row]
            if normalize:
                floats = _l2_normalize(floats)
            results.append(floats)
        return results

    def _embed_one_by_one(
        self, texts: List[str], *, normalize: bool
    ) -> List[Optional[List[float]]]:
        """Per-text fallback: isolate a bad row instead of failing the batch."""
        from mlx_embeddings import generate  # type: ignore

        results: List[Optional[List[float]]] = []
        for text in texts:
            try:
                output = generate(self._model, self._tokenizer, [text])
                vectors = self._extract_vectors(output)
                floats = [float(x) for x in vectors[0]]
                if normalize:
                    floats = _l2_normalize(floats)
                results.append(floats)
            except Exception:  # noqa: BLE001 -- isolate per-row failure
                results.append(None)
        return results

    @staticmethod
    def _extract_vectors(output) -> List[Sequence[float]]:
        """Pull pooled per-text vectors off an ``mlx_embeddings.generate`` result.

        Expected shape (mlx-embeddings README): ``output.text_embeds``, a
        ``(batch, dim)`` array-like of pooled, already-normalized vectors.
        Falls back to a couple of other plausible attribute names in case
        the installed version differs, so a minor API drift degrades
        gracefully instead of hard-crashing every call.
        """
        vectors = getattr(output, "text_embeds", None)
        if vectors is None:
            vectors = getattr(output, "embeddings", None)
        if vectors is None:
            vectors = output  # assume `output` is already the array

        # mlx arrays (and numpy arrays) support `.tolist()`; plain
        # lists/tuples don't, so guard the call.
        tolist = getattr(vectors, "tolist", None)
        if callable(tolist):
            vectors = tolist()
        return list(vectors)


# ---------------------------------------------------------------------------
# Process-global singleton registry (separate from the completion registry)
# ---------------------------------------------------------------------------
_EMBED_REGISTRY: Dict[Tuple[str, str], LocalEmbeddingEngine] = {}
_EMBED_REGISTRY_LOCK = threading.Lock()


def _build_embedding_engine(model: str, engine: str) -> LocalEmbeddingEngine:
    """Construct a fresh embedding engine for ``(model, engine)``.

    An explicit ``POLAR_LLAMA_LOCAL_ENGINE=fake`` environment override forces
    the dependency-free :class:`FakeEmbeddingEngine` regardless of the
    requested backend -- the SAME env var used by the completion-side
    ``inference_local`` engine seam, so one switch flips the whole local
    stack (generation + embeddings) to fake in CI.
    """
    override = os.environ.get("POLAR_LLAMA_LOCAL_ENGINE", "").strip().lower()
    if override in ("fake", "fake_engine", "test"):
        return FakeEmbeddingEngine(model)

    normalized = (engine or "").strip().lower()
    if normalized in ("fake", "fake_engine", "test"):
        return FakeEmbeddingEngine(model)
    if normalized in ("in_process", "mlx", "mlx_batch"):
        return MlxEmbeddingEngine(model)
    raise ValueError(
        f"Unknown local embedding engine {engine!r}; expected 'in_process' or 'fake'."
    )


def get_embedding_engine(
    model: str, engine: str = "in_process"
) -> LocalEmbeddingEngine:
    """Return a process-global singleton embedding engine for ``(model, engine)``.

    The model is loaded at most once per process. Safe to call from a
    ``map_batches`` UDF (which may run once per morsel): the first call
    builds and caches the engine, subsequent calls return the same instance.
    """
    key = (model, engine)

    existing = _EMBED_REGISTRY.get(key)
    if existing is not None:
        return existing

    with _EMBED_REGISTRY_LOCK:
        existing = _EMBED_REGISTRY.get(key)
        if existing is not None:
            return existing
        instance = _build_embedding_engine(model, engine)
        _EMBED_REGISTRY[key] = instance
        return instance


def register_embedding_engine(
    model: str,
    instance: LocalEmbeddingEngine,
    engine: str = "in_process",
) -> None:
    """Insert a pre-built embedding engine into the registry.

    Primarily for tests (and advanced users) that want to inject a
    :class:`FakeEmbeddingEngine` under an ``engine="in_process"`` key so
    ``embedding_local`` resolves to it without importing ``mlx``.
    """
    with _EMBED_REGISTRY_LOCK:
        _EMBED_REGISTRY[(model, engine)] = instance


def clear_embedding_registry() -> None:
    """Drop all cached embedding engines (used by tests for isolation)."""
    with _EMBED_REGISTRY_LOCK:
        _EMBED_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Column-chunking helper for very large columns
# ---------------------------------------------------------------------------
def embed_chunked(
    engine: LocalEmbeddingEngine,
    texts: Sequence[str],
    *,
    normalize: bool = True,
    chunk_size: int = DEFAULT_EMBED_CHUNK_SIZE,
) -> List[Optional[List[float]]]:
    """Run ``engine.embed`` over ``texts`` in row-aligned chunks.

    Splitting a very large column into bounded chunks keeps peak memory
    (the batch matrix built by the real mlx model) bounded while preserving
    overall input order. Mirrors
    :func:`polar_llama.local.engine.generate_chunked`.
    """
    text_list = list(texts)
    if len(text_list) <= chunk_size:
        return engine.embed(text_list, normalize=normalize)

    results: List[Optional[List[float]]] = []
    for chunk in iter_chunks(text_list, chunk_size):
        results.extend(engine.embed(chunk, normalize=normalize))
    return results


# ---------------------------------------------------------------------------
# Polars expression entry point
# ---------------------------------------------------------------------------
def _parse_into_expr(expr) -> pl.Expr:
    """Parse an input into an expression (str -> column name)."""
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def embedding_local(
    expr,
    *,
    model: str = DEFAULT_LOCAL_EMBED_MODEL,
    engine: str = "in_process",
    batch_size: int = DEFAULT_EMBED_CHUNK_SIZE,
    normalize: bool = True,
) -> pl.Expr:
    """Generate embeddings for a text column using a local (offline) model.

    The fully-offline counterpart of :func:`polar_llama.embedding_async`:
    runs an embedding model in this process via ``mlx_embeddings`` instead
    of calling a hosted provider API. Output dtype is ``List[Float64]``,
    identical to ``embedding_async`` -- a column produced by either function
    is a drop-in input to ``cosine_similarity``, ``knn_hnsw``,
    ``HnswIndex``, and ``cluster_embeddings``.

    Parameters
    ----------
    expr
        The text expression (or column name) to embed.
    model
        An ``mlx-community`` (or other mlx-embeddings-compatible) model
        repo id. Defaults to a small BGE conversion
        (``mlx-community/bge-small-en-v1.5-bf16``, 384 dims); downloads once
        from the HF Hub on first use and is cached under
        ``~/.cache/huggingface``. See docs/LOCAL_EMBEDDINGS.md.
    engine
        ``"in_process"`` (default, alias ``"mlx"``) uses the real
        ``mlx_embeddings`` model and requires the optional ``[local]``
        extra (``pip install polar-llama[local]``) on Apple silicon.
        ``"fake"`` forces the dependency-free :class:`FakeEmbeddingEngine`
        (also reachable via the ``POLAR_LLAMA_LOCAL_ENGINE=fake``
        environment override, which applies regardless of ``engine``).
    batch_size
        Number of texts handed to the model in a single call; larger
        columns are processed in row-aligned chunks of this size.
    normalize
        L2-normalize each output vector (default ``True``, matching the
        pre-normalized convention of BGE-style embedding models).

    Returns
    -------
    polars.Expr
        Expression with embeddings as ``List[Float64]``, in the original
        row order. A ``null`` input row produces a ``null`` output row; a
        per-row engine failure also produces a ``null`` output row (never
        raises and never aborts the rest of the batch).
    """
    parsed = _parse_into_expr(expr)

    def _udf(series: pl.Series) -> pl.Series:
        texts = series.to_list()

        # Only a `None` input row stays null -- mirrors `embedding_async`'s
        # Rust plugin (src/expressions.rs::embedding_async), which skips
        # `None` rows but sends `""` through to the model like any other
        # text.
        positions: List[int] = []
        non_null_texts: List[str] = []
        for idx, text in enumerate(texts):
            if text is None:
                continue
            positions.append(idx)
            non_null_texts.append(text)

        out: List[Optional[List[float]]] = [None] * len(texts)

        if non_null_texts:
            eng = get_embedding_engine(model, engine)
            vecs = embed_chunked(
                eng,
                non_null_texts,
                normalize=normalize,
                chunk_size=batch_size,
            )
            for pos, vec in zip(positions, vecs):
                out[pos] = vec

        return pl.Series(series.name, out, dtype=pl.List(pl.Float64))

    return parsed.map_batches(_udf, return_dtype=pl.List(pl.Float64))
