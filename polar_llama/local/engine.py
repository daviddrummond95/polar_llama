"""In-process local inference engines for polar-llama.

This module defines the ``LocalEngine`` seam used by the
``engine="in_process"`` backend of :func:`polar_llama.local.expr.inference_local`.

Two implementations are provided:

* :class:`FakeEngine` -- a deterministic, dependency-free engine used by CI.
  It never imports ``mlx`` and simply echoes each prompt (honoring
  ``max_tokens`` by truncation). It exercises the batching / ordering / error
  isolation logic on plain Linux CPU without a GPU.
* :class:`MlxBatchEngine` -- the real Apple-silicon path that wraps
  ``mlx_lm``'s continuous-batching ``BatchGenerator`` / ``batch_generate``.
  Every ``mlx`` import is guarded behind :func:`polar_llama.local.require_mlx`
  (with a local fallback), so importing this module never pulls in ``mlx``.

A process-global singleton registry keyed by ``(model, engine)`` ensures each
model is loaded exactly once per process. This matters because, under the
Polars streaming engine, a ``map_batches`` UDF may be invoked once per morsel;
constructing a fresh engine per call would reload the model repeatedly.

Per-row failures are surfaced as error-JSON strings mirroring
``create_error_response`` in ``src/model_client/mod.rs`` (shape
``{"_error": ..., "_details": ..., "_raw"?: ...}``) instead of raising an
exception that would abort the whole batch.
"""

from __future__ import annotations

import json
import os
import threading
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

# Default number of prompts handed to the engine in a single ``generate`` call.
# Very large columns are split into chunks of this size so a single call never
# holds an unbounded number of prompts / KV caches in memory at once.
DEFAULT_CHUNK_SIZE = 256

# Sentinel substring understood by :class:`FakeEngine`. Any prompt containing
# this marker raises inside ``generate`` so per-row error isolation can be
# exercised deterministically (including end-to-end through a DataFrame).
FAKE_FAIL_MARKER = "<<FAIL>>"


# ---------------------------------------------------------------------------
# Error payloads (mirror create_error_response in src/model_client/mod.rs)
# ---------------------------------------------------------------------------
def error_response(error_type: str, details: str, raw: Optional[str] = None) -> str:
    """Build an error-JSON payload matching the Rust ``create_error_response``.

    The Rust side emits ``{"_error", "_details"}`` and additionally ``"_raw"``
    when raw content is available. We mirror that key ordering and fallback.
    """
    if raw is not None:
        error_obj = {"_error": error_type, "_details": details, "_raw": raw}
    else:
        error_obj = {"_error": error_type, "_details": details}
    try:
        return json.dumps(error_obj)
    except (TypeError, ValueError):
        return '{"_error": "%s"}' % error_type


# ---------------------------------------------------------------------------
# The engine seam
# ---------------------------------------------------------------------------
@runtime_checkable
class LocalEngine(Protocol):
    """Protocol for a local, in-process completion engine.

    Implementations must return one completion per input prompt, in the same
    order as the input (row-aligned). A failure affecting a single prompt must
    be returned as an error-JSON string (see :func:`error_response`) rather than
    raised, so one bad row cannot abort the batch.
    """

    def generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Return one completion per prompt, in the original order."""
        ...


# ---------------------------------------------------------------------------
# FakeEngine -- deterministic, no mlx, for CI
# ---------------------------------------------------------------------------
class FakeEngine:
    """Deterministic engine for CI and unit tests. Never imports ``mlx``.

    Each prompt is echoed as ``"echo:" + prompt``, truncated to ``max_tokens``
    whitespace-delimited tokens (a cheap stand-in for real token budgeting) and
    trimmed at the first ``stop`` string if any is supplied.

    A prompt containing :data:`FAKE_FAIL_MARKER` (or matching the optional
    ``fail_on`` predicate) raises internally; that failure is caught per row and
    converted to an error-JSON payload, demonstrating that a single bad row does
    not abort the batch.
    """

    def __init__(
        self,
        model: str = "fake",
        *,
        fail_on: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.model = model
        self._fail_on = fail_on

    def generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                text = self._one(
                    prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate per-row failure
                text = error_response("local_generation_error", str(exc))
            results.append(text)
        return results

    def _one(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> str:
        if FAKE_FAIL_MARKER in prompt:
            raise ValueError(f"simulated failure for prompt: {prompt!r}")
        if self._fail_on is not None and self._fail_on(prompt):
            raise ValueError(f"simulated failure for prompt: {prompt!r}")

        text = "echo:" + prompt

        # Honor max_tokens by truncating to that many whitespace tokens.
        if max_tokens is not None and max_tokens >= 0:
            tokens = text.split(" ")
            text = " ".join(tokens[:max_tokens])

        # Honor stop sequences by trimming at the earliest occurrence.
        if stop:
            cut = len(text)
            for marker in stop:
                if not marker:
                    continue
                idx = text.find(marker)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        return text


# ---------------------------------------------------------------------------
# mlx guard (defensive: works even if polar_llama.local.__init__ isn't ready)
# ---------------------------------------------------------------------------
def _require_mlx() -> None:
    """Ensure ``mlx``/``mlx_lm`` are importable, else raise a helpful error.

    Prefers the package-level :func:`polar_llama.local.require_mlx` (owned by the
    packaging agent) but falls back to a self-contained guard so this module is
    robust to import ordering and never fails to import merely because the
    optional extra is missing.
    """
    package_guard = None
    try:
        from polar_llama.local import require_mlx as package_guard  # type: ignore
    except Exception:  # noqa: BLE001 -- __init__ may not define it yet
        package_guard = None

    if package_guard is not None:
        package_guard()
        return

    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The in-process local engine requires the optional 'mlx' and "
            "'mlx-lm' packages, which are only available on Apple silicon. "
            "Install them with: pip install 'polar-llama[local]'"
        ) from exc


# ---------------------------------------------------------------------------
# MlxBatchEngine -- the real Apple-silicon path (BLOCKED-ON-GPU / import-guarded)
# ---------------------------------------------------------------------------
class MlxBatchEngine:
    """Real in-process engine wrapping ``mlx_lm`` continuous batching.

    NOTE: This path requires Apple silicon + the ``[local]`` extra and cannot be
    exercised in CI (no ``mlx``, no GPU). All ``mlx`` imports are deferred to
    call time and guarded via :func:`_require_mlx`, so importing this module is
    always safe.

    API shapes verified against mlx-lm 0.31.3 (isolated smoke install):
      * ``batch_generate(model, tokenizer, prompts, ...)`` takes ``prompts`` as
        a list of **token-id lists** (``List[List[int]]``); passing raw strings
        raises ``ZeroDivisionError``.
      * It returns a ``BatchResponse`` with ``.texts`` / ``.stats`` / ``.caches``
        (prefix caches surface on ``.caches`` with ``return_prompt_caches=True``,
        NOT ``.prompt_caches``).
      * Per-request ``stop`` is generator-level, not per-``insert``.

    Because of that, the default real path is the high-level ``batch_generate``
    (row order preserved internally). The low-level streaming ``BatchGenerator``
    path -- which tags rows with ``uid``s and calls ``close()`` in a ``finally``
    to restore the Metal wired-memory limit -- is opt-in via
    ``POLAR_LLAMA_LOCAL_STREAMING=1`` and its per-``insert`` kwargs remain
    unverified against 0.31.3.
    """

    def __init__(
        self,
        model: str,
        *,
        tokenizer_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
    ) -> None:
        self.model_id = model
        self._tokenizer_config = tokenizer_config or {}
        self._model_config = model_config or {}
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
            _require_mlx()
            from mlx_lm import load  # type: ignore

            self._model, self._tokenizer = load(
                self.model_id,
                tokenizer_config=self._tokenizer_config,
                model_config=self._model_config,
            )
            self._loaded = True

    # -- sampling param conversion -----------------------------------------
    def _make_sampler(self, temperature: float, top_p: float):
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        return make_sampler(temp=float(temperature), top_p=float(top_p))

    def generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        if not prompts:
            return []

        self._ensure_loaded()

        sampler = self._make_sampler(temperature, top_p)

        # Default to the high-level ``batch_generate`` path: it is the exact shape
        # smoke-verified against mlx-lm 0.31.3 (token-id-list prompts, results on
        # ``.texts``) and preserves row order internally. The low-level streaming
        # ``BatchGenerator`` path (per-row TTFT, uid re-ordering, explicit
        # close()) is opt-in because its per-insert kwargs are not yet verified
        # against 0.31.3 -- notably per-request ``stop`` is generator-level.
        if os.environ.get("POLAR_LLAMA_LOCAL_STREAMING") == "1":
            try:
                from mlx_lm.generate import BatchGenerator  # type: ignore
            except Exception:  # noqa: BLE001
                BatchGenerator = None  # type: ignore
            if BatchGenerator is not None:
                return self._generate_batchgen(
                    BatchGenerator,
                    prompts,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    stop=stop,
                )
        return self._generate_batch_generate(
            prompts,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    # -- low-level path: uids + explicit close() ---------------------------
    def _generate_batchgen(
        self,
        BatchGenerator,
        prompts: List[str],
        *,
        max_tokens: int,
        sampler,
        stop: Optional[List[str]],
    ) -> List[str]:
        tokenizer = self._tokenizer

        # Tokenize each prompt (apply the chat template when the tokenizer
        # exposes one; otherwise encode raw text).
        prompt_tokens: List[Sequence[int]] = []
        for prompt in prompts:
            prompt_tokens.append(self._encode(prompt))

        generator = BatchGenerator(self._model, tokenizer)
        collected: Dict[int, List[int]] = {uid: [] for uid in range(len(prompts))}
        try:
            for uid, toks in enumerate(prompt_tokens):
                generator.insert(
                    toks,
                    uid=uid,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    stop=stop,
                )
            # Drain the generator; each step yields (uid, token) events. Row
            # order is reconstructed via the uid, independent of completion
            # order or any left-padding inside the batched KV cache.
            for event in generator:
                uid, token = self._unpack_event(event)
                if uid is not None:
                    collected.setdefault(uid, []).append(token)
        finally:
            # Restores the Metal wired-memory limit set by BatchGenerator.
            try:
                generator.close()
            except Exception:  # noqa: BLE001
                pass

        results: List[str] = []
        for uid in range(len(prompts)):
            try:
                results.append(tokenizer.decode(collected.get(uid, [])))
            except Exception as exc:  # noqa: BLE001 -- isolate per-row failure
                results.append(error_response("local_decode_error", str(exc)))
        return results

    # -- high-level fallback: order preserved by batch_generate ------------
    def _generate_batch_generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int,
        sampler,
    ) -> List[str]:
        from mlx_lm import batch_generate  # type: ignore

        # mlx-lm 0.31.3's batch_generate takes prompts as token-id lists
        # (List[List[int]]); raw strings raise ZeroDivisionError. Encode each
        # prompt (applying the chat template when available) exactly as the
        # verified benchmark (benchmarks/local_mlx_gate.py) does, and pass them
        # positionally.
        prompt_tokens = [list(self._encode(p)) for p in prompts]

        try:
            outputs = batch_generate(
                self._model,
                self._tokenizer,
                prompt_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except TypeError:
            # Guard the unverified ``sampler``/``verbose`` kwargs: fall back to
            # the minimal positional form the benchmark proved.
            outputs = batch_generate(
                self._model,
                self._tokenizer,
                prompt_tokens,
                max_tokens=max_tokens,
            )
        texts = getattr(outputs, "texts", outputs)
        return [str(t) for t in texts]

    # -- helpers ------------------------------------------------------------
    def _encode(self, prompt: str) -> Sequence[int]:
        tokenizer = self._tokenizer
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001 -- fall back to raw encode
                pass
        return tokenizer.encode(prompt)

    @staticmethod
    def _unpack_event(event) -> Tuple[Optional[int], Optional[int]]:
        """Best-effort unpack of a BatchGenerator step event into (uid, token)."""
        uid = getattr(event, "uid", None)
        token = getattr(event, "token", None)
        if uid is not None:
            return uid, token
        if isinstance(event, tuple) and len(event) >= 2:
            return event[0], event[1]
        return None, None


# ---------------------------------------------------------------------------
# Process-global singleton registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[Tuple[str, str], LocalEngine] = {}
_REGISTRY_LOCK = threading.Lock()


def _build_engine(model: str, engine: str) -> LocalEngine:
    """Construct a fresh engine for ``(model, engine)``.

    An explicit ``POLAR_LLAMA_LOCAL_ENGINE=fake`` environment override forces the
    dependency-free :class:`FakeEngine` regardless of the requested backend. This
    makes the in-process path exercisable end-to-end in CI (and lets users force
    a dry run) without importing ``mlx``.
    """
    override = os.environ.get("POLAR_LLAMA_LOCAL_ENGINE", "").strip().lower()
    if override in ("fake", "fake_engine", "test"):
        return FakeEngine(model)

    normalized = (engine or "").strip().lower()
    if normalized in ("fake", "fake_engine", "test"):
        return FakeEngine(model)
    if normalized in ("in_process", "mlx", "mlx_batch"):
        return MlxBatchEngine(model)
    raise ValueError(
        f"Unknown local engine {engine!r}; expected 'in_process' or 'fake'."
    )


def get_engine(model: str, engine: str = "in_process") -> LocalEngine:
    """Return a process-global singleton engine for ``(model, engine)``.

    The model is loaded at most once per process. Safe to call from a
    ``map_batches`` UDF (which may run once per morsel): the first call builds
    and caches the engine, subsequent calls return the same instance.
    """
    key = (model, engine)

    # Fast path without holding the lock.
    existing = _REGISTRY.get(key)
    if existing is not None:
        return existing

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None:
            return existing
        instance = _build_engine(model, engine)
        _REGISTRY[key] = instance
        return instance


def register_engine(
    model: str,
    instance: LocalEngine,
    engine: str = "in_process",
) -> None:
    """Insert a pre-built engine into the registry.

    Primarily for tests (and advanced users) that want to inject a
    :class:`FakeEngine` under an ``engine="in_process"`` key so the in-process
    ``inference_local`` path resolves to it without importing ``mlx``.
    """
    with _REGISTRY_LOCK:
        _REGISTRY[(model, engine)] = instance


def clear_registry() -> None:
    """Drop all cached engines (used by tests for isolation)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Column-chunking helper for very large columns
# ---------------------------------------------------------------------------
def iter_chunks(
    items: Sequence[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[List[str]]:
    """Yield successive ``chunk_size``-sized slices of ``items``."""
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def generate_chunked(
    engine: LocalEngine,
    prompts: Sequence[str],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[str]:
    """Run ``engine.generate`` over ``prompts`` in row-aligned chunks.

    Splitting a very large column into bounded chunks keeps peak memory (and the
    number of simultaneously live KV caches on the real engine) bounded while
    preserving overall input order.
    """
    prompt_list = list(prompts)
    if len(prompt_list) <= chunk_size:
        return engine.generate(
            prompt_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    results: List[str] = []
    for chunk in iter_chunks(prompt_list, chunk_size):
        results.extend(
            engine.generate(
                chunk,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        )
    return results
