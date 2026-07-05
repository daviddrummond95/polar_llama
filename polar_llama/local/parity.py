"""Parity / regression harness for the in-process local backend.

The prefix-cache layer's failure mode is SILENTLY WRONG OUTPUT, so parity
is defined as runnable code, not prose. Three checks:

1. **Batched vs sequential** (:func:`compare_batched_vs_sequential`):
   greedy decoding (temperature=0.0) with a fixed seed; the same prompts
   generated as one batch and one-at-a-time are compared token-by-token.
   Threshold:

   - Fake engine / CI: ``0.0``. A deterministic engine has no excuse.
   - Real MLX: ``0.02`` mean per-row token-divergence rate. Batched
     decode left-pads (BatchKVCache) and changes fp16 matmul reduction
     order, which can flip near-tied greedy argmaxes; one early flip
     diverges the whole remainder of that row, so a small number of
     rows with near-tied logits produces a small nonzero mean. 2% is an
     empirical regression tripwire, not a correctness license -- the
     real-model number must be measured and pinned (BLOCKED-ON-GPU).

2. **With-cache vs without-cache** (:func:`compare_cache_parity`):
   threshold is EXACTLY 0 -- byte-identical strings. Reusing a prefix
   cache replays the same tokens at the same absolute positions, so any
   difference means an invariant (token boundary, no-trim, positional
   alignment) is broken. This is the check that catches silent corruption.

3. **Warm vs cold TTFT** (:func:`measure_ttft`): per-row time-to-first-
   token measured from ROW ADMISSION (immediately before the engine call
   for that row), not from batch start -- batch-start timing flatters the
   cache by hiding queueing.

Divergence metric: ``token_divergence_rate(a, b) = 1 - lcp/max(len)``
where ``lcp`` is the longest common prefix length. Greedy decode is
autoregressive, so positions after the first flip are conditioned on
different histories; counting positionwise mismatches would double-count
one flip. Prefix-based divergence measures exactly "how soon did the
trajectories split".

Runs against the in-module ``FakeEngine``/``FakeTokenizer`` in CI (no mlx,
no GPU) and against real mlx-lm when available (import-guarded)::

    uv run python -m polar_llama.local.parity            # fake, CI-safe
    uv run python -m polar_llama.local.parity --real MODEL_PATH  # needs mlx
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .prefix_cache import (
    PrefixStore,
    error_payload,
    verify_token_boundary,
)

__all__ = [
    "GenParams",
    "FakeTokenizer",
    "FakePromptCache",
    "FakeEngine",
    "RowDivergence",
    "ParityReport",
    "TTFTSample",
    "token_divergence_rate",
    "compare_batched_vs_sequential",
    "compare_cache_parity",
    "measure_ttft",
    "run_fake_parity_suite",
    "FAKE_DIVERGENCE_THRESHOLD",
    "REAL_DIVERGENCE_THRESHOLD",
]

# Documented thresholds (rationale in module docstring / design doc).
FAKE_DIVERGENCE_THRESHOLD = 0.0
REAL_DIVERGENCE_THRESHOLD = 0.02
CACHE_PARITY_THRESHOLD = 0.0  # with-cache vs without-cache: identical, always.


@dataclass(frozen=True)
class GenParams:
    """Decoding parameters. Parity runs MUST be greedy and seeded."""

    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[Sequence[str]] = None
    seed: int = 0


# ---------------------------------------------------------------------------
# Fakes (CI test doubles; no mlx imports anywhere in this module's top level)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Greedy longest-match tokenizer over a small vocabulary.

    Deliberately exhibits BPE-style context dependence: with ``"walking"``
    in the vocab, ``encode("walk") == [walk]`` but
    ``encode("walk" + "ing") == [walking]`` -- a cross-seam merge that
    :func:`polar_llama.local.prefix_cache.verify_token_boundary` must catch.
    Unknown characters fall back to single-character tokens.
    """

    DEFAULT_VOCAB = ("walking", "walk", "ing", "<|user|>", "<|assistant|>", "\n")

    def __init__(self, vocab: Optional[Sequence[str]] = None) -> None:
        words = list(vocab if vocab is not None else self.DEFAULT_VOCAB)
        # Longest-match-first; stable ids.
        self._by_len = sorted(set(words), key=len, reverse=True)
        self._ids = {w: i for i, w in enumerate(sorted(set(words)))}
        self._char_base = len(self._ids)

    def encode(self, text: str) -> list:
        tokens: list = []
        i = 0
        while i < len(text):
            for w in self._by_len:
                if text.startswith(w, i):
                    tokens.append(self._ids[w])
                    i += len(w)
                    break
            else:
                tokens.append(self._char_base + ord(text[i]))
                i += 1
        return tokens


@dataclass
class FakePromptCache:
    """Stands in for an mlx-lm prompt-cache: remembers the prefix it was
    prefilled with (mutable ``consumed`` mimics in-place decode mutation,
    which is why replicate_into_batch must produce independent copies)."""

    prefix_text: str
    token_count: int
    consumed: int = 0
    log: list = field(default_factory=list)


class FakeEngine:
    """Deterministic engine implementing the LocalEngine seam.

    ``generate`` output depends ONLY on the full logical prompt
    (cached prefix + suffix, or the raw prompt), decoding params and seed --
    never on batch size or arrival order -- so batched==sequential and
    with-cache==without-cache hold by construction and any harness failure
    indicates a harness/store bug. ``fail_substring`` triggers per-row
    failures to exercise the error-payload path.
    """

    def __init__(
        self,
        tokenizer: Optional[FakeTokenizer] = None,
        fail_substring: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer or FakeTokenizer()
        self.fail_substring = fail_substring
        self.prefill_count = 0

    # -- engine seam -------------------------------------------------------

    def prefill(self, prefix_text: str) -> FakePromptCache:
        self.prefill_count += 1
        return FakePromptCache(
            prefix_text=prefix_text,
            token_count=len(self.tokenizer.encode(prefix_text)),
        )

    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        prompt_caches: Optional[Sequence[Optional[FakePromptCache]]] = None,
    ) -> list:
        if prompt_caches is not None and len(prompt_caches) != len(prompts):
            raise ValueError("prompt_caches must be row-aligned with prompts")
        out = []
        for i, prompt in enumerate(prompts):
            cache = prompt_caches[i] if prompt_caches is not None else None
            full = (cache.prefix_text if cache is not None else "") + prompt
            if self.fail_substring is not None and self.fail_substring in full:
                out.append(error_payload("engine_error", "fake per-row failure"))
                continue
            if cache is not None:
                cache.consumed += 1  # in-place mutation, like real decode
            tokens = self.tokenizer.encode(full)
            out.append(
                f"echo[seed={params.seed}]:{full[-48:]}#ntok={min(len(tokens), params.max_tokens)}"
            )
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_divergence_rate(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> float:
    """1 - longest_common_prefix / max(len). 0.0 iff identical."""
    if not tokens_a and not tokens_b:
        return 0.0
    denom = max(len(tokens_a), len(tokens_b))
    lcp = 0
    for x, y in zip(tokens_a, tokens_b):
        if x != y:
            break
        lcp += 1
    return 1.0 - (lcp / denom)


@dataclass
class RowDivergence:
    row: int
    rate: float
    a: str
    b: str


@dataclass
class ParityReport:
    name: str
    threshold: float
    mean_rate: float
    max_rate: float
    rows: int
    divergent: list  # list[RowDivergence] with rate > 0

    @property
    def passed(self) -> bool:
        return self.mean_rate <= self.threshold

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "threshold": self.threshold,
            "mean_rate": self.mean_rate,
            "max_rate": self.max_rate,
            "rows": self.rows,
            "divergent_rows": [d.row for d in self.divergent],
        }


def _report(name, threshold, tokenizer, outputs_a, outputs_b) -> ParityReport:
    if len(outputs_a) != len(outputs_b):
        raise ValueError("output lists must be row-aligned")
    divs = []
    rates = []
    for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        rate = token_divergence_rate(tokenizer.encode(a), tokenizer.encode(b))
        rates.append(rate)
        if rate > 0.0:
            divs.append(RowDivergence(row=i, rate=rate, a=a, b=b))
    n = len(rates)
    return ParityReport(
        name=name,
        threshold=threshold,
        mean_rate=(sum(rates) / n) if n else 0.0,
        max_rate=max(rates) if rates else 0.0,
        rows=n,
        divergent=divs,
    )


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------


def compare_batched_vs_sequential(
    engine: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    params: Optional[GenParams] = None,
    *,
    threshold: float = FAKE_DIVERGENCE_THRESHOLD,
) -> ParityReport:
    """Same prompts, one batch vs one-at-a-time; greedy + fixed seed."""
    params = params or GenParams()
    if params.temperature != 0.0:
        raise ValueError("parity runs must use greedy decoding (temperature=0.0)")
    batched = engine.generate(list(prompts), params)
    sequential = [engine.generate([p], params)[0] for p in prompts]
    return _report("batched_vs_sequential", threshold, tokenizer, batched, sequential)


def compare_cache_parity(
    engine: Any,
    tokenizer: Any,
    prefix_text: str,
    suffixes: Sequence[str],
    params: Optional[GenParams] = None,
    *,
    model: str = "fake-model",
    store: Optional[PrefixStore] = None,
) -> ParityReport:
    """With-prefix-cache vs without-prefix-cache; MUST be identical (0.0).

    Without-cache: each row generates from ``prefix_text + suffix`` cold.
    With-cache: the prefix is prefilled ONCE, stored, replicated per row
    via ``replicate_into_batch``, and rows generate from ``suffix`` only.
    The token-boundary invariant is verified for every row first; rows
    that fail it are generated cold in BOTH arms (never a mixed basis).
    """
    params = params or GenParams()
    if params.temperature != 0.0:
        raise ValueError("parity runs must use greedy decoding (temperature=0.0)")
    store = store or PrefixStore(tokenizer=tokenizer)

    # Arm A: cold, full prompts.
    full_prompts = [prefix_text + s for s in suffixes]
    without = engine.generate(full_prompts, params)

    # Arm B: prefill once, replicate, decode suffixes on top.
    entry = store.get_exact(model, prefix_text)
    if entry is None:
        cache_obj = engine.prefill(prefix_text)
        entry = store.put(
            model, prefix_text, cache_obj, token_count=cache_obj.token_count
        )
    boundary_ok = [verify_token_boundary(tokenizer, prefix_text, s) for s in suffixes]
    caches = entry.replicate_into_batch(len(suffixes))
    row_caches: list = []
    row_prompts: list = []
    for ok, cache, suffix, full in zip(boundary_ok, caches, suffixes, full_prompts):
        if ok:
            row_caches.append(cache)
            row_prompts.append(suffix)
        else:  # boundary violation: cold fallback, correctness over speed
            row_caches.append(None)
            row_prompts.append(full)
    with_cache = engine.generate(row_prompts, params, prompt_caches=row_caches)

    return _report(
        "with_cache_vs_without_cache",
        CACHE_PARITY_THRESHOLD,
        tokenizer,
        with_cache,
        without,
    )


@dataclass
class TTFTSample:
    row: int
    cold_s: float
    warm_s: float

    @property
    def speedup(self) -> float:
        return self.cold_s / self.warm_s if self.warm_s > 0 else float("inf")


def measure_ttft(
    engine: Any,
    tokenizer: Any,
    prefix_text: str,
    suffixes: Sequence[str],
    *,
    model: str = "fake-model",
    clock: Callable[[], float] = time.perf_counter,
) -> list:
    """Per-row warm-vs-cold TTFT, measured from row admission.

    TTFT is approximated as the latency of a ``max_tokens=1`` generation
    (prefill + first decode step), timed from immediately before THAT
    row's engine call -- not from batch start, which would credit the
    cache with queueing time it did not save. Cold: full prompt, no cache.
    Warm: prefix prefilled once beforehand (prefill excluded from warm
    timing -- it is amortized across the batch by design), suffix decoded
    on a replicated cache.
    """
    params = GenParams(max_tokens=1)
    store = PrefixStore(tokenizer=tokenizer)
    cache_obj = engine.prefill(prefix_text)
    entry = store.put(model, prefix_text, cache_obj, token_count=cache_obj.token_count)

    samples = []
    for i, suffix in enumerate(suffixes):
        t0 = clock()  # row admission (cold)
        engine.generate([prefix_text + suffix], params)
        cold = clock() - t0

        row_cache = entry.replicate_into_batch(1)
        t1 = clock()  # row admission (warm)
        engine.generate([suffix], params, prompt_caches=row_cache)
        warm = clock() - t1
        samples.append(TTFTSample(row=i, cold_s=cold, warm_s=warm))
    return samples


# ---------------------------------------------------------------------------
# Runnable suites
# ---------------------------------------------------------------------------


def run_fake_parity_suite() -> list:
    """CI-safe end-to-end parity suite on the fakes. Returns reports."""
    tokenizer = FakeTokenizer()
    engine = FakeEngine(tokenizer)
    prefix = "<|system|>\nYou are terse.\n"
    suffixes = [f"<|user|>\nq{i}\n<|assistant|>\n" for i in range(6)]
    prompts = [prefix + s for s in suffixes]
    reports = [
        compare_batched_vs_sequential(
            engine, tokenizer, prompts, threshold=FAKE_DIVERGENCE_THRESHOLD
        ),
        compare_cache_parity(engine, tokenizer, prefix, suffixes),
    ]
    return reports


def _run_real_parity_suite(model_path: str) -> list:  # pragma: no cover
    """Real-model parity (BLOCKED-ON-GPU for validation; import-guarded).

    Requires the [local] extra and Apple Silicon. Uses mlx-lm's own
    batch_generate for the batched arm and generate for the sequential arm.
    """
    from .prefix_cache import require_mlx

    mlx_lm = require_mlx()
    model, tokenizer = mlx_lm.load(model_path)
    raise NotImplementedError(
        "real-model parity wiring lands with the MlxBatchEngine "
        "(BLOCKED-ON-GPU validation); the harness definition above is final"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", metavar="MODEL", default=None)
    args = parser.parse_args(argv)
    reports = (
        _run_real_parity_suite(args.real) if args.real else run_fake_parity_suite()
    )
    print(json.dumps([r.to_dict() for r in reports], indent=2))
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
