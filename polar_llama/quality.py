"""
Survey data-quality flags for Polar Llama (issue #80).

Design summary -- see `docs/QUALITY_FLAGS.md` for the full walkthrough:

1. **Two tiers, one contract.** Every score in this module is a graded
   `Float64` in `[0, 1]` (higher = more suspicious), never a hard filter.
   Booleans (`flag`) are derived by thresholding a score in `quality_report`
   -- a null score (not enough signal: too few grid answers, too little
   text, missing duration) always resolves to `flag = False`, never `True`.
   **No function in this module ever drops a row.**

   - **Heuristic tier** (`straightlining_score`, `gibberish_score`,
     `duplicate_answer_score`, `response_length_score`, `speeder_score`):
     zero API calls. The first three are Rust plugin expressions
     (`src/quality.rs` pure functions + `src/expressions.rs` plumbing,
     mirroring the `src/metrics.rs` / issue #79 split); the last two are a
     few lines of pure Polars (median/quantile/rank) -- there is nothing
     for Rust to buy here.
   - **Embedding/LLM tier** (`near_duplicate`, `likely_ai` via
     `ai_likelihood`): opt-in (`QualityConfig.llm_tier=True`, default
     `False`), pure Python composition of existing primitives
     (`embedding_async`, `cosine_similarity`, `knn_hnsw`,
     `inference_messages(..., response_model=...)`) -- exactly the
     `induce_codebook` pattern (`polar_llama/codebook.py`).

2. **`quality_report(df, config) -> QualityReport(df, summary)`** is a
   DataFrame-in/DataFrame-out orchestration function, not a Polars
   expression, because a single expression cannot return two different
   heights (per-row flags *and* a per-flag summary) at once -- same
   reasoning as `induce_codebook`. `.df` is the input DataFrame plus one
   struct column (`quality` by default); `.summary` is one row per active
   flag (+ a final `any_flag` row). Row count and order are always
   preserved (`assert out.height == df.height` internally) -- flagging
   never filters.

3. **All default thresholds are SUBJECTIVE conventions**, not validated
   statistical cutoffs -- tune them to your own panel. This is called out
   per-threshold below and centrally in `docs/QUALITY_FLAGS.md`.

4. **AI-text detection is a flag, not a verdict.** `likely_ai`/
   `ai_likelihood` carry a mandatory limitation warning (see their
   docstrings and `docs/QUALITY_FLAGS.md` section 7): published AI-text
   detectors have high false-positive rates, especially for non-native
   English speakers, formal registers, and short texts. Never use this
   score as sole grounds for excluding a respondent or sanctioning a
   panelist.

`embedding_async`, `cosine_similarity`, `knn_hnsw`, `inference_messages`,
`combine_messages`, and `string_to_message` are imported lazily inside
function bodies (not at module scope) to avoid a circular import with
`polar_llama/__init__.py`, which imports this module -- the same pattern
`polar_llama/codebook.py` and `polar_llama/reliability.py` use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from pydantic import BaseModel
    from polars.type_aliases import IntoExpr

    from polar_llama import Provider


# ============================================================================
# Heuristic tier -- Rust plugin expressions (straightlining / gibberish /
# duplicate-answer) -- src/quality.rs + src/expressions.rs
# ============================================================================


def straightlining_score(
    grid_cols: Sequence["IntoExpr"],
    *,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    min_answers: int = 3,
) -> pl.Expr:
    """
    Straightlining/flatlining suspicion score across a Likert grid battery.

    Zero API calls. Backed by `crate::quality::straightline_score`
    (`src/quality.rs`): for each respondent's non-null answers across
    `grid_cols`, combines `mode_frac` (share of answers equal to the most
    frequent value) and `1 - var_norm` (variance normalized by the maximum
    possible variance for the scale) via `max` -- either component alone is
    enough to be suspicious. An alternating pattern (e.g. 1,5,1,5) has high
    variance, so it is *not* flagged despite having few distinct values.

    Parameters
    ----------
    grid_cols : sequence of IntoExpr
        The Likert grid columns (numeric), one row per respondent.
    scale_min, scale_max : float, optional
        The Likert scale's bounds (e.g. 1.0/5.0), used to normalize the
        variance component. When `None` (default), each call infers them
        from the observed min/max across every value in every column
        passed to *that call* -- a whole-batch computation. `quality_report`
        always computes and passes explicit bounds once over the full
        configured column set so this inference is never actually
        exercised by the DataFrame-level API; pass `scale_min`/`scale_max`
        explicitly yourself for the same determinism when calling this
        standalone.
    min_answers : int
        Minimum number of non-null grid answers required to score a row;
        below this, the score is null (not enough signal) -- never `0.0`.
        Default `3` (SUBJECTIVE -- see `docs/QUALITY_FLAGS.md`).

    Returns
    -------
    polars.Expr
        `Float64` in `[0, 1]`, higher = more suspicious. Null when fewer
        than `min_answers` non-null grid answers are present.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import straightlining_score
    >>> df = pl.DataFrame({
    ...     "g1": [4, 2], "g2": [4, 4], "g3": [4, 3], "g4": [4, 5], "g5": [4, 1],
    ... })
    >>> df.select(score=straightlining_score(["g1", "g2", "g3", "g4", "g5"]))
    """
    exprs = [parse_into_expr(c) for c in grid_cols]
    if not exprs:
        raise ValueError("straightlining_score: grid_cols must have at least 1 column")

    kwargs: Dict[str, Any] = {
        "scale_min": float(scale_min) if scale_min is not None else None,
        "scale_max": float(scale_max) if scale_max is not None else None,
        "min_answers": int(min_answers),
    }

    from polar_llama.expressions import get_lib_path

    return register_plugin(
        args=exprs,
        symbol="straightlining_score",
        is_elementwise=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )


def gibberish_score(col: "IntoExpr", *, min_chars: int = 8) -> pl.Expr:
    """
    Keyboard-mash / gibberish suspicion score for one open-end text column.

    Zero API calls. Backed by `crate::quality::gibberish_score`
    (`src/quality.rs`): an **English-only** heuristic combining a
    consecutive-consonant-run score, a deviation from English's ~0.38
    letter-level vowel ratio, and normalized character-bigram entropy
    (weights `0.5`/`0.3`/`0.2` respectively -- SUBJECTIVE, pinned in
    `docs/QUALITY_FLAGS.md`; unit-tested for *ordering* against a fixture,
    not for the exact weight values).

    Text that normalizes to fewer than `min_chars` Latin letters (including
    non-Latin scripts, which strip to ~nothing) scores null rather than
    being false-flagged as gibberish -- see `docs/QUALITY_FLAGS.md` section
    7 for this documented limitation.

    Parameters
    ----------
    col : IntoExpr
        The open-end text column.
    min_chars : int
        Minimum surviving letters (after normalization) required to score a
        row. Default `8` (SUBJECTIVE).

    Returns
    -------
    polars.Expr
        `Float64` in `[0, 1]`, higher = more suspicious. Null for null
        input or too-short/non-Latin text.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import gibberish_score
    >>> df = pl.DataFrame({"oe": ["asdkjfhaslkdjf", "The service was great"]})
    >>> df.select(score=gibberish_score("oe"))
    """
    expr = parse_into_expr(col)
    kwargs: Dict[str, Any] = {"min_chars": int(min_chars)}

    from polar_llama.expressions import get_lib_path

    return register_plugin(
        args=[expr],
        symbol="gibberish_score",
        is_elementwise=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )


def duplicate_answer_score(
    text_cols: Sequence["IntoExpr"], *, min_answer_chars: int = 10
) -> pl.Expr:
    """
    Cross-answer near-duplicate suspicion score across a respondent's
    open-end answers.

    Zero API calls. Backed by `crate::quality::duplicate_answer_score`
    (`src/quality.rs`): normalizes each non-null answer (lowercase, collapse
    whitespace/punctuation), drops answers shorter than `min_answer_chars`
    (so legitimate short repeats like "yes"/"n/a" across several open-ends
    don't trip this), and scores the maximum pairwise token-set Jaccard
    similarity over the survivors. `1.0` = a verbatim (post-normalization)
    repeat; the same mechanism catches "near" duplicates via partial
    Jaccard overlap.

    Parameters
    ----------
    text_cols : sequence of IntoExpr
        The open-end text columns to compare pairwise, one row per
        respondent.
    min_answer_chars : int
        Minimum normalized length for an answer to be compared. Default
        `10` (SUBJECTIVE).

    Returns
    -------
    polars.Expr
        `Float64` in `[0, 1]`, higher = more suspicious. Null when fewer
        than 2 answers survive the length filter.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import duplicate_answer_score
    >>> df = pl.DataFrame({
    ...     "oe1": ["The product is great and I would recommend it"],
    ...     "oe2": ["the product is great and I would recommend it."],
    ... })
    >>> df.select(score=duplicate_answer_score(["oe1", "oe2"]))
    """
    exprs = [parse_into_expr(c) for c in text_cols]
    if not exprs:
        raise ValueError("duplicate_answer_score: text_cols must have at least 1 column")

    kwargs: Dict[str, Any] = {"min_answer_chars": int(min_answer_chars)}

    from polar_llama.expressions import get_lib_path

    return register_plugin(
        args=exprs,
        symbol="duplicate_answer_score",
        is_elementwise=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )


# ============================================================================
# Heuristic tier -- pure Polars (response length / speeder)
# ============================================================================


def _response_length_score_and_z(
    col: "IntoExpr", rz_cap: float
) -> Tuple[pl.Expr, pl.Expr]:
    """Shared implementation for `response_length_score` and the
    `length_outlier` sub-struct in `quality_report` (which also needs the
    signed `z` alongside the score). A robust z-score over the whole
    (non-null) column: `(len - median) / (IQR / 1.349)`, so a single
    extreme value can't blow out the scale the way a mean/stddev z-score
    would.
    """
    expr = parse_into_expr(col)
    lengths = expr.str.len_chars().cast(pl.Float64)
    median = lengths.median()
    iqr = lengths.quantile(0.75) - lengths.quantile(0.25)

    rz = (
        pl.when(lengths.is_null())
        .then(None)
        .when(iqr == 0)
        .then(0.0)
        .otherwise((lengths - median) / (iqr / 1.349))
    )
    score = (
        pl.when(rz.is_null())
        .then(None)
        .otherwise(pl.min_horizontal(rz.abs() / rz_cap, pl.lit(1.0)))
    )
    return score, rz


def response_length_score(col: "IntoExpr", *, rz_cap: float = 3.0) -> pl.Expr:
    """
    Response-length outlier suspicion score for one open-end text column.

    Zero API calls, pure Polars (median/quantile -- there is nothing for a
    Rust plugin to buy here). A robust z-score of character length --
    `(len - median) / (IQR / 1.349)` -- so a single extreme value can't
    distort the scale (unlike a mean/stddev z-score). Symmetric: catches
    both suspiciously short *and* suspiciously long answers.

    Parameters
    ----------
    col : IntoExpr
        The open-end text column.
    rz_cap : float
        `|robust z| >= rz_cap` saturates the score at `1.0`. Default `3.0`
        (SUBJECTIVE).

    Returns
    -------
    polars.Expr
        `Float64` in `[0, 1]`, higher = more suspicious. Null for null
        input, or for the whole column when its interquartile range is 0
        (every non-null answer the same length, other than length 0 which
        legitimately scores 0 via the IQR==0 branch).

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import response_length_score
    >>> df = pl.DataFrame({"oe": ["fine"] * 10 + ["x" * 2000]})
    >>> df.select(score=response_length_score("oe"))
    """
    if rz_cap <= 0:
        raise ValueError(f"response_length_score: rz_cap must be > 0; got {rz_cap!r}")
    score, _z = _response_length_score_and_z(col, rz_cap)
    return score


def speeder_score(
    duration_col: "IntoExpr",
    *,
    percentile: float = 0.05,
    min_duration_seconds: Optional[float] = None,
    median_fraction: Optional[float] = None,
) -> pl.Expr:
    """
    Speeder ("rushed through the survey") suspicion score for a duration
    column.

    Zero API calls, pure Polars. Two modes:

    - **Percentile mode** (default): `p = rank(duration) / count`, score
      `= clamp((percentile - p) / percentile, 0, 1)`. The fastest
      respondent scores near `1.0`; a respondent at (or above) the
      `percentile`-th percentile scores `0.0`.
    - **Median-fraction mode** (`median_fraction` given): score
      `= clamp(1 - duration / (median * median_fraction), 0, 1)` -- the
      "faster than a fraction of median LOI" industry convention (e.g.
      `median_fraction=1/3`).

    `min_duration_seconds`, if given, forces the score to `1.0` for any
    duration below that absolute floor, regardless of mode.

    Parameters
    ----------
    duration_col : IntoExpr
        Survey completion duration in seconds (numeric).
    percentile : float
        Percentile-mode cutoff, in `(0, 1]`. Default `0.05` (SUBJECTIVE --
        `docs/QUALITY_FLAGS.md` notes the median-fraction alternative is
        also a common industry convention).
    min_duration_seconds : float, optional
        Absolute floor; durations below it always score `1.0`.
    median_fraction : float, optional
        Switches to median-fraction mode when given (must be > 0).

    Returns
    -------
    polars.Expr
        `Float64` in `[0, 1]`, higher = more suspicious ("more of a
        speeder"). Null for null input.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import speeder_score
    >>> df = pl.DataFrame({"duration_s": [300, 310, 295, 12, 305]})
    >>> df.select(score=speeder_score("duration_s"))
    """
    if not (0.0 < percentile <= 1.0):
        raise ValueError(f"speeder_score: percentile must be in (0, 1]; got {percentile!r}")
    if median_fraction is not None and median_fraction <= 0:
        raise ValueError(
            f"speeder_score: median_fraction must be > 0; got {median_fraction!r}"
        )

    expr = parse_into_expr(duration_col)

    if median_fraction is not None:
        med = expr.median()
        raw = 1.0 - expr / (med * median_fraction)
    else:
        n = expr.count()
        rank = expr.rank(method="average")
        p = rank / n
        raw = (percentile - p) / percentile

    score = (
        pl.when(expr.is_null())
        .then(None)
        .otherwise(pl.min_horizontal(pl.max_horizontal(raw, pl.lit(0.0)), pl.lit(1.0)))
    )

    if min_duration_seconds is not None:
        score = (
            pl.when(expr.is_null())
            .then(None)
            .when(expr < min_duration_seconds)
            .then(1.0)
            .otherwise(score)
        )

    return score


# ============================================================================
# Embedding/LLM tier -- response model + prompt
# ============================================================================

#: Mandatory limitation wording (issue #80, section 2.2) -- must also appear
#: (verbatim or near-verbatim) in `docs/QUALITY_FLAGS.md` section 7 and in
#: every docstring that discusses `likely_ai`/`ai_likelihood`.
AI_DETECTION_LIMITATION = (
    "AI-text detection is unreliable. This score is a stylistic heuristic, "
    "not evidence. Published detectors (including OpenAI's own, withdrawn "
    "in 2023) show high false-positive rates, especially for non-native "
    "English speakers, formal registers, and short texts. Treat `likely_ai` "
    "as a review-prioritization signal only -- never as sole grounds for "
    "exclusion or panelist sanction."
)


def _ai_likelihood_model() -> Type["BaseModel"]:
    """Response model for `ai_likelihood`/`likely_ai`. Every field required,
    no `Dict`-typed field -- OpenAI-strict-mode compliant by construction
    (see `_validate_strict_mode_schema`, the issue #51 lesson).
    """
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "Pydantic is required for the quality-flags LLM tier. Install with: "
            "pip install pydantic>=2.0.0"
        )

    class AILikelihood(BaseModel):
        score: float = Field(
            ...,
            description=(
                "Likelihood this text is AI-generated, 0.0 to 1.0. Calibrate "
                "to 0.5 when you cannot tell -- never claim certainty in "
                "either direction."
            ),
        )
        rationale: str = Field(
            ...,
            description=(
                "At most two sentences citing concrete stylistic markers "
                "(or their absence) that informed the score."
            ),
        )

    return AILikelihood


def _build_ai_likelihood_system_prompt() -> str:
    return "\n".join(
        [
            "You are assessing whether a short survey open-end response was",
            "written by an AI language model rather than a human respondent.",
            "",
            "Score stylistic markers only -- you cannot verify authorship:",
            "- Hedged, both-sides framing ('on one hand... on the other hand')",
            "- Unusually uniform sentence length and structure",
            "- Absence of typos, fragments, slang, or colloquialisms",
            "- List-like or formulaic structure (e.g. 'firstly... secondly...')",
            "- Generic, non-specific phrasing with no concrete personal detail",
            "",
            "Calibrate `score` to 0.5 when you genuinely cannot tell -- do not",
            "default to a high or low score out of a desire to seem decisive.",
            "Never claim certainty in either direction; this is a stylistic",
            "signal for human review, not a verdict.",
            "",
            AI_DETECTION_LIMITATION,
        ]
    )


def ai_likelihood(
    col: "IntoExpr",
    *,
    provider: Optional[Union[str, "Provider"]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    AI-generated-text likelihood score for one text column (flag, not
    verdict).

    One `inference_messages(..., response_model=...)` structured-output
    call per row, scoring stylistic markers (hedged both-sides framing,
    uniform sentence length, absence of typos/colloquialisms, list-like
    structure) and calibrated to `0.5` when the model is uncertain.

    **AI-text detection is unreliable. This score is a stylistic heuristic,
    not evidence.** Published detectors (including OpenAI's own, withdrawn
    in 2023) show high false-positive rates, especially for non-native
    English speakers, formal registers, and short texts. Treat this as a
    review-prioritization signal only -- never as sole grounds for
    exclusion or panelist sanction. See `docs/QUALITY_FLAGS.md` section 7.

    Parameters
    ----------
    col : IntoExpr
        The text column to assess.
    provider, model : optional
        Provider/model for the structured-output call.

    Returns
    -------
    polars.Expr
        `Struct{score: Float64, rationale: Utf8}`. `score` is clamped to
        `[0, 1]`; both fields are null on a parse/inference failure for that
        row (never an error, never a dropped row).

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import ai_likelihood, Provider  # doctest: +SKIP
    >>> df = pl.DataFrame({"oe": ["On one hand... on the other hand, overall satisfactory."]})  # doctest: +SKIP
    >>> df.with_columns(ai=ai_likelihood("oe", provider=Provider.OPENAI))  # doctest: +SKIP
    """
    expr = parse_into_expr(col)

    from polar_llama import combine_messages, inference_messages, string_to_message

    system_prompt = _build_ai_likelihood_system_prompt()
    system_message_expr = expr.map_batches(
        lambda s: pl.Series([system_prompt] * len(s)), return_dtype=pl.Utf8
    ).pipe(string_to_message, message_type="system")
    user_message_expr = expr.pipe(string_to_message, message_type="user")
    messages_expr = combine_messages(system_message_expr, user_message_expr)

    response_model = _ai_likelihood_model()
    result = inference_messages(
        messages_expr, provider=provider, model=model, response_model=response_model
    )

    ok = result.struct.field("_error").is_null()
    score = (
        pl.when(ok)
        .then(pl.min_horizontal(pl.max_horizontal(result.struct.field("score"), pl.lit(0.0)), pl.lit(1.0)))
        .otherwise(None)
    )
    rationale = pl.when(ok).then(result.struct.field("rationale")).otherwise(None)
    return pl.struct([score.alias("score"), rationale.alias("rationale")])


# ============================================================================
# QualityConfig / QualityReport
# ============================================================================

_THRESHOLD_FIELDS = (
    "straightlining_threshold",
    "length_outlier_threshold",
    "gibberish_threshold",
    "duplicate_threshold",
    "speeder_percentile",
    "near_duplicate_threshold",
    "ai_likelihood_threshold",
)


@dataclass
class QualityConfig:
    """
    Configuration for `quality_report`. All thresholds default to
    documented, SUBJECTIVE conventions -- see `docs/QUALITY_FLAGS.md`
    section 7 and tune to your own panel.

    The heuristic tier (grid/text/duration scoring) needs zero API calls
    and never drops a row. The embedding/LLM tier (`llm_tier=True`) is
    strictly opt-in -- default `False`, zero network calls.
    """

    id_column: Optional[str] = None
    grid_columns: Sequence[str] = field(default_factory=tuple)
    text_columns: Sequence[str] = field(default_factory=tuple)
    duration_column: Optional[str] = None

    # Likert scale bounds for straightlining's variance normalization.
    # None (default) infers the bounds from the observed data.
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None

    # Thresholds: score >= threshold => flag (speeder is `score > 0`
    # instead -- see `speeder_score`'s docstring). All SUBJECTIVE defaults.
    straightlining_threshold: float = 0.85
    length_outlier_threshold: float = 1.0  # == |robust z| >= 3
    gibberish_threshold: float = 0.65
    duplicate_threshold: float = 0.8
    speeder_percentile: float = 0.05
    min_duration_seconds: Optional[float] = None
    speeder_median_fraction: Optional[float] = None

    # Scoring knobs (rarely need tuning; see the standalone functions).
    straightlining_min_answers: int = 3
    gibberish_min_chars: int = 8
    duplicate_min_answer_chars: int = 10
    response_length_rz_cap: float = 3.0

    # Embedding/LLM tier -- off by default (zero API calls).
    llm_tier: bool = False
    near_duplicate_threshold: float = 0.97
    ai_likelihood_threshold: float = 0.75
    embedding_column: Optional[str] = None
    embedding_provider: Optional[Union[str, "Provider"]] = None
    embedding_model: Optional[str] = None
    provider: Optional[Union[str, "Provider"]] = None  # for likely_ai
    model: Optional[str] = None

    def __post_init__(self) -> None:
        for name in _THRESHOLD_FIELDS:
            value = getattr(self, name)
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"QualityConfig.{name} must be in the open-closed interval "
                    f"(0, 1]; got {value!r}"
                )

        if not (self.grid_columns or self.text_columns or self.duration_column):
            raise ValueError(
                "QualityConfig: at least one of grid_columns, text_columns, "
                "or duration_column must be configured"
            )

        if (
            self.scale_min is not None
            and self.scale_max is not None
            and self.scale_max <= self.scale_min
        ):
            raise ValueError(
                f"QualityConfig: scale_max ({self.scale_max!r}) must be > "
                f"scale_min ({self.scale_min!r})"
            )

        if self.llm_tier and not self.text_columns:
            raise ValueError(
                "QualityConfig: llm_tier=True requires text_columns to be "
                "configured (near_duplicate and likely_ai both operate on "
                "open-end text)"
            )


@dataclass
class QualityReport:
    """Return value of `quality_report`.

    `df`: the input DataFrame plus one struct column (`quality` by
    default) -- same height and row order as the input, always (flagging
    never filters). `summary`: one row per active flag, plus a final
    `any_flag` row.
    """

    df: pl.DataFrame
    summary: pl.DataFrame


# ============================================================================
# Embedding/LLM tier -- DataFrame-shaped private helpers
# ============================================================================


def _near_duplicate_flags(df: pl.DataFrame, config: QualityConfig) -> pl.DataFrame:
    """Cross-respondent near-duplicate detection over the concatenated
    open-end answers (or a precomputed `config.embedding_column`). Returns
    `df` plus `__q_near_duplicate_score`/`__q_near_duplicate_neighbor_id`
    columns, same height and order as `df`.
    """
    from polar_llama import cosine_similarity, embedding_async, knn_hnsw

    work = df.with_columns(__q_row_nr=pl.int_range(pl.len()))
    id_expr = pl.col(config.id_column) if config.id_column is not None else pl.col("__q_row_nr")

    if config.embedding_column is not None:
        work = work.with_columns(__nd_emb=pl.col(config.embedding_column))
    else:
        concat_expr = pl.concat_str(
            [pl.col(c).fill_null("") for c in config.text_columns], separator=" || "
        )
        work = work.with_columns(__nd_text=concat_expr).with_columns(
            __nd_emb=embedding_async(
                pl.col("__nd_text"),
                provider=config.embedding_provider,
                model=config.embedding_model,
            )
        )

    non_null = work.filter(
        pl.col("__nd_emb").is_not_null() & (pl.col("__nd_emb").list.len() > 0)
    )

    if non_null.height < 2:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("__q_near_duplicate_score"),
            pl.lit(None).alias("__q_near_duplicate_neighbor_id"),
        )

    non_null = non_null.with_columns(
        __nd_neighbors=knn_hnsw(pl.col("__nd_emb"), pl.col("__nd_emb"), k=2)
    ).with_columns(
        __nd_neighbor_idx=pl.when(pl.col("__nd_neighbors").list.len() >= 2)
        .then(pl.col("__nd_neighbors").list.get(1))
        .otherwise(None)
    )
    non_null = non_null.with_columns(
        __nd_neighbor_idx_filled=pl.col("__nd_neighbor_idx").fill_null(0)
    ).with_columns(
        __nd_neighbor_emb=pl.col("__nd_emb").gather(pl.col("__nd_neighbor_idx_filled")),
        __nd_neighbor_id=id_expr.gather(pl.col("__nd_neighbor_idx_filled")),
    )
    non_null = non_null.with_columns(
        __q_near_duplicate_score=pl.when(pl.col("__nd_neighbor_idx").is_not_null())
        .then(cosine_similarity(pl.col("__nd_emb"), pl.col("__nd_neighbor_emb")))
        .otherwise(None),
        __q_near_duplicate_neighbor_id=pl.when(pl.col("__nd_neighbor_idx").is_not_null())
        .then(pl.col("__nd_neighbor_id"))
        .otherwise(None),
    )

    keyed = non_null.select(
        "__q_row_nr", "__q_near_duplicate_score", "__q_near_duplicate_neighbor_id"
    )
    joined = work.select("__q_row_nr").join(keyed, on="__q_row_nr", how="left")
    assert joined.height == df.height

    return df.with_columns(
        joined["__q_near_duplicate_score"].alias("__q_near_duplicate_score"),
        joined["__q_near_duplicate_neighbor_id"].alias("__q_near_duplicate_neighbor_id"),
    )


def _ai_likelihood_flags(df: pl.DataFrame, config: QualityConfig) -> pl.DataFrame:
    """Per-respondent AI-likelihood scoring over the concatenated open-end
    answers. Returns `df` plus `__q_likely_ai_score`/`__q_likely_ai_rationale`
    columns, same height and order as `df`. Rows with no non-empty text
    across `config.text_columns` score null without an inference call
    happening to matter for their result (never flagged on missing data).
    """
    concat_expr = pl.concat_str(
        [pl.col(c).fill_null("") for c in config.text_columns], separator="\n\n"
    )
    has_text_expr = pl.any_horizontal(
        [
            pl.col(c).is_not_null() & (pl.col(c).str.len_chars() > 0)
            for c in config.text_columns
        ]
    )

    work = df.with_columns(__ai_text=concat_expr, __ai_has_text=has_text_expr)
    work = work.with_columns(
        __ai_result=ai_likelihood(
            pl.col("__ai_text"), provider=config.provider, model=config.model
        )
    )
    work = work.with_columns(
        __q_likely_ai_score=pl.when(pl.col("__ai_has_text"))
        .then(pl.col("__ai_result").struct.field("score"))
        .otherwise(None),
        __q_likely_ai_rationale=pl.when(pl.col("__ai_has_text"))
        .then(pl.col("__ai_result").struct.field("rationale"))
        .otherwise(None),
    )

    return df.with_columns(
        work["__q_likely_ai_score"].alias("__q_likely_ai_score"),
        work["__q_likely_ai_rationale"].alias("__q_likely_ai_rationale"),
    )


# ============================================================================
# quality_report: DataFrame-in / DataFrame-out orchestration
# ============================================================================


def _validate_columns(df: pl.DataFrame, config: QualityConfig) -> None:
    missing: List[str] = []
    for c in (*config.grid_columns, *config.text_columns):
        if c not in df.columns:
            missing.append(c)
    if config.duration_column is not None and config.duration_column not in df.columns:
        missing.append(config.duration_column)
    if config.id_column is not None and config.id_column not in df.columns:
        missing.append(config.id_column)
    if config.embedding_column is not None and config.embedding_column not in df.columns:
        missing.append(config.embedding_column)
    if missing:
        raise ValueError(
            f"quality_report: column(s) not found in dataframe: {sorted(set(missing))!r}"
        )


def _infer_grid_scale(df: pl.DataFrame, config: QualityConfig) -> Tuple[Optional[float], Optional[float]]:
    if config.scale_min is not None and config.scale_max is not None:
        return config.scale_min, config.scale_max
    stacked = pl.concat([df[c].cast(pl.Float64, strict=False) for c in config.grid_columns])
    inferred_min = stacked.min()
    inferred_max = stacked.max()
    scale_min = config.scale_min if config.scale_min is not None else inferred_min
    scale_max = config.scale_max if config.scale_max is not None else inferred_max
    return scale_min, scale_max


def quality_report(
    df: pl.DataFrame,
    config: QualityConfig,
    *,
    output_column: str = "quality",
) -> QualityReport:
    """
    Score every respondent (row) in `df` for common survey data-quality
    issues and return a per-flag summary. DataFrame-in / DataFrame-out.

    **Never drops a row.** Every flag is derived from a graded `[0, 1]`
    score (`score >= threshold`, except `speeder` which is `score > 0`);
    a null score (not enough signal) always resolves to `flag = False`.
    The heuristic tier (grid/text/duration scoring) makes zero API calls;
    the embedding/LLM tier (`config.llm_tier=True`) is strictly opt-in.

    Parameters
    ----------
    df : polars.DataFrame
        Input data, one row per respondent.
    config : QualityConfig
        Which columns to score and every threshold. See `QualityConfig`.
    output_column : str
        Name of the appended struct column. Default `"quality"`.

    Returns
    -------
    QualityReport
        `.df`: `df` plus the `output_column` struct (same height and row
        order as `df`, always). Sub-structs for unconfigured inputs (e.g.
        no `duration_column`) are omitted entirely, not null-filled --
        the struct's schema reflects `config`. `.summary`: one row per
        active flag (`flag, n_scored, n_flagged, rate, threshold,
        mean_score`), plus a final `any_flag` row.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import QualityConfig, quality_report
    >>> df = pl.DataFrame({
    ...     "g1": [4, 2], "g2": [4, 4], "g3": [4, 3], "g4": [4, 5], "g5": [4, 1],
    ...     "oe1": ["fine", "The product exceeded my expectations, truly"],
    ...     "duration_s": [300, 290],
    ... })
    >>> config = QualityConfig(
    ...     grid_columns=["g1", "g2", "g3", "g4", "g5"],
    ...     text_columns=["oe1"],
    ...     duration_column="duration_s",
    ... )
    >>> result = quality_report(df, config)
    >>> result.df.height == df.height
    True
    >>> result.summary.columns
    ['flag', 'n_scored', 'n_flagged', 'rate', 'threshold', 'mean_score']
    """
    _validate_columns(df, config)

    work = df
    # name -> {"score_col", "threshold", "extra": {field_name: col_name}, "flag_rule"}
    active: "Dict[str, Dict[str, Any]]" = {}

    if config.grid_columns:
        scale_min, scale_max = _infer_grid_scale(df, config)
        work = work.with_columns(
            straightlining_score(
                list(config.grid_columns),
                scale_min=scale_min,
                scale_max=scale_max,
                min_answers=config.straightlining_min_answers,
            ).alias("__q_straightlining_score")
        )
        active["straightlining"] = {
            "score_col": "__q_straightlining_score",
            "threshold": config.straightlining_threshold,
        }

    if config.text_columns:
        gib_cols = []
        for i, c in enumerate(config.text_columns):
            colname = f"__q_gibberish_{i}"
            work = work.with_columns(
                gibberish_score(pl.col(c), min_chars=config.gibberish_min_chars).alias(colname)
            )
            gib_cols.append(colname)
        work = work.with_columns(
            pl.max_horizontal(*[pl.col(c) for c in gib_cols]).alias("__q_gibberish_score")
        )
        active["gibberish"] = {
            "score_col": "__q_gibberish_score",
            "threshold": config.gibberish_threshold,
        }

        len_score_cols, len_z_cols = [], []
        for i, c in enumerate(config.text_columns):
            s_col, z_col = f"__q_len_score_{i}", f"__q_len_z_{i}"
            s_expr, z_expr = _response_length_score_and_z(
                pl.col(c), config.response_length_rz_cap
            )
            work = work.with_columns(s_expr.alias(s_col), z_expr.alias(z_col))
            len_score_cols.append(s_col)
            len_z_cols.append(z_col)
        work = work.with_columns(
            pl.max_horizontal(*[pl.col(c) for c in len_score_cols]).alias(
                "__q_length_outlier_score"
            )
        )
        work = work.with_columns(
            pl.coalesce(
                [
                    pl.when(pl.col(sc) == pl.col("__q_length_outlier_score")).then(pl.col(zc))
                    for sc, zc in zip(len_score_cols, len_z_cols)
                ]
            ).alias("__q_length_outlier_z")
        )
        active["length_outlier"] = {
            "score_col": "__q_length_outlier_score",
            "threshold": config.length_outlier_threshold,
            "extra": {"z": "__q_length_outlier_z"},
        }

        work = work.with_columns(
            duplicate_answer_score(
                list(config.text_columns), min_answer_chars=config.duplicate_min_answer_chars
            ).alias("__q_duplicate_score")
        )
        active["duplicate_answers"] = {
            "score_col": "__q_duplicate_score",
            "threshold": config.duplicate_threshold,
        }

    if config.duration_column:
        work = work.with_columns(
            speeder_score(
                pl.col(config.duration_column),
                percentile=config.speeder_percentile,
                min_duration_seconds=config.min_duration_seconds,
                median_fraction=config.speeder_median_fraction,
            ).alias("__q_speeder_score")
        )
        active["speeder"] = {
            "score_col": "__q_speeder_score",
            "threshold": 0.0,
            "flag_rule": "gt",
        }

    if config.llm_tier:
        nd = _near_duplicate_flags(df, config)
        ai = _ai_likelihood_flags(df, config)
        assert nd.height == df.height, "near-duplicate helper must preserve row count"
        assert ai.height == df.height, "ai-likelihood helper must preserve row count"

        work = work.with_columns(
            nd["__q_near_duplicate_score"].alias("__q_near_duplicate_score"),
            nd["__q_near_duplicate_neighbor_id"].alias("__q_near_duplicate_neighbor_id"),
            ai["__q_likely_ai_score"].alias("__q_likely_ai_score"),
            ai["__q_likely_ai_rationale"].alias("__q_likely_ai_rationale"),
        )
        active["near_duplicate"] = {
            "score_col": "__q_near_duplicate_score",
            "threshold": config.near_duplicate_threshold,
            "extra": {"neighbor_id": "__q_near_duplicate_neighbor_id"},
        }
        active["likely_ai"] = {
            "score_col": "__q_likely_ai_score",
            "threshold": config.ai_likelihood_threshold,
            "extra": {"rationale": "__q_likely_ai_rationale"},
        }

    # -- Flag columns: score >= threshold (speeder: score > 0), null-safe --
    for name, info in active.items():
        score_col = info["score_col"]
        threshold = info["threshold"]
        if info.get("flag_rule") == "gt":
            flag_expr = (pl.col(score_col) > threshold).fill_null(False)
        else:
            flag_expr = (pl.col(score_col) >= threshold).fill_null(False)
        flag_col = f"{score_col}__flag"
        work = work.with_columns(flag_expr.alias(flag_col))
        info["flag_col"] = flag_col

    # -- Assemble the nested `quality` struct --
    struct_field_exprs = []
    for name, info in active.items():
        parts = {"score": pl.col(info["score_col"]), "flag": pl.col(info["flag_col"])}
        extra = info.get("extra", {})
        for field_name, col_name in extra.items():
            parts[field_name] = pl.col(col_name)
        ordered = ["score", *extra.keys(), "flag"]
        struct_field_exprs.append(
            pl.struct([parts[n].alias(n) for n in ordered]).alias(name)
        )

    if active:
        flag_cols = [pl.col(info["flag_col"]) for info in active.values()]
        any_flag_expr = pl.any_horizontal(*flag_cols)
        n_flags_expr = pl.sum_horizontal(*[c.cast(pl.UInt32) for c in flag_cols])
    else:
        any_flag_expr = pl.lit(False)
        n_flags_expr = pl.lit(0).cast(pl.UInt32)

    work = work.with_columns(
        any_flag_expr.alias("__q_any_flag"), n_flags_expr.alias("__q_n_flags")
    )

    final_struct = pl.struct(
        [*struct_field_exprs, pl.col("__q_any_flag").alias("any_flag"), pl.col("__q_n_flags").alias("n_flags")]
    ).alias(output_column)
    work = work.with_columns(final_struct)

    result_df = df.with_columns(**{output_column: work[output_column]})
    assert result_df.height == df.height, "quality_report must never change row count"

    # -- Summary: one row per active flag + a final any_flag row --
    summary_rows: List[Dict[str, Any]] = []
    for name, info in active.items():
        score_col, flag_col = info["score_col"], info["flag_col"]
        stats = work.select(
            n_scored=pl.col(score_col).is_not_null().sum(),
            n_flagged=pl.col(flag_col).sum(),
            mean_score=pl.col(score_col).mean(),
        ).row(0, named=True)
        n_scored = int(stats["n_scored"])
        n_flagged = int(stats["n_flagged"])
        summary_rows.append(
            {
                "flag": name,
                "n_scored": n_scored,
                "n_flagged": n_flagged,
                "rate": (n_flagged / n_scored) if n_scored else 0.0,
                "threshold": float(info["threshold"]),
                "mean_score": stats["mean_score"],
            }
        )

    n_scored_any = df.height
    n_flagged_any = int(work.select(pl.col("__q_any_flag").sum()).item())
    summary_rows.append(
        {
            "flag": "any_flag",
            "n_scored": n_scored_any,
            "n_flagged": n_flagged_any,
            "rate": (n_flagged_any / n_scored_any) if n_scored_any else 0.0,
            "threshold": None,
            "mean_score": None,
        }
    )

    summary_df = pl.DataFrame(
        summary_rows,
        schema={
            "flag": pl.Utf8,
            "n_scored": pl.UInt32,
            "n_flagged": pl.UInt32,
            "rate": pl.Float64,
            "threshold": pl.Float64,
            "mean_score": pl.Float64,
        },
    )

    return QualityReport(df=result_df, summary=summary_df)
