# Survey Data-Quality Flags

## Overview

`quality_report` scores every respondent (row) in a survey DataFrame for
common data-quality problems -- straightlining, gibberish/keyboard-mash
open-ends, duplicate answers, response-length outliers, speeding through
the survey, cross-respondent near-duplicates, and (opt-in) likely-AI-generated
text -- and returns a per-flag summary. It is DataFrame-in/DataFrame-out,
the same shape as `induce_codebook`: a single Polars expression can't return
both per-row flags and a per-flag summary table (two different heights) at
once.

**Every score is graded, in `[0, 1]`, higher = more suspicious. Booleans
(`flag`) are derived by thresholding a score; a null score (not enough
signal -- too few grid answers, too little text, missing duration) always
resolves to `flag = False`, never `True`. `quality_report` never drops a
row.**

Two tiers:

- **Heuristic tier** (always available, zero API calls): `straightlining_score`,
  `gibberish_score`, `duplicate_answer_score` are Rust plugin expressions
  (`src/quality.rs` pure functions, unit-tested there, plus the Polars
  `Series`/null plumbing in `src/expressions.rs` -- mirrors the
  `src/metrics.rs` / issue #79 split). `response_length_score` and
  `speeder_score` are a few lines of pure Polars (median/quantile/rank) --
  there is nothing for a Rust plugin to buy there.
- **Embedding/LLM tier** (opt-in, `QualityConfig(llm_tier=True)`, default
  `False`): cross-respondent near-duplicate detection (`embedding_async` +
  `knn_hnsw` + `cosine_similarity`) and a likely-AI-generated-text flag
  (`inference_messages(..., response_model=...)`), pure Python composition
  of existing primitives -- exactly the `induce_codebook` pattern.

## Quick start

```python
import polars as pl
from polar_llama import QualityConfig, quality_report

df = pl.DataFrame({
    "respondent_id": ["r1", "r2", "r3"],
    "g1": [4, 2, 3], "g2": [4, 4, 2], "g3": [4, 3, 5], "g4": [4, 5, 1], "g5": [4, 1, 4],
    "oe1": ["fine", "The delivery was quick and the packaging held up well.", "Support was slow but eventually helpful."],
    "duration_s": [40, 305, 298],
})

config = QualityConfig(
    id_column="respondent_id",
    grid_columns=["g1", "g2", "g3", "g4", "g5"],
    text_columns=["oe1"],
    duration_column="duration_s",
)

result = quality_report(df, config)
result.df.height == df.height  # True -- never drops a row
result.df.select("respondent_id", "quality").unnest("quality")
result.summary
```

`result.df` gains one struct column (`quality` by default, override with
`quality_report(df, config, output_column=...)`):

```
Struct{
    straightlining:    Struct{score, flag},
    gibberish:         Struct{score, flag},          # max over text_columns
    length_outlier:    Struct{score, z, flag},        # max over text_columns
    duplicate_answers: Struct{score, flag},
    speeder:           Struct{score, flag},
    near_duplicate:    Struct{score, neighbor_id, flag},   # only when llm_tier=True
    likely_ai:         Struct{score, rationale, flag},     # only when llm_tier=True
    any_flag: bool,
    n_flags: u32,
}
```

Sub-structs for unconfigured inputs are **omitted from the schema
entirely**, not null-filled -- e.g. no `duration_column` means no `speeder`
field at all.

`result.summary` is one row per active flag, plus a final `any_flag` row:

| flag | n_scored | n_flagged | rate | threshold | mean_score |
|---|---|---|---|---|---|

## Heuristic tier: formulas and default thresholds

All scores are Float64 in `[0, 1]`, higher = more suspicious. **Every
default threshold below is a documented, SUBJECTIVE convention** -- not a
validated statistical cutoff. Tune them to your own panel.

### `straightlining_score` (default threshold `0.85`)

Over a respondent's non-null grid (Likert) answers `x_1..x_m` (requires
`m >= min_answers`, default `3`, else null):

- `mode_frac = count(most frequent value) / m`.
- `var_norm = variance(x) / max_var`, where `max_var = ((scale_max - scale_min) / 2)^2`
  -- the maximum possible variance for a value bouncing between the scale's
  endpoints, clipped to `[0, 1]`.
- **score `= max(mode_frac, 1 - var_norm)`**, clipped to `[0, 1]`.

A pure straightliner (all 4s on a 1-5 scale) scores `1.0` on both
components. An alternating pattern (1, 5, 1, 5) has few distinct values but
*maximal* variance, so `1 - var_norm` is near 0 and the row is correctly
**not** flagged despite the low distinct-value count.

`scale_min`/`scale_max` are inferred from the observed min/max across the
configured grid columns when not given explicitly in `QualityConfig`.

### `gibberish_score` (default threshold `0.65`)

**English-only heuristic** -- normalizes to lowercase `[a-z ]` (strips
digits/punctuation/non-Latin scripts); text with fewer than `min_chars`
(default `8`) surviving letters scores **null**, not a false flag. Three
weighted components (weights `0.5`/`0.3`/`0.2` -- SUBJECTIVE, unit-tested
for ordering, not exact values):

- `consonant_run` (weight `0.5`): longest run of consecutive consonants,
  normalized by a 6-char cap.
- `vowel_dev` (weight `0.3`): relative deviation of the letter-only vowel
  ratio from English's ~0.38.
- `entropy_term` (weight `0.2`): normalized Shannon entropy of character
  bigrams -- weak alone (English isn't low-entropy either), a tie-breaker.

`gibberish_score("asdkjfhaslkdjf") > gibberish_score(<normal English sentence>) + 0.3`
is a pinned regression fixture (`src/quality.rs` and
`tests/test_quality_flags.py`).

**Non-Latin scripts normalize to ~nothing and score null** -- this
heuristic cannot judge non-English text and deliberately does not
false-flag it as gibberish. If your panel is multilingual, treat
`gibberish` as English-open-end-only and don't rely on it for other
languages.

### `duplicate_answer_score` (default threshold `0.8`)

Normalizes each non-null answer (lowercase, collapse whitespace/
punctuation) and drops answers shorter than `min_answer_chars` (default
`10` -- so legitimate short repeats like "yes"/"n/a" across several
open-ends don't trip this). **score `= max pairwise token-set Jaccard
similarity`** (`|A ∩ B| / |A ∪ B|`) over the survivors -- `1.0` for a
verbatim (post-normalization) repeat. One mechanism catches both "exact"
and "near" duplicates; requires >= 2 surviving answers, else null.

### `response_length_score` (default threshold `1.0`, i.e. `|robust z| >= 3`)

Pure Polars (median/quantile -- no Rust plugin). Robust z-score of
character length: `rz = (len - median) / (IQR / 1.349)` (`IQR == 0` scores
`0.0` for every length, since there's no spread to be an outlier from).
**score `= min(1, |rz| / rz_cap)`**, `rz_cap = 3.0` (SUBJECTIVE) -- `|rz| >= 3`
saturates at `1.0`. Symmetric: catches both suspiciously short *and*
suspiciously long answers. The signed `rz` is exposed as `length_outlier.z`
in the report struct so you can tell which direction tripped it.

### `speeder_score` (flags when `score > 0`, default `percentile = 0.05`)

Pure Polars. Two modes:

- **Percentile mode** (default): `p = rank(duration) / count`,
  **score `= clamp((percentile - p) / percentile, 0, 1)`**. The fastest
  respondent scores near `1.0`; anyone at or above the `percentile`-th
  percentile scores `0.0`. `percentile = 0.05` is SUBJECTIVE -- some panels
  instead use "faster than a fraction of median LOI"
  (`median_fraction=1/3`, say), available via the `median_fraction=`
  alternative mode: **score `= clamp(1 - duration / (median * median_fraction), 0, 1)`**.
- `min_duration_seconds`, if set, forces the score to `1.0` for any
  duration below that absolute floor, in either mode.

Note the flag rule for `speeder` is `score > 0` (not `score >= threshold`
like the others) -- the summary row reports `threshold = 0.0` accordingly.

## Standalone expressions and the `.llama` namespace

Every heuristic-tier score is also a standalone Polars expression (used
internally by `quality_report`, and directly usable on their own):

```python
from polar_llama import (
    straightlining_score, gibberish_score, duplicate_answer_score,
    response_length_score, speeder_score,
)

df.select(
    sl=straightlining_score(["g1", "g2", "g3", "g4", "g5"]),
    gib=gibberish_score("oe1"),
    dup=duplicate_answer_score(["oe1", "oe2", "oe3"]),
    len_out=response_length_score("oe1"),
    speed=speeder_score("duration_s"),
)
```

And via the `.llama` namespace (multi-column ones take an `others:`
sequence, mirroring `.llama.krippendorffs_alpha`):

```python
pl.col("g1").llama.straightlining_score(["g2", "g3", "g4", "g5"])
pl.col("oe1").llama.gibberish_score()
pl.col("oe1").llama.duplicate_answer_score(["oe2", "oe3"])
pl.col("oe1").llama.response_length_score()
pl.col("duration_s").llama.speeder_score()
```

## Embedding/LLM tier (opt-in)

Set `QualityConfig(llm_tier=True, text_columns=[...], ...)`. Requires
`text_columns` to be configured (raised as `ValueError` in
`QualityConfig.__post_init__` otherwise). Makes API calls -- embeddings for
every respondent (unless `embedding_column=` points at a precomputed
column) plus one structured-output call per respondent for `likely_ai`.

### `near_duplicate` (default threshold `0.97`)

Concatenates each respondent's open-end answers, embeds them
(`embedding_async`), finds each respondent's nearest *other* respondent by
cosine similarity (`knn_hnsw` + `cosine_similarity`), and scores
`near_duplicate.score = cosine_similarity(self, nearest_other)`.
`near_duplicate.neighbor_id` carries that nearest respondent's
`id_column` value (or row index if `id_column` isn't set).

`0.97` is **embedding-model-specific** (SUBJECTIVE) -- for
`text-embedding-3-small`, paraphrases typically land ~0.85-0.93 and
verbatim copies > 0.98; recalibrate for other embedding models.

### `likely_ai` (default threshold `0.75`) -- flag, not verdict

One `inference_messages(..., response_model=AILikelihood)` call per
respondent, scoring stylistic markers (hedged both-sides framing, uniform
sentence length, absence of typos/colloquialisms, list-like structure) and
calibrated to `0.5` when the model is uncertain. Also exposed standalone as
`ai_likelihood(col, provider=..., model=...)`, returning
`Struct{score, rationale}`.

> **AI-text detection is unreliable. This score is a stylistic heuristic,
> not evidence.** Published detectors (including OpenAI's own, withdrawn in
> 2023) show high false-positive rates, especially for non-native English
> speakers, formal registers, and short texts. Treat `likely_ai` as a
> review-prioritization signal only -- **never** as sole grounds for
> exclusion or panelist sanction.

This warning is not optional decoration: non-native-English-speaker text
tends to read as "unusually uniform" or "less colloquial" to these
heuristics, which is exactly the false-positive failure mode documented
above. Use `likely_ai` to prioritize which responses a human reviews, never
to auto-exclude a respondent or flag a panelist for sanction on its own.

## Config reference (`QualityConfig`)

```python
@dataclass
class QualityConfig:
    id_column: Optional[str] = None
    grid_columns: Sequence[str] = ()
    text_columns: Sequence[str] = ()
    duration_column: Optional[str] = None
    scale_min: Optional[float] = None       # None = infer from grid_columns
    scale_max: Optional[float] = None
    straightlining_threshold: float = 0.85  # SUBJECTIVE
    length_outlier_threshold: float = 1.0   # == |robust z| >= 3, SUBJECTIVE
    gibberish_threshold: float = 0.65       # SUBJECTIVE
    duplicate_threshold: float = 0.8        # SUBJECTIVE
    speeder_percentile: float = 0.05        # SUBJECTIVE
    min_duration_seconds: Optional[float] = None
    speeder_median_fraction: Optional[float] = None
    straightlining_min_answers: int = 3     # SUBJECTIVE floor
    gibberish_min_chars: int = 8
    duplicate_min_answer_chars: int = 10    # SUBJECTIVE floor
    response_length_rz_cap: float = 3.0
    llm_tier: bool = False                  # off by default => zero API calls
    near_duplicate_threshold: float = 0.97  # SUBJECTIVE, embedding-model-specific
    ai_likelihood_threshold: float = 0.75   # SUBJECTIVE
    embedding_column: Optional[str] = None
    embedding_provider: Optional[Union[str, Provider]] = None
    embedding_model: Optional[str] = None
    provider: Optional[Union[str, Provider]] = None   # for likely_ai
    model: Optional[str] = None
```

Validated in `__post_init__`: every threshold must be in `(0, 1]`; at least
one of `grid_columns`/`text_columns`/`duration_column` must be configured;
`scale_max` must be `> scale_min` when both are given; `llm_tier=True`
requires `text_columns`. Column-existence is checked in `quality_report`
itself (raises `ValueError` naming the missing column(s), since it needs
the DataFrame to check against).

## Section 7: subjective choices (must-read before tuning)

1. **All five heuristic-tier thresholds** (`0.85` straightlining / `|z| >= 3`
   length outlier / `0.65` gibberish / `0.8` duplicate / 5th-percentile
   speeder) are conventions, not validated cutoffs. Expect to tune every
   one of them against your own panel's false-positive rate.
2. **Gibberish component weights** (`0.5`/`0.3`/`0.2`) and the English
   vowel-ratio anchor (`0.38`) are English-only and subjective. Non-Latin
   scripts return **null** (normalization strips them below `min_chars`)
   rather than a false gibberish flag -- this heuristic simply cannot judge
   non-English text.
3. **`near_duplicate_threshold = 0.97`** is specific to the embedding model
   in use (documented for `text-embedding-3-small`; recalibrate for others).
4. **`ai_likelihood_threshold = 0.75` and the entire `likely_ai` flag**
   carry the false-positive limitation above, including the
   non-native-speaker bias warning, in every docstring that mentions it.
   Flag-not-verdict framing is mandatory wherever this score is surfaced.
5. **`straightlining_min_answers = 3`** and **`duplicate_min_answer_chars = 10`**
   are floors below which there's judged to be "not enough signal" -- both
   subjective, both documented here.

## Testing

- `tests/test_quality_flags.py`: CI-safe (zero API calls, no `mlx`), covers
  the full heuristic tier -- synthetic respondents each engineered to trip
  exactly one flag, graded-score ordering, height/order preservation,
  config validation, null-safety, and summary shape.
- `tests/test_quality_llm.py`: gated on `OPENAI_API_KEY`
  (`@pytest.mark.skipif`), exercises `near_duplicate` and `likely_ai`
  against the real API.
- `src/quality.rs`: `#[cfg(test)]` unit tests for the three Rust pure
  functions, including edge cases (empty input, all-null, unicode/non-Latin
  text, exact duplicates, degenerate scale bounds).
