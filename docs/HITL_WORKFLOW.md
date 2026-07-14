# Human-in-the-loop Review Loop

## Overview

Polar Llama's prompt-optimization engine (`polar_llama/optimize.py`:
`Signature`, `Predict`, `BootstrapFewShot`) tunes a module against a
labeled DataFrame. In practice the labels you have on day one are the
model's own predictions, not ground truth -- someone has to look at a
sample of what the LLM produced, correct the mistakes, and feed those
corrections back in. `polar_llama/hitl.py` (issue #81) is that loop, as
four DataFrame-in/DataFrame-out functions:

```
   code with an LLM
         |
         v
  export_review_sample   -- stratified, optionally confidence-weighted
         |                  sample -> CSV/XLSX for a human reviewer
         v
   (human reviews, fills in `corrected_code`/`review_notes`)
         |
         v
  import_corrections      -- joins corrections back onto the full
         |                   DataFrame; computes kappa/agreement stats
         |                   (reuses polar_llama.reliability.cohens_kappa)
         v
  retune_from_corrections -- corrections -> a BootstrapFewShot trainset
         |                   -> a compiled Predict (reuses
         |                   polar_llama.optimize.BootstrapFewShot)
         v
      evaluate(...) on a holdout set, then go again
```

**Like `induce_codebook`/`quality_report`, these are DataFrame-level
orchestration functions, not row-wise expressions** -- none of them are
added to the `.llama` expression namespace.

**No reimplementation.** `retune_from_corrections` builds a trainset and
hands it straight to the existing `BootstrapFewShot(...).compile(...)`;
`import_corrections` computes its agreement statistics with the existing
`cohens_kappa` aggregation expression. See `polar_llama/hitl.py`'s module
docstring for the exact contract each of those reuses depends on.

## Quick start

```python
import polars as pl
from polar_llama import (
    Predict, Signature, evaluate,
    export_review_sample, import_corrections, retune_from_corrections,
)

# 1. Code a DataFrame with an LLM module (see docs on Prompt Optimization
#    in the README for `Predict`/`Signature`).
module = Predict(Signature("text -> code"), provider="openai", model="gpt-4o-mini")
coded = module(df).rename({"pred_code": "code"})

# 2. Export a sample for human review -- stratified by an existing code
#    column, oversampling rows the model itself was least sure about.
sample = export_review_sample(
    coded,
    n=100,
    strata="code",
    confidence_column="confidence",       # optional
    oversample_low_confidence=4.0,        # optional
    path="review_batch_1.csv",
)

# 3. A human opens review_batch_1.csv, fills in `corrected_code` for the
#    rows that need it (leaves it blank for rows that were already right),
#    optionally jots `review_notes`, and hands the file back.
corrections = pl.read_csv("review_batch_1_reviewed.csv")

# 4. Join the corrections back onto the full coded DataFrame and score
#    human/LLM agreement.
result = import_corrections(coded, corrections, code_column="code")
print(result.kappa, result.agreement_rate, result.n_reviewed, result.n_changed)

# 5. Retune: mine few-shot demos from the corrected rows.
tuned = retune_from_corrections(
    result.df.filter(pl.col("was_reviewed")),
    Signature("text -> code"),
    provider="openai",
    model="gpt-4o-mini",
    max_demos=4,
)

# 6. Score the tuned module and go again.
print(evaluate(tuned, holdout_df, my_metric).score)
```

## `export_review_sample`

```python
export_review_sample(
    df: pl.DataFrame,
    *,
    n: int = 100,
    strata: str | Sequence[str] | None = None,
    allocation: Literal["proportional", "equal"] = "proportional",
    confidence_column: str | None = None,
    oversample_low_confidence: float = 0.0,
    seed: int = 0,
    include_review_columns: bool = True,
    path: str | Path | None = None,
    format: Literal["csv", "xlsx"] | None = None,
) -> pl.DataFrame
```

Deterministic (pure Python + `random` -- no numpy) stratified sampling
without replacement:

1. **Quotas.** `n` is capped at `df.height`, then split across strata by
   `allocation`:
   - `"proportional"` (default): largest-remainder method, so each
     stratum's quota is as close as integer rounding allows to
     `n * stratum_size / total_size`.
   - `"equal"`: `n` split as evenly as possible across strata, remainder
     to the largest strata.

   Either way, **every nonempty stratum gets at least 1 row** once
   `n >= ` the number of strata (bumping a zero-quota stratum to 1,
   deducting deterministically from the currently-largest allocation).
   When `n` is *smaller* than the number of strata, only the `n` largest
   strata are represented (one row each) and a `UserWarning` is raised --
   there's no way to touch every stratum with fewer draws than strata.

   A quota is never allowed to exceed its stratum's size: any overflow is
   capped and redistributed across the other, not-yet-capped strata
   proportional to their remaining capacity, iterated until stable.

2. **Within-stratum sampling** uses the **Efraimidis-Spirakis** algorithm
   (weighted sampling without replacement in one pass, no numpy): each row
   draws a uniform `u_i` and gets key `u_i ** (1 / w_i)`; the top-`k` keys
   are kept. `oversample_low_confidence=0.0` (default) makes every weight
   `1.0`, which reduces to plain simple random sampling.

   With `confidence_column` set and `oversample_low_confidence=f > 0`,
   row weight is `1 + f * (1 - confidence)` -- a confidence-`0` row is
   `(1 + f)` times as likely to be drawn as a confidence-`1` row.
   **Null confidence is treated as confidence `0.0`** (a model that
   couldn't/didn't report a confidence is exactly the case you most want
   a human to look at).

   Sampling is seeded per-stratum (`random.Random(f"{seed}:{stratum_index}")`,
   strata ordered by their sorted key) so the same `seed` over the same
   `df` always reproduces the same sample; a different `seed` reliably
   produces a different one.

3. **Output** is the sampled rows plus `_review_id` (an `Int64` row index
   into the *original* `df` -- the join key `import_corrections` expects)
   and, when `include_review_columns=True` (default), blank
   `corrected_code`/`review_notes` (`Utf8`, all-null) columns for the
   reviewer to fill in.

4. **Export.** Pass `path=` to also write the sample out. `format=None`
   (default) infers from `path`'s suffix, falling back to CSV. CSV
   (`pl.DataFrame.write_csv`) always works with no extra install. XLSX
   (`pl.DataFrame.write_excel`) needs the optional `xlsxwriter`
   dependency:

   ```bash
   pip install "polar-llama[excel]"
   ```

   Without it, `format="xlsx"` (or a `.xlsx` path) raises a clear
   `ImportError` naming that extra -- there is **no silent fallback to
   CSV bytes in an `.xlsx` file**. `xlsxwriter` is never a required
   dependency of the base package.

## `import_corrections`

```python
import_corrections(
    df: pl.DataFrame,
    corrections: pl.DataFrame,
    *,
    id_column: str = "_review_id",
    code_column: str,
    corrected_column: str = "corrected_code",
    on_unmatched: Literal["warn", "raise", "ignore"] = "warn",
) -> CorrectionResult
```

Joins the reviewed sample back onto the original DataFrame and scores
agreement:

- **A blank/null cell in `corrected_column` means "not reviewed"**, not
  "the code is the empty string" -- for a `Utf8` `corrected_column` (the
  normal case after a CSV round trip), whitespace-only cells count as
  blank too.
- `corrected_column` is cast to `df[code_column]`'s dtype (CSV round-trips
  int codes as strings) -- a `ValueError` names the failing cast if it
  doesn't parse cleanly, rather than raising Polars' raw error.
- **Duplicate ids** in either `df` or `corrections` raise `ValueError`
  (the join would be ambiguous about which row a correction belongs to).
- **Unmatched correction ids** (present in `corrections`, absent from
  `df`) are controlled by `on_unmatched`: `"warn"` (default, a
  `UserWarning`), `"raise"` (`ValueError`), or `"ignore"`. Always counted
  in `CorrectionResult.n_unmatched_corrections` regardless.
- Row count and order of `df` are always preserved -- reviewing a sample
  never filters the full DataFrame.

`CorrectionResult`:

| Field | Meaning |
|---|---|
| `df` | `df` + `corrected_column`, `was_reviewed`, `was_changed` |
| `kappa` | `cohens_kappa(code_column, corrected_column)` over reviewed rows -- **`NaN`** (not an error) when chance agreement is 0 (e.g. every reviewed row has the same code); see `docs/RELIABILITY_METRICS.md` |
| `agreement_rate` | mean(`code_column == corrected_column`) over reviewed rows |
| `n_reviewed` | corrections that matched a `df` row and had a non-blank corrected value |
| `n_changed` | reviewed rows where the correction differs from the original code |
| `n_unmatched_corrections` | correction ids absent from `df` |
| `n_unreviewed` | `df` rows with no correction |

`kappa`/`agreement_rate` are `NaN` (not raised) when `n_reviewed == 0` --
check with `math.isnan(...)` before treating them as a real number.

## `corrections_to_trainset` / `retune_from_corrections`

`BootstrapFewShot.compile(module, trainset)` (`polar_llama/optimize.py`)
requires the trainset to have **one column per signature input field and
one column per output field, named exactly like the field** --
`Signature.render_inputs`/`render_outputs` index the row by those literal
names. `corrections_to_trainset` builds exactly that:

```python
corrections_to_trainset(
    corrections: pl.DataFrame,
    signature: Signature | str | Predict,
    *,
    corrected_column: str = "corrected_code",
    column_map: dict[str, str] | None = None,
) -> pl.DataFrame
```

- Every input field maps from the `corrections` column of the same name,
  unless overridden in `column_map` (e.g. `{"text": "response_text"}`).
- A single-output signature's output maps from `corrected_column` by
  default. A multi-output signature requires **every** output field
  covered by `column_map`, or raises `ValueError` naming what's missing.
- Rows with any null mapped value (unreviewed rows) are dropped. An empty
  result raises `ValueError`.

`retune_from_corrections` is the one call that wraps this and the
existing optimizer:

```python
retune_from_corrections(
    corrections: pl.DataFrame,
    signature: Signature | str | Predict,   # Predict reuses its own backend
    *,
    corrected_column: str = "corrected_code",
    column_map: dict[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    inference_fn: InferenceFn | None = None,   # injectable, CI-safe
    metric: Metric | None = None,
    max_demos: int = 4,
    threshold: float = 0.0,
    use_gold_outputs: bool = True,
) -> Predict
```

Internally:

```python
module = signature if isinstance(signature, Predict) else Predict(signature, provider=..., model=..., inference_fn=...)
trainset = corrections_to_trainset(corrections, module.signature, ...)
return BootstrapFewShot(
    metric=metric or _default_metric(module.signature),
    max_demos=max_demos, threshold=threshold, use_gold_outputs=use_gold_outputs,
).compile(module, trainset)
```

No demo-selection or bootstrap logic is reimplemented -- this is a thin
adapter over `polar_llama.optimize.BootstrapFewShot`.

**Defaults intentionally diverge from `BootstrapFewShot`'s own defaults**
(`threshold=1.0`, `use_gold_outputs=False`). Corrections are, by
definition, the rows the model got *wrong* -- gating on the model's own
predictions passing a strict metric (the stock bootstrap default) would
throw away exactly the training signal this function exists to use.
`threshold=0.0` admits every corrected row (a metric score is always
`>= 0.0`), and `use_gold_outputs=True` uses the human's corrected label as
the demo output rather than the model's own (wrong) prediction -- i.e.
labeled few-shot from human corrections. Pass
`threshold=1.0, use_gold_outputs=False` yourself to recover classic
bootstrap semantics (demos drawn only from rows the model already gets
right, using its own predictions).

**Default metric** (used when `metric=` is omitted): normalized exact
match, averaged across every output field --
`str(value).strip().casefold()` equality per field. This generalizes the
single-output `exact_match` convention documented in
`polar_llama/optimize.py`'s own docstring to signatures with multiple
output fields.

**Cost note.** `BootstrapFewShot.compile` evaluates the module over the
trainset before selecting demos -- so `retune_from_corrections` still
costs one inference pass over the corrected rows. This is the accepted
price of reusing the existing optimizer rather than special-casing
label-only compilation.

## Caveats

- **Kappa can be `NaN`.** `cohens_kappa` returns `NaN` (matching
  `sklearn.metrics.cohen_kappa_score`'s own convention) when expected-by-
  chance agreement is zero -- most commonly, every reviewed row happens to
  share the same code. This is not a bug in `import_corrections`; check
  `math.isnan(result.kappa)` before treating it as a number, and prefer
  reviewing a more diverse sample if it happens often.
- **A measurable lift from retuning is stochastic on small samples.**
  Few-shot demos mined from a handful of human corrections can help,
  hurt, or do nothing on any particular holdout split -- there's no
  guarantee baked into `retune_from_corrections` that the tuned module
  beats the baseline. Evaluate on a real holdout set before trusting a
  retuned module, and don't expect a consistent lift from a small
  correction batch (tens of rows); the effect gets more reliable as the
  review sample (and therefore the trainset) grows.
- **`_review_id` is positional, not stable across `df` mutations.** It's
  assigned by `with_row_index` on the `df` passed to
  `export_review_sample` at that moment -- re-sorting or filtering `df`
  before calling `import_corrections` will misalign the join. Keep the
  original `df` (or re-derive it identically) between export and import.
