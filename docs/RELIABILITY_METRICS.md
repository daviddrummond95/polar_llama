# Inter-rater Reliability Metrics

## Overview

Polar Llama provides **Cohen's kappa** and **Krippendorff's alpha** as
Polars aggregation expressions -- the standard metrics for measuring how
much two or more raters (human annotators, LLM judges, or a mix of both)
agree on a set of categorical/ordinal/numeric ratings. They are useful any
time you're validating an LLM's labels against a human gold set, comparing
two LLM judges against each other, or checking agreement across a panel of
human annotators before trusting their labels.

Both metrics are implemented in pure Rust (`src/metrics.rs`, zero new
dependencies) and are numerically pinned to reproduce two well-known
reference implementations **bit-for-bit** on their own published fixtures:

- `cohens_kappa` matches [`sklearn.metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html).
- `krippendorffs_alpha` matches the [`krippendorff`](https://pypi.org/project/krippendorff/) PyPI package.

See `tests/test_irr_metrics.py` for the differential fixtures, and
`src/metrics.rs`'s own `#[cfg(test)]` module for the Rust-level unit tests
against the same values.

### Key Features

📐 **Two metrics**: Cohen's kappa (2 raters) and Krippendorff's alpha (2+
raters, handles missing data natively)

🎯 **Bit-for-bit numeric parity**: verified against `sklearn` and the
`krippendorff` package on published fixtures

📊 **Aggregation-shaped**: composes with `.select()` and
`group_by().agg()` -- one number per group, not a value per row

🔁 **Bootstrap confidence intervals**: optional, deterministic
(seed-reproducible) percentile CIs via case resampling

🎨 **Fluent API**: available via `.llama` namespace and functional API

## Quick Start

### Cohen's kappa: comparing an LLM judge to a human label

```python
import polars as pl
from polar_llama import cohens_kappa

df = pl.DataFrame({
    "llm_label": ["positive", "negative", "positive", "neutral", "positive"],
    "human_label": ["positive", "negative", "neutral", "neutral", "positive"],
})

result = df.select(kappa=cohens_kappa("llm_label", "human_label"))
print(result)  # shape: (1, 1)  kappa: f64
```

### Krippendorff's alpha: agreement across 3+ raters, with missing data

```python
from polar_llama import krippendorffs_alpha

df = pl.DataFrame({
    "rater_1": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    "rater_2": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    "rater_3": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
})

result = df.select(alpha=krippendorffs_alpha(["rater_1", "rater_2", "rater_3"]))
print(result)  # nominal alpha over the 12 units
```

### Per-group agreement with `group_by().agg()`

Because these are aggregation expressions, they drop straight into
`group_by().agg()`:

```python
df.group_by("batch").agg(
    kappa=cohens_kappa("llm_label", "human_label"),
    alpha=krippendorffs_alpha(["rater_1", "rater_2", "rater_3"]),
)
```

### `.llama` namespace

```python
df.select(
    kappa=pl.col("llm_label").llama.cohens_kappa("human_label"),
    alpha=pl.col("rater_1").llama.krippendorffs_alpha("rater_2", "rater_3"),
)
```

### Bootstrap confidence intervals

Pass `n_bootstrap=` to get a nonparametric case-resampling percentile CI
alongside the point estimate. This changes the return dtype from `Float64`
to `Struct{value, ci_low, ci_high}`:

```python
df.select(
    krippendorffs_alpha(
        ["rater_1", "rater_2", "rater_3"], n_bootstrap=1000, ci=0.95, seed=0
    ).struct.unnest()
)
# shape: (1, 3)  value: f64 | ci_low: f64 | ci_high: f64
```

## API

### `cohens_kappa`

```python
cohens_kappa(
    a: IntoExpr,
    b: IntoExpr,
    *,
    weights: Literal["linear", "quadratic"] | None = None,
    n_bootstrap: int | None = None,
    ci: float = 0.95,
    seed: int = 0,
) -> pl.Expr
```

- **`a`, `b`**: the two columns of categorical codes being compared. Any of
  Int*/UInt*/Float/String/Categorical/Enum (Categorical/Enum are cast to
  String first).
- **`weights`**: `None` (default, unweighted) is appropriate for purely
  categorical labels. `"linear"`/`"quadratic"` penalize disagreements by
  how far apart the labels are -- use these for ordinal ratings (e.g. a
  1-5 Likert scale, a severity rating).
- **`n_bootstrap`/`ci`/`seed`**: see [Bootstrap confidence intervals](#bootstrap-confidence-intervals-1)
  below.

**Null handling: pairwise-complete.** A row is dropped if either column is
null there -- there's no meaningful "the LLM abstained" concept for
Cohen's kappa the way there is for Krippendorff's alpha; sklearn has no
concept of a missing label either.

**Label weighting is by sorted-label *index*, not label value.** This is
sklearn's convention and it can surprise you: if your codes are
`{1, 5, 10}`, `"linear"` weights the gap between `1` and `5` the same as
the gap between `5` and `10` (both are 1 sorted-index step apart), *not*
proportional to `4` vs `5`. If your codes have non-contiguous numeric
gaps and you want the gap itself to matter, remap them to contiguous
ranks (`0, 1, 2, ...`) before calling `cohens_kappa`.

### `krippendorffs_alpha`

```python
krippendorffs_alpha(
    cols: Sequence[IntoExpr],
    *,
    level: Literal["nominal", "ordinal", "interval", "ratio"] = "nominal",
    n_bootstrap: int | None = None,
    ci: float = 0.95,
    seed: int = 0,
) -> pl.Expr
```

- **`cols`**: M >= 2 rater columns, one row per unit (item being rated).
- **`level`**: level of measurement, governing how "distance" between two
  categories is defined:
  - `"nominal"` (default): categories are either equal or not (distance
    is 0 or 1). Accepts numeric or String/Categorical/Enum columns.
  - `"ordinal"`: categories have a meaningful order but not meaningful
    spacing (e.g. "low/medium/high"). **Requires numeric columns** --
    encode your ordered categories as ranks (`0, 1, 2, ...`) first. The
    rank used is the numeric value's position in the sorted domain of
    *observed* values, not the raw numeric gap.
  - `"interval"`: categories are numeric with meaningful, evenly-spaced
    distance (e.g. a temperature, a 1-10 rating where each step is
    equally significant). Requires numeric columns.
  - `"ratio"`: like interval, but distances are normalized by magnitude
    (`(c-k)/(c+k)`) -- appropriate for ratio-scale numeric data where 0
    is a true zero (e.g. counts, durations). Requires numeric columns.
  - `"ordinal"`/`"interval"`/`"ratio"` raise `pl.exceptions.ComputeError`
    if any column isn't numeric.
- **`n_bootstrap`/`ci`/`seed`**: see [Bootstrap confidence intervals](#bootstrap-confidence-intervals-1)
  below.

**Null handling: nulls are missing ratings** -- this is the entire reason
Krippendorff's alpha exists (unlike Cohen's kappa, it was designed to
tolerate incomplete rating matrices). A unit (row) with fewer than 2
non-null ratings contributes nothing pairable and is excluded from the
computation.

### Return shape

Both functions return:
- **`Float64`** (the point estimate) when `n_bootstrap` is `None`
  (default).
- **`Struct{value: Float64, ci_low: Float64, ci_high: Float64}`** when
  `n_bootstrap` is given. Unnest with `.struct.unnest()` or access fields
  with `.struct.field("value")`.

### Bootstrap confidence intervals

Passing `n_bootstrap=B` runs a **nonparametric case (unit/row) bootstrap**:
the N units are resampled with replacement B times, the metric is
recomputed on each resample, and a percentile confidence interval is taken
at `((1-ci)/2, 1-(1-ci)/2)` using linear-interpolation quantiles (numpy's
default "type 7" convention).

> **Note:** this is *not* Krippendorff's own 2011 pair-bootstrap algorithm
> for alpha specifically. Case (unit) resampling is the standard,
> simply-defensible bootstrap and is what most practitioners expect from
> an `n_bootstrap=` parameter -- but if you need the original
> pair-bootstrap procedure for a publication that specifically requires
> it, you'll need to implement that separately.

The bootstrap PRNG is a deterministic, seeded `SplitMix64` generator (the
same one used elsewhere in this package, e.g. `cluster_embeddings`'s
k-means seeding) -- the same `seed` over the same input always produces
the same CI.

**Degenerate resamples** (a resample that collapses to a single category,
etc., where the metric is undefined) are dropped from the percentile
computation. If more than half of the `n_bootstrap` resamples are
degenerate, `ci_low`/`ci_high` are null -- but `value` is always the point
estimate computed on the *original*, non-resampled data, so it's never
affected by a noisy bootstrap.

## Edge cases and documented divergences from the reference implementations

Both `sklearn.metrics.cohen_kappa_score` and the `krippendorff` package
*raise exceptions* on certain degenerate inputs. A Polars expression can't
raise per-group inside `group_by().agg()` without aborting the whole
query, so these cases are handled by returning a specific value instead --
documented here so the divergence is never a silent surprise:

| Situation | Reference behavior | `polar_llama` behavior |
|---|---|---|
| `cohens_kappa`: expected-by-chance agreement is 0 (e.g. both columns have one single, identical observed category) | sklearn returns `nan` (a `0/0`) | Returns `NaN` too (not null) -- check with `.is_nan()`, same as `numpy.isnan()` |
| `cohens_kappa`: zero pairwise-complete rows | sklearn has nothing to score | Returns **null** |
| `krippendorffs_alpha`: no unit has >= 2 ratings (nothing pairable) | `krippendorff` raises `ValueError` | Returns **null** |
| `krippendorffs_alpha`: the observed domain has a single category (every rating, across every included unit, is identical) | `krippendorff` raises `ValueError("... value in the domain")` | Returns **`1.0`** ("trivially perfect agreement" -- returning null is also defensible, but `1.0` was chosen as the convention; see `src/metrics.rs::krippendorff_alpha`) |

Other conventions worth knowing about, both verified bit-for-bit against
the reference implementations on the fixtures in `tests/test_irr_metrics.py`:

- **Ordinal alpha's difference function uses coincidence-matrix
  marginals**, not raw category frequencies -- this is the classic
  from-scratch implementation mistake, and it's why the ordinal fixture in
  the test suite exists.
- Systematic *disagreement* (raters are anti-correlated) can produce a
  **negative** kappa or alpha -- this is expected and correct, not a bug
  (e.g. Krippendorff's own `[[1,2,1,2],[2,1,2,1]]` nominal fixture gives
  `alpha = -0.75`).

## Multi-label codes (MASI) -- documented strategy

Some coding tasks are multi-label: each unit can receive *a set* of codes
rather than one (e.g. a document tagged with `{"billing", "urgent"}`). The
standard metric for that shape is MASI (Measuring Agreement on Set-valued
Items, via `nltk.metrics.agreement.AnnotationTask` + `masi_distance`).

**This package does not implement `level="masi"` in this release.** MASI
needs a different input contract (set-valued cells, i.e. `List[String]`
columns) than the scalar-per-cell contract `cohens_kappa`/
`krippendorffs_alpha` use, and its reference implementation (NLTK) uses a
distance function (not a squared difference) with its own missing-data
quirks -- matching it exactly is a research task in its own right, not a
mechanical extension of the coincidence-matrix machinery above. Shipping a
MASI implementation with an unverified convention would be worse than not
shipping one.

Two supported strategies instead, both expressible today with the
existing `nominal` alpha:

### Strategy 1: per-label binary alpha (recommended for most cases)

Explode the label set into one 0/1 column per label per rater, then report
a separate nominal alpha per label. This is standard practice in content
analysis, and it gives you a per-label reliability breakdown (some labels
are usually agreed on more than others) rather than one opaque number.

```python
import polars as pl
from polar_llama import krippendorffs_alpha

# rater_1_labels / rater_2_labels: List[String] columns, e.g.
# rater_1_labels = [["billing", "urgent"], ["shipping"], None, ...]
all_labels = ["billing", "urgent", "shipping", "refund"]

df = df.with_columns(
    [
        pl.col("rater_1_labels").list.contains(label).cast(pl.Int8).alias(f"r1_{label}")
        for label in all_labels
    ]
    + [
        pl.col("rater_2_labels").list.contains(label).cast(pl.Int8).alias(f"r2_{label}")
        for label in all_labels
    ]
)

per_label_alpha = {
    label: df.select(
        krippendorffs_alpha([f"r1_{label}", f"r2_{label}"])
    ).item()
    for label in all_labels
}
```

Note: `list.contains` returns `False` (not null) for an empty list but
null for a null list -- a rater who didn't rate a unit at all still comes
through as a missing rating (null) in each per-label column, which is the
correct semantics for alpha's null-as-missing convention.

### Strategy 2: exact-set nominal alpha (strict)

Treat the whole label set as one atomic nominal category -- two raters
only "agree" if they picked the *exact same set*. This is stricter than
per-label alpha (it doesn't give partial credit for overlapping-but-not-
identical sets) but is sometimes what you want for a hard equality bar.

```python
df = df.with_columns(
    r1_set=pl.col("rater_1_labels").list.sort().list.join("|"),
    r2_set=pl.col("rater_2_labels").list.sort().list.join("|"),
)
exact_set_alpha = df.select(krippendorffs_alpha(["r1_set", "r2_set"])).item()
```

### Future: `level="masi"`

The coincidence-matrix machinery in `src/metrics.rs::krippendorff_alpha`
already accepts an arbitrary `delta2(c, k)` difference function over
hashable categories -- MASI is "just" another difference function (a set
Jaccard-style distance rather than equality/squared-difference), plus a
`List[String]`-shaped input path through `src/expressions.rs`. This is
being tracked as a follow-up issue rather than bundled into #79, so it can
be verified against NLTK's `masi_distance` on its own fixtures rather than
guessed at.
