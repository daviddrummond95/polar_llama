"""
Inter-rater reliability metrics for Polar Llama (issue #79): Cohen's kappa
and Krippendorff's alpha.

Design summary -- see `docs/RELIABILITY_METRICS.md` for the full walkthrough:

1. **Shape**: unlike the row-wise expressions elsewhere in this package,
   `cohens_kappa`/`krippendorffs_alpha` are *aggregations* -- one number
   over N rows, optionally per group. They are registered with
   `is_elementwise=False, returns_scalar=True`
   (`polar_llama.utils.register_plugin`), the standard Polars
   aggregation-plugin pattern, so they compose with `.select()`,
   `group_by().agg()`, and lazy frames for free:
   `df.group_by("segment").agg(pl.col("llm_a").llama.cohens_kappa("llm_b"))`.

2. **Numeric parity**: the Rust algorithms (`src/metrics.rs`) are pinned to
   reproduce `sklearn.metrics.cohen_kappa_score` and the `krippendorff`
   PyPI package bit-for-bit on their own published fixtures -- see
   `tests/test_irr_metrics.py` and `src/metrics.rs`'s own unit tests.
   Notable, *documented* divergences from those references (an expression
   must not raise/panic per-group the way the reference libraries do):
   - `cohens_kappa`: `0/0` expected-agreement (e.g. a single observed
     category) returns `NaN` (not null) -- matches sklearn's own
     `nan`-not-error convention.
   - `krippendorffs_alpha`: no unit with >= 2 ratings returns null (the
     `krippendorff` package raises `ValueError`); a single-category domain
     returns `1.0` (trivially perfect agreement; the package also raises
     there).

3. **Return dtype**: a plain `Float64` scalar by default; passing
   `n_bootstrap=` switches to `Struct{value, ci_low, ci_high}` (case
   resampling, percentile CI, `SplitMix64`-seeded and therefore
   deterministic for a fixed `seed`). `pyo3_polars`'
   `output_type_func` can't see kwargs, so this dispatches between two
   distinct Rust symbols per metric based on whether `n_bootstrap is None`.

4. **Multi-label / MASI**: deliberately not implemented here -- see the
   "Multi-label codes" section of `docs/RELIABILITY_METRICS.md` for the two
   documented strategies (per-label binary alpha, exact-set nominal alpha),
   which satisfy the "or documented strategy" acceptance arm of issue #79.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from typing import Literal

    from polars.type_aliases import IntoExpr

_KAPPA_WEIGHTS = ("linear", "quadratic")
_ALPHA_LEVELS = ("nominal", "ordinal", "interval", "ratio")


def _validate_ci(ci: float) -> None:
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in the open interval (0, 1); got {ci!r}")


def _validate_weights(weights: Optional[str]) -> None:
    if weights is not None and weights not in _KAPPA_WEIGHTS:
        raise ValueError(
            f"weights must be None, 'linear', or 'quadratic'; got {weights!r}"
        )


def _validate_level(level: str) -> None:
    if level not in _ALPHA_LEVELS:
        raise ValueError(
            f"level must be one of {_ALPHA_LEVELS!r}; got {level!r}"
        )


def cohens_kappa(
    a: "IntoExpr",
    b: "IntoExpr",
    *,
    weights: "Optional[Literal['linear', 'quadratic']]" = None,
    n_bootstrap: Optional[int] = None,
    ci: float = 0.95,
    seed: int = 0,
) -> pl.Expr:
    """
    Cohen's kappa between two columns of categorical ratings.

    Reproduces `sklearn.metrics.cohen_kappa_score` bit-for-bit: labels are
    the sorted union of both columns' observed values (numeric sort for
    numeric dtypes, lexicographic for strings), `weights` are applied over
    sorted-label *indices* (sklearn's convention -- this can surprise you
    with non-contiguous numeric codes; e.g. codes `{1, 5, 10}` weight as
    indices `{0, 1, 2}`, not by the raw label gap), and rows are
    pairwise-complete (a row is dropped if either column is null there).

    Parameters
    ----------
    a, b : IntoExpr
        The two columns of categorical codes being compared (Int/UInt/
        Float/String/Categorical/Enum; Categorical/Enum are cast to
        String).
    weights : {None, "linear", "quadratic"}, optional
        `None` (default) is unweighted kappa. `"linear"`/`"quadratic"`
        penalize disagreements by their distance in the sorted-label list
        -- appropriate for ordinal ratings (e.g. a 1-5 Likert scale).
    n_bootstrap : int, optional
        When given, also compute a nonparametric case (row-pair) bootstrap
        confidence interval: resample the pairwise-complete rows with
        replacement `n_bootstrap` times, recompute kappa on each resample,
        and take the percentile interval. Changes the return dtype -- see
        Returns below. `None` (default) skips the bootstrap.
    ci : float
        Confidence level for the bootstrap interval, e.g. `0.95` for a
        95% CI. Must be in `(0, 1)`. Only used when `n_bootstrap` is given.
    seed : int
        Seed for the (deterministic) bootstrap PRNG. Only used when
        `n_bootstrap` is given.

    Returns
    -------
    polars.Expr
        `Float64` (the point estimate) when `n_bootstrap is None`.
        `Struct{value: Float64, ci_low: Float64, ci_high: Float64}` when
        `n_bootstrap` is given -- unnest with `.struct.unnest()`.
        `NaN` (not null) when expected-by-chance agreement is zero (e.g.
        both columns have a single, identical observed category). Null
        when there are zero pairwise-complete rows.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import cohens_kappa
    >>> df = pl.DataFrame({"llm": [1, 2, 2, 1], "human": [1, 2, 1, 1]})
    >>> df.select(kappa=cohens_kappa("llm", "human"))
    >>> df.select(
    ...     cohens_kappa("llm", "human", n_bootstrap=1000).struct.unnest()
    ... )
    """
    _validate_weights(weights)
    _validate_ci(ci)

    a_expr = parse_into_expr(a)
    b_expr = parse_into_expr(b)

    kwargs: Dict[str, Any] = {
        "weights": weights,
        "n_bootstrap": int(n_bootstrap) if n_bootstrap is not None else None,
        "ci": float(ci),
        "seed": int(seed),
    }

    from polar_llama.expressions import get_lib_path

    symbol = "cohens_kappa" if n_bootstrap is None else "cohens_kappa_ci"
    return register_plugin(
        args=[a_expr, b_expr],
        symbol=symbol,
        is_elementwise=False,
        returns_scalar=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )


def krippendorffs_alpha(
    cols: Sequence["IntoExpr"],
    *,
    level: "Literal['nominal', 'ordinal', 'interval', 'ratio']" = "nominal",
    n_bootstrap: Optional[int] = None,
    ci: float = 0.95,
    seed: int = 0,
) -> pl.Expr:
    """
    Krippendorff's alpha across M >= 2 columns of ratings (one column per
    rater, one row per unit).

    Reproduces the `krippendorff` PyPI package's coincidence-matrix
    algorithm bit-for-bit. Nulls are treated as missing ratings (the
    reason alpha exists): a unit (row) with fewer than 2 non-null ratings
    is excluded from the computation.

    Parameters
    ----------
    cols : sequence of IntoExpr
        The M >= 2 rater columns, one row per unit.
    level : {"nominal", "ordinal", "interval", "ratio"}
        Level of measurement, governing the difference function between
        categories. `"nominal"` (default) accepts numeric or
        String/Categorical/Enum columns. `"ordinal"`/`"interval"`/
        `"ratio"` require numeric columns (raises `pl.exceptions.ComputeError`
        otherwise); for `"ordinal"` the rank is the numeric value's
        position in the sorted domain of observed values.
    n_bootstrap : int, optional
        When given, also compute a nonparametric case (unit) bootstrap
        confidence interval: resample the units with replacement
        `n_bootstrap` times, recompute alpha on each resample, and take
        the percentile interval. Changes the return dtype -- see Returns
        below. `None` (default) skips the bootstrap.
    ci : float
        Confidence level for the bootstrap interval. Must be in `(0, 1)`.
        Only used when `n_bootstrap` is given.
    seed : int
        Seed for the (deterministic) bootstrap PRNG. Only used when
        `n_bootstrap` is given.

    Returns
    -------
    polars.Expr
        `Float64` (the point estimate) when `n_bootstrap is None`.
        `Struct{value: Float64, ci_low: Float64, ci_high: Float64}` when
        `n_bootstrap` is given -- unnest with `.struct.unnest()`.
        `1.0` for a single-category domain (trivially perfect agreement,
        by documented convention; the reference package raises
        `ValueError` there instead). Null when no unit has >= 2 ratings
        (the reference package also raises there).

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import krippendorffs_alpha
    >>> df = pl.DataFrame({
    ...     "r1": [1, 2, 3, None],
    ...     "r2": [1, 2, 3, 3],
    ...     "r3": [1, 2, None, 3],
    ... })
    >>> df.select(alpha=krippendorffs_alpha(["r1", "r2", "r3"]))
    >>> df.group_by("topic").agg(
    ...     alpha=krippendorffs_alpha(["r1", "r2", "r3"], level="interval")
    ... )
    >>> df.select(
    ...     krippendorffs_alpha(
    ...         ["r1", "r2", "r3"], n_bootstrap=1000
    ...     ).struct.unnest()
    ... )
    """
    _validate_level(level)
    _validate_ci(ci)

    exprs: List[pl.Expr] = [parse_into_expr(c) for c in cols]
    if len(exprs) < 2:
        raise ValueError(
            f"krippendorffs_alpha needs at least 2 rater columns; got {len(exprs)}"
        )

    kwargs: Dict[str, Any] = {
        "level": level,
        "n_bootstrap": int(n_bootstrap) if n_bootstrap is not None else None,
        "ci": float(ci),
        "seed": int(seed),
    }

    from polar_llama.expressions import get_lib_path

    symbol = "krippendorffs_alpha" if n_bootstrap is None else "krippendorffs_alpha_ci"
    return register_plugin(
        args=exprs,
        symbol=symbol,
        is_elementwise=False,
        returns_scalar=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )
