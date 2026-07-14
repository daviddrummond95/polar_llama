"""
Human-in-the-loop review loop for Polar Llama (issue #81).

Design summary -- see `docs/HITL_WORKFLOW.md` for the full walkthrough:

1. **The loop**: LLM-coded DataFrame -> `export_review_sample` (a
   stratified, optionally confidence-weighted sample for a human to look
   at) -> a human fills in `corrected_code`/`review_notes` in a
   spreadsheet -> `import_corrections` (joins the corrections back onto
   the full DataFrame and computes agreement stats, reusing
   `polar_llama.reliability.cohens_kappa`) -> `retune_from_corrections`
   (turns the corrections into a `BootstrapFewShot` trainset and compiles
   an improved `Predict`, reusing `polar_llama.optimize` verbatim) ->
   evaluate the retuned module and go again.

2. **DataFrame-level orchestration, like `induce_codebook`/
   `quality_report`.** All four public names here (`export_review_sample`,
   `import_corrections`, `corrections_to_trainset`,
   `retune_from_corrections`) take and return whole DataFrames/dataclasses,
   not row-wise values -- so, deliberately, **none of them are added to the
   `.llama` expression namespace** (`polar_llama/__init__.py`'s
   `register_expr_namespace("llama")` block is expr-only by design; see
   `polar_llama/codebook.py`/`polar_llama/quality.py` for the same
   convention).

3. **No reimplementation.** The retune step is a thin adapter in front of
   the existing prompt-optimization engine (`polar_llama/optimize.py`):
   `corrections_to_trainset` builds a `pl.DataFrame` whose columns are
   exactly a `Signature`'s input and output field names (the contract
   `BootstrapFewShot.compile` -> `evaluate` -> `Signature.render_inputs`
   actually requires), and `retune_from_corrections` hands that trainset
   straight to `BootstrapFewShot(...).compile(module, trainset)` -- no
   duplicated bootstrap/demo-selection logic. Likewise, `import_corrections`
   computes its agreement statistics with
   `polar_llama.reliability.cohens_kappa`, not a bespoke kappa
   implementation; that module's `NaN`-on-zero-chance-agreement and
   null-on-zero-pairwise-rows conventions apply unchanged here.

   Unlike `codebook.py`/`quality.py`, this module imports
   `polar_llama.optimize` (`Signature`, `Predict`, `BootstrapFewShot`) and
   `polar_llama.reliability` (`cohens_kappa`) at module scope rather than
   lazily inside function bodies: neither of those modules imports the
   `polar_llama` package root at module scope (they reach into it lazily
   themselves, where needed), so importing them here from
   `polar_llama/__init__.py` creates no import cycle.

4. **Zero new required dependencies.** The stratified sampler
   (`export_review_sample`) is pure stdlib `random` + `polars` -- no numpy,
   matching this package's existing convention of hand-rolled,
   dependency-free sampling/allocation algorithms (e.g.
   `cluster_embeddings`'s k-means). XLSX export is the one optional
   extra: `pip install polar-llama[excel]` pulls in `xlsxwriter` (used by
   `pl.DataFrame.write_excel`); CSV export always works with no extra
   install, and an `.xlsx` `path`/`format="xlsx"` without the extra raises
   a clear `ImportError` rather than silently writing CSV bytes to a
   `.xlsx` file.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import polars as pl

from polar_llama.optimize import (
    BootstrapFewShot,
    InferenceFn,
    Metric,
    Predict,
    Signature,
)
from polar_llama.reliability import cohens_kappa

if TYPE_CHECKING:
    from typing import Literal

__all__ = [
    "CorrectionResult",
    "corrections_to_trainset",
    "export_review_sample",
    "import_corrections",
    "retune_from_corrections",
]

_ALLOCATIONS = ("proportional", "equal")
_FORMATS = ("csv", "xlsx")
_ON_UNMATCHED = ("warn", "raise", "ignore")


# ============================================================================
# export_review_sample -- stratified, confidence-weighted sampling
# ============================================================================


def _normalize_strata(strata: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if strata is None:
        return []
    if isinstance(strata, str):
        return [strata]
    return list(strata)


def _sort_key(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # `partition_by(..., as_dict=True)` keys are always tuples; sort with
    # Nones ordered last (Python can't compare None to other types).
    return tuple((v is None, v) for v in key)


def _largest_remainder(weights: Sequence[float], n: int) -> List[int]:
    """Largest-remainder-method allocation of `n` units across `weights`
    (proportional, integer, sums to exactly `n` when `n <= sum(weights)`
    is not required -- see `_cap_and_redistribute` for that guarantee).
    Ties broken deterministically by ascending index.
    """
    g = len(weights)
    total = sum(weights)
    if g == 0 or n <= 0 or total <= 0:
        return [0] * g

    exact = [n * w / total for w in weights]
    base = [int(math.floor(e)) for e in exact]
    remainder = n - sum(base)
    order = sorted(range(g), key=lambda i: (-(exact[i] - base[i]), i))
    alloc = list(base)
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


def _equal_split(counts: Sequence[int], n: int) -> List[int]:
    """Split `n` as evenly as possible across `len(counts)` strata; the
    remainder goes to the largest strata (by `counts`, descending),
    deterministically tie-broken by ascending index.
    """
    g = len(counts)
    if g == 0 or n <= 0:
        return [0] * g
    base_each = n // g
    alloc = [base_each] * g
    remainder = n - base_each * g
    order = sorted(range(g), key=lambda i: (-counts[i], i))
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


def _apply_min1_guarantee(alloc: List[int], counts: Sequence[int], n: int) -> List[int]:
    """Every nonempty stratum gets >= 1 when `n >= G` (donating from the
    largest current allocations, deterministically). When `n < G`, only
    the `n` largest strata are represented (one row each) -- warns.
    """
    g = len(counts)
    if g == 0:
        return list(alloc)

    if n >= g:
        alloc = list(alloc)
        for i in range(g):
            if alloc[i] > 0:
                continue
            candidates = [j for j in range(g) if j != i and alloc[j] > 1]
            if not candidates:
                continue
            donor = max(candidates, key=lambda j: (alloc[j], -j))
            alloc[donor] -= 1
            alloc[i] = 1
        return alloc

    warnings.warn(
        f"export_review_sample: n={n} is smaller than the number of strata "
        f"(G={g}); only the {n} largest stratum/strata will be represented "
        "(one row each). Increase n to cover every stratum.",
        UserWarning,
        stacklevel=4,
    )
    order = sorted(range(g), key=lambda i: (-counts[i], i))
    keep = set(order[:n])
    return [1 if i in keep else 0 for i in range(g)]


def _cap_and_redistribute(alloc: List[int], counts: Sequence[int]) -> List[int]:
    """Iteratively cap each stratum's allocation at its size and
    redistribute the deficit across not-yet-capped strata, proportional to
    their remaining capacity, until stable.
    """
    alloc = list(alloc)
    g = len(counts)
    capped = [False] * g

    while True:
        overflow = 0
        for i in range(g):
            if not capped[i] and alloc[i] > counts[i]:
                overflow += alloc[i] - counts[i]
                alloc[i] = counts[i]
                capped[i] = True
        if overflow == 0:
            break

        open_idxs = [i for i in range(g) if not capped[i]]
        remaining_capacity = [counts[i] - alloc[i] for i in open_idxs]
        if not open_idxs or sum(remaining_capacity) <= 0:
            # Nowhere left to put the overflow (shouldn't happen when
            # n <= sum(counts), the caller's invariant) -- give up silently
            # rather than looping forever.
            break

        add = _largest_remainder(remaining_capacity, min(overflow, sum(remaining_capacity)))
        for idx, a in zip(open_idxs, add):
            alloc[idx] += a

    return alloc


def _allocate_quotas(counts: Sequence[int], n: int, allocation: str) -> List[int]:
    if allocation == "proportional":
        alloc = _largest_remainder(counts, n)
    else:
        alloc = _equal_split(counts, n)
    alloc = _apply_min1_guarantee(alloc, counts, n)
    alloc = _cap_and_redistribute(alloc, counts)
    return alloc


def _row_weights(
    stratum: pl.DataFrame,
    confidence_column: Optional[str],
    oversample_low_confidence: float,
) -> List[float]:
    height = stratum.height
    if confidence_column is None or oversample_low_confidence == 0.0:
        return [1.0] * height
    # Null confidence is treated as confidence 0.0 (unknown = most in need
    # of review) -- documented in export_review_sample's docstring.
    conf = (
        stratum.get_column(confidence_column)
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .to_list()
    )
    return [1.0 + oversample_low_confidence * (1.0 - c) for c in conf]


def _weighted_sample_without_replacement(
    weights: Sequence[float], k: int, rng: random.Random
) -> List[int]:
    """Efraimidis-Spirakis weighted sampling without replacement: draw a
    key `u_i ** (1 / w_i)` per row (u_i ~ Uniform(0, 1)) and keep the `k`
    largest keys. `oversample_low_confidence=0.0` -> all weights equal ->
    reduces to plain SRS. Returns indices in the stratum's original order.
    """
    n = len(weights)
    k = min(max(k, 0), n)
    if k == 0:
        return []

    keyed: List[Tuple[float, int]] = []
    for i, w in enumerate(weights):
        u = rng.random()
        if u <= 0.0:
            u = 1e-300
        w_eff = w if w > 0.0 else 1e-12
        keyed.append((u ** (1.0 / w_eff), i))

    keyed.sort(key=lambda t: (-t[0], t[1]))
    return sorted(i for _, i in keyed[:k])


def _write_export(df: pl.DataFrame, path: Union[str, Path], fmt: Optional[str]) -> None:
    path = Path(path)
    resolved = fmt
    if resolved is None:
        suffix = path.suffix.lower().lstrip(".")
        resolved = suffix if suffix in _FORMATS else "csv"

    if resolved == "csv":
        df.write_csv(path)
        return

    try:
        import xlsxwriter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xlsx export requires xlsxwriter; install with: "
            "pip install polar-llama[excel] — or use format='csv'"
        ) from exc
    df.write_excel(path)


def export_review_sample(
    df: pl.DataFrame,
    *,
    n: int = 100,
    strata: Optional[Union[str, Sequence[str]]] = None,
    allocation: "Literal['proportional', 'equal']" = "proportional",
    confidence_column: Optional[str] = None,
    oversample_low_confidence: float = 0.0,
    seed: int = 0,
    include_review_columns: bool = True,
    path: Optional[Union[str, Path]] = None,
    format: Optional["Literal['csv', 'xlsx']"] = None,
) -> pl.DataFrame:
    """
    Draw a deterministic, stratified sample of `df` for human review.

    Parameters
    ----------
    df : pl.DataFrame
        The LLM-coded data to sample from.
    n : int
        Target sample size. Capped at `df.height` (a smaller sample if
        `df` has fewer rows). Must be positive.
    strata : str or sequence of str, optional
        Column name(s) to stratify by (e.g. a code column, a source
        column). `None` (default) draws one unstratified sample.
    allocation : {"proportional", "equal"}
        How `n` is split across strata. `"proportional"` (default) uses
        the largest-remainder method so each stratum's quota is
        proportional to its size. `"equal"` splits `n` evenly across
        strata (remainder to the largest strata). Either way, every
        nonempty stratum gets at least 1 row when `n >= ` the number of
        strata; when `n` is smaller than the number of strata, only the
        `n` largest strata are represented (one row each) and a
        `UserWarning` is raised. A stratum's quota is never allowed to
        exceed its size -- the deficit is redistributed across the other,
        not-yet-capped strata proportional to their remaining capacity.
    confidence_column : str, optional
        A `Float64`-like column (0..1) of model confidence per row. When
        given together with `oversample_low_confidence > 0`, rows are
        drawn with probability weighted toward low confidence within each
        stratum (still without replacement, still capped at n per
        stratum). **Null confidence is treated as confidence 0.0** (the
        model didn't/couldn't report a confidence -- treated as the row
        most in need of review).
    oversample_low_confidence : float
        Oversampling factor `f >= 0`. Per-row weight is
        `1 + f * (1 - confidence)`: a confidence-0 row is `(1 + f)` times
        as likely to be drawn (per unit of the Efraimidis-Spirakis key) as
        a confidence-1 row. `0.0` (default) is uniform sampling.
    seed : int
        Seed for the deterministic per-stratum PRNG. The same `seed` over
        the same `df` always produces the same sample.
    include_review_columns : bool
        When `True` (default), attach blank `corrected_code`/
        `review_notes` (`Utf8`, all-null) columns for the reviewer to
        fill in.
    path : str or Path, optional
        When given, also write the sample to this path.
    format : {"csv", "xlsx"}, optional
        Export format. `None` (default) infers from `path`'s suffix,
        falling back to `"csv"`. `"csv"` always works (`pl.write_csv`).
        `"xlsx"` requires the optional `xlsxwriter` dependency
        (`pip install polar-llama[excel]`) and raises a clear `ImportError`
        naming that extra if it isn't installed -- there is no silent
        fallback to CSV.

    Returns
    -------
    pl.DataFrame
        The sampled rows (original columns plus `_review_id`, an `Int64`
        row index into `df` -- the join key for `import_corrections` --
        and, when `include_review_columns=True`, blank `corrected_code`/
        `review_notes` columns).
    """
    if n <= 0:
        raise ValueError(f"export_review_sample: n must be positive, got {n!r}")
    if oversample_low_confidence < 0:
        raise ValueError(
            "export_review_sample: oversample_low_confidence must be >= 0, "
            f"got {oversample_low_confidence!r}"
        )
    if allocation not in _ALLOCATIONS:
        raise ValueError(
            f"export_review_sample: allocation must be one of {_ALLOCATIONS!r}, "
            f"got {allocation!r}"
        )
    if format is not None and format not in _FORMATS:
        raise ValueError(
            f"export_review_sample: format must be one of {_FORMATS!r} or None, "
            f"got {format!r}"
        )

    strata_cols = _normalize_strata(strata)
    for col in strata_cols:
        if col not in df.columns:
            raise ValueError(f"export_review_sample: unknown strata column {col!r}")
    if confidence_column is not None and confidence_column not in df.columns:
        raise ValueError(
            f"export_review_sample: unknown confidence_column {confidence_column!r}"
        )

    n_eff = min(n, df.height)

    indexed = df.with_row_index("_review_id").with_columns(
        pl.col("_review_id").cast(pl.Int64)
    )

    if strata_cols:
        groups: Dict[Tuple[Any, ...], pl.DataFrame] = indexed.partition_by(
            strata_cols, as_dict=True
        )
    else:
        groups = {(): indexed}

    sorted_keys = sorted(groups.keys(), key=_sort_key)
    counts = [groups[k].height for k in sorted_keys]

    alloc = _allocate_quotas(counts, n_eff, allocation)

    sampled_frames: List[pl.DataFrame] = []
    for g_idx, key in enumerate(sorted_keys):
        k = alloc[g_idx]
        if k <= 0:
            continue
        stratum = groups[key]
        weights = _row_weights(stratum, confidence_column, oversample_low_confidence)
        rng = random.Random(f"{seed}:{g_idx}")
        idx = _weighted_sample_without_replacement(weights, k, rng)
        sampled_frames.append(stratum[idx])

    sample = pl.concat(sampled_frames, how="vertical") if sampled_frames else indexed.clear()

    if include_review_columns:
        sample = sample.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("corrected_code"),
            pl.lit(None, dtype=pl.Utf8).alias("review_notes"),
        )

    if path is not None:
        _write_export(sample, path, format)

    return sample


# ============================================================================
# import_corrections -- join corrections back onto df, score agreement
# ============================================================================


@dataclass(frozen=True)
class CorrectionResult:
    """Return value of `import_corrections`.

    `df`: the input DataFrame plus `corrected_column` (cast to
    `code_column`'s dtype), `was_reviewed`, and `was_changed` -- same
    height and row order as the input. `kappa`/`agreement_rate` are
    computed over the reviewed rows only.
    """

    df: pl.DataFrame
    kappa: float
    agreement_rate: float
    n_reviewed: int
    n_changed: int
    n_unmatched_corrections: int
    n_unreviewed: int


def import_corrections(
    df: pl.DataFrame,
    corrections: pl.DataFrame,
    *,
    id_column: str = "_review_id",
    code_column: str,
    corrected_column: str = "corrected_code",
    on_unmatched: "Literal['warn', 'raise', 'ignore']" = "warn",
) -> CorrectionResult:
    """
    Join human corrections back onto `df` and score human/LLM agreement.

    Parameters
    ----------
    df : pl.DataFrame
        The (LLM-coded) DataFrame corrections apply to -- typically the
        `df` originally passed to `export_review_sample`.
    corrections : pl.DataFrame
        The reviewed sample, e.g. `export_review_sample`'s output after a
        human filled in `corrected_column`. A blank/null cell there means
        "not reviewed", not "the code is empty".
    id_column : str
        Join key present in both `df` and `corrections` (default
        `"_review_id"`, matching `export_review_sample`'s output).
        Duplicate ids in either frame raise `ValueError` (an ambiguous
        join).
    code_column : str
        The original LLM-assigned code column in `df` being corrected.
    corrected_column : str
        The reviewer's corrected value column in `corrections`.
    on_unmatched : {"warn", "raise", "ignore"}
        What to do when `corrections` contains ids absent from `df`:
        warn (default), raise `ValueError`, or silently ignore. Always
        counted in `CorrectionResult.n_unmatched_corrections` regardless.

    Returns
    -------
    CorrectionResult
    """
    if on_unmatched not in _ON_UNMATCHED:
        raise ValueError(
            f"import_corrections: on_unmatched must be one of {_ON_UNMATCHED!r}, "
            f"got {on_unmatched!r}"
        )
    for name, col in (("df", id_column), ("df", code_column)):
        if col not in df.columns:
            raise ValueError(f"import_corrections: {name} is missing column {col!r}")
    for name, col in (("corrections", id_column), ("corrections", corrected_column)):
        if col not in corrections.columns:
            raise ValueError(f"import_corrections: {name} is missing column {col!r}")

    if df.select(pl.col(id_column).is_duplicated().any()).item():
        raise ValueError(
            f"import_corrections: df has duplicate values in id_column {id_column!r}"
        )
    if corrections.select(pl.col(id_column).is_duplicated().any()).item():
        raise ValueError(
            f"import_corrections: corrections has duplicate values in id_column {id_column!r}"
        )

    # A blank cell means "not reviewed" -- for a Utf8 corrected_column
    # (the common case, e.g. a CSV round-trip) an empty/whitespace-only
    # string counts as blank too, not "the code is the empty string".
    not_null = pl.col(corrected_column).is_not_null()
    if corrections.schema[corrected_column] == pl.Utf8:
        not_null = not_null & (pl.col(corrected_column).str.strip_chars() != "")
    reviewed = corrections.filter(not_null)

    unmatched = reviewed.join(df.select(id_column), on=id_column, how="anti")
    n_unmatched = unmatched.height
    if n_unmatched:
        message = (
            f"import_corrections: {n_unmatched} correction id(s) not found in df "
            f"(id_column={id_column!r})"
        )
        if on_unmatched == "raise":
            raise ValueError(message)
        if on_unmatched == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)
        # "ignore": no-op

    target_dtype = df.schema[code_column]
    try:
        reviewed = reviewed.with_columns(
            pl.col(corrected_column).cast(target_dtype, strict=True)
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError(
            f"import_corrections: could not cast {corrected_column!r} to "
            f"{target_dtype!r} (df[{code_column!r}]'s dtype): {exc}"
        ) from exc

    # Join the incoming corrections under a private alias so we never collide
    # with a column already named `corrected_column` in `df` (which happens
    # when an exported review sample -- it carries a blank `corrected_code` --
    # or a prior CorrectionResult.df is fed back in). A plain join on the same
    # name would suffix the incoming column to `<name>_right` and silently read
    # df's original (blank) column, dropping every correction with no error.
    _CORR = "__pl_hitl_corrected__"
    result_df = df.join(
        reviewed.select([id_column, pl.col(corrected_column).alias(_CORR)]),
        on=id_column,
        how="left",
    )
    assert result_df.height == df.height  # join preserves row count/order

    result_df = result_df.with_columns(
        was_reviewed=pl.col(_CORR).is_not_null()
    ).with_columns(
        was_changed=(
            pl.col("was_reviewed") & (pl.col(_CORR) != pl.col(code_column))
        ).fill_null(False)
    )

    reviewed_rows = result_df.filter(pl.col("was_reviewed"))
    n_reviewed = reviewed_rows.height
    n_changed = int(reviewed_rows.select(pl.col("was_changed").sum()).item() or 0)
    n_unreviewed = df.height - n_reviewed

    if n_reviewed:
        kappa = reviewed_rows.select(
            cohens_kappa(code_column, _CORR)
        ).item()
        agreement_rate = reviewed_rows.select(
            (pl.col(code_column) == pl.col(_CORR)).mean()
        ).item()
    else:
        kappa = float("nan")
        agreement_rate = float("nan")

    # Expose the merged corrections under the requested name, replacing any
    # pre-existing (e.g. blank) column of that name that came in on `df`.
    if corrected_column in result_df.columns:
        result_df = result_df.drop(corrected_column)
    result_df = result_df.rename({_CORR: corrected_column})

    return CorrectionResult(
        df=result_df,
        kappa=kappa,
        agreement_rate=agreement_rate,
        n_reviewed=n_reviewed,
        n_changed=n_changed,
        n_unmatched_corrections=n_unmatched,
        n_unreviewed=n_unreviewed,
    )


# ============================================================================
# Retune -- corrections -> BootstrapFewShot trainset -> compiled Predict
# ============================================================================


def _resolve_signature(signature: Union[Signature, str, Predict]) -> Signature:
    if isinstance(signature, Predict):
        return signature.signature
    if isinstance(signature, Signature):
        return signature
    if isinstance(signature, str):
        return Signature(signature)
    raise TypeError(
        "expected a Signature, shorthand str, or Predict, got "
        f"{type(signature).__name__}"
    )


def corrections_to_trainset(
    corrections: pl.DataFrame,
    signature: Union[Signature, str, Predict],
    *,
    corrected_column: str = "corrected_code",
    column_map: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Map a corrections DataFrame onto the trainset shape `BootstrapFewShot`
    requires: one column per `signature` input field and per output field,
    named exactly like the field (see `polar_llama.optimize.Signature.
    render_inputs`/`BootstrapFewShot.compile`, which index the trainset by
    those literal names).

    Parameters
    ----------
    corrections : pl.DataFrame
        Typically `import_corrections(...).df` (optionally filtered to
        `was_reviewed`), or any DataFrame with the model's input text and
        a corrected-label column.
    signature : Signature, str, or Predict
        The task the trainset is being built for. A `Predict` contributes
        only its `.signature` here.
    corrected_column : str
        Source column for the signature's single output field, unless
        overridden via `column_map`.
    column_map : dict, optional
        Signature field name -> `corrections` column name, for any field
        whose source column doesn't share the field's name (e.g.
        `{"text": "response_text"}`). Every output field of a
        multi-output signature must be covered here (or by
        `corrected_column` for a single-output signature) -- otherwise
        raises `ValueError` naming the missing field(s).

    Returns
    -------
    pl.DataFrame
        Columns are exactly the signature's input names union output
        names. Rows with any null mapped value (i.e. unreviewed rows) are
        dropped. Raises `ValueError` if that leaves zero rows.
    """
    sig = _resolve_signature(signature)
    column_map = dict(column_map or {})

    mapping: Dict[str, str] = {name: column_map.get(name, name) for name in sig.inputs}

    output_names = list(sig.outputs)
    if len(output_names) == 1 and output_names[0] not in column_map:
        mapping[output_names[0]] = corrected_column
    else:
        missing = [name for name in output_names if name not in column_map]
        if missing:
            raise ValueError(
                "corrections_to_trainset: a multi-output signature requires every "
                f"output field covered by column_map; missing {missing!r}"
            )
        for name in output_names:
            mapping[name] = column_map[name]

    missing_cols = sorted({src for src in mapping.values() if src not in corrections.columns})
    if missing_cols:
        raise ValueError(
            f"corrections_to_trainset: corrections is missing column(s) {missing_cols!r}"
        )

    trainset = corrections.select(
        [pl.col(src).alias(field) for field, src in mapping.items()]
    )
    trainset = trainset.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in trainset.columns])
    )

    if trainset.height == 0:
        raise ValueError(
            "corrections_to_trainset: no rows have every mapped field non-null "
            "(empty trainset -- nothing reviewed yet?)"
        )

    return trainset


def _default_metric(signature: Signature) -> Metric:
    """Normalized exact match, averaged over all output fields:
    `str(value).strip().casefold()` comparison per field. Generalizes
    `polar_llama.optimize`'s single-output `exact_match` docstring
    convention to signatures with multiple output fields.
    """
    output_names = list(signature.outputs)

    def metric(example: Dict[str, Any], prediction: Dict[str, Any]) -> float:
        if not output_names:
            return 0.0
        hits = 0
        for name in output_names:
            gold = example.get(name)
            pred = prediction.get(name)
            gold_norm = "" if gold is None else str(gold).strip().casefold()
            pred_norm = "" if pred is None else str(pred).strip().casefold()
            if gold_norm == pred_norm:
                hits += 1
        return hits / len(output_names)

    return metric


def retune_from_corrections(
    corrections: pl.DataFrame,
    signature: Union[Signature, str, Predict],
    *,
    corrected_column: str = "corrected_code",
    column_map: Optional[Dict[str, str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    inference_fn: Optional[InferenceFn] = None,
    metric: Optional[Metric] = None,
    max_demos: int = 4,
    threshold: float = 0.0,
    use_gold_outputs: bool = True,
) -> Predict:
    """
    Compile a `Predict` module with few-shot demos mined from human
    corrections -- one call wrapping `corrections_to_trainset` and the
    existing `BootstrapFewShot` optimizer (`polar_llama.optimize`); no
    bootstrap/demo-selection logic is reimplemented here.

    Parameters
    ----------
    corrections : pl.DataFrame
        Typically `CorrectionResult.df` (optionally filtered to
        `was_reviewed`), or any DataFrame with input text plus a
        corrected-label column.
    signature : Signature, str, or Predict
        The task to tune. Pass a `Predict` to reuse its existing
        `provider`/`model`/`inference_fn` (in which case `provider`,
        `model`, and `inference_fn` below are ignored); pass a
        `Signature`/shorthand string to build a fresh `Predict` from
        `provider`/`model`/`inference_fn`.
    corrected_column, column_map : see `corrections_to_trainset`.
    provider, model, inference_fn : optional
        Backend for a freshly built `Predict` (ignored if `signature` is
        already a `Predict`). `inference_fn` is the injectable, CI-safe
        override used by this package's own test suite.
    metric : callable, optional
        Scoring function, as for `polar_llama.optimize.evaluate`.
        Defaults to a normalized exact match over every output field
        (see `_default_metric`).
    max_demos : int
        Passed through to `BootstrapFewShot`.
    threshold, use_gold_outputs : float, bool
        Passed through to `BootstrapFewShot`. The defaults here
        (`threshold=0.0`, `use_gold_outputs=True`) deliberately diverge
        from `BootstrapFewShot`'s own defaults (`threshold=1.0`,
        `use_gold_outputs=False`): corrections are exactly the rows the
        model got *wrong*, so gating on the model's own predictions
        passing a strict metric would discard precisely the training
        signal this function exists to use. `threshold=0.0` admits every
        corrected row (a metric score is always `>= 0.0`) with the human's
        gold label as the demo output -- i.e. labeled few-shot from
        corrections. Pass `threshold=1.0, use_gold_outputs=False`
        yourself to recover classic bootstrap semantics (demos are the
        model's own passing predictions). Note this still costs one
        inference pass over the trainset (`BootstrapFewShot.compile`
        evaluates the module before selecting demos) -- the accepted price
        of reusing the existing optimizer rather than special-casing this
        path.

    Returns
    -------
    Predict
        A new module (`with_demos(...)`) -- the input `signature`/`Predict`
        is never mutated.
    """
    if isinstance(signature, Predict):
        module = signature
    else:
        module = Predict(
            _resolve_signature(signature),
            provider=provider,
            model=model,
            inference_fn=inference_fn,
        )

    trainset = corrections_to_trainset(
        corrections,
        module.signature,
        corrected_column=corrected_column,
        column_map=column_map,
    )

    optimizer = BootstrapFewShot(
        metric=metric or _default_metric(module.signature),
        max_demos=max_demos,
        threshold=threshold,
        use_gold_outputs=use_gold_outputs,
    )
    return optimizer.compile(module, trainset)
