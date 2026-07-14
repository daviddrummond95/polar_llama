"""
Codebook induction for Polar Llama (issue #78).

Design summary -- see `docs/CODEBOOK_INDUCTION.md` for the full walkthrough:

1. **`cluster_embeddings`**: a whole-column Rust plugin expression (same
   shape as `knn_hnsw`, `src/expressions.rs`) that clusters a `List[Float64]`
   embeddings column with hand-rolled k-means++ / Lloyd's algorithm
   (`src/kmeans.rs`). Zero new Rust dependencies -- no `linfa`, `ndarray`,
   `rand`; a splitmix64 PRNG and a sampled-silhouette auto-`k` heuristic are
   hand-rolled there too. Returns `Struct{cluster, distance, k}` per row
   (`cluster`/`k` null for null/empty-vector input rows).

2. **`induce_codebook`**: DataFrame-in / DataFrame-out. Embeds (if needed),
   clusters, picks per-cluster exemplars (closest-to-centroid rows, via
   ordinary `sort` + `group_by().agg(...head(n))` -- expression primitives,
   no manual centroid math in Python), then asks the LLM to name and define
   each cluster (one `inference_messages` call per cluster, structured
   output). Codes that collide across clusters (the LLM independently
   proposes the same code twice) are merged. Returns the original DataFrame
   plus `cluster_id`/`cluster_distance` columns, and a `Codebook`.

3. **`apply_codebook`**: expression-in / expression-out multi-label coder.
   Unlike `tag_taxonomy` (single-label, one value chosen per taxonomy
   field -- reusing it here would mean one `inference_messages` call *per
   code*, which is both semantically wrong for multi-label coding and
   token-explosive for a codebook of any size), `apply_codebook` makes one
   `inference_messages` call per document against a response model that
   evaluates every candidate code in a single structured-output pass:
   `{"applications": [{"code", "applies", "confidence", "evidence"}, ...]}`
   -- a fixed-shape `List` of fixed-key structs, never a `Dict`/dynamic-key
   object, so it stays OpenAI strict-mode compliant (the issue #51 lesson;
   see `_validate_strict_mode_schema` / `_walk_assert_strict_compliant` in
   `tests/test_taxonomy_strict_schema.py`, and
   `tests/test_codebook.py::test_apply_codebook_response_model_is_strict_compliant`
   for the codebook-specific version of that check).

4. **`codebook_to_taxonomy`**: bridges an induced `Codebook` into the
   `{field_name: {description, values: {code: definition}}}` shape
   `tag_taxonomy` / `_create_taxonomy_pydantic_model` already understand, for
   callers who want single-label (mutually exclusive) classification against
   the induced codes instead of `apply_codebook`'s multi-label evaluation.

`inference_messages`, `embedding_async`, `string_to_message`, and
`combine_messages` are imported lazily inside function bodies (not at module
scope) to avoid a circular import with `polar_llama/__init__.py`, which
imports this module -- the same pattern `polar_llama/tools.py` uses for
`tool_results_to_message`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr
    from pydantic import BaseModel

    from polar_llama import Provider


# ============================================================================
# cluster_embeddings: whole-column k-means plugin expression
# ============================================================================

#: Decoded dtype of the `cluster_embeddings` Rust plugin's per-row JSON
#: output. `cluster`/`k` are null for null or empty-vector input rows.
CLUSTER_STRUCT_DTYPE = pl.Struct(
    {
        "cluster": pl.UInt32,
        "distance": pl.Float64,
        "k": pl.UInt32,
    }
)


def _decode_cluster_json(series: pl.Series) -> pl.Series:
    """Decode `cluster_embeddings`'s JSON string column into a Struct.

    Mirrors `polar_llama._parse_json_to_struct`'s schema-first-then-fallback
    approach (kept local to avoid importing from `polar_llama/__init__.py`,
    which would be circular -- this module is imported *by* that file).
    """
    try:
        return series.str.json_decode(dtype=CLUSTER_STRUCT_DTYPE)
    except Exception:
        try:
            parsed = series.str.json_decode()
            return parsed.cast(CLUSTER_STRUCT_DTYPE, strict=False)
        except Exception:
            return series.str.json_decode()


def cluster_embeddings(
    expr: "IntoExpr",
    *,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 20,
    max_iter: int = 100,
    n_init: int = 8,
    seed: int = 0,
    silhouette_sample: int = 200,
) -> pl.Expr:
    """
    Cluster a column of embeddings with k-means, in one whole-column pass.

    A hand-rolled k-means++ / Lloyd's algorithm (`src/kmeans.rs`) runs over
    the entire (non-null) embeddings column at once -- there is no external
    clustering dependency (no `linfa`/`ndarray`/`scikit-learn`). Distance is
    cosine distance, matching the metric already used for embeddings
    elsewhere in this package (`cosine_similarity`, `knn_hnsw`).

    Parameters
    ----------
    expr : polars.Expr
        Embeddings column (`List[Float64]`, e.g. from `embedding_async`).
    k : int, optional
        Fixed number of clusters. When `None` (the default), `k` is chosen
        automatically by searching `[k_min, k_max]` and picking the value
        with the highest sampled silhouette score.
    k_min, k_max : int
        Search range for automatic `k` selection (ignored when `k` is
        given). Both are clamped to `[1, n_rows]` internally.
    max_iter : int
        Maximum Lloyd's-algorithm iterations per run.
    n_init : int
        Number of k-means++ restarts per candidate `k`; the lowest-inertia
        restart is kept. Higher values are more robust to unlucky seeding at
        the cost of more compute.
    seed : int
        Seed for the (hand-rolled, deterministic) PRNG. The same `seed` over
        the same input always produces the same clustering.
    silhouette_sample : int
        Number of points sampled per candidate `k` when scoring with
        silhouette (auto-`k` only); caps the cost of the search on large
        inputs.

    Returns
    -------
    polars.Expr
        `Struct{cluster: UInt32, distance: Float64, k: UInt32}` -- `cluster`
        is the assigned cluster id (`0..k`), `distance` is the cosine
        distance to that cluster's centroid, and `k` is the number of
        clusters actually used (the chosen `k`, whether fixed or
        auto-selected). All three are null for a null or empty-vector input
        row.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, cluster_embeddings
    >>> df = pl.DataFrame({"text": ["a", "b", "c", "d"]}).with_columns(
    ...     emb=embedding_async(pl.col("text"))
    ... )
    >>> clustered = df.with_columns(
    ...     cluster=cluster_embeddings(pl.col("emb"), k=2).struct.field("cluster")
    ... )
    """
    expr = parse_into_expr(expr)

    kwargs: Dict[str, Any] = {
        "k": int(k) if k is not None else None,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "max_iter": int(max_iter),
        "n_init": int(n_init),
        "seed": int(seed),
        "silhouette_sample": int(silhouette_sample),
    }

    from polar_llama.expressions import get_lib_path

    json_expr = register_plugin(
        args=[expr],
        symbol="cluster_embeddings",
        is_elementwise=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )
    return json_expr.map_batches(
        _decode_cluster_json, return_dtype=CLUSTER_STRUCT_DTYPE
    )


# ============================================================================
# Response models (strict-schema-safe: fixed keys only, no Dict/dynamic map)
# ============================================================================


def _cluster_code_model() -> Type["BaseModel"]:
    """Per-cluster naming/definition response model, used by
    `induce_codebook` (one call per cluster). Every field is required (no
    defaults) and there is no `Dict`-typed field anywhere, so this is
    OpenAI-strict-mode compliant by construction -- see
    `_validate_strict_mode_schema` in `polar_llama/__init__.py`.
    """
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "Pydantic is required for codebook induction. Install with: pip install pydantic>=2.0.0"
        )

    class ClusterCodeResult(BaseModel):
        code: str = Field(
            ...,
            description=(
                "A short, distinct label for this cluster (2-5 words; "
                "Title Case or snake_case), suitable as a dictionary key"
            ),
        )
        definition: str = Field(
            ...,
            description=(
                "A precise one-to-two sentence definition of what this code "
                "captures -- specific enough to apply consistently to new, "
                "unseen documents"
            ),
        )
        rationale: str = Field(
            ...,
            description=(
                "Why the exemplar documents below share this code; note any "
                "notable internal variation among them"
            ),
        )

    return ClusterCodeResult


def _code_application_model() -> Type["BaseModel"]:
    """Multi-label apply response model, used by `apply_codebook` (one call
    per document, evaluating every candidate code). `applications` is a
    `List` of fixed-key `{code, applies, confidence, evidence}` structs --
    deliberately never a `Dict[code, ...]`, which would have no fixed
    `properties` and so would fail OpenAI strict mode (the issue #51
    lesson). See `tests/test_codebook.py` for the strict-mode-compliance
    regression test.
    """
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "Pydantic is required for codebook induction. Install with: pip install pydantic>=2.0.0"
        )

    class CodeApplication(BaseModel):
        code: str = Field(
            ...,
            description="The exact code name being evaluated (must match one of the candidate codes verbatim)",
        )
        applies: bool = Field(
            ..., description="Whether this code applies to the document"
        )
        confidence: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence that `applies` is correct (0.0 to 1.0)",
        )
        evidence: str = Field(
            ...,
            description=(
                "A short quote or paraphrase from the document supporting the "
                "decision (or a brief note on why the code does not apply)"
            ),
        )

    class ApplyCodebookResult(BaseModel):
        applications: List[CodeApplication] = Field(
            ...,
            description=(
                "Exactly one entry per candidate code, in the order given, "
                "evaluating whether it applies to the document"
            ),
        )

    return ApplyCodebookResult


# ============================================================================
# Codebook data structures
# ============================================================================


@dataclass
class CodebookEntry:
    """One induced (or hand-authored) code."""

    code: str
    definition: str
    rationale: str
    cluster_ids: List[int]
    size: int
    exemplars: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "definition": self.definition,
            "rationale": self.rationale,
            "cluster_ids": list(self.cluster_ids),
            "size": self.size,
            "exemplars": list(self.exemplars),
        }


class Codebook:
    """An ordered, code-deduplicated set of `CodebookEntry`.

    Read-only-list-like (`len()`, indexing, iteration) so it composes
    directly with `apply_codebook` and `codebook_to_taxonomy` -- e.g.
    `apply_codebook(pl.col("text"), result.codebook)`.
    """

    def __init__(self, entries: Sequence[CodebookEntry]):
        self._entries: List[CodebookEntry] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> CodebookEntry:
        return self._entries[index]

    def __iter__(self) -> Iterator[CodebookEntry]:
        return iter(self._entries)

    def __repr__(self) -> str:
        codes = ", ".join(e.code for e in self._entries)
        return f"Codebook({len(self._entries)} codes: {codes})"

    @property
    def codes(self) -> List[str]:
        return [e.code for e in self._entries]

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dicts())


@dataclass
class CodebookInductionResult:
    """Return value of `induce_codebook`: the input DataFrame with
    `cluster_id`/`cluster_distance` columns appended (same row count and
    order as the input -- no reshaping), plus the induced `Codebook`.
    """

    df: pl.DataFrame
    codebook: Codebook


def _normalize_code(code: str) -> str:
    return " ".join(code.strip().lower().split())


def dedupe_codebook_entries(
    entries: Iterable[CodebookEntry], *, max_exemplars: int = 10
) -> List[CodebookEntry]:
    """Merge entries whose `code` collides (case/whitespace-insensitively).

    `induce_codebook` makes one independent LLM call per cluster; nothing
    stops two different clusters from being named the same thing (e.g. two
    clusters of complaints both come back "billing_issue"). Merging keeps
    codes unique -- a hard requirement for `codebook_to_taxonomy`, whose
    `values` dict is keyed by code -- while preserving every contributing
    cluster's `cluster_ids`, summed `size`, and combined (deduplicated,
    capped) `exemplars`. The first entry's `code`/`definition` win; later
    duplicates' `rationale` is appended for traceability.
    """
    merged: Dict[str, CodebookEntry] = {}
    for entry in entries:
        key = _normalize_code(entry.code)
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            continue

        combined_exemplars = list(existing.exemplars)
        for ex in entry.exemplars:
            if ex not in combined_exemplars:
                combined_exemplars.append(ex)

        merged[key] = CodebookEntry(
            code=existing.code,
            definition=existing.definition,
            rationale=f"{existing.rationale} | (merged with cluster {entry.cluster_ids}: {entry.rationale})",
            cluster_ids=[*existing.cluster_ids, *entry.cluster_ids],
            size=existing.size + entry.size,
            exemplars=combined_exemplars[:max_exemplars],
        )
    return list(merged.values())


def codebook_to_taxonomy(
    codebook: Union[Codebook, Sequence[CodebookEntry]],
    *,
    field_name: str = "code",
    description: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Bridge an induced `Codebook` into a `tag_taxonomy`-compatible taxonomy
    dict, for single-label (mutually exclusive) classification against the
    induced codes -- feeds directly into
    `polar_llama._create_taxonomy_pydantic_model` / `tag_taxonomy`.

    For multi-label coding (several codes may apply to the same document),
    use `apply_codebook` instead -- reusing `tag_taxonomy` for that would
    mean one taxonomy *field* per code (one full reasoning pass each),
    which does not scale with codebook size.

    Parameters
    ----------
    codebook : Codebook or sequence of CodebookEntry
        The induced codebook (e.g. `induce_codebook(...).codebook`).
    field_name : str
        Name of the single taxonomy field to create. Default `"code"`.
    description : str, optional
        Taxonomy field description. Defaults to a generic description
        mentioning the number of codes.

    Returns
    -------
    dict
        `{field_name: {"description": ..., "values": {code: definition, ...}}}`

    Examples
    --------
    >>> from polar_llama import induce_codebook, codebook_to_taxonomy, tag_taxonomy
    >>> result = induce_codebook(df, "text")  # doctest: +SKIP
    >>> taxonomy = codebook_to_taxonomy(result.codebook, field_name="topic")  # doctest: +SKIP
    >>> tagged = result.df.with_columns(
    ...     topic=tag_taxonomy(pl.col("text"), taxonomy)
    ... )  # doctest: +SKIP
    """
    entries = list(codebook)
    if not entries:
        raise ValueError("codebook_to_taxonomy: codebook has no entries")

    values: Dict[str, str] = {}
    for entry in entries:
        if entry.code in values:
            raise ValueError(
                f"codebook_to_taxonomy: duplicate code {entry.code!r} in "
                "codebook -- codes must be unique (induce_codebook dedupes "
                "automatically via dedupe_codebook_entries; if you built "
                "this Codebook by hand, dedupe it yourself first)"
            )
        values[entry.code] = entry.definition

    return {
        field_name: {
            "description": description
            or (
                f"The single best-fitting code for this document, from an "
                f"induced codebook of {len(entries)} codes"
            ),
            "values": values,
        }
    }


# ============================================================================
# Prompt builders
# ============================================================================


def _build_induction_system_prompt() -> str:
    return "\n".join(
        [
            "You are an expert qualitative-coding analyst building a codebook",
            "from a corpus. You will be shown a set of exemplar documents that",
            "a clustering algorithm grouped together because they are similar.",
            "",
            "Propose ONE short code that captures what this cluster of",
            "documents has in common, along with a precise definition and your",
            "rationale. The code must be specific enough to apply consistently",
            "to new, unseen documents -- avoid vague labels like 'general' or",
            "'misc'.",
            "",
            "The exemplar documents are separated by '---' below.",
        ]
    )


def _build_apply_system_prompt(entries: Sequence[CodebookEntry]) -> str:
    lines = [
        "You are an expert document coder. You will be given a document and a",
        "fixed list of candidate codes, each with a definition. This is a",
        "MULTI-LABEL task: zero, one, or several codes may apply to the same",
        "document -- they are not mutually exclusive.",
        "",
        "# Candidate Codes",
        "",
    ]
    for entry in entries:
        lines.append(f"- **{entry.code}**: {entry.definition}")
    lines.extend(
        [
            "",
            "# Instructions",
            "",
            "For each candidate code above, add exactly one entry to",
            "`applications` (same order as listed) with:",
            "",
            "1. `code`: the exact code name, copied verbatim.",
            "2. `applies`: true if the document matches this code's",
            "   definition, else false.",
            "3. `confidence`: your confidence in that decision, 0.0 to 1.0.",
            "4. `evidence`: a short quote or paraphrase from the document",
            "   supporting the decision (or a brief note on why it does not",
            "   apply).",
            "",
            "Return exactly one `applications` entry per candidate code --",
            "never omit a code and never invent one that isn't listed above.",
        ]
    )
    return "\n".join(lines)


def _coerce_codebook_entries(
    codebook: Union[Codebook, Sequence[CodebookEntry], Sequence[Dict[str, Any]]],
) -> List[CodebookEntry]:
    entries: List[CodebookEntry] = []
    for item in codebook:
        if isinstance(item, CodebookEntry):
            entries.append(item)
        elif isinstance(item, dict):
            entries.append(
                CodebookEntry(
                    code=item["code"],
                    definition=item.get("definition", ""),
                    rationale=item.get("rationale", ""),
                    cluster_ids=list(item.get("cluster_ids", [])),
                    size=int(item.get("size", 0)),
                    exemplars=list(item.get("exemplars", [])),
                )
            )
        else:
            raise TypeError(
                f"apply_codebook: unsupported codebook entry type {type(item)!r} "
                "(expected a Codebook, CodebookEntry, or dict with a 'code' key)"
            )
    return entries


def _resolve_column_name(df: pl.DataFrame, column: Union[str, "IntoExpr"]) -> str:
    if isinstance(column, str):
        if column not in df.columns:
            raise ValueError(
                f"induce_codebook: column {column!r} not found in dataframe"
            )
        return column
    if isinstance(column, pl.Expr):
        try:
            name = column.meta.output_name()
        except Exception as e:
            raise TypeError(
                "induce_codebook: `column` must be a column name (str) or a "
                "simple `pl.col(name)` expression with a resolvable output "
                "name"
            ) from e
        if name not in df.columns:
            raise ValueError(f"induce_codebook: column {name!r} not found in dataframe")
        return name
    raise TypeError(f"induce_codebook: unsupported `column` type {type(column)!r}")


# ============================================================================
# induce_codebook / apply_codebook
# ============================================================================


def induce_codebook(
    df: pl.DataFrame,
    column: Union[str, "IntoExpr"],
    *,
    embedding_column: Optional[str] = None,
    provider: Optional[Union[str, "Provider"]] = None,
    model: Optional[str] = None,
    embedding_provider: Optional[Union[str, "Provider"]] = None,
    embedding_model: Optional[str] = None,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 20,
    n_exemplars: int = 5,
    seed: int = 0,
    max_iter: int = 100,
    n_init: int = 8,
) -> CodebookInductionResult:
    """
    Induce a codebook from a text column: embed, cluster, and name each
    cluster with an LLM. DataFrame-in / DataFrame-out.

    Pipeline (all built from existing expression primitives -- see
    `docs/CODEBOOK_INDUCTION.md`):

    1. `embedding_async` (unless `embedding_column` is given).
    2. `cluster_embeddings` (hand-rolled k-means; auto-`k` unless `k` is
       given).
    3. Per-cluster exemplar selection via `sort` + `group_by().agg(...head(n))`
       -- the `n_exemplars` rows closest to their cluster's centroid.
    4. One `inference_messages(..., response_model=...)` call per cluster to
       name and define it.
    5. Cross-cluster code-collision dedupe (`dedupe_codebook_entries`).

    Parameters
    ----------
    df : polars.DataFrame
        Input data.
    column : str or polars.Expr
        Name of the text column to induce a codebook from (or a bare
        `pl.col(name)` expression referencing it).
    embedding_column : str, optional
        Name of an existing `List[Float64]` embeddings column to cluster,
        instead of computing new embeddings from `column`.
    provider, model : optional
        Provider/model for the per-cluster naming calls (`inference_messages`).
    embedding_provider, embedding_model : optional
        Provider/model for `embedding_async`, when `embedding_column` is not
        given.
    k : int, optional
        Fixed number of clusters. `None` (default) auto-selects `k` in
        `[k_min, k_max]` via sampled silhouette.
    k_min, k_max : int
        Auto-`k` search range (ignored when `k` is given).
    n_exemplars : int
        Number of centroid-closest exemplar rows per cluster shown to the
        LLM when naming that cluster.
    seed : int
        k-means seed; the same `seed` over the same embeddings always
        produces the same clustering.
    max_iter, n_init : int
        k-means iteration cap / restart count -- see `cluster_embeddings`.

    Returns
    -------
    CodebookInductionResult
        `.df`: `df` plus `cluster_id` (`UInt32`, null if that row's
        embedding was null) and `cluster_distance` (`Float64`) columns, same
        row count and order as `df`. `.codebook`: the induced `Codebook`.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import induce_codebook, Provider
    >>> df = pl.DataFrame({"ticket": ["...", "...", "..."]})  # doctest: +SKIP
    >>> result = induce_codebook(
    ...     df, "ticket", provider=Provider.OPENAI, model="gpt-4o-mini", k=5
    ... )  # doctest: +SKIP
    >>> result.codebook.codes  # doctest: +SKIP
    ['billing_issue', 'login_failure', 'feature_request', ...]
    >>> result.df.select("ticket", "cluster_id")  # doctest: +SKIP
    """
    if df.height == 0:
        raise ValueError("induce_codebook: dataframe is empty")

    text_col = _resolve_column_name(df, column)

    from polar_llama import combine_messages, inference_messages, string_to_message

    work = df
    if embedding_column is not None:
        if embedding_column not in df.columns:
            raise ValueError(
                f"induce_codebook: embedding_column {embedding_column!r} not found in dataframe"
            )
        emb_col = embedding_column
    else:
        from polar_llama import embedding_async

        emb_col = "__codebook_embedding"
        work = work.with_columns(
            **{
                emb_col: embedding_async(
                    pl.col(text_col), provider=embedding_provider, model=embedding_model
                )
            }
        )

    work = work.with_columns(
        __codebook_cluster=cluster_embeddings(
            pl.col(emb_col),
            k=k,
            k_min=k_min,
            k_max=k_max,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed,
        )
    )
    work = work.with_columns(
        __codebook_cluster_id=pl.col("__codebook_cluster").struct.field("cluster"),
        __codebook_distance=pl.col("__codebook_cluster").struct.field("distance"),
    ).drop("__codebook_cluster")

    rows_with_cluster = work.filter(pl.col("__codebook_cluster_id").is_not_null())
    if rows_with_cluster.height == 0:
        raise ValueError(
            "induce_codebook: no row had a usable (non-null) embedding to cluster"
        )

    exemplars_df = (
        rows_with_cluster.sort(["__codebook_cluster_id", "__codebook_distance"])
        .group_by("__codebook_cluster_id", maintain_order=True)
        .agg(
            pl.col(text_col).head(n_exemplars).alias("__exemplars"),
            pl.len().alias("__cluster_size"),
        )
        .sort("__codebook_cluster_id")
        .with_columns(__prompt_text=pl.col("__exemplars").list.join("\n---\n"))
    )

    system_prompt = _build_induction_system_prompt()
    system_message_expr = (
        pl.col("__prompt_text")
        .map_batches(
            lambda s: pl.Series([system_prompt] * len(s)), return_dtype=pl.Utf8
        )
        .pipe(string_to_message, message_type="system")
    )
    user_message_expr = pl.col("__prompt_text").pipe(
        string_to_message, message_type="user"
    )
    messages_expr = combine_messages(system_message_expr, user_message_expr)

    response_model = _cluster_code_model()
    exemplars_df = exemplars_df.with_columns(
        __induced=inference_messages(
            messages_expr, provider=provider, model=model, response_model=response_model
        )
    )

    entries: List[CodebookEntry] = []
    for row in exemplars_df.select(
        pl.col("__codebook_cluster_id").alias("cluster_id"),
        pl.col("__cluster_size").alias("size"),
        pl.col("__exemplars").alias("exemplars"),
        pl.col("__induced").struct.field("code").alias("code"),
        pl.col("__induced").struct.field("definition").alias("definition"),
        pl.col("__induced").struct.field("rationale").alias("rationale"),
        pl.col("__induced").struct.field("_error").alias("_error"),
    ).iter_rows(named=True):
        if row["_error"] is not None or row["code"] is None:
            code = f"cluster_{row['cluster_id']}"
            definition = (
                "LLM induction failed for this cluster; inspect the exemplars manually."
            )
            rationale = row["_error"] or "unknown induction error"
        else:
            code = row["code"]
            definition = row["definition"]
            rationale = row["rationale"]
        entries.append(
            CodebookEntry(
                code=code,
                definition=definition,
                rationale=rationale,
                cluster_ids=[row["cluster_id"]],
                size=row["size"],
                exemplars=list(row["exemplars"]),
            )
        )

    entries = dedupe_codebook_entries(
        entries, max_exemplars=max(2 * n_exemplars, n_exemplars)
    )
    codebook = Codebook(entries)

    result_df = df.with_columns(
        cluster_id=work["__codebook_cluster_id"],
        cluster_distance=work["__codebook_distance"],
    )

    return CodebookInductionResult(df=result_df, codebook=codebook)


def apply_codebook(
    expr: "IntoExpr",
    codebook: Union[Codebook, Sequence[CodebookEntry], Sequence[Dict[str, Any]]],
    *,
    provider: Optional[Union[str, "Provider"]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Multi-label-apply a codebook to a text column: for each document,
    evaluate every candidate code independently in a single structured
    output call.

    Built on `inference_messages` + a strict-schema-safe response model
    (`List[{code, applies, confidence, evidence}]`) -- *not* `tag_taxonomy`,
    which is single-label and would need one call per code to express
    "several codes may apply" (token-explosive and semantically wrong for
    this task; see the module docstring).

    Parameters
    ----------
    expr : polars.Expr
        The document/text expression to code.
    codebook : Codebook, sequence of CodebookEntry, or sequence of dict
        The codebook to apply (e.g. `induce_codebook(...).codebook`, or a
        hand-authored list of `{"code": ..., "definition": ...}` dicts).
    provider, model : optional
        Provider/model for the coding calls.

    Returns
    -------
    polars.Expr
        `List[Struct{code, applies, confidence, evidence}]`, one entry per
        candidate code in `codebook`, in order. Composes directly onto any
        DataFrame holding the text column via `with_columns` -- no join or
        explode needed. In particular, this "round-trips" zero-reshaping
        onto the DataFrame `induce_codebook` returned:

        >>> from polar_llama import induce_codebook, apply_codebook  # doctest: +SKIP
        >>> result = induce_codebook(df, "text")  # doctest: +SKIP
        >>> coded = result.df.with_columns(
        ...     labels=apply_codebook(pl.col("text"), result.codebook)
        ... )  # doctest: +SKIP
    """
    entries = _coerce_codebook_entries(codebook)
    if not entries:
        raise ValueError("apply_codebook: codebook has no entries")

    expr = parse_into_expr(expr)

    from polar_llama import combine_messages, inference_messages, string_to_message

    system_prompt = _build_apply_system_prompt(entries)
    system_message_expr = expr.map_batches(
        lambda s: pl.Series([system_prompt] * len(s)), return_dtype=pl.Utf8
    ).pipe(string_to_message, message_type="system")
    user_message_expr = expr.pipe(string_to_message, message_type="user")
    messages_expr = combine_messages(system_message_expr, user_message_expr)

    response_model = _code_application_model()
    result = inference_messages(
        messages_expr, provider=provider, model=model, response_model=response_model
    )
    return result.struct.field("applications")
