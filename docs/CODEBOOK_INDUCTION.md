# Codebook Induction

Issue #78. Codebook induction discovers a small set of themes ("codes") in a
column of free text -- support tickets, survey responses, interview
transcripts -- and lets you apply those codes back as structured, multi-label
tags. It is the unsupervised counterpart to `tag_taxonomy`: instead of you
defining the categories up front, the pipeline proposes them from the data,
and you keep (or edit) the ones that make sense.

## Overview

Three functions, composing the same DataFrame-shaped, expression-primitive
style as the rest of polar-llama:

1. **`cluster_embeddings`** -- a whole-column Rust plugin expression that
   clusters an embeddings column with k-means. Zero clustering dependencies:
   k-means++ / Lloyd's algorithm, a splitmix64 PRNG, and a sampled-silhouette
   heuristic for automatic `k` selection are all hand-rolled in
   `src/kmeans.rs` (no `linfa`, `ndarray`, `rand`, scikit-learn, numpy, or
   UMAP).
2. **`induce_codebook`** -- DataFrame-in / DataFrame-out. Embeds (if needed),
   clusters, picks representative exemplars per cluster, and asks an LLM to
   name and define each cluster.
3. **`apply_codebook`** -- expression-in / expression-out. Multi-label-codes
   a text column against a codebook: for each document, evaluates every
   candidate code independently in a single structured-output call.

Plus a bridge, **`codebook_to_taxonomy`**, for when you want single-label
(mutually exclusive) classification against the induced codes instead of
`apply_codebook`'s multi-label evaluation -- it feeds an induced `Codebook`
into `tag_taxonomy`.

## Quick Start

```python
import polars as pl
from polar_llama import induce_codebook, apply_codebook, Provider

tickets = pl.DataFrame({
    "text": [
        "I was charged twice for my subscription this month.",
        "The invoice amount doesn't match what I agreed to pay.",
        "I can't log in -- the password reset link is broken.",
        "My account got locked out after too many login attempts.",
        # ... more rows
    ]
})

# 1. Embed + cluster + name each cluster with an LLM (auto-picks k).
result = induce_codebook(
    tickets, "text", provider=Provider.OPENAI, model="gpt-4o-mini"
)

print(result.codebook)
# Codebook(2 codes: billing_issue, login_issue)

print(result.df.select("text", "cluster_id", "cluster_distance"))
# same rows as `tickets`, plus cluster_id (which cluster) and
# cluster_distance (cosine distance to that cluster's centroid)

# 2. Apply the induced codebook back as multi-label tags. Composes directly
#    onto result.df -- no join or explode needed.
coded = result.df.with_columns(
    labels=apply_codebook(pl.col("text"), result.codebook, provider=Provider.OPENAI)
)
coded.select(
    "text",
    pl.col("labels").explode().struct.unnest(),
)
```

`labels` is `List[Struct{code, applies, confidence, evidence}]`, one entry
per code in the codebook, for every row.

## `cluster_embeddings`

```python
def cluster_embeddings(
    expr: IntoExpr,
    *,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 20,
    max_iter: int = 100,
    n_init: int = 8,
    seed: int = 0,
    silhouette_sample: int = 200,
) -> pl.Expr
```

`expr` is a `List[Float64]` embeddings column. The return value is
`Struct{cluster: UInt32, distance: Float64, k: UInt32}`:

- `cluster`: assigned cluster id (`0..k`).
- `distance`: cosine distance from the point to its cluster's centroid.
- `k`: the number of clusters actually used -- useful when `k` was
  auto-selected, so you can see what was chosen.

Null / empty-vector rows are excluded from clustering and come back as a
null struct.

**Fixed `k`:**

```python
df.with_columns(
    cluster_embeddings(pl.col("embedding"), k=5).alias("clusters")
).unnest("clusters")
```

**Automatic `k` (default):** searches `k_min..k_max`, running k-means at each
candidate and scoring with a *sampled* silhouette score (capped at
`silhouette_sample` points, so the search stays cheap even on large inputs);
picks the `k` with the highest score.

```python
df.with_columns(
    cluster_embeddings(pl.col("embedding"), k_min=2, k_max=15).alias("clusters")
)
```

**Determinism:** the same `seed` over the same input always produces the
same clustering (the PRNG is a hand-rolled, deterministic `splitmix64` --
`src/kmeans.rs`). `n_init` controls how many k-means++ restarts are tried
per candidate `k`, keeping the lowest-inertia run; raise it if clustering
looks unstable on your data.

**Algorithm notes:**
- Distance metric is cosine distance, matching `cosine_similarity` /
  `knn_hnsw` elsewhere in this package.
- Centroids are recomputed each iteration as the re-normalized mean of their
  assigned points ("spherical k-means"), which keeps centroids comparable to
  members under cosine distance.
- Empty-cluster repair: if an iteration leaves a cluster with no points, it
  is reseeded with the point currently farthest from its own centroid, so a
  requested `k` clusters always come back as `k` clusters (given at least
  `k` input points).

## `induce_codebook`

```python
def induce_codebook(
    df: pl.DataFrame,
    column: Union[str, IntoExpr],
    *,
    embedding_column: Optional[str] = None,
    provider=None,
    model: Optional[str] = None,
    embedding_provider=None,
    embedding_model: Optional[str] = None,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 20,
    n_exemplars: int = 5,
    seed: int = 0,
    max_iter: int = 100,
    n_init: int = 8,
) -> CodebookInductionResult
```

Pipeline, built entirely from existing expression primitives:

1. **Embed** -- `embedding_async(pl.col(column), ...)`, unless
   `embedding_column=` names an existing embeddings column to reuse.
2. **Cluster** -- `cluster_embeddings(...)` (fixed `k`, or auto-selected in
   `[k_min, k_max]`).
3. **Pick exemplars** -- for each cluster, the `n_exemplars` rows closest to
   that cluster's centroid, via `sort(["cluster_id", "cluster_distance"])`
   + `group_by("cluster_id").agg(pl.col(column).head(n_exemplars))` --
   ordinary Polars expressions, no manual centroid math in Python.
4. **Name each cluster** -- one `inference_messages(..., response_model=...)`
   call per cluster, showing the exemplars and asking for a `code`,
   `definition`, and `rationale`.
5. **Dedupe** -- clusters are named independently, so two different clusters
   can end up with the same code (e.g. two clusters of complaints both
   induce `"billing_issue"`). `dedupe_codebook_entries` merges same-code
   entries (case/whitespace-insensitive), combining `cluster_ids`, summing
   `size`, and merging (deduplicated, capped) `exemplars`.

Returns a `CodebookInductionResult`:

- `.df`: the input DataFrame with `cluster_id` and `cluster_distance`
  columns appended -- same row count and order as the input, no reshaping.
- `.codebook`: the induced `Codebook` (iterable of `CodebookEntry`:
  `code`, `definition`, `rationale`, `cluster_ids`, `size`, `exemplars`).

If a cluster's naming call fails (`_error` set), it falls back to a
placeholder code (`cluster_<id>`) rather than dropping the cluster or
raising -- inspect `entry.rationale` for the error message and re-run or
hand-edit that entry as needed.

## `apply_codebook`

```python
def apply_codebook(
    expr: IntoExpr,
    codebook: Union[Codebook, Sequence[CodebookEntry], Sequence[dict]],
    *,
    provider=None,
    model: Optional[str] = None,
) -> pl.Expr
```

Multi-label-codes `expr` (a text column) against `codebook`: one
`inference_messages` call per document, evaluating **every** candidate code
in a single structured-output pass. The response model is:

```
{"applications": [{"code", "applies", "confidence", "evidence"}, ...]}
```

-- a fixed-length `List` of fixed-key structs, one entry per candidate code,
**never** a `Dict` keyed by code name. This is deliberate: a `Dict`-typed
field has no fixed `properties` in its JSON schema, so OpenAI Structured
Outputs' strict mode rejects it outright -- exactly the failure mode fixed
for `tag_taxonomy`'s `thinking` field in issue #51. Because the schema shape
never depends on *which* or *how many* codes are in the codebook (only the
prompt content does), it stays strict-mode compliant for a codebook of any
size.

**Why not reuse `tag_taxonomy`?** `tag_taxonomy` is single-label: one
`value` chosen per taxonomy field. Coding "does code X apply, does code Y
apply, ..." independently for several codes would mean either one taxonomy
*field* per code (one full reasoning pass each -- token-explosive for a
codebook of any size) or forcing a single mutually-exclusive choice onto an
inherently multi-label task. `apply_codebook` instead evaluates the whole
codebook in one call per document.

`codebook` accepts a `Codebook` (e.g. `induce_codebook(...).codebook`), a
list of `CodebookEntry`, or a plain list of `{"code": ..., "definition": ...}`
dicts (for a hand-authored codebook, no induction needed).

**Zero-reshaping round trip:** because `induce_codebook` returns the
original DataFrame with only appended columns (same row count and order),
and `apply_codebook` returns a plain expression, applying a just-induced
codebook back onto its own rows is one `with_columns` -- no join, no
explode:

```python
result = induce_codebook(df, "text")
coded = result.df.with_columns(
    labels=apply_codebook(pl.col("text"), result.codebook)
)
```

## `codebook_to_taxonomy`

```python
def codebook_to_taxonomy(
    codebook: Union[Codebook, Sequence[CodebookEntry]],
    *,
    field_name: str = "code",
    description: str = "",
) -> Dict[str, Dict[str, Any]]
```

Bridges an induced `Codebook` into the taxonomy dict shape
`tag_taxonomy` / `_create_taxonomy_pydantic_model` already understand:

```python
from polar_llama import induce_codebook, codebook_to_taxonomy, tag_taxonomy

result = induce_codebook(df, "text")
taxonomy = codebook_to_taxonomy(result.codebook, field_name="topic")

tagged = result.df.with_columns(
    topic=tag_taxonomy(pl.col("text"), taxonomy, provider=Provider.OPENAI)
)
# topic.value is the single best-fitting code, with tag_taxonomy's usual
# thinking/reflection/confidence structure.
```

Use this when classification should be mutually exclusive (exactly one code
per document); use `apply_codebook` when several codes may legitimately
apply to the same document. Raises `ValueError` if the codebook has
duplicate codes (run through `dedupe_codebook_entries` first -- automatic
inside `induce_codebook`) or is empty.

## Data Structures

- **`CodebookEntry`**: `code`, `definition`, `rationale`, `cluster_ids`
  (list -- more than one after a dedupe merge), `size` (total row count
  across its cluster(s)), `exemplars` (representative source texts).
- **`Codebook`**: an ordered, code-deduplicated, read-only-list-like
  container of `CodebookEntry` (`len()`, indexing, iteration, `.codes`,
  `.to_dicts()`, `.to_polars()`).
- **`CodebookInductionResult`**: `.df` + `.codebook`, as returned by
  `induce_codebook`.
- **`dedupe_codebook_entries(entries)`**: the merge-by-code-collision
  helper `induce_codebook` runs automatically; exposed for hand-built
  codebooks or custom induction pipelines.

## The `.llama` Namespace

`cluster_embeddings` and `apply_codebook` are also available as fluent
methods, matching the rest of the package's namespace accessor:

```python
df.with_columns(
    clusters=pl.col("embedding").llama.cluster_embeddings(k=5)
)
coded = result.df.with_columns(
    labels=pl.col("text").llama.apply_codebook(result.codebook)
)
```

## Testing Notes

`cluster_embeddings` is pure Rust with no network calls, so
`tests/test_clustering.py` runs unconditionally in CI (synthetic seeded
Gaussian blobs, auto-`k` selection, determinism, null handling, `k`
override -- no API key, no `mlx`).

`tests/test_codebook.py` monkeypatches `polar_llama.inference_messages`
with a deterministic, keyword-based fake (decoding through the *real*
`_pydantic_to_json_schema` / `_json_schema_to_polars_dtype` /
`_parse_json_to_struct` helpers, so only the network call is faked) to
exercise `induce_codebook` / `apply_codebook` / `codebook_to_taxonomy` /
strict-schema compliance without a key. A separate real-API smoke test is
gated on `OPENAI_API_KEY` and skipped otherwise.
