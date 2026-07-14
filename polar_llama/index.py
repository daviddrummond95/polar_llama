"""Persistent, incrementally updatable HNSW index (issue #82).

Design summary -- see `docs/VECTOR_SIMILARITY_AND_ANN.md` for the full
write-up:

`instant-distance`'s `HnswMap` (the same type the stateless `knn_hnsw`
expression -- `src/ann.rs` / `src/expressions.rs`, both untouched by this
feature -- builds fresh on every call) is immutable once built: there is no
incremental insert or delete. `HnswIndex` is a Python-facing,
DataFrame-native wrapper (precedent: `Checkpoint` #75, `Codebook` #78 --
a stateful, serializable object, not a stateless plugin expression) around a
small Rust core (`src/index.rs::PyHnswIndex`, registered as the private
`_HnswIndexCore` pyclass) that layers add/remove on top of that immutable
graph:

- **`.build(df, id_col, embedding_col)`**: a fresh `HnswMap` from an entire
  DataFrame.
- **`.add(df, id_col, embedding_col)`**: upserts go into a small brute-force
  *staging* buffer, not the graph -- new/updated points are queryable
  immediately (`.query()`/`.query_one()` merge the graph and the staging
  buffer at search time), without touching the immutable `main` index.
- **`.remove(ids)`**: soft-delete via a *tombstone* set of internal ids;
  `.query()` over-fetches from the graph to compensate for tombstoned hits
  it can't yet physically remove.
- **`.compact()`** (also automatic -- `auto_compact=True` by default, once
  the staging buffer or tombstone set crosses a policy threshold): rebuilds
  `main` from scratch out of every currently-live point, then clears
  staging/tombstones. See `src/index.rs` module docs for the exact
  threshold formula and the internal-id bookkeeping that makes this safe.
- **`.save(path)` / `HnswIndex.load(path)`**: the whole index -- graph,
  pending staging buffer, tombstones, id maps, dimension, and the pinned
  build parameters (including the HNSW `seed`, for reproducible
  compaction) -- round-trips through a small `bincode` file (see
  `src/index.rs`; `bincode` is the only new Rust dependency this feature
  adds).

External ids are arbitrary caller-supplied strings (cast from whatever
dtype `id_col` is), stable across every operation above, including
save/load.

Ids and embeddings cross the Rust boundary as real Polars columns
(`pyo3_polars::{PySeries, PyDataFrame}`, zero-copy Arrow) -- never JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import polars as pl

from polar_llama.utils import parse_into_expr

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Import the compiled core from the extension module, matching the
# try/relative-then-absolute-import fallback pattern `polar_llama/__init__.py`
# uses for `Provider` / `_stream_inference_batch`.
try:
    from .polar_llama import _HnswIndexCore
except ImportError:  # pragma: no cover - exercised only in broken installs
    try:
        from polar_llama.polar_llama import _HnswIndexCore
    except ImportError:  # pragma: no cover
        _HnswIndexCore = None


#: Struct dtype of each element in the list `.knn()` returns per row.
NEIGHBOR_STRUCT_DTYPE = pl.Struct(
    {
        "neighbor_id": pl.Utf8,
        "distance": pl.Float64,
        "rank": pl.UInt32,
    }
)


def _require_core() -> None:
    if _HnswIndexCore is None:  # pragma: no cover - exercised only in broken installs
        raise ImportError(
            "HnswIndex requires the compiled polar_llama Rust extension "
            "(`_HnswIndexCore`), which failed to import. Reinstall "
            "polar-llama, or rebuild it locally with "
            "`maturin develop --release`."
        )


def _require_column(df: pl.DataFrame, col: str, what: str) -> None:
    if col not in df.columns:
        raise ValueError(f"{what} {col!r} not found in DataFrame columns {df.columns}")


def _require_embedding_dtype(df: pl.DataFrame, embedding_col: str) -> None:
    dtype = df.schema[embedding_col]
    if not isinstance(dtype, pl.List):
        raise ValueError(
            f"embedding_col {embedding_col!r} must have a List dtype "
            f"(e.g. List[Float64]/List[Float32]), got {dtype}"
        )


def _require_non_negative_k(k: int) -> None:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")


class HnswIndex:
    """A persistent, incrementally updatable approximate-nearest-neighbor
    index over embedding vectors, backed by `instant-distance`'s HNSW graph.

    Construct via `HnswIndex.build(...)` or `HnswIndex.load(...)` -- never
    the constructor directly.

    >>> import polars as pl
    >>> from polar_llama import HnswIndex
    >>> df = pl.DataFrame({
    ...     "id": ["a", "b", "c"],
    ...     "embedding": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
    ... })
    >>> index = HnswIndex.build(df, id_col="id", embedding_col="embedding")
    >>> index.query_one([1.0, 0.0], k=2)["neighbor_id"].to_list()
    ['a', 'c']
    >>> index.add(
    ...     pl.DataFrame({"id": ["d"], "embedding": [[1.0, 0.0]]}),
    ...     id_col="id",
    ...     embedding_col="embedding",
    ... )
    1
    >>> index.remove(["a"])
    1
    >>> "a" in index
    False
    """

    __slots__ = ("_core",)

    def __init__(self, core: "_HnswIndexCore") -> None:
        # Internal -- use `HnswIndex.build(...)` / `HnswIndex.load(...)`.
        self._core = core

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        df: pl.DataFrame,
        id_col: str,
        embedding_col: str,
        *,
        ef_construction: int = 200,
        ef_search: int = 200,
        seed: int = 42,
        auto_compact: bool = True,
        compact_staged_min: int = 1000,
        compact_staged_ratio: float = 0.10,
        compact_tombstone_min: int = 1000,
        compact_tombstone_ratio: float = 0.10,
    ) -> "HnswIndex":
        """Build a fresh index from every row of `df`.

        Parameters
        ----------
        df
            Source DataFrame.
        id_col
            Column of stable external ids. Cast to `Utf8` internally, so
            any dtype works, but a duplicate *string representation*
            within `df` resolves last-row-wins -- the same upsert semantics
            `.add()` uses for a repeated id.
        embedding_col
            Column of `List[Float32]`/`List[Float64]` embedding vectors,
            all the same length (the length of row 0 is taken as the
            index's dimensionality; a differing row raises `ValueError`
            naming it). Null or empty-vector rows are rejected -- filter
            them out first (e.g. `df.drop_nulls(embedding_col)`) if your
            data has them.
        ef_construction, ef_search
            `instant-distance` HNSW build-quality / search-breadth
            parameters (higher = more accurate, slower). `ef_search` also
            upper-bounds how many candidates a single query can pull from
            the main graph, which in turn caps how much `.query()`'s
            soft-delete over-fetch (`k + tombstone_len`, see
            `src/index.rs`) can compensate for -- an index accumulating a
            great many un-compacted removals wants a larger `ef_search`.
        seed
            Pinned HNSW build seed. Fixed (never re-randomized) so the
            graph -- and therefore query results -- is reproducible for a
            given point set; persisted, and reused by every future
            `.compact()` too.
        auto_compact
            Automatically call `.compact()` after `.add()`/`.remove()` once
            a policy threshold below is crossed. Default `True`. Set
            `False` to control compaction timing yourself (e.g. batch many
            adds, then `.compact()` once).
        compact_staged_min, compact_staged_ratio
            Auto-compact once the staging buffer (pending `.add()`s) grows
            past `max(compact_staged_min, compact_staged_ratio * len(index))`.
        compact_tombstone_min, compact_tombstone_ratio
            Same formula, for the tombstone (soft-deleted) count.

        Returns
        -------
        HnswIndex
        """
        _require_core()
        _require_column(df, id_col, "id_col")
        _require_column(df, embedding_col, "embedding_col")
        _require_embedding_dtype(df, embedding_col)
        if df.height == 0:
            raise ValueError("HnswIndex.build requires a non-empty DataFrame")

        core = _HnswIndexCore.build(
            df.get_column(id_col),
            df.get_column(embedding_col),
            ef_construction=ef_construction,
            ef_search=ef_search,
            seed=seed,
            auto_compact=auto_compact,
            compact_staged_min=compact_staged_min,
            compact_staged_ratio=compact_staged_ratio,
            compact_tombstone_min=compact_tombstone_min,
            compact_tombstone_ratio=compact_tombstone_ratio,
        )
        return cls(core)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HnswIndex":
        """Load an index previously written by `.save()`.

        Round-trips everything: the main graph, any pending staging-buffer
        rows and tombstones that hadn't been compacted yet, the id maps,
        dimension, and build parameters -- querying a freshly loaded index
        gives identical results to the index right before it was saved.
        """
        _require_core()
        return cls(_HnswIndexCore.load(str(path)))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, df: pl.DataFrame, id_col: str, embedding_col: str) -> int:
        """Upsert every row of `df` into the index; queryable immediately.

        An id already live in the index has its old vector soft-deleted
        (tombstoned) and the new vector inserted fresh into the staging
        buffer -- the underlying HNSW graph is immutable, so an upsert
        never mutates a point already baked into it. Embeddings must match
        this index's dimensionality (`.dim`).

        Returns the number of rows processed (0 for an empty `df`). May
        trigger an automatic `.compact()` -- see `auto_compact` on
        `.build()`.
        """
        _require_column(df, id_col, "id_col")
        _require_column(df, embedding_col, "embedding_col")
        _require_embedding_dtype(df, embedding_col)
        if df.height == 0:
            return 0
        return self._core.add(df.get_column(id_col), df.get_column(embedding_col))

    def remove(self, ids: Union[Sequence[str], "pl.Series"]) -> int:
        """Soft-delete (tombstone) `ids`.

        An id not currently live (unknown, or already removed) is silently
        skipped. Returns the number of ids actually removed. May trigger
        an automatic `.compact()` -- see `auto_compact` on `.build()`.
        """
        if not isinstance(ids, pl.Series):
            ids = pl.Series("id", list(ids))
        return self._core.remove(ids)

    def compact(self) -> None:
        """Rebuild the main graph from every currently live point, now.

        Runs regardless of `auto_compact`/the policy thresholds -- useful
        after `auto_compact=False` batch loading, or to force a query
        latency reset after many soft-deletes.
        """
        self._core.compact()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, query_df: pl.DataFrame, embedding_col: str, k: int = 10) -> pl.DataFrame:
        """Batch k-nearest-neighbor search.

        Parameters
        ----------
        query_df
            DataFrame holding the query embeddings. No id column is needed
            -- `query_id` in the result is `query_df`'s 0-based row index.
        embedding_col
            Column of `List[Float32]`/`List[Float64]` query vectors.
        k
            Neighbors per query row (`k=0` returns an empty DataFrame).

        Returns
        -------
        pl.DataFrame
            `query_id | neighbor_id | distance | rank`, `rank` 1-based
            (nearest first). A query row can return *fewer* than `k` rows
            (a small index, or every candidate tombstoned); a null or
            empty-vector query row contributes zero result rows, not an
            error.
        """
        _require_column(query_df, embedding_col, "embedding_col")
        _require_embedding_dtype(query_df, embedding_col)
        _require_non_negative_k(k)
        return self._core.query(query_df.get_column(embedding_col), k)

    def query_one(self, vector: Sequence[float], k: int = 10) -> pl.DataFrame:
        """k-nearest-neighbor search for a single query vector.

        Convenience form of `.query()` for one-off lookups outside a
        DataFrame pipeline.

        Returns
        -------
        pl.DataFrame
            `neighbor_id | distance | rank`, nearest first (no `query_id`
            column -- there is only one query).
        """
        _require_non_negative_k(k)
        return self._core.query_one(list(vector), k)

    def knn(self, embedding_expr: "IntoExpr", k: int = 10) -> pl.Expr:
        """Row-wise k-NN search against this index, as a lazy expression.

        Bridges `.query()`'s whole-batch core into a `map_batches`
        expression: for every row of `embedding_expr`'s column, returns a
        `List[Struct{neighbor_id, distance, rank}]` of that row's `k`
        nearest neighbors in this index (an empty list for a null/empty
        embedding row -- never `null`, so `.list.len()` always works).

        Each `map_batches` invocation makes exactly one batch call into the
        Rust core (same cost profile as `.query()`), not one call per row.

        >>> import polars as pl
        >>> from polar_llama import HnswIndex
        >>> corpus = pl.DataFrame({
        ...     "id": ["a", "b", "c"],
        ...     "embedding": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        ... })
        >>> index = HnswIndex.build(corpus, id_col="id", embedding_col="embedding")
        >>> queries = pl.DataFrame({"embedding": [[1.0, 0.0]]})
        >>> queries.with_columns(nbrs=index.knn(pl.col("embedding"), k=2))["nbrs"][0].struct.field("neighbor_id").to_list()  # doctest: +SKIP
        ['a', 'c']
        """
        _require_non_negative_k(k)
        core = self._core

        def _knn_batch(s: pl.Series) -> pl.Series:
            n = len(s)
            result = core.query(s, k)
            grouped = (
                result.select(
                    pl.col("query_id"),
                    pl.struct(["neighbor_id", "distance", "rank"]).alias("nbrs"),
                )
                .group_by("query_id", maintain_order=True)
                .agg(pl.col("nbrs"))
            )
            empty = pl.Series("nbrs", [[]] * n, dtype=pl.List(NEIGHBOR_STRUCT_DTYPE))
            base = pl.DataFrame(
                {
                    "query_id": pl.Series("query_id", range(n), dtype=pl.Int64),
                    "nbrs": empty,
                }
            )
            out = base.update(grouped, on="query_id").get_column("nbrs")
            return out.rename(s.name)

        return parse_into_expr(embedding_expr).map_batches(
            _knn_batch,
            return_dtype=pl.List(NEIGHBOR_STRUCT_DTYPE),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the whole index to `path`, atomically.

        Includes the main graph, any not-yet-compacted staging/tombstone
        state, the id maps, dimension, and build parameters (including the
        pinned HNSW seed) -- see the module docstring and `src/index.rs`.
        Writes via a temp-file-then-rename in `path`'s directory, so a
        crash mid-write can never leave a truncated file at `path`.
        """
        self._core.save(str(path))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimensionality this index was built with."""
        return self._core.dim

    @property
    def staged_len(self) -> int:
        """Number of points in the staging buffer (pending compaction)."""
        return self._core.staged_len

    @property
    def tombstone_len(self) -> int:
        """Number of soft-deleted internal ids (pending compaction)."""
        return self._core.tombstone_len

    @property
    def should_compact(self) -> bool:
        """Whether a policy threshold is currently crossed (see `.build()`'s
        `compact_*` parameters) -- `.compact()` would do real work right now."""
        return self._core.should_compact()

    @property
    def auto_compact(self) -> bool:
        return self._core.auto_compact

    @auto_compact.setter
    def auto_compact(self, value: bool) -> None:
        self._core.auto_compact = value

    def __len__(self) -> int:
        """Number of currently live points (survives add/remove/compact)."""
        return len(self._core)

    def __contains__(self, id_: object) -> bool:
        return self._core.contains(str(id_))

    def __repr__(self) -> str:
        return (
            f"HnswIndex(len={len(self)}, dim={self.dim}, "
            f"staged={self.staged_len}, tombstones={self.tombstone_len})"
        )
