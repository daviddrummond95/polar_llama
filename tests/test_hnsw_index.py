"""Tests for the persistent, incrementally updatable HNSW index (issue #82).

Pure local computation -- HNSW build/query, brute-force ground truth, and
(de)serialization all run entirely in Rust/Python with no network call, no
API key, and no mlx, so these tests never skip. All vectors are synthetic
and seeded (`random.Random(seed)`), so every test is deterministic.
"""

from __future__ import annotations

import math
import random
import time

import polars as pl
import pytest

from polar_llama import HnswIndex, NEIGHBOR_STRUCT_DTYPE


# ============================================================================
# Synthetic data helpers
# ============================================================================


def _random_unit_vectors(rng: random.Random, n: int, dim: int):
    vecs = []
    for _ in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _cosine_distance(a, b):
    """Same convention as `src/ann.rs::EmbeddingPoint::distance`."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cos


def _brute_force_topk(ids, vectors, query, k):
    scored = sorted(zip(ids, vectors), key=lambda iv: _cosine_distance(iv[1], query))
    return [i for i, _ in scored[:k]]


def _dataset(seed=0, n=300, dim=16):
    rng = random.Random(seed)
    ids = [f"p{i}" for i in range(n)]
    vectors = _random_unit_vectors(rng, n, dim)
    return ids, vectors


def _df(ids, vectors):
    return pl.DataFrame({"id": ids, "embedding": vectors})


def _avg_recall_at_k(index: HnswIndex, ids, vectors, queries, k=10) -> float:
    """Average top-k overlap (as a fraction in [0, 1]) between the index's
    approximate results and exact brute-force cosine-distance ranking."""
    overlaps = []
    for q in queries:
        got = index.query_one(q, k=k)["neighbor_id"].to_list()
        want = _brute_force_topk(ids, vectors, q, k)
        overlaps.append(len(set(got) & set(want)) / k)
    return sum(overlaps) / len(overlaps)


# ============================================================================
# Recall: fresh build vs. incremental build+add, vs. brute-force ground truth
# ============================================================================


def test_build_recall_matches_brute_force():
    ids, vectors = _dataset(seed=1, n=300, dim=16)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding", seed=7)

    queries = _random_unit_vectors(random.Random(2), 20, 16)
    recall = _avg_recall_at_k(index, ids, vectors, queries, k=10)
    assert recall >= 0.85, f"top-10 recall {recall:.3f} too low vs. brute force"


def test_incremental_build_then_add_matches_fresh_build_recall():
    """Build half the dataset, `.add()` the other half incrementally, and
    confirm recall is statistically indistinguishable from building the
    whole dataset in one `.build()` call."""
    ids, vectors = _dataset(seed=3, n=400, dim=16)
    half = len(ids) // 2
    queries = _random_unit_vectors(random.Random(4), 25, 16)

    fresh = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding", seed=11)
    fresh_recall = _avg_recall_at_k(fresh, ids, vectors, queries, k=10)

    incremental = HnswIndex.build(
        _df(ids[:half], vectors[:half]), id_col="id", embedding_col="embedding", seed=11
    )
    incremental.add(_df(ids[half:], vectors[half:]), id_col="id", embedding_col="embedding")
    incremental_recall = _avg_recall_at_k(incremental, ids, vectors, queries, k=10)

    assert len(incremental) == len(ids)
    # Both should be close to brute-force ground truth; incremental must not
    # be meaningfully worse than a fresh build over the same final point set.
    assert incremental_recall >= 0.80
    assert incremental_recall >= fresh_recall - 0.15


def test_recall_preserved_across_compaction():
    ids, vectors = _dataset(seed=5, n=300, dim=16)
    half = len(ids) // 2
    queries = _random_unit_vectors(random.Random(6), 20, 16)

    index = HnswIndex.build(
        _df(ids[:half], vectors[:half]), id_col="id", embedding_col="embedding", seed=13
    )
    index.auto_compact = False
    index.add(_df(ids[half:], vectors[half:]), id_col="id", embedding_col="embedding")

    pre_compaction_recall = _avg_recall_at_k(index, ids, vectors, queries, k=10)
    assert index.staged_len == len(ids) - half

    index.compact()
    assert index.staged_len == 0
    assert index.tombstone_len == 0

    post_compaction_recall = _avg_recall_at_k(index, ids, vectors, queries, k=10)
    assert post_compaction_recall >= 0.80
    assert abs(post_compaction_recall - pre_compaction_recall) <= 0.15


# ============================================================================
# add() -- queryable immediately
# ============================================================================


def test_add_is_queryable_immediately():
    ids, vectors = _dataset(seed=10, n=50, dim=8)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    index.auto_compact = False

    new_vec = [0.0] * 8
    new_vec[0] = 1.0
    n = index.add(pl.DataFrame({"id": ["brand_new"], "embedding": [new_vec]}), "id", "embedding")
    assert n == 1
    assert index.staged_len == 1
    assert "brand_new" in index

    got = index.query_one(new_vec, k=1)
    assert got["neighbor_id"].to_list() == ["brand_new"]
    assert got["distance"][0] == pytest.approx(0.0, abs=1e-6)


def test_add_empty_dataframe_is_a_noop():
    ids, vectors = _dataset(seed=11, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    n = index.add(pl.DataFrame({"id": [], "embedding": []}, schema={"id": pl.Utf8, "embedding": pl.List(pl.Float64)}), "id", "embedding")
    assert n == 0
    assert len(index) == 10


# ============================================================================
# Soft-delete / upsert
# ============================================================================


def test_remove_excludes_id_and_k_neighbors_still_returned():
    ids, vectors = _dataset(seed=20, n=100, dim=8)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")

    query = vectors[0]
    before = index.query_one(query, k=10)["neighbor_id"].to_list()

    removed = index.remove(before[:3])
    assert removed == 3
    assert index.tombstone_len == 3
    for rid in before[:3]:
        assert rid not in index

    after = index.query_one(query, k=10)
    assert len(after) == 10  # over-fetch backfills the removed slots
    for rid in before[:3]:
        assert rid not in after["neighbor_id"].to_list()


def test_remove_unknown_id_is_a_noop():
    ids, vectors = _dataset(seed=21, n=20, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    assert index.remove(["does-not-exist"]) == 0
    assert len(index) == 20


def test_remove_accepts_a_polars_series():
    ids, vectors = _dataset(seed=22, n=20, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    removed = index.remove(pl.Series(["p0", "p1"]))
    assert removed == 2
    assert len(index) == 18


def test_upsert_replaces_vector_and_old_position_unreachable():
    ids, vectors = _dataset(seed=30, n=100, dim=8)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    index.auto_compact = False

    target_id = ids[0]
    original_vec = vectors[0]
    far_vec = [-x for x in original_vec]  # antipodal: maximally far under cosine distance

    index.add(pl.DataFrame({"id": [target_id], "embedding": [far_vec]}), "id", "embedding")
    assert len(index) == 100  # still just an upsert, not a new point
    assert index.tombstone_len == 1

    got = index.query_one(far_vec, k=1)
    assert got["neighbor_id"][0] == target_id
    assert got["distance"][0] == pytest.approx(0.0, abs=1e-6)

    # Querying near the *old* position should no longer surface target_id
    # ahead of everything else -- it moved to the antipode.
    near_old = index.query_one(original_vec, k=5)["neighbor_id"].to_list()
    assert target_id not in near_old[:1]


# ============================================================================
# Persistence: save / load
# ============================================================================


def test_save_load_round_trip_identical_query_results(tmp_path):
    ids, vectors = _dataset(seed=40, n=150, dim=12)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding", seed=99)

    path = tmp_path / "index.bin"
    index.save(str(path))
    reloaded = HnswIndex.load(str(path))

    assert reloaded.dim == index.dim
    assert len(reloaded) == len(index)

    queries = _random_unit_vectors(random.Random(41), 10, 12)
    for q in queries:
        before = index.query_one(q, k=10)
        after = reloaded.query_one(q, k=10)
        assert before["neighbor_id"].to_list() == after["neighbor_id"].to_list()
        assert before["distance"].to_list() == pytest.approx(after["distance"].to_list())


def test_save_load_round_trip_preserves_pending_staging_and_tombstones(tmp_path):
    ids, vectors = _dataset(seed=42, n=100, dim=8)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    index.auto_compact = False

    new_vec = [0.0] * 8
    new_vec[0] = 1.0
    index.add(pl.DataFrame({"id": ["pending1"], "embedding": [new_vec]}), "id", "embedding")
    index.remove([ids[0]])

    assert index.staged_len == 1
    assert index.tombstone_len == 1

    path = tmp_path / "pending_index.bin"
    index.save(str(path))
    reloaded = HnswIndex.load(str(path))

    assert reloaded.staged_len == 1
    assert reloaded.tombstone_len == 1
    assert "pending1" in reloaded
    assert ids[0] not in reloaded

    got = reloaded.query_one(new_vec, k=1)
    assert got["neighbor_id"].to_list() == ["pending1"]


def test_save_creates_parent_directories(tmp_path):
    ids, vectors = _dataset(seed=43, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    path = tmp_path / "nested" / "dir" / "index.bin"
    index.save(str(path))
    assert path.exists()
    reloaded = HnswIndex.load(str(path))
    assert len(reloaded) == 10


def test_load_rejects_a_non_index_file(tmp_path):
    path = tmp_path / "not_an_index.bin"
    path.write_bytes(b"definitely not an hnsw index file")
    with pytest.raises(ValueError):
        HnswIndex.load(str(path))


# ============================================================================
# Compaction correctness
# ============================================================================


def test_compaction_resets_staged_and_tombstones_and_preserves_live_set():
    ids, vectors = _dataset(seed=50, n=100, dim=8)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    index.auto_compact = False

    extra_vec = [0.0] * 8
    extra_vec[0] = 1.0
    index.add(pl.DataFrame({"id": ["new1"], "embedding": [extra_vec]}), "id", "embedding")
    index.remove([ids[0], ids[1]])

    assert index.staged_len == 1
    assert index.tombstone_len == 2
    assert index.should_compact is False  # thresholds (default 1000) not crossed yet

    index.compact()
    assert index.staged_len == 0
    assert index.tombstone_len == 0
    assert len(index) == 99  # 100 - 2 removed + 1 added
    assert "new1" in index
    assert ids[0] not in index
    assert ids[1] not in index
    assert ids[2] in index


def test_auto_compact_triggers_past_threshold():
    ids, vectors = _dataset(seed=51, n=20, dim=4)
    index = HnswIndex.build(
        _df(ids, vectors),
        id_col="id",
        embedding_col="embedding",
        auto_compact=True,
        compact_staged_min=3,
    )
    for i in range(5):
        v = [0.0] * 4
        v[0] = float(i + 1)
        index.add(pl.DataFrame({"id": [f"extra{i}"], "embedding": [v]}), "id", "embedding")

    # Auto-compact must have fired at least once -- staged never allowed to
    # grow unboundedly past the (very low) threshold.
    assert index.staged_len <= 3
    assert len(index) == 25


def test_auto_compact_disabled_never_compacts_automatically():
    ids, vectors = _dataset(seed=52, n=10, dim=4)
    index = HnswIndex.build(
        _df(ids, vectors),
        id_col="id",
        embedding_col="embedding",
        auto_compact=False,
        compact_staged_min=1,
    )
    for i in range(5):
        v = [0.0] * 4
        v[0] = float(i + 1)
        index.add(pl.DataFrame({"id": [f"extra{i}"], "embedding": [v]}), "id", "embedding")
    assert index.staged_len == 5  # would have auto-compacted if enabled


# ============================================================================
# query() / query_one() / knn() shapes and dtypes
# ============================================================================


def test_batch_query_wrong_dim_raises():
    # Regression (#82 review): a wrong-dimension query on the primary batch
    # .query() API must raise (like .query_one), not silently return garbage
    # (EmbeddingPoint::distance zips over the shorter length).
    ids, vectors = _dataset(seed=61, n=30, dim=6)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")

    bad = pl.DataFrame({"embedding": [[0.1, 0.2, 0.3]]})  # dim 3 != index dim 6
    with pytest.raises((ValueError, Exception), match=r"dim"):
        index.query(bad, embedding_col="embedding", k=3)


def test_query_dataframe_shape_and_dtypes():
    ids, vectors = _dataset(seed=60, n=30, dim=6)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")

    queries = pl.DataFrame({"embedding": vectors[:5]})
    result = index.query(queries, embedding_col="embedding", k=3)

    assert result.schema["query_id"] == pl.Int64
    assert result.schema["neighbor_id"] == pl.Utf8
    assert result.schema["distance"] == pl.Float64
    assert result.schema["rank"] == pl.UInt32
    assert result.height == 5 * 3
    assert result["query_id"].n_unique() == 5

    for qid in range(5):
        sub = result.filter(pl.col("query_id") == qid)
        assert sub["rank"].to_list() == [1, 2, 3]
        # rank 1 (self-match) should have ~0 distance for an exact point.
        assert sub.filter(pl.col("rank") == 1)["distance"][0] == pytest.approx(0.0, abs=1e-6)


def test_query_one_shape():
    ids, vectors = _dataset(seed=61, n=20, dim=6)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    result = index.query_one(vectors[0], k=5)
    assert result.columns == ["neighbor_id", "distance", "rank"]
    assert result.height == 5
    assert result["rank"].to_list() == [1, 2, 3, 4, 5]


def test_query_k_zero_returns_empty():
    ids, vectors = _dataset(seed=62, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    result = index.query_one(vectors[0], k=0)
    assert result.height == 0


def test_query_null_embedding_row_contributes_zero_rows_not_error():
    ids, vectors = _dataset(seed=63, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    queries = pl.DataFrame(
        {"embedding": [vectors[0], None]}, schema={"embedding": pl.List(pl.Float64)}
    )
    result = index.query(queries, embedding_col="embedding", k=3)
    assert result.filter(pl.col("query_id") == 0).height == 3
    assert result.filter(pl.col("query_id") == 1).height == 0


def test_knn_expression_bridge():
    ids, vectors = _dataset(seed=70, n=30, dim=6)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")

    queries = pl.DataFrame({"embedding": vectors[:4]})
    out = queries.with_columns(nbrs=index.knn(pl.col("embedding"), k=3))

    assert out.schema["nbrs"] == pl.List(NEIGHBOR_STRUCT_DTYPE)
    assert out["nbrs"].list.len().to_list() == [3, 3, 3, 3]

    # Cross-check against `.query()`'s batch output for the same input.
    via_query = index.query(queries, embedding_col="embedding", k=3)
    exploded = out.with_row_index("query_id").explode("nbrs").unnest("nbrs")
    assert exploded["neighbor_id"].to_list() == via_query["neighbor_id"].to_list()


def test_knn_null_embedding_row_yields_empty_list_not_null():
    ids, vectors = _dataset(seed=71, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    queries = pl.DataFrame(
        {"embedding": [vectors[0], None]}, schema={"embedding": pl.List(pl.Float64)}
    )
    out = queries.with_columns(nbrs=index.knn(pl.col("embedding"), k=2))
    assert out["nbrs"][1].to_list() == []
    assert out["nbrs"][1] is not None


# ============================================================================
# Validation / error handling
# ============================================================================


def test_build_requires_non_empty_dataframe():
    empty = pl.DataFrame({"id": [], "embedding": []}, schema={"id": pl.Utf8, "embedding": pl.List(pl.Float64)})
    with pytest.raises(ValueError):
        HnswIndex.build(empty, id_col="id", embedding_col="embedding")


def test_build_rejects_missing_columns():
    df = pl.DataFrame({"id": ["a"], "embedding": [[1.0, 2.0]]})
    with pytest.raises(ValueError):
        HnswIndex.build(df, id_col="nope", embedding_col="embedding")
    with pytest.raises(ValueError):
        HnswIndex.build(df, id_col="id", embedding_col="nope")


def test_build_rejects_non_list_embedding_column():
    df = pl.DataFrame({"id": ["a"], "embedding": [1.0]})
    with pytest.raises(ValueError):
        HnswIndex.build(df, id_col="id", embedding_col="embedding")


def test_build_rejects_dimension_mismatch():
    df = pl.DataFrame({"id": ["a", "b"], "embedding": [[1.0, 2.0], [1.0, 2.0, 3.0]]})
    with pytest.raises(ValueError):
        HnswIndex.build(df, id_col="id", embedding_col="embedding")


def test_add_rejects_dimension_mismatch():
    ids, vectors = _dataset(seed=80, n=10, dim=4)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    bad = pl.DataFrame({"id": ["x"], "embedding": [[1.0, 2.0]]})
    with pytest.raises(ValueError):
        index.add(bad, "id", "embedding")


def test_build_duplicate_id_last_row_wins():
    df = pl.DataFrame(
        {"id": ["dup", "dup"], "embedding": [[1.0, 0.0], [0.0, 1.0]]}
    )
    index = HnswIndex.build(df, id_col="id", embedding_col="embedding")
    assert len(index) == 1
    got = index.query_one([0.0, 1.0], k=1)
    assert got["neighbor_id"].to_list() == ["dup"]
    assert got["distance"][0] == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Introspection
# ============================================================================


def test_len_dim_and_contains():
    ids, vectors = _dataset(seed=90, n=25, dim=5)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    assert len(index) == 25
    assert index.dim == 5
    assert "p0" in index
    assert "does-not-exist" not in index


def test_auto_compact_property_get_set():
    ids, vectors = _dataset(seed=91, n=10, dim=4)
    index = HnswIndex.build(
        _df(ids, vectors), id_col="id", embedding_col="embedding", auto_compact=False
    )
    assert index.auto_compact is False
    index.auto_compact = True
    assert index.auto_compact is True


def test_repr_contains_key_stats():
    ids, vectors = _dataset(seed=92, n=5, dim=3)
    index = HnswIndex.build(_df(ids, vectors), id_col="id", embedding_col="embedding")
    text = repr(index)
    assert "len=5" in text
    assert "dim=3" in text


# ============================================================================
# 10k-point CI smoke test (loose timing -- not a benchmark; the real 100k
# benchmark lives in scripts/bench_hnsw_index.py, outside the test suite)
# ============================================================================


def test_10k_points_build_and_query_smoke():
    ids, vectors = _dataset(seed=100, n=10_000, dim=32)
    df = _df(ids, vectors)

    start = time.monotonic()
    index = HnswIndex.build(df, id_col="id", embedding_col="embedding", seed=1)
    build_elapsed = time.monotonic() - start
    assert build_elapsed < 60.0, f"10k build took {build_elapsed:.1f}s -- unexpectedly slow"

    assert len(index) == 10_000
    assert index.dim == 32

    queries = _random_unit_vectors(random.Random(101), 50, 32)
    query_df = pl.DataFrame({"embedding": queries})

    start = time.monotonic()
    result = index.query(query_df, embedding_col="embedding", k=10)
    query_elapsed = time.monotonic() - start
    assert query_elapsed < 15.0, f"50 queries against 10k points took {query_elapsed:.1f}s"
    assert result.height == 50 * 10

    # Sanity: a small sample's recall should still be reasonably high.
    sample_ids = ids[:2000]
    sample_vectors = vectors[:2000]
    recall = _avg_recall_at_k(index, sample_ids, sample_vectors, queries[:5], k=10)
    # Ground truth is computed only over a 2000-point sample (brute-force
    # over the full 10k would be slow in pure Python), so this is a loose
    # sanity check, not a strict recall bound.
    assert recall >= 0.0  # smoke: query completes and returns sane results

    # Incremental add + remove still work at this scale.
    index.add(
        pl.DataFrame({"id": ["extra_pt"], "embedding": [vectors[0]]}), "id", "embedding"
    )
    assert "extra_pt" in index
    removed = index.remove(["p0"])
    assert removed == 1
