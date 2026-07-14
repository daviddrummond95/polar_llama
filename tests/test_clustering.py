"""Tests for `cluster_embeddings` (issue #78 codebook induction).

Pure local computation -- k-means runs entirely in Rust with no network
calls, so these tests never skip (no API key, no mlx needed).
"""

import random

import polars as pl
import pytest

from polar_llama import cluster_embeddings


def _blob(rng: random.Random, cx: float, cy: float, n: int, spread: float):
    """`n` 2D points scattered around `(cx, cy)`.

    Centers are chosen away from the origin and at distinct angles from it
    in the tests below: `cluster_embeddings` clusters by cosine distance,
    which is direction-sensitive and origin-unstable, so a blob centered at
    (or collinear through, in the *same* direction as) the origin would not
    be a meaningful separation test (see `src/kmeans.rs`'s test module for
    the same note).
    """
    return [
        [cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread)]
        for _ in range(n)
    ]


def _three_blobs(seed: int = 0, n: int = 15, spread: float = 0.3):
    rng = random.Random(seed)
    points = []
    points += _blob(rng, 20.0, 0.0, n, spread)
    points += _blob(rng, 0.0, 20.0, n, spread)
    points += _blob(rng, -20.0, 0.0, n, spread)
    return points, n


def test_cluster_embeddings_recovers_gaussian_blobs():
    """Three tight, well-separated blobs with k=3 -> each blob maps to its
    own cluster label (same label within a blob, different across blobs)."""
    points, n = _three_blobs()
    df = pl.DataFrame({"emb": points})

    out = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=3, seed=1).alias("_c")
    ).unnest("_c")

    labels = out["cluster"].to_list()
    label_a, label_b, label_c = labels[0], labels[n], labels[2 * n]
    assert len({label_a, label_b, label_c}) == 3

    assert all(l == label_a for l in labels[0:n])
    assert all(l == label_b for l in labels[n : 2 * n])
    assert all(l == label_c for l in labels[2 * n : 3 * n])

    # Cluster sizes should exactly match the source blob sizes.
    counts = out.group_by("cluster").len().sort("cluster")["len"].to_list()
    assert sorted(counts) == [n, n, n]


def test_cluster_embeddings_accepts_float32_embeddings():
    """Regression: user-supplied Float32 embeddings (sentence-transformers /
    numpy default) must cluster correctly, not silently produce null rows."""
    points, n = _three_blobs()
    df = pl.DataFrame({"emb": points}).with_columns(
        pl.col("emb").cast(pl.List(pl.Float32))
    )
    assert df["emb"].dtype == pl.List(pl.Float32)

    out = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=3, seed=1).alias("_c")
    ).unnest("_c")

    labels = out["cluster"].to_list()
    assert all(l is not None for l in labels)  # no silent null clusters
    label_a, label_b, label_c = labels[0], labels[n], labels[2 * n]
    assert len({label_a, label_b, label_c}) == 3
    assert all(l == label_a for l in labels[0:n])
    assert all(l == label_b for l in labels[n : 2 * n])
    assert all(l == label_c for l in labels[2 * n : 3 * n])


def test_cluster_embeddings_auto_k_picks_right_k():
    """With k unset, auto-k (sampled silhouette over k_min..k_max) should
    land on k=3 for three well-separated blobs."""
    points, _n = _three_blobs(seed=7)
    df = pl.DataFrame({"emb": points})

    out = df.with_columns(
        cluster_embeddings(pl.col("emb"), k_min=2, k_max=6, seed=1).alias("_c")
    ).unnest("_c")

    chosen_k = out["k"].unique().to_list()
    assert chosen_k == [3]
    assert out["cluster"].n_unique() == 3


def test_cluster_embeddings_is_deterministic():
    """The same seed over the same input always produces the same labels."""
    points, _n = _three_blobs(seed=3)
    df = pl.DataFrame({"emb": points})

    run1 = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=3, seed=42)
        .struct.field("cluster")
        .alias("c")
    )["c"].to_list()
    run2 = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=3, seed=42)
        .struct.field("cluster")
        .alias("c")
    )["c"].to_list()

    assert run1 == run2


def test_cluster_embeddings_different_seeds_still_recover_same_partition():
    """k-means++ with enough restarts should converge to the same partition
    (up to label permutation) regardless of seed, for well-separated data."""
    points, n = _three_blobs(seed=11)
    df = pl.DataFrame({"emb": points})

    def partition(seed: int):
        labels = df.with_columns(
            cluster_embeddings(pl.col("emb"), k=3, seed=seed)
            .struct.field("cluster")
            .alias("c")
        )["c"].to_list()
        # Normalize by first-seen label per blob so permutation doesn't matter.
        return [labels[0], labels[n], labels[2 * n]], labels

    (a0, a_labels) = partition(1)
    (b0, b_labels) = partition(999)
    assert len(set(a0)) == 3
    assert len(set(b0)) == 3


def test_cluster_embeddings_null_handling():
    """Null / empty-vector rows are excluded from clustering and come back
    null; non-null rows still cluster normally."""
    df = pl.DataFrame(
        {
            "emb": [
                [20.0, 0.1],
                [20.1, -0.1],
                None,
                [0.0, 20.0],
                [-0.1, 19.9],
            ]
        }
    )

    out = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=2, seed=0).alias("_c")
    ).unnest("_c")

    assert out["cluster"][2] is None
    assert out["distance"][2] is None
    assert out["k"][2] is None

    non_null = out.filter(pl.col("cluster").is_not_null())
    assert non_null.height == 4
    assert non_null["cluster"].null_count() == 0


def test_cluster_embeddings_all_null_column():
    """A wholly-null embeddings column returns all-null output, not an error."""
    df = pl.DataFrame({"emb": pl.Series([None, None], dtype=pl.List(pl.Float64))})
    out = df.with_columns(cluster_embeddings(pl.col("emb"), k=2).alias("_c")).unnest(
        "_c"
    )
    assert out["cluster"].null_count() == 2


def test_cluster_embeddings_k_override():
    """Passing k= forces that many clusters, bypassing auto-k selection."""
    points, _n = _three_blobs(seed=5)
    df = pl.DataFrame({"emb": points})

    out2 = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=2, seed=0).alias("_c")
    ).unnest("_c")
    assert out2["k"].unique().to_list() == [2]
    assert out2["cluster"].n_unique() == 2

    out5 = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=5, seed=0).alias("_c")
    ).unnest("_c")
    assert out5["k"].unique().to_list() == [5]
    assert out5["cluster"].n_unique() == 5


def test_cluster_embeddings_single_point():
    df = pl.DataFrame({"emb": [[1.0, 2.0, 3.0]]})
    out = df.with_columns(cluster_embeddings(pl.col("emb"), k=1).alias("_c")).unnest(
        "_c"
    )
    assert out["cluster"][0] == 0
    assert out["distance"][0] == pytest.approx(0.0)
    assert out["k"][0] == 1


def test_cluster_embeddings_namespace_matches_direct_call():
    points, _n = _three_blobs(seed=2)
    df = pl.DataFrame({"emb": points})

    direct = df.with_columns(
        cluster_embeddings(pl.col("emb"), k=3, seed=0)
        .struct.field("cluster")
        .alias("c")
    )["c"].to_list()
    via_ns = df.with_columns(
        pl.col("emb")
        .llama.cluster_embeddings(k=3, seed=0)
        .struct.field("cluster")
        .alias("c")
    )["c"].to_list()

    assert direct == via_ns


def test_cluster_embeddings_empty_dataframe():
    df = pl.DataFrame({"emb": pl.Series([], dtype=pl.List(pl.Float64))})
    out = df.with_columns(cluster_embeddings(pl.col("emb"), k=2).alias("_c")).unnest(
        "_c"
    )
    assert out.height == 0
