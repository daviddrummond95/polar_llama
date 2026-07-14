#!/usr/bin/env python
"""Benchmark `HnswIndex` (issue #82) at realistic scale (default 100k points).

Not part of the test suite -- deliberately slow (100k-point HNSW build +
`bincode` save/load), and its pass/fail bar is "reasonable ballpark", not a
strict regression gate. `tests/test_hnsw_index.py::test_10k_points_build_and_query_smoke`
covers the same code paths at a CI-friendly 10k points with loose timing
bounds.

Measures, over synthetic seeded unit vectors (no network, no API key):

- fresh `.build()` wall time
- `.query()` batch latency (ms/query, for a batch of queries)
- `.add()` incremental-insert latency (queryable immediately, pre-compaction)
- `.compact()` wall time after accumulating a staging buffer
- `.save()` / `.load()` wall time and on-disk file size
- approximate recall@10 vs. brute-force cosine-distance ground truth, over a
  small query sample (brute force itself is O(n) per query, so this stays
  small even at n=100k)

Usage::

    python scripts/bench_hnsw_index.py
    python scripts/bench_hnsw_index.py --n 200000 --dim 768 --queries 200
"""

from __future__ import annotations

import argparse
import math
import random
import tempfile
import time
from pathlib import Path

import polars as pl

from polar_llama import HnswIndex


def _random_unit_vectors(rng: random.Random, n: int, dim: int) -> list[list[float]]:
    vecs = []
    for _ in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cos


def _brute_force_topk(ids: list[str], vectors: list[list[float]], query: list[float], k: int):
    scored = sorted(zip(ids, vectors), key=lambda iv: _cosine_distance(iv[1], query))
    return [i for i, _ in scored[:k]]


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="number of indexed points")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimensionality")
    parser.add_argument("--queries", type=int, default=100, help="number of benchmark queries")
    parser.add_argument("--k", type=int, default=10, help="neighbors per query")
    parser.add_argument("--add-n", type=int, default=2000, help="points to `.add()` incrementally")
    parser.add_argument("--recall-sample", type=int, default=5000, help="brute-force ground-truth corpus size (kept small -- O(n) per query in pure Python)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"HnswIndex benchmark: n={args.n} dim={args.dim} k={args.k}\n")

    rng = random.Random(args.seed)
    ids = [f"pt{i}" for i in range(args.n)]
    print("Generating synthetic vectors...")
    t0 = time.monotonic()
    vectors = _random_unit_vectors(rng, args.n, args.dim)
    print(f"  {time.monotonic() - t0:.2f}s\n")

    df = pl.DataFrame({"id": ids, "embedding": vectors})

    print("build()...")
    t0 = time.monotonic()
    index = HnswIndex.build(df, id_col="id", embedding_col="embedding", seed=42)
    build_s = time.monotonic() - t0
    print(f"  {build_s:.2f}s ({args.n / build_s:,.0f} points/s)\n")

    queries = _random_unit_vectors(random.Random(args.seed + 1), args.queries, args.dim)
    query_df = pl.DataFrame({"embedding": queries})

    print("query() (batch)...")
    t0 = time.monotonic()
    result = index.query(query_df, embedding_col="embedding", k=args.k)
    query_s = time.monotonic() - t0
    per_query_ms = (query_s / args.queries) * 1000
    print(f"  {query_s * 1000:.1f}ms total, {per_query_ms:.3f}ms/query "
          f"({result.height} result rows)\n")

    print(f"add() ({args.add_n} incremental points, pre-compaction)...")
    index.auto_compact = False
    extra_ids = [f"extra{i}" for i in range(args.add_n)]
    extra_vectors = _random_unit_vectors(random.Random(args.seed + 2), args.add_n, args.dim)
    extra_df = pl.DataFrame({"id": extra_ids, "embedding": extra_vectors})
    t0 = time.monotonic()
    index.add(extra_df, "id", "embedding")
    add_s = time.monotonic() - t0
    print(f"  {add_s * 1000:.1f}ms total, {(add_s / args.add_n) * 1000:.4f}ms/point "
          f"(staged_len={index.staged_len})\n")

    print("query() with a non-trivial staging buffer (pre-compaction)...")
    t0 = time.monotonic()
    index.query(query_df, embedding_col="embedding", k=args.k)
    staged_query_s = time.monotonic() - t0
    print(f"  {staged_query_s * 1000:.1f}ms total "
          f"({(staged_query_s / args.queries) * 1000:.3f}ms/query)\n")

    print("compact()...")
    t0 = time.monotonic()
    index.compact()
    compact_s = time.monotonic() - t0
    print(f"  {compact_s:.2f}s (staged_len={index.staged_len}, "
          f"tombstone_len={index.tombstone_len}, len={len(index)})\n")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench_index.bin"

        print("save()...")
        t0 = time.monotonic()
        index.save(str(path))
        save_s = time.monotonic() - t0
        size = path.stat().st_size
        print(f"  {save_s:.2f}s, {_fmt_bytes(size)} on disk\n")

        print("load()...")
        t0 = time.monotonic()
        reloaded = HnswIndex.load(str(path))
        load_s = time.monotonic() - t0
        print(f"  {load_s:.2f}s (len={len(reloaded)})\n")

    print(f"Recall@{args.k} sanity check (a separate, smaller "
          f"{args.recall_sample}-point index -- brute force over the full "
          f"{args.n}-point main index in pure Python would dominate this "
          f"script's runtime, so recall is measured on a corpus size where "
          f"exact brute force stays cheap; the *same* corpus backs both the "
          f"approximate and exact search, so this is a real recall number, "
          f"not an artifact of comparing against a mismatched candidate "
          f"pool)...")
    recall_ids = ids[: args.recall_sample]
    recall_vectors = vectors[: args.recall_sample]
    recall_index = HnswIndex.build(
        pl.DataFrame({"id": recall_ids, "embedding": recall_vectors}),
        id_col="id",
        embedding_col="embedding",
        seed=42,
    )
    recall_queries = queries[: min(20, args.queries)]
    overlaps = []
    for q in recall_queries:
        got = recall_index.query_one(q, k=args.k)["neighbor_id"].to_list()
        want = _brute_force_topk(recall_ids, recall_vectors, q, args.k)
        overlaps.append(len(set(got) & set(want)) / args.k)
    recall = sum(overlaps) / len(overlaps)
    print(f"  recall@{args.k} ~= {recall:.3f}\n")

    print("Summary")
    print("-------")
    print(f"  build:            {build_s:.2f}s")
    print(f"  query (compacted):     {per_query_ms:.3f}ms/query")
    print(f"  query (staged):        {(staged_query_s / args.queries) * 1000:.3f}ms/query")
    print(f"  add (per point):       {(add_s / args.add_n) * 1000:.4f}ms")
    print(f"  compact:               {compact_s:.2f}s")
    print(f"  save:                  {save_s:.2f}s ({_fmt_bytes(size)})")
    print(f"  load:                  {load_s:.2f}s")
    print(f"  recall@{args.k} (sample): {recall:.3f}")


if __name__ == "__main__":
    main()
