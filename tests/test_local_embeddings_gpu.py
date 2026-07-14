"""GPU-gated tests for the real mlx_embeddings-backed local embedding engine
(issue #83).

Deselected in CI by the existing `-m "not local_gpu"` filter
(.github/workflows/CI.yml). Run these directly on an Apple-Silicon Mac with
the `[local]` extra installed:

    uv run pytest tests/test_local_embeddings_gpu.py -v

The default model (`mlx-community/bge-small-en-v1.5-bf16`, ~65MB) downloads
from the HF Hub on first use and is cached under `~/.cache/huggingface`;
subsequent runs are fully offline (no provider API key needed anywhere in
this pipeline).
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from polar_llama import embedding_local
from polar_llama.index import HnswIndex
from polar_llama.local.embed import (
    DEFAULT_LOCAL_EMBED_MODEL,
    clear_embedding_registry,
    get_embedding_engine,
)

pytestmark = pytest.mark.local_gpu


@pytest.fixture(autouse=True)
def _clean_embedding_registry():
    clear_embedding_registry()
    yield
    clear_embedding_registry()


# ---------------------------------------------------------------------------
# 12. Acceptance-criteria demo: fully offline embed -> HnswIndex -> query
# ---------------------------------------------------------------------------
def test_e2e_offline_embed_index_query():
    docs = pl.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "text": [
                "How do I reset my password?",
                "Steps to change your account password",
                "The quarterly earnings report shows growth",
                "Our cat needs to visit the veterinarian",
                "Forgot your password? Here's how to reset it",
                "Stock prices rose sharply this quarter",
                "Best recipes for a summer barbecue",
                "The dog ran happily in the park",
            ],
        }
    )

    t0 = time.perf_counter()
    df = docs.with_columns(emb=embedding_local(pl.col("text"), model=DEFAULT_LOCAL_EMBED_MODEL))
    elapsed = time.perf_counter() - t0
    docs_per_sec = len(docs) / elapsed if elapsed > 0 else float("inf")
    print(f"\n[local embeddings] {len(docs)} docs in {elapsed:.3f}s ({docs_per_sec:.1f} docs/sec)")

    assert df.schema["emb"] == pl.List(pl.Float64)
    dims = df["emb"].list.len().to_list()
    assert len(set(dims)) == 1  # consistent dimensionality across rows
    assert dims[0] > 0

    # Unit norms (default normalize=True).
    for vec in df["emb"].to_list():
        norm = sum(x * x for x in vec) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

    index = HnswIndex.build(df, id_col="id", embedding_col="emb")

    query_df = pl.DataFrame({"text": ["how do I reset my password"]}).with_columns(
        emb=embedding_local(pl.col("text"), model=DEFAULT_LOCAL_EMBED_MODEL)
    )
    result = index.query(query_df.select("emb"), "emb", k=3).sort("rank")

    top_id = result["neighbor_id"][0]
    assert top_id in ("1", "2", "5")  # the password-reset-related docs


def test_real_engine_singleton_load_once():
    eng1 = get_embedding_engine(DEFAULT_LOCAL_EMBED_MODEL, engine="in_process")
    eng2 = get_embedding_engine(DEFAULT_LOCAL_EMBED_MODEL, engine="in_process")
    assert eng1 is eng2
