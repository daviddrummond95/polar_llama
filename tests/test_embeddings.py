#!/usr/bin/env python3
"""
Test suite for parallel embedding operations in polar-llama.

This test verifies the parallelized, memory-efficient embedding functionality.
"""

import os
import polars as pl
import pytest

# Skip tests if no API key is available
skip_if_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@skip_if_no_openai
def test_basic_embedding():
    """Test basic embedding generation with OpenAI."""
    from polar_llama import embedding_async

    # Create a simple dataframe
    df = pl.DataFrame({
        "text": ["Hello world", "Machine learning is exciting"]
    })

    # Generate embeddings
    result = df.with_columns(
        embeddings=embedding_async(pl.col("text"))
    )

    # Verify structure
    assert "embeddings" in result.columns
    assert result["embeddings"].dtype == pl.List(pl.Float64)

    # Verify we got embeddings for all rows
    assert result.height == 2
    assert result["embeddings"][0] is not None
    assert result["embeddings"][1] is not None

    # Verify embedding dimensions (default OpenAI small model is 1536)
    assert len(result["embeddings"][0]) == 1536
    assert len(result["embeddings"][1]) == 1536

    print("✓ Basic embedding test passed")


@skip_if_no_openai
def test_embedding_with_model():
    """Test embedding generation with explicit model specification."""
    from polar_llama import embedding_async, Provider

    df = pl.DataFrame({
        "text": ["Test embedding"]
    })

    # Generate embeddings with specific model
    result = df.with_columns(
        embeddings=embedding_async(
            pl.col("text"),
            provider=Provider.OPENAI,
            model="text-embedding-3-small"
        )
    )

    assert result["embeddings"][0] is not None
    assert len(result["embeddings"][0]) == 1536

    print("✓ Embedding with model test passed")


@skip_if_no_openai
def test_parallel_embedding_performance():
    """Test parallel embedding generation for multiple texts."""
    from polar_llama import embedding_async
    import time

    # Create a dataframe with multiple texts
    texts = [
        "The quick brown fox",
        "Machine learning models",
        "Natural language processing",
        "Deep neural networks",
        "Transformer architecture"
    ]

    df = pl.DataFrame({"text": texts})

    # Measure time for parallel processing
    start = time.time()
    result = df.with_columns(
        embeddings=embedding_async(pl.col("text"))
    )
    duration = time.time() - start

    # Verify all embeddings were generated
    assert result.height == len(texts)
    for i in range(len(texts)):
        assert result["embeddings"][i] is not None
        assert len(result["embeddings"][i]) == 1536

    # Parallel processing should be reasonably fast
    # (not a strict requirement, just informational)
    print(f"✓ Parallel embedding test passed in {duration:.2f}s")


@skip_if_no_openai
def test_embedding_with_nulls():
    """Test handling of null values in input."""
    from polar_llama import embedding_async

    df = pl.DataFrame({
        "text": ["Hello world", None, "Test text"]
    })

    result = df.with_columns(
        embeddings=embedding_async(pl.col("text"))
    )

    # Verify structure
    assert result.height == 3
    assert result["embeddings"][0] is not None
    assert result["embeddings"][1] is None  # Null input should produce null output
    assert result["embeddings"][2] is not None

    print("✓ Null handling test passed")


@skip_if_no_openai
def test_embedding_namespace():
    """Test embedding generation using the .llama namespace."""
    from polar_llama import embedding_async

    df = pl.DataFrame({
        "text": ["Namespace test"]
    })

    # Test using the .llama namespace
    result = df.with_columns(
        embeddings=pl.col("text").llama.embedding()
    )

    assert result["embeddings"][0] is not None
    assert len(result["embeddings"][0]) == 1536

    print("✓ Namespace test passed")


def test_empty_dataframe():
    """Test handling of empty dataframes."""
    from polar_llama import embedding_async, Provider

    df = pl.DataFrame({
        "text": []
    }, schema={"text": pl.Utf8})

    result = df.with_columns(
        embeddings=embedding_async(pl.col("text"), provider=Provider.OPENAI)
    )

    assert result.height == 0
    assert result["embeddings"].dtype == pl.List(pl.Float64)

    print("✓ Empty dataframe test passed")


@skip_if_no_openai
def test_embedding_dimensions():
    """Test that embedding dimensions can be accessed."""
    from polar_llama import embedding_async

    df = pl.DataFrame({
        "text": ["Dimension test"]
    })

    result = df.with_columns(
        embeddings=embedding_async(pl.col("text"))
    ).with_columns(
        dimensions=pl.col("embeddings").list.len()
    )

    assert result["dimensions"][0] == 1536

    print("✓ Embedding dimensions test passed")


if __name__ == "__main__":
    # Run tests
    print("\n=== Running Embedding Tests ===\n")

    # Run basic tests
    test_empty_dataframe()

    # Run OpenAI tests if key is available
    if os.environ.get("OPENAI_API_KEY"):
        test_basic_embedding()
        test_embedding_with_model()
        test_embedding_with_nulls()
        test_embedding_namespace()
        test_embedding_dimensions()
        test_parallel_embedding_performance()
        print("\n✓ All embedding tests passed!")
    else:
        print("\n⚠ Skipping OpenAI tests (no API key)")
