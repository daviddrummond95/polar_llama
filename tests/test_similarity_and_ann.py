#!/usr/bin/env python3
"""
Test suite for similarity and approximate nearest neighbor operations in polar-llama.
"""

import polars as pl
import pytest
import math


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    from polar_llama import cosine_similarity

    # Create test vectors
    df = pl.DataFrame({
        "vec1": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "vec2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 4.0, 6.0]],
    })

    result = df.with_columns(
        similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
    )

    # Test identical vectors (similarity should be 1.0)
    assert abs(result["similarity"][0] - 1.0) < 1e-6, f"Expected 1.0, got {result['similarity'][0]}"

    # Test orthogonal vectors (similarity should be 0.0)
    # [1,0,0] and [0,1,0] are orthogonal
    assert abs(result["similarity"][1] - 0.0) < 1e-6, f"Expected 0.0, got {result['similarity'][1]}"

    # Test parallel vectors (similarity should be 1.0)
    # [1,2,3] and [2,4,6] are parallel (one is 2x the other)
    assert abs(result["similarity"][2] - 1.0) < 1e-6, f"Expected 1.0, got {result['similarity'][2]}"

    print("✓ Cosine similarity test passed")


def test_dot_product():
    """Test dot product calculation."""
    from polar_llama import dot_product

    df = pl.DataFrame({
        "vec1": [[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]],
        "vec2": [[4.0, 5.0, 6.0], [0.0, 1.0, 0.0]],
    })

    result = df.with_columns(
        dot_prod=dot_product(pl.col("vec1"), pl.col("vec2"))
    )

    # Test: [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert abs(result["dot_prod"][0] - 32.0) < 1e-6, f"Expected 32.0, got {result['dot_prod'][0]}"

    # Test: [1,0,0] · [0,1,0] = 0
    assert abs(result["dot_prod"][1] - 0.0) < 1e-6, f"Expected 0.0, got {result['dot_prod'][1]}"

    print("✓ Dot product test passed")


def test_euclidean_distance():
    """Test Euclidean distance calculation."""
    from polar_llama import euclidean_distance

    df = pl.DataFrame({
        "vec1": [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        "vec2": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
    })

    result = df.with_columns(
        distance=euclidean_distance(pl.col("vec1"), pl.col("vec2"))
    )

    # Test: distance from origin to (1,1,1) = sqrt(3) ≈ 1.732
    expected_dist1 = math.sqrt(3.0)
    assert abs(result["distance"][0] - expected_dist1) < 1e-6, \
        f"Expected {expected_dist1}, got {result['distance'][0]}"

    # Test: distance from (3,4,0) to origin = sqrt(9+16) = 5
    assert abs(result["distance"][1] - 5.0) < 1e-6, f"Expected 5.0, got {result['distance'][1]}"

    print("✓ Euclidean distance test passed")


def test_similarity_with_nulls():
    """Test similarity operations with null values."""
    from polar_llama import cosine_similarity, dot_product, euclidean_distance

    df = pl.DataFrame({
        "vec1": [[1.0, 0.0, 0.0], None, [1.0, 1.0, 0.0]],
        "vec2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], None],
    })

    result = df.with_columns(
        cos_sim=cosine_similarity(pl.col("vec1"), pl.col("vec2")),
        dot_prod=dot_product(pl.col("vec1"), pl.col("vec2")),
        distance=euclidean_distance(pl.col("vec1"), pl.col("vec2"))
    )

    # First row should have values
    assert result["cos_sim"][0] is not None
    assert result["dot_prod"][0] is not None
    assert result["distance"][0] is not None

    # Second row should be null (vec1 is null)
    assert result["cos_sim"][1] is None
    assert result["dot_prod"][1] is None
    assert result["distance"][1] is None

    # Third row should be null (vec2 is null)
    assert result["cos_sim"][2] is None
    assert result["dot_prod"][2] is None
    assert result["distance"][2] is None

    print("✓ Null handling test passed")


def test_knn_hnsw_basic():
    """Test basic HNSW k-nearest neighbors search."""
    from polar_llama import knn_hnsw

    # Create a corpus of reference vectors
    corpus = [
        [1.0, 0.0, 0.0],  # Index 0: exact match
        [0.9, 0.1, 0.0],  # Index 1: close match
        [0.0, 1.0, 0.0],  # Index 2: orthogonal
        [0.0, 0.0, 1.0],  # Index 3: orthogonal
    ]

    # Create two queries - each will search the same corpus
    df = pl.DataFrame({
        "query_emb": [
            [1.0, 0.0, 0.0],  # Should find indices 0, 1 as nearest
            [0.0, 1.0, 0.0],  # Should find index 2 as nearest
        ],
        "corpus_emb": [corpus, corpus]  # Same corpus for both queries
    })

    result = df.with_columns(
        neighbors=knn_hnsw(pl.col("query_emb"), pl.col("corpus_emb"), k=2)
    )

    # First query: nearest neighbors should be indices 0 and 1
    neighbors_0 = result["neighbors"][0]
    assert len(neighbors_0) == 2, f"Expected 2 neighbors, got {len(neighbors_0)}"
    assert 0 in neighbors_0, f"Expected index 0 in neighbors, got {neighbors_0}"

    # Second query: nearest neighbor should include index 2
    neighbors_1 = result["neighbors"][1]
    assert len(neighbors_1) == 2, f"Expected 2 neighbors, got {len(neighbors_1)}"
    assert 2 in neighbors_1, f"Expected index 2 in neighbors, got {neighbors_1}"

    print("✓ HNSW k-NN basic test passed")


def test_namespace_similarity_functions():
    """Test similarity functions using the .llama namespace."""
    from polar_llama import cosine_similarity, dot_product, euclidean_distance

    df = pl.DataFrame({
        "vec1": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        "vec2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    })

    # Test using namespace
    result = df.with_columns(
        cos_sim_ns=pl.col("vec1").llama.cosine_similarity(pl.col("vec2")),
        dot_prod_ns=pl.col("vec1").llama.dot_product(pl.col("vec2")),
        distance_ns=pl.col("vec1").llama.euclidean_distance(pl.col("vec2"))
    )

    # Test using direct functions
    result_direct = df.with_columns(
        cos_sim=cosine_similarity(pl.col("vec1"), pl.col("vec2")),
        dot_prod=dot_product(pl.col("vec1"), pl.col("vec2")),
        distance=euclidean_distance(pl.col("vec1"), pl.col("vec2"))
    )

    # Results should be the same
    assert result["cos_sim_ns"][0] == result_direct["cos_sim"][0]
    assert result["dot_prod_ns"][0] == result_direct["dot_prod"][0]
    assert result["distance_ns"][0] == result_direct["distance"][0]

    print("✓ Namespace similarity functions test passed")


def test_similarity_vector_length_mismatch():
    """Test error handling for vectors of different lengths."""
    from polar_llama import cosine_similarity

    df = pl.DataFrame({
        "vec1": [[1.0, 0.0, 0.0]],
        "vec2": [[1.0, 0.0]],  # Different length
    })

    # This should raise an error
    try:
        result = df.with_columns(
            similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
        )
        # If we get here without an exception, the test should fail
        assert False, "Expected an error for mismatched vector lengths"
    except Exception as e:
        # Expected to fail
        assert "same length" in str(e).lower() or "compute" in str(e).lower()
        print("✓ Vector length mismatch error handling test passed")


if __name__ == "__main__":
    # Run tests
    print("\n=== Running Similarity and ANN Tests ===\n")

    test_cosine_similarity()
    test_dot_product()
    test_euclidean_distance()
    test_similarity_with_nulls()
    test_knn_hnsw_basic()
    test_namespace_similarity_functions()
    test_similarity_vector_length_mismatch()

    print("\n✓ All similarity and ANN tests passed!")
