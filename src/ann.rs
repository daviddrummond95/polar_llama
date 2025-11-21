use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};

/// Wrapper for embedding vectors that implements the Point trait
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingPoint(pub Vec<f64>);

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Calculate cosine distance (1 - cosine similarity)
        // This is a common metric for embeddings
        let dot_product: f64 = self.0.iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f64 = self.0.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = other.0.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }

        let cosine_similarity = dot_product / (norm1 * norm2);
        (1.0 - cosine_similarity) as f32
    }
}

/// Build an HNSW index from a collection of embedding vectors
pub fn build_hnsw_index(embeddings: Vec<Vec<f64>>) -> HnswMap<EmbeddingPoint, usize> {
    let points: Vec<EmbeddingPoint> = embeddings
        .into_iter()
        .map(EmbeddingPoint)
        .collect();

    let values: Vec<usize> = (0..points.len()).collect();

    Builder::default().build(points, values)
}

/// Search for k nearest neighbors in the HNSW index
pub fn search_hnsw(
    index: &HnswMap<EmbeddingPoint, usize>,
    query: &[f64],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_point = EmbeddingPoint(query.to_vec());
    let mut search = Search::default();

    index
        .search(&query_point, &mut search)
        .take(k)
        .map(|item| (*item.value, item.distance))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_point_distance() {
        let p1 = EmbeddingPoint(vec![1.0, 0.0, 0.0]);
        let p2 = EmbeddingPoint(vec![0.0, 1.0, 0.0]);
        let p3 = EmbeddingPoint(vec![1.0, 0.0, 0.0]);

        // Orthogonal vectors should have distance 1.0 (cosine similarity 0)
        assert!((p1.distance(&p2) - 1.0).abs() < 1e-6);

        // Identical vectors should have distance 0.0 (cosine similarity 1)
        assert!((p1.distance(&p3) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_and_search_hnsw() {
        // Create some simple embeddings
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0],
        ];

        let index = build_hnsw_index(embeddings);

        // Query for nearest neighbors of [1.0, 0.0, 0.0]
        let query = vec![1.0, 0.0, 0.0];
        let results = search_hnsw(&index, &query, 2);

        // Should find index 0 first (exact match), then index 3 (similar)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Index of exact match
        assert!(results[0].1 < 1e-6); // Distance should be very small
    }
}
