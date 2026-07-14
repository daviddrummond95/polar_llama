//! Hand-rolled k-means++ / Lloyd's-algorithm clustering for embedding
//! vectors, plus a sampled-silhouette heuristic for automatic k selection.
//!
//! This module is intentionally dependency-free (no `linfa`, `ndarray`,
//! `rand`, or any BLAS/numpy-style crate) -- see issue #78 / the "ZERO new
//! dependencies" constraint in `docs/CODEBOOK_INDUCTION.md`. Randomness comes
//! from a hand-rolled `SplitMix64` PRNG (a tiny, well-known, public-domain
//! generator by Sebastiano Vigna), which is more than sufficient for k-means
//! seeding -- it does not need to be cryptographically secure, only fast and
//! reproducible for a given `seed`.
//!
//! Distance metric: cosine distance (`1 - cosine_similarity`, clamped to
//! `[0, 2]`), matching the metric already used for embeddings elsewhere in
//! this crate (`src/ann.rs::EmbeddingPoint::distance`). Centroids are
//! recomputed as the arithmetic mean of their assigned points and then
//! re-normalized to unit length ("spherical k-means"), which keeps the
//! centroid comparable to its members under cosine distance.
//!
//! Consumed by `cluster_embeddings` (`src/expressions.rs`), the whole-column
//! Polars plugin expression that backs `polar_llama.cluster_embeddings` /
//! `polar_llama.induce_codebook`.

/// Minimal, fast, deterministic PRNG (`splitmix64`, public domain, Vigna).
/// Not cryptographically secure -- used only to seed k-means++ and to build
/// the silhouette subsample, both of which only need reproducible,
/// reasonably-well-distributed randomness.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform usize in `[0, n)`. Panics if `n == 0`.
    pub fn next_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "next_below requires n > 0");
        (self.next_u64() % n as u64) as usize
    }
}

/// Cosine distance between two equal-length vectors, in `[0, 2]`. Zero
/// vectors are treated as maximally distant from everything (matching
/// `src/ann.rs::EmbeddingPoint::distance`'s convention).
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let cos = (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0);
    (1.0 - cos).clamp(0.0, 2.0)
}

fn mean_vector(points: &[&Vec<f64>]) -> Vec<f64> {
    let dim = points[0].len();
    let mut sum = vec![0.0f64; dim];
    for p in points {
        for (s, v) in sum.iter_mut().zip(p.iter()) {
            *s += v;
        }
    }
    let n = points.len() as f64;
    for s in sum.iter_mut() {
        *s /= n;
    }
    sum
}

fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Result of one k-means run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster label (`0..k`) assigned to each input point, in input order.
    pub labels: Vec<usize>,
    /// Final centroids, one per cluster.
    pub centroids: Vec<Vec<f64>>,
    /// Cosine distance from each point to its assigned centroid, in input
    /// order (parallel to `labels`).
    pub distances: Vec<f64>,
    /// Sum of `distances` -- the objective Lloyd's algorithm minimizes.
    pub inertia: f64,
}

/// k-means++ seeding: pick the first centroid uniformly at random, then
/// repeatedly pick the next centroid with probability proportional to its
/// squared distance from the nearest already-chosen centroid. This spreads
/// the initial centroids out, which is what makes k-means++ converge faster
/// and more reliably than picking `k` centroids uniformly at random.
fn kmeans_plusplus_init(points: &[Vec<f64>], k: usize, rng: &mut SplitMix64) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = rng.next_below(n);
    centroids.push(points[first].clone());

    let mut nearest_sq: Vec<f64> = points
        .iter()
        .map(|p| {
            let d = cosine_distance(p, &centroids[0]);
            d * d
        })
        .collect();

    while centroids.len() < k {
        let total: f64 = nearest_sq.iter().sum();
        let chosen_idx = if total <= 0.0 {
            // All remaining points are on top of an existing centroid
            // (degenerate/duplicate-heavy input); fall back to uniform pick.
            rng.next_below(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cumulative = 0.0f64;
            let mut idx = n - 1;
            for (i, d2) in nearest_sq.iter().enumerate() {
                cumulative += d2;
                if cumulative >= target {
                    idx = i;
                    break;
                }
            }
            idx
        };

        let new_centroid = points[chosen_idx].clone();
        for (i, p) in points.iter().enumerate() {
            let d = cosine_distance(p, &new_centroid);
            let d2 = d * d;
            if d2 < nearest_sq[i] {
                nearest_sq[i] = d2;
            }
        }
        centroids.push(new_centroid);
    }

    centroids
}

/// Assign every point to its nearest centroid. Returns `(labels, distances)`.
fn assign(points: &[Vec<f64>], centroids: &[Vec<f64>]) -> (Vec<usize>, Vec<f64>) {
    let mut labels = Vec::with_capacity(points.len());
    let mut distances = Vec::with_capacity(points.len());
    for p in points {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (c_idx, c) in centroids.iter().enumerate() {
            let d = cosine_distance(p, c);
            if d < best_dist {
                best_dist = d;
                best_idx = c_idx;
            }
        }
        labels.push(best_idx);
        distances.push(best_dist);
    }
    (labels, distances)
}

/// One Lloyd's-algorithm run from a fixed initial set of centroids.
fn lloyd(points: &[Vec<f64>], initial_centroids: Vec<Vec<f64>>, max_iter: usize) -> KMeansResult {
    let k = initial_centroids.len();
    let mut centroids = initial_centroids;
    let (mut labels, mut distances) = assign(points, &centroids);

    for _ in 0..max_iter {
        // Recompute centroids as the (re-normalized) mean of assigned points.
        let mut buckets: Vec<Vec<&Vec<f64>>> = vec![Vec::new(); k];
        for (p, &label) in points.iter().zip(labels.iter()) {
            buckets[label].push(p);
        }

        let mut new_centroids = Vec::with_capacity(k);
        for bucket in buckets.iter() {
            if bucket.is_empty() {
                // Empty-cluster repair: reseed with the point currently
                // farthest from its own assigned centroid, keeping k
                // centroids alive rather than letting a cluster vanish. Note:
                // on degenerate/duplicate-heavy inputs several empty clusters
                // can reseed to the same or coincident points, so the number
                // of *distinct* labels actually observed may be < k even when
                // n_points >= k. There is always at least one label.
                let farthest = distances
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                new_centroids.push(points[farthest].clone());
            } else {
                let mut c = mean_vector(bucket);
                normalize(&mut c);
                new_centroids.push(c);
            }
        }

        let (new_labels, new_distances) = assign(points, &new_centroids);
        let converged = new_labels == labels;
        centroids = new_centroids;
        labels = new_labels;
        distances = new_distances;
        if converged {
            break;
        }
    }

    let inertia = distances.iter().sum();
    KMeansResult {
        labels,
        centroids,
        distances,
        inertia,
    }
}

/// Run k-means with `n_init` k-means++ restarts, keeping the lowest-inertia
/// result (ties broken by the earliest restart, for determinism). `k` is
/// clamped to `[1, points.len()]`.
pub fn kmeans(
    points: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> KMeansResult {
    let n = points.len();
    assert!(n > 0, "kmeans requires at least one point");
    let k = k.clamp(1, n);

    if k == 1 {
        let mut c = mean_vector(&points.iter().collect::<Vec<_>>());
        normalize(&mut c);
        return lloyd(points, vec![c], max_iter);
    }

    if k == n {
        // Every point is its own cluster: skip the iterative search
        // (k-means++ with k == n_init candidates degenerates to "every
        // point picked exactly once" and Lloyd's algorithm as no-op work).
        let labels: Vec<usize> = (0..n).collect();
        let centroids = points.to_vec();
        let distances = vec![0.0; n];
        return KMeansResult {
            labels,
            centroids,
            distances,
            inertia: 0.0,
        };
    }

    let n_init = n_init.max(1);
    let mut best: Option<KMeansResult> = None;
    for init_idx in 0..n_init {
        let mut rng = SplitMix64::new(seed ^ (init_idx as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let init_centroids = kmeans_plusplus_init(points, k, &mut rng);
        let result = lloyd(points, init_centroids, max_iter);
        match &best {
            Some(b) if b.inertia <= result.inertia => {}
            _ => best = Some(result),
        }
    }
    best.expect("n_init >= 1 guarantees at least one candidate")
}

/// Sampled silhouette score for a clustering: the average, over up to
/// `sample_size` points chosen deterministically (via `seed`), of
/// `(b - a) / max(a, b)` where `a` is the point's mean distance to other
/// points in its own cluster and `b` is the lowest mean distance to any
/// other cluster's points. Ranges over `[-1, 1]`; higher is better
/// (well-separated, internally-tight clusters). Sampling the *outer* loop
/// (not the comparison set) keeps the cost `O(sample_size * n)` regardless
/// of how large `n` is, which is what makes this usable inside `auto_k`'s
/// per-candidate-`k` search.
pub fn sampled_silhouette(
    points: &[Vec<f64>],
    labels: &[usize],
    k: usize,
    sample_size: usize,
    seed: u64,
) -> f64 {
    let n = points.len();
    if k < 2 || k >= n || n == 0 {
        return -1.0;
    }

    let mut cluster_sizes = vec![0usize; k];
    for &l in labels {
        cluster_sizes[l] += 1;
    }

    let sample_size = sample_size.min(n).max(1);
    let mut rng = SplitMix64::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut sample_indices: Vec<usize> = Vec::with_capacity(sample_size);
    if sample_size == n {
        sample_indices.extend(0..n);
    } else {
        // Reservoir-free "sample without replacement" via a Fisher-Yates
        // partial shuffle over a fresh index vector -- simple and exact.
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..sample_size {
            let j = i + rng.next_below(n - i);
            idx.swap(i, j);
        }
        sample_indices.extend_from_slice(&idx[..sample_size]);
    }

    let mut total = 0.0f64;
    let mut counted = 0usize;
    for &i in &sample_indices {
        let own_cluster = labels[i];
        if cluster_sizes[own_cluster] <= 1 {
            // Singleton cluster: silhouette is conventionally 0 for that point.
            counted += 1;
            continue;
        }

        let mut a_sum = 0.0f64;
        let mut a_n = 0usize;
        let mut b_sums = vec![0.0f64; k];
        for (j, p) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = cosine_distance(&points[i], p);
            if labels[j] == own_cluster {
                a_sum += d;
                a_n += 1;
            } else {
                b_sums[labels[j]] += d;
            }
        }
        let a = if a_n > 0 { a_sum / a_n as f64 } else { 0.0 };
        let mut b = f64::INFINITY;
        for (c, &sum) in b_sums.iter().enumerate() {
            if c == own_cluster || cluster_sizes[c] == 0 {
                continue;
            }
            let mean = sum / cluster_sizes[c] as f64;
            if mean < b {
                b = mean;
            }
        }
        if !b.is_finite() {
            continue;
        }
        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        total += s;
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        total / counted as f64
    }
}

/// Search `k` in `[k_min, k_max]`, running `kmeans` at each candidate and
/// scoring it with `sampled_silhouette`; returns the `(k, KMeansResult)`
/// with the highest score (ties broken toward the smaller `k`, for a more
/// parsimonious codebook). Range is clamped to `[1, points.len()]`.
pub fn auto_k(
    points: &[Vec<f64>],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
    silhouette_sample: usize,
) -> (usize, KMeansResult) {
    let n = points.len();
    let k_min = k_min.max(1).min(n);
    let k_max = k_max.max(k_min).min(n);

    let mut best_k = k_min;
    let mut best_result = kmeans(points, k_min, max_iter, n_init, seed);
    let mut best_score =
        sampled_silhouette(points, &best_result.labels, k_min, silhouette_sample, seed);

    for k in (k_min + 1)..=k_max {
        let result = kmeans(points, k, max_iter, n_init, seed);
        let score = sampled_silhouette(points, &result.labels, k, silhouette_sample, seed);
        if score > best_score {
            best_score = score;
            best_k = k;
            best_result = result;
        }
    }

    (best_k, best_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(cx: f64, cy: f64, n: usize, spread: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let dx = (rng.next_f64() - 0.5) * spread;
                let dy = (rng.next_f64() - 0.5) * spread;
                vec![cx + dx, cy + dy]
            })
            .collect()
    }

    #[test]
    fn test_splitmix64_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_splitmix64_f64_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_cosine_distance_basics() {
        assert!((cosine_distance(&[1.0, 0.0], &[1.0, 0.0]) - 0.0).abs() < 1e-9);
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-9);
        assert!((cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-9);
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 0.0]), 1.0);
    }

    #[test]
    fn test_kmeans_recovers_well_separated_clusters() {
        // Three tight, well-separated blobs -- k=3 should perfectly recover
        // the three source clusters (same label within a blob). Centers are
        // placed well away from the origin and at distinct angles from it
        // (0 deg / 90 deg / 180 deg): the clustering metric is cosine
        // distance, which is direction-sensitive and origin-unstable, so a
        // blob centered *at* the origin (or two centers collinear with it in
        // the *same* direction) would not be a meaningful separation test.
        let mut points = Vec::new();
        points.extend(blob(20.0, 0.0, 20, 0.05, 1));
        points.extend(blob(0.0, 20.0, 20, 0.05, 2));
        points.extend(blob(-20.0, 0.0, 20, 0.05, 3));

        let result = kmeans(&points, 3, 100, 8, 123);

        let label_a = result.labels[0];
        let label_b = result.labels[20];
        let label_c = result.labels[40];
        assert_ne!(label_a, label_b);
        assert_ne!(label_a, label_c);
        assert_ne!(label_b, label_c);
        for i in 0..20 {
            assert_eq!(result.labels[i], label_a);
        }
        for i in 20..40 {
            assert_eq!(result.labels[i], label_b);
        }
        for i in 40..60 {
            assert_eq!(result.labels[i], label_c);
        }
    }

    #[test]
    fn test_kmeans_is_deterministic_for_a_fixed_seed() {
        let mut points = Vec::new();
        points.extend(blob(0.0, 0.0, 15, 0.2, 11));
        points.extend(blob(5.0, 5.0, 15, 0.2, 12));

        let r1 = kmeans(&points, 2, 50, 5, 999);
        let r2 = kmeans(&points, 2, 50, 5, 999);
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.inertia - r2.inertia).abs() < 1e-12);
    }

    #[test]
    fn test_kmeans_k_equals_1() {
        let points = blob(0.0, 0.0, 10, 1.0, 5);
        let result = kmeans(&points, 1, 50, 3, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
        assert_eq!(result.centroids.len(), 1);
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        let points = blob(0.0, 0.0, 5, 1.0, 5);
        let result = kmeans(&points, 5, 50, 3, 1);
        let mut labels = result.labels.clone();
        labels.sort_unstable();
        assert_eq!(labels, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_auto_k_recovers_correct_k() {
        // Four blobs at 0/90/180/270 degrees from the origin -- pairwise
        // distinct directions (see the note in
        // `test_kmeans_recovers_well_separated_clusters` on why cosine
        // distance needs centers that aren't collinear with the origin in
        // the same direction).
        let mut points = Vec::new();
        points.extend(blob(20.0, 0.0, 25, 0.05, 21));
        points.extend(blob(0.0, 20.0, 25, 0.05, 22));
        points.extend(blob(-20.0, 0.0, 25, 0.05, 23));
        points.extend(blob(0.0, -20.0, 25, 0.05, 24));

        let (chosen_k, result) = auto_k(&points, 2, 8, 100, 5, 42, 60);
        assert_eq!(chosen_k, 4);
        assert_eq!(result.labels.len(), points.len());
    }

    #[test]
    fn test_sampled_silhouette_prefers_well_separated_clustering() {
        let mut points = Vec::new();
        points.extend(blob(20.0, 0.0, 20, 0.05, 31));
        points.extend(blob(0.0, 20.0, 20, 0.05, 32));

        let good = kmeans(&points, 2, 100, 5, 7);
        let bad_labels: Vec<usize> = (0..points.len()).map(|i| i % 2).collect();

        let good_score = sampled_silhouette(&points, &good.labels, 2, 40, 1);
        let bad_score = sampled_silhouette(&points, &bad_labels, 2, 40, 1);
        assert!(good_score > bad_score);
    }

    #[test]
    fn test_kmeans_handles_duplicate_points_without_panicking() {
        let points = vec![vec![1.0, 0.0]; 10];
        let result = kmeans(&points, 3, 20, 3, 1);
        assert_eq!(result.labels.len(), 10);
    }
}
