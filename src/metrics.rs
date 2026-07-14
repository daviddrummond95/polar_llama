//! Inter-rater reliability metrics (issue #79): Cohen's kappa and
//! Krippendorff's alpha, plus a shared case-resampling bootstrap for
//! percentile confidence intervals.
//!
//! These are pure functions over plain Rust vectors -- no Polars types --
//! so they are trivially unit-testable here and reused by the plugin
//! expressions in `src/expressions.rs`, which handle the Polars `Series` /
//! null / label-encoding plumbing around them.
//!
//! Numeric conventions are pinned to match two reference implementations
//! bit-for-bit (verified against `sklearn.metrics.cohen_kappa_score` and the
//! `krippendorff` PyPI package on the fixtures in the `tests` module below):
//!
//! - Cohen's kappa: sklearn's weight-matrix formulation, including its
//!   `NaN` (not null/error) result when the expected-by-chance agreement is
//!   zero (e.g. a single observed category).
//! - Krippendorff's alpha: the coincidence-matrix construction, including
//!   the "ordinal" difference function using coincidence-matrix marginals
//!   (not raw category frequencies -- the classic implementation mistake).
//!
//! Both reference implementations *raise* on certain degenerate inputs
//! (single-category domain, no pairable units). A Polars expression must
//! not panic per-group, so this module documents and returns a value
//! instead in each case -- see the doc comments on `krippendorff_alpha` for
//! the exact divergence.

use crate::kmeans::SplitMix64;

// ============================================================================
// Cohen's kappa
// ============================================================================

/// Weighting scheme for Cohen's kappa, applied over sorted-label *indices*
/// (sklearn's convention -- not the label values themselves).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaWeights {
    None,
    Linear,
    Quadratic,
}

fn kappa_weight(weights: KappaWeights, i: usize, j: usize) -> f64 {
    match weights {
        KappaWeights::None => {
            if i == j {
                0.0
            } else {
                1.0
            }
        }
        KappaWeights::Linear => (i as f64 - j as f64).abs(),
        KappaWeights::Quadratic => {
            let d = i as f64 - j as f64;
            d * d
        }
    }
}

/// Cohen's kappa between two equal-length label-*index* sequences (indices
/// into a shared, sorted label list of size `n_labels` -- pre-encoded by the
/// caller; pairwise-incomplete rows must already be dropped).
///
/// Matches `sklearn.metrics.cohen_kappa_score` bit-for-bit: builds the
/// confusion matrix, weights it (see `KappaWeights`), and computes
/// `1 - sum(w*C) / sum(w*E)`. When the expected-agreement denominator is
/// zero (e.g. every rating in both columns is the same single category),
/// sklearn returns `nan` for the resulting `0/0`; this returns `f64::NAN`
/// too (not an `Option::None`) so the caller can propagate it into a
/// Polars-null-free `Float64` the same way sklearn does into a numpy float.
pub fn cohens_kappa(a: &[usize], b: &[usize], n_labels: usize, weights: KappaWeights) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 || n_labels == 0 {
        return f64::NAN;
    }

    let mut confusion = vec![0f64; n_labels * n_labels];
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        confusion[ai * n_labels + bi] += 1.0;
    }
    let n_f = n as f64;

    let mut row_sum = vec![0f64; n_labels];
    let mut col_sum = vec![0f64; n_labels];
    for i in 0..n_labels {
        for j in 0..n_labels {
            let c = confusion[i * n_labels + j];
            row_sum[i] += c;
            col_sum[j] += c;
        }
    }

    let mut weighted_observed = 0f64;
    let mut weighted_expected = 0f64;
    for i in 0..n_labels {
        for j in 0..n_labels {
            let w = kappa_weight(weights, i, j);
            if w == 0.0 {
                continue;
            }
            weighted_observed += w * confusion[i * n_labels + j];
            let expected = row_sum[i] * col_sum[j] / n_f;
            weighted_expected += w * expected;
        }
    }

    if weighted_expected == 0.0 {
        return f64::NAN;
    }

    1.0 - weighted_observed / weighted_expected
}

// ============================================================================
// Krippendorff's alpha
// ============================================================================

/// Level of measurement, governing the difference function `delta^2(c, k)`
/// used by the coincidence-matrix formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaLevel {
    Nominal,
    Ordinal,
    Interval,
    Ratio,
}

/// Krippendorff's alpha over `units`: each element is one unit's non-null
/// ratings (already filtered of nulls by the caller). For `Nominal`, values
/// are pre-encoded category ids as `f64`; for `Ordinal`/`Interval`/`Ratio`
/// they are the numeric ratings themselves.
///
/// Implements the coincidence-matrix construction: units with fewer than 2
/// ratings are excluded (they contribute no pairs), a coincidence matrix
/// `o[c][k]` is built by distributing `1 / (m_u - 1)` over every ordered
/// pair of a unit's rating *slots* (not distinct values -- two raters both
/// giving category `c` still contributes `o[c][c]`), and
/// `alpha = 1 - (n-1) * sum(o * delta2) / sum(n_marg (x) n_marg * delta2)`.
///
/// Two conventions diverge from the reference `krippendorff` PyPI package,
/// which raises `ValueError` in both cases (an expression must not panic
/// per-group):
/// - No unit has >= 2 ratings (nothing pairable): returns `None` (null),
///   where the package raises.
/// - The observed domain has a single category (every rating, across every
///   included unit, is identical) -- so `delta2` is 0 for every pair and the
///   expected-disagreement denominator is 0: returns `Some(1.0)` ("trivially
///   perfect agreement") by documented convention, where the package raises
///   `ValueError("value in domain")`.
pub fn krippendorff_alpha(units: &[Vec<f64>], level: AlphaLevel) -> Option<f64> {
    let included: Vec<&Vec<f64>> = units.iter().filter(|u| u.len() >= 2).collect();
    if included.is_empty() {
        return None;
    }

    // Sorted distinct observed values over included units -- the category
    // domain. Numeric sort (not NaN-safe by design: ratings are expected to
    // be finite; callers must filter NaN/inf upstream same as nulls).
    let mut domain: Vec<f64> = included.iter().flat_map(|u| u.iter().copied()).collect();
    // total_cmp is a total order over all f64 (incl. any NaN that slipped past
    // the upstream finiteness filter), so the sort/search can never panic.
    domain.sort_by(|a, b| a.total_cmp(b));
    domain.dedup();
    let n_cat = domain.len();

    let index_of = |v: f64| -> usize {
        domain
            .binary_search_by(|x| x.total_cmp(&v))
            .expect("value must be in domain by construction")
    };

    let mut coincidence = vec![0f64; n_cat * n_cat];
    for unit in &included {
        let m = unit.len();
        let denom = (m - 1) as f64;
        for i in 0..m {
            let ci = index_of(unit[i]);
            for j in 0..m {
                if i == j {
                    continue;
                }
                let cj = index_of(unit[j]);
                coincidence[ci * n_cat + cj] += 1.0 / denom;
            }
        }
    }

    let mut marginal = vec![0f64; n_cat];
    for c in 0..n_cat {
        for k in 0..n_cat {
            marginal[c] += coincidence[c * n_cat + k];
        }
    }
    let n_total: f64 = marginal.iter().sum();

    let delta2 = |c: usize, k: usize| -> f64 {
        match level {
            AlphaLevel::Nominal => {
                if c == k {
                    0.0
                } else {
                    1.0
                }
            }
            AlphaLevel::Ordinal => {
                // Sum of coincidence-matrix marginals over categories from
                // c to k inclusive, in sorted domain order -- not raw
                // frequencies. Using raw counts here is the classic
                // implementation mistake this fixture-tests against.
                let (lo, hi) = if c <= k { (c, k) } else { (k, c) };
                let span: f64 = marginal[lo..=hi].iter().sum();
                let term = span - (marginal[c] + marginal[k]) / 2.0;
                term * term
            }
            AlphaLevel::Interval => {
                let d = domain[c] - domain[k];
                d * d
            }
            AlphaLevel::Ratio => {
                let s = domain[c] + domain[k];
                if s == 0.0 {
                    0.0
                } else {
                    let d = (domain[c] - domain[k]) / s;
                    d * d
                }
            }
        }
    };

    let mut observed_disagreement = 0f64;
    let mut expected_disagreement = 0f64;
    for c in 0..n_cat {
        for k in 0..n_cat {
            let d2 = delta2(c, k);
            if d2 == 0.0 {
                continue;
            }
            observed_disagreement += coincidence[c * n_cat + k] * d2;
            expected_disagreement += marginal[c] * marginal[k] * d2;
        }
    }

    if expected_disagreement == 0.0 {
        // Single-category domain (or a level whose delta2 vanishes for
        // every pair present): agreement is trivially perfect by
        // convention. See the doc comment above.
        return Some(1.0);
    }

    Some(1.0 - (n_total - 1.0) * observed_disagreement / expected_disagreement)
}

// ============================================================================
// Bootstrap confidence intervals
// ============================================================================

/// Linear-interpolation quantile (numpy's default, "type 7"), over an
/// already-sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Nonparametric case (unit/row) bootstrap: resample `n_units` indices with
/// replacement `b` times, call `estimate` on each resample (indices into
/// whatever unit population the caller cares about -- pairwise-complete row
/// pairs for kappa, unit rows for alpha), and return a percentile CI at
/// `((1-ci)/2, 1-(1-ci)/2)` using linear-interpolation quantiles.
///
/// `estimate` may return `NaN` for a degenerate resample (e.g. a resample
/// that collapses to a single category); such resamples are dropped from
/// the percentile computation. If more than half of the `b` resamples are
/// degenerate, this returns `None` (the caller should surface null CI
/// fields, keeping the point estimate on the original data unaffected).
pub fn bootstrap_ci<F>(estimate: F, n_units: usize, b: usize, ci: f64, seed: u64) -> Option<(f64, f64)>
where
    F: Fn(&[usize]) -> f64,
{
    if n_units == 0 || b == 0 {
        return None;
    }

    let mut rng = SplitMix64::new(seed);
    let mut values: Vec<f64> = Vec::with_capacity(b);
    for _ in 0..b {
        let mut idx = Vec::with_capacity(n_units);
        for _ in 0..n_units {
            idx.push(rng.next_below(n_units));
        }
        let v = estimate(&idx);
        if v.is_finite() {
            values.push(v);
        }
    }

    if values.len() * 2 < b {
        return None;
    }

    values.sort_by(|a, b2| a.partial_cmp(b2).expect("non-finite bootstrap estimate"));
    let lo_q = (1.0 - ci) / 2.0;
    let hi_q = 1.0 - lo_q;
    Some((quantile_sorted(&values, lo_q), quantile_sorted(&values, hi_q)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    // ------------------------------------------------------------------
    // Cohen's kappa
    // ------------------------------------------------------------------

    #[test]
    fn test_kappa_2x2_exact() {
        // Confusion matrix [[20,5],[10,15]], n=50: p_o=0.7, p_e=0.5, kappa=0.4.
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..20 {
            a.push(0);
            b.push(0);
        }
        for _ in 0..5 {
            a.push(0);
            b.push(1);
        }
        for _ in 0..10 {
            a.push(1);
            b.push(0);
        }
        for _ in 0..15 {
            a.push(1);
            b.push(1);
        }
        let k = cohens_kappa(&a, &b, 2, KappaWeights::None);
        assert_close(k, 0.4, 1e-12);
    }

    #[test]
    fn test_kappa_weighted_fixture() {
        // y1=[0,0,0,1,1,1,2,2,2,1], y2=[0,0,1,1,1,2,2,2,2,0] -- confirmed
        // against sklearn.metrics.cohen_kappa_score.
        let a = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 1];
        let b = vec![0, 0, 1, 1, 1, 2, 2, 2, 2, 0];
        assert_close(
            cohens_kappa(&a, &b, 3, KappaWeights::None),
            0.5522388059701492,
            1e-9,
        );
        assert_close(
            cohens_kappa(&a, &b, 3, KappaWeights::Linear),
            0.6590909090909092,
            1e-9,
        );
        assert_close(
            cohens_kappa(&a, &b, 3, KappaWeights::Quadratic),
            0.7692307692307692,
            1e-9,
        );
    }

    #[test]
    fn test_kappa_single_category_is_nan() {
        let a = vec![0, 0, 0];
        let b = vec![0, 0, 0];
        let k = cohens_kappa(&a, &b, 1, KappaWeights::None);
        assert!(k.is_nan());
    }

    #[test]
    fn test_kappa_perfect_agreement() {
        let a = vec![0, 1, 0, 1];
        let b = vec![0, 1, 0, 1];
        assert_close(cohens_kappa(&a, &b, 2, KappaWeights::None), 1.0, 1e-12);
    }

    #[test]
    fn test_kappa_systematic_disagreement() {
        // labels [1,2] -> indices [0,1]; a=[0,1,0,1], b=[1,0,1,0]
        let a = vec![0, 1, 0, 1];
        let b = vec![1, 0, 1, 0];
        assert_close(cohens_kappa(&a, &b, 2, KappaWeights::None), -1.0, 1e-12);
    }

    // ------------------------------------------------------------------
    // Krippendorff's alpha
    // ------------------------------------------------------------------

    /// Krippendorff's canonical reliability-data example (2004/2011),
    /// 4 observers x 12 units; `None` = missing. Unit 12 (index 11) has
    /// only one rating and is excluded.
    fn fixture_a() -> Vec<Vec<f64>> {
        let raw: [[Option<f64>; 12]; 4] = [
            [
                Some(1.0), Some(2.0), Some(3.0), Some(3.0), Some(2.0), Some(1.0),
                Some(4.0), Some(1.0), Some(2.0), None, None, None,
            ],
            [
                Some(1.0), Some(2.0), Some(3.0), Some(3.0), Some(2.0), Some(2.0),
                Some(4.0), Some(1.0), Some(2.0), Some(5.0), None, Some(3.0),
            ],
            [
                None, Some(3.0), Some(3.0), Some(3.0), Some(2.0), Some(3.0),
                Some(4.0), Some(2.0), Some(2.0), Some(5.0), Some(1.0), None,
            ],
            [
                Some(1.0), Some(2.0), Some(3.0), Some(3.0), Some(2.0), Some(4.0),
                Some(4.0), Some(1.0), Some(2.0), Some(5.0), Some(1.0), None,
            ],
        ];
        let mut units = Vec::new();
        for col in 0..12 {
            let mut ratings = Vec::new();
            for row in raw.iter() {
                if let Some(v) = row[col] {
                    ratings.push(v);
                }
            }
            units.push(ratings);
        }
        units
    }

    /// The `krippendorff` PyPI README example, 3 coders x 15 units.
    fn fixture_b() -> Vec<Vec<f64>> {
        let raw: [[Option<f64>; 15]; 3] = [
            [
                None, None, None, None, None, Some(3.0), Some(4.0), Some(1.0),
                Some(2.0), Some(1.0), Some(1.0), Some(3.0), Some(3.0), None, Some(3.0),
            ],
            [
                Some(1.0), None, Some(2.0), Some(1.0), Some(3.0), Some(3.0), Some(4.0),
                Some(3.0), None, None, None, None, None, None, None,
            ],
            [
                None, None, Some(2.0), Some(1.0), Some(3.0), Some(4.0), Some(4.0),
                None, Some(2.0), Some(1.0), Some(1.0), Some(3.0), Some(3.0), None, Some(4.0),
            ],
        ];
        let mut units = Vec::new();
        for col in 0..15 {
            let mut ratings = Vec::new();
            for row in raw.iter() {
                if let Some(v) = row[col] {
                    ratings.push(v);
                }
            }
            units.push(ratings);
        }
        units
    }

    #[test]
    fn test_alpha_fixture_a_all_levels() {
        let units = fixture_a();
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap(),
            0.743421052631579,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Ordinal).unwrap(),
            0.8153875037548814,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Interval).unwrap(),
            0.8491071428571428,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Ratio).unwrap(),
            0.7974027747116121,
            1e-9,
        );
    }

    #[test]
    fn test_alpha_fixture_b_all_levels() {
        let units = fixture_b();
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap(),
            0.691358024691358,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Ordinal).unwrap(),
            0.8067214199413153,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Interval).unwrap(),
            0.8108448928121059,
            1e-9,
        );
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Ratio).unwrap(),
            0.8089436707842471,
            1e-9,
        );
    }

    #[test]
    fn test_alpha_perfect_agreement() {
        let units = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![1.0, 1.0, 1.0]];
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap(),
            1.0,
            1e-12,
        );
    }

    #[test]
    fn test_alpha_systematic_disagreement() {
        // [[1,2,1,2],[2,1,2,1]] nominal -> -0.75 (confirmed against the
        // krippendorff package).
        let units = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
        ];
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap(),
            -0.75,
            1e-9,
        );
    }

    #[test]
    fn test_alpha_single_category_is_one() {
        let units = vec![vec![1.0, 1.0], vec![1.0, 1.0, 1.0]];
        assert_close(
            krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap(),
            1.0,
            1e-12,
        );
    }

    #[test]
    fn test_alpha_no_pairable_units_is_none() {
        let units = vec![vec![1.0], vec![2.0]];
        assert_eq!(krippendorff_alpha(&units, AlphaLevel::Nominal), None);
    }

    #[test]
    fn test_alpha_empty_is_none() {
        let units: Vec<Vec<f64>> = vec![];
        assert_eq!(krippendorff_alpha(&units, AlphaLevel::Nominal), None);
    }

    // ------------------------------------------------------------------
    // Bootstrap
    // ------------------------------------------------------------------

    #[test]
    fn test_bootstrap_ci_deterministic_for_fixed_seed() {
        let units = fixture_a();
        let n = units.len();
        let estimate = |idx: &[usize]| {
            let resampled: Vec<Vec<f64>> = idx.iter().map(|&i| units[i].clone()).collect();
            krippendorff_alpha(&resampled, AlphaLevel::Nominal).unwrap_or(f64::NAN)
        };
        let ci1 = bootstrap_ci(estimate, n, 500, 0.95, 42).unwrap();
        let ci2 = bootstrap_ci(estimate, n, 500, 0.95, 42).unwrap();
        assert_eq!(ci1, ci2);
        assert!(ci1.0 <= ci1.1);
    }

    #[test]
    fn test_bootstrap_ci_brackets_point_estimate() {
        let units = fixture_a();
        let n = units.len();
        let point = krippendorff_alpha(&units, AlphaLevel::Nominal).unwrap();
        let estimate = |idx: &[usize]| {
            let resampled: Vec<Vec<f64>> = idx.iter().map(|&i| units[i].clone()).collect();
            krippendorff_alpha(&resampled, AlphaLevel::Nominal).unwrap_or(f64::NAN)
        };
        let (lo, hi) = bootstrap_ci(estimate, n, 1000, 0.95, 0).unwrap();
        assert!(lo <= point + 1e-6);
        assert!(hi >= point - 1e-6);
    }

    #[test]
    fn test_bootstrap_ci_empty_is_none() {
        let estimate = |_idx: &[usize]| 0.0;
        assert_eq!(bootstrap_ci(estimate, 0, 100, 0.95, 0), None);
        assert_eq!(bootstrap_ci(estimate, 5, 0, 0.95, 0), None);
    }
}
