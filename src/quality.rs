//! Survey data-quality heuristics (issue #80): straightlining, gibberish,
//! and duplicate-answer scoring.
//!
//! These are pure functions over plain Rust values -- no Polars types --
//! so they are trivially unit-testable here and reused by the plugin
//! expressions in `src/expressions.rs`, which handle the Polars `Series` /
//! null plumbing around them (mirrors `src/metrics.rs`'s split for the
//! inter-rater-reliability metrics, issue #79).
//!
//! Every score is `Option<f64>` in `[0, 1]` when `Some`, higher = more
//! suspicious. `None` means "not enough signal to judge" (too few answers,
//! too little text) -- callers must treat `None` as "do not flag", never as
//! zero. **These functions never drop or reject a row; they only ever
//! return a score (or a null score) for it.**

use std::collections::HashMap;

// ============================================================================
// Straightlining
// ============================================================================

/// Straightlining/flatlining suspicion score for one respondent's grid
/// answers (a Likert-style battery, e.g. 5 columns on a 1-5 scale).
///
/// `row` holds that respondent's non-null numeric answers across the grid
/// columns (arbitrary order -- grid column order is not meaningful, so this
/// looks at value distribution, not run-length). Returns `None` when fewer
/// than `min_answers` answers are present (not enough signal).
///
/// Two components, combined by `max` (either alone is enough to be
/// suspicious):
/// - `mode_frac`: share of answers equal to the most frequent value. A pure
///   straightliner (all identical) scores `1.0` here.
/// - `1 - var_norm`: `var_norm` is the row's population variance normalized
///   by `max_var = ((scale_max - scale_min) / 2)^2` (the maximum possible
///   variance for a value bouncing between the scale's endpoints), clipped
///   to `[0, 1]`. A respondent alternating between endpoints (e.g. 1,5,1,5)
///   has near-maximal variance, so `1 - var_norm` is near 0 -- correctly
///   *not* flagged as straightlining even though it isn't varied in the
///   "many distinct values" sense.
///
/// `scale_min`/`scale_max` are the Likert scale's bounds (e.g. `1.0`/`5.0`);
/// when `scale_max <= scale_min` (a degenerate/unconfigured scale), the
/// variance component is skipped (`var_norm = 0`) and the score falls back
/// to `mode_frac` alone.
pub fn straightline_score(
    row: &[f64],
    scale_min: f64,
    scale_max: f64,
    min_answers: usize,
) -> Option<f64> {
    let m = row.len();
    if m < min_answers.max(1) {
        return None;
    }

    // mode_frac: count of the most frequent value / m. Bucketed on the raw
    // bit pattern -- grid answers are expected to be small discrete codes
    // (Likert points), so exact float equality is the right notion of
    // "the same answer" here.
    let mut counts: HashMap<u64, usize> = HashMap::with_capacity(m);
    for &v in row {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    let mode_frac = max_count as f64 / m as f64;

    let mean: f64 = row.iter().sum::<f64>() / m as f64;
    let variance: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / m as f64;

    let half_range = (scale_max - scale_min) / 2.0;
    let max_var = half_range * half_range;

    let score = if max_var > 0.0 {
        let var_norm = (variance / max_var).clamp(0.0, 1.0);
        mode_frac.max(1.0 - var_norm)
    } else {
        // Degenerate/unconfigured scale (scale_max <= scale_min): the
        // variance component is meaningless (no "maximum possible
        // variance" to normalize against), so fall back to mode_frac alone
        // rather than letting `1 - var_norm` collapse to a vacuous 1.0.
        mode_frac
    };
    Some(score.clamp(0.0, 1.0))
}

// ============================================================================
// Gibberish detection
// ============================================================================

const ENGLISH_VOWEL_RATIO: f64 = 0.38;
const CONSONANT_RUN_CAP: f64 = 6.0;
const VOWELS: [char; 5] = ['a', 'e', 'i', 'o', 'u'];

/// Gibberish/keyboard-mash suspicion score for one open-end answer.
///
/// English-only heuristic (documented limitation): normalizes to lowercase
/// `[a-z ]` only (digits/punctuation/non-Latin scripts are stripped).
/// Returns `None` when fewer than `min_chars` letters survive normalization
/// -- both because there's too little signal to judge, and so that
/// non-Latin-script text (which normalizes to ~nothing) gets a null score
/// rather than a false "gibberish" flag.
///
/// Three components, weighted-summed (weights are a documented, subjective
/// convention -- see `docs/QUALITY_FLAGS.md`):
/// - `consonant_run` (weight 0.5): longest run of consecutive consonants
///   (a run breaks on any vowel or space), normalized by a 6-char cap.
///   Keyboard mash like "asdkjfh" has long consonant runs; English rarely
///   exceeds 3-4.
/// - `vowel_dev` (weight 0.3): relative deviation of the letter-only vowel
///   ratio from English's ~0.38.
/// - `entropy_term` (weight 0.2): normalized Shannon entropy of character
///   bigrams (over the whole normalized string, spaces included). Weak
///   alone (English isn't low-entropy either) but a useful tie-breaker.
pub fn gibberish_score(text: &str, min_chars: usize) -> Option<f64> {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || *c == ' ')
        .collect();

    let letters: Vec<char> = normalized
        .chars()
        .filter(|c| c.is_ascii_lowercase())
        .collect();
    if letters.len() < min_chars.max(1) {
        return None;
    }

    let vowel_count = letters.iter().filter(|c| VOWELS.contains(c)).count();
    let vowel_ratio = vowel_count as f64 / letters.len() as f64;
    let vowel_dev = ((vowel_ratio - ENGLISH_VOWEL_RATIO).abs() / ENGLISH_VOWEL_RATIO).min(1.0);

    let mut max_run = 0usize;
    let mut cur_run = 0usize;
    for c in normalized.chars() {
        if c.is_ascii_lowercase() && !VOWELS.contains(&c) {
            cur_run += 1;
            max_run = max_run.max(cur_run);
        } else {
            cur_run = 0;
        }
    }
    let consonant_run = (max_run as f64 / CONSONANT_RUN_CAP).min(1.0);

    let chars: Vec<char> = normalized.chars().collect();
    let entropy_term = if chars.len() < 2 {
        0.0
    } else {
        // BTreeMap (not HashMap): iteration order must be deterministic --
        // std HashMap randomizes its hasher per-instance, which would make
        // this floating-point sum (and therefore the score) vary in its
        // last bit from call to call on identical input.
        let mut bigram_counts: std::collections::BTreeMap<(char, char), usize> =
            std::collections::BTreeMap::new();
        for w in chars.windows(2) {
            *bigram_counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        let total: usize = bigram_counts.values().sum();
        if total == 0 {
            0.0
        } else {
            let h: f64 = bigram_counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    -p * p.log2()
                })
                .sum();
            let n_distinct = (bigram_counts.len() as f64).max(2.0);
            (h / n_distinct.log2()).min(1.0)
        }
    };

    let score = 0.5 * consonant_run + 0.3 * vowel_dev + 0.2 * entropy_term;
    Some(score.clamp(0.0, 1.0))
}

// ============================================================================
// Duplicate-answer detection
// ============================================================================

/// Normalize an answer for duplicate comparison: lowercase, collapse any
/// run of non-alphanumeric characters (whitespace or punctuation) into a
/// single space, and trim. `"The product is great."` and `"the product is
/// great"` normalize identically.
fn normalize_for_duplicate(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut pending_space = false;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        } else {
            pending_space = true;
        }
    }
    out
}

/// Cross-answer near-duplicate suspicion score for one respondent's set of
/// open-end answers (e.g. several open-ends in the same survey).
///
/// Normalizes each answer (see `normalize_for_duplicate`) and drops any
/// answer shorter than `min_answer_chars` (so legitimate short repeats like
/// "yes"/"n/a" across several open-ends don't trip this). Returns `None`
/// when fewer than 2 answers survive that filter.
///
/// Score is the maximum pairwise token-set Jaccard similarity
/// (`|A ∩ B| / |A ∪ B|`) over every pair of surviving answers -- `1.0` for a
/// verbatim (post-normalization) repeat, lower for partial overlap. One
/// mechanism covers both "exact" (Jaccard == 1.0) and "near" duplicates.
pub fn duplicate_answer_score(answers: &[&str], min_answer_chars: usize) -> Option<f64> {
    let normalized: Vec<String> = answers
        .iter()
        .map(|a| normalize_for_duplicate(a))
        .filter(|a| a.chars().count() >= min_answer_chars)
        .collect();
    if normalized.len() < 2 {
        return None;
    }

    let token_sets: Vec<std::collections::HashSet<&str>> = normalized
        .iter()
        .map(|a| a.split_whitespace().collect())
        .collect();

    let mut max_jaccard = 0.0f64;
    for i in 0..token_sets.len() {
        for j in (i + 1)..token_sets.len() {
            let inter = token_sets[i].intersection(&token_sets[j]).count();
            let union = token_sets[i].union(&token_sets[j]).count();
            let jaccard = if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            };
            if jaccard > max_jaccard {
                max_jaccard = jaccard;
            }
        }
    }
    Some(max_jaccard.clamp(0.0, 1.0))
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
    // straightline_score
    // ------------------------------------------------------------------

    #[test]
    fn test_straightline_all_identical_is_one() {
        let row = [4.0, 4.0, 4.0, 4.0, 4.0];
        assert_close(straightline_score(&row, 1.0, 5.0, 3).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn test_straightline_alternating_endpoints_is_low() {
        // Alternating 1,5,1,5,1: mode_frac = 3/5 = 0.6, but variance is
        // maximal (half_range^2 = 4), so `1 - var_norm` ~ 0 and mode_frac
        // dominates but is well below the 0.85 default threshold band we
        // document -- assert it's meaningfully lower than the all-identical
        // case (correctness property, not a pinned threshold).
        let identical = [4.0, 4.0, 4.0, 4.0, 4.0];
        let alternating = [1.0, 5.0, 1.0, 5.0, 1.0];
        let s_identical = straightline_score(&identical, 1.0, 5.0, 3).unwrap();
        let s_alternating = straightline_score(&alternating, 1.0, 5.0, 3).unwrap();
        assert!(s_alternating < s_identical);
        assert!(s_alternating < 0.65);
    }

    #[test]
    fn test_straightline_normal_variation_is_lower_than_straightliner() {
        let straightliner = [4.0, 4.0, 4.0, 4.0, 4.0];
        let normal = [2.0, 4.0, 3.0, 5.0, 1.0];
        let s_straight = straightline_score(&straightliner, 1.0, 5.0, 3).unwrap();
        let s_normal = straightline_score(&normal, 1.0, 5.0, 3).unwrap();
        assert!(s_straight > s_normal);
        assert_close(s_straight, 1.0, 1e-12);
    }

    #[test]
    fn test_straightline_below_min_answers_is_none() {
        let row = [4.0, 4.0];
        assert_eq!(straightline_score(&row, 1.0, 5.0, 3), None);
    }

    #[test]
    fn test_straightline_empty_is_none() {
        let row: [f64; 0] = [];
        assert_eq!(straightline_score(&row, 1.0, 5.0, 3), None);
    }

    #[test]
    fn test_straightline_degenerate_scale_falls_back_to_mode_frac() {
        // scale_max <= scale_min: variance component skipped.
        let row = [3.0, 3.0, 5.0];
        let score = straightline_score(&row, 2.0, 2.0, 3).unwrap();
        assert_close(score, 2.0 / 3.0, 1e-12);
    }

    // ------------------------------------------------------------------
    // gibberish_score
    // ------------------------------------------------------------------

    #[test]
    fn test_gibberish_keyboard_mash_scores_higher_than_english() {
        let mash = gibberish_score("asdkjfhaslkdjf", 8).unwrap();
        let english = gibberish_score("The service was slow but friendly", 8).unwrap();
        assert!(
            mash > english + 0.3,
            "mash={mash}, english={english} (want margin >= 0.3)"
        );
    }

    #[test]
    fn test_gibberish_repeated_char_is_bounded_and_vowel_deviant() {
        // "aaaaaaaa": vowel_ratio=1.0 maxes out vowel_dev (weight 0.3), but
        // consonant_run (weight 0.5) is 0 (never a consonant) -- the
        // dominant weight doesn't fire here by design (a repeated-vowel
        // string isn't the "keyboard mash" shape this heuristic targets;
        // see the module-level weighting discussion). Just pin it's a
        // valid, non-trivial score, not an ordering vs. English text.
        let score = gibberish_score("aaaaaaaa", 8).unwrap();
        assert!((0.0..=1.0).contains(&score));
        assert!(score > 0.25);
    }

    #[test]
    fn test_gibberish_too_short_is_none() {
        assert_eq!(gibberish_score("hi", 8), None);
    }

    #[test]
    fn test_gibberish_non_latin_script_is_none() {
        // Normalizes to ~nothing (no [a-z] survives) -- null, not a false
        // "gibberish" flag. Documented limitation (English-only heuristic).
        assert_eq!(gibberish_score("こんにちは世界", 8), None);
    }

    #[test]
    fn test_gibberish_empty_is_none() {
        assert_eq!(gibberish_score("", 8), None);
    }

    #[test]
    fn test_gibberish_score_bounded() {
        for text in ["asdkjfhaslkdjf qwpoeiruqpwoeiru", "The quick brown fox jumps"] {
            let s = gibberish_score(text, 8).unwrap();
            assert!((0.0..=1.0).contains(&s));
        }
    }

    // ------------------------------------------------------------------
    // duplicate_answer_score
    // ------------------------------------------------------------------

    #[test]
    fn test_duplicate_exact_repeat_is_one() {
        let a = "The product is great and I would recommend it";
        let b = "the product is great and I would recommend it.";
        let score = duplicate_answer_score(&[a, b], 10).unwrap();
        assert_close(score, 1.0, 1e-12);
    }

    #[test]
    fn test_duplicate_distinct_answers_is_low() {
        let a = "The product is great and I would recommend it";
        let b = "Shipping was slow and the box arrived damaged";
        let score = duplicate_answer_score(&[a, b], 10).unwrap();
        assert!(score < 0.3);
    }

    #[test]
    fn test_duplicate_partial_overlap() {
        let a = "the product is great and fast";
        let b = "the product is great but pricey";
        // tokens: {the,product,is,great,and,fast} vs {the,product,is,great,but,pricey}
        // intersection = {the,product,is,great} = 4, union = 8
        let score = duplicate_answer_score(&[a, b], 10).unwrap();
        assert_close(score, 4.0 / 8.0, 1e-12);
    }

    #[test]
    fn test_duplicate_short_answers_skipped() {
        // Both answers below min_answer_chars=10 -- legit "yes"/"n/a" repeats.
        let score = duplicate_answer_score(&["yes", "yes"], 10);
        assert_eq!(score, None);
    }

    #[test]
    fn test_duplicate_single_survivor_is_none() {
        let a = "The product is great and I would recommend it";
        let score = duplicate_answer_score(&[a, "n/a"], 10);
        assert_eq!(score, None);
    }

    #[test]
    fn test_duplicate_empty_is_none() {
        let score = duplicate_answer_score(&[], 10);
        assert_eq!(score, None);
    }

    #[test]
    fn test_duplicate_three_answers_max_pairwise() {
        let a = "the weather today is quite nice";
        let b = "the weather today is quite nice";
        let c = "completely unrelated statement about pricing";
        let score = duplicate_answer_score(&[a, b, c], 10).unwrap();
        assert_close(score, 1.0, 1e-12);
    }
}
