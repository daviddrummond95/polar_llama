"""Tests for inter-rater reliability metrics (issue #79): `cohens_kappa`,
`krippendorffs_alpha`.

All expected values are hardcoded floats, independently verified against
published reference fixtures / reference implementations
(`sklearn.metrics.cohen_kappa_score`, the `krippendorff` PyPI package) at
planning time -- this test file has no runtime dependency on either
package and is CI-safe (no network, no API key, no `mlx`).

Fixture A is Krippendorff's own canonical reliability-data example (2004 /
2011 "Computing Krippendorff's Alpha-Reliability"), 4 observers x 12 units.
Fixture B is the `krippendorff` PyPI package's own README example, 3
coders x 15 units.
"""

import math

import polars as pl
import pytest

from polar_llama import cohens_kappa, krippendorffs_alpha

TOL = 1e-9


def approx(expected: float) -> pytest.approx:
    return pytest.approx(expected, abs=TOL)


# ============================================================================
# Fixtures
# ============================================================================

# Krippendorff's canonical reliability-data example, 4 observers x 12
# units. Unit 12 has only one (non-null) rating and is excluded internally.
FIXTURE_A = {
    "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
}

# The krippendorff PyPI README example, 3 coders x 15 units.
FIXTURE_B = {
    "A": [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3],
    "B": [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None],
    "C": [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4],
}


# ============================================================================
# Krippendorff's alpha -- Fixture A (all 4 levels)
# ============================================================================


class TestAlphaFixtureA:
    df = pl.DataFrame(FIXTURE_A)

    def test_nominal(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C", "D"], level="nominal"))
        assert result["a"][0] == approx(0.743421052631579)

    def test_ordinal(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C", "D"], level="ordinal"))
        assert result["a"][0] == approx(0.8153875037548814)

    def test_interval(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C", "D"], level="interval"))
        assert result["a"][0] == approx(0.8491071428571428)

    def test_ratio(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C", "D"], level="ratio"))
        assert result["a"][0] == approx(0.7974027747116121)


# ============================================================================
# Krippendorff's alpha -- Fixture B (all 4 levels)
# ============================================================================


class TestAlphaFixtureB:
    df = pl.DataFrame(FIXTURE_B)

    def test_nominal(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C"], level="nominal"))
        assert result["a"][0] == approx(0.691358024691358)

    def test_ordinal(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C"], level="ordinal"))
        assert result["a"][0] == approx(0.8067214199413153)

    def test_interval(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C"], level="interval"))
        assert result["a"][0] == approx(0.8108448928121059)

    def test_ratio(self):
        result = self.df.select(a=krippendorffs_alpha(["A", "B", "C"], level="ratio"))
        assert result["a"][0] == approx(0.8089436707842471)


# ============================================================================
# Cohen's kappa -- hand-derivable + sklearn-confirmed fixtures
# ============================================================================


class TestKappaFixtures:
    def test_2x2_exact(self):
        # Confusion matrix [[20,5],[10,15]], n=50: p_o=0.7, p_e=0.5, kappa=0.4.
        a = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15
        b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        df = pl.DataFrame({"a": a, "b": b})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] == approx(0.4)

    def test_weighted_unweighted(self):
        y1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 1]
        y2 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]
        df = pl.DataFrame({"y1": y1, "y2": y2})
        result = df.select(k=cohens_kappa("y1", "y2"))
        assert result["k"][0] == approx(0.5522388059701492)

    def test_weighted_linear(self):
        y1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 1]
        y2 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]
        df = pl.DataFrame({"y1": y1, "y2": y2})
        result = df.select(k=cohens_kappa("y1", "y2", weights="linear"))
        assert result["k"][0] == approx(0.6590909090909092)

    def test_weighted_quadratic(self):
        y1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 1]
        y2 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]
        df = pl.DataFrame({"y1": y1, "y2": y2})
        result = df.select(k=cohens_kappa("y1", "y2", weights="quadratic"))
        assert result["k"][0] == approx(0.7692307692307692)

    def test_pairwise_null_interleaved_unchanged(self):
        # Same 2x2 fixture as test_2x2_exact, but with nulls interleaved
        # into both columns at matching row positions (so the pairwise-
        # complete subset is identical) -- result must be unchanged (0.4).
        a = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15
        b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        # Insert a handful of null rows (only one side null per row).
        a2 = a[:5] + [None] + a[5:10] + [None] + a[10:]
        b2 = b[:5] + [1] + b[5:10] + [None] + b[10:]
        df = pl.DataFrame({"a": a2, "b": b2})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] == approx(0.4)


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_kappa_single_category_is_nan(self):
        df = pl.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]})
        result = df.select(k=cohens_kappa("a", "b"))
        assert math.isnan(result["k"][0])

    def test_alpha_single_category_is_one(self):
        df = pl.DataFrame({"r1": [1, 1, 1], "r2": [1, 1, 1]})
        result = df.select(a=krippendorffs_alpha(["r1", "r2"]))
        assert result["a"][0] == approx(1.0)

    def test_kappa_perfect_agreement(self):
        df = pl.DataFrame({"a": [1, 2, 1, 2], "b": [1, 2, 1, 2]})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] == approx(1.0)

    def test_alpha_perfect_agreement(self):
        df = pl.DataFrame({"r1": [1, 2, 1, 2], "r2": [1, 2, 1, 2]})
        result = df.select(a=krippendorffs_alpha(["r1", "r2"]))
        assert result["a"][0] == approx(1.0)

    def test_kappa_systematic_disagreement(self):
        df = pl.DataFrame({"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] == approx(-1.0)

    def test_alpha_systematic_disagreement(self):
        # Krippendorff's own [[1,2,1,2],[2,1,2,1]] nominal fixture -> -0.75.
        df = pl.DataFrame({"r1": [1, 2, 1, 2], "r2": [2, 1, 2, 1]})
        result = df.select(a=krippendorffs_alpha(["r1", "r2"]))
        assert result["a"][0] == approx(-0.75)

    def test_kappa_zero_pairwise_complete_is_null(self):
        df = pl.DataFrame({"a": [1, None], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] is None

    def test_alpha_no_pairable_units_is_null(self):
        df = pl.DataFrame({"a": [1, None], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(a=krippendorffs_alpha(["a", "b"]))
        assert result["a"][0] is None

    def test_kappa_empty_columns_is_null(self):
        df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(k=cohens_kappa("a", "b"))
        assert result["k"][0] is None

    def test_alpha_all_null_columns_is_null(self):
        df = pl.DataFrame(
            {"a": [None, None], "b": [None, None]}, schema={"a": pl.Int64, "b": pl.Int64}
        )
        result = df.select(a=krippendorffs_alpha(["a", "b"]))
        assert result["a"][0] is None

    def test_alpha_missing_data_matches_fixture_a(self):
        # Re-assert Fixture A nominal specifically to document that missing
        # data (nulls) is exercised by the headline fixture, not just the
        # dedicated null-handling tests above.
        df = pl.DataFrame(FIXTURE_A)
        result = df.select(a=krippendorffs_alpha(["A", "B", "C", "D"]))
        assert result["a"][0] == approx(0.743421052631579)


# ============================================================================
# group_by context: per-group values match per-slice computation
# ============================================================================


class TestGroupByContext:
    def test_kappa_per_group_matches_slice(self):
        a1 = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15
        b1 = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        a2 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 1]
        b2 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]

        df = pl.DataFrame(
            {
                "group": ["g1"] * len(a1) + ["g2"] * len(a2),
                "a": a1 + a2,
                "b": b1 + b2,
            }
        )
        result = df.group_by("group", maintain_order=True).agg(k=cohens_kappa("a", "b"))
        values = dict(zip(result["group"], result["k"]))
        assert values["g1"] == approx(0.4)
        assert values["g2"] == approx(0.5522388059701492)

    def test_alpha_per_group_matches_slice(self):
        n_a = len(FIXTURE_A["A"])
        df = pl.DataFrame(
            {
                "group": ["g1"] * n_a + ["g2"] * n_a,
                "r1": FIXTURE_A["A"] + FIXTURE_A["A"],
                "r2": FIXTURE_A["B"] + FIXTURE_A["B"],
                "r3": FIXTURE_A["C"] + FIXTURE_A["C"],
                "r4": FIXTURE_A["D"] + FIXTURE_A["D"],
            }
        )
        result = df.group_by("group", maintain_order=True).agg(
            a=krippendorffs_alpha(["r1", "r2", "r3", "r4"])
        )
        for v in result["a"]:
            assert v == approx(0.743421052631579)


# ============================================================================
# Bootstrap CIs
# ============================================================================


class TestBootstrap:
    def test_kappa_bootstrap_returns_struct(self):
        a = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15
        b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        df = pl.DataFrame({"a": a, "b": b})
        result = df.select(
            cohens_kappa("a", "b", n_bootstrap=200, seed=0).struct.unnest()
        )
        assert result.columns == ["value", "ci_low", "ci_high"]
        assert result["value"][0] == approx(0.4)
        assert result["ci_low"][0] <= result["value"][0] + 1e-6
        assert result["ci_high"][0] >= result["value"][0] - 1e-6

    def test_alpha_bootstrap_determinism_same_seed(self):
        df = pl.DataFrame(FIXTURE_A)
        r1 = df.select(
            krippendorffs_alpha(
                ["A", "B", "C", "D"], n_bootstrap=1000, seed=42
            ).struct.unnest()
        )
        r2 = df.select(
            krippendorffs_alpha(
                ["A", "B", "C", "D"], n_bootstrap=1000, seed=42
            ).struct.unnest()
        )
        assert r1["ci_low"][0] == r2["ci_low"][0]
        assert r1["ci_high"][0] == r2["ci_high"][0]
        assert r1["value"][0] == approx(0.743421052631579)

    def test_alpha_bootstrap_ci_brackets_point_estimate(self):
        df = pl.DataFrame(FIXTURE_A)
        result = df.select(
            krippendorffs_alpha(
                ["A", "B", "C", "D"], n_bootstrap=1000, seed=0
            ).struct.unnest()
        )
        value, lo, hi = result["value"][0], result["ci_low"][0], result["ci_high"][0]
        assert lo <= value + 1e-6
        assert hi >= value - 1e-6

    def test_no_bootstrap_returns_plain_float(self):
        df = pl.DataFrame(FIXTURE_A)
        result = df.select(a=krippendorffs_alpha(["A", "B", "C", "D"]))
        assert result.schema["a"] == pl.Float64


# ============================================================================
# .llama namespace
# ============================================================================


class TestNamespace:
    def test_kappa_namespace(self):
        df = pl.DataFrame({"a": [1, 2, 1, 2], "b": [1, 2, 1, 2]})
        result = df.select(k=pl.col("a").llama.cohens_kappa("b"))
        assert result["k"][0] == approx(1.0)

    def test_alpha_namespace(self):
        df = pl.DataFrame(FIXTURE_A)
        result = df.select(a=pl.col("A").llama.krippendorffs_alpha("B", "C", "D"))
        assert result["a"][0] == approx(0.743421052631579)


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa("a", "b", weights="bogus")

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            krippendorffs_alpha(["a", "b"], level="bogus")

    def test_invalid_ci_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa("a", "b", n_bootstrap=10, ci=1.5)

    def test_too_few_alpha_columns_raises(self):
        with pytest.raises(ValueError):
            krippendorffs_alpha(["a"])

    def test_ordinal_alpha_rejects_string_columns(self):
        df = pl.DataFrame({"r1": ["low", "high"], "r2": ["low", "high"]})
        with pytest.raises(pl.exceptions.PolarsError):
            df.select(krippendorffs_alpha(["r1", "r2"], level="ordinal")).item()


def test_nan_ratings_treated_as_missing_no_panic():
    """Regression (#79 review): a NaN float in a rater column must be treated
    as missing (like the reference krippendorff package), not panic the plugin
    or become a spurious label."""
    # krippendorffs_alpha: NaN in a float column == missing, no query abort.
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, float("nan"), 4.0],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [1.0, float("nan"), 3.0, 4.0],
        }
    )
    a = df.select(a=krippendorffs_alpha(["A", "B", "C"], level="interval"))["a"][0]
    assert a is not None and math.isfinite(a)  # computed, not a crash/NaN

    # Same data with the NaNs written as nulls must give an identical alpha.
    df_null = pl.DataFrame(
        {
            "A": [1.0, 2.0, None, 4.0],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [1.0, None, 3.0, 4.0],
        }
    )
    a_null = df_null.select(a=krippendorffs_alpha(["A", "B", "C"], level="interval"))["a"][0]
    assert abs(a - a_null) < 1e-12

    # cohens_kappa: NaN rows dropped as missing (pairwise-complete), no panic.
    dfk = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 3.0], "b": [1.0, 2.0, 2.0, 3.0]})
    k = dfk.select(k=cohens_kappa("a", "b"))["k"][0]
    k_null = pl.DataFrame({"a": [1.0, 2.0, None, 3.0], "b": [1.0, 2.0, 2.0, 3.0]}).select(
        k=cohens_kappa("a", "b")
    )["k"][0]
    assert (k is None and k_null is None) or abs(k - k_null) < 1e-12
