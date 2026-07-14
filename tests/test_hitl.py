"""Tests for the human-in-the-loop review loop (issue #81):
`export_review_sample`, `import_corrections`, `corrections_to_trainset`,
`retune_from_corrections`.

CI-safe: `retune_from_corrections`/`corrections_to_trainset` tests use an
injected `inference_fn` (same convention as `tests/test_optimize.py`), so
this file requires no API key, no network, and no `mlx`. A gated real-API
proof lives at the bottom (`@pytest.mark.skipif(not os.getenv(
"OPENAI_API_KEY"), ...)`, matching `tests/test_codebook.py`/
`tests/test_quality_llm.py`).

The Cohen's kappa hand fixture below (`test_import_kappa_matches_hand_calc`)
is independently derived, not read off `cohens_kappa`'s own test suite:
10 rows, 2 categories, 6/10 raw agreement, both marginals 50/50 ->
po=0.6, pe=0.5, kappa=(po-pe)/(1-pe)=0.2 exactly.
"""

import json
import math
import os
import warnings

import polars as pl
import pytest

try:
    import xlsxwriter  # noqa: F401

    _HAS_XLSXWRITER = True
except ImportError:
    _HAS_XLSXWRITER = False

from polar_llama import (
    CorrectionResult,
    Predict,
    Signature,
    corrections_to_trainset,
    export_review_sample,
    import_corrections,
    retune_from_corrections,
)

TOL = 1e-9


# ============================================================================
# export_review_sample
# ============================================================================


def _skewed_df(n_a=90, n_b=8, n_c=2):
    return pl.DataFrame(
        {
            "group": ["A"] * n_a + ["B"] * n_b + ["C"] * n_c,
            "val": list(range(n_a + n_b + n_c)),
        }
    )


class TestExportReviewSample:
    def test_returns_exactly_n_rows(self):
        df = pl.DataFrame({"x": list(range(50))})
        sample = export_review_sample(df, n=10, seed=0, include_review_columns=False)
        assert sample.height == 10

    def test_n_greater_than_height_returns_full_df(self):
        df = pl.DataFrame({"x": list(range(5))})
        sample = export_review_sample(df, n=100, seed=0, include_review_columns=False)
        assert sample.height == 5
        assert sorted(sample["x"].to_list()) == [0, 1, 2, 3, 4]

    def test_each_stratum_represented(self):
        df = _skewed_df()
        sample = export_review_sample(
            df, n=10, strata="group", seed=0, include_review_columns=False
        )
        assert set(sample["group"].unique().to_list()) == {"A", "B", "C"}
        assert sample.height == 10

    def test_proportional_allocation_matches_hand_calc(self):
        df = _skewed_df()
        sample = export_review_sample(
            df,
            n=10,
            strata="group",
            allocation="proportional",
            seed=0,
            include_review_columns=False,
        )
        counts = dict(
            sample.group_by("group").len().sort("group").iter_rows()
        )
        # largest-remainder: exact=[9, 0.8, 0.2] -> base=[9,0,0] + remainder
        # to B (frac .8) -> [9,1,0] -> min-1 guarantee bumps C, donating
        # from A (largest) -> [8,1,1].
        assert counts == {"A": 8, "B": 1, "C": 1}
        assert sum(counts.values()) == 10

    def test_equal_allocation_matches_hand_calc(self):
        df = _skewed_df()
        sample = export_review_sample(
            df,
            n=10,
            strata="group",
            allocation="equal",
            seed=0,
            include_review_columns=False,
        )
        counts = dict(
            sample.group_by("group").len().sort("group").iter_rows()
        )
        # equal split: base 3 each + 1 remainder to largest (A) -> [4,3,3]
        # -> C (size 2) is over quota -> capped to 2, redistribute 1
        # deficit to A (largest remaining capacity) -> [5,3,2].
        assert counts == {"A": 5, "B": 3, "C": 2}
        assert sum(counts.values()) == 10

    def test_quota_capped_and_redistributed(self):
        # A tiny stratum (2 rows) whose naive quota would exceed its size.
        df = pl.DataFrame({"group": ["A"] * 20 + ["B"] * 2, "val": list(range(22))})
        sample = export_review_sample(
            df, n=11, strata="group", allocation="equal", seed=0,
            include_review_columns=False,
        )
        counts = dict(sample.group_by("group").len().sort("group").iter_rows())
        assert counts["B"] == 2  # capped at its own size
        assert counts["A"] == 9  # absorbed the deficit
        assert sum(counts.values()) == 11

    def test_deterministic_with_seed(self):
        df = pl.DataFrame({"x": list(range(200))})
        s1 = export_review_sample(df, n=30, seed=7, include_review_columns=False)
        s2 = export_review_sample(df, n=30, seed=7, include_review_columns=False)
        assert s1.equals(s2)

        s3 = export_review_sample(df, n=30, seed=8, include_review_columns=False)
        assert not s1.equals(s3)

    def test_oversampling_boosts_low_confidence(self):
        n_rows = 2000
        half = n_rows // 2
        df = pl.DataFrame(
            {
                "text": [f"t{i}" for i in range(n_rows)],
                "confidence": [0.0] * half + [1.0] * half,
            }
        )
        oversampled = export_review_sample(
            df,
            n=200,
            confidence_column="confidence",
            oversample_low_confidence=9.0,
            seed=42,
            include_review_columns=False,
        )
        uniform = export_review_sample(
            df,
            n=200,
            confidence_column="confidence",
            oversample_low_confidence=0.0,
            seed=42,
            include_review_columns=False,
        )
        over_share = (oversampled["confidence"] == 0.0).sum() / 200
        uniform_share = (uniform["confidence"] == 0.0).sum() / 200
        assert over_share > 0.75
        assert over_share > uniform_share

    def test_null_confidence_treated_as_low(self):
        n_rows = 2000
        half = n_rows // 2
        df = pl.DataFrame(
            {
                "text": [f"t{i}" for i in range(n_rows)],
                "confidence": [None] * half + [1.0] * half,
            }
        )
        sample = export_review_sample(
            df,
            n=200,
            confidence_column="confidence",
            oversample_low_confidence=9.0,
            seed=1,
            include_review_columns=False,
        )
        null_share = sample["confidence"].is_null().sum() / 200
        assert null_share > 0.75

    def test_csv_roundtrip(self, tmp_path):
        df = pl.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        out = tmp_path / "sample.csv"
        sample = export_review_sample(df, n=3, seed=0, path=str(out))
        assert out.exists()

        back = pl.read_csv(out)
        assert back.height == 3
        assert "_review_id" in back.columns
        assert "corrected_code" in back.columns
        assert "review_notes" in back.columns
        assert back["corrected_code"].null_count() == 3
        assert back["review_notes"].null_count() == 3
        assert sample.height == 3

    def test_xlsx_missing_dep_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(__import__("sys").modules, "xlsxwriter", None)
        df = pl.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ImportError, match=r"polar-llama\[excel\]"):
            export_review_sample(
                df, n=2, seed=0, path=str(tmp_path / "sample.xlsx")
            )

    @pytest.mark.skipif(not _HAS_XLSXWRITER, reason="xlsxwriter not installed")
    def test_xlsx_with_dep(self, tmp_path):
        df = pl.DataFrame({"text": ["a", "b", "c"]})
        out = tmp_path / "sample.xlsx"
        export_review_sample(df, n=2, seed=0, path=str(out))
        assert out.exists()

    def test_validation_errors(self):
        df = pl.DataFrame({"x": [1, 2, 3], "g": ["a", "b", "c"]})
        with pytest.raises(ValueError):
            export_review_sample(df, n=0)
        with pytest.raises(ValueError):
            export_review_sample(df, n=-5)
        with pytest.raises(ValueError):
            export_review_sample(df, n=2, oversample_low_confidence=-1.0)
        with pytest.raises(ValueError):
            export_review_sample(df, n=2, strata="nope")
        with pytest.raises(ValueError):
            export_review_sample(df, n=2, confidence_column="nope")
        with pytest.raises(ValueError):
            export_review_sample(df, n=2, format="json")
        with pytest.raises(ValueError):
            export_review_sample(df, n=2, allocation="bogus")


# ============================================================================
# import_corrections
# ============================================================================


class TestImportCorrections:
    def test_join_correctness(self):
        df = pl.DataFrame(
            {"_review_id": [0, 1, 2, 3], "code": ["cat", "dog", "cat", "bird"]}
        )
        corrections = pl.DataFrame(
            {
                "_review_id": [0, 1, 2, 3],
                "corrected_code": ["cat", "cat", None, "bird"],
            }
        )
        result = import_corrections(df, corrections, code_column="code")
        assert isinstance(result, CorrectionResult)
        out = result.df.sort("_review_id")
        assert out["was_reviewed"].to_list() == [True, True, False, True]
        assert out["was_changed"].to_list() == [False, True, False, False]
        # unreviewed row keeps its original code, unaffected
        assert out.filter(pl.col("_review_id") == 2)["code"].item() == "cat"
        assert result.n_reviewed == 3
        assert result.n_unreviewed == 1

    def test_df_already_has_corrected_column_no_silent_loss(self):
        # Regression (#81 review): when `df` already carries a (blank)
        # `corrected_code` column -- exactly what export_review_sample emits,
        # so re-feeding it is a natural path -- corrections must NOT be
        # silently dropped by a join-name collision.
        df = pl.DataFrame(
            {
                "_review_id": [0, 1, 2],
                "code": ["cat", "dog", "cat"],
                "corrected_code": [None, None, None],  # blank, as exported
            }
        )
        corrections = pl.DataFrame(
            {"_review_id": [0, 1, 2], "corrected_code": ["cat", "cat", "bird"]}
        )
        result = import_corrections(df, corrections, code_column="code")
        assert result.n_reviewed == 3  # NOT 0
        assert result.n_changed == 2  # dog->cat, cat->bird
        out = result.df.sort("_review_id")
        assert out["corrected_code"].to_list() == ["cat", "cat", "bird"]
        assert not math.isnan(result.kappa)

    def test_kappa_matches_hand_calc(self):
        # po=0.6, pe=0.5 -> kappa=0.2 exactly (see module docstring).
        llm = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        human = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        df = pl.DataFrame({"_review_id": list(range(10)), "code": llm})
        corrections = pl.DataFrame(
            {"_review_id": list(range(10)), "corrected_code": human}
        )
        result = import_corrections(df, corrections, code_column="code")
        assert result.kappa == pytest.approx(0.2, abs=TOL)
        assert result.agreement_rate == pytest.approx(0.6, abs=TOL)
        assert result.n_reviewed == 10
        assert result.n_changed == 4

    def test_unmatched_correction_ids(self):
        df = pl.DataFrame({"_review_id": [0, 1], "code": ["a", "b"]})
        corrections = pl.DataFrame(
            {"_review_id": [0, 1, 99], "corrected_code": ["a", "b", "z"]}
        )
        with pytest.warns(UserWarning):
            result = import_corrections(df, corrections, code_column="code")
        assert result.n_unmatched_corrections == 1

        with pytest.raises(ValueError):
            import_corrections(
                df, corrections, code_column="code", on_unmatched="raise"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = import_corrections(
                df, corrections, code_column="code", on_unmatched="ignore"
            )
        assert result.n_unmatched_corrections == 1

    def test_unreviewed_rows_counted(self):
        df = pl.DataFrame({"_review_id": [0, 1, 2], "code": ["a", "b", "c"]})
        corrections = pl.DataFrame({"_review_id": [0], "corrected_code": ["a"]})
        result = import_corrections(df, corrections, code_column="code")
        assert result.n_reviewed == 1
        assert result.n_unreviewed == 2

    def test_duplicate_ids_raise(self):
        df = pl.DataFrame({"_review_id": [0, 0, 1], "code": ["a", "a", "b"]})
        corrections = pl.DataFrame({"_review_id": [0], "corrected_code": ["a"]})
        with pytest.raises(ValueError):
            import_corrections(df, corrections, code_column="code")

        df2 = pl.DataFrame({"_review_id": [0, 1], "code": ["a", "b"]})
        corrections2 = pl.DataFrame(
            {"_review_id": [0, 0], "corrected_code": ["a", "x"]}
        )
        with pytest.raises(ValueError):
            import_corrections(df2, corrections2, code_column="code")

    def test_blank_corrections_are_unreviewed(self):
        df = pl.DataFrame({"_review_id": [0, 1, 2], "code": ["a", "b", "c"]})
        corrections = pl.DataFrame(
            {"_review_id": [0, 1, 2], "corrected_code": ["a", "  ", None]}
        )
        result = import_corrections(df, corrections, code_column="code")
        assert result.n_reviewed == 1
        assert result.n_unreviewed == 2

    def test_dtype_cast_from_csv_strings(self):
        df = pl.DataFrame({"_review_id": [0, 1], "code": [1, 2]})
        corrections = pl.DataFrame(
            {"_review_id": [0, 1], "corrected_code": ["1", "3"]}
        )
        result = import_corrections(df, corrections, code_column="code")
        assert result.df.sort("_review_id")["corrected_code"].to_list() == [1, 3]
        assert result.df.schema["corrected_code"] == df.schema["code"]

        bad_corrections = pl.DataFrame(
            {"_review_id": [0, 1], "corrected_code": ["not-a-number", "3"]}
        )
        with pytest.raises(ValueError):
            import_corrections(df, bad_corrections, code_column="code")

    def test_kappa_nan_single_category_documented(self):
        df = pl.DataFrame({"_review_id": [0, 1, 2], "code": [1, 1, 1]})
        corrections = pl.DataFrame(
            {"_review_id": [0, 1, 2], "corrected_code": [1, 1, 1]}
        )
        result = import_corrections(df, corrections, code_column="code")
        assert math.isnan(result.kappa)
        assert result.agreement_rate == 1.0


# ============================================================================
# corrections_to_trainset / retune_from_corrections
# ============================================================================


def uppercase_backend(messages, output_model):
    out = []
    for raw in messages:
        convo = json.loads(raw)
        user = [m for m in convo if m["role"] == "user"][-1]
        text = user["content"].split("text: ", 1)[-1]
        out.append(json.dumps({"code": text.upper()}))
    return out


class TestCorrectionsToTrainset:
    def test_columns_match_signature_fields_default_naming(self):
        sig = Signature("text -> code")
        corrections = pl.DataFrame(
            {"text": ["a", "b"], "corrected_code": ["A", "B"]}
        )
        trainset = corrections_to_trainset(corrections, sig)
        assert set(trainset.columns) == {"text", "code"}
        assert trainset["code"].to_list() == ["A", "B"]

    def test_columns_match_signature_fields_with_column_map(self):
        sig = Signature("text -> code")
        corrections = pl.DataFrame(
            {"response_text": ["a", "b"], "corrected_code": ["A", "B"]}
        )
        trainset = corrections_to_trainset(
            corrections, sig, column_map={"text": "response_text"}
        )
        assert set(trainset.columns) == {"text", "code"}
        assert trainset["text"].to_list() == ["a", "b"]

    def test_multi_output_requires_column_map(self):
        sig = Signature("text -> sentiment, confidence")
        corrections = pl.DataFrame(
            {"text": ["a"], "sentiment_gold": ["pos"], "confidence_gold": [0.9]}
        )
        with pytest.raises(ValueError):
            corrections_to_trainset(corrections, sig)

        trainset = corrections_to_trainset(
            corrections,
            sig,
            column_map={
                "sentiment": "sentiment_gold",
                "confidence": "confidence_gold",
            },
        )
        assert set(trainset.columns) == {"text", "sentiment", "confidence"}

    def test_missing_column_raises(self):
        sig = Signature("text -> code")
        corrections = pl.DataFrame({"nope": ["a"], "corrected_code": ["A"]})
        with pytest.raises(ValueError):
            corrections_to_trainset(corrections, sig)

    def test_null_rows_dropped(self):
        sig = Signature("text -> code")
        corrections = pl.DataFrame(
            {"text": ["a", "b", "c"], "corrected_code": ["A", None, "C"]}
        )
        trainset = corrections_to_trainset(corrections, sig)
        assert trainset.height == 2
        assert trainset["text"].to_list() == ["a", "c"]

    def test_empty_trainset_raises(self):
        sig = Signature("text -> code")
        corrections = pl.DataFrame(
            {"text": ["a", "b"], "corrected_code": [None, None]}
        )
        with pytest.raises(ValueError):
            corrections_to_trainset(corrections, sig)


class TestRetuneFromCorrections:
    def test_returns_predict_with_gold_demos(self):
        corrections = pl.DataFrame(
            {
                "text": ["hi", "there", "world"],
                "corrected_code": ["HI", "THERE", "WORLD"],
            }
        )
        compiled = retune_from_corrections(
            corrections,
            "text -> code",
            inference_fn=uppercase_backend,
            max_demos=2,
        )
        assert isinstance(compiled, Predict)
        assert len(compiled.demos) <= 2
        assert len(compiled.demos) > 0
        for demo in compiled.demos:
            assert demo["code"] == demo["text"].upper()

    def test_accepts_predict_and_reuses_its_backend(self):
        module = Predict("text -> code", inference_fn=uppercase_backend)
        corrections = pl.DataFrame(
            {"text": ["ab"], "corrected_code": ["AB"]}
        )
        compiled = retune_from_corrections(corrections, module, max_demos=1)
        assert compiled.inference_fn is uppercase_backend
        assert compiled.demos[0] == {"text": "ab", "code": "AB"}

    def test_default_metric_is_normalized(self):
        def echo_backend(messages, output_model):
            return [json.dumps({"code": "yes"}) for _ in messages]

        corrections = pl.DataFrame({"text": ["q1"], "corrected_code": [" Yes "]})
        compiled = retune_from_corrections(
            corrections, "text -> code", inference_fn=echo_backend, max_demos=1
        )
        # " Yes " (gold) normalizes equal to "yes" (prediction) -> passes
        # threshold=0.0 with gold outputs as the demo.
        assert compiled.demos[0]["code"] == " Yes "

    def test_ci_roundtrip(self):
        df = pl.DataFrame(
            {
                "text": ["hello", "world", "foo", "bar"],
                "pred_code": ["HELLO", "wrong", "FOO", "BAR"],
            }
        ).with_row_index("_review_id")
        sample = df.select(
            "_review_id", "text", corrected_code=pl.col("text").str.to_uppercase()
        )
        corrections = import_corrections(
            df.rename({"pred_code": "code"}), sample, code_column="code"
        )
        assert corrections.n_reviewed == 4

        trainset = corrections_to_trainset(corrections.df, "text -> code")
        assert trainset.height == 4

        compiled = retune_from_corrections(
            corrections.df, "text -> code", inference_fn=uppercase_backend, max_demos=4
        )
        assert isinstance(compiled, Predict)
        assert len(compiled.demos) > 0


# ============================================================================
# Real-API round trip (gated; never runs in CI without a key)
# ============================================================================


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_hitl_roundtrip_real_api():
    from polar_llama import Provider, evaluate

    base = [
        ("I absolutely love this product, best purchase all year!", "positive"),
        ("Amazing customer support and fast shipping.", "positive"),
        ("This exceeded every expectation I had.", "positive"),
        ("Terrible experience, it broke after one day.", "negative"),
        ("Waste of money, would not recommend.", "negative"),
        ("Customer service was rude and unhelpful.", "negative"),
        ("It's fine, does what it says, nothing special.", "neutral"),
        ("Average quality, works as expected.", "neutral"),
        ("Neither impressed nor disappointed.", "neutral"),
    ]
    rows = [(f"{text} (#{i})", label) for i in range(5) for text, label in base]
    df = pl.DataFrame(
        {"text": [r[0] for r in rows], "gold": [r[1] for r in rows]}
    ).with_row_index("_review_id")
    train_df, holdout_df = df[:35], df[35:]

    module = Predict(
        Signature(
            "text -> code",
            instructions=(
                "Classify the sentiment as exactly one of: positive, negative, "
                "neutral."
            ),
        ),
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
    )

    def exact_match(example, prediction):
        return float(
            (example.get("gold") or "").strip().casefold()
            == (prediction.get("code") or "").strip().casefold()
        )

    baseline = evaluate(module, holdout_df.rename({"gold": "code"}), exact_match)

    # Code the training split, then run the review loop: export a sample,
    # "review" it (corrections = the known gold labels), import, retune.
    labeled = module(train_df).rename({"pred_code": "code"})

    sample = export_review_sample(labeled, n=20, seed=0)
    corrections_input = sample.join(
        df.select("_review_id", "gold"), on="_review_id", how="left"
    ).with_columns(corrected_code=pl.col("gold"))

    result = import_corrections(labeled, corrections_input, code_column="code")
    assert result.n_reviewed > 0
    assert isinstance(result.kappa, float)

    tuned = retune_from_corrections(
        result.df.filter(pl.col("was_reviewed")),
        Signature("text -> code", instructions=module.signature.instructions),
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        max_demos=4,
    )
    assert len(tuned.demos) >= 1

    tuned_eval = evaluate(tuned, holdout_df.rename({"gold": "code"}), exact_match)
    assert isinstance(tuned_eval.score, float)
    assert 0.0 <= tuned_eval.score <= 1.0

    print(
        f"hitl real-api roundtrip: baseline={baseline.score:.3f} "
        f"tuned={tuned_eval.score:.3f}"
    )
