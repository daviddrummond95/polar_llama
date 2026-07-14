"""Tests for survey data-quality flags (issue #80): `quality_report`,
`QualityConfig`, and the standalone heuristic-tier expressions
(`straightlining_score`, `gibberish_score`, `duplicate_answer_score`,
`response_length_score`, `speeder_score`).

CI-safe: the heuristic tier makes zero API calls and needs no `mlx` /
API key, so every test in this file runs unconditionally. The
embedding/LLM tier (`QualityConfig(llm_tier=True)`) is exercised
separately in `tests/test_quality_llm.py`, gated on `OPENAI_API_KEY`.
"""

import random

import polars as pl
import pytest

from polar_llama import (
    QualityConfig,
    QualityReport,
    duplicate_answer_score,
    gibberish_score,
    quality_report,
    response_length_score,
    speeder_score,
    straightlining_score,
)

GRID_COLS = ["g1", "g2", "g3", "g4", "g5"]
TEXT_COLS = ["oe1", "oe2", "oe3"]
DURATION_COL = "duration_s"

_SENTENCE_POOL = [
    "The customer service team was really helpful when I called about my broken order last week.",
    "I think the app works fine most days but sometimes it crashes right after I open it.",
    "Overall I'm happy with the purchase although shipping took a bit longer than the site promised.",
    "My favorite part of the update is the new dashboard, it loads noticeably faster than before.",
    "The instructions were a little confusing at first but customer support cleared things up quickly.",
    "I would probably buy this again since the price felt fair for what you actually get.",
    "Delivery was fast and the packaging was solid, nothing got damaged during the whole trip.",
    "The interface takes some getting used to but once you learn it things move pretty smoothly.",
    "Not a huge fan of the new pricing tiers, they feel more expensive than the old plan.",
    "Support answered within an hour which honestly surprised me given how busy their queue looked.",
    "The onboarding flow was clear enough that I didn't need to contact anyone for help.",
    "Battery life dropped a bit after the last update but everything else feels about the same.",
]


def _make_filler_rows(n: int, *, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        # A shuffled permutation of 1..5 always has the same population
        # variance (2.0) regardless of order, so it can never brush the
        # straightlining threshold by chance the way independent uniform
        # draws occasionally can (e.g. 4-of-5 identical) -- deterministic,
        # not just "usually", non-suspicious grid answers.
        grid = [1, 2, 3, 4, 5]
        rng.shuffle(grid)
        sentences = rng.sample(_SENTENCE_POOL, 3)
        duration = rng.randint(200, 400)
        rows.append(
            {
                "respondent_id": f"filler_{i}",
                "g1": grid[0],
                "g2": grid[1],
                "g3": grid[2],
                "g4": grid[3],
                "g5": grid[4],
                "oe1": sentences[0],
                "oe2": sentences[1],
                "oe3": sentences[2],
                "duration_s": duration,
            }
        )
    return rows


def _build_fixture() -> pl.DataFrame:
    targeted_rows = [
        {
            "respondent_id": "straightliner",
            "g1": 4,
            "g2": 4,
            "g3": 4,
            "g4": 4,
            "g5": 4,
            "oe1": _SENTENCE_POOL[0],
            "oe2": _SENTENCE_POOL[1],
            "oe3": _SENTENCE_POOL[2],
            "duration_s": 300,
        },
        {
            "respondent_id": "gibberisher",
            "g1": 2,
            "g2": 4,
            "g3": 1,
            "g4": 5,
            "g5": 3,
            "oe1": "asdkjfhaslkdjf",
            "oe2": "qwpoeiruqpwoeiru zxmcnvb",
            "oe3": _SENTENCE_POOL[3],
            "duration_s": 280,
        },
        {
            "respondent_id": "duplicator",
            "g1": 1,
            "g2": 3,
            "g3": 5,
            "g4": 2,
            "g5": 4,
            "oe1": "The product is great and I would recommend it",
            "oe2": "the product is great and I would recommend it.",
            "oe3": _SENTENCE_POOL[4],
            "duration_s": 320,
        },
        {
            "respondent_id": "speeder",
            "g1": 3,
            "g2": 2,
            "g3": 4,
            "g4": 1,
            "g5": 5,
            "oe1": _SENTENCE_POOL[5],
            "oe2": _SENTENCE_POOL[6],
            "oe3": _SENTENCE_POOL[7],
            "duration_s": 12,
        },
        {
            "respondent_id": "normal",
            "g1": 2,
            "g2": 4,
            "g3": 3,
            "g4": 5,
            "g5": 1,
            "oe1": _SENTENCE_POOL[8],
            "oe2": _SENTENCE_POOL[9],
            "oe3": _SENTENCE_POOL[10],
            "duration_s": 290,
        },
        {
            "respondent_id": "long_ranter",
            "g1": 3,
            "g2": 4,
            "g3": 2,
            "g4": 5,
            "g5": 3,
            "oe1": "x" * 2000,
            "oe2": _SENTENCE_POOL[11],
            "oe3": _SENTENCE_POOL[0],
            "duration_s": 310,
        },
    ]
    rows = targeted_rows + _make_filler_rows(15)
    return pl.DataFrame(rows)


def _row(report: QualityReport, respondent_id: str) -> dict:
    return (
        report.df.filter(pl.col("respondent_id") == respondent_id)
        .select("quality")
        .to_series()[0]
    )


@pytest.fixture(scope="module")
def fixture_df() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture(scope="module")
def config() -> QualityConfig:
    return QualityConfig(
        id_column="respondent_id",
        grid_columns=GRID_COLS,
        text_columns=TEXT_COLS,
        duration_column=DURATION_COL,
    )


@pytest.fixture(scope="module")
def report(fixture_df, config) -> QualityReport:
    return quality_report(fixture_df, config)


# ============================================================================
# 1. Height / order preserved
# ============================================================================


def test_output_height_matches_input(fixture_df, report):
    assert report.df.height == fixture_df.height


def test_row_order_preserved(fixture_df, report):
    assert report.df["respondent_id"].to_list() == fixture_df["respondent_id"].to_list()


# ============================================================================
# 2. Each target respondent's flag fires -- and only that flag
# ============================================================================


def test_straightliner_flags_only_straightlining(report):
    q = _row(report, "straightliner")
    assert q["straightlining"]["flag"] is True
    assert q["gibberish"]["flag"] is False
    assert q["duplicate_answers"]["flag"] is False
    assert q["speeder"]["flag"] is False


def test_gibberisher_flags_only_gibberish(report):
    q = _row(report, "gibberisher")
    assert q["gibberish"]["flag"] is True
    assert q["straightlining"]["flag"] is False
    assert q["duplicate_answers"]["flag"] is False
    assert q["speeder"]["flag"] is False


def test_duplicator_flags_only_duplicate_answers(report):
    q = _row(report, "duplicator")
    assert q["duplicate_answers"]["flag"] is True
    assert q["duplicate_answers"]["score"] == pytest.approx(1.0, abs=1e-9)
    assert q["straightlining"]["flag"] is False
    assert q["gibberish"]["flag"] is False
    assert q["speeder"]["flag"] is False


def test_speeder_flags_only_speeder(report):
    q = _row(report, "speeder")
    assert q["speeder"]["flag"] is True
    assert q["straightlining"]["flag"] is False
    assert q["gibberish"]["flag"] is False
    assert q["duplicate_answers"]["flag"] is False


def test_long_ranter_flags_length_outlier(report):
    q = _row(report, "long_ranter")
    assert q["length_outlier"]["flag"] is True
    assert q["length_outlier"]["z"] > 0


def test_normal_respondent_has_no_flags(report):
    q = _row(report, "normal")
    assert q["any_flag"] is False
    assert q["n_flags"] == 0
    for name in ("straightlining", "gibberish", "duplicate_answers", "speeder", "length_outlier"):
        assert q[name]["flag"] is False


def test_filler_rows_mostly_unflagged(report):
    # Not a hard "zero false positives" guarantee (random fixture data can
    # rarely brush a subjective threshold), but the overwhelming majority of
    # 15 filler rows built from normal-length, non-repeating, non-gibberish
    # text and moderate grid variety should not trip anything.
    filler = report.df.filter(pl.col("respondent_id").str.starts_with("filler_"))
    n_flagged = filler.select(pl.col("quality").struct.field("any_flag").sum()).item()
    assert n_flagged <= 2, f"too many false positives among filler rows: {n_flagged}/15"


# ============================================================================
# 3. Graded ordering
# ============================================================================


def test_straightlining_score_ordering(report):
    straightliner = _row(report, "straightliner")["straightlining"]["score"]
    normal = _row(report, "normal")["straightlining"]["score"]
    assert straightliner > normal


def test_gibberish_score_ordering_and_margin():
    mash = pl.DataFrame({"t": ["asdkjfhaslkdjf"]}).select(
        s=gibberish_score("t")
    ).item()
    normal = pl.DataFrame({"t": ["The service was slow but friendly today overall"]}).select(
        s=gibberish_score("t")
    ).item()
    assert mash > normal + 0.3


def test_duplicate_score_is_one_for_verbatim_after_normalization():
    df = pl.DataFrame(
        {
            "a": ["The product is great and I would recommend it"],
            "b": ["the product is great and I would recommend it."],
        }
    )
    score = df.select(s=duplicate_answer_score(["a", "b"])).item()
    assert score == pytest.approx(1.0, abs=1e-9)


def test_speeder_score_is_max_for_fastest(report, fixture_df):
    scores = report.df.select(
        pl.col("respondent_id"), pl.col("quality").struct.field("speeder").struct.field("score")
    )
    speeder_score_value = scores.filter(pl.col("respondent_id") == "speeder")["score"][0]
    others_max = scores.filter(pl.col("respondent_id") != "speeder")["score"].max()
    assert speeder_score_value > 0
    assert speeder_score_value > (others_max or 0.0)


# ============================================================================
# 4. Standalone expressions agree with the report struct fields
# ============================================================================


def test_standalone_straightlining_matches_report(fixture_df, report, config):
    scale_vals = pl.concat([fixture_df[c].cast(pl.Float64) for c in GRID_COLS])
    standalone = fixture_df.select(
        s=straightlining_score(
            GRID_COLS, scale_min=scale_vals.min(), scale_max=scale_vals.max()
        )
    )["s"]
    report_scores = report.df.select(
        pl.col("quality").struct.field("straightlining").struct.field("score")
    )["score"]
    for a, b in zip(standalone.to_list(), report_scores.to_list()):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert a == pytest.approx(b, abs=1e-9)


def test_standalone_gibberish_matches_report(fixture_df, report):
    for col in TEXT_COLS:
        standalone = fixture_df.select(s=gibberish_score(col))["s"].to_list()
        # report's gibberish score is a max over columns, so only compare
        # rows where this column achieves the row max.
        report_scores = report.df.select(
            pl.col("quality").struct.field("gibberish").struct.field("score")
        )["score"].to_list()
        for a, b in zip(standalone, report_scores):
            if a is not None and b is not None:
                assert a <= b + 1e-9


def test_standalone_duplicate_matches_report(fixture_df, report):
    standalone = fixture_df.select(s=duplicate_answer_score(TEXT_COLS))["s"].to_list()
    report_scores = report.df.select(
        pl.col("quality").struct.field("duplicate_answers").struct.field("score")
    )["score"].to_list()
    for a, b in zip(standalone, report_scores):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert a == pytest.approx(b, abs=1e-9)


def test_standalone_response_length_score_bounded(fixture_df):
    scores = fixture_df.select(s=response_length_score("oe1"))["s"]
    for v in scores.to_list():
        if v is not None:
            assert 0.0 <= v <= 1.0


def test_standalone_speeder_matches_report(fixture_df, report):
    standalone = fixture_df.select(s=speeder_score(DURATION_COL))["s"].to_list()
    report_scores = report.df.select(
        pl.col("quality").struct.field("speeder").struct.field("score")
    )["score"].to_list()
    for a, b in zip(standalone, report_scores):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert a == pytest.approx(b, abs=1e-9)


def test_llama_namespace_matches_functional(fixture_df):
    functional = fixture_df.select(s=gibberish_score("oe1"))["s"]
    namespaced = fixture_df.select(s=pl.col("oe1").llama.gibberish_score())["s"]
    assert functional.to_list() == namespaced.to_list()

    functional_dup = fixture_df.select(s=duplicate_answer_score(TEXT_COLS))["s"]
    namespaced_dup = fixture_df.select(
        s=pl.col("oe1").llama.duplicate_answer_score(["oe2", "oe3"])
    )["s"]
    assert functional_dup.to_list() == namespaced_dup.to_list()

    functional_speed = fixture_df.select(s=speeder_score(DURATION_COL))["s"]
    namespaced_speed = fixture_df.select(s=pl.col(DURATION_COL).llama.speeder_score())["s"]
    assert functional_speed.to_list() == namespaced_speed.to_list()

    functional_len = fixture_df.select(s=response_length_score("oe1"))["s"]
    namespaced_len = fixture_df.select(s=pl.col("oe1").llama.response_length_score())["s"]
    assert functional_len.to_list() == namespaced_len.to_list()

    scale_vals = pl.concat([fixture_df[c].cast(pl.Float64) for c in GRID_COLS])
    functional_sl = fixture_df.select(
        s=straightlining_score(
            GRID_COLS, scale_min=scale_vals.min(), scale_max=scale_vals.max()
        )
    )["s"]
    namespaced_sl = fixture_df.select(
        s=pl.col("g1").llama.straightlining_score(
            ["g2", "g3", "g4", "g5"],
            scale_min=scale_vals.min(),
            scale_max=scale_vals.max(),
        )
    )["s"]
    assert functional_sl.to_list() == namespaced_sl.to_list()


# ============================================================================
# 5. Config validation
# ============================================================================


def test_config_rejects_empty_config():
    with pytest.raises(ValueError):
        QualityConfig()


def test_config_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        QualityConfig(grid_columns=GRID_COLS, straightlining_threshold=1.5)


def test_config_rejects_zero_threshold():
    with pytest.raises(ValueError):
        QualityConfig(grid_columns=GRID_COLS, gibberish_threshold=0.0)


def test_config_rejects_bad_scale_bounds():
    with pytest.raises(ValueError):
        QualityConfig(grid_columns=GRID_COLS, scale_min=5.0, scale_max=1.0)


def test_config_rejects_llm_tier_without_text_columns():
    with pytest.raises(ValueError):
        QualityConfig(grid_columns=GRID_COLS, llm_tier=True)


def test_quality_report_rejects_missing_column(fixture_df):
    bad_config = QualityConfig(grid_columns=["g1", "does_not_exist"])
    with pytest.raises(ValueError):
        quality_report(fixture_df, bad_config)


# ============================================================================
# 6. Nulls: never flagged, row still present
# ============================================================================


def test_all_null_text_row_has_null_scores_and_no_flags():
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "oe1": [None, "A perfectly normal open-end answer about the product quality."],
            "oe2": [None, "Shipping was on time and the box looked undamaged on arrival."],
        }
    )
    cfg = QualityConfig(id_column="id", text_columns=["oe1", "oe2"])
    result = quality_report(df, cfg)
    assert result.df.height == 2
    row = result.df.filter(pl.col("id") == "a").select("quality").to_series()[0]
    assert row["gibberish"]["score"] is None
    assert row["gibberish"]["flag"] is False
    assert row["duplicate_answers"]["score"] is None
    assert row["duplicate_answers"]["flag"] is False
    assert row["any_flag"] is False


# ============================================================================
# 7. Summary shape
# ============================================================================


def test_summary_shape(report):
    assert report.summary.columns == [
        "flag",
        "n_scored",
        "n_flagged",
        "rate",
        "threshold",
        "mean_score",
    ]
    flags = report.summary["flag"].to_list()
    assert "any_flag" in flags
    for expected in ("straightlining", "gibberish", "duplicate_answers", "speeder", "length_outlier"):
        assert expected in flags


def test_summary_n_flagged_matches_manual_count(report):
    row = report.summary.filter(pl.col("flag") == "straightlining").row(0, named=True)
    manual = report.df.select(
        pl.col("quality").struct.field("straightlining").struct.field("flag").sum()
    ).item()
    assert row["n_flagged"] == manual
    assert row["rate"] == pytest.approx(row["n_flagged"] / row["n_scored"])


def test_summary_any_flag_row(report, fixture_df):
    row = report.summary.filter(pl.col("flag") == "any_flag").row(0, named=True)
    assert row["n_scored"] == fixture_df.height
    manual = report.df.select(pl.col("quality").struct.field("any_flag").sum()).item()
    assert row["n_flagged"] == manual


# ============================================================================
# 8. llm_tier=False makes zero network calls (no key needed)
# ============================================================================


def test_llm_tier_false_needs_no_api_key(fixture_df, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = QualityConfig(
        grid_columns=GRID_COLS,
        text_columns=TEXT_COLS,
        duration_column=DURATION_COL,
        llm_tier=False,
    )
    result = quality_report(fixture_df, cfg)
    assert result.df.height == fixture_df.height
    q = result.df.select("quality").to_series()[0]
    assert "near_duplicate" not in q
    assert "likely_ai" not in q


# ============================================================================
# 9. quality_report never drops rows even with degenerate per-flag inputs
# ============================================================================


def test_single_row_dataframe_is_not_dropped():
    df = pl.DataFrame({"g1": [3], "g2": [3], "g3": [3]})
    cfg = QualityConfig(grid_columns=["g1", "g2", "g3"])
    result = quality_report(df, cfg)
    assert result.df.height == 1


def test_grid_only_config_omits_text_and_duration_substructs():
    df = pl.DataFrame({"g1": [3, 1], "g2": [3, 5], "g3": [3, 2]})
    cfg = QualityConfig(grid_columns=["g1", "g2", "g3"])
    result = quality_report(df, cfg)
    q = result.df.select("quality").to_series()[0]
    assert "straightlining" in q
    assert "gibberish" not in q
    assert "duplicate_answers" not in q
    assert "speeder" not in q
